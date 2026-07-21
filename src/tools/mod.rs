//! Agent tool execution: argument parsing, shared context, and dispatch.
//!
//! Port of the "Tool Argument Parsing And File Tool Helpers" and "Tool
//! Dispatch" sections of `ds4_agent.c`. Tool calls arrive as parsed
//! [`crate::dsml::ToolCall`] values; each tool returns the exact text the C
//! agent would feed back to the model as the tool-role result, including the
//! `Tool error: ...` convention for failures. The browser web tools
//! (`google_search`, `visit_page`) live in [`web`].

pub mod bash;
pub mod edit;
pub mod files;
pub mod mcp;
pub mod web;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::dsml::ToolCall;

/// Default timeout for bash commands, in seconds.
const BASH_DEFAULT_TIMEOUT_SEC: u64 = 3600;

/// Result of executing one tool call.
///
/// `output` is the model-visible observation text. `is_error` mirrors the C
/// convention: failures are plain text starting with `Tool error:`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResult {
    /// Model-visible observation text.
    pub output: String,
    /// True when the observation reports a tool failure.
    pub is_error: bool,
}

impl ToolResult {
    /// Wraps raw observation text, deriving `is_error` from the C convention.
    #[must_use]
    pub fn from_output(output: String) -> Self {
        let is_error = output.starts_with("Tool error:");
        Self { output, is_error }
    }
}

/// State for the `more` continuation tool: where the next read resumes.
#[derive(Debug, Clone)]
pub struct MoreState {
    /// Path of the file the previous truncated read came from.
    pub path: String,
    /// 1-based line the next chunk starts at.
    pub next_line: usize,
    /// True when the previous read was in raw (bare) mode.
    pub bare: bool,
}

/// Approval hook for the web tools, mirroring `agent_web_confirm`.
///
/// Receives the approval prompt and returns true to allow web access.
pub type WebConfirmFn = Box<dyn FnMut(&str) -> bool + Send>;

/// Mutable state shared by all tools of one agent worker.
pub struct ToolContext {
    /// Working directory relative paths are resolved against.
    pub cwd: PathBuf,
    /// Continuation state for the `more` tool, if a read was truncated.
    pub more: Option<MoreState>,
    /// Table of live and finished asynchronous bash jobs.
    pub bash: bash::BashJobs,
    /// Per-session web tool state (sticky approval flag).
    pub web: web::WebState,
    /// Web access approval hook; `None` auto-denies like non-interactive C.
    pub web_confirm: Option<WebConfirmFn>,
    /// Live MCP servers started from the `.mcp.json` config, if any.
    pub mcp: Vec<mcp::McpServer>,
    /// Resolved paths of recent successful `read` calls, oldest first, for
    /// post-compaction re-injection (`compact::build_reinjection`).
    pub recent_reads: Vec<PathBuf>,
    /// Command hooks (PreToolUse/PostToolUse/Stop) from hooks.json configs.
    pub hooks: crate::hooks::Hooks,
    /// Seatbelt sandbox policy for model-initiated bash commands.
    pub sandbox: crate::sandbox::Sandbox,
    /// User-only warnings from non-blocking hook failures, drained by the UI
    /// after each dispatch.
    pub hook_warnings: Vec<String>,
}

impl std::fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("cwd", &self.cwd)
            .field("more", &self.more)
            .field("bash", &self.bash)
            .field("web", &self.web)
            .field("web_confirm", &self.web_confirm.as_ref().map(|_| "<fn>"))
            .field("mcp", &self.mcp)
            .field("recent_reads", &self.recent_reads)
            .field("hooks", &self.hooks)
            .field("sandbox", &self.sandbox)
            .finish_non_exhaustive()
    }
}

impl ToolContext {
    /// Creates a context rooted at the given working directory.
    #[must_use]
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            more: None,
            bash: bash::BashJobs::default(),
            web: web::WebState::default(),
            web_confirm: None,
            mcp: Vec::new(),
            recent_reads: Vec::new(),
            hooks: crate::hooks::Hooks::default(),
            sandbox: crate::sandbox::Sandbox::default(),
            hook_warnings: Vec::new(),
        }
    }

    /// Records a successful file read for post-compaction re-injection:
    /// moves `path` to the newest slot and bounds the list.
    pub fn note_read(&mut self, path: PathBuf) {
        const RECENT_READS_CAP: usize = 16;
        self.recent_reads.retain(|p| *p != path);
        self.recent_reads.push(path);
        if self.recent_reads.len() > RECENT_READS_CAP {
            self.recent_reads
                .drain(..self.recent_reads.len() - RECENT_READS_CAP);
        }
    }

    /// Resolves a tool-provided path against the context working directory.
    #[must_use]
    pub fn resolve(&self, path: impl AsRef<Path>) -> PathBuf {
        let path = path.as_ref();
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        }
    }
}

/// Executes one parsed tool call and returns the model-visible result.
///
/// Mirrors `agent_execute_tool_call`: the same tool names the C agent
/// registers, minus the browser web tools.
pub fn dispatch(call: &ToolCall, ctx: &mut ToolContext) -> ToolResult {
    if call.name.is_empty() {
        return ToolResult::from_output("Tool error: missing tool name\n".to_string());
    }
    // PreToolUse hooks: exit 2 blocks the tool, its stderr becomes the
    // model-visible tool error.
    if !ctx.hooks.pre_tool_use.is_empty() {
        let input = crate::hooks::tool_event_input(
            "PreToolUse",
            &call.name,
            &mcp::args_to_json(call),
            None,
            &ctx.cwd,
        );
        let pre = crate::hooks::run_event(&ctx.hooks.pre_tool_use, &call.name, &input, &ctx.cwd);
        ctx.hook_warnings.extend(pre.warnings);
        if let Some(msg) = pre.block {
            return ToolResult::from_output(format!(
                "Tool error: blocked by PreToolUse hook: {msg}\n"
            ));
        }
    }
    let output = match call.name.as_str() {
        "read" => files::tool_read(ctx, call),
        "more" => files::tool_more(ctx, call),
        "write" => files::tool_write(ctx, call),
        "list" => files::tool_list(ctx, call),
        "glob" => files::tool_glob(ctx, call),
        "edit" => edit::tool_edit(ctx, call),
        "search" => edit::tool_search(ctx, call),
        "bash" => bash::tool_bash(ctx, call),
        "bash_status" => bash::tool_bash_status_or_stop(ctx, call, false),
        "bash_stop" => bash::tool_bash_status_or_stop(ctx, call, true),
        "google_search" => web::tool_google_search(ctx, call),
        "visit_page" => web::tool_visit_page(ctx, call),
        "mcp_describe" => mcp::tool_mcp_describe(&ctx.mcp, call),
        "mcp_list_resources" => mcp::tool_mcp_list_resources(&ctx.mcp, call),
        "mcp_read_resource" => mcp::tool_mcp_read_resource(&mut ctx.mcp, call),
        name if name.starts_with("mcp__") => mcp::tool_mcp_call(&mut ctx.mcp, call),
        other => format!("Tool error: unknown tool: {other}\n"),
    };
    // PostToolUse hooks: exit 2 appends stderr to the model's observation.
    let mut output = output;
    if !ctx.hooks.post_tool_use.is_empty() {
        let input = crate::hooks::tool_event_input(
            "PostToolUse",
            &call.name,
            &mcp::args_to_json(call),
            Some(&output),
            &ctx.cwd,
        );
        let post = crate::hooks::run_event(&ctx.hooks.post_tool_use, &call.name, &input, &ctx.cwd);
        ctx.hook_warnings.extend(post.warnings);
        if let Some(msg) = post.block {
            if !output.ends_with('\n') {
                output.push('\n');
            }
            let _ = writeln!(output, "[PostToolUse hook] {msg}");
        }
    }
    ToolResult::from_output(output)
}

/// Executes all calls of one DSML block, framing each result with its label.
///
/// Mirrors `agent_execute_tool_calls`, so the model can associate
/// observations with calls.
pub fn dispatch_all(calls: &[ToolCall], ctx: &mut ToolContext) -> String {
    if calls.is_empty() {
        return "Tool error: empty tool call block\n".to_string();
    }
    let mut all = String::new();
    for (i, call) in calls.iter().enumerate() {
        let res = dispatch(call, ctx);
        let name = if call.name.is_empty() {
            "unknown"
        } else {
            call.name.as_str()
        };
        let _ = writeln!(all, "Tool result {} ({}):", i + 1, name);
        all.push_str(&res.output);
        if !res.output.is_empty() && !res.output.ends_with('\n') {
            all.push('\n');
        }
    }
    all
}

/// Parses a bash timeout in seconds, clamped to `1..=86400`.
///
/// Mirrors `agent_parse_timeout`: missing or malformed values yield 3600.
#[must_use]
pub fn parse_timeout(s: Option<&str>) -> u64 {
    let Some(s) = s else {
        return BASH_DEFAULT_TIMEOUT_SEC;
    };
    let s = s.trim();
    // strtod stops at the first non-numeric byte; approximate by trying
    // progressively shorter prefixes of the leading float-looking run.
    let end = s
        .find(|c: char| !(c.is_ascii_digit() || "+-.eE".contains(c)))
        .unwrap_or(s.len());
    let Ok(v) = s[..end].parse::<f64>() else {
        return BASH_DEFAULT_TIMEOUT_SEC;
    };
    if v <= 0.0 || !v.is_finite() {
        return BASH_DEFAULT_TIMEOUT_SEC;
    }
    let v = v.clamp(1.0, 24.0 * 3600.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        v as u64
    }
}

/// Parses an integer argument with a default and clamping range.
///
/// Mirrors `agent_parse_int_default`: trailing whitespace is tolerated, any
/// other trailing text falls back to the default.
#[must_use]
pub fn parse_int_default(s: Option<&str>, def: i64, min: i64, max: i64) -> i64 {
    let Some(s) = s else { return def };
    let t = s.trim();
    if t.is_empty() {
        return def;
    }
    match t.parse::<i64>() {
        Ok(v) => v.clamp(min, max),
        Err(_) => def,
    }
}

/// Parses a boolean argument, accepting true/yes/1 and false/no/0.
#[must_use]
pub fn parse_bool_default(s: Option<&str>, def: bool) -> bool {
    let Some(s) = s else { return def };
    if s.is_empty() {
        return def;
    }
    if s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("yes") || s == "1" {
        return true;
    }
    if s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("no") || s == "0" {
        return false;
    }
    def
}

#[cfg(test)]
pub(crate) fn test_call(name: &str, args: &[(&str, &str)]) -> ToolCall {
    ToolCall {
        name: name.to_string(),
        args: args
            .iter()
            .map(|(n, v)| crate::dsml::ToolArg {
                name: (*n).to_string(),
                value: (*v).to_string(),
                is_string: true,
            })
            .collect(),
    }
}

#[cfg(test)]
pub(crate) fn test_ctx() -> (ToolContext, PathBuf) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "plank_tools_test_{}_{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    (ToolContext::new(&dir), dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_helpers_defaults() {
        assert_eq!(parse_timeout(None), 3600);
        assert_eq!(parse_timeout(Some("0")), 3600);
        assert_eq!(parse_timeout(Some("0.5")), 1);
        assert_eq!(parse_timeout(Some("999999")), 86400);
        assert_eq!(parse_int_default(Some("7"), 1, 1, 5), 5);
        assert_eq!(parse_int_default(Some("junk"), 9, 0, 100), 9);
        assert!(parse_bool_default(Some("YES"), false));
        assert!(!parse_bool_default(Some("0"), true));
        assert!(parse_bool_default(Some("maybe"), true));
    }

    #[test]
    fn dispatch_unknown_tool_errors() {
        let (mut ctx, dir) = test_ctx();
        let res = dispatch(&test_call("frobnicate", &[]), &mut ctx);
        assert!(res.is_error);
        assert_eq!(res.output, "Tool error: unknown tool: frobnicate\n");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn dispatch_all_frames_results() {
        let (mut ctx, dir) = test_ctx();
        let out = dispatch_all(&[test_call("nope", &[])], &mut ctx);
        assert!(out.starts_with("Tool result 1 (nope):\n"));
        assert_eq!(
            dispatch_all(&[], &mut ctx),
            "Tool error: empty tool call block\n"
        );
        std::fs::remove_dir_all(dir).ok();
    }
}
