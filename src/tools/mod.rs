// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Agent tool execution: argument parsing, shared context, and dispatch.
//!
//! Port of the "Tool Argument Parsing And File Tool Helpers" and "Tool
//! Dispatch" sections of `ds4_agent.c`. Tool calls arrive as parsed
//! [`crate::dsml::ToolCall`] values; each tool returns the exact text the C
//! agent would feed back to the model as the tool-role result, including the
//! `Tool error: ...` convention for failures. The browser web tools
//! (`google_search`, `visit_page`) live in [`web`].

pub mod ask;
pub mod bash;
pub mod diff;
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
    /// Skills the model may invoke via the `skill` tool (issue #36).
    pub skills: Vec<crate::skills::Skill>,
    /// Skill invocations so far this turn; the turn driver resets it to 0 at
    /// the start of each turn. Bounds runaway skill-invokes-skill recursion.
    pub skill_invocations: usize,
    /// Current sub-agent nesting depth (issue #50). The turn driver increments
    /// this around a delegated `agent` tool run and the `agent` tool refuses
    /// once it reaches [`SUBAGENT_DEPTH_CAP`], bounding agent-invokes-agent
    /// recursion the same way [`SKILL_DEPTH_CAP`] bounds skills.
    pub subagent_depth: usize,
    /// True while a read-only plan-mode gate is active (issue #50). Mutating
    /// tools refuse until `ExitPlanMode` clears it.
    pub plan_mode: bool,
    /// Which opt-in non-trained tools (`task`/`agent`/plan mode) are enabled,
    /// captured from settings when the context is built. Dispatch refuses a
    /// disabled tool even if the model calls it.
    pub tools: crate::settings::ToolSettings,
    /// Set by a tool hook's `{"continue": false}` response envelope; the turn
    /// driver halts the turn after the dispatch that produced it.
    pub hook_stop: Option<String>,
    /// Live model-visible task list (issue #35). The authoritative working copy
    /// during a turn; the driver mirrors it onto the session (which serializes
    /// it) so it survives compaction, `/resume`, and checkpoint rollback.
    pub tasks: crate::tasks::TaskList,
    /// Subjects of tasks the `task` tool just marked completed, drained by the
    /// UI after each dispatch to write the single dim completion log line.
    pub task_completions: Vec<String>,
    /// Diff previews from `edit`/`write` calls this dispatch, drained by the UI
    /// to render a git-style change card. Empty when nothing changed a file.
    pub edit_previews: Vec<diff::EditPreview>,
    /// Front end that presents `ask` questions (issue #34); `None` in
    /// non-interactive mode, where `ask` fast-fails instead of blocking.
    pub asker: Option<Box<dyn ask::Asker>>,
    /// UI-thread handle to the `ask` rendezvous, set only under the TUI (the
    /// worker's [`asker`](Self::asker) parks requests here for the event loop to
    /// render). `None` for the plain REPL (stdin asker) and non-interactive mode.
    pub ask_bridge: Option<ask::AskBridge>,
    /// Live browser session for the web tools, created lazily on first web use
    /// and reused across turns (like the C agent keeping Chrome alive). Only on
    /// `ds4_engine` builds; the curl path needs no handle.
    #[cfg(ds4_engine)]
    pub web_browser: Option<crate::ds4web::WebBrowser>,
}

/// Most `skill` invocations allowed within one turn before the tool refuses,
/// bounding a skill whose text tells the model to invoke another skill.
pub const SKILL_DEPTH_CAP: usize = 8;

/// Maximum sub-agent nesting depth (issue #50). A depth of 1 means a top-level
/// turn may delegate to a sub-agent, but that sub-agent may not delegate again;
/// this bounds runaway agent-invokes-agent recursion.
pub const SUBAGENT_DEPTH_CAP: usize = 1;

/// Tools that mutate the workspace and are therefore refused while plan mode is
/// active (issue #50). Read-only tools stay available so the model can research
/// before proposing a plan. `bash` is included because it can run arbitrary
/// side-effecting commands.
const PLAN_MODE_BLOCKED_TOOLS: &[&str] = &["write", "edit", "bash"];

/// True when `name` is a workspace-mutating tool blocked under plan mode.
#[must_use]
fn is_plan_mode_blocked(name: &str) -> bool {
    PLAN_MODE_BLOCKED_TOOLS.contains(&name)
}

/// True when `name` is an opt-in non-trained tool that is currently disabled.
/// (`agent` is gated at the `Agent` layer, not here.)
#[must_use]
fn is_tool_disabled(name: &str, tools: crate::settings::ToolSettings) -> bool {
    match name {
        "task" => !tools.task,
        "EnterPlanMode" | "ExitPlanMode" => !tools.plan_mode,
        _ => false,
    }
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
            skills: Vec::new(),
            skill_invocations: 0,
            subagent_depth: 0,
            plan_mode: false,
            tools: crate::settings::active().tools,
            hook_stop: None,
            tasks: crate::tasks::TaskList::new(),
            task_completions: Vec::new(),
            edit_previews: Vec::new(),
            asker: None,
            ask_bridge: None,
            #[cfg(ds4_engine)]
            web_browser: None,
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
#[allow(clippy::too_many_lines)]
pub fn dispatch(call: &ToolCall, ctx: &mut ToolContext) -> ToolResult {
    if call.name.is_empty() {
        return ToolResult::from_output("Tool error: missing tool name\n".to_string());
    }
    // Argument values feed argument matchers like `bash(git *)`.
    let arg_values: Vec<&str> = call.args.iter().map(|a| a.value.as_str()).collect();
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
        let pre = crate::hooks::run_event_args(
            &ctx.hooks.pre_tool_use,
            &call.name,
            &arg_values,
            &input,
            &ctx.cwd,
        );
        ctx.hook_warnings.extend(pre.warnings);
        ctx.hook_warnings.extend(pre.system_messages);
        if ctx.hook_stop.is_none() {
            ctx.hook_stop = pre.stop_reason;
        }
        if let Some(msg) = pre.block {
            return ToolResult::from_output(format!(
                "Tool error: blocked by PreToolUse hook: {msg}\n"
            ));
        }
    }
    // Non-trained tools are opt-in (see `ToolSettings`); if one is called while
    // disabled — a small model sometimes does — refuse cleanly instead of
    // running it, and never leave a stray plan-mode/task gate behind.
    if is_tool_disabled(&call.name, ctx.tools) {
        return ToolResult::from_output(format!(
            "Tool error: the {} tool is not enabled\n",
            call.name
        ));
    }
    // Plan mode (issue #50): while the read-only gate is active, refuse any
    // workspace-mutating tool so the model researches and proposes before it
    // edits. The gate itself is entered/exited by dedicated tools below.
    if ctx.plan_mode && is_plan_mode_blocked(&call.name) {
        return ToolResult::from_output(format!(
            "Tool error: plan mode is active — {} is read-only until you call ExitPlanMode with your proposed plan and it is approved\n",
            call.name
        ));
    }
    let output = match call.name.as_str() {
        "EnterPlanMode" => tool_enter_plan_mode(ctx),
        "ExitPlanMode" => tool_exit_plan_mode(ctx, call),
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
        "skill" => crate::skills::tool_skill(
            &ctx.skills,
            &mut ctx.skill_invocations,
            SKILL_DEPTH_CAP,
            call,
        ),
        "task" => crate::tasks::tool_task(&mut ctx.tasks, &mut ctx.task_completions, call),
        "ask" => ask::tool_ask(ctx.asker.as_mut(), call),
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
        let post = crate::hooks::run_event_args(
            &ctx.hooks.post_tool_use,
            &call.name,
            &arg_values,
            &input,
            &ctx.cwd,
        );
        ctx.hook_warnings.extend(post.warnings);
        ctx.hook_warnings.extend(post.system_messages);
        if ctx.hook_stop.is_none() {
            ctx.hook_stop = post.stop_reason;
        }
        if let Some(msg) = post.block {
            if !output.ends_with('\n') {
                output.push('\n');
            }
            let _ = writeln!(output, "[PostToolUse hook] {msg}");
        }
    }
    // PostToolUseFailure hooks: fire only when the tool failed (the C
    // `Tool error:` convention); success never reaches here.
    if output.starts_with("Tool error:") && !ctx.hooks.post_tool_use_failure.is_empty() {
        fire_post_tool_failure(ctx, call, &arg_values, &mut output);
    }
    ToolResult::from_output(output)
}

/// Fires the `PostToolUseFailure` hooks and appends any exit-2 block message to
/// `output`, mirroring the `PostToolUse` block framing. Split out of `dispatch`
/// to keep it under the function-length lint.
fn fire_post_tool_failure(
    ctx: &mut ToolContext,
    call: &ToolCall,
    arg_values: &[&str],
    output: &mut String,
) {
    // plank has no per-tool interrupt tracking in the dispatch path, so the
    // `is_interrupt` flag the reference carries is always false here; it is
    // still emitted so hooks can rely on the field being present.
    let base = crate::hooks::tool_event_input(
        "PostToolUseFailure",
        &call.name,
        &mcp::args_to_json(call),
        Some(output),
        &ctx.cwd,
    );
    let input = format!("{},\"is_interrupt\":false}}", &base[..base.len() - 1]);
    let fail = crate::hooks::run_event_args(
        &ctx.hooks.post_tool_use_failure,
        &call.name,
        arg_values,
        &input,
        &ctx.cwd,
    );
    ctx.hook_warnings.extend(fail.warnings);
    ctx.hook_warnings.extend(fail.system_messages);
    if ctx.hook_stop.is_none() {
        ctx.hook_stop = fail.stop_reason;
    }
    if let Some(msg) = fail.block {
        if !output.ends_with('\n') {
            output.push('\n');
        }
        let _ = writeln!(output, "[PostToolUseFailure hook] {msg}");
    }
}

/// Handles `EnterPlanMode`: turns on the read-only plan gate (issue #50).
///
/// Idempotent — entering plan mode when already in it just reaffirms the gate.
fn tool_enter_plan_mode(ctx: &mut ToolContext) -> String {
    ctx.plan_mode = true;
    "Plan mode is on. You are now read-only: research with read/list/glob/search \
     and the web/MCP tools, but do not modify the workspace. When you have a \
     concrete plan, call ExitPlanMode with the plan in its 'plan' argument to \
     ask the user for approval before making changes.\n"
        .to_string()
}

/// Handles `ExitPlanMode`: presents the proposed plan for approval and, when
/// approved, lifts the read-only gate (issue #50).
///
/// With an interactive [`ask::Asker`] the user approves or rejects; a rejection
/// keeps the gate on. Without one (non-interactive / headless) the plan is
/// auto-approved so scripted runs are not wedged, mirroring the `ask` tool's
/// non-interactive fast-path.
fn tool_exit_plan_mode(ctx: &mut ToolContext, call: &ToolCall) -> String {
    if !ctx.plan_mode {
        return "Tool error: ExitPlanMode called but plan mode is not active\n".to_string();
    }
    let plan = call.arg_value("plan").unwrap_or("").trim();
    if plan.is_empty() {
        return "Tool error: ExitPlanMode requires a non-empty 'plan' describing what you intend to do\n"
            .to_string();
    }
    let Some(asker) = ctx.asker.as_mut() else {
        // No interactive user to approve; lift the gate and proceed.
        ctx.plan_mode = false;
        return "No interactive user is available to approve the plan \
                (non-interactive mode); plan mode lifted, proceed.\n"
            .to_string();
    };
    let req = ask::AskRequest {
        question: format!("Approve this plan?\n\n{plan}"),
        header: "Plan".to_string(),
        options: vec![
            ask::AskOption {
                label: "Approve".to_string(),
                description: "Proceed with the plan and allow edits".to_string(),
            },
            ask::AskOption {
                label: "Keep planning".to_string(),
                description: "Stay read-only and refine the plan".to_string(),
            },
        ],
        multi: false,
    };
    match asker.ask(req) {
        ask::AskOutcome::Answered(labels) if labels.iter().any(|l| l == "Approve") => {
            ctx.plan_mode = false;
            "Plan approved. Plan mode is off; you may now modify the workspace to carry it out.\n"
                .to_string()
        }
        _ => {
            "Plan not approved; plan mode stays on. Refine the plan and call ExitPlanMode again.\n"
                .to_string()
        }
    }
}

/// Executes all calls of one DSML block, framing each result with its label.
///
/// Mirrors `agent_execute_tool_calls`, so the model can associate
/// observations with calls.
pub fn dispatch_all(calls: &[ToolCall], ctx: &mut ToolContext) -> String {
    if calls.is_empty() {
        return "Tool error: empty tool call block\n".to_string();
    }
    // Diff previews accumulate per dispatch; clear any a prior caller left
    // undrained so cards never leak between turns.
    ctx.edit_previews.clear();
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
    fn plan_mode_gates_mutating_tools_and_exits_on_approval() {
        let (mut ctx, dir) = test_ctx();
        ctx.tools.plan_mode = true; // opt-in tool (default off)
        // Entering plan mode turns on the read-only gate.
        let res = dispatch(&test_call("EnterPlanMode", &[]), &mut ctx);
        assert!(!res.is_error);
        assert!(ctx.plan_mode);
        // A mutating tool is now refused with a plan-mode error.
        let res = dispatch(
            &test_call("write", &[("path", "x.txt"), ("content", "hi")]),
            &mut ctx,
        );
        assert!(res.is_error);
        assert!(res.output.contains("plan mode is active"));
        // A read-only tool still works (list of the scratch dir).
        let res = dispatch(&test_call("list", &[]), &mut ctx);
        assert!(!res.is_error, "read-only tool blocked: {}", res.output);
        // ExitPlanMode requires a plan.
        let res = dispatch(&test_call("ExitPlanMode", &[]), &mut ctx);
        assert!(res.is_error);
        assert!(ctx.plan_mode, "gate must stay on without a plan");
        // With a plan and no asker (non-interactive), it auto-approves.
        let res = dispatch(
            &test_call("ExitPlanMode", &[("plan", "do the thing")]),
            &mut ctx,
        );
        assert!(!res.is_error);
        assert!(!ctx.plan_mode, "gate must lift after approval");
        // Now the mutating tool is allowed again.
        let res = dispatch(
            &test_call("write", &[("path", "x.txt"), ("content", "hi")]),
            &mut ctx,
        );
        assert!(!res.is_error, "write still blocked: {}", res.output);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn exit_plan_mode_errors_when_not_planning() {
        let (mut ctx, dir) = test_ctx();
        ctx.tools.plan_mode = true; // opt-in tool (default off)
        let res = dispatch(&test_call("ExitPlanMode", &[("plan", "p")]), &mut ctx);
        assert!(res.is_error);
        assert!(res.output.contains("plan mode is not active"));
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
