//! Command hooks: user shell commands run at lifecycle points.
//!
//! The first slice of the reference agent's hook system, command hooks only:
//!
//! - **`PreToolUse`** — before a tool executes; exit code 2 blocks the tool and
//!   its stderr becomes the model-visible tool error.
//! - **`PostToolUse`** — after a tool executes; exit code 2 appends its stderr
//!   to the observation the model sees.
//! - **`Stop`** — when a turn is about to conclude; exit code 2 feeds its
//!   stderr back to the model and the turn continues (once per turn).
//!
//! Any other nonzero exit shows stderr to the *user* only. Hook input is a
//! JSON object piped to the command's stdin. Configuration is JSON, merged
//! from `~/.plank/hooks.json` then `./.plank/hooks.json` (both lists run),
//! mirroring the `.mcp.json` layering:
//!
//! ```json
//! {
//!   "PreToolUse": [
//!     { "matcher": "bash|edit",
//!       "hooks": [ { "type": "command", "command": "check.sh", "timeout": 60 } ] }
//!   ]
//! }
//! ```
//!
//! A top-level `"hooks"` wrapper object (reference settings.json shape) is
//! also accepted. An empty or missing `matcher` matches every tool.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::tools::mcp::{Json, json_escape, json_parse};

/// Default hook timeout, in seconds.
const HOOK_DEFAULT_TIMEOUT_SEC: u64 = 60;

/// One command hook.
#[derive(Debug, Clone)]
pub struct HookDef {
    /// Shell command run via `/bin/sh -c`.
    pub command: String,
    /// Kill the hook after this many seconds.
    pub timeout_sec: u64,
    /// When true (config `"async": true`), the hook is spawned fire-and-forget
    /// and never blocks the turn; its exit code and output are ignored.
    pub is_async: bool,
}

/// A matcher group: hooks that run when the matcher accepts the target.
#[derive(Debug, Clone)]
pub struct HookMatcher {
    /// `|`-separated tool names; empty matches everything.
    pub matcher: String,
    /// Hooks run in order when the matcher accepts.
    pub hooks: Vec<HookDef>,
}

impl HookMatcher {
    /// True when this group applies to `target` (a tool name; Stop hooks use
    /// an empty target and match everything).
    #[must_use]
    pub fn matches(&self, target: &str) -> bool {
        let m = self.matcher.trim();
        m.is_empty() || m.split('|').any(|p| p.trim() == target)
    }
}

/// Every event name plank recognizes in a hooks config. A config naming any
/// other event is ignored with a warning rather than failing to load.
pub const KNOWN_EVENTS: &[&str] = &[
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "Stop",
    "UserPromptSubmit",
    "SessionStart",
    "SessionEnd",
    "PreCompact",
    "PostCompact",
];

/// All configured hooks, by event.
#[derive(Debug, Clone, Default)]
pub struct Hooks {
    /// Hooks run before each tool call.
    pub pre_tool_use: Vec<HookMatcher>,
    /// Hooks run after each tool call (success or failure).
    pub post_tool_use: Vec<HookMatcher>,
    /// Hooks run after a tool call that *failed* (carries the error).
    pub post_tool_use_failure: Vec<HookMatcher>,
    /// Hooks run when a turn is about to conclude.
    pub stop: Vec<HookMatcher>,
    /// Hooks run on every submitted user prompt; may inject turn context.
    pub user_prompt_submit: Vec<HookMatcher>,
    /// Hooks run when a session begins (startup|resume|clear|compact); may
    /// inject context.
    pub session_start: Vec<HookMatcher>,
    /// Hooks run when a session ends (carries the exit reason).
    pub session_end: Vec<HookMatcher>,
    /// Hooks run before a compaction pass (trigger manual|auto); may inject
    /// context.
    pub pre_compact: Vec<HookMatcher>,
    /// Hooks run after a compaction pass (carries the resulting summary); may
    /// inject context.
    pub post_compact: Vec<HookMatcher>,
    /// Warnings gathered while loading (e.g. unknown event names), surfaced to
    /// the user at startup rather than aborting the load.
    pub warnings: Vec<String>,
}

impl Hooks {
    /// True when no hooks are configured at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pre_tool_use.is_empty()
            && self.post_tool_use.is_empty()
            && self.post_tool_use_failure.is_empty()
            && self.stop.is_empty()
            && self.user_prompt_submit.is_empty()
            && self.session_start.is_empty()
            && self.session_end.is_empty()
            && self.pre_compact.is_empty()
            && self.post_compact.is_empty()
    }
}

/// Outcome of running the hooks for one event occurrence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HookOutcome {
    /// Exit-2 stderr, destined for the model; `Some` blocks/continues per event.
    pub block: Option<String>,
    /// Other-nonzero stderr lines, destined for the user only.
    pub warnings: Vec<String>,
    /// Exit-0 stdout of context-capable events (`UserPromptSubmit`,
    /// `SessionStart`, `PreCompact`, `PostCompact`), destined for the model as
    /// injected turn context. Empty for tool/stop events.
    pub context: Option<String>,
    /// Set by a `{"continue": false}` response envelope: the turn should halt,
    /// carrying the optional `stopReason` (or a default) for the user.
    pub stop_reason: Option<String>,
    /// `systemMessage` envelope values: user-visible notes that never block.
    pub system_messages: Vec<String>,
    /// `suppressOutput` envelope flag: keep this hook's stdout out of the
    /// transcript (no context injection).
    pub suppress_output: bool,
}

fn parse_hook_def(v: &Json) -> Option<HookDef> {
    // Only command hooks are supported; other types are ignored.
    if v.str_or("type", "command") != "command" {
        return None;
    }
    let command = v.str_or("command", "").to_string();
    if command.is_empty() {
        return None;
    }
    let is_async = matches!(v.get("async"), Some(Json::Bool(true)));
    // `asyncTimeout` overrides the kill deadline for async hooks; otherwise the
    // usual `timeout` applies.
    let timeout_key = if is_async && v.get("asyncTimeout").is_some() {
        "asyncTimeout"
    } else {
        "timeout"
    };
    let timeout_sec = match v.get(timeout_key) {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Some(Json::Num(n)) if *n >= 1.0 => n.round() as u64,
        _ => HOOK_DEFAULT_TIMEOUT_SEC,
    };
    Some(HookDef {
        command,
        timeout_sec,
        is_async,
    })
}

fn parse_event(root: &Json, event: &str) -> Vec<HookMatcher> {
    let Some(Json::Arr(groups)) = root.get(event) else {
        return Vec::new();
    };
    groups
        .iter()
        .filter_map(|g| {
            let hooks: Vec<HookDef> = match g.get("hooks") {
                Some(Json::Arr(list)) => list.iter().filter_map(parse_hook_def).collect(),
                _ => Vec::new(),
            };
            if hooks.is_empty() {
                return None;
            }
            Some(HookMatcher {
                matcher: g.str_or("matcher", "").to_string(),
                hooks,
            })
        })
        .collect()
}

/// Parses one hooks.json text; unknown hook types are ignored and unknown
/// event names are skipped with a warning (never a load failure).
#[must_use]
pub fn parse_config(text: &str) -> Hooks {
    let Some(mut root) = json_parse(text) else {
        return Hooks::default();
    };
    // Accept the reference settings.json shape with a top-level "hooks" key.
    if let Some(inner) = root.get("hooks") {
        root = inner.clone();
    }
    // Warn on any event key we do not recognize, so a typo or a config aimed
    // at a richer agent degrades gracefully instead of silently vanishing.
    let mut warnings = Vec::new();
    if let Json::Obj(members) = &root {
        for (key, _) in members {
            if !KNOWN_EVENTS.contains(&key.as_str()) {
                warnings.push(format!("hooks: ignoring unknown event \"{key}\""));
            }
        }
    }
    Hooks {
        pre_tool_use: parse_event(&root, "PreToolUse"),
        post_tool_use: parse_event(&root, "PostToolUse"),
        post_tool_use_failure: parse_event(&root, "PostToolUseFailure"),
        stop: parse_event(&root, "Stop"),
        user_prompt_submit: parse_event(&root, "UserPromptSubmit"),
        session_start: parse_event(&root, "SessionStart"),
        session_end: parse_event(&root, "SessionEnd"),
        pre_compact: parse_event(&root, "PreCompact"),
        post_compact: parse_event(&root, "PostCompact"),
        warnings,
    }
}

/// Loads and merges hooks from the given config files, in order. Unlike MCP
/// servers (keyed by name), hook lists concatenate: global hooks and project
/// hooks all run.
#[must_use]
pub fn load_from(paths: &[PathBuf]) -> Hooks {
    let mut merged = Hooks::default();
    for path in paths {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let h = parse_config(&text);
        merged.pre_tool_use.extend(h.pre_tool_use);
        merged.post_tool_use.extend(h.post_tool_use);
        merged.post_tool_use_failure.extend(h.post_tool_use_failure);
        merged.stop.extend(h.stop);
        merged.user_prompt_submit.extend(h.user_prompt_submit);
        merged.session_start.extend(h.session_start);
        merged.session_end.extend(h.session_end);
        merged.pre_compact.extend(h.pre_compact);
        merged.post_compact.extend(h.post_compact);
        merged.warnings.extend(h.warnings);
    }
    merged
}

/// Loads hooks from the default hierarchy: `~/.plank/hooks.json` then
/// `<cwd>/.plank/hooks.json`.
#[must_use]
pub fn load_default(cwd: &Path) -> Hooks {
    let mut paths = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        paths.push(PathBuf::from(home).join(".plank").join("hooks.json"));
    }
    paths.push(cwd.join(".plank").join("hooks.json"));
    load_from(&paths)
}

/// Builds the stdin JSON for a hook event. `response` is included as
/// `tool_response` for `PostToolUse`.
#[must_use]
pub fn tool_event_input(
    event: &str,
    tool_name: &str,
    args_json: &str,
    response: Option<&str>,
    cwd: &Path,
) -> String {
    let mut out = String::from("{\"hook_event_name\":");
    json_escape(&mut out, event);
    out.push_str(",\"tool_name\":");
    json_escape(&mut out, tool_name);
    out.push_str(",\"tool_input\":");
    out.push_str(if args_json.is_empty() {
        "{}"
    } else {
        args_json
    });
    if let Some(response) = response {
        out.push_str(",\"tool_response\":");
        json_escape(&mut out, response);
    }
    out.push_str(",\"cwd\":");
    json_escape(&mut out, &cwd.to_string_lossy());
    out.push('}');
    out
}

/// Builds the stdin JSON for a non-tool lifecycle event. `fields` are extra
/// string members specific to the event (e.g. `("source", "startup")` for
/// `SessionStart`, `("prompt", text)` for `UserPromptSubmit`), always emitted
/// as JSON strings alongside the common `hook_event_name` and `cwd`.
#[must_use]
pub fn lifecycle_event_input(event: &str, fields: &[(&str, &str)], cwd: &Path) -> String {
    let mut out = String::from("{\"hook_event_name\":");
    json_escape(&mut out, event);
    for (key, value) in fields {
        out.push(',');
        json_escape(&mut out, key);
        out.push(':');
        json_escape(&mut out, value);
    }
    out.push_str(",\"cwd\":");
    json_escape(&mut out, &cwd.to_string_lossy());
    out.push('}');
    out
}

/// Runs one hook command with `input` on stdin; returns (exit code, stdout,
/// stderr). A timeout or spawn failure reads as a user-visible warning (on
/// stderr), never a block.
fn run_hook(def: &HookDef, input: &str, cwd: &Path) -> (i32, String, String) {
    let child = Command::new("/bin/sh")
        .arg("-c")
        .arg(&def.command)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => return (1, String::new(), format!("hook failed to start: {e}")),
    };
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(input.as_bytes());
    }
    // Drain stdout and stderr on their own threads so a hook that fills one
    // pipe buffer cannot deadlock against our wait loop.
    let stdout = child.stdout.take();
    let out_reader = std::thread::spawn(move || read_all(stdout));
    let stderr = child.stderr.take();
    let err_reader = std::thread::spawn(move || read_all(stderr));
    let deadline = Instant::now() + Duration::from_secs(def.timeout_sec);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {}
            Err(_) => break None,
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let stdout_text = out_reader.join().unwrap_or_default();
    let stderr_text = err_reader.join().unwrap_or_default();
    match status {
        Some(s) => (s.code().unwrap_or(1), stdout_text, stderr_text),
        None => (
            1,
            stdout_text,
            format!("hook timed out after {}s", def.timeout_sec),
        ),
    }
}

/// Reads a child pipe to end as a lossy UTF-8 string.
fn read_all<R: std::io::Read>(pipe: Option<R>) -> String {
    let mut out = Vec::new();
    if let Some(mut s) = pipe {
        let _ = s.read_to_end(&mut out);
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Runs every matching hook of one event; the first exit-2 stderr becomes
/// `block` (remaining hooks still run), other nonzero exits accumulate
/// user-visible warnings.
#[must_use]
pub fn run_event(groups: &[HookMatcher], target: &str, input: &str, cwd: &Path) -> HookOutcome {
    run_event_inner(groups, target, input, cwd, false)
}

/// Like [`run_event`], but exit-0 stdout of each matching hook is collected as
/// injected turn context ([`HookOutcome::context`]). Used by the context-capable
/// lifecycle events (`UserPromptSubmit`, `SessionStart`, `PreCompact`,
/// `PostCompact`).
#[must_use]
pub fn run_event_ctx(groups: &[HookMatcher], target: &str, input: &str, cwd: &Path) -> HookOutcome {
    run_event_inner(groups, target, input, cwd, true)
}

fn run_event_inner(
    groups: &[HookMatcher],
    target: &str,
    input: &str,
    cwd: &Path,
    capture_context: bool,
) -> HookOutcome {
    let mut outcome = HookOutcome::default();
    let mut context = String::new();
    for group in groups.iter().filter(|g| g.matches(target)) {
        for def in &group.hooks {
            // async hooks are fire-and-forget: spawn, ignore result, never block.
            if def.is_async {
                let (def, input, cwd) = (def.clone(), input.to_owned(), cwd.to_path_buf());
                std::thread::spawn(move || {
                    let _ = run_hook(&def, &input, &cwd);
                });
                continue;
            }
            let (code, stdout, stderr) = run_hook(def, input, cwd);
            let stderr = stderr.trim().to_string();
            // Optional JSON response envelope on stdout. When absent, stdout is
            // plain context text and exit codes stay fully authoritative.
            let envelope = parse_envelope(&stdout);
            if let Some(env) = &envelope {
                if let Some(reason) = &env.stop_reason
                    && outcome.stop_reason.is_none()
                {
                    outcome.stop_reason = Some(reason.clone());
                }
                if let Some(msg) = &env.system_message {
                    outcome.system_messages.push(msg.clone());
                }
                if env.suppress_output {
                    outcome.suppress_output = true;
                }
            }
            match code {
                0 => {
                    // Plain (non-envelope) stdout of a context event is injected
                    // unless the hook asked to suppress its output.
                    if capture_context && envelope.is_none() && !outcome.suppress_output {
                        let stdout = stdout.trim();
                        if !stdout.is_empty() {
                            if !context.is_empty() {
                                context.push('\n');
                            }
                            context.push_str(stdout);
                        }
                    }
                }
                2 => {
                    if outcome.block.is_none() {
                        outcome.block = Some(if stderr.is_empty() {
                            format!("blocked by hook: {}", def.command)
                        } else {
                            stderr
                        });
                    }
                }
                _ => {
                    if !stderr.is_empty() {
                        outcome
                            .warnings
                            .push(format!("[hook: {}] {stderr}", def.command));
                    }
                }
            }
        }
    }
    if !context.is_empty() {
        outcome.context = Some(context);
    }
    outcome
}

/// Parsed subset of the response envelope plank honors.
struct Envelope {
    /// `Some` when `continue` is `false`; carries `stopReason` (or a default).
    stop_reason: Option<String>,
    /// `systemMessage` string, if present.
    system_message: Option<String>,
    /// `suppressOutput` flag.
    suppress_output: bool,
}

/// Parses a hook's stdout as a response envelope. Returns `None` unless the
/// stdout is a JSON object (the additive contract: no JSON means exit codes
/// alone decide). An object with none of the known keys still counts as an
/// envelope, so its stdout is not re-used as plain context.
fn parse_envelope(stdout: &str) -> Option<Envelope> {
    let trimmed = stdout.trim();
    if !trimmed.starts_with('{') {
        return None;
    }
    let Some(root @ Json::Obj(_)) = json_parse(trimmed) else {
        return None;
    };
    let stop_reason = match root.get("continue") {
        Some(Json::Bool(false)) => {
            Some(root.str_or("stopReason", "hook requested stop").to_string())
        }
        _ => None,
    };
    let system_message = match root.get("systemMessage") {
        Some(Json::Str(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    };
    let suppress_output = matches!(root.get("suppressOutput"), Some(Json::Bool(true)));
    Some(Envelope {
        stop_reason,
        system_message,
        suppress_output,
    })
}

/// Renders the `/hooks` listing.
#[must_use]
pub fn render_list(hooks: &Hooks) -> String {
    if hooks.is_empty() {
        return "no hooks configured (checked ~/.plank/hooks.json and ./.plank/hooks.json)\n"
            .to_string();
    }
    let mut out = String::from("Command hooks:\n");
    for (event, groups) in [
        ("PreToolUse", &hooks.pre_tool_use),
        ("PostToolUse", &hooks.post_tool_use),
        ("PostToolUseFailure", &hooks.post_tool_use_failure),
        ("Stop", &hooks.stop),
        ("UserPromptSubmit", &hooks.user_prompt_submit),
        ("SessionStart", &hooks.session_start),
        ("SessionEnd", &hooks.session_end),
        ("PreCompact", &hooks.pre_compact),
        ("PostCompact", &hooks.post_compact),
    ] {
        for g in groups {
            for h in &g.hooks {
                let scope = if g.matcher.trim().is_empty() {
                    "*".to_string()
                } else {
                    g.matcher.clone()
                };
                let _ = writeln!(
                    out,
                    "  {event} [{scope}] {} (timeout {}s)",
                    h.command, h.timeout_sec
                );
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"{
        "PreToolUse": [
            { "matcher": "bash|edit",
              "hooks": [ { "type": "command", "command": "exit 0" } ] }
        ],
        "PostToolUse": [
            { "hooks": [ { "type": "command", "command": "echo ok", "timeout": 5 } ] }
        ],
        "Stop": [
            { "hooks": [ { "type": "prompt", "prompt": "ignored" },
                          { "type": "command", "command": "check.sh" } ] }
        ]
    }"#;

    #[test]
    fn parses_events_matchers_and_ignores_non_command() {
        let h = parse_config(CONFIG);
        assert_eq!(h.pre_tool_use.len(), 1);
        assert_eq!(h.pre_tool_use[0].matcher, "bash|edit");
        assert_eq!(h.post_tool_use[0].hooks[0].timeout_sec, 5);
        // The prompt-type hook is ignored; the command survives.
        assert_eq!(h.stop[0].hooks.len(), 1);
        assert_eq!(h.stop[0].hooks[0].command, "check.sh");
        assert_eq!(h.stop[0].hooks[0].timeout_sec, HOOK_DEFAULT_TIMEOUT_SEC);
    }

    #[test]
    fn parses_new_lifecycle_events() {
        let cfg = r#"{
            "UserPromptSubmit": [ { "hooks": [ { "type": "command", "command": "u" } ] } ],
            "SessionStart":     [ { "hooks": [ { "type": "command", "command": "s" } ] } ],
            "SessionEnd":       [ { "hooks": [ { "type": "command", "command": "e" } ] } ],
            "PreCompact":       [ { "hooks": [ { "type": "command", "command": "pre" } ] } ],
            "PostCompact":      [ { "hooks": [ { "type": "command", "command": "post" } ] } ],
            "PostToolUseFailure": [ { "hooks": [ { "type": "command", "command": "f" } ] } ]
        }"#;
        let h = parse_config(cfg);
        assert_eq!(h.user_prompt_submit[0].hooks[0].command, "u");
        assert_eq!(h.session_start[0].hooks[0].command, "s");
        assert_eq!(h.session_end[0].hooks[0].command, "e");
        assert_eq!(h.pre_compact[0].hooks[0].command, "pre");
        assert_eq!(h.post_compact[0].hooks[0].command, "post");
        assert_eq!(h.post_tool_use_failure[0].hooks[0].command, "f");
        assert!(h.warnings.is_empty());
        assert!(!h.is_empty());
    }

    #[test]
    fn unknown_event_warns_but_loads() {
        let cfg = r#"{
            "Bogus": [ { "hooks": [ { "type": "command", "command": "x" } ] } ],
            "Stop":  [ { "hooks": [ { "type": "command", "command": "s" } ] } ]
        }"#;
        let h = parse_config(cfg);
        // The known event still loads; the unknown one is dropped with a warning.
        assert_eq!(h.stop[0].hooks[0].command, "s");
        assert_eq!(h.warnings.len(), 1);
        assert!(h.warnings[0].contains("Bogus"));
    }

    #[test]
    fn lifecycle_input_carries_fields() {
        let cwd = std::env::temp_dir();
        let input = lifecycle_event_input("SessionStart", &[("source", "startup")], &cwd);
        assert!(input.contains("\"hook_event_name\":\"SessionStart\""));
        assert!(input.contains("\"source\":\"startup\""));
        assert!(input.contains("\"cwd\":"));
    }

    #[test]
    fn context_capable_event_collects_stdout() {
        let cwd = std::env::temp_dir();
        // Exit-0 stdout becomes injected context; exit-0 stderr is ignored.
        let out = run_event_ctx(&one("echo hello-context", ""), "", "{}", &cwd);
        assert_eq!(out.context.as_deref(), Some("hello-context"));
        assert!(out.block.is_none());
        // Plain run_event never captures stdout as context.
        let plain = run_event(&one("echo hello-context", ""), "", "{}", &cwd);
        assert!(plain.context.is_none());
    }

    #[test]
    fn envelope_continue_false_halts_and_system_message_warns() {
        let cwd = std::env::temp_dir();
        let out = run_event(
            &one(
                r#"echo '{"continue": false, "stopReason": "stop now", "systemMessage": "heads up"}'"#,
                "",
            ),
            "bash",
            "{}",
            &cwd,
        );
        assert_eq!(out.stop_reason.as_deref(), Some("stop now"));
        assert_eq!(out.system_messages, vec!["heads up".to_string()]);
        // Envelope JSON is not re-used as plain context.
        assert!(out.context.is_none());
    }

    #[test]
    fn envelope_suppress_output_blocks_context_injection() {
        let cwd = std::env::temp_dir();
        // Plain stdout would inject; suppressOutput keeps it out. The JSON here
        // *is* the envelope, so there is no plain context anyway — assert the
        // flag is read.
        let out = run_event_ctx(
            &one(r#"echo '{"suppressOutput": true}'"#, ""),
            "",
            "{}",
            &cwd,
        );
        assert!(out.suppress_output);
        assert!(out.context.is_none());
    }

    #[test]
    fn no_envelope_keeps_exit_code_semantics() {
        let cwd = std::env::temp_dir();
        // Non-JSON stdout on exit 2 still blocks via stderr; stdout ignored.
        let out = run_event(
            &one("echo plain; echo err >&2; exit 2", ""),
            "bash",
            "{}",
            &cwd,
        );
        assert_eq!(out.block.as_deref(), Some("err"));
        assert!(out.stop_reason.is_none());
    }

    #[test]
    fn async_hook_does_not_block() {
        let cwd = std::env::temp_dir();
        let groups = vec![HookMatcher {
            matcher: String::new(),
            hooks: vec![HookDef {
                command: "sleep 30".to_string(),
                timeout_sec: 30,
                is_async: true,
            }],
        }];
        let start = Instant::now();
        let out = run_event(&groups, "bash", "{}", &cwd);
        // Fire-and-forget: returns immediately, contributes nothing.
        assert!(start.elapsed() < Duration::from_secs(2));
        assert_eq!(out, HookOutcome::default());
    }

    #[test]
    fn parses_async_flag_and_timeout() {
        let cfg = r#"{
            "Stop": [ { "hooks": [
                { "type": "command", "command": "x", "async": true, "asyncTimeout": 5 }
            ] } ]
        }"#;
        let h = parse_config(cfg);
        assert!(h.stop[0].hooks[0].is_async);
        assert_eq!(h.stop[0].hooks[0].timeout_sec, 5);
    }

    #[test]
    fn accepts_wrapped_hooks_key_and_bad_json() {
        let h = parse_config(&format!("{{\"hooks\":{CONFIG}}}"));
        assert_eq!(h.pre_tool_use.len(), 1);
        assert!(parse_config("not json").is_empty());
    }

    #[test]
    fn matcher_semantics() {
        let m = HookMatcher {
            matcher: "bash | edit".to_string(),
            hooks: Vec::new(),
        };
        assert!(m.matches("bash"));
        assert!(m.matches("edit"));
        assert!(!m.matches("read"));
        let all = HookMatcher {
            matcher: String::new(),
            hooks: Vec::new(),
        };
        assert!(all.matches("anything"));
    }

    fn one(cmd: &str, matcher: &str) -> Vec<HookMatcher> {
        vec![HookMatcher {
            matcher: matcher.to_string(),
            hooks: vec![HookDef {
                command: cmd.to_string(),
                timeout_sec: 5,
                is_async: false,
            }],
        }]
    }

    #[test]
    fn exit_codes_map_to_outcomes() {
        let cwd = std::env::temp_dir();
        // Exit 0: silent.
        let ok = run_event(&one("exit 0", ""), "bash", "{}", &cwd);
        assert_eq!(ok, HookOutcome::default());
        // Exit 2: blocks with stderr.
        let block = run_event(&one("echo nope >&2; exit 2", ""), "bash", "{}", &cwd);
        assert_eq!(block.block.as_deref(), Some("nope"));
        // Other nonzero: user warning only.
        let warn = run_event(&one("echo careful >&2; exit 1", ""), "bash", "{}", &cwd);
        assert!(warn.block.is_none());
        assert_eq!(warn.warnings.len(), 1);
        assert!(warn.warnings[0].contains("careful"));
        // Non-matching target: nothing runs.
        let skip = run_event(&one("exit 2", "edit"), "bash", "{}", &cwd);
        assert!(skip.block.is_none());
    }

    #[test]
    fn hook_reads_input_from_stdin() {
        let cwd = std::env::temp_dir();
        let input = tool_event_input("PreToolUse", "bash", "{\"command\":\"ls\"}", None, &cwd);
        let out = run_event(
            &one(
                "grep -q '\"tool_name\":\"bash\"' && exit 0 || { echo missing >&2; exit 2; }",
                "",
            ),
            "bash",
            &input,
            &cwd,
        );
        assert!(out.block.is_none(), "{out:?}");
    }

    #[test]
    fn timeout_is_a_warning_not_a_block() {
        let cwd = std::env::temp_dir();
        let groups = vec![HookMatcher {
            matcher: String::new(),
            hooks: vec![HookDef {
                command: "sleep 30".to_string(),
                timeout_sec: 1,
                is_async: false,
            }],
        }];
        let start = Instant::now();
        let out = run_event(&groups, "bash", "{}", &cwd);
        assert!(start.elapsed() < Duration::from_secs(5));
        assert!(out.block.is_none());
        assert_eq!(out.warnings.len(), 1);
        assert!(out.warnings[0].contains("timed out"));
    }
}
