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

/// All configured hooks, by event.
#[derive(Debug, Clone, Default)]
pub struct Hooks {
    /// Hooks run before each tool call.
    pub pre_tool_use: Vec<HookMatcher>,
    /// Hooks run after each tool call.
    pub post_tool_use: Vec<HookMatcher>,
    /// Hooks run when a turn is about to conclude.
    pub stop: Vec<HookMatcher>,
}

impl Hooks {
    /// True when no hooks are configured at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pre_tool_use.is_empty() && self.post_tool_use.is_empty() && self.stop.is_empty()
    }
}

/// Outcome of running the hooks for one event occurrence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HookOutcome {
    /// Exit-2 stderr, destined for the model; `Some` blocks/continues per event.
    pub block: Option<String>,
    /// Other-nonzero stderr lines, destined for the user only.
    pub warnings: Vec<String>,
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
    let timeout_sec = match v.get("timeout") {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Some(Json::Num(n)) if *n >= 1.0 => n.round() as u64,
        _ => HOOK_DEFAULT_TIMEOUT_SEC,
    };
    Some(HookDef {
        command,
        timeout_sec,
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

/// Parses one hooks.json text; unknown events and hook types are ignored.
#[must_use]
pub fn parse_config(text: &str) -> Hooks {
    let Some(mut root) = json_parse(text) else {
        return Hooks::default();
    };
    // Accept the reference settings.json shape with a top-level "hooks" key.
    if let Some(inner) = root.get("hooks") {
        root = inner.clone();
    }
    Hooks {
        pre_tool_use: parse_event(&root, "PreToolUse"),
        post_tool_use: parse_event(&root, "PostToolUse"),
        stop: parse_event(&root, "Stop"),
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
        merged.stop.extend(h.stop);
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

/// Runs one hook command with `input` on stdin; returns (exit code, stderr).
/// A timeout or spawn failure reads as a user-visible warning, never a block.
fn run_hook(def: &HookDef, input: &str, cwd: &Path) -> (i32, String) {
    let child = Command::new("/bin/sh")
        .arg("-c")
        .arg(&def.command)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => return (1, format!("hook failed to start: {e}")),
    };
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(input.as_bytes());
    }
    let stderr = child.stderr.take();
    let reader = std::thread::spawn(move || {
        let mut out = Vec::new();
        if let Some(mut s) = stderr {
            use std::io::Read as _;
            let _ = s.read_to_end(&mut out);
        }
        String::from_utf8_lossy(&out).into_owned()
    });
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
    let stderr_text = reader.join().unwrap_or_default();
    match status {
        Some(s) => (s.code().unwrap_or(1), stderr_text),
        None => (1, format!("hook timed out after {}s", def.timeout_sec)),
    }
}

/// Runs every matching hook of one event; the first exit-2 stderr becomes
/// `block` (remaining hooks still run), other nonzero exits accumulate
/// user-visible warnings.
#[must_use]
pub fn run_event(groups: &[HookMatcher], target: &str, input: &str, cwd: &Path) -> HookOutcome {
    let mut outcome = HookOutcome::default();
    for group in groups.iter().filter(|g| g.matches(target)) {
        for def in &group.hooks {
            let (code, stderr) = run_hook(def, input, cwd);
            let stderr = stderr.trim().to_string();
            match code {
                0 => {}
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
    outcome
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
        ("Stop", &hooks.stop),
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
