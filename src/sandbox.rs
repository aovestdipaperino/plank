// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! macOS Seatbelt sandbox for model-initiated shell commands (issue #17).
//!
//! When enabled, `bash` tool commands run under `/usr/bin/sandbox-exec` with
//! a generated profile: read everywhere, write only under the working
//! directory, temp dirs, and any configured extra roots. User-typed `!`
//! commands are never sandboxed — the user typing the command is the
//! authorization.
//!
//! Configured via `~/.plank/sandbox.json` overlaid by `./.plank/sandbox.json`:
//!
//! ```json
//! {
//!   "enabled": true,
//!   "writablePaths": ["/some/extra/root"],
//!   "excludedCommands": ["git push*", "brew *"]
//! }
//! ```
//!
//! Scalars come from the most specific file; list values concatenate (like
//! hooks.json). `excludedCommands` is a convenience escape hatch, not a
//! security boundary — a `*`-glob match against the whole command line skips
//! the sandbox for that command.
//!
//! `sandbox-exec` is deprecated by Apple but remains functional and is what
//! the reference agents use on macOS.

use crate::tools::mcp::{Json, json_parse};
use std::path::{Path, PathBuf};

/// Sandbox policy for model-initiated bash commands.
#[derive(Debug, Clone, Default)]
pub struct Sandbox {
    /// Master switch; default off.
    pub enabled: bool,
    /// Extra roots writable in addition to cwd and temp dirs.
    pub writable_paths: Vec<PathBuf>,
    /// `*`-glob patterns for commands that skip the sandbox entirely.
    pub excluded_commands: Vec<String>,
}

impl Sandbox {
    /// True when this command should run under `sandbox-exec`.
    #[must_use]
    pub fn should_sandbox(&self, cmd: &str) -> bool {
        if !self.enabled {
            return false;
        }
        let cmd = cmd.trim();
        !self
            .excluded_commands
            .iter()
            .any(|pat| glob_match(pat.trim(), cmd))
    }

    /// Builds the Seatbelt (SBPL) profile: allow everything, deny all file
    /// writes, then re-allow writes under cwd, temp roots, /dev, and the
    /// configured extra paths. Later rules win in SBPL, so the allow list
    /// punches holes in the write denial.
    #[must_use]
    pub fn profile(&self, cwd: &Path) -> String {
        let mut p = String::from("(version 1)\n(allow default)\n(deny file-write*)\n");
        p.push_str("(allow file-write*\n");
        let mut roots: Vec<PathBuf> = vec![
            cwd.to_path_buf(),
            PathBuf::from("/tmp"),
            PathBuf::from("/private/tmp"),
            PathBuf::from("/var/folders"),
            PathBuf::from("/private/var/folders"),
            PathBuf::from("/dev"),
        ];
        roots.extend(self.writable_paths.iter().cloned());
        for root in roots {
            // Resolve symlinks where possible: Seatbelt matches the real
            // path, and macOS cwds are often under the /tmp -> /private/tmp
            // or /var -> /private/var symlinks.
            let real = root.canonicalize().unwrap_or(root);
            p.push_str("  (subpath \"");
            p.push_str(&sbpl_escape(&real.to_string_lossy()));
            p.push_str("\")\n");
        }
        p.push_str(")\n");
        p
    }
}

/// Escapes a path for use inside a double-quoted SBPL string literal.
fn sbpl_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Matches `pat` (literal text with `*` wildcards) against the whole of
/// `text`. No escaping; `?`/character classes are not supported.
#[must_use]
pub fn glob_match(pat: &str, text: &str) -> bool {
    let segs: Vec<&str> = pat.split('*').collect();
    if segs.len() == 1 {
        return pat == text;
    }
    let mut rest = text;
    for (i, seg) in segs.iter().enumerate() {
        if seg.is_empty() {
            continue;
        }
        if i == 0 {
            let Some(r) = rest.strip_prefix(seg) else {
                return false;
            };
            rest = r;
        } else if i == segs.len() - 1 {
            return rest.ends_with(seg);
        } else {
            let Some(pos) = rest.find(seg) else {
                return false;
            };
            rest = &rest[pos + seg.len()..];
        }
    }
    // Pattern ends with '*' (last segment empty) or everything consumed.
    segs.last().is_some_and(|s| s.is_empty()) || rest.is_empty()
}

/// Parses one sandbox.json file into `sb`. Scalars overwrite, lists append.
fn apply_config(sb: &mut Sandbox, text: &str) {
    let Some(root) = json_parse(text) else {
        return;
    };
    if let Some(Json::Bool(b)) = root.get("enabled") {
        sb.enabled = *b;
    }
    if let Some(Json::Arr(items)) = root.get("writablePaths") {
        for item in items {
            if let Json::Str(s) = item {
                sb.writable_paths.push(PathBuf::from(s));
            }
        }
    }
    if let Some(Json::Arr(items)) = root.get("excludedCommands") {
        for item in items {
            if let Json::Str(s) = item {
                sb.excluded_commands.push(s.clone());
            }
        }
    }
}

/// Loads `~/.plank/sandbox.json` then `<cwd>/.plank/sandbox.json`; the
/// project file wins on scalars and appends to lists.
#[must_use]
pub fn load_default(cwd: &Path) -> Sandbox {
    let mut sb = Sandbox::default();
    if let Ok(home) = std::env::var("HOME")
        && let Ok(text) = std::fs::read_to_string(Path::new(&home).join(".plank/sandbox.json"))
    {
        apply_config(&mut sb, &text);
    }
    if let Ok(text) = std::fs::read_to_string(cwd.join(".plank/sandbox.json")) {
        apply_config(&mut sb, &text);
    }
    sb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_by_default() {
        let sb = Sandbox::default();
        assert!(!sb.should_sandbox("rm -rf /"));
    }

    #[test]
    fn glob_matching() {
        assert!(glob_match("git push*", "git push origin main"));
        assert!(glob_match("git push*", "git push"));
        assert!(!glob_match("git push*", "git pull"));
        assert!(glob_match("* --version", "clang --version"));
        assert!(glob_match("brew * plank", "brew install plank"));
        assert!(glob_match("exact", "exact"));
        assert!(!glob_match("exact", "exactly"));
        assert!(glob_match("*", "anything at all"));
    }

    #[test]
    fn excluded_commands_skip_sandbox() {
        let sb = Sandbox {
            enabled: true,
            writable_paths: Vec::new(),
            excluded_commands: vec!["git push*".to_string()],
        };
        assert!(sb.should_sandbox("cargo build"));
        assert!(!sb.should_sandbox("git push origin main"));
        assert!(!sb.should_sandbox("  git push  "));
    }

    #[test]
    fn profile_contains_cwd_and_escapes() {
        let sb = Sandbox {
            enabled: true,
            writable_paths: vec![PathBuf::from("/odd\"name")],
            excluded_commands: Vec::new(),
        };
        let p = sb.profile(Path::new("/nonexistent/work dir"));
        assert!(p.starts_with("(version 1)\n(allow default)\n(deny file-write*)\n"));
        assert!(p.contains("(subpath \"/nonexistent/work dir\")"));
        assert!(p.contains("(subpath \"/odd\\\"name\")"));
        assert!(p.contains("(subpath \"/dev\")"));
    }

    #[test]
    fn config_merge_appends_lists() {
        let mut sb = Sandbox::default();
        apply_config(
            &mut sb,
            r#"{"enabled": true, "writablePaths": ["/a"], "excludedCommands": ["x*"]}"#,
        );
        apply_config(
            &mut sb,
            r#"{"writablePaths": ["/b"], "excludedCommands": ["y"]}"#,
        );
        assert!(sb.enabled);
        assert_eq!(sb.writable_paths.len(), 2);
        assert_eq!(sb.excluded_commands, vec!["x*", "y"]);
    }
}
