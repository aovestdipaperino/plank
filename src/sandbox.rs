// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! macOS Seatbelt sandbox for model-initiated shell commands (issue #17).
//!
//! On by default on macOS: a model-chosen command should not be able to write
//! outside the project it was pointed at. `--no-sandbox` (or `"enabled": false`)
//! turns it off.
//!
//! When enabled, `bash` tool commands run under `/usr/bin/sandbox-exec` with
//! a generated profile: read everywhere, write only under the working
//! directory, temp dirs, and any configured extra roots. User-typed `!` and
//! `!!` commands are never sandboxed — the user typing the command is the
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
//! `~/.plank` is deliberately *not* writable by default: it holds the session
//! store, the KV cache, hooks and consent markers, so a model-chosen command
//! that can rewrite it can rewrite plank's own behaviour. It is instead granted
//! on request — when a sandboxed command names the plank home
//! ([`mentions_plank_home`]) the bash tool asks the user, and an "always allow"
//! answer sets [`Sandbox::plank_home_writable`] for the rest of the session
//! only. Nothing about that grant is written to disk.
//!
//! `sandbox-exec` is deprecated by Apple but remains functional and is what
//! the reference agents use on macOS.

use crate::tools::mcp::{Json, json_parse};
use std::path::{Path, PathBuf};

/// Sandbox policy for model-initiated bash commands.
#[derive(Debug, Clone)]
pub struct Sandbox {
    /// Master switch; on by default wherever `sandbox-exec` exists (macOS).
    pub enabled: bool,
    /// Extra roots writable in addition to cwd and temp dirs.
    pub writable_paths: Vec<PathBuf>,
    /// `*`-glob patterns for commands that skip the sandbox entirely.
    pub excluded_commands: Vec<String>,
    /// Session-scoped grant for writes under `~/.plank`, set by an "always
    /// allow" answer to the bash tool's prompt. In-memory only: a new session
    /// (or a `/resume` of this one) starts denied again.
    pub plank_home_writable: bool,
}

impl Default for Sandbox {
    /// On where Seatbelt exists, off elsewhere: `sandbox-exec` is macOS-only,
    /// and wrapping a command in a binary that is not there would fail every
    /// model-initiated command on other platforms.
    fn default() -> Self {
        Self {
            enabled: cfg!(target_os = "macos"),
            writable_paths: Vec::new(),
            excluded_commands: Vec::new(),
            plank_home_writable: false,
        }
    }
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
    ///
    /// `~/.plank` is included only when [`plank_home_writable`](Self::plank_home_writable)
    /// is set; see [`profile_allowing_plank_home`](Self::profile_allowing_plank_home)
    /// for the single-command grant.
    #[must_use]
    pub fn profile(&self, cwd: &Path) -> String {
        self.profile_allowing_plank_home(cwd, self.plank_home_writable)
    }

    /// Same as [`profile`](Self::profile) with the `~/.plank` write grant forced
    /// on or off, for a user who answered "Allow" for one command without
    /// granting the rest of the session.
    #[must_use]
    pub fn profile_allowing_plank_home(&self, cwd: &Path, allow_plank_home: bool) -> String {
        self.profile_with_plank_home(cwd, allow_plank_home.then(plank_home).flatten().as_deref())
    }

    /// The profile builder proper, with the plank home passed in rather than read
    /// from `HOME`, so tests need not mutate the environment other tests read.
    /// `None` withholds the grant.
    fn profile_with_plank_home(&self, cwd: &Path, plank_home: Option<&Path>) -> String {
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
        if let Some(home) = plank_home {
            roots.push(home.to_path_buf());
        }
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

/// The plank home directory, `$HOME/.plank`, or `None` when `HOME` is unset.
#[must_use]
pub fn plank_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".plank"))
}

/// True when `cmd` names the plank home, in any of the spellings a shell command
/// plausibly uses: `~/.plank`, `$HOME/.plank`, `${HOME}/.plank`, or the expanded
/// absolute path.
///
/// This is the trigger for the write-permission prompt. It reads the command
/// text because Seatbelt profiles are built before the command runs, so there is
/// no write to observe yet — which makes it a heuristic in both directions: a
/// command that only *reads* `~/.plank` still prompts, and one that reaches the
/// directory through a variable or a symlink is not caught. It is not the
/// security boundary; the boundary is the profile, which withholds the write
/// unless the user grants it.
#[must_use]
pub fn mentions_plank_home(cmd: &str) -> bool {
    mentions_plank_home_at(cmd, plank_home().as_deref())
}

/// The mention check proper, with the plank home passed in rather than read from
/// `HOME`, so tests need not mutate the environment other tests read.
fn mentions_plank_home_at(cmd: &str, plank_home: Option<&Path>) -> bool {
    let mut needles = vec!["~/.plank", "$HOME/.plank", "${HOME}/.plank"];
    let expanded = plank_home.map(|h| h.to_string_lossy().into_owned());
    if let Some(e) = expanded.as_deref() {
        needles.push(e);
    }
    needles.iter().any(|n| contains_path_prefix(cmd, n))
}

/// True when `needle` occurs in `text` as a whole path component prefix, so
/// `~/.plank` and `~/.plank/kvcache` match but `~/.plankton` does not.
fn contains_path_prefix(text: &str, needle: &str) -> bool {
    let mut from = 0;
    while let Some(pos) = text[from..].find(needle) {
        let end = from + pos + needle.len();
        let next = text[end..].chars().next();
        if !next.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '-') {
            return true;
        }
        from = end;
    }
    false
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
    fn enabled_by_default_on_macos_only() {
        let sb = Sandbox::default();
        assert_eq!(sb.should_sandbox("rm -rf /"), cfg!(target_os = "macos"));
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
            plank_home_writable: false,
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
            plank_home_writable: false,
        };
        let p = sb.profile(Path::new("/nonexistent/work dir"));
        assert!(p.starts_with("(version 1)\n(allow default)\n(deny file-write*)\n"));
        assert!(p.contains("(subpath \"/nonexistent/work dir\")"));
        assert!(p.contains("(subpath \"/odd\\\"name\")"));
        assert!(p.contains("(subpath \"/dev\")"));
    }

    /// A plank home that cannot exist, so `canonicalize` is a no-op and the
    /// profile carries the literal path.
    const FAKE_PLANK_HOME: &str = "/nonexistent/home/.plank";

    #[test]
    fn plank_home_is_not_writable_until_granted() {
        let mut sb = Sandbox {
            enabled: true,
            writable_paths: Vec::new(),
            excluded_commands: Vec::new(),
            plank_home_writable: false,
        };
        let cwd = Path::new("/nonexistent/work");
        let home = Path::new(FAKE_PLANK_HOME);
        let subpath = format!("(subpath \"{FAKE_PLANK_HOME}\")");

        // Denied (and the default): no grant reaches the profile at all.
        assert!(!sb.profile_with_plank_home(cwd, None).contains(".plank"));
        // A one-command "Allow" punches the hole for that command only...
        assert!(
            sb.profile_with_plank_home(cwd, Some(home))
                .contains(&subpath)
        );
        // ...without recording anything on the session.
        assert!(!sb.plank_home_writable);

        // "Always allow" sets the session flag, which is what `profile` reads.
        sb.plank_home_writable = true;
        assert!(sb.plank_home_writable);
        assert!(
            sb.profile_allowing_plank_home(cwd, false)
                .contains("(allow file-write*"),
            "an explicit false still builds a valid profile"
        );
    }

    #[test]
    fn plank_home_mentions_match_whole_path_components() {
        let home = Path::new(FAKE_PLANK_HOME);
        let m = |cmd: &str| mentions_plank_home_at(cmd, Some(home));
        assert!(m("cat ~/.plank/sandbox.json"));
        assert!(m("rm -rf $HOME/.plank"));
        assert!(m("ls ${HOME}/.plank/kvcache"));
        assert!(m("touch /nonexistent/home/.plank/x"));
        assert!(m("echo hi > ~/.plank"));
        // Neighbouring names that merely share the prefix must not prompt.
        assert!(!m("cat ~/.plankton/config"));
        assert!(!m("cat ~/.plank-old/config"));
        assert!(!m("cat ~/.plank_backup"));
        assert!(!m("ls /nonexistent/home/.plank-sandbox-test-1"));
        // Unrelated commands, including the project-local .plank directory.
        assert!(!m("cargo build"));
        assert!(!m("cat ./.plank/sandbox.json"));
        // The tilde spellings are recognised even with no HOME to expand.
        assert!(mentions_plank_home_at("cat ~/.plank/x", None));
        assert!(!mentions_plank_home_at(
            "touch /nonexistent/home/.plank/x",
            None
        ));
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
