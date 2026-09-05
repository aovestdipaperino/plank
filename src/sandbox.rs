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
//! hooks.json). The project file can only *tighten* the policy, though: its
//! `"enabled": false`, `writablePaths` and `excludedCommands` are ignored,
//! because a cloned checkout must not be able to relax the sandbox for the
//! user who opens it. `excludedCommands` is a convenience escape hatch, not a
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
        for real in self.write_roots_with_plank_home(cwd, plank_home) {
            p.push_str("  (subpath \"");
            p.push_str(&sbpl_escape(&real.to_string_lossy()));
            p.push_str("\")\n");
        }
        p.push_str(")\n");
        p
    }

    /// The directories a model-initiated write may land in: cwd, the temp
    /// roots, `/dev`, the configured extra paths, and `~/.plank` once granted.
    /// Symlinks are resolved where possible, so callers compare against
    /// canonical paths. This is the one list both the Seatbelt profile and the
    /// file tools' containment check ([`Sandbox::contains_write_target`]) are
    /// built from, so the two can never disagree.
    #[must_use]
    pub fn write_roots(&self, cwd: &Path) -> Vec<PathBuf> {
        self.write_roots_with_plank_home(
            cwd,
            self.plank_home_writable
                .then(plank_home)
                .flatten()
                .as_deref(),
        )
    }

    fn write_roots_with_plank_home(&self, cwd: &Path, plank_home: Option<&Path>) -> Vec<PathBuf> {
        let mut roots: Vec<PathBuf> = vec![
            cwd.to_path_buf(),
            PathBuf::from("/tmp"),
            PathBuf::from("/private/tmp"),
            PathBuf::from("/var/folders"),
            PathBuf::from("/private/var/folders"),
            PathBuf::from("/dev"),
            std::env::temp_dir(),
        ];
        roots.extend(self.writable_paths.iter().cloned());
        if let Some(home) = plank_home {
            roots.push(home.to_path_buf());
        }
        // Resolve symlinks where possible: Seatbelt matches the real path,
        // and macOS cwds are often under the /tmp -> /private/tmp or
        // /var -> /private/var symlinks.
        roots
            .into_iter()
            .map(|root| root.canonicalize().unwrap_or(root))
            .collect()
    }

    /// True when a file tool may write `target` (already resolved against
    /// `cwd`): the sandbox is off, or the target's real location lies under
    /// one of [`write_roots`](Self::write_roots). The target need not exist
    /// yet — a file about to be created is judged by its parent directory —
    /// and `..` segments are resolved before the comparison, so
    /// `<cwd>/../outside` cannot slip past.
    #[must_use]
    pub fn contains_write_target(&self, cwd: &Path, target: &Path) -> bool {
        if !self.enabled {
            return true;
        }
        let real = realpath_for_write(target);
        self.write_roots(cwd).iter().any(|r| real.starts_with(r))
    }
}

/// The real location a write to `target` would land at. An existing target
/// canonicalizes directly; a file about to be created canonicalizes its
/// parent and re-attaches the file name; when even the parent is missing the
/// path is normalised lexically (`.` and `..` folded) so a `..` escape is still
/// visible to the containment check.
fn realpath_for_write(target: &Path) -> PathBuf {
    if let Ok(real) = target.canonicalize() {
        return real;
    }
    if let (Some(parent), Some(name)) = (target.parent(), target.file_name())
        && let Ok(real_parent) = parent.canonicalize()
    {
        return real_parent.join(name);
    }
    lexical_normalize(target)
}

/// Folds `.` and `..` components without touching the filesystem.
fn lexical_normalize(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
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

/// True when `cmd` is provably read-only: every simple command in the line
/// starts with a utility from a small allowlist of readers, and no output
/// redirect appears anywhere. The [`mentions_plank_home`] prompt is skipped for
/// such commands, because `cat ~/.plank/settings.json` cannot write no matter
/// what the profile says.
///
/// Conservative in the safe direction: anything not recognised (an unknown
/// program, `sed -i`, `find -delete`, a subshell, a script) counts as a
/// potential write and keeps the prompt. A miss here therefore costs one extra
/// question, never a silent grant — the Seatbelt profile remains the boundary.
#[must_use]
pub fn is_read_only_command(cmd: &str) -> bool {
    const READERS: &[&str] = &[
        "cat",
        "ls",
        "head",
        "tail",
        "less",
        "more",
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "wc",
        "stat",
        "file",
        "du",
        "df",
        "echo",
        "printf",
        "sort",
        "uniq",
        "jq",
        "diff",
        "cmp",
        "test",
        "[",
        "readlink",
        "realpath",
        "tree",
        "cut",
        "tr",
        "column",
        "nl",
        "od",
        "xxd",
        "hexdump",
        "strings",
        "basename",
        "dirname",
        "pwd",
        "which",
        "type",
        "true",
        "false",
        "md5",
        "shasum",
        "sha256sum",
        "md5sum",
        "date",
        "env",
        "printenv",
    ];
    if cmd.contains('>') || cmd.contains('`') || cmd.contains("$(") {
        return false;
    }
    let mut any = false;
    for segment in cmd
        .split(['|', ';', '\n'])
        .flat_map(|s| s.split("&&"))
        .flat_map(|s| s.split("||"))
    {
        let mut words = segment.split_whitespace();
        let Some(first) = words.next() else {
            continue;
        };
        any = true;
        // `2>&1` never survives the '>' check above, so a leading env
        // assignment is the only prefix to skip.
        let head = if first.contains('=') && !first.starts_with('=') {
            match words.next() {
                Some(w) => w,
                None => return false,
            }
        } else {
            first
        };
        let name = head.rsplit('/').next().unwrap_or(head);
        if !READERS.contains(&name) {
            return false;
        }
    }
    any
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

/// Where a sandbox.json came from, which decides how much of it is believed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfigSource {
    /// `~/.plank/sandbox.json`: the user's own file, fully trusted.
    User,
    /// `<cwd>/.plank/sandbox.json`: ships with the checkout, so it may only
    /// tighten the policy. A cloned repository must not be able to switch the
    /// sandbox off, exclude commands from it, or widen the writable roots.
    Project,
}

/// Parses one sandbox.json file into `sb`. Scalars overwrite, lists append.
///
/// From a [`ConfigSource::Project`] file the relaxing keys are ignored:
/// `enabled` is honoured only when it is `true`, and `writablePaths` and
/// `excludedCommands` are dropped entirely, since every entry in either list
/// widens what a model-chosen command may do.
fn apply_config(sb: &mut Sandbox, text: &str, source: ConfigSource) {
    let Some(root) = json_parse(text) else {
        return;
    };
    if let Some(Json::Bool(b)) = root.get("enabled") {
        // A project file may turn the sandbox on, never off.
        if *b || source == ConfigSource::User {
            sb.enabled = *b;
        }
    }
    if source == ConfigSource::Project {
        return;
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

/// Loads `~/.plank/sandbox.json` then `<cwd>/.plank/sandbox.json`. Only the
/// user file can relax the sandbox; the project file can only tighten it
/// (`"enabled": true`), and its `writablePaths` / `excludedCommands` are
/// ignored. There is no warning channel at this layer, so the ignored keys
/// are dropped silently.
#[must_use]
pub fn load_default(cwd: &Path) -> Sandbox {
    let mut sb = Sandbox::default();
    if let Ok(home) = std::env::var("HOME")
        && let Ok(text) = std::fs::read_to_string(Path::new(&home).join(".plank/sandbox.json"))
    {
        apply_config(&mut sb, &text, ConfigSource::User);
    }
    if let Ok(text) = std::fs::read_to_string(cwd.join(".plank/sandbox.json")) {
        apply_config(&mut sb, &text, ConfigSource::Project);
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
    fn read_only_commands_are_recognised_conservatively() {
        // Pure readers, alone and in pipelines, with the usual noise.
        assert!(is_read_only_command(
            "cat ~/.plank/settings.json | head -40"
        ));
        assert!(is_read_only_command(
            "ls -la ~/.plank; echo ---; cat ~/.plank/x"
        ));
        assert!(is_read_only_command(
            "grep -n foo ~/.plank/settings.json && wc -l ~/.plank/x"
        ));
        assert!(is_read_only_command("/bin/cat ~/.plank/x"));
        assert!(is_read_only_command("LC_ALL=C sort ~/.plank/x"));
        // Redirects, substitutions and anything unrecognised keep the prompt.
        assert!(!is_read_only_command("cat x > ~/.plank/y"));
        assert!(!is_read_only_command("cat ~/.plank/x 2>/dev/null"));
        assert!(!is_read_only_command("echo $(rm -rf ~/.plank)"));
        assert!(!is_read_only_command("rm -rf ~/.plank"));
        assert!(!is_read_only_command("sed -i s/a/b/ ~/.plank/x"));
        assert!(!is_read_only_command("find ~/.plank -delete"));
        assert!(!is_read_only_command("cat ~/.plank/x | tee ~/.plank/y"));
        assert!(!is_read_only_command("plank --dump-config"));
        assert!(!is_read_only_command("FOO=1"));
        assert!(!is_read_only_command(""));
    }

    #[test]
    fn write_containment_follows_the_profile_roots() {
        let sb = Sandbox {
            enabled: true,
            writable_paths: vec![PathBuf::from("/nonexistent/extra")],
            excluded_commands: Vec::new(),
            plank_home_writable: false,
        };
        let cwd = Path::new("/nonexistent/work");
        // Inside cwd, including a not-yet-existing file and nested dirs.
        assert!(sb.contains_write_target(cwd, Path::new("/nonexistent/work/new.txt")));
        assert!(sb.contains_write_target(cwd, Path::new("/nonexistent/work/a/b/c.txt")));
        // `..` is folded before the comparison.
        assert!(!sb.contains_write_target(cwd, Path::new("/nonexistent/work/../outside")));
        assert!(sb.contains_write_target(cwd, Path::new("/nonexistent/work/a/../b.txt")));
        // Sibling directories that merely share a prefix do not count.
        assert!(!sb.contains_write_target(cwd, Path::new("/nonexistent/workspace/x")));
        // Configured extra roots and the temp dir are writable.
        assert!(sb.contains_write_target(cwd, Path::new("/nonexistent/extra/f")));
        assert!(sb.contains_write_target(cwd, &std::env::temp_dir().join("plank_x")));
        // Everything else is refused, `~/.plank` included, until granted.
        assert!(!sb.contains_write_target(cwd, Path::new("/nonexistent/home/.plank/x")));
        if let Some(home) = plank_home() {
            assert!(!sb.contains_write_target(cwd, &home.join("settings.json")));
            let granted = Sandbox {
                plank_home_writable: true,
                ..sb.clone()
            };
            assert!(granted.contains_write_target(cwd, &home.join("settings.json")));
        }
        // Disabled sandbox: no containment at all.
        let off = Sandbox {
            enabled: false,
            ..sb.clone()
        };
        assert!(off.contains_write_target(cwd, Path::new("/etc/passwd")));
    }

    #[test]
    fn config_merge_appends_lists() {
        let mut sb = Sandbox::default();
        apply_config(
            &mut sb,
            r#"{"enabled": true, "writablePaths": ["/a"], "excludedCommands": ["x*"]}"#,
            ConfigSource::User,
        );
        apply_config(
            &mut sb,
            r#"{"writablePaths": ["/b"], "excludedCommands": ["y"]}"#,
            ConfigSource::User,
        );
        assert!(sb.enabled);
        assert_eq!(sb.writable_paths.len(), 2);
        assert_eq!(sb.excluded_commands, vec!["x*", "y"]);
    }

    #[test]
    fn project_config_can_only_tighten_the_sandbox() {
        let mut sb = Sandbox {
            enabled: true,
            writable_paths: Vec::new(),
            excluded_commands: Vec::new(),
            plank_home_writable: false,
        };
        // Every relaxing key from a project file is ignored.
        apply_config(
            &mut sb,
            r#"{"enabled": false, "writablePaths": ["/"], "excludedCommands": ["*"]}"#,
            ConfigSource::Project,
        );
        assert!(sb.enabled, "a checkout must not switch the sandbox off");
        assert!(sb.writable_paths.is_empty());
        assert!(sb.excluded_commands.is_empty());
        assert!(sb.should_sandbox("rm -rf /"));

        // Turning it on from the project file is tightening, so it is honoured
        // even after the user disabled it.
        let mut off = Sandbox {
            enabled: false,
            writable_paths: Vec::new(),
            excluded_commands: Vec::new(),
            plank_home_writable: false,
        };
        apply_config(&mut off, r#"{"enabled": true}"#, ConfigSource::Project);
        assert!(off.enabled);

        // The user file keeps its full authority.
        apply_config(
            &mut sb,
            r#"{"enabled": false, "excludedCommands": ["git *"]}"#,
            ConfigSource::User,
        );
        assert!(!sb.enabled);
        assert_eq!(sb.excluded_commands, vec!["git *"]);
    }

    #[test]
    fn load_default_ignores_relaxing_keys_in_project_file() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let cwd = std::env::temp_dir().join(format!(
            "plank_sandbox_load_{}_{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(cwd.join(".plank")).unwrap();
        std::fs::write(
            cwd.join(".plank/sandbox.json"),
            r#"{"enabled": false, "writablePaths": ["/etc"], "excludedCommands": ["*"]}"#,
        )
        .unwrap();
        let sb = load_default(&cwd);
        // The project file cannot relax anything: whatever the user file and
        // platform default say, the result is at least that strict.
        assert!(!sb.writable_paths.iter().any(|p| p == Path::new("/etc")));
        assert!(!sb.excluded_commands.iter().any(|c| c == "*"));
        if cfg!(target_os = "macos") && !user_config_disables() {
            assert!(sb.enabled);
        }
        std::fs::remove_dir_all(&cwd).ok();
    }

    /// True when the developer's own `~/.plank/sandbox.json` turns the sandbox
    /// off, in which case `load_default` legitimately returns disabled.
    fn user_config_disables() -> bool {
        let Ok(home) = std::env::var("HOME") else {
            return false;
        };
        let Ok(text) = std::fs::read_to_string(Path::new(&home).join(".plank/sandbox.json")) else {
            return false;
        };
        let mut sb = Sandbox::default();
        apply_config(&mut sb, &text, ConfigSource::User);
        !sb.enabled
    }
}
