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
//! Two families of directory get special treatment beyond that.
//!
//! **Toolchain caches** are writable by default: `~/.cargo/registry`,
//! `~/.cargo/git`, the rustup download dirs, `~/.npm/_cacache`, the Go module
//! cache, `~/.cache` and `~/Library/Caches`. Building the project the model was
//! pointed at is part of what it was told to do, and a build that has to fetch
//! a dependency writes there, not into the project. Only caches: the write
//! costs disk and nothing else.
//!
//! **Protected roots** ([`Protected`]) are withheld and granted on request,
//! because a write there escalates past the project:
//!
//! - `~/.plank` holds the session store, the KV cache, hooks and consent
//!   markers, so a model-chosen command that can rewrite it can rewrite plank's
//!   own behaviour.
//! - `~/.cargo/bin`, `~/.local/bin` and `/usr/local/bin` are on the user's
//!   `PATH`: a binary installed there is one the user later runs. This is why
//!   the cache grant above names `~/.cargo/registry` and `~/.cargo/git` rather
//!   than `~/.cargo`, which would carry `bin` with it.
//!
//! When a sandboxed command names one of them ([`Protected::mentioned_by`]) the
//! bash tool asks the user, and an "always allow" answer records it in
//! [`Sandbox::granted`] for the rest of the session only. Nothing about that
//! grant is written to disk.
//!
//! `sandbox-exec` is deprecated by Apple but remains functional and is what
//! the reference agents use on macOS.

use crate::tools::mcp::{Json, json_parse};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// A directory kept out of the default write roots and granted only on the
/// user's say-so, because a model-chosen write there reaches past the project.
///
/// Each variant is a *family* of directories rather than one path: the prompt,
/// the mention check and the profile all speak in terms of the family, so a
/// user answering once about "binaries on your PATH" is not asked again per
/// directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Protected {
    /// `~/.plank`: the session store, KV cache, hooks and consent markers.
    PlankHome,
    /// `~/.cargo/bin`, `~/.local/bin`, `/usr/local/bin`: directories on the
    /// user's `PATH`, where an installed binary is one the user later runs.
    PathBin,
}

impl Protected {
    /// Every protected family, in prompt order.
    pub const ALL: [Self; 2] = [Self::PlankHome, Self::PathBin];

    /// The directories this family covers. `user_home` is `$HOME` and
    /// `plank_home` the resolved plank home (which is not always under
    /// `$HOME` — see [`crate::home`]); `None` drops the roots that need it.
    #[must_use]
    fn roots(self, user_home: Option<&Path>, plank_home: Option<&Path>) -> Vec<PathBuf> {
        match self {
            Self::PlankHome => plank_home.map(PathBuf::from).into_iter().collect(),
            Self::PathBin => {
                let mut roots = vec![PathBuf::from("/usr/local/bin")];
                if let Some(home) = user_home {
                    roots.push(cargo_home(home).join("bin"));
                    roots.push(home.join(".local/bin"));
                }
                roots
            }
        }
    }

    /// How this family is named in the permission prompt.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::PlankHome => "~/.plank",
            Self::PathBin => {
                "a directory on your PATH (~/.cargo/bin, ~/.local/bin, /usr/local/bin)"
            }
        }
    }

    /// Why plank withholds it, shown under the prompt.
    #[must_use]
    pub fn why(self) -> &'static str {
        match self {
            Self::PlankHome => "it holds plank's sessions, hooks and consent markers",
            Self::PathBin => "a binary installed there is one you later run",
        }
    }

    /// True when `cmd` names this family, in any of the spellings a shell
    /// command plausibly uses.
    ///
    /// This reads the command text because Seatbelt profiles are built before
    /// the command runs, so there is no write to observe yet — which makes it a
    /// heuristic in both directions: a command that only *reads* the directory
    /// still prompts, and one that reaches it through a variable or a symlink
    /// is not caught. It is not the security boundary; the boundary is the
    /// profile, which withholds the write unless the user grants it.
    #[must_use]
    pub fn mentioned_by(self, cmd: &str) -> bool {
        self.mentioned_by_at(cmd, user_home().as_deref(), plank_home().as_deref())
    }

    /// The mention check proper, with both homes passed in rather than read
    /// from the environment, so tests need not mutate what other tests read.
    fn mentioned_by_at(
        self,
        cmd: &str,
        user_home: Option<&Path>,
        plank_home: Option<&Path>,
    ) -> bool {
        let tildes: &[&str] = match self {
            Self::PlankHome => &[".plank"],
            Self::PathBin => &[".cargo/bin", ".local/bin"],
        };
        let mut needles: Vec<String> = Vec::new();
        for t in tildes {
            for prefix in ["~/", "$HOME/", "${HOME}/"] {
                needles.push(format!("{prefix}{t}"));
            }
            if let Some(home) = user_home {
                needles.push(home.join(t).to_string_lossy().into_owned());
            }
        }
        if self == Self::PathBin {
            needles.push("/usr/local/bin".to_string());
            // `cargo install` with no `--root` lands in `$CARGO_HOME/bin`
            // without ever naming it; catch the command instead of the path.
            if cargo_install_without_root(cmd) {
                return true;
            }
        }
        // The plank home is not always under `$HOME` (the shared-home
        // fallback), so match the resolved directory too.
        if self == Self::PlankHome
            && let Some(h) = plank_home
        {
            needles.push(h.to_string_lossy().into_owned());
        }
        needles.iter().any(|n| contains_path_prefix(cmd, n))
    }
}

/// True when `cmd` runs `cargo install` (or `cargo binstall`) with no `--root`
/// redirecting the installation, which means it writes `$CARGO_HOME/bin`.
fn cargo_install_without_root(cmd: &str) -> bool {
    cmd.split(['|', ';', '\n'])
        .flat_map(|s| s.split("&&"))
        .flat_map(|s| s.split("||"))
        .any(|segment| {
            let mut words = segment.split_whitespace().peekable();
            let Some(prog) = words.find(|w| !w.contains('=')) else {
                return false;
            };
            if prog.rsplit('/').next() != Some("cargo") {
                return false;
            }
            let rest: Vec<&str> = words.collect();
            let installs = rest
                .iter()
                .any(|w| *w == "install" || *w == "binstall" || *w == "uninstall");
            installs
                && !rest
                    .iter()
                    .any(|w| *w == "--root" || w.starts_with("--root="))
        })
}

/// `$CARGO_HOME`, or `~/.cargo`.
fn cargo_home(user_home: &Path) -> PathBuf {
    env_dir("CARGO_HOME").unwrap_or_else(|| user_home.join(".cargo"))
}

/// An environment variable read as a non-empty absolute-ish directory path.
fn env_dir(var: &str) -> Option<PathBuf> {
    std::env::var_os(var)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

/// The package-manager and compiler caches a build legitimately writes.
///
/// These are granted by default: a `cargo build` that has to fetch a
/// dependency, an `npm install`, a `go build` — all of them write here and
/// none of them write anything the user runs by name. `~/.cargo` itself is
/// deliberately *not* listed; `bin` lives under it and is
/// [`Protected::PathBin`].
///
/// Paths need not exist: a non-existent `(subpath ...)` in the profile is
/// inert, and listing it unconditionally keeps the root set deterministic.
fn toolchain_cache_roots(user_home: Option<&Path>) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(home) = user_home {
        let cargo = cargo_home(home);
        roots.push(cargo.join("registry"));
        roots.push(cargo.join("git"));
        let rustup = env_dir("RUSTUP_HOME").unwrap_or_else(|| home.join(".rustup"));
        roots.push(rustup.join("downloads"));
        roots.push(rustup.join("tmp"));
        roots.push(home.join(".npm/_cacache"));
        roots.push(home.join(".cache"));
        roots.push(home.join("Library/Caches"));
        let gopath = env_dir("GOPATH").unwrap_or_else(|| home.join("go"));
        roots.push(env_dir("GOMODCACHE").unwrap_or_else(|| gopath.join("pkg/mod")));
        if let Some(gocache) = env_dir("GOCACHE") {
            roots.push(gocache);
        }
    }
    roots
}

/// Sandbox policy for model-initiated bash commands.
#[derive(Debug, Clone)]
pub struct Sandbox {
    /// Master switch; on by default wherever `sandbox-exec` exists (macOS).
    pub enabled: bool,
    /// Extra roots writable in addition to cwd and temp dirs.
    pub writable_paths: Vec<PathBuf>,
    /// `*`-glob patterns for commands that skip the sandbox entirely.
    pub excluded_commands: Vec<String>,
    /// Session-scoped grants for the [`Protected`] families, set by an "always
    /// allow" answer to the bash tool's prompt. In-memory only: a new session
    /// (or a `/resume` of this one) starts denied again.
    pub granted: BTreeSet<Protected>,
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
            granted: BTreeSet::new(),
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
    /// writes, then re-allow writes under cwd, temp roots, /dev, the toolchain
    /// caches, and the configured extra paths. Later rules win in SBPL, so the
    /// allow list punches holes in the write denial.
    ///
    /// The [`Protected`] families are included only where
    /// [`granted`](Self::granted) says so; see
    /// [`profile_granting`](Self::profile_granting) for a single-command grant.
    #[must_use]
    pub fn profile(&self, cwd: &Path) -> String {
        self.profile_granting(cwd, &self.granted)
    }

    /// Same as [`profile`](Self::profile) with an explicit grant set, for a user
    /// who answered "Allow" for one command without granting the rest of the
    /// session.
    #[must_use]
    pub fn profile_granting(&self, cwd: &Path, granted: &BTreeSet<Protected>) -> String {
        self.profile_at(
            cwd,
            user_home().as_deref(),
            plank_home().as_deref(),
            granted,
        )
    }

    /// The profile builder proper, with both homes passed in rather than read
    /// from the environment, so tests need not mutate what other tests read.
    fn profile_at(
        &self,
        cwd: &Path,
        user_home: Option<&Path>,
        plank_home: Option<&Path>,
        granted: &BTreeSet<Protected>,
    ) -> String {
        let mut p = String::from("(version 1)\n(allow default)\n(deny file-write*)\n");
        p.push_str("(allow file-write*\n");
        for real in self.write_roots_at(cwd, user_home, plank_home, granted) {
            p.push_str("  (subpath \"");
            p.push_str(&sbpl_escape(&real.to_string_lossy()));
            p.push_str("\")\n");
        }
        p.push_str(")\n");
        p
    }

    /// The directories a model-initiated write may land in: cwd, the temp
    /// roots, `/dev`, the toolchain caches, the configured extra paths, and
    /// whichever [`Protected`] families have been granted. Symlinks are
    /// resolved where possible, so callers compare against canonical paths.
    /// This is the one list both the Seatbelt profile and the file tools'
    /// containment check ([`Sandbox::contains_write_target`]) are built from,
    /// so the two can never disagree.
    #[must_use]
    pub fn write_roots(&self, cwd: &Path) -> Vec<PathBuf> {
        self.write_roots_at(
            cwd,
            user_home().as_deref(),
            plank_home().as_deref(),
            &self.granted,
        )
    }

    fn write_roots_at(
        &self,
        cwd: &Path,
        user_home: Option<&Path>,
        plank_home: Option<&Path>,
        granted: &BTreeSet<Protected>,
    ) -> Vec<PathBuf> {
        let mut roots: Vec<PathBuf> = vec![
            cwd.to_path_buf(),
            PathBuf::from("/tmp"),
            PathBuf::from("/private/tmp"),
            PathBuf::from("/var/folders"),
            PathBuf::from("/private/var/folders"),
            PathBuf::from("/dev"),
            std::env::temp_dir(),
        ];
        roots.extend(toolchain_cache_roots(user_home));
        roots.extend(self.writable_paths.iter().cloned());
        roots.extend(worktree_git_roots(cwd));
        for p in granted {
            roots.extend(p.roots(user_home, plank_home));
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

/// The git metadata directories a checkout at `cwd` needs to be writable for
/// ordinary git commands to work.
///
/// In a normal clone `.git` sits inside the working directory and is already
/// covered by the cwd root. In a **linked worktree** it does not: `.git` is a
/// pointer file at `<repo>/.git/worktrees/<name>`, and the objects and refs a
/// commit writes live in the shared `<repo>/.git`. Without these roots a
/// sandboxed `git commit` inside a worktree fails on the worktree metadata,
/// which is not a containment win — the model was pointed at that checkout, so
/// its repository is part of what it was told to work on.
///
/// Returns nothing when `cwd` is not in a repository, or when the git
/// directory is already inside `cwd`.
fn worktree_git_roots(cwd: &Path) -> Vec<PathBuf> {
    let Some(root) = crate::worktree::find_git_root(cwd) else {
        return Vec::new();
    };
    let Some(git_dir) = crate::worktree::resolve_git_dir(&root) else {
        return Vec::new();
    };
    let mut roots = vec![git_dir.clone()];
    if let Some(common) = crate::worktree::common_dir(&git_dir) {
        roots.push(common);
    }
    roots
        .into_iter()
        .map(|r| r.canonicalize().unwrap_or(r))
        .filter(|r| !r.starts_with(cwd.canonicalize().as_deref().unwrap_or(cwd)))
        .collect()
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

/// The resolved plank home, or `None` when `HOME` is unset.
#[must_use]
pub fn plank_home() -> Option<PathBuf> {
    crate::home::plank_home_opt()
}

/// `$HOME`, or `None` when it is unset.
#[must_use]
fn user_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// The [`Protected`] families `cmd` names and `granted` does not already
/// cover — the ones the bash tool must ask about before running it.
#[must_use]
pub fn protected_mentions(cmd: &str, granted: &BTreeSet<Protected>) -> Vec<Protected> {
    Protected::ALL
        .into_iter()
        .filter(|p| !granted.contains(p) && p.mentioned_by(cmd))
        .collect()
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
    // `2>/dev/null` and `2>&1` only route stderr; they cannot create a file
    // outside `/dev`, so they are removed before the redirect check rather than
    // counted as a write. Every other `>` still counts.
    let cmd = strip_stderr_redirects(cmd);
    let cmd = cmd.as_str();
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
        if name == "find" {
            // `find` only reads unless asked to act on what it finds.
            if words.any(|w| FIND_MUTATORS.contains(&w)) {
                return false;
            }
            continue;
        }
        if !READERS.contains(&name) {
            return false;
        }
    }
    any
}

/// `find` primaries that run a program or remove a file; a `find` carrying
/// any of them is not read-only.
const FIND_MUTATORS: &[&str] = &[
    "-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls",
];

/// Removes `2>/dev/null` and `2>&1` (with or without a space before the
/// target) from `cmd`, leaving every other redirect in place.
fn strip_stderr_redirects(cmd: &str) -> String {
    let mut out = String::with_capacity(cmd.len());
    let mut rest = cmd;
    while let Some(i) = rest.find("2>") {
        let (before, after) = rest.split_at(i);
        out.push_str(before);
        let tail = after[2..].trim_start();
        if let Some(t) = tail.strip_prefix("&1") {
            rest = t;
        } else if let Some(t) = tail.strip_prefix("/dev/null") {
            rest = t;
        } else {
            // Some other stderr target: keep the `>` so the caller sees it.
            out.push_str("2>");
            rest = &after[2..];
        }
    }
    out.push_str(rest);
    out
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
    fn worktree_git_metadata_is_writable() {
        let tmp = std::env::temp_dir().join(format!("plank-sbwt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let repo = tmp.join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        let git = |args: &[&str], dir: &Path| {
            let ok = std::process::Command::new("git")
                .args(args)
                .current_dir(dir)
                .env("GIT_AUTHOR_NAME", "t")
                .env("GIT_AUTHOR_EMAIL", "t@e")
                .env("GIT_COMMITTER_NAME", "t")
                .env("GIT_COMMITTER_EMAIL", "t@e")
                .output()
                .unwrap()
                .status
                .success();
            assert!(ok, "git {args:?} failed");
        };
        git(&["init", "-q"], &repo);
        std::fs::write(repo.join("f"), "x").unwrap();
        git(&["add", "f"], &repo);
        git(&["commit", "-qm", "init"], &repo);
        let wt = tmp.join("wt");
        git(&["worktree", "add", "-q", wt.to_str().unwrap()], &repo);

        // The per-worktree metadata and the shared object store both sit
        // outside the worktree; a commit writes to both. (The whole tree here
        // is under the temp root, so assert on the extra roots themselves
        // rather than on `contains_write_target`, which temp alone satisfies.)
        let real = |p: &Path| p.canonicalize().unwrap();
        let roots = worktree_git_roots(&wt);
        assert!(
            roots.contains(&real(&repo.join(".git/worktrees/wt"))),
            "{roots:?}"
        );
        assert!(roots.contains(&real(&repo.join(".git"))), "{roots:?}");
        // A normal clone keeps `.git` inside the cwd and needs no extra root.
        assert!(worktree_git_roots(&repo).is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
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
            granted: BTreeSet::new(),
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
            granted: BTreeSet::new(),
        };
        let p = sb.profile(Path::new("/nonexistent/work dir"));
        assert!(p.starts_with("(version 1)\n(allow default)\n(deny file-write*)\n"));
        assert!(p.contains("(subpath \"/nonexistent/work dir\")"));
        assert!(p.contains("(subpath \"/odd\\\"name\")"));
        assert!(p.contains("(subpath \"/dev\")"));
    }

    /// Homes that cannot exist, so `canonicalize` is a no-op and the profile
    /// carries the literal paths.
    const FAKE_HOME: &str = "/nonexistent/home";
    const FAKE_PLANK_HOME: &str = "/nonexistent/home/.plank";

    fn test_sandbox() -> Sandbox {
        Sandbox {
            enabled: true,
            writable_paths: Vec::new(),
            excluded_commands: Vec::new(),
            granted: BTreeSet::new(),
        }
    }

    fn profile_for(sb: &Sandbox, granted: &[Protected]) -> String {
        sb.profile_at(
            Path::new("/nonexistent/work"),
            Some(Path::new(FAKE_HOME)),
            Some(Path::new(FAKE_PLANK_HOME)),
            &granted.iter().copied().collect(),
        )
    }

    #[test]
    fn protected_roots_are_not_writable_until_granted() {
        let mut sb = test_sandbox();
        let plank = format!("(subpath \"{FAKE_PLANK_HOME}\")");
        let cargo_bin = format!("(subpath \"{FAKE_HOME}/.cargo/bin\")");

        // Denied (and the default): no grant reaches the profile at all.
        let none = profile_for(&sb, &[]);
        assert!(!none.contains(&plank));
        assert!(!none.contains(&cargo_bin));
        assert!(!none.contains("(subpath \"/usr/local/bin\")"));

        // A one-command "Allow" punches the hole for that family only...
        let one = profile_for(&sb, &[Protected::PlankHome]);
        assert!(one.contains(&plank));
        assert!(
            !one.contains(&cargo_bin),
            "granting one family grants no other"
        );
        // ...without recording anything on the session.
        assert!(sb.granted.is_empty());

        let bins = profile_for(&sb, &[Protected::PathBin]);
        assert!(bins.contains(&cargo_bin));
        assert!(bins.contains(&format!("(subpath \"{FAKE_HOME}/.local/bin\")")));
        assert!(bins.contains("(subpath \"/usr/local/bin\")"));
        assert!(!bins.contains(&plank));

        // "Always allow" records the family, which is what `profile` reads.
        sb.granted.insert(Protected::PlankHome);
        assert!(profile_for(&sb, &sb.granted.iter().copied().collect::<Vec<_>>()).contains(&plank));
    }

    /// The bug this whole split exists for: a build that has to fetch a crate
    /// writes `~/.cargo/registry`, which must be allowed, while
    /// `cargo install` writes `~/.cargo/bin`, which must not — so the grant
    /// cannot simply name `~/.cargo`.
    #[test]
    fn toolchain_caches_are_writable_but_path_bins_are_not() {
        let sb = test_sandbox();
        let p = profile_for(&sb, &[]);
        for cache in [
            ".cargo/registry",
            ".cargo/git",
            ".rustup/downloads",
            ".rustup/tmp",
            ".npm/_cacache",
            ".cache",
            "Library/Caches",
            "go/pkg/mod",
        ] {
            assert!(
                p.contains(&format!("(subpath \"{FAKE_HOME}/{cache}\")")),
                "{cache} should be writable by default"
            );
        }
        // Never the whole of ~/.cargo, which would carry bin with it.
        assert!(!p.contains(&format!("(subpath \"{FAKE_HOME}/.cargo\")")));
        assert!(!p.contains(&format!("(subpath \"{FAKE_HOME}/.cargo/bin\")")));

        // And the containment check the file tools use agrees with the profile.
        let cwd = Path::new("/nonexistent/work");
        let granted = BTreeSet::new();
        let roots = sb.write_roots_at(cwd, Some(Path::new(FAKE_HOME)), None, &granted);
        let under = |p: &str| roots.iter().any(|r| Path::new(p).starts_with(r));
        assert!(under("/nonexistent/home/.cargo/registry/cache/x"));
        assert!(!under("/nonexistent/home/.cargo/bin/plank-replay"));
        assert!(!under("/nonexistent/home/.cargo/config.toml"));
    }

    #[test]
    fn plank_home_mentions_match_whole_path_components() {
        let m = |cmd: &str| {
            Protected::PlankHome.mentioned_by_at(
                cmd,
                Some(Path::new(FAKE_HOME)),
                Some(Path::new(FAKE_PLANK_HOME)),
            )
        };
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
        // The tilde spellings are recognised even with no homes to expand.
        assert!(Protected::PlankHome.mentioned_by_at("cat ~/.plank/x", None, None));
        assert!(!Protected::PlankHome.mentioned_by_at(
            "touch /nonexistent/home/.plank/x",
            None,
            None
        ));
    }

    #[test]
    fn path_bin_mentions_include_cargo_install() {
        let m = |cmd: &str| {
            Protected::PathBin.mentioned_by_at(
                cmd,
                Some(Path::new(FAKE_HOME)),
                Some(Path::new(FAKE_PLANK_HOME)),
            )
        };
        // Named outright, in every spelling.
        assert!(m("cp x ~/.cargo/bin/"));
        assert!(m("install -m755 x $HOME/.local/bin/x"));
        assert!(m("mv x /usr/local/bin/x"));
        assert!(m("touch /nonexistent/home/.cargo/bin/x"));
        // Named by implication: `cargo install` with no --root writes
        // $CARGO_HOME/bin without the command ever spelling the path.
        assert!(m("cargo install --path ."));
        assert!(m("cd sub && cargo install --locked ripgrep"));
        // ...but not when redirected somewhere writable.
        assert!(!m("cargo install --path . --root /tmp/plank-install"));
        assert!(!m("cargo install --path . --root=/tmp/plank-install"));
        // Ordinary builds are not install.
        assert!(!m("cargo build --release"));
        assert!(!m("cargo test --lib"));
        assert!(!m("cat ~/.cargo/config.toml"));
        assert!(!m("ls ~/.cargo/registry"));
    }

    /// The prompt loop asks about exactly the families a command names, and
    /// stops asking about one already granted for the session.
    #[test]
    fn protected_mentions_skips_granted_families() {
        let mut granted = BTreeSet::new();
        assert_eq!(protected_mentions("cargo build", &granted), Vec::new());
        assert_eq!(
            protected_mentions("cargo install --path .", &granted),
            vec![Protected::PathBin]
        );
        granted.insert(Protected::PathBin);
        assert_eq!(
            protected_mentions("cargo install --path .", &granted),
            Vec::new()
        );
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
        // Stderr-only redirects cannot create a file; every other `>` can.
        assert!(is_read_only_command("cat ~/.plank/x 2>/dev/null"));
        assert!(is_read_only_command("ls ~/.plank 2>&1 | head"));
        assert!(is_read_only_command("ls ~/.plank 2> /dev/null"));
        assert!(!is_read_only_command("cat ~/.plank/x 2>~/.plank/err"));
        assert!(!is_read_only_command("cat ~/.plank/x >/dev/null"));
        assert!(!is_read_only_command("cat ~/.plank/x &>/dev/null"));
        // `find` reads unless it acts on its matches.
        assert!(is_read_only_command("find ~/.plank -name '*.gguf'"));
        assert!(is_read_only_command("find ~/.plank -type f | wc -l"));
        assert!(!is_read_only_command("find ~/.plank -name x -exec rm {} +"));
        assert!(!is_read_only_command("find ~/.plank -fprint ~/.plank/list"));
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
            granted: BTreeSet::new(),
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
                granted: [Protected::PlankHome].into_iter().collect(),
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
            granted: BTreeSet::new(),
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
            granted: BTreeSet::new(),
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
