// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Installing Claude Code plugins: fetching one from a git repository, a
//! marketplace repository or a `.tar.gz`, checking it against what plank
//! actually implements, and copying it where the plugin loader will find it.
//!
//! Kept apart from [`crate::plugins`] because the two answer different
//! questions. `plugins` is about what a plugin *is* once it is on disk, and it
//! already understands the Claude Code spellings. This module is only about
//! getting a third-party tree onto disk safely, which is where all the
//! network, subprocess and trust decisions live.

use std::path::{Path, PathBuf};

use crate::tools::mcp::{Json, json_parse};

/// Where a plugin is being fetched from, chosen by the shape of the argument.
///
/// A two-variant enum rather than one per user-facing form: a marketplace repo
/// is indistinguishable from a plain one until it has been cloned and its
/// `.claude-plugin/marketplace.json` looked for, so that distinction belongs
/// after acquisition, not here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// A repository to `git clone --depth 1`.
    Git {
        /// The clone URL, already expanded from `owner/repo` if it was short.
        url: String,
    },
    /// A `.tar.gz` to download and extract.
    Archive {
        /// The archive URL, checked against the remote policy at fetch time.
        url: String,
    },
}

/// Classifies `arg` into the acquisition path it names.
///
/// The rules, in order: anything ending `.tar.gz` is an archive; any other URL
/// with a scheme is a git repository; and a bare `owner/repo` — exactly two
/// segments, no dot in the first — expands to GitHub. The dot test is what
/// keeps `example.com/p` from being silently rewritten into a `github.com`
/// URL, which would fetch from a server the user never named.
///
/// # Errors
/// Returns a message when `arg` is neither a URL nor `owner/repo`.
pub fn parse_source(arg: &str) -> Result<Source, String> {
    let arg = arg.trim();
    if arg.is_empty() {
        return Err("usage: /install-claude-plugin <url|owner/repo> [plugin-name]".to_string());
    }
    let has_scheme = arg.contains("://");
    if has_scheme {
        if arg.ends_with(".tar.gz") {
            return Ok(Source::Archive {
                url: arg.to_string(),
            });
        }
        return Ok(Source::Git {
            url: arg.to_string(),
        });
    }
    let parts: Vec<&str> = arg.split('/').collect();
    if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() && !parts[0].contains('.') {
        return Ok(Source::Git {
            url: format!("https://github.com/{}/{}", parts[0], parts[1]),
        });
    }
    Err(format!(
        "'{arg}' is neither a URL nor an owner/repo shorthand"
    ))
}

/// The directory `/install-claude-plugin` copies into, and one of the roots
/// the loader auto-scans.
///
/// Separate from `~/.plank/plugins/dev/` on purpose: a directory under
/// `claude/` is known to have arrived from someone else's repository and to be
/// unedited by hand, which is exactly the distinction a user needs when
/// deciding what to trust or remove.
#[must_use]
pub fn install_dir(home: &Path) -> PathBuf {
    home.join(".plank").join("plugins").join("claude")
}

/// Substitutes `${CLAUDE_PLUGIN_ROOT}` with `dest` in the two files whose
/// contents become subprocess command lines, returning whether anything
/// changed.
///
/// Claude Code hook and MCP commands reference the variable to find files
/// inside their own plugin. Plank's hook runner execs `/bin/sh` with no
/// injected environment, so left alone the variable expands to empty and the
/// command silently misfires.
///
/// This is done at install time rather than by injecting the variable at exec
/// time because [`crate::plugins`] flattens every source's hooks into one list
/// with no per-hook provenance: injecting would mean threading an owning root
/// through the hook types and the merge order, a change to load-bearing code
/// for a problem the boundary can solve. The cost is that the installed tree
/// differs from upstream and stops working if it is moved, which the install
/// output says out loud.
///
/// Only `hooks/hooks.json` and `.mcp.json` are touched. Skills, agents and
/// commands are model-facing text, and rewriting a path into them would change
/// what the model reads rather than what a subprocess runs.
///
/// # Errors
/// Returns a message when a file exists but cannot be read or written.
pub fn rewrite_plugin_root(dest: &Path) -> Result<bool, String> {
    let root = dest.display().to_string();
    let mut changed = false;
    for rel in ["hooks/hooks.json", ".mcp.json"] {
        let path = dest.join(rel);
        if !path.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        // Order does not matter here: `str::replace` matches the literal
        // pattern `$CLAUDE_PLUGIN_ROOT`, and inside `${CLAUDE_PLUGIN_ROOT}`
        // the character after `$` is `{`, not `C`, so the bare pattern can
        // never match there. Both spellings are replaced either way.
        let out = text
            .replace("${CLAUDE_PLUGIN_ROOT}", &root)
            .replace("$CLAUDE_PLUGIN_ROOT", &root);
        if out != text {
            std::fs::write(&path, &out)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
            changed = true;
        }
    }
    Ok(changed)
}

/// Event names in `root/hooks/hooks.json` that plank does not implement.
///
/// Claude Code defines `Notification` and `SubagentStop`, which plank has no
/// equivalent for, and a config may also carry a typo or an event from a
/// newer release. A hook under any of those names would be installed and then
/// never fire, which is the silent failure this check exists to turn into a
/// loud one at install time.
///
/// A missing file is not a problem — most plugins contribute no hooks — but an
/// unparseable one is, because a file that cannot be read cannot be cleared.
///
/// # Errors
/// Returns a message when `hooks/hooks.json` exists but cannot be read or does
/// not parse as a JSON object.
pub fn unsupported_hook_events(root: &Path) -> Result<Vec<String>, String> {
    let path = root.join("hooks").join("hooks.json");
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let Some(Json::Obj(members)) = json_parse(&text) else {
        return Err(format!(
            "{} does not parse as a JSON object",
            path.display()
        ));
    };
    Ok(members
        .iter()
        .map(|(k, _)| k.clone())
        .filter(|k| !crate::hooks::KNOWN_EVENTS.contains(&k.as_str()))
        .collect())
}

/// The directory inside `staged` holding the plugin to install.
///
/// A repository is one of two things and cannot be told apart before it is
/// fetched: a single plugin, whose root carries `.claude-plugin/plugin.json`,
/// or a marketplace, whose root carries `.claude-plugin/marketplace.json` and
/// which needs `want` to say which of its plugins is meant. When `want` is
/// missing or matches nothing, the error lists the names on offer, so the
/// second attempt is informed rather than guessed.
///
/// A marketplace entry's `source` is resolved under `staged` and then checked
/// to still be under it: `"../../etc"` in a fetched file must not become a
/// directory outside the tree this function was handed.
///
/// # Errors
/// Returns a message when the tree is neither a plugin nor a marketplace, when
/// a marketplace name is missing or unknown, or when an entry's `source`
/// escapes the tree or holds no plugin.
pub fn resolve_in_tree(staged: &Path, want: Option<&str>) -> Result<PathBuf, String> {
    let market = staged.join(".claude-plugin").join("marketplace.json");
    if market.is_file() {
        return resolve_marketplace(staged, &market, want);
    }
    if staged.join(".claude-plugin").join("plugin.json").is_file() {
        return Ok(staged.to_path_buf());
    }
    Err(
        "this is not a Claude Code plugin: no .claude-plugin/plugin.json and no \
         .claude-plugin/marketplace.json at its root"
            .to_string(),
    )
}

/// Picks `want` out of a marketplace manifest. Split out to keep
/// [`resolve_in_tree`]'s two cases readable side by side.
fn resolve_marketplace(
    staged: &Path,
    market: &Path,
    want: Option<&str>,
) -> Result<PathBuf, String> {
    let text = std::fs::read_to_string(market)
        .map_err(|e| format!("cannot read {}: {e}", market.display()))?;
    let Some(root) = json_parse(&text) else {
        return Err(format!("{} does not parse as JSON", market.display()));
    };
    let Some(Json::Arr(entries)) = root.get("plugins") else {
        return Err(format!("{} has no \"plugins\" array", market.display()));
    };
    let names: Vec<String> = entries
        .iter()
        .filter_map(|e| match e.get("name") {
            Some(Json::Str(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    let Some(want) = want else {
        return Err(format!(
            "this is a marketplace of several plugins; name the one you want: {}",
            names.join(", ")
        ));
    };
    let Some(entry) = entries
        .iter()
        .find(|e| matches!(e.get("name"), Some(Json::Str(s)) if s == want))
    else {
        return Err(format!(
            "this marketplace has no plugin '{want}'; it offers: {}",
            names.join(", ")
        ));
    };
    let source = match entry.get("source") {
        Some(Json::Str(s)) => s.clone(),
        // The documented default: a plugin lives in a directory named after
        // itself.
        _ => format!("./{want}"),
    };
    let dir = staged.join(source.trim_start_matches("./"));
    // String prefix on the *cleaned* paths, not on the raw join: `..` in the
    // manifest is the case this exists to catch, and it only shows up after
    // the path is resolved. A `source` that fails to canonicalize is refused
    // outright rather than falling back to the unresolved join: the fallback
    // still starts with `staged`'s own path text, so it would pass the
    // `starts_with` check below no matter how many `../` components `source`
    // held. A `source` that does not resolve to a real directory is not a
    // plugin we can install either way, so refusing it loses nothing.
    let canon_staged = staged
        .canonicalize()
        .unwrap_or_else(|_| staged.to_path_buf());
    let Ok(canon_dir) = dir.canonicalize() else {
        return Err(format!(
            "marketplace entry '{want}' points outside the repository, at {source}"
        ));
    };
    if !canon_dir.starts_with(&canon_staged) {
        return Err(format!(
            "marketplace entry '{want}' points outside the repository, at {source}"
        ));
    }
    if !canon_dir
        .join(".claude-plugin")
        .join("plugin.json")
        .is_file()
    {
        return Err(format!(
            "marketplace entry '{want}' has no .claude-plugin/plugin.json at {source}"
        ));
    }
    Ok(canon_dir)
}

/// What an install produced, for the command to report.
#[derive(Debug, Clone)]
pub struct Installed {
    /// The plugin's name, from its manifest.
    pub name: String,
    /// Where it was copied to.
    pub dest: PathBuf,
    /// Whether a `${CLAUDE_PLUGIN_ROOT}` reference was rewritten, which the
    /// user is told because it means the directory can no longer be moved.
    pub rewrote_plugin_root: bool,
    /// Hook events installed under `--force` that will never fire.
    pub skipped_hook_events: Vec<String>,
}

/// Validates the staged tree and copies it into `~/.plank/plugins/claude/`.
///
/// The order matters: everything that can refuse happens before anything is
/// written under `home`, so a refusal never leaves a partial install behind.
///
/// `force` waives only the unimplemented-hook refusal, in which case those
/// hooks are installed and simply never fire. The structural refusals — no
/// manifest, a symlink, an existing install — are not waivable, because none
/// of them describes a plugin the user could still want as it is.
///
/// # Errors
/// Returns a message when the tree is not a Claude Code plugin, contains a
/// symlink, names a hook event plank does not implement (without `force`), is
/// already installed, or cannot be copied.
pub fn install_staged(
    staged: &Path,
    want: Option<&str>,
    home: &Path,
    force: bool,
) -> Result<Installed, String> {
    let root = resolve_in_tree(staged, want)?;
    // Scanned before the copy, not after: `copy_tree` follows links, so a
    // `wasm -> ~/.ssh` entry would put a private key inside a plugin directory
    // and only then be noticed.
    crate::plugins::reject_symlinks(&root)?;
    let skipped = unsupported_hook_events(&root)?;
    if !skipped.is_empty() && !force {
        return Err(format!(
            "this plugin hooks events plank does not implement: {}\nthose hooks would never \
             fire; pass --force to install it anyway",
            skipped.join(", ")
        ));
    }
    let name = plugin_name(&root)?;
    let dest = install_dir(home).join(&name);
    if dest.exists() {
        return Err(format!(
            "'{name}' is already installed at {}; remove it first with /plugins remove {name}",
            dest.display()
        ));
    }
    crate::plugins::copy_tree(&root, &dest)
        .map_err(|e| format!("cannot install into {}: {e}", dest.display()))?;
    let rewrote = rewrite_plugin_root(&dest)?;
    Ok(Installed {
        name,
        dest,
        rewrote_plugin_root: rewrote,
        skipped_hook_events: skipped,
    })
}

/// Fetches the plugin `arg` names, validates it, and installs it.
///
/// The one entry point the slash command calls. Everything it does happens in
/// a staging directory outside every root [`crate::plugins::load_in`] scans,
/// which is removed on every exit path: a half-fetched or half-validated tree
/// left behind would otherwise be found by the next scan as a plugin nobody
/// asked for and never checked — including one that was about to be refused
/// for a symlink or an unimplemented hook event.
///
/// # Errors
/// Returns a message when the argument names nothing fetchable, the fetch
/// fails, or the tree does not pass [`install_staged`]'s checks.
pub fn install(
    arg: &str,
    want: Option<&str>,
    home: &Path,
    force: bool,
) -> Result<Installed, String> {
    let staging = staging_dir(home)?;
    let result = fetch(arg, &staging).and_then(|tree| install_staged(&tree, want, home, force));
    let _ = std::fs::remove_dir_all(&staging);
    result
}

/// A private, empty staging directory. Removed by [`install`] on every path.
///
/// Deliberately not under [`install_dir`]: that directory is one of
/// `load_in`'s scan roots (its subdirectories are loaded as installed
/// plugins), so a tree extracted there is a tree the next start could load
/// before it was ever validated, if plank were killed between extraction and
/// checking, or the closing `remove_dir_all` failed. `~/.plank/.claude-staging`
/// sits one level up, beside `plugins/` rather than inside it, so it is never
/// a subdirectory of anything `load_in` scans — `dev/`, `install_dir`'s
/// `plugins/claude/`, and the project's own `.plank/plugins/` are all rooted
/// under a project or `plugins/`, never above it.
fn staging_dir(home: &Path) -> Result<PathBuf, String> {
    let base = home.join(".plank").join(".claude-staging");
    // Fresh every time: reusing it would mix a previous failed fetch into a new
    // one, and the plugin root is found by looking at the tree.
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).map_err(|e| format!("cannot create {}: {e}", base.display()))?;
    Ok(base)
}

/// Puts the named plugin's tree inside `staging` and returns its root.
fn fetch(arg: &str, staging: &Path) -> Result<PathBuf, String> {
    // A local directory is checked before the URL forms: `git clone` takes a
    // path as happily as a URL, and it is how a plugin is tried before it is
    // published. Not every directory is a repository, though — a plain tree
    // assembled by hand or by a test has no `.git` and no bare-repo layout —
    // so a clone failure falls back to a plain copy rather than refusing.
    if Path::new(arg).is_dir() {
        if let Ok(dest) = clone(arg, staging) {
            return Ok(dest);
        }
        // Scanned before the copy, not after: `copy_tree` follows links, so a
        // `key -> ~/.ssh/id_rsa` entry would already be a plain file holding
        // the private key's bytes by the time a check ran on the destination,
        // and the refusal would find nothing left to refuse. This is the same
        // inversion `install_staged` guards against for the staged tree.
        crate::plugins::reject_symlinks(Path::new(arg))?;
        let dest = staging.join("tree");
        crate::plugins::copy_tree(Path::new(arg), &dest)
            .map_err(|e| format!("cannot copy {arg}: {e}"))?;
        return Ok(dest);
    }
    match parse_source(arg)? {
        Source::Git { url } => clone(&url, staging),
        Source::Archive { url } => {
            crate::plugins::fetch_archive(&url, staging)?;
            // Every conventionally produced tarball nests its contents under
            // one top-level directory — GitHub's codeload archives use
            // `repo-main/`, and a plain `tar czf x.tar.gz somedir` yields
            // `somedir/` — so the manifest is one level down more often than
            // it is at the root. The descent lives here, in the archive
            // branch only, rather than in `resolve_in_tree`: that function is
            // also reached by the git-clone and local-directory paths, whose
            // behaviour is already reviewed and correct, and it has no way to
            // tell "the caller is an archive" from "the caller is a repo that
            // happens to have no manifest at its root" — conflating the two
            // would make a git clone of a directory-of-directories resolve
            // somewhere it never has before.
            find_claude_root(staging).ok_or_else(|| {
                "this is not a Claude Code plugin: no .claude-plugin/plugin.json and no \
                 .claude-plugin/marketplace.json at its root or one level in"
                    .to_string()
            })
        }
    }
}

/// Whether `dir` itself carries a Claude Code manifest — either spelling,
/// since a tarball of a marketplace repository must resolve here too.
fn is_claude_manifest_root(dir: &Path) -> bool {
    dir.join(".claude-plugin").join("plugin.json").is_file()
        || dir
            .join(".claude-plugin")
            .join("marketplace.json")
            .is_file()
}

/// Finds the directory holding a Claude Code manifest, at `dir` itself or one
/// level in.
///
/// Mirrors [`crate::plugins`]'s private `find_plugin_root`, which solves the
/// identical problem for plank's own archive format: check the given
/// directory, else collect subdirectories that qualify, sort them, and take
/// the first. Kept as a separate copy rather than shared code because the two
/// check different manifest spellings and `plugins`'s version is private —
/// this module changes nothing outside itself.
fn find_claude_root(dir: &Path) -> Option<PathBuf> {
    if is_claude_manifest_root(dir) {
        return Some(dir.to_path_buf());
    }
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir() && is_claude_manifest_root(p))
        .collect();
    // Sorted so an archive holding two qualifying directories resolves the
    // same way twice rather than depending on directory order.
    candidates.sort();
    candidates.into_iter().next()
}

/// Shallow-clones `url` into `staging/repo` and drops its history.
fn clone(url: &str, staging: &Path) -> Result<PathBuf, String> {
    let dest = staging.join("repo");
    let status = std::process::Command::new("git")
        .arg("clone")
        .arg("--quiet")
        .arg("--depth")
        .arg("1")
        .arg("--") // Separator ensures url is treated as a path, not an option
        .arg(url)
        .arg(&dest)
        .status()
        .map_err(|e| format!("cannot run git: {e}"))?;
    if !status.success() {
        return Err(format!("cannot clone {url}"));
    }
    // The history is not part of the plugin, and it is a tree of files that
    // would otherwise be scanned for symlinks and copied into the user's home.
    let _ = std::fs::remove_dir_all(dest.join(".git"));
    crate::plugins::reject_symlinks(&dest)?;
    Ok(dest)
}

/// The plugin's name, from its manifest's `name` field, falling back to the
/// directory name. The name becomes a path component under `install_dir`, so
/// values that would not create a fresh subdirectory are refused rather than
/// sanitized. This includes path separators, `..`, `.` (which resolves to the
/// parent), and whitespace-only strings. A plugin calling itself `../x` or `.`
/// is not a naming style to accommodate.
fn plugin_name(root: &Path) -> Result<String, String> {
    let manifest = root.join(".claude-plugin").join("plugin.json");
    let text = std::fs::read_to_string(&manifest)
        .map_err(|e| format!("cannot read {}: {e}", manifest.display()))?;
    let from_manifest = match json_parse(&text).as_ref().and_then(|j| j.get("name")) {
        Some(Json::Str(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    };
    let name = from_manifest
        .or_else(|| root.file_name().map(|n| n.to_string_lossy().into_owned()))
        .ok_or_else(|| "this plugin has no name".to_string())?;
    if name.is_empty()
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || name == "."
        || name.trim().is_empty()
    {
        return Err(format!("'{name}' is not a usable plugin name"));
    }
    Ok(name)
}

/// The whole `/install-claude-plugin` command: parses its argument line, runs
/// the install, and renders the outcome — success or refusal — as the text
/// both front ends print.
///
/// Rendering lives here rather than in `ui.rs` because the two front ends must
/// say the same thing, and the plain and TUI paths in `ui.rs` are already two
/// places a message can drift between.
///
/// `home` is `None` when there is no home directory, which is the one case
/// where there is nowhere to install to at all.
#[must_use]
pub fn render_install(arg: &str, home: Option<&Path>) -> String {
    // `concat!`, not a `\`-continued literal: continuation strips the next
    // line's leading whitespace, which is exactly the kind of invisible trap
    // this project's text must not carry.
    const USAGE: &str = concat!(
        "usage: /install-claude-plugin <url|owner/repo> [plugin-name] [--force]\n",
        "a url may be a git repository, a marketplace repository, or a .tar.gz\n"
    );
    let mut force = false;
    let mut words: Vec<&str> = Vec::new();
    for word in arg.split_whitespace() {
        if word == "--force" {
            force = true;
        } else {
            words.push(word);
        }
    }
    let Some(target) = words.first() else {
        return USAGE.to_string();
    };
    let Some(home) = home else {
        return "no HOME, so there is nowhere to install a plugin\n".to_string();
    };
    match install(target, words.get(1).copied(), home, force) {
        Ok(out) => render_installed(&out),
        Err(e) => format!("{e}\n"),
    }
}

/// The success half of [`render_install`], kept separate so the happy path
/// reads as one list of what the user just granted.
fn render_installed(out: &Installed) -> String {
    use std::fmt::Write as _;
    let mut s = format!("installed '{}' to {}\n", out.name, out.dest.display());
    let parts = contributions(&out.dest);
    if parts.is_empty() {
        s.push_str("it contributes nothing plank recognizes\n");
    } else {
        let _ = writeln!(s, "it contributes: {}", parts.join(", "));
    }
    if !out.skipped_hook_events.is_empty() {
        let _ = writeln!(
            s,
            "warning: it hooks {}, which plank does not implement; those hooks never fire",
            out.skipped_hook_events.join(", ")
        );
    }
    if out.rewrote_plugin_root {
        // `concat!`, not a `\`-continued literal: continuation strips the next
        // line's leading whitespace, which is exactly the kind of invisible
        // trap this project's text must not carry.
        s.push_str(concat!(
            "CLAUDE_PLUGIN_ROOT was resolved to that path in its hooks and MCP config, so ",
            "moving the directory breaks them\n"
        ));
    }
    s.push_str("it is loaded on the next start\n");
    s
}

/// Human-readable labels for what an installed tree actually carries.
///
/// Read from the destination rather than from the manifest: a manifest can
/// claim anything, and what matters to the user is which files are now on
/// their disk.
fn contributions(dest: &Path) -> Vec<&'static str> {
    [
        ("skills", "skills"),
        ("commands", "commands"),
        ("agents", "agents"),
        ("hooks/hooks.json", "hooks"),
        (".mcp.json", "MCP servers"),
        ("settings.json", "settings"),
    ]
    .into_iter()
    .filter(|(rel, _)| dest.join(rel).exists())
    .map(|(_, label)| label)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Writes `body` to `root/rel`, creating parent directories.
    fn write(root: &Path, rel: &str, body: &str) {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().expect("has a parent")).expect("mkdir");
        std::fs::write(&path, body).expect("write");
    }

    /// A fresh empty directory under the system temp dir, named for the test.
    ///
    /// An atomic counter appended to the directory name ensures uniqueness even
    /// if a tag is duplicated across tests, preventing one test from removing
    /// another's directory mid-run.
    fn tmpdir(tag: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "plank-claudeplugin-{tag}-{}-{seq}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    #[test]
    fn install_dir_is_under_the_plank_home() {
        let dir = install_dir(Path::new("/tmp/h"));
        assert_eq!(dir, Path::new("/tmp/h/.plank/plugins/claude"));
    }

    #[test]
    fn a_tarball_url_is_an_archive() {
        let src = parse_source("https://example.com/p.tar.gz").expect("archive");
        assert!(
            matches!(src, Source::Archive { ref url } if url == "https://example.com/p.tar.gz")
        );
    }

    #[test]
    fn a_repo_url_is_a_git_clone() {
        let src = parse_source("https://github.com/owner/repo").expect("git");
        assert!(matches!(src, Source::Git { ref url } if url == "https://github.com/owner/repo"));
    }

    #[test]
    fn owner_repo_shorthand_expands_to_github() {
        let src = parse_source("anthropics/claude-plugins").expect("git");
        assert!(
            matches!(src, Source::Git { ref url } if url == "https://github.com/anthropics/claude-plugins")
        );
    }

    #[test]
    fn a_hostname_is_not_shorthand() {
        // A dot in the first segment means a host, not a GitHub owner: silently
        // rewriting `example.com/p` to a github.com URL would fetch from a
        // different server than the one the user named.
        let err = parse_source("example.com/p").expect_err("rejected");
        assert!(err.contains("owner/repo"), "{err}");
    }

    #[test]
    fn a_local_path_is_rejected() {
        let err = parse_source("/Users/me/plugins/demo").expect_err("rejected");
        assert!(err.contains("owner/repo"), "{err}");
    }

    #[test]
    fn a_three_segment_path_is_not_shorthand() {
        let err = parse_source("a/b/c").expect_err("rejected");
        assert!(err.contains("owner/repo"), "{err}");
    }

    #[test]
    fn an_empty_argument_is_rejected() {
        assert!(parse_source("").is_err());
    }

    #[test]
    fn no_hooks_file_is_no_unsupported_events() {
        let root = tmpdir("hooks-absent");
        assert_eq!(
            unsupported_hook_events(&root).expect("ok"),
            Vec::<String>::new()
        );
    }

    #[test]
    fn known_events_are_supported() {
        let root = tmpdir("hooks-known");
        write(
            &root,
            "hooks/hooks.json",
            r#"{"PreToolUse":[{"matcher":"Bash","hooks":[{"type":"command","command":"true"}]}],
                "SessionStart":[{"hooks":[{"type":"command","command":"true"}]}]}"#,
        );
        assert_eq!(
            unsupported_hook_events(&root).expect("ok"),
            Vec::<String>::new()
        );
    }

    #[test]
    fn claude_only_events_are_reported() {
        let root = tmpdir("hooks-unknown");
        write(
            &root,
            "hooks/hooks.json",
            r#"{"PreToolUse":[],"SubagentStop":[],"Notification":[]}"#,
        );
        let mut got = unsupported_hook_events(&root).expect("ok");
        got.sort();
        assert_eq!(
            got,
            vec!["Notification".to_string(), "SubagentStop".to_string()]
        );
    }

    #[test]
    fn a_malformed_hooks_file_is_an_error() {
        let root = tmpdir("hooks-malformed");
        write(&root, "hooks/hooks.json", "{not json");
        let err = unsupported_hook_events(&root).expect_err("rejected");
        assert!(err.contains("hooks.json"), "{err}");
    }

    #[test]
    fn plugin_root_is_substituted_in_hooks_and_mcp() {
        let dest = tmpdir("rewrite-both");
        write(
            &dest,
            "hooks/hooks.json",
            r#"{"PreToolUse":[{"hooks":[{"type":"command","command":"${CLAUDE_PLUGIN_ROOT}/bin/g"}]}]}"#,
        );
        write(
            &dest,
            ".mcp.json",
            r#"{"mcpServers":{"s":{"command":"$CLAUDE_PLUGIN_ROOT/bin/s"}}}"#,
        );
        assert!(rewrite_plugin_root(&dest).expect("ok"));
        let root = dest.display().to_string();
        let hooks = std::fs::read_to_string(dest.join("hooks/hooks.json")).expect("read");
        assert!(hooks.contains(&format!("{root}/bin/g")), "{hooks}");
        assert!(!hooks.contains("CLAUDE_PLUGIN_ROOT"), "{hooks}");
        let mcp = std::fs::read_to_string(dest.join(".mcp.json")).expect("read");
        assert!(mcp.contains(&format!("{root}/bin/s")), "{mcp}");
        assert!(!mcp.contains("CLAUDE_PLUGIN_ROOT"), "{mcp}");
    }

    #[test]
    fn nothing_to_substitute_reports_no_change() {
        let dest = tmpdir("rewrite-none");
        write(&dest, ".mcp.json", r#"{"mcpServers":{}}"#);
        assert!(!rewrite_plugin_root(&dest).expect("ok"));
    }

    #[test]
    fn other_files_are_left_alone() {
        // A skill body is model-facing text; substituting a path into it would
        // change what the model reads, not what a subprocess runs.
        let dest = tmpdir("rewrite-scope");
        write(&dest, "skills/s/SKILL.md", "run ${CLAUDE_PLUGIN_ROOT}/x\n");
        assert!(!rewrite_plugin_root(&dest).expect("ok"));
        let body = std::fs::read_to_string(dest.join("skills/s/SKILL.md")).expect("read");
        assert!(body.contains("${CLAUDE_PLUGIN_ROOT}"), "{body}");
    }

    #[test]
    fn a_plain_repo_resolves_to_itself() {
        let staged = tmpdir("resolve-plain");
        write(&staged, ".claude-plugin/plugin.json", r#"{"name":"demo"}"#);
        assert_eq!(resolve_in_tree(&staged, None).expect("ok"), staged);
    }

    #[test]
    fn a_marketplace_resolves_the_named_plugin() {
        let staged = tmpdir("resolve-named");
        write(
            &staged,
            ".claude-plugin/marketplace.json",
            r#"{"plugins":[{"name":"alpha"},{"name":"beta","source":"./pkgs/beta"}]}"#,
        );
        write(
            &staged,
            "alpha/.claude-plugin/plugin.json",
            r#"{"name":"alpha"}"#,
        );
        write(
            &staged,
            "pkgs/beta/.claude-plugin/plugin.json",
            r#"{"name":"beta"}"#,
        );
        assert_eq!(
            resolve_in_tree(&staged, Some("alpha")).expect("ok"),
            staged.join("alpha").canonicalize().expect("canon")
        );
        assert_eq!(
            resolve_in_tree(&staged, Some("beta")).expect("ok"),
            staged.join("pkgs/beta").canonicalize().expect("canon")
        );
    }

    #[test]
    fn a_marketplace_without_a_name_lists_what_it_offers() {
        let staged = tmpdir("resolve-unnamed");
        write(
            &staged,
            ".claude-plugin/marketplace.json",
            r#"{"plugins":[{"name":"alpha"},{"name":"beta"}]}"#,
        );
        let err = resolve_in_tree(&staged, None).expect_err("rejected");
        assert!(err.contains("alpha"), "{err}");
        assert!(err.contains("beta"), "{err}");
    }

    #[test]
    fn an_unknown_marketplace_name_lists_what_it_offers() {
        let staged = tmpdir("resolve-wrong");
        write(
            &staged,
            ".claude-plugin/marketplace.json",
            r#"{"plugins":[{"name":"alpha"}]}"#,
        );
        let err = resolve_in_tree(&staged, Some("nope")).expect_err("rejected");
        assert!(err.contains("alpha"), "{err}");
    }

    #[test]
    fn a_marketplace_source_may_not_escape_the_tree() {
        let staged = tmpdir("resolve-escape");
        write(
            &staged,
            ".claude-plugin/marketplace.json",
            r#"{"plugins":[{"name":"evil","source":"../../etc"}]}"#,
        );
        let err = resolve_in_tree(&staged, Some("evil")).expect_err("rejected");
        assert!(err.contains("outside"), "{err}");
    }

    /// A fresh empty directory under the crate's own `target/` directory,
    /// rather than the system temp dir.
    ///
    /// `std::env::temp_dir()` resolves through `/tmp` -> `/private/tmp` (and
    /// similarly for `/var`) on macOS, so a path built under it canonicalizes
    /// to a different string than the one the join produced even when nothing
    /// escaped the tree — which would mask the exact "candidate does not
    /// canonicalize" bypass this module's regression test exists to pin.
    /// `target/` sits inside the repository checkout, which has no such
    /// symlink hop.
    fn tmpdir_in_repo(tag: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("tmp")
            .join(format!(
                "plank-claudeplugin-{tag}-{}-{seq}",
                std::process::id()
            ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    #[test]
    fn a_marketplace_source_escaping_to_a_nonexistent_path_is_rejected() {
        // Pins the branch that a bare `canonicalize().unwrap_or_else(|_| ..)`
        // fallback missed: unlike `../../etc` (which exists), this target does
        // not, so the old code fell back to the unresolved join, which still
        // starts with `staged`'s own path text and slipped past `starts_with`.
        let staged = tmpdir_in_repo("resolve-escape-missing");
        write(
            &staged,
            ".claude-plugin/marketplace.json",
            r#"{"plugins":[{"name":"evil","source":"../../etc/nope-nonexistent"}]}"#,
        );
        let err = resolve_in_tree(&staged, Some("evil")).expect_err("rejected");
        assert!(err.contains("outside"), "{err}");
    }

    #[test]
    fn an_absolute_marketplace_source_may_not_escape_the_tree() {
        let staged = tmpdir_in_repo("resolve-escape-absolute");
        write(
            &staged,
            ".claude-plugin/marketplace.json",
            r#"{"plugins":[{"name":"evil","source":"/etc"}]}"#,
        );
        let err = resolve_in_tree(&staged, Some("evil")).expect_err("rejected");
        assert!(err.contains("outside"), "{err}");
    }

    #[test]
    fn a_tree_with_no_manifest_is_not_a_claude_plugin() {
        let staged = tmpdir("resolve-nomanifest");
        write(&staged, "README.md", "hi\n");
        let err = resolve_in_tree(&staged, None).expect_err("rejected");
        assert!(err.contains(".claude-plugin/plugin.json"), "{err}");
    }

    /// A minimal valid staged Claude plugin named `name`.
    fn staged_plugin(tag: &str, name: &str) -> PathBuf {
        let dir = tmpdir(tag);
        write(
            &dir,
            ".claude-plugin/plugin.json",
            &format!(r#"{{"name":"{name}","description":"d"}}"#),
        );
        dir
    }

    #[test]
    fn a_valid_tree_installs_under_the_claude_root() {
        let staged = staged_plugin("install-ok", "demo");
        write(&staged, "commands/note.md", "hi\n");
        let home = tmpdir("install-ok-home");
        let out = install_staged(&staged, None, &home, false).expect("installs");
        assert_eq!(out.name, "demo");
        assert_eq!(out.dest, install_dir(&home).join("demo"));
        assert!(out.dest.join(".claude-plugin/plugin.json").is_file());
        assert!(out.dest.join("commands/note.md").is_file());
        assert!(out.skipped_hook_events.is_empty());
        assert!(!out.rewrote_plugin_root);
    }

    #[test]
    fn an_unimplemented_hook_event_refuses() {
        let staged = staged_plugin("install-hook", "demo");
        write(&staged, "hooks/hooks.json", r#"{"SubagentStop":[]}"#);
        let home = tmpdir("install-hook-home");
        let err = install_staged(&staged, None, &home, false).expect_err("refused");
        assert!(err.contains("SubagentStop"), "{err}");
        assert!(err.contains("--force"), "{err}");
        assert!(
            !install_dir(&home).join("demo").exists(),
            "nothing installed"
        );
    }

    #[test]
    fn force_installs_past_an_unimplemented_hook_event() {
        let staged = staged_plugin("install-force", "demo");
        write(&staged, "hooks/hooks.json", r#"{"SubagentStop":[]}"#);
        let home = tmpdir("install-force-home");
        let out = install_staged(&staged, None, &home, true).expect("installs");
        assert_eq!(out.skipped_hook_events, vec!["SubagentStop".to_string()]);
        assert!(out.dest.join("hooks/hooks.json").is_file());
    }

    #[test]
    fn a_symlink_refuses_even_with_force() {
        let staged = staged_plugin("install-symlink", "demo");
        std::os::unix::fs::symlink("/etc/hosts", staged.join("link")).expect("symlink");
        let home = tmpdir("install-symlink-home");
        let err = install_staged(&staged, None, &home, true).expect_err("refused");
        assert!(err.contains("symlink"), "{err}");
        assert!(
            !install_dir(&home).join("demo").exists(),
            "nothing installed"
        );
    }

    #[test]
    fn a_local_directory_with_a_symlink_is_refused_and_the_target_never_copied() {
        // Exercises `fetch`'s local-directory fallback (a plain, non-git tree),
        // not `install_staged` directly: `copy_tree` follows symlinks and
        // materializes the target's bytes as a plain file, so a check run on
        // the copy instead of the source would find nothing to refuse — the
        // secret would already be sitting in the user's home.
        let secret_dir = tmpdir("install-local-symlink-secret");
        let secret = secret_dir.join("id_rsa");
        std::fs::write(&secret, "-----BEGIN PRIVATE KEY-----\nsekrit\n").expect("write secret");
        let staged = staged_plugin("install-local-symlink", "demo");
        std::os::unix::fs::symlink(&secret, staged.join("key")).expect("symlink");
        let home = tmpdir("install-local-symlink-home");
        let err = install(staged.to_str().expect("utf8"), None, &home, false).expect_err("refused");
        assert!(err.contains("symlink"), "{err}");
        assert!(
            !install_dir(&home).join("demo").exists(),
            "nothing installed"
        );
        // Walk every file plank could have written and make sure none of them
        // carry the secret's contents.
        assert!(
            !walk_for_secret(&install_dir(&home), "sekrit"),
            "the linked file's contents must not reach the install root"
        );
    }

    /// True if any non-symlink file under `dir` contains `needle`.
    fn walk_for_secret(dir: &Path, needle: &str) -> bool {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return false;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_symlink() {
                continue;
            }
            if path.is_dir() {
                if walk_for_secret(&path, needle) {
                    return true;
                }
            } else if let Ok(text) = std::fs::read_to_string(&path)
                && text.contains(needle)
            {
                return true;
            }
        }
        false
    }

    #[test]
    fn an_existing_install_refuses_even_with_force() {
        let staged = staged_plugin("install-dup", "demo");
        let home = tmpdir("install-dup-home");
        write(
            &install_dir(&home),
            "demo/.claude-plugin/plugin.json",
            r#"{"name":"demo","description":"the one already there"}"#,
        );
        let err = install_staged(&staged, None, &home, true).expect_err("refused");
        assert!(err.contains("already installed"), "{err}");
        let kept =
            std::fs::read_to_string(install_dir(&home).join("demo/.claude-plugin/plugin.json"))
                .expect("read");
        assert!(kept.contains("already there"), "not overwritten");
    }

    #[test]
    fn a_tree_with_no_claude_manifest_refuses() {
        let staged = tmpdir("install-nomanifest");
        write(&staged, ".plank-plugin/plugin.json", r#"{"name":"native"}"#);
        let home = tmpdir("install-nomanifest-home");
        let err = install_staged(&staged, None, &home, true).expect_err("refused");
        assert!(err.contains(".claude-plugin/plugin.json"), "{err}");
    }

    #[test]
    fn plugin_root_is_rewritten_to_the_final_destination() {
        let staged = staged_plugin("install-rewrite", "demo");
        write(
            &staged,
            ".mcp.json",
            r#"{"mcpServers":{"s":{"command":"${CLAUDE_PLUGIN_ROOT}/bin/s"}}}"#,
        );
        let home = tmpdir("install-rewrite-home");
        let out = install_staged(&staged, None, &home, false).expect("installs");
        assert!(out.rewrote_plugin_root);
        let mcp = std::fs::read_to_string(out.dest.join(".mcp.json")).expect("read");
        assert!(mcp.contains(&out.dest.display().to_string()), "{mcp}");
    }

    #[test]
    fn an_installed_plugin_is_found_by_the_loader() {
        let staged = staged_plugin("install-loads", "demo");
        let home = tmpdir("install-loads-home");
        let out = install_staged(&staged, None, &home, false).expect("installs");
        let set = crate::plugins::load_in(Some(&home), &tmpdir("install-loads-cwd"), &[]);
        let found = set
            .plugins
            .iter()
            .find(|p| p.name == "demo")
            .expect("loaded");
        assert_eq!(found.origin, crate::plugins::Origin::UserClaude);
        assert!(out.dest.is_dir());
    }

    #[test]
    fn plugin_named_dot_is_refused() {
        let staged = staged_plugin("install-dot-name", ".");
        let home = tmpdir("install-dot-name-home");
        let err = install_staged(&staged, None, &home, false).expect_err("refused");
        assert!(err.contains("not a usable plugin name"), "{err}");
        assert!(
            !install_dir(&home).exists(),
            "claude root should not be created"
        );
    }

    #[test]
    fn plugin_named_dot_is_refused_with_force() {
        let staged = staged_plugin("install-dot-force", ".");
        let home = tmpdir("install-dot-force-home");
        let err = install_staged(&staged, None, &home, true).expect_err("refused");
        assert!(err.contains("not a usable plugin name"), "{err}");
    }

    #[test]
    fn plugin_with_whitespace_only_name_is_refused() {
        let staged = staged_plugin("install-whitespace", "   ");
        let home = tmpdir("install-whitespace-home");
        let err = install_staged(&staged, None, &home, false).expect_err("refused");
        assert!(err.contains("not a usable plugin name"), "{err}");
    }

    #[test]
    fn a_git_source_clones_and_installs() {
        // A local bare repository, so the test needs no network.
        let root = tmpdir("git-install");
        let work = root.join("work");
        write(&work, ".claude-plugin/plugin.json", r#"{"name":"gitdemo"}"#);
        let bare = root.join("bare.git");
        for args in [
            vec!["init", "-q", "-b", "main", work.to_str().expect("utf8")],
            vec!["-C", work.to_str().expect("utf8"), "add", "-A"],
            vec![
                "-C",
                work.to_str().expect("utf8"),
                "-c",
                "user.email=t@example.com",
                "-c",
                "user.name=t",
                "commit",
                "-qm",
                "init",
            ],
            vec![
                "clone",
                "-q",
                "--bare",
                work.to_str().expect("utf8"),
                bare.to_str().expect("utf8"),
            ],
        ] {
            let ok = std::process::Command::new("git")
                .args(&args)
                .status()
                .expect("git runs")
                .success();
            assert!(ok, "git {args:?} failed");
        }
        let home = tmpdir("git-install-home");
        let out = install(bare.to_str().expect("utf8"), None, &home, false).expect("installs");
        assert_eq!(out.name, "gitdemo");
        assert!(out.dest.join(".claude-plugin/plugin.json").is_file());
        // The clone's own history is not part of the plugin.
        assert!(!out.dest.join(".git").exists(), ".git is not installed");
    }

    #[test]
    fn a_failed_install_leaves_no_staging_behind() {
        let root = tmpdir("staging-clean");
        let work = root.join("work");
        write(&work, "README.md", "not a plugin\n");
        let ok = std::process::Command::new("git")
            .args(["init", "-q", "-b", "main", work.to_str().expect("utf8")])
            .status()
            .expect("git runs")
            .success();
        assert!(ok);
        for args in [
            vec!["-C", work.to_str().expect("utf8"), "add", "-A"],
            vec![
                "-C",
                work.to_str().expect("utf8"),
                "-c",
                "user.email=t@example.com",
                "-c",
                "user.name=t",
                "commit",
                "-qm",
                "init",
            ],
        ] {
            assert!(
                std::process::Command::new("git")
                    .args(&args)
                    .status()
                    .expect("git runs")
                    .success()
            );
        }
        let home = tmpdir("staging-clean-home");
        let err = install(work.to_str().expect("utf8"), None, &home, false).expect_err("refused");
        assert!(err.contains(".claude-plugin"), "{err}");
        assert!(
            !home.join(".plank").join(".claude-staging").exists(),
            "staging removed on the failure path"
        );
    }

    #[test]
    fn an_archive_over_plain_http_to_a_public_host_is_refused() {
        let home = tmpdir("archive-tls");
        let err = install("http://plugins.example.com/x.tar.gz", None, &home, false)
            .expect_err("refused");
        assert!(
            err.to_lowercase().contains("https") || err.to_lowercase().contains("tls"),
            "{err}"
        );
    }

    #[test]
    fn an_archive_nested_under_a_top_level_directory_installs() {
        use std::io::{Read as _, Write as _};

        let root = tmpdir("archive-nested");
        let staged = root.join("stage").join("repo-main").join(".claude-plugin");
        std::fs::create_dir_all(&staged).expect("mkdir");
        std::fs::write(staged.join("plugin.json"), r#"{"name":"demo"}"#).expect("write");
        write(
            &root.join("stage").join("repo-main"),
            "commands/note.md",
            "hi\n",
        );
        let tarball = root.join("x.tar.gz");
        let ok = std::process::Command::new("tar")
            .arg("-czf")
            .arg(&tarball)
            .arg("-C")
            .arg(root.join("stage"))
            .arg("repo-main")
            .status()
            .expect("tar runs")
            .success();
        assert!(ok, "could not build the fixture tarball");
        let bytes = std::fs::read(&tarball).expect("read tarball");

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().expect("accept");
            let mut buf = [0u8; 4096];
            let _ = sock.read(&mut buf);
            let _ = sock.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/gzip\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    bytes.len()
                )
                .as_bytes(),
            );
            let _ = sock.write_all(&bytes);
            let _ = sock.flush();
        });

        let home = tmpdir("archive-nested-home");
        let out = install(
            &format!("http://127.0.0.1:{port}/x.tar.gz"),
            None,
            &home,
            false,
        )
        .expect("installs the plugin nested one level down");
        server.join().expect("server thread");

        assert_eq!(out.name, "demo");
        assert!(out.dest.join(".claude-plugin/plugin.json").is_file());
        assert!(out.dest.join("commands/note.md").is_file());
    }

    #[test]
    fn an_archive_with_the_plugin_at_the_top_level_still_installs() {
        // The nesting descent must not break the flat case: a hand-rolled
        // archive that puts the manifest straight at the root is exactly what
        // `resolve_in_tree` already handled before this fix, and it must keep
        // working unchanged.
        use std::io::{Read as _, Write as _};

        let root = tmpdir("archive-flat");
        write(
            &root.join("stage"),
            ".claude-plugin/plugin.json",
            r#"{"name":"flatdemo"}"#,
        );
        let tarball = root.join("x.tar.gz");
        let ok = std::process::Command::new("tar")
            .arg("-czf")
            .arg(&tarball)
            .arg("-C")
            .arg(root.join("stage"))
            .arg(".")
            .status()
            .expect("tar runs")
            .success();
        assert!(ok, "could not build the fixture tarball");
        let bytes = std::fs::read(&tarball).expect("read tarball");

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().expect("accept");
            let mut buf = [0u8; 4096];
            let _ = sock.read(&mut buf);
            let _ = sock.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/gzip\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    bytes.len()
                )
                .as_bytes(),
            );
            let _ = sock.write_all(&bytes);
            let _ = sock.flush();
        });

        let home = tmpdir("archive-flat-home");
        let out = install(
            &format!("http://127.0.0.1:{port}/x.tar.gz"),
            None,
            &home,
            false,
        )
        .expect("installs the plugin at the archive root");
        server.join().expect("server thread");

        assert_eq!(out.name, "flatdemo");
        assert!(out.dest.join(".claude-plugin/plugin.json").is_file());
    }

    #[test]
    fn a_failed_install_leaves_nothing_loadable() {
        // Staging must sit outside every root `plugins::load_in` scans: if it
        // did not, a tree that was extracted but refused installation (here,
        // an unimplemented hook event without `--force`) would still be found
        // by the very next scan, under whatever name its manifest claims.
        let staged = staged_plugin("staging-scope", "unwanted");
        write(&staged, "hooks/hooks.json", r#"{"SubagentStop":[]}"#);
        let home = tmpdir("staging-scope-home");
        let err = install(staged.to_str().expect("utf8"), None, &home, false).expect_err("refused");
        assert!(err.contains("SubagentStop"), "{err}");
        let set = crate::plugins::load_in(Some(&home), &tmpdir("staging-scope-cwd"), &[]);
        assert!(
            set.plugins.iter().all(|p| p.name != "unwanted"),
            "a refused install must not be loadable: {:?}",
            set.plugins.iter().map(|p| &p.name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn render_reports_what_was_installed() {
        let staged = staged_plugin("render-ok", "demo");
        write(&staged, "commands/note.md", "hi\n");
        write(&staged, "skills/s/SKILL.md", "s\n");
        let home = tmpdir("render-ok-home");
        let out = render_install(staged.to_str().expect("utf8"), Some(&home));
        assert!(out.contains("installed 'demo'"), "{out}");
        assert!(out.contains("next start"), "{out}");
        assert!(out.contains("commands"), "{out}");
        assert!(out.contains("skills"), "{out}");
    }

    #[test]
    fn render_reports_a_refusal_without_panicking() {
        let staged = tmpdir("render-bad");
        write(&staged, "README.md", "x\n");
        let home = tmpdir("render-bad-home");
        let out = render_install(staged.to_str().expect("utf8"), Some(&home));
        assert!(out.contains(".claude-plugin"), "{out}");
    }

    #[test]
    fn render_with_no_argument_prints_usage() {
        let home = tmpdir("render-usage-home");
        let out = render_install("", Some(&home));
        assert!(out.contains("usage: /install-claude-plugin"), "{out}");
    }

    #[test]
    fn render_parses_a_name_and_force_in_any_order() {
        let staged = staged_plugin("render-force", "demo");
        write(&staged, "hooks/hooks.json", r#"{"SubagentStop":[]}"#);
        let home = tmpdir("render-force-home");
        let line = format!("--force {}", staged.to_str().expect("utf8"));
        let out = render_install(&line, Some(&home));
        assert!(out.contains("installed 'demo'"), "{out}");
        assert!(out.contains("SubagentStop"), "{out}");
        assert!(out.contains("never fire"), "{out}");
    }

    #[test]
    fn render_says_when_the_plugin_root_was_rewritten() {
        let staged = staged_plugin("render-rewrite", "demo");
        write(
            &staged,
            ".mcp.json",
            r#"{"mcpServers":{"s":{"command":"${CLAUDE_PLUGIN_ROOT}/s"}}}"#,
        );
        let home = tmpdir("render-rewrite-home");
        let out = render_install(staged.to_str().expect("utf8"), Some(&home));
        assert!(out.contains("CLAUDE_PLUGIN_ROOT"), "{out}");
        assert!(out.contains("moving"), "{out}");
    }

    #[test]
    fn render_without_a_home_says_so() {
        let out = render_install("owner/repo", None);
        assert!(out.contains("HOME"), "{out}");
    }
}
