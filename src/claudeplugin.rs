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

use crate::tools::mcp::{Json, json_parse, json_write};

/// Unwraps Claude Code's nested `hooks/hooks.json` shape, if present.
///
/// plank's own hook runner (`src/hooks.rs`) reads event names from the TOP
/// level of the file. Claude Code plugins instead nest every event under one
/// outer `"hooks"` key, e.g. `{"hooks": {"SessionStart": [...]}}`. Left alone,
/// plank would see a single top-level key named `"hooks"`, refuse the plugin
/// as naming an event plank does not implement, and — worse, under `--force`
/// — install a file whose hooks can never fire, because `parse_config` would
/// never find `SessionStart` at the top level either.
///
/// Detection looks only at the outer shape: an object whose `"hooks"` member
/// is itself an object. That object's own contents — matcher groups, each
/// with its own inner `"hooks"` array of `{type, command, shell, async,
/// timeout}` — are plank's native format one level down and must survive
/// untouched; this function only ever removes the single outer wrapper, never
/// recurses into the members it returns.
///
/// Returns `None` when the value is not shaped this way, so a file that
/// already puts events at the top level (or an event named `hooks`, though
/// none of plank's or Claude Code's do) is reported as needing no change and
/// can be left byte-for-byte as it was written.
fn unwrap_nested_hooks(v: &Json) -> Option<Json> {
    let Json::Obj(members) = v else {
        return None;
    };
    members
        .iter()
        .find(|(k, val)| k == "hooks" && matches!(val, Json::Obj(_)))
        .map(|(_, inner)| inner.clone())
}

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
    /// A GitHub `/tree/<ref>/<path>` or `/blob/<ref>/<path>` URL: what a user
    /// gets by browsing to a subdirectory on github.com and copying the
    /// address bar, rather than the repository's own URL.
    ///
    /// This has to carry the ref and the subpath alongside the clone URL,
    /// because none of the three is recoverable from the others: the clone
    /// URL alone loses which commit the browser was looking at, and the
    /// subpath is only meaningful once the repository is actually on disk.
    GitSubpath {
        /// The repository's clone URL, with the `/tree/`-or-`/blob/` suffix
        /// stripped off.
        url: String,
        /// The branch, tag, or commit named after `tree`/`blob`.
        refname: String,
        /// The path within the repository the URL pointed at. Not
        /// necessarily the plugin root itself — see [`resolve_subpath`].
        subpath: String,
    },
    /// `owner/repo:folder`: one folder of a GitHub repository's default
    /// branch, the short form `--profile` takes for a repository that holds
    /// several profiles side by side.
    GitFolder {
        /// The clone URL, expanded from `owner/repo`.
        url: String,
        /// The folder within the repository, relative, with no `..`.
        folder: String,
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
        if let Some((url, refname, subpath)) = parse_github_tree_url(arg) {
            return Ok(Source::GitSubpath {
                url,
                refname,
                subpath,
            });
        }
        return Ok(Source::Git {
            url: arg.to_string(),
        });
    }
    if let Some((repo, folder)) = arg.split_once(':') {
        let url = github_shorthand(repo)
            .ok_or_else(|| format!("'{repo}' is not an owner/repo shorthand"))?;
        let folder = folder.trim_matches('/');
        if folder.is_empty()
            || Path::new(folder)
                .components()
                .any(|c| !matches!(c, std::path::Component::Normal(_)))
        {
            return Err(format!(
                "'{folder}' is not a folder inside the repository (use owner/repo:folder)"
            ));
        }
        return Ok(Source::GitFolder {
            url,
            folder: folder.to_string(),
        });
    }
    if let Some(url) = github_shorthand(arg) {
        return Ok(Source::Git { url });
    }
    Err(format!(
        "'{arg}' is neither a URL nor an owner/repo shorthand"
    ))
}

/// `owner/repo` as a GitHub clone URL: exactly two segments, no dot in the
/// first, which is what keeps `example.com/p` from being silently rewritten
/// into a `github.com` URL.
fn github_shorthand(arg: &str) -> Option<String> {
    let parts: Vec<&str> = arg.split('/').collect();
    (parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() && !parts[0].contains('.'))
        .then(|| format!("https://github.com/{}/{}", parts[0], parts[1]))
}

/// Recognizes a GitHub `/tree/<ref>/<path>` or `/blob/<ref>/<path>` URL,
/// returning the plain repository clone URL, the ref, and the subpath.
///
/// This exists because pasting a GitHub URL is the natural thing to do after
/// browsing to a plugin's subdirectory in a browser, and github.com's own
/// address bar puts exactly this shape there — `tree` for a directory,
/// `blob` for a file. Sent straight to `git clone` (the pre-existing
/// behaviour), it fails outright: neither form is a repository URL, and the
/// path after the ref is not something `git clone` understands at all.
fn parse_github_tree_url(arg: &str) -> Option<(String, String, String)> {
    let rest = arg
        .strip_prefix("https://github.com/")
        .or_else(|| arg.strip_prefix("http://github.com/"))?;
    let parts: Vec<&str> = rest.split('/').filter(|s| !s.is_empty()).collect();
    if parts.len() < 5 {
        return None;
    }
    let (owner, repo, kind, refname) = (parts[0], parts[1], parts[2], parts[3]);
    if kind != "tree" && kind != "blob" {
        return None;
    }
    let subpath = parts[4..].join("/");
    if subpath.is_empty() {
        return None;
    }
    Some((
        format!("https://github.com/{owner}/{repo}"),
        refname.to_string(),
        subpath,
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
    crate::home::plank_home_in(home)
        .join("plugins")
        .join("claude")
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
/// `hooks/hooks.json` gets one more treatment the other file does not:
/// [`unwrap_nested_hooks`] flattens Claude Code's nested shape into plank's
/// own, so the two rewrites compose on a file (like obra/superpowers') that
/// needs both. This is done here, at install time, rather than by teaching
/// `src/hooks.rs` to accept both shapes, for the same reason the variable
/// substitution is: it keeps the hook runner — shared with every other hook
/// source — ignorant of a quirk that belongs to one of its inputs. The
/// returned bool reports only the variable substitution, matching its doc
/// below; a flatten with no `${CLAUDE_PLUGIN_ROOT}` in the file still writes
/// the flattened content but leaves that bool false.
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
        // The substitution runs on the parsed tree's string values, never on
        // the JSON text: a root containing `"` or `\` spliced into the text
        // would leave an unparseable file, and the hooks would vanish without
        // a word. Rewriting the value and re-serializing lets the writer
        // escape it. A file that does not parse is left exactly as it was;
        // `unsupported_hook_events` (run before this, in `install_staged`) is
        // what refuses a malformed hooks file, and a malformed `.mcp.json`
        // is the MCP loader's to report.
        let Some(mut parsed) = json_parse(&text) else {
            continue;
        };
        let var_changed = substitute_plugin_root(&mut parsed, &root);
        if rel == "hooks/hooks.json"
            && let Some(unwrapped) = unwrap_nested_hooks(&parsed)
        {
            parsed = unwrapped;
        }
        let mut out = String::new();
        json_write(&mut out, &parsed);
        if out != text {
            std::fs::write(&path, &out)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        }
        changed |= var_changed;
    }
    Ok(changed)
}

/// Replaces `${CLAUDE_PLUGIN_ROOT}` and `$CLAUDE_PLUGIN_ROOT` with `root` in
/// every string value of `v`, recursively, returning whether anything changed.
///
/// Order does not matter: `str::replace` matches the literal pattern
/// `$CLAUDE_PLUGIN_ROOT`, and inside `${CLAUDE_PLUGIN_ROOT}` the character
/// after `$` is `{`, not `C`, so the bare pattern can never match there.
fn substitute_plugin_root(v: &mut Json, root: &str) -> bool {
    match v {
        Json::Str(s) => {
            let out = s
                .replace("${CLAUDE_PLUGIN_ROOT}", root)
                .replace("$CLAUDE_PLUGIN_ROOT", root);
            if out == *s {
                false
            } else {
                *s = out;
                true
            }
        }
        Json::Arr(items) => items
            .iter_mut()
            .fold(false, |acc, item| substitute_plugin_root(item, root) | acc),
        Json::Obj(members) => members.iter_mut().fold(false, |acc, (_, val)| {
            substitute_plugin_root(val, root) | acc
        }),
        Json::Null | Json::Bool(_) | Json::Num(_) => false,
    }
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
    let Some(parsed) = json_parse(&text) else {
        return Err(format!(
            "{} does not parse as a JSON object",
            path.display()
        ));
    };
    // Checked against the same unwrapped shape `rewrite_plugin_root` writes,
    // so a nested file's refusal (or acceptance) names the events it will
    // actually end up installed under, not the single outer "hooks" key.
    let effective = unwrap_nested_hooks(&parsed).unwrap_or(parsed);
    let Json::Obj(members) = effective else {
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
    if crate::plugins::manifest_path(staged).is_some() {
        return Ok(staged.to_path_buf());
    }
    Err("this is not a plank or Claude Code plugin: no .plank-plugin/plugin.json, no .claude-plugin/plugin.json and no .claude-plugin/marketplace.json at its root".to_string())
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
    if crate::plugins::manifest_path(&canon_dir).is_none() {
        return Err(format!(
            "marketplace entry '{want}' has no .plank-plugin/plugin.json or .claude-plugin/plugin.json at {source}"
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

/// Validates a staged tree and copies it into `dest_root`.
///
/// `dest_root` is the install root, not the plugin directory: the plugin's own
/// name is appended here, from its manifest. `/install-claude-plugin` passes
/// [`install_dir`]; `/install-profile` passes [`crate::profiles::dir`].
///
/// The order matters: everything that can refuse happens before anything is
/// written under `dest_root`, so a refusal never leaves a partial install
/// behind.
///
/// `force` waives only the unimplemented-hook refusal, in which case those
/// hooks are installed and simply never fire. The structural refusals — no
/// manifest, a symlink, an existing install — are not waivable, because none
/// of them describes a plugin the user could still want as it is.
///
/// # Errors
/// Returns a message when the tree holds an escaping symlink, hooks an event
/// plank does not implement (unless `force`), has no usable name, or names
/// something already installed.
pub fn install_staged(
    staged: &Path,
    want: Option<&str>,
    dest_root: &Path,
    force: bool,
) -> Result<Installed, String> {
    let root = resolve_in_tree(staged, want)?;
    // Scanned before the copy, not after: `copy_tree` follows links, so a
    // `wasm -> ~/.ssh` entry would put a private key inside a plugin directory
    // and only then be noticed. See `reject_escaping_symlinks`'s doc comment
    // for why a contained symlink (Claude plugins like superpowers ship
    // `AGENTS.md -> CLAUDE.md`) is fine while an escaping target (or a
    // symlinked directory) is refused.
    crate::plugins::reject_escaping_symlinks(&root)?;
    let skipped = unsupported_hook_events(&root)?;
    if !skipped.is_empty() && !force {
        return Err(format!(
            "this plugin hooks events plank does not implement: {}\nthose hooks would never \
             fire; pass --force to install it anyway",
            skipped.join(", ")
        ));
    }
    let name = plugin_name(&root)?;
    let dest = dest_root.join(&name);
    if dest.exists() {
        return Err(format!(
            "'{name}' is already installed at {}; remove it first",
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

/// Refuses a staged tree that is not an installable profile, then installs it
/// into the profiles root.
///
/// Split from [`install_profile`] so the gate is testable without fetching:
/// every refusal below is reachable from a directory on disk.
///
/// The prompt is validated *here*, at install time, rather than left to
/// startup. `main.rs` treats an unreadable or blank profile prompt as fatal —
/// deliberately, because silently running as plain plank when the user asked
/// for HAL is worse than not running — and a fatal launch is a bad place to
/// discover a bad download.
///
/// # Errors
/// Returns a message when the tree has no manifest, its manifest declares no
/// `profile` block, its `systemPrompt` is missing, unreadable or blank, or
/// [`install_staged`]'s own checks refuse it.
pub fn install_profile_staged(
    staged: &Path,
    want: Option<&str>,
    home: &Path,
    force: bool,
) -> Result<Installed, String> {
    let root = resolve_in_tree(staged, want)?;
    let name = plugin_name(&root)?;
    // `plugin_name`'s grammar is looser than `--profile` will later demand:
    // once installed, this same name (the directory `install_staged` creates
    // under `crate::profiles::dir`) becomes the plugin's namespace prefix in
    // `plugins::splice_profile`, which gates it with `plugins::valid_name`.
    // Refusing here, rather than letting `plugin_name` wave it through, is
    // what keeps `/install-profile` from reporting success for a name that
    // `--profile` would then refuse to load — the refusal belongs where the
    // user can still pick a different name.
    if !crate::plugins::valid_name(&name) {
        return Err(format!(
            "'{name}' is not usable as a profile name: it would make an ambiguous or unroutable \
             namespace prefix (no ':', '/', '\\', or '__' allowed); rename it in plugin.json"
        ));
    }
    let manifest = crate::plugins::manifest_path(&root)
        .ok_or_else(|| format!("no plugin.json in {}", root.display()))?;
    let text = std::fs::read_to_string(&manifest)
        .map_err(|e| format!("cannot read {}: {e}", manifest.display()))?;
    let spec = crate::profile::parse(&text, &root).ok_or_else(|| {
        format!(
            "plugin '{name}' declares no profile block; install it with /install-claude-plugin instead"
        )
    })?;
    let prompt = std::fs::read_to_string(&spec.system_prompt).map_err(|e| {
        format!(
            "profile '{name}': cannot read {}: {e}",
            spec.system_prompt.display()
        )
    })?;
    if prompt.trim().is_empty() {
        return Err(format!(
            "profile '{name}': {} is empty",
            spec.system_prompt.display()
        ));
    }
    install_staged(staged, want, &crate::profiles::dir(home), force)
}

/// Fetches the profile `arg` names, validates it, and installs it.
///
/// The `/install-profile` counterpart of [`install`], sharing its staging
/// discipline: everything happens in a directory outside every scan root, and
/// that directory is removed on every exit path.
///
/// # Errors
/// Returns a message when the argument names nothing fetchable, the fetch
/// fails, or the tree does not pass [`install_profile_staged`]'s checks.
pub fn install_profile(
    arg: &str,
    want: Option<&str>,
    home: &Path,
    force: bool,
) -> Result<Installed, String> {
    let staging = staging_dir(home)?;
    let result =
        fetch(arg, &staging).and_then(|tree| install_profile_staged(&tree, want, home, force));
    let _ = std::fs::remove_dir_all(&staging);
    // Recorded so `--profile <this same source>` launches the installed copy
    // instead of offering to install it again.
    if let Ok(installed) = &result {
        crate::profiles::record_source(&installed.dest, &crate::profiles::normalize_source(arg));
    }
    result
}

/// The URLs that serve the manifest of the profile `arg` names on GitHub, in
/// the order to try them, or `None` for a source that is not a GitHub
/// repository (or folder of one) and so cannot be checked without a clone.
///
/// The contents API comes first because `raw.githubusercontent.com` is a CDN
/// that can keep serving an old copy for minutes after a push, and not the
/// same old copy to every client: a version bump just released would read as
/// nothing new. The raw URLs follow as the fallback for when the API's
/// unauthenticated hourly limit is spent. Both name the default branch, as a
/// clone without `--branch` does (`HEAD` on the raw host, no `ref` on the
/// API), and both spellings of the manifest are tried, plank's first.
#[must_use]
pub fn raw_manifest_urls(arg: &str) -> Option<Vec<String>> {
    let (url, folder) = match parse_source(arg).ok()? {
        Source::GitFolder { url, folder } => (url, folder),
        Source::Git { url } => (url, String::new()),
        _ => return None,
    };
    let repo = url.strip_prefix("https://github.com/")?;
    let prefix = if folder.is_empty() {
        String::new()
    } else {
        format!("{folder}/")
    };
    let mut out = Vec::new();
    for manifest in [".plank-plugin/plugin.json", ".claude-plugin/plugin.json"] {
        out.push(format!(
            "https://api.github.com/repos/{repo}/contents/{prefix}{manifest}"
        ));
    }
    for manifest in [".plank-plugin/plugin.json", ".claude-plugin/plugin.json"] {
        out.push(format!(
            "https://raw.githubusercontent.com/{repo}/HEAD/{prefix}{manifest}"
        ));
    }
    Some(out)
}

/// The `version` the source `arg` currently offers, read without fetching
/// the profile: a local directory's manifest, or one small request for a
/// GitHub manifest, bounded by [`VERSION_CHECK_TIMEOUT_SECS`]. `None` when it
/// cannot be told, which the caller treats as nothing newer.
#[must_use]
pub fn source_version(arg: &str) -> Option<String> {
    if Path::new(arg).is_dir() {
        return crate::profiles::version_of(Path::new(arg));
    }
    raw_manifest_urls(arg)?
        .iter()
        .find_map(|url| fetch_text(url))
        .and_then(|text| crate::profiles::manifest_version(&text))
}

/// The bound on [`source_version`]'s request: launching must not hang on a
/// slow network just to learn that nothing changed.
#[cfg(not(test))]
const VERSION_CHECK_TIMEOUT_SECS: u64 = 3;

#[cfg(not(test))]
fn fetch_text(url: &str) -> Option<String> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(
            VERSION_CHECK_TIMEOUT_SECS,
        )))
        .build()
        .new_agent();
    let mut resp = agent
        .get(url)
        .header("User-Agent", concat!("plank/", env!("CARGO_PKG_VERSION")))
        .header("Accept", "application/vnd.github.raw+json")
        .call()
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.body_mut().read_to_string().ok()
}

/// Test builds never touch the network.
#[cfg(test)]
fn fetch_text(_url: &str) -> Option<String> {
    None
}

/// Replaces the installed profile `name` with a fresh install from `arg`.
///
/// The old copy is moved aside first, into `<profiles>/.replacing/`, rather
/// than the new one being installed elsewhere and renamed in: an install
/// rewrites `${CLAUDE_PLUGIN_ROOT}` to its own path, so it has to happen at
/// the final location. Any failure puts the old copy back, and a source that
/// now declares a different name is refused the same way, since installing
/// it would leave `name` missing while its record points somewhere else.
///
/// # Errors
/// The install's own error, a changed name, or a filesystem failure moving
/// the old copy.
pub fn replace_profile(arg: &str, name: &str, home: &Path) -> Result<Installed, String> {
    let dest = crate::profiles::dir(home).join(name);
    let aside_root = crate::profiles::dir(home).join(".replacing");
    std::fs::create_dir_all(&aside_root)
        .map_err(|e| format!("cannot create {}: {e}", aside_root.display()))?;
    let aside = aside_root.join(format!("{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&aside);
    std::fs::rename(&dest, &aside)
        .map_err(|e| format!("cannot move {} aside: {e}", dest.display()))?;
    let restore = |why: String| {
        let _ = std::fs::remove_dir_all(&dest);
        match std::fs::rename(&aside, &dest) {
            Ok(()) => why,
            Err(e) => format!("{why}; the previous copy is at {} ({e})", aside.display()),
        }
    };
    match install_profile(arg, None, home, false) {
        Ok(installed) if installed.name == name => {
            let _ = std::fs::remove_dir_all(&aside);
            Ok(installed)
        }
        Ok(installed) => {
            let _ = std::fs::remove_dir_all(&installed.dest);
            Err(restore(format!(
                "{arg} now declares the profile '{}', not '{name}'",
                installed.name
            )))
        }
        Err(e) => Err(restore(e)),
    }
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
    let result = fetch(arg, &staging)
        .and_then(|tree| install_staged(&tree, want, &install_dir(home), force));
    let _ = std::fs::remove_dir_all(&staging);
    result
}

/// A private, empty staging directory. Removed by [`install`] (and by
/// [`crate::plugins::install_from_url`], which stages here too) on every path.
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
pub(crate) fn staging_dir(home: &Path) -> Result<PathBuf, String> {
    let base = staging_dir_path(home);
    // Fresh every time: reusing it would mix a previous failed fetch into a new
    // one, and the plugin root is found by looking at the tree.
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).map_err(|e| format!("cannot create {}: {e}", base.display()))?;
    Ok(base)
}

/// Where [`staging_dir`] lives, without creating it; for tests that check it
/// was cleaned up.
#[must_use]
pub(crate) fn staging_dir_path(home: &Path) -> PathBuf {
    crate::home::plank_home_in(home).join(".claude-staging")
}

/// Puts the named plugin's tree inside `staging` and returns its root.
fn fetch(arg: &str, staging: &Path) -> Result<PathBuf, String> {
    // A local directory is checked before the URL forms: `git clone` takes a
    // path as happily as a URL, and it is how a plugin is tried before it is
    // published. Not every directory is a repository, though — a plain tree
    // assembled by hand or by a test has no `.git` and no bare-repo layout —
    // so a clone failure falls back to a plain copy rather than refusing.
    if Path::new(arg).is_dir() {
        if let Ok(dest) = clone(arg, None, staging) {
            return Ok(dest);
        }
        // Scanned before the copy, not after: `copy_tree` follows links, so a
        // `key -> ~/.ssh/id_rsa` entry would already be a plain file holding
        // the private key's bytes by the time a check ran on the destination,
        // and the refusal would find nothing left to refuse. This is the same
        // inversion `install_staged` guards against for the staged tree.
        crate::plugins::reject_escaping_symlinks(Path::new(arg))?;
        let dest = staging.join("tree");
        crate::plugins::copy_tree(Path::new(arg), &dest)
            .map_err(|e| format!("cannot copy {arg}: {e}"))?;
        return Ok(dest);
    }
    match parse_source(arg)? {
        Source::Git { url } => clone(&url, None, staging),
        Source::GitSubpath {
            url,
            refname,
            subpath,
        } => {
            let dest = clone(&url, Some(&refname), staging)?;
            resolve_subpath(&dest, &subpath)
        }
        Source::GitFolder { url, folder } => {
            let dest = clone(&url, None, staging)?;
            let root = dest.join(&folder);
            if is_claude_manifest_root(&root) {
                Ok(root)
            } else {
                Err(format!(
                    "{url} has no plugin manifest in its folder '{folder}'"
                ))
            }
        }
        Source::Archive { url } => {
            // `download_and_extract`, not `plugins::fetch_archive`: the latter
            // bundles in its own call to the same scan, and calling it here
            // too would just scan the tree twice.
            crate::plugins::download_and_extract(&url, staging)?;
            crate::plugins::reject_escaping_symlinks(staging)?;
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
                "this is not a plank or Claude Code plugin: no .plank-plugin/plugin.json, no .claude-plugin/plugin.json and no .claude-plugin/marketplace.json at its root or one level in".to_string()
            })
        }
    }
}

/// Whether `dir` is the root of an installable tree: a plugin manifest in
/// either spelling, or a marketplace manifest.
///
/// Both spellings, because `/install-profile` fetches profiles authored in
/// plank's own `.plank-plugin/` layout and `/install-claude-plugin` had no
/// reason to refuse them either — a plank-spelling plugin was previously
/// unfetchable by any command.
fn is_claude_manifest_root(dir: &Path) -> bool {
    crate::plugins::manifest_path(dir).is_some()
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

/// Shallow-clones `url` into `staging/repo`, optionally at `refname`, and
/// drops its history.
///
/// `refname` is `None` for a plain repository URL and `Some` for a
/// [`Source::GitSubpath`], where the browser URL named a specific branch,
/// tag, or commit and cloning the default branch instead would silently show
/// the wrong tree.
fn clone(url: &str, refname: Option<&str>, staging: &Path) -> Result<PathBuf, String> {
    let dest = staging.join("repo");
    let mut cmd = std::process::Command::new("git");
    cmd.arg("clone").arg("--quiet").arg("--depth").arg("1");
    if let Some(refname) = refname {
        cmd.arg("--branch").arg(refname);
    }
    let output = cmd
        .arg("--") // Separator ensures url is treated as a path, not an option
        .arg(url)
        .arg(&dest)
        .output()
        .map_err(|e| format!("cannot run git: {e}"))?;
    if !output.status.success() {
        // `.output()` rather than `.status()`: git's stderr says *why* the
        // clone failed (repository not found, ref not found, network
        // unreachable), and a bare "cannot clone <url>" throws that away —
        // exactly the useless message a pasted-but-wrong URL used to produce.
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr = stderr.trim();
        return Err(if stderr.is_empty() {
            format!("cannot clone {url}")
        } else {
            format!("cannot clone {url}: {stderr}")
        });
    }
    // The history is not part of the plugin, and it is a tree of files that
    // would otherwise be scanned for symlinks and copied into the user's home.
    let _ = std::fs::remove_dir_all(dest.join(".git"));
    crate::plugins::reject_escaping_symlinks(&dest)?;
    Ok(dest)
}

/// Finds the plugin root under `dest` named by `subpath`.
///
/// A GitHub tree URL's path is not always the plugin root itself: the real
/// example this fix was written for pointed at `.claude-plugin`, which is the
/// *manifest* directory, and the plugin root is its parent. So `subpath` is
/// tried first, and failing that, its parent — the manifest-directory case
/// covers both `.claude-plugin` and a bare `plugin.json` file (a `/blob/` URL
/// resolves to a file, whose parent is the manifest directory, whose parent in
/// turn is the plugin root; this function does one parent hop, which is
/// enough for the directory case and named explicitly in the refusal when
/// neither matches so the miss is legible rather than a raw filesystem error.
fn resolve_subpath(dest: &Path, subpath: &str) -> Result<PathBuf, String> {
    let candidate = dest.join(subpath);
    if is_claude_manifest_root(&candidate) {
        return Ok(candidate);
    }
    if let Some(parent) = candidate.parent()
        && is_claude_manifest_root(parent)
    {
        return Ok(parent.to_path_buf());
    }
    Err(format!(
        "no Claude Code plugin found at '{subpath}' or its parent directory in this repository"
    ))
}

/// The plugin's name, from its manifest's `name` field, falling back to the
/// directory name. The name becomes a path component under `install_dir`, so
/// values that would not create a fresh subdirectory are refused rather than
/// sanitized. This includes path separators, `..`, `.` (which resolves to the
/// parent), and whitespace-only strings. A plugin calling itself `../x` or `.`
/// is not a naming style to accommodate.
fn plugin_name(root: &Path) -> Result<String, String> {
    let manifest = crate::plugins::manifest_path(root)
        .ok_or_else(|| format!("no plugin.json in {}", root.display()))?;
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

/// Splits an install command's argument line into the `--force` flag and the
/// positional words, in order.
///
/// Shared by [`render_install`] and [`render_install_profile`]: the two render
/// different text but take the same argument shape, and a flag one of them
/// learned to accept and the other did not would be a silent divergence
/// between two commands users expect to behave alike.
fn split_force_flag(arg: &str) -> (bool, Vec<&str>) {
    let mut force = false;
    let mut words: Vec<&str> = Vec::new();
    for word in arg.split_whitespace() {
        if word == "--force" {
            force = true;
        } else {
            words.push(word);
        }
    }
    (force, words)
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
    let (force, words) = split_force_flag(arg);
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

/// The whole `/install-profile` command: parses its argument line, runs the
/// install, and renders the outcome as the text both front ends print.
///
/// Mirrors [`render_install`] rather than sharing its body: the two differ in
/// their usage line, in the closing hint (a profile is launched with
/// `--profile`, not merely loaded), and in how they are removed. Sharing would
/// mean threading three flags through one function to save a dozen lines.
#[must_use]
pub fn render_install_profile(arg: &str, home: Option<&Path>) -> String {
    // `concat!`, not a `\`-continued literal: continuation strips the next
    // line's leading whitespace.
    const USAGE: &str = concat!(
        "usage: /install-profile <url|owner/repo|path> [name] [--force]\n",
        "a url may be a git repository, a marketplace repository, or a .tar.gz\n"
    );
    let (force, words) = split_force_flag(arg);
    let Some(source) = words.first() else {
        return USAGE.to_string();
    };
    let Some(home) = home else {
        return "plank: no home directory; nowhere to install a profile\n".to_string();
    };
    match install_profile(source, words.get(1).copied(), home, force) {
        Ok(installed) => format!(
            "installed profile '{}' into {}\nrun it with: plank --profile {}\nremove it with: rm -rf {}\n",
            installed.name,
            installed.dest.display(),
            installed.name,
            installed.dest.display()
        ),
        Err(e) => format!("plank: {e}\n"),
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
    fn a_plank_spelling_tree_is_found_and_named() {
        let dir = tmpdir("plank-spelling");
        let root = dir.join("thing");
        std::fs::create_dir_all(root.join(".plank-plugin")).expect("mkdir");
        std::fs::write(
            root.join(".plank-plugin").join("plugin.json"),
            r#"{"name":"thing"}"#,
        )
        .expect("write");
        // Found one level in, exactly as the Claude spelling is.
        assert_eq!(find_claude_root(&dir), Some(root.clone()));
        // And named from the plank manifest rather than falling back to the
        // directory name, which would silently differ when the two disagree.
        assert_eq!(plugin_name(&root).as_deref(), Ok("thing"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_plank_manifest_wins_when_a_tree_carries_both() {
        let dir = tmpdir("both-spellings");
        std::fs::create_dir_all(dir.join(".plank-plugin")).expect("mkdir");
        std::fs::create_dir_all(dir.join(".claude-plugin")).expect("mkdir");
        std::fs::write(
            dir.join(".plank-plugin").join("plugin.json"),
            r#"{"name":"plank-name"}"#,
        )
        .expect("write");
        std::fs::write(
            dir.join(".claude-plugin").join("plugin.json"),
            r#"{"name":"claude-name"}"#,
        )
        .expect("write");
        // Same precedence `plugins::manifest_path` already uses everywhere else,
        // so a tree with both spellings loads under one name, not two.
        assert_eq!(plugin_name(&dir).as_deref(), Ok("plank-name"));
        let _ = std::fs::remove_dir_all(&dir);
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
    fn a_github_tree_url_parses_to_repo_ref_and_subpath() {
        let src = parse_source("https://github.com/obra/superpowers/tree/main/skills/foo")
            .expect("parses");
        assert_eq!(
            src,
            Source::GitSubpath {
                url: "https://github.com/obra/superpowers".to_string(),
                refname: "main".to_string(),
                subpath: "skills/foo".to_string(),
            }
        );
    }

    #[test]
    fn a_github_blob_url_parses_too() {
        let src = parse_source("https://github.com/obra/superpowers/blob/main/README.md")
            .expect("parses");
        assert!(matches!(
            src,
            Source::GitSubpath { ref refname, ref subpath, .. }
                if refname == "main" && subpath == "README.md"
        ));
    }

    #[test]
    fn a_github_tree_url_at_the_manifest_directory_still_resolves_via_its_parent() {
        // The real example this fix was written for: the user's browser URL
        // pointed at `.claude-plugin`, the manifest directory, whose *parent*
        // is the plugin root.
        let src = parse_source("https://github.com/obra/superpowers/tree/main/.claude-plugin")
            .expect("parses");
        assert!(matches!(
            src,
            Source::GitSubpath { ref subpath, .. } if subpath == ".claude-plugin"
        ));
    }

    #[test]
    fn a_github_source_checks_its_manifest_through_the_api_then_the_cdn() {
        assert_eq!(
            raw_manifest_urls("aovestdipaperino/plank-profiles:HAL").expect("github"),
            [
                "https://api.github.com/repos/aovestdipaperino/plank-profiles/contents/HAL/.plank-plugin/plugin.json",
                "https://api.github.com/repos/aovestdipaperino/plank-profiles/contents/HAL/.claude-plugin/plugin.json",
                "https://raw.githubusercontent.com/aovestdipaperino/plank-profiles/HEAD/HAL/.plank-plugin/plugin.json",
                "https://raw.githubusercontent.com/aovestdipaperino/plank-profiles/HEAD/HAL/.claude-plugin/plugin.json",
            ]
        );
        assert_eq!(
            raw_manifest_urls("o/r").expect("github")[0],
            "https://api.github.com/repos/o/r/contents/.plank-plugin/plugin.json"
        );
        assert_eq!(raw_manifest_urls("https://example.com/p.tar.gz"), None);
        assert_eq!(raw_manifest_urls("https://gitlab.com/o/r"), None);
    }

    /// A profile source directory whose manifest declares `version`.
    fn versioned_profile(tag: &str, name: &str, version: &str) -> PathBuf {
        let src = tmpdir(tag).join(name);
        write(
            &src,
            ".plank-plugin/plugin.json",
            &format!(
                r#"{{"name":"{name}","version":"{version}","profile":{{"systemPrompt":"prompt.md"}}}}"#
            ),
        );
        write(&src, "prompt.md", "You are a test profile.\n");
        src
    }

    #[test]
    fn a_local_source_reports_its_version() {
        let src = versioned_profile("source-version", "hal", "0.2.0");
        assert_eq!(
            source_version(src.to_str().expect("utf8")).as_deref(),
            Some("0.2.0")
        );
    }

    #[test]
    fn replacing_installs_the_new_copy_and_drops_the_old_one() {
        let home = tmpdir("replace-home");
        let v1 = versioned_profile("replace-v1", "hal", "0.1.0");
        install_profile(v1.to_str().expect("utf8"), None, &home, false).expect("v1");
        let v2 = versioned_profile("replace-v2", "hal", "0.2.0");
        let installed = replace_profile(v2.to_str().expect("utf8"), "hal", &home).expect("v2");
        assert_eq!(
            crate::profiles::version_of(&installed.dest).as_deref(),
            Some("0.2.0")
        );
        let aside = crate::profiles::dir(&home).join(".replacing");
        assert_eq!(std::fs::read_dir(&aside).map_or(0, Iterator::count), 0);
    }

    #[test]
    fn a_failed_replace_puts_the_old_copy_back() {
        let home = tmpdir("replace-fail-home");
        let v1 = versioned_profile("replace-fail-v1", "hal", "0.1.0");
        install_profile(v1.to_str().expect("utf8"), None, &home, false).expect("v1");
        let renamed = versioned_profile("replace-fail-v2", "opus", "0.2.0");
        assert!(replace_profile(renamed.to_str().expect("utf8"), "hal", &home).is_err());
        let dest = crate::profiles::dir(&home).join("hal");
        assert_eq!(crate::profiles::version_of(&dest).as_deref(), Some("0.1.0"));
        assert!(!crate::profiles::dir(&home).join("opus").exists());
        let broken = tmpdir("replace-fail-missing").join("nothing-here");
        assert!(replace_profile(broken.to_str().expect("utf8"), "hal", &home).is_err());
        assert_eq!(crate::profiles::version_of(&dest).as_deref(), Some("0.1.0"));
    }

    #[test]
    fn owner_repo_colon_folder_names_one_folder_of_a_github_repo() {
        assert_eq!(
            parse_source("aovestdipaperino/plank-profiles:HAL").expect("parses"),
            Source::GitFolder {
                url: "https://github.com/aovestdipaperino/plank-profiles".to_string(),
                folder: "HAL".to_string(),
            }
        );
        assert!(matches!(
            parse_source("o/r:nested/dir/").expect("parses"),
            Source::GitFolder { ref folder, .. } if folder == "nested/dir"
        ));
        for bad in ["o/r:", "o/r:../x", "o/r:/", "example.com/r:x", "o:x"] {
            assert!(parse_source(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn resolve_subpath_falls_back_to_the_parent_of_the_manifest_directory() {
        let dest = tmpdir("resolve-subpath-parent");
        write(&dest, ".claude-plugin/plugin.json", r#"{"name":"demo"}"#);
        let root = resolve_subpath(&dest, ".claude-plugin").expect("resolves");
        assert_eq!(root, dest);
    }

    #[test]
    fn resolve_subpath_uses_the_subpath_itself_when_it_is_already_the_root() {
        let dest = tmpdir("resolve-subpath-direct");
        write(
            &dest,
            "plugin/.claude-plugin/plugin.json",
            r#"{"name":"demo"}"#,
        );
        let root = resolve_subpath(&dest, "plugin").expect("resolves");
        assert_eq!(root, dest.join("plugin"));
    }

    #[test]
    fn resolve_subpath_names_what_it_looked_for_when_neither_matches() {
        let dest = tmpdir("resolve-subpath-miss");
        write(&dest, "README.md", "hi\n");
        let err = resolve_subpath(&dest, "docs").expect_err("refused");
        assert!(err.contains("docs"), "{err}");
    }

    #[test]
    fn a_non_github_url_is_not_a_tree_url() {
        assert!(parse_github_tree_url("https://example.com/o/r/tree/main/x").is_none());
    }

    #[test]
    fn a_plain_github_repo_url_is_not_a_tree_url() {
        assert!(parse_github_tree_url("https://github.com/owner/repo").is_none());
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

    /// The real `hooks/hooks.json` from obra/superpowers, nesting every event
    /// under one outer `"hooks"` key and using `${CLAUDE_PLUGIN_ROOT}`. This
    /// is the file that found the defect.
    const SUPERPOWERS_HOOKS_JSON: &str = r#"{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|clear|compact",
        "hooks": [
          {
            "type": "command",
            "command": "\"${CLAUDE_PLUGIN_ROOT}/hooks/run-hook.cmd\" session-start",
            "shell": "bash",
            "async": false
          }
        ]
      }
    ]
  }
}"#;

    #[test]
    fn a_nested_hooks_file_naming_only_known_events_is_supported() {
        let root = tmpdir("hooks-nested-known");
        write(&root, "hooks/hooks.json", SUPERPOWERS_HOOKS_JSON);
        assert_eq!(
            unsupported_hook_events(&root).expect("ok"),
            Vec::<String>::new()
        );
    }

    #[test]
    fn a_nested_hooks_file_naming_an_unknown_event_is_still_refused() {
        let root = tmpdir("hooks-nested-unknown");
        write(
            &root,
            "hooks/hooks.json",
            r#"{"hooks":{"SubagentStop":[{"hooks":[{"type":"command","command":"true"}]}]}}"#,
        );
        let got = unsupported_hook_events(&root).expect("ok");
        assert_eq!(got, vec!["SubagentStop".to_string()]);
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
    fn a_nested_hooks_file_is_flattened_and_still_parses_with_its_command_intact() {
        // The end-to-end proof this task is about: install the real
        // superpowers file, then hand the INSTALLED file to plank's own
        // `hooks::parse_config` and check the event and command survived,
        // not just that the text looks flat.
        let dest = tmpdir("rewrite-flatten-parity");
        write(&dest, "hooks/hooks.json", SUPERPOWERS_HOOKS_JSON);
        rewrite_plugin_root(&dest).expect("ok");
        let installed = std::fs::read_to_string(dest.join("hooks/hooks.json")).expect("read");
        assert!(!installed.contains("CLAUDE_PLUGIN_ROOT"), "{installed}");
        let hooks = crate::hooks::parse_config(&installed);
        assert!(hooks.warnings.is_empty(), "{:?}", hooks.warnings);
        assert_eq!(hooks.session_start.len(), 1);
        let group = &hooks.session_start[0];
        assert_eq!(group.matcher, "startup|clear|compact");
        assert_eq!(group.hooks.len(), 1);
        let root = dest.display().to_string();
        assert!(
            group.hooks[0]
                .command
                .contains(&format!("{root}/hooks/run-hook.cmd")),
            "{}",
            group.hooks[0].command
        );
    }

    #[test]
    fn a_nested_file_with_claude_plugin_root_ends_up_both_flat_and_substituted() {
        let dest = tmpdir("rewrite-flatten-and-substitute");
        write(
            &dest,
            "hooks/hooks.json",
            r#"{"hooks":{"PreToolUse":[{"matcher":"Bash","hooks":[{"type":"command","command":"${CLAUDE_PLUGIN_ROOT}/bin/g"}]}]}}"#,
        );
        assert!(rewrite_plugin_root(&dest).expect("ok"));
        let installed = std::fs::read_to_string(dest.join("hooks/hooks.json")).expect("read");
        assert!(!installed.contains("CLAUDE_PLUGIN_ROOT"), "{installed}");
        assert!(
            !installed.trim_start().starts_with("{\"hooks\":{"),
            "{installed}"
        );
        let root = dest.display().to_string();
        let hooks = crate::hooks::parse_config(&installed);
        assert_eq!(hooks.pre_tool_use.len(), 1);
        assert!(
            hooks.pre_tool_use[0].hooks[0]
                .command
                .contains(&format!("{root}/bin/g")),
            "{}",
            hooks.pre_tool_use[0].hooks[0].command
        );
    }

    #[test]
    fn an_already_flat_hooks_file_is_left_byte_for_byte_unchanged() {
        let dest = tmpdir("rewrite-flat-unchanged");
        let body =
            r#"{"PreToolUse":[{"matcher":"Bash","hooks":[{"type":"command","command":"true"}]}]}"#;
        write(&dest, "hooks/hooks.json", body);
        assert!(!rewrite_plugin_root(&dest).expect("ok"));
        let installed = std::fs::read_to_string(dest.join("hooks/hooks.json")).expect("read");
        assert_eq!(installed, body);
        let hooks = crate::hooks::parse_config(&installed);
        assert_eq!(hooks.pre_tool_use.len(), 1);
    }

    #[test]
    fn a_matcher_groups_inner_hooks_array_survives_the_unwrap_intact() {
        // The subtlest part of the task: the outer `"hooks"` wrapper must be
        // removed, but each matcher group's own `"hooks"` array — plank's
        // native format one level down — must not be touched by the same
        // name.
        let dest = tmpdir("rewrite-inner-hooks-survive");
        write(
            &dest,
            "hooks/hooks.json",
            r#"{"hooks":{"PreToolUse":[{"matcher":"Bash","hooks":[{"type":"command","command":"one"},{"type":"command","command":"two"}]}]}}"#,
        );
        rewrite_plugin_root(&dest).expect("ok");
        let installed = std::fs::read_to_string(dest.join("hooks/hooks.json")).expect("read");
        let hooks = crate::hooks::parse_config(&installed);
        assert_eq!(hooks.pre_tool_use.len(), 1);
        assert_eq!(hooks.pre_tool_use[0].hooks.len(), 2);
        assert_eq!(hooks.pre_tool_use[0].hooks[0].command, "one");
        assert_eq!(hooks.pre_tool_use[0].hooks[1].command, "two");
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
    fn install_staged_copies_into_the_destination_it_is_given() {
        let tmp = tmpdir("dest-root");
        let staged = tmp.join("staged");
        std::fs::create_dir_all(staged.join(".claude-plugin")).expect("mkdir");
        std::fs::write(
            staged.join(".claude-plugin").join("plugin.json"),
            r#"{"name":"thing"}"#,
        )
        .expect("write");
        let dest_root = tmp.join("somewhere-else");
        let installed = install_staged(&staged, None, &dest_root, false).expect("installs");
        assert_eq!(installed.dest, dest_root.join("thing"));
        assert!(dest_root.join("thing").join(".claude-plugin").is_dir());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn a_valid_tree_installs_under_the_claude_root() {
        let staged = staged_plugin("install-ok", "demo");
        write(&staged, "commands/note.md", "hi\n");
        let home = tmpdir("install-ok-home");
        let out = install_staged(&staged, None, &install_dir(&home), false).expect("installs");
        assert_eq!(out.name, "demo");
        assert_eq!(out.dest, install_dir(&home).join("demo"));
        assert!(out.dest.join(".claude-plugin/plugin.json").is_file());
        assert!(out.dest.join("commands/note.md").is_file());
        assert_eq!(out.skipped_hook_events, [] as [std::string::String; 0]);
        assert!(!out.rewrote_plugin_root);
    }

    #[test]
    fn an_unimplemented_hook_event_refuses() {
        let staged = staged_plugin("install-hook", "demo");
        write(&staged, "hooks/hooks.json", r#"{"SubagentStop":[]}"#);
        let home = tmpdir("install-hook-home");
        let err = install_staged(&staged, None, &install_dir(&home), false).expect_err("refused");
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
        let out = install_staged(&staged, None, &install_dir(&home), true).expect("installs");
        assert_eq!(out.skipped_hook_events, vec!["SubagentStop".to_string()]);
        assert!(out.dest.join("hooks/hooks.json").is_file());
    }

    #[test]
    fn a_symlink_refuses_even_with_force() {
        let staged = staged_plugin("install-symlink", "demo");
        std::os::unix::fs::symlink("/etc/hosts", staged.join("link")).expect("symlink");
        let home = tmpdir("install-symlink-home");
        let err = install_staged(&staged, None, &install_dir(&home), true).expect_err("refused");
        assert!(err.contains("symlink"), "{err}");
        assert!(
            !install_dir(&home).join("demo").exists(),
            "nothing installed"
        );
    }

    #[test]
    fn a_contained_file_symlink_is_accepted_and_installs() {
        // The `obra/superpowers` case: `AGENTS.md -> CLAUDE.md`, a relative
        // link resolving inside the tree — the ordinary "same file under two
        // names" idiom, not a private-key exfiltration attempt.
        let staged = staged_plugin("install-contained-symlink", "demo");
        write(&staged, "CLAUDE.md", "the real content\n");
        std::os::unix::fs::symlink("CLAUDE.md", staged.join("AGENTS.md")).expect("symlink");
        let home = tmpdir("install-contained-symlink-home");
        let out = install_staged(&staged, None, &install_dir(&home), false).expect("installs");
        let installed = std::fs::read_to_string(out.dest.join("AGENTS.md")).expect("read");
        assert_eq!(installed, "the real content\n");
    }

    #[test]
    fn an_absolute_escaping_symlink_is_refused() {
        let staged = staged_plugin("install-escape-absolute", "demo");
        std::os::unix::fs::symlink("/etc/hosts", staged.join("link")).expect("symlink");
        let home = tmpdir("install-escape-absolute-home");
        let err = install_staged(&staged, None, &install_dir(&home), false).expect_err("refused");
        assert!(err.contains("symlink"), "{err}");
        assert!(!install_dir(&home).join("demo").exists());
    }

    #[test]
    fn a_relative_escaping_symlink_is_refused() {
        let staged = staged_plugin("install-escape-relative", "demo");
        let secret_dir = tmpdir("install-escape-relative-secret");
        std::fs::write(secret_dir.join("secret.txt"), "sekrit-payload\n").expect("write");
        // `../../<secret dir name>/secret.txt` walks out of `staged` itself.
        let secret_name = secret_dir.file_name().expect("has a name");
        let rel = Path::new("..")
            .join("..")
            .join(
                secret_dir
                    .parent()
                    .expect("has a parent")
                    .file_name()
                    .expect("name"),
            )
            .join(secret_name)
            .join("secret.txt");
        std::os::unix::fs::symlink(&rel, staged.join("link")).expect("symlink");
        let home = tmpdir("install-escape-relative-home");
        let err = install_staged(&staged, None, &install_dir(&home), false).expect_err("refused");
        assert!(err.contains("symlink"), "{err}");
        assert!(!install_dir(&home).join("demo").exists());
        assert!(
            !walk_for_secret(&install_dir(&home), "sekrit-payload"),
            "the escaping target's contents must not reach the install root"
        );
    }

    #[test]
    fn removing_the_containment_check_would_let_the_escaping_symlink_through() {
        // A direct regression pin on `scan_unsafe_symlinks`'s containment
        // test, independent of `install_staged`'s plumbing: a symlink whose
        // canonicalized target is outside `canon_root` must be refused, and
        // this asserts the specific condition rather than just "something
        // failed".
        let root = tmpdir("scan-containment-root");
        let outside = tmpdir("scan-containment-outside");
        std::fs::write(outside.join("x"), "hi\n").expect("write");
        std::os::unix::fs::symlink(outside.join("x"), root.join("link")).expect("symlink");
        let err = crate::plugins::reject_escaping_symlinks(&root).expect_err("refused");
        assert!(err.contains("outside"), "{err}");
    }

    #[test]
    fn a_contained_directory_symlink_is_refused_with_a_clear_message() {
        let staged = staged_plugin("install-dir-symlink", "demo");
        std::fs::create_dir_all(staged.join("real_dir")).expect("mkdir");
        std::fs::write(staged.join("real_dir/f.txt"), "hi\n").expect("write");
        std::os::unix::fs::symlink("real_dir", staged.join("link_dir")).expect("symlink");
        let home = tmpdir("install-dir-symlink-home");
        let err = install_staged(&staged, None, &install_dir(&home), false).expect_err("refused");
        assert!(err.contains("symlink"), "{err}");
        assert!(err.contains("directory"), "{err}");
        assert!(!install_dir(&home).join("demo").exists());
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
        let err = install_staged(&staged, None, &install_dir(&home), true).expect_err("refused");
        assert!(err.contains("already installed"), "{err}");
        let kept =
            std::fs::read_to_string(install_dir(&home).join("demo/.claude-plugin/plugin.json"))
                .expect("read");
        assert!(kept.contains("already there"), "not overwritten");
    }

    #[test]
    fn a_plank_spelling_only_tree_installs() {
        let staged = tmpdir("install-plankspelling");
        write(&staged, ".plank-plugin/plugin.json", r#"{"name":"native"}"#);
        let home = tmpdir("install-plankspelling-home");
        let installed = install_staged(&staged, None, &install_dir(&home), true).expect("installs");
        assert!(
            installed
                .dest
                .join(".plank-plugin")
                .join("plugin.json")
                .is_file()
        );
    }

    #[test]
    fn a_tree_with_no_manifest_at_all_refuses() {
        let staged = tmpdir("install-nomanifest");
        write(&staged, "README.md", "nothing here\n");
        let home = tmpdir("install-nomanifest-home");
        let err = install_staged(&staged, None, &install_dir(&home), true).expect_err("refused");
        assert!(err.contains(".plank-plugin/plugin.json"), "{err}");
        assert!(err.contains(".claude-plugin/plugin.json"), "{err}");
        assert!(err.contains(".claude-plugin/marketplace.json"), "{err}");
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
        let out = install_staged(&staged, None, &install_dir(&home), false).expect("installs");
        assert!(out.rewrote_plugin_root);
        let mcp = std::fs::read_to_string(out.dest.join(".mcp.json")).expect("read");
        assert!(mcp.contains(&out.dest.display().to_string()), "{mcp}");
    }

    #[test]
    fn an_installed_plugin_is_found_by_the_loader() {
        let staged = staged_plugin("install-loads", "demo");
        let home = tmpdir("install-loads-home");
        let out = install_staged(&staged, None, &install_dir(&home), false).expect("installs");
        let set = crate::plugins::load_in(Some(&home), &tmpdir("install-loads-cwd"));
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
        let err = install_staged(&staged, None, &install_dir(&home), false).expect_err("refused");
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
        let err = install_staged(&staged, None, &install_dir(&home), true).expect_err("refused");
        assert!(err.contains("not a usable plugin name"), "{err}");
    }

    #[test]
    fn plugin_with_whitespace_only_name_is_refused() {
        let staged = staged_plugin("install-whitespace", "   ");
        let home = tmpdir("install-whitespace-home");
        let err = install_staged(&staged, None, &install_dir(&home), false).expect_err("refused");
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

    /// A root with a `"` or `\` in it spliced into the JSON text would leave
    /// an unparseable file; the rewrite goes through the tree so the writer
    /// escapes it.
    #[test]
    fn rewrite_escapes_a_root_containing_a_quote() {
        let base = tmpdir("rewrite-quote");
        let dest = base.join(r#"odd"name\dir"#);
        write(
            &dest,
            "hooks/hooks.json",
            r#"{"PreToolUse":[{"hooks":[{"type":"command","command":"${CLAUDE_PLUGIN_ROOT}/bin/g"}]}]}"#,
        );
        write(
            &dest,
            ".mcp.json",
            r#"{"mcpServers":{"s":{"command":"$CLAUDE_PLUGIN_ROOT/bin/s","args":["${CLAUDE_PLUGIN_ROOT}/a"]}}}"#,
        );
        assert!(rewrite_plugin_root(&dest).expect("rewrite"));
        let root = dest.display().to_string();

        let hooks = std::fs::read_to_string(dest.join("hooks/hooks.json")).unwrap();
        let parsed = json_parse(&hooks).expect("hooks.json still parses");
        let cmd = parsed
            .get("PreToolUse")
            .and_then(|v| match v {
                Json::Arr(items) => items.first(),
                _ => None,
            })
            .and_then(|e| e.get("hooks"))
            .and_then(|v| match v {
                Json::Arr(items) => items.first(),
                _ => None,
            })
            .and_then(|h| h.get("command"));
        assert_eq!(cmd, Some(&Json::Str(format!("{root}/bin/g"))), "{hooks}");

        let mcp = std::fs::read_to_string(dest.join(".mcp.json")).unwrap();
        let parsed = json_parse(&mcp).expect(".mcp.json still parses");
        let server = parsed
            .get("mcpServers")
            .and_then(|v| v.get("s"))
            .expect("server");
        assert_eq!(
            server.get("command"),
            Some(&Json::Str(format!("{root}/bin/s"))),
            "{mcp}"
        );
        assert_eq!(
            server.get("args"),
            Some(&Json::Arr(vec![Json::Str(format!("{root}/a"))])),
            "{mcp}"
        );
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
        let set = crate::plugins::load_in(Some(&home), &tmpdir("staging-scope-cwd"));
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

    /// Builds a staged tree at `<tmp>/staged` and returns it. `block` is the
    /// `profile` member's JSON text, or `None` for no profile block. `prompt` is
    /// the prompt file's contents, or `None` to write no prompt file at all.
    fn staged_profile(tmp: &Path, block: Option<&str>, prompt: Option<&str>) -> PathBuf {
        let staged = tmp.join("staged");
        std::fs::create_dir_all(staged.join(".plank-plugin")).expect("mkdir");
        if let Some(p) = prompt {
            std::fs::write(staged.join("prompt.md"), p).expect("write");
        }
        let manifest = match block {
            Some(b) => format!("{{\"name\":\"hal\",\"profile\":{b}}}"),
            None => "{\"name\":\"hal\"}".to_string(),
        };
        std::fs::write(staged.join(".plank-plugin").join("plugin.json"), manifest).expect("write");
        staged
    }

    const GOOD_BLOCK: &str = r#"{"systemPrompt":"prompt.md"}"#;

    #[test]
    fn a_tree_with_no_profile_block_is_refused() {
        let tmp = tmpdir("gate-no-block");
        let staged = staged_profile(&tmp, None, Some("hi\n"));
        let err = install_profile_staged(&staged, None, &tmp, false).expect_err("refused");
        assert!(err.contains("declares no profile block"), "{err}");
        assert!(err.contains("/install-claude-plugin"), "{err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn a_profile_whose_prompt_file_is_missing_is_refused() {
        let tmp = tmpdir("gate-no-prompt");
        let staged = staged_profile(&tmp, Some(GOOD_BLOCK), None);
        let err = install_profile_staged(&staged, None, &tmp, false).expect_err("refused");
        assert!(err.contains("cannot read"), "{err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn a_profile_whose_prompt_file_is_blank_is_refused() {
        let tmp = tmpdir("gate-blank-prompt");
        let staged = staged_profile(&tmp, Some(GOOD_BLOCK), Some("   \n\n"));
        let err = install_profile_staged(&staged, None, &tmp, false).expect_err("refused");
        assert!(err.contains("is empty"), "{err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// N1: a name `plugin_name` admits but `plugins::valid_name` refuses
    /// (double underscore, which mints an unroutable MCP server namespace)
    /// must be caught here, at install time, rather than installed
    /// successfully only to be refused later by `--profile`.
    #[test]
    fn a_profile_whose_name_fails_the_namespace_grammar_is_refused() {
        let tmp = tmpdir("gate-bad-namespace-name");
        let staged = tmp.join("staged");
        std::fs::create_dir_all(staged.join(".plank-plugin")).expect("mkdir");
        std::fs::write(staged.join("prompt.md"), "You are HAL.\n").expect("write");
        std::fs::write(
            staged.join(".plank-plugin").join("plugin.json"),
            r#"{"name":"a__b","profile":{"systemPrompt":"prompt.md"}}"#,
        )
        .expect("write");
        let home = tmp.join("home");
        let err = install_profile_staged(&staged, None, &home, false).expect_err("refused");
        assert!(err.contains("a__b"), "{err}");
        assert!(
            !crate::profiles::dir(&home).join("a__b").exists(),
            "a refused name must leave nothing installed"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn a_good_profile_lands_in_the_profiles_root() {
        let tmp = tmpdir("gate-good");
        let home = tmp.join("home");
        let staged = staged_profile(&tmp, Some(GOOD_BLOCK), Some("You are HAL.\n"));
        let installed = install_profile_staged(&staged, None, &home, false).expect("installs");
        assert_eq!(installed.dest, crate::profiles::dir(&home).join("hal"));
        assert!(installed.dest.join("prompt.md").is_file());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The gate runs before the copy: a refused profile leaves nothing behind for
    /// the next `--profile` listing to find.
    #[test]
    fn a_refused_profile_is_not_partially_installed() {
        let tmp = tmpdir("gate-atomic");
        let home = tmp.join("home");
        let staged = staged_profile(&tmp, None, Some("hi\n"));
        let _ = install_profile_staged(&staged, None, &home, false);
        assert!(!crate::profiles::dir(&home).join("hal").exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn the_profile_usage_line_names_its_own_command() {
        let out = render_install_profile("", Some(Path::new("/tmp/h")));
        assert!(out.contains("usage: /install-profile"), "{out}");
    }
}
