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

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes `body` to `root/rel`, creating parent directories.
    fn write(root: &Path, rel: &str, body: &str) {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().expect("has a parent")).expect("mkdir");
        std::fs::write(&path, body).expect("write");
    }

    /// A fresh empty directory under the system temp dir, named for the test.
    fn tmpdir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("plank-claudeplugin-{tag}-{}", std::process::id()));
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
}
