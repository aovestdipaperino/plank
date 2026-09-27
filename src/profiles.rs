//! The `~/.plank/profiles/` root: where `/install-profile` puts a profile and
//! where `--profile` looks for one.
//!
//! Deliberately not a plugin scan root. [`crate::plugins::load_in`] never
//! visits it, so installing a profile adds nothing to an ordinary session — no
//! skills, no agents, no hooks, and in particular no MCP server starting
//! behind the user's back. The directory is read in exactly two situations:
//! listing the names `--profile` accepts, and loading the one profile
//! `--profile` named.

use std::path::{Path, PathBuf};

/// The directory `/install-profile` copies into and `--profile` falls back to.
///
/// Separate from every `plugins/` root on purpose: what lives here is an
/// identity you launch as, not a contribution to whatever session happens to
/// be running.
#[must_use]
pub fn dir(home: &Path) -> PathBuf {
    home.join(".plank").join("profiles")
}

/// Whether `name` is a single path segment, and therefore safe to join onto
/// the profiles root.
///
/// `--profile` takes its argument from the command line, so `../../etc` would
/// otherwise resolve to a directory outside the root and be loaded as a
/// plugin. Checked here rather than at the call site because every caller
/// joins, and one that forgets is a path traversal.
fn is_one_segment(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains('/') && !name.contains('\\')
}

/// Whether `dir` holds a manifest declaring a `profile` block.
///
/// Parses rather than greps: a `"profile"` substring in a description is not a
/// profile, and [`crate::profile::parse`] is the same reader activation uses,
/// so listing and activation cannot disagree about what qualifies.
fn has_profile_block(dir: &Path) -> bool {
    let Some(manifest) = crate::plugins::manifest_path(dir) else {
        return false;
    };
    let Ok(text) = std::fs::read_to_string(&manifest) else {
        return false;
    };
    crate::profile::parse(&text, dir).is_some()
}

/// The installed profile directory called `name`, if it is one.
///
/// `None` covers all of: no profiles root, no such directory, a directory that
/// is not a profile, and a name that is not a single path segment.
#[must_use]
pub fn find(home: &Path, name: &str) -> Option<PathBuf> {
    if !is_one_segment(name) {
        return None;
    }
    let candidate = dir(home).join(name);
    if !candidate.is_dir() || !has_profile_block(&candidate) {
        return None;
    }
    Some(candidate)
}

/// Every installed profile's name, sorted.
///
/// Reads manifests only; loads nothing.
#[must_use]
pub fn names(home: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir(home)) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir() && has_profile_block(p))
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect();
    out.sort();
    out
}

/// The file an install writes into the installed directory, holding the
/// source it came from, so `--profile <that source>` finds it again.
const SOURCE_FILE: &str = ".plank-source";

/// Whether a `--profile` argument names a source to install from rather than
/// an installed name: a path (anything with a `/`, or starting with `.` or
/// `~`) or `owner/repo:folder`. An installed name is one segment with neither.
#[must_use]
pub fn is_source(arg: &str) -> bool {
    arg.contains('/') || arg.contains(':') || arg.starts_with('.') || arg.starts_with('~')
}

/// The form a source is recorded and compared in: a directory that exists is
/// canonicalized, so `./hal` and its absolute path match; anything else is
/// kept as typed, trimmed.
#[must_use]
pub fn normalize_source(arg: &str) -> String {
    let arg = arg.trim();
    std::fs::canonicalize(arg)
        .ok()
        .filter(|p| p.is_dir())
        .map_or_else(|| arg.to_string(), |p| p.to_string_lossy().into_owned())
}

/// Records where the profile installed at `dest` came from. Best-effort: a
/// profile without the record still launches by name; only the by-source
/// lookup misses, which means one more install prompt.
pub fn record_source(dest: &Path, source: &str) {
    let _ = std::fs::write(dest.join(SOURCE_FILE), format!("{source}\n"));
}

/// The installed profile whose recorded source is `source` (already
/// normalized), as its name and directory.
#[must_use]
pub fn find_by_source(home: &Path, source: &str) -> Option<(String, PathBuf)> {
    names(home).into_iter().find_map(|name| {
        let dir = dir(home).join(&name);
        let recorded = std::fs::read_to_string(dir.join(SOURCE_FILE)).ok()?;
        (recorded.trim() == source).then_some((name, dir))
    })
}

/// The `version` a profile's manifest declares, as written.
#[must_use]
pub fn version_of(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(crate::plugins::manifest_path(dir)?).ok()?;
    manifest_version(&text)
}

/// The top-level `version` of a manifest's text.
#[must_use]
pub fn manifest_version(text: &str) -> Option<String> {
    match crate::tools::mcp::json_parse(text)?.get("version")? {
        crate::tools::mcp::Json::Str(v) if !v.trim().is_empty() => Some(v.trim().to_string()),
        _ => None,
    }
}

/// Whether an installed profile is behind the version its source offers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Freshness {
    /// Nothing newer, or no way to tell: launch what is installed.
    Current,
    /// The source declares a higher `MAJOR.MINOR.PATCH`.
    Newer {
        /// What is installed, or `unversioned`.
        installed: String,
        /// What the source offers.
        available: String,
    },
}

/// Compares an installed version with the one its source offers.
///
/// Only a source version that parses and is strictly higher counts as newer.
/// An unreadable or unparsable source (offline, no `version`, not
/// `MAJOR.MINOR.PATCH`) is [`Freshness::Current`], so a failed check never
/// costs a download. An installed copy with no version is behind any source
/// that has one.
#[must_use]
pub fn freshness(installed: Option<&str>, available: Option<&str>) -> Freshness {
    let Some(available) = available else {
        return Freshness::Current;
    };
    let Some(new) = crate::upgrade::parse_version(available) else {
        return Freshness::Current;
    };
    let old = installed.and_then(crate::upgrade::parse_version);
    if old.is_some_and(|old| old >= new) {
        return Freshness::Current;
    }
    Freshness::Newer {
        installed: installed.unwrap_or("unversioned").to_string(),
        available: available.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A fresh empty directory under the system temp dir, named for the test.
    fn tmpdir(tag: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("plank-profiles-{tag}-{}-{seq}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    /// Writes `<home>/.plank/profiles/<name>/` with a manifest and a prompt.
    /// `block` is the `profile` member's JSON text, or `None` for a manifest
    /// with no profile block at all.
    fn seed(home: &Path, name: &str, block: Option<&str>) -> PathBuf {
        let root = dir(home).join(name);
        std::fs::create_dir_all(root.join(".plank-plugin")).expect("mkdir");
        std::fs::write(root.join("prompt.md"), "You are a test profile.\n").expect("write");
        let manifest = match block {
            Some(b) => format!("{{\"name\":\"{name}\",\"profile\":{b}}}"),
            None => format!("{{\"name\":\"{name}\"}}"),
        };
        std::fs::write(root.join(".plank-plugin").join("plugin.json"), manifest).expect("write");
        root
    }

    const BLOCK: &str = r#"{"systemPrompt":"prompt.md"}"#;

    #[test]
    fn the_root_is_under_the_plank_home() {
        assert_eq!(
            dir(Path::new("/tmp/h")),
            Path::new("/tmp/h/.plank/profiles")
        );
    }

    #[test]
    fn find_resolves_a_profile_by_name() {
        let home = tmpdir("find");
        let want = seed(&home, "hal", Some(BLOCK));
        assert_eq!(find(&home, "hal"), Some(want));
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn find_ignores_a_directory_whose_manifest_has_no_profile_block() {
        let home = tmpdir("find-plain");
        seed(&home, "plain", None);
        assert_eq!(find(&home, "plain"), None);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn find_returns_nothing_for_a_name_that_is_not_installed() {
        let home = tmpdir("find-missing");
        seed(&home, "hal", Some(BLOCK));
        assert_eq!(find(&home, "nope"), None);
        let _ = std::fs::remove_dir_all(&home);
    }

    /// A name is one path segment. Without this, `--profile ../../etc` would
    /// have `find` join its way out of the profiles root entirely.
    #[test]
    fn find_refuses_a_name_that_is_not_a_single_segment() {
        let home = tmpdir("find-traversal");
        seed(&home, "hal", Some(BLOCK));
        assert_eq!(find(&home, "../hal"), None);
        assert_eq!(find(&home, "a/b"), None);
        assert_eq!(find(&home, ".."), None);
        assert_eq!(find(&home, ""), None);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn names_lists_only_profile_bearing_directories_sorted() {
        let home = tmpdir("names");
        seed(&home, "zeta", Some(BLOCK));
        seed(&home, "hal", Some(BLOCK));
        seed(&home, "plain", None);
        assert_eq!(names(&home), vec!["hal".to_string(), "zeta".to_string()]);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn names_is_empty_when_the_root_does_not_exist() {
        let home = tmpdir("names-empty");
        assert!(names(&home).is_empty());
        let _ = std::fs::remove_dir_all(&home);
    }

    /// The Claude Code spelling works too: an installed profile may have come
    /// from a repository that uses it.
    #[test]
    fn find_accepts_the_claude_manifest_spelling() {
        let home = tmpdir("find-claude");
        let root = dir(&home).join("hal");
        std::fs::create_dir_all(root.join(".claude-plugin")).expect("mkdir");
        std::fs::write(root.join("prompt.md"), "x\n").expect("write");
        std::fs::write(
            root.join(".claude-plugin").join("plugin.json"),
            format!("{{\"name\":\"hal\",\"profile\":{BLOCK}}}"),
        )
        .expect("write");
        assert_eq!(find(&home, "hal"), Some(root));
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn a_recorded_source_finds_its_installed_profile() {
        let home = tmpdir("by-source");
        let dir = seed(&home, "hal", Some(r#"{"systemPrompt":"p.md"}"#));
        assert_eq!(find_by_source(&home, "o/r:HAL"), None);
        record_source(&dir, "o/r:HAL");
        assert_eq!(
            find_by_source(&home, "o/r:HAL"),
            Some(("hal".to_string(), dir))
        );
        assert_eq!(find_by_source(&home, "o/r:OTHER"), None);
    }

    #[test]
    fn a_local_source_is_recorded_as_its_canonical_path() {
        let home = tmpdir("normalize");
        let dir = seed(&home, "hal", Some(r#"{"systemPrompt":"p.md"}"#));
        let canonical = std::fs::canonicalize(&dir).expect("canonical");
        let dotted = format!("{}/./", dir.display());
        assert_eq!(normalize_source(&dotted), canonical.to_string_lossy());
        assert_eq!(normalize_source(" o/r:HAL "), "o/r:HAL");
    }

    #[test]
    fn a_source_is_told_apart_from_an_installed_name() {
        for source in [
            "o/r:HAL",
            "./hal",
            "~/p/hal",
            "examples/profiles/chatbgt",
            "/abs",
        ] {
            assert!(is_source(source), "{source}");
        }
        for name in ["hal", "chat-bgt"] {
            assert!(!is_source(name), "{name}");
        }
    }

    #[test]
    fn only_a_strictly_higher_source_version_is_newer() {
        let newer = |i, a| freshness(i, a) != Freshness::Current;
        assert!(newer(Some("0.1.0"), Some("0.1.1")));
        assert!(newer(Some("0.9.9"), Some("1.0.0")));
        assert!(newer(None, Some("0.1.0")));
        assert!(newer(Some("garbage"), Some("0.1.0")));
        assert!(!newer(Some("0.1.1"), Some("0.1.1")));
        assert!(!newer(Some("0.2.0"), Some("0.1.9")));
        assert!(!newer(Some("0.1.0"), None));
        assert!(!newer(Some("0.1.0"), Some("latest")));
        assert_eq!(
            freshness(Some("0.1.0"), Some("0.2.0")),
            Freshness::Newer {
                installed: "0.1.0".to_string(),
                available: "0.2.0".to_string()
            }
        );
    }

    #[test]
    fn the_version_is_read_from_the_manifest() {
        let home = tmpdir("version");
        let root = dir(&home).join("hal");
        std::fs::create_dir_all(root.join(".plank-plugin")).expect("mkdir");
        std::fs::write(
            root.join(".plank-plugin/plugin.json"),
            r#"{"name":"hal","version":" 0.3.1 ","profile":{"systemPrompt":"p.md"}}"#,
        )
        .expect("write");
        assert_eq!(version_of(&root).as_deref(), Some("0.3.1"));
        assert_eq!(manifest_version(r#"{"name":"x"}"#), None);
        assert_eq!(manifest_version(r#"{"version":7}"#), None);
    }
}
