//! Version-transition maintenance for the `~/.plank` cache directory.
//!
//! Plank persists rebuildable state under `~/.plank`: the system-prompt KV
//! checkpoint (`kvcache/sysprompt.kv`) and the pasted-image cache
//! (`image-cache/`). Their on-disk formats follow the binary, so after an
//! upgrade a stale file could be read by code that no longer understands it.
//! Instead of asking the user to clean up, the version delta itself encodes
//! the maintenance the new binary performs on first launch (the tokensave
//! "maintenance-based versioning" idea):
//!
//! - **patch** bump — nothing to do; only the recorded marker advances.
//! - **minor** bump — drop the sysprompt KV checkpoint so it is rebuilt.
//! - **major** bump, downgrade, or unreadable marker — drop the sysprompt
//!   checkpoint *and* the image cache.
//!
//! Session transcripts (`kvcache/*.session`) are user data and are never
//! touched. Everything removed here is rebuilt automatically on demand, so a
//! wrong classification costs one warm-up, never data.

use std::path::Path;

/// Name of the marker file under `~/.plank` recording the last version run.
const MARKER_FILE: &str = "version";

/// Maintenance implied by a version transition, in increasing order of work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Transition {
    /// Same version: nothing to do at all.
    None,
    /// Patch bump: advance the marker only.
    Patch,
    /// Minor bump: rebuild the system-prompt KV checkpoint.
    Minor,
    /// Major bump, downgrade, or unknown previous version: full cache reset.
    Major,
}

/// Parses `MAJOR.MINOR.PATCH`, ignoring any `-pre`/`+build` suffix.
fn parse_version(v: impl AsRef<str>) -> Option<(u64, u64, u64)> {
    let v = v.as_ref().trim();
    let core = v.split(['-', '+']).next()?;
    let mut it = core.split('.');
    let major = it.next()?.parse().ok()?;
    let minor = it.next()?.parse().ok()?;
    let patch = it.next()?.parse().ok()?;
    if it.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}

/// Classifies the transition from `previous` to `current`.
///
/// A missing or malformed `previous` classifies as [`Transition::Major`]:
/// a full cache reset is cheap, guessing wrong is not. Downgrades are also
/// major, since older formats cannot be assumed forward-compatible.
#[must_use]
pub fn classify(previous: Option<&str>, current: &str) -> Transition {
    let Some(cur) = parse_version(current) else {
        return Transition::None; // dev build with an odd version: leave caches alone
    };
    let Some(prev) = previous.and_then(parse_version) else {
        return Transition::Major;
    };
    if prev == cur {
        Transition::None
    } else if prev > cur || prev.0 != cur.0 {
        Transition::Major
    } else if prev.1 != cur.1 {
        Transition::Minor
    } else {
        Transition::Patch
    }
}

/// Runs startup maintenance on `plank_dir` (normally `~/.plank`).
///
/// Reads the version marker, classifies the transition to `current`, performs
/// the implied cleanup, and rewrites the marker. A `plank_dir` that does not
/// exist yet is a fresh install: the marker is written and nothing is removed.
/// All filesystem failures are ignored — the caches are rebuildable and the
/// worst case is repeating the maintenance next launch.
///
/// Returns the classified transition so the caller can log what happened.
#[must_use = "log the transition so cache resets are visible"]
pub fn run_startup_maintenance(plank_dir: &Path, current: &str) -> Transition {
    let marker = plank_dir.join(MARKER_FILE);
    if !plank_dir.is_dir() {
        // Fresh install: nothing cached yet, just record the version.
        if std::fs::create_dir_all(plank_dir).is_ok() {
            let _ = std::fs::write(&marker, current);
        }
        return Transition::None;
    }
    let previous = std::fs::read_to_string(&marker).ok();
    let transition = classify(previous.as_deref(), current);
    match transition {
        Transition::None | Transition::Patch => {}
        Transition::Minor => {
            let _ = std::fs::remove_file(plank_dir.join("kvcache").join("sysprompt.kv"));
        }
        Transition::Major => {
            let _ = std::fs::remove_file(plank_dir.join("kvcache").join("sysprompt.kv"));
            let _ = std::fs::remove_dir_all(plank_dir.join("image-cache"));
        }
    }
    if transition != Transition::None || previous.is_none() {
        let _ = std::fs::write(&marker, current);
    }
    transition
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_handles_core_and_suffixes() {
        assert_eq!(parse_version("1.2.3"), Some((1, 2, 3)));
        assert_eq!(parse_version("10.0.1-beta.2"), Some((10, 0, 1)));
        assert_eq!(parse_version("1.2.3+abc"), Some((1, 2, 3)));
        assert_eq!(parse_version("1.2"), None);
        assert_eq!(parse_version("1.2.3.4"), None);
        assert_eq!(parse_version("x.y.z"), None);
    }

    #[test]
    fn classify_tiers() {
        assert_eq!(classify(Some("1.2.3"), "1.2.3"), Transition::None);
        assert_eq!(classify(Some("1.2.3"), "1.2.4"), Transition::Patch);
        assert_eq!(classify(Some("1.2.3"), "1.3.0"), Transition::Minor);
        assert_eq!(classify(Some("1.2.3"), "2.0.0"), Transition::Major);
        // Downgrade and unknown previous are major.
        assert_eq!(classify(Some("2.0.0"), "1.9.9"), Transition::Major);
        assert_eq!(classify(Some("1.2.4"), "1.2.3"), Transition::Major);
        assert_eq!(classify(None, "1.2.3"), Transition::Major);
        assert_eq!(classify(Some("garbage"), "1.2.3"), Transition::Major);
        // Unparseable current (dev builds) never triggers maintenance.
        assert_eq!(classify(Some("1.2.3"), "dev"), Transition::None);
    }

    fn setup(dir: &Path, prev: &str) {
        std::fs::create_dir_all(dir.join("kvcache")).unwrap();
        std::fs::create_dir_all(dir.join("image-cache")).unwrap();
        std::fs::write(dir.join("kvcache").join("sysprompt.kv"), b"kv").unwrap();
        std::fs::write(dir.join("kvcache").join("abc.session"), b"s").unwrap();
        std::fs::write(dir.join("image-cache").join("img.png"), b"p").unwrap();
        std::fs::write(dir.join(MARKER_FILE), prev).unwrap();
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("plank-upgrade-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn fresh_install_writes_marker_only() {
        let dir = tmp("fresh");
        assert_eq!(run_startup_maintenance(&dir, "1.0.0"), Transition::None);
        assert_eq!(
            std::fs::read_to_string(dir.join(MARKER_FILE)).unwrap(),
            "1.0.0"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn minor_drops_sysprompt_keeps_sessions_and_images() {
        let dir = tmp("minor");
        setup(&dir, "1.2.3");
        assert_eq!(run_startup_maintenance(&dir, "1.3.0"), Transition::Minor);
        assert!(!dir.join("kvcache").join("sysprompt.kv").exists());
        assert!(dir.join("kvcache").join("abc.session").exists());
        assert!(dir.join("image-cache").join("img.png").exists());
        assert_eq!(
            std::fs::read_to_string(dir.join(MARKER_FILE)).unwrap(),
            "1.3.0"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn major_also_drops_image_cache_never_sessions() {
        let dir = tmp("major");
        setup(&dir, "1.2.3");
        assert_eq!(run_startup_maintenance(&dir, "2.0.0"), Transition::Major);
        assert!(!dir.join("kvcache").join("sysprompt.kv").exists());
        assert!(!dir.join("image-cache").exists());
        assert!(dir.join("kvcache").join("abc.session").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_and_same_version_touch_nothing() {
        let dir = tmp("patch");
        setup(&dir, "1.2.3");
        assert_eq!(run_startup_maintenance(&dir, "1.2.4"), Transition::Patch);
        assert!(dir.join("kvcache").join("sysprompt.kv").exists());
        assert_eq!(
            std::fs::read_to_string(dir.join(MARKER_FILE)).unwrap(),
            "1.2.4"
        );
        assert_eq!(run_startup_maintenance(&dir, "1.2.4"), Transition::None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
