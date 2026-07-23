// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Version-transition maintenance for the `~/.plank` cache directory.
//!
//! Plank persists rebuildable state under `~/.plank`: the system-prompt KV
//! checkpoint (`kvcache/sysprompt.kv`), the per-session KV payload sidecars
//! (`kvcache/*.payload`), and the pasted-image cache (`image-cache/`).
//!
//! The **KV caches are never dropped on a version change.** Their validity is
//! self-describing, independent of the plank version:
//!
//! - *Content* — both the sysprompt checkpoint and the payload sidecars are
//!   prefixed with a textual fingerprint (`model ‖ system [‖ transcript]`); a
//!   changed prompt misses and rebuilds on its own.
//! - *Format* — the serialized snapshot carries its own
//!   `DS4_SESSION_PAYLOAD_MAGIC`/`DS4_SESSION_PAYLOAD_VERSION` plus layout
//!   invariants (context size, DS4 layout, ring/graph chunk shape). Loading an
//!   incompatible snapshot returns a graceful error ("unsupported session
//!   payload version"), so the warm-up rebuilds rather than trusting it.
//!
//! Tying the KV cache to the plank version was actively harmful: two
//! co-installed versions (e.g. a homebrew build and a dev build) sharing one
//! `~/.plank/kvcache` mutually deleted `sysprompt.kv` on every switch, forcing
//! a ~130 MB cold rebuild even though the prompt was byte-identical. The
//! fingerprint and the payload format-version already provide every guarantee
//! the version delta was standing in for.
//!
//! Only the image cache remains version-gated, since it has no such
//! self-validation:
//!
//! - **patch / minor** bump — nothing to do; only the recorded marker advances.
//! - **major** bump, downgrade, or unreadable marker — drop the image cache.
//!
//! Session transcripts (`kvcache/*.kv`) are user data and are never touched.
//! Everything removed here is rebuilt automatically on demand, so a wrong
//! classification costs one warm-up, never data.

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
    // The KV caches (sysprompt.kv, *.payload) self-validate by fingerprint and
    // snapshot format-version, so no version transition touches them. Only the
    // image cache, which has no such guard, is dropped on a major transition.
    if transition == Transition::Major {
        let _ = std::fs::remove_dir_all(plank_dir.join("image-cache"));
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
        std::fs::write(dir.join("kvcache").join("abc.payload"), b"p").unwrap();
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
    fn minor_keeps_all_kv_caches_and_images() {
        // The KV caches self-validate (fingerprint + snapshot format-version),
        // so a version bump never drops them — this is the fix for two
        // co-installed versions churning sysprompt.kv on every switch.
        let dir = tmp("minor");
        setup(&dir, "1.2.3");
        assert_eq!(run_startup_maintenance(&dir, "1.3.0"), Transition::Minor);
        assert!(dir.join("kvcache").join("sysprompt.kv").exists());
        assert!(dir.join("kvcache").join("abc.payload").exists());
        assert!(dir.join("kvcache").join("abc.session").exists());
        assert!(dir.join("image-cache").join("img.png").exists());
        assert_eq!(
            std::fs::read_to_string(dir.join(MARKER_FILE)).unwrap(),
            "1.3.0"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn major_drops_only_image_cache_keeping_kv_and_sessions() {
        let dir = tmp("major");
        setup(&dir, "1.2.3");
        assert_eq!(run_startup_maintenance(&dir, "2.0.0"), Transition::Major);
        // KV caches survive even a major bump — they self-validate on load.
        assert!(dir.join("kvcache").join("sysprompt.kv").exists());
        assert!(dir.join("kvcache").join("abc.payload").exists());
        assert!(dir.join("kvcache").join("abc.session").exists());
        // Only the image cache, which has no self-validation, is dropped.
        assert!(!dir.join("image-cache").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patch_and_same_version_touch_nothing() {
        let dir = tmp("patch");
        setup(&dir, "1.2.3");
        assert_eq!(run_startup_maintenance(&dir, "1.2.4"), Transition::Patch);
        assert!(dir.join("kvcache").join("sysprompt.kv").exists());
        assert!(dir.join("kvcache").join("abc.payload").exists());
        assert_eq!(
            std::fs::read_to_string(dir.join(MARKER_FILE)).unwrap(),
            "1.2.4"
        );
        assert_eq!(run_startup_maintenance(&dir, "1.2.4"), Transition::None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
