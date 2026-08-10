// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Version-transition maintenance for the `~/.plank` cache directory.
//!
//! Plank persists rebuildable state under `~/.plank`: the KV blob bodies
//! (`kvcache/*.kv_raw`, covering the Tier 1 system-prompt checkpoints, the
//! Tier 2 per-project ones, and the per-session KV payloads) with their
//! advisory `kvcache/*.json` metadata sidecars, and the pasted-image cache
//! (`image-cache/`).
//!
//! The **KV caches are never dropped on a version change.** Their validity is
//! self-describing, independent of the plank version:
//!
//! - *Content* — every blob body is prefixed with a textual fingerprint
//!   (`model ‖ system [‖ transcript]`); a changed prompt misses and rebuilds on
//!   its own.
//! - *Format* — the serialized snapshot carries its own
//!   `DS4_SESSION_PAYLOAD_MAGIC`/`DS4_SESSION_PAYLOAD_VERSION` plus layout
//!   invariants (context size, DS4 layout, ring/graph chunk shape). Loading an
//!   incompatible snapshot returns a graceful error ("unsupported session
//!   payload version"), so the warm-up rebuilds rather than trusting it. The
//!   metadata sidecars carry no trust weight at all, so they cannot make a
//!   version decision necessary either.
//!
//! Tying the KV cache to the plank version was actively harmful: two
//! co-installed versions (e.g. a homebrew build and a dev build) sharing one
//! `~/.plank/kvcache` mutually deleted the Tier 1 checkpoint on every switch,
//! forcing a ~130 MB cold rebuild even though the prompt was byte-identical. The
//! fingerprint and the blob format-version already provide every guarantee
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
use std::sync::OnceLock;

/// Name of the marker file under `~/.plank` recording the last version run.
const MARKER_FILE: &str = "version";

/// Name of the cache file under `~/.plank` recording the last update check:
/// line 1 is the check's Unix epoch (seconds), line 2 the latest known tag.
const UPDATE_CHECK_FILE: &str = "update-check";

/// GitHub Releases API endpoint for the latest published release.
#[cfg(not(test))]
const RELEASES_URL: &str = "https://api.github.com/repos/aovestdipaperino/plank/releases/latest";

/// Minimum seconds between network update checks (once per day).
const UPDATE_CHECK_INTERVAL_SECS: u64 = 24 * 60 * 60;

/// Bounded timeout for the update-check network request. Startup must never
/// stall on a slow or unreachable network, so this is deliberately short.
#[cfg(not(test))]
const UPDATE_CHECK_TIMEOUT_SECS: u64 = 3;

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
    // The KV caches (*.kv_raw and their *.json sidecars) self-validate by
    // fingerprint and snapshot format-version, so no version transition touches
    // them. Only the image cache, which has no such guard, is dropped on a major
    // transition.
    if transition == Transition::Major {
        let _ = std::fs::remove_dir_all(plank_dir.join("image-cache"));
    }
    if transition != Transition::None || previous.is_none() {
        let _ = std::fs::write(&marker, current);
    }
    transition
}

// ---------------------------------------------------------------------------
// Update-available detection (issue #56)
//
// Complementary to the post-upgrade `Transition` maintenance above: this checks
// the GitHub Releases API for a newer release than the running binary and, when
// one exists, surfaces a non-intrusive hint. It is best-effort and offline-safe
// — any network failure or timeout is silent — and rate-limited to at most one
// network request per day via a cache file under `~/.plank`.

/// True when `latest` parses to a strictly greater version than `current`.
///
/// Either version being unparseable (e.g. a dev build) yields `false`: we never
/// nag when we cannot be sure a real newer release exists.
#[must_use]
pub fn is_newer(current: &str, latest: &str) -> bool {
    match (parse_version(current), parse_version(latest)) {
        (Some(cur), Some(lat)) => lat > cur,
        _ => false,
    }
}

/// Whether a network check is due, given the last check's epoch (if any), the
/// current epoch, and the minimum interval. A missing/unreadable last check is
/// always due; the subtraction saturates so a clock that went backwards is due.
#[must_use]
fn check_due(last: Option<u64>, now: u64, interval: u64) -> bool {
    match last {
        None => true,
        Some(t) => now.saturating_sub(t) >= interval,
    }
}

/// Extracts the `tag_name` string from a GitHub `releases/latest` JSON body.
///
/// Deliberately a tiny hand-rolled scan rather than a full JSON parse: the field
/// is a flat top-level string and this keeps the check dependency-free and
/// robust to the rest of the (large) payload.
#[must_use]
fn parse_latest_tag(json: &str) -> Option<String> {
    let key = "\"tag_name\"";
    let after = &json[json.find(key)? + key.len()..];
    let after = &after[after.find(':')? + 1..];
    let start = after.find('"')? + 1;
    let rest = &after[start..];
    let end = rest.find('"')?;
    let tag = rest[..end].trim();
    // Tags are conventionally `vX.Y.Z`; normalize to the bare version so
    // `parse_version` accepts it.
    let ver = tag
        .strip_prefix('v')
        .or_else(|| tag.strip_prefix('V'))
        .unwrap_or(tag);
    (!ver.is_empty()).then(|| ver.to_string())
}

/// Current Unix time in whole seconds, or 0 if the clock predates the epoch.
fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Reads the update-check cache: `(last_check_epoch, latest_known_version)`.
/// A missing, unreadable, or malformed file reads as "never checked".
fn read_check_cache(path: &Path) -> (Option<u64>, Option<String>) {
    let Ok(text) = std::fs::read_to_string(path) else {
        return (None, None);
    };
    let mut lines = text.lines();
    let when = lines.next().and_then(|l| l.trim().parse::<u64>().ok());
    let latest = lines
        .next()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    (when, latest)
}

/// Writes the update-check cache. Failures are ignored — the worst case is a
/// repeated check next launch.
fn write_check_cache(path: &Path, when: u64, latest: &str) {
    let _ = std::fs::write(path, format!("{when}\n{latest}\n"));
}

/// Fetches the latest release tag from GitHub, bounded by
/// [`UPDATE_CHECK_TIMEOUT_SECS`]. Returns `None` on any error — offline,
/// timeout, HTTP error, or unparseable body — so callers stay silent.
#[cfg(not(test))]
fn fetch_latest_version() -> Option<String> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(
            UPDATE_CHECK_TIMEOUT_SECS,
        )))
        .build()
        .new_agent();
    let mut resp = agent
        .get(RELEASES_URL)
        .header("Accept", "application/vnd.github+json")
        .header("User-Agent", concat!("plank/", env!("CARGO_PKG_VERSION")))
        .call()
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body = resp.body_mut().read_to_string().ok()?;
    parse_latest_tag(&body)
}

/// Test builds never touch the network; the check logic is exercised directly
/// against `decide_update_notice` with synthetic inputs.
#[cfg(test)]
fn fetch_latest_version() -> Option<String> {
    None
}

/// Pure decision: given the cache state, the clock, and the running version,
/// decide whether a network fetch is due and what (if any) notice to surface.
///
/// Returns `(refreshed_cache, notice)` where `refreshed_cache` is
/// `Some((epoch, latest))` to persist when a fetch happened, else `None`.
/// `fetch` is injected so tests can drive it without the network.
fn decide_update_notice(
    current: &str,
    last: Option<u64>,
    cached_latest: Option<String>,
    now: u64,
    fetch: impl FnOnce() -> Option<String>,
) -> (Option<(u64, String)>, Option<String>) {
    let mut latest = cached_latest;
    let mut refreshed = None;
    if check_due(last, now, UPDATE_CHECK_INTERVAL_SECS) {
        // Prefer a fresh fetch; on failure keep the previously known latest so
        // we still surface a hint learned earlier while offline. Either way we
        // record the attempt time so the check stays rate-limited to once/day.
        if let Some(fetched) = fetch() {
            latest = Some(fetched);
        }
        let to_store = latest.clone().unwrap_or_default();
        refreshed = Some((now, to_store));
    }
    let notice = latest
        .filter(|l| is_newer(current, l))
        .map(|l| update_message(current, &l));
    (refreshed, notice)
}

/// The one-line, non-intrusive upgrade hint.
#[must_use]
fn update_message(current: &str, latest: &str) -> String {
    format!(
        "plank {latest} is available (you have {current}) — see \
         https://github.com/aovestdipaperino/plank/releases"
    )
}

/// Best-effort update check run at startup. Reads the cache under `plank_dir`,
/// performs at most one network request per day (bounded, silent on failure),
/// and returns a hint string when a newer release exists.
///
/// Never blocks startup perceptibly and never surfaces errors: a missing
/// network, a slow endpoint, or a broken cache all resolve to `None` (or a
/// stale-but-valid hint from a prior check).
#[must_use]
pub fn check_for_update(plank_dir: &Path, current: &str) -> Option<String> {
    let path = plank_dir.join(UPDATE_CHECK_FILE);
    let (last, cached_latest) = read_check_cache(&path);
    let now = now_epoch();
    let (refreshed, notice) =
        decide_update_notice(current, last, cached_latest, now, fetch_latest_version);
    if let Some((when, latest)) = refreshed {
        write_check_cache(&path, when, &latest);
    }
    notice
}

/// Process-global stash for the startup update notice, so it can be computed
/// once in `main` and rendered by whichever front-end (plain REPL or TUI) comes
/// up. Set once before the UI starts; read once by the front-end.
static UPDATE_NOTICE: OnceLock<Option<String>> = OnceLock::new();

/// Records the startup update notice (or its absence). First call wins.
pub fn set_update_notice(notice: Option<String>) {
    let _ = UPDATE_NOTICE.set(notice);
}

/// The startup update notice, if one was recorded and a newer release exists.
#[must_use]
pub fn update_notice() -> Option<&'static str> {
    UPDATE_NOTICE.get().and_then(|o| o.as_deref())
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
        // Deliberately the *legacy* blob names. The claim under test is that no
        // version transition unlinks anything in `kvcache/`, and it has to hold
        // for files an older plank left behind just as much as for `.kv_raw`
        // bodies this one writes.
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

    // --- update-available detection (issue #56) -----------------------------

    #[test]
    fn is_newer_compares_semver_and_ignores_junk() {
        assert!(is_newer("2.6.0", "2.6.1"));
        assert!(is_newer("2.6.0", "2.7.0"));
        assert!(is_newer("2.6.0", "3.0.0"));
        assert!(!is_newer("2.6.0", "2.6.0"));
        assert!(!is_newer("2.6.1", "2.6.0")); // older release, no nag
        // Pre-release/build suffixes are ignored by the parser.
        assert!(!is_newer("2.6.0", "2.6.0-beta.1"));
        assert!(is_newer("2.6.0", "2.7.0-beta.1"));
        // Unparseable either side never nags.
        assert!(!is_newer("dev", "2.7.0"));
        assert!(!is_newer("2.6.0", "garbage"));
    }

    #[test]
    fn check_due_respects_the_interval() {
        let day = UPDATE_CHECK_INTERVAL_SECS;
        assert!(check_due(None, 1000, day), "never checked -> due");
        assert!(check_due(Some(0), day, day), "exactly one interval -> due");
        assert!(check_due(Some(0), day + 1, day));
        assert!(!check_due(Some(1000), 1000 + day - 1, day), "too soon");
        assert!(!check_due(Some(1000), 1000, day));
        // Clock went backwards: saturating sub keeps it not-due, never panics.
        assert!(!check_due(Some(5000), 1000, day));
    }

    #[test]
    fn parse_latest_tag_extracts_and_normalizes() {
        assert_eq!(
            parse_latest_tag(r#"{"url":"x","tag_name":"v2.7.0","name":"2.7.0"}"#).as_deref(),
            Some("2.7.0")
        );
        assert_eq!(
            parse_latest_tag(r#"{ "tag_name" : "2.7.0" }"#).as_deref(),
            Some("2.7.0")
        );
        assert_eq!(
            parse_latest_tag(r#"{"tag_name":"V3.0.0-beta.1"}"#).as_deref(),
            Some("3.0.0-beta.1")
        );
        assert_eq!(parse_latest_tag(r#"{"name":"no tag here"}"#), None);
        assert_eq!(parse_latest_tag("not json"), None);
        assert_eq!(parse_latest_tag(r#"{"tag_name":""}"#), None);
    }

    #[test]
    fn decide_skips_fetch_when_not_due_but_still_hints_from_cache() {
        let now = 10_000_000;
        // Checked just now; a newer version is already cached.
        let (refreshed, notice) =
            decide_update_notice("2.6.0", Some(now), Some("2.7.0".to_string()), now, || {
                panic!("must not fetch when not due")
            });
        assert!(refreshed.is_none(), "no cache rewrite when not due");
        assert!(notice.unwrap().contains("2.7.0"));
    }

    #[test]
    fn decide_fetches_when_due_and_records_attempt() {
        let now = 10_000_000;
        let (refreshed, notice) =
            decide_update_notice("2.6.0", None, None, now, || Some("2.7.0".to_string()));
        assert_eq!(refreshed, Some((now, "2.7.0".to_string())));
        assert!(notice.unwrap().contains("2.7.0"));
    }

    #[test]
    fn decide_offline_keeps_prior_hint_and_still_rate_limits() {
        let now = 10_000_000;
        // Due for a check, but the network fetch fails; a prior check learned
        // 2.7.0, so we still surface it — and we record `now` so the failed
        // attempt still counts against the once-per-day budget.
        let (refreshed, notice) =
            decide_update_notice("2.6.0", Some(0), Some("2.7.0".to_string()), now, || None);
        assert_eq!(refreshed, Some((now, "2.7.0".to_string())));
        assert!(notice.unwrap().contains("2.7.0"));
    }

    #[test]
    fn decide_no_notice_when_current_is_latest() {
        let now = 10_000_000;
        let (_r, notice) =
            decide_update_notice("2.7.0", None, None, now, || Some("2.7.0".to_string()));
        assert_eq!(notice, None);
    }

    #[test]
    fn check_cache_round_trips_and_tolerates_garbage() {
        let dir = tmp("update-cache");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(UPDATE_CHECK_FILE);
        write_check_cache(&path, 12345, "2.7.0");
        assert_eq!(read_check_cache(&path), (Some(12345), Some("2.7.0".into())));
        // Empty latest reads back as None.
        write_check_cache(&path, 12345, "");
        assert_eq!(read_check_cache(&path), (Some(12345), None));
        // Missing file -> never checked.
        assert_eq!(read_check_cache(&dir.join("nope")), (None, None));
        // Garbage timestamp is ignored.
        std::fs::write(&path, "notanumber\n2.7.0\n").unwrap();
        assert_eq!(read_check_cache(&path), (None, Some("2.7.0".into())));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn check_for_update_is_silent_without_network_in_tests() {
        // In test builds `fetch_latest_version` returns None, so a fresh cache
        // yields no notice and startup stays silent.
        let dir = tmp("update-check-e2e");
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(check_for_update(&dir, "2.6.0"), None);
        // A cache that already knows a newer version surfaces the hint even with
        // no network (and without re-checking, since it was just written).
        let now = now_epoch();
        write_check_cache(&dir.join(UPDATE_CHECK_FILE), now, "9.9.9");
        assert!(check_for_update(&dir, "2.6.0").unwrap().contains("9.9.9"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
