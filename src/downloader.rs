//! The background model downloader: its state file, its cancel flag, and (from
//! Task 4 on) its fetch loop.
//!
//! The downloader is a *detached* child process, so nothing here may assume a
//! live parent. Progress is published by rewriting a small JSON file, and
//! cancellation arrives as a file the helper polls — both chosen over
//! in-memory channels or signals precisely because the plank that started the
//! download is free to exit at any moment.

use std::path::PathBuf;

/// How long a state file may go unrefreshed before a *dead* writer makes it
/// stale. A live writer is never stale however old its state: a helper stalled
/// on a slow socket still owns the download.
pub const STALE_AFTER_SECS: u64 = 10;

/// What the helper is doing right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    /// Re-reading an existing `.part` to rebuild the SHA-256 state.
    Rehashing,
    /// Streaming bytes.
    Downloading,
    /// Comparing the finished digest to the manifest.
    Verifying,
    /// Every artifact is verified and waiting in staging for the next launch.
    Staged,
    /// Gave up. `State::error` says why.
    Failed,
    /// Stopped on request.
    Cancelled,
}

/// The helper's published progress.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct State {
    /// The helper's pid, so a reader can tell a stalled helper from a dead one.
    pub pid: u32,
    /// Manifest version being installed.
    pub version: u32,
    /// Artifact kind in flight.
    pub current: String,
    /// 1-based position of `current` in the set.
    pub index: usize,
    /// How many artifacts the whole job covers.
    pub of: usize,
    /// Bytes of `current` on disk.
    pub done_bytes: u64,
    /// Expected length of `current`.
    pub total_bytes: u64,
    /// Recent throughput, bytes per second.
    pub rate_bps: u64,
    /// What the helper is doing.
    pub phase: Phase,
    /// Failure detail, when `phase` is [`Phase::Failed`].
    pub error: Option<String>,
    /// Unix seconds when this was written.
    pub updated: u64,
}

/// How to treat the partial files when cancelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cancel {
    /// Leave the `.part` files; the next launch resumes them.
    Keep,
    /// Delete the `.part` files and the staging directory.
    Delete,
}

/// Current Unix time in whole seconds, or 0 if the clock predates the epoch.
#[must_use]
pub fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Whether `pid` names a live process.
///
/// `kill(pid, 0)` performs the permission and existence checks without
/// delivering anything. Pid 0 means "the whole process group" to `kill`, which
/// is emphatically not the question being asked, so it is refused up front.
#[must_use]
pub fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    let pid = libc::pid_t::try_from(pid).unwrap_or(-1);
    // SAFETY: signal 0 delivers nothing; this is a pure existence probe.
    let rc = unsafe { libc::kill(pid, 0) };
    rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// `~/.plank/downloads/state.json`.
#[must_use]
pub fn state_path() -> PathBuf {
    crate::manifest::downloads_dir().join("state.json")
}

/// `~/.plank/downloads/cancel`.
#[must_use]
pub fn cancel_path() -> PathBuf {
    crate::manifest::downloads_dir().join("cancel")
}

/// Publishes `state`, atomically.
///
/// Written to a sibling temp file and renamed, so a reader polling twice a
/// second never sees a half-written JSON object.
///
/// # Errors
/// Propagates any filesystem error from creating the directory, writing the
/// temp file, or renaming it.
pub fn write_state(state: &State) -> std::io::Result<()> {
    let path = state_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string(state)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, &path)
}

/// Reads the published state, whatever its age. `None` when absent or corrupt.
#[must_use]
pub fn read_state() -> Option<State> {
    serde_json::from_str(&std::fs::read_to_string(state_path()).ok()?).ok()
}

/// Whether `state` should be ignored: too old *and* written by a process that
/// is gone.
///
/// Both conditions are required. Age alone would hide a helper that is merely
/// blocked on a slow read or a slow socket, not dead; a dead pid alone would
/// hide the last state a finished helper published, which is exactly the
/// "staged" or "failed" line worth showing after it exits. The age
/// subtraction is saturating so a clock that jumps backwards produces zero,
/// never an underflowed huge age that would wrongly call a fresh state stale.
#[must_use]
pub fn is_stale(state: &State, now: u64, alive: &dyn Fn(u32) -> bool) -> bool {
    now.saturating_sub(state.updated) > STALE_AFTER_SECS && !alive(state.pid)
}

/// The published state, if it is worth trusting.
#[must_use]
pub fn live_state() -> Option<State> {
    let state = read_state()?;
    (!is_stale(&state, now_epoch(), &pid_alive)).then_some(state)
}

/// Asks the running helper to stop.
///
/// # Errors
/// Propagates any filesystem error from creating the directory or writing the
/// flag.
pub fn request_cancel(how: Cancel) -> std::io::Result<()> {
    let path = cancel_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        path,
        match how {
            Cancel::Keep => "keep",
            Cancel::Delete => "delete",
        },
    )
}

/// Parses a cancel-flag body. Anything unrecognized is `None` — a corrupt flag
/// must not be guessed into `Delete`.
#[must_use]
fn parse_cancel(text: &str) -> Option<Cancel> {
    match text.trim() {
        "keep" => Some(Cancel::Keep),
        "delete" => Some(Cancel::Delete),
        _ => None,
    }
}

/// The pending cancel request, if any.
#[must_use]
pub fn read_cancel() -> Option<Cancel> {
    parse_cancel(&std::fs::read_to_string(cancel_path()).ok()?)
}

/// Clears any pending cancel request. Best-effort: a stale flag that cannot be
/// removed is caught by the helper's own startup, which clears it before work
/// begins.
pub fn clear_cancel() {
    let _ = std::fs::remove_file(cancel_path());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a_state() -> State {
        State {
            pid: 4711,
            version: 3,
            current: "main".to_string(),
            index: 1,
            of: 3,
            done_bytes: 38,
            total_bytes: 100,
            rate_bps: 12,
            phase: Phase::Downloading,
            error: None,
            updated: 1_000_000,
        }
    }

    #[test]
    fn state_round_trips_through_json() {
        let json = serde_json::to_string(&a_state()).expect("serializes");
        let back: State = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back, a_state());
    }

    #[test]
    fn every_phase_round_trips() {
        for phase in [
            Phase::Rehashing,
            Phase::Downloading,
            Phase::Verifying,
            Phase::Staged,
            Phase::Failed,
            Phase::Cancelled,
        ] {
            let json = serde_json::to_string(&phase).expect("serializes");
            let back: Phase = serde_json::from_str(&json).expect("deserializes");
            assert_eq!(back, phase);
        }
    }

    #[test]
    fn a_fresh_state_is_never_stale() {
        let st = a_state();
        assert!(!is_stale(&st, st.updated, &|_| false));
        assert!(!is_stale(&st, st.updated + 3, &|_| false));
    }

    #[test]
    fn an_old_state_with_a_live_pid_is_not_stale() {
        // A helper stalled on a slow read still owns the download. Killing the
        // display because it went quiet for eleven seconds would be wrong.
        let st = a_state();
        assert!(!is_stale(&st, st.updated + 600, &|_| true));
    }

    #[test]
    fn an_old_state_with_a_dead_pid_is_stale() {
        let st = a_state();
        assert!(is_stale(&st, st.updated + STALE_AFTER_SECS + 1, &|_| false));
    }

    #[test]
    fn a_clock_that_went_backwards_is_not_stale() {
        // Saturating arithmetic, not a panic or an underflowed huge age.
        let st = a_state();
        assert!(!is_stale(&st, st.updated - 500, &|_| false));
    }

    #[test]
    fn cancel_words_parse_and_anything_else_does_not() {
        assert_eq!(parse_cancel("keep"), Some(Cancel::Keep));
        assert_eq!(parse_cancel("delete"), Some(Cancel::Delete));
        assert_eq!(parse_cancel("  delete\n"), Some(Cancel::Delete));
        assert_eq!(parse_cancel(""), None);
        assert_eq!(parse_cancel("rm -rf"), None);
    }

    #[test]
    fn our_own_pid_is_alive_and_pid_zero_is_not() {
        assert!(pid_alive(std::process::id()));
        assert!(!pid_alive(0));
    }
}
