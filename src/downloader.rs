//! The background model downloader: its state file, its cancel flag, and (from
//! Task 4 on) its fetch loop.
//!
//! The downloader is a *detached* child process, so nothing here may assume a
//! live parent. Progress is published by rewriting a small JSON file, and
//! cancellation arrives as a file the helper polls — both chosen over
//! in-memory channels or signals precisely because the plank that started the
//! download is free to exit at any moment.

use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

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

/// Read buffer, matching `download::fetch`'s 1 MiB.
const CHUNK: usize = 1 << 20;
/// How often the helper republishes its state. Twice a second is fast enough
/// for a status bar and slow enough that the rename costs nothing.
const PUBLISH_EVERY: std::time::Duration = std::time::Duration::from_millis(500);

/// In-flight bytes for `kind`.
#[must_use]
pub fn part_path(kind: &str) -> PathBuf {
    crate::manifest::staging_dir().join(format!("{kind}.part"))
}

/// Verified-but-not-yet-installed bytes for `kind`.
#[must_use]
pub fn staged_path(kind: &str) -> PathBuf {
    crate::manifest::staging_dir().join(format!("{kind}.gguf"))
}

/// Lowercase hex.
#[must_use]
pub fn hex(digest: &[u8]) -> String {
    use std::fmt::Write as _;
    digest.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// Feeds an existing `.part` back through `hasher`, returning its length.
///
/// This is what makes resume and incremental hashing coexist: `sha2` has no
/// serializable state, so rather than invent a sidecar format (and a new way to
/// be silently wrong), the bytes already on local disk are read once more. It
/// is disk-bound, it needs no format, and it validates what it hashes on the
/// way past.
///
/// An absent `.part` is 0 bytes and an untouched hasher, not an error.
///
/// # Errors
/// Propagates read errors other than "not found".
pub fn rehash(part: &Path, hasher: &mut Sha256) -> std::io::Result<u64> {
    let mut file = match File::open(part) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e),
    };
    let mut buf = vec![0u8; CHUNK];
    let mut total = 0u64;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            return Ok(total);
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
}

/// How a job ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Every artifact is verified and staged.
    Verified,
    /// Stopped on request.
    Cancelled,
    /// Gave up, with a reason.
    Failed(String),
}

/// A fetcher: given a URL and a resume offset, returns an owned byte stream or
/// a failure message. Injected so tests drive [`run_job`] without a socket.
type Fetcher = dyn Fn(&str, u64) -> Result<Box<dyn Read + Send>, String>;

/// Opens `url` at byte `offset`, resuming where the server allows it.
///
/// Injected into [`run_job`] so tests drive the whole loop without a socket,
/// mirroring `download::fetch_tree`'s `#[cfg(test)]` stub.
///
/// # Errors
/// Returns a message on any transport or status failure.
pub fn http_fetch(url: &str, offset: u64) -> Result<Box<dyn Read + Send>, String> {
    let mut request = ureq::get(url);
    if offset > 0 {
        request = request.header("Range", format!("bytes={offset}-"));
    }
    let response = request
        .call()
        .map_err(|e| format!("download failed: {e}"))?;
    // A server that ignores Range answers 200 with the whole body. The caller
    // detects that by the offset it asked for versus what it gets, so signal it
    // by refusing: `run_job` truncates and restarts rather than appending a
    // second copy of the head to the .part.
    if offset > 0 && response.status().as_u16() != 206 {
        return Err("resume-not-supported".to_string());
    }
    // The reader must OWN the body: `body_mut().as_reader()` (what
    // `download::fetch` uses) borrows the response, which cannot escape into a
    // `Box<dyn Read + Send>`. In ureq 3 the owning form is
    // `response.into_body().into_reader()`, verified against the vendored
    // ureq 3.3.0 source (`body/mod.rs`: `Body::into_reader(self) -> BodyReader<'static>`).
    Ok(Box::new(response.into_body().into_reader()))
}

/// Downloads, verifies and stages every artifact in `manifest` that this build
/// knows how to install.
///
/// Artifacts already sitting verified in staging are skipped, so a job that
/// died after two of three files does not restart those two. When everything is
/// staged, the manifest itself is written into staging *last*, which is what
/// makes the set self-describing for the swap in Task 6.
#[must_use]
pub fn run_job(manifest: &crate::manifest::Manifest, fetch: &Fetcher) -> Outcome {
    let staging = crate::manifest::staging_dir();
    if let Err(e) = std::fs::create_dir_all(&staging) {
        return Outcome::Failed(format!("cannot create {}: {e}", staging.display()));
    }

    let jobs: Vec<(&str, &crate::manifest::FileEntry)> = crate::manifest::KINDS
        .iter()
        .filter_map(|kind| manifest.files.get(*kind).map(|e| (*kind, e)))
        .collect();
    let of = jobs.len();

    for (index, (kind, entry)) in jobs.iter().enumerate() {
        if let Some(how) = read_cancel() {
            return finish_cancel(how, manifest, &jobs);
        }
        if staged_path(kind).exists() {
            continue;
        }
        match one_artifact(manifest, kind, entry, index + 1, of, fetch) {
            Ok(None) => {}
            Ok(Some(how)) => return finish_cancel(how, manifest, &jobs),
            Err(e) => {
                publish(
                    manifest,
                    kind,
                    index + 1,
                    of,
                    0,
                    entry.bytes,
                    0,
                    Phase::Failed,
                    Some(e.clone()),
                );
                return Outcome::Failed(e);
            }
        }
    }

    // Written last: its presence is the swap's proof that the whole set landed.
    if let Err(e) = std::fs::write(staging.join("ds4.manifest"), &manifest.raw) {
        return Outcome::Failed(format!("cannot stage the manifest: {e}"));
    }
    publish(manifest, "", of, of, 0, 0, 0, Phase::Staged, None);
    Outcome::Verified
}

/// Downloads and verifies one artifact. `Ok(Some(how))` means a cancel was
/// observed mid-stream.
#[allow(clippy::too_many_lines)]
fn one_artifact(
    manifest: &crate::manifest::Manifest,
    kind: &str,
    entry: &crate::manifest::FileEntry,
    index: usize,
    of: usize,
    fetch: &Fetcher,
) -> Result<Option<Cancel>, String> {
    let part = part_path(kind);
    let mut hasher = Sha256::new();

    publish(
        manifest,
        kind,
        index,
        of,
        0,
        entry.bytes,
        0,
        Phase::Rehashing,
        None,
    );
    let mut done = rehash(&part, &mut hasher)
        .map_err(|e| format!("cannot re-read {}: {e}", part.display()))?;

    // A .part longer than the manifest says is not a resume point, it is
    // garbage from a different release. Start over rather than append to it.
    if done > entry.bytes {
        let _ = std::fs::remove_file(&part);
        hasher = Sha256::new();
        done = 0;
    }

    let mut reader = match fetch(&entry.url, done) {
        Ok(r) => r,
        Err(e) if e == "resume-not-supported" => {
            // The server sent the whole body despite the Range ask. Throw the
            // partial away and restart the hash, as `download::fetch` does.
            let _ = std::fs::remove_file(&part);
            hasher = Sha256::new();
            done = 0;
            fetch(&entry.url, 0)?
        }
        Err(e) => return Err(e),
    };

    let mut file = if done > 0 {
        OpenOptions::new()
            .append(true)
            .open(&part)
            .map_err(|e| e.to_string())?
    } else {
        File::create(&part).map_err(|e| e.to_string())?
    };

    publish(
        manifest,
        kind,
        index,
        of,
        done,
        entry.bytes,
        0,
        Phase::Downloading,
        None,
    );
    let mut buf = vec![0u8; CHUNK];
    let mut last_publish = Instant::now();
    let mut window_start = Instant::now();
    let mut window_bytes = 0u64;
    let mut rate = 0u64;
    loop {
        if let Some(how) = read_cancel() {
            let _ = file.flush();
            return Ok(Some(how));
        }
        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        hasher.update(&buf[..n]);
        done += n as u64;
        window_bytes += n as u64;
        if last_publish.elapsed() >= PUBLISH_EVERY {
            let secs = window_start.elapsed().as_secs_f64();
            if secs > 0.0 {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss
                )]
                {
                    rate = (window_bytes as f64 / secs) as u64;
                }
            }
            window_start = Instant::now();
            window_bytes = 0;
            last_publish = Instant::now();
            publish(
                manifest,
                kind,
                index,
                of,
                done,
                entry.bytes,
                rate,
                Phase::Downloading,
                None,
            );
        }
    }
    file.flush().map_err(|e| e.to_string())?;

    // A body that ends early reads as a clean EOF. Leave the .part alone: those
    // bytes are good, just incomplete, and the next run resumes them.
    if done != entry.bytes {
        return Err(format!(
            "{kind}: got {done} of {} bytes; it will resume on the next launch",
            entry.bytes
        ));
    }

    publish(
        manifest,
        kind,
        index,
        of,
        done,
        entry.bytes,
        rate,
        Phase::Verifying,
        None,
    );
    let got = hex(&hasher.finalize());
    if got != entry.sha256 {
        // Wrong bytes cannot be fixed by resuming, so the .part must not
        // survive to be resumed. Fail once; nothing retries automatically.
        let _ = std::fs::remove_file(&part);
        return Err(format!(
            "{kind}: checksum mismatch (got {got}, want {})",
            entry.sha256
        ));
    }
    std::fs::rename(&part, staged_path(kind)).map_err(|e| e.to_string())?;
    Ok(None)
}

/// Clears the flag, optionally removes partial work, and publishes the stop.
fn finish_cancel(
    how: Cancel,
    manifest: &crate::manifest::Manifest,
    jobs: &[(&str, &crate::manifest::FileEntry)],
) -> Outcome {
    if how == Cancel::Delete {
        for (kind, _) in jobs {
            let _ = std::fs::remove_file(part_path(kind));
            let _ = std::fs::remove_file(staged_path(kind));
        }
        let _ = std::fs::remove_file(crate::manifest::staging_dir().join("ds4.manifest"));
        // Only if it is now empty: a directory with anything else in it is not
        // ours to remove.
        let _ = std::fs::remove_dir(crate::manifest::staging_dir());
    }
    clear_cancel();
    publish(manifest, "", 0, jobs.len(), 0, 0, 0, Phase::Cancelled, None);
    Outcome::Cancelled
}

/// Publishes one state snapshot. Best-effort: a download that is going fine is
/// not worth aborting over an unwritable status file.
#[allow(clippy::too_many_arguments)]
fn publish(
    manifest: &crate::manifest::Manifest,
    current: &str,
    index: usize,
    of: usize,
    done_bytes: u64,
    total_bytes: u64,
    rate_bps: u64,
    phase: Phase,
    error: Option<String>,
) {
    let _ = write_state(&State {
        pid: std::process::id(),
        version: manifest.version,
        current: current.to_string(),
        index,
        of,
        done_bytes,
        total_bytes,
        rate_bps,
        phase,
        error,
        updated: now_epoch(),
    });
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

    /// SHA-256 of the empty input, as a control.
    const EMPTY_SHA: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    /// SHA-256 of `b"abc"`, the canonical test vector.
    const ABC_SHA: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    #[test]
    fn hex_renders_lowercase_and_full_width() {
        assert_eq!(hex(&[0x00, 0x0f, 0xff]), "000fff");
    }

    #[test]
    fn rehash_of_an_absent_part_is_zero_bytes_and_the_empty_digest() {
        let dir = tempdir();
        let mut hasher = Sha256::new();
        let n = rehash(&dir.join("absent.part"), &mut hasher).expect("absent is not an error");
        assert_eq!(n, 0);
        assert_eq!(hex(&hasher.finalize()), EMPTY_SHA);
    }

    #[test]
    fn rehash_matches_hashing_the_whole_input_at_once() {
        let dir = tempdir();
        let part = dir.join("x.part");
        std::fs::write(&part, b"abc").expect("write");
        let mut hasher = Sha256::new();
        let n = rehash(&part, &mut hasher).expect("rehash");
        assert_eq!(n, 3);
        assert_eq!(hex(&hasher.finalize()), ABC_SHA);
    }

    #[test]
    fn a_resumed_download_produces_the_same_digest_as_a_fresh_one() {
        // The whole point of Task 4: `.part` bytes rehashed from disk plus the
        // remaining bytes streamed must equal hashing the file in one pass.
        let dir = tempdir();
        let part = dir.join("y.part");
        std::fs::write(&part, b"a").expect("write");
        let mut hasher = Sha256::new();
        let offset = rehash(&part, &mut hasher).expect("rehash");
        assert_eq!(offset, 1);
        hasher.update(b"bc");
        assert_eq!(hex(&hasher.finalize()), ABC_SHA);
    }

    #[test]
    fn run_job_verifies_stages_and_reports_done() {
        let dir = tempdir();
        let _guard = with_plank_home(&dir);
        let m = manifest_for(&[("main", b"abc".as_slice(), ABC_SHA)]);
        let outcome = run_job(&m, &serving(&[("main", b"abc".to_vec())]));
        assert_eq!(outcome, Outcome::Verified);
        assert_eq!(
            std::fs::read(staged_path("main")).expect("staged file"),
            b"abc"
        );
        assert!(!part_path("main").exists(), "the .part is consumed");
        assert!(
            crate::manifest::staging_dir().join("ds4.manifest").exists(),
            "the manifest is staged alongside the artifacts"
        );
    }

    #[test]
    fn a_hash_mismatch_deletes_the_part_and_fails_once() {
        // Wrong bytes cannot be fixed by resuming, so the .part must not survive
        // to be resumed on the next launch.
        let dir = tempdir();
        let _guard = with_plank_home(&dir);
        let m = manifest_for(&[("main", b"abc".as_slice(), EMPTY_SHA)]);
        let outcome = run_job(&m, &serving(&[("main", b"abc".to_vec())]));
        assert!(matches!(outcome, Outcome::Failed(_)), "got {outcome:?}");
        assert!(!part_path("main").exists(), "bad bytes are discarded");
        assert!(!staged_path("main").exists(), "nothing is staged");
    }

    #[test]
    fn a_short_body_leaves_the_part_for_the_next_run() {
        // A truncated body reads as a clean EOF. It must not be verified, and it
        // must not be deleted: those bytes are good, just incomplete.
        let dir = tempdir();
        let _guard = with_plank_home(&dir);
        let mut m = manifest_for(&[("main", b"abc".as_slice(), ABC_SHA)]);
        m.files.get_mut("main").expect("main").bytes = 10;
        let outcome = run_job(&m, &serving(&[("main", b"abc".to_vec())]));
        assert!(matches!(outcome, Outcome::Failed(_)), "got {outcome:?}");
        assert_eq!(
            std::fs::read(part_path("main")).expect("part survives"),
            b"abc"
        );
    }

    #[test]
    fn a_pending_cancel_stops_before_any_bytes_and_keep_preserves_partials() {
        let dir = tempdir();
        let _guard = with_plank_home(&dir);
        std::fs::create_dir_all(crate::manifest::staging_dir()).expect("staging");
        std::fs::write(part_path("main"), b"a").expect("seed a partial");
        request_cancel(Cancel::Keep).expect("flag");
        let m = manifest_for(&[("main", b"abc".as_slice(), ABC_SHA)]);
        let outcome = run_job(&m, &serving(&[("main", b"abc".to_vec())]));
        assert_eq!(outcome, Outcome::Cancelled);
        assert!(part_path("main").exists(), "keep means keep");
    }

    #[test]
    fn cancel_with_delete_removes_the_partials() {
        let dir = tempdir();
        let _guard = with_plank_home(&dir);
        std::fs::create_dir_all(crate::manifest::staging_dir()).expect("staging");
        std::fs::write(part_path("main"), b"a").expect("seed a partial");
        request_cancel(Cancel::Delete).expect("flag");
        let m = manifest_for(&[("main", b"abc".as_slice(), ABC_SHA)]);
        let outcome = run_job(&m, &serving(&[("main", b"abc".to_vec())]));
        assert_eq!(outcome, Outcome::Cancelled);
        assert!(!part_path("main").exists(), "delete means delete");
    }

    #[test]
    fn an_already_staged_artifact_is_not_refetched() {
        // Resuming a job that got through two of three files must not restart the
        // two that are done.
        let dir = tempdir();
        let _guard = with_plank_home(&dir);
        std::fs::create_dir_all(crate::manifest::staging_dir()).expect("staging");
        std::fs::write(staged_path("main"), b"abc").expect("pre-staged");
        let m = manifest_for(&[("main", b"abc".as_slice(), ABC_SHA)]);
        // A fetcher that would panic if called proves nothing was refetched.
        let never = |_: &str, _: u64| -> Result<Box<dyn Read + Send>, String> {
            panic!("must not refetch a staged artifact")
        };
        assert_eq!(run_job(&m, &never), Outcome::Verified);
    }

    /// A unique scratch directory under the system temp dir, removed on drop by
    /// [`HomeGuard`]. The repo has no `tempfile` dependency, so this is hand-rolled
    /// the way `src/imagepaste.rs`'s tests do it.
    fn tempdir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "plank-dl-{}-{}",
            std::process::id(),
            now_epoch_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// Nanosecond counter, so two `tempdir()` calls in the same second differ.
    fn now_epoch_nanos() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    }

    /// Points `HOME` at `dir` for the duration, so `manifest::plank_dir()` and
    /// everything under it resolve inside the sandbox.
    ///
    /// `set_var` is process-global, so these tests must not run concurrently with
    /// each other; the guard's mutex enforces that. (`FINDINGS.md` records the same
    /// hazard for the spill tests.)
    struct HomeGuard {
        previous: Option<std::ffi::OsString>,
        dir: std::path::PathBuf,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    static HOME_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_plank_home(dir: &std::path::Path) -> HomeGuard {
        let lock = HOME_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = std::env::var_os("HOME");
        // SAFETY: serialized by HOME_LOCK; no other thread in these tests reads HOME.
        unsafe { std::env::set_var("HOME", dir) };
        HomeGuard {
            previous,
            dir: dir.to_path_buf(),
            _lock: lock,
        }
    }

    impl Drop for HomeGuard {
        fn drop(&mut self) {
            // SAFETY: still holding HOME_LOCK.
            unsafe {
                match &self.previous {
                    Some(v) => std::env::set_var("HOME", v),
                    None => std::env::remove_var("HOME"),
                }
            }
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// Builds a manifest over `(kind, bytes, sha256)` triples.
    fn manifest_for(entries: &[(&str, &[u8], &str)]) -> crate::manifest::Manifest {
        let files: Vec<String> = entries
            .iter()
            .map(|(kind, bytes, sha)| {
                format!(
                    r#""{kind}": {{ "name": "{kind}.gguf", "url": "https://example.invalid/{kind}", "bytes": {}, "sha256": "{sha}" }}"#,
                    bytes.len()
                )
            })
            .collect();
        let text = format!(
            r#"{{"version":3,"released":"t","notes":"","files":{{{}}}}}"#,
            files.join(",")
        );
        crate::manifest::parse(&text).expect("test manifest parses")
    }

    /// A fetcher serving fixed bodies, honoring the resume offset the way a
    /// Range-capable server would.
    fn serving(
        bodies: &[(&str, Vec<u8>)],
    ) -> impl Fn(&str, u64) -> Result<Box<dyn Read + Send>, String> + 'static {
        let bodies: Vec<(String, Vec<u8>)> = bodies
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect();
        move |url: &str, offset: u64| {
            let kind = url.rsplit('/').next().unwrap_or_default();
            let body = bodies
                .iter()
                .find(|(k, _)| k == kind)
                .map(|(_, v)| v.clone())
                .ok_or_else(|| format!("no body for {kind}"))?;
            let start = usize::try_from(offset)
                .unwrap_or(usize::MAX)
                .min(body.len());
            Ok(Box::new(std::io::Cursor::new(body[start..].to_vec())))
        }
    }
}
