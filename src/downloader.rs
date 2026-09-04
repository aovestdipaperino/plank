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

/// `<root>/downloads/state.json`.
#[must_use]
pub fn state_path_in(root: &Path) -> PathBuf {
    crate::manifest::downloads_dir_in(root).join("state.json")
}

/// `~/.plank/downloads/state.json`.
#[must_use]
pub fn state_path() -> PathBuf {
    state_path_in(&crate::manifest::plank_dir())
}

/// `<root>/downloads/cancel`.
#[must_use]
pub fn cancel_path_in(root: &Path) -> PathBuf {
    crate::manifest::downloads_dir_in(root).join("cancel")
}

/// `~/.plank/downloads/cancel`.
#[must_use]
pub fn cancel_path() -> PathBuf {
    cancel_path_in(&crate::manifest::plank_dir())
}

/// Publishes `state` under `root`, atomically.
///
/// Written to a sibling temp file and renamed, so a reader polling twice a
/// second never sees a half-written JSON object.
///
/// # Errors
/// Propagates any filesystem error from creating the directory, writing the
/// temp file, or renaming it.
pub fn write_state_in(root: &Path, state: &State) -> std::io::Result<()> {
    let path = state_path_in(root);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string(state)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, &path)
}

/// Publishes `state`, atomically.
///
/// # Errors
/// Propagates any filesystem error from creating the directory, writing the
/// temp file, or renaming it.
pub fn write_state(state: &State) -> std::io::Result<()> {
    write_state_in(&crate::manifest::plank_dir(), state)
}

/// Reads the published state under `root`, whatever its age. `None` when
/// absent or corrupt.
#[must_use]
pub fn read_state_in(root: &Path) -> Option<State> {
    serde_json::from_str(&std::fs::read_to_string(state_path_in(root)).ok()?).ok()
}

/// Reads the published state, whatever its age. `None` when absent or corrupt.
#[must_use]
pub fn read_state() -> Option<State> {
    read_state_in(&crate::manifest::plank_dir())
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

/// Asks the running helper to stop, under `root`.
///
/// # Errors
/// Propagates any filesystem error from creating the directory or writing the
/// flag.
pub fn request_cancel_in(root: &Path, how: Cancel) -> std::io::Result<()> {
    let path = cancel_path_in(root);
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

/// Asks the running helper to stop.
///
/// # Errors
/// Propagates any filesystem error from creating the directory or writing the
/// flag.
pub fn request_cancel(how: Cancel) -> std::io::Result<()> {
    request_cancel_in(&crate::manifest::plank_dir(), how)
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

/// The pending cancel request under `root`, if any.
#[must_use]
pub fn read_cancel_in(root: &Path) -> Option<Cancel> {
    parse_cancel(&std::fs::read_to_string(cancel_path_in(root)).ok()?)
}

/// The pending cancel request, if any.
#[must_use]
pub fn read_cancel() -> Option<Cancel> {
    read_cancel_in(&crate::manifest::plank_dir())
}

/// Clears any pending cancel request under `root`. Best-effort: a stale flag
/// that cannot be removed is caught by the helper's own startup, which clears
/// it before work begins.
pub fn clear_cancel_in(root: &Path) {
    let _ = std::fs::remove_file(cancel_path_in(root));
}

/// Clears any pending cancel request. Best-effort: a stale flag that cannot be
/// removed is caught by the helper's own startup, which clears it before work
/// begins.
pub fn clear_cancel() {
    clear_cancel_in(&crate::manifest::plank_dir());
}

/// Read buffer, matching `download::fetch`'s 1 MiB.
const CHUNK: usize = 1 << 20;
/// How often the helper republishes its state. Twice a second is fast enough
/// for a status bar and slow enough that the rename costs nothing.
const PUBLISH_EVERY: std::time::Duration = std::time::Duration::from_millis(500);

/// In-flight bytes for `kind`, under `root`.
#[must_use]
pub fn part_path_in(root: &Path, kind: &str) -> PathBuf {
    crate::manifest::staging_dir_in(root).join(format!("{kind}.part"))
}

/// In-flight bytes for `kind`.
#[must_use]
pub fn part_path(kind: &str) -> PathBuf {
    part_path_in(&crate::manifest::plank_dir(), kind)
}

/// Verified-but-not-yet-installed bytes for `kind`, under `root`.
#[must_use]
pub fn staged_path_in(root: &Path, kind: &str) -> PathBuf {
    crate::manifest::staging_dir_in(root).join(format!("{kind}.gguf"))
}

/// Verified-but-not-yet-installed bytes for `kind`.
#[must_use]
pub fn staged_path(kind: &str) -> PathBuf {
    staged_path_in(&crate::manifest::plank_dir(), kind)
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

/// Feeds an existing `.part` back through `hasher`, returning its length, or
/// `None` if `should_stop` reports true partway through.
///
/// This is what makes resume and incremental hashing coexist: `sha2` has no
/// serializable state, so rather than invent a sidecar format (and a new way to
/// be silently wrong), the bytes already on local disk are read once more. It
/// is disk-bound, it needs no format, and it validates what it hashes on the
/// way past.
///
/// An absent `.part` is 0 bytes and an untouched hasher, not an error.
///
/// For an interrupted multi-gigabyte download this read alone can take
/// minutes, so `should_stop` is polled once per chunk rather than only after
/// the whole file is read.
///
/// # Errors
/// Propagates read errors other than "not found".
pub fn rehash(
    part: &Path,
    hasher: &mut Sha256,
    should_stop: &dyn Fn() -> bool,
) -> std::io::Result<Option<u64>> {
    let mut file = match File::open(part) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Some(0)),
        Err(e) => return Err(e),
    };
    let mut buf = vec![0u8; CHUNK];
    let mut total = 0u64;
    loop {
        if should_stop() {
            return Ok(None);
        }
        let n = file.read(&mut buf)?;
        if n == 0 {
            return Ok(Some(total));
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
pub fn run_job(root: &Path, manifest: &crate::manifest::Manifest, fetch: &Fetcher) -> Outcome {
    let staging = crate::manifest::staging_dir_in(root);
    if let Err(e) = std::fs::create_dir_all(&staging) {
        return Outcome::Failed(format!("cannot create {}: {e}", staging.display()));
    }

    let jobs: Vec<(&str, &crate::manifest::FileEntry)> = crate::manifest::KINDS
        .iter()
        .filter_map(|kind| manifest.files.get(*kind).map(|e| (*kind, e)))
        .collect();
    let of = jobs.len();

    for (index, (kind, entry)) in jobs.iter().enumerate() {
        if let Some(how) = read_cancel_in(root) {
            return finish_cancel(root, how, manifest, &jobs);
        }
        if staged_path_in(root, kind).exists() {
            continue;
        }
        match one_artifact(root, manifest, kind, entry, index + 1, of, fetch) {
            Ok(None) => {}
            Ok(Some(how)) => return finish_cancel(root, how, manifest, &jobs),
            Err(e) => {
                publish(
                    root,
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
    publish(root, manifest, "", of, of, 0, 0, 0, Phase::Staged, None);
    Outcome::Verified
}

/// Downloads and verifies one artifact. `Ok(Some(how))` means a cancel was
/// observed mid-stream.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn one_artifact(
    root: &Path,
    manifest: &crate::manifest::Manifest,
    kind: &str,
    entry: &crate::manifest::FileEntry,
    index: usize,
    of: usize,
    fetch: &Fetcher,
) -> Result<Option<Cancel>, String> {
    let part = part_path_in(root, kind);
    let mut hasher = Sha256::new();

    publish(
        root,
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
    let should_stop = || read_cancel_in(root).is_some();
    let Some(mut done) = rehash(&part, &mut hasher, &should_stop)
        .map_err(|e| format!("cannot re-read {}: {e}", part.display()))?
    else {
        // A cancel arrived while rehashing an existing .part. Whatever the
        // pending request says (keep or delete), it is handled by run_job's
        // loop after this artifact returns — surface it the same way a
        // cancel mid-stream does.
        let how = read_cancel_in(root).unwrap_or(Cancel::Keep);
        return Ok(Some(how));
    };

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
        root,
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
        if let Some(how) = read_cancel_in(root) {
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
                root,
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
        root,
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
    std::fs::rename(&part, staged_path_in(root, kind)).map_err(|e| e.to_string())?;
    Ok(None)
}

/// Clears the flag, optionally removes partial work, and publishes the stop.
fn finish_cancel(
    root: &Path,
    how: Cancel,
    manifest: &crate::manifest::Manifest,
    jobs: &[(&str, &crate::manifest::FileEntry)],
) -> Outcome {
    if how == Cancel::Delete {
        for (kind, _) in jobs {
            let _ = std::fs::remove_file(part_path_in(root, kind));
            let _ = std::fs::remove_file(staged_path_in(root, kind));
        }
        let _ = std::fs::remove_file(crate::manifest::staging_dir_in(root).join("ds4.manifest"));
        // Only if it is now empty: a directory with anything else in it is not
        // ours to remove.
        let _ = std::fs::remove_dir(crate::manifest::staging_dir_in(root));
    }
    clear_cancel_in(root);
    publish(
        root,
        manifest,
        "",
        0,
        jobs.len(),
        0,
        0,
        0,
        Phase::Cancelled,
        None,
    );
    Outcome::Cancelled
}

/// Publishes one state snapshot. Best-effort: a download that is going fine is
/// not worth aborting over an unwritable status file.
#[allow(clippy::too_many_arguments)]
fn publish(
    root: &Path,
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
    let _ = write_state_in(
        root,
        &State {
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
        },
    );
}

/// The flock file that guarantees one downloader per machine.
///
/// Machine-wide, not per-project: the artifacts live in `~/.plank`, so two
/// planks in two repos downloading the same 87 GB file into the same staging
/// directory would corrupt each other's `.part`.
#[must_use]
pub fn lock_path_in(root: &Path) -> PathBuf {
    crate::manifest::downloads_dir_in(root).join("lock")
}

/// `~/.plank/downloads/lock`.
#[must_use]
pub fn lock_path() -> PathBuf {
    lock_path_in(&crate::manifest::plank_dir())
}

/// The manifest the running (or next) helper is installing.
#[must_use]
pub fn job_path_in(root: &Path) -> PathBuf {
    crate::manifest::downloads_dir_in(root).join("job.json")
}

/// `~/.plank/downloads/job.json`.
#[must_use]
pub fn job_path() -> PathBuf {
    job_path_in(&crate::manifest::plank_dir())
}

/// Where a detached helper's stderr goes, since it has no terminal.
#[must_use]
pub fn log_path_in(root: &Path) -> PathBuf {
    crate::manifest::downloads_dir_in(root).join("log")
}

/// `~/.plank/downloads/log`.
#[must_use]
pub fn log_path() -> PathBuf {
    log_path_in(&crate::manifest::plank_dir())
}

/// Records `manifest` as the job for the helper to pick up.
///
/// Stored as the manifest's own bytes rather than a wrapper struct: the helper
/// needs exactly the manifest, and a wrapper would be a second format to keep
/// in sync with `ds4.manifest` for no gain.
///
/// # Errors
/// Propagates filesystem errors.
pub fn write_job_in(root: &Path, manifest: &crate::manifest::Manifest) -> std::io::Result<()> {
    let path = job_path_in(root);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, &manifest.raw)
}

/// Records `manifest` as the job for the helper to pick up, under `~/.plank`.
///
/// # Errors
/// Propagates filesystem errors.
pub fn write_job(manifest: &crate::manifest::Manifest) -> std::io::Result<()> {
    write_job_in(&crate::manifest::plank_dir(), manifest)
}

/// The pending job, if one is recorded and still parses.
#[must_use]
pub fn read_job_in(root: &Path) -> Option<crate::manifest::Manifest> {
    crate::manifest::read_at(&job_path_in(root))
}

/// The pending job under `~/.plank`, if one is recorded and still parses.
#[must_use]
pub fn read_job() -> Option<crate::manifest::Manifest> {
    read_job_in(&crate::manifest::plank_dir())
}

/// Whether a helper currently holds the lock at `root`.
#[must_use]
pub fn running_in(root: &Path) -> bool {
    matches!(
        crate::singleton::probe_lock(&lock_path_in(root)),
        crate::singleton::LockProbe::Contended
    )
}

/// Whether a helper currently holds the lock under `~/.plank`.
#[must_use]
pub fn running() -> bool {
    running_in(&crate::manifest::plank_dir())
}

/// Spawns the detached helper for `manifest`, unless one is already running.
///
/// Detached deliberately: the download outlives the plank that started it, and
/// outlives the terminal that plank was typed into. `setsid` in the child is
/// what breaks it off the controlling terminal, and all three standard streams
/// go to `/dev/null` so nothing it writes can ever land in a user's session —
/// diagnostics go to [`log_path`] instead.
///
/// # Errors
/// Returns a message when the job cannot be recorded or the child cannot be
/// spawned. There is deliberately no in-process fallback — a foreground 87 GB
/// download is exactly what this whole feature exists to avoid — so callers
/// report the error and nothing downloads; `/model download` is how the user
/// retries.
pub fn spawn_detached(manifest: &crate::manifest::Manifest) -> Result<(), String> {
    if running() {
        return Ok(());
    }
    write_job(manifest).map_err(|e| format!("cannot record the download job: {e}"))?;
    clear_cancel();
    let exe = std::env::current_exe().map_err(|e| format!("cannot find plank's own path: {e}"))?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--model-downloader")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    detach(&mut cmd);
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("cannot start the downloader: {e}"))?;
    // Reaped immediately: `setsid` already made the grandchild session leader,
    // so nothing here needs to wait on it, and not reaping would leave a zombie
    // for as long as this plank runs.
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

/// Puts the child in its own session, so it survives the terminal closing.
#[cfg(unix)]
fn detach(cmd: &mut std::process::Command) {
    use std::os::unix::process::CommandExt as _;
    // SAFETY: `setsid` is async-signal-safe and is the only call made between
    // fork and exec.
    unsafe {
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        })
    };
}

/// No session concept to detach from; the caller's fallback path applies.
#[cfg(not(unix))]
fn detach(_cmd: &mut std::process::Command) {}

/// The `plank --model-downloader` entry point. Returns a process exit code.
///
/// Takes the lock for its whole life, so a second helper started by another
/// plank exits immediately instead of fighting over the same `.part` files.
#[must_use]
pub fn run_helper() -> i32 {
    let root = crate::manifest::plank_dir();
    let path = lock_path_in(&root);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let Ok(lock) = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&path)
    else {
        return 1;
    };
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd as _;
        // SAFETY: `lock` owns a valid fd; LOCK_NB makes this non-blocking.
        let rc = unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            // Another helper owns the download. Not an error worth reporting.
            return 0;
        }
    }
    let Some(manifest) = read_job_in(&root) else {
        log_line("no job recorded; nothing to download");
        return 1;
    };
    // A cancel flag left over from a previous run must not stop this one before
    // it starts.
    clear_cancel();
    let outcome = run_job(&root, &manifest, &http_fetch);
    log_line(&format!(
        "job for version {} ended: {outcome:?}",
        manifest.version
    ));
    // The lock drops with `lock` here, releasing the flock.
    drop(lock);
    match outcome {
        Outcome::Verified | Outcome::Cancelled => 0,
        Outcome::Failed(_) => 1,
    }
}

/// Appends one timestamped line to the helper's log. Best-effort.
fn log_line(msg: &str) {
    let path = log_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "[{}] {msg}", now_epoch());
    }
}

/// Installs a complete staged set under `root`, if one is waiting.
///
/// This runs at launch rather than the instant a download verifies, because a
/// running plank has the 87 GB main model mapped: overwriting it under a live
/// process is exactly the kind of thing that works in testing and corrupts a
/// session in the field.
///
/// The ordering is the whole design. Files move first, the installed manifest
/// moves last, so the manifest's presence is proof the entire set landed. A
/// crash between two renames leaves no installed manifest and leaves the
/// remaining artifacts in staging, so the next launch simply finishes the job —
/// artifacts already moved are no longer in staging and are skipped.
///
/// Nothing is re-hashed here. Staging is ours, and every byte in it was
/// verified on the way in.
///
/// Returns the version installed, or `None` when there was nothing complete to
/// install.
///
/// # Errors
/// Returns a message only when a rename of a verified artifact fails — a real
/// filesystem problem the user needs to hear about.
pub fn swap_staged_in(root: &Path) -> Result<Option<u32>, String> {
    let staged_manifest = crate::manifest::staging_dir_in(root).join("ds4.manifest");
    let Some(manifest) = crate::manifest::read_at(&staged_manifest) else {
        return Ok(None);
    };

    // Everything this build installs, paired with where it goes.
    let jobs: Vec<(&str, &crate::manifest::FileEntry, PathBuf)> = crate::manifest::KINDS
        .iter()
        .filter_map(|kind| {
            let entry = manifest.files.get(*kind)?;
            let dest = crate::manifest::local_path_for_in(root, kind)?;
            Some((*kind, entry, dest))
        })
        .collect();

    // Each artifact must be either already installed at the right size (a
    // resumed swap) or staged at the right size. Anything else means the set is
    // incomplete, and a partial swap would leave a mismatched trio on disk.
    let complete = jobs.iter().all(|(kind, entry, dest)| {
        let size = |p: &Path| std::fs::metadata(p).map(|m| m.len()).ok();
        size(&staged_path_in(root, kind)) == Some(entry.bytes) || size(dest) == Some(entry.bytes)
    });
    if !complete {
        return Ok(None);
    }

    if let Err(e) = std::fs::create_dir_all(root) {
        return Err(format!("cannot create the plank directory: {e}"));
    }
    for (kind, _, dest) in &jobs {
        let from = staged_path_in(root, kind);
        if !from.exists() {
            // Already moved by an interrupted earlier swap.
            continue;
        }
        std::fs::rename(&from, dest)
            .map_err(|e| format!("cannot install {kind} at {}: {e}", dest.display()))?;
    }
    // Last, and only now: this file is the claim that the set is installed.
    std::fs::rename(&staged_manifest, crate::manifest::installed_path_in(root))
        .map_err(|e| format!("cannot record the installed manifest: {e}"))?;
    let _ = std::fs::remove_dir(crate::manifest::staging_dir_in(root));
    let _ = std::fs::remove_file(state_path_in(root));
    Ok(Some(manifest.version))
}

/// Installs a complete staged set under `~/.plank`, if one is waiting.
///
/// # Errors
/// Returns a message only when a rename of a verified artifact fails — a real
/// filesystem problem the user needs to hear about.
pub fn swap_staged() -> Result<Option<u32>, String> {
    swap_staged_in(&crate::manifest::plank_dir())
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
        let n = rehash(&dir.join("absent.part"), &mut hasher, &|| false)
            .expect("absent is not an error")
            .expect("not cancelled");
        assert_eq!(n, 0);
        assert_eq!(hex(&hasher.finalize()), EMPTY_SHA);
    }

    #[test]
    fn rehash_matches_hashing_the_whole_input_at_once() {
        let dir = tempdir();
        let part = dir.join("x.part");
        std::fs::write(&part, b"abc").expect("write");
        let mut hasher = Sha256::new();
        let n = rehash(&part, &mut hasher, &|| false)
            .expect("rehash")
            .expect("not cancelled");
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
        let offset = rehash(&part, &mut hasher, &|| false)
            .expect("rehash")
            .expect("not cancelled");
        assert_eq!(offset, 1);
        hasher.update(b"bc");
        assert_eq!(hex(&hasher.finalize()), ABC_SHA);
    }

    #[test]
    fn rehash_polls_should_stop_and_reports_cancellation() {
        // An interrupted multi-gigabyte .part can take minutes to re-read; the
        // caller must be able to abandon that read partway through, not only
        // after it finishes.
        let dir = tempdir();
        let part = dir.join("z.part");
        // Several chunks' worth so should_stop is polled more than once, in
        // spirit; CHUNK is 1 MiB so a few KB here still exercises one read.
        std::fs::write(&part, b"abc").expect("write");
        let mut hasher = Sha256::new();
        let n = rehash(&part, &mut hasher, &|| true).expect("read succeeds");
        assert_eq!(n, None, "should_stop firing immediately cancels the rehash");
    }

    #[test]
    fn run_job_verifies_stages_and_reports_done() {
        let root = tempdir();
        let m = manifest_for(&[("main", b"abc".as_slice(), ABC_SHA)]);
        let outcome = run_job(&root, &m, &serving(&[("main", b"abc".to_vec())]));
        assert_eq!(outcome, Outcome::Verified);
        assert_eq!(
            std::fs::read(staged_path_in(&root, "main")).expect("staged file"),
            b"abc"
        );
        assert!(
            !part_path_in(&root, "main").exists(),
            "the .part is consumed"
        );
        assert!(
            crate::manifest::staging_dir_in(&root)
                .join("ds4.manifest")
                .exists(),
            "the manifest is staged alongside the artifacts"
        );
    }

    #[test]
    fn a_hash_mismatch_deletes_the_part_and_fails_once() {
        // Wrong bytes cannot be fixed by resuming, so the .part must not survive
        // to be resumed on the next launch.
        let root = tempdir();
        let m = manifest_for(&[("main", b"abc".as_slice(), EMPTY_SHA)]);
        let outcome = run_job(&root, &m, &serving(&[("main", b"abc".to_vec())]));
        assert!(matches!(outcome, Outcome::Failed(_)), "got {outcome:?}");
        assert!(
            !part_path_in(&root, "main").exists(),
            "bad bytes are discarded"
        );
        assert!(!staged_path_in(&root, "main").exists(), "nothing is staged");
    }

    #[test]
    fn a_short_body_leaves_the_part_for_the_next_run() {
        // A truncated body reads as a clean EOF. It must not be verified, and it
        // must not be deleted: those bytes are good, just incomplete.
        let root = tempdir();
        let mut m = manifest_for(&[("main", b"abc".as_slice(), ABC_SHA)]);
        m.files.get_mut("main").expect("main").bytes = 10;
        let outcome = run_job(&root, &m, &serving(&[("main", b"abc".to_vec())]));
        assert!(matches!(outcome, Outcome::Failed(_)), "got {outcome:?}");
        assert_eq!(
            std::fs::read(part_path_in(&root, "main")).expect("part survives"),
            b"abc"
        );
    }

    #[test]
    fn a_pending_cancel_stops_before_any_bytes_and_keep_preserves_partials() {
        let root = tempdir();
        std::fs::create_dir_all(crate::manifest::staging_dir_in(&root)).expect("staging");
        std::fs::write(part_path_in(&root, "main"), b"a").expect("seed a partial");
        request_cancel_in(&root, Cancel::Keep).expect("flag");
        let m = manifest_for(&[("main", b"abc".as_slice(), ABC_SHA)]);
        let outcome = run_job(&root, &m, &serving(&[("main", b"abc".to_vec())]));
        assert_eq!(outcome, Outcome::Cancelled);
        assert!(part_path_in(&root, "main").exists(), "keep means keep");
    }

    #[test]
    fn cancel_with_delete_removes_the_partials() {
        let root = tempdir();
        std::fs::create_dir_all(crate::manifest::staging_dir_in(&root)).expect("staging");
        std::fs::write(part_path_in(&root, "main"), b"a").expect("seed a partial");
        request_cancel_in(&root, Cancel::Delete).expect("flag");
        let m = manifest_for(&[("main", b"abc".as_slice(), ABC_SHA)]);
        let outcome = run_job(&root, &m, &serving(&[("main", b"abc".to_vec())]));
        assert_eq!(outcome, Outcome::Cancelled);
        assert!(!part_path_in(&root, "main").exists(), "delete means delete");
    }

    #[test]
    fn an_already_staged_artifact_is_not_refetched() {
        // Resuming a job that got through two of three files must not restart the
        // two that are done.
        let root = tempdir();
        std::fs::create_dir_all(crate::manifest::staging_dir_in(&root)).expect("staging");
        std::fs::write(staged_path_in(&root, "main"), b"abc").expect("pre-staged");
        let m = manifest_for(&[("main", b"abc".as_slice(), ABC_SHA)]);
        // A fetcher that would panic if called proves nothing was refetched.
        let never = |_: &str, _: u64| -> Result<Box<dyn Read + Send>, String> {
            panic!("must not refetch a staged artifact")
        };
        assert_eq!(run_job(&root, &m, &never), Outcome::Verified);
    }

    #[test]
    fn resuming_after_a_truncated_body_ends_up_correctly_staged_and_verified() {
        // Finding 3: the digest-equality guarantee for resume, exercised
        // end-to-end through run_job rather than only at the rehash unit
        // level. The first run gets a truncated body and leaves a .part; the
        // second run is served only the remainder, from the offset run_job
        // actually asks for. This is the test that would catch an off-by-one
        // between the resume offset and the hasher state.
        let root = tempdir();
        let full: &[u8] = b"the quick brown fox jumps over the lazy dog";
        let mut hasher = Sha256::new();
        hasher.update(full);
        let sha = hex(&hasher.finalize());
        let m = manifest_for(&[("main", full, &sha)]);

        // First run: serve only the first half, then EOF (a clean truncation,
        // not an error) so the .part is left behind for resume.
        let half = full.len() / 2;
        let first = serving(&[("main", full[..half].to_vec())]);
        let outcome1 = run_job(&root, &m, &first);
        assert!(matches!(outcome1, Outcome::Failed(_)), "got {outcome1:?}");
        assert!(part_path_in(&root, "main").exists(), "partial is kept");
        assert_eq!(
            std::fs::read(part_path_in(&root, "main")).expect("part"),
            full[..half]
        );

        // Second run: a fetcher that only knows how to serve the tail starting
        // exactly at `half`, so if run_job asked for the wrong offset this
        // fetcher would either panic (offset 0 request) or serve the wrong
        // slice (any other wrong offset), and the final digest check would
        // catch it either way.
        let full_owned = full.to_vec();
        let resume_only = move |url: &str, offset: u64| -> Result<Box<dyn Read + Send>, String> {
            let kind = url.rsplit('/').next().unwrap_or_default();
            assert_eq!(kind, "main");
            let offset = usize::try_from(offset).expect("offset fits in usize in this test");
            assert_eq!(
                offset, half,
                "run_job must resume from exactly the bytes already on disk"
            );
            Ok(Box::new(std::io::Cursor::new(
                full_owned[offset..].to_vec(),
            )))
        };
        let outcome2 = run_job(&root, &m, &resume_only);
        assert_eq!(outcome2, Outcome::Verified);
        assert_eq!(
            std::fs::read(staged_path_in(&root, "main")).expect("staged"),
            full
        );
        assert!(
            !part_path_in(&root, "main").exists(),
            "the .part is consumed"
        );
    }

    /// A reader that, after its first read, sets the cancel flag before any
    /// further reads are served. `run_job`'s per-chunk cancel poll then stops
    /// it before the body is exhausted.
    struct CancelAfterFirstChunk {
        root: std::path::PathBuf,
        data: Vec<u8>,
        pos: usize,
        fired: bool,
    }

    impl Read for CancelAfterFirstChunk {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.pos >= self.data.len() {
                return Ok(0);
            }
            if !self.fired {
                self.fired = true;
                // First chunk: hand back a few bytes and arm the cancel so
                // the *next* poll in run_job's loop sees it before reading
                // any more.
                let n = 3.min(self.data.len() - self.pos);
                buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
                self.pos += n;
                request_cancel_in(&self.root, Cancel::Keep).expect("flag");
                return Ok(n);
            }
            panic!("run_job must stop polling before reading further chunks");
        }
    }

    #[test]
    fn a_cancel_set_mid_stream_stops_immediately_and_keeps_the_bytes_so_far() {
        // Finding 4: existing tests only covered a cancel checked between
        // artifacts. This drives one set from inside the fetcher's reader,
        // partway through the byte stream itself.
        let root = tempdir();
        std::fs::create_dir_all(crate::manifest::staging_dir_in(&root)).expect("staging");
        let m = manifest_for(&[("main", b"abcdefghij".as_slice(), EMPTY_SHA)]);

        let root_for_reader = root.clone();
        let fetcher = move |_: &str, offset: u64| -> Result<Box<dyn Read + Send>, String> {
            assert_eq!(offset, 0);
            Ok(Box::new(CancelAfterFirstChunk {
                root: root_for_reader.clone(),
                data: b"abcdefghij".to_vec(),
                pos: 0,
                fired: false,
            }) as Box<dyn Read + Send>)
        };

        let outcome = run_job(&root, &m, &fetcher);
        assert_eq!(outcome, Outcome::Cancelled);
        assert!(
            part_path_in(&root, "main").exists(),
            "Keep leaves the .part in place"
        );
        assert_eq!(
            std::fs::read(part_path_in(&root, "main")).expect("partial bytes"),
            b"abc",
            "only the bytes received before the cancel was observed are kept"
        );
    }

    /// A unique scratch directory under the system temp dir, removed on drop.
    /// The repo has no `tempfile` dependency, so this is hand-rolled the way
    /// `src/imagepaste.rs`'s tests do it. Every test passes this root
    /// explicitly to the `_in` functions under test; none of them touch
    /// process environment, so they are safe under cargo's default test
    /// parallelism (see `FINDINGS.md`'s note on the spill tests' hazard, and
    /// `src/spill.rs`'s `_in` pattern this mirrors).
    fn tempdir() -> std::path::PathBuf {
        // A nanosecond timestamp alone is not enough: cargo runs tests on
        // multiple threads, and two calls on different threads can land in
        // the same nanosecond, colliding on the same directory and letting
        // one test's cancel flag or state file leak into another's. The
        // atomic counter guarantees uniqueness regardless of timing.
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "plank-dl-{}-{}-{n}",
            std::process::id(),
            now_epoch_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        TempDirs::register(&dir);
        dir
    }

    /// Nanosecond counter, so two `tempdir()` calls in the same second differ.
    fn now_epoch_nanos() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    }

    /// Sweeps every directory `tempdir()` created, on process exit, via a
    /// `Drop` guard stashed in a `OnceLock` the first time it is needed.
    /// Avoids leaking scratch directories across a whole `cargo test` run
    /// without requiring each test to remember to clean up after itself.
    struct TempDirs {
        dirs: std::sync::Mutex<Vec<std::path::PathBuf>>,
    }

    impl Drop for TempDirs {
        fn drop(&mut self) {
            if let Ok(dirs) = self.dirs.lock() {
                for dir in dirs.iter() {
                    let _ = std::fs::remove_dir_all(dir);
                }
            }
        }
    }

    impl TempDirs {
        fn register(dir: &std::path::Path) {
            static REGISTRY: std::sync::OnceLock<TempDirs> = std::sync::OnceLock::new();
            let registry = REGISTRY.get_or_init(|| TempDirs {
                dirs: std::sync::Mutex::new(Vec::new()),
            });
            if let Ok(mut dirs) = registry.dirs.lock() {
                dirs.push(dir.to_path_buf());
            }
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

    #[test]
    fn helper_paths_sit_under_the_downloads_directory() {
        let root = tempdir();
        assert_eq!(lock_path_in(&root), root.join("downloads").join("lock"));
        assert_eq!(job_path_in(&root), root.join("downloads").join("job.json"));
        assert_eq!(log_path_in(&root), root.join("downloads").join("log"));
    }

    #[test]
    fn a_job_written_is_a_job_read_back() {
        let root = tempdir();
        let m = manifest_for(&[("main", b"abc".as_slice(), ABC_SHA)]);
        write_job_in(&root, &m).expect("write job");
        let back = read_job_in(&root).expect("read job");
        assert_eq!(back.version, m.version);
        assert_eq!(back.raw, m.raw, "the job is the manifest bytes, verbatim");
    }

    #[test]
    fn an_absent_job_reads_as_none() {
        let root = tempdir();
        assert!(read_job_in(&root).is_none());
    }

    #[test]
    fn nothing_is_running_when_no_lock_is_held() {
        let root = tempdir();
        assert!(!running_in(&root));
    }

    /// Stages a complete set for `manifest` with the given bodies, under `root`.
    fn stage(root: &Path, manifest: &crate::manifest::Manifest, bodies: &[(&str, &[u8])]) {
        std::fs::create_dir_all(crate::manifest::staging_dir_in(root)).expect("staging");
        for (kind, body) in bodies {
            std::fs::write(staged_path_in(root, kind), body).expect("stage artifact");
        }
        std::fs::write(
            crate::manifest::staging_dir_in(root).join("ds4.manifest"),
            &manifest.raw,
        )
        .expect("stage manifest");
    }

    #[test]
    fn swap_moves_every_artifact_and_records_the_manifest_last() {
        let root = tempdir();
        let m = manifest_for(&[("main", b"abc".as_slice(), ABC_SHA)]);
        stage(&root, &m, &[("main", b"abc")]);

        let swapped = swap_staged_in(&root).expect("swap succeeds");
        assert_eq!(swapped, Some(3));
        let installed = crate::manifest::local_path_for_in(&root, "main").expect("main path");
        assert_eq!(std::fs::read(installed).expect("installed"), b"abc");
        assert!(
            !staged_path_in(&root, "main").exists(),
            "staging is drained"
        );
        let recorded = crate::manifest::read_at(&crate::manifest::installed_path_in(&root))
            .expect("installed manifest");
        assert_eq!(recorded.version, 3);
    }

    #[test]
    fn swap_with_nothing_staged_is_a_no_op() {
        let root = tempdir();
        assert_eq!(swap_staged_in(&root).expect("no-op"), None);
        assert!(!crate::manifest::installed_path_in(&root).exists());
    }

    #[test]
    fn swap_refuses_an_incomplete_set() {
        // The staged manifest lists two artifacts but only one is present: a
        // partial swap would leave a mismatched main/vision pair on disk.
        let root = tempdir();
        let m = manifest_for(&[
            ("main", b"abc".as_slice(), ABC_SHA),
            ("vision", b"abc".as_slice(), ABC_SHA),
        ]);
        stage(&root, &m, &[("main", b"abc")]);

        assert_eq!(swap_staged_in(&root).expect("no-op"), None);
        assert!(
            !crate::manifest::installed_path_in(&root).exists(),
            "nothing recorded"
        );
        assert!(
            staged_path_in(&root, "main").exists(),
            "staging is left intact for a retry"
        );
    }

    #[test]
    fn swap_refuses_a_staged_file_of_the_wrong_size() {
        let root = tempdir();
        let m = manifest_for(&[("main", b"abc".as_slice(), ABC_SHA)]);
        stage(&root, &m, &[("main", b"ab")]);
        assert_eq!(swap_staged_in(&root).expect("no-op"), None);
        assert!(!crate::manifest::installed_path_in(&root).exists());
    }

    #[test]
    fn a_swap_interrupted_midway_completes_on_the_next_run() {
        // Simulates a crash after the first rename: `main` is already installed
        // and gone from staging, `vision` is still waiting. The installed
        // manifest has not been written, so the re-run must finish the job
        // rather than skip it.
        let root = tempdir();
        let m = manifest_for(&[
            ("main", b"abc".as_slice(), ABC_SHA),
            ("vision", b"abc".as_slice(), ABC_SHA),
        ]);
        stage(&root, &m, &[("main", b"abc"), ("vision", b"abc")]);
        // Perform the "already happened" half by hand.
        std::fs::create_dir_all(&root).expect("root");
        std::fs::rename(
            staged_path_in(&root, "main"),
            crate::manifest::local_path_for_in(&root, "main").expect("main path"),
        )
        .expect("first rename");

        assert_eq!(swap_staged_in(&root).expect("completes"), Some(3));
        assert!(
            crate::manifest::local_path_for_in(&root, "vision")
                .expect("v")
                .exists()
        );
        assert_eq!(
            crate::manifest::read_at(&crate::manifest::installed_path_in(&root))
                .expect("installed")
                .version,
            3
        );
    }
}
