// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! GPU yield: unloading the local model so a child command can have the GPU.
//!
//! A command that needs the GPU plank's loaded model occupies (the first is
//! `mex`, which runs a diffusion model on Metal) follows a small protocol.
//! plank exports [`ENV_VAR`]`=1` into every bash-tool job, and into the
//! user's `!` and `!!` shell escapes, so the tool knows plank can step aside,
//! and [`FILE_ENV_VAR`] naming a fresh signal file for that one run. The tool
//! asks for the GPU by writing one line into that file (the primary signal:
//! it survives pipes, `2>&1` and `| tail`, which replace the exit status), or
//! by exiting with [`EXIT_CODE`] and printing a line starting with [`MARKER`].
//! When a foreground bash call or a shell escape ends either way, the agent saves the
//! live KV, drops the engine (which frees the Metal state and the model lock),
//! runs the command once more, reopens the model with its startup parameters
//! and restores the KV (`docs/ARCHITECTURE.md`, "GPU yield").
//!
//! This module holds the pure half: the detection rule, the step order
//! ([`run_cycle`] over a [`CycleHost`]), the placeholder engine that fills the
//! slot while the model is out, and the snapshot file. The agent supplies the
//! host in `ui.rs`.

use std::path::{Path, PathBuf};

use crate::engine::{Engine, EngineError, EngineEvent, GenerationOptions, GenerationStats, Prompt};

/// Exit status a command uses to say it could not get the GPU (`EX_TEMPFAIL`).
pub const EXIT_CODE: i64 = 75;

/// Start of the line a command prints alongside [`EXIT_CODE`].
pub const MARKER: &str = "GPU not available";

/// Environment variable plank sets to `1` in the bash tool's jobs.
pub const ENV_VAR: &str = "PLANK_GPU_YIELD";

/// Environment variable naming the run's signal file (see [`signal_path`]).
/// plank never creates the file; a tool that wants the GPU writes one line
/// into it, whatever it then exits with.
pub const FILE_ENV_VAR: &str = "PLANK_GPU_YIELD_FILE";

/// Longest signal line kept for the notice, in characters.
const SIGNAL_LINE_MAX: usize = 200;

/// Most bytes of a signal file ever read: one line is all it may carry.
const SIGNAL_READ_MAX: u64 = 4096;

/// A fresh signal-file path for one command run, in a directory only this
/// user can enter (`$TMPDIR/plank-gpu-yield-<uid>`, mode 0700). The file
/// itself is not created. `None` when the directory cannot be made safe, in
/// which case the run gets no [`FILE_ENV_VAR`] and only the exit-status rule
/// applies.
#[must_use]
pub fn signal_path() -> Option<PathBuf> {
    let dir = signal_dir(&std::env::temp_dir())?;
    Some(dir.join(format!("signal-{}", nonce())))
}

/// Creates (or checks) the per-user signal directory under `base`: a real
/// directory, owned by this user, mode 0700.
fn signal_dir(base: &Path) -> Option<PathBuf> {
    use std::os::unix::fs::{DirBuilderExt as _, MetadataExt as _, PermissionsExt as _};
    // SAFETY: getuid(2) has no preconditions and cannot fail.
    let uid = unsafe { libc::getuid() };
    let dir = base.join(format!("plank-gpu-yield-{uid}"));
    match std::fs::DirBuilder::new().mode(0o700).create(&dir) {
        Ok(()) => return Some(dir),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(_) => return None,
    }
    let meta = std::fs::symlink_metadata(&dir).ok()?;
    if !meta.is_dir() || meta.uid() != uid {
        return None;
    }
    if meta.permissions().mode() & 0o777 != 0o700 {
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).ok()?;
    }
    Some(dir)
}

/// Reads a run's signal file and deletes it, whatever it held. Returns the
/// first non-blank line (control characters dropped, capped at
/// [`SIGNAL_LINE_MAX`] characters) when the file exists and is non-empty, or
/// [`MARKER`] when it holds only blanks; `None` when there is no file or it
/// is empty.
#[must_use]
pub fn take_signal(path: &Path) -> Option<String> {
    use std::io::Read as _;
    // A FIFO (or anything else a hostile or confused tool leaves at this
    // path) must never be opened for a blocking read here: nothing on the
    // other end may ever write or close it, and plank would hang. Check the
    // file type first with symlink_metadata (which does not open the file)
    // and only read when it is a regular file. Whatever is found, the path
    // is removed afterward so a stale entry never lingers.
    let is_file = std::fs::symlink_metadata(path).is_ok_and(|m| m.is_file());
    let mut bytes = Vec::new();
    let read = is_file
        && std::fs::File::open(path)
            .and_then(|f| f.take(SIGNAL_READ_MAX).read_to_end(&mut bytes))
            .is_ok();
    let _ = std::fs::remove_file(path);
    if !read || bytes.is_empty() {
        return None;
    }
    let text = String::from_utf8_lossy(&bytes);
    let line = text
        .lines()
        .map(|l| {
            l.chars()
                .filter(|c| !c.is_control())
                .take(SIGNAL_LINE_MAX)
                .collect::<String>()
        })
        .map(|l| l.trim().to_owned())
        .find(|l| !l.is_empty());
    Some(line.unwrap_or_else(|| MARKER.to_owned()))
}

/// Age past which an untouched `signal-*` file in the signal directory is
/// swept as stray: one hour, generously longer than any real command's run.
const STALE_SIGNAL_AGE: std::time::Duration = std::time::Duration::from_secs(3600);

/// Removes every `signal-*` file under `dir` last modified more than `max_age`
/// before `now`, ignoring anything else in the directory (including a
/// subdirectory, or a name `take_signal` would never have produced) and any
/// error reading a single entry, since this is a best-effort tidy-up, not a
/// correctness requirement.
///
/// A file can be left behind when its command wrote it but plank never
/// called [`take_signal`] on it (a crash between write and read, or a job
/// dropped without going through the foreground path). Pure and taking an
/// injected directory and clock so it is testable without touching a real
/// `~/.plank`; see [`sweep_stray_signals`] for the real wiring.
fn sweep_stale_signals_in(dir: &Path, now: std::time::SystemTime, max_age: std::time::Duration) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with("signal-") {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        let Ok(modified) = meta.modified() else {
            continue;
        };
        let Ok(age) = now.duration_since(modified) else {
            continue;
        };
        if age > max_age {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Sweeps stray `signal-*` files older than [`STALE_SIGNAL_AGE`] out of the
/// real per-user signal directory. Best-effort: a directory that cannot be
/// created or read (see [`signal_dir`]) simply means nothing is swept.
///
/// Meant to run once, at startup or at first use of the GPU-yield machinery,
/// so a command that wrote a signal file and then crashed before plank read
/// it does not leave that file behind forever.
pub fn sweep_stray_signals() {
    if let Some(dir) = signal_dir(&std::env::temp_dir()) {
        sweep_stale_signals_in(&dir, std::time::SystemTime::now(), STALE_SIGNAL_AGE);
    }
}

/// Reopens the local engine with the parameters it was first opened with.
///
/// Built once at startup by the front end that opened the model, and called
/// only after the previous engine has been dropped: two live models in one
/// process would fight over the instance lock.
pub type ReopenFn = Box<dyn FnMut() -> Result<Box<dyn Engine>, String> + Send>;

/// Whether a finished command asked for the GPU: exit status [`EXIT_CODE`]
/// *and* some line of its output starting with [`MARKER`].
///
/// Both are required. Exit 75 alone is a generic "try again later", and the
/// marker alone could be a command merely printing someone else's message. The
/// marker must open its line, so a log line quoting it does not count.
#[must_use]
pub fn needs_gpu(exit_status: i64, output: &str) -> bool {
    exit_status == EXIT_CODE && output.lines().any(|l| l.starts_with(MARKER))
}

/// [`needs_gpu`] over output captured as two streams (a `!` or `!!` shell
/// escape): the marker counts on either.
#[must_use]
pub fn streams_need_gpu(exit_status: i64, stdout: &str, stderr: &str) -> bool {
    needs_gpu(exit_status, stdout) || needs_gpu(exit_status, stderr)
}

/// [`needs_gpu`] over an output file too large to hold as one string: the
/// bash tool's observation may show only a head, but the file has everything.
/// An unreadable file is no marker.
#[must_use]
pub fn output_file_needs_gpu(exit_status: i64, path: &Path) -> bool {
    use std::io::BufRead as _;
    if exit_status != EXIT_CODE {
        return false;
    }
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    std::io::BufReader::new(file)
        .split(b'\n')
        .map_while(Result::ok)
        .any(|line| line.starts_with(MARKER.as_bytes()))
}

/// Checks that every file a reopen will map is still a readable file, each
/// named by its role (`model`, `DSpark draft model`, `vision encoder`).
///
/// The C `model_open` calls `exit` on a file it cannot open or map instead of
/// returning an error, so a model deleted or moved while it was unloaded would
/// end plank with no message. Checked on the Rust side first, the same case is
/// an ordinary failed reopen: the placeholder stays and says which file.
///
/// # Errors
/// Names the first file that is missing, unreadable or not a regular file.
pub fn check_model_files<'p>(
    files: impl IntoIterator<Item = (&'p str, &'p Path)>,
) -> Result<(), String> {
    for (role, path) in files {
        let checked = std::fs::File::open(path)
            .and_then(|f| f.metadata())
            .and_then(|m| {
                if m.is_file() {
                    Ok(())
                } else {
                    Err(std::io::Error::other("not a regular file"))
                }
            });
        if let Err(e) = checked {
            return Err(format!(
                "the {role} file {} cannot be read ({e})",
                path.display()
            ));
        }
    }
    Ok(())
}

/// How a foreground bash call's command ended, recorded by the bash tool when
/// the command finished inside the call. A job still running when the call
/// returned (a background job) leaves no record, so it never cycles.
#[derive(Debug, Clone)]
pub struct ForegroundExit {
    /// The command's exit status.
    pub exit_status: i64,
    /// Whether the command asked for the GPU: its signal file, or
    /// [`needs_gpu`] over its whole output.
    pub needs_gpu: bool,
    /// The line the command wrote into its signal file, when it did.
    pub signal: Option<String>,
    /// The sandbox policy the command actually ran under, grants included, so
    /// the re-run needs no second permission prompt. `None` ran unsandboxed.
    pub sandbox: Option<crate::sandbox::Sandbox>,
}

/// The one line shown when the cycle starts, quoting the command's signal
/// line when it wrote one.
#[must_use]
pub fn notice(command: &str, signal: Option<&str>) -> String {
    let first = command.lines().next().unwrap_or("").trim();
    let name = first.split_whitespace().next().unwrap_or("the command");
    match signal {
        Some(line) => format!(
            "plank: {name} needs the GPU ({line}); unloading the model and running it again"
        ),
        None => format!("plank: {name} needs the GPU; unloading the model and running it again"),
    }
}

/// The steps of one cycle, as the agent performs them. Split out so the order
/// is a tested property of [`run_cycle`] rather than of one long method.
pub trait CycleHost {
    /// What [`save`](Self::save) captured.
    type Snapshot;
    /// What [`rerun`](Self::rerun) produces: a tool result for a bash call,
    /// the captured streams for a shell escape.
    type Output;
    /// Shows the notice through the front end's status path.
    fn notice(&mut self, text: &str);
    /// Captures the live KV, or `None` when there is none to capture (no
    /// session yet, or state a snapshot cannot carry, such as images).
    fn save(&mut self) -> Option<Self::Snapshot>;
    /// Drops the engine, freeing the GPU and the model lock.
    fn release(&mut self);
    /// Runs the command (again, after a refusal; for the first time, for a
    /// `suspend_model` call), to exit, and returns its result.
    fn rerun(&mut self) -> Self::Output;
    /// Reopens the engine.
    ///
    /// # Errors
    /// Returns why the model could not be loaded.
    fn reopen(&mut self) -> Result<(), String>;
    /// Restores `snapshot` into the reopened engine.
    ///
    /// # Errors
    /// Returns why the engine refused the bytes.
    fn restore(&mut self, snapshot: Self::Snapshot) -> Result<(), String>;
    /// Leaves the reopened engine to rebuild from the transcript, after a
    /// missing or refused snapshot.
    fn rebuild(&mut self, why: &str);
    /// Keeps `snapshot` for a later reopen attempt, after this one failed.
    fn park(&mut self, snapshot: Option<Self::Snapshot>, error: &str);
}

/// How a cycle ended, for the caller's diagnostics and for tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CycleEnd {
    /// Reopened and the KV came back exactly.
    Restored,
    /// Reopened, but the next pass rebuilds from the transcript.
    Rebuilt,
    /// The model could not be reopened; the engine slot holds a placeholder.
    Unloaded(String),
}

/// Runs one cycle: notice, save, release, re-run, reopen, restore. Returns the
/// re-run's result and how the engine came back. `signal` is the line the
/// first run wrote into its signal file, for the notice.
///
/// The re-run happens exactly once, whatever it returns: a command that fails
/// the same way again gets that result, and the model is reloaded regardless.
pub fn run_cycle<H: CycleHost>(
    host: &mut H,
    command: &str,
    signal: Option<&str>,
) -> (H::Output, CycleEnd) {
    cycle_after_notice(host, &notice(command, signal))
}

/// The notice for a `bash` call the model sent with `suspend_model`.
pub const SUSPEND_NOTICE: &str = "plank: suspending the model for this command";

/// Runs the proactive cycle for a `bash` call sent with `suspend_model`: the
/// same steps as [`run_cycle`], but there was no failed first run, so
/// [`CycleHost::rerun`] is the command's only run and [`SUSPEND_NOTICE`] the
/// notice.
pub fn run_suspended<H: CycleHost>(host: &mut H) -> (H::Output, CycleEnd) {
    cycle_after_notice(host, SUSPEND_NOTICE)
}

/// The shared body of [`run_cycle`] and [`run_suspended`].
fn cycle_after_notice<H: CycleHost>(host: &mut H, text: &str) -> (H::Output, CycleEnd) {
    host.notice(text);
    let snapshot = host.save();
    host.release();
    let output = host.rerun();
    if let Err(e) = host.reopen() {
        host.park(snapshot, &e);
        return (output, CycleEnd::Unloaded(e));
    }
    let end = if let Some(snap) = snapshot {
        match host.restore(snap) {
            Ok(()) => CycleEnd::Restored,
            Err(e) => {
                host.rebuild(&format!("KV restore refused ({e})"));
                CycleEnd::Rebuilt
            }
        }
    } else {
        host.rebuild("no KV snapshot to restore");
        CycleEnd::Rebuilt
    };
    (output, end)
}

/// A KV snapshot taken for the cycle.
///
/// Written to a file and dropped from memory where possible: on unified
/// memory, a multi-gigabyte blob held in RAM is memory the child needs.
#[derive(Debug)]
pub enum Snapshot {
    /// Persisted under a one-time signature, which is the only key the file
    /// is trusted by.
    Disk {
        /// The snapshot file.
        path: PathBuf,
        /// The nonce it was signed with.
        signature: String,
    },
    /// Kept in memory because the file could not be written.
    Memory(crate::kvcache::KVCache),
}

impl Snapshot {
    /// Persists `cache` to a fresh file under `dir`, falling back to keeping
    /// it in memory.
    #[must_use]
    pub fn store(cache: crate::kvcache::KVCache, dir: &Path) -> Self {
        let signature = nonce();
        let path = dir.join(format!("plank-gpu-yield-{signature}.kv"));
        match cache.persist(&path, &signature) {
            Ok(()) => Self::Disk { path, signature },
            Err(_) => Self::Memory(cache),
        }
    }

    /// Reads the snapshot back, removing its file either way.
    ///
    /// # Errors
    /// Returns a message when the file is missing or does not decode under its
    /// signature.
    pub fn load(self) -> Result<crate::kvcache::KVCache, String> {
        match self {
            Self::Memory(cache) => Ok(cache),
            Self::Disk { path, signature } => {
                let cache = crate::kvcache::KVCache::from_file(&path, &signature);
                let _ = std::fs::remove_file(&path);
                cache.ok_or_else(|| format!("snapshot {} did not read back", path.display()))
            }
        }
    }

    /// Deletes the snapshot's file without reading it.
    pub fn discard(self) {
        if let Self::Disk { path, .. } = self {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// A signature no other snapshot shares: pid, a process-wide counter and the
/// clock, so two cycles in one process, or two processes, never collide.
fn nonce() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!(
        "{}-{}-{nanos}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    )
}

/// Fills an engine slot while the model is unloaded.
///
/// Normally present only for the length of the re-run. When the reopen fails
/// it stays, answering every generation with an error that says what happened,
/// until the next turn's retry succeeds.
#[derive(Debug)]
pub struct UnloadedEngine {
    ctx_size: i32,
    model_name: String,
    reason: String,
}

impl UnloadedEngine {
    /// A placeholder standing in for an engine with this context size and
    /// model name, failing generations with `reason`.
    #[must_use]
    pub fn new(ctx_size: i32, model_name: String, reason: String) -> Self {
        Self {
            ctx_size,
            model_name,
            reason,
        }
    }
}

impl Engine for UnloadedEngine {
    fn generate(
        &mut self,
        _prompt: Prompt<'_>,
        _opts: &GenerationOptions,
        _interrupt: &dyn Fn() -> bool,
        _greedy: &dyn Fn() -> bool,
        _on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError> {
        Err(EngineError::new(self.reason.clone()))
    }

    fn ctx_size(&self) -> i32 {
        self.ctx_size
    }

    fn model_name(&self) -> String {
        self.model_name.clone()
    }

    fn is_local(&self) -> bool {
        true
    }

    fn is_gpu_placeholder(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_75_with_the_marker_at_line_start_needs_the_gpu() {
        let out = "loading\nGPU not available: held by plank (PID 1234); rerun later\n";
        assert!(needs_gpu(75, out));
    }

    #[test]
    fn exit_75_without_the_marker_does_not() {
        assert!(!needs_gpu(75, "temporary failure, try again\n"));
        assert!(!needs_gpu(75, ""));
    }

    #[test]
    fn the_marker_with_another_exit_status_does_not() {
        assert!(!needs_gpu(0, "GPU not available: whatever\n"));
        assert!(!needs_gpu(1, "GPU not available: whatever\n"));
    }

    #[test]
    fn the_marker_must_open_its_line() {
        assert!(!needs_gpu(75, "error: GPU not available: busy\n"));
        assert!(!needs_gpu(75, "  GPU not available\n"));
        // A CRLF line still starts with it.
        assert!(needs_gpu(75, "x\r\nGPU not available\r\n"));
    }

    #[test]
    fn either_captured_stream_can_carry_the_marker() {
        assert!(streams_need_gpu(75, "GPU not available\n", ""));
        assert!(streams_need_gpu(75, "", "GPU not available: held\n"));
        assert!(!streams_need_gpu(75, "busy\n", "try later\n"));
        assert!(!streams_need_gpu(1, "", "GPU not available\n"));
    }

    #[test]
    fn the_file_scan_matches_the_string_rule() {
        let dir = std::env::temp_dir().join(format!("plank-gpuyield-scan-{}", nonce()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out");
        std::fs::write(&path, "a\nsaid GPU not available\n").unwrap();
        assert!(!output_file_needs_gpu(75, &path));
        std::fs::write(&path, "a\nGPU not available: held\n").unwrap();
        assert!(output_file_needs_gpu(75, &path));
        assert!(!output_file_needs_gpu(0, &path));
        assert!(!output_file_needs_gpu(75, &dir.join("missing")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_or_unreadable_model_file_is_named() {
        let dir = std::env::temp_dir().join(format!("plank-gpuyield-files-{}", nonce()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("m.gguf");
        std::fs::write(&model, b"GGUF").unwrap();
        let mtp = dir.join("m.mtp.gguf");
        assert!(check_model_files([("model", model.as_path())]).is_ok());
        let err = check_model_files([("model", model.as_path()), ("DSpark draft model", &mtp)])
            .unwrap_err();
        assert!(err.contains("DSpark draft model"), "{err}");
        assert!(err.contains("m.mtp.gguf"), "{err}");
        // A directory where the file was is not a model either.
        let err = check_model_files([("vision encoder", dir.as_path())]).unwrap_err();
        assert!(err.contains("vision encoder"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_notice_names_the_command() {
        assert_eq!(
            notice("mex generate --prompt cat", None),
            "plank: mex needs the GPU; unloading the model and running it again"
        );
        assert_eq!(
            notice("mex x.md", Some("GPU not available: held by plank")),
            "plank: mex needs the GPU (GPU not available: held by plank); \
             unloading the model and running it again"
        );
    }

    #[test]
    fn each_signal_path_is_fresh_in_a_private_directory_and_not_created() {
        use std::os::unix::fs::PermissionsExt as _;
        let a = signal_path().expect("a path");
        let b = signal_path().expect("a path");
        assert_ne!(a, b);
        assert_eq!(a.parent(), b.parent());
        assert!(!a.exists(), "plank never creates the file");
        let dir = a.parent().unwrap();
        assert!(dir.starts_with(std::env::temp_dir()));
        let mode = std::fs::metadata(dir).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o700);
    }

    #[test]
    fn a_loose_signal_directory_is_tightened_and_a_file_in_its_place_refused() {
        use std::os::unix::fs::PermissionsExt as _;
        let base = std::env::temp_dir().join(format!("plank-gpuyield-sigdir-{}", nonce()));
        std::fs::create_dir_all(&base).unwrap();
        let dir = signal_dir(&base).expect("created");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(signal_dir(&base).as_deref(), Some(dir.as_path()));
        let mode = std::fs::metadata(&dir).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o700);
        std::fs::remove_dir(&dir).unwrap();
        std::fs::write(&dir, b"").unwrap();
        assert!(signal_dir(&base).is_none(), "not a directory");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_signal_is_read_once_and_the_file_always_goes() {
        let dir = std::env::temp_dir().join(format!("plank-gpuyield-sig-{}", nonce()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("signal");
        assert_eq!(take_signal(&path), None, "no file");
        std::fs::write(&path, "\nGPU not available: held\x07 by plank\nmore\n").unwrap();
        assert_eq!(
            take_signal(&path).as_deref(),
            Some("GPU not available: held by plank")
        );
        assert!(!path.exists());
        std::fs::write(&path, "").unwrap();
        assert_eq!(take_signal(&path), None, "an empty file is no signal");
        assert!(!path.exists(), "deleted anyway");
        std::fs::write(&path, " \n").unwrap();
        assert_eq!(take_signal(&path).as_deref(), Some(MARKER));
        std::fs::write(&path, "x".repeat(10_000)).unwrap();
        assert_eq!(take_signal(&path).map(|l| l.len()), Some(SIGNAL_LINE_MAX));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stray_signal_sweep_removes_only_old_matching_files() {
        let dir = std::env::temp_dir().join(format!("plank-gpuyield-stale-{}", nonce()));
        std::fs::create_dir_all(&dir).unwrap();
        let old = dir.join("signal-old");
        let young = dir.join("signal-young");
        let other = dir.join("not-a-signal");
        std::fs::write(&old, "x").unwrap();
        std::fs::write(&young, "x").unwrap();
        std::fs::write(&other, "x").unwrap();
        let now = std::time::SystemTime::now();
        let hour = std::time::Duration::from_secs(3600);
        // Backdate the "old" file's mtime by two hours; leave the others alone.
        let times = std::fs::FileTimes::new().set_modified(now - hour * 2);
        std::fs::File::options()
            .write(true)
            .open(&old)
            .unwrap()
            .set_times(times)
            .unwrap();
        sweep_stale_signals_in(&dir, now, hour);
        assert!(!old.exists(), "old signal file is removed");
        assert!(young.exists(), "young signal file is kept");
        assert!(other.exists(), "a non-matching name is left alone");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_fifo_at_the_signal_path_is_not_read_and_is_removed() {
        let dir = std::env::temp_dir().join(format!("plank-gpuyield-fifo-{}", nonce()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("signal");
        let c_path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
        // SAFETY: c_path is a valid, NUL-terminated string for a path in a
        // directory we just created; mkfifo has no other preconditions.
        let rc = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
        assert_eq!(rc, 0, "mkfifo failed: {}", std::io::Error::last_os_error());
        // If take_signal opened this for a blocking read, this call would
        // hang forever (nothing ever opens the FIFO for writing). Returning
        // at all is the test.
        assert_eq!(take_signal(&path), None);
        assert!(!path.exists(), "the FIFO is removed either way");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Records every step in order, with scripted outcomes.
    #[derive(Default)]
    struct Recorder {
        log: Vec<String>,
        have_snapshot: bool,
        reopen_fails: bool,
        restore_fails: bool,
        reruns: usize,
    }

    impl CycleHost for Recorder {
        type Snapshot = &'static str;
        type Output = String;
        fn notice(&mut self, _text: &str) {
            self.log.push("notice".into());
        }
        fn save(&mut self) -> Option<&'static str> {
            self.log.push("save".into());
            self.have_snapshot.then_some("kv")
        }
        fn release(&mut self) {
            self.log.push("release".into());
        }
        fn rerun(&mut self) -> String {
            self.reruns += 1;
            self.log.push("rerun".into());
            "second run\n".into()
        }
        fn reopen(&mut self) -> Result<(), String> {
            self.log.push("reopen".into());
            if self.reopen_fails {
                Err("no model".into())
            } else {
                Ok(())
            }
        }
        fn restore(&mut self, snap: &'static str) -> Result<(), String> {
            self.log.push(format!("restore:{snap}"));
            if self.restore_fails {
                Err("signature".into())
            } else {
                Ok(())
            }
        }
        fn rebuild(&mut self, _why: &str) {
            self.log.push("rebuild".into());
        }
        fn park(&mut self, snap: Option<&'static str>, _error: &str) {
            self.log.push(format!("park:{}", snap.unwrap_or("-")));
        }
    }

    #[test]
    fn the_cycle_runs_save_release_rerun_reopen_restore_in_order() {
        let mut host = Recorder {
            have_snapshot: true,
            ..Recorder::default()
        };
        let (out, end) = run_cycle(&mut host, "mex x", None);
        assert_eq!(out, "second run\n");
        assert_eq!(end, CycleEnd::Restored);
        assert_eq!(
            host.log,
            ["notice", "save", "release", "rerun", "reopen", "restore:kv"]
        );
        assert_eq!(host.reruns, 1);
    }

    #[test]
    fn a_suspended_run_takes_the_same_steps_with_its_own_notice() {
        let mut host = Recorder {
            have_snapshot: true,
            ..Recorder::default()
        };
        let (out, end) = run_suspended(&mut host);
        assert_eq!(out, "second run\n");
        assert_eq!(end, CycleEnd::Restored);
        assert_eq!(
            host.log,
            ["notice", "save", "release", "rerun", "reopen", "restore:kv"]
        );
        assert_eq!(host.reruns, 1);
    }

    #[test]
    fn a_refused_restore_falls_back_to_rebuild() {
        let mut host = Recorder {
            have_snapshot: true,
            restore_fails: true,
            ..Recorder::default()
        };
        let (_, end) = run_cycle(&mut host, "mex", None);
        assert_eq!(end, CycleEnd::Rebuilt);
        assert_eq!(host.log.last().map(String::as_str), Some("rebuild"));
    }

    #[test]
    fn no_snapshot_rebuilds_without_a_restore() {
        let mut host = Recorder::default();
        let (_, end) = run_cycle(&mut host, "mex", None);
        assert_eq!(end, CycleEnd::Rebuilt);
        assert!(!host.log.iter().any(|s| s.starts_with("restore")));
    }

    #[test]
    fn a_failed_reopen_parks_the_snapshot_and_still_returns_the_rerun() {
        let mut host = Recorder {
            have_snapshot: true,
            reopen_fails: true,
            ..Recorder::default()
        };
        let (out, end) = run_cycle(&mut host, "mex", None);
        assert_eq!(out, "second run\n");
        assert_eq!(end, CycleEnd::Unloaded("no model".into()));
        assert_eq!(host.log.last().map(String::as_str), Some("park:kv"));
    }

    #[test]
    fn a_snapshot_round_trips_through_its_file_and_the_file_goes() {
        let dir = std::env::temp_dir().join(format!("plank-gpuyield-snap-{}", nonce()));
        let cache = crate::kvcache::KVCache::new(
            vec![1, 2, 3, 4],
            crate::ds4tokens::TokenTranscript::new(),
        );
        let snap = Snapshot::store(cache, &dir);
        let Snapshot::Disk { path, .. } = &snap else {
            panic!("expected a file snapshot");
        };
        let path = path.clone();
        assert!(path.exists());
        let back = snap.load().expect("reads back");
        assert_eq!(back.kv(), &[1, 2, 3, 4]);
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_placeholder_refuses_to_generate_and_keeps_the_shape() {
        let mut e = UnloadedEngine::new(4096, "ds4".into(), "model unloaded".into());
        assert_eq!(e.ctx_size(), 4096);
        assert!(!e.can_release_gpu());
        assert!(e.is_gpu_placeholder());
        let err = e
            .generate(
                Prompt::Flat(""),
                &GenerationOptions::default(),
                &|| false,
                &|| false,
                &mut |_| {},
            )
            .unwrap_err();
        assert_eq!(err.to_string(), "model unloaded");
    }
}
