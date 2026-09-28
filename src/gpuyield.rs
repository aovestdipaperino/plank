// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! GPU yield: unloading the local model so a child command can have the GPU.
//!
//! A command that needs the GPU plank's loaded model occupies (the first is
//! `mex`, which runs a diffusion model on Metal) follows a small protocol: it
//! exits with [`EXIT_CODE`] and prints a line starting with [`MARKER`]. plank
//! exports [`ENV_VAR`]`=1` into every bash-tool job so the tool knows plank can
//! step aside. When a foreground bash call ends that way, the agent saves the
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
    /// [`needs_gpu`] over the command's whole output.
    pub needs_gpu: bool,
    /// The sandbox policy the command actually ran under, grants included, so
    /// the re-run needs no second permission prompt. `None` ran unsandboxed.
    pub sandbox: Option<crate::sandbox::Sandbox>,
}

/// The one line shown when the cycle starts.
#[must_use]
pub fn notice(command: &str) -> String {
    let first = command.lines().next().unwrap_or("").trim();
    let name = first.split_whitespace().next().unwrap_or("the command");
    format!("plank: {name} needs the GPU; unloading the model and running it again")
}

/// The steps of one cycle, as the agent performs them. Split out so the order
/// is a tested property of [`run_cycle`] rather than of one long method.
pub trait CycleHost {
    /// What [`save`](Self::save) captured.
    type Snapshot;
    /// Shows the notice through the front end's status path.
    fn notice(&mut self, text: &str);
    /// Captures the live KV, or `None` when there is none to capture (no
    /// session yet, or state a snapshot cannot carry, such as images).
    fn save(&mut self) -> Option<Self::Snapshot>;
    /// Drops the engine, freeing the GPU and the model lock.
    fn release(&mut self);
    /// Runs the command again and returns its tool result.
    fn rerun(&mut self) -> String;
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
/// re-run's tool result and how the engine came back.
///
/// The re-run happens exactly once, whatever it returns: a command that fails
/// the same way again gets that result, and the model is reloaded regardless.
pub fn run_cycle<H: CycleHost>(host: &mut H, command: &str) -> (String, CycleEnd) {
    host.notice(&notice(command));
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
            notice("mex generate --prompt cat"),
            "plank: mex needs the GPU; unloading the model and running it again"
        );
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
        let (out, end) = run_cycle(&mut host, "mex x");
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
        let (_, end) = run_cycle(&mut host, "mex");
        assert_eq!(end, CycleEnd::Rebuilt);
        assert_eq!(host.log.last().map(String::as_str), Some("rebuild"));
    }

    #[test]
    fn no_snapshot_rebuilds_without_a_restore() {
        let mut host = Recorder::default();
        let (_, end) = run_cycle(&mut host, "mex");
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
        let (out, end) = run_cycle(&mut host, "mex");
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
