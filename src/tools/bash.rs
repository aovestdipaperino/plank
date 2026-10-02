// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Asynchronous bash jobs: spawn, poll output, and stop shell commands.
//!
//! Port of the "Asynchronous Bash Jobs" section of `ds4_agent.c`. Bash
//! commands are tracked jobs, not blocking one-shot calls. Each job owns a
//! process, reader threads, and a temp output file. The first observation is
//! head-biased so headers and early errors are visible; later observations
//! are tail-biased.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::io::Read as _;
use std::os::unix::process::CommandExt as _;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::dsml::ToolCall;
use crate::sandbox::Protected;

use super::{ToolContext, parse_int_default, parse_timeout};

const BASH_HEAD_BYTES: usize = 8 * 1024;
const BASH_HEAD_LINES: usize = 100;
const BASH_TAIL_BYTES: usize = 32 * 1024;
const BASH_PROGRESS_TAIL_LINES: usize = 4;
const BASH_FINAL_TAIL_LINES: usize = 20;

/// Output counters updated by the reader threads.
#[derive(Debug, Default)]
struct Stats {
    bytes: u64,
    newlines: u64,
    last_byte: u8,
}

/// State shared between a job and its stream reader threads.
#[derive(Debug)]
struct Shared {
    sink: Mutex<(std::fs::File, Stats)>,
}

/// One tracked background shell command.
#[derive(Debug)]
// running/timed_out/sandboxed are independent process facts, not a state enum.
#[allow(clippy::struct_excessive_bools)]
struct BashJob {
    id: i64,
    pid: u32,
    child: Child,
    path: PathBuf,
    /// The run's GPU-yield signal file (`PLANK_GPU_YIELD_FILE`), read once
    /// by a foreground call and deleted when the job is reaped at the latest.
    signal: Option<PathBuf>,
    start: Instant,
    timeout: Duration,
    shared: Arc<Shared>,
    observed_once: bool,
    exit_status: i64,
    running: bool,
    timed_out: bool,
    sandboxed: bool,
}

/// Table of live and finished asynchronous bash jobs.
#[derive(Debug, Default)]
pub struct BashJobs {
    jobs: Vec<BashJob>,
    next_id: i64,
    /// How the last `bash` call's command ended, when it ended inside the
    /// call (`gpuyield`). Cleared by the agent before each dispatch; a job
    /// still running when the call returned leaves it `None`.
    pub last_foreground: Option<crate::gpuyield::ForegroundExit>,
    /// Set by the agent for a GPU-yield re-run: the sandbox decision the first
    /// run made, reused as-is so a one-command grant is not asked twice.
    pub replay_sandbox: Option<DecidedSandbox>,
    /// Set by the agent for a GPU-yield re-run: the next `bash` call waits
    /// for its command to exit (or hit its own timeout, or be interrupted)
    /// instead of returning after `refresh_sec`, because the model is reopened
    /// right after and must not share the GPU with a command still running.
    /// Consumed by that call.
    pub wait_to_exit: bool,
}

/// A sandbox decision already made for one command: the policy it ran under,
/// or `None` when it ran unsandboxed.
#[derive(Debug, Clone)]
pub struct DecidedSandbox(pub Option<crate::sandbox::Sandbox>);

fn spawn_reader(shared: &Arc<Shared>, mut stream: impl std::io::Read + Send + 'static) {
    let shared = Arc::clone(shared);
    std::thread::spawn(move || {
        let mut buf = [0_u8; 4096];
        loop {
            match stream.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    let chunk = &buf[..n];
                    let mut sink = shared.sink.lock().expect("bash output sink poisoned");
                    let (file, stats) = &mut *sink;
                    let _ = std::io::Write::write_all(file, chunk);
                    stats.bytes += n as u64;
                    stats.newlines += chunk
                        .iter()
                        .fold(0_u64, |acc, &b| acc + u64::from(b == b'\n'));
                    stats.last_byte = chunk[n - 1];
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => break,
            }
        }
    });
}

fn make_output_file(id: u64) -> Result<(PathBuf, std::fs::File), String> {
    for attempt in 0..100_u32 {
        let path = std::env::temp_dir().join(format!(
            "ds4_agent_output_{}_{id}_{attempt}",
            std::process::id()
        ));
        match std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
        {
            Ok(file) => return Ok((path, file)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => {
                return Err(format!("failed to create temporary output file: {e}"));
            }
        }
    }
    Err("failed to create temporary output file: too many collisions".to_string())
}

/// A process's exit code, or `128 + signal` when a signal ended it.
fn status_code(status: std::process::ExitStatus) -> i64 {
    status.code().map_or_else(
        || {
            use std::os::unix::process::ExitStatusExt;
            status.signal().map_or(-1, |sig| 128 + i64::from(sig))
        },
        i64::from,
    )
}

/// Exports the GPU-yield promise into `command` (`gpuyield`): `PLANK_GPU_YIELD`
/// and a fresh signal-file path, which is returned so the caller can read it
/// back and delete it.
fn export_gpu_yield(command: &mut Command) -> Option<PathBuf> {
    command.env(crate::gpuyield::ENV_VAR, "1");
    let signal = crate::gpuyield::signal_path();
    if let Some(path) = &signal {
        command.env(crate::gpuyield::FILE_ENV_VAR, path);
    }
    signal
}

/// How long a process group gets to exit after SIGTERM before SIGKILL.
const GROUP_KILL_GRACE: Duration = Duration::from_millis(500);

/// Terminates `child` together with every process in its process group and
/// reaps it.
///
/// Only meaningful for children spawned with `process_group(0)`, which makes
/// the child the leader of a group of its own (its pgid equals its pid, and
/// `sandbox-exec` keeps that pid because it execs the shell in place). A plain
/// `Child::kill` reaches only the shell, orphaning `sleep 9999 & wait` or
/// `cmd | tee` pipelines, which then outlive plank. SIGTERM goes first so
/// well-behaved tools can clean up; after [`GROUP_KILL_GRACE`] the group gets
/// SIGKILL. `wait()` always runs so nothing is left as a zombie. Signalling a
/// group that has already gone is harmless (ESRCH is ignored).
///
/// If the child has already exited on its own, it is reaped without
/// signalling: the PID may have been reused by an unrelated process, and
/// `kill(-pgid, ...)` would hit the wrong group.
pub(crate) fn kill_process_group(child: &mut Child) -> Option<std::process::ExitStatus> {
    // Reap an already-exited child before signalling. Without this, a
    // BashJob whose child died between polls would Drop, call terminate(),
    // and send SIGTERM to a PID that may have been reused by another test's
    // process — the root cause of cross-test SIGTERM flakes.
    if let Ok(Some(status)) = child.try_wait() {
        return Some(status);
    }
    #[allow(clippy::cast_possible_wrap)]
    let pgid = child.id() as libc::pid_t;
    if pgid <= 0 {
        return child.try_wait().ok().flatten();
    }
    // SAFETY: kill(2) with a negative pid signals a process group; it has no
    // memory-safety preconditions and any error (ESRCH, EPERM) is ignored.
    unsafe { libc::kill(-pgid, libc::SIGTERM) };
    let deadline = Instant::now() + GROUP_KILL_GRACE;
    let mut status = None;
    while Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(st)) => {
                status = Some(st);
                break;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(_) => break,
        }
    }
    // Even when the leader exited on SIGTERM, a grandchild may have ignored
    // it: SIGKILL the group unconditionally. The group id stays valid until
    // its last member is gone, so this is safe after the leader's reap.
    // SAFETY: as above.
    unsafe { libc::kill(-pgid, libc::SIGKILL) };
    status.or_else(|| child.wait().ok())
}

impl BashJob {
    fn stats(&self) -> (u64, u64, u8) {
        let sink = self.shared.sink.lock().expect("bash output sink poisoned");
        (sink.1.bytes, sink.1.newlines, sink.1.last_byte)
    }

    fn display_lines(&self) -> u64 {
        let (bytes, newlines, last_byte) = self.stats();
        if bytes == 0 {
            0
        } else {
            newlines + u64::from(last_byte != b'\n')
        }
    }

    fn finalize(&mut self, status: std::process::ExitStatus) {
        self.exit_status = status_code(status);
        self.running = false;
    }

    /// Drains output, notices process exit, and enforces the timeout.
    ///
    /// Called opportunistically by status/wait instead of a reaper thread,
    /// mirroring `agent_bash_poll`. Output draining is continuous via the
    /// reader threads.
    fn poll(&mut self) {
        if !self.running {
            return;
        }
        match self.child.try_wait() {
            Ok(Some(status)) => {
                self.finalize(status);
                return;
            }
            Ok(None) => {}
            Err(_) => {
                self.exit_status = -1;
                self.running = false;
                return;
            }
        }
        if self.start.elapsed() >= self.timeout {
            self.timed_out = true;
            self.terminate();
        }
    }

    /// Kills the job's whole process group and records its exit.
    fn terminate(&mut self) {
        if !self.running {
            return;
        }
        if let Some(status) = kill_process_group(&mut self.child) {
            self.finalize(status);
        } else {
            self.exit_status = -1;
            self.running = false;
        }
    }

    /// Reads the first `max_lines` of output with a byte cap, mirroring
    /// `agent_bash_read_head`.
    fn read_head(&self, max_lines: usize, max_bytes: usize) -> (String, u64, bool) {
        let Ok(mut file) = std::fs::File::open(&self.path) else {
            return ("<failed to reopen output file>\n".to_string(), 0, false);
        };
        let mut out = Vec::new();
        let mut lines = 0_usize;
        let mut buf = [0_u8; 4096];
        let mut byte_limited = false;
        'read: loop {
            let n = match file.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            };
            for &b in &buf[..n] {
                if lines >= max_lines || out.len() >= max_bytes {
                    byte_limited = out.len() >= max_bytes;
                    break 'read;
                }
                out.push(b);
                if b == b'\n' {
                    lines += 1;
                }
            }
        }
        let shown = lines as u64 + u64::from(!out.is_empty() && *out.last().unwrap() != b'\n');
        (
            String::from_utf8_lossy(&out).into_owned(),
            shown,
            byte_limited,
        )
    }

    /// Reads the last `max_lines` of output, mirroring
    /// `agent_bash_read_tail_lines`.
    fn read_tail_lines(&self, max_lines: usize) -> String {
        let Ok(mut file) = std::fs::File::open(&self.path) else {
            return "<failed to reopen output file>\n".to_string();
        };
        let mut tail: Vec<u8> = Vec::new();
        let mut buf = [0_u8; 4096];
        loop {
            let n = match file.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            };
            tail.extend_from_slice(&buf[..n]);
            if tail.len() > BASH_TAIL_BYTES {
                let drop = tail.len() - BASH_TAIL_BYTES;
                tail.drain(..drop);
            }
        }
        let mut start = 0;
        let mut newlines = 0;
        for (i, &b) in tail.iter().enumerate().rev() {
            if b == b'\n' {
                newlines += 1;
                if newlines > max_lines {
                    start = i + 1;
                    break;
                }
            }
        }
        String::from_utf8_lossy(&tail[start..]).into_owned()
    }

    /// Builds the tool result text, mirroring `agent_bash_observation`.
    fn observation(&mut self, mark_observed: bool) -> String {
        self.poll();
        let first_observation = !self.observed_once;
        let display_lines = self.display_lines();
        let (bytes, _, _) = self.stats();
        let elapsed = self.start.elapsed().as_secs_f64();

        let mut out = String::new();
        if self.running {
            let _ = writeln!(
                out,
                "bash job={} pid={} status=running elapsed_sec={elapsed:.1} timeout_sec={:.0}",
                self.id,
                self.pid,
                self.timeout.as_secs_f64()
            );
        } else {
            let _ = writeln!(
                out,
                "bash job={} pid={} status=done elapsed_sec={elapsed:.1} timed_out={}",
                self.id,
                self.pid,
                i32::from(self.timed_out)
            );
            let _ = writeln!(out, "exit_status={}", self.exit_status);
        }

        if bytes == 0 {
            out.push_str("<output>\n</output>\n");
        } else if first_observation {
            let (head, shown_lines, byte_limited) =
                self.read_head(BASH_HEAD_LINES, BASH_HEAD_BYTES);
            let truncated = byte_limited || display_lines > shown_lines;
            if !self.running && !truncated {
                out.push_str("<output>\n");
                out.push_str(&head);
                if !head.is_empty() && !head.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str("</output>\n");
            } else {
                let _ = writeln!(
                    out,
                    "output_path={} ({bytes} bytes, {display_lines} lines)",
                    self.path.display()
                );
                let _ = writeln!(out, "<head -{BASH_HEAD_LINES} {}>", self.path.display());
                out.push_str(&head);
                if !head.is_empty() && !head.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str("</head>\n");
            }
        } else {
            let tail_lines = if self.running {
                BASH_PROGRESS_TAIL_LINES
            } else {
                BASH_FINAL_TAIL_LINES
            };
            let tail = self.read_tail_lines(tail_lines);
            let _ = writeln!(
                out,
                "output_path={} ({bytes} bytes, {display_lines} lines)",
                self.path.display()
            );
            let _ = writeln!(out, "<tail -{tail_lines} {}>", self.path.display());
            out.push_str(&tail);
            if !tail.is_empty() && !tail.ends_with('\n') {
                out.push('\n');
            }
            out.push_str("</tail>\n");
        }
        if self.sandboxed
            && !self.running
            && self.exit_status != 0
            && self
                .read_tail_lines(BASH_FINAL_TAIL_LINES)
                .contains("Operation not permitted")
        {
            let _ = writeln!(
                out,
                "[sandbox blocked: this command ran under plank's write sandbox \
                 (writes allowed only under the working directory and temp dirs). \
                 If the failure is a legitimate write elsewhere, ask the user to add \
                 the path to writablePaths in ~/.plank/sandbox.json — the project's \
                 own .plank/sandbox.json cannot widen the sandbox.]"
            );
        }
        if self.running {
            let _ = writeln!(
                out,
                "\nUse bash_status job={} to get info before refresh time; \
                 use bash_stop job={} to stop execution",
                self.id, self.id
            );
        }
        if mark_observed {
            self.observed_once = true;
        }
        out
    }

    /// Waits up to `refresh_sec` for the job to finish, polling as it goes.
    ///
    /// A pending user interrupt (Ctrl-C in the REPL, Esc in the TUI) kills
    /// the job instead of waiting the refresh out: the shell runs in its own
    /// process group, so the terminal's SIGINT no longer reaches it and plank
    /// has to forward the intent itself. The flag is left set for the turn
    /// loop, which is what stops the generation.
    fn refresh_for(&mut self, refresh_sec: u64) {
        let deadline = Instant::now() + Duration::from_secs(refresh_sec);
        while self.running && Instant::now() < deadline {
            self.poll();
            if !self.running {
                break;
            }
            if crate::interrupt::pending() {
                self.terminate();
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        self.poll();
    }

    /// Waits until the job is no longer running, however long that takes.
    ///
    /// The job's own timeout still ends it (`poll` enforces it) and a user
    /// interrupt kills it as in [`refresh_for`](Self::refresh_for); nothing
    /// else does. `refresh_sec` is capped at an hour while a timeout may run to
    /// a day, so the refresh wait cannot stand in for this.
    fn wait_for_exit(&mut self) {
        while self.running {
            self.poll();
            if !self.running {
                break;
            }
            if crate::interrupt::pending() {
                self.terminate();
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

impl Drop for BashJob {
    fn drop(&mut self) {
        // Kill the group, not just the shell, so nothing outlives plank.
        self.terminate();
        // A background job that asked for the GPU is not retried, but its
        // signal file must not outlive it.
        if let Some(signal) = &self.signal {
            let _ = std::fs::remove_file(signal);
        }
        // The temp output file is intentionally kept: its path was shown to
        // the model as output_path and may be read with the file tools.
    }
}

impl BashJobs {
    /// Spawns a shell command as a new tracked job; returns its id.
    ///
    /// Mirrors `agent_bash_start`. stdin is `/dev/null` so the shell cannot
    /// disturb the interactive terminal.
    ///
    /// # Errors
    ///
    /// Returns a message describing why the process could not be started.
    pub fn start(
        &mut self,
        ctx_cwd: &std::path::Path,
        cmd: &str,
        timeout_sec: u64,
        sandbox: Option<&crate::sandbox::Sandbox>,
    ) -> Result<i64, String> {
        if self.next_id <= 0 {
            self.next_id = 1;
        }
        let id = self.next_id;
        let (path, file) = make_output_file(u64::try_from(id).unwrap_or(0))?;
        // When a sandbox policy applies, wrap the shell in sandbox-exec with
        // a generated Seatbelt profile (read everywhere, write only under
        // cwd/temp/configured roots). See src/sandbox.rs.
        let mut command = if let Some(sb) = sandbox {
            let mut c = Command::new("/usr/bin/sandbox-exec");
            c.arg("-p").arg(sb.profile(ctx_cwd)).arg("/bin/sh");
            c
        } else {
            Command::new("/bin/sh")
        };
        command.arg("-c").arg(cmd).current_dir(ctx_cwd);
        // Tells a GPU-bound tool that plank can unload its model on request:
        // a line in the signal file, or exit 75 plus the marker line.
        let signal = export_gpu_yield(&mut command);
        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Own process group: stop/timeout/Drop kill every descendant, and
            // the terminal's Ctrl-C reaches plank alone (see refresh_for).
            .process_group(0)
            .spawn()
            .map_err(|e| {
                std::fs::remove_file(&path).ok();
                format!("failed to fork: {e}")
            })?;
        self.next_id += 1;

        let shared = Arc::new(Shared {
            sink: Mutex::new((file, Stats::default())),
        });
        if let Some(stdout) = child.stdout.take() {
            spawn_reader(&shared, stdout);
        }
        if let Some(stderr) = child.stderr.take() {
            spawn_reader(&shared, stderr);
        }
        let pid = child.id();
        self.jobs.push(BashJob {
            id,
            pid,
            child,
            path,
            signal,
            start: Instant::now(),
            timeout: Duration::from_secs(timeout_sec),
            shared,
            observed_once: false,
            exit_status: -1,
            running: true,
            sandboxed: sandbox.is_some(),
            timed_out: false,
        });
        Ok(id)
    }

    /// Polls every job once, so timeouts are enforced even for jobs the model
    /// never asks about again.
    ///
    /// The per-job timeout used to run only inside that job's own poll: a
    /// `refresh_sec` job that was never revisited ran until session drop.
    /// Every bash-family tool call sweeps the whole table first.
    pub fn sweep(&mut self) {
        for job in &mut self.jobs {
            job.poll();
        }
    }

    /// Number of jobs still running: the footer's job count.
    #[must_use]
    pub fn running_count(&self) -> usize {
        self.jobs.iter().filter(|j| j.running).count()
    }

    /// A snapshot of every tracked job, for `/jobs` and the UI thread's
    /// panel (which cannot reach the table while the worker owns it).
    #[must_use]
    pub fn rows(&self) -> Vec<JobRow> {
        self.jobs
            .iter()
            .map(|job| JobRow {
                id: job.id,
                pid: job.pid,
                started: job.start,
                state: if job.running {
                    JobState::Running
                } else if job.timed_out {
                    JobState::TimedOut(job.exit_status)
                } else {
                    JobState::Done(job.exit_status)
                },
                path: job.path.clone(),
            })
            .collect()
    }

    /// One line per tracked job for `/jobs`; a fixed sentence when empty.
    #[must_use]
    pub fn render_table(&self) -> String {
        render_rows(&self.rows())
    }

    /// Whether any job in the table has finished (after a `sweep`).
    #[must_use]
    pub fn has_finished(&self) -> bool {
        self.jobs.iter().any(|j| !j.running)
    }

    /// Polls every job and removes those that finished without the model
    /// seeing the exit, returning each one's final observation.
    ///
    /// The table is the sole source of truth for "unannounced": a job the
    /// model observed as `status=done` through `bash`, `bash_status` or
    /// `bash_stop` is removed at that observation (`job_tool_result`), so
    /// anything still present and not running finished on its own. Each job
    /// therefore produces at most one notification, and none if the model
    /// polled it to completion itself (see `docs/BACKGROUND-TASKS.md`).
    pub fn take_finished(&mut self) -> Vec<String> {
        self.sweep();
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.jobs.len() {
            if self.jobs[i].running {
                i += 1;
                continue;
            }
            let mut job = self.jobs.remove(i);
            out.push(job.observation(true));
        }
        out
    }

    fn find(&self, id: i64, pid: u32) -> Option<usize> {
        self.jobs
            .iter()
            .position(|job| (id > 0 && job.id == id) || (id <= 0 && pid > 0 && job.pid == pid))
    }

    /// Common result path for `bash`, `bash_status`, and `bash_stop`.
    ///
    /// Mirrors `agent_bash_job_tool_result`.
    fn job_tool_result(
        &mut self,
        idx: usize,
        wait: bool,
        refresh_sec: u64,
        stop: bool,
        remove_if_done: bool,
    ) -> String {
        let job = &mut self.jobs[idx];
        if stop {
            job.terminate();
        }
        if wait || stop {
            job.refresh_for(refresh_sec);
        } else {
            job.poll();
        }
        let obs = job.observation(true);
        if remove_if_done && !job.running {
            self.jobs.remove(idx);
        }
        obs
    }

    /// The `bash` tool's wait-and-observe, which also records how the command
    /// ended in [`last_foreground`](Self::last_foreground) when it ended
    /// inside the call. The signal file is read (and deleted) first; the GPU
    /// marker is looked for in the whole output file, not in the observation,
    /// which may show only its head.
    fn foreground_result(
        &mut self,
        idx: usize,
        refresh_sec: u64,
        sandbox: Option<crate::sandbox::Sandbox>,
        wait_to_exit: bool,
    ) -> String {
        if wait_to_exit {
            self.jobs[idx].wait_for_exit();
        } else {
            self.jobs[idx].refresh_for(refresh_sec);
        }
        let job = &self.jobs[idx];
        // A pending interrupt means refresh_for/wait_for_exit killed the job
        // itself: the signal file or exit code may still say "needs the
        // GPU" (the kill can race a write, or land right after exit 75), but
        // trusting that would unload and reload the model for a run the user
        // asked to stop, not one that wants to try again. Esc/Ctrl-C must
        // never lead to a cycle.
        let interrupted = crate::interrupt::pending();
        self.last_foreground = (!job.running).then(|| {
            let signal = job.signal.as_deref().and_then(crate::gpuyield::take_signal);
            crate::gpuyield::ForegroundExit {
                exit_status: job.exit_status,
                needs_gpu: !interrupted
                    && (signal.is_some()
                        || crate::gpuyield::output_file_needs_gpu(job.exit_status, &job.path)),
                signal,
                sandbox,
            }
        });
        self.job_tool_result(idx, false, 0, false, true)
    }
}

/// Lifecycle of a tracked job as the `/jobs` panel reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobState {
    /// Still running.
    Running,
    /// Exited on its own with this status.
    Done(i64),
    /// Killed by its own timeout; the status is what the kill produced.
    TimedOut(i64),
}

/// One job as the `/jobs` panel sees it: a copy, so the UI thread can render
/// it while the worker thread owns the live table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobRow {
    /// The `job=` id the model uses.
    pub id: i64,
    /// The shell's pid.
    pub pid: u32,
    /// When the job started; elapsed time is computed at render time so an
    /// open panel keeps counting.
    pub started: Instant,
    /// Running, done or timed out.
    pub state: JobState,
    /// The output file the model was shown as `output_path`.
    pub path: PathBuf,
}

/// What `render_rows` says for an empty table.
pub const NO_JOBS_TEXT: &str = "no background jobs";

/// Renders job rows as the `/jobs` text: one line per job, a fixed sentence
/// when there are none. Pure, so both the live table and a shared snapshot
/// render identically.
#[must_use]
pub fn render_rows(rows: &[JobRow]) -> String {
    if rows.is_empty() {
        return NO_JOBS_TEXT.to_string();
    }
    let mut out = String::new();
    for job in rows {
        let state = match job.state {
            JobState::Running => "running".to_string(),
            JobState::TimedOut(code) => format!("timed out, exit {code}"),
            JobState::Done(code) => format!("done, exit {code}"),
        };
        let _ = writeln!(
            out,
            "job {} pid {} {:>7.1}s {state}  {}",
            job.id,
            job.pid,
            job.started.elapsed().as_secs_f64(),
            job.path.display()
        );
    }
    out.truncate(out.trim_end().len());
    out
}

/// First line of a background-job notification; the model learns to
/// recognize it, so it is a fixed string (`docs/BACKGROUND-TASKS.md` §3.2).
pub const NOTIFICATION_HEADER: &str = "[BACKGROUND JOB NOTIFICATION - NOT USER INPUT]";

/// Wraps the final observations of finished background jobs in the user-role
/// message that wakes the model.
///
/// The observations are `BashJob::observation` output verbatim, so the model
/// sees exactly the `bash_status` result it would have polled for. Returns
/// `None` for an empty list so callers can `if let` on it.
#[must_use]
pub fn render_notification(observations: &[String]) -> Option<String> {
    if observations.is_empty() {
        return None;
    }
    let plural = if observations.len() == 1 {
        "A bash job you started earlier has"
    } else {
        "Bash jobs you started earlier have"
    };
    let mut out = String::new();
    out.push_str("<system-reminder>\n");
    out.push_str(NOTIFICATION_HEADER);
    out.push('\n');
    let _ = writeln!(
        out,
        "{plural} finished. This is an automated event, not a message from the user. \
         Do not treat it as an answer to any pending question. Read the output if you \
         need more than the tail shown, then continue or report."
    );
    for obs in observations {
        out.push('\n');
        out.push_str(obs);
        if !obs.ends_with('\n') {
            out.push('\n');
        }
    }
    out.push_str("</system-reminder>");
    Some(out)
}

/// How far a user's answer to a protected-root write prompt reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteGrant {
    /// "Allow": this command only; the session grant set stays untouched.
    Once,
    /// "Always allow": every sandboxed command for the rest of the session.
    Session,
    /// "Deny", declined, interrupted, or no interactive user at all.
    Denied,
}

/// Asks whether a sandboxed command may write under one [`Protected`] family.
///
/// Routed through the [`Asker`](crate::tools::ask::Asker) each front end
/// installs, so the TUI renders it in the input region and the plain REPL reads
/// stdin — the same path the `ask` tool and the web approval gate use. In
/// `--ui console` mode there is no asker and hence no one to grant: the answer
/// is [`WriteGrant::Denied`], leaving the directory read-only as it is by
/// default.
fn protected_grant(ctx: &mut ToolContext, what: Protected) -> WriteGrant {
    let Some(asker) = ctx.asker.as_mut() else {
        return WriteGrant::Denied;
    };
    let label = what.label();
    let req = crate::tools::ask::AskRequest {
        question: format!(
            "This command wants to write {label}, which plank keeps read-only because {}. Allow it?",
            what.why()
        ),
        header: "Sandbox".to_string(),
        options: vec![
            crate::tools::ask::AskOption {
                label: "Allow".to_string(),
                description: format!("Allow writes to {label} for this command only"),
            },
            crate::tools::ask::AskOption {
                label: "Always allow".to_string(),
                description: format!("Allow writes to {label} for the rest of this session"),
            },
            crate::tools::ask::AskOption {
                label: "Deny".to_string(),
                description: format!("Run the command with {label} read-only"),
            },
        ],
        multi: false,
        allow_chat: false,
    };
    match asker.ask(req) {
        crate::tools::ask::AskOutcome::Answered(labels)
            if labels.iter().any(|l| l == "Always allow") =>
        {
            WriteGrant::Session
        }
        crate::tools::ask::AskOutcome::Answered(labels) if labels.iter().any(|l| l == "Allow") => {
            WriteGrant::Once
        }
        _ => WriteGrant::Denied,
    }
}

/// The `bash` tool's `suspend_model` parameter (`gpuyield`). Deliberately
/// absent from the trained bash schema, which must stay byte-identical to the
/// C (FINDINGS.md); plank's own prompt note teaches it instead.
pub const SUSPEND_MODEL_PARAM: &str = "suspend_model";

/// Whether `call` is a `bash` call asking plank to unload the model before
/// running it. Liberal in what it reads: `true`, `1` or `yes` in any case,
/// surrounding blanks ignored; any other value, or none, is false.
#[must_use]
pub fn suspend_model_requested(call: &ToolCall) -> bool {
    call.name == "bash"
        && call.arg_value(SUSPEND_MODEL_PARAM).is_some_and(|v| {
            let v = v.trim();
            ["true", "1", "yes"]
                .iter()
                .any(|t| v.eq_ignore_ascii_case(t))
        })
}

/// Whether `cmd` puts itself in the background with a trailing `&` (not
/// `&&`): the shell exits at once and the work goes on after the call, so
/// unloading the model around it would buy nothing.
#[must_use]
pub fn backgrounds_itself(cmd: &str) -> bool {
    let chars: Vec<char> = cmd.chars().collect();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if escaped {
            escaped = false;
            i += 1;
            continue;
        }
        match c {
            '\\' if !in_single => escaped = true,
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '&' if !in_single && !in_double => {
                if bare_ampersand_backgrounds(&chars, i) {
                    return true;
                }
                // `&&`, `>&` or `&>` consume the second character too, so the
                // loop below never re-examines it as its own operator.
                if chars.get(i + 1) == Some(&'&') || chars.get(i + 1) == Some(&'>') {
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    false
}

/// Whether the `&` at `chars[i]` is a bare backgrounding operator rather than
/// half of `&&`, `>&`, `&>` or `|&` (none of which end a foreground
/// subcommand). Quoting and escaping are resolved by the caller; this only
/// looks at the immediate neighbors.
fn bare_ampersand_backgrounds(chars: &[char], i: usize) -> bool {
    let prev = i.checked_sub(1).and_then(|j| chars.get(j));
    let next = chars.get(i + 1);
    !matches!(next, Some('&' | '>')) && !matches!(prev, Some('>' | '|'))
}

/// Implements the `bash` tool: start a job and wait up to `refresh_sec`.
pub fn tool_bash(ctx: &mut ToolContext, call: &ToolCall) -> String {
    let cmd = call.arg_value("command").unwrap_or("");
    if cmd.is_empty() {
        return "Tool error: bash requires command\n".to_string();
    }
    let timeout = parse_timeout(call.arg_value("timeout_sec"));
    let refresh = u64::try_from(parse_int_default(
        call.arg_value("refresh_sec"),
        60,
        1,
        3600,
    ))
    .unwrap_or(60);
    // A GPU-yield re-run reuses the first run's decision verbatim: asking
    // about a protected root again would turn one grant into two prompts.
    let sandbox = match ctx.bash.replay_sandbox.take() {
        Some(DecidedSandbox(decided)) => decided,
        None => resolve_sandbox(ctx, cmd),
    };
    ctx.bash.sweep();
    if let Err(err) = ctx
        .bash
        .start(&ctx.cwd.clone(), cmd, timeout, sandbox.as_ref())
    {
        return format!("Tool error: bash failed to start: {err}\n");
    }
    let idx = ctx.bash.jobs.len() - 1;
    let wait = std::mem::take(&mut ctx.bash.wait_to_exit);
    ctx.bash.foreground_result(idx, refresh, sandbox, wait)
}

/// Decides the sandbox policy one command runs under, asking about any
/// protected root it names. `None` runs it unsandboxed.
fn resolve_sandbox(ctx: &mut ToolContext, cmd: &str) -> Option<crate::sandbox::Sandbox> {
    // The protected roots are read-only under the sandbox unless the user says
    // otherwise. Ask only about the families the command actually names, and
    // only when it is not provably read-only, so ordinary commands and
    // `cat ~/.plank/...` never see a prompt.
    let mut once: BTreeSet<Protected> = BTreeSet::new();
    if ctx.sandbox.should_sandbox(cmd) && !crate::sandbox::is_read_only_command(cmd) {
        for what in crate::sandbox::protected_mentions(cmd, &ctx.sandbox.granted) {
            match protected_grant(ctx, what) {
                WriteGrant::Session => {
                    ctx.sandbox.granted.insert(what);
                }
                WriteGrant::Once => {
                    once.insert(what);
                }
                WriteGrant::Denied => {
                    ctx.publish_status(&format!(
                        "{} stays read-only for this command",
                        what.label()
                    ));
                }
            }
        }
    }
    if !ctx.sandbox.should_sandbox(cmd) {
        return None;
    }
    // A one-command grant rides on a throwaway copy of the policy, leaving the
    // session's own grant set clear.
    Some(if once.is_empty() {
        ctx.sandbox.clone()
    } else {
        crate::sandbox::Sandbox {
            granted: ctx.sandbox.granted.union(&once).copied().collect(),
            ..ctx.sandbox.clone()
        }
    })
}

/// Implements `bash_status` and (`stop = true`) `bash_stop`.
pub fn tool_bash_status_or_stop(ctx: &mut ToolContext, call: &ToolCall, stop: bool) -> String {
    let job_id = parse_int_default(call.arg_value("job"), 0, 0, i64::MAX);
    let pid = u32::try_from(parse_int_default(
        call.arg_value("pid"),
        0,
        0,
        i64::from(u32::MAX),
    ))
    .unwrap_or(0);
    ctx.bash.sweep();
    let Some(idx) = ctx.bash.find(job_id, pid) else {
        return format!("Tool error: bash job not found: job={job_id} pid={pid}\n");
    };
    // Mirrors the C: `bash_status` returns immediately unless a positive
    // refresh_sec asks it to wait; `bash_stop` always waits at least 1 s so
    // the observation reflects the termination.
    let mut refresh =
        u64::try_from(parse_int_default(call.arg_value("refresh_sec"), 0, 0, 3600)).unwrap_or(0);
    let wait = stop || refresh > 0;
    if stop && refresh == 0 {
        refresh = 1;
    }
    ctx.bash.job_tool_result(idx, wait, refresh, stop, true)
}

/// Outcome of an immediate (`!`-prefixed) shell command.
#[derive(Debug)]
pub struct ImmediateOutput {
    /// Captured standard output (lossy UTF-8).
    pub stdout: String,
    /// Captured standard error (lossy UTF-8).
    pub stderr: String,
    /// Exit code; `128 + signal` when signal-killed, like `BashJob`.
    pub exit_code: i64,
    /// True when the user interrupted the command before it finished.
    pub interrupted: bool,
    /// The line the command wrote into its GPU-yield signal file
    /// (`PLANK_GPU_YIELD_FILE`), when it wrote one. The file is gone by the
    /// time this is returned.
    pub gpu_signal: Option<String>,
}

/// Which stream a line of `!` output arrived on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    /// Standard output.
    Stdout,
    /// Standard error.
    Stderr,
}

/// Receives a running `!` command's output and drives the caller's redraw.
///
/// One trait rather than two closures because both halves need `&mut` access
/// to the same UI state — the output log — which two closures cannot share.
pub trait ImmediateSink {
    /// One complete line of output, newline already stripped.
    fn line(&mut self, stream: Stream, text: &str);
    /// Called on each poll tick; return `true` to interrupt the command.
    fn tick(&mut self) -> bool;
}

/// An [`ImmediateSink`] that only reports interrupts, discarding lines as they
/// stream (they are still accumulated into [`ImmediateOutput`]).
#[derive(Debug)]
pub struct InterruptOnly<F: FnMut() -> bool>(pub F);

impl<F: FnMut() -> bool> ImmediateSink for InterruptOnly<F> {
    fn line(&mut self, _stream: Stream, _text: &str) {}
    fn tick(&mut self) -> bool {
        (self.0)()
    }
}

/// Splits a byte stream into complete lines, holding any partial trailing line
/// until the newline arrives (or the stream ends).
#[derive(Default)]
struct LineSplitter {
    /// Everything seen so far, for `ImmediateOutput`.
    full: String,
    /// Bytes after the last newline, not yet a complete line.
    pending: String,
}

impl LineSplitter {
    /// Absorbs a chunk, handing every newly completed line to `emit`.
    fn push(&mut self, chunk: &[u8], mut emit: impl FnMut(&str)) {
        let text = String::from_utf8_lossy(chunk);
        self.full.push_str(&text);
        self.pending.push_str(&text);
        while let Some(nl) = self.pending.find('\n') {
            let line: String = self.pending.drain(..=nl).collect();
            emit(line.trim_end_matches(['\n', '\r']));
        }
    }

    /// Emits any trailing line that never got its newline.
    fn flush(&mut self, mut emit: impl FnMut(&str)) {
        if !self.pending.is_empty() {
            let line = std::mem::take(&mut self.pending);
            emit(&line);
        }
    }
}

/// Runs a user-typed `!` command to completion in `cwd`, separate from the
/// model's bash job table: stdout and stderr are captured independently and
/// `interrupt` is polled so Ctrl-C/Esc can kill a runaway command.
///
/// # Errors
/// Returns an error string when the shell fails to spawn.
///
/// # Panics
/// Panics only if the child's piped stdout/stderr handles are missing, which
/// cannot happen with `Stdio::piped`.
pub fn run_immediate(
    cwd: &std::path::Path,
    cmd: &str,
    sink: &mut dyn ImmediateSink,
) -> Result<ImmediateOutput, String> {
    use std::sync::mpsc::{Sender, channel};

    /// Reads a stream in chunks, forwarding each to the collector as it
    /// arrives. `read_to_end` would deliver everything only at exit, which is
    /// exactly what issue #22 was about.
    fn pump(
        stream: impl std::io::Read + Send + 'static,
        which: Stream,
        tx: Sender<(Stream, Vec<u8>)>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let mut stream = stream;
            let mut buf = [0u8; 8192];
            loop {
                match stream.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if tx.send((which, buf[..n].to_vec())).is_err() {
                            break;
                        }
                    }
                }
            }
        })
    }

    let mut command = Command::new("/bin/sh");
    command.arg("-c").arg(cmd).current_dir(cwd);
    // The same promise the bash tool's jobs get: a `!` or `!!` that writes
    // its signal file, or exits 75 with the marker line, gets the model
    // unloaded for a second run.
    let signal = export_gpu_yield(&mut command);
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Own group so an interrupt kills pipelines and backgrounded
        // grandchildren, not just the shell.
        .process_group(0)
        .spawn()
        .map_err(|e| format!("failed to start: {e}"))?;

    let (tx, rx) = channel::<(Stream, Vec<u8>)>();
    // The pipes are always present with Stdio::piped.
    let out_reader = pump(
        child.stdout.take().expect("piped stdout"),
        Stream::Stdout,
        tx.clone(),
    );
    let err_reader = pump(
        child.stderr.take().expect("piped stderr"),
        Stream::Stderr,
        tx,
    );

    let mut out = LineSplitter::default();
    let mut err = LineSplitter::default();
    let drain = |out: &mut LineSplitter, err: &mut LineSplitter, sink: &mut dyn ImmediateSink| {
        while let Ok((which, chunk)) = rx.try_recv() {
            match which {
                Stream::Stdout => out.push(&chunk, |l| sink.line(Stream::Stdout, l)),
                Stream::Stderr => err.push(&chunk, |l| sink.line(Stream::Stderr, l)),
            }
        }
    };

    let mut interrupted = false;
    let status = loop {
        drain(&mut out, &mut err, sink);
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {}
            Err(e) => {
                if let Some(path) = &signal {
                    let _ = std::fs::remove_file(path);
                }
                return Err(format!("wait failed: {e}"));
            }
        }
        if sink.tick() {
            interrupted = true;
            break kill_process_group(&mut child);
        }
        std::thread::sleep(Duration::from_millis(25));
    };

    // The readers end when their pipes close, which the child's exit
    // guarantees; joining first makes the final drain see every last byte.
    let _ = out_reader.join();
    let _ = err_reader.join();
    drain(&mut out, &mut err, sink);
    out.flush(|l| sink.line(Stream::Stdout, l));
    err.flush(|l| sink.line(Stream::Stderr, l));

    let exit_code = status.map_or(-1, status_code);
    Ok(ImmediateOutput {
        stdout: out.full,
        stderr: err.full,
        exit_code,
        interrupted,
        gpu_signal: signal.as_deref().and_then(crate::gpuyield::take_signal),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{home_writable, test_call, test_ctx};

    #[test]
    fn bash_echo_round_trip() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        let out = tool_bash(&mut ctx, &test_call("bash", &[("command", "echo hello")]));
        assert!(out.starts_with("bash job=1 pid="), "got: {out}");
        assert!(out.contains(" status=done "));
        assert!(out.contains("exit_status=0\n"));
        assert!(out.contains("<output>\nhello\n</output>\n"));
        assert!(ctx.bash.jobs.is_empty(), "finished job should be removed");
        std::fs::remove_dir_all(dir).ok();
    }

    fn parse_dsml(stanza: &str) -> ToolCall {
        let mut p = crate::dsml::DsmlParser::new();
        p.feed(stanza);
        assert!(p.error().is_empty(), "{}", p.error());
        p.calls().first().cloned().expect("one call")
    }

    fn dsml_bash(value: &str) -> String {
        format!(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n\
             <｜DSML｜parameter name=\"command\" string=\"true\">mex post.md</｜DSML｜parameter>\n\
             <｜DSML｜parameter name=\"suspend_model\" string=\"false\">{value}</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        )
    }

    #[test]
    fn the_dsml_parser_hands_suspend_model_through_on_bash() {
        for yes in ["true", "TRUE", " True ", "1", "yes", "Yes"] {
            let call = parse_dsml(&dsml_bash(yes));
            assert_eq!(call.arg_value("command"), Some("mex post.md"));
            assert!(suspend_model_requested(&call), "{yes:?}");
        }
        for no in ["false", "0", "no", "", "y", "truee"] {
            assert!(
                !suspend_model_requested(&parse_dsml(&dsml_bash(no))),
                "{no:?}"
            );
        }
        let plain = test_call("bash", &[("command", "mex post.md")]);
        assert!(!suspend_model_requested(&plain), "absent is false");
        let other = test_call("read", &[("path", "x"), ("suspend_model", "true")]);
        assert!(!suspend_model_requested(&other), "bash only");
    }

    #[test]
    fn the_qwen_parser_hands_suspend_model_through_on_bash() {
        let parse = |value: &str| {
            let mut p = trace_stream::qwen::QwenParser::new();
            p.feed(format!(
                "<tool_call>\n<function=bash>\n<parameter=command>\nmex post.md\n</parameter>\n\
                 <parameter=suspend_model>\n{value}\n</parameter>\n</function>\n</tool_call>"
            ));
            p.finish();
            assert!(p.error().is_none(), "{:?}", p.error());
            p.calls().first().cloned().expect("one call")
        };
        assert!(suspend_model_requested(&parse("true")));
        assert!(suspend_model_requested(&parse("YES")));
        assert!(!suspend_model_requested(&parse("false")));
        assert!(!suspend_model_requested(&parse("later")));
    }

    #[test]
    fn a_trailing_ampersand_backgrounds_a_command_and_a_double_one_does_not() {
        assert!(backgrounds_itself("mex post.md &"));
        assert!(backgrounds_itself("nohup mex post.md > log 2>&1 &  \n"));
        assert!(!backgrounds_itself("make && mex post.md"));
        assert!(!backgrounds_itself("mex post.md 2>&1 | tail"));
        assert!(!backgrounds_itself("true &&"));
    }

    #[test]
    fn a_bare_ampersand_anywhere_in_the_command_backgrounds_a_subcommand() {
        assert!(backgrounds_itself("(mex x &)"));
        assert!(backgrounds_itself("mex x & disown"));
        assert!(backgrounds_itself("mex x & echo started"));
        assert!(backgrounds_itself("mex x & sleep 1"));
    }

    #[test]
    fn two_char_operators_and_quoted_ampersands_are_not_backgrounding() {
        assert!(!backgrounds_itself("a && b"));
        assert!(!backgrounds_itself("x >&2"));
        assert!(!backgrounds_itself("x &> f"));
        assert!(!backgrounds_itself(r#"echo "a & b""#));
        assert!(!backgrounds_itself("echo 'a & b'"));
        assert!(!backgrounds_itself("x |& tail"));
        assert!(!backgrounds_itself(r"echo a \& b"));
    }

    #[test]
    fn a_bash_job_sees_plank_gpu_yield() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        let out = tool_bash(
            &mut ctx,
            &test_call("bash", &[("command", "echo \"yield=$PLANK_GPU_YIELD\"")]),
        );
        assert!(out.contains("<output>\nyield=1\n</output>\n"), "got: {out}");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_foreground_call_records_how_its_command_ended() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        let cmd = "echo 'GPU not available: held by plank (PID 1)' >&2; exit 75";
        tool_bash(&mut ctx, &test_call("bash", &[("command", cmd)]));
        let exit = ctx
            .bash
            .last_foreground
            .take()
            .expect("a finished call records");
        assert_eq!(exit.exit_status, 75);
        assert!(exit.needs_gpu, "the marker on stderr counts");

        tool_bash(&mut ctx, &test_call("bash", &[("command", "exit 75")]));
        let exit = ctx.bash.last_foreground.take().expect("recorded");
        assert!(
            !exit.needs_gpu,
            "75 without the marker is not a GPU request"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// Runs `cmd` as a foreground `bash` call after recording the signal
    /// path it was handed into `<dir>/path`; returns that path.
    fn foreground_with_path(ctx: &mut ToolContext, dir: &std::path::Path, cmd: &str) -> PathBuf {
        let rec = dir.join("path");
        let full = format!(
            "printf %s \"$PLANK_GPU_YIELD_FILE\" > '{}'; {cmd}",
            rec.display()
        );
        tool_bash(ctx, &test_call("bash", &[("command", &full)]));
        PathBuf::from(std::fs::read_to_string(&rec).expect("path recorded"))
    }

    #[test]
    fn each_bash_command_gets_its_own_signal_path_and_none_is_left_behind() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        let a = foreground_with_path(&mut ctx, &dir, "true");
        let b = foreground_with_path(&mut ctx, &dir, "true");
        assert!(a.is_absolute(), "{}", a.display());
        assert_ne!(a, b);
        assert!(a.starts_with(std::env::temp_dir()), "{}", a.display());
        assert!(!a.exists() && !b.exists());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_signal_file_asks_for_the_gpu_whatever_the_exit_status() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        let cmd = "echo 'GPU not available: held by plank' > \"$PLANK_GPU_YIELD_FILE\"; exit 0";
        let path = foreground_with_path(&mut ctx, &dir, cmd);
        let exit = ctx.bash.last_foreground.take().expect("recorded");
        assert_eq!(exit.exit_status, 0);
        assert!(exit.needs_gpu);
        assert_eq!(
            exit.signal.as_deref(),
            Some("GPU not available: held by plank")
        );
        assert!(!path.exists(), "read and deleted");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_signal_file_survives_a_pipe_that_replaces_the_exit_status() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        let cmd = "sh -c 'echo \"GPU not available: x\" > \"$PLANK_GPU_YIELD_FILE\"; exit 3' 2>&1 | tail -20";
        let path = foreground_with_path(&mut ctx, &dir, cmd);
        let exit = ctx.bash.last_foreground.take().expect("recorded");
        assert_eq!(exit.exit_status, 0, "tail's status");
        assert!(exit.needs_gpu);
        assert_eq!(exit.signal.as_deref(), Some("GPU not available: x"));
        assert!(!path.exists());
        std::fs::remove_dir_all(dir).ok();
    }

    /// Clears the process-wide interrupt flag on drop (panic included), so a
    /// failing assertion between raising it and clearing it cannot leave it
    /// set for every other test sharing the process.
    struct ClearInterruptOnDrop;
    impl Drop for ClearInterruptOnDrop {
        fn drop(&mut self) {
            crate::interrupt::clear();
        }
    }

    #[test]
    fn an_interrupt_during_the_run_is_never_recorded_as_needing_the_gpu() {
        // Held for the whole body, not just the run: the flag is live from
        // just before `request()` to the final `clear()`, and every other
        // bash-spawning test in this module takes the same guard, so none of
        // them can observe this test's interrupt (or vice versa).
        let _interrupt_guard = crate::interrupt::test_guard();
        crate::interrupt::clear();
        let _clear_on_drop = ClearInterruptOnDrop;
        let (mut ctx, dir) = test_ctx();
        // Writes the signal file immediately, then keeps running: without the
        // interrupt this would be an unmistakable GPU request.
        let cmd = "echo 'GPU not available: held' > \"$PLANK_GPU_YIELD_FILE\"; sleep 5; exit 75";
        std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(150));
            crate::interrupt::request();
        });
        let start = Instant::now();
        foreground_with_path(&mut ctx, &dir, cmd);
        assert!(
            start.elapsed() < Duration::from_secs(4),
            "the interrupt should have killed the job long before its own sleep finished"
        );
        let exit = ctx.bash.last_foreground.take().expect("recorded");
        assert!(
            !exit.needs_gpu,
            "an interrupted run must never start the GPU-yield cycle, \
             even though it wrote a signal file"
        );
        crate::interrupt::clear();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn no_signal_file_and_exit_0_is_no_gpu_request() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        // The marker without the file or exit 75 is still not a request.
        foreground_with_path(&mut ctx, &dir, "echo 'GPU not available: x'");
        let exit = ctx.bash.last_foreground.take().expect("recorded");
        assert!(!exit.needs_gpu);
        assert!(exit.signal.is_none());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_background_jobs_signal_file_goes_when_the_job_is_reaped() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        let cmd = "sleep 2; echo 'GPU not available: x' > \"$PLANK_GPU_YIELD_FILE\"";
        let rec = dir.join("path");
        let full = format!(
            "printf %s \"$PLANK_GPU_YIELD_FILE\" > '{}'; {cmd}",
            rec.display()
        );
        let out = tool_bash(
            &mut ctx,
            &test_call("bash", &[("command", &full), ("refresh_sec", "1")]),
        );
        assert!(out.contains("status=running"), "got: {out}");
        assert!(ctx.bash.last_foreground.is_none(), "never cycles");
        let path = PathBuf::from(std::fs::read_to_string(&rec).unwrap());
        let deadline = Instant::now() + Duration::from_secs(10);
        while !path.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(path.exists(), "the job wrote its signal");
        let out = tool_bash_status_or_stop(
            &mut ctx,
            &test_call("bash_status", &[("job", "1"), ("refresh_sec", "5")]),
            false,
        );
        assert!(out.contains("status=done"), "got: {out}");
        assert!(ctx.bash.jobs.is_empty(), "reaped");
        assert!(!path.exists(), "the reap deleted it");
        assert!(ctx.bash.last_foreground.is_none());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn the_gpu_marker_is_found_past_the_observations_head() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        // Far more lines than the head shows, marker last.
        let cmd = "seq 1 5000; echo 'GPU not available: busy'; exit 75";
        let out = tool_bash(&mut ctx, &test_call("bash", &[("command", cmd)]));
        assert!(
            !out.contains("GPU not available"),
            "the head hides it: {out}"
        );
        assert!(ctx.bash.last_foreground.take().expect("recorded").needs_gpu);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn wait_to_exit_outlasts_the_refresh_and_is_consumed() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        ctx.bash.wait_to_exit = true;
        let call = test_call(
            "bash",
            &[("command", "sleep 2; echo late"), ("refresh_sec", "1")],
        );
        let out = tool_bash(&mut ctx, &call);
        assert!(out.contains("status=done"), "got: {out}");
        assert!(out.contains("late"), "got: {out}");
        assert!(!ctx.bash.wait_to_exit, "one call only");
        assert!(
            ctx.bash.last_foreground.is_some(),
            "finished inside the call"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_background_job_records_nothing_even_when_it_later_asks_for_the_gpu() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        let cmd = "sleep 2; echo 'GPU not available: busy'; exit 75";
        let out = tool_bash(
            &mut ctx,
            &test_call("bash", &[("command", cmd), ("refresh_sec", "1")]),
        );
        assert!(out.contains("status=running"), "got: {out}");
        assert!(ctx.bash.last_foreground.is_none());
        // Observing it finish later is `bash_status`, which never records.
        let out = tool_bash_status_or_stop(
            &mut ctx,
            &test_call("bash_status", &[("job", "1"), ("refresh_sec", "5")]),
            false,
        );
        assert!(out.contains("exit_status=75"), "got: {out}");
        assert!(ctx.bash.last_foreground.is_none());
        std::fs::remove_dir_all(dir).ok();
    }

    /// A job that exits after the model last saw it running is returned once
    /// by `take_finished`, with its final observation, and then forgotten.
    #[test]
    fn take_finished_announces_a_background_exit_once() {
        let (mut ctx, dir) = test_ctx();
        // `refresh_sec` is clamped to >= 1 s, so start the job directly to
        // leave it running when the table is first inspected.
        let id = ctx
            .bash
            .start(&ctx.cwd.clone(), "sleep 0.3; echo late", 30, None)
            .unwrap();
        assert!(ctx.bash.take_finished().is_empty(), "still running");
        assert_eq!(ctx.bash.running_count(), 1);
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut got = Vec::new();
        while got.is_empty() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
            got = ctx.bash.take_finished();
        }
        assert_eq!(got.len(), 1, "exactly one announcement");
        let obs = &got[0];
        assert!(
            obs.starts_with(&format!("bash job={id} pid=")),
            "got: {obs}"
        );
        assert!(obs.contains(" status=done "));
        assert!(obs.contains("exit_status=0\n"));
        assert!(obs.contains("late\n"));
        assert!(ctx.bash.take_finished().is_empty(), "never announced twice");
        assert!(ctx.bash.jobs.is_empty());
        std::fs::remove_dir_all(dir).ok();
    }

    /// A job the model polled to completion itself is removed by that poll,
    /// so it is never announced.
    #[test]
    fn take_finished_skips_jobs_the_model_observed_done() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        let out = tool_bash(&mut ctx, &test_call("bash", &[("command", "echo now")]));
        assert!(out.contains(" status=done "));
        assert_eq!(ctx.bash.take_finished(), [] as [std::string::String; 0]);
        std::fs::remove_dir_all(dir).ok();
    }

    /// A job killed by its own timeout is announced with `timed_out=1`.
    #[test]
    fn take_finished_announces_timeouts() {
        let (mut ctx, dir) = test_ctx();
        let id = ctx
            .bash
            .start(&ctx.cwd.clone(), "sleep 30", 1, None)
            .unwrap();
        std::thread::sleep(Duration::from_millis(1100));
        let got = ctx.bash.take_finished();
        assert_eq!(got.len(), 1);
        assert!(got[0].starts_with(&format!("bash job={id} pid=")));
        assert!(got[0].contains("timed_out=1\n"), "got: {}", got[0]);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn render_rows_lists_each_job_state() {
        assert_eq!(render_rows(&[]), "no background jobs");
        let now = Instant::now();
        let rows = vec![
            JobRow {
                id: 1,
                pid: 10,
                started: now,
                state: JobState::Running,
                path: PathBuf::from("/tmp/a"),
            },
            JobRow {
                id: 2,
                pid: 11,
                started: now,
                state: JobState::Done(0),
                path: PathBuf::from("/tmp/b"),
            },
            JobRow {
                id: 3,
                pid: 12,
                started: now,
                state: JobState::TimedOut(143),
                path: PathBuf::from("/tmp/c"),
            },
        ];
        let text = render_rows(&rows);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with("job 1 pid 10 ") && lines[0].contains("running  /tmp/a"));
        assert!(lines[1].contains("done, exit 0  /tmp/b"));
        assert!(lines[2].contains("timed out, exit 143  /tmp/c"));
    }

    #[test]
    fn render_notification_wraps_observations() {
        assert!(render_notification(&[]).is_none());
        let one = render_notification(&["bash job=3 pid=1 status=done\n".to_string()]).unwrap();
        assert!(
            one.starts_with("<system-reminder>\n[BACKGROUND JOB NOTIFICATION - NOT USER INPUT]\n")
        );
        assert!(one.contains("A bash job you started earlier has finished."));
        assert!(one.contains("\nbash job=3 pid=1 status=done\n"));
        assert!(one.ends_with("</system-reminder>"));
        let two = render_notification(&["a".to_string(), "b".to_string()]).unwrap();
        assert!(two.contains("Bash jobs you started earlier have finished."));
        assert!(two.contains("\na\n\nb\n"));
    }

    #[test]
    fn bash_nonzero_exit_and_stderr_capture() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        let out = tool_bash(
            &mut ctx,
            &test_call("bash", &[("command", "echo oops >&2; exit 3")]),
        );
        assert!(out.contains("exit_status=3\n"));
        assert!(out.contains("oops\n"));
        std::fs::remove_dir_all(dir).ok();
    }

    /// End-to-end Seatbelt check: with the sandbox on, a write inside cwd
    /// succeeds while a write outside it is denied with EPERM and the
    /// observation carries the `[sandbox blocked: ...]` hint. Requires
    /// /usr/bin/sandbox-exec, so macOS only.
    #[cfg(target_os = "macos")]
    #[test]
    fn bash_sandbox_blocks_writes_outside_cwd() {
        let _interrupt_guard = crate::interrupt::test_guard();
        // The escape target lives under `$HOME` (outside cwd and temp), and
        // `sandbox-exec` itself can't apply a profile from inside a nested
        // sandbox — so skip both when `$HOME` isn't writable.
        if !home_writable() {
            return;
        }
        let (mut ctx, dir) = test_ctx();
        ctx.sandbox.enabled = true;

        let ok = tool_bash(
            &mut ctx,
            &test_call("bash", &[("command", "echo inside > inside.txt")]),
        );
        assert!(ok.contains("exit_status=0\n"), "cwd write failed: {ok}");

        // The scratch dir lives under temp_dir(), which the profile always
        // allows — the escape target must sit outside both cwd and temp, so
        // use a scratch dir under $HOME.
        let home = std::env::var("HOME").expect("HOME set");
        let outside =
            std::path::Path::new(&home).join(format!(".plank-sandbox-test-{}", std::process::id()));
        std::fs::create_dir_all(&outside).unwrap();
        let cmd = format!("echo escape > '{}/escape.txt'", outside.display());
        let blocked = tool_bash(&mut ctx, &test_call("bash", &[("command", &cmd)]));
        assert!(
            !blocked.contains("exit_status=0\n"),
            "outside write should fail: {blocked}"
        );
        assert!(
            blocked.contains("[sandbox blocked:"),
            "missing violation hint: {blocked}"
        );
        assert!(!outside.join("escape.txt").exists());

        // A toolchain cache under the same $HOME is writable without any
        // grant: this is the `cargo build fetches a crate` path, and the one
        // the escape check above must not have closed off.
        let cache = std::path::Path::new(&home).join(".cache/plank-sandbox-test");
        std::fs::create_dir_all(&cache).unwrap();
        let cmd_cache = format!("echo cached > '{}/probe.txt'", cache.display());
        let cached = tool_bash(&mut ctx, &test_call("bash", &[("command", &cmd_cache)]));
        assert!(
            cached.contains("exit_status=0\n"),
            "toolchain cache write should be allowed: {cached}"
        );
        std::fs::remove_dir_all(&cache).ok();

        // ...while a PATH directory under the same $HOME stays denied, since
        // no asker is installed in this test to grant it.
        let bin = std::path::Path::new(&home).join(".cargo/bin");
        if bin.is_dir() {
            let cmd_bin = format!(
                "echo nope > '{}/plank-sandbox-test-{}'",
                bin.display(),
                std::process::id()
            );
            let denied = tool_bash(&mut ctx, &test_call("bash", &[("command", &cmd_bin)]));
            assert!(
                !denied.contains("exit_status=0\n"),
                "PATH bin write should be denied without a grant: {denied}"
            );
        }

        // Excluded commands bypass the sandbox entirely.
        ctx.sandbox.excluded_commands.push("echo *".to_string());
        let bypass = tool_bash(&mut ctx, &test_call("bash", &[("command", &cmd)]));
        assert!(
            bypass.contains("exit_status=0\n"),
            "excluded command should bypass sandbox: {bypass}"
        );
        std::fs::remove_dir_all(outside).ok();
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn bash_missing_command_errors() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        assert_eq!(
            tool_bash(&mut ctx, &test_call("bash", &[])),
            "Tool error: bash requires command\n"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn async_job_spawn_poll_and_stop() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        // refresh_sec=1 returns while the job is still running.
        let out = tool_bash(
            &mut ctx,
            &test_call(
                "bash",
                &[("command", "echo started; sleep 30"), ("refresh_sec", "1")],
            ),
        );
        assert!(out.contains("status=running"), "got: {out}");
        assert!(out.contains("Use bash_status job=1"));
        assert_eq!(ctx.bash.jobs.len(), 1);

        let out =
            tool_bash_status_or_stop(&mut ctx, &test_call("bash_status", &[("job", "1")]), false);
        assert!(out.contains("status=running"));
        // Second observation is tail-biased.
        assert!(out.contains("<tail -4 "), "got: {out}");

        let out = tool_bash_status_or_stop(
            &mut ctx,
            &test_call("bash_stop", &[("job", "1"), ("refresh_sec", "1")]),
            true,
        );
        assert!(out.contains("status=done"), "got: {out}");
        assert!(ctx.bash.jobs.is_empty());

        let out =
            tool_bash_status_or_stop(&mut ctx, &test_call("bash_status", &[("job", "1")]), false);
        assert_eq!(out, "Tool error: bash job not found: job=1 pid=0\n");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn bash_status_waits_for_refresh_sec() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        let out = tool_bash(
            &mut ctx,
            &test_call(
                "bash",
                &[("command", "sleep 2; echo finished"), ("refresh_sec", "1")],
            ),
        );
        assert!(out.contains("status=running"), "got: {out}");

        // A positive refresh_sec waits, so the job finishes within the poll.
        let started = std::time::Instant::now();
        let out = tool_bash_status_or_stop(
            &mut ctx,
            &test_call("bash_status", &[("job", "1"), ("refresh_sec", "5")]),
            false,
        );
        assert!(out.contains("status=done"), "got: {out}");
        assert!(out.contains("finished"), "got: {out}");
        assert!(started.elapsed() < std::time::Duration::from_secs(4));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn bash_status_without_refresh_returns_immediately() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        tool_bash(
            &mut ctx,
            &test_call("bash", &[("command", "sleep 30"), ("refresh_sec", "1")]),
        );
        let started = std::time::Instant::now();
        let out =
            tool_bash_status_or_stop(&mut ctx, &test_call("bash_status", &[("job", "1")]), false);
        assert!(out.contains("status=running"), "got: {out}");
        assert!(started.elapsed() < std::time::Duration::from_millis(500));
        tool_bash_status_or_stop(&mut ctx, &test_call("bash_stop", &[("job", "1")]), true);
        assert!(ctx.bash.jobs.is_empty());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn immediate_command_captures_streams_and_exit() {
        let out = run_immediate(
            std::path::Path::new("/tmp"),
            "echo out; echo err >&2; exit 3",
            &mut InterruptOnly(|| false),
        )
        .unwrap();
        assert_eq!(out.stdout, "out\n");
        assert_eq!(out.stderr, "err\n");
        assert_eq!(out.exit_code, 3);
        assert!(!out.interrupted);
    }

    #[test]
    fn an_immediate_command_sees_the_gpu_yield_variable() {
        let out = run_immediate(
            std::path::Path::new("/tmp"),
            "echo \"yield=$PLANK_GPU_YIELD\"",
            &mut InterruptOnly(|| false),
        )
        .unwrap();
        assert_eq!(out.stdout, "yield=1\n");
    }

    #[test]
    fn an_immediate_command_gets_a_fresh_signal_path_and_its_signal_back() {
        let run = |cmd: &str| {
            run_immediate(
                std::path::Path::new("/tmp"),
                cmd,
                &mut InterruptOnly(|| false),
            )
            .unwrap()
        };
        let a = run("printf %s \"$PLANK_GPU_YIELD_FILE\"");
        let b = run("printf %s \"$PLANK_GPU_YIELD_FILE\"");
        assert_ne!(a.stdout, "");
        assert_ne!(a.stdout, b.stdout);
        assert!(a.gpu_signal.is_none());
        let asked = run("printf %s \"$PLANK_GPU_YIELD_FILE\"; \
             echo 'GPU not available: held' > \"$PLANK_GPU_YIELD_FILE\"");
        assert_eq!(asked.exit_code, 0);
        assert_eq!(asked.gpu_signal.as_deref(), Some("GPU not available: held"));
        assert!(!std::path::Path::new(&asked.stdout).exists(), "deleted");
    }

    /// Records each line with the moment it arrived, for the streaming tests.
    struct Recorder {
        lines: Vec<(Stream, String, Instant)>,
        ticks: usize,
    }

    impl ImmediateSink for Recorder {
        fn line(&mut self, stream: Stream, text: &str) {
            self.lines.push((stream, text.to_owned(), Instant::now()));
        }
        fn tick(&mut self) -> bool {
            self.ticks += 1;
            false
        }
    }

    #[test]
    fn immediate_output_streams_before_the_command_exits() {
        // The regression #22 fixed: `read_to_end` delivered everything only at
        // exit. The first line must arrive well before the process ends.
        let mut rec = Recorder {
            lines: Vec::new(),
            ticks: 0,
        };
        let start = Instant::now();
        let out = run_immediate(
            std::path::Path::new("/tmp"),
            "echo first; sleep 1; echo second",
            &mut rec,
        )
        .unwrap();
        let total = start.elapsed();

        assert_eq!(rec.lines.len(), 2, "{:?}", rec.lines);
        assert_eq!(rec.lines[0].1, "first");
        assert_eq!(rec.lines[1].1, "second");
        let first_at = rec.lines[0].2.duration_since(start);
        assert!(
            first_at < total / 2,
            "first line arrived at {first_at:?} of {total:?} - not streaming"
        );
        assert_eq!(out.stdout, "first\nsecond\n", "accumulation still works");
    }

    #[test]
    fn immediate_output_separates_the_two_streams() {
        let mut rec = Recorder {
            lines: Vec::new(),
            ticks: 0,
        };
        run_immediate(
            std::path::Path::new("/tmp"),
            "echo out; echo err 1>&2",
            &mut rec,
        )
        .unwrap();
        let by_stream: Vec<(Stream, &str)> =
            rec.lines.iter().map(|(s, t, _)| (*s, t.as_str())).collect();
        assert!(
            by_stream.contains(&(Stream::Stdout, "out")),
            "{by_stream:?}"
        );
        assert!(
            by_stream.contains(&(Stream::Stderr, "err")),
            "{by_stream:?}"
        );
    }

    #[test]
    fn a_trailing_line_without_a_newline_is_still_emitted() {
        let mut rec = Recorder {
            lines: Vec::new(),
            ticks: 0,
        };
        let out = run_immediate(
            std::path::Path::new("/tmp"),
            "printf 'no-newline'",
            &mut rec,
        )
        .unwrap();
        assert_eq!(rec.lines.len(), 1, "{:?}", rec.lines);
        assert_eq!(rec.lines[0].1, "no-newline");
        assert_eq!(out.stdout, "no-newline");
    }

    #[test]
    fn crlf_is_stripped_from_streamed_lines() {
        let mut rec = Recorder {
            lines: Vec::new(),
            ticks: 0,
        };
        run_immediate(
            std::path::Path::new("/tmp"),
            "printf 'a\\r\\nb\\n'",
            &mut rec,
        )
        .unwrap();
        assert_eq!(rec.lines[0].1, "a");
        assert_eq!(rec.lines[1].1, "b");
    }

    #[test]
    fn immediate_command_interrupt_kills() {
        let mut polls = 0;
        let start = Instant::now();
        let out = run_immediate(
            std::path::Path::new("/tmp"),
            "sleep 30",
            &mut InterruptOnly(|| {
                polls += 1;
                polls > 2
            }),
        )
        .unwrap();
        assert!(out.interrupted);
        assert!(start.elapsed() < Duration::from_secs(5));
    }

    /// Extracts the first all-digit line from an observation: the pid a test
    /// command echoed with `$!`.
    fn echoed_pid(obs: &str) -> libc::pid_t {
        obs.lines()
            .find_map(|l| l.trim().parse::<libc::pid_t>().ok())
            .unwrap_or_else(|| panic!("no pid line in: {obs}"))
    }

    /// Waits up to two seconds for `pid` to disappear (kill(pid, 0) fails).
    fn wait_gone(pid: libc::pid_t) -> bool {
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if unsafe { libc::kill(pid, 0) } != 0 {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        false
    }

    #[test]
    fn stopping_a_job_kills_its_grandchildren() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        let out = tool_bash(
            &mut ctx,
            &test_call(
                "bash",
                &[
                    ("command", "sleep 30 & echo $!; wait"),
                    ("refresh_sec", "1"),
                ],
            ),
        );
        assert!(out.contains("status=running"), "got: {out}");
        let grandchild = echoed_pid(&out);
        assert_eq!(unsafe { libc::kill(grandchild, 0) }, 0, "sleep not running");

        let out = tool_bash_status_or_stop(
            &mut ctx,
            &test_call("bash_stop", &[("job", "1"), ("refresh_sec", "1")]),
            true,
        );
        assert!(out.contains("status=done"), "got: {out}");
        assert!(
            wait_gone(grandchild),
            "sleep {grandchild} outlived bash_stop"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn an_unpolled_timed_out_job_is_swept_by_the_next_bash_call() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        let out = tool_bash(
            &mut ctx,
            &test_call(
                "bash",
                &[
                    ("command", "sleep 30 & echo $!; wait"),
                    ("timeout_sec", "2"),
                    ("refresh_sec", "1"),
                ],
            ),
        );
        assert!(out.contains("status=running"), "got: {out}");
        let grandchild = echoed_pid(&out);
        std::thread::sleep(Duration::from_millis(1500));
        // Never poll job 1 again; an unrelated bash call must reap it.
        let other = tool_bash(&mut ctx, &test_call("bash", &[("command", "echo other")]));
        assert!(other.contains("status=done"), "got: {other}");
        let stale = &ctx.bash.jobs[0];
        assert_eq!(stale.id, 1);
        assert!(!stale.running, "timed-out job still marked running");
        assert!(stale.timed_out);
        assert!(
            wait_gone(grandchild),
            "sleep {grandchild} outlived the timeout"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn interrupting_an_immediate_command_kills_its_grandchildren() {
        // Give the shell a few ticks to echo the pid before interrupting.
        let mut polls = 0;
        let out = run_immediate(
            std::path::Path::new("/tmp"),
            "sleep 30 & echo $!; wait",
            &mut InterruptOnly(|| {
                polls += 1;
                polls > 8
            }),
        )
        .unwrap();
        assert!(out.interrupted);
        let grandchild = echoed_pid(&out.stdout);
        assert!(
            wait_gone(grandchild),
            "sleep {grandchild} outlived the interrupt"
        );
    }

    #[test]
    fn bash_timeout_kills_job() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        let out = tool_bash(
            &mut ctx,
            &test_call(
                "bash",
                &[
                    ("command", "sleep 30"),
                    ("timeout_sec", "1"),
                    ("refresh_sec", "3"),
                ],
            ),
        );
        assert!(out.contains("timed_out=1"), "got: {out}");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn bash_runs_in_context_cwd() {
        let _interrupt_guard = crate::interrupt::test_guard();
        let (mut ctx, dir) = test_ctx();
        std::fs::write(dir.join("marker.txt"), "x").unwrap();
        let out = tool_bash(
            &mut ctx,
            &test_call("bash", &[("command", "ls marker.txt")]),
        );
        assert!(out.contains("marker.txt\n"), "got: {out}");
        std::fs::remove_dir_all(dir).ok();
    }
}
