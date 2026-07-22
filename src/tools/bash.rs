// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Asynchronous bash jobs: spawn, poll output, and stop shell commands.
//!
//! Port of the "Asynchronous Bash Jobs" section of `ds4_agent.c`. Bash
//! commands are tracked jobs, not blocking one-shot calls. Each job owns a
//! process, reader threads, and a temp output file. The first observation is
//! head-biased so headers and early errors are visible; later observations
//! are tail-biased.

use std::fmt::Write as _;
use std::io::Read as _;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::dsml::ToolCall;

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
}

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
        self.exit_status = status.code().map_or_else(
            || {
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    status.signal().map_or(-1, |sig| 128 + i64::from(sig))
                }
                #[cfg(not(unix))]
                {
                    -1
                }
            },
            i64::from,
        );
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
            let _ = self.child.kill();
            if let Ok(status) = self.child.wait() {
                self.finalize(status);
            } else {
                self.exit_status = -1;
                self.running = false;
            }
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
                 the path to writablePaths in .plank/sandbox.json.]"
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
    fn refresh_for(&mut self, refresh_sec: u64) {
        let deadline = Instant::now() + Duration::from_secs(refresh_sec);
        while self.running && Instant::now() < deadline {
            self.poll();
            if !self.running {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        self.poll();
    }
}

impl Drop for BashJob {
    fn drop(&mut self) {
        if self.running {
            let _ = self.child.kill();
            let _ = self.child.wait();
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
        let mut child = command
            .arg("-c")
            .arg(cmd)
            .current_dir(ctx_cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
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
        if stop && job.running {
            let _ = job.child.kill();
            let deadline = Instant::now() + Duration::from_secs(1);
            while job.running && Instant::now() < deadline {
                job.poll();
                if !job.running {
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
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
    let sandbox = ctx.sandbox.should_sandbox(cmd).then_some(&ctx.sandbox);
    if let Err(err) = ctx.bash.start(&ctx.cwd.clone(), cmd, timeout, sandbox) {
        return format!("Tool error: bash failed to start: {err}\n");
    }
    let idx = ctx.bash.jobs.len() - 1;
    ctx.bash.job_tool_result(idx, true, refresh, false, true)
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
    let Some(idx) = ctx.bash.find(job_id, pid) else {
        return format!("Tool error: bash job not found: job={job_id} pid={pid}\n");
    };
    let refresh = u64::try_from(parse_int_default(
        call.arg_value("refresh_sec"),
        60,
        1,
        3600,
    ))
    .unwrap_or(60);
    ctx.bash.job_tool_result(idx, stop, refresh, stop, true)
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

    let mut child = Command::new("/bin/sh")
        .arg("-c")
        .arg(cmd)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
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
            Err(e) => return Err(format!("wait failed: {e}")),
        }
        if sink.tick() {
            interrupted = true;
            let _ = child.kill();
            break child.wait().ok();
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

    let exit_code = status.map_or(-1, |s| {
        s.code().map_or_else(
            || {
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    s.signal().map_or(-1, |sig| 128 + i64::from(sig))
                }
                #[cfg(not(unix))]
                {
                    -1
                }
            },
            i64::from,
        )
    });
    Ok(ImmediateOutput {
        stdout: out.full,
        stderr: err.full,
        exit_code,
        interrupted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{test_call, test_ctx};

    #[test]
    fn bash_echo_round_trip() {
        let (mut ctx, dir) = test_ctx();
        let out = tool_bash(&mut ctx, &test_call("bash", &[("command", "echo hello")]));
        assert!(out.starts_with("bash job=1 pid="), "got: {out}");
        assert!(out.contains(" status=done "));
        assert!(out.contains("exit_status=0\n"));
        assert!(out.contains("<output>\nhello\n</output>\n"));
        assert!(ctx.bash.jobs.is_empty(), "finished job should be removed");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn bash_nonzero_exit_and_stderr_capture() {
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
        let (mut ctx, dir) = test_ctx();
        assert_eq!(
            tool_bash(&mut ctx, &test_call("bash", &[])),
            "Tool error: bash requires command\n"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn async_job_spawn_poll_and_stop() {
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

    #[test]
    fn bash_timeout_kills_job() {
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
