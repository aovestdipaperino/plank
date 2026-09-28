// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! In-place stderr log rendering for the noisy C-engine load phase.
//!
//! The ds4 C library prints its startup diagnostics ("ds4: ...") directly to
//! stderr, one line each. While a [`StderrLineReplacer`] guard is alive,
//! stderr is redirected into a pipe and a reader thread repaints each line in
//! place on the real terminal (carriage return + clear), so the load phase
//! occupies a single screen row instead of scrolling. Dropping the guard
//! restores stderr and clears the row.

use std::io::Read;
use std::os::fd::{FromRawFd, RawFd};
use std::path::Path;

/// Guard that renders stderr lines in place until dropped.
#[derive(Debug)]
pub struct StderrLineReplacer {
    saved: RawFd,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl StderrLineReplacer {
    /// Starts replacing stderr lines; returns `None` when stderr is not a
    /// terminal (logs then flow through untouched).
    #[must_use]
    pub fn start() -> Option<Self> {
        // SAFETY: isatty/dup/pipe/dup2 on process-owned fds.
        unsafe {
            if libc::isatty(libc::STDERR_FILENO) == 0 {
                return None;
            }
            let saved = libc::dup(libc::STDERR_FILENO);
            if saved < 0 {
                return None;
            }
            let mut fds = [0_i32; 2];
            if libc::pipe(fds.as_mut_ptr()) != 0 {
                libc::close(saved);
                return None;
            }
            if libc::dup2(fds[1], libc::STDERR_FILENO) < 0 {
                libc::close(saved);
                libc::close(fds[0]);
                libc::close(fds[1]);
                return None;
            }
            libc::close(fds[1]);
            let reader = std::fs::File::from_raw_fd(fds[0]);
            let thread = std::thread::spawn(move || render_lines(reader, saved));
            Some(Self {
                saved,
                thread: Some(thread),
            })
        }
    }
}

impl Drop for StderrLineReplacer {
    fn drop(&mut self) {
        // SAFETY: restoring the saved stderr fd; this closes the pipe's only
        // write end (fd 2), so the reader thread sees EOF and exits.
        unsafe {
            libc::dup2(self.saved, libc::STDERR_FILENO);
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        // SAFETY: the reader thread has exited; nothing else uses `saved`.
        unsafe {
            libc::close(self.saved);
        }
    }
}

/// Writes `bytes` to `fd`, ignoring errors (best-effort terminal paint).
fn write_all(fd: RawFd, bytes: &[u8]) {
    // SAFETY: fd is the saved terminal fd, valid while the thread runs.
    let _ = unsafe { libc::write(fd, bytes.as_ptr().cast(), bytes.len()) };
}

/// Terminal column count for `fd`, defaulting to 80.
fn term_cols(fd: RawFd) -> usize {
    // SAFETY: winsize is plain-old-data; ioctl fills it on success.
    let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
    // SAFETY: fd valid; ws is a writable winsize.
    let rc = unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, &raw mut ws) };
    if rc == 0 && ws.ws_col > 0 {
        ws.ws_col as usize
    } else {
        80
    }
}

/// Repaints the current line in place, truncated to the terminal width so a
/// wrapped line cannot leave residue on the row above when replaced.
fn repaint(fd: RawFd, line: &[u8]) {
    let cols = term_cols(fd).saturating_sub(1).max(1);
    let text = String::from_utf8_lossy(line);
    let shown: String = text.chars().take(cols).collect();
    write_all(fd, b"\r\x1b[K");
    write_all(fd, shown.as_bytes());
}

/// Reads the redirected stderr and paints each (partial) line in place.
fn render_lines(mut reader: std::fs::File, out: RawFd) {
    let mut line: Vec<u8> = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        let n = match reader.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        for &b in &chunk[..n] {
            if b == b'\n' {
                repaint(out, &line);
                line.clear();
            } else {
                line.push(b);
            }
        }
        // Show partial lines too, so "requesting residency... done" style
        // messages that arrive in two writes stay live.
        if !line.is_empty() {
            repaint(out, &line);
        }
    }
    write_all(out, b"\r\x1b[K");
}

/// Serializes [`discarding`] and [`logging_to`]. Its own lock, not
/// [`CAPTURE_LOCK`]: a session created inside `f` takes that one through
/// [`without_ds4_chatter`], and the same thread locking it twice would
/// deadlock.
static DISCARD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Runs `f` with fd 2 pointed at `/dev/null`.
///
/// For the C engine's teardown in the middle of a session (the GPU-yield
/// cycle): its log would land on whatever the front end is drawing, the
/// TUI's alternate screen included. fd 2 comes back even when `f` panics.
pub fn discarding<T>(f: impl FnOnce() -> T) -> T {
    // The phase is never recorded: with no log, `redirected` never arms the
    // exit hook, so this value is not read.
    redirected(None, Phase::Release, f)
}

/// Which side of a GPU-yield cycle a [`logging_to`] redirect covers, printed
/// in the exit report so a crash during teardown is not blamed on "reloading".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Tearing the old engine down (`ds4_engine_close`).
    Release,
    /// Bringing the engine back up after the yield.
    Reload,
}

impl Phase {
    fn verb(self) -> &'static str {
        match self {
            Phase::Release => "releasing",
            Phase::Reload => "reloading",
        }
    }
}

/// Runs `f` with fd 2 appended to the file at `log`, falling back to
/// `/dev/null` when it cannot be opened.
///
/// For the C engine's reload in the middle of a session. The C `model_open`
/// calls `exit` on a model it cannot map and on a contended instance lock, so
/// while `f` runs an `atexit` hook stands ready: should the process end
/// inside `f`, it hands fd 2 back, resets the terminal (a TUI would otherwise
/// be left on the alternate screen in raw mode) and prints the last line of
/// the log, so the user sees why plank stopped. Failures that return normally
/// still reach the user through `f`'s own result.
pub fn logging_to<T>(log: &Path, phase: Phase, f: impl FnOnce() -> T) -> T {
    redirected(Some(log), phase, f)
}

/// Opens where [`redirected`] points fd 2: `log` for appending, else (or when
/// that fails) `/dev/null`. Returns the fd and whether it is the log.
fn open_sink(log: Option<&Path>) -> (RawFd, bool) {
    use std::os::unix::ffi::OsStrExt as _;
    if let Some(path) = log
        && let Ok(c) = std::ffi::CString::new(path.as_os_str().as_bytes())
    {
        // SAFETY: `c` is a valid NUL-terminated path.
        let fd = unsafe {
            libc::open(
                c.as_ptr(),
                libc::O_WRONLY | libc::O_CREAT | libc::O_APPEND,
                0o600,
            )
        };
        if fd >= 0 {
            return (fd, true);
        }
    }
    // SAFETY: a constant NUL-terminated path.
    (
        unsafe { libc::open(c"/dev/null".as_ptr(), libc::O_WRONLY) },
        false,
    )
}

/// Puts the real fd 2 back when dropped, panics included.
struct Fd2Restore {
    saved: RawFd,
}

impl Drop for Fd2Restore {
    fn drop(&mut self) {
        // Disarmed first: from here on an exit is not inside the reload.
        EXIT_SAVED_FD.store(-1, std::sync::atomic::Ordering::SeqCst);
        // SAFETY: `saved` is our dup of the real fd 2. Flush C stdio first so
        // a buffered line cannot slip out after the fd comes back.
        unsafe {
            libc::fflush(std::ptr::null_mut());
            libc::dup2(self.saved, libc::STDERR_FILENO);
            libc::close(self.saved);
        }
    }
}

/// The real fd 2 while a [`logging_to`] redirect is live, `-1` otherwise.
/// Read by [`report_exit_inside_redirect`].
static EXIT_SAVED_FD: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

/// The log a live [`logging_to`] redirect writes to; `None` both before a
/// redirect and when the log could not be opened (rule 5: the hook still
/// arms and reports, just without a last-line quote).
static EXIT_LOG: std::sync::Mutex<Option<std::path::PathBuf>> = std::sync::Mutex::new(None);

/// The [`Phase`] of the live [`logging_to`] redirect, encoded as `0` for
/// [`Phase::Release`] and `1` for [`Phase::Reload`].
static EXIT_PHASE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The session id to name in `/resume`, or `None` when nothing has been
/// saved. Set by [`set_exit_resume`], read (via `try_lock`, never blocking)
/// by [`report_exit_inside_redirect`].
static EXIT_RESUME: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Whether the TUI currently owns the terminal (alternate screen, raw mode).
/// Set by the TUI front end on entry and cleared on exit; the exit hook only
/// writes the terminal-reset sequence while this is `true`, so a plain-REPL
/// crash does not paint alt-screen/mouse-mode escapes into a scrolling shell
/// that never turned them on.
static TUI_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Marks whether the TUI currently owns the terminal. See [`TUI_ACTIVE`].
pub fn set_tui_active(active: bool) {
    TUI_ACTIVE.store(active, std::sync::atomic::Ordering::SeqCst);
}

/// Disarms the exit-report hook: a `process::exit` right after this call is
/// a deliberate plank exit, not the C engine dying mid-redirect, so no
/// misleading "ended the process while reloading" report should print.
pub fn disarm_exit_report() {
    EXIT_SAVED_FD.store(-1, std::sync::atomic::Ordering::SeqCst);
}

/// Records what `/resume` should say: the id of the session saved before a
/// GPU-yield unload, or `None` when nothing was saved (rule 2). Uses a
/// blocking lock, poison recovered, because this always runs outside the
/// exit hook; the hook itself only ever `try_lock`s.
pub fn set_exit_resume(id: Option<String>) {
    let mut slot = EXIT_RESUME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *slot = id;
}

fn redirected<T>(log: Option<&Path>, phase: Phase, f: impl FnOnce() -> T) -> T {
    // A panic inside an earlier `f` poisons the lock, but the drop guard put
    // fd 2 back and the lock guards no data, so it is safe to go on using.
    let _lock = DISCARD_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (sink, is_log) = open_sink(log);
    // SAFETY: dup/dup2/close on process-owned fds; every fd opened here is
    // closed on every path (`saved` by the guard).
    let saved = unsafe {
        let saved = libc::dup(libc::STDERR_FILENO);
        if saved < 0 || sink < 0 || libc::dup2(sink, libc::STDERR_FILENO) < 0 {
            if saved >= 0 {
                libc::close(saved);
            }
            if sink >= 0 {
                libc::close(sink);
            }
            None
        } else {
            libc::close(sink);
            Some(saved)
        }
    };
    let Some(saved) = saved else {
        return f();
    };
    let _restore = Fd2Restore { saved };
    // Arm whenever the caller asked to log at all (`log.is_some()`), even if
    // the file itself could not be opened: an exit is exactly as real either
    // way, only the last-line quote is unavailable (rule 5).
    if log.is_some() {
        if let Ok(mut slot) = EXIT_LOG.lock() {
            *slot = if is_log {
                log.map(Path::to_path_buf)
            } else {
                None
            };
        }
        EXIT_PHASE.store(
            u8::from(matches!(phase, Phase::Reload)),
            std::sync::atomic::Ordering::SeqCst,
        );
        arm_exit_report(saved);
    }
    f()
}

/// Registers [`report_exit_inside_redirect`] once and arms it for `saved`.
fn arm_exit_report(saved: RawFd) {
    static REGISTERED: std::sync::Once = std::sync::Once::new();
    REGISTERED.call_once(|| {
        // SAFETY: registering a plain `extern "C"` function with no captures.
        unsafe {
            libc::atexit(report_exit_inside_redirect);
        }
    });
    EXIT_SAVED_FD.store(saved, std::sync::atomic::Ordering::SeqCst);
}

/// Terminal modes the TUI turns on, turned off: keyboard enhancement flags,
/// focus events, bracketed paste, mouse capture, the alternate screen, and
/// a hidden cursor. Each is harmless on a terminal that never had it on.
const TERMINAL_RESET: &str = "\x1b[<1u\x1b[?1004l\x1b[?2004l\x1b[?1006l\x1b[?1015l\
                              \x1b[?1003l\x1b[?1002l\x1b[?1000l\x1b[?1049l\x1b[?25h";

/// What the exit hook prints: the phase, the last non-empty line of `log`
/// when there is one, and the `/resume` hint when `resume` names a saved
/// session.
#[must_use]
fn exit_report(log: Option<&Path>, phase: Phase, resume: Option<&str>) -> String {
    use std::fmt::Write as _;
    let mut out = format!(
        "\r\nplank: the model engine ended the process while {} the model after a GPU yield",
        phase.verb()
    );
    match log {
        Some(log) => {
            let text = std::fs::read_to_string(log).unwrap_or_default();
            let last = text.lines().rev().find(|l| !l.trim().is_empty());
            if let Some(line) = last {
                out.push_str(": ");
                out.push_str(line.trim());
            }
            let _ = write!(out, "\r\nplank: engine log: {}", log.display());
        }
        None => out.push('.'),
    }
    match resume {
        Some(id) => {
            let _ = write!(
                out,
                "\r\nplank: run plank again and /resume {id} to restore the session saved \
                 before the unload\r\n"
            );
        }
        None => out.push_str(
            "\r\nplank: the session could not be saved before the unload; /resume is not \
             available\r\n",
        ),
    }
    out
}

/// The `atexit` hook behind [`logging_to`]. A no-op unless the process is
/// exiting while a redirect is live, which only the C engine does.
extern "C" fn report_exit_inside_redirect() {
    let saved = EXIT_SAVED_FD.swap(-1, std::sync::atomic::Ordering::SeqCst);
    if saved < 0 {
        return;
    }
    let log = EXIT_LOG.try_lock().ok().and_then(|g| g.clone());
    let resume = EXIT_RESUME.try_lock().ok().and_then(|g| g.clone());
    let phase = if EXIT_PHASE.load(std::sync::atomic::Ordering::SeqCst) == 1 {
        Phase::Reload
    } else {
        Phase::Release
    };
    let tui_active = TUI_ACTIVE.load(std::sync::atomic::Ordering::SeqCst);
    // SAFETY: plain writes and termios calls on process-owned fds at exit.
    unsafe {
        libc::fflush(std::ptr::null_mut());
        libc::dup2(saved, libc::STDERR_FILENO);
        if tui_active && libc::isatty(libc::STDOUT_FILENO) != 0 {
            write_all(libc::STDOUT_FILENO, TERMINAL_RESET.as_bytes());
        }
    }
    if tui_active {
        let _ = ratatui::crossterm::terminal::disable_raw_mode();
    }
    write_all(
        libc::STDERR_FILENO,
        exit_report(log.as_deref(), phase, resume.as_deref()).as_bytes(),
    );
}

/// Known ds4 chatter: lines the C library prints on every session creation
/// that say nothing a user of plank can act on.
///
/// Matched by prefix against a whole line. Kept deliberately short — anything
/// not listed here is passed through, because a diagnostic swallowed is worse
/// than a diagnostic repeated.
const DS4_CHATTER: &[&str] = &["ds4: DSpark target-hidden capture enabled:"];

/// Serializes the fd-2 swap in [`without_ds4_chatter`], so two threads
/// creating sessions at once cannot restore each other's stderr.
static CAPTURE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Runs `f` with stderr captured, then re-emits everything it printed except
/// the [`DS4_CHATTER`] lines.
///
/// The C engine announces its `DSpark` capture configuration on stderr every
/// time a session is created — at startup, on `/clear`, for every aside and
/// sub-agent — and that lands in the middle of the user's screen. The line is
/// a build detail, not news, so plank drops it here while the upstream print
/// is still unconditional. Nothing else is dropped: whatever else `f` wrote,
/// including the failure messages that matter, is written straight back out.
pub fn without_ds4_chatter<T>(f: impl FnOnce() -> T) -> T {
    let Ok(_guard) = CAPTURE_LOCK.lock() else {
        // A poisoned lock means some other capture panicked mid-swap; leaving
        // stderr alone is the safe response, chatter and all.
        return f();
    };
    // SAFETY: dup/pipe/dup2 on process-owned fds; every fd opened here is
    // closed on both the success and the early-return paths.
    let saved_and_pipe = unsafe {
        let saved = libc::dup(libc::STDERR_FILENO);
        if saved < 0 {
            None
        } else {
            let mut fds = [0_i32; 2];
            if libc::pipe(fds.as_mut_ptr()) != 0 || libc::dup2(fds[1], libc::STDERR_FILENO) < 0 {
                libc::close(saved);
                None
            } else {
                libc::close(fds[1]);
                Some((saved, fds[0]))
            }
        }
    };
    let Some((saved, read_fd)) = saved_and_pipe else {
        return f();
    };
    // Drained on a thread: `f` writing more than the pipe buffer holds would
    // otherwise block forever on a pipe nobody is reading yet.
    let reader = std::thread::spawn(move || {
        // SAFETY: read_fd is owned here and closed by the File's drop.
        let mut file = unsafe { std::fs::File::from_raw_fd(read_fd) };
        let mut buf = Vec::new();
        let _ = file.read_to_end(&mut buf);
        buf
    });
    let out = f();
    // SAFETY: restoring the real stderr closes the pipe's last write end, so
    // the reader above sees EOF.
    unsafe {
        libc::dup2(saved, libc::STDERR_FILENO);
        libc::close(saved);
    }
    if let Ok(buf) = reader.join() {
        let text = String::from_utf8_lossy(&buf);
        for line in text.lines() {
            if is_ds4_chatter(line) {
                continue;
            }
            eprintln!("{line}");
        }
    }
    out
}

/// Whether a captured stderr line is chatter plank drops. See [`DS4_CHATTER`].
fn is_ds4_chatter(line: &str) -> bool {
    DS4_CHATTER.iter().any(|p| line.starts_with(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_listed_chatter_is_dropped() {
        assert!(is_ds4_chatter(
            "ds4: DSpark target-hidden capture enabled: layers=40,41,42"
        ));
        // A failure from the same code path reads almost the same and must
        // still reach the user.
        assert!(!is_ds4_chatter(
            "ds4: failed to configure DSpark target-hidden capture"
        ));
        assert!(!is_ds4_chatter("ds4: out of memory"));
        assert!(!is_ds4_chatter(""));
    }

    #[test]
    fn the_capture_returns_the_value_and_leaves_stderr_usable() {
        let out = without_ds4_chatter(|| {
            eprintln!("ds4: DSpark target-hidden capture enabled: layers=1");
            41 + 1
        });
        assert_eq!(out, 42);
        // Restored: the harness's own stderr still works afterwards, which is
        // the failure this would otherwise cause everywhere at once.
        eprint!("");
    }

    #[test]
    fn the_exit_report_names_the_last_log_line_and_the_log() {
        let dir = std::env::temp_dir().join(format!("plank-stderrline-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("gpu-yield.log");
        std::fs::write(&log, "ds4: loading\nds4: failed to mmap model.gguf\n\n").unwrap();
        let report = exit_report(Some(&log), Phase::Reload, Some("brave-hopper"));
        assert!(report.contains("failed to mmap model.gguf"), "{report}");
        assert!(report.contains(&log.display().to_string()), "{report}");
        assert!(report.contains("reloading"), "{report}");
        assert!(report.contains("/resume brave-hopper"), "{report}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_exit_report_names_the_phase() {
        let report = exit_report(None, Phase::Release, Some("brave-hopper"));
        assert!(report.contains("releasing"), "{report}");
        assert!(!report.contains("reloading"), "{report}");
    }

    #[test]
    fn the_exit_report_says_so_when_nothing_was_saved() {
        let report = exit_report(None, Phase::Reload, None);
        assert!(
            report.contains("could not be saved"),
            "{report}: expected the unsaved wording"
        );
        assert!(!report.contains("run plank again and /resume"), "{report}");
    }

    #[test]
    fn the_exit_report_has_no_log_line_when_there_is_no_log() {
        let report = exit_report(None, Phase::Reload, Some("brave-hopper"));
        assert!(!report.contains("engine log:"), "{report}");
        assert!(report.contains("/resume brave-hopper"), "{report}");
    }

    #[test]
    fn a_redirect_writes_to_its_log_and_restores_fd_2_after_a_panic() {
        let dir = std::env::temp_dir().join(format!("plank-stderrline-p-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("gpu-yield.log");
        let caught = std::panic::catch_unwind(|| {
            logging_to(&log, Phase::Reload, || {
                // Straight to fd 2, the way the C engine writes.
                write_all(libc::STDERR_FILENO, b"ds4: reload line\n");
                panic!("inside the redirect");
            })
        });
        assert!(caught.is_err());
        assert!(
            std::fs::read_to_string(&log)
                .unwrap()
                .contains("ds4: reload line")
        );
        {
            // Under the lock: another test's redirect arms the hook too.
            let _held = DISCARD_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert_eq!(
                EXIT_SAVED_FD.load(std::sync::atomic::Ordering::SeqCst),
                -1,
                "the exit hook is disarmed"
            );
        }
        // fd 2 is the real one again: a write lands nowhere near the log.
        write_all(libc::STDERR_FILENO, b"");
        let after = std::fs::read_to_string(&log).unwrap();
        discarding(|| write_all(libc::STDERR_FILENO, b"ds4: discarded\n"));
        assert_eq!(std::fs::read_to_string(&log).unwrap(), after);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn disarm_exit_report_clears_the_armed_fd() {
        let dir = std::env::temp_dir().join(format!("plank-stderrline-d-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("gpu-yield.log");
        logging_to(&log, Phase::Release, || {
            assert_ne!(
                EXIT_SAVED_FD.load(std::sync::atomic::Ordering::SeqCst),
                -1,
                "armed while the redirect is live"
            );
            disarm_exit_report();
            assert_eq!(
                EXIT_SAVED_FD.load(std::sync::atomic::Ordering::SeqCst),
                -1,
                "disarmed by a deliberate exit such as force_quit"
            );
        });
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_log_that_fails_to_open_still_arms_the_hook() {
        // A directory can't be opened O_WRONLY|O_CREAT, so `open_sink` falls
        // back to `/dev/null`; the hook must arm anyway (rule 5).
        let dir =
            std::env::temp_dir().join(format!("plank-stderrline-noopen-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        logging_to(&dir, Phase::Release, || {
            assert_ne!(
                EXIT_SAVED_FD.load(std::sync::atomic::Ordering::SeqCst),
                -1,
                "armed even though the log could not be opened"
            );
            let recorded = EXIT_LOG
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            assert_eq!(recorded, None, "nothing to quote a last line from");
        });
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn set_exit_resume_round_trips() {
        set_exit_resume(Some("calm-otter".to_owned()));
        assert_eq!(
            EXIT_RESUME
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone(),
            Some("calm-otter".to_owned())
        );
        set_exit_resume(None);
        assert_eq!(
            EXIT_RESUME
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone(),
            None
        );
    }

    #[test]
    fn tui_active_flag_round_trips() {
        // Gates the reset sequence in `report_exit_inside_redirect`: only
        // checkable here as a plain get/set, since the write itself only
        // happens from the real `atexit` hook.
        set_tui_active(true);
        assert!(TUI_ACTIVE.load(std::sync::atomic::Ordering::SeqCst));
        set_tui_active(false);
        assert!(!TUI_ACTIVE.load(std::sync::atomic::Ordering::SeqCst));
    }
}
