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
}
