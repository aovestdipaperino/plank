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
