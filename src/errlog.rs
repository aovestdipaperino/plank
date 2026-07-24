// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Persistent error log under `~/.plank/errors.log`.
//!
//! Tool failures shown to the model (and to the user) are terse one-liners
//! like `Tool error: visit_page failed: ...`; the full detail — which
//! subsystem failed, for which URL, with the complete error text — is
//! appended here so it can be inspected after the fact. Writing is
//! best-effort: logging must never turn a recoverable tool error into a
//! crash, so every failure to log is silently ignored.

use std::io::Write as _;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Log file name under `~/.plank`.
const ERROR_LOG_FILE: &str = "errors.log";

/// Path to the error log, or `None` when `$HOME` is unset.
#[must_use]
pub fn error_log_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").filter(|h| !h.is_empty())?;
    Some(PathBuf::from(home).join(".plank").join(ERROR_LOG_FILE))
}

/// Appends one timestamped entry to the error log.
///
/// `source` names the failing subsystem (e.g. `visit_page`, `obscura`);
/// `detail` is the full error text, which may span multiple lines —
/// continuation lines are indented so entries stay visually grouped.
/// Best-effort: all I/O errors are swallowed.
pub fn log_error(source: &str, detail: &str) {
    let Some(path) = error_log_path() else {
        return;
    };
    if let Some(dir) = path.parent()
        && std::fs::create_dir_all(dir).is_err()
    {
        return;
    }
    let Ok(mut f) = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(&path)
    else {
        return;
    };
    let ts = timestamp();
    let mut entry = format!("[{ts}] {source}: ");
    let mut lines = detail.lines();
    entry.push_str(lines.next().unwrap_or(""));
    entry.push('\n');
    for line in lines {
        entry.push_str("    ");
        entry.push_str(line);
        entry.push('\n');
    }
    let _ = f.write_all(entry.as_bytes());
}

/// Local-time timestamp `YYYY-MM-DD HH:MM:SS.mmm` (same shape as trace.rs).
fn timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let millis = now.subsec_millis();
    let secs = now.as_secs();
    // SAFETY: localtime_r with valid pointers; tm is fully initialized on
    // success and zeroed on failure.
    let (y, mo, d, h, mi, s) = unsafe {
        let t = libc::time_t::try_from(secs).unwrap_or(0);
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&raw const t, &raw mut tm);
        (
            tm.tm_year + 1900,
            tm.tm_mon + 1,
            tm.tm_mday,
            tm.tm_hour,
            tm.tm_min,
            tm.tm_sec,
        )
    };
    format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02}.{millis:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_timestamped_multiline_entries() {
        let tmp = std::env::temp_dir().join(format!("plank-errlog-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        // SAFETY: single-threaded test; restored before returning.
        let prev = std::env::var_os("HOME");
        unsafe { std::env::set_var("HOME", &tmp) };

        log_error("visit_page", "failed to load https://x: boom");
        log_error("obscura", "line one\nline two");
        let text = std::fs::read_to_string(error_log_path().unwrap()).unwrap();
        assert!(text.contains("] visit_page: failed to load https://x: boom\n"));
        assert!(text.contains("] obscura: line one\n    line two\n"));
        // Two entries, each starting with a `[YYYY-` timestamp.
        assert_eq!(text.lines().filter(|l| l.starts_with('[')).count(), 2);

        match prev {
            Some(v) => unsafe { std::env::set_var("HOME", v) },
            None => unsafe { std::env::remove_var("HOME") },
        }
        std::fs::remove_dir_all(&tmp).ok();
    }
}
