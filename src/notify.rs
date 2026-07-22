// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Native macOS desktop notifications fired at turn lifecycle points.
//!
//! Best-effort and non-blocking: a failed or slow notification must never
//! affect a turn. Delivery is `osascript display notification` (silent, no
//! sound); off macOS `notify` is a no-op. The enable flag is a module-level
//! atomic seeded from `settings.ui.notifications` at startup and flipped by
//! the `/notify` command (session-only; the persisted default lives in
//! settings).

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

static ENABLED: AtomicBool = AtomicBool::new(true);

/// Enable or disable notifications for the rest of the session.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// Whether notifications are currently enabled.
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// True when a completed turn lasting `elapsed` should notify given the
/// configured `after_secs` threshold.
#[must_use]
pub fn should_notify_complete(elapsed: Duration, after_secs: u64) -> bool {
    elapsed.as_secs() >= after_secs
}

/// Escape a string for inclusion inside an `AppleScript` double-quoted literal:
/// backslash and double-quote are backslash-escaped; CR/LF become spaces so
/// the body stays a single line and cannot terminate or inject the script.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn applescript_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_was_cr = false;
    for c in s.chars() {
        match c {
            '\\' => {
                out.push_str(r"\\");
                prev_was_cr = false;
            }
            '"' => {
                out.push_str(r#"\""#);
                prev_was_cr = false;
            }
            '\n' => {
                if !prev_was_cr {
                    out.push(' ');
                }
                prev_was_cr = false;
            }
            '\r' => {
                out.push(' ');
                prev_was_cr = true;
            }
            _ => {
                out.push(c);
                prev_was_cr = false;
            }
        }
    }
    out
}

/// Fire a desktop notification. No-op when disabled or off macOS.
pub fn notify(title: &str, body: &str) {
    if !enabled() {
        return;
    }
    #[cfg(target_os = "macos")]
    {
        let script = format!(
            "display notification \"{}\" with title \"{}\"",
            applescript_escape(body),
            applescript_escape(title),
        );
        // Spawn and detach: never wait, so a slow osascript cannot stall the
        // caller. Ignore spawn errors — notifications are best-effort.
        let _ = std::process::Command::new("osascript")
            .arg("-e")
            .arg(script)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (title, body);
    }
}

/// Serializes tests that touch the process-global `ENABLED` flag, so
/// `cargo test --lib` (which runs tests concurrently) cannot interleave
/// `set_enabled` calls between this module's test and `ui`'s
/// `notify_command_toggles_and_reports`. Lives at module level (not inside
/// `mod tests`) so both take the *same* guard — two separate mutexes would
/// not serialize anything.
#[cfg(test)]
pub(crate) static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn escapes_quotes_and_backslashes() {
        assert_eq!(applescript_escape(r#"a"b\c"#), r#"a\"b\\c"#);
    }

    #[test]
    fn strips_newlines_so_body_stays_one_line() {
        assert_eq!(applescript_escape("line1\nline2\r\n"), "line1 line2 ");
    }

    #[test]
    fn complete_gate_respects_threshold() {
        assert!(!should_notify_complete(Duration::from_secs(9), 10));
        assert!(should_notify_complete(Duration::from_secs(10), 10));
        assert!(should_notify_complete(Duration::from_secs(30), 10));
    }

    #[test]
    fn enable_flag_round_trips() {
        let _g = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        set_enabled(false);
        assert!(!enabled());
        set_enabled(true);
        assert!(enabled());
    }
}
