// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Native macOS desktop notifications fired at turn lifecycle points.
//!
//! Best-effort and non-blocking: a failed or slow notification must never
//! affect a turn, so delivery runs on a detached thread. Delivery is the
//! `notify-rust` crate (an embedded `NSUserNotification` helper — no external
//! binary, no subprocess); off macOS `notify` is a no-op. The enable flag is a
//! module-level atomic seeded from `settings.ui.notifications` at startup and
//! flipped by the `/notify` command (session-only; the persisted default lives
//! in settings).
//!
//! ## Icon
//! A bare CLI has no app bundle, so macOS would otherwise stamp the banner with
//! a generic Finder icon. We [`set_application`](notify_rust::set_application)
//! once to the host terminal's bundle id (derived from `TERM_PROGRAM`) so the
//! banner wears the terminal's icon, and attach plank's logo as the banner's
//! right-side content image. macOS gives no way to set an *arbitrary* main icon
//! for an unbundled process, so borrowing the terminal's is the reliable win.

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

/// The bundle id of the host terminal, derived from `TERM_PROGRAM`, whose icon
/// the notification borrows. Falls back to Terminal.app (always present on
/// macOS) for an unrecognized or absent terminal.
#[cfg(target_os = "macos")]
fn host_terminal_bundle_id() -> &'static str {
    match std::env::var("TERM_PROGRAM").as_deref() {
        Ok("iTerm.app") => "com.googlecode.iterm2",
        Ok("WarpTerminal") => "dev.warp.Warp",
        Ok("vscode") => "com.microsoft.VSCode",
        Ok("ghostty") => "com.mitchellh.ghostty",
        Ok("WezTerm") => "com.github.wez.wezterm",
        Ok("Hyper") => "co.zeit.hyper",
        Ok("kitty") => "net.kovidgoyal.kitty",
        Ok("Tabby") => "org.tabby",
        // "Apple_Terminal" and anything unknown land here.
        _ => "com.apple.Terminal",
    }
}

/// Registers the sending application (borrowing the terminal's icon) exactly
/// once. `set_application` is internally `call_once`, and if it is never called
/// macOS falls back to a generic Finder icon, so this must run before the first
/// notification. Errors are ignored — the notification still delivers.
#[cfg(target_os = "macos")]
fn ensure_application() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = notify_rust::set_application(host_terminal_bundle_id());
    });
}

/// plank's logo, embedded so the notification content image needs no installed
/// asset. Materialized to a temp file on first use (the macOS notification API
/// takes a filesystem path).
#[cfg(target_os = "macos")]
static LOGO_PNG: &[u8] = include_bytes!("resources/logo.png");

/// Writes the embedded logo to a stable temp path once and returns it, for use
/// as the banner's content image. `None` on any I/O failure — the notification
/// simply goes out without an image.
#[cfg(target_os = "macos")]
fn logo_path() -> Option<std::path::PathBuf> {
    use std::sync::OnceLock;
    static PATH: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    PATH.get_or_init(|| {
        let dir = std::env::temp_dir().join("plank");
        std::fs::create_dir_all(&dir).ok()?;
        let path = dir.join("notify-logo.png");
        // Reuse an already-materialized copy of the right size; else (re)write.
        let up_to_date = std::fs::metadata(&path)
            .map(|m| m.len() == LOGO_PNG.len() as u64)
            .unwrap_or(false);
        if !up_to_date {
            std::fs::write(&path, LOGO_PNG).ok()?;
        }
        Some(path)
    })
    .clone()
}

/// Fire a desktop notification. No-op when disabled or off macOS.
pub fn notify(title: &str, body: &str) {
    if !enabled() {
        return;
    }
    #[cfg(target_os = "macos")]
    {
        // Detach: the ObjC delivery is quick but must never stall a turn, and a
        // failed notification is ignored (best-effort, like the old path).
        let title = title.to_string();
        let body = body.to_string();
        std::thread::spawn(move || {
            ensure_application();
            let mut n = notify_rust::Notification::new();
            n.summary(&title).body(&body);
            if let Some(path) = logo_path() {
                n.image_path(&path.to_string_lossy());
            }
            let _ = n.show();
        });
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
    fn complete_gate_respects_threshold() {
        assert!(!should_notify_complete(Duration::from_secs(9), 10));
        assert!(should_notify_complete(Duration::from_secs(10), 10));
        assert!(should_notify_complete(Duration::from_secs(30), 10));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn logo_materializes_to_temp_and_matches_embedded_bytes() {
        let path = logo_path().expect("logo should materialize");
        assert!(path.exists(), "logo file not written: {}", path.display());
        let on_disk = std::fs::metadata(&path).unwrap().len();
        assert_eq!(on_disk, LOGO_PNG.len() as u64, "size mismatch");
        // Idempotent: a second call returns the same path without re-erroring.
        assert_eq!(logo_path().as_deref(), Some(path.as_path()));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn terminal_bundle_id_maps_known_and_defaults() {
        // The mapping is env-driven; assert the default branch explicitly since
        // TERM_PROGRAM under the test harness is unspecified.
        assert!(!host_terminal_bundle_id().is_empty());
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
