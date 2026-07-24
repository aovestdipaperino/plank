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

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::Duration;

/// When desktop notifications fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NotifyMode {
    /// On every qualifying turn event.
    #[default]
    Always,
    /// Only while the terminal window is not focused — the case where a
    /// banner is informative rather than redundant.
    Unfocused,
    /// Never.
    Never,
}

impl NotifyMode {
    /// The settings-file spelling of the mode.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            NotifyMode::Always => "always",
            NotifyMode::Unfocused => "unfocused",
            NotifyMode::Never => "never",
        }
    }

    /// Parses a settings value; `true`/`false` are accepted as the pre-mode
    /// boolean spellings (always/never).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "always" | "true" | "on" => Some(NotifyMode::Always),
            "unfocused" => Some(NotifyMode::Unfocused),
            "never" | "false" | "off" => Some(NotifyMode::Never),
            _ => None,
        }
    }
}

static MODE: AtomicU8 = AtomicU8::new(0);

/// Whether the terminal window is focused. Only the TUI receives focus
/// events; it seeds this true at startup and tracks changes. The plain REPL
/// never sets it, so it stays false there and `Unfocused` behaves like
/// `Always` — better a redundant banner than a silently missing one.
static FOCUSED: AtomicBool = AtomicBool::new(false);

/// Sets the notification mode for the rest of the session.
pub fn set_mode(mode: NotifyMode) {
    MODE.store(mode as u8, Ordering::Relaxed);
}

/// The current notification mode.
#[must_use]
pub fn mode() -> NotifyMode {
    match MODE.load(Ordering::Relaxed) {
        1 => NotifyMode::Unfocused,
        2 => NotifyMode::Never,
        _ => NotifyMode::Always,
    }
}

/// Enable (`Always`) or disable (`Never`) notifications for the rest of the
/// session — the `/notify` toggle.
pub fn set_enabled(on: bool) {
    set_mode(if on {
        NotifyMode::Always
    } else {
        NotifyMode::Never
    });
}

/// Whether notifications are currently enabled at all (any mode but `Never`).
#[must_use]
pub fn enabled() -> bool {
    mode() != NotifyMode::Never
}

/// Records whether the terminal window is focused (TUI focus events).
pub fn set_focused(focused: bool) {
    FOCUSED.store(focused, Ordering::Relaxed);
}

/// Whether a notification should be delivered right now under the current
/// mode and focus state.
fn should_deliver() -> bool {
    match mode() {
        NotifyMode::Always => true,
        NotifyMode::Never => false,
        NotifyMode::Unfocused => !FOCUSED.load(Ordering::Relaxed),
    }
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
    notify_full(title, None, body);
}

/// Fire a desktop notification with an optional subtitle. No-op when disabled
/// or off macOS.
///
/// macOS renders the `summary` in bold and the `subtitle` in regular weight
/// beneath it, so callers wanting a bold line (e.g. the user's prompt) pass it
/// as `summary`. The subtitle is ignored by notification servers that lack the
/// concept.
pub fn notify_full(summary: &str, subtitle: Option<&str>, body: &str) {
    deliver(summary, subtitle, body, false);
}

/// Like [`notify_full`], but the banner stays on screen until the user
/// dismisses it instead of auto-fading after a few seconds.
///
/// macOS gives no public control over banner duration — it is a per-app style
/// choice in System Settings. The escape hatch (the same one terminal-notifier
/// uses) is that a notification carrying dropdown actions makes the helper set
/// the private `_showsButtons` key, which forces the persistent *alert* style
/// regardless of the borrowed app's setting. The actions themselves do nothing;
/// they exist only to trigger that style.
///
/// # Platform caveat (macOS 13+)
///
/// Apple deprecated `NSUserNotificationCenter` and stopped honoring its
/// private keys (including `_showsButtons`) on macOS 13 (Ventura) and later.
/// On those versions the override is ignored and the banner renders in
/// whatever style the *borrowed* app (the host terminal — see
/// [`host_terminal_bundle_id`]) is configured for in System Settings, which
/// ships as the auto-fading **Banner** style by default. The sticky banner
/// therefore still auto-fades after ~5s on current macOS despite the actions
/// being attached.
///
/// The only supported fix is to migrate to `UNUserNotificationCenter` (notify-rust's
/// `preview-macos-un` feature / the `mac-usernotifications` crate), but that API
/// keys everything off `NSBundle::mainBundle().bundleIdentifier()` and refuses to
/// deliver (`check_bundle` returns `Err`) when the binary is **not** a bundled,
/// code-signed `.app`. Plank ships as a bare Homebrew binary with no bundle, so
/// the UN path silently drops every notification — strictly worse than the
/// current auto-fade. Adopting it would require redistributing plank as a signed
/// `.app` bundle (a packaging change, not a `notify.rs` change).
///
/// **User-side workaround**: in System Settings → Notifications → [Terminal /
/// iTerm / Warp / …] set the alert style from "Banners" to **"Alerts"**. Alerts
/// persist until dismissed, restoring sticky behavior without a code change.
pub fn notify_sticky(summary: &str, subtitle: Option<&str>, body: &str) {
    deliver(summary, subtitle, body, true);
}

fn deliver(summary: &str, subtitle: Option<&str>, body: &str, sticky: bool) {
    if !should_deliver() {
        return;
    }
    // Remember the last delivered notification so `/renotify` can re-show it
    // (e.g. to screenshot the banner). Recorded only when we actually deliver,
    // so a suppressed (disabled / unfocused) call does not clobber a prior one.
    record_last(summary, subtitle, body);
    deliver_raw(summary, subtitle, body, sticky);
}

/// Platform delivery with no mode/focus gating. Used by [`deliver`] (after the
/// gate + recording) and by [`renotify`] (an explicit user request, always on).
fn deliver_raw(summary: &str, subtitle: Option<&str>, body: &str, sticky: bool) {
    #[cfg(target_os = "macos")]
    {
        // Detach: the ObjC delivery is quick but must never stall a turn, and a
        // failed notification is ignored (best-effort, like the old path).
        let summary = summary.to_string();
        let subtitle = subtitle.map(str::to_string);
        let body = body.to_string();
        std::thread::spawn(move || {
            ensure_application();
            let mut n = notify_rust::Notification::new();
            n.summary(&summary).body(&body);
            if let Some(sub) = &subtitle {
                n.subtitle(sub);
            }
            if sticky {
                // Two actions → a dropdown → `_showsButtons` → alert style.
                // One action alone only sets `hasActionButton`, which does not
                // flip the style.
                n.action("dismiss", "Dismiss");
                n.action("close", "Close");
            }
            if let Some(path) = logo_path() {
                n.image_path(&path.to_string_lossy());
            }
            let _ = n.show();
        });
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (summary, subtitle, body, sticky);
    }
}

/// The last notification plank actually delivered, remembered so [`renotify`]
/// can re-show it on demand (used by the non-advertised `/renotify` command to
/// get a fresh banner on screen for screenshotting).
#[derive(Clone)]
struct LastNotify {
    summary: String,
    subtitle: Option<String>,
    body: String,
}

static LAST: std::sync::Mutex<Option<LastNotify>> = std::sync::Mutex::new(None);

fn record_last(summary: &str, subtitle: Option<&str>, body: &str) {
    *LAST.lock().unwrap() = Some(LastNotify {
        summary: summary.to_string(),
        subtitle: subtitle.map(str::to_string),
        body: body.to_string(),
    });
}

/// Re-shows the last delivered notification as a sticky banner, for
/// screenshotting. Returns `false` if no notification has fired yet in this
/// session. Bypasses [`should_deliver`] — the user asked for it explicitly, so
/// focus/mode gating does not apply. Always re-delivers as sticky so the
/// banner stays on screen long enough to capture.
///
/// Not advertised in `/help`; invoked by the hidden `/renotify` slash command.
#[must_use]
pub fn renotify() -> bool {
    // Recover from a poisoned mutex (only possible after a prior panic in a
    // test that held the lock) rather than propagating the panic into a turn.
    let guard = LAST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(last) = guard.clone() else {
        return false;
    };
    drop(guard);
    deliver_raw(&last.summary, last.subtitle.as_deref(), &last.body, true);
    true
}

/// Condenses a user prompt into a single-line notification headline: newlines
/// and runs of whitespace collapse to single spaces, and the result is
/// truncated with an ellipsis past [`PROMPT_SUMMARY_MAX`] characters. Empty
/// (or whitespace-only) prompts yield `"plank"` so the banner is never blank.
#[must_use]
pub fn prompt_summary(prompt: &str) -> String {
    const PROMPT_SUMMARY_MAX: usize = 80;
    let collapsed = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return "plank".to_string();
    }
    match collapsed.char_indices().nth(PROMPT_SUMMARY_MAX) {
        Some((i, _)) => format!("{}…", &collapsed[..i]),
        None => collapsed,
    }
}

/// Formats the "task finished" notification headline: the prompt (collapsed to
/// one line) wrapped in single quotes and truncated with `...` past
/// [`TITLE_PROMPT_MAX`] characters, followed by the outcome verb (`finished`,
/// or `interrupted` for a user-aborted turn) — e.g.
/// `'add .DS_Store to .gitignor...' finished`. Empty prompts fall back to
/// `"plank"` via [`prompt_summary`]'s convention.
#[must_use]
pub fn finished_title(prompt: &str, interrupted: bool) -> String {
    const TITLE_PROMPT_MAX: usize = 26;
    let verb = if interrupted {
        "interrupted"
    } else {
        "finished"
    };
    let collapsed = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return format!("plank {verb}");
    }
    match collapsed.char_indices().nth(TITLE_PROMPT_MAX) {
        Some((i, _)) => format!("'{}...' {verb}", collapsed[..i].trim_end()),
        None => format!("'{collapsed}' {verb}"),
    }
}

/// Formats the "task finished" notification body: `Latest output: ` followed
/// by the *tail* of the assistant's final output (the end is where the
/// conclusion lives), prefixed with `...` when the head was cut. Newlines and
/// whitespace runs collapse to single spaces. Empty output yields a plain
/// `Task complete` (or `Task interrupted`).
#[must_use]
pub fn latest_output_body(output: &str, interrupted: bool) -> String {
    const BODY_TAIL_MAX: usize = 160;
    let collapsed = output.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return if interrupted {
            "Task interrupted".to_string()
        } else {
            "Task complete".to_string()
        };
    }
    let total = collapsed.chars().count();
    if total <= BODY_TAIL_MAX {
        return format!("Latest output: {collapsed}");
    }
    let start = collapsed
        .char_indices()
        .nth(total - BODY_TAIL_MAX)
        .map_or(0, |(i, _)| i);
    format!("Latest output: ...{}", &collapsed[start..])
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
    fn prompt_summary_collapses_and_truncates() {
        assert_eq!(prompt_summary("  fix   the\n bug  "), "fix the bug");
        assert_eq!(prompt_summary("   "), "plank");
        assert_eq!(prompt_summary(""), "plank");
        let long = "a".repeat(100);
        let out = prompt_summary(&long);
        assert!(out.ends_with('…'));
        assert_eq!(out.chars().count(), 81); // 80 chars + ellipsis
    }

    #[test]
    fn finished_title_quotes_and_truncates() {
        assert_eq!(
            finished_title("fix the bug", false),
            "'fix the bug' finished"
        );
        assert_eq!(
            finished_title("add .DS_Store to .gitignore and untrack copies", false),
            "'add .DS_Store to .gitignor...' finished"
        );
        assert_eq!(finished_title("  ", false), "plank finished");
        assert_eq!(
            finished_title("fix the bug", true),
            "'fix the bug' interrupted"
        );
    }

    #[test]
    fn latest_output_body_takes_tail() {
        assert_eq!(latest_output_body("", false), "Task complete");
        assert_eq!(latest_output_body("", true), "Task interrupted");
        assert_eq!(
            latest_output_body("all done\nok", false),
            "Latest output: all done ok"
        );
        let long = format!("{} THE END", "x".repeat(300));
        let body = latest_output_body(&long, false);
        assert!(body.starts_with("Latest output: ..."));
        assert!(body.ends_with("THE END"));
    }

    #[test]
    fn renotify_returns_false_with_no_prior_notification() {
        // No notification has fired in a fresh test process, so there is nothing
        // to re-show. (Serlializes against other tests that might set LAST.)
        let _g = TEST_LOCK.lock().unwrap();
        *LAST.lock().unwrap() = None;
        assert!(!renotify());
    }

    #[test]
    fn renotify_replays_last_after_a_deliver() {
        let _g = TEST_LOCK.lock().unwrap();
        // Force the gate on so `deliver` records + delivers (platform delivery
        // is a detached no-op off-macOS / best-effort on macOS; only the
        // recording matters here).
        set_mode(NotifyMode::Always);
        notify_sticky("hello", Some("sub"), "world");
        let last = LAST.lock().unwrap().clone();
        assert_eq!(last.as_ref().unwrap().summary, "hello");
        assert_eq!(last.as_ref().unwrap().subtitle.as_deref(), Some("sub"));
        assert_eq!(last.as_ref().unwrap().body, "world");
        // renotify reports success and preserves the recorded content.
        assert!(renotify());
        let last2 = LAST.lock().unwrap().clone();
        assert_eq!(last2.as_ref().unwrap().summary, "hello");
    }

    #[test]
    fn suppressed_deliver_does_not_clobber_last() {
        let _g = TEST_LOCK.lock().unwrap();
        set_mode(NotifyMode::Always);
        notify_sticky("first", None, "body1");
        // Now disable: a subsequent notify must not overwrite the recorded last.
        set_mode(NotifyMode::Never);
        notify_full("second", None, "body2");
        let last = LAST.lock().unwrap().clone();
        assert_eq!(last.as_ref().unwrap().summary, "first");
        assert_eq!(last.as_ref().unwrap().body, "body1");
    }

    #[test]
    fn mode_parses_and_round_trips() {
        assert_eq!(NotifyMode::parse("always"), Some(NotifyMode::Always));
        assert_eq!(NotifyMode::parse("Unfocused"), Some(NotifyMode::Unfocused));
        assert_eq!(NotifyMode::parse("never"), Some(NotifyMode::Never));
        // Pre-mode boolean spellings stay accepted.
        assert_eq!(NotifyMode::parse("true"), Some(NotifyMode::Always));
        assert_eq!(NotifyMode::parse("false"), Some(NotifyMode::Never));
        assert_eq!(NotifyMode::parse("sometimes"), None);
        for m in [NotifyMode::Always, NotifyMode::Unfocused, NotifyMode::Never] {
            assert_eq!(NotifyMode::parse(m.as_str()), Some(m));
        }
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

        // Unfocused delivers only while the window is not focused.
        set_mode(NotifyMode::Unfocused);
        set_focused(true);
        assert!(!should_deliver());
        set_focused(false);
        assert!(should_deliver());
        // Restore the process-global defaults for other tests.
        set_mode(NotifyMode::Always);
        set_focused(false);
    }
}
