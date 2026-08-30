//! Mirrors the raw model stream to `turbo-debug-console`, an external Turbo
//! Vision window, whenever `ui.showThinking` is off.
//!
//! The console is optional infrastructure a developer may or may not have
//! running: nothing here may ever block a turn, panic, or spam retries. The
//! design is deliberately dumb:
//!
//! - One process-wide connection slot. plank has exactly one live session at
//!   a time (per the TUI and plain-REPL front ends this ships), so a single
//!   slot is enough; a second `Agent` in the same process (tests) shares it,
//!   which is harmless since [`reconcile`] is idempotent.
//! - [`reconcile`] is the only thing that ever dials out, and it makes at most
//!   one connection attempt per call — never a retry loop. It is called (a)
//!   whenever the settings are swapped in (`settings::install`/`reinstall`),
//!   which is the "immediately" of the showThinking toggle, and (b)
//!   defensively at the start of each turn, which is where a console that
//!   started up *after* plank gets picked up, without plank needing a
//!   restart. Neither call site is a hot path: turns start far less often
//!   than tokens stream.
//! - [`push`] never retries. A write failure (console closed mid-generation)
//!   just drops the connection so the rest of the turn streams normally; the
//!   next reconcile (next turn, or the next settings change) will try again.
//!
//! What gets mirrored is the *whole* raw model stream — thinking, visible
//! answer, tool-call markup, byte for byte — because the console runs its own
//! copy of `trace-stream` and renders it exactly as plank would. Filtering to
//! "just the thinking" would leave the console unable to reproduce plank's
//! own rendering, which is the entire point of pointing it at the raw bytes.

use std::io::Write as _;
use std::net::TcpStream;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU16, Ordering};

use turbo_debug_client::StreamKind;

static MIRROR: Mutex<Option<TcpStream>> = Mutex::new(None);

// Overridable only by tests, so they can point `reconcile` at a console
// listening on an ephemeral port instead of the real 7878. Never touched by
// production code.
static CONTROL_PORT: AtomicU16 = AtomicU16::new(turbo_debug_client::CONTROL_PORT);

/// The session name to hand the console at the next connection, kept as
/// `Option` so "no session minted yet" is distinguishable from "named the
/// empty string" — the former falls back to [`FALLBACK_NAME`], deliberately,
/// rather than refusing to connect.
static CURRENT_SESSION_NAME: Mutex<Option<String>> = Mutex::new(None);

/// Used to name the console window before any session id exists —
/// `reconcile()` can run this early (see module docs: `settings::install` at
/// startup, and the defensive per-turn call before the first turn's session
/// rename lands). A window under this name is retired the moment a real
/// session name is known, since [`set_session_id`] reconnects on change.
const FALLBACK_NAME: &str = "plank-unnamed";

/// Records plank's current session name (the `adjective-celebrity` slug from
/// `SessionStore::mint_id`, or a rename) as the name to present at the next
/// handshake, and — if it actually changed since the last call — tears down
/// any live connection and reconciles immediately.
///
/// This is the "changing sessions" half of naming the console window: without
/// it, starting a new session would keep mirroring into a window titled after
/// the session that just ended, since the console reuses a window by name and
/// a stale connection never re-sends `HELLO`. Called wherever `ui.rs` learns
/// the session's id (session start, `/new`, `/rename`, `/resume`), it is cheap
/// when the name is unchanged: a mutex lock and a string comparison.
pub fn set_session_id(id: &str) {
    let name = sanitize_name(id);
    let mut cur = CURRENT_SESSION_NAME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if cur.as_deref() == Some(name.as_str()) {
        return; // Same session as last time; no reconnect needed.
    }
    *cur = Some(name);
    drop(cur);
    // Drop any live connection under the old name so `reconcile` below dials a
    // fresh one under the new name rather than treating "already connected" as
    // done.
    *MIRROR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    reconcile();
}

/// Constrains a candidate window name to what the console's handshake
/// accepts: 1-64 bytes of printable ASCII, no whitespace. Session slugs
/// (`adjective-celebrity`) already satisfy this, but a renamed session takes
/// arbitrary user text (`crate::session::validate_name` only forbids path
/// separators and a few reserved characters), so this does not trust that
/// blindly — a name that would not survive the handshake falls back rather
/// than silently losing the mirror.
fn sanitize_name(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .filter(char::is_ascii_graphic)
        .take(64)
        .collect();
    if cleaned.is_empty() {
        FALLBACK_NAME.to_string()
    } else {
        cleaned
    }
}

/// The name to present at the next handshake: the current session's name if
/// one has been recorded via [`set_session_id`], else [`FALLBACK_NAME`] —
/// covers `reconcile()` running before any session exists (see module docs).
fn session_name() -> String {
    CURRENT_SESSION_NAME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .unwrap_or_else(|| FALLBACK_NAME.to_string())
}

/// Reconciles the mirror connection with the current `ui.showThinking`
/// setting. Cheap to call often: when the desired state already matches
/// (connected-and-wanted, or disconnected-and-not-wanted) this is a mutex
/// lock and a comparison, no I/O.
///
/// This is the "immediately" half of the toggle: a settings change is
/// reflected in the connection the moment this runs (called from
/// `settings::install`/`reinstall`). The *display* half — whether a given
/// generation renders thinking — is fixed per `StreamRenderer` at
/// construction (`StreamRenderer::set_show_thinking`), so a generation
/// already in flight keeps rendering the way it started; only the next
/// generation picks up a mid-turn toggle. That split is intentional (see
/// `stream_generation` / `worker_generate_kind` in `ui.rs`) rather than an
/// oversight: swapping a renderer's mode mid-stream would risk splitting a
/// `<think>` block across two display modes.
pub fn reconcile() {
    let want_mirror = !crate::settings::active().ui.show_thinking;
    let mut slot = MIRROR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !want_mirror {
        *slot = None; // showThinking is on: mirror nothing, hold no socket.
        return;
    }
    if slot.is_some() {
        return; // Already connected; nothing to reconcile.
    }
    // Best-effort, single attempt. Nothing listening on the control port is
    // the overwhelmingly common case (no console running) and must be
    // silent: this is optional dev tooling, not a required dependency.
    let port = CONTROL_PORT.load(Ordering::Relaxed);
    if let Ok(stream) = turbo_debug_client::connect_on(port, StreamKind::Tokens, &session_name()) {
        *slot = Some(stream);
    }
}

/// Mirrors a synthetic opening `<think>` tag so the console's own
/// `StreamRenderer` enters the thinking state before any bytes arrive.
///
/// This is not the mirror going verbatim: with a local model, the chat
/// template pre-opens `<think>` in the prefill prefix without emitting the
/// tag, so the raw stream has no marker to key off. plank compensates for its
/// own rendering by calling `StreamRenderer::begin_in_think` directly (see the
/// call site in `ui.rs`, guarded on thinking being enabled and the engine
/// being local); the console gets no such direct call; it only ever sees
/// bytes. So this injects the one tag the console needs to reach the same
/// state, under the identical guard — call it only where plank's own renderer
/// gets `begin_in_think()`, never unconditionally, or a provider engine's
/// stream (which does emit real `<think>`/`</think>` tags) would be
/// mis-colored from the first token. The model always emits the closing
/// `</think>` itself, so the mirrored stream stays balanced.
pub fn begin_in_think() {
    push("<think>");
}

/// Mirrors one chunk of raw model bytes, exactly as fed to the local
/// `StreamRenderer`. A no-op when not connected (showThinking on, or no
/// console reachable). A write failure drops the connection rather than
/// erroring or retrying — the turn that owns these bytes must never notice.
pub fn push(text: &str) {
    let mut slot = MIRROR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(stream) = slot.as_mut()
        && stream.write_all(text.as_bytes()).is_err()
    {
        *slot = None;
    }
}

/// Flushes the mirror at the end of a turn. Best-effort like [`push`]: a
/// failure here just drops the (already-dead) connection.
pub fn flush() {
    let mut slot = MIRROR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(stream) = slot.as_mut()
        && stream.flush().is_err()
    {
        *slot = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::TcpListener;

    // Tests share the process-wide MIRROR slot and settings' process-wide
    // ACTIVE slot, so they must not run concurrently with each other or with
    // anything else touching `settings::install_for_test` for showThinking.
    // `settings::install_for_test` is thread-local (see settings.rs), which
    // is exactly what makes that safe across the suite; MIRROR itself is not
    // thread-local, so within *this* module's tests we serialize by taking a
    // lock for the duration of each test.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn reset() {
        *MIRROR.lock().unwrap() = None;
        *CURRENT_SESSION_NAME.lock().unwrap() = None;
        CONTROL_PORT.store(0, Ordering::Relaxed); // nothing listens on 0
    }

    /// Spins up a stand-in for `turbo-debug-console`'s control port that
    /// accepts exactly one handshake, hands back the `HELLO` line it
    /// received, and returns a connected data-port socket to the caller. Used
    /// by every test that needs to inspect what name a connection presents.
    fn fake_console() -> (u16, std::sync::mpsc::Receiver<String>) {
        let control = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let control_port = control.local_addr().unwrap().port();
        let data = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let data_port = data.local_addr().unwrap().port();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            loop {
                let Ok((mut sock, _)) = control.accept() else {
                    return;
                };
                let mut line = String::new();
                if BufReader::new(sock.try_clone().unwrap())
                    .read_line(&mut line)
                    .is_err()
                {
                    return;
                }
                let _ = writeln!(sock, "PORT {data_port}");
                let _ = data.accept();
                if tx.send(line).is_err() {
                    return;
                }
            }
        });
        (control_port, rx)
    }

    #[test]
    fn reconcile_does_not_connect_when_show_thinking_is_on() {
        let _g = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let mut s = crate::settings::Settings::default();
        s.ui.show_thinking = true;
        crate::settings::install_for_test(s);

        reconcile();

        assert!(
            MIRROR.lock().unwrap().is_none(),
            "showThinking on: no connection should be attempted"
        );
    }

    #[test]
    fn a_failed_connection_leaves_no_mirror_and_does_not_panic() {
        let _g = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let mut s = crate::settings::Settings::default();
        s.ui.show_thinking = false;
        crate::settings::install_for_test(s);

        reconcile(); // port 0: nothing is listening, must not panic/hang.

        assert!(MIRROR.lock().unwrap().is_none());
        // And pushing/flushing with no connection is a harmless no-op.
        push("hello");
        flush();
    }

    #[test]
    fn reconcile_connects_when_show_thinking_is_off_and_a_console_is_up() {
        let _g = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();

        // Stand in for turbo-debug-console's control port: reply with a data
        // port and accept one connection there.
        let control = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let control_port = control.local_addr().unwrap().port();
        let data = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let data_port = data.local_addr().unwrap().port();
        let accepted = std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            let (mut sock, _) = control.accept().unwrap();
            let mut line = String::new();
            BufReader::new(sock.try_clone().unwrap())
                .read_line(&mut line)
                .unwrap();
            assert!(line.starts_with("HELLO "), "{line}");
            writeln!(sock, "PORT {data_port}").unwrap();
            data.accept().unwrap().0
        });

        CONTROL_PORT.store(control_port, Ordering::Relaxed);
        let mut s = crate::settings::Settings::default();
        s.ui.show_thinking = false;
        crate::settings::install_for_test(s);

        reconcile();
        assert!(MIRROR.lock().unwrap().is_some(), "should have connected");

        push("thinking and answer bytes");
        flush();

        let mut server_side = accepted.join().unwrap();
        let mut buf = [0u8; 64];
        let n = server_side.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"thinking and answer bytes");

        // Flip the setting off (i.e. show_thinking back to default true) and
        // reconcile: the mirror must be dropped, not left dangling.
        let mut on = crate::settings::Settings::default();
        on.ui.show_thinking = true;
        crate::settings::install_for_test(on);
        reconcile();
        assert!(
            MIRROR.lock().unwrap().is_none(),
            "showThinking back on: mirror must be torn down"
        );

        reset();
    }

    /// The handshake must carry plank's session name, not a pid-based
    /// identity: that's the whole point of issue #1 — the window the user
    /// sees has to match the name shown above their prompt.
    #[test]
    fn handshake_carries_the_session_name_not_a_pid() {
        let _g = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let (control_port, rx) = fake_console();
        CONTROL_PORT.store(control_port, Ordering::Relaxed);
        let mut s = crate::settings::Settings::default();
        s.ui.show_thinking = false;
        crate::settings::install_for_test(s);

        set_session_id("spunky-oppenheimer");
        let line = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        assert!(
            line.contains("spunky-oppenheimer"),
            "handshake should carry the session name: {line}"
        );
        assert!(
            !line.contains(&std::process::id().to_string()),
            "handshake should not fall back to a pid-based name: {line}"
        );

        reset();
    }

    /// Before any session id is known (`reconcile()` running from
    /// `settings::install` at startup), the mirror must still connect rather
    /// than refuse — under a sensible fallback name.
    #[test]
    fn reconcile_before_any_session_falls_back_to_a_sensible_name() {
        let _g = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let (control_port, rx) = fake_console();
        CONTROL_PORT.store(control_port, Ordering::Relaxed);
        let mut s = crate::settings::Settings::default();
        s.ui.show_thinking = false;
        crate::settings::install_for_test(s);

        reconcile();
        let line = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        assert!(
            line.contains(FALLBACK_NAME),
            "no session yet: should fall back to {FALLBACK_NAME}: {line}"
        );

        reset();
    }

    /// Minting a new session (or a rename) must reconnect the mirror so the
    /// console's window follows the session the user is actually looking at,
    /// instead of keeping the old name forever.
    #[test]
    fn a_session_name_change_reconnects_under_the_new_name() {
        let _g = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let (control_port, rx) = fake_console();
        CONTROL_PORT.store(control_port, Ordering::Relaxed);
        let mut s = crate::settings::Settings::default();
        s.ui.show_thinking = false;
        crate::settings::install_for_test(s);

        set_session_id("parser-hunt");
        let first = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        assert!(first.contains("parser-hunt"), "{first}");

        // Same name again: no reason to tear down and redial.
        set_session_id("parser-hunt");
        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(100))
                .is_err(),
            "an unchanged session name must not reconnect"
        );

        set_session_id("spunky-oppenheimer");
        let second = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        assert!(
            second.contains("spunky-oppenheimer"),
            "new session should reconnect under its own name: {second}"
        );

        reset();
    }

    /// The console rejects anything outside 1-64 printable-ASCII bytes with
    /// no whitespace; a session name that would not survive the handshake
    /// (e.g. a `/rename` with spaces) must fall back rather than silently
    /// losing the mirror.
    #[test]
    fn an_unsafe_name_falls_back_instead_of_refusing_to_connect() {
        let _g = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let (control_port, rx) = fake_console();
        CONTROL_PORT.store(control_port, Ordering::Relaxed);
        let mut s = crate::settings::Settings::default();
        s.ui.show_thinking = false;
        crate::settings::install_for_test(s);

        // Sanitizing strips whitespace rather than refusing outright — a name
        // that is nothing *but* whitespace is the case that must fall back.
        set_session_id("   \t\t  ");
        let line = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        assert!(
            line.contains(FALLBACK_NAME),
            "all-whitespace name should fall back: {line}"
        );

        reset();
    }

    /// [`begin_in_think`] is a named wrapper over pushing the literal tag —
    /// exercised directly since the guard itself lives in `ui.rs`, not here.
    #[test]
    fn begin_in_think_mirrors_a_synthetic_open_tag() {
        let _g = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();

        let control = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let control_port = control.local_addr().unwrap().port();
        let data = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let data_port = data.local_addr().unwrap().port();
        let accepted = std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            let (mut sock, _) = control.accept().unwrap();
            let mut line = String::new();
            BufReader::new(sock.try_clone().unwrap())
                .read_line(&mut line)
                .unwrap();
            writeln!(sock, "PORT {data_port}").unwrap();
            data.accept().unwrap().0
        });

        CONTROL_PORT.store(control_port, Ordering::Relaxed);
        let mut s = crate::settings::Settings::default();
        s.ui.show_thinking = false;
        crate::settings::install_for_test(s);
        reconcile();
        assert!(MIRROR.lock().unwrap().is_some());

        begin_in_think();
        flush();

        let mut server_side = accepted.join().unwrap();
        let mut buf = [0u8; 16];
        let n = server_side.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"<think>");

        reset();
    }
}
