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
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Mutex, OnceLock};

use turbo_debug_client::StreamKind;

static MIRROR: Mutex<Option<TcpStream>> = Mutex::new(None);

// Overridable only by tests, so they can point `reconcile` at a console
// listening on an ephemeral port instead of the real 7878. Never touched by
// production code.
static CONTROL_PORT: AtomicU16 = AtomicU16::new(turbo_debug_client::CONTROL_PORT);

/// A name that identifies this plank instance in the console's window list.
///
/// Chosen as `<project-dir-name>-<pid>`: readable (shows which project the
/// session belongs to) and unique per process, so two plank instances in
/// different worktrees — or two runs in the same one — get distinct windows
/// rather than fighting over one. The console reuses a window per name across
/// reconnects, which is exactly what we want across a `/config` toggle that
/// disconnects and reconnects the mirror within a single plank run.
fn session_name() -> &'static str {
    static NAME: OnceLock<String> = OnceLock::new();
    NAME.get_or_init(|| {
        let dir = std::env::current_dir()
            .ok()
            .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "plank".to_string());
        format!("{dir}-{}", std::process::id())
    })
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
    if let Ok(stream) = turbo_debug_client::connect_on(port, StreamKind::Tokens, session_name()) {
        *slot = Some(stream);
    }
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
        CONTROL_PORT.store(0, Ordering::Relaxed); // nothing listens on 0
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
}
