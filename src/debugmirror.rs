//! Mirrors the raw model stream to `turbo-debug-console`, an external Turbo
//! Vision window, whenever `ui.showThinking` is off.
//!
//! The console is optional infrastructure a developer may or may not have
//! running: nothing here may ever block a turn, panic, or spam retries. The
//! design is deliberately dumb:
//!
//! - A registry of connections keyed by [`MirrorId`], with
//!   [`MirrorId::PARENT`] for the session's own window and one entry per live
//!   sub-agent. Which one [`push`] writes to is a thread-local, defaulting to
//!   the parent, so code that knows nothing about sub-agents is unaffected.
//!   Sub-agents need this because a fan-out generates several at once
//!   (`generate_fanout_round` spawns a thread per slot) and each wants its own
//!   named window: `plank:<session>:subagent-<ordinal>`, ordinal monotonic
//!   within a session and reset when the session changes.
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

use std::cell::Cell;
use std::collections::BTreeMap;
use std::io::Write as _;
use std::net::TcpStream;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};

use turbo_debug_client::StreamKind;

/// Identifies one console connection. [`MirrorId::PARENT`] is the main
/// session's window; every sub-agent gets a fresh id from `NEXT_ORDINAL`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct MirrorId(usize);

impl MirrorId {
    /// The main session's window — the thread-local default, so every
    /// pre-existing `push` call site keeps writing to it untouched.
    pub const PARENT: MirrorId = MirrorId(0);
}

/// Every live console connection. Was a single `Option<TcpStream>`: one
/// process-wide connection could not name a window per sub-agent, and
/// sub-agents generate concurrently (`generate_fanout_round`), so a slot
/// would have been contended as well as unnameable.
// A `BTreeMap` rather than a `HashMap` purely so this can be a plain `static`:
// `HashMap::new` is not const (it seeds a `RandomState`), which would force a
// `LazyLock` for a map that never holds more than a handful of entries.
static MIRRORS: Mutex<BTreeMap<MirrorId, TcpStream>> = Mutex::new(BTreeMap::new());

thread_local! {
    /// Which connection this thread's [`push`] writes to. Defaults to the
    /// parent on every thread, so code that knows nothing about sub-agents
    /// behaves exactly as it did before the registry existed.
    static CURRENT: Cell<MirrorId> = const { Cell::new(MirrorId::PARENT) };
}

fn current() -> MirrorId {
    CURRENT.with(Cell::get)
}

/// Ordinal for the next sub-agent window this session. Starts at 1 (so the
/// first window is `subagent-1`) and resets whenever the session changes.
static NEXT_ORDINAL: AtomicUsize = AtomicUsize::new(1);

// Overridable only by tests, so they can point `reconcile` at a console
// listening on an ephemeral port instead of the real 7878. Never touched by
// production code.
// Zero under `cfg(test)`: nothing listens on port 0, so no test can dial out
// to a console the developer has running for real work. Without this, any test
// reaching `open_subagent` (every sub-agent sidechain test does, via
// `run_subagent_loop`) would open junk windows in that live console on every
// `cargo test` run. Tests that *want* a connection point this at their own
// `fake_console` listener.
static CONTROL_PORT: AtomicU16 = AtomicU16::new(if cfg!(test) {
    0
} else {
    turbo_debug_client::CONTROL_PORT
});

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
const FALLBACK_NAME: &str = "plank:unnamed";

/// Prefixes every window name so a console shared by several tools shows at a
/// glance which windows are plank's. `:` is printable, non-whitespace ASCII,
/// so it satisfies the console's name rules (1-64 bytes, printable ASCII, no
/// whitespace) and needs no protocol change.
const NAME_PREFIX: &str = "plank:";

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
    let mut cur = CURRENT_SESSION_NAME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if cur.as_deref() == Some(id) {
        return; // Same session as last time; no reconnect needed.
    }
    // Stored raw, decorated on read: `subagent_name` applies the prefix
    // itself, and handing it an already-prefixed name would produce
    // `plank:plank:<session>:subagent-1`.
    *cur = Some(id.to_owned());
    drop(cur);
    // Drop every live connection under the old name so `reconcile` below dials
    // a fresh one under the new name rather than treating "already connected"
    // as done. Sub-agent windows go too: their ordinals belong to the session
    // that just ended.
    MIRRORS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    NEXT_ORDINAL.store(1, Ordering::Relaxed);
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
    name_with_suffix(raw, "")
}

/// The window name for one sub-agent: the parent's name plus `:subagent-<n>`.
///
/// The suffix is budgeted *before* the session is truncated, for the same
/// reason the prefix is (see [`name_with_suffix`]): a wire name over 64 bytes
/// gets the handshake refused, which shows up as a missing window rather than
/// an error.
fn subagent_name(raw: &str, ordinal: usize) -> String {
    name_with_suffix(raw, &format!(":subagent-{ordinal}"))
}

/// Builds `<prefix><session><suffix>`, truncating only the session part so the
/// whole thing fits the console's 64-byte limit. An empty session falls back to
/// [`FALLBACK_NAME`], suffix still applied.
fn name_with_suffix(raw: &str, suffix: &str) -> String {
    // Budget the truncation against the prefix *and* the suffix, not the raw
    // id: taking 64 first and decorating after would push the wire name past
    // the console's 64-byte limit and get the handshake refused, which shows
    // up as a silently missing window rather than an error.
    let budget = 64 - NAME_PREFIX.len() - suffix.len();
    let cleaned: String = raw
        .chars()
        .filter(char::is_ascii_graphic)
        .take(budget)
        .collect();
    if cleaned.is_empty() {
        format!("{FALLBACK_NAME}{suffix}")
    } else {
        format!("{NAME_PREFIX}{cleaned}{suffix}")
    }
}

/// The name to present at the next handshake: the current session's name if
/// one has been recorded via [`set_session_id`], else [`FALLBACK_NAME`] —
/// covers `reconcile()` running before any session exists (see module docs).
fn session_name() -> String {
    sanitize_name(&raw_session_name())
}

/// The current session's raw id, before prefixing. [`subagent_name`] and
/// [`sanitize_name`] each apply their own decoration, so they need the
/// undecorated id rather than each other's output.
fn raw_session_name() -> String {
    CURRENT_SESSION_NAME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .unwrap_or_default()
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
    let mut reg = MIRRORS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !want_mirror {
        // showThinking is on: mirror nothing, hold no socket. Sub-agent
        // connections go too — they are gated on the same setting, and a
        // sub-agent outliving the toggle would keep a window alive that the
        // user just asked to stop seeing.
        reg.clear();
        return;
    }
    if reg.contains_key(&MirrorId::PARENT) {
        return; // Already connected; nothing to reconcile.
    }
    // Best-effort, single attempt. Nothing listening on the control port is
    // the overwhelmingly common case (no console running) and must be
    // silent: this is optional dev tooling, not a required dependency.
    let port = CONTROL_PORT.load(Ordering::Relaxed);
    if let Ok(stream) = turbo_debug_client::connect_on(port, StreamKind::Tokens, &session_name()) {
        reg.insert(MirrorId::PARENT, stream);
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
    let id = current();
    let mut reg = MIRRORS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(stream) = reg.get_mut(&id)
        && stream.write_all(text.as_bytes()).is_err()
    {
        reg.remove(&id);
    }
}

/// Flushes the mirror at the end of a turn. Best-effort like [`push`]: a
/// failure here just drops the (already-dead) connection.
pub fn flush() {
    let id = current();
    let mut reg = MIRRORS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(stream) = reg.get_mut(&id)
        && stream.flush().is_err()
    {
        reg.remove(&id);
    }
}

/// A console window belonging to one sub-agent, held for that sub-agent's
/// lifetime.
///
/// Always returned, even when no console is listening (the common case), so
/// callers never branch on console availability — the same discipline
/// [`reconcile`] follows. Dropping it retires the window and reclaims the
/// socket: ordinals are monotonic within a session, so without reaping, a
/// session that ran thirty sub-agents would hold thirty live sockets.
#[derive(Debug)]
pub struct SubagentMirror {
    id: MirrorId,
}

impl SubagentMirror {
    /// This window's id, for routing and for tests.
    #[must_use]
    pub fn id(&self) -> MirrorId {
        self.id
    }

    /// Routes *this thread's* [`push`] and [`flush`] to this window until the
    /// returned guard drops.
    ///
    /// A thread-local rather than a parameter because the fan-out boundary is
    /// already a thread boundary (`generate_fanout_round` spawns one thread per
    /// slot), and because the alternative is a seventh parameter on
    /// `generate_pass` that every non-sub-agent caller would have to pass and
    /// ignore.
    #[must_use]
    pub fn activate(&self) -> ActiveMirror {
        let previous = CURRENT.with(|c| c.replace(self.id));
        ActiveMirror { previous }
    }
}

impl Drop for SubagentMirror {
    fn drop(&mut self) {
        MIRRORS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.id);
    }
}

/// Restores the previous routing target when dropped, including on unwind — a
/// panicking sub-agent must not leave its thread pointed at a retired window,
/// or every later byte from that thread would vanish.
#[derive(Debug)]
pub struct ActiveMirror {
    previous: MirrorId,
}

impl Drop for ActiveMirror {
    fn drop(&mut self) {
        CURRENT.with(|c| c.set(self.previous));
    }
}

/// Opens a console window for the next sub-agent of this session, named
/// `plank:<session>:subagent-<ordinal>`.
///
/// Best-effort and single-attempt, exactly like [`reconcile`]: no console
/// means no connection and no complaint. Honours the same `ui.showThinking`
/// gate as the parent mirror, so turning thinking display on stops sub-agent
/// mirroring too.
#[must_use]
pub fn open_subagent() -> SubagentMirror {
    let ordinal = NEXT_ORDINAL.fetch_add(1, Ordering::Relaxed);
    let id = MirrorId(ordinal);
    if crate::settings::active().ui.show_thinking {
        return SubagentMirror { id };
    }
    let name = subagent_name(&raw_session_name(), ordinal);
    let port = CONTROL_PORT.load(Ordering::Relaxed);
    if let Ok(stream) = turbo_debug_client::connect_on(port, StreamKind::Tokens, &name) {
        MIRRORS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id, stream);
    }
    SubagentMirror { id }
}

/// The current routing target. Test-only accessor: production code never needs
/// to ask, it just calls [`push`].
#[cfg(test)]
#[must_use]
pub fn current_for_test() -> MirrorId {
    current()
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_window_name_is_prefixed_so_plank_s_windows_are_identifiable() {
        assert_eq!(sanitize_name("mellow-pauling"), "plank:mellow-pauling");
    }

    #[test]
    fn a_long_session_id_still_fits_the_console_s_64_byte_name_limit() {
        // Truncating to 64 and *then* prefixing would exceed the limit and get
        // the handshake refused, which surfaces as a missing window rather
        // than an error, so the budget has to account for the prefix.
        let name = sanitize_name(&"x".repeat(200));
        assert!(name.starts_with(NAME_PREFIX));
        assert!(
            name.len() <= 64,
            "wire name must fit the console's limit, got {} bytes: {name}",
            name.len()
        );
    }

    #[test]
    fn a_subagent_name_carries_the_session_and_the_ordinal() {
        assert_eq!(
            subagent_name("mellow-pauling", 3),
            "plank:mellow-pauling:subagent-3"
        );
    }

    /// The suffix has to come out of the same 64-byte budget as the prefix.
    /// Truncating the session to 58 and *then* appending `:subagent-12` would
    /// push the wire name past the console's limit and get the handshake
    /// refused -- a silently missing window rather than an error.
    #[test]
    fn a_long_session_id_still_fits_once_the_subagent_suffix_is_added() {
        let name = subagent_name(&"x".repeat(200), 12);
        assert!(name.starts_with(NAME_PREFIX), "{name}");
        assert!(name.ends_with(":subagent-12"), "{name}");
        assert!(
            name.len() <= 64,
            "wire name must fit the console's limit, got {} bytes: {name}",
            name.len()
        );
    }

    /// A session whose id sanitizes to nothing still gets a usable, distinct
    /// window per sub-agent rather than collapsing onto the parent's fallback.
    #[test]
    fn a_subagent_of_an_unnamed_session_still_gets_its_own_name() {
        let name = subagent_name("   ", 1);
        assert_eq!(name, "plank:unnamed:subagent-1");
    }

    #[test]
    fn a_name_that_sanitizes_to_nothing_falls_back_and_is_still_prefixed() {
        let name = sanitize_name("   ");
        assert_eq!(name, FALLBACK_NAME);
        assert!(
            name.starts_with(NAME_PREFIX),
            "fallback must look like plank's"
        );
    }

    use super::*;
    use std::io::Read;
    use std::net::TcpListener;

    // Tests share the process-wide MIRRORS registry and settings' process-wide
    // ACTIVE slot, so they must not run concurrently with each other or with
    // anything else touching `settings::install_for_test` for showThinking.
    // `settings::install_for_test` is thread-local (see settings.rs), which
    // is exactly what makes that safe across the suite; MIRRORS itself is not
    // thread-local, so within *this* module's tests we serialize by taking a
    // lock for the duration of each test.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn reset() {
        MIRRORS.lock().unwrap().clear();
        *CURRENT_SESSION_NAME.lock().unwrap() = None;
        NEXT_ORDINAL.store(1, Ordering::Relaxed);
        CURRENT.with(|c| c.set(MirrorId::PARENT));
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

    /// Like [`fake_console`], but hands back each accepted *data* socket so a
    /// test can read what plank wrote to that particular connection. Needed
    /// once there is more than one connection to tell apart.
    fn fake_console_keeping_sockets() -> (u16, std::sync::mpsc::Receiver<(String, TcpStream)>) {
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
                let Ok((data_sock, _)) = data.accept() else {
                    return;
                };
                if tx.send((line, data_sock)).is_err() {
                    return;
                }
            }
        });
        (control_port, rx)
    }

    /// The registry replaces a single slot, so the parent must still be
    /// reachable under a stable well-known id and still be what an
    /// un-redirected `push` writes to.
    #[test]
    fn the_parent_connection_lives_under_the_parent_id() {
        let _g = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let (control_port, rx) = fake_console_keeping_sockets();
        CONTROL_PORT.store(control_port, Ordering::Relaxed);
        let mut s = crate::settings::Settings::default();
        s.ui.show_thinking = false;
        crate::settings::install_for_test(s);

        reconcile();
        assert!(
            MIRRORS.lock().unwrap().contains_key(&MirrorId::PARENT),
            "reconcile must register the parent connection"
        );

        push("parent bytes");
        flush();

        let (_hello, mut sock) = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        let mut buf = [0u8; 64];
        let n = sock.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"parent bytes");

        reset();
    }

    /// Each sub-agent gets its own window, so ordinals must not repeat and the
    /// names must differ.
    #[test]
    fn each_subagent_gets_a_distinct_ordinal_and_window_name() {
        let _g = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let (control_port, rx) = fake_console_keeping_sockets();
        CONTROL_PORT.store(control_port, Ordering::Relaxed);
        let mut s = crate::settings::Settings::default();
        s.ui.show_thinking = false;
        crate::settings::install_for_test(s);
        set_session_id("bouncy-phelps");
        // Drain the parent's handshake so the two below are the sub-agents'.
        let _parent = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();

        let a = open_subagent();
        let b = open_subagent();
        assert_ne!(a.id(), b.id(), "ordinals must not repeat");

        let (hello_a, _sa) = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        let (hello_b, _sb) = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        assert!(
            hello_a.contains("plank:bouncy-phelps:subagent-1"),
            "{hello_a}"
        );
        assert!(
            hello_b.contains("plank:bouncy-phelps:subagent-2"),
            "{hello_b}"
        );

        reset();
    }

    /// The whole point of the guard: a sub-agent's bytes must reach its own
    /// window and must not contaminate the parent's.
    #[test]
    fn an_active_guard_routes_bytes_to_the_subagent_and_not_the_parent() {
        let _g = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let (control_port, rx) = fake_console_keeping_sockets();
        CONTROL_PORT.store(control_port, Ordering::Relaxed);
        let mut s = crate::settings::Settings::default();
        s.ui.show_thinking = false;
        crate::settings::install_for_test(s);
        set_session_id("bouncy-phelps");
        let (_h, mut parent_sock) = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();

        let sub = open_subagent();
        let (_h2, mut sub_sock) = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();

        {
            let _active = sub.activate();
            push("sub bytes");
            flush();
        }
        // Guard dropped: back to the parent.
        push("parent bytes");
        flush();

        let mut buf = [0u8; 64];
        let n = sub_sock.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"sub bytes");

        let n = parent_sock.read(&mut buf).unwrap();
        assert_eq!(
            &buf[..n],
            b"parent bytes",
            "the parent must see only its own bytes"
        );

        reset();
    }

    /// A sub-agent that panics must not leave this thread routed at a dead
    /// window — every later parent byte would vanish.
    #[test]
    fn the_guard_restores_the_previous_target_on_unwind() {
        let _g = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();

        let sub = open_subagent();
        let id = sub.id();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _active = sub.activate();
            assert_eq!(current(), id);
            panic!("sub-agent blew up");
        }));
        assert!(caught.is_err());
        assert_eq!(
            current(),
            MirrorId::PARENT,
            "the guard must restore the parent target even on unwind"
        );

        reset();
    }

    /// Ordinals are monotonic, so without reaping, a long session accumulates
    /// one live socket per sub-agent it ever ran.
    #[test]
    fn dropping_a_handle_removes_its_registry_entry() {
        let _g = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let (control_port, rx) = fake_console_keeping_sockets();
        CONTROL_PORT.store(control_port, Ordering::Relaxed);
        let mut s = crate::settings::Settings::default();
        s.ui.show_thinking = false;
        crate::settings::install_for_test(s);
        set_session_id("bouncy-phelps");
        let _parent = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();

        let sub = open_subagent();
        let id = sub.id();
        let _s = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        assert!(MIRRORS.lock().unwrap().contains_key(&id));

        drop(sub);
        assert!(
            !MIRRORS.lock().unwrap().contains_key(&id),
            "a dropped handle must not leak its socket"
        );

        reset();
    }

    /// A new session restarts numbering and retires the old session's
    /// sub-agent windows along with its parent window.
    #[test]
    fn a_new_session_resets_ordinals_and_clears_subagent_windows() {
        let _g = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let (control_port, rx) = fake_console_keeping_sockets();
        CONTROL_PORT.store(control_port, Ordering::Relaxed);
        let mut s = crate::settings::Settings::default();
        s.ui.show_thinking = false;
        crate::settings::install_for_test(s);

        set_session_id("first-session");
        let _p1 = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        let first = open_subagent();
        let _s1 = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        let first_id = first.id();
        std::mem::forget(first); // Simulate a handle still held across the switch.

        set_session_id("second-session");
        let _p2 = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        assert!(
            !MIRRORS.lock().unwrap().contains_key(&first_id),
            "the previous session's sub-agent windows must be retired"
        );

        let next = open_subagent();
        let (hello, _sock) = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        assert!(
            hello.contains("plank:second-session:subagent-1"),
            "ordinals restart with the session: {hello}"
        );
        drop(next);

        reset();
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
            MIRRORS.lock().unwrap().is_empty(),
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

        assert!(MIRRORS.lock().unwrap().is_empty());
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
        assert!(
            MIRRORS.lock().unwrap().contains_key(&MirrorId::PARENT),
            "should have connected"
        );

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
            MIRRORS.lock().unwrap().is_empty(),
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
        assert!(MIRRORS.lock().unwrap().contains_key(&MirrorId::PARENT));

        begin_in_think();
        flush();

        let mut server_side = accepted.join().unwrap();
        let mut buf = [0u8; 16];
        let n = server_side.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"<think>");

        reset();
    }
}
