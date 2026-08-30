// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Named, reconnectable stream sessions over loopback TCP.
//!
//! One control listener performs the handshake and allocates a dedicated data
//! listener per session. A session outlives its data socket, so a client that
//! drops can reconnect to the same port and rejoin the same window with its
//! transcript intact.
//!
//! Thread growth is unbounded by design at this scope: one thread per control
//! connection, one per session's accept loop, and one per attached data
//! connection. Acceptable given the expected session counts; not a resource
//! pool.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::proto::{HelloError, parse_hello};

/// How long the handshake line may take to arrive.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
/// Read chunk size for a data socket.
const READ_CHUNK: usize = 8192;

/// Stable identifier for a session, independent of its name.
pub type SessionId = u64;

/// Something the UI needs to know about.
#[derive(Debug, Clone)]
pub enum ServerEvent {
    /// A new session exists; open a window for it.
    Opened {
        id: SessionId,
        name: String,
        port: u16,
    },
    /// A data socket just attached to a session (`live` flipped false ->
    /// true). Fires for the ordinary first attach of a brand-new session
    /// *and* every later reattach — `reattached` tells the two apart so the
    /// UI draws its "-- reconnected --" rule only for a genuine rejoin.
    /// This subsumes the old handshake-level `Reconnected` event: a
    /// repeat `HELLO` only ever matters to the UI once the client's data
    /// socket actually attaches, so attach is the single source of truth
    /// for "connected" (see `.superpowers/sdd/lifecycle-fixes-report.md`,
    /// defect 1).
    Attached { id: SessionId, reattached: bool },
    /// Stream bytes for a session.
    Bytes { id: SessionId, data: Vec<u8> },
    /// The data socket closed; the session and its port stay alive.
    Disconnected { id: SessionId },
    /// The session's idle TTL expired and it was dropped; close its window.
    /// Sent by [`Server::reap`], on the same channel as every other
    /// lifecycle event.
    Closed { id: SessionId },
}

#[derive(Debug)]
struct Session {
    id: SessionId,
    port: u16,
    /// True while a data socket is attached; guards against two writers.
    live: Arc<AtomicBool>,
    /// Bumped by one on every successful attach (never reset). A value of
    /// `1` means "first ever attach" (so `Attached.reattached` is false);
    /// anything higher is a genuine reattach. A `LiveGuard` captures the
    /// value at its own attach and, on drop, only tears the attachment
    /// down if this counter still matches — otherwise a newer attachment
    /// already owns the session and the old, asynchronously-observed EOF
    /// must not clobber it (defect 3: reconnect race).
    generation: Arc<AtomicU64>,
    /// Set when the data socket detaches (or the session is first created,
    /// before anything ever attaches); cleared the instant a client
    /// attaches — via a fresh `HELLO` reconnect or a new data-socket
    /// accept — so a session currently in use is never a reap candidate.
    idle_since: Option<Instant>,
    /// Told to `true` by [`Server::reap`] to stop this session's data-port
    /// accept loop (unused, always `false`, for an anonymous session, which
    /// has no listener of its own — see the `NotHello` branch of
    /// `handle_control`).
    shutdown: Arc<AtomicBool>,
}

/// Shared teardown for one attached data connection: whatever ends `pump`
/// (clean EOF, read error, or a panic unwinding through the caller), the
/// live flag must come back down and `idle_since` must be set — a stuck
/// `true` would permanently refuse every future reconnect attempt for this
/// session, which is the exact failure this design exists to prevent. A
/// guard makes that unconditional, and is shared by the named-session
/// accept loop and the anonymous raw-stream path so both age out the same
/// way (see finding 3 in `.superpowers/sdd/task-9-report.md`).
struct LiveGuard {
    sessions: Arc<Mutex<HashMap<String, Session>>>,
    name: String,
    tx: Sender<ServerEvent>,
    id: SessionId,
    /// The generation this guard's attachment owns; see `Session::generation`.
    generation: u64,
}

impl Drop for LiveGuard {
    fn drop(&mut self) {
        // The live-flag flip, the idle_since update and the "was this guard
        // outrun by a newer attach" check must happen as one atomic step
        // under the session-map lock — not as separate unguarded accesses —
        // or a newer attacher's own lock-held transaction (see
        // `open_or_reuse`'s accept loop) could interleave with this one and
        // reintroduce exactly the torn read/write defect 3 exists to close.
        let mut map = self.sessions.lock().unwrap();
        let Some(s) = map.get_mut(&self.name) else {
            return;
        };
        if s.generation.load(Ordering::SeqCst) != self.generation {
            // A newer attachment has already taken over this session; it
            // owns `live` and `idle_since` now. Tearing them down here
            // would be exactly the stale-guard bug: it would mark a
            // currently-attached session both not-live and idle, and emit
            // a spurious Disconnected for a connection that never left.
            return;
        }
        s.live.store(false, Ordering::SeqCst);
        s.idle_since = Some(Instant::now());
        drop(map);
        let _ = self.tx.send(ServerEvent::Disconnected { id: self.id });
    }
}

/// The listening server.
///
/// `bind` spawns a background thread that owns the control [`TcpListener`]
/// and accepts connections for the life of the process; dropping `Server`
/// does **not** stop that thread or close the control port — there is
/// currently no shutdown handshake for the control listener itself. Only
/// per-session data listeners are torn down early, and only through
/// [`Server::reap`]. This is a deliberate, narrower scope than "dropping it
/// stops accepting" would suggest; widening it is future work, not a claim
/// this code already makes good on.
#[derive(Debug)]
pub struct Server {
    control_port: u16,
    sessions: Arc<Mutex<HashMap<String, Session>>>,
    tx: Sender<ServerEvent>,
    rx: Receiver<ServerEvent>,
}

impl Server {
    /// Binds the control port. Pass `0` to let the OS choose (tests do).
    ///
    /// # Errors
    /// Returns the OS error when the address cannot be bound.
    pub fn bind(port: u16) -> std::io::Result<Self> {
        let control = TcpListener::bind(("127.0.0.1", port))?;
        let control_port = control.local_addr()?.port();
        let (tx, rx) = channel();
        let sessions: Arc<Mutex<HashMap<String, Session>>> = Arc::new(Mutex::new(HashMap::new()));

        let s = Arc::clone(&sessions);
        let t = tx.clone();
        std::thread::spawn(move || {
            for stream in control.incoming().flatten() {
                let s = Arc::clone(&s);
                let t = t.clone();
                std::thread::spawn(move || handle_control(stream, &s, &t));
            }
        });

        Ok(Self {
            control_port,
            sessions,
            tx,
            rx,
        })
    }

    /// The port clients send `HELLO` to.
    #[must_use]
    pub fn control_port(&self) -> u16 {
        self.control_port
    }

    /// Events for the UI to drain.
    #[must_use]
    pub fn events(&self) -> &Receiver<ServerEvent> {
        &self.rx
    }

    /// Number of sessions with a live data socket.
    ///
    /// # Panics
    /// If the internal session-map mutex is poisoned.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.sessions
            .lock()
            .unwrap()
            .values()
            .filter(|s| s.live.load(Ordering::SeqCst))
            .count()
    }

    /// Drops sessions that have been detached longer than `ttl`, sending a
    /// [`ServerEvent::Closed`] for each one on the same channel as every
    /// other lifecycle event — the coherent way for a caller draining
    /// [`Server::events`] to learn what to close, whether it comes from a
    /// live connection or from reaping.
    ///
    /// Each reaped session's data-port listener thread is told to stop and
    /// its listener is dropped, releasing the port: a session's TCP port
    /// does not outlive the session. Simply dropping a `TcpListener` while
    /// another thread is blocked in `accept()` does not reliably wake that
    /// thread (the behaviour is platform-dependent), so a shutdown flag is
    /// set first and then a throwaway self-connect forces one more
    /// iteration of the accept loop, which observes the flag and exits.
    ///
    /// A connected session, or one that has reconnected since it last
    /// detached, never expires: `idle_since` is `None` in both cases.
    ///
    /// # Panics
    /// If the internal session-map mutex is poisoned.
    pub fn reap(&mut self, ttl: Duration) {
        let mut sessions = self.sessions.lock().unwrap();
        let mut reaped: Vec<(SessionId, u16, Arc<AtomicBool>)> = Vec::new();
        sessions.retain(|_, s| {
            let expired = s
                .idle_since
                .is_some_and(|t| t.elapsed() > ttl && !s.live.load(Ordering::SeqCst));
            if expired {
                reaped.push((s.id, s.port, Arc::clone(&s.shutdown)));
            }
            !expired
        });
        drop(sessions);

        for (id, port, shutdown) in reaped {
            wake_and_shutdown(id, port, &shutdown);
            let _ = self.tx.send(ServerEvent::Closed { id });
        }
    }

    /// Tears a session down immediately, regardless of its idle state:
    /// used when the UI window is closed by the user rather than by the
    /// idle TTL. Reuses `reap`'s shutdown mechanism (shutdown flag plus a
    /// self-connect to wake the blocked accept loop) so the session's data
    /// port and accept thread are actually released, not just forgotten —
    /// see defect 2 in `.superpowers/sdd/lifecycle-fixes-report.md` for why
    /// closing a window must tear the server-side session down rather than
    /// leaving it live forever with no window to show it.
    ///
    /// No event is sent: the caller (the UI) already knows it is closing
    /// this session and is not waiting to be told.
    ///
    /// # Panics
    /// If the internal session-map mutex is poisoned.
    pub fn close_session(&mut self, id: SessionId) {
        let removed = {
            let mut sessions = self.sessions.lock().unwrap();
            let name = sessions
                .iter()
                .find(|(_, s)| s.id == id)
                .map(|(n, _)| n.clone());
            name.and_then(|n| sessions.remove(&n))
        };
        let Some(session) = removed else { return };
        wake_and_shutdown(id, session.port, &session.shutdown);
    }
}

/// Tells a session's accept loop to stop and forces it to notice: dropping
/// a `TcpListener` does not reliably wake a thread blocked in `accept()` on
/// all platforms, so the shutdown flag is set first and then a throwaway
/// self-connect forces one more iteration of the accept loop, which
/// observes the flag and exits, releasing the port. Retries on transient
/// failures (EMFILE, fd exhaustion, etc). A no-op for an anonymous session
/// (`port == 0`), which has no listener of its own.
fn wake_and_shutdown(id: SessionId, port: u16, shutdown: &Arc<AtomicBool>) {
    shutdown.store(true, Ordering::SeqCst);
    if port != 0 {
        const MAX_ATTEMPTS: u32 = 3;
        for attempt in 1..=MAX_ATTEMPTS {
            match TcpStream::connect(("127.0.0.1", port)) {
                Ok(_) => break,
                Err(e) if attempt == MAX_ATTEMPTS => {
                    eprintln!("Failed to wake session {id} on port {port}: {e}");
                }
                Err(_) => {
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
    }
}

/// Result of one accept-loop attach attempt, decided atomically under the
/// session-map lock (see the accept loop in `open_or_reuse`).
enum AttachOutcome {
    /// The session was removed (closed or reaped) out from under this
    /// listener; stop accepting.
    SessionGone,
    /// Another data socket is already live for this session.
    Rejected,
    /// This connection is now the session's live attachment, at this
    /// generation.
    Attached { generation: u64 },
}

/// Next session id and anonymous-name counter.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_ANON: AtomicU64 = AtomicU64::new(1);

/// Runs the handshake on one control connection.
fn handle_control(
    stream: TcpStream,
    sessions: &Arc<Mutex<HashMap<String, Session>>>,
    tx: &Sender<ServerEvent>,
) {
    let _ = stream.set_read_timeout(Some(HANDSHAKE_TIMEOUT));
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    });
    let mut writer = stream;

    let mut line = String::new();
    if reader.read_line(&mut line).is_err() {
        return;
    }

    match parse_hello(&line) {
        Ok(name) => match open_or_reuse(&name, sessions, tx) {
            Ok(port) => {
                let _ = writeln!(writer, "PORT {port}");
            }
            Err(_) => {
                let _ = writeln!(writer, "ERR no port");
            }
        },
        Err(HelloError::BadName) => {
            let _ = writeln!(writer, "{}", HelloError::BadName.wire());
        }
        Err(HelloError::NotHello) => {
            // Not a handshake: an anonymous raw stream. The line already read
            // is part of the stream and must not be lost.
            let n = NEXT_ANON.fetch_add(1, Ordering::SeqCst);
            let name = format!("anon-{n}");
            let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
            // An anonymous session is a one-shot: this connection is its
            // only ever attachment, so generation is fixed at 1 (first and
            // only attach) for the life of the session.
            sessions.lock().unwrap().insert(
                name.clone(),
                Session {
                    id,
                    port: 0,
                    live: Arc::new(AtomicBool::new(true)),
                    generation: Arc::new(AtomicU64::new(1)),
                    idle_since: None,
                    shutdown: Arc::new(AtomicBool::new(false)),
                },
            );
            let _ = tx.send(ServerEvent::Opened {
                id,
                name: name.clone(),
                port: 0,
            });
            let _ = tx.send(ServerEvent::Attached {
                id,
                reattached: false,
            });
            let _ = tx.send(ServerEvent::Bytes {
                id,
                data: line.into_bytes(),
            });
            let _ = writer.set_read_timeout(None);
            // The guard's drop sends Disconnected and sets idle_since,
            // giving this anonymous session the same reapable lifecycle as
            // a named session's data socket (see finding 3).
            let _guard = LiveGuard {
                sessions: Arc::clone(sessions),
                name,
                tx: tx.clone(),
                id,
                generation: 1,
            };
            pump(reader, id, tx);
        }
    }
}

/// Returns the data port for `name`, creating the session if it is new.
fn open_or_reuse(
    name: &str,
    sessions: &Arc<Mutex<HashMap<String, Session>>>,
    tx: &Sender<ServerEvent>,
) -> std::io::Result<u16> {
    {
        let mut map = sessions.lock().unwrap();
        if let Some(existing) = map.get_mut(name) {
            // A HELLO reconnect counts as the client showing up again, even
            // before a new data socket attaches: clear idle_since now so a
            // concurrent reap cannot drop the session out from under the
            // client that is about to dial the data port. The UI is not
            // told anything here — `ServerEvent::Attached` (sent once the
            // data socket actually attaches) is the sole source of truth
            // for "connected", so there is nothing to notify yet.
            existing.idle_since = None;
            let port = existing.port;
            return Ok(port);
        }
    }

    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
    let shutdown = Arc::new(AtomicBool::new(false));

    sessions.lock().unwrap().insert(
        name.to_string(),
        Session {
            id,
            port,
            live: Arc::new(AtomicBool::new(false)),
            generation: Arc::new(AtomicU64::new(0)),
            idle_since: Some(Instant::now()),
            shutdown: Arc::clone(&shutdown),
        },
    );
    let _ = tx.send(ServerEvent::Opened {
        id,
        name: name.to_string(),
        port,
    });

    let tx = tx.clone();
    let sessions = Arc::clone(sessions);
    let name = name.to_string();
    std::thread::spawn(move || {
        // Each accepted connection gets its own thread: `pump` blocks on
        // reading that socket until it closes, and the accept loop must
        // keep running underneath it — otherwise a first, still-open
        // writer would starve `incoming()` and a genuine second writer
        // (or a reconnect after a clean disconnect) could never be
        // accepted at all.
        for stream in listener.incoming().flatten() {
            if shutdown.load(Ordering::SeqCst) {
                // Reaped: `Server::reap` set the flag and forced this wake
                // with a throwaway self-connect. Stop accepting and drop
                // `listener` (falling out of this closure), which releases
                // the port. The stream that woke us is discarded.
                break;
            }

            // The "is someone already attached" check and the attach
            // itself (flipping `live`, bumping `generation`, clearing
            // `idle_since`) must happen as one atomic transaction under the
            // session-map lock: doing the check and the flip as separate
            // unguarded atomic ops (the old `live.swap`) is exactly the
            // torn read/write that let a stale `LiveGuard::drop` race a
            // fresh attach (defect 3).
            let attach = {
                let mut map = sessions.lock().unwrap();
                match map.get_mut(&name) {
                    None => AttachOutcome::SessionGone,
                    Some(s) if s.live.load(Ordering::SeqCst) => AttachOutcome::Rejected,
                    Some(s) => {
                        s.live.store(true, Ordering::SeqCst);
                        let generation = s.generation.fetch_add(1, Ordering::SeqCst) + 1;
                        s.idle_since = None;
                        AttachOutcome::Attached { generation }
                    }
                }
            };

            match attach {
                AttachOutcome::SessionGone => break,
                AttachOutcome::Rejected => {
                    // Already streaming: one writer per session.
                    let mut s = stream;
                    let _ = writeln!(s, "ERR already attached");
                }
                AttachOutcome::Attached { generation } => {
                    let reattached = generation > 1;
                    let _ = tx.send(ServerEvent::Attached { id, reattached });

                    let tx = tx.clone();
                    let sessions = Arc::clone(&sessions);
                    let name = name.clone();
                    std::thread::spawn(move || {
                        let _guard = LiveGuard {
                            sessions,
                            name,
                            tx: tx.clone(),
                            id,
                            generation,
                        };
                        pump(BufReader::new(stream), id, &tx);
                    });
                }
            }
        }
    });

    Ok(port)
}

/// Reads a data socket to EOF, forwarding chunks as events.
fn pump(mut reader: BufReader<TcpStream>, id: SessionId, tx: &Sender<ServerEvent>) {
    let _ = reader.get_ref().set_read_timeout(None);
    let mut buf = vec![0u8; READ_CHUNK];
    loop {
        match reader.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if tx
                    .send(ServerEvent::Bytes {
                        id,
                        data: buf[..n].to_vec(),
                    })
                    .is_err()
                {
                    break;
                }
            }
        }
    }
}

/// Deterministic, white-box coverage for defect 3 (the reconnect race).
///
/// `close_then_immediate_redial_ends_up_attached_with_one_writer_and_no_spurious_disconnect`
/// in `tests/protocol.rs` drives the real race over loopback TCP under many
/// rapid iterations, but the actual window — an old `LiveGuard::drop`
/// racing a brand-new attach — is a matter of thread-scheduling luck: it
/// reproduced reliably against the pre-fix code the first few times this
/// was tried, but is not *guaranteed* to reproduce on every machine or
/// every run (confirmed here: 500 iterations of the deliberately
/// reintroduced pre-fix logic passed clean on this machine in one run).
/// These tests instead construct the exact ordering directly — no network,
/// no scheduler dependency — so the invariant is checked every time, not
/// "usually".
#[cfg(test)]
mod guard_generation_tests {
    use super::*;
    use std::sync::mpsc::channel;

    fn make_session(port: u16) -> Session {
        Session {
            id: 1,
            port,
            live: Arc::new(AtomicBool::new(true)),
            generation: Arc::new(AtomicU64::new(1)),
            idle_since: None,
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// The exact ordering defect 3 describes: an old attachment's guard
    /// drops *after* a newer attachment has already taken the session over
    /// (generation bumped, `live` still true). The old guard must not clear
    /// `live`, must not set `idle_since`, and must not send a
    /// `Disconnected` — any of those would tear down or misreport a
    /// connection that is still genuinely attached.
    #[test]
    fn outrun_guard_does_not_clobber_a_newer_attachment() {
        let sessions: Arc<Mutex<HashMap<String, Session>>> = Arc::new(Mutex::new(HashMap::new()));
        let name = "race".to_string();
        sessions.lock().unwrap().insert(name.clone(), make_session(4242));

        let (tx, rx) = channel();
        let outrun_guard = LiveGuard {
            sessions: Arc::clone(&sessions),
            name: name.clone(),
            tx,
            id: 1,
            generation: 1,
        };

        // A newer attachment takes over exactly as the accept loop's
        // atomic transaction does: bump generation, `live` stays true.
        {
            let map = sessions.lock().unwrap();
            let s = map.get(&name).unwrap();
            s.generation.fetch_add(1, Ordering::SeqCst);
            assert!(s.live.load(Ordering::SeqCst));
        }

        drop(outrun_guard);

        let map = sessions.lock().unwrap();
        let s = map.get(&name).unwrap();
        assert!(
            s.live.load(Ordering::SeqCst),
            "an outrun guard cleared `live` out from under the new attachment"
        );
        assert!(
            s.idle_since.is_none(),
            "an outrun guard marked a currently-attached session idle"
        );
        drop(map);
        assert!(
            rx.try_recv().is_err(),
            "an outrun guard sent a spurious Disconnected"
        );
    }

    /// The mirror case: a guard whose generation is still current (nothing
    /// newer has attached) must tear the attachment down exactly as before
    /// — this is not a change in the ordinary, non-racing path.
    #[test]
    fn current_guard_tears_down_normally() {
        let sessions: Arc<Mutex<HashMap<String, Session>>> = Arc::new(Mutex::new(HashMap::new()));
        let name = "race".to_string();
        sessions.lock().unwrap().insert(name.clone(), make_session(4242));

        let (tx, rx) = channel();
        let guard = LiveGuard {
            sessions: Arc::clone(&sessions),
            name: name.clone(),
            tx,
            id: 1,
            generation: 1,
        };
        drop(guard);

        let map = sessions.lock().unwrap();
        let s = map.get(&name).unwrap();
        assert!(!s.live.load(Ordering::SeqCst));
        assert!(s.idle_since.is_some());
        drop(map);
        assert!(matches!(
            rx.try_recv(),
            Ok(ServerEvent::Disconnected { id: 1 })
        ));
    }
}
