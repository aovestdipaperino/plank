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
    /// A known session's client attached again.
    Reconnected { id: SessionId },
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
    live: Arc<AtomicBool>,
    sessions: Arc<Mutex<HashMap<String, Session>>>,
    name: String,
    tx: Sender<ServerEvent>,
    id: SessionId,
}

impl Drop for LiveGuard {
    fn drop(&mut self) {
        self.live.store(false, Ordering::SeqCst);
        if let Some(s) = self.sessions.lock().unwrap().get_mut(&self.name) {
            s.idle_since = Some(Instant::now());
        }
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
            shutdown.store(true, Ordering::SeqCst);
            if port != 0 {
                // Dropping a TcpListener does not reliably wake a thread blocked in
                // accept() on all platforms, so we force one more iteration of the
                // accept loop with a self-connect. This wakes the blocked thread so
                // it observes the shutdown flag and exits cleanly, releasing the port.
                // Retry on transient failures (EMFILE, fd exhaustion, etc).
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
            let _ = self.tx.send(ServerEvent::Closed { id });
        }
    }
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
            let attached = Arc::new(AtomicBool::new(true));
            sessions.lock().unwrap().insert(
                name.clone(),
                Session {
                    id,
                    port: 0,
                    live: Arc::clone(&attached),
                    idle_since: None,
                    shutdown: Arc::new(AtomicBool::new(false)),
                },
            );
            let _ = tx.send(ServerEvent::Opened {
                id,
                name: name.clone(),
                port: 0,
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
                live: attached,
                sessions: Arc::clone(sessions),
                name,
                tx: tx.clone(),
                id,
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
            // client that is about to dial the data port.
            existing.idle_since = None;
            let id = existing.id;
            let port = existing.port;
            drop(map);
            let _ = tx.send(ServerEvent::Reconnected { id });
            return Ok(port);
        }
    }

    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
    let live = Arc::new(AtomicBool::new(false));
    let shutdown = Arc::new(AtomicBool::new(false));

    sessions.lock().unwrap().insert(
        name.to_string(),
        Session {
            id,
            port,
            live: Arc::clone(&live),
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
            if live.swap(true, Ordering::SeqCst) {
                // Already streaming: one writer per session.
                let mut s = stream;
                let _ = writeln!(s, "ERR already attached");
                continue;
            }
            // A client just attached: this session is in use, not idle.
            if let Some(s) = sessions.lock().unwrap().get_mut(&name) {
                s.idle_since = None;
            }

            let live = Arc::clone(&live);
            let tx = tx.clone();
            let sessions = Arc::clone(&sessions);
            let name = name.clone();
            std::thread::spawn(move || {
                let _guard = LiveGuard {
                    live,
                    sessions,
                    name,
                    tx: tx.clone(),
                    id,
                };
                pump(BufReader::new(stream), id, &tx);
            });
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
