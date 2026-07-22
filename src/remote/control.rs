// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Remote-control interface (issue #25): a loopback WebSocket server that lets
//! another process or machine mirror a running plank instance and, by policy,
//! drive it. This is the CLI-only, backend-free variant specified in
//! `docs/REMOTE-CONTROL-DESIGN.md`.
//!
//! A remote client is *another front-end* over the existing
//! [`crate::worker::UiEvent`] stream: the worker broadcasts events to a
//! [`crate::worker::BroadcastBus`], each connection subscribes and pumps them
//! out as JSON frames, and inbound frames push prompts / interrupts / `/btw`
//! questions into the shared [`crate::worker::TurnShared`] — exactly what the
//! local UI does. There is one engine, one session, one transcript (design §7).
//!
//! Transport is blocking [`tungstenite`] on dedicated threads (no async
//! runtime), matching plank's synchronous worker idiom (design §4.7). The
//! server binds `127.0.0.1` only; off-box reach is the user's SSH tunnel.
//!
//! ## What this module implements
//! - The versioned JSON wire protocol ([`ServerFrame`] / [`ClientFrame`]), a
//!   near-1:1 image of `UiEvent` + `TurnShared` (design §4.3), with lossless
//!   round-trip.
//! - Token auth: generation, constant-time comparison, mandatory first-frame
//!   handshake (design §4.6).
//! - The single-controller [`ControlPolicy`] (one controller, many mirrors;
//!   design §4.4).
//! - The accept/connection threads: auth, `hello`, `snapshot` replay, live
//!   mirroring, `status` coalescing, and inbound control into `TurnShared`.
//!
//! ## Live wiring (issue #25, done)
//! - The worker's stream events mirror onto the bus and remote
//!   `prompt`/`command`/`btw`/`interrupt` frames drive the real agent through
//!   both `ui.rs` turn-loop paths: the TUI mirrors inline in `busy_ui_loop` and
//!   drives turns from the idle loop; the headless path runs a dedicated
//!   remote-serve loop (`run_remote_headless`).
//! - `command` (slash) frames route through the shared slash dispatcher: they
//!   land in the shared queue and the turn loop sends `/`-prefixed lines through
//!   the same path the local REPL/TUI uses.
//! - Reconnect grace window ([`CONTROL_GRACE`]): a dropped controller's slot is
//!   held so a brief drop can reclaim it via `resume_from`.
//!
//! ## Follow-up (issue #25, done)
//! - The plain-REPL fallback (piped stdin, no TTY) now interleaves remote input:
//!   a helper thread turns the blocking `read_line` into channel sends so the
//!   loop can `select` stdin against the remote queue, draining `pump_remote`
//!   and echoing the bus to stdout (see `ui::run_repl_plain_remote`).
//! - The CLI client `plank remote <url>` connects to this server, authenticates,
//!   mirrors output, and sends `prompt`/`command`/`btw`/`interrupt` frames (see
//!   [`crate::remote::client`]).
//!
//! ## Hardening (issue #25, done)
//! - `Origin` allow-list on the WebSocket upgrade: missing / loopback Origins
//!   are always allowed (native clients send none), other browser Origins must
//!   be allow-listed via `--control-origin` or the upgrade is refused with an
//!   HTTP 403 (see [`origin_allowed`]).
//! - Bounded per-client outbound queue: the socket write buffer is capped
//!   (`--control-queue-max`); a slow/stalled client whose unsent output exceeds
//!   the cap is evicted (slow-consumer backpressure) instead of buffered
//!   without bound. The bus prunes the dropped subscriber on the next
//!   broadcast.
//! - A minimal self-contained static web client served at `GET /` (see
//!   [`WEB_CLIENT_HTML`]); it speaks the exact JSON frames below.

use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tungstenite::Message;

use crate::status::{Status, WorkerState};
use crate::worker::{BroadcastBus, SeqEvent, TurnShared, UiEvent};

/// Wire protocol version carried in every frame's `v` field. Adding frame
/// types is backward-compatible; changing existing ones bumps this (design §7).
pub const PROTOCOL_VERSION: u32 = 1;

/// How often the connection loop wakes to pump the bus and poll the socket.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Minimum spacing between `status` frames per connection (coalescing, §4.9):
/// at most ~10/s. Text frames are never coalesced.
const STATUS_MIN_INTERVAL: Duration = Duration::from_millis(100);

/// The self-contained static web client served at `GET /`. No external deps:
/// all CSS/JS is inline. It authenticates with a token and speaks the same
/// [`PROTOCOL_VERSION`] JSON frames as the native client.
pub const WEB_CLIENT_HTML: &str = include_str!("web_client.html");

// --- Protocol ---------------------------------------------------------------

/// Flattened, serializable view of [`Status`] for `status` frames (§4.3). The
/// worker state is a lowercase string so the schema is language-neutral.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StatusWire {
    /// Worker state name (`idle`, `prefill`, `generating`, ...).
    pub state: String,
    /// Prefill tokens done / total.
    pub prefill_done: i32,
    /// Prefill tokens total.
    pub prefill_total: i32,
    /// Tokens generated so far.
    pub generated: i32,
    /// Generation throughput, tokens per second.
    pub gen_tps: f64,
    /// Prefill throughput, tokens per second.
    pub prefill_tps: f64,
    /// Seconds elapsed in the current operation.
    pub elapsed_secs: f64,
    /// Context tokens in use.
    pub ctx_used: i32,
    /// Context window size.
    pub ctx_size: i32,
    /// Error text (empty unless `state == "error"`).
    pub error: String,
}

fn state_name(s: WorkerState) -> &'static str {
    match s {
        WorkerState::Idle => "idle",
        WorkerState::Prefill => "prefill",
        WorkerState::Generating => "generating",
        WorkerState::Compacting => "compacting",
        WorkerState::Saving => "saving",
        WorkerState::Error => "error",
        WorkerState::Stopped => "stopped",
    }
}

impl From<&Status> for StatusWire {
    fn from(s: &Status) -> Self {
        Self {
            state: state_name(s.state).to_owned(),
            prefill_done: s.prefill_done,
            prefill_total: s.prefill_total,
            generated: s.generated,
            gen_tps: s.gen_tps,
            prefill_tps: s.prefill_tps,
            elapsed_secs: s.elapsed_secs,
            ctx_used: s.ctx_used,
            ctx_size: s.ctx_size,
            error: s.error.clone(),
        }
    }
}

/// One scrollback entry in a `snapshot` frame: a prior server message with its
/// bus sequence id.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScrollbackEntry {
    /// Bus sequence id of the original event.
    pub id: u64,
    /// The replayed message.
    #[serde(flatten)]
    pub msg: ServerMsg,
}

/// Server → client message body, discriminated by `type` (design §4.3).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMsg {
    /// Sent once on connect.
    Hello {
        /// Protocol version the server speaks.
        protocol_version: u32,
        /// plank crate version.
        plank_version: String,
        /// Numeric session id assigned to this connection.
        session_id: u64,
        /// Whether this session currently holds control.
        controller: bool,
    },
    /// Scrollback replay on connect / resume.
    Snapshot {
        /// Prior output frames (oldest first).
        scrollback: Vec<ScrollbackEntry>,
        /// Highest sequence id represented, for the client to `resume_from`.
        highest_id: Option<u64>,
    },
    /// Rendered assistant text.
    Visible {
        /// Text payload.
        text: String,
    },
    /// Thinking text.
    Think {
        /// Text payload.
        text: String,
    },
    /// Tool banner text.
    Tool {
        /// Text payload.
        text: String,
    },
    /// Stream error text.
    Error {
        /// Text payload.
        text: String,
    },
    /// A dim log line.
    Dim {
        /// Text payload.
        text: String,
    },
    /// A plain log line.
    Plain {
        /// Text payload.
        text: String,
    },
    /// A user-echo line.
    UserEcho {
        /// Text payload.
        text: String,
    },
    /// Terminate the in-progress rendered line.
    EndLine,
    /// A `/btw` side answer is starting.
    BtwBegin,
    /// The `/btw` answer finished.
    BtwEnd,
    /// Main-pass checkpoint marker.
    MainCheckpoint,
    /// Main-pass rollback marker.
    MainRollback,
    /// Status footer snapshot (coalesced, §4.9).
    Status {
        /// Flattened status fields.
        status: StatusWire,
    },
    /// Task-list counter snapshot (issue #35). The contextual strip is
    /// local-TUI-only; the remote wire carries just the completed/total counter.
    Tasks {
        /// Completed task count.
        completed: usize,
        /// Total task count.
        total: usize,
    },
    /// A control request from a non-controller was refused.
    ControlDenied {
        /// Human-readable reason.
        reason: String,
    },
    /// Reply to a client `ping`.
    Pong,
    /// The server is closing this session.
    Bye {
        /// Human-readable reason.
        reason: String,
    },
}

impl ServerMsg {
    /// Maps a worker [`UiEvent`] to its wire message. Total and lossless.
    #[must_use]
    pub fn from_event(ev: &UiEvent) -> Self {
        match ev {
            UiEvent::Visible(t) => Self::Visible { text: t.clone() },
            UiEvent::Think(t) => Self::Think { text: t.clone() },
            UiEvent::Tool(t) => Self::Tool { text: t.clone() },
            UiEvent::Error(t) => Self::Error { text: t.clone() },
            UiEvent::Dim(t) => Self::Dim { text: t.clone() },
            // Remote clients render the card as a pre-styled ANSI text block.
            UiEvent::EditCard(p) => Self::Dim {
                text: p.to_ansi(true),
            },
            UiEvent::Plain(t) => Self::Plain { text: t.clone() },
            UiEvent::UserEcho(t) => Self::UserEcho { text: t.clone() },
            UiEvent::EndLine => Self::EndLine,
            UiEvent::BtwBegin => Self::BtwBegin,
            UiEvent::BtwEnd => Self::BtwEnd,
            UiEvent::MainCheckpoint => Self::MainCheckpoint,
            UiEvent::MainRollback => Self::MainRollback,
            UiEvent::Status(s) => Self::Status {
                status: StatusWire::from(s),
            },
            UiEvent::Tasks(tv) => {
                let (completed, total) = tv.counter().unwrap_or((0, 0));
                Self::Tasks { completed, total }
            }
        }
    }

    /// True for a `status` frame (used by the connection loop's coalescing).
    #[must_use]
    fn is_status(&self) -> bool {
        matches!(self, Self::Status { .. })
    }
}

/// A server → client frame: the versioned envelope wrapping a [`ServerMsg`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServerFrame {
    /// Protocol version.
    pub v: u32,
    /// Sequence id (bus id for mirrored events; `None` for control frames).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub id: Option<u64>,
    /// The message body.
    #[serde(flatten)]
    pub msg: ServerMsg,
}

impl ServerFrame {
    /// A frame with no sequence id (control/handshake frames).
    #[must_use]
    pub fn control(msg: ServerMsg) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id: None,
            msg,
        }
    }

    /// A frame carrying a bus sequence id (mirrored events).
    #[must_use]
    pub fn seq(id: u64, msg: ServerMsg) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id: Some(id),
            msg,
        }
    }

    /// Serializes to a JSON text string.
    ///
    /// # Errors
    /// Returns the `serde_json` error if serialization fails (never expected
    /// for these types).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// Client → server message body, discriminated by `type` (design §4.3).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMsg {
    /// Mandatory first frame.
    Auth {
        /// Shared bearer token.
        token: String,
        /// Optional resume point: replay only events with a greater id.
        #[serde(default)]
        resume_from: Option<u64>,
    },
    /// Submit a prompt (starts a turn or queues while busy).
    Prompt {
        /// Prompt text.
        text: String,
    },
    /// Submit a `/btw` side question (ephemeral; allowed from mirrors).
    Btw {
        /// Question text.
        text: String,
    },
    /// Interrupt the current turn.
    Interrupt,
    /// A `/slash` command (routed through the slash dispatcher; see TODO).
    Command {
        /// Command text including the leading slash.
        text: String,
    },
    /// Ask to become the controller.
    RequestControl,
    /// Give up control.
    ReleaseControl,
    /// Liveness probe; the server replies `pong`.
    Ping,
}

/// A client → server frame: the versioned envelope wrapping a [`ClientMsg`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClientFrame {
    /// Protocol version.
    pub v: u32,
    /// Optional client-assigned id (echoed in errors; currently informational).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub id: Option<u64>,
    /// The message body.
    #[serde(flatten)]
    pub msg: ClientMsg,
}

impl ClientFrame {
    /// A frame wrapping `msg` at the current protocol version.
    #[must_use]
    pub fn new(msg: ClientMsg) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id: None,
            msg,
        }
    }

    /// Serializes to a JSON text string.
    ///
    /// # Errors
    /// Returns the `serde_json` error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Parses a frame from a JSON text string.
    ///
    /// # Errors
    /// Returns the `serde_json` error if the text is not a valid frame.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

// --- Auth -------------------------------------------------------------------

/// Generates a 32-byte, base64url (unpadded) bearer token from the OS CSPRNG.
/// No default token exists; this is called when `--control` is given without one
/// (design §4.6).
#[must_use]
pub fn generate_token() -> String {
    let mut bytes = [0u8; 32];
    if fill_random(&mut bytes).is_err() {
        // Extremely unlikely; fall back to time+addr entropy so we never hand
        // out an all-zero (predictable) token.
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0u128, |d| d.as_nanos());
        // Deliberate byte-wise truncation to spread entropy across the buffer.
        #[allow(clippy::cast_possible_truncation)]
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((t >> (i % 16 * 8)) as u8) ^ (i as u8).wrapping_mul(31);
        }
    }
    base64url(&bytes)
}

fn fill_random(buf: &mut [u8]) -> std::io::Result<()> {
    use std::io::Read;
    let mut f = std::fs::File::open("/dev/urandom")?;
    f.read_exact(buf)
}

/// Base64url encoding without padding (RFC 4648 §5).
fn base64url(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = chunk.get(1).map_or(0, |b| u32::from(*b));
        let b2 = chunk.get(2).map_or(0, |b| u32::from(*b));
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[((n >> 6) & 63) as usize] as char);
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(n & 63) as usize] as char);
        }
    }
    out
}

/// Constant-time byte-slice equality. Unequal lengths short-circuit (token
/// length is fixed and not secret), equal lengths compare without early exit.
#[must_use]
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

// --- Control policy ---------------------------------------------------------

/// Who currently holds control (may submit prompts / interrupts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Holder {
    /// The local TUI/REPL user.
    Local,
    /// A remote session, by session id.
    Remote(u64),
    /// No one holds control (headless, between remote controllers).
    Free,
}

/// Outcome of a remote `request_control`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestOutcome {
    /// Control was granted to the requester.
    Granted,
    /// Denied; carries a reason for a `control_denied` frame.
    Denied(String),
    /// A local user is present and did not pre-authorize transfer: the request
    /// must be surfaced locally and wait for an explicit `/grant`.
    NeedsLocalGrant,
}

/// How long a dropped controller's slot is held for reconnection before it is
/// released to the local user / freed (design §4.8 reconnect grace window).
pub const CONTROL_GRACE: Duration = Duration::from_secs(15);

/// The single-controller coexistence policy (design §4.4): one controller,
/// many mirrors. Pure state machine; the server holds it behind a `Mutex`.
#[derive(Debug)]
pub struct ControlPolicy {
    holder: Holder,
    local_present: bool,
    allow_control: bool,
    /// A controller that dropped and whose slot is reserved until the deadline
    /// for a reconnecting client to reclaim via `resume_from` (design §4.8).
    grace: Option<(u64, std::time::Instant)>,
}

impl ControlPolicy {
    /// New policy. With a local front-end present the local user holds control
    /// initially; headless starts [`Holder::Free`]. `allow_control` lets a
    /// remote take control without an explicit local `/grant`.
    #[must_use]
    pub fn new(local_present: bool, allow_control: bool) -> Self {
        Self {
            holder: if local_present {
                Holder::Local
            } else {
                Holder::Free
            },
            local_present,
            allow_control,
            grace: None,
        }
    }

    /// Current holder.
    #[must_use]
    pub fn holder(&self) -> Holder {
        self.holder
    }

    /// Drops a stale grace reservation once its window has elapsed, releasing
    /// the reserved slot to the local user (if present) or freeing it. Called
    /// at the start of every decision so a lapsed reconnect never blocks.
    fn expire_grace(&mut self) {
        if let Some((session, deadline)) = self.grace
            && std::time::Instant::now() >= deadline
        {
            self.grace = None;
            if self.holder == Holder::Remote(session) {
                self.holder = if self.local_present {
                    Holder::Local
                } else {
                    Holder::Free
                };
            }
        }
    }

    /// Begins the reconnect grace window for a dropping controller: the slot is
    /// held (holder unchanged) until [`CONTROL_GRACE`] elapses, letting a
    /// reconnecting client reclaim it with [`ControlPolicy::reclaim`]. A
    /// non-controller disconnect just releases as before (design §4.8).
    pub fn disconnect(&mut self, session: u64) {
        if self.holder == Holder::Remote(session) {
            self.grace = Some((session, std::time::Instant::now() + CONTROL_GRACE));
        } else {
            self.release(session);
        }
    }

    /// A reconnecting client reclaims a slot still inside its grace window,
    /// transferring control to the new session id. Returns whether control was
    /// reclaimed (design §4.8).
    pub fn reclaim(&mut self, new_session: u64) -> bool {
        self.expire_grace();
        if let Some((prev, _)) = self.grace
            && self.holder == Holder::Remote(prev)
        {
            self.holder = Holder::Remote(new_session);
            self.grace = None;
            return true;
        }
        false
    }

    /// Whether the given remote session may currently submit control frames.
    #[must_use]
    pub fn remote_can_control(&self, session: u64) -> bool {
        self.holder == Holder::Remote(session)
    }

    /// A remote session requests control.
    pub fn request(&mut self, session: u64) -> RequestOutcome {
        self.expire_grace();
        if self.holder == Holder::Remote(session) {
            return RequestOutcome::Granted;
        }
        match self.holder {
            Holder::Remote(_) => RequestOutcome::Denied("another client holds control".to_owned()),
            Holder::Local if self.local_present && !self.allow_control => {
                RequestOutcome::NeedsLocalGrant
            }
            _ => {
                self.holder = Holder::Remote(session);
                RequestOutcome::Granted
            }
        }
    }

    /// The local operator grants control to a remote session (via `/grant`).
    pub fn grant(&mut self, session: u64) {
        self.holder = Holder::Remote(session);
    }

    /// A remote session releases control (explicitly or on disconnect). Control
    /// returns to the local user if present, else becomes free.
    pub fn release(&mut self, session: u64) {
        if self.holder == Holder::Remote(session) {
            self.holder = if self.local_present {
                Holder::Local
            } else {
                Holder::Free
            };
        }
    }
}

// --- Server -----------------------------------------------------------------

/// Shared state a running remote server exposes to its connection threads and
/// to the rest of plank.
#[derive(Debug)]
pub struct RemoteState {
    /// The event fan-out bus the worker broadcasts into.
    pub bus: Arc<BroadcastBus>,
    /// Per-turn shared state (interrupt / queued / btw).
    pub shared: Arc<TurnShared>,
    /// The single-controller policy.
    pub control: Mutex<ControlPolicy>,
    token: String,
    /// Browser `Origin`s allowed on the WS upgrade (besides missing/loopback).
    allowed_origins: Vec<String>,
    /// Per-client outbound queue cap in bytes (slow-consumer eviction).
    queue_max: usize,
    session_ids: AtomicU64,
    shutdown: AtomicBool,
}

impl RemoteState {
    fn next_session(&self) -> u64 {
        self.session_ids.fetch_add(1, Ordering::Relaxed)
    }
}

/// A running remote-control server. Dropping it signals shutdown; call
/// [`RemoteServer::shutdown`] to stop deterministically.
#[derive(Debug)]
pub struct RemoteServer {
    /// Shared state (bus, turn state, control policy).
    pub state: Arc<RemoteState>,
    /// The actual bound address (useful when binding to port 0 in tests).
    pub local_addr: std::net::SocketAddr,
    accept: Option<JoinHandle<()>>,
}

/// Server-side configuration for a [`RemoteServer`]. Groups the bearer token,
/// control-policy seed, and the hardening knobs (`Origin` allow-list and the
/// per-client outbound queue cap) so [`RemoteServer::start`] stays a small,
/// stable signature.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Shared bearer secret every client must present.
    pub token: String,
    /// Whether a local front-end is present (seeds the control policy).
    pub local_present: bool,
    /// Whether a remote may take control without an explicit local `/grant`.
    pub allow_control: bool,
    /// Browser `Origin`s allowed on the WS upgrade (besides missing/loopback).
    pub allowed_origins: Vec<String>,
    /// Per-client outbound queue cap in bytes (slow-consumer eviction).
    pub queue_max: usize,
}

impl RemoteServer {
    /// Binds `addr` (loopback expected) and starts the accept thread. See
    /// [`ServerConfig`] for the auth / policy / hardening knobs.
    ///
    /// # Errors
    /// Returns an error if the address cannot be bound.
    pub fn start(
        addr: &str,
        cfg: ServerConfig,
        bus: Arc<BroadcastBus>,
        shared: Arc<TurnShared>,
    ) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        // A short accept timeout lets the accept loop observe shutdown.
        listener.set_nonblocking(false)?;
        let local_addr = listener.local_addr()?;
        let state = Arc::new(RemoteState {
            bus,
            shared,
            control: Mutex::new(ControlPolicy::new(cfg.local_present, cfg.allow_control)),
            token: cfg.token,
            allowed_origins: cfg.allowed_origins,
            queue_max: cfg.queue_max,
            session_ids: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        });
        let accept_state = Arc::clone(&state);
        let accept = std::thread::Builder::new()
            .name("plank-remote-accept".into())
            .spawn(move || accept_loop(&listener, &accept_state))?;
        Ok(Self {
            state,
            local_addr,
            accept: Some(accept),
        })
    }

    /// Signals shutdown and joins the accept thread.
    pub fn shutdown(&mut self) {
        self.state.shutdown.store(true, Ordering::Relaxed);
        // Nudge the blocking accept() by opening a throwaway connection.
        let _ = TcpStream::connect(self.local_addr);
        if let Some(h) = self.accept.take() {
            let _ = h.join();
        }
    }
}

impl Drop for RemoteServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn accept_loop(listener: &TcpListener, state: &Arc<RemoteState>) {
    for stream in listener.incoming() {
        if state.shutdown.load(Ordering::Relaxed) {
            break;
        }
        let Ok(stream) = stream else { continue };
        let conn_state = Arc::clone(state);
        let _ = std::thread::Builder::new()
            .name("plank-remote-conn".into())
            .spawn(move || {
                if let Err(e) = handle_connection(stream, &conn_state) {
                    // Connection errors are per-client and non-fatal.
                    let _ = e;
                }
            });
    }
}

/// Per-connection handler: reads the HTTP request head, serves the static web
/// client for a plain `GET /`, enforces the `Origin` allow-list on a WebSocket
/// upgrade, then runs auth + the mirror/control loop. Runs on its own thread;
/// a slow or dead client only affects itself.
fn handle_connection(mut stream: TcpStream, state: &Arc<RemoteState>) -> Result<(), String> {
    stream
        .set_read_timeout(Some(POLL_INTERVAL))
        .map_err(|e| e.to_string())?;
    // A bounded write timeout lets the outbound buffer build up (rather than
    // the thread blocking forever) when a client stops reading, so the queue
    // cap can evict it.
    let _ = stream.set_write_timeout(Some(POLL_INTERVAL));

    // Read the request head so we can both serve the static web client and
    // inspect `Origin` before completing any WebSocket upgrade.
    let head = read_http_head(&mut stream)?;
    let req = parse_request_head(&head);

    if !req.is_websocket_upgrade() {
        // Plain HTTP: serve the web client at `/`, else 404.
        serve_http(&mut stream, &req);
        return Ok(());
    }

    // Origin allow-list: missing / loopback Origins pass (native clients send
    // none); other browser Origins must be allow-listed or the upgrade is
    // refused with an HTTP 403 (design §8).
    if !origin_allowed(req.origin.as_deref(), &state.allowed_origins) {
        let _ = write_http_response(
            &mut stream,
            403,
            "Forbidden",
            "text/plain; charset=utf-8",
            b"origin not allowed\n",
        );
        return Ok(());
    }

    // Replay the consumed head bytes so tungstenite can parse the handshake
    // itself, and cap the outbound buffer for per-client backpressure.
    let prefixed = PrefixStream::new(head, stream);
    let config = tungstenite::protocol::WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: state.queue_max.max(1),
        ..Default::default()
    };
    let mut ws =
        tungstenite::accept_with_config(prefixed, Some(config)).map_err(|e| e.to_string())?;

    let Some(session_id) = do_handshake(&mut ws, state)? else {
        return Ok(()); // unauthorized; connection already closed
    };
    let loop_result = mirror_loop(&mut ws, state, session_id);

    // On disconnect (including slow-consumer eviction) a controller's slot is
    // held for the reconnect grace window (§4.8) so a brief drop can reclaim it
    // via `resume_from`; a non-controller just releases.
    if let Ok(mut c) = state.control.lock() {
        c.disconnect(session_id);
    }
    let _ = ws.flush();
    loop_result
}

/// A minimal parsed HTTP request head: method, path, and the header values the
/// server cares about.
struct ParsedRequest {
    method: String,
    path: String,
    origin: Option<String>,
    upgrade_websocket: bool,
}

impl ParsedRequest {
    fn is_websocket_upgrade(&self) -> bool {
        self.upgrade_websocket
    }
}

/// Reads bytes from `stream` until the end of the HTTP header block
/// (`\r\n\r\n`) or a bounded deadline, tolerating read-timeout wake-ups. The
/// returned buffer is the raw head, replayed to tungstenite via [`PrefixStream`]
/// for a WebSocket upgrade.
fn read_http_head(stream: &mut TcpStream) -> Result<Vec<u8>, String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let mut buf = Vec::with_capacity(1024);
    let mut chunk = [0u8; 512];
    loop {
        if std::time::Instant::now() > deadline {
            return Err("timed out reading request head".to_owned());
        }
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        // Guard against an unbounded head.
        if buf.len() > 64 * 1024 {
            return Err("request head too large".to_owned());
        }
        match stream.read(&mut chunk) {
            Ok(0) => break, // EOF
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(buf)
}

/// Parses the request line and the headers we care about from a raw head.
/// Header names are matched case-insensitively (HTTP header names are
/// case-insensitive).
fn parse_request_head(head: &[u8]) -> ParsedRequest {
    let text = String::from_utf8_lossy(head);
    let mut lines = text.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_owned();
    let path = parts.next().unwrap_or("/").to_owned();

    let mut origin = None;
    let mut upgrade_websocket = false;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        match name.as_str() {
            "origin" => origin = Some(value.to_owned()),
            "upgrade" if value.eq_ignore_ascii_case("websocket") => {
                upgrade_websocket = true;
            }
            _ => {}
        }
    }
    ParsedRequest {
        method,
        path,
        origin,
        upgrade_websocket,
    }
}

/// Decides whether a WebSocket upgrade with the given `Origin` is allowed.
/// Default policy (design §8): a missing Origin (native clients send none) and
/// any loopback Origin are allowed; every other browser Origin must appear in
/// the configured allow-list.
#[must_use]
pub fn origin_allowed(origin: Option<&str>, allowed: &[String]) -> bool {
    match origin {
        None => true,
        // Browsers send the literal "null" for opaque origins (e.g. a page
        // opened from a file:// URL); treat it as a non-loopback browser origin
        // that must be explicitly allow-listed.
        Some(o) => is_loopback_origin(o) || allowed.iter().any(|a| a.eq_ignore_ascii_case(o)),
    }
}

/// True if an `Origin` value refers to a loopback host (any scheme / port).
fn is_loopback_origin(origin: &str) -> bool {
    let after_scheme = origin.split_once("://").map_or(origin, |(_, rest)| rest);
    let host_port = after_scheme.split('/').next().unwrap_or(after_scheme);
    // Strip a trailing :port, taking care not to break an IPv6 literal.
    let host = if let Some(stripped) = host_port.strip_prefix('[') {
        stripped.split(']').next().unwrap_or(stripped)
    } else {
        host_port.rsplit_once(':').map_or(host_port, |(h, _)| h)
    };
    matches!(host, "localhost" | "127.0.0.1" | "::1")
}

/// Serves a plain HTTP request: the web client at `/` (or `/index.html`), else
/// a 404. Best-effort; write errors just drop the connection.
fn serve_http(stream: &mut TcpStream, req: &ParsedRequest) {
    let path = req.path.split('?').next().unwrap_or(&req.path);
    if req.method.eq_ignore_ascii_case("GET") && matches!(path, "/" | "/index.html") {
        let _ = write_http_response(
            stream,
            200,
            "OK",
            "text/html; charset=utf-8",
            WEB_CLIENT_HTML.as_bytes(),
        );
    } else {
        let _ = write_http_response(
            stream,
            404,
            "Not Found",
            "text/plain; charset=utf-8",
            b"not found\n",
        );
    }
}

/// Writes a minimal HTTP/1.1 response and closes (no keep-alive).
fn write_http_response(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n",
        len = body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

/// A read/write adapter that replays a consumed byte prefix before delegating
/// to the underlying stream. Lets us peek the HTTP head (for static serving and
/// `Origin` inspection) and still hand the full handshake request to
/// tungstenite.
struct PrefixStream<S> {
    prefix: std::io::Cursor<Vec<u8>>,
    inner: S,
}

impl<S> PrefixStream<S> {
    fn new(prefix: Vec<u8>, inner: S) -> Self {
        Self {
            prefix: std::io::Cursor::new(prefix),
            inner,
        }
    }
}

impl<S: Read> Read for PrefixStream<S> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.prefix.read(buf)?;
        if n > 0 {
            return Ok(n);
        }
        self.inner.read(buf)
    }
}

impl<S: Write> Write for PrefixStream<S> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Authenticates, assigns a session id, and sends the `hello` + `snapshot`
/// frames. Returns the session id on success, or `None` if the client was
/// unauthorized (the connection is closed in that case).
fn do_handshake<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
    state: &Arc<RemoteState>,
) -> Result<Option<u64>, String> {
    let (resume_from, ok) = authenticate(ws, state)?;
    if !ok {
        let _ = ws.close(Some(tungstenite::protocol::CloseFrame {
            code: tungstenite::protocol::frame::coding::CloseCode::Library(4401),
            reason: "unauthorized".into(),
        }));
        let _ = ws.flush();
        return Ok(None);
    }

    let session_id = state.next_session();

    // A client presenting `resume_from` is reconnecting: reclaim the dropped
    // controller's slot if it is still inside its grace window (§4.8). Failing
    // that, headless (no local front-end) auto-requests control for
    // scriptability (design open-question §8, leaning auto-grant).
    let controller = {
        let mut c = state.control.lock().map_err(|e| e.to_string())?;
        if resume_from.is_some() && c.reclaim(session_id) {
            true
        } else if matches!(c.holder(), Holder::Free) {
            matches!(c.request(session_id), RequestOutcome::Granted)
        } else {
            c.remote_can_control(session_id)
        }
    };

    send(
        ws,
        &ServerFrame::control(ServerMsg::Hello {
            protocol_version: PROTOCOL_VERSION,
            plank_version: env!("CARGO_PKG_VERSION").to_owned(),
            session_id,
            controller,
        }),
    )?;

    // Snapshot: replay scrollback tail (or events newer than resume_from).
    let (tail, highest_id) = state.bus.scrollback_since(resume_from);
    let scrollback = tail
        .into_iter()
        .map(|s| ScrollbackEntry {
            id: s.id,
            msg: ServerMsg::from_event(&s.event),
        })
        .collect();
    send(
        ws,
        &ServerFrame::control(ServerMsg::Snapshot {
            scrollback,
            highest_id,
        }),
    )?;
    Ok(Some(session_id))
}

/// The post-handshake loop: pump bus events to the socket and handle inbound
/// control frames until the client disconnects or the server shuts down.
fn mirror_loop<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
    state: &Arc<RemoteState>,
    session_id: u64,
) -> Result<(), String> {
    // Subscribe after the snapshot so no event is missed or duplicated.
    let rx = state.bus.subscribe();
    let mut last_status_at: Option<std::time::Instant> = None;
    loop {
        if state.shutdown.load(Ordering::Relaxed) {
            let _ = send(
                ws,
                &ServerFrame::control(ServerMsg::Bye {
                    reason: "server shutting down".to_owned(),
                }),
            );
            break;
        }
        // Pump bus → socket, coalescing status frames.
        pump_bus(ws, &rx, &mut last_status_at)?;

        // Poll the socket for one inbound frame (times out per POLL_INTERVAL).
        match ws.read() {
            Ok(Message::Text(txt)) => {
                if handle_client_frame(ws, state, session_id, &txt)? {
                    break; // client asked to close / fatal
                }
            }
            Ok(Message::Close(_)) => break,
            Ok(Message::Ping(p)) => {
                let _ = ws.send(Message::Pong(p));
            }
            Ok(_) => {}
            Err(tungstenite::Error::Io(e))
                if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(_) => break,
        }
    }
    Ok(())
}

/// Reads and validates the mandatory `auth` first frame. Returns
/// `(resume_from, authorized)`. A non-`auth` first frame is a policy violation
/// (returned as unauthorized).
fn authenticate<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
    state: &RemoteState,
) -> Result<(Option<u64>, bool), String> {
    // Wait (bounded) for the first text frame.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        if std::time::Instant::now() > deadline {
            return Ok((None, false));
        }
        match ws.read() {
            Ok(Message::Text(txt)) => {
                let Ok(frame) = ClientFrame::from_json(&txt) else {
                    return Ok((None, false));
                };
                return match frame.msg {
                    ClientMsg::Auth { token, resume_from } => {
                        let ok = constant_time_eq(token.as_bytes(), state.token.as_bytes());
                        Ok((resume_from, ok))
                    }
                    // Anything but auth first is a policy violation.
                    _ => Ok((None, false)),
                };
            }
            Ok(Message::Close(_)) => return Ok((None, false)),
            Ok(Message::Ping(p)) => {
                let _ = ws.send(Message::Pong(p));
            }
            Ok(_) => {}
            Err(tungstenite::Error::Io(e))
                if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(e) => return Err(e.to_string()),
        }
    }
}

/// Drains available bus events to the socket. Status frames are coalesced to at
/// most one per [`STATUS_MIN_INTERVAL`] (keep the latest, drop intermediates).
fn pump_bus<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
    rx: &std::sync::mpsc::Receiver<SeqEvent>,
    last_status_at: &mut Option<std::time::Instant>,
) -> Result<(), String> {
    let mut pending_status: Option<SeqEvent> = None;
    while let Ok(seq) = rx.try_recv() {
        let msg = ServerMsg::from_event(&seq.event);
        if msg.is_status() {
            pending_status = Some(seq); // keep only the newest
            continue;
        }
        send(ws, &ServerFrame::seq(seq.id, msg))?;
    }
    if let Some(seq) = pending_status {
        let now = std::time::Instant::now();
        let due = last_status_at.is_none_or(|prev| now.duration_since(prev) >= STATUS_MIN_INTERVAL);
        if due {
            send(
                ws,
                &ServerFrame::seq(seq.id, ServerMsg::from_event(&seq.event)),
            )?;
            *last_status_at = Some(now);
        }
    }
    // Push any buffered data from a prior transient stall; overflow evicts.
    flush_ws(ws)?;
    Ok(())
}

/// Handles one inbound client frame. Returns `Ok(true)` to close the connection.
fn handle_client_frame<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
    state: &Arc<RemoteState>,
    session_id: u64,
    txt: &str,
) -> Result<bool, String> {
    let Ok(frame) = ClientFrame::from_json(txt) else {
        // Ignore unparseable frames rather than dropping the connection.
        return Ok(false);
    };
    match frame.msg {
        // Re-auth mid-session is a no-op (already authenticated).
        ClientMsg::Auth { .. } => {}
        ClientMsg::Ping => send(ws, &ServerFrame::control(ServerMsg::Pong))?,
        // `/btw` is ephemeral and allowed from mirrors (design §4.4/§7).
        ClientMsg::Btw { text } => {
            let _ = state.shared.push_btw(text);
        }
        ClientMsg::Prompt { text } | ClientMsg::Command { text } => {
            if is_controller(state, session_id)? {
                // Both land in the shared queue the turn loop drains: `ui.rs`
                // starts a turn when idle and routes `/`-prefixed lines (the
                // `command` frames) through the slash dispatcher (issue #25).
                state.shared.push_queued(text);
            } else {
                send(
                    ws,
                    &ServerFrame::control(ServerMsg::ControlDenied {
                        reason: "not the controller".to_owned(),
                    }),
                )?;
            }
        }
        ClientMsg::Interrupt => {
            if is_controller(state, session_id)? {
                state.shared.interrupt.store(true, Ordering::Relaxed);
            } else {
                send(
                    ws,
                    &ServerFrame::control(ServerMsg::ControlDenied {
                        reason: "not the controller".to_owned(),
                    }),
                )?;
            }
        }
        ClientMsg::RequestControl => {
            let outcome = state
                .control
                .lock()
                .map_err(|e| e.to_string())?
                .request(session_id);
            match outcome {
                RequestOutcome::Granted => {}
                RequestOutcome::Denied(reason) => {
                    send(
                        ws,
                        &ServerFrame::control(ServerMsg::ControlDenied { reason }),
                    )?;
                }
                RequestOutcome::NeedsLocalGrant => {
                    // Surface the request to the local user; grant happens via
                    // the local `/grant` command (wiring is a ui.rs TODO).
                    state.bus.broadcast(UiEvent::Dim(format!(
                        "[remote session {session_id} wants control — /grant to allow]"
                    )));
                    send(
                        ws,
                        &ServerFrame::control(ServerMsg::ControlDenied {
                            reason: "awaiting local /grant".to_owned(),
                        }),
                    )?;
                }
            }
        }
        ClientMsg::ReleaseControl => {
            state
                .control
                .lock()
                .map_err(|e| e.to_string())?
                .release(session_id);
        }
    }
    Ok(false)
}

fn is_controller(state: &Arc<RemoteState>, session_id: u64) -> Result<bool, String> {
    Ok(state
        .control
        .lock()
        .map_err(|e| e.to_string())?
        .remote_can_control(session_id))
}

/// Error string used to signal slow-consumer eviction: the client's bounded
/// outbound queue overflowed. Returned up through the pump so the connection
/// loop tears down (the bus prunes the dropped subscriber on the next
/// broadcast).
const EVICT_SLOW: &str = "outbound queue full (slow consumer evicted)";

/// True for a transient write stall (buffered, retry later), not a hard error.
fn is_would_block(e: &tungstenite::Error) -> bool {
    matches!(e, tungstenite::Error::Io(io) if matches!(io.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut))
}

/// Queues `frame` on the connection's bounded outbound buffer and tries to
/// flush. A transient write stall keeps the frame buffered (returns `Ok`); an
/// overflow of the configured cap returns [`EVICT_SLOW`].
fn send<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
    frame: &ServerFrame,
) -> Result<(), String> {
    let json = frame.to_json().map_err(|e| e.to_string())?;
    match ws.write(Message::Text(json)) {
        Ok(()) => {}
        Err(tungstenite::Error::WriteBufferFull(_)) => return Err(EVICT_SLOW.to_owned()),
        Err(e) if is_would_block(&e) => return Ok(()), // buffered; flush later
        Err(e) => return Err(e.to_string()),
    }
    flush_ws(ws)
}

/// Flushes buffered outbound data. A transient stall is not an error; a cap
/// overflow evicts.
fn flush_ws<S: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<S>,
) -> Result<(), String> {
    match ws.flush() {
        Ok(()) => Ok(()),
        Err(tungstenite::Error::WriteBufferFull(_)) => Err(EVICT_SLOW.to_owned()),
        Err(e) if is_would_block(&e) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- protocol round-trip ---

    #[test]
    fn frame_roundtrip_all_events() {
        let events = [
            UiEvent::Visible("v".into()),
            UiEvent::Think("t".into()),
            UiEvent::Tool("tool".into()),
            UiEvent::Error("e".into()),
            UiEvent::Dim("d".into()),
            UiEvent::Plain("p".into()),
            UiEvent::UserEcho("u".into()),
            UiEvent::EndLine,
            UiEvent::BtwBegin,
            UiEvent::BtwEnd,
            UiEvent::MainCheckpoint,
            UiEvent::MainRollback,
            UiEvent::Status(Status {
                state: WorkerState::Generating,
                generated: 7,
                ctx_used: 10,
                ctx_size: 100,
                ..Status::default()
            }),
        ];
        for (i, ev) in events.iter().enumerate() {
            let frame = ServerFrame::seq(i as u64, ServerMsg::from_event(ev));
            let json = frame.to_json().unwrap();
            let back: ServerFrame = serde_json::from_str(&json).unwrap();
            assert_eq!(frame, back, "roundtrip for {ev:?}");
            assert_eq!(back.v, PROTOCOL_VERSION);
            assert!(json.contains("\"v\":1"));
        }
    }

    #[test]
    fn client_frame_roundtrip() {
        let msgs = [
            ClientMsg::Auth {
                token: "abc".into(),
                resume_from: Some(5),
            },
            ClientMsg::Prompt { text: "hi".into() },
            ClientMsg::Btw { text: "q".into() },
            ClientMsg::Interrupt,
            ClientMsg::Command {
                text: "/help".into(),
            },
            ClientMsg::RequestControl,
            ClientMsg::ReleaseControl,
            ClientMsg::Ping,
        ];
        for msg in msgs {
            let frame = ClientFrame::new(msg);
            let json = frame.to_json().unwrap();
            assert_eq!(ClientFrame::from_json(&json).unwrap(), frame);
        }
    }

    #[test]
    fn auth_defaults_resume_from_absent() {
        let f = ClientFrame::from_json(r#"{"v":1,"type":"auth","token":"x"}"#).unwrap();
        assert_eq!(
            f.msg,
            ClientMsg::Auth {
                token: "x".into(),
                resume_from: None
            }
        );
    }

    // --- auth primitives ---

    #[test]
    fn constant_time_eq_semantics() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secret2"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn generated_tokens_are_unique_and_urlsafe() {
        let a = generate_token();
        let b = generate_token();
        assert_ne!(a, b);
        assert_eq!(a.len(), 43); // 32 bytes base64url unpadded
        assert!(
            a.bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
        );
    }

    #[test]
    fn base64url_known_vector() {
        // "foobar" → Zm9vYmFy in standard/url base64.
        assert_eq!(base64url(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64url(b"fo"), "Zm8");
    }

    // --- control policy ---

    #[test]
    fn headless_first_requester_becomes_controller() {
        let mut p = ControlPolicy::new(false, false);
        assert_eq!(p.holder(), Holder::Free);
        assert_eq!(p.request(1), RequestOutcome::Granted);
        assert!(p.remote_can_control(1));
        // A second client is denied while the first holds control.
        assert_eq!(
            p.request(2),
            RequestOutcome::Denied("another client holds control".to_owned())
        );
        assert!(!p.remote_can_control(2));
        // Release frees it for the next requester.
        p.release(1);
        assert_eq!(p.holder(), Holder::Free);
        assert_eq!(p.request(2), RequestOutcome::Granted);
    }

    #[test]
    fn local_present_requires_grant() {
        let mut p = ControlPolicy::new(true, false);
        assert_eq!(p.holder(), Holder::Local);
        assert_eq!(p.request(1), RequestOutcome::NeedsLocalGrant);
        assert!(!p.remote_can_control(1));
        // Explicit local grant transfers control.
        p.grant(1);
        assert!(p.remote_can_control(1));
        // Releasing returns control to the local user.
        p.release(1);
        assert_eq!(p.holder(), Holder::Local);
    }

    #[test]
    fn allow_control_lets_remote_take_from_local() {
        let mut p = ControlPolicy::new(true, true);
        assert_eq!(p.request(1), RequestOutcome::Granted);
        assert!(p.remote_can_control(1));
    }

    #[test]
    fn disconnect_holds_slot_and_reconnect_reclaims() {
        let mut p = ControlPolicy::new(false, false);
        assert_eq!(p.request(1), RequestOutcome::Granted);
        // A drop keeps the slot reserved (grace window): another client cannot
        // grab it, but the reconnecting session can reclaim it.
        p.disconnect(1);
        assert_eq!(
            p.request(2),
            RequestOutcome::Denied("another client holds control".to_owned())
        );
        assert!(p.reclaim(3), "reconnect reclaims within grace");
        assert!(p.remote_can_control(3));
        assert!(!p.remote_can_control(2));
    }

    #[test]
    fn expired_grace_frees_the_slot() {
        let mut p = ControlPolicy::new(false, false);
        assert_eq!(p.request(1), RequestOutcome::Granted);
        p.disconnect(1);
        // Force the grace deadline into the past, then a new request succeeds
        // and a stale reclaim fails.
        p.grace = Some((
            1,
            std::time::Instant::now()
                .checked_sub(Duration::from_secs(1))
                .unwrap(),
        ));
        assert_eq!(p.request(2), RequestOutcome::Granted);
        assert!(!p.reclaim(3));
        assert!(p.remote_can_control(2));
    }

    #[test]
    fn non_controller_disconnect_just_releases() {
        let mut p = ControlPolicy::new(true, false);
        p.grant(1);
        // A session that is not the holder disconnecting must not reserve a slot.
        p.disconnect(2);
        assert!(p.remote_can_control(1));
        p.disconnect(1);
        // Holder drop reserves; local user reclaims after expiry.
        p.grace = Some((
            1,
            std::time::Instant::now()
                .checked_sub(Duration::from_secs(1))
                .unwrap(),
        ));
        p.request(2); // triggers expire_grace
        assert_eq!(p.holder(), Holder::Local);
    }

    // --- integration: a real loopback server + tungstenite client ---

    fn test_server(local_present: bool, allow_control: bool) -> RemoteServer {
        test_server_cfg(ServerConfig {
            token: "tok".to_owned(),
            local_present,
            allow_control,
            allowed_origins: Vec::new(),
            queue_max: 1 << 20,
        })
    }

    fn test_server_cfg(cfg: ServerConfig) -> RemoteServer {
        RemoteServer::start(
            "127.0.0.1:0",
            cfg,
            Arc::new(BroadcastBus::new()),
            Arc::new(TurnShared::default()),
        )
        .expect("server binds")
    }

    fn connect(addr: std::net::SocketAddr) -> tungstenite::WebSocket<std::net::TcpStream> {
        let url = format!("ws://{addr}/");
        let stream = TcpStream::connect(addr).unwrap();
        let (ws, _resp) =
            tungstenite::client(url.parse::<tungstenite::http::Uri>().unwrap(), stream)
                .expect("ws handshake");
        ws
    }

    fn send_client(ws: &mut tungstenite::WebSocket<std::net::TcpStream>, msg: ClientMsg) {
        ws.send(Message::Text(ClientFrame::new(msg).to_json().unwrap()))
            .unwrap();
        ws.flush().unwrap();
    }

    fn read_server(ws: &mut tungstenite::WebSocket<std::net::TcpStream>) -> Option<ServerFrame> {
        loop {
            match ws.read() {
                Ok(Message::Text(t)) => return Some(serde_json::from_str(&t).unwrap()),
                Ok(Message::Close(_)) | Err(_) => return None,
                Ok(_) => {}
            }
        }
    }

    #[test]
    fn auth_required_and_hello_snapshot_flow() {
        let server = test_server(false, false);
        let addr = server.local_addr;
        // Seed some scrollback so the snapshot is non-empty.
        server
            .state
            .bus
            .broadcast(UiEvent::Visible("earlier".into()));

        let mut ws = connect(addr);
        send_client(
            &mut ws,
            ClientMsg::Auth {
                token: "tok".into(),
                resume_from: None,
            },
        );
        let hello = read_server(&mut ws).expect("hello");
        match hello.msg {
            ServerMsg::Hello {
                controller,
                protocol_version,
                ..
            } => {
                assert_eq!(protocol_version, PROTOCOL_VERSION);
                assert!(controller, "headless first client auto-controls");
            }
            other => panic!("expected hello, got {other:?}"),
        }
        let snap = read_server(&mut ws).expect("snapshot");
        match snap.msg {
            ServerMsg::Snapshot { scrollback, .. } => {
                assert_eq!(scrollback.len(), 1);
                assert_eq!(
                    scrollback[0].msg,
                    ServerMsg::Visible {
                        text: "earlier".into()
                    }
                );
            }
            other => panic!("expected snapshot, got {other:?}"),
        }
        // A live event after connect is mirrored.
        server.state.bus.broadcast(UiEvent::Visible("live".into()));
        let live = read_server(&mut ws).expect("live frame");
        assert_eq!(
            live.msg,
            ServerMsg::Visible {
                text: "live".into()
            }
        );
    }

    #[test]
    fn auth_rejects_bad_token() {
        let server = test_server(false, false);
        let mut ws = connect(server.local_addr);
        send_client(
            &mut ws,
            ClientMsg::Auth {
                token: "wrong".into(),
                resume_from: None,
            },
        );
        // Server closes the connection without a hello.
        assert!(read_server(&mut ws).is_none());
    }

    #[test]
    fn remote_prompt_queues_and_interrupt_sets_flag() {
        let server = test_server(false, false);
        let mut ws = connect(server.local_addr);
        send_client(
            &mut ws,
            ClientMsg::Auth {
                token: "tok".into(),
                resume_from: None,
            },
        );
        let _ = read_server(&mut ws); // hello
        let _ = read_server(&mut ws); // snapshot

        send_client(
            &mut ws,
            ClientMsg::Prompt {
                text: "do it".into(),
            },
        );
        send_client(&mut ws, ClientMsg::Ping);
        // Wait for pong to guarantee the prompt was processed first.
        assert_eq!(read_server(&mut ws).map(|f| f.msg), Some(ServerMsg::Pong));
        assert_eq!(server.state.shared.take_queued(), vec!["do it"]);

        send_client(&mut ws, ClientMsg::Interrupt);
        send_client(&mut ws, ClientMsg::Ping);
        assert_eq!(read_server(&mut ws).map(|f| f.msg), Some(ServerMsg::Pong));
        assert!(server.state.shared.interrupt.load(Ordering::Relaxed));
    }

    #[test]
    fn non_controller_prompt_is_denied_but_btw_allowed() {
        let server = test_server(false, false);
        let addr = server.local_addr;
        // First client grabs control.
        let mut c1 = connect(addr);
        send_client(
            &mut c1,
            ClientMsg::Auth {
                token: "tok".into(),
                resume_from: None,
            },
        );
        let _ = read_server(&mut c1);
        let _ = read_server(&mut c1);

        // Second client is a mirror.
        let mut c2 = connect(addr);
        send_client(
            &mut c2,
            ClientMsg::Auth {
                token: "tok".into(),
                resume_from: None,
            },
        );
        let _ = read_server(&mut c2);
        let _ = read_server(&mut c2);

        send_client(
            &mut c2,
            ClientMsg::Prompt {
                text: "nope".into(),
            },
        );
        let denied = read_server(&mut c2).expect("denied frame");
        assert!(matches!(denied.msg, ServerMsg::ControlDenied { .. }));

        // But a mirror's /btw is accepted.
        send_client(
            &mut c2,
            ClientMsg::Btw {
                text: "why?".into(),
            },
        );
        send_client(&mut c2, ClientMsg::Ping);
        assert_eq!(read_server(&mut c2).map(|f| f.msg), Some(ServerMsg::Pong));
        assert_eq!(server.state.shared.take_btw(), vec!["why?"]);
    }

    // --- Origin allow-list ---

    #[test]
    fn origin_policy_unit() {
        let allow = vec!["https://app.example.com".to_owned()];
        // Missing Origin (native clients): allowed.
        assert!(origin_allowed(None, &allow));
        // Loopback Origins: always allowed.
        assert!(origin_allowed(Some("http://localhost:31415"), &allow));
        assert!(origin_allowed(Some("http://127.0.0.1"), &allow));
        assert!(origin_allowed(Some("http://[::1]:8080"), &allow));
        // Allow-listed browser Origin (case-insensitive).
        assert!(origin_allowed(Some("https://app.example.com"), &allow));
        assert!(origin_allowed(Some("HTTPS://APP.EXAMPLE.COM"), &allow));
        // A non-allow-listed browser Origin is refused.
        assert!(!origin_allowed(Some("https://evil.example.com"), &allow));
        assert!(!origin_allowed(Some("null"), &allow));
        // With no allow-list, only missing / loopback pass.
        assert!(origin_allowed(Some("http://localhost"), &[]));
        assert!(!origin_allowed(Some("https://app.example.com"), &[]));
    }

    /// Builds a WS handshake request carrying an optional `Origin` and attempts
    /// the upgrade. Returns whether the handshake succeeded.
    fn try_ws_with_origin(addr: std::net::SocketAddr, origin: Option<&str>) -> bool {
        let stream = TcpStream::connect(addr).unwrap();
        let key = tungstenite::handshake::client::generate_key();
        let mut builder = tungstenite::http::Request::builder()
            .uri(format!("ws://{addr}/"))
            .header("Host", addr.to_string())
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header("Sec-WebSocket-Version", "13")
            .header("Sec-WebSocket-Key", key);
        if let Some(o) = origin {
            builder = builder.header("Origin", o);
        }
        let req = builder.body(()).unwrap();
        tungstenite::client(req, stream).is_ok()
    }

    #[test]
    fn origin_allowlist_accepts_and_rejects_on_upgrade() {
        let server = test_server_cfg(ServerConfig {
            token: "tok".to_owned(),
            local_present: false,
            allow_control: false,
            allowed_origins: vec!["https://app.example.com".to_owned()],
            queue_max: 1 << 20,
        });
        let addr = server.local_addr;
        // No Origin (native client): accepted.
        assert!(try_ws_with_origin(addr, None));
        // Loopback Origin: accepted.
        assert!(try_ws_with_origin(addr, Some("http://127.0.0.1:31415")));
        // Allow-listed browser Origin: accepted.
        assert!(try_ws_with_origin(addr, Some("https://app.example.com")));
        // Non-allow-listed browser Origin: rejected (HTTP 403, no upgrade).
        assert!(!try_ws_with_origin(addr, Some("https://evil.example.com")));
    }

    // --- static web client ---

    /// Minimal blocking HTTP GET over a raw socket; returns (status line,
    /// headers-block, body).
    fn http_get(addr: std::net::SocketAddr, path: &str) -> (String, String, String) {
        use std::io::{Read, Write};
        let mut stream = TcpStream::connect(addr).unwrap();
        stream
            .write_all(
                format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n")
                    .as_bytes(),
            )
            .unwrap();
        stream.flush().unwrap();
        let mut raw = String::new();
        stream.read_to_string(&mut raw).unwrap();
        let (head, body) = raw.split_once("\r\n\r\n").unwrap_or((&raw, ""));
        let (status, headers) = head.split_once("\r\n").unwrap_or((head, ""));
        (status.to_owned(), headers.to_owned(), body.to_owned())
    }

    #[test]
    fn serves_web_client_at_root() {
        let server = test_server(false, false);
        let addr = server.local_addr;
        let (status, headers, body) = http_get(addr, "/");
        assert!(status.contains("200"), "status: {status}");
        assert!(
            headers
                .to_ascii_lowercase()
                .contains("content-type: text/html"),
            "headers: {headers}"
        );
        assert!(body.contains("plank remote"), "body missing marker");
        // A bogus path 404s.
        let (status, _, _) = http_get(addr, "/nope");
        assert!(status.contains("404"), "status: {status}");
    }

    // --- bounded outbound queue / slow-consumer eviction ---

    #[test]
    fn slow_client_is_evicted_while_others_keep_receiving() {
        // Small per-client cap so a stalled client trips it quickly.
        let server = test_server_cfg(ServerConfig {
            token: "tok".to_owned(),
            local_present: false,
            allow_control: false,
            allowed_origins: Vec::new(),
            queue_max: 64 * 1024,
        });
        let addr = server.local_addr;

        // Healthy client: drained continuously on a background thread.
        let mut healthy = connect(addr);
        send_client(
            &mut healthy,
            ClientMsg::Auth {
                token: "tok".into(),
                resume_from: None,
            },
        );
        let _ = read_server(&mut healthy); // hello
        let _ = read_server(&mut healthy); // snapshot
        let healthy_count = Arc::new(AtomicU64::new(0));
        let hc = Arc::clone(&healthy_count);
        let drain = std::thread::spawn(move || {
            while let Some(f) = read_server(&mut healthy) {
                if matches!(f.msg, ServerMsg::Visible { .. }) {
                    hc.fetch_add(1, Ordering::Relaxed);
                }
                if matches!(f.msg, ServerMsg::Bye { .. }) {
                    break;
                }
            }
        });

        // Slow client: authenticates, then stops reading entirely.
        let mut slow = connect(addr);
        send_client(
            &mut slow,
            ClientMsg::Auth {
                token: "tok".into(),
                resume_from: None,
            },
        );
        // Do NOT read hello/snapshot: its socket buffer fills and the server's
        // bounded outbound queue for this client overflows.

        // Wait until both connections are registered as subscribers.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while server.state.bus.subscriber_count() < 2 {
            assert!(
                std::time::Instant::now() < deadline,
                "both never subscribed"
            );
            std::thread::sleep(Duration::from_millis(10));
        }

        // Flood enough data to overflow OS buffers plus the 64 KiB cap.
        let big = "x".repeat(2048);
        for _ in 0..2000 {
            server.state.bus.broadcast(UiEvent::Visible(big.clone()));
        }

        // The slow client is evicted: subscriber count drops back to just the
        // healthy one (the bus prunes the dropped subscriber on broadcast).
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            // Keep broadcasting so the prune (which happens on send) fires.
            server.state.bus.broadcast(UiEvent::Visible("tick".into()));
            if server.state.bus.subscriber_count() <= 1 {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "slow client was not evicted"
            );
            std::thread::sleep(Duration::from_millis(20));
        }

        // The healthy client kept receiving throughout.
        assert!(
            healthy_count.load(Ordering::Relaxed) > 0,
            "healthy client received nothing"
        );

        drop(server); // sends Bye, unblocks the drain thread
        let _ = drain.join();
    }
}
