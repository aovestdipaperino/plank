// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! `plank serve` — host a plank engine behind HTTP+SSE (flavor a, issue #26).
//!
//! The server is a thin adapter: it wraps whatever [`Engine`] `make_engine`
//! built (the Metal `Ds4Engine` on a real box, `EchoEngine` elsewhere) and
//! exposes the wire protocol in [`crate::remote::proto`]. All prompt bytes,
//! DSML framing and KV discipline live inside that engine, unchanged — the
//! client (`RemoteDs4Engine`) is a dumb transport.
//!
//! v1 is single-tenant: one shared engine behind a `Mutex`, so generations are
//! serialized (matching the one-user plank workflow). Each TCP connection is
//! handled on its own thread so a `DELETE /generate/{id}` cancel can arrive
//! while a `/generate` stream is in flight.
//!
//! Written on `std::net` only — no async runtime — to match the synchronous
//! `Engine` contract and keep the dependency surface minimal.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::engine::{Engine, EngineEvent, GenerationOptions};
use crate::host::{EngineHost, SessionHandle};
use crate::remote::control::constant_time_eq;
use crate::remote::proto::{
    GenerateRequest, InfoResponse, PROTOCOL_VERSION, SessionStatus, SharedStatus, TokenizeRequest,
    TokenizeResponse, WireEvent, WireStats,
};

/// Options for the `serve` subcommand.
#[derive(Debug, Clone)]
pub struct ServeConfig {
    /// Listen address, e.g. `0.0.0.0:8080`.
    pub listen: String,
    /// Optional bearer token; when set, every request must present it.
    pub token: Option<String>,
}

/// Largest request body the server will read. A transcript for the biggest
/// context plank supports is well under this; anything larger is refused with
/// `413` before a single byte of it is buffered.
pub const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

/// How long a connection may sit idle mid-request before the read is abandoned.
/// A peer that sends a header and then stalls holds a thread, nothing more.
const READ_TIMEOUT: Duration = Duration::from_secs(30);

/// How long a shared-mode host session may go unused before its handle is
/// dropped (detaching it from the host and freeing its KV). A client that comes
/// back later simply re-attaches.
pub const SESSION_IDLE_TTL: Duration = Duration::from_mins(30);

/// Header a `RemoteDs4Engine` sends to identify itself across turns. Its value
/// keys the shared-mode host session and namespaces cancel ids; a request
/// without it (an older client) falls back to keying by `session_id` alone.
pub const CLIENT_ID_HEADER: &str = "x-plank-client-id";

/// Decides whether `listen` may be served without a bearer token.
///
/// A token-less server on a non-loopback address hands the model to anyone who
/// can reach the port, so that combination is refused unless the operator
/// passed `--insecure` explicitly. Loopback binds (`127.0.0.1`, `[::1]`,
/// `localhost`) are always fine. An address that cannot be parsed is treated as
/// exposed, so a typo errs toward refusing.
///
/// # Errors
/// Returns the message to print when startup must be refused.
pub fn check_exposure(listen: &str, has_token: bool, insecure: bool) -> Result<(), String> {
    if has_token || insecure || is_loopback_listen(listen) {
        return Ok(());
    }
    Err(format!(
        "refusing to listen on {listen} without a bearer token: anyone who can reach that \
         address could drive the model. Pass --token <token> (or set PLANK_REMOTE_TOKEN), \
         bind to 127.0.0.1, or pass --insecure to serve unauthenticated anyway"
    ))
}

/// True when `listen` names a loopback interface.
fn is_loopback_listen(listen: &str) -> bool {
    if let Ok(addr) = listen.parse::<std::net::SocketAddr>() {
        return addr.ip().is_loopback();
    }
    // `localhost:PORT` does not parse as a `SocketAddr`; accept the literal
    // host name, and nothing else that fails to parse.
    let host = listen.rsplit_once(':').map_or(listen, |(h, _)| h);
    host.eq_ignore_ascii_case("localhost")
}

/// Registry of in-flight turns to their cancel flags, keyed by
/// [`Request::cancel_key`].
type Cancels = Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>;

/// Runs the server until the process is killed (blocking).
///
/// # Errors
/// Returns a message when the listen address cannot be bound.
pub fn run(engine: Box<dyn Engine>, cfg: &ServeConfig) -> Result<(), String> {
    // Seed the notification enable flag once at server startup so headless
    // `plank serve` honors `ui.notifications`, mirroring `run_interactive`.
    crate::notify::set_mode(crate::settings::active().ui.notifications);
    let listener =
        TcpListener::bind(&cfg.listen).map_err(|e| format!("serve: bind {}: {e}", cfg.listen))?;
    eprintln!(
        "plank serve: listening on {} (model: {})",
        cfg.listen,
        engine.model_name()
    );
    let engine = Arc::new(Mutex::new(engine));
    let cancels: Cancels = Arc::new(Mutex::new(HashMap::new()));
    let token = cfg.token.clone();

    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let engine = Arc::clone(&engine);
        let cancels = Arc::clone(&cancels);
        let token = token.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle_conn(stream, &engine, &cancels, token.as_deref()) {
                eprintln!("plank serve: connection error: {e}");
            }
        });
    }
    Ok(())
}

/// Runs the server in shared-engine mode (issue #28): one [`EngineHost`] backs
/// many per-`session_id` [`SessionHandle`]s, all sharing the single model on the
/// host's one GPU thread. Requests for distinct sessions run concurrently
/// through the cooperative scheduler instead of serializing behind one mutex.
///
/// # Errors
/// Returns a message when the listen address cannot be bound.
pub fn run_shared(host: EngineHost, cfg: &ServeConfig) -> Result<(), String> {
    // Seed the notification enable flag once at server startup so headless
    // `plank serve` honors `ui.notifications`, mirroring `run_interactive`.
    crate::notify::set_mode(crate::settings::active().ui.notifications);
    let listener =
        TcpListener::bind(&cfg.listen).map_err(|e| format!("serve: bind {}: {e}", cfg.listen))?;
    eprintln!(
        "plank serve: listening on {} (shared engine, model: {})",
        cfg.listen,
        host.model_name()
    );
    let host = Arc::new(host);
    let sessions: Sessions = Arc::new(Mutex::new(HashMap::new()));
    let cancels: Cancels = Arc::new(Mutex::new(HashMap::new()));
    let token = cfg.token.clone();

    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let host = Arc::clone(&host);
        let sessions = Arc::clone(&sessions);
        let cancels = Arc::clone(&cancels);
        let token = token.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle_conn_shared(stream, &host, &sessions, &cancels, token.as_deref())
            {
                eprintln!("plank serve: connection error: {e}");
            }
        });
    }
    Ok(())
}

/// One shared-mode host session with the time it was last used, so idle
/// entries can be swept (see [`SESSION_IDLE_TTL`]).
struct SessionEntry {
    handle: Arc<SessionHandle>,
    last_used: Instant,
}

/// Per-client handle registry for shared mode, keyed by
/// [`Request::session_key`].
type Sessions = Arc<Mutex<HashMap<String, SessionEntry>>>;

/// Drops every session unused for longer than `ttl`. A turn in flight holds
/// its own `Arc`, so sweeping its entry only defers the detach to turn end.
fn sweep_idle_sessions(map: &mut HashMap<String, SessionEntry>, ttl: Duration, now: Instant) {
    map.retain(|_, e| now.duration_since(e.last_used) < ttl);
}

fn handle_conn_shared(
    stream: TcpStream,
    host: &Arc<EngineHost>,
    sessions: &Sessions,
    cancels: &Cancels,
    token: Option<&str>,
) -> std::io::Result<()> {
    stream.set_read_timeout(Some(READ_TIMEOUT))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut out = stream;
    let req = match read_request(&mut reader, token)? {
        Parsed::Closed => return Ok(()),
        Parsed::Reject(code, msg) => return write_status(&mut out, code, msg),
        Parsed::Request(req) => req,
    };

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/info") => {
            // Read the scheduler's published accounting snapshot cheaply (design
            // §9 step 5): one mutex read, no GPU-thread contention.
            let st = host.status();
            let info = InfoResponse {
                model_name: host.model_name(),
                ctx_size: host.ctx_size(),
                protocol_version: PROTOCOL_VERSION,
                shared: Some(SharedStatus {
                    live_sessions: st.live_sessions,
                    max_sessions: st.max_sessions,
                    resident_ctx_tokens: st.resident_ctx_tokens,
                    kv_bytes: st.kv_bytes,
                    kv_budget_bytes: st.kv_budget_bytes,
                    sessions: st
                        .sessions
                        .into_iter()
                        .map(|s| SessionStatus {
                            id: s.id,
                            ctx_size: s.ctx_size,
                            ctx_tokens: s.ctx_tokens,
                            reclaimed: s.reclaimed,
                        })
                        .collect(),
                }),
            };
            write_json(&mut out, &serde_json::to_string(&info).unwrap_or_default())
        }
        ("POST", "/tokenize") => {
            let n = serde_json::from_str::<TokenizeRequest>(&req.body)
                .map_or(0, |r| host.count_tokens(&r.text));
            let resp = TokenizeResponse { n_tokens: n };
            write_json(&mut out, &serde_json::to_string(&resp).unwrap_or_default())
        }
        ("POST", "/generate" | "/warm") => {
            handle_generate_shared(&req, &mut out, host, sessions, cancels, req.path == "/warm")
        }
        ("DELETE", path) if path.starts_with("/generate/") => {
            let id = path.trim_start_matches("/generate/");
            if let Some(flag) = cancels.lock().unwrap().get(&req.cancel_key(id)) {
                flag.store(true, Ordering::Relaxed);
            }
            write_status(&mut out, 200, "cancelled")
        }
        _ => write_status(&mut out, 404, "not found"),
    }
}

fn handle_generate_shared<W: Write>(
    req: &Request,
    out: &mut W,
    host: &Arc<EngineHost>,
    sessions: &Sessions,
    cancels: &Cancels,
    warm: bool,
) -> std::io::Result<()> {
    let Ok(gen_req) = serde_json::from_str::<GenerateRequest>(&req.body) else {
        return write_status(out, 400, "bad request");
    };

    // A `/warm` in shared mode is a no-op: the host warms the shared system
    // prompt once at startup and each attach restores it (design §6).
    if warm {
        out.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nCache-Control: no-cache\r\n\r\n",
        )?;
        out.flush()?;
        let terminal = WireEvent::Done {
            stats: WireStats::from(&crate::engine::GenerationStats::default()),
        };
        send_frame(out, &terminal)?;
        return out.flush();
    }

    // Get-or-attach the session for this client. Keyed by the stable client id
    // when the client sends one (so one client keeps one host session across
    // turns), else by the per-turn `session_id` for older clients.
    let session_key = req.session_key(&gen_req.session_id);
    let handle = {
        let mut map = sessions.lock().unwrap();
        let now = Instant::now();
        sweep_idle_sessions(&mut map, SESSION_IDLE_TTL, now);
        if let Some(e) = map.get_mut(&session_key) {
            e.last_used = now;
            Arc::clone(&e.handle)
        } else {
            // Per-client context sizing (design §7, v2): honor a positive
            // requested `ctx_size` from the client's options, else let the host
            // apply its configured default. The host clamps to the model max.
            let requested = (gen_req.opts.ctx_size > 0).then_some(gen_req.opts.ctx_size);
            match host.attach_sized(requested) {
                Ok(h) => {
                    let h = Arc::new(h);
                    map.insert(
                        session_key.clone(),
                        SessionEntry {
                            handle: Arc::clone(&h),
                            last_used: now,
                        },
                    );
                    h
                }
                Err(e) => {
                    drop(map);
                    return write_status(out, 503, &e.to_string());
                }
            }
        }
    };

    out.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nCache-Control: no-cache\r\n\r\n",
    )?;
    out.flush()?;

    let cancel_key = req.cancel_key(&gen_req.session_id);
    let cancel = Arc::new(AtomicBool::new(false));
    cancels
        .lock()
        .unwrap()
        .insert(cancel_key.clone(), Arc::clone(&cancel));

    let opts: GenerationOptions = (&gen_req.opts).into();
    let mut write_err: Option<std::io::Error> = None;
    let result = {
        let mut on_event = |ev: EngineEvent| {
            if write_err.is_some() {
                return;
            }
            let Some(frame) = WireEvent::from_engine_event(&ev) else {
                return;
            };
            if let Err(e) = send_frame(out, &frame) {
                write_err = Some(e);
            }
        };
        handle.generate(
            &gen_req.transcript,
            &opts,
            Arc::clone(&cancel),
            &mut on_event,
        )
    };

    cancels.lock().unwrap().remove(&cancel_key);
    if let Some(e) = sessions.lock().unwrap().get_mut(&session_key) {
        e.last_used = Instant::now();
    }
    if let Some(e) = write_err {
        return Err(e);
    }

    let terminal = match result {
        Ok(stats) => WireEvent::Done {
            stats: WireStats::from(&stats),
        },
        Err(e) => WireEvent::Error {
            message: e.to_string(),
        },
    };
    send_frame(out, &terminal)?;
    out.flush()
}

/// Parsed HTTP request essentials.
struct Request {
    method: String,
    path: String,
    /// The `X-Plank-Client-Id` header, when the client sent one.
    client_id: Option<String>,
    body: String,
}

impl Request {
    /// Key under which this request's host session is stored: the stable
    /// client id when present, else the per-turn `session_id` (older clients).
    fn session_key(&self, session_id: &str) -> String {
        self.client_id
            .clone()
            .unwrap_or_else(|| session_id.to_string())
    }

    /// Key under which an in-flight turn's cancel flag is registered. The client
    /// id namespaces it so two clients' `turn-1` cannot cancel each other.
    fn cancel_key(&self, session_id: &str) -> String {
        match &self.client_id {
            Some(c) => format!("{c}:{session_id}"),
            None => session_id.to_string(),
        }
    }
}

/// Outcome of reading one request off a connection.
enum Parsed {
    /// The peer closed before sending a request line.
    Closed,
    /// The request was refused before its body was read; reply with this
    /// status and close.
    Reject(u16, &'static str),
    /// A complete request that passed the token check.
    Request(Request),
}

/// Reads the request line, headers and body. The bearer token (when one is
/// configured) is checked and the `Content-Length` capped *before* the body is
/// allocated or read, so an unauthenticated peer cannot make the server buffer
/// anything.
fn read_request<R: BufRead>(
    reader: &mut R,
    expected_token: Option<&str>,
) -> std::io::Result<Parsed> {
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(Parsed::Closed);
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();

    let mut content_length = 0usize;
    let mut auth_header: Option<String> = None;
    let mut client_id: Option<String> = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        let Some((name, value)) = trimmed.split_once(':') else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        match name.as_str() {
            "content-length" => content_length = value.parse().unwrap_or(0),
            "authorization" => auth_header = Some(value.to_string()),
            CLIENT_ID_HEADER if !value.is_empty() => client_id = Some(value.to_string()),
            _ => {}
        }
    }
    if let Some(t) = expected_token {
        let expected = format!("Bearer {t}");
        let ok = auth_header
            .as_deref()
            .is_some_and(|got| constant_time_eq(got.as_bytes(), expected.as_bytes()));
        if !ok {
            return Ok(Parsed::Reject(401, "unauthorized"));
        }
    }
    if content_length > MAX_BODY_BYTES {
        return Ok(Parsed::Reject(413, "payload too large"));
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok(Parsed::Request(Request {
        method,
        path,
        client_id,
        body: String::from_utf8_lossy(&body).into_owned(),
    }))
}

fn handle_conn(
    stream: TcpStream,
    engine: &Arc<Mutex<Box<dyn Engine>>>,
    cancels: &Cancels,
    token: Option<&str>,
) -> std::io::Result<()> {
    stream.set_read_timeout(Some(READ_TIMEOUT))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut out = stream;
    let req = match read_request(&mut reader, token)? {
        Parsed::Closed => return Ok(()),
        Parsed::Reject(code, msg) => return write_status(&mut out, code, msg),
        Parsed::Request(req) => req,
    };

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/info") => {
            let eng = engine.lock().unwrap();
            let info = InfoResponse {
                model_name: eng.model_name(),
                ctx_size: eng.ctx_size(),
                protocol_version: PROTOCOL_VERSION,
                // Single-owner serve has no host/scheduler; no shared accounting.
                shared: None,
            };
            write_json(&mut out, &serde_json::to_string(&info).unwrap_or_default())
        }
        ("POST", "/tokenize") => {
            let n = serde_json::from_str::<TokenizeRequest>(&req.body)
                .map_or(0, |r| engine.lock().unwrap().count_tokens(&r.text));
            let resp = TokenizeResponse { n_tokens: n };
            write_json(&mut out, &serde_json::to_string(&resp).unwrap_or_default())
        }
        ("POST", "/generate" | "/warm") => {
            handle_generate(&req, &mut out, engine, cancels, req.path == "/warm")
        }
        ("DELETE", path) if path.starts_with("/generate/") => {
            let id = path.trim_start_matches("/generate/");
            if let Some(flag) = cancels.lock().unwrap().get(&req.cancel_key(id)) {
                flag.store(true, Ordering::Relaxed);
            }
            write_status(&mut out, 200, "cancelled")
        }
        _ => write_status(&mut out, 404, "not found"),
    }
}

fn handle_generate<W: Write>(
    req: &Request,
    out: &mut W,
    engine: &Arc<Mutex<Box<dyn Engine>>>,
    cancels: &Cancels,
    warm: bool,
) -> std::io::Result<()> {
    let Ok(gen_req) = serde_json::from_str::<GenerateRequest>(&req.body) else {
        return write_status(out, 400, "bad request");
    };
    // SSE stream header; the body streams until the connection closes.
    out.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nCache-Control: no-cache\r\n\r\n",
    )?;
    out.flush()?;

    let cancel_key = req.cancel_key(&gen_req.session_id);
    let cancel = Arc::new(AtomicBool::new(false));
    cancels
        .lock()
        .unwrap()
        .insert(cancel_key.clone(), Arc::clone(&cancel));

    let opts: GenerationOptions = (&gen_req.opts).into();
    let interrupt = {
        let cancel = Arc::clone(&cancel);
        move || cancel.load(Ordering::Relaxed)
    };

    // Any socket write failure aborts the turn (client hung up); recorded so we
    // can stop pumping the engine.
    let mut write_err: Option<std::io::Error> = None;
    let result = {
        let mut eng = engine.lock().unwrap();
        let mut on_event = |ev: EngineEvent| {
            if write_err.is_some() {
                return;
            }
            let Some(frame) = WireEvent::from_engine_event(&ev) else {
                return;
            };
            if let Err(e) = send_frame(out, &frame) {
                write_err = Some(e);
            }
        };
        // The client owns greedy display state; server samples per opts. Greedy
        // stanza determinism is reproduced by the engine's own streaming parser.
        let greedy = || false;
        if warm {
            // One append per tier, in order: the client's `warm_append` calls
            // are replayed here rather than concatenated, so the server's token
            // buffer is framed exactly as a local engine's would be.
            eng.warm_reset(&gen_req.transcript)
                .and_then(|()| {
                    gen_req
                        .warm_appends
                        .iter()
                        .try_for_each(|t| eng.warm_append(Some(t)))
                })
                .and_then(|()| eng.warm_sync(&mut on_event))
                .map(|_| crate::engine::GenerationStats::default())
        } else {
            eng.generate(
                crate::engine::Prompt::Flat(&gen_req.transcript),
                &opts,
                &interrupt,
                &greedy,
                &mut on_event,
            )
        }
    };

    cancels.lock().unwrap().remove(&cancel_key);
    if let Some(e) = write_err {
        return Err(e);
    }

    let terminal = match result {
        Ok(stats) => WireEvent::Done {
            stats: WireStats::from(&stats),
        },
        Err(e) => WireEvent::Error {
            message: e.to_string(),
        },
    };
    send_frame(out, &terminal)?;
    out.flush()
}

fn send_frame<W: Write>(out: &mut W, frame: &WireEvent) -> std::io::Result<()> {
    let json = serde_json::to_string(frame).unwrap_or_default();
    out.write_all(format!("data: {json}\n\n").as_bytes())?;
    out.flush()
}

fn write_json<W: Write>(out: &mut W, body: &str) -> std::io::Result<()> {
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    out.write_all(resp.as_bytes())?;
    out.flush()
}

fn write_status<W: Write>(out: &mut W, code: u16, msg: &str) -> std::io::Result<()> {
    let resp = format!(
        "HTTP/1.1 {code} {msg}\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{msg}",
        msg.len()
    );
    out.write_all(resp.as_bytes())?;
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use crate::host::{EchoSharedModel, HostConfig};
    use crate::remote::proto::WireOptions;

    fn parse(raw: &str, token: Option<&str>) -> std::io::Result<Parsed> {
        read_request(
            &mut BufReader::new(Cursor::new(raw.as_bytes().to_vec())),
            token,
        )
    }

    #[test]
    fn exposure_check_allows_loopback_without_token() {
        assert!(check_exposure("127.0.0.1:8080", false, false).is_ok());
        assert!(check_exposure("[::1]:8080", false, false).is_ok());
        assert!(check_exposure("localhost:8080", false, false).is_ok());
    }

    #[test]
    fn exposure_check_refuses_public_bind_without_token() {
        let err = check_exposure("0.0.0.0:8080", false, false).unwrap_err();
        assert!(err.contains("--insecure"), "{err}");
        assert!(check_exposure("[::]:8080", false, false).is_err());
        assert!(check_exposure("192.168.1.10:8080", false, false).is_err());
        // Unparsable addresses are treated as exposed rather than trusted.
        assert!(check_exposure("not-an-address", false, false).is_err());
    }

    #[test]
    fn exposure_check_honors_token_or_insecure() {
        assert!(check_exposure("0.0.0.0:8080", true, false).is_ok());
        assert!(check_exposure("0.0.0.0:8080", false, true).is_ok());
    }

    #[test]
    fn oversized_content_length_is_rejected_before_body_is_read() {
        // No body follows the headers: had the server tried to read (or
        // allocate) `Content-Length` bytes, `read_exact` would fail with EOF
        // instead of producing a clean 413.
        let raw = format!(
            "POST /generate HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY_BYTES + 1
        );
        match parse(&raw, None).unwrap() {
            Parsed::Reject(413, _) => {}
            _ => panic!("expected 413"),
        }
        // Exactly the cap is still accepted (body then read as usual).
        let raw = "POST /x HTTP/1.1\r\nContent-Length: 3\r\n\r\nabc";
        match parse(raw, None).unwrap() {
            Parsed::Request(r) => assert_eq!(r.body, "abc"),
            _ => panic!("expected request"),
        }
    }

    #[test]
    fn wrong_token_is_rejected_before_body_is_read() {
        // Body is declared but absent: a 401 proves the token was checked
        // before any body read was attempted.
        let raw =
            "POST /generate HTTP/1.1\r\nAuthorization: Bearer nope\r\nContent-Length: 10\r\n\r\n";
        match parse(raw, Some("secret")).unwrap() {
            Parsed::Reject(401, _) => {}
            _ => panic!("expected 401"),
        }
        // Missing header entirely is also a 401.
        let raw = "POST /generate HTTP/1.1\r\nContent-Length: 10\r\n\r\n";
        assert!(matches!(
            parse(raw, Some("secret")).unwrap(),
            Parsed::Reject(401, _)
        ));
        // The right token reads the body.
        let raw = "POST /generate HTTP/1.1\r\nAuthorization: Bearer secret\r\nContent-Length: 2\r\n\r\nhi";
        match parse(raw, Some("secret")).unwrap() {
            Parsed::Request(r) => assert_eq!(r.body, "hi"),
            _ => panic!("expected request"),
        }
    }

    #[test]
    fn client_id_header_is_parsed_and_namespaces_keys() {
        let raw = "DELETE /generate/turn-1 HTTP/1.1\r\nX-Plank-Client-Id: c0ffee\r\n\r\n";
        let Parsed::Request(r) = parse(raw, None).unwrap() else {
            panic!("expected request")
        };
        assert_eq!(r.client_id.as_deref(), Some("c0ffee"));
        assert_eq!(r.session_key("turn-1"), "c0ffee");
        assert_eq!(r.cancel_key("turn-1"), "c0ffee:turn-1");

        let raw = "DELETE /generate/turn-1 HTTP/1.1\r\n\r\n";
        let Parsed::Request(r) = parse(raw, None).unwrap() else {
            panic!("expected request")
        };
        assert_eq!(r.client_id, None);
        assert_eq!(r.session_key("turn-1"), "turn-1");
        assert_eq!(r.cancel_key("turn-1"), "turn-1");
    }

    fn gen_request(client_id: Option<&str>, turn: u32) -> Request {
        let body = GenerateRequest {
            session_id: format!("turn-{turn}"),
            transcript: "hello".to_string(),
            opts: WireOptions::from(&GenerationOptions::default()),
            warm_appends: Vec::new(),
        };
        Request {
            method: "POST".to_string(),
            path: "/generate".to_string(),
            client_id: client_id.map(str::to_string),
            body: serde_json::to_string(&body).unwrap(),
        }
    }

    fn echo_host() -> Arc<EngineHost> {
        Arc::new(EngineHost::new(
            Arc::new(EchoSharedModel::new(4096)),
            HostConfig::default(),
        ))
    }

    #[test]
    fn turns_from_one_client_share_one_host_session() {
        let host = echo_host();
        let sessions: Sessions = Arc::new(Mutex::new(HashMap::new()));
        let cancels: Cancels = Arc::new(Mutex::new(HashMap::new()));
        for turn in 1..=5 {
            let req = gen_request(Some("client-a"), turn);
            let mut out = Vec::new();
            handle_generate_shared(&req, &mut out, &host, &sessions, &cancels, false).unwrap();
            let text = String::from_utf8_lossy(&out);
            assert!(text.starts_with("HTTP/1.1 200"), "turn {turn}: {text}");
        }
        assert_eq!(sessions.lock().unwrap().len(), 1);
        assert!(sessions.lock().unwrap().contains_key("client-a"));
        assert!(cancels.lock().unwrap().is_empty());
    }

    #[test]
    fn two_clients_get_two_host_sessions() {
        let host = echo_host();
        let sessions: Sessions = Arc::new(Mutex::new(HashMap::new()));
        let cancels: Cancels = Arc::new(Mutex::new(HashMap::new()));
        for client in ["a", "b"] {
            let req = gen_request(Some(client), 1);
            handle_generate_shared(&req, &mut Vec::new(), &host, &sessions, &cancels, false)
                .unwrap();
        }
        assert_eq!(sessions.lock().unwrap().len(), 2);
    }

    #[test]
    fn legacy_client_without_id_falls_back_to_session_id_key() {
        let host = echo_host();
        let sessions: Sessions = Arc::new(Mutex::new(HashMap::new()));
        let cancels: Cancels = Arc::new(Mutex::new(HashMap::new()));
        for turn in 1..=2 {
            let req = gen_request(None, turn);
            handle_generate_shared(&req, &mut Vec::new(), &host, &sessions, &cancels, false)
                .unwrap();
        }
        let map = sessions.lock().unwrap();
        assert!(map.contains_key("turn-1"));
        assert!(map.contains_key("turn-2"));
    }

    #[test]
    fn idle_sessions_are_swept() {
        let host = echo_host();
        let now = Instant::now();
        let mut map = HashMap::new();
        map.insert(
            "old".to_string(),
            SessionEntry {
                handle: Arc::new(host.attach().unwrap()),
                last_used: now.checked_sub(Duration::from_secs(3600)).unwrap(),
            },
        );
        map.insert(
            "fresh".to_string(),
            SessionEntry {
                handle: Arc::new(host.attach().unwrap()),
                last_used: now,
            },
        );
        sweep_idle_sessions(&mut map, Duration::from_mins(30), now);
        assert_eq!(map.len(), 1);
        assert!(map.contains_key("fresh"));
    }
}
