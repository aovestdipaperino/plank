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
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::engine::{Engine, EngineEvent, GenerationOptions};
use crate::host::{EngineHost, SessionHandle};
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

/// Registry of in-flight turns to their cancel flags, keyed by session id.
type Cancels = Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>;

/// Runs the server until the process is killed (blocking).
///
/// # Errors
/// Returns a message when the listen address cannot be bound.
pub fn run(engine: Box<dyn Engine>, cfg: &ServeConfig) -> Result<(), String> {
    // Seed the notification enable flag once at server startup so headless
    // `plank serve` honors `ui.notifications`, mirroring `run_interactive`.
    crate::notify::set_enabled(crate::settings::active().ui.notifications);
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
    crate::notify::set_enabled(crate::settings::active().ui.notifications);
    let listener =
        TcpListener::bind(&cfg.listen).map_err(|e| format!("serve: bind {}: {e}", cfg.listen))?;
    eprintln!(
        "plank serve: listening on {} (shared engine, model: {})",
        cfg.listen,
        host.model_name()
    );
    let host = Arc::new(host);
    let sessions: Arc<Mutex<HashMap<String, Arc<SessionHandle>>>> =
        Arc::new(Mutex::new(HashMap::new()));
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

/// Per-session-id handle registry for shared mode.
type Sessions = Arc<Mutex<HashMap<String, Arc<SessionHandle>>>>;

fn handle_conn_shared(
    stream: TcpStream,
    host: &Arc<EngineHost>,
    sessions: &Sessions,
    cancels: &Cancels,
    token: Option<&str>,
) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut out = stream;
    let Some(req) = read_request(&mut reader, token)? else {
        return Ok(());
    };
    if !req.authorized {
        return write_status(&mut out, 401, "unauthorized");
    }

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
            if let Some(flag) = cancels.lock().unwrap().get(id) {
                flag.store(true, Ordering::Relaxed);
            }
            write_status(&mut out, 200, "cancelled")
        }
        _ => write_status(&mut out, 404, "not found"),
    }
}

fn handle_generate_shared(
    req: &Request,
    out: &mut TcpStream,
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

    // Get-or-attach the session for this session_id.
    let handle = {
        let mut map = sessions.lock().unwrap();
        if let Some(h) = map.get(&gen_req.session_id) {
            Arc::clone(h)
        } else {
            // Per-client context sizing (design §7, v2): honor a positive
            // requested `ctx_size` from the client's options, else let the host
            // apply its configured default. The host clamps to the model max.
            let requested = (gen_req.opts.ctx_size > 0).then_some(gen_req.opts.ctx_size);
            match host.attach_sized(requested) {
                Ok(h) => {
                    let h = Arc::new(h);
                    map.insert(gen_req.session_id.clone(), Arc::clone(&h));
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

    let cancel = Arc::new(AtomicBool::new(false));
    cancels
        .lock()
        .unwrap()
        .insert(gen_req.session_id.clone(), Arc::clone(&cancel));

    let opts: GenerationOptions = (&gen_req.opts).into();
    let mut write_err: Option<std::io::Error> = None;
    let result = {
        let mut on_event = |ev: EngineEvent| {
            if write_err.is_some() {
                return;
            }
            let frame = WireEvent::from_engine_event(&ev);
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

    cancels.lock().unwrap().remove(&gen_req.session_id);
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
    authorized: bool,
    body: String,
}

fn read_request(
    reader: &mut BufReader<TcpStream>,
    expected_token: Option<&str>,
) -> std::io::Result<Option<Request>> {
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(None);
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();

    let mut content_length = 0usize;
    let mut auth_header: Option<String> = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        let lower = trimmed.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = trimmed.get("authorization:".len()..)
            && lower.starts_with("authorization:")
        {
            auth_header = Some(v.trim().to_string());
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    let authorized = match expected_token {
        None => true,
        Some(t) => auth_header.as_deref() == Some(format!("Bearer {t}").as_str()),
    };
    Ok(Some(Request {
        method,
        path,
        authorized,
        body: String::from_utf8_lossy(&body).into_owned(),
    }))
}

fn handle_conn(
    stream: TcpStream,
    engine: &Arc<Mutex<Box<dyn Engine>>>,
    cancels: &Cancels,
    token: Option<&str>,
) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut out = stream;
    let Some(req) = read_request(&mut reader, token)? else {
        return Ok(());
    };
    if !req.authorized {
        return write_status(&mut out, 401, "unauthorized");
    }

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
            if let Some(flag) = cancels.lock().unwrap().get(id) {
                flag.store(true, Ordering::Relaxed);
            }
            write_status(&mut out, 200, "cancelled")
        }
        _ => write_status(&mut out, 404, "not found"),
    }
}

fn handle_generate(
    req: &Request,
    out: &mut TcpStream,
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

    let cancel = Arc::new(AtomicBool::new(false));
    cancels
        .lock()
        .unwrap()
        .insert(gen_req.session_id.clone(), Arc::clone(&cancel));

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
            let frame = WireEvent::from_engine_event(&ev);
            if let Err(e) = send_frame(out, &frame) {
                write_err = Some(e);
            }
        };
        // The client owns greedy display state; server samples per opts. Greedy
        // stanza determinism is reproduced by the engine's own streaming parser.
        let greedy = || false;
        if warm {
            eng.warm_system_prompt(&gen_req.transcript, None, &mut on_event)
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

    cancels.lock().unwrap().remove(&gen_req.session_id);
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

fn send_frame(out: &mut TcpStream, frame: &WireEvent) -> std::io::Result<()> {
    let json = serde_json::to_string(frame).unwrap_or_default();
    out.write_all(format!("data: {json}\n\n").as_bytes())?;
    out.flush()
}

fn write_json(out: &mut TcpStream, body: &str) -> std::io::Result<()> {
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    out.write_all(resp.as_bytes())?;
    out.flush()
}

fn write_status(out: &mut TcpStream, code: u16, msg: &str) -> std::io::Result<()> {
    let resp = format!(
        "HTTP/1.1 {code} {msg}\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{msg}",
        msg.len()
    );
    out.write_all(resp.as_bytes())?;
    out.flush()
}
