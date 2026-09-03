// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Flavor (a) client: [`RemoteDs4Engine`].
//!
//! A dumb transport over `ureq` for a plank engine hosted by [`crate::serve`].
//! It implements the full [`Engine`] surface by translating each method into an
//! HTTP call:
//!
//! - `generate` → `POST /generate`, then reads the SSE stream, mapping each
//!   frame onto `on_event`, polling `interrupt` between frames and firing
//!   `DELETE /generate/{id}` on interrupt.
//! - warming (`warm_reset` / `warm_append` / `warm_sync`) → `POST /warm`. The KV
//!   lives on the server, so the client has nothing of its own to prefill; what
//!   it does have is the tier walk, which only it knows. `warm_reset`/
//!   `warm_append` therefore buffer the tier texts and `warm_sync` ships the
//!   whole buffer in one request, streaming the server's prefill frames back so
//!   the warm-up bar moves for a remote engine as it does for a local one. The
//!   trait defaults (silent no-ops) used to stand in here, which meant the
//!   server prefilled the system prompt lazily inside the first turn instead —
//!   attributing the whole cold prefill to that turn, with no warm-up bar.
//! - `count_tokens` → `POST /tokenize` (short LRU-free cache), degrading to the
//!   trait default (`len()/4`) on transport error so accounting never aborts.
//! - `ctx_size` / `model_name` → cached from the `/info` handshake.
//!
//! DSML, prompt bytes and KV discipline are untouched: the server tokenizes the
//! identical `render_transcript` bytes and streams DSML tool calls back as
//! text, so the existing `viz`/`dsml` pipeline parses them unchanged.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::engine::{Engine, EngineError, EngineEvent, GenerationOptions, GenerationStats};
use crate::remote::proto::{
    GenerateRequest, InfoResponse, PROTOCOL_VERSION, TokenizeRequest, TokenizeResponse, WireEvent,
    WireOptions,
};

/// Monotonic per-turn id source, so a `DELETE` cancel targets the right stream.
static TURN_SEQ: AtomicU64 = AtomicU64::new(1);

/// Header carrying the stable per-engine client id (see [`new_client_id`]).
/// Mirrors `crate::serve::CLIENT_ID_HEADER`, spelled here in wire case.
const CLIENT_ID_HEADER: &str = "X-Plank-Client-Id";

/// Mints a random client id: 16 bytes from the OS CSPRNG as 32 hex chars. The
/// id is generated once per [`RemoteDs4Engine`] and sent on every request so a
/// shared-engine server keeps exactly one host session per client, no matter
/// how many turns run (each turn still has its own `turn-N` cancel id). Falls
/// back to time + pid + a counter should `/dev/urandom` be unavailable, so the
/// id is never a constant that two clients could collide on.
fn new_client_id() -> String {
    use std::fmt::Write as _;
    use std::io::Read as _;
    let mut bytes = [0u8; 16];
    let ok = std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut bytes))
        .is_ok();
    if !ok {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0u128, |d| d.as_nanos());
        let mix = t
            ^ (u128::from(std::process::id()) << 64)
            ^ u128::from(TURN_SEQ.fetch_add(1, Ordering::Relaxed)) << 96;
        bytes = mix.to_le_bytes();
    }
    bytes.iter().fold(String::with_capacity(32), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// HTTP+SSE client engine talking to a `plank serve` host.
#[derive(Debug)]
pub struct RemoteDs4Engine {
    /// Base URL with no trailing slash, e.g. `https://box:8080`.
    base: String,
    /// Optional bearer token sent as `Authorization: Bearer …`.
    token: Option<String>,
    /// Stable random id for this engine instance, sent as `X-Plank-Client-Id`
    /// so the server keys its host session by client rather than by turn.
    client_id: String,
    /// Cached model name from `/info`.
    model_name: String,
    /// Cached context size from `/info`.
    ctx_size: i32,
    /// Small token-count memo so repeated `count_tokens` on stable prefixes
    /// avoid a round-trip.
    token_cache: RefCell<HashMap<u64, i32>>,
    /// The tier walk accumulated by `warm_reset`/`warm_append`, shipped to
    /// `/warm` by `warm_sync`. `.0` is the system tier, `.1` every tier below
    /// it in order.
    warm: (String, Vec<String>),
    /// Whether anything has been buffered since the last `warm_sync`. A second
    /// sync with nothing new must not re-POST: the server would re-run the same
    /// prefill decision and the warm-up bar would replay from zero.
    warm_dirty: bool,
}

impl RemoteDs4Engine {
    /// Connects to `base_url`, performing the `/info` handshake to cache the
    /// model name and context size and to verify the protocol version.
    ///
    /// # Errors
    /// Returns [`EngineError`] when the handshake fails or the server speaks an
    /// incompatible protocol version.
    pub fn connect(base_url: &str, token: Option<String>) -> Result<Self, EngineError> {
        let base = base_url.trim_end_matches('/').to_string();
        let info = fetch_info(&base, token.as_deref())?;
        if info.protocol_version != PROTOCOL_VERSION {
            return Err(EngineError::new(format!(
                "remote plank speaks protocol v{} but this client is v{PROTOCOL_VERSION}; \
                 upgrade the older side",
                info.protocol_version
            )));
        }
        // Surface the shared-engine accounting (issue #28, design §9 step 5)
        // when the server is running one; a single-owner server sends
        // `shared: None` and this is silently skipped.
        if let Some(line) = info.shared_status_line() {
            eprintln!("plank: {line}");
        }
        Ok(Self {
            base,
            token,
            client_id: new_client_id(),
            model_name: info.model_name,
            ctx_size: info.ctx_size,
            token_cache: RefCell::new(HashMap::new()),
            warm: (String::new(), Vec::new()),
            warm_dirty: false,
        })
    }

    /// Drives one streaming endpoint (`/generate` or `/warm`), mapping frames
    /// onto `on_event`. Returns the terminal stats.
    fn stream_turn(
        &mut self,
        path_body: (&str, &GenerateRequest),
        interrupt: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError> {
        let (path, body) = path_body;
        let payload = serde_json::to_string(body)
            .map_err(|e| EngineError::new(format!("serialize request: {e}")))?;
        let url = format!("{}{path}", self.base);
        // Connect and header timeouts, for the same reason as the provider
        // engine: `ureq` defaults every timeout to `None`, so a network drop
        // before the stream starts would park this thread indefinitely. Neither
        // bounds the body — a long generation is the idle timeout's business.
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_connect(Some(std::time::Duration::from_secs(30)))
            .timeout_recv_response(Some(std::time::Duration::from_mins(2)))
            .build()
            .into();
        let mut req = agent
            .post(&url)
            .header("Content-Type", "application/json")
            .header(CLIENT_ID_HEADER, &self.client_id);
        if let Some(t) = &self.token {
            req = req.header("Authorization", format!("Bearer {t}"));
        }
        let resp = req
            .send(payload.as_str())
            .map_err(|e| EngineError::new(format!("remote {path}: {e}")))?;

        let mut stats: Option<GenerationStats> = None;
        let mut stream_err: Option<String> = None;
        // Read on its own thread and consumed through a channel so `interrupt`
        // is polled on a clock, not per frame: a dropped network delivers no
        // frames at all, and the old per-frame check could never fire.
        let rx = crate::remote::spawn_sse_reader(resp.into_body().into_reader());
        let end = crate::remote::pump_sse(
            &rx,
            crate::remote::STREAM_IDLE_TIMEOUT,
            crate::remote::STREAM_POLL_INTERVAL,
            interrupt,
            |data| match serde_json::from_str::<WireEvent>(data) {
                Ok(WireEvent::Done { stats: s }) => {
                    stats = Some(s.into());
                    false
                }
                Ok(WireEvent::Error { message }) => {
                    stream_err = Some(message);
                    false
                }
                Ok(ev) => {
                    if let Some(engine_ev) = ev.to_engine_event() {
                        on_event(engine_ev);
                    }
                    true
                }
                Err(e) => {
                    stream_err = Some(format!("malformed server frame: {e}"));
                    false
                }
            },
        )
        .map_err(EngineError::new)?;

        if end == crate::remote::SseEnd::Interrupted {
            // The DELETE is what actually cancels; the reader thread ends when
            // the server closes the stream in response.
            self.cancel(&body.session_id);
            return Ok(GenerationStats {
                interrupted: true,
                ..GenerationStats::default()
            });
        }
        if let Some(msg) = stream_err {
            return Err(EngineError::new(msg));
        }
        stats.ok_or_else(|| EngineError::new("remote stream ended without a Done frame"))
    }

    /// Best-effort cancel of an in-flight turn.
    fn cancel(&self, session_id: &str) {
        let url = format!("{}/generate/{session_id}", self.base);
        // The server namespaces cancel ids by client id, so the DELETE must
        // carry the same header the `/generate` did.
        let req = ureq::delete(&url).header(CLIENT_ID_HEADER, &self.client_id);
        let req = match &self.token {
            Some(t) => req.header("Authorization", &format!("Bearer {t}")),
            None => req,
        };
        // Fire and forget: a failed cancel is not fatal (the dropped connection
        // already signals abandonment to a well-behaved server).
        let _ = req.call();
    }
}

/// Performs the `/info` handshake.
fn fetch_info(base: &str, token: Option<&str>) -> Result<InfoResponse, EngineError> {
    let url = format!("{base}/info");
    let req = match token {
        Some(t) => ureq::get(&url).header("Authorization", &format!("Bearer {t}")),
        None => ureq::get(&url),
    };
    let mut resp = req
        .call()
        .map_err(|e| EngineError::new(format!("remote /info: {e}")))?;
    let text = resp
        .body_mut()
        .read_to_string()
        .map_err(|e| EngineError::new(format!("remote /info body: {e}")))?;
    serde_json::from_str(&text).map_err(|e| EngineError::new(format!("remote /info parse: {e}")))
}

/// FNV-1a over the text, keying the token-count memo without storing the text.
fn hash_text(text: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl Engine for RemoteDs4Engine {
    fn generate(
        &mut self,
        prompt: crate::engine::Prompt<'_>,
        opts: &GenerationOptions,
        interrupt: &dyn Fn() -> bool,
        // The server owns greedy state (it runs the same streaming parser over
        // its own output), so the client sends no greedy hint — see design §4.1.
        _greedy: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError> {
        let session_id = format!("turn-{}", TURN_SEQ.fetch_add(1, Ordering::Relaxed));
        let body = GenerateRequest {
            session_id,
            transcript: prompt.flat().to_string(),
            opts: WireOptions::from(opts),
            warm_appends: Vec::new(),
        };
        self.stream_turn(("/generate", &body), interrupt, on_event)
    }

    fn warm_reset(&mut self, system: &str) -> Result<(), EngineError> {
        self.warm = (system.to_string(), Vec::new());
        self.warm_dirty = true;
        Ok(())
    }

    fn warm_append(&mut self, text: Option<&str>) -> Result<(), EngineError> {
        if let Some(text) = text {
            self.warm.1.push(text.to_string());
            self.warm_dirty = true;
        }
        Ok(())
    }

    fn warm_sync(&mut self, on_event: &mut dyn FnMut(EngineEvent)) -> Result<bool, EngineError> {
        if !self.warm_dirty || self.warm.0.is_empty() {
            return Ok(false);
        }
        let body = GenerateRequest {
            session_id: format!("warm-{}", TURN_SEQ.fetch_add(1, Ordering::Relaxed)),
            transcript: self.warm.0.clone(),
            // Warming is not a generation; the server ignores these, but the
            // field is not optional on the wire.
            opts: WireOptions::from(&GenerationOptions::default()),
            warm_appends: self.warm.1.clone(),
        };
        // A prefill frame is the server's own report that it really prefilled —
        // a cache hit streams none — so it is what the `bool` is derived from,
        // rather than a second wire field that could disagree with it.
        let mut prefilled = false;
        // Warming is uninterruptible by the trait's contract, matching the
        // local engine's `interrupt: &|| false`.
        self.stream_turn(("/warm", &body), &|| false, &mut |ev| {
            if matches!(ev, EngineEvent::Prefill(_)) {
                prefilled = true;
            }
            on_event(ev);
        })?;
        self.warm_dirty = false;
        Ok(prefilled)
    }

    fn count_tokens(&self, text: &str) -> i32 {
        let key = hash_text(text);
        if let Some(n) = self.token_cache.borrow().get(&key) {
            return *n;
        }
        let url = format!("{}/tokenize", self.base);
        let mut req = ureq::post(&url).header("Content-Type", "application/json");
        if let Some(t) = &self.token {
            req = req.header("Authorization", format!("Bearer {t}"));
        }
        let fallback = i32::try_from(text.len() / 4).unwrap_or(i32::MAX);
        let Ok(payload) = serde_json::to_string(&TokenizeRequest {
            text: text.to_string(),
        }) else {
            return fallback;
        };
        // Degrade rather than fail (design constraint 8): any transport or parse
        // error falls back to the ~4-bytes-per-token estimate.
        let n = req
            .send(payload.as_str())
            .ok()
            .and_then(|mut r| r.body_mut().read_to_string().ok())
            .and_then(|t| serde_json::from_str::<TokenizeResponse>(&t).ok())
            .map_or(fallback, |r| r.n_tokens);
        self.token_cache.borrow_mut().insert(key, n);
        n
    }

    fn ctx_size(&self) -> i32 {
        self.ctx_size
    }

    fn model_name(&self) -> String {
        self.model_name.clone()
    }
}

/// Idle-read timeout suggestion for callers that build their own agent. Kept
/// here to document intent; the default `ureq` calls above use the library
/// default (no global timeout, so long generations are not cut off).
#[allow(dead_code)]
pub const SUGGESTED_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_ids_are_hex_and_distinct() {
        let a = new_client_id();
        let b = new_client_id();
        assert_eq!(a.len(), 32);
        assert!(a.bytes().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
    }

    #[test]
    fn hash_is_stable_and_distinguishes() {
        assert_eq!(hash_text("abc"), hash_text("abc"));
        assert_ne!(hash_text("abc"), hash_text("abd"));
    }
}
