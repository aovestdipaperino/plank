//! Flavor (a) client: [`RemoteDs4Engine`].
//!
//! A dumb transport over `ureq` for a plank engine hosted by [`crate::serve`].
//! It implements the full [`Engine`] surface by translating each method into an
//! HTTP call:
//!
//! - `generate` → `POST /generate`, then reads the SSE stream, mapping each
//!   frame onto `on_event`, polling `interrupt` between frames and firing
//!   `DELETE /generate/{id}` on interrupt.
//! - `warm_system_prompt` → `POST /warm`; the `checkpoint` path is ignored (it
//!   is a server-side file), progress streams through, returns whether a
//!   prefill happened.
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
use crate::remote::read_sse;

/// Monotonic per-turn id source, so a `DELETE` cancel targets the right stream.
static TURN_SEQ: AtomicU64 = AtomicU64::new(1);

/// HTTP+SSE client engine talking to a `plank serve` host.
#[derive(Debug)]
pub struct RemoteDs4Engine {
    /// Base URL with no trailing slash, e.g. `https://box:8080`.
    base: String,
    /// Optional bearer token sent as `Authorization: Bearer …`.
    token: Option<String>,
    /// Cached model name from `/info`.
    model_name: String,
    /// Cached context size from `/info`.
    ctx_size: i32,
    /// Small token-count memo so repeated `count_tokens` on stable prefixes
    /// avoid a round-trip.
    token_cache: RefCell<HashMap<u64, i32>>,
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
        Ok(Self {
            base,
            token,
            model_name: info.model_name,
            ctx_size: info.ctx_size,
            token_cache: RefCell::new(HashMap::new()),
        })
    }

    /// Drives one streaming endpoint (`/generate` or `/warm`), mapping frames
    /// onto `on_event`. Returns the terminal stats plus whether any `Text`/gen
    /// frame was seen (used by `warm` to report a cache miss).
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
        let mut req = ureq::post(&url).header("Content-Type", "application/json");
        if let Some(t) = &self.token {
            req = req.header("Authorization", format!("Bearer {t}"));
        }
        let mut resp = req
            .send(payload.as_str())
            .map_err(|e| EngineError::new(format!("remote {path}: {e}")))?;

        let mut stats: Option<GenerationStats> = None;
        let mut stream_err: Option<String> = None;
        let reader = resp.body_mut().as_reader();
        // Interrupt is checked before dispatching each frame; on interrupt we
        // stop reading (dropping `resp` closes the connection) and fire DELETE.
        let mut interrupted = false;
        read_sse(reader, |data| {
            if interrupt() {
                interrupted = true;
                return false;
            }
            match serde_json::from_str::<WireEvent>(data) {
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
            }
        })
        .map_err(|e| EngineError::new(format!("remote stream read: {e}")))?;

        if interrupted {
            drop(resp);
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
        let req = match &self.token {
            Some(t) => ureq::delete(&url).header("Authorization", &format!("Bearer {t}")),
            None => ureq::delete(&url),
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
        transcript: &str,
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
            transcript: transcript.to_string(),
            opts: WireOptions::from(opts),
        };
        self.stream_turn(("/generate", &body), interrupt, on_event)
    }

    fn warm_system_prompt(
        &mut self,
        system: &str,
        _checkpoint: Option<&std::path::Path>,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<bool, EngineError> {
        let session_id = format!("warm-{}", TURN_SEQ.fetch_add(1, Ordering::Relaxed));
        let body = GenerateRequest {
            session_id,
            transcript: system.to_string(),
            opts: WireOptions::from(&GenerationOptions::default()),
        };
        // Never interrupted; warm is a fast prefill.
        let never = || false;
        let stats = self.stream_turn(("/warm", &body), &never, on_event)?;
        // A cache miss prefilled tokens; a hit returns generated == 0.
        Ok(stats.generated > 0 || stats.ctx_used > 0)
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
    fn hash_is_stable_and_distinguishes() {
        assert_eq!(hash_text("abc"), hash_text("abc"));
        assert_ne!(hash_text("abc"), hash_text("abd"));
    }
}
