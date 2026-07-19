//! Wire types for flavor (a) — hosting plank's own ds4 engine behind an HTTP
//! socket (`plank serve` + `RemoteDs4Engine`).
//!
//! These serde types are the shared contract between the server (`serve.rs`,
//! macOS/ds4-only) and the client (`ds4_client.rs`, all platforms). The
//! transcript sent over the wire is the *exact* `render_transcript` byte string
//! plank would feed a local `Ds4Engine`, so byte parity and DSML framing are
//! untouched: the client is a dumb transport, the server is just `Ds4Engine`.

use serde::{Deserialize, Serialize};

use crate::engine::{EngineEvent, GenerationOptions, GenerationStats, PrefillProgress, ThinkMode};

/// Bumped whenever the wire contract changes incompatibly; the client refuses a
/// server whose major version differs.
pub const PROTOCOL_VERSION: u32 = 1;

/// Reply to `GET /info`, cached by the client at construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfoResponse {
    /// Human-readable model name for the status bar.
    pub model_name: String,
    /// Context window size in tokens.
    pub ctx_size: i32,
    /// Server protocol version; see [`PROTOCOL_VERSION`].
    pub protocol_version: u32,
}

/// Serde mirror of [`ThinkMode`] (the engine enum is not itself serializable).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WireThinkMode {
    Auto,
    On,
    Off,
}

impl From<ThinkMode> for WireThinkMode {
    fn from(m: ThinkMode) -> Self {
        match m {
            ThinkMode::Auto => Self::Auto,
            ThinkMode::On => Self::On,
            ThinkMode::Off => Self::Off,
        }
    }
}

impl From<WireThinkMode> for ThinkMode {
    fn from(m: WireThinkMode) -> Self {
        match m {
            WireThinkMode::Auto => Self::Auto,
            WireThinkMode::On => Self::On,
            WireThinkMode::Off => Self::Off,
        }
    }
}

/// Serde mirror of [`GenerationOptions`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireOptions {
    pub n_predict: i32,
    pub ctx_size: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
    pub think_mode: WireThinkMode,
}

impl From<&GenerationOptions> for WireOptions {
    fn from(o: &GenerationOptions) -> Self {
        Self {
            n_predict: o.n_predict,
            ctx_size: o.ctx_size,
            temperature: o.temperature,
            top_p: o.top_p,
            min_p: o.min_p,
            seed: o.seed,
            think_mode: o.think_mode.into(),
        }
    }
}

impl From<&WireOptions> for GenerationOptions {
    fn from(o: &WireOptions) -> Self {
        Self {
            n_predict: o.n_predict,
            ctx_size: o.ctx_size,
            temperature: o.temperature,
            top_p: o.top_p,
            min_p: o.min_p,
            seed: o.seed,
            think_mode: o.think_mode.into(),
        }
    }
}

/// Request body for `POST /generate` and `POST /warm`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// Opaque per-turn id; the server uses it to route a `DELETE /generate/{id}`
    /// cancel. The client generates a fresh id per `generate` call.
    pub session_id: String,
    /// The full rendered transcript (verbatim `render_transcript` bytes).
    pub transcript: String,
    /// Generation options.
    pub opts: WireOptions,
}

/// Request body for `POST /tokenize`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizeRequest {
    pub text: String,
}

/// Reply to `POST /tokenize`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizeResponse {
    pub n_tokens: i32,
}

/// Serde mirror of [`GenerationStats`], carried in the terminal `Done` event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WireStats {
    pub generated: i32,
    pub tps: f64,
    pub ctx_used: i32,
    pub interrupted: bool,
}

impl From<&GenerationStats> for WireStats {
    fn from(s: &GenerationStats) -> Self {
        Self {
            generated: s.generated,
            tps: s.tps,
            ctx_used: s.ctx_used,
            interrupted: s.interrupted,
        }
    }
}

impl From<WireStats> for GenerationStats {
    fn from(s: WireStats) -> Self {
        Self {
            generated: s.generated,
            tps: s.tps,
            ctx_used: s.ctx_used,
            interrupted: s.interrupted,
        }
    }
}

/// One SSE frame streamed from `/generate` (and prefill frames from `/warm`).
///
/// Serialized as a single JSON object per SSE `data:` line, tagged by `type`.
/// The three streaming variants map 1:1 onto [`EngineEvent`] plus a terminal
/// `Done` carrying [`WireStats`]; `Error` surfaces a server-side failure that
/// the client turns into an `EngineError`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum WireEvent {
    Prefill { done: i32, total: i32, tps: f64 },
    Text { s: String },
    Done { stats: WireStats },
    Error { message: String },
}

impl WireEvent {
    /// Maps a streaming [`EngineEvent`] onto its wire frame.
    #[must_use]
    pub fn from_engine_event(ev: &EngineEvent) -> Self {
        match ev {
            EngineEvent::Prefill(p) => Self::Prefill {
                done: p.done,
                total: p.total,
                tps: p.tps,
            },
            EngineEvent::Text(s) => Self::Text { s: s.clone() },
        }
    }

    /// Converts a streaming frame into an [`EngineEvent`]. Returns `None` for
    /// the terminal `Done`/`Error` frames, which the caller handles separately.
    #[must_use]
    pub fn to_engine_event(&self) -> Option<EngineEvent> {
        match self {
            Self::Prefill { done, total, tps } => Some(EngineEvent::Prefill(PrefillProgress {
                done: *done,
                total: *total,
                tps: *tps,
            })),
            Self::Text { s } => Some(EngineEvent::Text(s.clone())),
            Self::Done { .. } | Self::Error { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_roundtrips_through_json() {
        let cases = [
            WireEvent::Prefill {
                done: 3,
                total: 10,
                tps: 12.5,
            },
            WireEvent::Text {
                s: "hello <｜DSML｜tool_calls>".to_string(),
            },
            WireEvent::Done {
                stats: WireStats {
                    generated: 7,
                    tps: 4.0,
                    ctx_used: 42,
                    interrupted: false,
                },
            },
            WireEvent::Error {
                message: "boom".to_string(),
            },
        ];
        for ev in cases {
            let json = serde_json::to_string(&ev).unwrap();
            let back: WireEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(serde_json::to_string(&back).unwrap(), json);
        }
    }

    #[test]
    fn engine_event_mapping_is_lossless() {
        let text = EngineEvent::Text("abc".to_string());
        let wire = WireEvent::from_engine_event(&text);
        match wire.to_engine_event() {
            Some(EngineEvent::Text(s)) => assert_eq!(s, "abc"),
            other => panic!("unexpected: {other:?}"),
        }
        assert!(
            WireEvent::Done {
                stats: WireStats::default()
            }
            .to_engine_event()
            .is_none()
        );
    }

    #[test]
    fn options_roundtrip_preserves_think_mode() {
        let opts = GenerationOptions {
            think_mode: ThinkMode::Off,
            seed: 99,
            ..GenerationOptions::default()
        };
        let wire = WireOptions::from(&opts);
        let back = GenerationOptions::from(&wire);
        assert_eq!(back.think_mode, ThinkMode::Off);
        assert_eq!(back.seed, 99);
    }
}
