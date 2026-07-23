// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

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
    /// Shared-engine accounting (issue #28, design §9 step 5). All fields are
    /// `#[serde(default)]` so a pre-#28 client parses a newer server and the
    /// single-engine `/info` path (which leaves them zero/empty) round-trips.
    #[serde(default)]
    pub shared: Option<SharedStatus>,
}

/// Live shared-engine status: how many sessions are attached against the cap,
/// and per-session KV/context accounting. Read cheaply off the scheduler thread
/// (design §7, §9 step 5).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SharedStatus {
    /// Number of currently attached sessions (resident + idle-reclaimed).
    pub live_sessions: usize,
    /// Admission cap on concurrently attached sessions (`--max-sessions`).
    pub max_sessions: usize,
    /// Aggregate resident KV, in tokens, summed over non-reclaimed sessions.
    pub resident_ctx_tokens: i64,
    /// Aggregate estimated KV bytes granted across attached sessions (design
    /// §7, v2). `#[serde(default)]` so an older server (no KV accounting) still
    /// round-trips to zero.
    #[serde(default)]
    pub kv_bytes: u64,
    /// Configured aggregate KV-bytes budget, if any (`--kv-budget-bytes`).
    #[serde(default)]
    pub kv_budget_bytes: Option<u64>,
    /// Per-session accounting, one entry per attached session.
    pub sessions: Vec<SessionStatus>,
}

/// Per-session KV/context accounting for the shared engine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionStatus {
    /// Opaque host-assigned session id.
    pub id: u64,
    /// Configured per-session context window, in tokens (design §7, v2).
    #[serde(default)]
    pub ctx_size: i32,
    /// Resident context size, in tokens (0 while reclaimed to disk).
    pub ctx_tokens: i32,
    /// True when this session's KV has been snapshotted to disk and its live
    /// context reclaimed; it restores transparently on the next request.
    pub reclaimed: bool,
}

/// Formats a byte count as a short human-readable string (B/KiB/MiB/GiB).
fn human_bytes(n: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    #[allow(clippy::cast_precision_loss)]
    let bytes = n as f64;
    if bytes >= GIB {
        format!("{:.1} GiB", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes / KIB)
    } else {
        format!("{n} B")
    }
}

impl SharedStatus {
    /// Renders a one-line human-readable summary of the shared-engine state for
    /// a client status line: live/max sessions, aggregate KV usage against any
    /// budget, and how many sessions are currently reclaimed to disk (design §9
    /// step 5).
    #[must_use]
    pub fn status_line(&self) -> String {
        let reclaimed = self.sessions.iter().filter(|s| s.reclaimed).count();
        let kv = match self.kv_budget_bytes {
            Some(budget) => format!(
                "KV {} / {}",
                human_bytes(self.kv_bytes),
                human_bytes(budget)
            ),
            None => format!("KV {}", human_bytes(self.kv_bytes)),
        };
        format!(
            "shared engine: {}/{} sessions, {kv}, {} resident tokens, {reclaimed} reclaimed",
            self.live_sessions, self.max_sessions, self.resident_ctx_tokens
        )
    }
}

impl InfoResponse {
    /// The shared-engine status line, or `None` for a single-owner server (which
    /// sends `shared: None`) — so a client renders it only when present and
    /// degrades gracefully otherwise.
    #[must_use]
    pub fn shared_status_line(&self) -> Option<String> {
        self.shared.as_ref().map(SharedStatus::status_line)
    }
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
            usage: None,
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
            // Text, and (defensively) a warm Notice — the serve path warms with
            // checkpoint=None, so a Notice never actually reaches the wire; if
            // that changes, surfacing it as text is a reasonable fallback.
            EngineEvent::Text(s) | EngineEvent::Notice(s) => Self::Text { s: s.clone() },
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
    fn shared_status_renders_and_absent_degrades() {
        // A shared server: the status line reports sessions, KV vs. budget, and
        // reclaimed count (design §9 step 5).
        let info = InfoResponse {
            model_name: "m".into(),
            ctx_size: 4096,
            protocol_version: PROTOCOL_VERSION,
            shared: Some(SharedStatus {
                live_sessions: 2,
                max_sessions: 8,
                resident_ctx_tokens: 100,
                kv_bytes: 512 * 1024 * 1024,
                kv_budget_bytes: Some(1024 * 1024 * 1024),
                sessions: vec![
                    SessionStatus {
                        id: 1,
                        ctx_size: 4096,
                        ctx_tokens: 100,
                        reclaimed: false,
                    },
                    SessionStatus {
                        id: 2,
                        ctx_size: 2048,
                        ctx_tokens: 0,
                        reclaimed: true,
                    },
                ],
            }),
        };
        let line = info.shared_status_line().expect("shared present renders");
        assert!(line.contains("2/8 sessions"));
        assert!(line.contains("512.0 MiB"));
        assert!(line.contains("1.0 GiB"));
        assert!(line.contains("1 reclaimed"));

        // A single-owner server sends `shared: None`; the client degrades to no
        // status line rather than erroring.
        let single = InfoResponse {
            model_name: "m".into(),
            ctx_size: 4096,
            protocol_version: PROTOCOL_VERSION,
            shared: None,
        };
        assert!(single.shared_status_line().is_none());
    }

    #[test]
    fn info_roundtrips_shared_block_and_old_client_default() {
        // A newer server's shared block survives a JSON round-trip, and a
        // payload with no `shared` field parses to None (pre-#28 compatibility).
        let info = InfoResponse {
            model_name: "m".into(),
            ctx_size: 4096,
            protocol_version: PROTOCOL_VERSION,
            shared: Some(SharedStatus {
                live_sessions: 1,
                max_sessions: 4,
                resident_ctx_tokens: 42,
                kv_bytes: 4096,
                kv_budget_bytes: None,
                sessions: vec![SessionStatus {
                    id: 7,
                    ctx_size: 1024,
                    ctx_tokens: 42,
                    reclaimed: false,
                }],
            }),
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: InfoResponse = serde_json::from_str(&json).unwrap();
        let s = back.shared.expect("shared round-trips");
        assert_eq!(s.kv_bytes, 4096);
        assert_eq!(s.sessions[0].ctx_size, 1024);

        let legacy: InfoResponse =
            serde_json::from_str(r#"{"model_name":"m","ctx_size":4096,"protocol_version":1}"#)
                .unwrap();
        assert!(legacy.shared.is_none());
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
