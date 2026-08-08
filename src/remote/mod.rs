// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Remote engines (issue #26, `docs/REMOTE-ENGINE-DESIGN.md`).
//!
//! Two independent `Engine` implementations share this module:
//!
//! - **Flavor (a)** — [`ds4_client::RemoteDs4Engine`], a thin sync HTTP+SSE
//!   client for plank's own engine hosted by [`crate::serve`]. The transport is
//!   the already-vendored blocking `ureq` client, which matches the *synchronous*
//!   `Engine::generate` contract directly: SSE frames arrive per token, and the
//!   `interrupt` closure is polled between frames. No async runtime is needed.
//! - **Flavor (b)** — [`provider::ProviderEngine`], an adapter for third-party
//!   LLM APIs (OpenAI-compatible in v1; Anthropic reserved). It reads the
//!   structured-input boundary from §4.4 ([`crate::engine::Prompt::Structured`],
//!   [`crate::sysprompt::provider_system_prompt`], and the tool registry) and
//!   re-emits native tool calls as synthesized DSML, so the dispatch/renderer
//!   stack stays backend-agnostic. It reuses the same blocking `ureq` transport
//!   and [`read_sse`] reader as flavor (a); no async runtime is pulled in
//!   (`refs/llms-sdk` is the wire-format reference, not a runtime dependency —
//!   depending on it would force `tokio`/`reqwest`, which the sync `Engine`
//!   contract does not need).
//!
//! This module holds the pieces both flavors share: URL validation and the SSE
//! frame reader.

pub mod proto;

// Available on every platform: pure Rust + HTTP, no C engine needed. A Linux or
// Windows user can drive a remote ds4 box without building the Metal engine.
pub mod ds4_client;

// Flavor (b): third-party provider engine (OpenAI-compatible / Anthropic).
pub mod provider;

// Remote-control server (issue #25): the loopback WebSocket mirror/drive
// interface. Formerly the standalone `src/remote.rs`; folded in here so #25 and
// #26 share the one `remote` module. Re-exported at the module root so callers
// keep using `crate::remote::{RemoteServer, generate_token}`.
pub mod control;
pub use control::{RemoteServer, generate_token};

/// Bind address for a `/remote-control` activation: loopback, ephemeral port.
/// A fixed port would collide with another plank instance or a stale listener;
/// the real port is read back from `RemoteServer::local_addr`.
pub const LOOPBACK_EPHEMERAL: &str = "127.0.0.1:0";

// Remote-control CLI client (issue #25): `plank remote <url>` connects to a
// running instance's control WebSocket, mirrors its output, and drives it.
pub mod client;

use std::io::{BufRead, Read};
use std::time::Duration;

/// Validates a `--remote` URL: TLS is required for any non-loopback host
/// (design §4.7 / constraint 6). `insecure` waives that for `http://` to a
/// loopback address only.
///
/// # Errors
/// Returns a message when the scheme is unsupported or when plaintext HTTP is
/// used against a non-loopback host without `--insecure`.
pub fn validate_remote_url(url: &str, insecure: bool) -> Result<(), String> {
    let (scheme, rest) = url
        .split_once("://")
        .ok_or_else(|| format!("invalid remote URL (no scheme): {url}"))?;
    match scheme {
        "https" => Ok(()),
        "http" => {
            let host = rest.split(['/', ':']).next().unwrap_or("");
            let is_loopback = host == "localhost"
                || host == "127.0.0.1"
                || host == "::1"
                || host.starts_with("127.");
            if is_loopback || insecure {
                Ok(())
            } else {
                Err(format!(
                    "refusing plaintext http:// to non-loopback host {host}; use https:// \
                     or pass --insecure for a trusted localhost tunnel"
                ))
            }
        }
        other => Err(format!("unsupported remote URL scheme: {other}")),
    }
}

/// Reads Server-Sent-Event `data:` payloads from `reader`, invoking `on_data`
/// with each complete event's concatenated data lines.
///
/// SSE framing: lines starting with `data:` accumulate (one event may span
/// several `data:` lines); a blank line dispatches the accumulated event.
/// `on_data` returns `false` to stop early (e.g. terminal frame seen or the
/// caller was interrupted), which ends the loop without draining the rest.
///
/// # Errors
/// Propagates the first read error from the underlying stream.
pub fn read_sse<R: Read>(reader: R, mut on_data: impl FnMut(&str) -> bool) -> std::io::Result<()> {
    let buf = std::io::BufReader::new(reader);
    let mut data = String::new();
    for line in buf.lines() {
        let line = line?;
        if let Some(payload) = line.strip_prefix("data:") {
            // A single leading space after the colon is part of the framing.
            data.push_str(payload.strip_prefix(' ').unwrap_or(payload));
        } else if line.is_empty() && !data.is_empty() {
            let keep_going = on_data(&data);
            data.clear();
            if !keep_going {
                return Ok(());
            }
        }
        // Other SSE fields (event:, id:, comments starting ':') are ignored.
    }
    // Flush a trailing event that was not terminated by a blank line.
    if !data.is_empty() {
        on_data(&data);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Stall-proof SSE consumption
// ---------------------------------------------------------------------------

/// One item from a [`spawn_sse_reader`] thread.
#[derive(Debug)]
pub enum SseItem {
    /// A complete SSE event's concatenated `data:` payload.
    Data(String),
    /// The reader finished; carries its final result (a clean EOF or the read
    /// error that ended it).
    End(std::io::Result<()>),
}

/// How a [`pump_sse`] run ended, short of an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SseEnd {
    /// The stream ended on its own, or `on_data` asked to stop.
    Finished,
    /// The `interrupt` closure returned true.
    Interrupted,
}

/// How long a provider stream may go silent before it is declared dead.
///
/// Both providers keepalive their SSE streams — Anthropic emits `event: ping`,
/// `OpenAI` emits comment frames — so a healthy generation is never quiet for
/// anything like this long. Silence past it means the connection is gone.
///
/// This is deliberately an *idle* timeout rather than a cap on the whole
/// response: `ureq`'s own `timeout_recv_body` bounds the total body duration,
/// which cannot distinguish a dead socket from a long-but-healthy generation,
/// so any value safe for the latter is useless against the former.
pub const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(90);

/// How often [`pump_sse`] wakes to poll `interrupt` while no data is arriving.
pub const STREAM_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Reads SSE events off `reader` on a dedicated thread, forwarding each
/// payload down the returned channel.
///
/// The read itself is what blocks — a sudden network drop leaves the socket
/// established but black-holed, with no unacked data for TCP to retransmit and
/// therefore no kernel timeout to save us, so the read can park forever. That
/// is survivable only if it parks somewhere the turn does not need: this
/// thread. It ends when the socket eventually errors, or leaks harmlessly for
/// the life of the process. [`pump_sse`] on the caller's side stays responsive
/// either way.
pub fn spawn_sse_reader<R: Read + Send + 'static>(reader: R) -> std::sync::mpsc::Receiver<SseItem> {
    // Bounded so a fast stream cannot outrun the consumer into unbounded
    // memory; 64 events is far more slack than the renderer ever needs.
    let (tx, rx) = std::sync::mpsc::sync_channel(64);
    std::thread::spawn(move || {
        let res = read_sse(reader, |data| {
            tx.send(SseItem::Data(data.to_string())).is_ok()
        });
        // Best-effort: the receiver is already gone if the turn was cancelled.
        let _ = tx.send(SseItem::End(res));
    });
    rx
}

/// Drives a [`spawn_sse_reader`] channel, feeding each payload to `on_data`.
///
/// The point of this loop is that it polls `interrupt` on a *clock* rather
/// than on data arrival: `recv_timeout` wakes every `poll` whether or not the
/// provider has sent anything, so Ctrl-C is honoured within one poll interval
/// even when the stream is completely dead. Silence past `idle_timeout` is
/// reported as an error so the turn ends with a diagnosis instead of hanging.
///
/// `on_data` returns false to stop early (a terminal frame), which ends the
/// pump as [`SseEnd::Finished`].
///
/// # Errors
/// Returns the reader's own read error, or a "stalled" message when nothing
/// arrives for `idle_timeout`.
pub fn pump_sse(
    rx: &std::sync::mpsc::Receiver<SseItem>,
    idle_timeout: Duration,
    poll: Duration,
    interrupt: &dyn Fn() -> bool,
    mut on_data: impl FnMut(&str) -> bool,
) -> Result<SseEnd, String> {
    use std::sync::mpsc::RecvTimeoutError;
    let mut last = std::time::Instant::now();
    loop {
        // Checked before each blocking wait as well as after a timeout, so an
        // interrupt raised while events are still flowing is honoured at the
        // same boundary the old per-event check used.
        if interrupt() {
            return Ok(SseEnd::Interrupted);
        }
        match rx.recv_timeout(poll) {
            Ok(SseItem::Data(data)) => {
                last = std::time::Instant::now();
                if !on_data(&data) {
                    return Ok(SseEnd::Finished);
                }
            }
            Ok(SseItem::End(res)) => {
                return res
                    .map(|()| SseEnd::Finished)
                    .map_err(|e| format!("provider stream read: {e}"));
            }
            // The reader thread is gone without an `End` — it panicked, or the
            // channel was dropped. Treat it as a clean end: whatever arrived is
            // what there is.
            Err(RecvTimeoutError::Disconnected) => return Ok(SseEnd::Finished),
            Err(RecvTimeoutError::Timeout) => {
                if last.elapsed() >= idle_timeout {
                    return Err(format!(
                        "provider stream stalled: no data for {}s (network dropped?)",
                        idle_timeout.as_secs()
                    ));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A silent stream must not swallow the user's interrupt. This is the
    /// regression the freeze-on-network-drop bug came down to: the old reader
    /// polled `interrupt` only when an SSE event arrived, so a black-holed
    /// socket — which delivers nothing, ever — could never be cancelled.
    #[test]
    fn a_silent_stream_still_observes_the_interrupt() {
        let (_tx, rx) = std::sync::mpsc::channel::<SseItem>();
        let end = pump_sse(
            &rx,
            Duration::from_mins(1),
            Duration::from_millis(5),
            &|| true,
            |_| true,
        )
        .expect("an interrupt is not an error");
        assert_eq!(end, SseEnd::Interrupted);
    }

    /// Silence past the idle timeout is a fault, not patience: both providers
    /// keepalive their streams, so a gap this long means the connection is
    /// gone even though the socket still looks open.
    #[test]
    fn a_stalled_stream_becomes_an_error_rather_than_hanging() {
        let (_tx, rx) = std::sync::mpsc::channel::<SseItem>();
        let err = pump_sse(
            &rx,
            Duration::from_millis(20),
            Duration::from_millis(5),
            &|| false,
            |_| true,
        )
        .expect_err("a stalled stream must not report success");
        assert!(err.contains("stalled"), "unhelpful message: {err}");
    }

    /// The happy path still delivers every payload in order and reports the
    /// reader's own clean end.
    #[test]
    fn pump_forwards_every_payload_then_finishes() {
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(SseItem::Data("one".to_string())).unwrap();
        tx.send(SseItem::Data("two".to_string())).unwrap();
        tx.send(SseItem::End(Ok(()))).unwrap();
        let mut got = Vec::new();
        let end = pump_sse(
            &rx,
            Duration::from_mins(1),
            Duration::from_millis(5),
            &|| false,
            |d| {
                got.push(d.to_string());
                true
            },
        )
        .unwrap();
        assert_eq!(got, vec!["one", "two"]);
        assert_eq!(end, SseEnd::Finished);
    }

    /// A terminal frame (`on_data` returning false) ends the pump normally —
    /// it is the stream saying it is done, not an interrupt and not a fault.
    #[test]
    fn a_terminal_frame_finishes_without_draining_the_rest() {
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(SseItem::Data("last".to_string())).unwrap();
        tx.send(SseItem::Data("never read".to_string())).unwrap();
        let mut seen = 0;
        let end = pump_sse(
            &rx,
            Duration::from_mins(1),
            Duration::from_millis(5),
            &|| false,
            |_| {
                seen += 1;
                false
            },
        )
        .unwrap();
        assert_eq!(seen, 1, "the pump kept reading past the terminal frame");
        assert_eq!(end, SseEnd::Finished);
    }

    /// A read error from the reader thread surfaces as an error rather than a
    /// silently truncated (and therefore plausible-looking) generation.
    #[test]
    fn a_reader_error_is_propagated() {
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(SseItem::End(Err(std::io::Error::other("connection reset"))))
            .unwrap();
        let err = pump_sse(
            &rx,
            Duration::from_mins(1),
            Duration::from_millis(5),
            &|| false,
            |_| true,
        )
        .expect_err("a read error must not look like a clean finish");
        assert!(err.contains("connection reset"), "lost the cause: {err}");
    }

    #[test]
    fn url_validation_rules() {
        assert!(validate_remote_url("https://box.example.com:8080", false).is_ok());
        assert!(validate_remote_url("http://localhost:9000", false).is_ok());
        assert!(validate_remote_url("http://127.0.0.1:9000/generate", false).is_ok());
        assert!(validate_remote_url("http://box.example.com", false).is_err());
        assert!(validate_remote_url("http://box.example.com", true).is_ok());
        assert!(validate_remote_url("ftp://host", false).is_err());
        assert!(validate_remote_url("no-scheme", false).is_err());
    }

    #[test]
    fn sse_reader_splits_events_and_joins_data_lines() {
        let raw = "data: one\n\ndata: two-a\ndata: two-b\n\ndata: three\n\n";
        let mut got = Vec::new();
        read_sse(raw.as_bytes(), |d| {
            got.push(d.to_string());
            true
        })
        .unwrap();
        assert_eq!(got, vec!["one", "two-atwo-b", "three"]);
    }

    #[test]
    fn sse_reader_stops_when_callback_returns_false() {
        let raw = "data: one\n\ndata: two\n\ndata: three\n\n";
        let mut got = Vec::new();
        read_sse(raw.as_bytes(), |d| {
            got.push(d.to_string());
            d != "two"
        })
        .unwrap();
        assert_eq!(got, vec!["one", "two"]);
    }
}
