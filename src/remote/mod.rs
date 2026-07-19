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

use std::io::{BufRead, Read};

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

#[cfg(test)]
mod tests {
    use super::*;

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
