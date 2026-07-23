// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Warp terminal agent-notification protocol (OSC 777).
//!
//! Warp renders CLI agents in its notification mailbox, tab badges, and
//! toasts when the agent emits `ESC ] 777 ; notify ; warp://cli-agent ;
//! <json> BEL` sequences. The JSON envelope is `{v, agent, event,
//! session_id, cwd, project}` plus event-specific fields. Emission is
//! best-effort and gated on Warp advertising the protocol via the
//! `WARP_CLI_AGENT_PROTOCOL_VERSION` environment variable (with
//! `WARP_CLIENT_VERSION` as a sanity check), so on every other terminal
//! this module is a no-op and nothing leaks into the byte stream.

use std::io::Write;
use std::sync::OnceLock;

/// Protocol version plank speaks.
const PROTOCOL_VERSION: u32 = 1;
/// Agent slug shown to Warp. Warp bundles icons only for agents it knows;
/// unknown slugs still get generic notification treatment.
const AGENT: &str = "plank";

/// True when the hosting terminal is Warp with the CLI-agent protocol
/// enabled. Computed once; env vars cannot change mid-process in a way we
/// care about.
pub fn available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::env::var_os("WARP_CLI_AGENT_PROTOCOL_VERSION").is_some()
            && std::env::var_os("WARP_CLIENT_VERSION").is_some()
    })
}

/// Builds the full escape sequence for `event`. Pure so tests can assert
/// the exact bytes; `cwd`/`project` are passed in rather than read from the
/// environment.
#[must_use]
fn sequence(event: &str, session_id: &str, cwd: &str, project: &str) -> String {
    let body = serde_json::json!({
        "v": PROTOCOL_VERSION,
        "agent": AGENT,
        "event": event,
        "session_id": session_id,
        "cwd": cwd,
        "project": project,
    });
    format!("\x1b]777;notify;warp://cli-agent;{body}\x07")
}

/// Emits `event` to Warp. No-op outside Warp; a write failure is ignored —
/// a lost badge must never affect a turn.
pub fn emit(event: &str, session_id: &str) {
    if !available() {
        return;
    }
    let cwd = std::env::current_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    let project = std::path::Path::new(&cwd)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let seq = sequence(event, session_id, &cwd, &project);
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(seq.as_bytes());
    let _ = out.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_has_osc777_framing_and_envelope() {
        let s = sequence("stop", "sid-1", "/Users/x/proj", "proj");
        assert!(s.starts_with("\x1b]777;notify;warp://cli-agent;{"));
        assert!(s.ends_with('\x07'));
        let json = &s["\x1b]777;notify;warp://cli-agent;".len()..s.len() - 1];
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(v["v"], 1);
        assert_eq!(v["agent"], "plank");
        assert_eq!(v["event"], "stop");
        assert_eq!(v["session_id"], "sid-1");
        assert_eq!(v["cwd"], "/Users/x/proj");
        assert_eq!(v["project"], "proj");
    }

    #[test]
    fn sequence_escapes_json_specials_in_paths() {
        let s = sequence("stop", "", "/tmp/a\"b", "a\"b");
        // The quote must be JSON-escaped so the payload stays one valid object.
        assert!(s.contains(r#"/tmp/a\"b"#));
        let json = &s["\x1b]777;notify;warp://cli-agent;".len()..s.len() - 1];
        assert!(serde_json::from_str::<serde_json::Value>(json).is_ok());
    }
}
