// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The smallest thing that is a plank plugin: it asserts an ABI, echoes, and
//! — on demand — misbehaves, so the host's budget enforcement has something
//! real to stop.

use extism_pdk::*;

/// The mandatory handshake. Returned as decimal text rather than packed bytes:
/// the host parses it before it trusts anything else about this module, so the
/// one export that must never be ambiguous is also the one whose encoding
/// needs no shared header to read.
#[plugin_fn]
pub fn plank_abi() -> FnResult<String> {
    Ok("1".to_string())
}

/// Proves a payload survives the round trip in both directions.
#[plugin_fn]
pub fn echo(input: String) -> FnResult<String> {
    Ok(input)
}

/// Never returns. The host must stop this without taking the session with it —
/// the load-bearing claim behind "a plugin that breaks degrades the feature,
/// never the session".
#[plugin_fn]
pub fn spin() -> FnResult<String> {
    loop {
        std::hint::spin_loop();
    }
}

/// The `command` surface's first export: what slash commands this component
/// contributes. Read once at load, never per keystroke.
#[plugin_fn]
pub fn command_specs() -> FnResult<String> {
    Ok(r#"[{"name": "greet", "args": "<who>", "desc": "say hello from wasm"}]"#.to_string())
}

/// The second: run one. The reply asks the host to print, and — when given an
/// argument — to submit a prompt, so both halves of `CmdOutput` are exercised.
#[plugin_fn]
pub fn command_run(input: String) -> FnResult<String> {
    // The host sends {"name": ..., "args": ...}; this guest only has one
    // command, so it reads the args and ignores the name.
    let args = input
        .split_once("\"args\":")
        .and_then(|(_, rest)| rest.trim().strip_prefix('"'))
        .and_then(|rest| rest.split('"').next())
        .unwrap_or("")
        .to_string();
    if args.is_empty() {
        return Ok(r#"{"print": ["hello from wasm"]}"#.to_string());
    }
    Ok(format!(
        r#"{{"print": ["greeting {args}"], "prompt": "say hello to {args}"}}"#
    ))
}
