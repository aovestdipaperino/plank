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

// The capability imports. Declared unconditionally: plank provides every host
// function to every component and checks the grant when called, so importing
// one this component was not granted is not a load failure — the call simply
// comes back with a refusal.
#[host_fn]
extern "ExtismHost" {
    fn plank_print(text: String) -> String;
    fn plank_state_get(key: String) -> Vec<u8>;
    fn plank_state_set(key: String, value: Vec<u8>) -> String;
}

/// Prints through the host and reports what the host said. An empty reply
/// means it worked; anything else is the refusal, which the test asserts on.
#[plugin_fn]
pub fn cap_print(text: String) -> FnResult<String> {
    let reply = unsafe { plank_print(text)? };
    Ok(reply)
}

/// A counter in the component's own state: reads, increments, writes back.
/// Proves state survives across calls and belongs to this component alone.
#[plugin_fn]
pub fn cap_bump(_: ()) -> FnResult<String> {
    let current = unsafe { plank_state_get("counter".to_string())? };
    let n: u32 = String::from_utf8(current)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let next = n + 1;
    let err = unsafe { plank_state_set("counter".to_string(), next.to_string().into_bytes())? };
    if err.is_empty() {
        Ok(next.to_string())
    } else {
        Ok(err)
    }
}

/// The one event handler. A guest matches on the event name rather than
/// exporting a function per event — see the host's `wasmevents` docs for why.
///
/// This one is a test fixture, so its behaviour is driven by the payload: it
/// blocks anything mentioning "forbidden", rewrites anything mentioning
/// "rewrite", and otherwise just counts what it saw into its own state.
#[plugin_fn]
pub fn on_event(input: String) -> FnResult<String> {
    let field = |key: &str| -> String {
        input
            .split_once(&format!("\"{key}\":"))
            .and_then(|(_, rest)| rest.trim_start().strip_prefix('"'))
            .and_then(|rest| rest.split('"').next())
            .unwrap_or("")
            .to_string()
    };
    let event = field("event");
    let seen = unsafe { plank_state_get("events".to_string())? };
    let seen = String::from_utf8(seen).unwrap_or_default();
    let _ =
        unsafe { plank_state_set("events".to_string(), format!("{seen}{event};").into_bytes())? };

    if input.contains("forbidden") {
        return Ok(r#"{"block": "the fixture refuses anything forbidden"}"#.to_string());
    }
    if input.contains("rewrite") {
        return Ok(r#"{"replace": "rewritten by the fixture"}"#.to_string());
    }
    Ok("{}".to_string())
}

/// The `tool` surface: what this component offers the model.
#[plugin_fn]
pub fn tool_specs() -> FnResult<String> {
    Ok(r#"[{
        "name": "wordcount",
        "description": "count words in a string",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                       "required": ["text"]}
    }]"#
    .to_string())
}

/// Runs it. The host sends `{"name": ..., "args": {...}}`; the reply is the
/// observation the model sees, verbatim.
#[plugin_fn]
pub fn tool_call(input: String) -> FnResult<String> {
    let text = input
        .split_once("\"text\":")
        .and_then(|(_, rest)| rest.trim_start().strip_prefix('"'))
        .and_then(|rest| rest.split('"').next())
        .unwrap_or("");
    Ok(format!("{} words\n", text.split_whitespace().count()))
}
