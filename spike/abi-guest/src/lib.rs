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
    Ok(
        r#"[{"name": "greet", "args": "<who>", "desc": "say hello from wasm"},
           {"name": "bounce", "args": "", "desc": "open the fast bouncer"}]"#
            .to_string(),
    )
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
    if input.contains("\"bounce\"") {
        // A command that opens its own component's frame, picking the face.
        return Ok(r#"{"open": "fast"}"#.to_string());
    }
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

/// The `segment` surface: one status-bar cell. Reads the payload it is given
/// so the test can prove the host actually passes session facts in.
#[plugin_fn]
pub fn segment_render(input: String) -> FnResult<String> {
    let messages = input
        .split_once("\"messages\":")
        .and_then(|(_, rest)| rest.trim_start().split(&[',', '}'][..]).next())
        .unwrap_or("?")
        .trim()
        .to_string();
    Ok(format!(r#"{{"text": "msgs {messages}", "priority": 200}}"#))
}

// ---------------------------------------------------------------------------
// The `frame` surface: a bouncing character, in as few lines as the contract
// allows. It exists to exercise the packed glyph buffer and the open/step/
// key/close lifecycle, not to be a game.
// ---------------------------------------------------------------------------

/// Frame state. A guest has no ambient clock and no ambient randomness — both
/// arrive in the payload — so a static is all the state there is.
static mut FRAME: (f32, f32, f32, f32, u16, u16, u32) = (0.0, 0.0, 24.0, 12.0, 0, 0, 0);

#[plugin_fn]
pub fn frame_open(input: String) -> FnResult<String> {
    let num = |key: &str| -> f32 {
        input
            .split_once(&format!("\"{key}\":"))
            .and_then(|(_, rest)| {
                rest.trim_start()
                    .split(&[',', '}'][..])
                    .next()
                    .and_then(|n| n.trim().parse::<f32>().ok())
            })
            .unwrap_or(0.0)
    };
    let (w, h, seed) = (num("w"), num("h"), num("seed"));
    // One module can serve several frames; the face arrives as `arg`. This
    // fixture has two, so a test can prove the selector reaches the guest.
    let fast = input.contains("\"arg\": \"fast\"");
    unsafe {
        // The seed picks a starting corner, so two opens with different seeds
        // are visibly different and the same seed replays exactly.
        let s = seed as u32;
        FRAME = (
            if s % 2 == 0 { 1.0 } else { w - 2.0 },
            if (s / 2) % 2 == 0 { 1.0 } else { h - 2.0 },
            if s % 2 == 0 { 24.0 } else { -24.0 } * if fast { 4.0 } else { 1.0 },
            if (s / 2) % 2 == 0 { 12.0 } else { -12.0 } * if fast { 4.0 } else { 1.0 },
            w as u16,
            h as u16,
            0,
        );
    }
    Ok(r#"{"ok": true}"#.to_string())
}

#[plugin_fn]
pub fn frame_step(input: String) -> FnResult<Vec<u8>> {
    let num = |key: &str| -> f32 {
        input
            .split_once(&format!("\"{key}\":"))
            .and_then(|(_, rest)| {
                rest.trim_start()
                    .split(&[',', '}'][..])
                    .next()
                    .and_then(|n| n.trim().parse::<f32>().ok())
            })
            .unwrap_or(0.0)
    };
    let (dt, w, h) = (num("dt_ms") / 1000.0, num("w"), num("h"));
    let (x, y, steps) = unsafe {
        let f = &mut *(&raw mut FRAME);
        f.4 = w as u16;
        f.5 = h as u16;
        f.0 += f.2 * dt;
        f.1 += f.3 * dt;
        if f.0 < 0.0 || f.0 > w - 1.0 {
            f.2 = -f.2;
            f.0 = f.0.clamp(0.0, (w - 1.0).max(0.0));
        }
        if f.1 < 0.0 || f.1 > h - 1.0 {
            f.3 = -f.3;
            f.1 = f.1.clamp(0.0, (h - 1.0).max(0.0));
        }
        f.6 += 1;
        (f.0 as u16, f.1 as u16, f.6)
    };

    // The packed layout, written by hand: a guest in another language would do
    // exactly this, so the fixture proves the format is writable without the
    // host's own encoder.
    let mut out = Vec::with_capacity(22);
    out.extend_from_slice(b"PGLY");
    out.extend_from_slice(&1u16.to_le_bytes()); // version
    out.extend_from_slice(&1u16.to_le_bytes()); // count
    out.extend_from_slice(&(w as u16).to_le_bytes());
    out.extend_from_slice(&(h as u16).to_le_bytes());
    out.extend_from_slice(&x.to_le_bytes());
    out.extend_from_slice(&y.to_le_bytes());
    out.extend_from_slice(&u32::from('●').to_le_bytes());
    // Colour drifts with the step count so a test can tell two frames apart.
    out.extend_from_slice(&[(steps % 256) as u8, 200, 100, 1 /* bold */]);
    Ok(out)
}

#[plugin_fn]
pub fn frame_key(input: String) -> FnResult<String> {
    if input.contains("\"q\"") || input.contains("escape") {
        return Ok(r#"{"close": "the bouncer says goodbye"}"#.to_string());
    }
    Ok(r#"{"stay": true}"#.to_string())
}

#[plugin_fn]
pub fn frame_close() -> FnResult<String> {
    let steps = unsafe { (*(&raw const FRAME)).6 };
    Ok(format!(r#"{{"scrollback": "bounced for {steps} steps"}}"#))
}
