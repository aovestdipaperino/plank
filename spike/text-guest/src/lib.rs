// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! A `frame` component that draws with `frame_step_text` and never packs a
//! glyph buffer.
//!
//! This is the shape the design offers an author for their first afternoon: a
//! JSON array of rows, no binary layout to get wrong. It exists as its own
//! guest because the host chooses between the packed and text exports by which
//! one a module *has*, and a module that has both would never exercise the
//! text path — `abi-guest` exports `frame_step`, so it cannot test this.

use extism_pdk::*;

#[plugin_fn]
pub fn plank_abi() -> FnResult<String> {
    Ok("1".to_string())
}

/// Stashes what `OpenParams.config` carried, so a test can prove the host
/// resolved the declaration rather than merely accepting it.
static mut SEEN_CONFIG: Option<String> = None;

#[plugin_fn]
pub fn frame_open(input: String) -> FnResult<String> {
    unsafe { SEEN_CONFIG = Some(input) };
    Ok(r#"{"veiled": false}"#.to_string())
}

/// Reports the config the host delivered, through the command surface so the
/// test can read it back without a second export in the frame ABI.
#[plugin_fn]
pub fn command_specs() -> FnResult<String> {
    Ok(r#"[{"name": "seen", "args": "", "desc": "what config arrived"}]"#.to_string())
}

#[plugin_fn]
pub fn command_run(_input: String) -> FnResult<String> {
    let seen = unsafe { (*(&raw const SEEN_CONFIG)).clone() };
    let seen = seen.unwrap_or_else(|| "never opened".to_string());
    // Reported through `print`, the command surface's own channel, so the test
    // reads it as a scrollback line rather than needing a bespoke export. The
    // payload is JSON inside a JSON string, hence the escaping.
    let escaped = seen.replace('\\', "\\\\").replace('"', "\\\"");
    Ok(format!(r#"{{"print": ["{escaped}"]}}"#))
}

/// One step, as rows of text. The second row carries a colour and bold so a
/// test can prove those survive the conversion rather than being defaulted.
#[plugin_fn]
pub fn frame_step_text(_input: String) -> FnResult<String> {
    Ok(r#"{"lines": [
        {"text": "hello"},
        {"text": "world", "fg": [1, 2, 3], "bold": true}
    ]}"#
    .to_string())
}

#[plugin_fn]
pub fn frame_key(input: String) -> FnResult<String> {
    if input.contains("\"q\"") {
        return Ok(r#"{"close": "text frame done"}"#.to_string());
    }
    Ok(r#"{"stay": true}"#.to_string())
}

#[plugin_fn]
pub fn frame_close() -> FnResult<String> {
    Ok("{}".to_string())
}
