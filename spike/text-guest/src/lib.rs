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

#[plugin_fn]
pub fn frame_open(_input: String) -> FnResult<String> {
    Ok(r#"{"veiled": false}"#.to_string())
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
