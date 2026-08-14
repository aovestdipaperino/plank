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
