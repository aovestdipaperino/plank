// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! A valid plugin that implements no surface at all: it asserts the ABI and
//! stops there. Exists so the host's "you claimed a surface you did not
//! implement" check has something real to refuse — a check that can only be
//! tested against a module that genuinely lacks the exports.

use extism_pdk::*;

#[plugin_fn]
pub fn plank_abi() -> FnResult<String> {
    Ok("1".to_string())
}
