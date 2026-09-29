// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Blocking frames: a tool call that opens a WASM frame component and waits
//! for the user to close it, the way `ask` waits for an answer.

/// How a blocking frame ended, from the tool call's side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameResult {
    /// The file on the component's disk differs from what was staged.
    Saved(Vec<u8>),
    /// The file is as staged, or the frame deleted it.
    Unchanged,
    /// Nothing was opened, for this reason.
    Refused(String),
    /// The frame failed to open or trapped while up.
    Failed(String),
}
