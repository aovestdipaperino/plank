// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Library half of `plank-console`, so integration tests can drive the
//! pipeline and the protocol without a terminal.

pub mod ansiasm;
pub mod cmd;
pub mod pipeline;
pub mod proto;
pub mod registry;
pub mod session;
pub mod streamview;
