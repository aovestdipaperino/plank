// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The ds4 C engine: `DeepSeek` V4 and V4.1 Flash, Qwen3.8-Flash-Next and GLM
//! on Metal.
//!
//! [`Model`] and [`Session`] here are the backend's own safe types; the
//! crate-level [`crate::Model`] wraps them alongside Gemma. Reach for these
//! when a ds4-only capability matters, such as the activation dumps described
//! in the crate docs.

#[cfg(ds4_engine)]
mod engine;
#[cfg(not(ds4_engine))]
#[path = "stub.rs"]
mod engine;

pub use engine::{Model, Session};
