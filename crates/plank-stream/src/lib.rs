// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! plank's model-token stream renderer, shared by the `plank` agent and the
//! `plank-console` Turbo Vision monitor.
//!
//! The pipeline is three layers, each usable on its own:
//!
//! - [`dsml`] — the strict parser for the DSML tool-call syntax the model was
//!   trained on.
//! - [`viz`] — [`viz::StreamRenderer`], which splits a raw byte stream into
//!   visible text, thinking text, tool banners and error banners, driving a
//!   [`viz::RenderSink`].
//! - [`render`] — [`render::TokenRenderer`], the byte-at-a-time ANSI markdown
//!   and syntax-highlighting renderer (the ds4 C parity path).

pub mod dsml;
