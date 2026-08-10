// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Plank: a Rust port of the ds4 agent.
//!
//! The port proceeds functionality-by-functionality from the C reference in
//! `refs/ds4/ds4_agent.c`, not line-by-line. Each module maps to one functional
//! section of the original agent.

pub mod agents;
pub mod anim;
pub mod arcade;
pub mod branch;
pub mod checkpoint;
pub mod compact;
pub mod complete;
pub mod config;
pub mod configform;
pub mod consent;
pub mod context;
/// Document ingestion for the `read` tool (PDF → Markdown). Always compiled so
/// the extension routing is CI-tested; the conversion itself needs `docparse`.
pub mod doc;
pub mod download;
#[cfg(ds4_engine)]
pub mod ds4engine;
/// Token-primary transcript core for the ds4 backend (issue #58). FFI-free and
/// always compiled so its reconciliation/persistence logic is CI-tested; the
/// gated `ds4engine` drives it.
pub mod ds4tokens;
#[cfg(ds4_engine)]
pub mod ds4web;
pub mod dsml;
pub mod editor;
pub mod engine;
pub mod errlog;
pub mod export;
#[cfg(ds4_engine)]
pub mod ffi;
pub mod hooks;
pub mod host;
pub mod imagepaste;
pub mod insights;
pub mod interrupt;
pub mod kvcache;
pub mod kvmeta;
/// Volatility-tiered KV cache planning (issues #60, #64). FFI-free and always
/// compiled so the tier walk is CI-tested; the gated `ds4engine` executes it.
pub mod kvtier;
pub mod logo;
pub mod memory;
#[cfg(feature = "builtin_editor")]
pub mod miniedit;
pub mod names;
pub mod notify;
#[cfg(feature = "use_obscura")]
pub mod obscura_web;
pub mod openfile;
pub mod remote;
pub mod render;
pub mod repro;
pub mod sandbox;
pub mod serve;
pub mod session;
pub mod settings;
pub mod singleton;
pub mod skills;
pub mod slashmenu;
/// Cooperative round-robin over several sessions on the caller's thread — the
/// hostless counterpart to `EngineHost`'s scheduler, for fanning out forks.
pub mod slice;
pub mod snapshot;
pub mod speeds;
pub mod status;
pub mod statusbar;
pub mod stderrline;
pub mod sysprompt;
pub mod tasks;
pub mod templates;
pub mod title;
pub mod tools;
pub mod trace;
pub mod tui;
pub mod ui;
pub mod uiremote;
pub mod upgrade;
pub mod viz;
pub mod warp;
pub mod worker;
pub mod worktree;
