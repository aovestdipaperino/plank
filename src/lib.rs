//! Plank: a Rust port of the ds4 agent.
//!
//! The port proceeds functionality-by-functionality from the C reference in
//! `refs/ds4/ds4_agent.c`, not line-by-line. Each module maps to one functional
//! section of the original agent.

pub mod agents;
pub mod checkpoint;
pub mod compact;
pub mod complete;
pub mod config;
pub mod context;
pub mod download;
#[cfg(ds4_engine)]
pub mod ds4engine;
pub mod dsml;
pub mod editor;
pub mod engine;
#[cfg(ds4_engine)]
pub mod ffi;
pub mod hooks;
pub mod host;
pub mod imagepaste;
pub mod interrupt;
pub mod logo;
pub mod memory;
pub mod names;
pub mod remote;
pub mod render;
pub mod repro;
pub mod sandbox;
pub mod serve;
pub mod session;
pub mod singleton;
pub mod skills;
pub mod snapshot;
pub mod status;
pub mod statusbar;
pub mod stderrline;
pub mod sysprompt;
pub mod tools;
pub mod trace;
pub mod tui;
pub mod ui;
pub mod upgrade;
pub mod viz;
pub mod worker;
