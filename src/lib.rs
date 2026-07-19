//! Plank: a Rust port of the ds4 agent.
//!
//! The port proceeds functionality-by-functionality from the C reference in
//! `ds4-ref/ds4_agent.c`, not line-by-line. Each module maps to one functional
//! section of the original agent.

pub mod compact;
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
pub mod imagepaste;
pub mod interrupt;
pub mod logo;
pub mod render;
pub mod session;
pub mod skills;
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
