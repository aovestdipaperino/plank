// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Windows build of [`crate::stderrline`]: the same API, passing stderr through.
//!
//! Every redirect in the Unix module exists for the ds4 C engine, which prints
//! straight to fd 2 and is not built on Windows (`crates/local-inference-engine`
//! compiles it on macOS only). With nothing writing to fd 2 behind plank's
//! back, there is nothing to repaint, discard or filter, so each entry point
//! simply runs its closure.

use std::path::Path;

/// Never constructed on Windows: [`StderrLineReplacer::start`] returns `None`,
/// the same answer the Unix module gives when stderr is not a terminal.
#[derive(Debug)]
pub struct StderrLineReplacer(());

impl StderrLineReplacer {
    /// Always `None`; stderr flows through untouched.
    #[must_use]
    pub fn start() -> Option<Self> {
        None
    }
}

/// Runs `f`; there is no C engine log to discard.
pub fn discarding<T>(f: impl FnOnce() -> T) -> T {
    f()
}

/// Which side of a GPU-yield cycle a [`logging_to`] redirect covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Tearing the old engine down.
    Release,
    /// Bringing the engine back up after the yield.
    Reload,
}

/// Runs `f`; there is no C engine log to capture.
pub fn logging_to<T>(_log: &Path, _phase: Phase, f: impl FnOnce() -> T) -> T {
    f()
}

/// No exit hook to inform on Windows.
pub fn set_tui_active(_active: bool) {}

/// No exit hook to disarm on Windows.
pub fn disarm_exit_report() {}

/// No exit hook to inform on Windows.
pub fn set_exit_resume(_id: Option<String>) {}

/// Runs `f`; the chatter it filters comes only from the C engine.
pub fn without_ds4_chatter<T>(f: impl FnOnce() -> T) -> T {
    f()
}
