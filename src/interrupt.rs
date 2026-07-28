// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! SIGINT handling: Ctrl-C interrupts a generation instead of killing plank.
//!
//! Port of `agent_sigint_handler`: a signal-async-safe flag the generation
//! loop polls between tokens. At the prompt, Ctrl-C is handled by the line
//! editor instead.

use std::sync::atomic::{AtomicBool, Ordering};

static SIGINT_PENDING: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_sigint(_sig: libc::c_int) {
    // Only an atomic store: async-signal-safe.
    SIGINT_PENDING.store(true, Ordering::SeqCst);
}

/// Installs the SIGINT handler; call once at startup.
pub fn install() {
    // SAFETY: handle_sigint only performs an atomic store, which is
    // async-signal-safe; libc::signal itself has no other preconditions.
    unsafe {
        libc::signal(
            libc::SIGINT,
            handle_sigint as *const () as libc::sighandler_t,
        );
    }
}

/// Raises the interrupt flag from something other than a signal.
///
/// The TUI reads keys itself, so an `Esc` pressed during a long command has to
/// reach the generation loop by the same route Ctrl-C takes — the loop polls
/// one flag, and it should not care which of the two set it.
pub fn request() {
    SIGINT_PENDING.store(true, Ordering::SeqCst);
}

/// True when a Ctrl-C arrived since the last [`clear`].
#[must_use]
pub fn pending() -> bool {
    SIGINT_PENDING.load(Ordering::SeqCst)
}

/// Clears the pending flag, returning whether it was set.
pub fn clear() -> bool {
    SIGINT_PENDING.swap(false, Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_raises_the_same_flag_as_a_signal() {
        clear();
        assert!(!pending());
        request();
        assert!(pending(), "an Esc must look like a Ctrl-C to the poller");
        clear();
    }

    #[test]
    fn flag_roundtrip() {
        SIGINT_PENDING.store(true, std::sync::atomic::Ordering::SeqCst);
        assert!(pending());
        assert!(clear());
        assert!(!pending());
        assert!(!clear());
    }
}
