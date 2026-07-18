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
    fn flag_roundtrip() {
        SIGINT_PENDING.store(true, std::sync::atomic::Ordering::SeqCst);
        assert!(pending());
        assert!(clear());
        assert!(!pending());
        assert!(!clear());
    }
}
