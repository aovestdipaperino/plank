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

/// Serializes tests that raise [`request`] to simulate an interrupt: the flag
/// is one process-wide atomic, so two such tests running concurrently (the
/// default under `cargo test`) could otherwise see each other's flag. Callers
/// hold the guard for the whole time the flag may be set, and should still
/// [`clear`] it before dropping the guard so the next test starts clean.
///
/// Tests that merely *read* the flag need it too: a memory-pass test asks
/// `pending()` and sees "interrupted" if another test has raised it at that
/// moment. The guard is reentrant on a thread, so a helper that takes it and a
/// test body that takes it as well cannot deadlock each other.
#[cfg(test)]
pub(crate) fn test_guard() -> TestGuard {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let outermost = TEST_GUARD_DEPTH.with(|d| {
        d.set(d.get() + 1);
        d.get() == 1
    });
    TestGuard {
        _lock: outermost.then(|| {
            LOCK.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }),
    }
}

#[cfg(test)]
thread_local! {
    /// How many [`TestGuard`]s this thread holds, so only the outermost takes
    /// the process-wide lock.
    static TEST_GUARD_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// What [`test_guard`] returns: holds the process-wide lock on the outermost
/// acquisition for a thread and nothing on a nested one.
#[cfg(test)]
pub(crate) struct TestGuard {
    _lock: Option<std::sync::MutexGuard<'static, ()>>,
}

#[cfg(test)]
impl Drop for TestGuard {
    fn drop(&mut self) {
        TEST_GUARD_DEPTH.with(|d| d.set(d.get() - 1));
    }
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
