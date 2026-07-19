//! Session-snapshot primitive and an unconditional-restore RAII guard.
//!
//! The core [`RestoreOnDrop`] guard is always available (and unit-testable
//! without a model): it runs a closure on drop, no matter how the scope exits
//! — normal return, `?`, or panic — which is what makes a suspended `/btw`
//! aside safe (BTW-SUSPEND-DESIGN §4.2 step 3, §4.5: restore must run even on
//! the interrupt/error path).
//!
//! Under the `ds4_engine` cfg it also exposes [`SessionSnapshot`], a safe
//! wrapper over the `ds4_session_save_snapshot` / `load_snapshot` /
//! `snapshot_free` FFI. It is deliberately more than `generate_aside` needs:
//! `as_bytes` / `restore_bytes` let #29 (`/checkpoint`) and #12 (per-session
//! KV payloads) reuse the same primitive for on-disk persistence.

/// Runs `restore` exactly once when dropped, unconditionally.
///
/// Use it to guarantee a captured session snapshot is reloaded on *every* exit
/// path from an aside — success, error, or interrupt. Dropping is the only way
/// it fires, so there is no "disarm": the aside path always wants the restore.
#[derive(Debug)]
pub struct RestoreOnDrop<F: FnMut()> {
    restore: Option<F>,
}

impl<F: FnMut()> RestoreOnDrop<F> {
    /// Wraps a restore action to run when the guard drops.
    pub fn new(restore: F) -> Self {
        Self {
            restore: Some(restore),
        }
    }
}

impl<F: FnMut()> Drop for RestoreOnDrop<F> {
    fn drop(&mut self) {
        if let Some(mut restore) = self.restore.take() {
            restore();
        }
    }
}

#[cfg(ds4_engine)]
pub use ds4::SessionSnapshot;

#[cfg(ds4_engine)]
mod ds4 {
    use std::ffi::CStr;

    use crate::engine::EngineError;
    use crate::ffi;

    /// Owns a serialized copy of a session's KV state.
    ///
    /// [`capture`](Self::capture) allocates the buffer through the engine and
    /// frees it on drop; [`restore`](Self::restore) reloads it into a session.
    /// [`as_bytes`](Self::as_bytes) exposes the payload for persistence, and
    /// [`restore_bytes`](Self::restore_bytes) reloads a payload read back from
    /// disk (a non-owning path that must never call the engine's free).
    #[derive(Debug)]
    pub struct SessionSnapshot {
        inner: ffi::Ds4SessionSnapshot,
    }

    // SAFETY: the snapshot is an owned byte buffer with no interior thread
    // state; it moves with the single-threaded engine like the session does.
    unsafe impl Send for SessionSnapshot {}

    impl SessionSnapshot {
        /// Captures the current KV state of `session`.
        ///
        /// # Errors
        /// Returns [`EngineError`] if the engine fails to serialize the state.
        #[allow(clippy::not_unsafe_ptr_arg_deref)] // FFI boundary; safety documented below.
        pub fn capture(session: *mut ffi::Ds4Session) -> Result<Self, EngineError> {
            let mut inner = ffi::Ds4SessionSnapshot::default();
            let mut err = [0_i8; 512];
            // SAFETY: session valid; inner is a valid out-ptr the engine fills.
            let rc = unsafe {
                ffi::ds4_session_save_snapshot(session, &raw mut inner, err.as_mut_ptr(), err.len())
            };
            if rc != 0 || inner.ptr.is_null() {
                return Err(EngineError::new(cstr_message(
                    &err,
                    "session snapshot failed",
                )));
            }
            Ok(Self { inner })
        }

        /// Reloads this snapshot into `session`, replacing its KV state.
        ///
        /// # Errors
        /// Returns [`EngineError`] if the engine rejects the snapshot.
        #[allow(clippy::not_unsafe_ptr_arg_deref)] // FFI boundary; safety documented below.
        pub fn restore(&self, session: *mut ffi::Ds4Session) -> Result<(), EngineError> {
            let mut err = [0_i8; 512];
            // SAFETY: session valid; inner is a live snapshot we own.
            let rc = unsafe {
                ffi::ds4_session_load_snapshot(
                    session,
                    &raw const self.inner,
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            if rc != 0 {
                return Err(EngineError::new(cstr_message(
                    &err,
                    "session restore failed",
                )));
            }
            Ok(())
        }

        /// The serialized snapshot bytes, for persistence (#29, #12).
        #[must_use]
        pub fn as_bytes(&self) -> &[u8] {
            let len = usize::try_from(self.inner.len).unwrap_or(0);
            if self.inner.ptr.is_null() || len == 0 {
                return &[];
            }
            // SAFETY: ptr points to len bytes owned by this snapshot.
            unsafe { std::slice::from_raw_parts(self.inner.ptr, len) }
        }

        /// Reloads a snapshot payload read back from disk into `session`.
        ///
        /// Unlike [`restore`](Self::restore) this does not own the buffer: it
        /// wraps the caller's bytes in a transient FFI struct and never calls
        /// the engine's free (which would try to free the caller's `Vec`).
        ///
        /// # Errors
        /// Returns [`EngineError`] if the engine rejects the payload.
        #[allow(clippy::not_unsafe_ptr_arg_deref)] // FFI boundary; safety documented below.
        pub fn restore_bytes(
            session: *mut ffi::Ds4Session,
            bytes: &[u8],
        ) -> Result<(), EngineError> {
            let mut buf = bytes.to_vec();
            let snap = ffi::Ds4SessionSnapshot {
                ptr: buf.as_mut_ptr(),
                len: buf.len() as u64,
                cap: buf.capacity() as u64,
            };
            let mut err = [0_i8; 512];
            // SAFETY: session valid; snap borrows `buf`, which outlives the call.
            let rc = unsafe {
                ffi::ds4_session_load_snapshot(
                    session,
                    &raw const snap,
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            drop(buf);
            if rc != 0 {
                return Err(EngineError::new(cstr_message(
                    &err,
                    "session restore failed",
                )));
            }
            Ok(())
        }
    }

    impl Drop for SessionSnapshot {
        fn drop(&mut self) {
            // SAFETY: inner was filled by ds4_session_save_snapshot; freeing an
            // engine-owned buffer. Never called on the restore_bytes path,
            // whose transient struct is not a SessionSnapshot.
            unsafe { ffi::ds4_session_snapshot_free(&raw mut self.inner) };
        }
    }

    fn cstr_message(buf: &[i8], fallback: &str) -> String {
        if buf.first().copied().unwrap_or(0) == 0 {
            return fallback.to_string();
        }
        // SAFETY: buf is NUL-terminated within its length by the C callee.
        let s = unsafe { CStr::from_ptr(buf.as_ptr()) };
        s.to_string_lossy().into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::RestoreOnDrop;
    use std::cell::Cell;

    #[test]
    fn guard_restores_on_normal_exit() {
        let restored = Cell::new(false);
        {
            let _g = RestoreOnDrop::new(|| restored.set(true));
        }
        assert!(restored.get(), "guard must restore when the scope ends");
    }

    // Restore must run even when the aside loop is interrupted (§4.2 step 3):
    // simulate an interrupt that returns early from the scope and assert the
    // guard still fired.
    #[test]
    fn aside_restores_on_interrupt() {
        let restored = Cell::new(false);
        let interrupted = |flag: bool| -> Option<()> {
            let _g = RestoreOnDrop::new(|| restored.set(true));
            if flag {
                // Aside was interrupted: bail out of the scope early.
                return None;
            }
            Some(())
        };
        assert_eq!(interrupted(true), None);
        assert!(
            restored.get(),
            "restore must run on the interrupt/early-return path"
        );
    }

    // The main task's state (transcript + KV, modeled here as a String) is
    // byte-identical before and after a suspended aside that mutated it.
    #[test]
    fn aside_leaves_transcript_untouched() {
        use std::cell::RefCell;
        let state = RefCell::new(String::from("[user]\nmain task\n"));
        let snapshot = state.borrow().clone();
        {
            let _g = RestoreOnDrop::new(|| *state.borrow_mut() = snapshot.clone());
            // Destructive aside overwrites the shared state.
            *state.borrow_mut() = String::from("[user]\nbtw question\n");
            assert_eq!(*state.borrow(), "[user]\nbtw question\n");
        }
        assert_eq!(
            *state.borrow(),
            "[user]\nmain task\n",
            "aside must leave the main transcript untouched"
        );
    }
}
