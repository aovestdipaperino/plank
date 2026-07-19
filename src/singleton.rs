//! Single-instance model-lock probe.
//!
//! The ds4 engine refuses to start a second process (the model maps tens of
//! GiB) by `flock`-ing a lock file and calling `exit(2)` if it is already held
//! — which kills the whole plank process before Rust can report anything. To
//! turn that abrupt exit into a clean error, plank probes the same lock file
//! *before* opening the engine (see `acquire_model_lock` in `main.rs`).
//!
//! The probe is non-destructive: it takes the lock only to test contention and
//! releases it immediately, so the engine can acquire it itself a moment later.
//! Holding it here would make the engine's own in-process acquire fail, since
//! `flock` is keyed to the open file description, not the process.

use std::path::Path;

/// Outcome of [`probe_lock`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockProbe {
    /// The lock is free (or could not be probed) — startup may proceed.
    Free,
    /// Another process holds the lock — a second instance must not start.
    Contended,
}

/// Non-destructively checks whether another process holds an exclusive
/// `flock` on `path`. Returns [`LockProbe::Contended`] when it does; otherwise
/// [`LockProbe::Free`], having released any lock it briefly took. An inability
/// to open/probe the path is reported as `Free` (the engine's own guard still
/// backstops it).
#[cfg(unix)]
#[must_use]
pub fn probe_lock(path: &Path) -> LockProbe {
    use std::os::unix::io::AsRawFd;

    let Ok(file) = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
    else {
        return LockProbe::Free;
    };
    // SAFETY: `file` owns a valid fd; LOCK_NB makes flock non-blocking.
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc != 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EWOULDBLOCK) {
        return LockProbe::Contended;
    }
    // Release whatever we took so the engine can lock it itself; dropping the
    // File closes the fd and drops the flock.
    drop(file);
    LockProbe::Free
}

/// Non-unix fallback: no advisory locking, so never reports contention.
#[cfg(not(unix))]
#[must_use]
pub fn probe_lock(_path: &Path) -> LockProbe {
    LockProbe::Free
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::io::AsRawFd;

    #[test]
    fn probe_detects_a_held_lock_and_clears_when_released() {
        let path =
            std::env::temp_dir().join(format!("plank-singleton-{}.lock", std::process::id()));
        std::fs::remove_file(&path).ok();

        // No holder yet: free.
        assert_eq!(probe_lock(&path), LockProbe::Free);

        // Hold the lock from a separate open file description (as another
        // process would), then the probe must see contention.
        let holder = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .unwrap();
        // SAFETY: valid fd; non-blocking exclusive lock.
        let rc = unsafe { libc::flock(holder.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        assert_eq!(rc, 0, "test setup should acquire the lock");
        assert_eq!(probe_lock(&path), LockProbe::Contended);

        // Once released, the probe reports free again (and left nothing held).
        drop(holder);
        assert_eq!(probe_lock(&path), LockProbe::Free);
        assert_eq!(probe_lock(&path), LockProbe::Free);

        std::fs::remove_file(&path).ok();
    }
}
