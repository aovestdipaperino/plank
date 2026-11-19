// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Landlock write sandbox for model-initiated shell commands on Linux.
//!
//! The Linux counterpart of the Seatbelt profile in `sandbox.rs`, with the
//! same shape: reads stay unrestricted, and every write-like filesystem access
//! is denied except beneath [`Sandbox::write_roots`](crate::sandbox::Sandbox::write_roots).
//! Landlock is unprivileged and stacks on whatever confines plank already.
//!
//! The ruleset is built in plank ([`Ruleset::for_roots`]) and enforced in the
//! child between `fork` and `exec` ([`Ruleset::apply_in_child`]), which makes
//! only two syscalls (`prctl(PR_SET_NO_NEW_PRIVS)` and
//! `landlock_restrict_self`), both safe to call there. A child that cannot be
//! restricted fails to spawn rather than running unconfined.
//!
//! Written against the raw syscalls (`linux/landlock.h`) rather than a crate:
//! the ABI is a handful of constants and two structs.

use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
use std::path::{Path, PathBuf};

// Syscall numbers are the same on every architecture (the unified table).
const SYS_CREATE_RULESET: libc::c_long = 444;
const SYS_ADD_RULE: libc::c_long = 445;
const SYS_RESTRICT_SELF: libc::c_long = 446;

const CREATE_RULESET_VERSION: u32 = 1;
const RULE_PATH_BENEATH: libc::c_int = 1;

const WRITE_FILE: u64 = 1 << 1;
const REMOVE_DIR: u64 = 1 << 4;
const REMOVE_FILE: u64 = 1 << 5;
const MAKE_CHAR: u64 = 1 << 6;
const MAKE_DIR: u64 = 1 << 7;
const MAKE_REG: u64 = 1 << 8;
const MAKE_SOCK: u64 = 1 << 9;
const MAKE_FIFO: u64 = 1 << 10;
const MAKE_BLOCK: u64 = 1 << 11;
const MAKE_SYM: u64 = 1 << 12;
/// ABI 2: linking or renaming a file into another directory.
const REFER: u64 = 1 << 13;
/// ABI 3: `truncate(2)` and `O_TRUNC`.
const TRUNCATE: u64 = 1 << 14;

/// The rights a rule may grant on a regular file (the rest are directory
/// rights, which the kernel rejects on a file with `EINVAL`).
const FILE_RIGHTS: u64 = WRITE_FILE | TRUNCATE;

#[repr(C)]
struct RulesetAttr {
    handled_access_fs: u64,
}

#[repr(C, packed)]
struct PathBeneathAttr {
    allowed_access: u64,
    parent_fd: i32,
}

/// The write rights this kernel's Landlock ABI knows: what the ruleset
/// handles (denies by default) and what each writable root gets back. `0` for
/// no Landlock at all.
#[must_use]
pub fn write_rights(abi: i64) -> u64 {
    let mut rights = 0;
    if abi >= 1 {
        rights |= WRITE_FILE
            | REMOVE_DIR
            | REMOVE_FILE
            | MAKE_CHAR
            | MAKE_DIR
            | MAKE_REG
            | MAKE_SOCK
            | MAKE_FIFO
            | MAKE_BLOCK
            | MAKE_SYM;
    }
    if abi >= 2 {
        rights |= REFER;
    }
    if abi >= 3 {
        rights |= TRUNCATE;
    }
    rights
}

/// The kernel's Landlock ABI version, or the reason there is none (an older
/// kernel, or Landlock left out of the boot's LSM list).
///
/// # Errors
/// A human-readable reason Landlock is unavailable.
pub fn abi_version() -> Result<i64, String> {
    // SAFETY: the version query takes no attribute pointer and touches no memory.
    let r = unsafe {
        libc::syscall(
            SYS_CREATE_RULESET,
            std::ptr::null::<RulesetAttr>(),
            0usize,
            CREATE_RULESET_VERSION,
        )
    };
    if r > 0 {
        return Ok(r);
    }
    let err = std::io::Error::last_os_error();
    Err(match err.raw_os_error() {
        Some(libc::ENOSYS) => "the kernel has no Landlock (Linux 5.13 or later)".to_string(),
        Some(libc::EOPNOTSUPP) => "Landlock is disabled in this kernel's boot options".to_string(),
        _ => format!("Landlock is unavailable: {err}"),
    })
}

/// A Landlock ruleset ready to enforce on a child process.
#[derive(Debug)]
pub struct Ruleset {
    fd: OwnedFd,
}

impl Ruleset {
    /// A ruleset denying every write right `abi` knows, except beneath
    /// `roots`. A root that does not exist is skipped: nothing can be written
    /// under it anyway, and it cannot be created without a writable parent.
    ///
    /// # Errors
    /// The failing syscall's error.
    pub fn for_roots(abi: i64, roots: &[PathBuf]) -> std::io::Result<Self> {
        let rights = write_rights(abi);
        let attr = RulesetAttr {
            handled_access_fs: rights,
        };
        // SAFETY: `attr` is a live, correctly sized ruleset attribute.
        let fd = unsafe {
            libc::syscall(
                SYS_CREATE_RULESET,
                &raw const attr,
                size_of::<RulesetAttr>(),
                0u32,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        let fd = i32::try_from(fd).map_err(std::io::Error::other)?;
        // SAFETY: a fresh descriptor (close-on-exec) that this ruleset owns.
        let ruleset = Self {
            fd: unsafe { OwnedFd::from_raw_fd(fd) },
        };
        for root in roots {
            ruleset.allow_beneath(root, rights)?;
        }
        Ok(ruleset)
    }

    fn allow_beneath(&self, root: &Path, rights: u64) -> std::io::Result<()> {
        use std::os::unix::fs::OpenOptionsExt as _;
        let Ok(dir) = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_PATH | libc::O_CLOEXEC)
            .open(root)
        else {
            return Ok(());
        };
        let is_dir = dir.metadata().is_ok_and(|m| m.is_dir());
        let attr = PathBeneathAttr {
            allowed_access: if is_dir { rights } else { rights & FILE_RIGHTS },
            parent_fd: dir.as_raw_fd(),
        };
        // SAFETY: both descriptors are live and `attr` is a packed
        // path-beneath attribute, as the kernel expects.
        let r = unsafe {
            libc::syscall(
                SYS_ADD_RULE,
                self.fd.as_raw_fd(),
                RULE_PATH_BENEATH,
                &raw const attr,
                0u32,
            )
        };
        if r < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    /// Restricts the calling process, which must be a freshly forked child
    /// about to `exec`: called from `pre_exec`, it allocates nothing and makes
    /// only async-signal-safe syscalls. `no_new_privs` is what lets an
    /// unprivileged process restrict itself, and it also keeps a setuid
    /// binary the command runs from gaining privileges past the sandbox.
    ///
    /// # Errors
    /// The failing syscall's error; the spawn then fails with it.
    pub fn apply_in_child(&self) -> std::io::Result<()> {
        // SAFETY: `prctl` and `syscall` take plain integers here; neither
        // touches memory nor allocates.
        unsafe {
            if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::syscall(SYS_RESTRICT_SELF, self.fd.as_raw_fd(), 0u32) != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rights_grow_with_the_abi() {
        assert_eq!(write_rights(0), 0);
        let v1 = write_rights(1);
        assert_eq!(v1 & (REFER | TRUNCATE), 0);
        assert_ne!(v1 & WRITE_FILE, 0);
        assert_eq!(write_rights(2), v1 | REFER);
        assert_eq!(write_rights(3), v1 | REFER | TRUNCATE);
        // Later ABIs add no write rights plank handles.
        assert_eq!(write_rights(7), write_rights(3));
        // Reading and executing are never handled, so stay unrestricted.
        assert_eq!(write_rights(7) & 0b1101, 0);
    }

    #[test]
    fn path_beneath_attr_is_packed() {
        assert_eq!(size_of::<PathBeneathAttr>(), 12);
    }

    /// End to end on a kernel with Landlock: a write beneath the root
    /// succeeds, one beside it is refused, and reads stay open. Skipped where
    /// the kernel has none.
    #[test]
    fn a_restricted_child_writes_only_beneath_its_roots() {
        use std::os::unix::process::CommandExt as _;
        let Ok(abi) = abi_version() else {
            return;
        };
        let base = std::env::temp_dir().join(format!("plank-landlock-{}", std::process::id()));
        let inside = base.join("inside");
        let outside = base.join("outside");
        std::fs::create_dir_all(&inside).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("readme"), "visible").unwrap();
        let ruleset = Ruleset::for_roots(abi, &[inside.clone(), PathBuf::from("/dev")]).unwrap();
        let script = format!(
            "echo ok > '{i}/a' && cat '{o}/readme' >/dev/null && ! echo no > '{o}/b' 2>/dev/null",
            i = inside.display(),
            o = outside.display()
        );
        let mut cmd = std::process::Command::new("/bin/sh");
        cmd.arg("-c").arg(&script);
        // SAFETY: `apply_in_child` is async-signal-safe (see its docs).
        unsafe {
            cmd.pre_exec(move || ruleset.apply_in_child());
        }
        let status = cmd.status().unwrap();
        assert!(status.success(), "{script}");
        assert!(inside.join("a").exists());
        assert!(!outside.join("b").exists());
        // The parent itself was never restricted.
        std::fs::write(outside.join("c"), "x").unwrap();
        std::fs::remove_dir_all(&base).ok();
    }
}
