// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The Windows `bash` sandbox: a Low-integrity token, and Low labels on the
//! write roots for as long as plank runs.
//!
//! Windows has no per-process filesystem allowlist like Seatbelt or Landlock.
//! What it has is mandatory integrity control: a process at Low integrity may
//! read what it could read before, but may write only to objects labelled Low
//! (the default "no write up" policy). So a confined command is started with
//! a duplicate of plank's token lowered to Low, and the write roots get a Low
//! label (`S:(ML;OICI;NW;;;LW)`) at the first confined command. Everything
//! else a medium-integrity user owns stays out of reach.
//!
//! Why not a write-restricted token, which needs no label: Git Bash (msys)
//! cannot run under one. msys gives its own pipes and named objects a security
//! descriptor naming only the user SID, which a write-restricted access check
//! never accepts, so `sh.exe` dies creating its signal pipe (FINDINGS.md).
//!
//! The price of labels is that any other Low-integrity process on the machine
//! (a browser sandbox, for one) can write under a labelled root while the
//! label is set. So labels are temporary: each one plank adds is recorded in
//! `~/.plank/sandbox/low-labels`, with plank's pid, and removed when plank
//! exits ([`release_mine`]), or at the next launch after a crash
//! ([`sweep_stale`]). A root another live plank still uses keeps its label
//! until that one exits too. A root that was already labelled before plank
//! touched it is never recorded, so plank never removes a label it did not
//! set.
//!
//! The pure half (the ledger and which roots may be labelled) is compiled and
//! tested on every OS; the Windows calls are `cfg(windows)`.

use std::path::{Path, PathBuf};

/// One label plank set: the root and the pid of the plank that set it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub pid: u32,
    pub root: PathBuf,
}

/// Reads the ledger's lines, `<pid>\t<root>`; malformed lines are skipped.
#[must_use]
pub fn parse_ledger(text: &str) -> Vec<Entry> {
    text.lines()
        .filter_map(|line| {
            let (pid, root) = line.split_once('\t')?;
            Some(Entry {
                pid: pid.trim().parse().ok()?,
                root: PathBuf::from(root),
            })
        })
        .collect()
}

/// The ledger text for `entries`.
#[must_use]
pub fn render_ledger(entries: &[Entry]) -> String {
    use std::fmt::Write as _;
    entries.iter().fold(String::new(), |mut out, e| {
        let _ = writeln!(out, "{}\t{}", e.pid, e.root.display());
        out
    })
}

/// Splits `entries` into the ones to keep and the roots to unlabel, after
/// dropping every entry `gone` says is finished (a dead pid, or plank's own
/// at exit). A root loses its label only when no kept entry still names it.
#[must_use]
pub fn release_plan(
    entries: Vec<Entry>,
    gone: &dyn Fn(&Entry) -> bool,
) -> (Vec<Entry>, Vec<PathBuf>) {
    let (dropped, kept): (Vec<Entry>, Vec<Entry>) = entries.into_iter().partition(|e| gone(e));
    let mut unlabel: Vec<PathBuf> = Vec::new();
    for e in dropped {
        if !kept.iter().any(|k| k.root == e.root) && !unlabel.contains(&e.root) {
            unlabel.push(e.root);
        }
    }
    (kept, unlabel)
}

/// Whether plank may put a Low label on `root`: it exists as a directory, and
/// it is not a drive root, the user's home or one of its ancestors, or the
/// temp directory (confined commands get their own, [`TEMP_DIR_NAME`], so the
/// rest of `%TEMP%` stays out of reach).
#[must_use]
pub fn labelable(
    root: &Path,
    home: Option<&Path>,
    temp: &Path,
    is_dir: &dyn Fn(&Path) -> bool,
) -> bool {
    root.parent().is_some()
        && is_dir(root)
        && home.is_none_or(|h| !h.starts_with(root))
        && root != temp
}

/// The temp directory a confined command gets, under the user's `%TEMP%`.
pub const TEMP_DIR_NAME: &str = "plank-low";

/// The environment variable that marks a `Command` to be started at Low
/// integrity by `platform::CommandExt::spawn_tree`. The child sees it too,
/// which tells a tool it runs sandboxed.
pub const MARKER_VAR: &str = "PLANK_SANDBOX";

/// [`MARKER_VAR`]'s value for a Low-integrity command.
pub const MARKER_VALUE: &str = "low-integrity";

#[cfg(windows)]
pub use imp::{confine, ensure_labelled, release_mine, sweep_stale, temp_dir};

#[cfg(windows)]
mod imp {
    use super::{Entry, labelable, parse_ledger, release_plan, render_ledger};
    use std::io;
    use std::os::windows::io::{AsRawHandle as _, FromRawHandle as _, OwnedHandle};
    use std::path::{Path, PathBuf};
    use windows_sys::Win32::Foundation::{HANDLE, LocalFree};
    use windows_sys::Win32::Security::Authorization::{
        ConvertStringSecurityDescriptorToSecurityDescriptorW, ConvertStringSidToSidW,
        GetNamedSecurityInfoW, SE_FILE_OBJECT, SetNamedSecurityInfoW,
    };
    use windows_sys::Win32::Security::{
        ACL, ACL_REVISION, DuplicateTokenEx, GetLengthSid, GetSecurityDescriptorSacl,
        InitializeAcl, LABEL_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR, PSID, SID_AND_ATTRIBUTES,
        SecurityImpersonation, SetTokenInformation, TOKEN_ALL_ACCESS, TOKEN_MANDATORY_LABEL,
        TokenIntegrityLevel, TokenPrimary,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn NtSetInformationProcess(
            process: HANDLE,
            class: u32,
            info: *const core::ffi::c_void,
            len: u32,
        ) -> i32;
        fn NtResumeProcess(process: HANDLE) -> i32;
    }

    /// `PROCESS_ACCESS_TOKEN`, for `ProcessAccessToken` (class 9).
    #[repr(C)]
    struct ProcessAccessToken {
        token: HANDLE,
        thread: HANDLE,
    }

    const PROCESS_ACCESS_TOKEN_CLASS: u32 = 9;
    /// `SE_GROUP_INTEGRITY`.
    const SE_GROUP_INTEGRITY: u32 = 0x20;
    /// A Low mandatory label, inherited by files and folders, no write up.
    const LOW_LABEL_SDDL: &str = "S:(ML;OICI;NW;;;LW)";

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(Some(0)).collect()
    }

    fn wide_path(p: &Path) -> Vec<u16> {
        wide(&p.to_string_lossy())
    }

    /// Starts `child`, created suspended, at Low integrity: swaps a Low copy
    /// of plank's token in before its first instruction, then resumes it.
    /// Windows allows a process to give its child a lowered copy of its own
    /// token without `SeAssignPrimaryTokenPrivilege`.
    ///
    /// # Errors
    /// The token could not be made or assigned; the caller kills the child.
    pub fn confine(child: &std::process::Child) -> io::Result<()> {
        let token = low_token()?;
        let info = ProcessAccessToken {
            token: token.as_raw_handle() as HANDLE,
            thread: std::ptr::null_mut(),
        };
        // SAFETY: the child handle is live, `info` is the documented layout
        // for this class, and the token handle outlives the call.
        let status = unsafe {
            NtSetInformationProcess(
                child.as_raw_handle() as HANDLE,
                PROCESS_ACCESS_TOKEN_CLASS,
                (&raw const info).cast(),
                u32::try_from(size_of::<ProcessAccessToken>()).unwrap_or(0),
            )
        };
        if status != 0 {
            return Err(io::Error::other(format!(
                "could not lower the command to Low integrity (NTSTATUS {status:#x})"
            )));
        }
        // SAFETY: the child handle is live.
        let status = unsafe { NtResumeProcess(child.as_raw_handle() as HANDLE) };
        if status != 0 {
            return Err(io::Error::other(format!(
                "could not resume the confined command (NTSTATUS {status:#x})"
            )));
        }
        Ok(())
    }

    /// A primary token like plank's, at Low integrity.
    fn low_token() -> io::Result<OwnedHandle> {
        let mut own: HANDLE = std::ptr::null_mut();
        // SAFETY: the pseudo-handle needs no closing; `own` is a valid out-ptr.
        if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_ALL_ACCESS, &raw mut own) } == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `own` is a fresh handle this function now owns.
        let own = unsafe { OwnedHandle::from_raw_handle(own.cast()) };
        let mut dup: HANDLE = std::ptr::null_mut();
        // SAFETY: `own` is live; `dup` is a valid out-pointer.
        let ok = unsafe {
            DuplicateTokenEx(
                own.as_raw_handle() as HANDLE,
                TOKEN_ALL_ACCESS,
                std::ptr::null(),
                SecurityImpersonation,
                TokenPrimary,
                &raw mut dup,
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `dup` is a fresh handle this function now owns.
        let dup = unsafe { OwnedHandle::from_raw_handle(dup.cast()) };
        let mut low: PSID = std::ptr::null_mut();
        // SAFETY: a NUL-terminated SID string and a valid out-pointer.
        if unsafe { ConvertStringSidToSidW(wide("S-1-16-4096").as_ptr(), &raw mut low) } == 0 {
            return Err(io::Error::last_os_error());
        }
        let label = TOKEN_MANDATORY_LABEL {
            Label: SID_AND_ATTRIBUTES {
                Sid: low,
                Attributes: SE_GROUP_INTEGRITY,
            },
        };
        // SAFETY: `low` is the SID just converted, sized by `GetLengthSid`.
        let len = size_of::<TOKEN_MANDATORY_LABEL>() + unsafe { GetLengthSid(low) } as usize;
        // SAFETY: `label` and the SID it points to are live for the call.
        let ok = unsafe {
            SetTokenInformation(
                dup.as_raw_handle() as HANDLE,
                TokenIntegrityLevel,
                (&raw const label).cast(),
                u32::try_from(len).unwrap_or(0),
            )
        };
        let err = io::Error::last_os_error();
        // SAFETY: `low` came from `ConvertStringSidToSidW`.
        unsafe { LocalFree(low.cast()) };
        if ok == 0 {
            return Err(err);
        }
        Ok(dup)
    }

    /// Whether `path` carries a mandatory label, explicit or inherited.
    fn has_label(path: &Path) -> bool {
        let mut sacl: *mut ACL = std::ptr::null_mut();
        let mut sd: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
        // SAFETY: a NUL-terminated path and valid out-pointers.
        let rc = unsafe {
            GetNamedSecurityInfoW(
                wide_path(path).as_ptr(),
                SE_FILE_OBJECT,
                LABEL_SECURITY_INFORMATION,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &raw mut sacl,
                &raw mut sd,
            )
        };
        // SAFETY: when non-null, `sacl` points into `sd`, which is live here.
        let labelled = rc == 0 && !sacl.is_null() && unsafe { (*sacl).AceCount } > 0;
        if !sd.is_null() {
            // SAFETY: `sd` came from `GetNamedSecurityInfoW`.
            unsafe { LocalFree(sd.cast()) };
        }
        labelled
    }

    /// Sets (with `Some`) or clears (with `None`) `path`'s label, propagated
    /// to everything under it.
    fn set_label(path: &Path, sacl: *mut ACL) -> io::Result<()> {
        // SAFETY: a NUL-terminated path; `sacl` is a valid ACL.
        let rc = unsafe {
            SetNamedSecurityInfoW(
                wide_path(path).as_ptr(),
                SE_FILE_OBJECT,
                LABEL_SECURITY_INFORMATION,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                sacl,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(io::Error::from_raw_os_error(
                i32::try_from(rc).unwrap_or(i32::MAX),
            ))
        }
    }

    fn label_low(path: &Path) -> io::Result<()> {
        let mut sd: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
        // SAFETY: a NUL-terminated SDDL string and a valid out-pointer.
        let ok = unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                wide(LOW_LABEL_SDDL).as_ptr(),
                1,
                &raw mut sd,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        let (mut present, mut defaulted) = (0, 0);
        let mut sacl: *mut ACL = std::ptr::null_mut();
        // SAFETY: `sd` is the descriptor just built; out-pointers are valid.
        unsafe {
            GetSecurityDescriptorSacl(sd, &raw mut present, &raw mut sacl, &raw mut defaulted)
        };
        let result = set_label(path, sacl);
        // SAFETY: `sd` came from the conversion above; `sacl` points into it.
        unsafe { LocalFree(sd.cast()) };
        result
    }

    fn unlabel(path: &Path) -> io::Result<()> {
        // An empty SACL: the label goes, and with it the inherited copies.
        let mut acl = [0u8; 64];
        let acl_ptr: *mut ACL = acl.as_mut_ptr().cast();
        // SAFETY: `acl` is a writable buffer of the size passed.
        if unsafe { InitializeAcl(acl_ptr, 64, ACL_REVISION) } == 0 {
            return Err(io::Error::last_os_error());
        }
        set_label(path, acl_ptr)
    }

    #[cfg(test)]
    pub(super) fn has_label_for_tests(path: &Path) -> bool {
        has_label(path)
    }

    fn ledger_dir() -> Option<PathBuf> {
        crate::home::plank_home_opt().map(|h| h.join("sandbox"))
    }

    /// Runs `f` on the ledger under its lock, writing back what it returns.
    fn with_ledger(f: impl FnOnce(Vec<Entry>) -> Vec<Entry>) -> io::Result<()> {
        let dir = ledger_dir().ok_or_else(|| io::Error::other("no plank home"))?;
        std::fs::create_dir_all(&dir)?;
        let lock = std::fs::File::create(dir.join("low-labels.lock"))?;
        lock.lock()?;
        let path = dir.join("low-labels");
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let out = f(parse_ledger(&text));
        std::fs::write(&path, render_ledger(&out))
    }

    /// The temp directory confined commands use, created and labelled Low on
    /// first use (and recorded like any other root).
    ///
    /// # Errors
    /// It could not be created or labelled.
    pub fn temp_dir() -> io::Result<PathBuf> {
        let dir = std::env::temp_dir().join(super::TEMP_DIR_NAME);
        std::fs::create_dir_all(&dir)?;
        ensure_labelled(std::slice::from_ref(&dir), Some(&dir))?;
        Ok(dir)
    }

    /// Labels Low each of `roots` that [`labelable`] allows and that carries
    /// no label yet, recording it under plank's pid. Roots that cannot be
    /// labelled are skipped, which leaves them unwritable to confined
    /// commands, except `required`, whose failure is returned.
    ///
    /// # Errors
    /// `required` could not be labelled, or the ledger could not be updated.
    pub fn ensure_labelled(roots: &[PathBuf], required: Option<&Path>) -> io::Result<()> {
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let temp = std::env::temp_dir();
        let me = std::process::id();
        let mut failure = None;
        with_ledger(|mut entries| {
            for root in roots {
                let ours = root.ends_with(super::TEMP_DIR_NAME);
                if !ours && !labelable(root, home.as_deref(), &temp, &|p| p.is_dir()) {
                    continue;
                }
                if entries.iter().any(|e| e.pid == me && &e.root == root) || has_label(root) {
                    continue;
                }
                match label_low(root) {
                    Ok(()) => entries.push(Entry {
                        pid: me,
                        root: root.clone(),
                    }),
                    Err(e) if required == Some(root.as_path()) => failure = Some(e),
                    Err(_) => {}
                }
            }
            entries
        })?;
        failure.map_or(Ok(()), Err)
    }

    /// Removes the labels this plank set, except where another live plank
    /// still uses the root. Called as plank exits.
    pub fn release_mine() {
        let me = std::process::id();
        release(&|e: &Entry| e.pid == me || !crate::platform::pid_alive(e.pid));
    }

    /// Removes the labels a plank that is gone set (one that crashed or was
    /// killed). Called at startup.
    pub fn sweep_stale() {
        release(&|e: &Entry| !crate::platform::pid_alive(e.pid));
    }

    fn release(gone: &dyn Fn(&Entry) -> bool) {
        let _ = with_ledger(|entries| {
            let (kept, unlabel_roots) = release_plan(entries, gone);
            for root in unlabel_roots {
                let _ = unlabel(&root);
            }
            kept
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(pid: u32, root: &str) -> Entry {
        Entry {
            pid,
            root: PathBuf::from(root),
        }
    }

    /// The real thing, on Windows with Git Bash: a confined `sh` writes
    /// inside a labelled root, is refused next to it, and the label goes when
    /// plank releases it.
    #[cfg(windows)]
    #[test]
    fn a_confined_shell_writes_only_inside_a_labelled_root() {
        use crate::platform::CommandExt as _;
        let base = std::env::temp_dir().join(format!("plank-lowil-test-{}", std::process::id()));
        let (work, outside) = (base.join("work"), base.join("outside"));
        std::fs::create_dir_all(&work).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let mut probe = crate::platform::shell_command();
        if probe.arg("-c").arg("true").output().is_err() {
            return; // No POSIX shell on this machine.
        }
        ensure_labelled(std::slice::from_ref(&work), Some(&work)).unwrap();
        let fwd = |p: &Path| p.display().to_string().replace('\\', "/");
        let script = format!(
            "echo in > '{}/a'; echo out > '{}/b'; echo done",
            fwd(&work),
            fwd(&outside)
        );
        let out = crate::platform::shell_command()
            .arg("-c")
            .arg(&script)
            .env(MARKER_VAR, MARKER_VALUE)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn_tree()
            .unwrap()
            .wait_with_output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("done"),
            "{stderr}"
        );
        assert!(
            work.join("a").exists(),
            "the write inside the root landed: {stderr}"
        );
        assert!(!outside.join("b").exists(), "the write outside was refused");
        assert!(stderr.contains("Permission denied"), "{stderr}");
        release_mine();
        assert!(!imp::has_label_for_tests(&work), "the label was removed");
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn the_ledger_round_trips_and_skips_bad_lines() {
        let entries = vec![e(7, r"C:\work"), e(9, r"C:\Users\me\.cargo\registry")];
        let text = render_ledger(&entries);
        assert_eq!(parse_ledger(&format!("{text}junk\nx\ty\n")), entries);
    }

    #[test]
    fn a_root_another_live_plank_uses_keeps_its_label() {
        let entries = vec![e(1, "/w"), e(2, "/w"), e(1, "/cache"), e(3, "/old")];
        // Plank 1 exits; plank 3 is long dead; plank 2 is alive.
        let (kept, unlabel) = release_plan(entries, &|x| x.pid == 1 || x.pid == 3);
        assert_eq!(kept, vec![e(2, "/w")]);
        assert_eq!(
            unlabel,
            vec![PathBuf::from("/cache"), PathBuf::from("/old")]
        );
    }

    #[test]
    fn drive_roots_home_and_temp_are_never_labelled() {
        let dir = |_: &Path| true;
        let home = Path::new("/home/me");
        let temp = Path::new("/tmp");
        assert!(labelable(
            Path::new("/home/me/work"),
            Some(home),
            temp,
            &dir
        ));
        assert!(!labelable(Path::new("/"), Some(home), temp, &dir));
        assert!(!labelable(home, Some(home), temp, &dir));
        assert!(!labelable(Path::new("/home"), Some(home), temp, &dir));
        assert!(!labelable(temp, Some(home), temp, &dir));
        assert!(!labelable(
            Path::new("/home/me/gone"),
            Some(home),
            temp,
            &|_| false
        ));
    }
}
