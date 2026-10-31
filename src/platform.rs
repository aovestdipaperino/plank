// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! OS primitives behind one portable API (Linux and Windows port, issue #89).
//!
//! Each function here replaced a direct `libc` call that only exists on Unix.
//! The Unix implementation is the call it replaced, unchanged, so macOS and
//! Linux behave exactly as before; Windows gets its own implementation beside
//! it. Callers stay free of `cfg` blocks.

use std::path::Path;

/// Positional file I/O: `read_exact_at` / `write_all_at`.
///
/// On Unix this is std's own `FileExt` (`pread`/`pwrite`). On Windows it is
/// built on `seek_read`/`seek_write`, which also move the file cursor, so do
/// not mix it with cursor-relative reads on the same handle.
#[cfg(unix)]
pub use std::os::unix::fs::FileExt;

/// See the Unix re-export above.
#[cfg(windows)]
pub trait FileExt {
    /// Reads exactly `buf.len()` bytes starting at `offset`.
    ///
    /// # Errors
    /// `UnexpectedEof` when the file ends first, or the OS error.
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()>;
    /// Writes all of `buf` starting at `offset`.
    ///
    /// # Errors
    /// `WriteZero` when the OS accepts nothing, or the OS error.
    fn write_all_at(&self, buf: &[u8], offset: u64) -> std::io::Result<()>;
}

#[cfg(windows)]
impl FileExt for std::fs::File {
    fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
        use std::os::windows::fs::FileExt as _;
        while !buf.is_empty() {
            match self.seek_read(buf, offset) {
                Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => {
                    buf = &mut buf[n..];
                    offset += n as u64;
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    fn write_all_at(&self, mut buf: &[u8], mut offset: u64) -> std::io::Result<()> {
        use std::os::windows::fs::FileExt as _;
        while !buf.is_empty() {
            match self.seek_write(buf, offset) {
                Ok(0) => return Err(std::io::ErrorKind::WriteZero.into()),
                Ok(n) => {
                    buf = &buf[n..];
                    offset += n as u64;
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

/// A broken-down local time, as `localtime_r` reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalTime {
    /// Full year, e.g. 2026.
    pub year: i32,
    /// Month, 1–12.
    pub month: i32,
    /// Day of the month, 1–31.
    pub day: i32,
    /// Hour, 0–23.
    pub hour: i32,
    /// Minute, 0–59.
    pub minute: i32,
    /// Second, 0–60 (leap second).
    pub second: i32,
    /// Seconds east of UTC in effect at this instant.
    pub utc_offset: i64,
}

impl LocalTime {
    /// `YYYY-MM-DD`.
    #[must_use]
    pub fn date(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }

    /// `YYYY-MM-DD HH:MM:SS`.
    #[must_use]
    pub fn date_time(&self) -> String {
        format!(
            "{} {:02}:{:02}:{:02}",
            self.date(),
            self.hour,
            self.minute,
            self.second
        )
    }
}

/// Converts Unix seconds to local time, or `None` when the C library cannot.
#[must_use]
pub fn local_time(secs: i64) -> Option<LocalTime> {
    let t = libc::time_t::try_from(secs).ok()?;
    // SAFETY: `tm` is plain-old-data; zeroed is valid and the call overwrites it.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    #[cfg(unix)]
    {
        // SAFETY: `t` and `tm` are valid, distinct pointers for the call;
        // localtime_r fills `tm` or returns NULL.
        if unsafe { libc::localtime_r(&raw const t, &raw mut tm) }.is_null() {
            return None;
        }
    }
    #[cfg(windows)]
    {
        // SAFETY: as above; localtime_s returns non-zero on failure.
        if unsafe { libc::localtime_s(&raw mut tm, &raw const t) } != 0 {
            return None;
        }
    }
    let local = LocalTime {
        year: tm.tm_year + 1900,
        month: tm.tm_mon + 1,
        day: tm.tm_mday,
        hour: tm.tm_hour,
        minute: tm.tm_min,
        second: tm.tm_sec,
        utc_offset: 0,
    };
    #[cfg(unix)]
    #[allow(
        clippy::useless_conversion,
        reason = "`c_long` is `i32` on 32-bit Unix"
    )]
    let utc_offset = i64::from(tm.tm_gmtoff);
    // The MSVC `tm` has no `tm_gmtoff`; the offset is the local wall clock
    // read back as if it were UTC, minus the instant itself.
    #[cfg(windows)]
    let utc_offset = civil_to_unix(&local) - secs;
    Some(LocalTime {
        utc_offset,
        ..local
    })
}

/// Seconds since the epoch of `t`'s wall-clock fields read as UTC.
#[cfg_attr(not(windows), allow(dead_code))]
fn civil_to_unix(t: &LocalTime) -> i64 {
    // Howard Hinnant's days_from_civil.
    let (y, m) = (i64::from(t.year), i64::from(t.month));
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + i64::from(t.day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    days * 86_400 + i64::from(t.hour) * 3600 + i64::from(t.minute) * 60 + i64::from(t.second)
}

/// The terminal size behind stdout as `(cols, rows)`, or `None` when stdout is
/// not a terminal or reports zero.
#[must_use]
pub fn stdout_size() -> Option<(u16, u16)> {
    #[cfg(unix)]
    {
        let ws = stdout_winsize()?;
        (ws.ws_col > 0 && ws.ws_row > 0).then_some((ws.ws_col, ws.ws_row))
    }
    #[cfg(windows)]
    {
        ratatui::crossterm::terminal::size()
            .ok()
            .filter(|&(c, r)| c > 0 && r > 0)
    }
}

/// The terminal size behind stdout in pixels as `(width, height)`, or `None`
/// when the terminal does not report it (always, on Windows consoles).
#[must_use]
pub fn stdout_pixels() -> Option<(u16, u16)> {
    #[cfg(unix)]
    {
        let ws = stdout_winsize()?;
        (ws.ws_xpixel > 0 && ws.ws_ypixel > 0).then_some((ws.ws_xpixel, ws.ws_ypixel))
    }
    #[cfg(windows)]
    {
        None
    }
}

#[cfg(unix)]
fn stdout_winsize() -> Option<libc::winsize> {
    // SAFETY: winsize is plain-old-data; zeroed is valid and ioctl overwrites.
    let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
    // SAFETY: stdout fd is valid and `ws` is a writable winsize buffer.
    let rc = unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &raw mut ws) };
    (rc == 0).then_some(ws)
}

/// The VM page size in bytes, or `None` when the OS will not say.
#[must_use]
pub fn page_size() -> Option<u64> {
    #[cfg(unix)]
    {
        // SAFETY: sysconf has no preconditions.
        let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        u64::try_from(n).ok().filter(|&n| n > 0)
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
        // SAFETY: SYSTEM_INFO is plain-old-data that GetSystemInfo fills in.
        let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
        // SAFETY: `info` is a valid out-parameter.
        unsafe { GetSystemInfo(&raw mut info) };
        Some(u64::from(info.dwPageSize)).filter(|&n| n > 0)
    }
}

/// Bytes of physical RAM, or `None` when the OS will not say.
#[must_use]
pub fn physical_memory() -> Option<u64> {
    #[cfg(unix)]
    {
        // SAFETY: sysconf only reads system configuration.
        let (pages, size) = unsafe {
            (
                libc::sysconf(libc::_SC_PHYS_PAGES),
                libc::sysconf(libc::_SC_PAGESIZE),
            )
        };
        let (p, s) = (u64::try_from(pages).ok()?, u64::try_from(size).ok()?);
        Some(p.saturating_mul(s))
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        // SAFETY: MEMORYSTATUSEX is plain-old-data; dwLength must be set first.
        let mut st: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        st.dwLength = u32::try_from(std::mem::size_of::<MEMORYSTATUSEX>()).ok()?;
        // SAFETY: `st` is a valid, initialized out-parameter.
        (unsafe { GlobalMemoryStatusEx(&raw mut st) } != 0).then_some(st.ullTotalPhys)
    }
}

/// Free bytes available to this user on the volume holding `dir`.
#[must_use]
pub fn free_bytes(dir: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt as _;
        let c = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
        let mut st = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        // SAFETY: `c` is a NUL-terminated path and `st` a valid out-parameter
        // of the right type; it is read only when the call reports success.
        let rc = unsafe { libc::statvfs(c.as_ptr(), st.as_mut_ptr()) };
        if rc != 0 {
            return None;
        }
        // SAFETY: statvfs returned 0, so it filled `st`.
        let st = unsafe { st.assume_init() };
        #[allow(clippy::useless_conversion)]
        let (avail, frsize) = (u64::from(st.f_bavail), u64::from(st.f_frsize));
        Some(avail.saturating_mul(frsize))
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt as _;
        use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
        let wide: Vec<u16> = dir.as_os_str().encode_wide().chain([0]).collect();
        let mut avail = 0u64;
        // SAFETY: `wide` is NUL-terminated and outlives the call; the other
        // out-parameters may be null.
        let ok = unsafe {
            GetDiskFreeSpaceExW(
                wide.as_ptr(),
                &raw mut avail,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        (ok != 0).then_some(avail)
    }
}

/// Whether `pid` names a live process. Pid 0 is always `false`.
#[must_use]
pub fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    #[cfg(unix)]
    {
        let pid = libc::pid_t::try_from(pid).unwrap_or(-1);
        // SAFETY: signal 0 delivers nothing; this is a pure existence probe.
        let rc = unsafe { libc::kill(pid, 0) };
        rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
        use windows_sys::Win32::System::Threading::{
            GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        };
        // SAFETY: OpenProcess has no pointer arguments; a null handle is failure.
        let h = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if h.is_null() {
            // Access denied still means the process exists, as EPERM does above.
            return std::io::Error::last_os_error().raw_os_error() == Some(5);
        }
        let mut code = 0u32;
        // SAFETY: `h` is a valid process handle and `code` a valid out-parameter.
        let ok = unsafe { GetExitCodeProcess(h, &raw mut code) };
        // SAFETY: `h` is a valid handle owned here.
        unsafe { CloseHandle(h) };
        ok != 0 && code == STILL_ACTIVE.cast_unsigned()
    }
}

/// Windows: sets `HOME` from `USERPROFILE` when it is unset.
///
/// plank reads `$HOME` for `~/.plank`, `~` expansion and the sandbox roots, and
/// hands it to the shells it starts. Windows itself only sets `USERPROFILE`;
/// adopting it as `HOME` (the convention Git Bash and MSYS follow) keeps every
/// one of those readers, and every child shell, agreeing on one directory.
/// A no-op on Unix, and whenever `HOME` is already set.
///
/// Call first thing in `main`, before any thread exists: changing the
/// environment is only sound while nothing else can read it.
pub fn ensure_home() {
    #[cfg(windows)]
    if std::env::var_os("HOME").is_none_or(|h| h.is_empty())
        && let Some(profile) = std::env::var_os("USERPROFILE").filter(|p| !p.is_empty())
    {
        // SAFETY: called at the top of `main`, before plank starts a thread.
        unsafe { std::env::set_var("HOME", profile) };
    }
}

/// Makes `link` name the file `target` (relative to `link`'s directory).
///
/// Unix: a symlink, as always. Windows: a file symlink when the account may
/// create one (Developer Mode or elevation), else a hard link, which needs no
/// privilege and still follows edits made in place. A hard link does not follow
/// an editor that replaces the target with a new file; that is the price of
/// not requiring Developer Mode.
///
/// # Errors
/// The OS error from the last attempt.
pub fn link_file(target: &str, link: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link)
    }
    #[cfg(windows)]
    {
        std::os::windows::fs::symlink_file(target, link).or_else(|_| {
            let dir = link.parent().unwrap_or_else(|| Path::new("."));
            std::fs::hard_link(dir.join(target), link)
        })
    }
}

/// A C-runtime file descriptor: `RawFd` on Unix, a CRT fd on Windows (both
/// `c_int`).
pub type Fd = libc::c_int;
/// Standard input's descriptor.
pub const STDIN_FD: Fd = 0;
/// Standard output's descriptor.
pub const STDOUT_FD: Fd = 1;

/// One `write(2)` of `buf` to `fd`; the byte count, or negative on error.
#[must_use]
pub fn write_fd(fd: Fd, buf: &[u8]) -> isize {
    #[cfg(unix)]
    {
        // SAFETY: writing `buf.len()` bytes from a live slice to a descriptor
        // the caller keeps open for the duration of the call.
        unsafe { libc::write(fd, buf.as_ptr().cast(), buf.len()) }
    }
    #[cfg(windows)]
    {
        let len = libc::c_uint::try_from(buf.len()).unwrap_or(libc::c_uint::MAX);
        // SAFETY: as above; the CRT takes the count as `unsigned int`.
        isize::try_from(unsafe { libc::write(fd, buf.as_ptr().cast(), len) }).unwrap_or(-1)
    }
}

/// Whether `fd` is a terminal.
#[must_use]
pub fn isatty(fd: Fd) -> bool {
    // SAFETY: `isatty` only inspects the descriptor.
    unsafe { libc::isatty(fd) == 1 }
}

/// Drops input the terminal sent that nobody read (late mouse or focus
/// reports), so the shell does not see it as typed text after plank exits.
pub fn flush_terminal_input() {
    #[cfg(unix)]
    {
        // SAFETY: `tcflush` takes a descriptor and a queue selector and touches
        // no memory; stdin is open for the life of the process.
        unsafe {
            libc::tcflush(libc::STDIN_FILENO, libc::TCIFLUSH);
        }
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Console::{
            FlushConsoleInputBuffer, GetStdHandle, STD_INPUT_HANDLE,
        };
        // SAFETY: GetStdHandle has no preconditions; flushing an invalid or
        // non-console handle fails harmlessly.
        unsafe {
            FlushConsoleInputBuffer(GetStdHandle(STD_INPUT_HANDLE));
        }
    }
}

/// A `Command` for the POSIX shell that runs model and user commands (`sh -c`).
///
/// Unix: `/bin/sh`, as always. Windows has no system POSIX shell, and the
/// model writes POSIX shell, so this looks for one in order: `PLANK_SHELL`, an
/// `sh.exe` or `bash.exe` on `PATH` (skipping the WSL launcher in System32,
/// which would run the command in a different filesystem), then Git for
/// Windows under the program-files folders. When none is found it falls back
/// to `sh`, so the spawn error names the missing shell.
#[must_use]
pub fn shell_command() -> std::process::Command {
    std::process::Command::new(shell_path())
}

#[cfg(unix)]
fn shell_path() -> std::path::PathBuf {
    std::path::PathBuf::from("/bin/sh")
}

#[cfg(windows)]
fn shell_path() -> std::path::PathBuf {
    use std::path::PathBuf;
    static SHELL: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    SHELL
        .get_or_init(|| {
            if let Some(s) = std::env::var_os("PLANK_SHELL").filter(|s| !s.is_empty()) {
                return PathBuf::from(s);
            }
            let system32 = std::env::var_os("SystemRoot")
                .map(|r| PathBuf::from(r).join("System32"))
                .map(|p| p.to_string_lossy().to_lowercase());
            let on_path = std::env::var_os("PATH")
                .into_iter()
                .flat_map(|p| std::env::split_paths(&p).collect::<Vec<_>>());
            for dir in on_path {
                let skip = system32
                    .as_deref()
                    .is_some_and(|s| dir.to_string_lossy().to_lowercase().starts_with(s));
                if skip {
                    continue;
                }
                for exe in ["sh.exe", "bash.exe"] {
                    let candidate = dir.join(exe);
                    if candidate.is_file() {
                        return candidate;
                    }
                }
            }
            for var in [
                "ProgramFiles",
                "ProgramW6432",
                "ProgramFiles(x86)",
                "LOCALAPPDATA",
            ] {
                let Some(root) = std::env::var_os(var) else {
                    continue;
                };
                let mut git = PathBuf::from(root);
                if var == "LOCALAPPDATA" {
                    git.push("Programs");
                }
                git.push("Git");
                git.push("bin");
                git.push("sh.exe");
                if git.is_file() {
                    return git;
                }
            }
            PathBuf::from("sh")
        })
        .clone()
}

/// Puts a child in a process group of its own, so the terminal's Ctrl-C
/// reaches plank alone and [`kill_tree`] can end every descendant.
pub trait CommandExt {
    /// `process_group(0)` on Unix, `CREATE_NEW_PROCESS_GROUP` on Windows.
    fn own_process_group(&mut self) -> &mut Self;
}

impl CommandExt for std::process::Command {
    fn own_process_group(&mut self) -> &mut Self {
        #[cfg(unix)]
        {
            std::os::unix::process::CommandExt::process_group(self, 0)
        }
        #[cfg(windows)]
        {
            const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
            std::os::windows::process::CommandExt::creation_flags(self, CREATE_NEW_PROCESS_GROUP)
        }
    }
}

/// Waits up to `timeout` for `pipe` (a child's stdout) to have bytes to read,
/// or to reach EOF. `false` on timeout or error.
#[must_use]
pub fn wait_readable(pipe: &std::process::ChildStdout, timeout: std::time::Duration) -> bool {
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd as _;
        let mut pfd = libc::pollfd {
            fd: pipe.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        let timeout_ms = libc::c_int::try_from(timeout.as_millis())
            .unwrap_or(libc::c_int::MAX)
            .saturating_add(1);
        // SAFETY: `pfd` is one valid pollfd for the duration of the call.
        unsafe { libc::poll(&raw mut pfd, 1, timeout_ms) > 0 }
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle as _;
        use windows_sys::Win32::System::Pipes::PeekNamedPipe;
        // Anonymous pipes cannot be waited on, so this polls their fill level.
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let mut avail = 0u32;
            // SAFETY: the handle is a live pipe owned by `pipe`; the buffer
            // pointers may be null when only the byte count is asked for.
            let ok = unsafe {
                PeekNamedPipe(
                    pipe.as_raw_handle(),
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null_mut(),
                    &raw mut avail,
                    std::ptr::null_mut(),
                )
            };
            // A failed peek is a broken (closed) pipe: readable, as EOF.
            if ok == 0 || avail > 0 {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
}

/// Windows: forcibly ends the process tree rooted at `pid` (`taskkill /T /F`).
/// There is no SIGTERM to offer first; console tools get no grace period.
#[cfg(windows)]
pub fn kill_tree(pid: u32) {
    let _ = std::process::Command::new("taskkill")
        .args(["/T", "/F", "/PID", &pid.to_string()])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_round_trips_known_instants() {
        let t = |year, month, day, hour, minute, second| LocalTime {
            year,
            month,
            day,
            hour,
            minute,
            second,
            utc_offset: 0,
        };
        assert_eq!(civil_to_unix(&t(1970, 1, 1, 0, 0, 0)), 0);
        assert_eq!(civil_to_unix(&t(2000, 3, 1, 0, 0, 0)), 951_868_800);
        assert_eq!(civil_to_unix(&t(2026, 10, 31, 12, 34, 56)), 1_793_450_096);
    }

    #[test]
    fn local_time_matches_utc_plus_offset() {
        let secs = 1_793_450_096;
        let lt = local_time(secs).expect("local time");
        assert_eq!(civil_to_unix(&lt) - lt.utc_offset, secs);
    }

    #[test]
    fn formats_pad_fields() {
        let lt = LocalTime {
            year: 2026,
            month: 3,
            day: 4,
            hour: 5,
            minute: 6,
            second: 7,
            utc_offset: 0,
        };
        assert_eq!(lt.date(), "2026-03-04");
        assert_eq!(lt.date_time(), "2026-03-04 05:06:07");
    }

    #[test]
    fn own_pid_is_alive_and_zero_is_not() {
        assert!(pid_alive(std::process::id()));
        assert!(!pid_alive(0));
    }

    #[test]
    fn reports_some_memory_and_page_size() {
        assert!(physical_memory().is_some_and(|b| b > 0));
        assert!(page_size().is_some_and(|b| b > 0));
        assert!(free_bytes(&std::env::temp_dir()).is_some());
    }
}
