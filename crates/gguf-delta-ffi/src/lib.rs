//! C ABI over the `gguf-delta` crate. See `include/gguf_delta.h` for the
//! contract; this file is its implementation, function for function.
//!
//! Conventions: a handle is an opaque `Box<Delta>` behind a raw pointer;
//! strings handed out are `CString`s owned by the handle; every fallible call
//! takes a caller-provided error buffer and returns `-1` or null on failure.
//! Paths cross the boundary as NUL-terminated byte strings, which on Unix are
//! the OS path bytes verbatim.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{CStr, CString, c_char};
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;

use gguf_delta::{ChunkInfo, CreateOptions, Header};

/// An opened delta: header, chunk table, and the strings the accessors hand
/// out, allocated once.
#[derive(Debug)]
pub struct Delta {
    path: PathBuf,
    header: Header,
    chunks: Vec<ChunkInfo>,
    label: CString,
    base_name: CString,
    base_model: CString,
    base_source_url: CString,
    base_source_rev: CString,
    target_name: CString,
    target_sha256: CString,
}

/// Mirrors `ggd_report` in the header.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct Report {
    pub tensors_changed: u64,
    pub chunks: u64,
    pub bytes_spanned: u64,
    pub bytes_changed: u64,
    pub payload_bytes: u64,
}

fn cstring(s: &str) -> CString {
    CString::new(s.replace('\0', "")).unwrap_or_default()
}

/// Copies `msg` into the caller's buffer, NUL-terminated and truncated.
unsafe fn put_err(err: *mut c_char, err_len: usize, msg: &str) {
    if err.is_null() || err_len == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let n = bytes.len().min(err_len - 1);
    // SAFETY: the caller promised `err` points at `err_len` writable bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), err.cast::<u8>(), n);
        *err.add(n) = 0;
    }
}

/// Copies `s` into an out buffer the same way.
unsafe fn put_str(out: *mut c_char, out_len: usize, s: &[u8]) -> bool {
    if out.is_null() || out_len == 0 {
        return false;
    }
    let n = s.len().min(out_len - 1);
    // SAFETY: as `put_err`.
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), out.cast::<u8>(), n);
        *out.add(n) = 0;
    }
    n == s.len()
}

unsafe fn path_arg(p: *const c_char) -> Option<PathBuf> {
    if p.is_null() {
        return None;
    }
    // SAFETY: the caller passed a NUL-terminated string.
    let bytes = unsafe { CStr::from_ptr(p) }.to_bytes();
    Some(PathBuf::from(std::ffi::OsStr::from_bytes(bytes)))
}

unsafe fn handle<'a>(d: *const Delta) -> Option<&'a Delta> {
    // SAFETY: a non-null pointer came from `ggd_open` and has not been closed.
    unsafe { d.as_ref() }
}

/// `ggd_version`.
#[unsafe(no_mangle)]
pub extern "C" fn ggd_version() -> *const c_char {
    static V: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    V.as_ptr().cast()
}

/// `ggd_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggd_open(
    path: *const c_char,
    err: *mut c_char,
    err_len: usize,
) -> *mut Delta {
    // SAFETY: forwarded caller guarantees.
    let Some(path) = (unsafe { path_arg(path) }) else {
        unsafe { put_err(err, err_len, "path is NULL") };
        return std::ptr::null_mut();
    };
    let opened = gguf_delta::read_header(&path)
        .and_then(|(h, first)| gguf_delta::read_chunks(&path, &h, first).map(|c| (h, c)));
    match opened {
        Ok((header, chunks)) => {
            let d = Delta {
                label: cstring(&header.label),
                base_name: cstring(&header.base_name),
                base_model: cstring(&header.base_general),
                base_source_url: cstring(&header.base_source_url),
                base_source_rev: cstring(&header.base_source_rev),
                target_name: cstring(&header.target_name),
                target_sha256: cstring(&gguf_delta::hex(&header.target_sha256)),
                path,
                header,
                chunks,
            };
            Box::into_raw(Box::new(d))
        }
        Err(e) => {
            unsafe { put_err(err, err_len, &e.to_string()) };
            std::ptr::null_mut()
        }
    }
}

/// `ggd_close`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggd_close(d: *mut Delta) {
    if !d.is_null() {
        // SAFETY: `d` came from `Box::into_raw` in `ggd_open` and is closed once.
        drop(unsafe { Box::from_raw(d) });
    }
}

macro_rules! str_accessor {
    ($name:ident, $field:ident) => {
        #[doc = concat!("`", stringify!($name), "`.")]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(d: *const Delta) -> *const c_char {
            // SAFETY: forwarded caller guarantees.
            match unsafe { handle(d) } {
                Some(d) => d.$field.as_ptr(),
                None => c"".as_ptr(),
            }
        }
    };
}

str_accessor!(ggd_label, label);
str_accessor!(ggd_base_name, base_name);
str_accessor!(ggd_base_model, base_model);
str_accessor!(ggd_base_source_url, base_source_url);
str_accessor!(ggd_base_source_revision, base_source_rev);
str_accessor!(ggd_target_name, target_name);
str_accessor!(ggd_target_sha256, target_sha256);

/// `ggd_base_size`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggd_base_size(d: *const Delta) -> u64 {
    // SAFETY: forwarded caller guarantees.
    unsafe { handle(d) }.map_or(0, |d| d.header.base_size)
}

/// `ggd_chunk_count`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggd_chunk_count(d: *const Delta) -> usize {
    // SAFETY: forwarded caller guarantees.
    unsafe { handle(d) }.map_or(0, |d| d.chunks.len())
}

/// `ggd_chunk`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggd_chunk(
    d: *const Delta,
    i: usize,
    offset: *mut u64,
    len: *mut u64,
    payload_len: *mut u64,
) -> i32 {
    // SAFETY: forwarded caller guarantees; out-params are written only when non-null.
    let Some(c) = (unsafe { handle(d) }).and_then(|d| d.chunks.get(i)) else {
        return -1;
    };
    unsafe {
        if !offset.is_null() {
            *offset = c.offset;
        }
        if !len.is_null() {
            *len = c.len;
        }
        if !payload_len.is_null() {
            *payload_len = c.payload_len;
        }
    }
    0
}

/// `ggd_find_base`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggd_find_base(
    d: *const Delta,
    out: *mut c_char,
    out_len: usize,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded caller guarantees.
    let Some(d) = (unsafe { handle(d) }) else {
        unsafe { put_err(err, err_len, "handle is NULL") };
        return -1;
    };
    match gguf_delta::find_base(&d.header, &d.path, &[]) {
        Ok(base) => {
            if unsafe { put_str(out, out_len, base.as_os_str().as_bytes()) } {
                0
            } else {
                unsafe { put_err(err, err_len, "output buffer too small") };
                -1
            }
        }
        Err(e) => {
            unsafe { put_err(err, err_len, &e.to_string()) };
            -1
        }
    }
}

/// `ggd_check_base`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggd_check_base(
    d: *const Delta,
    base_path: *const c_char,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded caller guarantees.
    let (Some(d), Some(base)) = (unsafe { handle(d) }, unsafe { path_arg(base_path) }) else {
        unsafe { put_err(err, err_len, "NULL argument") };
        return -1;
    };
    match gguf_delta::check_base(&d.header, &base) {
        Ok(()) => 0,
        Err(e) => {
            unsafe { put_err(err, err_len, &e.to_string()) };
            -1
        }
    }
}

/// `ggd_materialize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggd_materialize(
    d: *const Delta,
    out_path: *const c_char,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded caller guarantees.
    let (Some(d), Some(out)) = (unsafe { handle(d) }, unsafe { path_arg(out_path) }) else {
        unsafe { put_err(err, err_len, "NULL argument") };
        return -1;
    };
    match gguf_delta::materialize(&d.path, &out, &[]) {
        Ok(_) => 0,
        Err(e) => {
            unsafe { put_err(err, err_len, &e.to_string()) };
            -1
        }
    }
}

/// `ggd_apply`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggd_apply(
    d: *const Delta,
    clone_path: *const c_char,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded caller guarantees.
    let (Some(d), Some(clone)) = (unsafe { handle(d) }, unsafe { path_arg(clone_path) }) else {
        unsafe { put_err(err, err_len, "NULL argument") };
        return -1;
    };
    let applied = gguf_delta::read_header(&d.path)
        .and_then(|(h, first)| gguf_delta::apply(&d.path, &h, first, &clone));
    match applied {
        Ok(()) => 0,
        Err(e) => {
            unsafe { put_err(err, err_len, &e.to_string()) };
            -1
        }
    }
}

/// `ggd_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggd_create(
    base_path: *const c_char,
    target_path: *const c_char,
    out_path: *const c_char,
    label: *const c_char,
    hash_base: i32,
    report: *mut Report,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded caller guarantees.
    let (Some(base), Some(target), Some(out)) = (
        unsafe { path_arg(base_path) },
        unsafe { path_arg(target_path) },
        unsafe { path_arg(out_path) },
    ) else {
        unsafe { put_err(err, err_len, "NULL path") };
        return -1;
    };
    let label = if label.is_null() {
        None
    } else {
        // SAFETY: non-null, NUL-terminated per the header contract.
        Some(
            unsafe { CStr::from_ptr(label) }
                .to_string_lossy()
                .into_owned(),
        )
    };
    let opts = CreateOptions {
        label,
        hash_base: hash_base != 0,
    };
    match gguf_delta::write_delta(&base, &target, &out, &opts) {
        Ok(r) => {
            if !report.is_null() {
                // SAFETY: caller-provided struct of the declared layout.
                unsafe {
                    *report = Report {
                        tensors_changed: r.tensors_changed as u64,
                        chunks: r.chunks as u64,
                        bytes_spanned: r.bytes_spanned,
                        bytes_changed: r.bytes_changed,
                        payload_bytes: r.payload_bytes,
                    };
                }
            }
            0
        }
        Err(e) => {
            unsafe { put_err(err, err_len, &e.to_string()) };
            -1
        }
    }
}

#[cfg(test)]
#[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use gguf_delta::testing::Gguf;
    use std::fs;

    fn cpath(p: &std::path::Path) -> CString {
        CString::new(p.as_os_str().as_bytes()).unwrap()
    }

    #[test]
    fn the_c_api_round_trips() {
        let d = std::env::temp_dir().join(format!("gguf-delta-ffi-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        let base_bytes = Gguf::default()
            .str_val("general.name", "Tiny")
            .tensor("a", &[64], 0, &[1u8; 64])
            .tensor("b", &[64], 0, &[2u8; 64])
            .bytes();
        let mut target_bytes = base_bytes.clone();
        let n = target_bytes.len();
        target_bytes[n - 64..].fill(9);
        let base = d.join("tiny.gguf");
        let target = d.join("tiny-edit.gguf");
        let out = d.join("tiny.ggd");
        fs::write(&base, &base_bytes).unwrap();
        fs::write(&target, &target_bytes).unwrap();

        let mut err = [0 as c_char; 256];
        let mut report = Report::default();
        let rc = unsafe {
            ggd_create(
                cpath(&base).as_ptr(),
                cpath(&target).as_ptr(),
                cpath(&out).as_ptr(),
                std::ptr::null(),
                0,
                &raw mut report,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        assert_eq!(rc, 0);
        assert_eq!(report.chunks, 1);
        assert_eq!(report.bytes_changed, 64);

        let h = unsafe { ggd_open(cpath(&out).as_ptr(), err.as_mut_ptr(), err.len()) };
        assert!(!h.is_null());
        let label = unsafe { CStr::from_ptr(ggd_label(h)) }.to_str().unwrap();
        assert_eq!(label, "edit");
        assert_eq!(
            unsafe { CStr::from_ptr(ggd_base_name(h)) }
                .to_str()
                .unwrap(),
            "tiny.gguf"
        );
        assert_eq!(
            unsafe { CStr::from_ptr(ggd_base_model(h)) }
                .to_str()
                .unwrap(),
            "Tiny"
        );
        assert_eq!(unsafe { ggd_base_size(h) }, base_bytes.len() as u64);
        assert_eq!(unsafe { ggd_chunk_count(h) }, 1);
        let (mut off, mut len) = (0u64, 0u64);
        assert_eq!(
            unsafe { ggd_chunk(h, 0, &raw mut off, &raw mut len, std::ptr::null_mut()) },
            0
        );
        assert_eq!((off as usize, len as usize), (n - 64, 64));
        assert_eq!(
            unsafe {
                ggd_chunk(
                    h,
                    1,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            },
            -1
        );
        assert_eq!(
            unsafe { CStr::from_ptr(ggd_target_sha256(h)) }
                .to_bytes()
                .len(),
            64
        );

        let mut found = [0 as c_char; 512];
        assert_eq!(
            unsafe {
                ggd_find_base(
                    h,
                    found.as_mut_ptr(),
                    found.len(),
                    err.as_mut_ptr(),
                    err.len(),
                )
            },
            0
        );
        assert_eq!(
            unsafe { CStr::from_ptr(found.as_ptr()) }.to_bytes(),
            base.as_os_str().as_bytes()
        );
        assert_eq!(
            unsafe { ggd_check_base(h, cpath(&base).as_ptr(), err.as_mut_ptr(), err.len()) },
            0
        );
        assert_eq!(
            unsafe { ggd_check_base(h, cpath(&target).as_ptr(), err.as_mut_ptr(), err.len()) },
            0,
            "same header"
        );
        assert_eq!(
            unsafe { ggd_check_base(h, cpath(&out).as_ptr(), err.as_mut_ptr(), err.len()) },
            -1
        );
        assert!(
            unsafe { CStr::from_ptr(err.as_ptr()) }
                .to_str()
                .unwrap()
                .contains("wrong size")
        );

        let built = d.join("built.gguf");
        assert_eq!(
            unsafe { ggd_materialize(h, cpath(&built).as_ptr(), err.as_mut_ptr(), err.len()) },
            0
        );
        assert_eq!(fs::read(&built).unwrap(), target_bytes);

        let clone = d.join("clone.gguf");
        fs::write(&clone, &base_bytes).unwrap();
        assert_eq!(
            unsafe { ggd_apply(h, cpath(&clone).as_ptr(), err.as_mut_ptr(), err.len()) },
            0
        );
        assert_eq!(fs::read(&clone).unwrap(), target_bytes);
        unsafe { ggd_close(h) };

        // Errors land in the buffer, truncated and terminated.
        let mut small = [0 as c_char; 8];
        let bad = unsafe { ggd_open(cpath(&base).as_ptr(), small.as_mut_ptr(), small.len()) };
        assert!(bad.is_null());
        assert_eq!(
            unsafe { CStr::from_ptr(small.as_ptr()) }.to_bytes().len(),
            7
        );
        assert!(
            !unsafe { CStr::from_ptr(ggd_version()) }
                .to_str()
                .unwrap()
                .is_empty()
        );
        let _ = fs::remove_dir_all(&d);
    }
}
