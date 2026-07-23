// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Safe wrapper over the ds4 C engine's browser subsystem (`ds4_web.c`).
//!
//! The C `ds4_web` drives a real (headless-capable) Chrome over the Chrome
//! `DevTools` Protocol and extracts Markdown from the live page — the transport
//! plank's curl-based [`tools::web`](crate::tools::web) path deliberately does
//! not use. On `ds4_engine` builds plank routes `google_search`/`visit_page`
//! through this wrapper so results come from a genuine browser (which dodges
//! the bot challenges plain curl trips).
//!
//! Approval is handled entirely on the Rust side (the interactive gate with
//! its "Always allow" global consent), so the C `confirm` callback here always
//! answers yes — the browser is only ever created after Rust has approved it.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;

use crate::ffi;

/// A live browser session, owning a `*mut ds4_web`.
///
/// Created lazily on first web use and kept alive across turns (like the C
/// agent, which leaves Chrome running so repeat calls are cheap).
#[derive(Debug)]
pub struct WebBrowser {
    raw: *mut ffi::Ds4Web,
    // Keeps the home-dir C string alive for the lifetime of `raw`; the C side
    // copies it in `ds4_web_create`, but holding it is harmless and explicit.
    _home: CString,
}

// The handle is a plain owned pointer into C state that is only ever touched
// through `&mut self`; moving it between threads (as `ToolContext` may be) is
// sound because there is no shared aliasing.
unsafe impl Send for WebBrowser {}

/// Rust owns the approval decision, so the C-side confirm always allows.
unsafe extern "C" fn always_confirm(
    _privdata: *mut c_void,
    _message: *const c_char,
    _err: *mut c_char,
    _err_len: usize,
) -> c_int {
    1
}

impl WebBrowser {
    /// Creates a browser subsystem rooted at `home` (its profile lives under
    /// `<home>/.ds4/browser`). Chrome itself is not spawned until the first
    /// search/visit.
    ///
    /// # Errors
    /// Returns a message if the C subsystem could not be allocated.
    pub fn new(home: &Path) -> Result<Self, String> {
        let home_c = CString::new(home.to_string_lossy().as_bytes())
            .map_err(|_| "home path contains a NUL byte".to_string())?;
        let cfg = ffi::Ds4WebConfig {
            home_dir: home_c.as_ptr(),
            port: 0, // default debug port
            confirm: Some(always_confirm),
            confirm_privdata: std::ptr::null_mut(),
            log: None,
            log_privdata: std::ptr::null_mut(),
            cancel: None,
            cancel_privdata: std::ptr::null_mut(),
        };
        // SAFETY: `cfg` is a valid, fully-initialized config; the C side copies
        // the fields it keeps.
        let raw = unsafe { ffi::ds4_web_create(&raw const cfg) };
        if raw.is_null() {
            return Err("failed to initialize the web subsystem".to_string());
        }
        Ok(Self { raw, _home: home_c })
    }

    /// Runs a Google search through the live browser, returning the Markdown
    /// the C extractor produces.
    ///
    /// # Errors
    /// Returns the C error text (browser launch failure, navigation timeout…).
    pub fn google_search(&mut self, query: &str) -> Result<String, String> {
        // SAFETY: `raw` is a valid handle; `call` copies the C string.
        self.call(query, |web, q, err, len| unsafe {
            ffi::ds4_web_google_search(web, q, err, len)
        })
    }

    /// Renders a URL through the live browser to Markdown.
    ///
    /// # Errors
    /// Returns the C error text.
    pub fn visit_page(&mut self, url: &str) -> Result<String, String> {
        // SAFETY: `raw` is a valid handle; `call` copies the C string.
        self.call(url, |web, u, err, len| unsafe {
            ffi::ds4_web_visit_page(web, u, err, len)
        })
    }

    /// Shared FFI glue: pass `arg` as a C string, run `f`, and turn the
    /// malloc'd result / error buffer into an owned `Result<String, String>`.
    fn call(
        &mut self,
        arg: &str,
        f: impl FnOnce(*mut ffi::Ds4Web, *const c_char, *mut c_char, usize) -> *mut c_char,
    ) -> Result<String, String> {
        let arg_c = CString::new(arg).map_err(|_| "argument contains a NUL byte".to_string())?;
        let mut err = [0 as c_char; 512];
        let out = f(self.raw, arg_c.as_ptr(), err.as_mut_ptr(), err.len());
        if out.is_null() {
            // SAFETY: `err` is a NUL-terminated buffer the C side wrote into.
            let msg = unsafe { CStr::from_ptr(err.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err(if msg.is_empty() {
                "web tool failed".to_string()
            } else {
                msg
            });
        }
        // SAFETY: `out` is a malloc'd, NUL-terminated string owned by us now.
        let s = unsafe { CStr::from_ptr(out) }
            .to_string_lossy()
            .into_owned();
        // SAFETY: `out` was allocated by the C side for the caller to free.
        unsafe { libc::free(out.cast()) };
        Ok(s)
    }
}

impl Drop for WebBrowser {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `ds4_web_create` and is freed exactly once.
        unsafe { ffi::ds4_web_free(self.raw) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercises the FFI linkage and `ds4_web_config` layout: create allocates
    /// the subsystem (Chrome is not spawned until a search/visit), then drop
    /// frees it. A layout mismatch or missing symbol would crash or fail here.
    #[test]
    fn create_and_drop_does_not_spawn_chrome() {
        let dir = std::env::temp_dir().join(format!("plank-ds4web-{}", std::process::id()));
        let browser = WebBrowser::new(&dir).expect("web subsystem should allocate");
        drop(browser);
    }
}
