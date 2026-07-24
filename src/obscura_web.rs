// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Embedded obscura headless-browser transport for the web tools.
//!
//! With the `use_obscura` feature (default) the `google_search` and
//! `visit_page` tools fetch pages through the obscura headless browser
//! (`refs/obscura` submodule) statically linked into the plank binary — no
//! external Chrome and no separate obscura binary. Obscura's async API is
//! driven from plank's synchronous tool code by a small current-thread tokio
//! runtime created per fetch; the browser itself is cached per session so
//! cookies and per-site state survive across tool calls.
//!
//! Only the transport changes: the fetched HTML flows into the same pure-Rust
//! extraction and rendering code in [`crate::tools::web`], so the
//! model-visible output formats are unchanged.

use std::sync::Mutex;

/// Milliseconds obscura waits for the page network to settle after load.
const SETTLE_MS: u64 = 2_000;

/// Session-cached browser instance (obscura pages are cheap; the browser
/// carries the cookie jar and stealth state worth keeping warm).
static BROWSER: Mutex<Option<obscura::Browser>> = Mutex::new(None);

/// Fetches `url` in the embedded headless browser and returns the page HTML
/// after JavaScript has run and the network has settled.
///
/// # Errors
///
/// Returns a message when the runtime, browser, or navigation fails.
///
/// # Panics
///
/// Panics if the internal browser cache mutex is poisoned.
pub fn fetch(url: &str) -> Result<String, String> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("failed to start async runtime: {e}"))?;
    let mut guard = BROWSER.lock().expect("obscura browser mutex poisoned");
    if guard.is_none() {
        *guard = Some(
            obscura::Browser::builder()
                .stealth(true)
                .build()
                .map_err(|e| format!("failed to start obscura browser: {e}"))?,
        );
    }
    let browser = guard.as_ref().expect("browser was just set");
    rt.block_on(async {
        let mut page = browser
            .new_page()
            .await
            .map_err(|e| format!("failed to open page: {e}"))?;
        page.goto(url)
            .await
            .map_err(|e| format!("failed to load {url}: {e}"))?;
        page.settle(SETTLE_MS).await;
        Ok(page.content())
    })
}
