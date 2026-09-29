// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Blocking frames: a tool call that opens a WASM frame component and waits
//! for the user to close it, the way `ask` waits for an answer.

/// How a blocking frame ended, from the tool call's side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameResult {
    /// The file on the component's disk differs from what was staged.
    Saved(Vec<u8>),
    /// The file is as staged, or the frame deleted it.
    Unchanged,
    /// Nothing was opened, for this reason.
    Refused(String),
    /// The frame failed to open or trapped while up.
    Failed(String),
}

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// What the worker asks the UI to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameRequest {
    /// The frame component's id.
    pub component: String,
    /// The file on its RAM disk, passed to `frame_open` as the `arg`.
    pub file: String,
}

/// How the UI side says the frame ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameClose {
    /// The component closed it.
    Closed,
    /// It failed to open, or trapped while up.
    Failed(String),
}

#[derive(Debug, Default)]
struct Inner {
    request: Mutex<Option<(crate::wasmreg::Session, FrameRequest)>>,
    response: Mutex<Option<(crate::wasmreg::Session, FrameClose)>>,
    pending: AtomicBool,
}

/// The rendezvous between a tool call, blocked on the worker, and the TUI's
/// busy loop. The worker owns the agent, and so the WASM session, for the
/// whole turn; a frame needs `&mut` access to it on every key and tick. So
/// the worker *lends* the session: it parks it here and waits, the UI drives
/// the frame with it, and hands it back when the frame closes. Polls like
/// `AskBridge` does, for the same reasons: the worker has nothing else to do,
/// and the type stays trivially `Send`.
#[derive(Debug, Clone, Default)]
pub struct FrameBridge {
    inner: Arc<Inner>,
}

impl FrameBridge {
    /// A fresh, idle bridge.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Worker side: lends `session` for `req` and blocks until it comes back.
    #[must_use]
    pub fn lend(
        &self,
        session: crate::wasmreg::Session,
        req: FrameRequest,
    ) -> (crate::wasmreg::Session, FrameClose) {
        set(&self.inner.request, Some((session, req)));
        self.inner.pending.store(true, Ordering::SeqCst);
        loop {
            if let Some(back) = take(&self.inner.response) {
                self.inner.pending.store(false, Ordering::SeqCst);
                return back;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// UI side: true while a lent session waits to be driven or returned.
    #[must_use]
    pub fn is_pending(&self) -> bool {
        self.inner.pending.load(Ordering::SeqCst)
    }

    /// UI side: takes the lent session and its request.
    #[must_use]
    pub fn take(&self) -> Option<(crate::wasmreg::Session, FrameRequest)> {
        take(&self.inner.request)
    }

    /// UI side: returns the session, unblocking the worker.
    pub fn give_back(&self, session: crate::wasmreg::Session, close: FrameClose) {
        set(&self.inner.response, Some((session, close)));
    }
}

fn set<T>(m: &Mutex<Option<T>>, v: Option<T>) {
    *m.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = v;
}

fn take<T>(m: &Mutex<Option<T>>) -> Option<T> {
    m.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lent_session_comes_back_with_the_close() {
        let bridge = FrameBridge::new();
        let ui = bridge.clone();
        let handle = std::thread::spawn(move || {
            loop {
                if let Some((session, req)) = ui.take() {
                    assert_eq!(req.file, "data.csv");
                    ui.give_back(session, FrameClose::Closed);
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        });
        let req = FrameRequest {
            component: "dev.plank.x".into(),
            file: "data.csv".into(),
        };
        let (_session, close) = bridge.lend(crate::wasmreg::Session::default(), req);
        assert_eq!(close, FrameClose::Closed);
        assert!(!bridge.is_pending());
        handle.join().unwrap();
    }
}
