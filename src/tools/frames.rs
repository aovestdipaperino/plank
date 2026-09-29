// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Running a WASM frame component as an editor inside a tool call.

use super::ToolContext;
use crate::framebridge::{FrameClose, FrameRequest, FrameResult};

/// Runs `component` as an editor on `bytes`, staged as `file` on its RAM
/// disk, and blocks until the user closes it.
///
/// Refused, with nothing staged or lent, outside the TUI, when the agent set
/// a refusal, inside a sub-agent, when an editor is already open, or when the
/// component cannot edit. The file is removed from the disk in every case.
pub fn run_frame_blocking(
    ctx: &mut ToolContext,
    component: &str,
    file: &str,
    bytes: &[u8],
) -> FrameResult {
    let Some(bridge) = ctx.frame_bridge.clone() else {
        return FrameResult::Refused("editors need the interactive TUI".to_string());
    };
    if let Some(reason) = ctx.editor_refusal.clone() {
        return FrameResult::Refused(reason);
    }
    if ctx.subagent_depth > 0 {
        return FrameResult::Refused("editors cannot open inside a sub-agent".to_string());
    }
    if bridge.is_pending() || ctx.wasm.frame_is_open() {
        return FrameResult::Refused("an editor is already open".to_string());
    }
    if let Err(reason) = ctx.wasm.check_editor(component) {
        return FrameResult::Refused(reason);
    }
    if let Err(reason) = ctx.wasm.stage_editor_file(component, file, bytes) {
        return FrameResult::Refused(reason);
    }
    let session = std::mem::take(&mut ctx.wasm);
    let req = FrameRequest {
        component: component.to_string(),
        file: file.to_string(),
    };
    let (session, close) = bridge.lend(session, req);
    ctx.wasm = session;
    let collected = ctx.wasm.collect_editor_file(component, file, bytes);
    match close {
        FrameClose::Closed => collected,
        FrameClose::Failed(e) => FrameResult::Failed(e),
    }
}

/// A context whose WASM session holds one editor component,
/// `dev.plank.csvedit`, with a frame surface and the `fs` capability.
#[cfg(test)]
pub(crate) fn editor_ctx() -> ToolContext {
    use crate::wasmreg::test_support::{editor_component, editor_session};
    use crate::wasmreg::{Capability, Surface};
    let mut ctx = ToolContext::new(std::env::temp_dir());
    ctx.wasm = editor_session(vec![editor_component(
        vec![Surface::Frame],
        vec![Capability::Fs],
    )]);
    ctx
}

/// How long a test waits for either side of a lend: the stand-in TUI for a
/// session to be lent, and the lender for the call to return. Generous, so
/// only a lend that never happens or never comes back trips it.
#[cfg(test)]
pub(crate) const LEND_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// Gives a taken session back when dropped, so a stand-in TUI that panics
/// between `take` and `give_back` still releases the lender (with
/// `FrameClose::Failed`) instead of leaving it spinning.
#[cfg(test)]
struct GiveBack {
    bridge: crate::framebridge::FrameBridge,
    session: Option<crate::wasmreg::Session>,
    close: FrameClose,
}

#[cfg(test)]
impl Drop for GiveBack {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            let close = if std::thread::panicking() {
                FrameClose::Failed("the stand-in TUI panicked".to_string())
            } else {
                self.close.clone()
            };
            self.bridge.give_back(session, close);
        }
    }
}

/// A thread standing in for the TUI: it serves one lend per entry of
/// `edits`, in order, each time letting the entry change the staged file on
/// the RAM disk and giving the session back with `close`.
///
/// The give-back runs from a drop guard, so a panic after `take` still
/// releases the lender. The thread panics if a lend does not arrive within
/// [`LEND_WAIT`]; that only makes `join` fail, and does not unblock a lender
/// that lends later, so the lending side must be bounded too
/// ([`lend_bounded`]).
#[cfg(test)]
pub(crate) fn ui_thread_serving(
    bridge: crate::framebridge::FrameBridge,
    edits: Vec<Option<&'static [u8]>>,
    close: FrameClose,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        for edit in edits {
            let deadline = std::time::Instant::now() + LEND_WAIT;
            let (session, req) = loop {
                if let Some(lent) = bridge.take() {
                    break lent;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "no session was lent within {LEND_WAIT:?}"
                );
                std::thread::sleep(std::time::Duration::from_millis(1));
            };
            let mut guard = GiveBack {
                bridge: bridge.clone(),
                session: Some(session),
                close: close.clone(),
            };
            if let (Some(bytes), Some(session)) = (edit, guard.session.as_mut()) {
                session
                    .host
                    .ram_write(&req.component, &req.file, bytes)
                    .unwrap();
            }
        }
    })
}

/// [`ui_thread_serving`] for a single lend.
#[cfg(test)]
pub(crate) fn ui_thread(
    bridge: crate::framebridge::FrameBridge,
    edit: Option<&'static [u8]>,
    close: FrameClose,
) -> std::thread::JoinHandle<()> {
    ui_thread_serving(bridge, vec![edit], close)
}

/// Runs `call` on `ctx` from a thread of its own, as the worker would, and
/// waits at most [`LEND_WAIT`] for it: a lend nobody gives back becomes a
/// test failure instead of a hung run (the stuck thread is leaked).
#[cfg(test)]
pub(crate) fn lend_bounded<T: Send + 'static>(
    mut ctx: ToolContext,
    call: impl FnOnce(&mut ToolContext) -> T + Send + 'static,
) -> (ToolContext, T) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let out = call(&mut ctx);
        let _ = tx.send((ctx, out));
    });
    rx.recv_timeout(LEND_WAIT)
        .expect("the lent session never came back")
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "dev.plank.csvedit";

    #[test]
    fn without_a_bridge_nothing_opens() {
        let mut c = editor_ctx();
        assert_eq!(
            run_frame_blocking(&mut c, ID, "data.csv", b"a\n"),
            FrameResult::Refused("editors need the interactive TUI".into())
        );
    }

    #[test]
    fn the_agents_refusal_wins_and_lends_nothing() {
        let mut c = editor_ctx();
        c.frame_bridge = Some(crate::framebridge::FrameBridge::new());
        c.editor_refusal = Some("the editor needs the local screen".into());
        assert_eq!(
            run_frame_blocking(&mut c, ID, "data.csv", b"a\n"),
            FrameResult::Refused("the editor needs the local screen".into())
        );
        assert_eq!(c.wasm.host.ram_file(ID, "data.csv"), None, "nothing staged");
    }

    #[test]
    fn a_sub_agent_cannot_open_an_editor() {
        let mut c = editor_ctx();
        c.frame_bridge = Some(crate::framebridge::FrameBridge::new());
        c.subagent_depth = 1;
        assert_eq!(
            run_frame_blocking(&mut c, ID, "data.csv", b"a\n"),
            FrameResult::Refused("editors cannot open inside a sub-agent".into())
        );
    }

    #[test]
    fn a_component_check_failure_is_a_refusal() {
        let mut c = editor_ctx();
        c.frame_bridge = Some(crate::framebridge::FrameBridge::new());
        assert_eq!(
            run_frame_blocking(&mut c, "dev.plank.nobody", "data.csv", b"a\n"),
            FrameResult::Refused("dev.plank.nobody is not loaded".into())
        );
    }

    #[test]
    fn an_edit_comes_back_saved_and_the_session_is_restored() {
        let mut c = editor_ctx();
        let bridge = crate::framebridge::FrameBridge::new();
        c.frame_bridge = Some(bridge.clone());
        let h = ui_thread(bridge, Some(b"a\n1\n"), FrameClose::Closed);
        let (c, out) = lend_bounded(c, |c| run_frame_blocking(c, ID, "data.csv", b"a\n"));
        assert_eq!(out, FrameResult::Saved(b"a\n1\n".to_vec()));
        h.join().unwrap();
        assert!(
            c.wasm.check_editor(ID).is_ok(),
            "the real session came back"
        );
        assert_eq!(
            c.wasm.host.ram_file(ID, "data.csv"),
            None,
            "the file is removed"
        );
    }

    #[test]
    fn an_untouched_editor_is_unchanged_and_a_failure_says_why() {
        let mut c = editor_ctx();
        let bridge = crate::framebridge::FrameBridge::new();
        c.frame_bridge = Some(bridge.clone());
        let h = ui_thread(bridge.clone(), None, FrameClose::Closed);
        let (c, out) = lend_bounded(c, |c| run_frame_blocking(c, ID, "data.csv", b"a\n"));
        assert_eq!(out, FrameResult::Unchanged);
        h.join().unwrap();
        let h = ui_thread(bridge, None, FrameClose::Failed("trapped".into()));
        let (c, out) = lend_bounded(c, |c| run_frame_blocking(c, ID, "data.csv", b"a\n"));
        assert_eq!(out, FrameResult::Failed("trapped".into()));
        h.join().unwrap();
        assert_eq!(c.wasm.host.ram_file(ID, "data.csv"), None);
    }
}
