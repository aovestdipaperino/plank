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

/// A thread standing in for the TUI: it takes the session `bridge` lends,
/// lets `edit` change the staged file on the RAM disk, and gives it back with
/// `close`. It panics if nothing is lent within ten seconds, so a regression
/// that never lends fails at `join` instead of hanging the test run.
#[cfg(test)]
pub(crate) fn ui_thread(
    bridge: crate::framebridge::FrameBridge,
    edit: Option<&'static [u8]>,
    close: FrameClose,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if let Some((mut session, req)) = bridge.take() {
                if let Some(bytes) = edit {
                    session
                        .host
                        .ram_write(&req.component, &req.file, bytes)
                        .unwrap();
                }
                bridge.give_back(session, close);
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "no session was lent within ten seconds"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    })
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
        assert_eq!(
            run_frame_blocking(&mut c, ID, "data.csv", b"a\n"),
            FrameResult::Saved(b"a\n1\n".to_vec())
        );
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
        assert_eq!(
            run_frame_blocking(&mut c, ID, "data.csv", b"a\n"),
            FrameResult::Unchanged
        );
        h.join().unwrap();
        let h = ui_thread(bridge, None, FrameClose::Failed("trapped".into()));
        assert_eq!(
            run_frame_blocking(&mut c, ID, "data.csv", b"a\n"),
            FrameResult::Failed("trapped".into())
        );
        h.join().unwrap();
        assert_eq!(c.wasm.host.ram_file(ID, "data.csv"), None);
    }
}
