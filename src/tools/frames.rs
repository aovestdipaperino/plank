// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Running a WASM frame component as an editor inside a tool call.

use super::ToolContext;
use crate::framebridge::{FrameClose, FrameRequest, FrameResult};

/// A `tool_call` reply asking plank to run the component's frame on a real
/// file: `{"frame": {"path": <as the model gave it>, "file": <RAM-disk name>}}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameDirective {
    /// The file to edit, as the model named it.
    pub path: String,
    /// The name it is staged under on the component's RAM disk.
    pub file: String,
}

/// Recognises a frame directive; anything else is an ordinary tool output.
#[must_use]
pub fn parse_frame_directive(output: &str) -> Option<FrameDirective> {
    use crate::tools::mcp::{Json, json_parse};
    let json = json_parse(output.trim())?;
    let frame = json.get("frame")?;
    let field = |k: &str| match frame.get(k) {
        Some(Json::Str(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    };
    Some(FrameDirective {
        path: field("path")?,
        file: field("file")?,
    })
}

/// Honours `d` for `component`: checks the `files` grant and the path for
/// writing before anything else, refuses a target that is itself a symlink
/// (nothing here canonicalises it for us), stages the file (empty when
/// missing), runs the editor, re-checks containment and the symlink guard
/// immediately before the write (the editor runs for however long the human
/// takes, and the first check is stale by then), writes back only a changed
/// file, atomically, and lets the component report through its optional
/// `tool_resume` export.
pub fn run_edit_directive(ctx: &mut ToolContext, component: &str, d: FrameDirective) -> String {
    let FrameDirective { path, file } = d;
    let has_files = ctx.wasm.registry.loaded.iter().any(|l| {
        l.component.manifest.id == component
            && l.component
                .manifest
                .capabilities
                .contains(&crate::wasmreg::Capability::Files)
    });
    if !has_files {
        return format!("Tool error: {component} has no files grant\n");
    }
    let target = match resolve_write_target(ctx, &path) {
        Ok(p) => p,
        Err(e) => return format!("Tool error: {e}\n"),
    };
    let bytes = match std::fs::read(&target) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return format!("Tool error: cannot read {}: {e}\n", target.display()),
    };
    if bytes.len() > crate::wasmcaps::FS_MAX_FILE_BYTES {
        return format!(
            "Tool error: {} is too large for the editor ({} bytes; the limit is {})\n",
            target.display(),
            bytes.len(),
            crate::wasmcaps::FS_MAX_FILE_BYTES
        );
    }
    let (changed, written, error) = match run_frame_blocking(ctx, component, &file, &bytes) {
        // The editor was open for however long a human takes to close it;
        // `target` was resolved and checked before that wait started, so it
        // is re-resolved and re-checked here rather than trusted. A
        // background job can turn a parent directory into a symlink to
        // somewhere outside the write roots in the minutes in between, and
        // plank's own process is not sandboxed the way a model-initiated
        // `bash` call is — nothing else stands between a stale path and an
        // unsandboxed write.
        FrameResult::Saved(new) => match resolve_write_target(ctx, &path) {
            Ok(revalidated) => match write_atomically(&revalidated, &new) {
                Ok(()) => (true, true, None),
                Err(e) => (true, false, Some(e)),
            },
            Err(e) => (true, false, Some(e)),
        },
        FrameResult::Unchanged => (false, false, None),
        FrameResult::Refused(r) | FrameResult::Failed(r) => {
            return format!("Tool error: {r}\n");
        }
    };
    resume(ctx, component, &target, changed, written, error.as_deref())
}

/// Resolves `path` for writing and refuses a target that is itself a
/// symlink. Returns a bare reason (no `"Tool error: "` prefix, no trailing
/// newline) so every caller — the first check and the pre-write recheck
/// alike — can fold it into a uniform error line itself.
///
/// `resolve_for_write` joins `cwd` with the raw path; it does not
/// canonicalise its result, so a target that is itself a symlink would
/// otherwise be read through and then silently replaced by
/// `write_atomically`'s rename (which drops the link and leaves a plain
/// file in its place). Refuse it outright instead of guessing which side
/// the model meant.
fn resolve_write_target(ctx: &ToolContext, path: &str) -> Result<std::path::PathBuf, String> {
    let target = match ctx.resolve_for_write("frame", path) {
        Ok(p) => p,
        Err(e) => {
            let reason = e
                .strip_prefix("Tool error: ")
                .unwrap_or(&e)
                .trim_end_matches('\n')
                .to_string();
            return Err(reason);
        }
    };
    if let Ok(meta) = std::fs::symlink_metadata(&target)
        && meta.file_type().is_symlink()
    {
        return Err(format!(
            "{} is a symlink; refusing to edit through it",
            target.display()
        ));
    }
    Ok(target)
}

/// A sibling temp name unique to this process and call, so `a.csv` and
/// `a.tsv` (or two concurrent edits of the same file name in different
/// directories) never collide.
static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Builds that unique sibling name for `target`, from its full file name
/// (not `with_extension`, so two different extensions on the same stem
/// cannot collide) plus this process's id and a monotonic counter.
fn temp_sibling(target: &std::path::Path) -> std::path::PathBuf {
    let n = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let name = target
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("file");
    target.with_file_name(format!("{name}.{}.{n}.plank-edit.tmp", std::process::id()))
}

/// Opens `path` for writing only if nothing is there yet: `create_new`
/// refuses an existing path outright, including a dangling or live symlink,
/// rather than following it. This is what keeps a predictable temp name from
/// becoming a write-anywhere primitive.
fn create_new(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
}

/// Writes `bytes` to a fresh sibling of `target` and renames it into place.
///
/// The sibling is opened with [`create_new`], which refuses an already
/// existing path — including a dangling or live symlink planted at that
/// name — instead of following it, closing the containment hole an
/// attacker-controlled cwd (e.g. via a sandboxed `bash` call) would
/// otherwise have through a predictable temp name. The temp file is removed
/// on every failure path, including a failed write, and the original
/// file's permissions (when it existed) are carried over before the
/// rename so the replacement is not left more permissive than the file it
/// replaces.
fn write_atomically(target: &std::path::Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write as _;
    let tmp = temp_sibling(target);
    let mut file = create_new(&tmp).map_err(|e| format!("cannot create {}: {e}", tmp.display()))?;
    if let Err(e) = file.write_all(bytes) {
        drop(file);
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("cannot write {}: {e}", tmp.display()));
    }
    drop(file);
    if let Ok(original) = std::fs::metadata(target)
        && let Err(e) = std::fs::set_permissions(&tmp, original.permissions())
    {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!(
            "cannot preserve permissions on {}: {e}",
            tmp.display()
        ));
    }
    std::fs::rename(&tmp, target).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("cannot replace {}: {e}", target.display())
    })
}

/// The tool's final output: the component's `tool_resume` reply when it has
/// the export, else plank's own line. A trapping `tool_resume` costs the
/// component a strike, the same accounting `Registry::run_tool` applies to
/// a trapping `tool_call` (`Registry::strike`, `wasmreg.rs`), and still
/// reports the real outcome through plank's own line rather than swallowing
/// it. The trap's own text is not otherwise surfaced anywhere the model or
/// the user would see it — the same "log it, don't propagate it" trade-off
/// `Registry::dispatch` makes for a trapping event subscriber — so it is
/// printed to stderr, in the same `"{id}: {e}"` shape those call sites use.
fn resume(
    ctx: &mut ToolContext,
    component: &str,
    target: &std::path::Path,
    changed: bool,
    written: bool,
    error: Option<&str>,
) -> String {
    use crate::wasmreg::json_str;
    let path = target.display().to_string();
    let fallback = |error: Option<&str>| match (error, changed) {
        (Some(e), _) => format!("Tool error: {e}\n"),
        (None, true) => format!("saved changes to {path}\n"),
        (None, false) => format!("no changes to {path}\n"),
    };
    if ctx.wasm.host.has_export(component, "tool_resume") {
        let payload = format!(
            "{{\"path\": {}, \"changed\": {changed}, \"written\": {written}, \"error\": {}}}",
            json_str(&path),
            error.map_or_else(|| "null".to_string(), json_str)
        );
        match ctx
            .wasm
            .host
            .call(component, "tool_resume", payload.as_bytes())
        {
            Ok(bytes) => {
                let mut out = String::from_utf8_lossy(&bytes).into_owned();
                if !out.ends_with('\n') {
                    out.push('\n');
                }
                return out;
            }
            Err(e) => {
                eprintln!("plank: {component}: tool_resume trapped: {e}");
                ctx.wasm.registry.strike(component);
            }
        }
    }
    fallback(error)
}

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

/// Removes any old copy first, so a test that never cleans up after a panic
/// cannot leak state into a later run.
#[cfg(test)]
fn tempdir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("plank-frames-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
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
    fn a_frame_directive_is_recognised_and_anything_else_is_not() {
        assert_eq!(
            parse_frame_directive(r#"{"frame": {"path": "a.csv", "file": "data.csv"}}"#),
            Some(FrameDirective {
                path: "a.csv".into(),
                file: "data.csv".into()
            })
        );
        assert_eq!(parse_frame_directive("plain text"), None);
        assert_eq!(parse_frame_directive(r#"{"output": "x"}"#), None);
        assert_eq!(
            parse_frame_directive(r#"{"frame": {"path": "a.csv"}}"#),
            None
        );
    }

    fn files_ctx(dir: &std::path::Path) -> ToolContext {
        use crate::wasmreg::test_support::{editor_component, editor_session};
        use crate::wasmreg::{Capability, Surface};
        let mut ctx = ToolContext::new(dir.to_path_buf());
        ctx.wasm = editor_session(vec![editor_component(
            vec![Surface::Frame, Surface::Tool],
            vec![Capability::Fs, Capability::Files],
        )]);
        ctx
    }

    #[test]
    fn a_directive_without_the_files_grant_is_an_error() {
        use crate::wasmreg::test_support::{editor_component, editor_session};
        use crate::wasmreg::{Capability, Surface};
        let dir = tempdir("directive-nogrant");
        let mut ctx = ToolContext::new(dir.clone());
        ctx.wasm = editor_session(vec![editor_component(
            vec![Surface::Frame],
            vec![Capability::Fs],
        )]);
        let out = run_edit_directive(
            &mut ctx,
            ID,
            FrameDirective {
                path: "a.csv".into(),
                file: "data.csv".into(),
            },
        );
        assert!(out.contains("has no files grant"), "{out}");
    }

    #[test]
    fn an_edited_file_is_written_back_and_a_missing_one_starts_empty() {
        let dir = tempdir("directive-edit");
        let mut ctx = files_ctx(&dir);
        let bridge = crate::framebridge::FrameBridge::new();
        ctx.frame_bridge = Some(bridge.clone());
        let h = ui_thread(bridge, Some(b"a,b\n1,2\n"), FrameClose::Closed);
        let (_ctx, out) = lend_bounded(ctx, |ctx| {
            run_edit_directive(
                ctx,
                ID,
                FrameDirective {
                    path: "new.csv".into(),
                    file: "data.csv".into(),
                },
            )
        });
        h.join().unwrap();
        assert_eq!(std::fs::read(dir.join("new.csv")).unwrap(), b"a,b\n1,2\n");
        assert_eq!(
            out,
            format!("saved changes to {}\n", dir.join("new.csv").display())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unchanged_file_is_not_rewritten() {
        let dir = tempdir("directive-same");
        std::fs::write(dir.join("a.csv"), b"x\n").unwrap();
        let before = std::fs::metadata(dir.join("a.csv"))
            .unwrap()
            .modified()
            .unwrap();
        let mut ctx = files_ctx(&dir);
        let bridge = crate::framebridge::FrameBridge::new();
        ctx.frame_bridge = Some(bridge.clone());
        let h = ui_thread(bridge, None, FrameClose::Closed);
        let (_ctx, out) = lend_bounded(ctx, |ctx| {
            run_edit_directive(
                ctx,
                ID,
                FrameDirective {
                    path: "a.csv".into(),
                    file: "data.csv".into(),
                },
            )
        });
        h.join().unwrap();
        assert_eq!(
            std::fs::metadata(dir.join("a.csv"))
                .unwrap()
                .modified()
                .unwrap(),
            before
        );
        assert_eq!(
            out,
            format!("no changes to {}\n", dir.join("a.csv").display())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// No UI thread is ever started here on purpose: a quota check that
    /// staged (and so lent) before rejecting the file would hang instead of
    /// failing, and `lend_bounded` turns that hang into a timeout panic.
    #[test]
    fn a_file_over_the_disk_quota_is_refused_before_editing() {
        let dir = tempdir("directive-big");
        let big = vec![b'x'; crate::wasmcaps::FS_MAX_FILE_BYTES + 1];
        std::fs::write(dir.join("big.csv"), &big).unwrap();
        let mut ctx = files_ctx(&dir);
        ctx.frame_bridge = Some(crate::framebridge::FrameBridge::new());
        let (_ctx, out) = lend_bounded(ctx, |ctx| {
            run_edit_directive(
                ctx,
                ID,
                FrameDirective {
                    path: "big.csv".into(),
                    file: "data.csv".into(),
                },
            )
        });
        assert!(out.starts_with("Tool error: "), "{out}");
        assert!(out.contains("too large"), "{out}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A path outside every writable root is refused by `resolve_for_write`
    /// itself, before the file is read, the quota is checked, or anything
    /// is lent. No UI thread runs, and `lend_bounded` bounds the call so a
    /// regression that started lending anyway fails instead of hanging.
    #[test]
    fn a_directive_outside_the_write_roots_is_refused_and_never_lends() {
        let Some(home) = std::env::var_os("HOME") else {
            return;
        };
        let dir = tempdir("directive-escape");
        let mut ctx = files_ctx(&dir);
        ctx.sandbox.enabled = true;
        ctx.frame_bridge = Some(crate::framebridge::FrameBridge::new());
        let target =
            std::path::PathBuf::from(home).join(".plank/plank_frames_containment_test.csv");
        let directive_path = target.to_string_lossy().into_owned();
        let (_ctx, out) = lend_bounded(ctx, move |ctx| {
            run_edit_directive(
                ctx,
                ID,
                FrameDirective {
                    path: directive_path,
                    file: "data.csv".into(),
                },
            )
        });
        assert_eq!(
            out,
            format!(
                "Tool error: frame path escapes workspace: {}\n",
                target.display()
            )
        );
        assert!(!target.exists(), "a refused directive must not write");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A pre-existing symlink at the target itself must not be silently
    /// replaced by a regular file: `resolve_for_write` does not canonicalise
    /// its result (see `run_edit_directive`'s comment), so the check has to
    /// happen explicitly.
    #[cfg(unix)]
    #[test]
    fn a_symlinked_target_is_refused_not_replaced() {
        let dir = tempdir("directive-target-symlink");
        let outside = dir.join("outside.csv");
        std::fs::write(&outside, b"untouched\n").unwrap();
        std::os::unix::fs::symlink(&outside, dir.join("link.csv")).unwrap();
        let mut ctx = files_ctx(&dir);
        ctx.frame_bridge = Some(crate::framebridge::FrameBridge::new());
        let (_ctx, out) = lend_bounded(ctx, |ctx| {
            run_edit_directive(
                ctx,
                ID,
                FrameDirective {
                    path: "link.csv".into(),
                    file: "data.csv".into(),
                },
            )
        });
        assert!(out.contains("is a symlink"), "{out}");
        assert!(
            std::fs::symlink_metadata(dir.join("link.csv"))
                .unwrap()
                .file_type()
                .is_symlink(),
            "the link must survive untouched"
        );
        assert_eq!(std::fs::read(&outside).unwrap(), b"untouched\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `create_new` is the one thing standing between a predictable temp
    /// name and a write-anywhere primitive: it must refuse an existing path,
    /// including a symlink, rather than follow it.
    #[cfg(unix)]
    #[test]
    fn create_new_refuses_an_existing_path_including_a_symlink() {
        let dir = tempdir("create-new-symlink");
        let outside = dir.join("outside.txt");
        std::fs::write(&outside, b"untouched\n").unwrap();
        let link = dir.join("link");
        std::os::unix::fs::symlink(&outside, &link).unwrap();
        assert!(create_new(&link).is_err());
        assert_eq!(std::fs::read(&outside).unwrap(), b"untouched\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Containment is checked once before the editor opens, and the editor
    /// then runs for however long a human takes to close it — minutes, in
    /// the real TUI. A background `bash` job in the same session can spend
    /// that window replacing a parent directory of the target with a
    /// symlink to somewhere outside the write roots; plank's own process is
    /// not sandboxed the way a model-initiated `bash` call is, so a stale
    /// path is a real write-anywhere primitive, not just a theoretical one.
    ///
    /// The swap happens from inside the stand-in UI thread's edit step
    /// (after it has taken the lent session and written the edit, before
    /// giving it back), which is the one place guaranteed to run strictly
    /// between the first containment check and the rewrite: `run_edit_directive`
    /// finishes every check before the lend, and cannot resume past the
    /// `Saved` arm until the give-back unblocks it.
    #[cfg(unix)]
    #[test]
    fn a_parent_symlinked_to_outside_the_write_roots_while_the_editor_is_open_is_caught_before_the_write()
     {
        let Some(home) = std::env::var_os("HOME") else {
            return;
        };
        let dir = tempdir("directive-toctou");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        let mut ctx = files_ctx(&dir);
        ctx.sandbox.enabled = true;
        let bridge = crate::framebridge::FrameBridge::new();
        ctx.frame_bridge = Some(bridge.clone());
        let outside = std::path::PathBuf::from(home).join(format!(
            "plank_frames_toctou_outside_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&outside).unwrap();
        let sub = dir.join("sub");
        let outside_for_thread = outside.clone();
        let h = std::thread::spawn(move || {
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
                close: FrameClose::Closed,
            };
            if let Some(session) = guard.session.as_mut() {
                session
                    .host
                    .ram_write(&req.component, &req.file, b"a,b\n9,9\n")
                    .unwrap();
            }
            // The attack: a background job replaces `sub` with a symlink to
            // somewhere outside every write root while the editor is open.
            std::fs::remove_dir_all(&sub).unwrap();
            std::os::unix::fs::symlink(&outside_for_thread, &sub).unwrap();
        });
        let (_ctx, out) = lend_bounded(ctx, |ctx| {
            run_edit_directive(
                ctx,
                ID,
                FrameDirective {
                    path: "sub/a.csv".into(),
                    file: "data.csv".into(),
                },
            )
        });
        h.join().unwrap();
        assert!(
            !outside.join("a.csv").exists(),
            "must not write outside the write roots"
        );
        assert!(out.starts_with("Tool error: "), "{out}");
        assert!(out.contains("escapes workspace"), "{out}");
        let _ = std::fs::remove_dir_all(&outside);
        let _ = std::fs::remove_file(dir.join("sub"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A host whose `tool_resume` export exists but always traps, so the
    /// fallback path and the strike accounting can be tested without a real
    /// runtime. Everything else behaves like [`test_support::DiskHost`]'s
    /// RAM disk.
    #[derive(Debug, Default)]
    struct ResumeTrapHost {
        files: std::collections::BTreeMap<(String, String), Vec<u8>>,
    }

    impl crate::wasmhost::WasmHost for ResumeTrapHost {
        fn load(
            &mut self,
            _source: &str,
            _wasm: &[u8],
            _granted: &[&str],
        ) -> Result<crate::wasmhost::LoadedPlugin, crate::wasmhost::WasmError> {
            Err(crate::wasmhost::WasmError::Unsupported)
        }

        fn call(
            &mut self,
            _id: &str,
            export: &str,
            _input: &[u8],
        ) -> Result<Vec<u8>, crate::wasmhost::WasmError> {
            if export == "tool_resume" {
                return Err(crate::wasmhost::WasmError::Trap("resume boomed".into()));
            }
            Ok(b"{}".to_vec())
        }

        fn has_export(&self, _id: &str, export: &str) -> bool {
            export == "tool_resume"
        }

        fn ram_write(&mut self, id: &str, path: &str, bytes: &[u8]) -> Result<(), String> {
            let path = crate::wasmcaps::normalize_fs_path(path)?;
            self.files.insert((id.to_string(), path), bytes.to_vec());
            Ok(())
        }

        fn ram_remove(&mut self, id: &str, path: &str) {
            if let Ok(path) = crate::wasmcaps::normalize_fs_path(path) {
                self.files.remove(&(id.to_string(), path));
            }
        }

        fn ram_file(&self, id: &str, path: &str) -> Option<Vec<u8>> {
            let path = crate::wasmcaps::normalize_fs_path(path).ok()?;
            self.files.get(&(id.to_string(), path)).cloned()
        }
    }

    /// A trapping `tool_resume` must not be swallowed: the fallback line
    /// still reports the real outcome, and the component is struck the same
    /// way a trapping `tool_call` would be (`Registry::run_tool`).
    #[test]
    fn a_trapping_tool_resume_falls_back_and_strikes() {
        let dir = tempdir("directive-resume-trap");
        let mut ctx = files_ctx(&dir);
        ctx.wasm.host = Box::new(ResumeTrapHost::default());
        let bridge = crate::framebridge::FrameBridge::new();
        ctx.frame_bridge = Some(bridge.clone());
        let h = ui_thread(bridge, Some(b"a,b\n1,2\n"), FrameClose::Closed);
        let (ctx, out) = lend_bounded(ctx, |ctx| {
            run_edit_directive(
                ctx,
                ID,
                FrameDirective {
                    path: "a.csv".into(),
                    file: "data.csv".into(),
                },
            )
        });
        h.join().unwrap();
        assert_eq!(
            out,
            format!("saved changes to {}\n", dir.join("a.csv").display())
        );
        let strikes = ctx
            .wasm
            .registry
            .loaded
            .iter()
            .find(|l| l.component.manifest.id == ID)
            .unwrap()
            .strikes;
        assert_eq!(strikes, 1, "a trapping tool_resume costs a strike");
        let _ = std::fs::remove_dir_all(&dir);
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
