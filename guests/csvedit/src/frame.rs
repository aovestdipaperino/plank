//! The Extism exports: plank's frame and command surfaces over one Session.
//!
//! Every reply is shaped to what the host parses in `src/wasmreg.rs`:
//! `command_run` answers `{"open": "<arg>"}` to open this component's own
//! frame, `frame_key` answers `{"close": "<line>"}` (a string) to close and
//! anything else to stay, and `frame_close` answers `{"scrollback": "<line>"}`
//! (a string, not an array), which the host prefers over the key's line.

use std::cell::RefCell;

use extism_pdk::*;
use plank_guest_support::{encode_cells, int, text};

use crate::disk::PlankDisk;
use crate::editor::Session;
use crate::keys::{payload_text, translate};

thread_local! {
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
    /// The file name and its text as they stood when the frame opened, so
    /// `frame_close` can diff against what is on disk when it closes.
    static ORIGINAL: RefCell<Option<(String, String)>> = const { RefCell::new(None) };
    /// The row-count summary from the last `frame_close`, consumed once by
    /// the next `tool_resume`.
    static LAST_SUMMARY: RefCell<Option<String>> = const { RefCell::new(None) };
    /// Set by `tool_call` to the RAM-disk name it named in the frame
    /// directive, and always cleared by the very next `frame_open` — but
    /// `ORIGINAL` is only recorded when that open's own `arg` matches, so a
    /// directive plank refused before calling `frame_open` at all (no
    /// bridge, a sub-agent, `editor_refusal`, a missing `files` grant, a
    /// containment/symlink refusal, over quota) cannot attach itself to a
    /// later, unrelated open. It is also cleared by `command_run` (a slash
    /// command means the tool path is over) and at the start of every
    /// `tool_call` (a fresh call starts clean, in case a previous one never
    /// got its `frame_open` either).
    ///
    /// The remaining edge case — a refused `edit_csv` directive followed by
    /// a later `frame_open` whose `arg` happens to also be `"data.csv"` —
    /// is accepted: grid-bridge files are never named `data.csv` and a
    /// `/csvedit` open uses the user's own file name, so only another
    /// tool-invoked open of the RAM-disk staging name could collide, and
    /// that is exactly the case this flag is meant to recognise anyway.
    static PENDING_TOOL_OPEN: RefCell<Option<String>> = const { RefCell::new(None) };
}

#[plugin_fn]
pub fn plank_abi() -> FnResult<String> {
    Ok("1".to_string())
}

#[plugin_fn]
pub fn command_specs() -> FnResult<String> {
    Ok(r#"[{"name": "new", "args": "", "desc": "edit a new CSV table"},
           {"name": "open", "args": "<name.csv>", "desc": "edit a CSV table from the scratch disk"}]"#
        .to_string())
}

/// `new` opens the frame with an empty arg, `open NAME` with arg `NAME`, and a
/// bare `open` (no name) with the [`crate::editor::OPEN_DIALOG_ARG`] sentinel
/// so the frame comes up showing the Open dialog. The host only ever opens
/// the calling component's own frame, so `open` names an arg, never a
/// component. An unknown command name is reported rather than silently
/// treated as `new`.
#[plugin_fn]
pub fn command_run(input: String) -> FnResult<String> {
    // A slash command means the tool path is over: a pending tool-invoked
    // open (if any) is now stale.
    PENDING_TOOL_OPEN.with(|p| *p.borrow_mut() = None);
    let name = text(&input, "name");
    let args = text(&input, "args");
    let arg = args.trim();
    Ok(match name.as_str() {
        "new" => r#"{"open": ""}"#.to_string(),
        "open" if arg.is_empty() => {
            format!(
                "{{\"open\": {}}}",
                json_string(crate::editor::OPEN_DIALOG_ARG)
            )
        }
        "open" => format!("{{\"open\": {}}}", json_string(arg)),
        other => format!(
            "{{\"print\": [{}]}}",
            json_string(&format!("csvedit: unknown command \"{other}\""))
        ),
    })
}

#[plugin_fn]
pub fn frame_open(input: String) -> FnResult<String> {
    let (w, h) = (dim(int(&input, "w")), dim(int(&input, "h")));
    let arg = text(&input, "arg");
    // Always clear the pending marker on the very next open, whatever it
    // opens: only when it names the same file the tool call staged is the
    // marker actually acted on, which is what rules out a stale marker from
    // a directive plank never honoured attaching to a later, unrelated open.
    let pending = PENDING_TOOL_OPEN.with(|p| p.borrow_mut().take());
    if pending.as_deref() == Some(arg.as_str()) {
        let original = crate::disk::Disk::read(&PlankDisk, &arg).unwrap_or_default();
        ORIGINAL.with(|o| *o.borrow_mut() = Some((arg.clone(), original)));
    }
    SESSION.with(|s| *s.borrow_mut() = Some(Session::open(w, h, &arg, Box::new(PlankDisk))));
    Ok("{}".to_string())
}

#[plugin_fn]
pub fn frame_key(input: String) -> FnResult<String> {
    let code = text(&input, "code");
    let Some(event) = translate(&code, payload_text(&input)) else {
        return Ok(r#"{"stay": true}"#.to_string());
    };
    let closing = SESSION.with(|s| s.borrow_mut().as_mut().and_then(|s| s.key(event)));
    Ok(match closing {
        Some(line) => format!("{{\"close\": {}}}", json_string(&line)),
        None => r#"{"stay": true}"#.to_string(),
    })
}

#[plugin_fn]
pub fn frame_step(input: String) -> FnResult<Vec<u8>> {
    let (w, h) = (dim(int(&input, "w")), dim(int(&input, "h")));
    let cells = SESSION.with(|s| {
        let mut s = s.borrow_mut();
        let session = s.as_mut()?;
        session.step(w, h);
        Some(session.cells())
    });
    Ok(encode_cells(&cells.unwrap_or_default(), w, h))
}

#[plugin_fn]
pub fn frame_close() -> FnResult<String> {
    if let Some((name, before)) = ORIGINAL.with(|o| o.borrow_mut().take()) {
        let now = crate::disk::Disk::read(&PlankDisk, &name).unwrap_or_default();
        LAST_SUMMARY.with(|s| *s.borrow_mut() = Some(crate::summary::summarize(&before, &now)));
    }
    let line = SESSION.with(|s| s.borrow_mut().take().map(|s| s.close_line()));
    Ok(match line {
        Some(l) => format!("{{\"scrollback\": {}}}", json_string(&l)),
        None => "{}".to_string(),
    })
}

/// Reads a string field out of the `"args": {...}` sub-object of a flat JSON
/// payload, the shape plank sends `tool_call` (`{"name": ..., "args":
/// {...}}`) in. `text` only ever finds the *first* occurrence of a key
/// anywhere in the payload, so scoping to `args` first keeps a same-named
/// top-level field from shadowing it (and vice versa).
///
/// This is a brace-depth scan, not a JSON parser: a path containing a
/// literal brace character would miscount depth and truncate. Acceptable
/// for `edit_csv`'s one string field, a file path, where a brace is not a
/// realistic input.
fn args_field(input: &str, key: &str) -> String {
    let Some((_, rest)) = input.split_once("\"args\":") else {
        return String::new();
    };
    let rest = rest.trim_start();
    let Some(mut body) = rest.strip_prefix('{') else {
        return String::new();
    };
    let mut depth = 1usize;
    let mut end = 0usize;
    for (i, c) in body.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }
    body = &body[..end];
    text(&format!("{{{body}}}"), key)
}

#[plugin_fn]
pub fn tool_specs() -> FnResult<String> {
    Ok(r#"[{"name": "edit_csv", "description": "Open a CSV file in a grid editor for the user to edit. Blocks until they close it and reports what changed; the file's contents are not returned. A missing file opens as an empty table.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The CSV file to edit."}}, "required": ["path"]}}]"#.to_string())
}

#[plugin_fn]
pub fn tool_call(input: String) -> FnResult<String> {
    // A fresh call starts clean, in case an earlier one's frame_open never
    // arrived (the host refused the directive before opening the frame).
    PENDING_TOOL_OPEN.with(|p| *p.borrow_mut() = None);
    let path = args_field(&input, "path");
    if path.is_empty() {
        return Ok("error: edit_csv needs a path".to_string());
    }
    let file = "data.csv";
    PENDING_TOOL_OPEN.with(|p| *p.borrow_mut() = Some(file.to_string()));
    Ok(format!(
        "{{\"frame\": {{\"path\": {}, \"file\": {}}}}}",
        json_string(&path),
        json_string(file)
    ))
}

#[plugin_fn]
pub fn tool_resume(input: String) -> FnResult<String> {
    let path = text(&input, "path");
    let error = text(&input, "error");
    let summary = LAST_SUMMARY.with(|s| s.borrow_mut().take());
    let rewrote =
        crate::summary::flag(&input, "changed") || crate::summary::flag(&input, "written");
    Ok(crate::summary::resume_line(
        &path,
        summary.as_deref(),
        &error,
        rewrote,
    ))
}

fn dim(v: u64) -> u16 {
    u16::try_from(v).unwrap_or(u16::MAX).max(1)
}

fn json_string(s: &str) -> String {
    use std::fmt::Write as _;
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
