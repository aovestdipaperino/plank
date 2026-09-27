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
    let line = SESSION.with(|s| s.borrow_mut().take().map(|s| s.close_line()));
    Ok(match line {
        Some(l) => format!("{{\"scrollback\": {}}}", json_string(&l)),
        None => "{}".to_string(),
    })
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
