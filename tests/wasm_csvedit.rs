//! End-to-end tests for the `guests/csvedit` frame component.
//!
//! These load the real `.wasm` into plank's host, so they run only with
//! `--features plugins` **and** only when `guests/build.sh` has built the
//! guest. Without the artifact they skip and say why, unless
//! `PLANK_REQUIRE_GUESTS` is set, in which case a missing guest fails.

#![cfg(feature = "plugins")]

use plank::wasmhost::{WasmHost, host};

const ID: &str = "dev.plank.csvedit";

/// The built guest, or `None` when nobody has built it.
fn guest() -> Option<Vec<u8>> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/guests/csvedit/target/wasm32-wasip1/release/plank_csvedit.wasm"
    );
    std::fs::read(path).ok()
}

/// True when a missing guest must fail rather than skip.
fn guests_required() -> bool {
    std::env::var_os("PLANK_REQUIRE_GUESTS").is_some_and(|v| v != "0")
}

macro_rules! guest_or_skip {
    () => {
        match guest() {
            Some(w) => w,
            None => {
                assert!(
                    !guests_required(),
                    "PLANK_REQUIRE_GUESTS is set but the guest is missing: \
                     run guests/build.sh before the test step"
                );
                eprintln!("skipping: run guests/build.sh first");
                return;
            }
        }
    };
}

/// Sends one `frame_key`, shaped as the host sends it, and returns the reply.
fn key(h: &mut (dyn WasmHost + Send), code: &str, text: Option<char>) -> String {
    let payload = match text {
        Some(c) => format!("{{\"code\": \"{code}\", \"text\": \"{c}\"}}"),
        None => format!("{{\"code\": \"{code}\"}}"),
    };
    String::from_utf8(
        h.call(ID, "frame_key", payload.as_bytes())
            .expect("frame_key"),
    )
    .unwrap()
}

#[test]
fn csvedit_edits_a_cell_and_saves_to_the_ram_disk() {
    let wasm = guest_or_skip!();
    let mut h = host(None);
    h.load(ID, &wasm, &["fs", "log"]).expect("load");
    h.call(
        ID,
        "frame_open",
        br#"{"w": 80, "h": 24, "seed": 1, "arg": "", "config": {}}"#,
    )
    .expect("frame_open");
    let frame = h
        .call(
            ID,
            "frame_step",
            br#"{"dt_ms": 16, "w": 80, "h": 24, "now_ms": 0}"#,
        )
        .expect("frame_step");
    let decoded = plank::wasmglyph::decode(&frame).expect("a glyph buffer");
    assert_eq!(decoded.glyphs.len(), 80 * 24, "every cell painted");
    assert!(
        decoded.bg.iter().all(Option::is_some),
        "backgrounds included"
    );

    let h = h.as_mut();
    key(h, "enter", None);
    for c in "42".chars() {
        key(h, &c.to_string(), Some(c));
    }
    key(h, "enter", None);
    key(h, "ctrl-s", None);
    for c in "t.csv".chars() {
        key(h, &c.to_string(), Some(c));
    }
    key(h, "enter", None);
    assert_eq!(
        h.ram_file(ID, "/t.csv").as_deref(),
        Some(&b"A,B,C\n42,,\n,,\n,,\n"[..])
    );
    let out = key(h, "alt-x", None);
    assert!(
        out.starts_with(r#"{"close": "csvedit:"#),
        "expected a close reply naming the editor: {out}"
    );

    // The host calls frame_close after a closing key and prefers its line.
    let closed = String::from_utf8(h.call(ID, "frame_close", b"").expect("frame_close")).unwrap();
    assert!(
        closed.contains("\"scrollback\": \"csvedit: saved"),
        "{closed}"
    );
}

/// The Mac chord for Exit: a stock Mac terminal types Option+X as a
/// character, so Ctrl+Q is the exit a Mac keyboard can reach.
#[test]
fn csvedit_ctrl_q_closes_a_clean_frame() {
    let wasm = guest_or_skip!();
    let mut h = host(None);
    h.load(ID, &wasm, &["fs", "log"]).expect("load");
    h.call(
        ID,
        "frame_open",
        br#"{"w": 80, "h": 24, "seed": 1, "arg": "", "config": {}}"#,
    )
    .expect("frame_open");
    let out = key(h.as_mut(), "ctrl-q", None);
    assert!(
        out.starts_with(r#"{"close": "csvedit:"#),
        "expected a close reply naming the editor: {out}"
    );
}

#[test]
fn csvedit_command_specs_name_both_commands() {
    let wasm = guest_or_skip!();
    let mut h = host(None);
    h.load(ID, &wasm, &["fs", "log"]).expect("load");
    let specs = String::from_utf8(h.call(ID, "command_specs", b"").unwrap()).unwrap();
    assert!(
        specs.contains("\"new\"") && specs.contains("\"open\""),
        "{specs}"
    );
}

#[test]
fn csvedit_commands_open_the_frame_with_the_file_name() {
    let wasm = guest_or_skip!();
    let mut h = host(None);
    h.load(ID, &wasm, &["fs", "log"]).expect("load");
    let run = |h: &mut Box<dyn WasmHost + Send>, payload: &str| {
        String::from_utf8(h.call(ID, "command_run", payload.as_bytes()).unwrap()).unwrap()
    };
    assert_eq!(
        run(&mut h, r#"{"name": "new", "args": ""}"#),
        r#"{"open": ""}"#
    );
    assert_eq!(
        run(&mut h, r#"{"name": "open", "args": "  t.csv "}"#),
        r#"{"open": "t.csv"}"#
    );
    // A bare "open" (no name) opens the frame with the "/" sentinel, which
    // Session::open reads as "blank doc, show the Open dialog".
    assert_eq!(
        run(&mut h, r#"{"name": "open", "args": ""}"#),
        r#"{"open": "/"}"#
    );
    assert_eq!(
        run(&mut h, r#"{"name": "open", "args": "   "}"#),
        r#"{"open": "/"}"#
    );
    // An unknown command is reported rather than silently opening a document.
    let unknown = run(&mut h, r#"{"name": "bogus", "args": ""}"#);
    assert!(unknown.contains("\"print\""), "{unknown}");
    assert!(unknown.contains("bogus"), "{unknown}");
}

#[test]
fn csvedit_tool_call_replies_with_a_frame_directive() {
    let wasm = guest_or_skip!();
    let mut h = host(None);
    h.load(ID, &wasm, &["fs", "files", "log"]).expect("load");
    let out = String::from_utf8(
        h.call(
            ID,
            "tool_call",
            br#"{"name": "edit_csv", "args": {"path": "x.csv"}}"#,
        )
        .expect("tool_call"),
    )
    .unwrap();
    let directive = plank::tools::frames::parse_frame_directive(&out).expect("a frame directive");
    assert_eq!(directive.path, "x.csv");
    assert_eq!(directive.file, "data.csv");
}

/// The full tool round-trip: `tool_call` stages the directive, `frame_open`
/// loads the staged file, a key sequence edits one cell and saves, and
/// `frame_close` + `tool_resume` reports the row-count summary — only
/// because `tool_call` ran first (the gate from fix round 1: an ordinary
/// grid-bridge or `/csvedit` open never pays for this).
#[test]
fn csvedit_tool_edit_reports_a_changed_row_through_resume() {
    let wasm = guest_or_skip!();
    let mut h = host(None);
    h.load(ID, &wasm, &["fs", "files", "log"]).expect("load");

    let out = String::from_utf8(
        h.call(
            ID,
            "tool_call",
            br#"{"name": "edit_csv", "args": {"path": "x.csv"}}"#,
        )
        .expect("tool_call"),
    )
    .unwrap();
    let directive = plank::tools::frames::parse_frame_directive(&out).expect("a frame directive");
    assert_eq!(directive.file, "data.csv");

    h.ram_write(ID, "/data.csv", b"A,B,C\n1,2,3\n")
        .expect("stage the file plank would have written");

    h.call(
        ID,
        "frame_open",
        br#"{"w": 80, "h": 24, "seed": 1, "arg": "data.csv", "config": {}}"#,
    )
    .expect("frame_open");

    let hm = h.as_mut();
    key(hm, "enter", None); // begin editing the first cell
    for c in "42".chars() {
        key(hm, &c.to_string(), Some(c));
    }
    key(hm, "enter", None); // commit the cell
    key(hm, "ctrl-s", None); // save: name is already "data.csv", no dialog

    assert_eq!(
        hm.ram_file(ID, "/data.csv").as_deref(),
        Some(&b"A,B,C\n42,2,3\n"[..]),
        "the edit landed on the RAM disk"
    );

    let closed = String::from_utf8(h.call(ID, "frame_close", b"").unwrap()).unwrap();
    assert!(closed.contains("saved"), "{closed}");

    let resumed = String::from_utf8(
        h.call(
            ID,
            "tool_resume",
            br#"{"path": "x.csv", "changed": true, "written": true, "error": null}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(resumed, "1 row changed in x.csv");
}

/// A close with no edits at all reports "no changes", not a stale summary
/// from an earlier tool-invoked edit.
#[test]
fn csvedit_tool_resume_reports_no_changes_after_a_clean_close() {
    let wasm = guest_or_skip!();
    let mut h = host(None);
    h.load(ID, &wasm, &["fs", "files", "log"]).expect("load");

    h.call(
        ID,
        "tool_call",
        br#"{"name": "edit_csv", "args": {"path": "x.csv"}}"#,
    )
    .expect("tool_call");
    h.ram_write(ID, "/data.csv", b"A,B,C\n1,2,3\n")
        .expect("stage");
    h.call(
        ID,
        "frame_open",
        br#"{"w": 80, "h": 24, "seed": 1, "arg": "data.csv", "config": {}}"#,
    )
    .expect("frame_open");
    h.call(ID, "frame_close", b"").expect("frame_close");

    let resumed = String::from_utf8(
        h.call(
            ID,
            "tool_resume",
            br#"{"path": "x.csv", "changed": false, "written": false, "error": null}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(resumed, "no changes to x.csv");
}

/// Saving an untouched CRLF file rewrites it as LF (the writer always emits
/// LF) while every row compares equal, so the host reports `changed` and the
/// row summary is empty: the resume line must say the file was rewritten, not
/// `no changes`.
#[test]
fn csvedit_tool_resume_reports_a_rewrite_with_no_row_changes() {
    let wasm = guest_or_skip!();
    let mut h = host(None);
    h.load(ID, &wasm, &["fs", "files", "log"]).expect("load");

    h.call(
        ID,
        "tool_call",
        br#"{"name": "edit_csv", "args": {"path": "x.csv"}}"#,
    )
    .expect("tool_call");
    h.ram_write(ID, "/data.csv", b"A,B,C\r\n1,2,3\r\n")
        .expect("stage");
    h.call(
        ID,
        "frame_open",
        br#"{"w": 80, "h": 24, "seed": 1, "arg": "data.csv", "config": {}}"#,
    )
    .expect("frame_open");
    key(h.as_mut(), "ctrl-s", None);
    assert_eq!(
        h.as_mut().ram_file(ID, "/data.csv").as_deref(),
        Some(&b"A,B,C\n1,2,3\n"[..]),
        "the save rewrote the line endings"
    );
    h.call(ID, "frame_close", b"").expect("frame_close");

    let resumed = String::from_utf8(
        h.call(
            ID,
            "tool_resume",
            br#"{"path": "x.csv", "changed": true, "written": true, "error": null}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(resumed, "rewrote x.csv (no row changes)");
}

/// A host-reported error (e.g. the write failed) is surfaced alongside
/// whatever the editor itself observed.
#[test]
fn csvedit_tool_resume_surfaces_a_host_error() {
    let wasm = guest_or_skip!();
    let mut h = host(None);
    h.load(ID, &wasm, &["fs", "files", "log"]).expect("load");

    h.call(
        ID,
        "tool_call",
        br#"{"name": "edit_csv", "args": {"path": "x.csv"}}"#,
    )
    .expect("tool_call");
    h.ram_write(ID, "/data.csv", b"A,B,C\n1,2,3\n")
        .expect("stage");
    h.call(
        ID,
        "frame_open",
        br#"{"w": 80, "h": 24, "seed": 1, "arg": "data.csv", "config": {}}"#,
    )
    .expect("frame_open");
    h.call(ID, "frame_close", b"").expect("frame_close");

    let resumed = String::from_utf8(
        h.call(
            ID,
            "tool_resume",
            br#"{"path": "x.csv", "changed": false, "written": false, "error": "cannot write x.csv: refused"}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        resumed,
        "error: no changes in the editor, but cannot write x.csv: refused"
    );
}

/// A grid-bridge / `/csvedit` open with no preceding `tool_call` never
/// records `ORIGINAL`, so it never pays for a diff and leaves `tool_resume`
/// with no row summary: the host's `changed` flag is all it reports
/// ("saved changes"), never row counts from a diff it did not run.
#[test]
fn csvedit_a_non_tool_open_never_records_a_summary() {
    let wasm = guest_or_skip!();
    let mut h = host(None);
    h.load(ID, &wasm, &["fs", "files", "log"]).expect("load");

    h.ram_write(ID, "/data.csv", b"A,B,C\n1,2,3\n")
        .expect("stage");
    h.call(
        ID,
        "frame_open",
        br#"{"w": 80, "h": 24, "seed": 1, "arg": "data.csv", "config": {}}"#,
    )
    .expect("frame_open");
    let hm = h.as_mut();
    key(hm, "enter", None);
    for c in "42".chars() {
        key(hm, &c.to_string(), Some(c));
    }
    key(hm, "enter", None);
    key(hm, "ctrl-s", None);
    h.call(ID, "frame_close", b"").expect("frame_close");

    let resumed = String::from_utf8(
        h.call(
            ID,
            "tool_resume",
            br#"{"path": "x.csv", "changed": true, "written": true, "error": null}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(resumed, "saved changes to x.csv");
}

/// A `tool_call` whose directive plank never honours (refused before
/// `frame_open`: no bridge, a sub-agent, `editor_refusal`, a missing
/// `files` grant, containment/symlink, over quota — none reproduced here,
/// since the point is the guest side of the leak) must not attach its
/// pending marker to a later, unrelated `frame_open`. This drives a
/// grid-style open of a different file name straight after the `tool_call`,
/// with no `frame_open` of `data.csv` in between.
#[test]
fn csvedit_a_refused_directive_never_leaks_into_a_later_grid_open() {
    let wasm = guest_or_skip!();
    let mut h = host(None);
    h.load(ID, &wasm, &["fs", "files", "log"]).expect("load");

    h.call(
        ID,
        "tool_call",
        br#"{"name": "edit_csv", "args": {"path": "x.csv"}}"#,
    )
    .expect("tool_call");
    // No frame_open of data.csv follows: plank refused the directive.

    h.ram_write(ID, "/other.csv", b"A,B,C\n1,2,3\n")
        .expect("stage");
    h.call(
        ID,
        "frame_open",
        br#"{"w": 80, "h": 24, "seed": 1, "arg": "other.csv", "config": {}}"#,
    )
    .expect("frame_open");
    let hm = h.as_mut();
    key(hm, "enter", None);
    for c in "42".chars() {
        key(hm, &c.to_string(), Some(c));
    }
    key(hm, "enter", None);
    key(hm, "ctrl-s", None);
    h.call(ID, "frame_close", b"").expect("frame_close");

    let resumed = String::from_utf8(
        h.call(
            ID,
            "tool_resume",
            br#"{"path": "x.csv", "changed": true, "written": true, "error": null}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        resumed, "saved changes to x.csv",
        "the stale marker from the refused tool_call must not have attached to this open"
    );
}

/// A `tool_call` followed by a slash command (the tool path abandoned in
/// favour of `/csvedit`) must not leave its pending marker to attach to
/// that command's own `frame_open`, even when it happens to open
/// `data.csv` too.
#[test]
fn csvedit_a_slash_command_after_tool_call_never_leaks_into_its_own_open() {
    let wasm = guest_or_skip!();
    let mut h = host(None);
    h.load(ID, &wasm, &["fs", "files", "log"]).expect("load");

    h.call(
        ID,
        "tool_call",
        br#"{"name": "edit_csv", "args": {"path": "x.csv"}}"#,
    )
    .expect("tool_call");
    h.call(
        ID,
        "command_run",
        br#"{"name": "open", "args": "data.csv"}"#,
    )
    .expect("command_run");

    h.ram_write(ID, "/data.csv", b"A,B,C\n1,2,3\n")
        .expect("stage");
    h.call(
        ID,
        "frame_open",
        br#"{"w": 80, "h": 24, "seed": 1, "arg": "data.csv", "config": {}}"#,
    )
    .expect("frame_open");
    let hm = h.as_mut();
    key(hm, "enter", None);
    for c in "42".chars() {
        key(hm, &c.to_string(), Some(c));
    }
    key(hm, "enter", None);
    key(hm, "ctrl-s", None);
    h.call(ID, "frame_close", b"").expect("frame_close");

    let resumed = String::from_utf8(
        h.call(
            ID,
            "tool_resume",
            br#"{"path": "x.csv", "changed": true, "written": true, "error": null}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        resumed, "saved changes to x.csv",
        "command_run must have cleared the marker before this open"
    );
}

#[test]
fn csvedit_command_run_round_trips_a_quoted_and_backslashed_name() {
    let wasm = guest_or_skip!();
    let mut h = host(None);
    h.load(ID, &wasm, &["fs", "log"]).expect("load");
    // The name contains a JSON-escaped quote and backslash, exercising the
    // command_run -> frame_open hop's text() decoding.
    let payload = "{\"name\": \"open\", \"args\": \"a\\\"b\\\\c.csv\"}";
    let out = String::from_utf8(h.call(ID, "command_run", payload.as_bytes()).unwrap()).unwrap();
    assert_eq!(out, "{\"open\": \"a\\\"b\\\\c.csv\"}");
}
