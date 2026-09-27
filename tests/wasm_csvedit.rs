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
    assert!(out.contains("close"), "{out}");

    // The host calls frame_close after a closing key and prefers its line.
    let closed = String::from_utf8(h.call(ID, "frame_close", b"").expect("frame_close")).unwrap();
    assert!(
        closed.contains("\"scrollback\": \"csvedit: saved"),
        "{closed}"
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
}
