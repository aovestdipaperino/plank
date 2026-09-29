//! End to end through the grid bridge, host and guest halves together: a grid
//! goes onto csvedit's RAM disk, opens in its frame, is edited and saved
//! there, and comes back out of `collect_editor_file` when the frame closes.
//!
//! The MCP server, the write-back call and the tool call that lends the
//! session to the TUI are not here: what is under test is
//! the crossing of the component's RAM disk, which is the part only the real
//! guest can prove. Like `wasm_csvedit`, this runs only with
//! `--features plugins` and a guest built by `guests/build.sh`; without the
//! artifact it skips, unless `PLANK_REQUIRE_GUESTS` is set.

#![cfg(feature = "plugins")]

use std::path::PathBuf;

use plank::framebridge::FrameResult;
use plank::plugins::Origin;
use plank::wasmreg::{
    Capability, FrameKind, FrameOutcome, Loaded, OpenFrame, Registry, Session, Surface,
    WasmComponent, WasmManifest,
};

const ID: &str = "dev.plank.csvedit";
const FILE: &str = "transactions.csv";
const W: u16 = 80;
const H: u16 = 24;

const HEADER: &str = "#,date,provider,description,amount,currency,category,note";
const ROW_1: &str = "1,2026-09-01,bbva,Coffee,-3.50,EUR,food,";
const ROW_2: &str = "2,2026-09-02,bbva,Salary,2500.00,EUR,income,monthly";
/// Index of `category` in [`HEADER`].
const CATEGORY: usize = 6;

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

/// A session with the real csvedit loaded and admitted exactly as its
/// manifest declares it: a `frame` with `fs`, not a screensaver.
fn session(wasm: &[u8]) -> Session {
    let mut session = Session::new(None);
    session
        .host
        .load(ID, wasm, &["fs", "log"])
        .expect("load csvedit");
    session.registry = Registry::with_loaded(vec![Loaded {
        component: WasmComponent {
            plugin: "csvedit".to_string(),
            origin: Origin::UserScan,
            path: PathBuf::from("/csvedit.wasm"),
            manifest: WasmManifest {
                id: ID.to_string(),
                abi: 1,
                module: "csvedit.wasm".to_string(),
                surfaces: vec![Surface::Command, Surface::Frame],
                capabilities: vec![Capability::Log, Capability::Fs],
                kind: FrameKind::Arcade,
                veiled: false,
                min_size: (60, 16),
                frames: Vec::new(),
                config: Vec::new(),
                events: Vec::new(),
            },
        },
        strikes: 0,
        tools: Vec::new(),
        commands: Vec::new(),
    }]);
    session
}

/// The grid as a server exports it.
fn staged() -> String {
    format!("{HEADER}\n{ROW_1}\n{ROW_2}\n")
}

/// Stages the grid on csvedit's disk and opens its frame on it, the way a
/// lent session is driven once the TUI takes it.
fn stage_and_open(session: &mut Session) -> OpenFrame {
    session
        .stage_editor_file(ID, FILE, staged().as_bytes())
        .expect("staged on the RAM disk");
    let mut frame = session.open_frame(ID, FILE, W, H, 0).expect("open");
    session
        .step_frame(&mut frame, 16, W, H, 0)
        .expect("first paint");
    frame
}

fn press(session: &mut Session, frame: &OpenFrame, code: &str) -> FrameOutcome {
    session.frame_key(frame, code, None).expect("frame_key")
}

fn type_str(session: &mut Session, frame: &OpenFrame, text: &str) {
    for c in text.chars() {
        session
            .frame_key(frame, &c.to_string(), Some(c))
            .expect("frame_key");
    }
}

/// Exits with ctrl-q, closes the frame as the UI does after a closing key,
/// and collects the file.
fn quit(session: &mut Session, frame: &OpenFrame) -> FrameResult {
    assert!(
        matches!(press(session, frame, "ctrl-q"), FrameOutcome::Close(_)),
        "ctrl-q closes a saved or untouched grid"
    );
    session.close_frame(frame);
    session.collect_editor_file(ID, FILE, staged().as_bytes())
}

fn cells(line: &str) -> Vec<&str> {
    line.split(',').collect()
}

#[test]
fn a_grid_staged_edited_in_csvedit_and_handed_back() {
    let wasm = guest_or_skip!();
    let mut session = session(&wasm);
    let frame = stage_and_open(&mut session);

    // The selection starts on row 1's `#` cell; walk right to its category.
    for _ in 0..CATEGORY {
        press(&mut session, &frame, "right");
    }
    press(&mut session, &frame, "enter");
    // The edit dialog opens holding the cell's text, cursor at its end.
    for _ in 0.."food".len() {
        press(&mut session, &frame, "backspace");
    }
    type_str(&mut session, &frame, "coffee");
    press(&mut session, &frame, "enter");
    // Bridged, ctrl-s saves in place under the staged name.
    assert_eq!(press(&mut session, &frame, "ctrl-s"), FrameOutcome::Stay);

    let FrameResult::Saved(bytes) = quit(&mut session, &frame) else {
        panic!("an edited grid comes back saved");
    };
    let csv = String::from_utf8(bytes).expect("UTF-8");
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), 3, "{csv}");
    assert_eq!(cells(lines[0]), cells(HEADER), "{csv}");
    let mut edited = cells(ROW_1);
    edited[CATEGORY] = "coffee";
    assert_eq!(cells(lines[1]), edited, "{csv}");
    assert_eq!(cells(lines[2]), cells(ROW_2), "{csv}");

    // The grid is over: its file is gone from the disk.
    assert!(session.host.ram_file(ID, FILE).is_none());
}

#[test]
fn an_untouched_grid_hands_nothing_back() {
    let wasm = guest_or_skip!();
    let mut session = session(&wasm);
    let frame = stage_and_open(&mut session);
    assert_eq!(quit(&mut session, &frame), FrameResult::Unchanged);
    assert!(session.host.ram_file(ID, FILE).is_none());
}
