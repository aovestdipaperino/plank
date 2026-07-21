//! End-to-end coverage of `--ui-remote` against a headless render.
//!
//! This exercises `plank::uiremote` directly rather than driving the real
//! `Agent::tui_loop`: that loop is hardwired to
//! `ratatui::DefaultTerminal` (`Terminal<CrosstermBackend<Stdout>>`), so
//! there is no in-process way to hand it a `TestBackend` without changing
//! its signature — out of scope for a test-only change. What's covered
//! instead is two of the three pieces a harness leans on: draw-time region
//! recording (`uitree`) and screen serialization (`snapshot`), both driven
//! headlessly through a `ratatui::backend::TestBackend`.
//!
//! The third piece — the deferral that makes `keypress` then `snapshot`
//! return the *post-key* screen without a sleep — is **not** covered here.
//! It needs a real key-loop tick, which needs the signature change above.
//! Unit tests in `src/ui.rs` drive `UiRemote::capture` directly; issue #46
//! tracks closing the gap properly.
//!
//! Only `uitree_reports_the_popup_selection` touches the process-global
//! `RECORDING` flag (via `set_recording`/`region`); the other test never
//! calls into that state. With a single recorder-touching test in this
//! binary, there is nothing else in-process that can race it, regardless of
//! how the test runner schedules threads — so no lock or `--test-threads=1`
//! is needed here. `crate::uiremote::TEST_LOCK` is `pub(crate)` and cfg(test)
//! only, so it is not visible to this external test binary anyway; if a
//! second recorder-touching test is ever added here, it must be merged into
//! one `#[test]` (or the lock made available to integration tests) rather
//! than left to run concurrently.

use plank::uiremote;

/// Renders a frame with a popup region recorded, then reads it back.
#[test]
fn uitree_reports_the_popup_selection() {
    uiremote::set_recording(true);
    uiremote::begin_frame();
    uiremote::region("root", ratatui::layout::Rect::new(0, 0, 100, 30), &[]);
    uiremote::region(
        "popup",
        ratatui::layout::Rect::new(0, 14, 100, 15),
        &[
            ("rows", plank::tools::mcp::Json::Num(15.0)),
            ("selected", plank::tools::mcp::Json::Num(3.0)),
        ],
    );
    let tree = uiremote::frame_tree();
    uiremote::set_recording(false);
    assert!(tree.contains(r#""name":"popup""#), "{tree}");
    assert!(tree.contains(r#""selected":3"#), "{tree}");
}

/// Renders a frame with `TestBackend` (no terminal needed) and confirms
/// `buffer_to_ansi` round-trips its visible text.
#[test]
fn snapshot_round_trips_a_rendered_frame() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let mut term = Terminal::new(TestBackend::new(20, 3)).expect("terminal");
    term.draw(|f| {
        f.render_widget(ratatui::widgets::Paragraph::new("hello"), f.area());
    })
    .expect("draw");
    let ansi = uiremote::buffer_to_ansi(term.backend().buffer());
    assert!(ansi.starts_with("hello"), "{ansi:?}");
    assert_eq!(ansi.lines().count(), 3);
}
