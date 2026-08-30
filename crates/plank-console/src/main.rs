// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! `plank-console` — a Turbo Vision monitor for plank model-token streams.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant};

use plank_console::cmd;
use plank_console::registry::{Server, ServerEvent, SessionId};
use plank_console::session::{Sessions, SharedStreamView, format_title};
use plank_console::streamview::StreamView;
use plank_stream::render::RenderOptions;
use turbo_vision::app::Application;
use turbo_vision::core::command::{CM_CASCADE, CM_CLOSE, CM_NEXT, CM_QUIT, CM_TILE};
use turbo_vision::core::event::{EventType, KB_ALT_X, KB_F6, KB_F10};
use turbo_vision::core::geometry::Rect;
use turbo_vision::core::menu_data::{Menu, MenuItem};
use turbo_vision::core::state::{SF_CLOSED, shadow_size};
use turbo_vision::views::file_dialog::FileDialog;
use turbo_vision::views::menu_bar::{MenuBar, SubMenu};
use turbo_vision::views::msgbox;
use turbo_vision::views::status_line::{StatusItem, StatusLine};
use turbo_vision::views::view::ViewId;
use turbo_vision::views::window::{Window, WindowBuilder};

/// A bounded number of pending stream-server events drained per main-loop
/// iteration. `Server::events()` is an unbounded channel, so a client
/// streaming faster than the pipeline/draw can keep up (`cat bigfile | nc
/// ...`) would otherwise spin this inner loop forever and never return to
/// `app.get_event()` — starving keystrokes, redraws and Alt-X. A message
/// count is a simpler, more predictable budget here than a time slice (no
/// clock reads on a hot loop, and behavior does not depend on how fast the
/// draw happens to be on the day it is run).
const MAX_EVENTS_PER_TICK: usize = 64;

fn main() -> turbo_vision::core::error::Result<()> {
    let control_port = parse_port_arg();

    let mut app = Application::new()?;
    let (width, height) = app.terminal.size();
    app.set_menu_bar(build_menu_bar(width));
    app.set_status_line(build_status_line(width, height, 0));

    let mut server = match Server::bind(control_port) {
        Ok(s) => s,
        Err(e) => {
            // The terminal is already in raw mode; drop out of it before
            // printing, or the message lands in a half-torn-down screen.
            drop(app);
            eprintln!("plank-console: cannot bind 127.0.0.1:{control_port}: {e}");
            std::process::exit(1);
        }
    };

    let mut console = Console::default();

    app.draw();
    let _ = app.terminal.flush();

    let mut last_reap = Instant::now();
    let mut last_status_tick = Instant::now();

    while app.running {
        let mut dirty = false;

        if let Some(mut event) = app.get_event() {
            app.handle_event(&mut event);
            dirty = true;
            if event.what == EventType::Command {
                console.handle_command(&mut app, event.command);
            }
        }

        for _ in 0..MAX_EVENTS_PER_TICK {
            let Ok(ev) = server.events().try_recv() else {
                break;
            };
            dirty = true;
            console.handle_server_event(&mut app, ev);
        }

        if last_reap.elapsed() > Duration::from_secs(60) {
            last_reap = Instant::now();
            server.reap(Duration::from_mins(30));
        }

        if last_status_tick.elapsed() >= Duration::from_secs(1) {
            last_status_tick = Instant::now();
            let (w, h) = app.terminal.size();
            app.set_status_line(build_status_line(w, h, server.live_count()));
            dirty = true;
        }

        if dirty {
            app.draw();
            let _ = app.terminal.flush();
        }
        // `Application::run()` calls this each iteration; the hand-rolled
        // loop must too, or a moved window's move-tracking state is never
        // cleared and `get_redraw_union()` returns stale data indefinitely.
        app.desktop.handle_moved_windows(&mut app.terminal);

        if app.desktop.remove_closed_windows() {
            console.forget_closed_windows(&app);
            app.draw();
            let _ = app.terminal.flush();
        }
    }

    Ok(())
}

/// Everything the main loop mutates across frames: live sessions, the
/// `ViewId <-> SessionId` maps needed to find the focused window's session,
/// and the current render options.
///
/// `window_ids` maps a desktop `ViewId` to the session it belongs to, and is
/// how the focused window's session id is found each frame. **Deviation
/// from the brief:** rather than tracking a `HashMap<u8, SessionId>` keyed
/// by a Borland-style window number (this turbo-vision version does not
/// assign window numbers automatically), focus is read directly from
/// `Desktop::top_view_id()`, which the library already keeps correct across
/// `CM_NEXT`, clicks, and window close — a strictly more reliable source of
/// truth for "which window is focused" than a number this crate would have
/// to assign and track itself.
#[derive(Default)]
struct Console {
    sessions: Sessions,
    window_ids: HashMap<ViewId, SessionId>,
    session_windows: HashMap<SessionId, ViewId>,
    opts: RenderOptionsState,
}

/// What a decided [`ServerEvent`] means for the desktop: create a window for
/// a session, retitle one, or close one. Carries no `Application` reference,
/// so deciding which of these applies is testable without a TTY; only
/// *applying* one touches the desktop.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ConsoleIntent {
    CreateWindow {
        id: SessionId,
        name: String,
        port: u16,
    },
    Retitle {
        view_id: ViewId,
        title: String,
    },
    CloseWindow {
        view_id: ViewId,
    },
}

/// `RenderOptions` doesn't implement `Default`; this does, so `Console` can.
struct RenderOptionsState(RenderOptions);

impl Default for RenderOptionsState {
    fn default() -> Self {
        Self(RenderOptions {
            use_color: true,
            format_thinking: true,
            format_markdown: true,
        })
    }
}

impl Console {
    fn handle_command(
        &mut self,
        app: &mut Application,
        command: turbo_vision::core::command::CommandId,
    ) {
        let focused_id = app
            .desktop
            .top_view_id()
            .and_then(|id| self.window_ids.get(&id).copied());

        match command {
            cmd::CM_SHOW_THINKING => {
                self.opts.0.format_thinking = !self.opts.0.format_thinking;
                self.sessions.set_options(self.opts.0);
            }
            cmd::CM_SHOW_MARKDOWN => {
                self.opts.0.format_markdown = !self.opts.0.format_markdown;
                self.sessions.set_options(self.opts.0);
            }
            cmd::CM_CLEAR_WINDOW => {
                if let Some(id) = focused_id {
                    self.sessions.clear(id);
                }
            }
            cmd::CM_SAVE_AS => {
                if let Some(id) = focused_id {
                    self.save_as(app, id);
                }
            }
            cmd::CM_OPEN_CAPTURE => self.open_capture(app),
            _ => {}
        }
    }

    /// Applies a live stream-server event to a desktop, via the pure
    /// decision made by [`Self::decide_server_event`].
    fn handle_server_event(&mut self, app: &mut Application, ev: ServerEvent) {
        let Some(intent) = self.decide_server_event(ev) else {
            return;
        };
        self.apply_intent(app, intent);
    }

    /// Decides what a `ServerEvent` means for this console's bookkeeping and
    /// what (if anything) the caller must do to the desktop. Pure decision
    /// logic: it touches `self.sessions` / `self.window_ids` /
    /// `self.session_windows` only, never an `Application`, so it is
    /// unit-testable without a TTY.
    fn decide_server_event(&mut self, ev: ServerEvent) -> Option<ConsoleIntent> {
        match ev {
            ServerEvent::Opened { id, name, port } => {
                Some(ConsoleIntent::CreateWindow { id, name, port })
            }
            ServerEvent::Reconnected { id } => {
                self.sessions.mark_reconnected(id);
                self.retitle_intent(id)
            }
            ServerEvent::Bytes { id, data } => {
                self.sessions.feed(id, &data);
                None
            }
            ServerEvent::Disconnected { id } => {
                self.sessions.mark_disconnected(id);
                self.retitle_intent(id)
            }
            ServerEvent::Closed { id } => {
                // The server sends this once per session, on its own
                // 30-minute idle TTL (`Server::reap`) — invisible in a short
                // manual test, but certain in a long-running session. Drop
                // both bookkeeping maps here so a stale entry can never
                // misattribute a later command, and hand back an intent so
                // the caller actually removes the window: otherwise it sits
                // on screen forever, frozen on its last frame.
                self.sessions.remove(id);
                let view_id = self.session_windows.remove(&id)?;
                self.window_ids.remove(&view_id);
                Some(ConsoleIntent::CloseWindow { view_id })
            }
        }
    }

    /// The retitle intent for a session's window, if it still has one and
    /// still has a title to show (both `None` for e.g. a capture window,
    /// which isn't tracked in `session_windows`).
    fn retitle_intent(&self, id: SessionId) -> Option<ConsoleIntent> {
        let view_id = *self.session_windows.get(&id)?;
        let title = self.sessions.window_title(id)?;
        Some(ConsoleIntent::Retitle { view_id, title })
    }

    /// Performs the turbo-vision side effects a decided [`ConsoleIntent`]
    /// calls for.
    fn apply_intent(&mut self, app: &mut Application, intent: ConsoleIntent) {
        match intent {
            ConsoleIntent::CreateWindow { id, name, port } => {
                let window_bounds = tile_window_bounds(app);
                let view = Rc::new(RefCell::new(StreamView::new(session_view_bounds(
                    window_bounds,
                ))));
                self.sessions
                    .insert(id, name.clone(), port, Rc::clone(&view), self.opts.0);
                let mut window = WindowBuilder::new()
                    .bounds(window_bounds)
                    .title(format_title(&name, port))
                    .build();
                apply_session_window_palette(&mut window);
                window.add(Box::new(SharedStreamView(view)));
                let view_id = app.desktop.add(Box::new(window));
                self.window_ids.insert(view_id, id);
                self.session_windows.insert(id, view_id);
            }
            ConsoleIntent::Retitle { view_id, title } => {
                if let Some(view) = app.desktop.child_by_id_mut(view_id)
                    && let Some(window) = view.as_any_mut().downcast_mut::<Window>()
                {
                    window.set_title(&title);
                }
            }
            ConsoleIntent::CloseWindow { view_id } => {
                // Mark the window SF_CLOSED so the desktop's normal
                // `remove_closed_windows` sweep (already run each loop
                // iteration) picks it up, exactly as a user-driven close
                // does — no separate removal path to keep in sync.
                if let Some(view) = app.desktop.child_by_id_mut(view_id) {
                    view.set_state_flag(SF_CLOSED, true);
                }
            }
        }
    }

    /// A window may have been closed by the user (frame close box,
    /// Window > Close) rather than by the server. Drop any mapping whose
    /// `ViewId` is no longer on the desktop so a stale entry cannot
    /// misattribute a later command to the wrong session.
    fn forget_closed_windows(&mut self, app: &Application) {
        self.window_ids
            .retain(|view_id, _| app.desktop.contains_id(*view_id));
        self.session_windows
            .retain(|_, view_id| app.desktop.contains_id(*view_id));
    }

    fn save_as(&self, app: &mut Application, id: SessionId) {
        let Some(text) = self.sessions.plain_text(id) else {
            return;
        };
        let mut dialog = build_file_dialog(app, "Save As")
            .with_button_label("~S~ave")
            .build();
        if let Some(path) = dialog.execute(app)
            && let Err(e) = std::fs::write(&path, text)
        {
            msgbox::message_box_error(app, &format!("Cannot write {}: {e}", path.display()));
        }
    }

    fn open_capture(&mut self, app: &mut Application) {
        let mut dialog = build_file_dialog(app, "Open Capture").build();
        let Some(path) = dialog.execute(app) else {
            return;
        };
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                msgbox::message_box_error(app, &format!("Cannot read {}: {e}", path.display()));
                return;
            }
        };

        let name = basename(&path);
        let window_bounds = tile_window_bounds(app);
        let view = Rc::new(RefCell::new(StreamView::new(session_view_bounds(
            window_bounds,
        ))));
        let id = NEXT_CAPTURE_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        self.sessions
            .insert(id, name.clone(), 0, Rc::clone(&view), self.opts.0);
        if let Some(state) = self.sessions.get_mut(id) {
            state.connected = true;
            state.pipeline.feed(&bytes, &mut view.borrow_mut());
            state.pipeline.finish(&mut view.borrow_mut());
        }

        let mut window = WindowBuilder::new()
            .bounds(window_bounds)
            .title(name)
            .build();
        apply_session_window_palette(&mut window);
        window.add(Box::new(SharedStreamView(view)));
        let view_id = app.desktop.add(Box::new(window));
        self.window_ids.insert(view_id, id);
    }
}

/// Bounds for a new stream window: the full tile rect, shrunk on the
/// bottom-right by the window shadow. Every window shows its shadow by
/// default (`SF_SHADOW`), and a window placed at the *exact* desktop bounds
/// gets pushed up-and-left by `constrain_to_limits` to make room for that
/// shadow — hiding its entire top row (title bar included) above row 0 and
/// off screen. Reserving that room up front keeps the window, title bar
/// included, fully on screen.
fn tile_window_bounds(app: &Application) -> Rect {
    let mut bounds = app.get_tile_rect();
    let (shadow_x, shadow_y) = shadow_size();
    bounds.b.x -= shadow_x;
    bounds.b.y -= shadow_y;
    bounds
}

/// Bounds for a session's `StreamView`, given the *window's* outer bounds.
///
/// `Window::add` places children in the window's interior `Group`, whose
/// children take bounds relative to the interior (starting at `(0, 0)`) —
/// `Group::add` converts relative to absolute using the interior's own
/// origin. The interior itself is the window bounds grown by `(-1, -1)`
/// (one row/column inset for the frame on every side), so with `Rect::b`
/// exclusive the interior is exactly two narrower and two shorter than the
/// outer window. Passing the outer window bounds here (as before) drew the
/// view over the frame instead of inside it. See
/// `turbo-vision-4-rust/src/views/log_window.rs` for the same idiom.
fn session_view_bounds(window_bounds: Rect) -> Rect {
    Rect::new(0, 0, window_bounds.width() - 2, window_bounds.height() - 2)
}

/// Points a session window's palette at the app palette's "Black Window"
/// region (`CP_APP_COLOR` entries 97-104) instead of the default Blue
/// window colors, per the user's request for a black background.
///
/// Only the 8 window-palette entries (frame passive/active/icon, scrollbar
/// page/arrows, normal/selected text, reserved) are overridden — matching
/// `CP_BLUE_WINDOW`'s own first 8 entries and the `LogWindowBuilder`
/// precedent in `log_window.rs`. `Palette::get` returns the error color 0
/// (not a panic or garbage read) for any index beyond the slice's length,
/// so the 11 trailing syntax-highlighting entries (9-19) that `CP_BLUE_WINDOW`
/// carries are simply left unmapped here; `StreamView` paints its own
/// `Cell`s with explicit `Attr`s from the ANSI parser, so those entries are
/// never consulted for the stream text, and only the frame/scrollbar read
/// the 8 entries we do provide. Frame passive/active (entries 1-2, mapped
/// to app colors 97-98) keep the same layout as the Blue palette's own
/// frame entries, so the title stays legible focused and unfocused.
fn apply_session_window_palette(window: &mut Window) {
    window.set_custom_palette(vec![97, 98, 99, 100, 101, 102, 103, 104]);
}

/// Reads `--port <n>` from argv; defaults to 7878.
fn parse_port_arg() -> u16 {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--port"
            && let Some(v) = args.get(i + 1)
            && let Ok(p) = v.parse::<u16>()
        {
            return p;
        }
        i += 1;
    }
    7878
}

fn build_menu_bar(width: i16) -> MenuBar {
    let mut menu_bar = MenuBar::new(Rect::new(0, 0, width, 1));

    menu_bar.add_submenu(SubMenu::new(
        "~F~ile",
        Menu::from_items(vec![
            MenuItem::new("~O~pen capture...", cmd::CM_OPEN_CAPTURE, 0, 0),
            MenuItem::new("~S~ave As...", cmd::CM_SAVE_AS, 0, 0),
            MenuItem::separator(),
            MenuItem::new("E~x~it", CM_QUIT, 0, 0),
        ]),
    ));
    menu_bar.add_submenu(SubMenu::new(
        "~V~iew",
        Menu::from_items(vec![
            MenuItem::new("Show ~t~hinking", cmd::CM_SHOW_THINKING, 0, 0),
            MenuItem::new("~M~arkdown", cmd::CM_SHOW_MARKDOWN, 0, 0),
            MenuItem::new("~C~lear window", cmd::CM_CLEAR_WINDOW, 0, 0),
        ]),
    ));
    menu_bar.add_submenu(SubMenu::new(
        "~W~indow",
        Menu::from_items(vec![
            MenuItem::new("~N~ext", CM_NEXT, 0, 0),
            MenuItem::new("~T~ile", CM_TILE, 0, 0),
            MenuItem::new("C~a~scade", CM_CASCADE, 0, 0),
            MenuItem::new("~C~lose", CM_CLOSE, 0, 0),
        ]),
    ));

    menu_bar
}

fn build_status_line(width: i16, height: i16, live: usize) -> StatusLine {
    StatusLine::new(
        Rect::new(0, height - 1, width, height),
        vec![
            StatusItem::new("~F6~ Next", KB_F6, CM_NEXT),
            StatusItem::new("~F10~ Menu", KB_F10, 0),
            StatusItem::new("~Alt-X~ Exit", KB_ALT_X, CM_QUIT),
            StatusItem::new(&format!("{live} conn"), 0, 0),
        ],
    )
}

/// A centered file dialog sized to the current terminal, showing all files.
fn build_file_dialog(app: &Application, title: &str) -> FileDialog {
    let (width, height) = app.terminal.size();
    let dialog_width = 62.min(width);
    let dialog_height = 20.min(height);
    let dialog_x = (width - dialog_width) / 2;
    let dialog_y = (height - dialog_height) / 2;
    FileDialog::new(
        Rect::new(
            dialog_x,
            dialog_y,
            dialog_x + dialog_width,
            dialog_y + dialog_height,
        ),
        title,
        "*",
        None,
    )
}

fn basename(path: &std::path::Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

/// A synthetic, negative-space session id: capture windows never collide
/// with server-assigned ids because those start at 1 and only increase, so
/// a fixed high band is a simple, adequate allocator here.
static NEXT_CAPTURE_ID: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(u64::MAX / 2);

#[cfg(test)]
mod console_decision_tests {
    use super::*;
    use plank_console::registry::ServerEvent;

    #[test]
    fn opened_decides_to_create_a_window() {
        let mut console = Console::default();
        let intent = console.decide_server_event(ServerEvent::Opened {
            id: 1,
            name: "demo".into(),
            port: 61278,
        });
        assert_eq!(
            intent,
            Some(ConsoleIntent::CreateWindow {
                id: 1,
                name: "demo".into(),
                port: 61278,
            })
        );
    }

    #[test]
    fn bytes_decides_nothing_but_still_feeds_the_session() {
        let mut console = Console::default();
        console
            .sessions
            .insert(1, "demo".into(), 4242, test_view(), console.opts.0);
        let intent = console.decide_server_event(ServerEvent::Bytes {
            id: 1,
            data: b"hello\n".to_vec(),
        });
        assert_eq!(intent, None);
        assert!(console.sessions.plain_text(1).unwrap().contains("hello"));
    }

    #[test]
    fn reconnected_decides_to_retitle_the_mapped_window() {
        let mut console = Console::default();
        console
            .sessions
            .insert(1, "demo".into(), 4242, test_view(), console.opts.0);
        let view_id = ViewId::from_u16(7);
        console.session_windows.insert(1, view_id);
        console.window_ids.insert(view_id, 1);

        let intent = console.decide_server_event(ServerEvent::Reconnected { id: 1 });

        assert_eq!(
            intent,
            Some(ConsoleIntent::Retitle {
                view_id,
                title: "demo :4242".into(),
            })
        );
    }

    /// Regression test for the "Closed leaks an orphan window" finding:
    /// `Server::reap`'s 30-minute idle TTL sends `Closed` once per session,
    /// and the decision must produce a close-the-window intent — not just
    /// drop the bookkeeping maps and leave the window on screen forever.
    #[test]
    fn closed_decides_to_close_the_mapped_window_and_forgets_it() {
        let mut console = Console::default();
        console
            .sessions
            .insert(1, "demo".into(), 4242, test_view(), console.opts.0);
        let view_id = ViewId::from_u16(9);
        console.session_windows.insert(1, view_id);
        console.window_ids.insert(view_id, 1);

        let intent = console.decide_server_event(ServerEvent::Closed { id: 1 });

        assert_eq!(intent, Some(ConsoleIntent::CloseWindow { view_id }));
        assert!(!console.session_windows.contains_key(&1));
        assert!(!console.window_ids.contains_key(&view_id));
        assert!(console.sessions.plain_text(1).is_none());
    }

    #[test]
    fn closed_with_no_mapped_window_decides_nothing() {
        let mut console = Console::default();
        console
            .sessions
            .insert(1, "demo".into(), 4242, test_view(), console.opts.0);
        let intent = console.decide_server_event(ServerEvent::Closed { id: 1 });
        assert_eq!(intent, None);
    }

    fn test_view() -> plank_console::session::SharedView {
        std::rc::Rc::new(std::cell::RefCell::new(StreamView::new(Rect::new(
            0, 0, 80, 24,
        ))))
    }
}

/// Headless regression test for the "window title bars render empty"
/// finding. `Application::new()` needs a real TTY, so this exercises
/// `tile_window_bounds` + `WindowBuilder` + `Desktop::add` + `Frame::draw`
/// directly against a `Terminal` running a `NullBackend`, which is enough
/// to reproduce and pin the bug: a window placed at the *exact* desktop
/// bounds gets pushed up by `constrain_to_limits` to make room for its
/// shadow, hiding the title bar off the top of the screen.
#[cfg(test)]
mod title_render_tests {
    use std::io;
    use std::time::Duration;

    use turbo_vision::core::event::Event;
    use turbo_vision::core::geometry::Rect;
    use turbo_vision::terminal::{Backend, Terminal};
    use turbo_vision::views::desktop::Desktop;
    use turbo_vision::views::view::View;
    use turbo_vision::views::window::WindowBuilder;

    struct NullBackend;
    impl Backend for NullBackend {
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        fn init(&mut self) -> io::Result<()> {
            Ok(())
        }
        fn cleanup(&mut self) -> io::Result<()> {
            Ok(())
        }
        fn size(&self) -> io::Result<(u16, u16)> {
            Ok((80, 25))
        }
        fn poll_event(&mut self, _timeout: Duration) -> io::Result<Option<Event>> {
            Ok(None)
        }
        fn write_raw(&mut self, _data: &[u8]) -> io::Result<()> {
            Ok(())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
        fn show_cursor(&mut self, _x: u16, _y: u16) -> io::Result<()> {
            Ok(())
        }
        fn hide_cursor(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// A window whose bounds equal the tile rect returned by
    /// `tile_window_bounds` (shadow-shrunk) renders its title on row 0 —
    /// not pushed off screen.
    #[test]
    fn shadow_shrunk_bounds_keep_the_title_bar_on_screen() {
        let mut terminal = Terminal::with_backend(Box::new(NullBackend)).unwrap();
        let mut desktop = Desktop::new(Rect::new(0, 0, 80, 25));

        let mut bounds = desktop.bounds();
        let (shadow_x, shadow_y) = turbo_vision::core::state::shadow_size();
        bounds.b.x -= shadow_x;
        bounds.b.y -= shadow_y;

        let window = WindowBuilder::new()
            .bounds(bounds)
            .title("demo :61278")
            .build();
        desktop.add(Box::new(window));
        desktop.draw(&mut terminal);

        let row0: String = terminal.buffer()[0].iter().map(|c| c.ch).collect();
        assert!(row0.contains("demo :61278"), "title not on row 0: {row0:?}");
    }

    /// Without the shadow-shrink, the same window (bounds == full desktop)
    /// gets constrained up by one row to make room for its shadow, and the
    /// title bar (row 0 of the frame) scrolls off screen entirely. This
    /// pins the actual root cause: not `Frame`/`set_title`, but window
    /// sizing versus the shadow.
    #[test]
    fn unshrunk_bounds_push_the_title_bar_off_screen() {
        let mut terminal = Terminal::with_backend(Box::new(NullBackend)).unwrap();
        let mut desktop = Desktop::new(Rect::new(0, 0, 80, 25));

        let window = WindowBuilder::new()
            .bounds(desktop.bounds())
            .title("demo :61278")
            .build();
        desktop.add(Box::new(window));
        desktop.draw(&mut terminal);

        let row0: String = terminal.buffer()[0].iter().map(|c| c.ch).collect();
        assert!(
            !row0.contains("demo :61278"),
            "expected the unshrunk-bounds title to be scrolled off row 0, but found it: {row0:?}"
        );
    }

    /// Regression test for the "`StreamView` drawn over the frame" bug:
    /// `StreamView::new(app.get_tile_rect())` handed the view the window's
    /// *outer* bounds, so `Group::add` (relative -> absolute) placed it at
    /// the window's own origin, on top of the frame. A view added to a
    /// window's interior must be given bounds relative to the interior —
    /// `(0, 0, w - 2, h - 2)` for a window of width `w`, height `h` — so
    /// that after `Window::add` its absolute bounds land one row/column
    /// inside the frame on every side.
    #[test]
    fn stream_view_bounds_land_inside_the_frame_not_over_it() {
        use plank_console::streamview::StreamView;

        let window_bounds = Rect::new(0, 0, 40, 20);
        let view_bounds = super::session_view_bounds(window_bounds);
        // Interior-relative: must start at the origin, not the window's.
        assert_eq!(view_bounds, Rect::new(0, 0, 38, 18));

        let mut window = WindowBuilder::new()
            .bounds(window_bounds)
            .title("demo")
            .build();
        window.add(Box::new(StreamView::new(view_bounds)));

        // After Group::add's relative -> absolute conversion, the child's
        // absolute bounds must be the window bounds inset by one cell of
        // frame on every side (Rect::b is exclusive), not the window's
        // outer bounds.
        let absolute = window.child_at(0).bounds();
        assert_eq!(absolute, Rect::new(1, 1, 39, 19));
    }

    /// Regression test for the actual pre-fix bug: passing the window's
    /// *outer* bounds (unshrunk) as the child's relative bounds. `Group::add`
    /// still offsets the child's top-left by the interior's own origin
    /// (`(1, 1)` here), so the left/top edge lands correctly by
    /// coincidence — the bug is that the child's bottom-right is then two
    /// cells too large in each dimension, so it overlaps the frame's right
    /// and bottom border instead of stopping at the interior's edge.
    #[test]
    fn outer_bounds_overflow_the_interior_by_the_frame_width() {
        use plank_console::streamview::StreamView;

        let window_bounds = Rect::new(0, 0, 40, 20);
        let mut window = WindowBuilder::new()
            .bounds(window_bounds)
            .title("demo")
            .build();
        window.add(Box::new(StreamView::new(window_bounds)));

        let absolute = window.child_at(0).bounds();
        // The interior is (1, 1, 39, 19); the buggy child bounds instead
        // reach to (41, 21) — two cells past the interior on each side.
        assert_eq!(absolute, Rect::new(1, 1, 41, 21));
        assert!(
            absolute.b.x > 39 && absolute.b.y > 19,
            "expected the unfixed bounds to overflow the interior (1,1,39,19), got {absolute:?}"
        );
    }
}
