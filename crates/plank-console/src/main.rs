// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! `plank-console` — a Turbo Vision monitor for plank model-token streams.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant};

use plank_console::cmd;
use plank_console::registry::{Server, ServerEvent, SessionId};
use plank_console::session::{Sessions, SharedStreamView};
use plank_console::streamview::StreamView;
use plank_stream::render::RenderOptions;
use turbo_vision::app::Application;
use turbo_vision::core::command::{CM_CASCADE, CM_CLOSE, CM_NEXT, CM_QUIT, CM_TILE};
use turbo_vision::core::event::{EventType, KB_ALT_X, KB_F6, KB_F10};
use turbo_vision::core::geometry::Rect;
use turbo_vision::core::menu_data::{Menu, MenuItem};
use turbo_vision::views::file_dialog::FileDialog;
use turbo_vision::views::menu_bar::{MenuBar, SubMenu};
use turbo_vision::views::msgbox;
use turbo_vision::views::status_line::{StatusItem, StatusLine};
use turbo_vision::views::view::ViewId;
use turbo_vision::views::window::{Window, WindowBuilder};

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

        while let Ok(ev) = server.events().try_recv() {
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

    fn handle_server_event(&mut self, app: &mut Application, ev: ServerEvent) {
        match ev {
            ServerEvent::Opened { id, name, port } => {
                let view = Rc::new(RefCell::new(StreamView::new(app.get_tile_rect())));
                self.sessions
                    .insert(id, name.clone(), port, Rc::clone(&view), self.opts.0);
                let mut window = WindowBuilder::new()
                    .bounds(app.get_tile_rect())
                    .title(format!("{name} :{port}"))
                    .build();
                window.add(Box::new(SharedStreamView(view)));
                let view_id = app.desktop.add(Box::new(window));
                self.window_ids.insert(view_id, id);
                self.session_windows.insert(id, view_id);
            }
            ServerEvent::Reconnected { id } => {
                self.sessions.mark_reconnected(id);
                self.retitle(app, id);
            }
            ServerEvent::Bytes { id, data } => self.sessions.feed(id, &data),
            ServerEvent::Disconnected { id } => {
                self.sessions.mark_disconnected(id);
                self.retitle(app, id);
            }
            ServerEvent::Closed { id } => {
                self.sessions.remove(id);
                if let Some(view_id) = self.session_windows.remove(&id) {
                    self.window_ids.remove(&view_id);
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

    /// Looks up a session's window on the desktop and applies its current
    /// title text (reflecting connected/disconnected state).
    fn retitle(&self, app: &mut Application, id: SessionId) {
        let Some(view_id) = self.session_windows.get(&id).copied() else {
            return;
        };
        let Some(title) = self.sessions.window_title(id) else {
            return;
        };
        if let Some(view) = app.desktop.child_by_id_mut(view_id)
            && let Some(window) = view.as_any_mut().downcast_mut::<Window>()
        {
            window.set_title(&title);
        }
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
        let view = Rc::new(RefCell::new(StreamView::new(app.get_tile_rect())));
        let id = NEXT_CAPTURE_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        self.sessions
            .insert(id, name.clone(), 0, Rc::clone(&view), self.opts.0);
        if let Some(state) = self.sessions.get_mut(id) {
            state.connected = true;
            state.pipeline.feed(&bytes, &mut view.borrow_mut());
            state.pipeline.finish(&mut view.borrow_mut());
        }

        let mut window = WindowBuilder::new()
            .bounds(app.get_tile_rect())
            .title(name)
            .build();
        window.add(Box::new(SharedStreamView(view)));
        let view_id = app.desktop.add(Box::new(window));
        self.window_ids.insert(view_id, id);
    }
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
