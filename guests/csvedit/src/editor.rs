//! The editor: one window holding a Table over a CsvDoc, a menu bar, a status
//! line, and non-modal dialogs for editing, naming and confirming.
//!
//! plank pushes one key per call and asks for a screen per step, so this is a
//! Turbo Vision `Application` driven by [`Application::pump`], never by a
//! blocking run loop, and every dialog is a plain desktop window whose
//! buttons send commands back to the [`State`] handler.

use turbo_vision::app::{AppHandler, Application};
use turbo_vision::core::command::{CM_CLOSE, CM_QUIT, CommandId};
use turbo_vision::core::event::{
    Event, EventType, KB_ALT_X, KB_CTRL_K, KB_CTRL_L, KB_CTRL_O, KB_CTRL_Q, KB_CTRL_R, KB_CTRL_S,
    KB_CTRL_Y, KB_DEL, KB_ENTER, KB_ESC, KB_INS,
};
use turbo_vision::core::geometry::Rect;
use turbo_vision::core::menu_data::{Menu, MenuItem, MenuItemBuilder};
use turbo_vision::core::status_data::StatusItemBuilder;
use turbo_vision::terminal::{HostBackend, HostInput, Terminal};
use turbo_vision::views::View;
use turbo_vision::views::group::GroupLike;
use turbo_vision::views::handle::Handle;
use turbo_vision::views::menu_bar::{MenuBar, SubMenu};
use turbo_vision::views::paramtext::ParamText;
use turbo_vision::views::status_line::StatusLine;
use turbo_vision::views::table::{Column, Table};
use turbo_vision::views::window::{Window, WindowBuilder};

use crate::commands::{
    CMD_COL_DEL, CMD_COL_INS, CMD_DISCARD, CMD_DLG_CANCEL, CMD_DLG_OK, CMD_EDIT_CELL, CMD_EXIT,
    CMD_NEW, CMD_OPEN, CMD_OPEN_PICK, CMD_RENAME_COL, CMD_ROW_DEL, CMD_ROW_INS, CMD_SAVE,
    CMD_SAVE_AS,
};
use crate::dialogs::{self, Overlay};
use crate::disk::Disk;
use crate::doc::CsvDoc;
use crate::keys::is_ctrl;

/// What to do once a save or a discard has resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum After {
    Nothing,
    New,
    Open,
    Exit,
}

/// The app state the handler owns.
struct State {
    doc: CsvDoc,
    /// `None` while untitled.
    name: Option<String>,
    disk: Box<dyn Disk>,
    window: Handle<Window>,
    table: Handle<Table>,
    message: Handle<ParamText>,
    overlay: Option<Overlay>,
    /// Set when the editor wants the frame closed.
    closing: Option<String>,
    /// The name last saved to, cleared by the next change.
    last_saved: Option<String>,
    /// Set at load when the doc's first header cell is `#`: a grid the
    /// server owns the shape and name of.
    bridged: bool,
}

/// One editing session: the application, its input, and the state.
pub struct Session {
    app: Application,
    input: HostInput,
    state: State,
}

/// Width of a column: its widest text plus a space, within `[4, 24]`.
fn column_width(doc: &CsvDoc, col: usize) -> u16 {
    let widest = doc
        .body()
        .iter()
        .map(|r| r.get(col).map_or(0, |c| c.chars().count()))
        .chain([doc.header().get(col).map_or(0, |h| h.chars().count())])
        .max()
        .unwrap_or(0);
    u16::try_from((widest + 1).clamp(4, 24)).unwrap_or(24)
}

fn columns(doc: &CsvDoc) -> Vec<Column> {
    doc.header()
        .iter()
        .enumerate()
        .map(|(i, h)| Column::new(h.clone(), column_width(doc, i)))
        .collect()
}

/// The window's bounds in desktop space: all of it.
fn window_bounds(app: &Application) -> Rect {
    let d = app.desktop.bounds();
    Rect::new(0, 0, d.width(), d.height())
}

/// The table and the message row, in the window's interior space.
fn interior_bounds(window: Rect) -> (Rect, Rect) {
    let iw = (window.width() - 2).max(1);
    let ih = (window.height() - 2).max(2);
    (Rect::new(0, 0, iw, ih - 1), Rect::new(0, ih - 1, iw, ih))
}

/// The key hints the menus show. plank is macOS-only, and a stock Mac
/// terminal types Option as a character (no Alt+X), has no Insert key, and
/// keeps F10 behind fn, so each hint names the Control chord a Mac keyboard
/// can type. The PC keys (Alt+X, Ins, Del, Ctrl+Ins, Ctrl+Del) still work.
fn menu_bar(width: i16) -> MenuBar {
    let item = |text: &str, command: CommandId, shortcut: Option<&str>| {
        let b = MenuItemBuilder::new().text(text).command(command);
        match shortcut {
            Some(s) => b.shortcut(s).build(),
            None => b.build(),
        }
    };
    let file = SubMenu::new(
        "~F~ile",
        Menu::from_items(vec![
            item("~N~ew", CMD_NEW, None),
            item("~O~pen...", CMD_OPEN, Some("Ctrl+O")),
            item("~S~ave", CMD_SAVE, Some("Ctrl+S")),
            item("Save ~a~s...", CMD_SAVE_AS, None),
            MenuItem::separator(),
            item("E~x~it", CMD_EXIT, Some("Ctrl+Q")),
        ]),
    );
    let edit = SubMenu::new(
        "~E~dit",
        Menu::from_items(vec![
            item("~E~dit cell", CMD_EDIT_CELL, Some("Enter")),
            item("~R~ename column", CMD_RENAME_COL, None),
        ]),
    );
    let row = SubMenu::new(
        "~R~ow",
        Menu::from_items(vec![
            item("~I~nsert", CMD_ROW_INS, Some("Ctrl+R")),
            item("~D~elete", CMD_ROW_DEL, Some("Ctrl+Y")),
        ]),
    );
    let col = SubMenu::new(
        "~C~olumn",
        Menu::from_items(vec![
            item("~I~nsert", CMD_COL_INS, Some("Ctrl+L")),
            item("~D~elete", CMD_COL_DEL, Some("Ctrl+K")),
        ]),
    );
    let mut bar = MenuBar::new(Rect::new(0, 0, width, 1));
    for sub in [file, edit, row, col] {
        bar.add_submenu(sub);
    }
    bar
}

fn status_line(width: i16, height: i16) -> StatusLine {
    // Hint text only: every key here is claimed in `pre_event`, so the items
    // carry the commands for mouse clicks and bind no key of their own. Each
    // item takes its text plus four columns, so all five would need 70; at
    // the 60-column minimum Enter (the obvious key) gives way to F10, the one
    // way into the menus. F10 Menu has no command: the menu bar owns F10.
    let item = |text: &str, command: CommandId| {
        StatusItemBuilder::new().text(text).command(command).build()
    };
    StatusLine::new(
        Rect::new(0, height - 1, width, height),
        vec![
            item("~Ctrl-R~ Row", CMD_ROW_INS),
            item("~Ctrl-S~ Save", CMD_SAVE),
            item("~Ctrl-Q~ Exit", CMD_EXIT),
            item("~F10~ Menu", 0),
        ],
    )
}

/// The sentinel `command_run` sends for a bare `/csvedit:open` (no name): the
/// RAM disk refuses `/` as a file name, so it can never collide with a real
/// one, and `Session::open` reads it as "start blank, show the Open dialog".
pub const OPEN_DIALOG_ARG: &str = "/";

/// The document an argument names, its name, and a message to show.
fn load(arg: &str, disk: &dyn Disk) -> (CsvDoc, Option<String>, String) {
    if arg.is_empty() || arg == OPEN_DIALOG_ARG {
        return (CsvDoc::new_blank(3, 3), None, String::new());
    }
    match disk.read(arg) {
        Ok(text) => {
            let (doc, error) = CsvDoc::from_text(&text);
            let message = error.map_or_else(String::new, |line| {
                format!("parse error at line {line}; saving will drop the rest")
            });
            (doc, Some(arg.to_string()), message)
        }
        Err(e) => (CsvDoc::new_blank(3, 3), Some(arg.to_string()), e),
    }
}

impl Session {
    /// A session over `arg` (empty for a new document) on a `w` x `h` screen.
    ///
    /// # Panics
    /// Never in practice: a `HostBackend` terminal cannot fail to build.
    #[must_use]
    pub fn open(w: u16, h: u16, arg: &str, disk: Box<dyn Disk>) -> Self {
        let (backend, input) = HostBackend::new(w, h);
        let terminal = Terminal::with_backend(Box::new(backend)).expect("host terminal");
        let mut app = Application::with_terminal(terminal);
        let (sw, sh) = app.terminal.size();
        app.set_menu_bar(menu_bar(sw));
        app.set_status_line(status_line(sw, sh));

        let (doc, name, message) = load(arg, &*disk);
        let bridged = doc.header().first().is_some_and(|h| h == "#");

        let bounds = window_bounds(&app);
        let mut window = WindowBuilder::new().bounds(bounds).title("").build();
        // Closing the window is closing the editor, which asks first.
        window.set_auto_close(false);
        let (table_r, message_r) = interior_bounds(bounds);
        let mut grid = Table::new(table_r, CMD_EDIT_CELL);
        grid.set_column_separator(true);
        let table = window.add_typed(grid);
        let message = window.add_typed(ParamText::new(message_r, &message));
        let window = app.desktop.add_typed(window);

        let mut state = State {
            doc,
            name,
            disk,
            window,
            table,
            message,
            overlay: None,
            closing: None,
            last_saved: None,
            bridged,
        };
        state.refresh(&mut app);
        if bridged {
            // Disabling dims the menu items; the refusal messages below are
            // what actually stops the action, keys included.
            for c in [
                CMD_NEW,
                CMD_OPEN,
                CMD_SAVE_AS,
                CMD_RENAME_COL,
                CMD_COL_INS,
                CMD_COL_DEL,
            ] {
                app.disable_command(c);
            }
        }
        if arg == OPEN_DIALOG_ARG {
            state.open_picker(&mut app);
        }
        Self { app, input, state }
    }

    /// Handles one key. `Some(line)` means the editor wants to close, with
    /// `line` for the scrollback.
    pub fn key(&mut self, ev: Event) -> Option<String> {
        self.input.push(ev);
        let running = self.app.pump(&mut self.state);
        if self.state.closing.is_none() && !running {
            self.state.closing = Some(self.state.close_line());
        }
        self.state.closing.clone()
    }

    /// Draws a frame at `w` x `h`, re-laying out after a resize.
    pub fn step(&mut self, w: u16, h: u16) {
        let before = self.app.terminal.size();
        self.input.set_size(w, h);
        self.app.pump(&mut self.state);
        if self.app.terminal.size() != before {
            self.state.relayout(&mut self.app);
            self.app.draw();
        }
    }

    /// The last frame, every cell.
    #[must_use]
    pub fn cells(&self) -> Vec<plank_guest_support::CellGlyph> {
        crate::paint::cells(self.app.terminal.buffer())
    }

    /// The scrollback line for closing now.
    #[must_use]
    pub fn close_line(&self) -> String {
        self.state.close_line()
    }

    #[must_use]
    pub fn doc(&self) -> &CsvDoc {
        &self.state.doc
    }

    #[must_use]
    pub fn into_disk(self) -> Box<dyn Disk> {
        self.state.disk
    }
}

impl State {
    fn close_line(&self) -> String {
        match &self.last_saved {
            Some(name) if !self.doc.is_modified() => format!(
                "csvedit: saved {name} ({} rows \u{d7} {} columns)",
                self.doc.height(),
                self.doc.width()
            ),
            _ => "csvedit: closed without saving".to_string(),
        }
    }

    fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or("untitled.csv")
    }

    fn table<'a>(&self, app: &'a mut Application) -> Option<&'a mut Table> {
        app.desktop.get_mut(self.window)?.get_mut(self.table)
    }

    /// The selected cell, `(row, col)`.
    fn selection(&self, app: &mut Application) -> (usize, usize) {
        self.table(app).map_or((0, 0), |t| {
            (t.selected_row().unwrap_or(0), t.selected_col())
        })
    }

    pub(crate) fn set_message(&self, app: &mut Application, text: &str) {
        if let Some(m) = app
            .desktop
            .get_mut(self.window)
            .and_then(|w| w.get_mut(self.message))
        {
            m.set_template(text);
        }
    }

    /// Rebuilds the table from the document and retitles the window.
    fn refresh(&mut self, app: &mut Application) {
        if self.doc.is_modified() {
            self.last_saved = None;
        }
        let (row, col) = self.selection(app);
        let title = format!(
            "{}{}",
            self.display_name(),
            if self.doc.is_modified() { "*" } else { "" }
        );
        if let Some(t) = self.table(app) {
            t.set_columns(columns(&self.doc));
            t.set_rows(self.doc.body().to_vec());
            t.set_selected_row(row);
            t.set_selected_col(col);
        }
        if let Some(w) = app.desktop.get_mut(self.window) {
            w.set_title(&title);
        }
    }

    /// Fits the window, table and message to the desktop.
    fn relayout(&self, app: &mut Application) {
        let bounds = window_bounds(app);
        let (table_r, message_r) = interior_bounds(bounds);
        if let Some(w) = app.desktop.get_mut(self.window) {
            w.set_bounds(bounds);
            if let Some(t) = w.get_mut(self.table) {
                t.set_bounds(table_r);
            }
            if let Some(m) = w.get_mut(self.message) {
                m.set_bounds(message_r);
            }
        }
    }

    fn open_overlay(&mut self, overlay: Overlay) {
        self.overlay = Some(overlay);
    }

    /// Removes the open dialog and gives the focus back to the window.
    fn close_overlay(&mut self, app: &mut Application) {
        if let Some(o) = self.overlay.take() {
            app.desktop.remove_child_by_id(o.dialog_id());
            app.desktop.bring_to_front(self.window.id());
            if let Some(w) = app.desktop.get_mut(self.window) {
                w.set_focus(true);
            }
            app.needs_redraw();
        }
    }

    /// Writes the document under `name`. True on success.
    fn save_to(&mut self, app: &mut Application, name: &str) -> bool {
        match self.disk.write(name, &self.doc.to_text()) {
            Ok(()) => {
                self.doc.mark_saved();
                self.name = Some(name.to_string());
                self.last_saved = Some(name.to_string());
                self.set_message(app, &format!("saved {name}"));
                self.refresh(app);
                true
            }
            Err(e) => {
                self.set_message(app, &e);
                false
            }
        }
    }

    /// The step a save or a discard was waiting for.
    fn run(&mut self, app: &mut Application, then: After) {
        match then {
            After::Nothing => {}
            After::New => {
                self.doc = CsvDoc::new_blank(3, 3);
                self.name = None;
                self.last_saved = None;
                self.set_message(app, "");
                self.refresh(app);
            }
            After::Open => self.open_picker(app),
            After::Exit => self.closing = Some(self.close_line()),
        }
    }

    /// Runs `then` now, or asks first when there are unsaved changes.
    fn guard(&mut self, app: &mut Application, then: After) {
        if self.doc.is_modified() {
            let o = dialogs::unsaved(app, self.display_name(), then);
            self.open_overlay(o);
        } else {
            self.run(app, then);
        }
    }

    fn open_picker(&mut self, app: &mut Application) {
        match self.disk.list() {
            Ok(names) if names.is_empty() => self.set_message(app, "no saved files yet"),
            Ok(names) => {
                let o = dialogs::open(app, names);
                self.open_overlay(o);
            }
            Err(e) => self.set_message(app, &e),
        }
    }

    fn save_as(&mut self, app: &mut Application, then: After) {
        let preset = self.name.clone().unwrap_or_default();
        let o = dialogs::save_as(app, &preset, then);
        self.open_overlay(o);
    }

    /// `CMD_DLG_OK`: commit what the open dialog was editing.
    fn accept(&mut self, app: &mut Application) {
        let Some(overlay) = self.overlay.clone() else {
            return;
        };
        let value = overlay.value(app);
        match overlay {
            Overlay::EditCell { row, col, .. } => {
                self.close_overlay(app);
                self.doc.set_cell(row, col, value);
                self.refresh(app);
            }
            Overlay::RenameCol { col, .. } => {
                self.close_overlay(app);
                if !value.is_empty() {
                    self.doc.rename_col(col, value);
                }
                self.refresh(app);
            }
            Overlay::SaveAs { then, .. } => {
                let name = value.trim().to_string();
                if name.is_empty() {
                    self.set_message(app, "a file name is needed");
                    return;
                }
                // The Open dialog lists only the disk's root, so a file saved
                // in a subfolder could never be opened again.
                if name.contains('/') {
                    self.set_message(app, "names cannot contain '/'");
                    return;
                }
                let name = if name.to_ascii_lowercase().ends_with(".csv") {
                    name
                } else {
                    format!("{name}.csv")
                };
                // Keep the dialog open (with the name still in it) on a
                // refused write, so the typed name is not lost; only a
                // successful save closes it.
                if self.save_to(app, &name) {
                    self.close_overlay(app);
                    self.run(app, then);
                }
            }
            Overlay::Open { .. } => {
                self.close_overlay(app);
                if value.is_empty() {
                    return;
                }
                let (doc, name, message) = load(&value, &*self.disk);
                self.doc = doc;
                self.name = name;
                self.last_saved = None;
                self.set_message(app, &message);
                if let Some(t) = self.table(app) {
                    t.set_selected_row(0);
                    t.set_selected_col(0);
                }
                self.refresh(app);
            }
            Overlay::Unsaved { then, .. } => {
                self.close_overlay(app);
                match self.name.clone() {
                    Some(name) => {
                        if self.save_to(app, &name) {
                            self.run(app, then);
                        }
                    }
                    None => self.save_as(app, then),
                }
            }
        }
    }

    fn edit_cell(&mut self, app: &mut Application) {
        let (row, col) = self.selection(app);
        if self.bridged && col == 0 {
            self.set_message(app, "the # column is fixed");
            return;
        }
        let o = dialogs::edit_cell(app, self.doc.cell(row, col), row, col);
        self.open_overlay(o);
    }

    fn rename_col(&mut self, app: &mut Application) {
        if self.bridged {
            self.set_message(app, "this grid's columns are fixed");
            return;
        }
        let (_, col) = self.selection(app);
        let current = self.doc.header().get(col).cloned().unwrap_or_default();
        let o = dialogs::rename_col(app, &current, col);
        self.open_overlay(o);
    }

    /// A row or column command at the selection.
    fn reshape(&mut self, app: &mut Application, command: CommandId) {
        if self.bridged && matches!(command, CMD_COL_INS | CMD_COL_DEL) {
            self.set_message(app, "this grid's columns are fixed");
            return;
        }
        let (row, col) = self.selection(app);
        match command {
            CMD_ROW_INS => self.doc.insert_row(row),
            CMD_ROW_DEL => self.doc.delete_row(row),
            CMD_COL_INS => self.doc.insert_col(col),
            _ => self.doc.delete_col(col),
        }
        self.refresh(app);
    }

    /// The message for a bridged doc's refused New/Open/Save As.
    fn stays_message(&self) -> String {
        format!("this grid stays on {}", self.display_name())
    }
}

impl AppHandler for State {
    fn pre_event(&mut self, app: &mut Application, event: &mut Event) {
        // Alt+X quits inside Application::handle_event before any handler
        // sees it, and a CM_QUIT command is consumed there too; both become
        // CMD_EXIT so the unsaved check runs first.
        let quit_key = event.what == EventType::Keyboard && event.key_code == KB_ALT_X;
        let quit_cmd = event.what == EventType::Command && event.command == CM_QUIT;
        if quit_key || quit_cmd {
            *event = Event::command(CMD_EXIT);
            return;
        }
        // Keys claimed here would never reach an open menu.
        if app.menu_is_open() || event.what != EventType::Keyboard {
            return;
        }
        if let Some(overlay) = &self.overlay {
            match event.key_code {
                KB_ESC => *event = Event::command(CMD_DLG_CANCEL),
                // A non-modal dialog leaves Enter alone, so Enter in its
                // input line is taken here as OK; a focused button still
                // turns Enter into its own command.
                KB_ENTER if overlay.input_has_focus(app) => {
                    *event = Event::command(CMD_DLG_OK);
                }
                _ => {}
            }
            return;
        }
        let command = match event.key_code {
            KB_INS if is_ctrl(event) => CMD_COL_INS,
            KB_DEL if is_ctrl(event) => CMD_COL_DEL,
            KB_INS => CMD_ROW_INS,
            KB_DEL => CMD_ROW_DEL,
            KB_CTRL_S => CMD_SAVE,
            // The same commands on chords a Mac keyboard can type.
            KB_CTRL_Q => CMD_EXIT,
            KB_CTRL_O => CMD_OPEN,
            KB_CTRL_R => CMD_ROW_INS,
            KB_CTRL_Y => CMD_ROW_DEL,
            KB_CTRL_L => CMD_COL_INS,
            KB_CTRL_K => CMD_COL_DEL,
            _ => return,
        };
        *event = Event::command(command);
    }

    fn handle_command(&mut self, app: &mut Application, command: CommandId, _: &Event) -> bool {
        // A frame's close box (or Alt+F3) on a dialog cancels it; on the
        // editor's window it is Exit. Neither window closes itself.
        if command == CM_CLOSE {
            if self.overlay.is_some() {
                self.close_overlay(app);
            } else {
                self.guard(app, After::Exit);
            }
            return true;
        }
        // Only the dialog commands act while a dialog is open; a menu pick
        // behind it must not stack a second one.
        let dialog_command = matches!(
            command,
            CMD_DLG_OK | CMD_DLG_CANCEL | CMD_DISCARD | CMD_OPEN_PICK
        );
        if self.overlay.is_some() != dialog_command {
            return is_ours(command);
        }
        match command {
            CMD_EDIT_CELL => self.edit_cell(app),
            CMD_RENAME_COL => self.rename_col(app),
            CMD_DLG_OK | CMD_OPEN_PICK => self.accept(app),
            CMD_DLG_CANCEL => self.close_overlay(app),
            CMD_DISCARD => {
                let then = self.overlay.as_ref().map_or(After::Nothing, Overlay::then);
                self.close_overlay(app);
                self.run(app, then);
            }
            CMD_SAVE => match self.name.clone() {
                Some(name) => {
                    self.save_to(app, &name);
                }
                None => self.save_as(app, After::Nothing),
            },
            CMD_SAVE_AS if self.bridged => {
                let msg = self.stays_message();
                self.set_message(app, &msg);
            }
            CMD_SAVE_AS => self.save_as(app, After::Nothing),
            CMD_NEW if self.bridged => {
                let msg = self.stays_message();
                self.set_message(app, &msg);
            }
            CMD_NEW => self.guard(app, After::New),
            CMD_OPEN if self.bridged => {
                let msg = self.stays_message();
                self.set_message(app, &msg);
            }
            CMD_OPEN => self.guard(app, After::Open),
            CMD_EXIT => self.guard(app, After::Exit),
            CMD_ROW_INS | CMD_ROW_DEL | CMD_COL_INS | CMD_COL_DEL => self.reshape(app, command),
            _ => return false,
        }
        true
    }
}

/// Whether a command is one of the editor's own.
fn is_ours(command: CommandId) -> bool {
    (CMD_NEW..=CMD_OPEN_PICK).contains(&command)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disk::MemDisk;
    use crate::keys::translate;

    fn type_str(s: &mut Session, text: &str) {
        for c in text.chars() {
            assert!(
                s.key(translate(&c.to_lowercase().to_string(), Some(c)).unwrap())
                    .is_none()
            );
        }
    }
    fn press(s: &mut Session, code: &str) -> Option<String> {
        s.key(translate(code, None).unwrap())
    }
    fn screen(s: &Session) -> String {
        let mut rows = vec![vec![' '; 80]; 24];
        for c in s.cells() {
            if let Some(r) = rows.get_mut(usize::from(c.y))
                && let Some(cell) = r.get_mut(usize::from(c.x))
            {
                *cell = c.ch;
            }
        }
        rows.into_iter()
            .map(|r| r.into_iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn new_session() -> Session {
        let mut s = Session::open(80, 24, "", Box::new(MemDisk::default()));
        s.step(80, 24);
        s
    }

    #[test]
    fn a_new_session_shows_the_menu_title_and_columns() {
        let s = new_session();
        let text = screen(&s);
        assert!(text.contains("File"), "{text}");
        assert!(text.contains("untitled.csv"), "{text}");
        assert!(text.contains(" A ") || text.contains("A "), "{text}");
    }

    #[test]
    fn enter_edits_a_cell_and_enter_commits() {
        let mut s = new_session();
        press(&mut s, "enter");
        type_str(&mut s, "Hi, there");
        press(&mut s, "enter");
        s.step(80, 24);
        assert_eq!(s.doc().cell(0, 0), "Hi, there");
        assert!(screen(&s).contains("untitled.csv*"));
    }

    #[test]
    fn escape_cancels_an_edit_and_never_closes_the_editor() {
        let mut s = new_session();
        press(&mut s, "enter");
        type_str(&mut s, "nope");
        assert!(press(&mut s, "escape").is_none());
        assert_eq!(s.doc().cell(0, 0), "");
        assert!(
            press(&mut s, "escape").is_none(),
            "Esc with no dialog does nothing"
        );
    }

    #[test]
    fn insert_and_delete_rows_and_columns() {
        let mut s = new_session();
        press(&mut s, "insert");
        assert_eq!(s.doc().height(), 4);
        press(&mut s, "delete");
        assert_eq!(s.doc().height(), 3);
        press(&mut s, "ctrl-insert");
        assert_eq!(s.doc().width(), 4);
        press(&mut s, "ctrl-delete");
        assert_eq!(s.doc().width(), 3);
    }

    #[test]
    fn save_as_then_reopen_from_the_disk() {
        let mut s = new_session();
        press(&mut s, "enter");
        type_str(&mut s, "v");
        press(&mut s, "enter");
        press(&mut s, "ctrl-s"); // untitled -> Save As dialog
        type_str(&mut s, "budget.csv");
        press(&mut s, "enter");
        s.step(80, 24);
        assert!(!s.doc().is_modified());
        assert!(screen(&s).contains("budget.csv"));
        let disk = s.into_disk();
        assert_eq!(disk.read("budget.csv").unwrap(), "A,B,C\nv,,\n,,\n,,\n");

        let mut again = Session::open(80, 24, "budget.csv", disk);
        again.step(80, 24);
        assert_eq!(again.doc().cell(0, 0), "v");
    }

    #[test]
    fn exit_with_unsaved_changes_asks_and_discard_closes() {
        let mut s = new_session();
        press(&mut s, "enter");
        type_str(&mut s, "x");
        press(&mut s, "enter");
        assert!(press(&mut s, "alt-x").is_none(), "asks first");
        s.step(80, 24);
        assert!(screen(&s).contains("Discard"), "{}", screen(&s));
        // The dialog's default button is Save; Tab to Discard, then Enter.
        press(&mut s, "tab");
        let line = press(&mut s, "enter").expect("closes");
        assert!(line.contains("closed without saving"), "{line}");
    }

    #[test]
    fn exit_when_clean_closes_at_once() {
        let mut s = new_session();
        let line = press(&mut s, "alt-x").expect("closes");
        assert!(line.contains("csvedit"), "{line}");
    }

    #[test]
    fn a_missing_file_opens_blank_with_a_message() {
        let mut s = Session::open(80, 24, "nope.csv", Box::new(MemDisk::default()));
        s.step(80, 24);
        assert!(screen(&s).contains("no such file"), "{}", screen(&s));
        assert_eq!(s.doc().width(), 3);
    }

    #[test]
    fn a_parse_error_names_the_line() {
        let mut disk = MemDisk::default();
        crate::disk::Disk::write(&mut disk, "bad.csv", "a\n\"open").unwrap();
        let mut s = Session::open(80, 24, "bad.csv", Box::new(disk));
        s.step(80, 24);
        assert!(screen(&s).contains("line 2"), "{}", screen(&s));
    }

    /// A disk whose writes are always refused, like a plank quota error.
    /// Reads and listing behave like the `MemDisk` it wraps.
    struct FailDisk(MemDisk);

    impl Disk for FailDisk {
        fn read(&self, path: &str) -> Result<String, String> {
            self.0.read(path)
        }
        fn write(&mut self, _path: &str, _text: &str) -> Result<(), String> {
            Err("'dev.plank.csvedit' disk would grow to ... limit".into())
        }
        fn list(&self) -> Result<Vec<String>, String> {
            self.0.list()
        }
    }

    #[test]
    fn a_refused_save_keeps_the_doc_modified_and_shows_the_error() {
        let mut s = Session::open(80, 24, "", Box::new(FailDisk(MemDisk::default())));
        s.step(80, 24);
        press(&mut s, "enter");
        type_str(&mut s, "x");
        press(&mut s, "enter");
        assert!(press(&mut s, "ctrl-s").is_none(), "untitled -> Save As");
        type_str(&mut s, "x.csv");
        assert!(
            press(&mut s, "enter").is_none(),
            "refused save must not close"
        );
        s.step(80, 24);
        assert!(s.doc().is_modified());
        let text = screen(&s);
        assert!(text.contains("limit"), "{text}");
        // Save As fix: the dialog stays open with the typed name preserved,
        // instead of losing it when the write fails.
        assert!(text.contains("x.csv"), "{text}");
    }

    #[test]
    fn exit_with_a_refused_save_does_not_close() {
        let mut s = Session::open(80, 24, "", Box::new(FailDisk(MemDisk::default())));
        s.step(80, 24);
        press(&mut s, "enter");
        type_str(&mut s, "x");
        press(&mut s, "enter");
        assert!(press(&mut s, "alt-x").is_none(), "asks first");
        assert!(
            press(&mut s, "enter").is_none(),
            "Save on untitled asks for a name"
        );
        type_str(&mut s, "out");
        assert!(
            press(&mut s, "enter").is_none(),
            "a refused save must not close the editor"
        );
        s.step(80, 24);
        assert!(s.doc().is_modified());
        assert!(screen(&s).contains("limit"), "{}", screen(&s));
    }

    fn disk_with(files: &[(&str, &str)]) -> Box<dyn Disk> {
        let mut disk = MemDisk::default();
        for (name, text) in files {
            crate::disk::Disk::write(&mut disk, name, text).unwrap();
        }
        Box::new(disk)
    }

    #[test]
    fn rename_column_replaces_the_header() {
        let mut s = new_session();
        assert!(s.key(Event::command(CMD_RENAME_COL)).is_none());
        press(&mut s, "backspace"); // the preset "A"
        type_str(&mut s, "Total");
        press(&mut s, "enter");
        assert_eq!(s.doc().header()[0], "Total");
    }

    #[test]
    fn ctrl_s_on_a_named_file_saves_at_once_and_exit_reports_it() {
        let mut s = Session::open(80, 24, "t.csv", disk_with(&[("t.csv", "a,b\n1,2\n")]));
        s.step(80, 24);
        press(&mut s, "enter");
        press(&mut s, "backspace");
        type_str(&mut s, "9");
        press(&mut s, "enter");
        assert!(s.doc().is_modified());
        press(&mut s, "ctrl-s");
        s.step(80, 24);
        assert!(!s.doc().is_modified());
        assert!(screen(&s).contains("saved t.csv"), "{}", screen(&s));
        let line = press(&mut s, "alt-x").expect("closes");
        assert_eq!(line, "csvedit: saved t.csv (1 rows \u{d7} 2 columns)");
        assert_eq!(s.into_disk().read("t.csv").unwrap(), "a,b\n9,2\n");
    }

    #[test]
    fn saving_from_the_unsaved_dialog_goes_through_save_as_then_exits() {
        let mut s = new_session();
        press(&mut s, "enter");
        type_str(&mut s, "x");
        press(&mut s, "enter");
        assert!(press(&mut s, "alt-x").is_none());
        assert!(
            press(&mut s, "enter").is_none(),
            "Save on untitled asks for a name"
        );
        type_str(&mut s, "out");
        let line = press(&mut s, "enter").expect("saved, then closed");
        assert!(line.contains("saved out.csv"), "{line}");
        assert!(s.into_disk().read("out.csv").is_ok());
    }

    #[test]
    fn cancel_in_the_unsaved_dialog_keeps_editing() {
        let mut s = new_session();
        press(&mut s, "insert");
        assert!(press(&mut s, "alt-x").is_none());
        press(&mut s, "tab");
        press(&mut s, "tab");
        assert!(press(&mut s, "enter").is_none(), "Cancel");
        s.step(80, 24);
        assert!(!screen(&s).contains("Discard"));
        press(&mut s, "insert");
        assert_eq!(s.doc().height(), 5, "keys reach the table again");
    }

    #[test]
    fn open_lists_saved_files_and_enter_opens_the_selected_one() {
        let mut s = Session::open(
            80,
            24,
            "",
            disk_with(&[("a.csv", "h\n1\n"), ("b.csv", "k\n2\n")]),
        );
        s.step(80, 24);
        assert!(s.key(Event::command(CMD_OPEN)).is_none());
        s.step(80, 24);
        assert!(screen(&s).contains("b.csv"), "{}", screen(&s));
        press(&mut s, "down");
        press(&mut s, "enter");
        s.step(80, 24);
        assert_eq!(s.doc().header()[0], "k");
        assert!(screen(&s).contains("b.csv"));
    }

    #[test]
    fn open_with_nothing_saved_says_so() {
        let mut s = new_session();
        s.key(Event::command(CMD_OPEN));
        s.step(80, 24);
        assert!(screen(&s).contains("no saved files yet"), "{}", screen(&s));
    }

    #[test]
    fn opening_with_the_slash_sentinel_shows_a_blank_doc_and_the_open_dialog() {
        let mut s = Session::open(80, 24, "/", disk_with(&[("a.csv", "h\n1\n")]));
        s.step(80, 24);
        let text = screen(&s);
        assert!(text.contains("untitled.csv"), "{text}");
        assert!(text.contains("a.csv"), "{text}");
        assert_eq!(s.doc().width(), 3);
        press(&mut s, "enter");
        s.step(80, 24);
        assert_eq!(s.doc().header()[0], "h");
    }

    #[test]
    fn opening_with_the_slash_sentinel_and_no_saved_files_says_so() {
        let mut s = Session::open(80, 24, "/", Box::new(MemDisk::default()));
        s.step(80, 24);
        assert!(screen(&s).contains("no saved files yet"), "{}", screen(&s));
        assert_eq!(s.doc().width(), 3);
    }

    #[test]
    fn a_close_box_cancels_the_dialog_and_on_the_window_exits() {
        let mut s = new_session();
        press(&mut s, "enter");
        assert!(s.key(Event::command(CM_CLOSE)).is_none());
        s.step(80, 24);
        assert!(!screen(&s).contains("Edit cell"), "{}", screen(&s));
        let line = s.key(Event::command(CM_CLOSE)).expect("clean exit");
        assert!(line.contains("closed without saving"), "{line}");
    }

    #[test]
    fn step_relayouts_to_a_new_size() {
        let mut s = new_session();
        s.step(60, 15);
        let cells = s.cells();
        assert_eq!(cells.len(), 60 * 15);
        let row1: String = cells.iter().filter(|c| c.y == 1).map(|c| c.ch).collect();
        assert!(row1.contains("untitled.csv"), "{row1}");
        let right = cells.iter().find(|c| c.y == 1 && c.x == 59).map(|c| c.ch);
        assert_eq!(
            right,
            Some('\u{2557}'),
            "the frame's corner is at the new edge: {row1}"
        );
    }

    #[test]
    fn mac_chords_insert_and_delete_rows_and_columns() {
        let mut s = new_session();
        assert!(press(&mut s, "ctrl-r").is_none());
        assert_eq!(s.doc().height(), 4);
        assert!(press(&mut s, "ctrl-y").is_none());
        assert_eq!(s.doc().height(), 3);
        assert!(press(&mut s, "ctrl-l").is_none());
        assert_eq!(s.doc().width(), 4);
        assert!(press(&mut s, "ctrl-k").is_none());
        assert_eq!(s.doc().width(), 3);
    }

    #[test]
    fn ctrl_o_shows_the_open_dialog() {
        let mut s = Session::open(80, 24, "", disk_with(&[("a.csv", "h\n1\n")]));
        s.step(80, 24);
        assert!(press(&mut s, "ctrl-o").is_none());
        s.step(80, 24);
        assert!(screen(&s).contains("a.csv"), "{}", screen(&s));
        press(&mut s, "enter");
        assert_eq!(s.doc().header()[0], "h");
    }

    #[test]
    fn ctrl_q_with_unsaved_changes_asks_first() {
        let mut s = new_session();
        press(&mut s, "ctrl-r");
        assert!(press(&mut s, "ctrl-q").is_none(), "asks first");
        s.step(80, 24);
        assert!(screen(&s).contains("Discard"), "{}", screen(&s));
        press(&mut s, "tab");
        let line = press(&mut s, "enter").expect("closes");
        assert!(line.contains("closed without saving"), "{line}");
    }

    #[test]
    fn ctrl_q_when_clean_closes_at_once() {
        let mut s = new_session();
        let line = press(&mut s, "ctrl-q").expect("closes");
        assert!(line.contains("csvedit"), "{line}");
    }

    #[test]
    fn ctrl_q_inside_a_dialog_does_not_exit() {
        let mut s = new_session();
        press(&mut s, "enter");
        assert!(press(&mut s, "ctrl-q").is_none());
        s.step(80, 24);
        assert!(screen(&s).contains("Edit cell"), "{}", screen(&s));
    }

    #[test]
    fn the_status_line_shows_the_mac_chords_at_the_minimum_width() {
        let mut s = new_session();
        s.step(60, 16);
        let bottom: String = s
            .cells()
            .iter()
            .filter(|c| c.y == 15)
            .map(|c| c.ch)
            .collect();
        for hint in ["Ctrl-R Row", "Ctrl-S Save", "Ctrl-Q Exit", "F10 Menu"] {
            assert!(bottom.contains(hint), "{hint}: {bottom}");
        }
    }

    /// A disk holding a bridged grid: its header's first cell is `#`.
    fn bridged_disk() -> Box<dyn Disk> {
        disk_with(&[("grid.csv", "#,name\n1,alice\n2,bob\n")])
    }

    fn bridged_session() -> Session {
        let mut s = Session::open(80, 24, "grid.csv", bridged_disk());
        s.step(80, 24);
        s
    }

    #[test]
    fn bridged_column_commands_are_refused() {
        let mut s = bridged_session();
        assert!(press(&mut s, "ctrl-l").is_none());
        assert_eq!(s.doc().width(), 2, "no column inserted");
        s.step(80, 24);
        assert!(
            screen(&s).contains("this grid's columns are fixed"),
            "{}",
            screen(&s)
        );

        assert!(press(&mut s, "ctrl-k").is_none());
        assert_eq!(s.doc().width(), 2, "no column deleted");
        s.step(80, 24);
        assert!(
            screen(&s).contains("this grid's columns are fixed"),
            "{}",
            screen(&s)
        );

        assert!(s.key(Event::command(CMD_RENAME_COL)).is_none());
        s.step(80, 24);
        assert!(
            screen(&s).contains("this grid's columns are fixed"),
            "{}",
            screen(&s)
        );
        assert_eq!(s.doc().header()[0], "#", "no rename either");
    }

    #[test]
    fn bridged_hash_cell_edit_is_refused() {
        let mut s = bridged_session();
        // Selection starts at (0, 0): the # cell.
        assert!(press(&mut s, "enter").is_none());
        s.step(80, 24);
        assert!(
            screen(&s).contains("the # column is fixed"),
            "{}",
            screen(&s)
        );
        assert!(!screen(&s).contains("Edit cell"), "no dialog opened");
        assert_eq!(s.doc().cell(0, 0), "1");
    }

    #[test]
    fn bridged_non_hash_cell_edit_still_works() {
        let mut s = bridged_session();
        press(&mut s, "right"); // move to column 1 ("name")
        assert!(press(&mut s, "enter").is_none());
        press(&mut s, "backspace");
        press(&mut s, "backspace");
        press(&mut s, "backspace");
        press(&mut s, "backspace");
        press(&mut s, "backspace");
        type_str(&mut s, "carol");
        press(&mut s, "enter");
        s.step(80, 24);
        assert_eq!(s.doc().cell(0, 1), "carol");
    }

    #[test]
    fn bridged_ctrl_s_saves_to_the_same_name_with_no_save_as() {
        let mut s = bridged_session();
        press(&mut s, "right");
        press(&mut s, "enter");
        type_str(&mut s, "!");
        press(&mut s, "enter");
        assert!(press(&mut s, "ctrl-s").is_none(), "saves at once");
        s.step(80, 24);
        assert!(!s.doc().is_modified());
        assert!(screen(&s).contains("saved grid.csv"), "{}", screen(&s));
    }

    #[test]
    fn bridged_new_open_save_as_are_refused() {
        let mut s = bridged_session();
        assert!(s.key(Event::command(CMD_NEW)).is_none());
        s.step(80, 24);
        assert!(
            screen(&s).contains("this grid stays on grid.csv"),
            "{}",
            screen(&s)
        );
        assert_eq!(s.doc().header()[0], "#", "New refused");

        assert!(s.key(Event::command(CMD_OPEN)).is_none());
        s.step(80, 24);
        assert!(
            screen(&s).contains("this grid stays on grid.csv"),
            "{}",
            screen(&s)
        );
        assert!(!screen(&s).contains("Open"), "no Open dialog");

        assert!(s.key(Event::command(CMD_SAVE_AS)).is_none());
        s.step(80, 24);
        assert!(
            screen(&s).contains("this grid stays on grid.csv"),
            "{}",
            screen(&s)
        );
        assert!(!screen(&s).contains("Save as"), "no Save As dialog");
    }

    #[test]
    fn bridged_row_insert_leaves_the_hash_cell_empty_and_delete_still_works() {
        let mut s = bridged_session();
        assert!(press(&mut s, "insert").is_none());
        assert_eq!(s.doc().height(), 3);
        assert_eq!(s.doc().cell(0, 0), "", "new row's # cell is empty");

        assert!(press(&mut s, "delete").is_none());
        assert_eq!(s.doc().height(), 2);
    }

    #[test]
    fn a_non_hash_doc_is_unaffected_by_bridged_mode() {
        // Existing behavior: an ordinary doc allows column insert, rename
        // and Save As without any refusal message.
        let mut s = new_session();
        assert!(press(&mut s, "ctrl-l").is_none());
        assert_eq!(s.doc().width(), 4);
        s.step(80, 24);
        assert!(!screen(&s).contains("columns are fixed"), "{}", screen(&s));
    }

    #[test]
    fn save_as_refuses_a_name_with_a_slash_and_keeps_the_dialog() {
        let mut s = new_session();
        press(&mut s, "ctrl-r");
        assert!(press(&mut s, "ctrl-s").is_none(), "untitled -> Save As");
        type_str(&mut s, "sub/x.csv");
        assert!(press(&mut s, "enter").is_none());
        s.step(80, 24);
        let text = screen(&s);
        assert!(text.contains("names cannot contain '/'"), "{text}");
        assert!(text.contains("sub/x.csv"), "the typed name stays: {text}");
        assert!(s.doc().is_modified());
        assert!(s.into_disk().list().unwrap().is_empty());
    }
}
