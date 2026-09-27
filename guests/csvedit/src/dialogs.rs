//! The editor's dialogs: plain, non-modal `Dialog`s on the desktop whose
//! buttons send the `CMD_DLG_*` commands back to the editor's handler.

use turbo_vision::app::Application;
use turbo_vision::core::geometry::Rect;
use turbo_vision::views::View;
use turbo_vision::views::button::Button;
use turbo_vision::views::dialog::Dialog;
use turbo_vision::views::group::GroupLike;
use turbo_vision::views::handle::Handle;
use turbo_vision::views::input_line::InputLine;
use turbo_vision::views::listbox::ListBox;
use turbo_vision::views::static_text::StaticText;
use turbo_vision::views::view::ViewId;
use turbo_vision::views::window::WindowLike;

use crate::commands::{CMD_DISCARD, CMD_DLG_CANCEL, CMD_DLG_OK, CMD_OPEN_PICK};
use crate::editor::After;

/// The longest text an input line takes.
const INPUT_MAX: usize = 1024;

/// What an open dialog is for.
#[derive(Debug, Clone)]
pub(crate) enum Overlay {
    EditCell {
        dialog: Handle<Dialog>,
        input: Handle<InputLine>,
        row: usize,
        col: usize,
    },
    RenameCol {
        dialog: Handle<Dialog>,
        input: Handle<InputLine>,
        col: usize,
    },
    SaveAs {
        dialog: Handle<Dialog>,
        input: Handle<InputLine>,
        then: After,
    },
    Open {
        dialog: Handle<Dialog>,
        list: Handle<ListBox>,
        names: Vec<String>,
    },
    Unsaved {
        dialog: Handle<Dialog>,
        then: After,
    },
}

impl Overlay {
    fn dialog(&self) -> Handle<Dialog> {
        match self {
            Self::EditCell { dialog, .. }
            | Self::RenameCol { dialog, .. }
            | Self::SaveAs { dialog, .. }
            | Self::Open { dialog, .. }
            | Self::Unsaved { dialog, .. } => *dialog,
        }
    }

    pub(crate) fn dialog_id(&self) -> ViewId {
        self.dialog().id()
    }

    fn input(&self) -> Option<Handle<InputLine>> {
        match self {
            Self::EditCell { input, .. }
            | Self::RenameCol { input, .. }
            | Self::SaveAs { input, .. } => Some(*input),
            Self::Open { .. } | Self::Unsaved { .. } => None,
        }
    }

    /// What a save or a discard from this dialog leads to.
    pub(crate) fn then(&self) -> After {
        match self {
            Self::SaveAs { then, .. } | Self::Unsaved { then, .. } => *then,
            _ => After::Nothing,
        }
    }

    /// The input line's text, or the list's selected name; empty otherwise.
    pub(crate) fn value(&self, app: &Application) -> String {
        let Some(dialog) = app.desktop.get(self.dialog()) else {
            return String::new();
        };
        if let Some(input) = self.input() {
            return dialog
                .get(input)
                .map_or_else(String::new, |i| i.text().to_string());
        }
        match self {
            Self::Open { list, names, .. } => dialog
                .get(*list)
                .and_then(ListBox::get_selection)
                .and_then(|i| names.get(i))
                .cloned()
                .unwrap_or_default(),
            _ => String::new(),
        }
    }

    /// Whether the dialog's input line holds the focus, so Enter means OK.
    pub(crate) fn input_has_focus(&self, app: &Application) -> bool {
        let (Some(input), Some(dialog)) = (self.input(), app.desktop.get(self.dialog())) else {
            return false;
        };
        dialog.group().focused_child().is_some_and(|c| {
            c.as_any().downcast_ref::<InputLine>().is_some() && dialog.get(input).is_some()
        })
    }
}

/// A `w` x `h` dialog centred on the desktop.
fn centred(app: &Application, w: i16, h: i16, title: &str) -> Dialog {
    let d = app.desktop.bounds();
    let w = w.min(d.width());
    let h = h.min(d.height());
    let x = (d.width() - w) / 2;
    let y = (d.height() - h) / 2;
    let mut dialog = Dialog::new(Rect::new(x, y, x + w, y + h), title);
    // Its close box sends CM_CLOSE on to the editor, which closes the overlay.
    dialog.window_mut().set_auto_close(false);
    dialog
}

/// Buttons centred on interior row `y` of a dialog `w` wide, in order.
fn buttons(dialog: &mut Dialog, w: i16, y: i16, specs: &[(&str, u16, bool)]) {
    let bw: i16 = 12;
    let gap: i16 = 2;
    let n = i16::try_from(specs.len()).unwrap_or(1);
    let total = n * bw + (n - 1) * gap;
    let mut x = ((w - 2 - total) / 2).max(0);
    for (title, command, default) in specs {
        dialog.add(Button::new(
            Rect::new(x, y, x + bw, y + 2),
            title,
            *command,
            *default,
        ));
        x += bw + gap;
    }
}

/// A dialog with a prompt, an input line preset with `text`, and OK/Cancel.
fn prompt(
    app: &mut Application,
    title: &str,
    label: &str,
    text: &str,
) -> (Handle<Dialog>, Handle<InputLine>) {
    let w: i16 = 52;
    let mut dialog = centred(app, w, 9, title);
    dialog.add(StaticText::new(Rect::new(1, 0, w - 3, 1), label));
    let mut line = InputLine::new(Rect::new(1, 2, w - 3, 3), INPUT_MAX);
    line.set_text(text);
    let input = dialog.add_typed(line);
    buttons(
        &mut dialog,
        w,
        4,
        &[
            ("~O~K", CMD_DLG_OK, true),
            ("Cancel", CMD_DLG_CANCEL, false),
        ],
    );
    dialog.set_initial_focus();
    (app.desktop.add_typed(dialog), input)
}

pub(crate) fn edit_cell(app: &mut Application, text: &str, row: usize, col: usize) -> Overlay {
    let (dialog, input) = prompt(
        app,
        "Edit cell",
        &format!("Row {}, column {}:", row + 1, col + 1),
        text,
    );
    Overlay::EditCell {
        dialog,
        input,
        row,
        col,
    }
}

pub(crate) fn rename_col(app: &mut Application, name: &str, col: usize) -> Overlay {
    let (dialog, input) = prompt(app, "Rename column", "Column name:", name);
    Overlay::RenameCol { dialog, input, col }
}

pub(crate) fn save_as(app: &mut Application, name: &str, then: After) -> Overlay {
    let (dialog, input) = prompt(app, "Save as", "File name:", name);
    Overlay::SaveAs {
        dialog,
        input,
        then,
    }
}

pub(crate) fn open(app: &mut Application, names: Vec<String>) -> Overlay {
    let w: i16 = 44;
    let rows = i16::try_from(names.len()).unwrap_or(i16::MAX).clamp(3, 10);
    let mut dialog = centred(app, w, rows + 6, "Open");
    let mut list = ListBox::new(Rect::new(1, 0, w - 3, rows), CMD_OPEN_PICK);
    list.set_items(names.clone());
    let list = dialog.add_typed(list);
    buttons(
        &mut dialog,
        w,
        rows + 1,
        &[
            ("~O~pen", CMD_DLG_OK, true),
            ("Cancel", CMD_DLG_CANCEL, false),
        ],
    );
    dialog.set_initial_focus();
    Overlay::Open {
        dialog: app.desktop.add_typed(dialog),
        list,
        names,
    }
}

pub(crate) fn unsaved(app: &mut Application, name: &str, then: After) -> Overlay {
    let w: i16 = 50;
    let mut dialog = centred(app, w, 8, "Unsaved changes");
    dialog.add(StaticText::new(
        Rect::new(1, 1, w - 3, 2),
        &format!("Save changes to {name}?"),
    ));
    buttons(
        &mut dialog,
        w,
        3,
        &[
            ("~S~ave", CMD_DLG_OK, true),
            ("~D~iscard", CMD_DISCARD, false),
            ("Cancel", CMD_DLG_CANCEL, false),
        ],
    );
    dialog.set_initial_focus();
    Overlay::Unsaved {
        dialog: app.desktop.add_typed(dialog),
        then,
    }
}
