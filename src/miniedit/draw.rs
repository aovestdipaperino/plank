// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT
//
// Layout adapted from Microsoft Edit's `bin/edit/draw_editor.rs` and
// `bin/edit/draw_statusbar.rs` (MIT, (c) Microsoft Corporation), reduced to a
// single in-memory document with no file affordances.

//! One frame of the built-in editor: the text area, a one-line status bar,
//! and the two ways out.
//!
//! `edit`'s TUI is immediate mode: this runs several times per input batch and
//! must stay a pure function of [`State`] plus the context's input.

use edit::buffer::SearchOptions;
use edit::framebuffer::IndexedColor;
use edit::helpers::{COORD_TYPE_SAFE_MAX, Rect, Size};
use edit::input::{kbmod, vk};
use edit::tui::{ButtonStyle, Context, Position};

use crate::miniedit::state::{Dialog, FocusRequest, Outcome, Search, State};

/// Draws one frame and folds this frame's input into `state`.
///
/// Order matters, because `edit`'s TUI is immediate mode and the first widget
/// to call `consume_shortcut` wins the key. The menubar goes first (it owns
/// F10 and closes its own menus on Esc), then the search panel (Esc closes
/// it), and only then does a still-unconsumed Esc mean "cancel the session".
pub fn draw(ctx: &mut Context, state: &mut State) {
    menubar(ctx, state);

    // Search panel toggle. The menu items below register the same shortcuts,
    // but these keep working while the menubar has never been opened.
    if ctx.consume_shortcut(kbmod::CTRL | vk::F) {
        state.open_search(Search::Find);
    } else if ctx.consume_shortcut(kbmod::CTRL | vk::R) {
        state.open_search(Search::Replace);
    }
    if state.search != Search::Hidden {
        search_panel(ctx, state);
    }

    // Global exits, checked before the textarea so it cannot swallow them.
    // Esc reaching this point means no menu and no search panel wanted it.
    // While a dialog is up it owns the keyboard, so these stand down and Esc
    // dismisses the dialog instead of the session.
    if state.dialog == Dialog::None {
        if ctx.consume_shortcut(kbmod::CTRL | vk::S) {
            state.outcome = Outcome::Accept;
        } else if ctx.consume_shortcut(vk::ESCAPE) || ctx.consume_shortcut(kbmod::CTRL | vk::Q) {
            state.request_cancel();
        }
    }

    let size = ctx.size();
    // Menubar and status bar take a row each; the search panel takes a field
    // row plus its action row, and Replace adds a second field row.
    let chrome = 2 + match state.search {
        Search::Hidden => 0,
        Search::Find => 2,
        Search::Replace => 3,
    };

    ctx.block_begin("editor");
    {
        ctx.textarea("textarea", state.buffer.clone());
        ctx.inherit_focus();
        // The search panel took the focus while it was up; claim it back on
        // the frame after it closes, or keystrokes land on a widget that is
        // no longer drawn and are silently lost.
        if state.take_focus(FocusRequest::Editor) {
            ctx.steal_focus();
        }
        ctx.attr_intrinsic_size(Size {
            width: 0,
            height: size.height - chrome,
        });
    }
    ctx.block_end();

    status_bar(ctx, state);

    if state.dialog == Dialog::ConfirmDiscard {
        confirm_discard(ctx, state);
    }
}

/// Asks before throwing away edits.
///
/// Only raised when the text actually differs from the prompt the session
/// started with — discarding an untouched buffer costs nothing, so that path
/// leaves without a question. Dismissing the dialog (Esc, or clicking away)
/// means "keep editing": the safe answer is the one that keeps the work.
fn confirm_discard(ctx: &mut Context, state: &mut State) {
    let mut discard = false;
    let mut keep = false;

    ctx.modal_begin("confirm-discard", "Discard changes?");
    ctx.attr_background_rgba(ctx.indexed(IndexedColor::Red));
    ctx.attr_foreground_rgba(ctx.indexed(IndexedColor::BrightWhite));
    {
        let contains_focus = ctx.contains_focus();

        ctx.label(
            "description",
            "Your edits will be lost and the prompt will keep its original text.",
        );
        ctx.attr_padding(Rect::three(1, 2, 1));

        ctx.table_begin("choices");
        ctx.inherit_focus();
        ctx.attr_padding(Rect::three(0, 2, 1));
        ctx.attr_position(Position::Center);
        ctx.table_set_cell_gap(Size {
            width: 2,
            height: 0,
        });
        {
            ctx.table_next_row();
            ctx.inherit_focus();

            // "Keep editing" comes first so it holds the initial focus: the
            // destructive button should not be one stray Enter away.
            if ctx.button(
                "keep",
                "Keep editing",
                ButtonStyle::default().accelerator('K'),
            ) {
                keep = true;
            }
            ctx.inherit_focus();
            if ctx.button(
                "discard",
                "Discard",
                ButtonStyle::default().accelerator('D'),
            ) {
                discard = true;
            }

            if contains_focus {
                if ctx.consume_shortcut(vk::D) {
                    discard = true;
                } else if ctx.consume_shortcut(vk::K) {
                    keep = true;
                }
            }
        }
        ctx.table_end();
    }
    // True when the modal was dismissed (Esc or a click outside).
    if ctx.modal_end() {
        keep = true;
    }

    if discard {
        state.outcome = Outcome::Cancel;
    } else if keep {
        state.dismiss_dialog();
    }
}

/// The top menu bar: what miniedit can do, and where the keyboard shortcuts
/// are discoverable.
///
/// F10 focuses it. On macOS `edit` disables the Alt accelerators (see
/// `Context::menubar_menu_begin`), so F10 and the mouse are the ways in.
fn menubar(ctx: &mut Context, state: &mut State) {
    ctx.menubar_begin();
    ctx.attr_background_rgba(ctx.indexed(IndexedColor::Blue));
    ctx.attr_foreground_rgba(ctx.indexed(IndexedColor::BrightWhite));
    {
        let contains_focus = ctx.contains_focus();

        if ctx.menubar_menu_begin("Prompt", 'P') {
            menu_prompt(ctx, state);
        }
        if !contains_focus && ctx.consume_shortcut(vk::F10) {
            ctx.steal_focus();
        }
        // With a menu open, Esc means "close the menu" and must not reach the
        // session-level cancel below. `edit`'s menubar does not claim Esc
        // itself, so consume it here and hand focus back to the text area.
        if contains_focus && ctx.consume_shortcut(vk::ESCAPE) {
            ctx.toss_focus_up();
        }
        if ctx.menubar_menu_begin("Edit", 'E') {
            menu_edit(ctx, state);
        }
        if ctx.menubar_menu_begin("View", 'V') {
            menu_view(ctx, state);
        }
    }
    ctx.menubar_end();
}

/// The two ways out. There is no Save: the text goes back to the prompt, not
/// to a file.
fn menu_prompt(ctx: &mut Context, state: &mut State) {
    if ctx.menubar_menu_button("Use this text", 'U', kbmod::CTRL | vk::S) {
        state.outcome = Outcome::Accept;
    }
    if ctx.menubar_menu_button("Discard changes (Esc)", 'D', vk::NULL) {
        state.request_cancel();
    }
    ctx.menubar_menu_end();
}

/// Editing commands. Every one of these is a `TextBuffer` operation; the menu
/// exists to make them discoverable and to register their shortcuts.
fn menu_edit(ctx: &mut Context, state: &mut State) {
    let mut tb = state.buffer.borrow_mut();

    if ctx.menubar_menu_button("Undo", 'U', kbmod::CTRL | vk::Z) {
        tb.undo();
        ctx.needs_rerender();
    }
    if ctx.menubar_menu_button("Redo", 'R', kbmod::CTRL | vk::Y) {
        tb.redo();
        ctx.needs_rerender();
    }
    if ctx.menubar_menu_button("Cut", 'T', kbmod::CTRL | vk::X) {
        tb.cut(ctx.clipboard_mut());
        ctx.needs_rerender();
    }
    if ctx.menubar_menu_button("Copy", 'C', kbmod::CTRL | vk::C) {
        tb.copy(ctx.clipboard_mut());
        ctx.needs_rerender();
    }
    if ctx.menubar_menu_button("Paste", 'P', kbmod::CTRL | vk::V) {
        tb.paste(ctx.clipboard_ref(), false);
        ctx.needs_rerender();
    }
    if ctx.menubar_menu_button("Select All", 'A', kbmod::CTRL | vk::A) {
        tb.select_all();
        ctx.needs_rerender();
    }
    // The search panel borrows the buffer itself, so release it first.
    drop(tb);
    if ctx.menubar_menu_button("Find", 'F', kbmod::CTRL | vk::F) {
        state.open_search(Search::Find);
    }
    if ctx.menubar_menu_button("Replace", 'L', kbmod::CTRL | vk::R) {
        state.open_search(Search::Replace);
    }
    ctx.menubar_menu_end();
}

/// Display toggles.
fn menu_view(ctx: &mut Context, state: &mut State) {
    let word_wrap = state.buffer.borrow().is_word_wrap_enabled();
    if ctx.menubar_menu_checkbox("Word Wrap", 'W', kbmod::ALT | vk::Z, word_wrap) {
        state.buffer.borrow_mut().set_word_wrap(!word_wrap);
        ctx.needs_rerender();
    }
    if ctx.menubar_menu_checkbox("Line Numbers", 'L', vk::NULL, state.line_numbers) {
        state.line_numbers = !state.line_numbers;
        state
            .buffer
            .borrow_mut()
            .set_margin_enabled(state.line_numbers);
        ctx.needs_rerender();
    }
    ctx.menubar_menu_end();
}

/// The one-line footer: cursor position, modified marker, the two exits, and
/// any pending error.
fn status_bar(ctx: &mut Context, state: &mut State) {
    ctx.table_begin("status");
    ctx.attr_background_rgba(ctx.indexed(IndexedColor::BrightBlack));
    ctx.attr_foreground_rgba(ctx.indexed(IndexedColor::BrightWhite));
    ctx.table_set_cell_gap(Size {
        width: 2,
        height: 0,
    });
    {
        ctx.table_next_row();

        let pos = state.buffer.borrow().cursor_logical_pos();
        let marker = if state.is_modified() { "*" } else { " " };
        ctx.label(
            "pos",
            &format!("{marker} Ln {}, Col {}", pos.y + 1, pos.x + 1),
        );

        // A prompt has no name worth showing; a file does, and it is the only
        // thing on screen that says which file is being written.
        if let Some(title) = state.title.clone() {
            ctx.label("title", &title);
        }

        if let Some(err) = state.error.clone() {
            ctx.label("error", &err);
        }

        ctx.label("menu-hint", "F10 menu");

        // "Use text" is prompt language: it hands the text back to the prompt.
        // In file mode the button writes to disk, so it says so.
        let accept = match state.mode {
            crate::miniedit::Mode::Prompt => "Use text (Ctrl+S)",
            crate::miniedit::Mode::File => "Save (Ctrl+S)",
        };
        if ctx.button("accept", accept, ButtonStyle::default()) {
            state.outcome = Outcome::Accept;
        }
        if ctx.button("cancel", "Discard (Esc)", ButtonStyle::default()) {
            state.request_cancel();
        }
    }
    ctx.table_end();
}

/// Find / find-and-replace strip between the menubar and the text area.
///
/// Search goes through `edit`'s ICU binding, which loads the library at
/// runtime. When it is unavailable the panel closes and says so in the status
/// bar rather than failing the session — the editor is still fully usable.
///
/// The `attr_intrinsic_size` calls after each field are load-bearing: without
/// them an `editline` collapses to zero width and the panel renders as a bare
/// label with nothing to type into.
fn search_panel(ctx: &mut Context, state: &mut State) {
    if edit::icu::init().is_err() {
        state.close_search();
        state.error = Some("search unavailable (ICU not found)".to_string());
        return;
    }

    let mut run_search = false;
    let mut replace_all = false;

    ctx.block_begin("search");
    ctx.attr_focus_well();
    ctx.attr_background_rgba(ctx.indexed(IndexedColor::White));
    ctx.attr_foreground_rgba(ctx.indexed(IndexedColor::Black));
    {
        // Whenever the panel is up, Esc closes it — `draw` only treats Esc as
        // "cancel the session" once nothing else has claimed it.
        if ctx.consume_shortcut(vk::ESCAPE) {
            state.close_search();
        }

        ctx.table_begin("fields");
        ctx.table_set_cell_gap(Size {
            width: 1,
            height: 0,
        });
        {
            ctx.table_next_row();
            ctx.label("find-label", "Find");
            ctx.editline("needle", &mut state.needle);
            ctx.attr_intrinsic_size(Size {
                width: COORD_TYPE_SAFE_MAX,
                height: 1,
            });
            // Focus the needle on the frame the panel opens, so the user can
            // just start typing.
            if state.take_focus(FocusRequest::Search) {
                ctx.steal_focus();
            }
            // Search on Enter rather than per keystroke: an incremental search
            // moves the selection under the user mid-word.
            if ctx.is_focused() && ctx.consume_shortcut(vk::RETURN) {
                run_search = true;
            }

            if state.search == Search::Replace {
                ctx.table_next_row();
                ctx.label("replace-label", "Replace");
                ctx.editline("replacement", &mut state.replacement);
                ctx.attr_intrinsic_size(Size {
                    width: COORD_TYPE_SAFE_MAX,
                    height: 1,
                });
            }
        }
        ctx.table_end();

        ctx.table_begin("actions");
        ctx.table_set_cell_gap(Size {
            width: 2,
            height: 0,
        });
        {
            ctx.table_next_row();
            let mut changed = ctx.checkbox("match-case", "Match case", &mut state.match_case);
            changed |= ctx.checkbox("whole-word", "Whole word", &mut state.whole_word);
            if changed {
                run_search = true;
            }
            if ctx.button("find-next", "Find Next", ButtonStyle::default()) {
                run_search = true;
            }
            if state.search == Search::Replace
                && ctx.button("replace-all", "Replace All", ButtonStyle::default())
            {
                replace_all = true;
            }
            if ctx.button("close", "Close", ButtonStyle::default()) {
                state.close_search();
            }
        }
        ctx.table_end();
    }
    ctx.block_end();

    if run_search || replace_all {
        let options = SearchOptions {
            match_case: state.match_case,
            whole_word: state.whole_word,
            use_regex: false,
        };
        let mut tb = state.buffer.borrow_mut();
        let outcome = if replace_all {
            tb.find_and_replace_all(&state.needle, options, state.replacement.as_bytes())
        } else {
            tb.find_and_select(&state.needle, options)
        };
        drop(tb);
        state.error = if outcome.is_err() {
            Some(format!("no match for {:?}", state.needle))
        } else {
            None
        };
        ctx.needs_rerender();
    }
}
