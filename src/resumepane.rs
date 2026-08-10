// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! `/resume`: the state behind the interactive session picker.
//!
//! Mirrors [`crate::kvpane`]: this module owns rows, the search query,
//! selection and key handling; `tui::draw_resume` only paints what
//! [`ResumePane::rows`] returns. Nothing here touches the filesystem — the
//! pane *asks*, via [`Outcome`], and `ui.rs` performs the I/O. That is what
//! keeps the whole pane unit-testable without a terminal or a store.

use std::collections::HashMap;

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::session::SessionEntry;

/// One render-ready session, as three lines.
#[derive(Debug, Clone)]
pub struct Row {
    /// Session name, plus ` [tag]` when tagged.
    pub label: String,
    /// Dim second line: age and file size.
    pub detail: String,
    /// Dim trailing lines: the last prompt, or the preview when expanded.
    pub extra: Vec<String>,
    /// The cursor is on this row.
    pub selected: bool,
    /// The session this row acts on.
    ///
    /// A durable handle, deliberately not a row index: the listing is re-taken
    /// after every rename and delete, so a position can name a different
    /// session than the one the user was looking at. Same lesson as
    /// [`crate::kvpane::Row::fingerprint`].
    pub id: String,
}

/// What a key press asks the caller to do.
#[derive(Debug)]
pub enum Outcome {
    /// Stay open; the pane absorbed the key.
    Stay,
    /// Close the picker without resuming anything.
    Close,
    /// Resume this session.
    Resume(String),
    /// Rename the session (first field) to a new name (second).
    Rename(String, String),
    /// Delete this session; the pane already took the confirmation.
    Delete(String),
    /// Fill in this session's preview text via [`ResumePane::set_preview`].
    LoadPreview(String),
}

/// Interactive state over the saved-session listing.
#[derive(Debug)]
pub struct ResumePane {
    /// The listing, most-recent first, uncapped.
    entries: Vec<SessionEntry>,
    /// Search text; empty shows everything.
    query: String,
    /// Indices into `entries` matching `query`, recomputed on every edit.
    visible: Vec<usize>,
    /// Cursor over `visible`, not `entries`.
    cursor: usize,
    /// Rendered preview text by session id, so a re-toggle is free.
    previews: HashMap<String, String>,
    /// The selected row is showing its preview.
    preview_open: bool,
    /// A delete press is awaiting confirmation.
    pending_delete: bool,
    /// When `Some`, the search box is a rename buffer for the selected session.
    rename: Option<String>,
    /// Wall clock the ages render against.
    now: u64,
}

impl ResumePane {
    /// Builds a pane over a listing. `now` is passed in rather than read, so
    /// the ages a test asserts on are the ages the test chose.
    #[must_use]
    pub fn new(entries: Vec<SessionEntry>, now: u64) -> Self {
        let visible = (0..entries.len()).collect();
        Self {
            entries,
            query: String::new(),
            visible,
            cursor: 0,
            previews: HashMap::new(),
            preview_open: false,
            pending_delete: false,
            rename: None,
            now,
        }
    }

    /// The search text currently typed.
    #[must_use]
    pub fn query(&self) -> &str {
        &self.query
    }

    /// The rename text being edited, or `None` when the box is the search box.
    #[must_use]
    pub fn rename_buffer(&self) -> Option<&str> {
        self.rename.as_deref()
    }

    /// A delete press is armed and awaiting its confirmation.
    #[must_use]
    pub fn pending_delete(&self) -> bool {
        self.pending_delete
    }

    /// No session matches the query.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.visible.is_empty()
    }

    /// The selected entry, or `None` when the query matches nothing.
    fn selected(&self) -> Option<&SessionEntry> {
        self.entries.get(*self.visible.get(self.cursor)?)
    }

    /// Recomputes `visible` from `query` and pulls the cursor back into range.
    ///
    /// The clamp is the point: without it a narrowing query leaves the cursor
    /// past the end and the selection names a row that is not on screen.
    fn refilter(&mut self) {
        let needle = self.query.to_lowercase();
        self.visible = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                needle.is_empty()
                    || [&e.id, &e.title, &e.tag, &e.last_prompt]
                        .iter()
                        .any(|f| f.to_lowercase().contains(&needle))
            })
            .map(|(i, _)| i)
            .collect();
        self.cursor = self.cursor.min(self.visible.len().saturating_sub(1));
        self.preview_open = false;
        self.pending_delete = false;
    }

    /// Supplies the preview text the pane asked for with
    /// [`Outcome::LoadPreview`]. Cached by id, so re-opening the same row
    /// never asks again.
    pub fn set_preview(&mut self, id: &str, text: String) {
        self.previews.insert(id.to_owned(), text);
    }

    /// Header line: cursor position within the visible set.
    #[must_use]
    pub fn header(&self) -> String {
        let n = if self.visible.is_empty() {
            0
        } else {
            self.cursor + 1
        };
        format!("Resume session ({n} of {})", self.visible.len())
    }

    /// Footer hint line.
    #[must_use]
    pub fn footer(&self) -> String {
        if self.rename.is_some() {
            return "Enter to rename · Esc to go back".to_owned();
        }
        if self.pending_delete {
            return "Ctrl+X again to delete · any other key cancels".to_owned();
        }
        "Ctrl+R to rename · Ctrl+X to delete · Space to preview · Type to search · Esc to cancel"
            .to_owned()
    }

    /// Render-ready rows for the visible sessions.
    #[must_use]
    pub fn rows(&self) -> Vec<Row> {
        self.visible
            .iter()
            .enumerate()
            .filter_map(|(i, &e)| {
                let entry = self.entries.get(e)?;
                let selected = i == self.cursor;
                let label = if entry.tag.is_empty() {
                    entry.id.clone()
                } else {
                    format!("{} [{}]", entry.id, entry.tag)
                };
                // `last_used` is zero on a pre-metadata or unreadable file;
                // creation time is the only age it can offer.
                let when = if entry.last_used == 0 {
                    entry.created_at
                } else {
                    entry.last_used
                };
                let detail = format!(
                    "{} · {}",
                    crate::session::format_age(when, self.now),
                    crate::kvpane::human_bytes(entry.file_size)
                );
                let mut extra = Vec::new();
                if selected && self.preview_open {
                    match self.previews.get(&entry.id) {
                        Some(text) => extra.extend(text.lines().map(str::to_owned)),
                        None => extra.push("loading preview…".to_owned()),
                    }
                } else if !entry.last_prompt.is_empty() {
                    extra.push(format!("last: {}", entry.last_prompt));
                }
                Some(Row {
                    label,
                    detail,
                    extra,
                    selected,
                    id: entry.id.clone(),
                })
            })
            .collect()
    }

    /// Handles one key press.
    ///
    /// Control keys first: `Ctrl+R`/`Ctrl+X` must not be mistaken for the
    /// characters `r` and `x` going into the search box.
    pub fn handle_key(&mut self, key: KeyEvent) -> Outcome {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('r') if ctrl => {
                // Nothing selected means nothing to rename.
                if let Some(e) = self.selected() {
                    self.rename = Some(e.id.clone());
                    self.preview_open = false;
                    self.pending_delete = false;
                }
                Outcome::Stay
            }
            // Delete is unreachable behind a rename box: the two would fight
            // over Enter, and confirming a delete you cannot see is worse.
            KeyCode::Char('x') if ctrl && self.rename.is_none() => {
                if !self.pending_delete {
                    self.pending_delete = self.selected().is_some();
                    return Outcome::Stay;
                }
                self.pending_delete = false;
                match self.selected() {
                    Some(e) => Outcome::Delete(e.id.clone()),
                    None => Outcome::Stay,
                }
            }
            // Every other control chord is swallowed rather than falling through
            // to the text arms below. Without this, `Ctrl+X` behind a rename box
            // types a literal `x` into the name.
            KeyCode::Char(_) if ctrl => Outcome::Stay,
            KeyCode::Up => {
                self.cursor = self.cursor.saturating_sub(1);
                self.preview_open = false;
                self.pending_delete = false;
                Outcome::Stay
            }
            KeyCode::Down => {
                if self.cursor + 1 < self.visible.len() {
                    self.cursor += 1;
                }
                self.preview_open = false;
                self.pending_delete = false;
                Outcome::Stay
            }
            KeyCode::Enter => {
                self.pending_delete = false;
                let Some(id) = self.selected().map(|e| e.id.clone()) else {
                    return Outcome::Stay;
                };
                match self.rename.clone() {
                    // An empty name is refused here rather than round-tripping
                    // through the store just to be rejected; the box stays open.
                    Some(new) if new.trim().is_empty() => Outcome::Stay,
                    Some(new) => {
                        self.rename = None;
                        Outcome::Rename(id, new)
                    }
                    None => Outcome::Resume(id),
                }
            }
            // Esc backs out one level: rename box first, then the picker.
            KeyCode::Esc if self.rename.is_some() => {
                self.rename = None;
                Outcome::Stay
            }
            KeyCode::Esc => Outcome::Close,
            KeyCode::Backspace => {
                self.pending_delete = false;
                if let Some(buf) = self.rename.as_mut() {
                    buf.pop();
                } else {
                    self.query.pop();
                    self.refilter();
                }
                Outcome::Stay
            }
            KeyCode::Char(' ') if self.query.is_empty() && self.rename.is_none() => {
                self.pending_delete = false;
                self.preview_open = !self.preview_open;
                match self.selected().map(|e| e.id.clone()) {
                    Some(id) if self.preview_open && !self.previews.contains_key(&id) => {
                        Outcome::LoadPreview(id)
                    }
                    _ => Outcome::Stay,
                }
            }
            KeyCode::Char(c) => {
                self.pending_delete = false;
                if let Some(buf) = self.rename.as_mut() {
                    buf.push(c);
                } else {
                    self.query.push(c);
                    self.refilter();
                }
                Outcome::Stay
            }
            _ => Outcome::Stay,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent};

    fn entry(
        id: &str,
        title: &str,
        tag: &str,
        last: &str,
        used: u64,
    ) -> crate::session::SessionEntry {
        crate::session::SessionEntry {
            id: id.to_owned(),
            title: title.to_owned(),
            created_at: used,
            last_used: used,
            file_size: 1024,
            tag: tag.to_owned(),
            last_prompt: last.to_owned(),
            payload_bytes: 0,
            path: std::path::PathBuf::from(format!("/tmp/{id}.kv")),
        }
    }

    /// Three sessions, most-recent first, as `SessionStore::list` returns them.
    fn pane() -> ResumePane {
        ResumePane::new(
            vec![
                entry(
                    "kv-cache-design",
                    "Design KV cache",
                    "wip",
                    "make it scroll",
                    900,
                ),
                entry(
                    "guide-update",
                    "Update user guide",
                    "",
                    "rewrite intro",
                    600,
                ),
                entry("session-names", "Name a session", "done", "mint an id", 300),
            ],
            1000,
        )
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn typed(pane: &mut ResumePane, text: &str) {
        for c in text.chars() {
            pane.handle_key(key(KeyCode::Char(c)));
        }
    }

    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    fn ids(pane: &ResumePane) -> Vec<String> {
        pane.rows().into_iter().map(|r| r.id).collect()
    }

    #[test]
    fn a_fresh_pane_shows_every_session_in_listing_order() {
        let p = pane();
        assert_eq!(
            ids(&p),
            ["kv-cache-design", "guide-update", "session-names"]
        );
        assert!(p.rows()[0].selected, "cursor starts on the most recent");
        assert_eq!(p.header(), "Resume session (1 of 3)");
    }

    #[test]
    fn a_row_shows_the_name_tag_age_size_and_last_prompt() {
        let p = pane();
        let row = &p.rows()[0];
        assert_eq!(row.label, "kv-cache-design [wip]");
        assert!(
            row.detail.contains("1024") || row.detail.contains("1.0 KB"),
            "{}",
            row.detail
        );
        assert!(
            row.extra.iter().any(|l| l.contains("make it scroll")),
            "{:?}",
            row.extra
        );
        // No tag means no brackets rather than an empty pair.
        assert_eq!(p.rows()[1].label, "guide-update");
    }

    #[test]
    fn typing_filters_on_id_title_tag_and_last_prompt_case_insensitively() {
        let mut p = pane();
        typed(&mut p, "KV");
        assert_eq!(ids(&p), ["kv-cache-design"], "matches the id");
        p = pane();
        typed(&mut p, "user guide");
        assert_eq!(ids(&p), ["guide-update"], "matches the title");
        p = pane();
        typed(&mut p, "done");
        assert_eq!(ids(&p), ["session-names"], "matches the tag");
        p = pane();
        typed(&mut p, "intro");
        assert_eq!(ids(&p), ["guide-update"], "matches the last prompt");
    }

    #[test]
    fn backspace_widens_the_filter_again() {
        let mut p = pane();
        typed(&mut p, "kvx");
        assert!(p.is_empty(), "no session matches");
        p.handle_key(key(KeyCode::Backspace));
        assert_eq!(ids(&p), ["kv-cache-design"]);
        assert_eq!(p.query(), "kv");
    }

    #[test]
    fn the_cursor_moves_within_the_visible_rows_and_stops_at_the_ends() {
        let mut p = pane();
        p.handle_key(key(KeyCode::Up));
        assert!(p.rows()[0].selected, "already at the top, stays");
        p.handle_key(key(KeyCode::Down));
        p.handle_key(key(KeyCode::Down));
        p.handle_key(key(KeyCode::Down));
        assert!(p.rows()[2].selected, "clamped at the last row");
        assert_eq!(p.header(), "Resume session (3 of 3)");
    }

    /// The bug this guards: a query edit that shrinks the visible set must pull
    /// the cursor back in, or the selection names a row that is not shown.
    #[test]
    fn a_narrowing_filter_clamps_the_cursor_back_into_range() {
        let mut p = pane();
        p.handle_key(key(KeyCode::Down));
        p.handle_key(key(KeyCode::Down));
        typed(&mut p, "kv");
        let rows = p.rows();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].selected, "cursor followed the shrink");
        assert_eq!(p.header(), "Resume session (1 of 1)");
    }

    #[test]
    fn enter_resumes_the_selected_session_and_esc_closes() {
        let mut p = pane();
        p.handle_key(key(KeyCode::Down));
        assert!(matches!(p.handle_key(key(KeyCode::Enter)),
                         Outcome::Resume(id) if id == "guide-update"));
        assert!(matches!(p.handle_key(key(KeyCode::Esc)), Outcome::Close));
    }

    #[test]
    fn an_empty_result_set_has_nothing_to_resume() {
        let mut p = pane();
        typed(&mut p, "nothing-matches-this");
        assert!(p.is_empty());
        assert!(p.rows().is_empty());
        assert!(matches!(p.handle_key(key(KeyCode::Enter)), Outcome::Stay));
        assert_eq!(p.header(), "Resume session (0 of 0)");
    }

    #[test]
    fn an_unknown_key_is_absorbed_without_disturbing_the_pane() {
        let mut p = pane();
        assert!(matches!(p.handle_key(key(KeyCode::F(5))), Outcome::Stay));
        assert_eq!(
            ids(&p),
            ["kv-cache-design", "guide-update", "session-names"]
        );
    }

    #[test]
    fn space_on_an_empty_query_asks_the_caller_for_a_preview() {
        let mut p = pane();
        let out = p.handle_key(key(KeyCode::Char(' ')));
        assert!(matches!(out, Outcome::LoadPreview(id) if id == "kv-cache-design"));
        assert_eq!(p.query(), "", "Space did not land in the search box");
        // Until the caller answers, the row says so rather than showing stale text.
        assert!(p.rows()[0].extra.iter().any(|l| l.contains("loading")));

        p.set_preview(
            "kv-cache-design",
            "user: make it scroll\nplank: done".to_owned(),
        );
        let extra = p.rows()[0].extra.clone();
        assert_eq!(extra, ["user: make it scroll", "plank: done"]);
    }

    #[test]
    fn a_second_space_closes_the_preview_and_does_not_reload_it() {
        let mut p = pane();
        p.handle_key(key(KeyCode::Char(' ')));
        p.set_preview("kv-cache-design", "cached".to_owned());
        assert!(matches!(
            p.handle_key(key(KeyCode::Char(' '))),
            Outcome::Stay
        ));
        assert!(
            p.rows()[0].extra.iter().all(|l| l != "cached"),
            "preview closed"
        );
        // Reopening is free: the text is already in hand, so no reload is asked for.
        assert!(matches!(
            p.handle_key(key(KeyCode::Char(' '))),
            Outcome::Stay
        ));
        assert!(p.rows()[0].extra.iter().any(|l| l == "cached"));
    }

    #[test]
    fn space_types_a_space_once_the_query_is_not_empty() {
        let mut p = pane();
        typed(&mut p, "user");
        let out = p.handle_key(key(KeyCode::Char(' ')));
        assert!(matches!(out, Outcome::Stay));
        assert_eq!(p.query(), "user ");
        typed(&mut p, "guide");
        assert_eq!(ids(&p), ["guide-update"]);
    }

    #[test]
    fn moving_the_cursor_closes_the_preview() {
        let mut p = pane();
        p.handle_key(key(KeyCode::Char(' ')));
        p.set_preview("kv-cache-design", "cached".to_owned());
        p.handle_key(key(KeyCode::Down));
        assert!(p.rows()[0].extra.iter().all(|l| l != "cached"));
    }

    #[test]
    fn ctrl_r_opens_a_rename_prefilled_with_the_current_name() {
        let mut p = pane();
        p.handle_key(key(KeyCode::Down));
        assert!(matches!(p.handle_key(ctrl('r')), Outcome::Stay));
        assert_eq!(p.rename_buffer(), Some("guide-update"));
        // Editing goes to the rename buffer, not the search box.
        for _ in 0..6 {
            p.handle_key(key(KeyCode::Backspace));
        }
        typed(&mut p, "docs");
        assert_eq!(p.rename_buffer(), Some("guide-docs"));
        assert_eq!(p.query(), "", "the filter is untouched while renaming");
        assert!(matches!(p.handle_key(key(KeyCode::Enter)),
                         Outcome::Rename(id, new) if id == "guide-update" && new == "guide-docs"));
    }

    #[test]
    fn esc_leaves_rename_mode_without_closing_the_picker() {
        let mut p = pane();
        p.handle_key(ctrl('r'));
        typed(&mut p, "-x");
        assert!(matches!(p.handle_key(key(KeyCode::Esc)), Outcome::Stay));
        assert_eq!(p.rename_buffer(), None, "back to search");
        // A second Esc, now in search mode, closes the picker.
        assert!(matches!(p.handle_key(key(KeyCode::Esc)), Outcome::Close));
    }

    #[test]
    fn renaming_to_the_empty_string_is_refused_by_the_pane() {
        let mut p = pane();
        p.handle_key(ctrl('r'));
        for _ in 0..40 {
            p.handle_key(key(KeyCode::Backspace));
        }
        assert_eq!(p.rename_buffer(), Some(""));
        assert!(
            matches!(p.handle_key(key(KeyCode::Enter)), Outcome::Stay),
            "an empty name never reaches the store"
        );
        assert_eq!(p.rename_buffer(), Some(""), "still renaming");
    }

    #[test]
    fn ctrl_r_with_nothing_selected_does_nothing() {
        let mut p = pane();
        typed(&mut p, "no-such-session");
        assert!(matches!(p.handle_key(ctrl('r')), Outcome::Stay));
        assert_eq!(p.rename_buffer(), None);
    }

    #[test]
    fn ctrl_x_twice_deletes_the_selected_session() {
        let mut p = pane();
        p.handle_key(key(KeyCode::Down));
        assert!(
            matches!(p.handle_key(ctrl('x')), Outcome::Stay),
            "armed, not fired"
        );
        assert!(p.pending_delete());
        assert!(p.footer().contains("again"), "{}", p.footer());
        assert!(matches!(p.handle_key(ctrl('x')),
                         Outcome::Delete(id) if id == "guide-update"));
    }

    #[test]
    fn any_other_key_disarms_a_pending_delete() {
        let mut p = pane();
        p.handle_key(ctrl('x'));
        assert!(p.pending_delete());
        p.handle_key(key(KeyCode::Down));
        assert!(!p.pending_delete(), "moving away cancels");
        // And a second Ctrl+X after the disarm only re-arms, it does not delete.
        assert!(matches!(p.handle_key(ctrl('x')), Outcome::Stay));
    }

    #[test]
    fn ctrl_x_does_nothing_while_renaming() {
        let mut p = pane();
        p.handle_key(ctrl('r'));
        assert!(matches!(p.handle_key(ctrl('x')), Outcome::Stay));
        assert!(!p.pending_delete(), "no delete armed behind a rename box");
        assert_eq!(
            p.rename_buffer(),
            Some("kv-cache-design"),
            "and the chord did not type a literal x into the name"
        );
    }
}
