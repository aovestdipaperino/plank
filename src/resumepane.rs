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

use ratatui::crossterm::event::{KeyCode, KeyEvent};

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
    pub fn handle_key(&mut self, key: KeyEvent) -> Outcome {
        match key.code {
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
            KeyCode::Enter => match self.selected() {
                Some(e) => Outcome::Resume(e.id.clone()),
                None => Outcome::Stay,
            },
            KeyCode::Esc => Outcome::Close,
            KeyCode::Backspace => {
                self.query.pop();
                self.refilter();
                Outcome::Stay
            }
            KeyCode::Char(c) => {
                self.query.push(c);
                self.refilter();
                Outcome::Stay
            }
            _ => Outcome::Stay,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

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
}
