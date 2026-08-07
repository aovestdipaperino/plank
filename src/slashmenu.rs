// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The `/` command menu for the Ratatui TUI prompt.
//!
//! Sibling of [`crate::complete`], which does the same job for `@` file
//! completion. The two never overlap: `@` needs whitespace (or the start of the
//! input) before it and this one only opens on a `/` at byte 0, so at most one
//! is up at a time.
//!
//! Everything here is synchronous. The candidate set is a fixed table
//! ([`crate::config::SLASH_COMMANDS`]) plus the skills and templates already
//! loaded at startup, so there is no index to build and no worker to wait on —
//! the menu is fully populated on the first keypress.

use crate::editor::LineBuffer;
use nucleo_matcher::{Config, Matcher, Utf32Str};
use ratatui::crossterm::event::{KeyCode, KeyEvent};

/// Where a menu entry came from, which is also how it is labelled on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// A command dispatch implements itself.
    Builtin,
    /// A `SKILL.md` discovered under `.plank/skills` (or the user directory).
    Skill,
    /// A prompt template discovered under `.plank/templates`.
    Template,
}

impl Source {
    /// The right-hand tag shown after the description; empty for built-ins,
    /// which are the unremarkable case.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::Builtin => "",
            Self::Skill => "skill",
            Self::Template => "template",
        }
    }
}

/// One row of the menu: what to type, what it takes, and what it does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Command including the leading slash, e.g. `/compact`.
    pub name: String,
    /// Argument hint shown after the name; empty when it takes none.
    pub args: String,
    /// One-line help shown beside the name.
    pub desc: String,
    /// Which of the three registries contributed this entry.
    pub source: Source,
}

impl Entry {
    /// `"/name <args>"`, the left column as it is drawn.
    #[must_use]
    pub fn label(&self) -> String {
        if self.args.is_empty() {
            self.name.clone()
        } else {
            format!("{} {}", self.name, self.args)
        }
    }
}

/// Builds the full candidate list: built-ins first, then skills, then
/// templates.
///
/// Skills and templates keep the resolution order `Agent::slash_message` uses
/// (skills win over templates, and neither shadows a built-in), so a name that
/// would never actually reach its template is not offered twice: an entry whose
/// name is already taken by an earlier source is dropped.
#[must_use]
pub fn catalog(
    skills: &[crate::skills::Skill],
    templates: &[crate::templates::Template],
) -> Vec<Entry> {
    let mut out: Vec<Entry> = crate::config::SLASH_COMMANDS
        .iter()
        .map(|c| Entry {
            name: c.name.to_string(),
            args: c.args.to_string(),
            desc: c.desc.to_string(),
            source: Source::Builtin,
        })
        .collect();
    let push = |name: &str, args: &str, desc: &str, source: Source, out: &mut Vec<Entry>| {
        let name = format!("/{name}");
        if out.iter().any(|e| e.name == name) {
            return;
        }
        out.push(Entry {
            name,
            args: args.to_string(),
            desc: desc.to_string(),
            source,
        });
    };
    for s in skills {
        push(
            &s.name,
            &s.argument_hint,
            &s.description,
            Source::Skill,
            &mut out,
        );
    }
    for t in templates {
        push(
            &t.name,
            &t.argument_hint,
            &t.description,
            Source::Template,
            &mut out,
        );
    }
    out
}

/// A `/`-prefixed command token being typed at the start of the input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlashToken {
    /// Text typed after the `/`.
    pub query: String,
}

/// Finds the command token when the text left of the cursor is one.
///
/// A slash command is only ever the *whole* input line, so unlike `@` this
/// anchors at byte 0 rather than at any word boundary: `see /help` is prose and
/// must not raise a menu. Whitespace ends the token — once arguments are being
/// typed the command is settled and the menu is out of the way.
#[must_use]
pub fn detect_slash_token(left: &str) -> Option<SlashToken> {
    let rest = left.strip_prefix('/')?;
    if rest.contains(char::is_whitespace) {
        return None;
    }
    Some(SlashToken {
        query: rest.to_string(),
    })
}

/// Ranks `entries` against `query`, best first.
///
/// Scored on the name without its slash, so typing `/comp` scores against
/// `compact`. Ties break toward the shorter name and then alphabetically, so
/// the order never depends on the catalog's internal ordering — except for the
/// empty query, which keeps the catalog order deliberately (built-ins first, in
/// the curated order they are declared in).
#[must_use]
pub fn rank(query: &str, entries: &[Entry]) -> Vec<usize> {
    if query.is_empty() {
        return (0..entries.len()).collect();
    }
    let mut matcher = Matcher::new(Config::DEFAULT);
    let needle = query.to_lowercase();
    let mut scored: Vec<(u16, usize)> = entries
        .iter()
        .enumerate()
        .filter_map(|(i, e)| {
            let text = e.name.strip_prefix('/').unwrap_or(&e.name);
            let mut hay = Vec::new();
            let mut ndl = Vec::new();
            let score = matcher.fuzzy_match(
                Utf32Str::new(text, &mut hay),
                Utf32Str::new(&needle, &mut ndl),
            )?;
            Some((score, i))
        })
        .collect();
    scored.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| entries[a.1].name.len().cmp(&entries[b.1].name.len()))
            .then_with(|| entries[a.1].name.cmp(&entries[b.1].name))
    });
    scored.into_iter().map(|(_, i)| i).collect()
}

/// What the caller must do after offering a key to the menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuAction {
    /// The menu ignored the key; handle it normally.
    Passthrough,
    /// The menu handled the key; skip the normal binding.
    Consumed,
    /// The menu handled the key and should now close.
    Dismissed,
}

/// The open `/` menu: a filtered, ranked view over a catalog.
#[derive(Debug, Clone)]
pub struct SlashMenu {
    entries: Vec<Entry>,
    /// Indices into `entries`, ranked best first — the rows on screen.
    rows: Vec<usize>,
    selected: usize,
    /// The text typed after the `/`, kept so Enter can tell "finish this name"
    /// from "run the name I already finished".
    query: String,
}

impl SlashMenu {
    /// Opens a menu over `entries`, filtered by `query`.
    #[must_use]
    pub fn new(entries: Vec<Entry>, query: &str) -> Self {
        let rows = rank(query, &entries);
        Self {
            entries,
            rows,
            selected: 0,
            query: query.to_string(),
        }
    }

    /// Re-filters against a new query, keeping the highlighted *entry* when it
    /// survives the new filter.
    ///
    /// Following the entry rather than the row index matters while typing: the
    /// list shrinks under the cursor, and resetting to row 0 on every keystroke
    /// would make a deliberate Down immediately undone by the next character.
    pub fn retarget(&mut self, query: &str) {
        let keep = self.rows.get(self.selected).copied();
        self.rows = rank(query, &self.entries);
        self.selected = keep
            .and_then(|e| self.rows.iter().position(|&r| r == e))
            .unwrap_or(0);
        query.clone_into(&mut self.query);
    }

    /// True when what has been typed is already a runnable command, so Enter
    /// means "run it" rather than "complete the highlighted suggestion".
    ///
    /// This is deliberately wider than "the highlight equals the query". The
    /// menu ranks fuzzily, so a name it does not carry at all can still leave
    /// some unrelated entry highlighted — `mat` is a subsequence of `compact`.
    /// Completing there would silently swap the command the user typed for a
    /// different one on a single Enter. Asking dispatch instead means anything
    /// it recognizes runs as typed, including the commands kept out of the menu
    /// on purpose (the arcade easter eggs, `/matrix`).
    fn query_is_runnable(&self) -> bool {
        crate::config::slash_command_known(&format!("/{}", self.query))
    }

    /// The entries currently displayed, best match first.
    #[must_use]
    pub fn rows(&self) -> Vec<&Entry> {
        self.rows
            .iter()
            .filter_map(|&i| self.entries.get(i))
            .collect()
    }

    /// True when nothing matches — the caller closes the menu rather than
    /// showing an empty box.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Index of the highlighted row.
    #[must_use]
    pub fn selected(&self) -> usize {
        self.selected
    }

    /// The highlighted entry, if any.
    #[must_use]
    pub fn selected_entry(&self) -> Option<&Entry> {
        self.entries.get(*self.rows.get(self.selected)?)
    }

    /// Handles one key while the menu is open.
    ///
    /// Up/Down move the highlight and Esc dismisses. *Both* Tab and Enter
    /// accept — unlike the `@` popup, where Tab inserts a common prefix. The
    /// command list is small and closed, so a partial completion would only ever
    /// need a second key to finish; accepting outright is the shorter path and
    /// leaves Tab meaning "take the highlighted one" in both menus.
    ///
    /// Accepting replaces the whole command token with `"/name "` — with the
    /// trailing space, so a command that takes arguments is ready for them and
    /// one that does not still submits (dispatch trims the line).
    ///
    /// The one exception is Enter on a line that *already parses as a command*:
    /// it passes through to the caller's submit binding, so `/help` + Enter runs
    /// the command rather than demanding a second Enter to get past the menu,
    /// and a fuzzy highlight can never displace something the user finished
    /// typing (see [`SlashMenu::query_is_runnable`]). Tab there still completes,
    /// adding the trailing space and nothing else.
    pub fn handle_key(&mut self, key: KeyEvent, buf: &mut LineBuffer) -> MenuAction {
        match key.code {
            KeyCode::Esc => MenuAction::Dismissed,
            KeyCode::Up => {
                self.selected = self.selected.saturating_sub(1);
                MenuAction::Consumed
            }
            KeyCode::Down => {
                if self.selected + 1 < self.rows.len() {
                    self.selected += 1;
                }
                MenuAction::Consumed
            }
            KeyCode::Enter if self.query_is_runnable() => MenuAction::Passthrough,
            KeyCode::Tab | KeyCode::Enter => {
                let Some(entry) = self.selected_entry() else {
                    return MenuAction::Dismissed;
                };
                let end = buf.cursor();
                buf.replace_range(0, end, format!("{} ", entry.name));
                MenuAction::Dismissed
            }
            _ => MenuAction::Passthrough,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::KeyModifiers;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn entries() -> Vec<Entry> {
        ["compact", "context", "config", "clear"]
            .iter()
            .map(|n| Entry {
                name: format!("/{n}"),
                args: String::new(),
                desc: format!("does {n}"),
                source: Source::Builtin,
            })
            .collect()
    }

    #[test]
    fn a_bare_slash_opens_the_menu_with_an_empty_query() {
        let t = detect_slash_token("/").expect("token");
        assert_eq!(t.query, "");
    }

    #[test]
    fn a_slash_mid_line_is_prose_not_a_command() {
        assert!(detect_slash_token("see /help").is_none());
        assert!(detect_slash_token("a/b").is_none());
    }

    #[test]
    fn whitespace_ends_the_token_so_arguments_close_the_menu() {
        assert_eq!(detect_slash_token("/comp").expect("token").query, "comp");
        assert!(detect_slash_token("/compact now").is_none());
        assert!(detect_slash_token("/compact ").is_none());
    }

    #[test]
    fn an_empty_query_keeps_the_catalog_order() {
        let e = entries();
        assert_eq!(rank("", &e), vec![0, 1, 2, 3]);
    }

    #[test]
    fn ranking_puts_the_closest_command_first() {
        let e = entries();
        let ranked = rank("conf", &e);
        assert_eq!(e[ranked[0]].name, "/config");
    }

    #[test]
    fn a_query_matching_nothing_ranks_empty() {
        assert!(rank("zzzz", &entries()).is_empty());
        assert!(SlashMenu::new(entries(), "zzzz").is_empty());
    }

    #[test]
    fn accepting_replaces_the_token_and_leaves_a_trailing_space() {
        let mut menu = SlashMenu::new(entries(), "conf");
        let mut buf = LineBuffer::new();
        buf.insert("/conf");
        assert_eq!(
            menu.handle_key(key(KeyCode::Enter), &mut buf),
            MenuAction::Dismissed
        );
        assert_eq!(buf.text(), "/config ");
        assert_eq!(buf.cursor(), buf.text().len());
    }

    #[test]
    fn tab_accepts_exactly_like_enter() {
        let mut by_tab = SlashMenu::new(entries(), "comp");
        let mut by_enter = SlashMenu::new(entries(), "comp");
        let (mut a, mut b) = (LineBuffer::new(), LineBuffer::new());
        a.insert("/comp");
        b.insert("/comp");
        by_tab.handle_key(key(KeyCode::Tab), &mut a);
        by_enter.handle_key(key(KeyCode::Enter), &mut b);
        assert_eq!(a.text(), b.text());
        assert_eq!(a.text(), "/compact ");
    }

    #[test]
    fn up_and_down_move_the_highlight_without_running_past_the_ends() {
        let mut menu = SlashMenu::new(entries(), "");
        let mut buf = LineBuffer::new();
        assert_eq!(
            menu.handle_key(key(KeyCode::Up), &mut buf),
            MenuAction::Consumed
        );
        assert_eq!(menu.selected(), 0, "Up at the top stays put");
        for _ in 0..10 {
            menu.handle_key(key(KeyCode::Down), &mut buf);
        }
        assert_eq!(menu.selected(), 3, "Down stops at the last row");
    }

    #[test]
    fn retargeting_follows_the_highlighted_entry_rather_than_the_row() {
        let mut menu = SlashMenu::new(entries(), "");
        let mut buf = LineBuffer::new();
        menu.handle_key(key(KeyCode::Down), &mut buf);
        menu.handle_key(key(KeyCode::Down), &mut buf);
        let picked = menu.selected_entry().expect("entry").name.clone();
        assert_eq!(picked, "/config");
        // `/config` survives the narrower query, so the highlight stays on it
        // even though its row index changed.
        menu.retarget("con");
        assert_eq!(menu.selected_entry().expect("entry").name, picked);
    }

    #[test]
    fn retargeting_falls_back_to_the_top_when_the_selection_is_filtered_out() {
        let mut menu = SlashMenu::new(entries(), "");
        let mut buf = LineBuffer::new();
        menu.handle_key(key(KeyCode::Down), &mut buf);
        menu.retarget("clear");
        assert_eq!(menu.selected(), 0);
        assert_eq!(menu.selected_entry().expect("entry").name, "/clear");
    }

    #[test]
    fn unrelated_keys_pass_through_so_typing_keeps_working() {
        let mut menu = SlashMenu::new(entries(), "");
        let mut buf = LineBuffer::new();
        assert_eq!(
            menu.handle_key(key(KeyCode::Char('x')), &mut buf),
            MenuAction::Passthrough
        );
    }

    #[test]
    fn skills_and_templates_join_the_catalog_without_shadowing_builtins() {
        let skill = crate::skills::Skill {
            name: "review".into(),
            description: "review the diff".into(),
            argument_hint: "[path]".into(),
            body: String::new(),
            dir: std::path::PathBuf::new(),
        };
        // A skill named after a built-in must not appear twice: dispatch would
        // never route to it anyway.
        let shadow = crate::skills::Skill {
            name: "help".into(),
            description: "not reachable".into(),
            argument_hint: String::new(),
            body: String::new(),
            dir: std::path::PathBuf::new(),
        };
        let tpl = crate::templates::Template {
            name: "bugfix".into(),
            description: "start a bug fix".into(),
            argument_hint: "<issue>".into(),
            body: String::new(),
            path: std::path::PathBuf::new(),
        };
        let cat = catalog(&[skill, shadow], &[tpl]);
        let find = |n: &str| cat.iter().find(|e| e.name == n).expect("entry").clone();
        assert_eq!(find("/review").source, Source::Skill);
        assert_eq!(find("/review").desc, "review the diff");
        assert_eq!(find("/bugfix").source, Source::Template);
        assert_eq!(find("/help").source, Source::Builtin);
        assert_eq!(cat.iter().filter(|e| e.name == "/help").count(), 1);
    }

    #[test]
    fn the_label_folds_the_argument_hint_in() {
        let e = &catalog(&[], &[])
            .into_iter()
            .find(|e| e.name == "/compact")
            .expect("compact");
        assert_eq!(e.label(), "/compact [instructions]");
    }
}
