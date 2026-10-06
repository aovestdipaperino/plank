//! The engine menu: every catalog engine, the ones that cannot run here
//! dimmed with their reason. Runs before any engine is created, on its own
//! alternate screen like the download screen.
//!
//! Everything but [`run`] is pure, so the trigger, the keys and the cursor are
//! tested without a terminal.

use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
};

use crate::enginefit::{EngineRow, Fit, size_label};

/// Whether launch should show the menu: a terminal on both ends, a local run,
/// and either `--pick-engine` or an engine that is not on disk and was not
/// named on the command line (that one keeps its `[Y/n]`).
#[must_use]
#[allow(clippy::fn_params_excessive_bools)] // five independent launch facts; the call site names each
pub fn should_pick(
    forced: bool,
    from_cli: bool,
    main_exists: bool,
    interactive: bool,
    local: bool,
) -> bool {
    interactive && local && (forced || (!from_cli && !main_exists))
}

/// What a key press asks the menu to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Up,
    Down,
    Pick,
    Cancel,
    Ignore,
}

/// Maps a key press to an [`Action`].
#[must_use]
pub fn classify(key: KeyEvent) -> Action {
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => Action::Cancel,
        KeyCode::Up | KeyCode::Char('k') => Action::Up,
        KeyCode::Down | KeyCode::Char('j') => Action::Down,
        KeyCode::Enter => Action::Pick,
        KeyCode::Esc | KeyCode::Char('q') => Action::Cancel,
        _ => Action::Ignore,
    }
}

/// Where the cursor starts: on `current` when it is selectable, else on the
/// first selectable row; `None` when nothing is.
#[must_use]
pub fn initial_cursor(rows: &[EngineRow], current: Option<&str>) -> Option<usize> {
    current
        .and_then(|c| rows.iter().position(|r| r.name == c && r.selectable()))
        .or_else(|| rows.iter().position(EngineRow::selectable))
}

/// The next selectable row from `from` in the given direction, or `from`
/// itself at either end.
#[must_use]
pub fn step(rows: &[EngineRow], from: usize, down: bool) -> usize {
    let found = if down {
        (from + 1..rows.len()).find(|&i| rows[i].selectable())
    } else {
        (0..from).rev().find(|&i| rows[i].selectable())
    };
    found.unwrap_or(from)
}

/// The right-hand column of a row.
#[must_use]
pub fn state_label(fit: &Fit) -> String {
    match fit {
        Fit::Installed => "installed".to_owned(),
        Fit::Download { bytes } => format!("download {}", size_label(*bytes)),
        Fit::Disabled { reason } => reason.clone(),
    }
}

/// Shows the menu and returns the picked engine's name, or `None` on cancel.
///
/// # Errors
/// When the terminal cannot be read.
pub fn run(rows: &[EngineRow], current: Option<&str>) -> Result<Option<String>, String> {
    let mut state = ListState::default();
    state.select(initial_cursor(rows, current));
    let mut terminal = ratatui::init();
    let result = loop {
        let _ = terminal.draw(|f| draw(f, rows, &mut state));
        if !event::poll(Duration::from_millis(250)).map_err(|e| e.to_string())? {
            continue;
        }
        let Ok(Event::Key(k)) = event::read() else {
            continue;
        };
        if k.kind != KeyEventKind::Press {
            continue;
        }
        let at = state.selected();
        match classify(k) {
            Action::Up => state.select(at.map(|i| step(rows, i, false))),
            Action::Down => state.select(at.map(|i| step(rows, i, true))),
            Action::Pick => {
                if let Some(i) = at {
                    break Ok(Some(rows[i].name.clone()));
                }
            }
            Action::Cancel => break Ok(None),
            Action::Ignore => {}
        }
    };
    ratatui::restore();
    result
}

fn draw(frame: &mut Frame, rows: &[EngineRow], state: &mut ListState) {
    let [title, list, help] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(3),
        Constraint::Length(1),
    ])
    .areas(frame.area());
    frame.render_widget(
        Paragraph::new("Choose an engine. It becomes the default for future runs."),
        title,
    );
    let items: Vec<ListItem> = rows
        .iter()
        .map(|r| {
            let dim = !r.selectable();
            let style = if dim {
                Style::default().add_modifier(Modifier::DIM)
            } else {
                Style::default()
            };
            let mut spans = vec![Span::styled(format!("{:<14}", r.name), style)];
            if !r.notes.is_empty() {
                spans.push(Span::styled(format!("{}  ", r.notes), style));
            }
            spans.push(Span::styled(
                state_label(&r.fit),
                style.add_modifier(Modifier::ITALIC),
            ));
            ListItem::new(Line::from(spans))
        })
        .collect();
    frame.render_stateful_widget(
        List::new(items)
            .block(Block::default().borders(Borders::ALL).title(" engines "))
            .highlight_style(Style::default().add_modifier(Modifier::REVERSED))
            .highlight_symbol("> "),
        list,
        state,
    );
    frame.render_widget(
        Paragraph::new("Up/Down move  Enter pick  Esc cancel")
            .style(Style::default().add_modifier(Modifier::DIM)),
        help,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enginefit::{EngineRow, Fit};
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn rows() -> Vec<EngineRow> {
        let r = |name: &str, fit| EngineRow {
            name: name.into(),
            notes: String::new(),
            fit,
        };
        vec![
            r(
                "a",
                Fit::Disabled {
                    reason: "needs 96 GB RAM (this machine: 16 GB)".into(),
                },
            ),
            r(
                "b",
                Fit::Download {
                    bytes: Some(5 * crate::enginefit::GIB),
                },
            ),
            r(
                "c",
                Fit::Disabled {
                    reason: "not supported by this build".into(),
                },
            ),
            r("d", Fit::Installed),
        ]
    }

    #[test]
    fn the_menu_shows_only_when_it_has_a_job_and_a_terminal() {
        // Selected engine missing, chosen by settings or default.
        assert!(should_pick(false, false, false, true, true));
        // Installed: nothing to do unless forced.
        assert!(!should_pick(false, false, true, true, true));
        assert!(should_pick(true, false, true, true, true));
        // --model named it: today's [Y/n] for that engine.
        assert!(!should_pick(false, true, false, true, true));
        // No terminal, or a remote/provider run: never.
        assert!(!should_pick(true, false, false, false, true));
        assert!(!should_pick(true, false, false, true, false));
    }

    #[test]
    fn the_cursor_starts_on_the_current_engine_when_selectable() {
        assert_eq!(initial_cursor(&rows(), Some("d")), Some(3));
        assert_eq!(initial_cursor(&rows(), Some("a")), Some(1));
        assert_eq!(initial_cursor(&rows(), None), Some(1));
        let none: Vec<EngineRow> = rows().into_iter().filter(|r| !r.selectable()).collect();
        assert_eq!(initial_cursor(&none, None), None);
    }

    #[test]
    fn moving_skips_disabled_rows_and_stops_at_the_ends() {
        let r = rows();
        assert_eq!(step(&r, 1, true), 3);
        assert_eq!(step(&r, 3, false), 1);
        assert_eq!(step(&r, 3, true), 3);
        assert_eq!(step(&r, 1, false), 1);
    }

    #[test]
    fn keys_map_to_actions() {
        let k = |c| KeyEvent::new(c, KeyModifiers::NONE);
        assert_eq!(classify(k(KeyCode::Up)), Action::Up);
        assert_eq!(classify(k(KeyCode::Char('k'))), Action::Up);
        assert_eq!(classify(k(KeyCode::Down)), Action::Down);
        assert_eq!(classify(k(KeyCode::Char('j'))), Action::Down);
        assert_eq!(classify(k(KeyCode::Enter)), Action::Pick);
        assert_eq!(classify(k(KeyCode::Esc)), Action::Cancel);
        assert_eq!(
            classify(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Action::Cancel
        );
        assert_eq!(classify(k(KeyCode::Char('x'))), Action::Ignore);
    }

    #[test]
    fn rows_say_installed_download_or_why_not() {
        let r = rows();
        assert_eq!(state_label(&r[3].fit), "installed");
        assert_eq!(state_label(&r[1].fit), "download 5.0 GB");
        assert_eq!(
            state_label(&r[0].fit),
            "needs 96 GB RAM (this machine: 16 GB)"
        );
    }
}
