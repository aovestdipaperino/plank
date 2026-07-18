//! Ratatui-based full-screen interactive UI.
//!
//! Uses the alternate screen buffer so every terminal — including block-based
//! ones like Warp that reflow normal output — treats plank as a proper TUI and
//! renders it cleanly. Replaces the hand-rolled raw-mode editor, scroll
//! regions, and in-place redraws.
//!
//! This module holds the presentational pieces: the styled scrollback log and
//! the per-frame layout. The interactive event loop lives in [`crate::ui`].

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Paragraph, Wrap};

use crate::viz::RenderSink;

/// Style for ordinary assistant/visible output.
fn visible_style() -> Style {
    Style::default()
}

/// Barely-visible gray for thinking text.
fn think_style() -> Style {
    Style::default().fg(Color::Indexed(238))
}

/// Scrollback of styled lines plus the line currently being streamed.
///
/// Implements [`RenderSink`] so the viz stream renderer appends directly:
/// visible output in the default style, thinking text in gray.
#[derive(Debug, Default)]
pub struct OutputLog {
    lines: Vec<Line<'static>>,
    current: Vec<Span<'static>>,
}

impl OutputLog {
    /// Creates an empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn append(&mut self, text: &str, style: Style) {
        for (i, part) in text.split('\n').enumerate() {
            if i > 0 {
                self.newline();
            }
            if !part.is_empty() {
                self.current.push(Span::styled(part.to_string(), style));
            }
        }
    }

    fn newline(&mut self) {
        self.lines
            .push(Line::from(std::mem::take(&mut self.current)));
    }

    /// Appends a fully-styled standalone line (e.g. the user echo).
    pub fn push_spans(&mut self, spans: Vec<Span<'static>>) {
        if !self.current.is_empty() {
            self.newline();
        }
        self.lines.push(Line::from(spans));
    }

    /// Appends a plain system line.
    pub fn push_plain(&mut self, text: impl Into<String>) {
        self.push_spans(vec![Span::raw(text.into())]);
    }

    /// Ensures the streamed output ends on a fresh line.
    pub fn end_line(&mut self) {
        if !self.current.is_empty() {
            self.newline();
        }
    }

    /// Renders the log (including the in-progress line) as ratatui text.
    #[must_use]
    pub fn to_text(&self) -> Text<'static> {
        let mut lines = self.lines.clone();
        if !self.current.is_empty() {
            lines.push(Line::from(self.current.clone()));
        }
        Text::from(lines)
    }
}

impl RenderSink for OutputLog {
    fn visible_text(&mut self, text: &str) {
        self.append(text, visible_style());
    }
    fn think_text(&mut self, text: &str) {
        self.append(text, think_style());
    }
}

/// Builds the styled user-echo line shown for a submitted prompt.
#[must_use]
pub fn user_echo_spans(text: &str) -> Vec<Span<'static>> {
    vec![
        Span::styled(
            "* ",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            text.to_string(),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
    ]
}

/// Draws one frame: output log, input line, and status bar.
///
/// `input` is the current prompt text and `cursor_col` its display column.
pub fn draw(frame: &mut Frame, log: &OutputLog, input: &str, cursor_col: u16, status: &str) {
    let area = frame.area();
    let rows = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(area);

    // Output area, scrolled so the latest lines stay visible.
    let text = log.to_text();
    let width = rows[0].width.max(1) as usize;
    let total: usize = text
        .lines
        .iter()
        .map(|l| l.width().div_ceil(width).max(1))
        .sum();
    let scroll = u16::try_from(total.saturating_sub(rows[0].height as usize)).unwrap_or(u16::MAX);
    let para = Paragraph::new(text)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(para, rows[0]);

    // Input line.
    let prompt = "plank> ";
    let input_line = Line::from(vec![
        Span::styled(prompt, Style::default().fg(Color::Cyan)),
        Span::raw(input.to_string()),
    ]);
    frame.render_widget(Paragraph::new(input_line), rows[1]);
    let cursor_x = rows[1].x + u16::try_from(prompt.len()).unwrap_or(0) + cursor_col;
    frame.set_cursor_position(Position::new(
        cursor_x.min(area.right().saturating_sub(1)),
        rows[1].y,
    ));

    // Status bar, reverse-styled across the full width.
    let status_style = Style::default()
        .bg(Color::Indexed(238))
        .fg(Color::Indexed(252));
    frame.render_widget(
        Paragraph::new(status.to_string()).style(status_style),
        rows[2],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_splits_on_newlines() {
        let mut log = OutputLog::new();
        log.visible_text("hello\nworld");
        log.end_line();
        // "hello" and "world" become two lines.
        assert_eq!(log.lines.len(), 2);
    }

    #[test]
    fn think_and_visible_are_styled_differently() {
        let mut log = OutputLog::new();
        log.think_text("pondering");
        log.end_line();
        let spans = &log.lines[0];
        assert_eq!(spans.spans[0].style.fg, Some(Color::Indexed(238)));
    }

    #[test]
    fn user_echo_is_bold() {
        let spans = user_echo_spans("hi");
        assert!(spans[0].style.add_modifier.contains(Modifier::BOLD));
    }
}
