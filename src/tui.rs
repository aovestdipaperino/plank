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

use std::sync::{Arc, OnceLock};

use ratatui_markdown::ThemeConfig;
use ratatui_markdown::highlight::{HighlightHooks, TreeSitterHighlighter};
use ratatui_markdown::markdown::MarkdownRenderer;

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
/// visible output rendered as markdown (via `ratatui-markdown`, including
/// code-block syntax highlighting), thinking and tool text in gray/plain.
#[derive(Debug, Default)]
pub struct OutputLog {
    lines: Vec<Line<'static>>,
    current: Vec<Span<'static>>,
    /// Raw markdown of the visible segment currently streaming, plus the
    /// index in `lines` where its rendered form starts. Re-rendered whole on
    /// each append so partial emphasis/fences resolve as more text arrives.
    md_buf: String,
    md_start: Option<usize>,
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

    /// Ends the streaming markdown segment; later appends start a new one.
    fn md_close(&mut self) {
        self.md_buf.clear();
        self.md_start = None;
    }

    /// Re-renders the whole in-progress markdown segment in place.
    fn md_render(&mut self) {
        static HIGHLIGHTER: OnceLock<Arc<TreeSitterHighlighter>> = OnceLock::new();
        let Some(start) = self.md_start else { return };
        let width = ratatui::crossterm::terminal::size()
            .map_or(80, |(w, _)| w as usize)
            .max(20);
        let hl = HIGHLIGHTER
            .get_or_init(|| Arc::new(TreeSitterHighlighter::new()))
            .clone();
        let md = MarkdownRenderer::new(width)
            .with_render_hooks(Box::new(HighlightHooks::new(hl, width)));
        let blocks = md.parse(&self.md_buf);
        self.lines.truncate(start);
        self.lines.extend(md.render(&blocks, &ThemeConfig::new()));
    }

    /// Appends a fully-styled standalone line (e.g. the user echo).
    pub fn push_spans(&mut self, spans: Vec<Span<'static>>) {
        self.md_close();
        if !self.current.is_empty() {
            self.newline();
        }
        self.lines.push(Line::from(spans));
    }

    /// Appends a plain system line.
    pub fn push_plain(&mut self, text: impl Into<String>) {
        self.push_spans(vec![Span::raw(text.into())]);
    }

    /// Appends a line in the thinking gray, for tool and debug output.
    pub fn push_dim(&mut self, text: impl Into<String>) {
        self.push_spans(vec![Span::styled(text.into(), think_style())]);
    }

    /// Appends ANSI-colored text, one log line per input line.
    pub fn push_ansi(&mut self, text: &str) {
        self.md_close();
        self.end_line();
        self.lines.extend(ansi_to_lines(text));
    }

    /// Removes the most recent completed line (e.g. a transient status note).
    pub fn pop_line(&mut self) {
        self.md_close();
        self.lines.pop();
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
        if self.md_start.is_none() {
            self.end_line();
            self.md_start = Some(self.lines.len());
        }
        self.md_buf.push_str(text);
        self.md_render();
    }
    fn think_text(&mut self, text: &str) {
        self.md_close();
        self.append(text, think_style());
    }
    fn tool_text(&mut self, text: &str) {
        self.md_close();
        self.append(text, visible_style());
    }
}

/// Converts true-color ANSI art (from `logo-art`) into styled ratatui lines.
///
/// Understands the SGR subset the crate emits: `38;2;r;g;b` / `48;2;r;g;b`
/// truecolor, `39`/`49` defaults, and `\x1b[m` reset. Other bytes are text.
#[must_use]
pub fn ansi_to_lines(art: &str) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut run = String::new();
    let mut style = Style::default();
    let mut chars = art.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\x1b' if chars.peek() == Some(&'[') => {
                chars.next(); // consume '['
                let mut params = String::new();
                for pc in chars.by_ref() {
                    if pc == 'm' {
                        break;
                    }
                    params.push(pc);
                }
                if !run.is_empty() {
                    spans.push(Span::styled(std::mem::take(&mut run), style));
                }
                style = apply_sgr(style, &params);
            }
            '\n' => {
                if !run.is_empty() {
                    spans.push(Span::styled(std::mem::take(&mut run), style));
                }
                lines.push(Line::from(std::mem::take(&mut spans)));
            }
            other => run.push(other),
        }
    }
    if !run.is_empty() {
        spans.push(Span::styled(run, style));
    }
    if !spans.is_empty() {
        lines.push(Line::from(spans));
    }
    lines
}

/// Applies one SGR parameter string to a style (truecolor, 256-color, and
/// fg/bg reset only).
fn apply_sgr(mut style: Style, params: &str) -> Style {
    if params.is_empty() {
        return Style::default();
    }
    let parts: Vec<&str> = params.split(';').collect();
    let rgb = |i: usize| -> Color {
        let c = |k: usize| parts.get(k).and_then(|s| s.parse::<u8>().ok()).unwrap_or(0);
        Color::Rgb(c(i), c(i + 1), c(i + 2))
    };
    let mut i = 0;
    while i < parts.len() {
        match parts[i] {
            "" | "0" => {
                style = Style::default();
                i += 1;
            }
            "39" => {
                style = style.fg(Color::Reset);
                i += 1;
            }
            "49" => {
                style = style.bg(Color::Reset);
                i += 1;
            }
            "38" if parts.get(i + 1) == Some(&"2") => {
                style = style.fg(rgb(i + 2));
                i += 5;
            }
            "48" if parts.get(i + 1) == Some(&"2") => {
                style = style.bg(rgb(i + 2));
                i += 5;
            }
            "38" if parts.get(i + 1) == Some(&"5") => {
                if let Some(n) = parts.get(i + 2).and_then(|s| s.parse::<u8>().ok()) {
                    style = style.fg(Color::Indexed(n));
                }
                i += 3;
            }
            "48" if parts.get(i + 1) == Some(&"5") => {
                if let Some(n) = parts.get(i + 2).and_then(|s| s.parse::<u8>().ok()) {
                    style = style.bg(Color::Indexed(n));
                }
                i += 3;
            }
            _ => i += 1,
        }
    }
    style
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
/// `scroll_back` is how many wrapped lines above the bottom the view sits;
/// it is clamped in place to the scrollable range.
pub fn draw(
    frame: &mut Frame,
    log: &OutputLog,
    input: &str,
    cursor_col: u16,
    status: &str,
    scroll_back: &mut usize,
) {
    let area = frame.area();
    let rows = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(area);

    // Output area, scrolled so the latest lines stay visible unless the user
    // has wheeled back into the buffer.
    let text = log.to_text();
    let width = rows[0].width.max(1) as usize;
    let total: usize = text
        .lines
        .iter()
        .map(|l| l.width().div_ceil(width).max(1))
        .sum();
    let max_back = total.saturating_sub(rows[0].height as usize);
    *scroll_back = (*scroll_back).min(max_back);
    let scroll = u16::try_from(max_back - *scroll_back).unwrap_or(u16::MAX);
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

    // Status bar, reverse-styled across the full width, with a magenta bar.
    let status_style = Style::default()
        .bg(Color::Indexed(238))
        .fg(Color::Indexed(252));
    frame.render_widget(
        Paragraph::new(status_bar_line(status, status_style)).style(status_style),
        rows[2],
    );
}

/// Pushes spans for `seg`, painting the accent word — `prefill` before the
/// bar, or the trailing-`…` spinner verb — in the theme color.
fn push_accented(spans: &mut Vec<Span<'static>>, seg: &str, base: Style, theme: Style) {
    let range = seg
        .find("prefill")
        .map(|i| (i, i + "prefill".len()))
        .or_else(|| {
            seg.find('…').map(|e| {
                let start = seg[..e].rfind(' ').map_or(0, |i| i + 1);
                (start, e + '…'.len_utf8())
            })
        });
    if let Some((start, end)) = range {
        spans.push(Span::styled(seg[..start].to_string(), base));
        spans.push(Span::styled(seg[start..end].to_string(), theme));
        spans.push(Span::styled(seg[end..].to_string(), base));
    } else {
        spans.push(Span::styled(seg.to_string(), base));
    }
}

/// Builds the status line, coloring the progress bar's filled arrows and the
/// accent word (operation name or spinner verb) in the theme color.
///
/// The bar segment lives between `[` and `]`; `▶` cells render in the theme
/// color (military green) and `·` cells a dim gray.
fn status_bar_line(text: &str, base: Style) -> Line<'static> {
    let theme = base
        .fg(Color::Indexed(crate::status::THEME_COLOR))
        .add_modifier(Modifier::BOLD);
    let bar = text
        .find('[')
        .and_then(|open| text[open..].find(']').map(|i| (open, open + i)));
    let Some((open, close)) = bar else {
        let mut spans = Vec::new();
        push_accented(&mut spans, text, base, theme);
        return Line::from(spans);
    };
    let mut spans = Vec::new();
    push_accented(&mut spans, &text[..=open], base, theme);
    for ch in text[open + 1..close].chars() {
        let style = match ch {
            '▶' => theme,
            '·' => base.fg(Color::Indexed(240)),
            _ => base,
        };
        spans.push(Span::styled(ch.to_string(), style));
    }
    spans.push(Span::styled(text[close..].to_string(), base));
    Line::from(spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ansi_to_lines_parses_truecolor_cells() {
        // Two cells (bg red '▄', bg green ' ') then newline.
        let art = "\x1b[48;2;255;0;0m▄\x1b[48;2;0;255;0m \x1b[m\n";
        let lines = ansi_to_lines(art);
        assert_eq!(lines.len(), 1);
        let spans = &lines[0].spans;
        assert_eq!(spans[0].content.as_ref(), "▄");
        assert_eq!(spans[0].style.bg, Some(Color::Rgb(255, 0, 0)));
        assert_eq!(spans[1].style.bg, Some(Color::Rgb(0, 255, 0)));
    }

    #[test]
    fn ansi_to_lines_parses_256_color() {
        let art = "\x1b[38;5;105m⛁\x1b[0m \x1b[48;5;44mx\x1b[m\n";
        let lines = ansi_to_lines(art);
        assert_eq!(lines.len(), 1);
        let spans = &lines[0].spans;
        assert_eq!(spans[0].content.as_ref(), "⛁");
        assert_eq!(spans[0].style.fg, Some(Color::Indexed(105)));
        assert_eq!(spans.last().unwrap().style.bg, Some(Color::Indexed(44)));
    }

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
    fn visible_text_renders_markdown_emphasis() {
        let mut log = OutputLog::new();
        log.visible_text("some **bold** words");
        let spans = &log.lines[0].spans;
        assert!(
            spans
                .iter()
                .any(|s| s.content.as_ref() == "bold"
                    && s.style.add_modifier.contains(Modifier::BOLD))
        );
    }

    #[test]
    fn visible_text_highlights_code_blocks() {
        let mut log = OutputLog::new();
        log.visible_text("```rust\nfn main() {}\n```\n");
        // Real highlighting produces multiple distinct colors (keyword vs
        // identifier), not one flat code color.
        let mut colors: Vec<String> = log
            .lines
            .iter()
            .flat_map(|l| &l.spans)
            .filter_map(|s| s.style.fg.map(|c| format!("{c:?}")))
            .collect();
        colors.sort_unstable();
        colors.dedup();
        assert!(
            colors.len() >= 2,
            "expected multi-color highlighted code: {:?}",
            log.lines
        );
    }

    #[test]
    fn tool_text_is_plain_and_closes_markdown_segment() {
        let mut log = OutputLog::new();
        log.visible_text("**a**");
        log.tool_text("\n$ ls **not markdown**\n");
        log.visible_text("**b**");
        // The banner line keeps its literal asterisks.
        assert!(
            log.lines
                .iter()
                .flat_map(|l| &l.spans)
                .any(|s| s.content.contains("**not markdown**"))
        );
        // The second segment re-renders independently as bold "b".
        assert!(
            log.lines
                .iter()
                .flat_map(|l| &l.spans)
                .any(|s| s.content.as_ref() == "b"
                    && s.style.add_modifier.contains(Modifier::BOLD))
        );
    }

    #[test]
    fn user_echo_is_bold() {
        let spans = user_echo_spans("hi");
        assert!(spans[0].style.add_modifier.contains(Modifier::BOLD));
    }
}
