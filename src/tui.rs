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
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Position, Rect};
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

/// Bold red for error banners, matching the C renderer's `\x1b[1;31m`.
fn error_style() -> Style {
    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
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

    /// Snapshots the current committed line count, for [`truncate_to`]. Ends
    /// any in-progress line first so the checkpoint sits on a line boundary.
    pub fn checkpoint(&mut self) -> usize {
        self.md_close();
        self.end_line();
        self.lines.len()
    }

    /// Rolls the log back to a [`checkpoint`](Self::checkpoint), discarding
    /// every line appended since (used to drop a preempted generation pass).
    pub fn truncate_to(&mut self, len: usize) {
        self.md_close();
        self.current.clear();
        self.lines.truncate(len);
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
    fn error_text(&mut self, text: &str) {
        self.md_close();
        self.append(text, error_style());
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

/// A mouse selection over screen cells, in reading order: inclusive
/// `(x, y)` start and end positions.
pub type Selection = ((u16, u16), (u16, u16));

/// Orders two drag endpoints into reading order (top-to-bottom, then
/// left-to-right), returning a normalized [`Selection`].
#[must_use]
pub fn normalize_selection(a: (u16, u16), b: (u16, u16)) -> Selection {
    if (a.1, a.0) <= (b.1, b.0) {
        (a, b)
    } else {
        (b, a)
    }
}

/// Column bounds (inclusive) the selection covers on screen row `y`, clamped
/// to `area`; `None` when the row is outside the selection or the area.
fn selection_row_bounds(sel: Selection, area: Rect, y: u16) -> Option<(u16, u16)> {
    let ((sx, sy), (ex, ey)) = sel;
    if y < sy || y > ey || y < area.top() || y >= area.bottom() {
        return None;
    }
    let x0 = if y == sy { sx } else { area.left() };
    let x1 = if y == ey {
        ex
    } else {
        area.right().saturating_sub(1)
    };
    let x0 = x0.max(area.left());
    let x1 = x1.min(area.right().saturating_sub(1));
    (x0 <= x1).then_some((x0, x1))
}

/// Paints the selection as reversed video over the rendered cells.
pub fn highlight_selection(buf: &mut Buffer, area: Rect, sel: Selection) {
    for y in area.top()..area.bottom() {
        if let Some((x0, x1)) = selection_row_bounds(sel, area, y) {
            let row = Rect::new(x0, y, x1 - x0 + 1, 1);
            buf.set_style(row, Style::default().add_modifier(Modifier::REVERSED));
        }
    }
}

/// Extracts the selected text from the rendered screen buffer, one line per
/// screen row with trailing whitespace trimmed (WYSIWYG copy).
#[must_use]
pub fn selection_text(buf: &Buffer, area: Rect, sel: Selection) -> String {
    let mut out = Vec::new();
    for y in area.top()..area.bottom() {
        let Some((x0, x1)) = selection_row_bounds(sel, area, y) else {
            continue;
        };
        let mut line = String::new();
        for x in x0..=x1 {
            if let Some(cell) = buf.cell(ratatui::layout::Position::new(x, y)) {
                line.push_str(cell.symbol());
            }
        }
        out.push(line.trim_end().to_owned());
    }
    out.join("\n")
}

/// Minimal standard base64 (with padding) for the OSC 52 payload.
fn base64(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        let n = (u32::from(chunk[0]) << 16) | (u32::from(b1) << 8) | u32::from(b2);
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Copies `text` to the system clipboard: `pbcopy` for the local macOS
/// clipboard, plus an OSC 52 escape so it also works over SSH in terminals
/// that support it. Best-effort on both paths.
pub fn copy_to_clipboard(text: &str) {
    use std::io::Write as _;
    if let Ok(mut child) = std::process::Command::new("pbcopy")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        if let Some(stdin) = child.stdin.as_mut() {
            let _ = stdin.write_all(text.as_bytes());
        }
        drop(child.stdin.take());
        let _ = child.wait();
    }
    let mut out = std::io::stdout();
    let _ = write!(out, "\x1b]52;c;{}\x1b\\", base64(text.as_bytes()));
    let _ = out.flush();
}

/// Scroll state of the output log viewport.
///
/// While `follow` is set the view tracks the bottom of the log (the streaming
/// default). Scrolling back pins the viewport at wrapped-line offset `top`,
/// which stays put while new output arrives; `draw` clamps `top` in place and
/// re-enters follow mode once the view reaches the bottom again.
#[derive(Debug, Clone, Copy)]
pub struct OutputView {
    /// First wrapped log line shown, updated by `draw` every frame.
    pub top: usize,
    /// True when the view tracks the newest output.
    pub follow: bool,
}

impl Default for OutputView {
    fn default() -> Self {
        Self {
            top: 0,
            follow: true,
        }
    }
}

/// Theme green, used for the prompt separator rule and panel accents.
const THEME_GREEN: Color = Color::Indexed(114);

/// Splits `area` into `(output, input, status)` rows, giving the input
/// `input_rows` rows. When `has_prompt`, a
/// one-row green rule is inserted just above the input line (and drawn here),
/// separating the scrollback from the resting prompt; while the agent is busy
/// (no prompt) the rule is omitted.
fn frame_rows(
    frame: &mut Frame,
    area: Rect,
    has_prompt: bool,
    input_rows: u16,
) -> (Rect, Rect, Rect) {
    let (output, input, status, rule) = frame_geom(area, has_prompt, input_rows);
    // Draw-site instrumentation for `--ui-remote`. This is the one place both
    // `draw` and `draw_btw_split` funnel through, so the frame is reset and
    // the structural regions published here; `render_input` and `render_popup`
    // append their own regions later in the same pass.
    crate::uiremote::begin_frame();
    crate::uiremote::region("root", area, &[]);
    crate::uiremote::region("output", output, &[]);
    crate::uiremote::region("status", status, &[]);
    if let Some(rule) = rule {
        let text = "─".repeat(rule.width as usize);
        frame.render_widget(
            Paragraph::new(Span::styled(text, Style::default().fg(THEME_GREEN))),
            rule,
        );
    }
    (output, input, status)
}

/// Pure geometry behind [`frame_rows`]: returns `(output, input, status, rule)`
/// where `rule` is the separator row, present only when `has_prompt`.
///
/// Split out so layout can be computed (and tested) without a `Frame`.
fn frame_geom(area: Rect, has_prompt: bool, input_rows: u16) -> (Rect, Rect, Rect, Option<Rect>) {
    if has_prompt {
        let r = Layout::vertical([
            Constraint::Min(1),             // output
            Constraint::Length(1),          // separator rule
            Constraint::Length(input_rows), // input
            Constraint::Length(1),          // status
        ])
        .split(area);
        (r[0], r[2], r[3], Some(r[1]))
    } else {
        let r = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);
        (r[0], r[1], r[2], None)
    }
}

/// Splits the resting-prompt input into styled spans so a valid command is
/// highlighted live as the user types: a known `/command` token in theme green,
/// and the `!` shell-escape marker in red. Anything else stays default-styled.
///
/// Validity mirrors dispatch: the green highlight appears only when the whole
/// line parses as a known command ([`crate::config::slash_command_known`]), so
/// partial (`/hel`) and unknown (`/nope`) inputs stay plain until complete.
fn input_spans(input: &str) -> Vec<Span<'static>> {
    // Shell escape (`!cmd`): only the `!` marker is colored, red; the shell
    // command text after it stays plain (any non-empty command is "valid").
    if let Some(rest) = input.strip_prefix('!') {
        let mut spans = vec![Span::styled(
            "!".to_string(),
            Style::default().fg(Color::Red),
        )];
        if !rest.is_empty() {
            spans.push(Span::raw(rest.to_string()));
        }
        return spans;
    }
    // Slash command: highlight the leading command token green, but only when
    // the line as a whole is a known command invocation.
    if input.starts_with('/') && crate::config::slash_command_known(input) {
        let token_len = input.find(char::is_whitespace).unwrap_or(input.len());
        let (cmd, rest) = input.split_at(token_len);
        let mut spans = vec![Span::styled(
            cmd.to_string(),
            Style::default().fg(THEME_GREEN),
        )];
        if !rest.is_empty() {
            spans.push(Span::raw(rest.to_string()));
        }
        return spans;
    }
    vec![Span::raw(input.to_string())]
}

/// Computes the popup rect: it floats up from the top edge of the input,
/// overlaying the output pane, so it never reaches the status bar. When fewer
/// rows fit above the input than requested it shrinks rather than moving down.
///
/// The bottom row deliberately overlays the green separator rule drawn by
/// [`frame_rows`]: the popup then sits flush against the prompt it is completing
/// (leaving a blank gap instead reads as a detached, floating box), and the rule
/// is redrawn the moment the popup closes.
#[must_use]
pub fn popup_rect(output: Rect, input: Rect, rows: u16) -> Rect {
    let space_above = input.y.saturating_sub(output.y);
    let h = rows
        .min(space_above)
        .min(output.height)
        .min(u16::try_from(crate::complete::MAX_ROWS).unwrap_or(u16::MAX));
    if h == output.height && output.height < space_above {
        // The output pane itself is the limiting factor (a tall multi-line
        // input has squeezed it down): anchor to its top rather than
        // floating with a gap between the popup and the input above it.
        Rect::new(output.x, output.y, output.width, h)
    } else {
        Rect::new(output.x, input.y.saturating_sub(h), output.width, h)
    }
}

/// Draws the `@` suggestion popup over the output pane.
///
/// Clears the region first so the scrollback underneath does not bleed
/// through; the selected row is highlighted in the theme green.
fn render_popup(frame: &mut Frame, area: Rect, popup: &crate::complete::Popup) {
    use ratatui::widgets::{Clear, List, ListItem, ListState, StatefulWidget};
    if area.height == 0 || popup.rows().is_empty() {
        return;
    }
    crate::uiremote::region(
        "popup",
        area,
        &[
            (
                "rows",
                crate::tools::mcp::Json::Num(f64::from(
                    u32::try_from(popup.rows().len()).unwrap_or(u32::MAX),
                )),
            ),
            (
                "selected",
                crate::tools::mcp::Json::Num(f64::from(
                    u32::try_from(popup.selected()).unwrap_or(u32::MAX),
                )),
            ),
        ],
    );
    frame.render_widget(Clear, area);
    let items: Vec<ListItem> = popup
        .rows()
        .iter()
        .map(|m| {
            let marker = match m.kind {
                crate::complete::Kind::Dir => "/",
                crate::complete::Kind::Resource => "@",
                crate::complete::Kind::File => " ",
            };
            ListItem::new(Span::raw(format!("{marker} {}", m.text)))
        })
        .collect();
    let list = List::new(items)
        .highlight_style(Style::default().fg(THEME_GREEN))
        .highlight_symbol("> ");
    let mut state = ListState::default();
    state.select(Some(popup.selected()));
    StatefulWidget::render(list, area, frame.buffer_mut(), &mut state);
}

/// Draws the `@` popup over the frame just rendered by [`draw`] or
/// [`draw_btw_split`], recomputing the same layout those use so the popup lands
/// directly above the input line.
///
/// `input_text` must be the same prompt text passed to the draw call, so the
/// input's height (and therefore the popup's anchor) matches.
pub fn draw_popup(frame: &mut Frame, input_text: &str, popup: &crate::complete::Popup) {
    let (output, input, _, _) = frame_geom(frame.area(), true, input_height(input_text));
    let rows = u16::try_from(popup.rows().len()).unwrap_or(u16::MAX);
    render_popup(frame, popup_rect(output, input, rows), popup);
}

/// Horizontal scroll offset for the input line, in display columns.
///
/// The prompt glyph is always drawn, so the text gets `width - prompt_width`
/// cells. When the cursor would fall past the last usable column, the view
/// scrolls just far enough to keep the cursor (and thus the freshly typed
/// character) on screen.
fn input_scroll(cursor_col: u16, row_width: u16, prompt_width: u16) -> u16 {
    let avail = row_width.saturating_sub(prompt_width);
    if avail == 0 {
        return cursor_col;
    }
    cursor_col.saturating_sub(avail.saturating_sub(1))
}

/// Number of rows the prompt needs for `input` — one per embedded newline.
///
/// The input area grows with the text, taking rows away from the scrollback.
#[must_use]
pub fn input_height(input: &str) -> u16 {
    u16::try_from(input.matches('\n').count() + 1).unwrap_or(u16::MAX)
}

/// Draws the prompt glyph and the (possibly multi-line) input text into
/// `input_area`, placing the terminal cursor at `cursor` = `(row, col)`.
///
/// Rows after the first are indented under the prompt, and every row shares
/// the cursor row's horizontal scroll so a long line stays visible.
fn render_input(frame: &mut Frame, input_area: Rect, input: &str, cursor: (u16, u16)) {
    // The input region carries its text, so a harness can assert on what is
    // typed without decoding the ANSI snapshot. It is registered here rather
    // than in `frame_rows` because only this function sees the text; while the
    // agent is busy no prompt is drawn and no `input` region appears.
    crate::uiremote::region(
        "input",
        input_area,
        &[("text", crate::tools::mcp::Json::Str(input.to_string()))],
    );
    let prompt = crate::status::prompt_text();
    let prompt_span = Span::styled(prompt, Style::default().fg(Color::Cyan));
    let prompt_width = u16::try_from(prompt_span.width()).unwrap_or(0);
    let prompt_row = Rect {
        height: 1,
        ..input_area
    };
    frame.render_widget(Paragraph::new(Line::from(vec![prompt_span])), prompt_row);

    // Only the first line can be a slash command or a `!` shell escape, so the
    // continuation lines render plain.
    let lines: Vec<Line<'static>> = input
        .split('\n')
        .enumerate()
        .map(|(i, l)| {
            if i == 0 {
                Line::from(input_spans(l))
            } else {
                Line::raw(l.to_string())
            }
        })
        .collect();

    let (cursor_row, cursor_col) = cursor;
    let scroll = input_scroll(cursor_col, input_area.width, prompt_width);
    let text_area = Rect {
        x: input_area.x + prompt_width,
        y: input_area.y,
        width: input_area.width.saturating_sub(prompt_width),
        height: input_area.height,
    };
    frame.render_widget(Paragraph::new(lines).scroll((0, scroll)), text_area);

    let cursor_x = text_area.x + cursor_col.saturating_sub(scroll);
    frame.set_cursor_position(Position::new(
        cursor_x.min(input_area.right().saturating_sub(1)),
        input_area.y + cursor_row.min(input_area.height.saturating_sub(1)),
    ));
}

/// Draws one frame: output log, input line, and status bar.
///
/// `input` is the current prompt text and `cursor` its `(row, col)` position.
/// `input` is `None` while the agent is busy (prefill/generation): the prompt
/// line renders empty and the cursor stays hidden until input is accepted again.
/// `view` is the scroll state; it is clamped in place to the scrollable range
/// and a jump-to-bottom hint is shown while it is pinned above the bottom.
pub fn draw(
    frame: &mut Frame,
    log: &OutputLog,
    input: Option<&str>,
    cursor: (u16, u16),
    status: &str,
    view: &mut OutputView,
    selection: Option<Selection>,
) {
    let area = frame.area();
    let (output, input_row, status_row) =
        frame_rows(frame, area, input.is_some(), input.map_or(1, input_height));

    render_output(frame, output, log, view, selection);

    // Input line: hidden entirely (no prompt, no cursor) while the agent is busy.
    if let Some(input) = input {
        render_input(frame, input_row, input, cursor);
    }

    // Status bar, reverse-styled across the full width, with a magenta bar.
    let status_style = Style::default()
        .bg(Color::Indexed(238))
        .fg(Color::Indexed(252));
    frame.render_widget(
        Paragraph::new(status_bar_line(
            &with_remote_marker(status),
            anim_tick_ms(),
            status_style,
        ))
        .style(status_style),
        status_row,
    );
}

/// Renders a scrollback log into `area`, clamping `view` to the scrollable
/// range and following the newest output unless the user has scrolled back.
fn render_output(
    frame: &mut Frame,
    area: Rect,
    log: &OutputLog,
    view: &mut OutputView,
    selection: Option<Selection>,
) {
    let text = log.to_text();
    let width = area.width.max(1) as usize;
    let total: usize = text
        .lines
        .iter()
        .map(|l| l.width().div_ceil(width).max(1))
        .sum();
    let max_top = total.saturating_sub(area.height as usize);
    if view.follow || view.top >= max_top {
        view.top = max_top;
        view.follow = true;
    }
    let scroll = u16::try_from(view.top).unwrap_or(u16::MAX);
    let para = Paragraph::new(text)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(para, area);
    if !view.follow {
        draw_jump_hint(frame, area);
    }
    if let Some(sel) = selection {
        highlight_selection(frame.buffer_mut(), area, sel);
    }
}

/// Draws one frame with the output area split into two columns: the main
/// conversation (60%) on the left and the live `/btw` side answer (40%) on
/// the right, separated by a labelled left border. The input line and status
/// bar span the full width below, as in [`draw`]. Used while a `/btw` panel
/// is active; pressing Esc cancels the side answer and returns to [`draw`].
#[allow(clippy::too_many_arguments)]
pub fn draw_btw_split(
    frame: &mut Frame,
    log: &OutputLog,
    btw_log: &OutputLog,
    btw_view: &mut OutputView,
    input: Option<&str>,
    cursor: (u16, u16),
    status: &str,
    view: &mut OutputView,
) {
    use ratatui::widgets::{Block, Borders};

    let area = frame.area();
    let (output, input_row, status_row) =
        frame_rows(frame, area, input.is_some(), input.map_or(1, input_height));
    let cols =
        Layout::horizontal([Constraint::Percentage(60), Constraint::Percentage(40)]).split(output);

    render_output(frame, cols[0], log, view, None);

    // The btw panel: a left border acts as the vertical separator and carries
    // a "btw · Esc closes" title; the answer streams inside.
    let block = Block::default()
        .borders(Borders::LEFT)
        .border_style(Style::default().fg(Color::Indexed(238)))
        .title(Span::styled(
            " btw · Esc closes ",
            Style::default()
                .fg(THEME_GREEN)
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(cols[1]);
    frame.render_widget(block, cols[1]);
    render_output(frame, inner, btw_log, btw_view, None);

    // Input line and status bar span the full width, identical to `draw`.
    if let Some(input) = input {
        render_input(frame, input_row, input, cursor);
    }
    let status_style = Style::default()
        .bg(Color::Indexed(238))
        .fg(Color::Indexed(252));
    frame.render_widget(
        Paragraph::new(status_bar_line(
            &with_remote_marker(status),
            anim_tick_ms(),
            status_style,
        ))
        .style(status_style),
        status_row,
    );
}

/// Overlays the jump-to-bottom affordance on the output area's bottom-right
/// corner while the view is pinned above the newest output.
fn draw_jump_hint(frame: &mut Frame, area: Rect) {
    const HINT: &str = " ↓ End: jump to bottom ";
    let hint_width = u16::try_from(HINT.chars().count()).unwrap_or(u16::MAX);
    if area.width < hint_width || area.height == 0 {
        return;
    }
    let rect = Rect::new(area.right() - hint_width, area.bottom() - 1, hint_width, 1);
    let style = Style::default()
        .bg(Color::Indexed(238))
        .fg(Color::Indexed(252))
        .add_modifier(Modifier::BOLD);
    frame.render_widget(Paragraph::new(Span::styled(HINT, style)), rect);
}

/// Milliseconds since the first frame, driving the shimmer sweep.
fn anim_tick_ms() -> u64 {
    use std::sync::OnceLock;
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    u64::try_from(
        EPOCH
            .get_or_init(std::time::Instant::now)
            .elapsed()
            .as_millis(),
    )
    .unwrap_or(0)
}

/// Pushes the accent word with a shimmer: a 3-column bright highlight sweeps
/// right-to-left across the word, one column per `SHIMMER_STEP_MS`, over a
/// cycle of word width + 20 columns (so the highlight rests off-text between
/// sweeps).
fn push_shimmered(spans: &mut Vec<Span<'static>>, word: &str, tick_ms: u64, theme: Style) {
    let shimmer = theme.fg(Color::Indexed(crate::status::SHIMMER_COLOR));
    let width = i64::try_from(word.chars().count()).unwrap_or(0);
    let cycle = width + 20;
    let step = i64::try_from(tick_ms / crate::status::SHIMMER_STEP_MS).unwrap_or(0);
    let center = width + 10 - step % cycle;
    // Split into before / shimmer / after segments by char column.
    let (mut before, mut shim, mut after) = (String::new(), String::new(), String::new());
    for (col, ch) in word.chars().enumerate() {
        let col = i64::try_from(col).unwrap_or(i64::MAX);
        if col < center - 1 {
            before.push(ch);
        } else if col <= center + 1 {
            shim.push(ch);
        } else {
            after.push(ch);
        }
    }
    for (text, style) in [(before, theme), (shim, shimmer), (after, theme)] {
        if !text.is_empty() {
            spans.push(Span::styled(text, style));
        }
    }
}

/// Pushes spans for `seg`, painting the accent word — `prefill` before the
/// bar, or the trailing-`…` spinner verb — in the theme color with the
/// shimmer animation sweeping across it.
fn push_accented(
    spans: &mut Vec<Span<'static>>,
    seg: &str,
    tick_ms: u64,
    base: Style,
    theme: Style,
) {
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
        push_shimmered(spans, &seg[start..end], tick_ms, theme);
        spans.push(Span::styled(seg[end..].to_string(), base));
    } else {
        spans.push(Span::styled(seg.to_string(), base));
    }
}

/// Appends a visible marker to the status text while `--ui-remote` is active.
///
/// A session that can be typed into from outside must say so on screen.
fn with_remote_marker(status: &str) -> std::borrow::Cow<'_, str> {
    if crate::uiremote::recording_enabled() {
        std::borrow::Cow::Owned(format!("{status} | remote"))
    } else {
        std::borrow::Cow::Borrowed(status)
    }
}

/// Builds the status line, coloring the progress bar's filled arrows and the
/// accent word (operation name or spinner verb) in the theme color.
///
/// The bar segment lives between `[` and `]`; `▶` cells render in the theme
/// color (military green) and `·` cells a dim gray.
fn status_bar_line(text: &str, tick_ms: u64, base: Style) -> Line<'static> {
    let theme = base
        .fg(Color::Indexed(crate::status::THEME_COLOR))
        .add_modifier(Modifier::BOLD);
    let bar = text
        .find('[')
        .and_then(|open| text[open..].find(']').map(|i| (open, open + i)));
    let Some((open, close)) = bar else {
        let mut spans = Vec::new();
        push_accented(&mut spans, text, tick_ms, base, theme);
        return Line::from(spans);
    };
    let mut spans = Vec::new();
    push_accented(&mut spans, &text[..=open], tick_ms, base, theme);
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

    /// `(content, fg)` pairs for each span, for terse assertions.
    fn parts(input: &str) -> Vec<(String, Option<Color>)> {
        input_spans(input)
            .into_iter()
            .map(|s| (s.content.into_owned(), s.style.fg))
            .collect()
    }

    #[test]
    fn input_spans_highlights_known_slash_command_green() {
        // A bare known command: whole token green.
        assert_eq!(
            parts("/help"),
            vec![("/help".to_owned(), Some(THEME_GREEN))]
        );
        // Known command with args: only the token is green, the rest plain.
        assert_eq!(
            parts("/btw what is this"),
            vec![
                ("/btw".to_owned(), Some(THEME_GREEN)),
                (" what is this".to_owned(), None),
            ]
        );
        assert_eq!(
            parts("/checkpoint before-refactor"),
            vec![
                ("/checkpoint".to_owned(), Some(THEME_GREEN)),
                (" before-refactor".to_owned(), None),
            ]
        );
    }

    #[test]
    fn input_spans_leaves_partial_or_unknown_slash_plain() {
        // Partial (not yet a full command) and unknown stay default-styled.
        assert_eq!(parts("/hel"), vec![("/hel".to_owned(), None)]);
        assert_eq!(parts("/nope"), vec![("/nope".to_owned(), None)]);
        // A no-arg command given args is not a valid invocation: no highlight.
        assert_eq!(parts("/help me"), vec![("/help me".to_owned(), None)]);
    }

    #[test]
    fn input_spans_colors_only_the_bang_red() {
        assert_eq!(
            parts("!ls -la"),
            vec![
                ("!".to_owned(), Some(Color::Red)),
                ("ls -la".to_owned(), None),
            ]
        );
        // A lone `!` still colors the marker.
        assert_eq!(parts("!"), vec![("!".to_owned(), Some(Color::Red))]);
    }

    #[test]
    fn input_spans_plain_text_is_unstyled() {
        assert_eq!(parts("hello world"), vec![("hello world".to_owned(), None)]);
    }

    #[test]
    fn input_height_is_one_row_per_line() {
        assert_eq!(input_height(""), 1);
        assert_eq!(input_height("one line"), 1);
        assert_eq!(input_height("two\nlines"), 2);
        // A trailing newline opens a new (empty) row to type on.
        assert_eq!(input_height("trailing\n"), 2);
    }

    #[test]
    fn input_scroll_holds_still_until_the_cursor_leaves_the_row() {
        // 20 columns, a 2-wide prompt: 18 usable, last is column 17.
        assert_eq!(input_scroll(0, 20, 2), 0);
        assert_eq!(input_scroll(17, 20, 2), 0);
        assert_eq!(input_scroll(18, 20, 2), 1);
        assert_eq!(input_scroll(30, 20, 2), 13);
    }

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
    fn verb_shimmer_sweeps_across_the_word() {
        let base = Style::default();
        let shimmer = Color::Indexed(crate::status::SHIMMER_COLOR);
        // Collect the shimmer segment text at each step of one full cycle.
        let text = "◆ Pondering… 3s";
        let mut highlights = Vec::new();
        for step in 0..40u64 {
            let line = status_bar_line(text, step * crate::status::SHIMMER_STEP_MS, base);
            let hit: String = line
                .spans
                .iter()
                .filter(|s| s.style.fg == Some(shimmer))
                .map(|s| s.content.as_ref())
                .collect();
            highlights.push(hit);
        }
        // The highlight moves: several distinct segments appear, including
        // off-text (empty) phases and at least one mid-word slice.
        highlights.sort_unstable();
        highlights.dedup();
        assert!(highlights.len() > 3, "static shimmer: {highlights:?}");
        assert!(highlights.iter().any(String::is_empty));
        assert!(highlights.iter().any(|h| h.contains("nde")));
    }

    #[test]
    fn user_echo_is_bold() {
        let spans = user_echo_spans("hi");
        assert!(spans[0].style.add_modifier.contains(Modifier::BOLD));
    }

    /// Row index of the input line (the prompt), found by its cyan prompt glyph.
    fn input_row(buf: &Buffer) -> Option<u16> {
        let prompt = crate::status::prompt_text();
        let head = prompt.chars().next()?;
        (0..buf.area.height).find(|&y| {
            let cell = &buf[(0, y)];
            cell.symbol().starts_with(head) && cell.style().fg == Some(Color::Cyan)
        })
    }

    #[test]
    fn draw_publishes_the_frame_regions_for_ui_remote() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let _guard = crate::uiremote::TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::uiremote::set_recording(true);
        let mut log = OutputLog::new();
        log.push_plain("some output");
        let mut view = OutputView::default();
        let mut term = Terminal::new(TestBackend::new(24, 8)).unwrap();
        term.draw(|f| draw(f, &log, Some("hi"), (0, 2), "idle", &mut view, None))
            .unwrap();
        let tree = crate::uiremote::frame_tree();
        crate::uiremote::set_recording(false);

        // One top-level region (`root`, the whole frame) with the rest nested
        // inside it, so the shape a harness sees is a single object.
        assert!(tree.starts_with(r#"{"name":"root""#), "{tree}");
        for name in ["output", "input", "status"] {
            assert!(tree.contains(&format!(r#""name":"{name}""#)), "{tree}");
        }
        assert!(tree.contains(r#""text":"hi""#), "{tree}");
    }

    #[test]
    fn status_bar_marks_an_active_remote_session() {
        let _guard = crate::uiremote::TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::uiremote::set_recording(false);
        assert_eq!(with_remote_marker("idle"), "idle");
        crate::uiremote::set_recording(true);
        assert_eq!(with_remote_marker("idle"), "idle | remote");
        crate::uiremote::set_recording(false);
    }

    #[test]
    fn green_rule_separates_output_from_the_visible_prompt() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut log = OutputLog::new();
        log.push_plain("some output");
        let mut view = OutputView::default();

        // Prompt visible: a green ─ rule sits on the row directly above input.
        let mut term = Terminal::new(TestBackend::new(24, 8)).unwrap();
        term.draw(|f| draw(f, &log, Some("hi"), (0, 2), "idle", &mut view, None))
            .unwrap();
        let buf = term.backend().buffer();
        let prompt_y = input_row(buf).expect("prompt row present");
        let rule_y = prompt_y - 1;
        let rule = &buf[(0, rule_y)];
        assert_eq!(rule.symbol(), "─");
        assert_eq!(rule.style().fg, Some(THEME_GREEN));

        // Prompt hidden (agent busy): no rule — the row above the (empty)
        // input line is ordinary output, never the green ─.
        let mut term = Terminal::new(TestBackend::new(24, 8)).unwrap();
        term.draw(|f| draw(f, &log, None, (0, 0), "generating", &mut view, None))
            .unwrap();
        let buf = term.backend().buffer();
        let has_rule = (0..buf.area.height).any(|y| {
            let c = &buf[(0, y)];
            c.symbol() == "─" && c.style().fg == Some(THEME_GREEN)
        });
        assert!(!has_rule, "no separator while the prompt is hidden");
    }

    #[test]
    fn multiline_input_renders_every_row_and_places_the_cursor() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let log = OutputLog::new();
        let mut view = OutputView::default();
        let mut term = Terminal::new(TestBackend::new(24, 8)).unwrap();
        // Cursor sits at the end of the second line ("bb").
        term.draw(|f| draw(f, &log, Some("aa\nbb"), (1, 2), "idle", &mut view, None))
            .unwrap();

        let buf = term.backend().buffer();
        let prompt_y = input_row(buf).expect("prompt row present");
        let row = |y: u16| -> String {
            (0..buf.area.width)
                .map(|x| buf[(x, y)].symbol())
                .collect::<String>()
        };
        assert!(
            row(prompt_y).contains("aa"),
            "first line: {}",
            row(prompt_y)
        );
        assert!(
            row(prompt_y + 1).contains("bb"),
            "second line: {}",
            row(prompt_y + 1)
        );
        // The status bar was pushed down to make room; the cursor is on row 2
        // of the input, indented past the prompt.
        let (cx, cy) = term.get_cursor_position().unwrap().into();
        assert_eq!(cy, prompt_y + 1);
        let prompt_width = u16::try_from(Span::raw(crate::status::prompt_text()).width()).unwrap();
        assert_eq!(cx, prompt_width + 2);
    }

    #[test]
    fn popup_sits_above_the_input_and_never_touches_the_status_bar() {
        // 24-row screen: output 0..20, input 21, status 23.
        let output = Rect::new(0, 0, 80, 20);
        let input = Rect::new(0, 21, 80, 1);
        let r = popup_rect(output, input, 5);
        assert_eq!(r.height, 5);
        assert_eq!(r.y + r.height, input.y, "bottom edge meets the input top");
        assert!(r.y >= output.y);
    }

    #[test]
    fn popup_shrinks_rather_than_moving_down_when_space_is_tight() {
        // Only 2 rows of output above a tall multi-line input.
        let output = Rect::new(0, 0, 80, 2);
        let input = Rect::new(0, 3, 80, 6);
        let r = popup_rect(output, input, 15);
        assert_eq!(r.height, 2, "clamped to the output pane");
        assert_eq!(r.y, output.y);
        assert!(r.y + r.height <= input.y);
    }

    #[test]
    fn popup_is_empty_when_no_rows_fit() {
        let output = Rect::new(0, 0, 80, 0);
        let input = Rect::new(0, 0, 80, 1);
        assert_eq!(popup_rect(output, input, 5).height, 0);
    }

    #[test]
    fn popup_geometry_matches_the_real_frame_layout() {
        // Drive popup_rect with the actual frame_geom split rather than
        // hand-made rects, for a one-row and a tall multi-row input.
        for (input_text, rows) in [("@src", 5u16), ("a\nb\nc\n@src", 15)] {
            let screen = Rect::new(0, 0, 80, 24);
            let (output, input, status, rule) = frame_geom(screen, true, input_height(input_text));
            let rule = rule.expect("prompt showing means a rule row");
            let r = popup_rect(output, input, rows);
            assert!(r.y >= output.y, "popup starts inside the output pane");
            assert!(
                r.y + r.height <= input.y,
                "popup never reaches the input line"
            );
            assert!(
                r.y + r.height <= status.y,
                "popup never touches the status bar"
            );
            // Deliberate: the bottom-anchored popup overlays the separator rule
            // (see popup_rect docs); it must never spill past it.
            assert!(r.y + r.height <= rule.y + 1);
            assert!(r.height > 0, "some rows fit on a 24-row screen");
        }
    }

    #[test]
    fn popup_never_exceeds_the_row_cap() {
        let output = Rect::new(0, 0, 80, 40);
        let input = Rect::new(0, 41, 80, 1);
        let r = popup_rect(
            output,
            input,
            u16::try_from(crate::complete::MAX_ROWS).unwrap(),
        );
        assert_eq!(r.height, 15);
    }
}
