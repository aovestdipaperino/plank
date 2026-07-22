// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

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
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use unicode_width::UnicodeWidthStr;

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
/// Header marker `ratatui-markdown` opens a fenced code block with (`╭─ lang`).
const CODE_HEADER_MARK: char = '╭';
/// Footer marker that closes a fenced code block (`╰─`).
const CODE_FOOTER_MARK: char = '╰';
/// The `│ ` gutter each code body line carries; stripped to recover the source.
const CODE_BODY_GUTTER: &str = "│ ";
/// Clickable control appended to a code block's header, next to the language.
const CODE_COPY_LABEL: &str = " ⧉ copy";

/// A rendered fenced code block: its logical `lines` range, the raw code it
/// holds (`│ ` gutter stripped, trailing whitespace trimmed, WYSIWYG), and the
/// inclusive screen columns of the header's `⧉ copy` control.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeBlockRegion {
    /// Logical index (into `lines`) of the `╭` header row.
    pub header: usize,
    /// Inclusive screen-column span of the header's copy control.
    pub copy_cols: (u16, u16),
    /// Block contents, one body line per row, ready for the clipboard.
    pub code: String,
}

/// Concatenates a line's span contents into its plain text.
fn line_text(line: &Line<'_>) -> String {
    line.spans.iter().map(|s| s.content.as_ref()).collect()
}

/// Appends the `⧉ copy` control to every code-block header in `lines[from..]`
/// and returns the regions found there (with absolute `lines` indices). A block
/// runs from a `╭` header to its `╰` footer; body rows contribute their text
/// with the `│ ` gutter stripped.
fn annotate_code_blocks(lines: &mut [Line<'static>], from: usize) -> Vec<CodeBlockRegion> {
    let mut regions = Vec::new();
    let mut i = from;
    while i < lines.len() {
        let header_text = line_text(&lines[i]);
        if !header_text.starts_with(CODE_HEADER_MARK) {
            i += 1;
            continue;
        }
        let header = i;
        let mut code_lines: Vec<String> = Vec::new();
        let mut j = i + 1;
        while j < lines.len() {
            let body = line_text(&lines[j]);
            if body.starts_with(CODE_FOOTER_MARK) {
                break;
            }
            let stripped = body.strip_prefix(CODE_BODY_GUTTER).unwrap_or(&body);
            code_lines.push(stripped.trim_end().to_owned());
            j += 1;
        }
        let start_col = u16::try_from(UnicodeWidthStr::width(header_text.as_str())).unwrap_or(0);
        let end_col = start_col
            .saturating_add(u16::try_from(UnicodeWidthStr::width(CODE_COPY_LABEL)).unwrap_or(0));
        lines[header].spans.push(Span::styled(
            CODE_COPY_LABEL.to_owned(),
            Style::default()
                .fg(Color::Indexed(245))
                .add_modifier(Modifier::DIM),
        ));
        regions.push(CodeBlockRegion {
            header,
            copy_cols: (start_col, end_col.saturating_sub(1)),
            code: code_lines.join("\n"),
        });
        // Resume past the footer (or at EOF for a still-streaming block).
        i = j + 1;
    }
    regions
}

#[derive(Debug, Default)]
pub struct OutputLog {
    lines: Vec<Line<'static>>,
    /// Rendered fenced code blocks, each carrying its raw text and the screen
    /// columns of its header's `⧉ copy` control, so a click on that control
    /// copies the block verbatim. Rebuilt alongside `lines` in `md_render`.
    code_blocks: Vec<CodeBlockRegion>,
    current: Vec<Span<'static>>,
    /// Raw markdown of the visible segment currently streaming, plus the
    /// index in `lines` where its rendered form starts. Re-rendered whole on
    /// each append so partial emphasis/fences resolve as more text arrives.
    md_buf: String,
    md_start: Option<usize>,
    /// Transient progress line pinned below the scrollback (throbber + verb +
    /// stats), shown while the worker runs so activity stays visible even when
    /// no text is streaming. Not part of the persistent `lines`; cleared when
    /// the turn ends.
    progress: Option<Line<'static>>,
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
        // Rebuild the code-block registry for the re-rendered segment; blocks
        // in committed earlier segments keep their (stable) indices.
        self.code_blocks.retain(|r| r.header < start);
        self.code_blocks
            .extend(annotate_code_blocks(&mut self.lines, start));
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
        self.code_blocks.retain(|r| r.header < len);
    }

    /// Sets (or clears) the transient progress line pinned below the output.
    pub fn set_progress(&mut self, line: Option<Line<'static>>) {
        self.progress = line;
    }

    /// Renders the log (including the in-progress line and any pinned progress
    /// line) as ratatui text.
    #[must_use]
    pub fn to_text(&self) -> Text<'static> {
        let mut lines = self.lines.clone();
        if !self.current.is_empty() {
            lines.push(Line::from(self.current.clone()));
        }
        if let Some(progress) = &self.progress {
            lines.push(progress.clone());
        }
        Text::from(lines)
    }

    /// Maps a click at output-area cell (`col`, `row`) — with the log scrolled
    /// so its first visible wrapped row is `top` and wrapped at `width` — to the
    /// raw text of a code block, when the click lands on that block's header
    /// `⧉ copy` control. `None` otherwise.
    #[must_use]
    pub fn code_copy_at(&self, width: u16, top: usize, col: u16, row: u16) -> Option<String> {
        if self.code_blocks.is_empty() {
            return None;
        }
        let width = width.max(1);
        let target = top.checked_add(row as usize)?;
        let mut acc = 0usize;
        for (idx, line) in self.lines.iter().enumerate() {
            // Each logical line wraps independently (Wrap { trim: false }), so
            // its screen height matches how `render_output` lays it out.
            let height = Paragraph::new(Text::from(line.clone()))
                .wrap(Wrap { trim: false })
                .line_count(width)
                .max(1);
            if target < acc + height {
                // The header sits on the block's first wrapped row.
                if target != acc {
                    return None;
                }
                return self
                    .code_blocks
                    .iter()
                    .find(|r| r.header == idx && col >= r.copy_cols.0 && col <= r.copy_cols.1)
                    .map(|r| r.code.clone());
            }
            acc += height;
        }
        None
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

/// A cheap, cloneable snapshot of the task list for rendering (issue #35): the
/// status-bar counter plus the strip rows. Sent worker→UI over
/// [`crate::worker::UiEvent::Tasks`] and passed straight into [`draw`], so
/// neither thread needs to reach into session state during a frame.
#[derive(Debug, Clone, Default)]
pub struct TaskView {
    completed: usize,
    total: usize,
    /// `(text, is_active)` rows for the contextual strip, already capped.
    rows: Vec<(String, bool)>,
}

impl From<&crate::tasks::TaskList> for TaskView {
    fn from(list: &crate::tasks::TaskList) -> Self {
        let (completed, total) = list.counter().unwrap_or((0, 0));
        Self {
            completed,
            total,
            rows: list.strip_rows(),
        }
    }
}

impl TaskView {
    /// `(completed, total)` for the status-bar counter, or `None` when empty.
    #[must_use]
    pub fn counter(&self) -> Option<(usize, usize)> {
        if self.total == 0 {
            None
        } else {
            Some((self.completed, self.total))
        }
    }

    /// True when the list is non-empty and fully completed (counter goes dim).
    #[must_use]
    pub fn all_done(&self) -> bool {
        self.total > 0 && self.completed == self.total
    }

    /// Strip rows above the separator rule, `(text, is_active)`, already capped
    /// at three by [`crate::tasks::TaskList::strip_rows`].
    #[must_use]
    pub fn strip_rows(&self) -> &[(String, bool)] {
        &self.rows
    }
}

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
    tasks: &TaskView,
) -> (Rect, Rect, Rect) {
    // The task strip (issue #35) sits directly above the rule; it appears only
    // at rest (with a prompt) and only when a task is in flight, capped at
    // three rows so it never crowds the scrollback.
    let strip = if has_prompt { tasks.strip_rows() } else { &[] };
    let strip_rows = u16::try_from(strip.len()).unwrap_or(0);
    let (output, input, status, rule_top, rule_bottom, strip_area) =
        frame_geom(area, has_prompt, input_rows, strip_rows);
    // Draw-site instrumentation for `--ui-remote`. This is the one place both
    // `draw` and `draw_btw_split` funnel through, so the frame is reset and
    // the structural regions published here; `render_input` and `render_popup`
    // append their own regions later in the same pass.
    crate::uiremote::begin_frame();
    if crate::uiremote::recording_enabled() {
        crate::uiremote::region("root", area, &[]);
        crate::uiremote::region("output", output, &[]);
        crate::uiremote::region("status", status, &[]);
    }
    if let Some(strip_area) = strip_area {
        render_task_strip(frame, strip_area, strip);
    }
    // Both rules bracket the resting prompt (above and below the input).
    for rule in [rule_top, rule_bottom].into_iter().flatten() {
        let text = "─".repeat(rule.width as usize);
        frame.render_widget(
            Paragraph::new(Span::styled(text, Style::default().fg(THEME_GREEN))),
            rule,
        );
    }
    (output, input, status)
}

/// Draws the contextual task strip: the active task in the theme green the rule
/// uses, pending tasks in the `Indexed(238)` gray thinking text uses.
fn render_task_strip(frame: &mut Frame, area: Rect, rows: &[(String, bool)]) {
    for (i, (text, is_active)) in rows.iter().enumerate() {
        let Some(y) = area.y.checked_add(u16::try_from(i).unwrap_or(u16::MAX)) else {
            break;
        };
        if y >= area.bottom() {
            break;
        }
        let style = if *is_active {
            Style::default().fg(THEME_GREEN)
        } else {
            Style::default().fg(Color::Indexed(238))
        };
        let marker = if *is_active { "▸ " } else { "  " };
        let row = Rect::new(area.x, y, area.width, 1);
        frame.render_widget(
            Paragraph::new(Span::styled(format!("{marker}{text}"), style)),
            row,
        );
    }
}

/// Pure geometry behind [`frame_rows`]: returns
/// `(output, input, status, rule_top, rule_bottom, strip)`. The rules bracket
/// the resting prompt (one above, one below) and are present only when
/// `has_prompt`; `strip` only when `strip_rows > 0`.
///
/// Split out so layout can be computed (and tested) without a `Frame`.
fn frame_geom(
    area: Rect,
    has_prompt: bool,
    input_rows: u16,
    strip_rows: u16,
) -> (Rect, Rect, Rect, Option<Rect>, Option<Rect>, Option<Rect>) {
    if has_prompt {
        let r = Layout::vertical([
            Constraint::Min(1),             // output
            Constraint::Length(strip_rows), // task strip (0 when idle-empty)
            Constraint::Length(1),          // top rule
            Constraint::Length(input_rows), // input
            Constraint::Length(1),          // bottom rule
            Constraint::Length(1),          // status
        ])
        .split(area);
        let strip = if strip_rows > 0 { Some(r[1]) } else { None };
        (r[0], r[3], r[5], Some(r[2]), Some(r[4]), strip)
    } else {
        let r = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);
        (r[0], r[1], r[2], None, None, None)
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
        .min(u16::try_from(crate::complete::max_rows()).unwrap_or(u16::MAX));
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
/// Trims `text` to `budget` display columns, dropping characters from the
/// *left* and marking the cut with `…`.
///
/// Completion rows are paths, so the informative end is the basename. Clipping
/// on the right (ratatui's default) hides exactly the part the user is reading
/// — see issue #42.
fn elide_left(text: &str, budget: usize) -> String {
    use unicode_width::UnicodeWidthStr;
    if text.width() <= budget {
        return text.to_string();
    }
    if budget <= 1 {
        return "…".repeat(budget);
    }
    // Take characters from the end until the ellipsis plus the tail fills the
    // budget; a wide character that would overflow simply stops the loop.
    let mut tail = String::new();
    for c in text.chars().rev() {
        let mut next = String::from(c);
        next.push_str(&tail);
        if next.width() + 1 > budget {
            break;
        }
        tail = next;
    }
    format!("…{tail}")
}

fn render_popup(frame: &mut Frame, area: Rect, popup: &crate::complete::Popup) {
    use ratatui::widgets::{Clear, List, ListItem, ListState, StatefulWidget};
    if area.height == 0 || popup.rows().is_empty() {
        return;
    }
    if crate::uiremote::recording_enabled() {
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
    }
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
            // The highlight symbol ("> ") and the kind marker plus its space
            // each eat two columns of every row.
            let budget = usize::from(area.width).saturating_sub(4);
            ListItem::new(Span::raw(format!(
                "{marker} {}",
                elide_left(&m.text, budget)
            )))
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
    let tw = input_text_width(frame.area().width);
    let (output, input, _, _, _, _) =
        frame_geom(frame.area(), true, input_height(input_text, tw), 0);
    let rows = u16::try_from(popup.rows().len()).unwrap_or(u16::MAX);
    render_popup(frame, popup_rect(output, input, rows), popup);
}

/// Display width of the prompt glyph (`🪵> `), the left indent shared by every
/// input row.
fn prompt_width() -> u16 {
    u16::try_from(UnicodeWidthStr::width(crate::status::prompt_text())).unwrap_or(0)
}

/// Columns available for the wrapped input text at the given frame width.
fn input_text_width(frame_width: u16) -> u16 {
    frame_width.saturating_sub(prompt_width()).max(1)
}

/// Display width of a char, treating control chars as zero.
fn char_width(c: char) -> usize {
    unicode_width::UnicodeWidthChar::width(c).unwrap_or(0)
}

/// Start offsets (in chars) of each visual segment when wrapping `chars` at
/// `width` cells: a word wrap that breaks at the last space before an overflow,
/// or hard-breaks a token too long to fit. Always starts with `0`.
fn wrap_offsets(chars: &[char], width: usize) -> Vec<usize> {
    let width = width.max(1);
    let mut starts = vec![0usize];
    let mut seg_start = 0usize;
    let mut col = 0usize;
    let mut last_space: Option<usize> = None;
    let mut i = 0usize;
    while i < chars.len() {
        let w = char_width(chars[i]);
        if col + w > width && i > seg_start {
            let brk = match last_space {
                Some(s) if s + 1 > seg_start => s + 1,
                _ => i,
            };
            starts.push(brk);
            seg_start = brk;
            col = chars[seg_start..i].iter().copied().map(char_width).sum();
            last_space = (seg_start..i).rev().find(|&k| chars[k] == ' ');
            continue;
        }
        if chars[i] == ' ' {
            last_space = Some(i);
        }
        col += w;
        i += 1;
    }
    starts
}

/// Number of visual rows the prompt needs for `input` wrapped at `width` cells
/// — one per wrapped segment across all logical (newline-separated) lines.
#[must_use]
pub fn input_height(input: &str, width: u16) -> u16 {
    let width = width as usize;
    let rows: usize = input
        .split('\n')
        .map(|line| wrap_offsets(&line.chars().collect::<Vec<_>>(), width).len())
        .sum();
    u16::try_from(rows.max(1)).unwrap_or(u16::MAX)
}

/// Coalesces styled cells into spans, merging runs of the same style.
fn cells_to_spans(cells: &[(char, Style)]) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut run = String::new();
    let mut run_style: Option<Style> = None;
    for &(c, st) in cells {
        if run_style == Some(st) {
            run.push(c);
        } else {
            if let Some(s) = run_style {
                spans.push(Span::styled(std::mem::take(&mut run), s));
            }
            run.push(c);
            run_style = Some(st);
        }
    }
    if let (Some(s), false) = (run_style, run.is_empty()) {
        spans.push(Span::styled(run, s));
    }
    spans
}

/// Word-wraps `input` into styled visual rows and locates the cursor's visual
/// `(row, col)` for the char index `cursor_char`. Line 0 keeps its
/// command/`!` highlighting; continuation lines render plain.
fn wrap_input(input: &str, width: u16, cursor_char: usize) -> (Vec<Line<'static>>, u16, u16) {
    let width = (width as usize).max(1);
    let mut lines: Vec<Line<'static>> = Vec::new();
    let (mut cur_row, mut cur_col) = (0u16, 0u16);
    let mut base = 0usize; // input char index at the start of the logical line
    for (li, logical) in input.split('\n').enumerate() {
        let styled: Vec<(char, Style)> = if li == 0 {
            input_spans(logical)
                .into_iter()
                .flat_map(|s| {
                    let st = s.style;
                    s.content.chars().map(move |c| (c, st)).collect::<Vec<_>>()
                })
                .collect()
        } else {
            logical.chars().map(|c| (c, Style::default())).collect()
        };
        let chars: Vec<char> = styled.iter().map(|&(c, _)| c).collect();
        let offsets = wrap_offsets(&chars, width);
        let len = chars.len();
        for (si, &start) in offsets.iter().enumerate() {
            let end = offsets.get(si + 1).copied().unwrap_or(len);
            let is_last = si + 1 == offsets.len();
            // The cursor sits in this segment when its index falls within
            // [start, end) — or exactly at `end` on the final segment (line end).
            if cursor_char >= base + start
                && (cursor_char < base + end || (is_last && cursor_char <= base + end))
            {
                cur_row = u16::try_from(lines.len()).unwrap_or(u16::MAX);
                let off = cursor_char - (base + start);
                let w: usize = chars[start..start + off]
                    .iter()
                    .copied()
                    .map(char_width)
                    .sum();
                cur_col = u16::try_from(w).unwrap_or(u16::MAX);
            }
            lines.push(Line::from(cells_to_spans(&styled[start..end])));
        }
        base += len + 1; // +1 for the consumed newline
    }
    (lines, cur_row, cur_col)
}

/// Draws the prompt glyph and the word-wrapped input text into `input_area`,
/// placing the terminal cursor for the char index `cursor_char`.
///
/// The text is indented under the prompt and wraps to the next row instead of
/// scrolling horizontally.
fn render_input(frame: &mut Frame, input_area: Rect, input: &str, cursor_char: usize) {
    // The input region carries its text, so a harness can assert on what is
    // typed without decoding the ANSI snapshot. It is registered here rather
    // than in `frame_rows` because only this function sees the text; while the
    // agent is busy no prompt is drawn and no `input` region appears.
    if crate::uiremote::recording_enabled() {
        crate::uiremote::region(
            "input",
            input_area,
            &[("text", crate::tools::mcp::Json::Str(input.to_string()))],
        );
    }
    let prompt_span = Span::styled(
        crate::status::prompt_text(),
        Style::default().fg(Color::Cyan),
    );
    let pw = prompt_width();
    frame.render_widget(
        Paragraph::new(Line::from(vec![prompt_span])),
        Rect {
            height: 1,
            ..input_area
        },
    );

    let text_area = Rect {
        x: input_area.x + pw,
        y: input_area.y,
        width: input_area.width.saturating_sub(pw),
        height: input_area.height,
    };
    let (lines, cur_row, cur_col) = wrap_input(input, text_area.width, cursor_char);
    frame.render_widget(Paragraph::new(lines), text_area);

    let cursor = Position::new(
        (text_area.x + cur_col).min(input_area.right().saturating_sub(1)),
        input_area.y + cur_row.min(input_area.height.saturating_sub(1)),
    );
    frame.set_cursor_position(cursor);
    // ratatui 0.29 keeps `Frame::cursor_position` private with no getter, so
    // the snapshot's cursor field is recorded here, at the one site that sets
    // it, rather than read back off the frame.
    if crate::uiremote::recording_enabled() {
        crate::uiremote::set_cursor(cursor.x, cursor.y);
    }
}

/// Renders a git-style diff card for a changed file into the output log: a
/// bold `Update(path)` / `Create(path)` header, an added/removed summary, then
/// `@@` hunks with red-background removals and green-background additions.
pub fn render_diff_card(log: &mut OutputLog, p: &crate::tools::diff::EditPreview) {
    use crate::tools::diff::{DiffRow, gutter, human_size, plural};
    let verb = if p.created { "Create" } else { "Update" };
    let mut head = vec![
        Span::styled("● ", Style::default().fg(THEME_GREEN)),
        Span::styled(
            format!("{verb}({})", p.path),
            Style::default().add_modifier(Modifier::BOLD),
        ),
    ];
    if let Some(bytes) = p.bytes {
        head.push(Span::styled(
            format!(" · {}", human_size(bytes)),
            Style::default().fg(Color::Indexed(240)),
        ));
    }
    log.push_spans(head);

    let dim = Style::default().fg(Color::Indexed(240));
    log.push_spans(vec![Span::styled(
        format!(
            "  └ Added {} {}, removed {} {}",
            p.added,
            plural(p.added),
            p.removed,
            plural(p.removed)
        ),
        dim,
    )]);

    let del = Style::default()
        .bg(Color::Indexed(52))
        .fg(Color::Indexed(224));
    let add = Style::default()
        .bg(Color::Indexed(22))
        .fg(Color::Indexed(194));
    for row in &p.rows {
        match row {
            DiffRow::Hunk {
                old_start,
                old_len,
                new_start,
                new_len,
            } => log.push_spans(vec![Span::styled(
                format!("  @@ -{old_start},{old_len} +{new_start},{new_len} @@"),
                Style::default().fg(Color::Indexed(44)),
            )]),
            DiffRow::Context { text, .. } => log.push_spans(vec![
                Span::styled(format!("{}   ", gutter(row.gutter())), dim),
                Span::raw(text.clone()),
            ]),
            DiffRow::Del { text, .. } => log.push_spans(vec![Span::styled(
                format!("{} - {text}", gutter(row.gutter())),
                del,
            )]),
            DiffRow::Add { text, .. } => log.push_spans(vec![Span::styled(
                format!("{} + {text}", gutter(row.gutter())),
                add,
            )]),
            DiffRow::Elision(n) => {
                log.push_spans(vec![Span::styled(format!("      ⋯ {n} more lines ⋯"), dim)]);
            }
        }
    }
    log.push_spans(vec![]);
}

/// Minimal pre-UI screen shown while the system-prompt KV cache is (re)built at
/// launch: a centered note and a simple progress bar. The full UI is withheld
/// until warming finishes, so the user sees clear progress instead of an idle
/// screen during the one slow step.
pub fn draw_warm(frame: &mut Frame, done: i32, total: i32, tps: f64) {
    let total = total.max(1);
    let done = done.clamp(0, total);
    let pct = u16::try_from(i64::from(done) * 100 / i64::from(total)).unwrap_or(100);
    let bar = crate::status::progress_bar(done, total, tps, false);
    let area = frame.area();
    let rows = Layout::vertical([
        Constraint::Percentage(45),
        Constraint::Length(2),
        Constraint::Min(0),
    ])
    .split(area);
    let text = Text::from(vec![
        Line::from(Span::styled(
            "Updating system prompt cache…",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ))
        .centered(),
        Line::from(format!("{bar}  {pct}%")).centered(),
    ]);
    frame.render_widget(Paragraph::new(text), rows[1]);
}

/// Draws the interactive `/config` editor as a centered modal overlay.
///
/// Rows come from [`crate::configform::ConfigForm::rows`]: section headers are
/// dimmed, the selected field is reversed, and a field being edited shows its
/// live buffer with a caret. A footer carries the key hints and any error.
pub fn draw_config(frame: &mut Frame, form: &crate::configform::ConfigForm) {
    let rows = form.rows();
    let mut lines: Vec<Line<'static>> = Vec::with_capacity(rows.len() + 2);
    for row in &rows {
        if row.header {
            lines.push(Line::from(Span::styled(
                format!("[{}]", row.label),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )));
            continue;
        }
        let marker = if row.selected { "▸ " } else { "  " };
        let value = if row.editing {
            format!("{}▏", row.value)
        } else {
            row.value.clone()
        };
        let label = format!("{marker}{:<24} {}", row.label, value);
        let style = if row.selected {
            Style::default().add_modifier(Modifier::REVERSED)
        } else {
            Style::default()
        };
        lines.push(Line::from(Span::styled(label, style)));
    }
    lines.push(Line::from(""));
    let footer = match form.status() {
        Some(err) => Span::styled(
            format!("  ! {err}"),
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ),
        None if form.editing() => Span::styled(
            "  type value · ⏎ commit · Esc cancel edit",
            Style::default().fg(Color::DarkGray),
        ),
        None => Span::styled(
            "  ↑↓ move · ⏎/Space edit·toggle · Esc save & close · q cancel",
            Style::default().fg(Color::DarkGray),
        ),
    };
    lines.push(Line::from(footer));

    let area = frame.area();
    let width = 66.min(area.width.saturating_sub(4)).max(20);
    let height = u16::try_from(lines.len() + 2)
        .unwrap_or(u16::MAX)
        .min(area.height.saturating_sub(2))
        .max(3);
    let x = area.x + (area.width.saturating_sub(width)) / 2;
    let y = area.y + (area.height.saturating_sub(height)) / 2;
    let rect = Rect {
        x,
        y,
        width,
        height,
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" config → ./.plank/settings.json ")
        .title_style(Style::default().add_modifier(Modifier::BOLD));
    frame.render_widget(Clear, rect);
    frame.render_widget(Paragraph::new(Text::from(lines)).block(block), rect);
}

/// Draws one frame: output log, input line, and status bar.
///
/// `input` is the current prompt text and `cursor` its `(row, col)` position.
/// `input` is `None` while the agent is busy (prefill/generation): the prompt
/// line renders empty and the cursor stays hidden until input is accepted again.
/// `view` is the scroll state; it is clamped in place to the scrollable range
/// and a jump-to-bottom hint is shown while it is pinned above the bottom.
#[allow(clippy::too_many_arguments)]
pub fn draw(
    frame: &mut Frame,
    log: &OutputLog,
    input: Option<&str>,
    cursor: usize,
    status: &str,
    view: &mut OutputView,
    selection: Option<Selection>,
    tasks: &TaskView,
) {
    let area = frame.area();
    let tw = input_text_width(area.width);
    let (output, input_row, status_row) = frame_rows(
        frame,
        area,
        input.is_some(),
        input.map_or(1, |t| input_height(t, tw)),
        tasks,
    );

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
            tasks,
        ))
        .style(status_style),
        status_row,
    );
}

/// Draws one frame while an `ask` question (issue #34) is up: the output log
/// on top, the interactive question panel in the input region, and the status
/// bar below it. The panel is sized from the option count so it never overlaps
/// the status bar, and it coexists with the same layout the resting prompt uses.
pub fn draw_ask(
    frame: &mut Frame,
    log: &OutputLog,
    req: &crate::tools::ask::AskRequest,
    state: &crate::tools::ask::AskState,
    status: &str,
    view: &mut OutputView,
    tasks: &TaskView,
) {
    let area = frame.area();
    let panel_rows = crate::tools::ask::panel_rows(req.options.len())
        // Never let the panel eat the whole screen: leave at least one output row.
        .min(area.height.saturating_sub(2));
    let r = Layout::vertical([
        Constraint::Min(1),             // output
        Constraint::Length(panel_rows), // question panel
        Constraint::Length(1),          // status
    ])
    .split(area);
    render_output(frame, r[0], log, view, None);
    render_ask_panel(frame, r[1], req, state);

    let status_style = Style::default()
        .bg(Color::Indexed(238))
        .fg(Color::Indexed(252));
    frame.render_widget(
        Paragraph::new(status_bar_line(
            &with_remote_marker(status),
            anim_tick_ms(),
            status_style,
            tasks,
        ))
        .style(status_style),
        r[2],
    );
}

/// Renders the question panel: a header chip and question, then the options as a
/// selectable list (arrow keys move, Enter accepts, Space toggles in multi
/// mode), and a key-hint footer. The highlighted row is reverse-styled; ticked
/// multi-select rows carry a checkbox.
fn render_ask_panel(
    frame: &mut Frame,
    area: Rect,
    req: &crate::tools::ask::AskRequest,
    state: &crate::tools::ask::AskState,
) {
    let mut lines: Vec<Line<'static>> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled(
            format!(" {} ", req.header),
            Style::default()
                .bg(THEME_GREEN)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            req.question.clone(),
            Style::default().add_modifier(Modifier::BOLD),
        ),
    ]));
    lines.push(Line::raw(String::new()));
    for (i, opt) in req.options.iter().enumerate() {
        let is_cursor = i == state.cursor;
        let ticked = state.selected.get(i).copied().unwrap_or(false);
        let marker = if req.multi {
            if ticked { "[x] " } else { "[ ] " }
        } else if is_cursor {
            "> "
        } else {
            "  "
        };
        let mut style = Style::default();
        if is_cursor {
            style = style.fg(THEME_GREEN).add_modifier(Modifier::BOLD);
        }
        let mut spans = vec![Span::styled(format!("{marker}{}", opt.label), style)];
        if !opt.description.is_empty() {
            spans.push(Span::styled(
                format!("  — {}", opt.description),
                Style::default().fg(Color::Indexed(245)),
            ));
        }
        lines.push(Line::from(spans));
    }
    let hint = if req.multi {
        "↑/↓ move · Space toggle · Enter accept · Esc decline"
    } else {
        "↑/↓ move · Enter accept · Esc decline"
    };
    lines.push(Line::from(Span::styled(
        hint.to_string(),
        Style::default().fg(Color::Indexed(238)),
    )));
    frame.render_widget(Paragraph::new(lines), area);
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
    let width = area.width.max(1);
    let para = Paragraph::new(text).wrap(Wrap { trim: false });
    // Exact wrapped-line count from ratatui itself: a char-packing estimate
    // undercounts word-wrapped rows, leaving the view unable to reach the
    // bottom (e.g. the long `/context` report).
    let total = para.line_count(width);
    let max_top = total.saturating_sub(area.height as usize);
    if view.follow || view.top >= max_top {
        view.top = max_top;
        view.follow = true;
    }
    let scroll = u16::try_from(view.top).unwrap_or(u16::MAX);
    let para = para.scroll((scroll, 0));
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
    cursor: usize,
    status: &str,
    view: &mut OutputView,
    tasks: &TaskView,
) {
    use ratatui::widgets::{Block, Borders};

    let area = frame.area();
    let tw = input_text_width(area.width);
    let (output, input_row, status_row) = frame_rows(
        frame,
        area,
        input.is_some(),
        input.map_or(1, |t| input_height(t, tw)),
        tasks,
    );
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
            tasks,
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
/// Themes the leading directory segment of the status line. `prefix` looks like
/// `"<path> | "` or `"<path> ⎇ <branch> | "`; the path and branch render in the
/// theme green while the powerline glyph, spaces, and `|` separators stay plain.
fn push_dir_prefix(spans: &mut Vec<Span<'static>>, prefix: &str, base: Style, theme: Style) {
    let glyph = crate::status::POWERLINE_BRANCH;
    // Trailing " | " separator that hands off to the "ctx …" body.
    let (segment, sep) = prefix
        .rfind(" | ")
        .map_or((prefix, ""), |i| (&prefix[..i], &prefix[i..]));
    if let Some(gi) = segment.find(glyph) {
        let path = segment[..gi].trim_end();
        let branch = segment[gi + glyph.len_utf8()..].trim();
        spans.push(Span::styled(path.to_string(), theme));
        spans.push(Span::styled(format!(" {glyph} "), base));
        spans.push(Span::styled(branch.to_string(), theme));
    } else {
        spans.push(Span::styled(segment.trim_end().to_string(), theme));
    }
    spans.push(Span::styled(sep.to_string(), base));
}

/// Styles the plain progress text (`⠹ Verb… (stats)`) as a standalone output
/// line: the spinner verb shimmers in the theme green, the rest stays default.
/// Used to render the progress on a line below the output.
#[must_use]
pub fn progress_line(text: &str) -> Line<'static> {
    let base = Style::default();
    let theme = base.fg(THEME_GREEN).add_modifier(Modifier::BOLD);
    let mut spans = Vec::new();
    push_accented(&mut spans, text, anim_tick_ms(), base, theme);
    Line::from(spans)
}

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
fn status_bar_line(text: &str, tick_ms: u64, base: Style, tasks: &TaskView) -> Line<'static> {
    let theme = base
        .fg(Color::Indexed(crate::status::THEME_COLOR))
        .add_modifier(Modifier::BOLD);
    let mut spans = Vec::new();
    // Peel the leading "<path> ⎇ <branch> | " directory segment and theme the
    // path and branch green; the powerline glyph and separators stay plain.
    let text = if let Some(idx) = text.find("ctx ").filter(|&i| i > 0) {
        push_dir_prefix(&mut spans, &text[..idx], base, theme);
        &text[idx..]
    } else {
        text
    };
    let bar = text
        .find('[')
        .and_then(|open| text[open..].find(']').map(|i| (open, open + i)));
    if let Some((open, close)) = bar {
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
    } else {
        push_accented(&mut spans, text, tick_ms, base, theme);
    }
    // Task counter (issue #35): appended to the bracketed status region, themed
    // green while work is in flight and dim gray once the list is complete. An
    // empty list adds nothing.
    if let Some((done, total)) = tasks.counter() {
        let counter_style = if tasks.all_done() {
            base.fg(Color::Indexed(240))
        } else {
            theme
        };
        spans.push(Span::styled(" | ".to_string(), base));
        spans.push(Span::styled(format!("✓ {done}/{total}"), counter_style));
    }
    // Rotating yellow tip at the tail. It changes every few seconds off the
    // animation clock; on a narrow terminal the line truncates and drops it.
    let tip = crate::status::rotating_tip(tick_ms);
    if !tip.is_empty() {
        spans.push(Span::styled(" | ".to_string(), base));
        spans.push(Span::styled(
            format!("💡 {tip}"),
            base.fg(Color::Yellow).add_modifier(Modifier::BOLD),
        ));
    }
    Line::from(spans)
}

#[cfg(test)]
mod tests {
    use unicode_width::UnicodeWidthStr;

    #[test]
    fn elide_left_keeps_the_basename_visible() {
        let long = "deeply/nested/directory/structure/for/testing/a-very-long-filename.txt";
        let out = super::elide_left(long, 20);
        assert!(out.starts_with('…'), "{out:?}");
        assert!(out.ends_with("filename.txt"), "{out:?}");
        assert!(out.width() <= 20, "{out:?} is {} wide", out.width());
    }

    #[test]
    fn elide_left_leaves_a_fitting_path_alone() {
        assert_eq!(super::elide_left("src/ui.rs", 20), "src/ui.rs");
        assert_eq!(super::elide_left("src/ui.rs", 9), "src/ui.rs");
    }

    #[test]
    fn elide_left_never_exceeds_the_budget_with_wide_characters() {
        // A wide character that cannot fit beside the ellipsis must be
        // dropped, not half-drawn.
        for budget in 0..12 {
            let out = super::elide_left("世界世界世界", budget);
            assert!(
                out.width() <= budget,
                "budget {budget}: {out:?} is {} wide",
                out.width()
            );
        }
    }

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
    fn input_height_counts_newlines_and_wrapped_rows() {
        // Wide enough to never wrap: one row per logical line.
        assert_eq!(input_height("", 80), 1);
        assert_eq!(input_height("one line", 80), 1);
        assert_eq!(input_height("two\nlines", 80), 2);
        // A trailing newline opens a new (empty) row to type on.
        assert_eq!(input_height("trailing\n", 80), 2);
        // Narrow width wraps a long line onto extra rows.
        assert_eq!(input_height("abcdefghij", 4), 3); // 4+4+2
        assert_eq!(input_height("aaaa\nbbbbbb", 5), 3); // 1 + 2
    }

    #[test]
    fn word_wrap_breaks_at_spaces_and_maps_the_cursor() {
        // "hello world" at width 8 wraps after "hello " → "hello ", "world".
        let (lines, row, col) = wrap_input("hello world", 8, 11);
        assert_eq!(lines.len(), 2);
        let row0: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(row0, "hello ");
        // Cursor at end (char 11) lands on the second row after "world".
        assert_eq!((row, col), (1, 5));
    }

    #[test]
    fn word_wrap_hard_breaks_a_too_long_token() {
        // No spaces: a hard break at the width boundary.
        let (lines, _, _) = wrap_input("abcdefgh", 4, 0);
        let texts: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert_eq!(texts, vec!["abcd".to_string(), "efgh".to_string()]);
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
    fn progress_line_is_pinned_below_output_and_clears() {
        let mut log = OutputLog::new();
        log.visible_text("answer");
        log.end_line();
        let base = log.to_text().lines.len();

        // Pinned progress adds one trailing line without touching scrollback.
        log.set_progress(Some(super::progress_line(
            "⠹ Cooking… (2s · ↓ 5 tokens · 4.0 t/s)",
        )));
        let with = log.to_text();
        assert_eq!(with.lines.len(), base + 1);
        let last = with.lines.last().unwrap();
        let text: String = last.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains("Cooking…"), "{text}");

        // Clearing removes it again.
        log.set_progress(None);
        assert_eq!(log.to_text().lines.len(), base);
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
    fn code_block_records_region_and_copy_control() {
        let mut log = OutputLog::new();
        log.visible_text("```rust\nfn main() {}\nlet x = 1;\n```\n");
        assert_eq!(log.code_blocks.len(), 1, "one block recorded");
        let region = &log.code_blocks[0];
        // The raw code round-trips with the `│ ` gutter stripped.
        assert_eq!(region.code, "fn main() {}\nlet x = 1;");
        // The header carries the `⧉ copy` control after the language label.
        let header = line_text(&log.lines[region.header]);
        assert!(header.starts_with("╭"), "header: {header:?}");
        assert!(header.contains("rust"), "header: {header:?}");
        assert!(header.contains("copy"), "header: {header:?}");
    }

    #[test]
    fn code_copy_at_hits_control_and_misses_elsewhere() {
        let mut log = OutputLog::new();
        log.visible_text("```rust\nfn main() {}\n```\n");
        let region = log.code_blocks[0].clone();
        let (c0, c1) = region.copy_cols;

        // A click inside the control's columns on the header row copies it.
        assert_eq!(
            log.code_copy_at(80, 0, c0, 0).as_deref(),
            Some("fn main() {}")
        );
        assert_eq!(
            log.code_copy_at(80, 0, c1, 0).as_deref(),
            Some("fn main() {}")
        );
        // The language label (column 0) is not the control.
        assert_eq!(log.code_copy_at(80, 0, 0, 0), None);
        // Just past the control is a miss.
        assert_eq!(log.code_copy_at(80, 0, c1 + 1, 0), None);
        // A body row is not the header.
        assert_eq!(log.code_copy_at(80, 0, c0, 1), None);
    }

    #[test]
    fn code_copy_at_respects_scroll_offset() {
        let mut log = OutputLog::new();
        // Push a plain line, then a code block; scrolling shifts the header up.
        log.visible_text("intro line\n\n```sh\necho hi\n```\n");
        let region = log.code_blocks[0].clone();
        let header_row = region.header;
        let (c0, _) = region.copy_cols;
        // With the header scrolled to the top visible row, the click lands.
        assert_eq!(
            log.code_copy_at(80, header_row, c0, 0).as_deref(),
            Some("echo hi")
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

    fn task_view_with(entries: &[(&str, crate::tasks::TaskStatus)]) -> TaskView {
        let mut list = crate::tasks::TaskList::new();
        for (subject, status) in entries {
            let id = list.add(*subject, None);
            list.update(id, Some(*status), None, None).unwrap();
        }
        TaskView::from(&list)
    }

    #[test]
    fn status_bar_counter_is_themed_in_flight_and_dim_when_done() {
        use crate::tasks::TaskStatus::{Completed, InProgress};
        let base = Style::default();
        // An empty list adds no task counter to the status bar (the "✓ n/n"
        // segment); rotating tips may still contribute other spans.
        let empty = status_bar_line("idle", 0, base, &TaskView::default());
        assert!(!empty.spans.iter().any(|s| s.content.contains('✓')));

        // In flight: the counter carries the theme color.
        let theme = Color::Indexed(crate::status::THEME_COLOR);
        let tv = task_view_with(&[("a", Completed), ("b", InProgress)]);
        let line = status_bar_line("idle", 0, base, &tv);
        let counter = line
            .spans
            .iter()
            .find(|s| s.content.contains("1/2"))
            .expect("counter span present");
        assert_eq!(counter.style.fg, Some(theme));

        // Fully complete: the counter goes dim gray, not theme.
        let tv = task_view_with(&[("a", Completed)]);
        let line = status_bar_line("idle", 0, base, &tv);
        let counter = line
            .spans
            .iter()
            .find(|s| s.content.contains("1/1"))
            .unwrap();
        assert_eq!(counter.style.fg, Some(Color::Indexed(240)));
    }

    #[test]
    fn status_bar_themes_path_and_branch_but_not_the_powerline_glyph() {
        let base = Style::default();
        let theme = Color::Indexed(crate::status::THEME_COLOR);
        let glyph = crate::status::POWERLINE_BRANCH;
        let text = format!("~/Code/plank {glyph} main | ctx 12% | idle");
        let line = status_bar_line(&text, 0, base, &TaskView::default());

        let path = line
            .spans
            .iter()
            .find(|s| s.content == "~/Code/plank")
            .expect("path span");
        assert_eq!(path.style.fg, Some(theme));

        let branch = line
            .spans
            .iter()
            .find(|s| s.content == "main")
            .expect("branch span");
        assert_eq!(branch.style.fg, Some(theme));

        // The powerline glyph is not themed green.
        let glyph_span = line
            .spans
            .iter()
            .find(|s| s.content.contains(glyph))
            .expect("glyph span");
        assert_ne!(glyph_span.style.fg, Some(theme));
    }

    #[test]
    fn frame_geom_reserves_strip_rows_only_when_present() {
        let area = Rect::new(0, 0, 80, 24);
        // No strip: the top rule sits directly above the input, the bottom
        // rule directly below it (above the status bar).
        let (out0, in0, st0, rule0, rule_bot0, strip0) = frame_geom(area, true, 1, 0);
        assert!(strip0.is_none());
        let rule0 = rule0.unwrap();
        let rule_bot0 = rule_bot0.expect("bottom rule present");
        assert_eq!(rule0.y + 1, in0.y, "top rule directly above input");
        assert_eq!(
            in0.bottom(),
            rule_bot0.y,
            "bottom rule directly below input"
        );
        assert_eq!(
            rule_bot0.bottom(),
            st0.y,
            "bottom rule directly above status"
        );
        // Three strip rows: reserved between the output and the rule, and the
        // output pane shrinks by exactly three rows.
        let (out3, _in3, _st3, rule3, _rule_bot3, strip3) = frame_geom(area, true, 1, 3);
        let strip3 = strip3.expect("strip present");
        let rule3 = rule3.expect("rule present");
        assert_eq!(strip3.height, 3);
        assert_eq!(out3.height + 3, out0.height);
        assert_eq!(strip3.y, out3.bottom());
        assert_eq!(rule3.y, strip3.bottom());
        // The rule/input/status band is fixed at the bottom; the strip is
        // absorbed by shrinking the output, so the rule row does not move.
        assert_eq!(rule3.y, rule0.y);
    }

    #[test]
    fn verb_shimmer_sweeps_across_the_word() {
        let base = Style::default();
        let shimmer = Color::Indexed(crate::status::SHIMMER_COLOR);
        // Collect the shimmer segment text at each step of one full cycle.
        let text = "◆ Pondering… 3s";
        let mut highlights = Vec::new();
        for step in 0..40u64 {
            let line = status_bar_line(
                text,
                step * crate::status::SHIMMER_STEP_MS,
                base,
                &TaskView::default(),
            );
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
        term.draw(|f| {
            draw(
                f,
                &log,
                Some("hi"),
                2,
                "idle",
                &mut view,
                None,
                &TaskView::default(),
            );
        })
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
        term.draw(|f| {
            draw(
                f,
                &log,
                Some("hi"),
                2,
                "idle",
                &mut view,
                None,
                &TaskView::default(),
            );
        })
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
        term.draw(|f| {
            draw(
                f,
                &log,
                None,
                0,
                "generating",
                &mut view,
                None,
                &TaskView::default(),
            );
        })
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
        term.draw(|f| {
            draw(
                f,
                &log,
                Some("aa\nbb"),
                5,
                "idle",
                &mut view,
                None,
                &TaskView::default(),
            );
        })
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
            let (output, input, status, rule, _rule_bot, _strip) =
                frame_geom(screen, true, input_height(input_text, 78), 0);
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
            u16::try_from(crate::complete::max_rows()).unwrap(),
        );
        assert_eq!(r.height, 15);
    }
}
