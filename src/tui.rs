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
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use ratatui_markdown::ThemeConfig;
use ratatui_markdown::highlight::{HighlightHooks, TreeSitterHighlighter};
use ratatui_markdown::markdown::{MarkdownBlock, MarkdownRenderer};

use crate::viz::RenderSink;

/// Minimum wall-clock gap between markdown re-renders of a streaming segment.
/// Bounds highlighting cost to a fixed cadence regardless of token rate (see
/// [`OutputLog::md_render_throttled`]); ~10 renders/sec keeps live syntax
/// highlighting smooth without re-highlighting every token.
const MD_RENDER_MIN_GAP: Duration = Duration::from_millis(100);

/// Style for ordinary assistant/visible output.
fn visible_style() -> Style {
    Style::default()
}

/// Barely-visible gray, italic, for thinking text.
/// Theme green, for the note that a `!` command finished.
#[must_use]
pub fn done_style() -> Style {
    Style::default().fg(THEME_GREEN)
}

/// Bold grey, for the turn's closing "Planked for …" line: present enough to
/// find when scrolling back to a turn boundary, quiet enough not to compete
/// with the model's own output.
#[must_use]
pub fn turn_footer_style() -> Style {
    Style::default()
        .fg(Color::Indexed(245))
        .add_modifier(Modifier::BOLD)
}

/// Formats a turn's wall-clock duration as `Xh YYm ZZs`.
///
/// Every unit is always shown, hours unpadded and the rest to two digits, so
/// the line has one shape and consecutive turns align when scrolling back.
#[must_use]
pub fn fmt_turn_duration(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    format!(
        "{}h {:02}m {:02}s",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

/// The turn's closing line: a marker, and how long the turn took.
#[must_use]
pub fn turn_footer(d: std::time::Duration) -> String {
    format!("\u{273b} Planked for {}", fmt_turn_duration(d))
}

fn think_style() -> Style {
    Style::default()
        .fg(Color::Indexed(238))
        .add_modifier(Modifier::ITALIC)
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
/// runs from a `╭` header to its `╰` footer. Each region's copied text is the
/// verbatim source from `raw_codes` (the Nth `╭` header pairs with the Nth
/// entry, both in document order); if that pairing is unavailable it falls back
/// to the rendered body rows with the `│ ` gutter stripped — which may include
/// soft-wrap breaks, so it is a last resort only.
fn annotate_code_blocks(
    lines: &mut [Line<'static>],
    from: usize,
    raw_codes: &[&str],
) -> Vec<CodeBlockRegion> {
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
        let code = raw_codes.get(regions.len()).map_or_else(
            || code_lines.join("\n"),
            |raw| raw.trim_end_matches('\n').to_owned(),
        );
        regions.push(CodeBlockRegion {
            header,
            copy_cols: (start_col, end_col.saturating_sub(1)),
            code,
        });
        // Resume past the footer (or at EOF for a still-streaming block).
        i = j + 1;
    }
    regions
}

/// The sub-agent output pane: a second [`OutputLog`] holding the most recent
/// sub-agent run, plus its own scroll state so switching panes does not disturb
/// the main transcript's position. Retention is last-run-only — [`Self::begin`]
/// clears the buffer — and a finished run stays readable until the next one.
#[derive(Debug, Default)]
pub struct SubPane {
    /// The sub-agent's rendered output.
    pub log: OutputLog,
    /// Scroll/follow state for `log`, independent of the main pane's.
    pub view: OutputView,
    /// Display label of the most recent run; `None` before any run.
    pub label: Option<String>,
    /// Whether a sub-agent is running right now.
    pub running: bool,
    /// Whether the pane is the one currently on screen.
    pub active: bool,
    /// Set while a `/subagent` turn is in flight: the turn's own render events
    /// are applied to `log` instead of the main transcript.
    pub adopt_turn: bool,
}

impl SubPane {
    /// Starts a run: clears the buffer and records the label.
    pub fn begin(&mut self, label: String) {
        self.log = OutputLog::new();
        self.view = OutputView::default();
        self.label = Some(label);
        self.running = true;
    }

    /// Ends a run, leaving the buffer readable.
    pub fn end(&mut self) {
        self.running = false;
    }

    /// Handles a `SubStart` from the worker. A nested `agent` tool call made by
    /// the model *inside* a `/subagent` turn also emits `SubStart`; honouring it
    /// would clear the outer run's output, steal its label, and (through the
    /// matching `SubEnd`) mark the still-streaming outer run as finished. While
    /// `adopt_turn` is set the outer run owns the pane, so the nested lifecycle
    /// is ignored and its output simply continues into the same buffer.
    pub fn on_sub_start(&mut self, label: String) {
        if self.adopt_turn {
            return;
        }
        self.begin(label);
    }

    /// Handles a `SubEnd` from the worker; the counterpart of
    /// [`Self::on_sub_start`] and ignored for the same reason.
    pub fn on_sub_end(&mut self) {
        if self.adopt_turn {
            return;
        }
        self.end();
    }

    /// Drops everything the pane holds. Used by the session-resetting commands
    /// (`/clear`, `/new`, `/resume`, `/switch`): a pane left over from the old
    /// session would keep drawing (and, while `active`, would swallow the newly
    /// cleared main log) as if it belonged to the new one.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// The log the user is currently looking at: this pane's while it is on
    /// screen, otherwise `main`. Single decision point for input routing, so a
    /// selection or hit test never reads the pane that is not displayed.
    #[must_use]
    pub fn active_log<'a>(&'a self, main: &'a OutputLog) -> &'a OutputLog {
        if self.active { &self.log } else { main }
    }

    /// The scroll state the visible pane owns — the mutable counterpart of
    /// [`Self::active_log`]. Every scroll/follow gesture goes through this so
    /// the hidden pane's position is never disturbed.
    pub fn active_view<'a>(&'a mut self, main: &'a mut OutputView) -> &'a mut OutputView {
        if self.active { &mut self.view } else { main }
    }

    /// Flips which pane is on screen. Returns `false` (changing nothing) when
    /// no sub-agent has ever run, so there is nothing to switch to.
    pub fn toggle(&mut self) -> bool {
        if self.label.is_none() {
            return false;
        }
        self.active = !self.active;
        true
    }
}

/// The one-shot dim line pushed into the main transcript when a sub-agent run
/// starts. Single source of the wording so the `/subagent` command, the model's
/// `agent` tool, and anything replaying the transcript all say the same thing.
#[must_use]
pub fn subagent_signpost(label: &str) -> String {
    format!("[sub-agent: {label} — ctrl+o to follow]")
}

/// The dim line pushed into the main transcript when several sub-agents run
/// concurrently; the plural counterpart of [`subagent_signpost`].
#[must_use]
pub fn subagents_signpost(labels: &[&str]) -> String {
    format!("[sub-agents: {} — ctrl+o to follow]", labels.join(", "))
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
    /// Wall-clock of the last markdown re-render, and whether tokens have been
    /// appended since. The render is throttled to [`MD_RENDER_MIN_GAP`] while
    /// streaming (see [`OutputLog::md_render_throttled`]); a boundary
    /// [`OutputLog::flush_md`] guarantees the deferred tail still renders.
    last_md_render: Option<Instant>,
    md_dirty: bool,
    /// Count of actual re-renders performed — test-only observability that the
    /// throttle collapses a token burst into few renders, not one per token.
    #[cfg(test)]
    renders: usize,
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
    /// Flushes any throttle-deferred render first so the committed lines hold
    /// the complete segment before the buffer is dropped.
    fn md_close(&mut self) {
        self.flush_md();
        self.md_buf.clear();
        self.md_start = None;
        self.last_md_render = None;
        self.md_dirty = false;
    }

    /// Re-renders the streaming segment at most once per [`MD_RENDER_MIN_GAP`].
    /// Highlighting a code block recompiles a tree-sitter query per render in
    /// the markdown crate (tens of ms), so rendering on every streamed token is
    /// quadratic in the block's length and pins the UI. Rendering on a bounded
    /// cadence keeps highlighting live; deferred tokens ride the `md_dirty` flag
    /// until the next render or a [`flush_md`](Self::flush_md) boundary.
    fn md_render_throttled(&mut self) {
        let due = self
            .last_md_render
            .is_none_or(|t| t.elapsed() >= MD_RENDER_MIN_GAP);
        if due {
            self.md_render();
        } else {
            self.md_dirty = true;
        }
    }

    /// Forces a render when the throttle has deferred appended tokens, so a
    /// segment boundary (tool/think text, end of turn, checkpoint) always
    /// commits the full buffer. No-op when nothing is pending.
    fn flush_md(&mut self) {
        if self.md_dirty && self.md_start.is_some() {
            self.md_render();
        }
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
        // The verbatim source of each top-level code block, in document order.
        // Copying must use this, not the rendered rows: the renderer soft-wraps
        // long lines to the width, and reading those wrapped rows back would
        // turn every wrap point into a spurious newline. Blockquoted code blocks
        // are excluded here just as they are skipped by `annotate_code_blocks`
        // (their headers carry a gutter prefix), so the two lists stay aligned.
        let raw_codes: Vec<&str> = blocks
            .iter()
            .filter_map(|b| match b {
                MarkdownBlock::CodeBlock { code, .. } => Some(code.as_str()),
                _ => None,
            })
            .collect();
        self.lines.truncate(start);
        self.lines.extend(md.render(&blocks, &ThemeConfig::new()));
        // Rebuild the code-block registry for the re-rendered segment; blocks
        // in committed earlier segments keep their (stable) indices.
        self.code_blocks.retain(|r| r.header < start);
        self.code_blocks
            .extend(annotate_code_blocks(&mut self.lines, start, &raw_codes));
        self.last_md_render = Some(Instant::now());
        self.md_dirty = false;
        #[cfg(test)]
        {
            self.renders += 1;
        }
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

    /// Number of committed lines, not counting an in-progress streamed line.
    /// A read-only peek for callers (like tests) that just need to observe
    /// whether the log grew, without the side effects [`checkpoint`] has
    /// (flushing markdown, ending the current line).
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.lines.len()
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

    /// Ensures the streamed output ends on a fresh line. Flushes any
    /// throttle-deferred markdown render first so the turn's final tokens are
    /// committed (the worker sends `EndLine` at the end of every segment).
    pub fn end_line(&mut self) {
        self.flush_md();
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

    /// Drops every line, code block, and in-flight streaming state, returning
    /// the log to its freshly-constructed form. Used by `/clear` and `/new`, so
    /// the screen reflects the fresh session rather than the old conversation.
    pub fn clear(&mut self) {
        *self = Self::new();
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
        self.md_dirty = true;
        self.md_render_throttled();
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

/// A drag selection stored in content space: an inclusive `(x, row)` start and
/// end where `row` is the absolute wrapped-line index (independent of scroll),
/// so the selection tracks the text as the viewport scrolls.
pub type ContentSelection = ((u16, usize), (u16, usize));

/// Orders two content-space endpoints into reading order (top-to-bottom, then
/// left-to-right).
fn order_content(sel: ContentSelection) -> ContentSelection {
    let (a, b) = sel;
    if (a.1, a.0) <= (b.1, b.0) {
        (a, b)
    } else {
        (b, a)
    }
}

/// Projects a [`ContentSelection`] onto the visible output area given the
/// current scroll `top` (in wrapped rows) and the area `height`. Endpoints
/// above or below the viewport clamp to its edges as full-width boundary rows,
/// so a selection larger than one screen still highlights its visible portion.
/// Returns `None` when the whole selection is off-screen. Screen rows are
/// relative to the area's top (the output area starts at row 0).
#[must_use]
pub fn selection_screen(sel: ContentSelection, top: usize, height: u16) -> Option<Selection> {
    let ((sx, sy), (ex, ey)) = order_content(sel);
    let bottom = top + height as usize; // exclusive
    if ey < top || sy >= bottom {
        return None;
    }
    // Above the viewport: start at the top-left, so row 0 selects from the left.
    let start = if sy < top {
        (0, 0)
    } else {
        (sx, u16::try_from(sy - top).unwrap_or(u16::MAX))
    };
    // Below the viewport: end at the bottom-right. `selection_row_bounds` clips
    // the column to the area, so `u16::MAX` is a safe "to end of row" sentinel.
    let end = if ey >= bottom {
        (u16::MAX, height.saturating_sub(1))
    } else {
        (ex, u16::try_from(ey - top).unwrap_or(u16::MAX))
    };
    Some((start, end))
}

/// Extracts the selected text for a [`ContentSelection`] by rendering just the
/// selected wrapped-row range into an off-screen buffer (reusing ratatui's own
/// wrapping) and reading it back — so a selection spanning more than the
/// visible viewport still copies in full. `width` is the wrapped output width.
#[must_use]
pub fn selection_text_content(log: &OutputLog, width: u16, sel: ContentSelection) -> String {
    use ratatui::widgets::Widget as _;
    let width = width.max(1);
    let ((sx, sy), (ex, ey)) = order_content(sel);
    let para = Paragraph::new(log.to_text()).wrap(Wrap { trim: false });
    let total = para.line_count(width);
    if total == 0 || sy >= total {
        return String::new();
    }
    let ey = ey.min(total - 1);
    let height = u16::try_from(ey - sy + 1).unwrap_or(u16::MAX);
    let rect = Rect::new(0, 0, width, height);
    let mut buf = Buffer::empty(rect);
    let scroll = u16::try_from(sy).unwrap_or(u16::MAX);
    para.scroll((scroll, 0)).render(rect, &mut buf);
    // Rebase into the buffer's own space (its row 0 is content row `sy`).
    let local: Selection = ((sx, 0), (ex, height.saturating_sub(1)));
    selection_text(&buf, rect, local)
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

/// Reads the system clipboard as text, for an explicit Ctrl-V.
///
/// `pbpaste` only, with no OSC 52 counterpart: reading the clipboard back over
/// an escape sequence needs a terminal reply, which would mean draining the
/// event stream mid-keypress. Over SSH the terminal's own paste (which arrives
/// as a bracketed [`ratatui::crossterm::event::Event::Paste`]) is the working
/// path, and it is already handled.
///
/// Returns `None` when the clipboard is empty or holds no text — an image, for
/// instance, which the paste handler routes to `crate::imagepaste` instead.
#[must_use]
pub fn paste_from_clipboard() -> Option<String> {
    let out = std::process::Command::new("pbpaste")
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    (!text.is_empty()).then_some(text)
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
    /// Screen rect of the jump-to-bottom hint drawn on the last frame, or
    /// `None` when the hint is hidden (the view already follows the bottom).
    /// Set by `render_output`; read by the mouse handler to make the hint
    /// clickable.
    pub jump_hint_rect: Option<Rect>,
}

impl Default for OutputView {
    fn default() -> Self {
        Self {
            top: 0,
            follow: true,
            jump_hint_rect: None,
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
/// Rows the status bar occupies: the directory/branch row plus the volatile row
/// (see [`status_bar_lines`]). Named so the layout and the renderer cannot
/// disagree about the height.
const STATUS_ROWS: u16 = 2;

fn frame_geom(
    area: Rect,
    has_prompt: bool,
    input_rows: u16,
    strip_rows: u16,
) -> (Rect, Rect, Rect, Option<Rect>, Option<Rect>, Option<Rect>) {
    if has_prompt {
        let r = Layout::vertical([
            Constraint::Min(1),              // output
            Constraint::Length(strip_rows),  // task strip (0 when idle-empty)
            Constraint::Length(1),           // top rule
            Constraint::Length(input_rows),  // input
            Constraint::Length(1),           // bottom rule
            Constraint::Length(STATUS_ROWS), // status (two rows: see status_bar_lines)
        ])
        .split(area);
        let strip = if strip_rows > 0 { Some(r[1]) } else { None };
        (r[0], r[3], r[5], Some(r[2]), Some(r[4]), strip)
    } else {
        let r = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(STATUS_ROWS),
        ])
        .split(area);
        (r[0], r[1], r[2], None, None, None)
    }
}

/// Splits the resting-prompt input into styled spans so a valid command is
/// highlighted live as the user types: a known `/command` token in theme green,
/// and the shell-escape marker colored by where its output goes. Anything else
/// stays default-styled.
///
/// Validity mirrors dispatch: the green highlight appears only when the whole
/// line parses as a known command ([`crate::config::slash_command_known`]), so
/// partial (`/hel`) and unknown (`/nope`) inputs stay plain until complete.
///
/// `/subagent:<name>` splits further — see [`subagent_spans`] — because the
/// command being known says nothing about the name being known.
fn input_spans(input: &str) -> Vec<Span<'static>> {
    // Shell escape: the marker is colored by consequence, which is the one
    // thing the two forms differ in and the one thing that is invisible once
    // typed. Red `!` feeds the command and its output to the model as history;
    // green `!!` keeps it between the user and the shell. Only the marker is
    // colored — any non-empty command is "valid", so the text after it stays
    // plain.
    if let Some(rest) = input.strip_prefix('!') {
        let (marker, color, rest) = match rest.strip_prefix('!') {
            Some(rest) => ("!!", THEME_GREEN, rest),
            None => ("!", Color::Red, rest),
        };
        let mut spans = vec![Span::styled(marker.to_string(), Style::default().fg(color))];
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
        let mut spans = subagent_spans(cmd).unwrap_or_else(|| {
            vec![Span::styled(
                cmd.to_string(),
                Style::default().fg(THEME_GREEN),
            )]
        });
        if !rest.is_empty() {
            spans.push(Span::raw(rest.to_string()));
        }
        return spans;
    }
    vec![Span::raw(input.to_string())]
}

/// Splits a `/subagent:<name>` token into a green `/subagent` and a `:<name>`
/// coloured by whether that definition exists — green when it resolves, red
/// when it does not.
///
/// `None` for any other command, so the caller falls back to colouring the
/// whole token green. The point is to answer "did I spell it right?" while the
/// line is still being typed: dispatch rejects an unknown name outright, and
/// finding that out after pressing Enter is finding out too late.
fn subagent_spans(cmd: &str) -> Option<Vec<Span<'static>>> {
    let name = crate::agents::command_name(cmd)?;
    let color = if crate::agents::is_known(name) {
        THEME_GREEN
    } else {
        Color::Red
    };
    Some(vec![
        Span::styled(
            crate::agents::SUBAGENT_COMMAND.to_string(),
            Style::default().fg(THEME_GREEN),
        ),
        Span::styled(format!(":{name}"), Style::default().fg(color)),
    ])
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

/// Splits a slash-menu row into its command column width and the description
/// budget left over, given the total `width` a row has to work with.
///
/// The command column is sized to the widest label actually on screen (so the
/// descriptions line up as one block) but never takes more than half the row —
/// one long `argument_hint` must not squeeze every description out of view.
#[must_use]
fn slash_columns(labels: &[String], width: usize) -> (usize, usize) {
    use unicode_width::UnicodeWidthStr;
    let widest = labels.iter().map(|l| l.width()).max().unwrap_or(0);
    let cmd = widest.min(width / 2);
    // Two spaces separate the columns.
    (cmd, width.saturating_sub(cmd + 2))
}

/// Draws the `/` command menu: the command (with its argument hint) on the
/// left, its one-line description dimmed on the right, and the source tag for
/// skills and templates after that.
fn render_slash_menu(frame: &mut Frame, area: Rect, menu: &crate::slashmenu::SlashMenu) {
    use ratatui::widgets::{Clear, List, ListItem, ListState, StatefulWidget};
    let rows = menu.rows();
    if area.height == 0 || rows.is_empty() {
        return;
    }
    if crate::uiremote::recording_enabled() {
        crate::uiremote::region(
            "slashmenu",
            area,
            &[
                (
                    "rows",
                    crate::tools::mcp::Json::Num(f64::from(
                        u32::try_from(rows.len()).unwrap_or(u32::MAX),
                    )),
                ),
                (
                    "selected",
                    crate::tools::mcp::Json::Num(f64::from(
                        u32::try_from(menu.selected()).unwrap_or(u32::MAX),
                    )),
                ),
            ],
        );
    }
    frame.render_widget(Clear, area);
    let labels: Vec<String> = rows.iter().map(|e| e.label()).collect();
    // The highlight symbol ("> ") eats two columns of every row.
    let (cmd_w, desc_w) = slash_columns(&labels, usize::from(area.width).saturating_sub(2));
    let items: Vec<ListItem> = rows
        .iter()
        .zip(&labels)
        .map(|(e, label)| {
            let tag = e.source.tag();
            let desc = if tag.is_empty() {
                e.desc.clone()
            } else {
                format!("{} · {tag}", e.desc)
            };
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{:<cmd_w$}", elide_right(label, cmd_w)),
                    Style::default().fg(THEME_GREEN),
                ),
                Span::raw("  "),
                Span::styled(
                    elide_right(&desc, desc_w),
                    Style::default().fg(Color::Indexed(245)),
                ),
            ]))
        })
        .collect();
    let list = List::new(items)
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED))
        .highlight_symbol("> ");
    let mut state = ListState::default();
    state.select(Some(menu.selected()));
    StatefulWidget::render(list, area, frame.buffer_mut(), &mut state);
}

/// Trims `text` to `budget` display columns from the *right*, marking the cut
/// with `…`. Mirror of [`elide_left`], for text whose informative end is the
/// start (command names, prose descriptions).
fn elide_right(text: &str, budget: usize) -> String {
    use unicode_width::UnicodeWidthStr;
    if text.width() <= budget {
        return text.to_string();
    }
    if budget <= 1 {
        return "…".repeat(budget);
    }
    let mut head = String::new();
    for c in text.chars() {
        if head.width() + char_width(c) + 1 > budget {
            break;
        }
        head.push(c);
    }
    format!("{head}…")
}

/// Draws the `/` menu over the frame just rendered, anchored above the input
/// exactly like [`draw_popup`].
pub fn draw_slash_menu(frame: &mut Frame, input_text: &str, menu: &crate::slashmenu::SlashMenu) {
    let tw = input_text_width(frame.area().width);
    let (output, input, _, _, _, _) =
        frame_geom(frame.area(), true, input_height(input_text, tw), 0);
    let rows = u16::try_from(menu.rows().len()).unwrap_or(u16::MAX);
    render_slash_menu(frame, popup_rect(output, input, rows), menu);
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
///
/// `sel` is the selected char range (half-open, in char indices over the whole
/// input), painted as reversed video on top of whatever styling a cell already
/// had — the same treatment [`highlight_selection`] gives the output pane, so
/// selections read identically wherever they are made.
fn wrap_input(
    input: &str,
    width: u16,
    cursor_char: usize,
    sel: Option<(usize, usize)>,
) -> (Vec<Line<'static>>, u16, u16) {
    let width = (width as usize).max(1);
    let mut lines: Vec<Line<'static>> = Vec::new();
    let (mut cur_row, mut cur_col) = (0u16, 0u16);
    let mut base = 0usize; // input char index at the start of the logical line
    for (li, logical) in input.split('\n').enumerate() {
        let mut styled: Vec<(char, Style)> = if li == 0 {
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
        if let Some((lo, hi)) = sel {
            for (k, (_, st)) in styled.iter_mut().enumerate() {
                if (lo..hi).contains(&(base + k)) {
                    *st = st.add_modifier(Modifier::REVERSED);
                }
            }
        }
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

/// The prompt's text plus where the cursor and any selection sit inside it.
///
/// Bundled rather than passed as three parallel arguments so a caller cannot
/// hand [`draw`] a cursor or a selection belonging to different text than the
/// one it renders.
#[derive(Debug, Clone, Copy)]
pub struct InputState<'a> {
    /// The prompt text being edited.
    pub text: &'a str,
    /// Cursor position as a char index into `text`.
    pub cursor: usize,
    /// Selected char range (half-open), when one is active.
    pub sel: Option<(usize, usize)>,
}

impl<'a> InputState<'a> {
    /// A plain cursor-only state, for callers with nothing selected.
    #[must_use]
    pub fn new(text: &'a str, cursor: usize) -> Self {
        Self {
            text,
            cursor,
            sel: None,
        }
    }
}

/// Screen rect the prompt text last occupied — everything right of the prompt
/// glyph, which is the region [`input_hit`] maps clicks over.
///
/// Recorded at render time rather than recomputed, because the input's position
/// depends on the task strip's height, which the mouse handler does not
/// otherwise know. `None` until a prompt has been drawn (and while the agent is
/// busy with the prompt hidden), which is exactly when a click cannot land in
/// it anyway.
static INPUT_TEXT_RECT: std::sync::Mutex<Option<Rect>> = std::sync::Mutex::new(None);

/// The prompt text rect from the last drawn frame, for mouse hit-testing.
#[must_use]
pub fn last_input_rect() -> Option<Rect> {
    INPUT_TEXT_RECT.lock().ok().and_then(|r| *r)
}

/// Records (or, with `None`, forgets) the prompt's text rect. Forgetting it is
/// what keeps a click from steering an invisible cursor on a frame drawn
/// without a prompt.
fn set_input_rect(rect: Option<Rect>) {
    if let Ok(mut slot) = INPUT_TEXT_RECT.lock() {
        *slot = rect;
    }
}

/// Maps a screen cell to a char index into `input`, using the same word wrap
/// [`wrap_input`] draws with.
///
/// Returns `None` only when the *row* misses the prompt entirely — the column
/// is clamped into `area`, so a click on the prompt glyph left of the text
/// lands at the start of its row and one past the end of a row lands at that
/// row's end. Dragging off either edge therefore selects to the boundary
/// instead of doing nothing.
#[must_use]
pub fn input_hit(area: Rect, input: &str, col: u16, row: u16) -> Option<usize> {
    if area.width == 0 || area.height == 0 || row < area.y || row >= area.bottom() {
        return None;
    }
    let width = usize::from(area.width).max(1);
    let target = usize::from(row - area.y);
    let want_col = usize::from(col.clamp(area.x, area.right().saturating_sub(1)) - area.x);
    let mut visual = 0usize;
    let mut base = 0usize;
    let mut last_end = 0usize;
    for logical in input.split('\n') {
        let chars: Vec<char> = logical.chars().collect();
        let offsets = wrap_offsets(&chars, width);
        for (si, &start) in offsets.iter().enumerate() {
            let end = offsets.get(si + 1).copied().unwrap_or(chars.len());
            if visual == target {
                // Walk the row's cells until the click column is covered; a
                // wide char claims the columns it spans.
                let mut w = 0usize;
                for (k, c) in chars[start..end].iter().enumerate() {
                    let cw = char_width(*c).max(1);
                    if want_col < w + cw {
                        return Some(base + start + k);
                    }
                    w += cw;
                }
                return Some(base + end);
            }
            visual += 1;
            last_end = base + end;
        }
        base += chars.len() + 1; // +1 for the consumed newline
    }
    // Below the last drawn row: clamp to the end of the text.
    Some(last_end)
}

/// Draws the prompt glyph and the word-wrapped input text into `input_area`,
/// placing the terminal cursor for `state.cursor` and reversing `state.sel`.
///
/// The text is indented under the prompt and wraps to the next row instead of
/// scrolling horizontally.
fn render_input(frame: &mut Frame, input_area: Rect, state: InputState<'_>) {
    let input = state.text;
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
    set_input_rect(Some(text_area));
    let (lines, cur_row, cur_col) = wrap_input(input, text_area.width, state.cursor, state.sel);
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
    // Emphasis for the changed span of a word-diffed pair: brighter bg + bold so
    // the surrounding common text (base bg) recedes. Mirrors `EditPreview::to_ansi`.
    let del_emph = Style::default()
        .bg(Color::Indexed(88))
        .fg(Color::Indexed(231))
        .add_modifier(Modifier::BOLD);
    let add_emph = Style::default()
        .bg(Color::Indexed(28))
        .fg(Color::Indexed(231))
        .add_modifier(Modifier::BOLD);
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
            DiffRow::Del { text, segments, .. } => log.push_spans(diff_line_spans(
                &format!("{} - ", gutter(row.gutter())),
                text,
                segments.as_deref(),
                del,
                del_emph,
            )),
            DiffRow::Add { text, segments, .. } => log.push_spans(diff_line_spans(
                &format!("{} + ", gutter(row.gutter())),
                text,
                segments.as_deref(),
                add,
                add_emph,
            )),
            DiffRow::Elision(n) => {
                log.push_spans(vec![Span::styled(format!("      ⋯ {n} more lines ⋯"), dim)]);
            }
        }
    }
    log.push_spans(vec![]);
}

/// Builds the styled spans for one Del/Add diff row. With word-level `segments`,
/// the `prefix` (gutter + sigil) and common runs take `base` while changed runs
/// take `emph`; without segments the whole line is one `base` span, matching the
/// prior line-level rendering.
fn diff_line_spans(
    prefix: &str,
    text: &str,
    segments: Option<&[crate::tools::diff::Segment]>,
    base: Style,
    emph: Style,
) -> Vec<Span<'static>> {
    use crate::tools::diff::SegKind;
    match segments {
        Some(segs) => {
            let mut spans = vec![Span::styled(prefix.to_string(), base)];
            for seg in segs {
                let style = match seg.kind {
                    SegKind::Removed | SegKind::Added => emph,
                    SegKind::Common => base,
                };
                spans.push(Span::styled(seg.text.clone(), style));
            }
            spans
        }
        None => vec![Span::styled(format!("{prefix}{text}"), base)],
    }
}

/// Minimal pre-UI screen shown while the KV cache is (re)built at launch: a
/// centered note and a simple progress bar. The full UI is withheld until
/// warming finishes, so the user sees clear progress instead of an idle screen
/// during the one slow step. `stage` names the tier currently prefilling (see
/// [`crate::kvtier::TierKind::warm_label`]) so the note does not claim the
/// system prompt is being rebuilt while a cheaper context tier is the one
/// running.
pub fn draw_warm(
    frame: &mut Frame,
    done: i32,
    total: i32,
    tps: f64,
    stage: &str,
    notice: Option<&str>,
) {
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
            format!("{stage}…"),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ))
        .centered(),
        Line::from(format!("{bar}  {pct}%")).centered(),
    ]);
    frame.render_widget(Paragraph::new(text), rows[1]);
    // Reason for the rebuild (cache missing / prompt changed + diff), below the
    // progress bar in the reserved region.
    if let Some(notice) = notice {
        // First line (the reason) is centered yellow; diff rows below use the
        // same red/green backgrounds as the code-diff cards (del bg 52 / fg 224,
        // add bg 22 / fg 194) so a `-`/`+` diff reads the same everywhere. The
        // elision summary is dimmed.
        let lines: Vec<Line> = notice
            .lines()
            .enumerate()
            .map(|(i, l)| {
                if i == 0 {
                    return Line::from(Span::styled(
                        l.to_owned(),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ))
                    .centered();
                }
                let style = if l.starts_with("- ") {
                    Style::default()
                        .bg(Color::Indexed(52))
                        .fg(Color::Indexed(224))
                } else if l.starts_with("+ ") {
                    Style::default()
                        .bg(Color::Indexed(22))
                        .fg(Color::Indexed(194))
                } else {
                    Style::default().fg(Color::Indexed(240))
                };
                Line::from(Span::styled(l.to_owned(), style))
            })
            .collect();
        frame.render_widget(Paragraph::new(Text::from(lines)), rows[2]);
    }
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

/// How far a veiled cell's color is pulled toward black.
const VEIL_KEEP: f32 = 0.32;

/// Pushes everything already drawn in `area` back into the background, so an
/// overlay painted on top reads as the foreground layer.
///
/// RGB colors are scaled; the palette colors (named and indexed) have no
/// numeric brightness to scale, so they collapse to one dim gray. Backgrounds
/// and attributes are dropped outright — a leftover highlight or bold run would
/// punch straight back through the veil.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // scaling a u8 down stays in u8
fn veil(buf: &mut Buffer, area: Rect) {
    const DIM: Color = Color::Indexed(236);
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            let Some(cell) = buf.cell_mut(Position::new(x, y)) else {
                continue;
            };
            // `Color::Reset` is the common case — ordinary output carries the
            // terminal's default foreground — and it must dim like the rest,
            // or most of the screen would shine straight through the veil.
            let fg = match cell.fg {
                Color::Rgb(r, g, b) => Color::Rgb(
                    (f32::from(r) * VEIL_KEEP) as u8,
                    (f32::from(g) * VEIL_KEEP) as u8,
                    (f32::from(b) * VEIL_KEEP) as u8,
                ),
                _ => DIM,
            };
            cell.set_style(Style::reset().fg(fg));
        }
    }
}

/// Lays out the arcade's bottom line: status and key hints on the left, the
/// exit hint pinned to the right edge.
///
/// A full status line runs past 80 columns, and plain truncation would cut off
/// exactly the part that says how to get out. So the exit hint is placed first
/// and the status is trimmed to whatever is left over.
fn arcade_footer_line(arcade: &crate::arcade::Arcade, width: u16) -> String {
    let exit = crate::arcade::Arcade::EXIT_HINT;
    let width = usize::from(width);
    if width <= exit.width() + 2 {
        // Too narrow for both: the exit hint is the one that must survive.
        return exit.chars().take(width).collect();
    }
    let room = width - exit.width() - 2;
    let mut left = String::new();
    for ch in arcade.footer().chars() {
        if left.width() + ch.width().unwrap_or(0) > room {
            break;
        }
        left.push(ch);
    }
    let pad = width - exit.width() - left.width();
    format!("{left}{}{exit}", " ".repeat(pad))
}

/// Draws an open arcade easter egg (`/stars`, `/pelota`) over the whole frame.
///
/// The arcade hands back a flat list of [`crate::arcade::Glyph`]s in area
/// coordinates; this paints them straight into the buffer rather than building
/// `Line`s, because the content is sparse — a starfield touches a few hundred
/// cells out of several thousand. The bottom row is reserved for the hint line
/// and any centered banner is stamped over the middle.
pub fn draw_arcade(frame: &mut Frame, arcade: &crate::arcade::Arcade) {
    let area = frame.area();
    if arcade.translucent {
        // No Clear: the frame already holds the live UI, and the glyphs land in
        // the gaps between its characters. Pushing everything underneath down
        // to a dim gray is what sells the layer — see `Arcade::translucent` for
        // why this, and not alpha, is how a terminal does translucency.
        veil(frame.buffer_mut(), area);
    } else {
        frame.render_widget(Clear, area);
        // An explicit RGB triple, not `Color::Black`: that is ANSI index 0,
        // which themes remap freely and most render as a dark grey — the
        // starfield came up on grey. Night sky wants a real black.
        frame.render_widget(
            Block::default().style(Style::default().bg(Color::Rgb(0, 0, 0))),
            area,
        );
    }
    if area.height < 2 {
        return;
    }
    let play = Rect {
        height: area.height - 1,
        ..area
    };

    let buf = frame.buffer_mut();
    for g in arcade.glyphs(play.width, play.height) {
        let (x, y) = (play.x + g.x, play.y + g.y);
        if let Some(cell) = buf.cell_mut(Position::new(x, y)) {
            let (r, gr, b) = g.color;
            cell.set_char(g.ch).set_fg(Color::Rgb(r, gr, b));
        }
    }

    if let Some(text) = arcade.banner(play.width, play.height) {
        let width = u16::try_from(text.width()).unwrap_or(u16::MAX);
        let rect = Rect {
            x: play.x + play.width.saturating_sub(width + 2) / 2,
            y: play.y + play.height / 2,
            width: (width + 2).min(play.width),
            height: 1,
        };
        frame.render_widget(Clear, rect);
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!(" {text} "),
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Rgb(255, 214, 120))
                    .add_modifier(Modifier::BOLD),
            ))),
            rect,
        );
    }

    // The screensaver has no controls to hint at — any key puts the UI back —
    // so it gets the whole screen, stars and nothing else.
    if arcade.is_screensaver() {
        return;
    }

    let footer = Rect {
        y: area.y + area.height - 1,
        height: 1,
        ..area
    };
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            arcade_footer_line(arcade, footer.width),
            Style::default().fg(Color::Indexed(245)),
        ))),
        footer,
    );
}

/// Draws one frame: output log, input line, and status bar.
///
/// `input` carries the prompt text with its cursor and selection.
/// `input` is `None` while the agent is busy (prefill/generation): the prompt
/// line renders empty and the cursor stays hidden until input is accepted again.
/// `view` is the scroll state; it is clamped in place to the scrollable range
/// and a jump-to-bottom hint is shown while it is pinned above the bottom.
/// `sub_label` names the sub-agent whose pane is being shown, adding the frame
/// title and the `ctrl+o: back to main` hint; `None` draws the main transcript.
#[allow(clippy::too_many_arguments)]
pub fn draw(
    frame: &mut Frame,
    log: &OutputLog,
    input: Option<InputState<'_>>,
    status: &str,
    view: &mut OutputView,
    selection: Option<ContentSelection>,
    tasks: &TaskView,
    sub_label: Option<&str>,
) {
    let area = frame.area();
    let tw = input_text_width(area.width);
    let (output, input_row, status_row) = frame_rows(
        frame,
        area,
        input.is_some(),
        input.map_or(1, |s| input_height(s.text, tw)),
        tasks,
    );

    render_output(frame, output, log, view, selection);
    // `Some` only while the sub-agent pane is the one on screen; the main
    // transcript draws exactly as before.
    if let Some(label) = sub_label {
        draw_sub_header(frame, output, label);
    }

    // Input line: hidden entirely (no prompt, no cursor) while the agent is busy.
    match input {
        Some(input) => render_input(frame, input_row, input),
        // No prompt on this frame: forget its rect so a stray click cannot
        // steer a cursor that is not on screen.
        None => set_input_rect(None),
    }

    // Status bar, reverse-styled across the full width, with a magenta bar.
    let status_style = Style::default()
        .bg(Color::Indexed(238))
        .fg(Color::Indexed(252));
    frame.render_widget(
        Paragraph::new(status_bar_lines(
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
        Paragraph::new(status_bar_lines(
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
    selection: Option<ContentSelection>,
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
    view.jump_hint_rect = if view.follow {
        None
    } else {
        draw_jump_hint(frame, area)
    };
    // Project the content-space selection onto the (post-clamp) viewport.
    if let Some(sel) = selection.and_then(|s| selection_screen(s, view.top, area.height)) {
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
    input: Option<InputState<'_>>,
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
        input.map_or(1, |s| input_height(s.text, tw)),
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
    match input {
        Some(input) => render_input(frame, input_row, input),
        // No prompt on this frame: forget its rect so a stray click cannot
        // steer a cursor that is not on screen.
        None => set_input_rect(None),
    }
    let status_style = Style::default()
        .bg(Color::Indexed(238))
        .fg(Color::Indexed(252));
    frame.render_widget(
        Paragraph::new(status_bar_lines(
            &with_remote_marker(status),
            anim_tick_ms(),
            status_style,
            tasks,
        ))
        .style(status_style),
        status_row,
    );
}

/// Overlays the sub-agent pane's identity on the output area's top row: the
/// frame title `[sub-agent: <label>]` on the left and the `ctrl+o: back to
/// main` hint on the right. Drawn as an overlay (like [`draw_jump_hint`])
/// rather than as a bordered block so the output area keeps its geometry —
/// scroll math and mouse hit-testing map screen rows to content rows directly,
/// and a consumed border row would shift both by one. Persistent while the
/// pane is on screen, so a user who has scrolled past the one-shot signpost
/// can still see the way back.
fn draw_sub_header(frame: &mut Frame, area: Rect, label: &str) {
    const HINT: &str = " ctrl+o: back to main ";
    if area.height == 0 {
        return;
    }
    let title = format!(" [sub-agent: {label}] ");
    let title_width = u16::try_from(title.chars().count()).unwrap_or(u16::MAX);
    let bar = Style::default()
        .bg(Color::Indexed(238))
        .fg(Color::Indexed(252));
    if area.width >= title_width {
        frame.render_widget(
            Paragraph::new(Span::styled(
                title,
                bar.fg(THEME_GREEN).add_modifier(Modifier::BOLD),
            )),
            Rect::new(area.x, area.y, title_width, 1),
        );
    }
    let hint_width = u16::try_from(HINT.chars().count()).unwrap_or(u16::MAX);
    // Only when both fit side by side: a truncated hint reads as noise.
    if area.width >= title_width.saturating_add(hint_width) {
        frame.render_widget(
            Paragraph::new(Span::styled(HINT, bar)),
            Rect::new(area.right() - hint_width, area.y, hint_width, 1),
        );
    }
}

/// Overlays the jump-to-bottom affordance on the output area's bottom-right
/// corner while the view is pinned above the newest output. Returns the screen
/// rect it drew (so the mouse handler can make it clickable), or `None` when
/// the area is too small to show the hint.
fn draw_jump_hint(frame: &mut Frame, area: Rect) -> Option<Rect> {
    const HINT: &str = " ↓ End/click: jump to bottom ";
    let hint_width = u16::try_from(HINT.chars().count()).unwrap_or(u16::MAX);
    if area.width < hint_width || area.height == 0 {
        return None;
    }
    let rect = Rect::new(area.right() - hint_width, area.bottom() - 1, hint_width, 1);
    let style = Style::default()
        .bg(Color::Indexed(238))
        .fg(Color::Indexed(252))
        .add_modifier(Modifier::BOLD);
    frame.render_widget(Paragraph::new(Span::styled(HINT, style)), rect);
    Some(rect)
}

/// Milliseconds off the shared 20 Hz animation clock, driving the shimmer
/// sweep. Returns `0` (a frozen, static render) when reduced motion is active,
/// so every effect collapses to its fallback from one branch point.
fn anim_tick_ms() -> u64 {
    crate::anim::clock_ms().unwrap_or(0)
}

/// Pushes the accent word with a shimmer: a graded highlight sweeps
/// right-to-left across the word, one column per `SHIMMER_STEP_MS`, over a
/// cycle of word width + 20 columns (so the highlight rests off-text between
/// sweeps).
///
/// Each column inside the window takes its own shade from
/// [`crate::status::SHIMMER_RAMP`] by distance from the center — brightest in
/// the middle, easing back into the theme color at the edges — so the sweep
/// looks like light travelling over the word rather than a white block sliding
/// along it.
fn push_shimmered(spans: &mut Vec<Span<'static>>, word: &str, tick_ms: u64, theme: Style) {
    let ramp = crate::status::SHIMMER_RAMP;
    let half = i64::try_from(ramp.len().saturating_sub(1)).unwrap_or(0);
    let width = i64::try_from(word.chars().count()).unwrap_or(0);
    let cycle = width + 20;
    let step = i64::try_from(tick_ms / crate::status::SHIMMER_STEP_MS).unwrap_or(0);
    let center = width + 10 - step % cycle;
    // One shade per column, then coalesce equal-styled neighbours so a sweep
    // costs a handful of spans rather than one per character.
    let mut runs: Vec<(String, Style)> = Vec::new();
    for (col, ch) in word.chars().enumerate() {
        let col = i64::try_from(col).unwrap_or(i64::MAX);
        let dist = (col - center).abs();
        let style = if dist <= half {
            // dist 0 is the center, which takes the last (brightest) shade.
            let idx = ramp.len() - 1 - usize::try_from(dist).unwrap_or(0);
            theme.fg(Color::Indexed(ramp[idx]))
        } else {
            theme
        };
        match runs.last_mut() {
            Some((text, prev)) if *prev == style => text.push(ch),
            _ => runs.push((ch.to_string(), style)),
        }
    }
    for (text, style) in runs {
        spans.push(Span::styled(text, style));
    }
}

/// Pushes spans for `seg`, painting the accent word — `prefill` before the
/// bar, or the trailing-`…` spinner verb — in the theme color with the
/// shimmer animation sweeping across it.
/// Themes the leading directory segment across the status bar's two rows.
/// `prefix` looks like `"<path> | "` or `"<path> ⎇ <branch> | <origin> | "`; the
/// path and branch go to `first` (row one) in the theme green with the powerline
/// glyph plain, and the engine origin heads `spans` (row two).
fn push_dir_prefix(
    first: &mut Vec<Span<'static>>,
    spans: &mut Vec<Span<'static>>,
    prefix: &str,
    base: Style,
    theme: Style,
) {
    let glyph = crate::status::POWERLINE_BRANCH;
    // Trailing " | " separator that hands off to the "ctx …" body.
    let (segment, sep) = prefix
        .rfind(" | ")
        .map_or((prefix, ""), |i| (&prefix[..i], &prefix[i..]));
    // The engine origin is its own bar-separated segment after the path/branch,
    // and stays plain so it never reads as part of the branch name. Peel it here
    // for the same reason the think segment is peeled: `rfind` above lands on the
    // separator *before* the origin, not the one before the body.
    let origin = crate::status::engine_origin_label();
    let (segment, origin) = match segment.strip_suffix(origin.as_str()) {
        Some(head) => {
            let head = head.trim_end();
            let head = head.strip_suffix('|').map_or(head, str::trim_end);
            if head.is_empty() {
                ("", origin.clone())
            } else {
                (head, format!(" | {origin}"))
            }
        }
        None => (segment, String::new()),
    };
    if let Some(gi) = segment.find(glyph) {
        let path = segment[..gi].trim_end();
        let branch = segment[gi + glyph.len_utf8()..].trim();
        first.push(Span::styled(path.to_string(), theme));
        first.push(Span::styled(format!(" {glyph} "), base));
        first.push(Span::styled(branch.to_string(), theme));
    } else {
        first.push(Span::styled(segment.trim_end().to_string(), theme));
    }
    // The origin heads the *second* row rather than trailing the first: row one
    // answers "which tree am I in", and only that, so it stays readable at a
    // glance while everything volatile lives below it.
    if !origin.is_empty() {
        spans.push(Span::styled(
            origin.trim_start_matches(" | ").to_owned(),
            base,
        ));
    }
    let _ = sep;
    if !spans.is_empty() {
        spans.push(Span::styled(" | ".to_string(), base));
    }
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

/// The compaction progress line, shown in place of the throbber/spinner-verb
/// segment pinned below the output while a compaction pass runs. Same slot, same
/// role: it is what the turn is doing right now.
#[must_use]
pub fn compact_progress_line(frac: f64) -> Line<'static> {
    Line::from(compact_slot_spans(frac, anim_tick_ms(), Style::default()))
}

/// Builds the compaction indicator: a flashing `compacting` label, then the
/// bar, then the percentage.
///
/// Only the *label* flashes. The bar and the percentage hold steady, because a
/// blinking progress bar reads as a glitch rather than as progress — and both
/// keep a fixed width, so the line does not reflow as the bar fills.
fn compact_slot_spans(frac: f64, tick_ms: u64, base: Style) -> Vec<Span<'static>> {
    let (filled, empty, pct) = crate::status::compact_bar(frac);
    let dim = base.fg(Color::Indexed(240));
    vec![
        Span::styled(
            "compacting ".to_string(),
            if crate::status::tool_blink_on(tick_ms) {
                base.fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                dim
            },
        ),
        Span::styled(filled, base.add_modifier(Modifier::BOLD)),
        Span::styled(empty, dim),
        Span::styled(format!(" {pct}%"), base.add_modifier(Modifier::BOLD)),
    ]
}

/// Builds the status bar's two rows, coloring the progress bar's filled arrows
/// and the accent word (operation name or spinner verb) in the theme color.
///
/// Row one is the working directory and git branch, and nothing else: the answer
/// to "which tree am I in" holds still while the rest of the bar churns. Row two
/// carries everything volatile — engine origin, think level, context gauge,
/// progress or state, task counter, power suffix, remote marker — in the order
/// the single-row bar used.
///
/// The bar segment lives between `[` and `]`; `▶` cells render in the theme
/// color (military green) and `·` cells a dim gray.
fn status_bar_lines(text: &str, tick_ms: u64, base: Style, tasks: &TaskView) -> Vec<Line<'static>> {
    let theme = base
        .fg(Color::Indexed(crate::status::THEME_COLOR))
        .add_modifier(Modifier::BOLD);
    let mut spans = Vec::new();
    let mut first = Vec::new();
    // Peel the leading "<path> ⎇ <branch> | " directory segment onto row one and
    // theme the path and branch green; the powerline glyph and separators stay
    // plain.
    //
    // The boundary is the think segment when present, else the ctx gauge. It
    // cannot be the ctx gauge unconditionally: `push_dir_prefix` splits on the
    // *last* " | " it finds, so leaving "🧠 medium | " inside the prefix slice
    // would make the branch read "main | 🧠 medium".
    let think_mark = crate::status::THINK_MARK;
    let boundary = text
        .find(think_mark)
        .or_else(|| text.find("ctx "))
        .filter(|&i| i > 0);
    let mut text = if let Some(idx) = boundary {
        push_dir_prefix(&mut first, &mut spans, &text[..idx], base, theme);
        &text[idx..]
    } else {
        text
    };
    // The think segment is its own span: plain, like the ctx gauge and power
    // suffix it sits beside, and kept away from `push_accented`'s verb shimmer.
    //
    // While the *local* engine is prefilling or generating, its brain blinks —
    // the one on-screen signal that says which engine is actually working, which
    // is otherwise invisible for a `provider: local` sidechain under a remote
    // main agent.
    //
    // The dark half replaces the glyph with its own width in spaces, so the
    // brain genuinely disappears and the rest of the bar holds still. Styling
    // the emoji is not an option: `THINK_MARK` is a color emoji, and a terminal
    // paints those from the glyph's own palette — an earlier version dimmed its
    // foreground, which a terminal simply does not render.
    //
    // Phased off the pass's own elapsed time rather than `tick_ms`: the status
    // bar redraws when a prefill/generation event lands — the same event that
    // moves the `9s` and `t/s` readouts — so tying the blink to that interval
    // keeps the two in step and makes every pass start with the brain showing.
    if text.starts_with(think_mark)
        && let Some(i) = text.find(" | ")
    {
        let segment = &text[..i];
        let showing = !crate::status::local_pass_active()
            || crate::anim::reduced_motion()
            || crate::status::brain_blink_on(crate::status::local_pass_ms());
        if showing {
            spans.push(Span::styled(segment.to_string(), base));
        } else {
            let rest = segment.strip_prefix(think_mark).unwrap_or(segment);
            spans.push(Span::styled(
                " ".repeat(UnicodeWidthStr::width(think_mark)),
                base,
            ));
            spans.push(Span::styled(rest.to_string(), base));
        }
        spans.push(Span::styled(" | ".to_string(), base));
        text = &text[i + " | ".len()..];
    }
    let text = text;
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
    // Tail notification slot. A running tool owns it for the whole run (no
    // timed window), shimmering off the animation clock, so nothing else shows
    // there until the tool finishes. Otherwise a transient "flash" (e.g. a copy
    // confirmation) takes over for its window; otherwise the rotating yellow
    // tip shows, changing every few seconds off the animation clock. On a
    // narrow terminal the line truncates and drops whichever tip is last.
    if let Some(running) = crate::status::tool_activity() {
        spans.push(Span::styled(" | ".to_string(), base));
        if crate::status::tool_blink_on(tick_ms) {
            push_shimmered(&mut spans, &running, tick_ms, theme);
        } else {
            // Off half of the blink: same glyphs at the same width, dimmed, so
            // the label pulses without the line jittering around it.
            spans.push(Span::styled(running, base.fg(Color::Indexed(240))));
        }
    } else if let Some(flash) = crate::status::flash_tip() {
        spans.push(Span::styled(" | ".to_string(), base));
        spans.push(Span::styled(
            flash,
            base.fg(Color::Green).add_modifier(Modifier::BOLD),
        ));
    } else {
        let tip = crate::status::rotating_tip(tick_ms);
        if !tip.is_empty() {
            spans.push(Span::styled(" | ".to_string(), base));
            spans.push(Span::styled(
                format!("💡 {tip}"),
                base.fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ));
        }
    }
    vec![Line::from(first), Line::from(spans)]
}

#[cfg(test)]
mod tests {
    use super::SubPane;
    use unicode_width::UnicodeWidthStr;

    #[test]
    fn sub_pane_begin_clears_the_buffer_and_toggle_needs_a_run() {
        let mut pane = SubPane::default();
        // Nothing has run yet: there is nothing to switch to.
        assert!(!pane.toggle());
        assert!(!pane.active);

        pane.begin("research".to_string());
        assert_eq!(pane.label.as_deref(), Some("research"));
        assert!(pane.running);
        pane.log.push_plain("first run output");
        let after_first = pane.log.line_count();
        assert!(after_first > 0);

        // Now it is followable, and toggling is reversible.
        assert!(pane.toggle());
        assert!(pane.active);
        assert!(pane.toggle());
        assert!(!pane.active);

        pane.end();
        assert!(!pane.running);
        // The finished run stays readable.
        assert_eq!(pane.log.line_count(), after_first);

        // A second run starts from an empty buffer (last-run-only retention).
        pane.begin("other".to_string());
        assert_eq!(pane.log.line_count(), 0);
        assert_eq!(pane.label.as_deref(), Some("other"));
    }

    #[test]
    fn sub_pane_begin_resets_the_scroll_view_not_just_the_log() {
        let mut pane = SubPane::default();
        pane.begin("first".to_string());
        pane.log.push_plain("a long first run");
        // The user scrolled up in the sub pane during the first run.
        pane.view.follow = false;
        pane.view.top = 42;

        // A new run must start at the top in follow mode: a stale `top` from
        // the previous (now discarded) buffer would point past the new content.
        pane.begin("second".to_string());
        assert_eq!(pane.view.top, 0);
        assert!(pane.view.follow);
        assert_eq!(pane.log.line_count(), 0);
    }

    #[test]
    fn sub_start_and_end_never_clobber_an_adopted_run() {
        // A `/subagent` turn owns the pane (`adopt_turn`). The model inside it
        // may still call the `agent` tool, which emits SubStart/SubEnd; those
        // must not clear the buffer, steal the label, or mark the outer run
        // finished while it is still streaming.
        let mut pane = SubPane::default();
        pane.begin("reviewer".to_string());
        pane.adopt_turn = true;
        pane.log.push_plain("outer /subagent output");
        let before = pane.log.line_count();

        pane.on_sub_start("nested".to_string());
        assert_eq!(pane.label.as_deref(), Some("reviewer"), "label kept");
        assert_eq!(pane.log.line_count(), before, "buffer not cleared");
        assert!(pane.running, "outer run still running");

        pane.log.push_plain("more outer output");
        pane.on_sub_end();
        assert!(pane.running, "inner SubEnd must not end the outer run");
        assert_eq!(pane.log.line_count(), before + 1);

        // Outside an adopted run the events do their normal work.
        pane.adopt_turn = false;
        pane.on_sub_start("nested".to_string());
        assert_eq!(pane.label.as_deref(), Some("nested"));
        assert_eq!(pane.log.line_count(), 0);
        assert!(pane.running);
        pane.on_sub_end();
        assert!(!pane.running);
    }

    #[test]
    fn sub_pane_header_shows_the_label_and_the_way_back() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        fn render(sub_label: Option<&str>) -> String {
            let mut log = OutputLog::new();
            log.push_plain("sub output");
            let mut view = OutputView::default();
            let mut term = Terminal::new(TestBackend::new(60, 8)).unwrap();
            term.draw(|f| {
                draw(
                    f,
                    &log,
                    Some(InputState::new("", 0)),
                    "idle",
                    &mut view,
                    None,
                    &TaskView::default(),
                    sub_label,
                );
            })
            .unwrap();
            let buf = term.backend().buffer().clone();
            (0..buf.area.height)
                .map(|y| {
                    (0..buf.area.width)
                        .map(|x| buf[(x, y)].symbol())
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
                .join("\n")
        }

        let with = render(Some("research"));
        assert!(with.contains("[sub-agent: research]"), "{with}");
        assert!(with.contains("ctrl+o: back to main"), "{with}");

        // The main transcript is untouched: no title, no hint, and the output
        // text still starts on the very first row.
        let without = render(None);
        assert!(!without.contains("sub-agent"), "{without}");
        assert!(!without.contains("ctrl+o"), "{without}");
        assert!(without.starts_with("sub output"), "{without}");
    }

    #[test]
    fn reset_drops_everything_the_old_session_left_behind() {
        let mut pane = SubPane::default();
        pane.begin("research".to_string());
        pane.log.push_plain("old session output");
        pane.adopt_turn = true;
        assert!(pane.toggle());
        assert!(pane.active);

        pane.reset();
        assert_eq!(pane.log.line_count(), 0);
        assert!(pane.label.is_none());
        assert!(
            !pane.active,
            "a hidden pane must not swallow the new session"
        );
        assert!(!pane.running);
        assert!(!pane.adopt_turn);
        // Nothing to switch to again, exactly as at launch.
        assert!(!pane.toggle());
    }

    #[test]
    fn sub_pane_active_log_and_view_follow_which_pane_is_on_screen() {
        let mut main_log = super::OutputLog::new();
        main_log.push_plain("main transcript");
        let mut main_view = super::OutputView {
            top: 7,
            ..Default::default()
        };

        let mut pane = SubPane::default();
        pane.begin("research".to_string());
        pane.log.push_plain("sub output");
        pane.log.push_plain("sub output two");
        pane.view.top = 3;

        // Inactive: everything routes to the main pane, unchanged.
        assert_eq!(pane.active_log(&main_log).line_count(), 1);
        assert_eq!(pane.active_view(&mut main_view).top, 7);

        assert!(pane.toggle());
        assert_eq!(pane.active_log(&main_log).line_count(), 2);
        let v = pane.active_view(&mut main_view);
        assert_eq!(v.top, 3);
        // Scrolling the active pane leaves the hidden one alone.
        v.top = 9;
        assert_eq!(pane.view.top, 9);
        assert_eq!(main_view.top, 7);
    }

    /// Issue #72: `/clear` and `/new` must wipe the TUI scrollback, including
    /// an unfinished streaming line, a pinned progress line, and the code-block
    /// registry that backs click-to-copy.
    #[test]
    fn clear_empties_the_output_log() {
        use crate::viz::RenderSink;
        let mut log = super::OutputLog::new();
        log.push_plain("older conversation");
        log.visible_text("```rust\nfn main() {}\n```\nstreaming tail");
        log.set_progress(Some(ratatui::text::Line::from("working…")));
        assert!(!log.to_text().lines.is_empty());

        log.clear();

        assert!(log.to_text().lines.is_empty(), "{:?}", log.to_text());
        assert!(log.code_copy_at(80, 0, 0, 0).is_none());
        assert_eq!(log.checkpoint(), 0);
    }

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
    fn compact_progress_line_flashes_the_label_but_holds_the_bar_steady() {
        let base = Style::default();
        let lit = crate::status::TOOL_BLINK_MS / 4; // lit half of the blink
        let dark = crate::status::TOOL_BLINK_MS * 3 / 4; // dark half
        let spans = compact_slot_spans(0.21, lit, base);
        let text: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.starts_with("compacting "), "{text}");
        assert!(text.ends_with(" 21%"), "{text}");
        assert_eq!(spans[0].style.fg, Some(Color::Yellow), "label lit");
        // Bar cells: filled + empty always add up to the full width.
        assert_eq!(
            spans[1].content.chars().count() + spans[2].content.chars().count(),
            crate::status::COMPACT_BAR_WIDTH
        );
        assert!(spans[1].content.chars().all(|c| c == '▰'));
        assert!(spans[2].content.chars().all(|c| c == '▱'));

        // Off half of the blink: the label dims, everything else is unchanged
        // (same glyphs, same width), so the line does not jitter.
        let off = compact_slot_spans(0.21, dark, base);
        assert_eq!(off[0].style.fg, Some(Color::Indexed(240)), "label dimmed");
        let off_text: String = off.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(off_text, text);
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
    fn input_spans_colors_the_subagent_name_by_whether_it_exists() {
        // The roster is a process-global; these names are distinctive enough
        // not to collide with anything a concurrent test would publish.
        crate::agents::set_roster_for_test(&["spanreviewer"]);

        // A name that resolves: both halves green.
        assert_eq!(
            parts("/subagent:spanreviewer check the diff"),
            vec![
                ("/subagent".to_owned(), Some(THEME_GREEN)),
                (":spanreviewer".to_owned(), Some(THEME_GREEN)),
                (" check the diff".to_owned(), None),
            ]
        );
        // A name that does not: the command stays green, the name goes red —
        // the command *is* valid, it is the name that is wrong.
        assert_eq!(
            parts("/subagent:nosuchagent check the diff"),
            vec![
                ("/subagent".to_owned(), Some(THEME_GREEN)),
                (":nosuchagent".to_owned(), Some(Color::Red)),
                (" check the diff".to_owned(), None),
            ]
        );
        // The name is coloured while the task is still unwritten, which is the
        // whole point: the answer arrives before Enter, not after.
        assert_eq!(
            parts("/subagent:nosuchagent"),
            vec![
                ("/subagent".to_owned(), Some(THEME_GREEN)),
                (":nosuchagent".to_owned(), Some(Color::Red)),
            ]
        );
        // The bare form has no name to judge and stays one green token.
        assert_eq!(
            parts("/subagent check the diff"),
            vec![
                ("/subagent".to_owned(), Some(THEME_GREEN)),
                (" check the diff".to_owned(), None),
            ]
        );
    }

    /// The colours that actually reach the screen for `input`, as
    /// `(char, fg)` for the drawn input row.
    ///
    /// Goes through `render_input` and a real ratatui backend rather than
    /// calling `input_spans` directly: the span-to-cell path in `wrap_input`
    /// is exactly where a style can be dropped without any unit test noticing.
    fn drawn_input_colors(input: &str) -> Vec<(char, Color)> {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let mut term = Terminal::new(TestBackend::new(60, 1)).unwrap();
        term.draw(|f| {
            render_input(f, Rect::new(0, 0, 60, 1), InputState::new(input, 0));
        })
        .unwrap();
        let buf = term.backend().buffer().clone();
        (0..60)
            .map(|x| {
                let cell = &buf[(x, 0)];
                (cell.symbol().chars().next().unwrap_or(' '), cell.fg)
            })
            .collect()
    }

    #[test]
    fn the_subagent_name_reaches_the_screen_in_its_colour() {
        crate::agents::set_roster_for_test(&["drawnreviewer"]);

        // The colour of the `r` in `:drawnreviewer` is the whole question:
        // green when the definition exists, red when it does not.
        let known = drawn_input_colors("/subagent:drawnreviewer check the diff");
        let unknown = drawn_input_colors("/subagent:nosuchagent check the diff");
        let name_cell = |cells: &[(char, Color)], name: &str| -> Color {
            let text: String = cells.iter().map(|&(c, _)| c).collect();
            let at = text.find(name).expect("the name is drawn");
            cells[at].1
        };
        assert_eq!(name_cell(&known, "drawnreviewer"), THEME_GREEN);
        assert_eq!(name_cell(&unknown, "nosuchagent"), Color::Red);
        // The command half stays green in both cases: the command is valid
        // either way, and only the name is in question.
        assert_eq!(name_cell(&known, "subagent"), THEME_GREEN);
        assert_eq!(name_cell(&unknown, "subagent"), THEME_GREEN);
        // The task text is not coloured at all.
        assert_eq!(name_cell(&known, "check"), Color::Reset);
    }

    #[test]
    fn a_known_command_reaches_the_screen_green() {
        // The plain case, drawn rather than computed — this is what catches a
        // regression between `input_spans` and the terminal.
        let cells = drawn_input_colors("/btw what is this");
        let text: String = cells.iter().map(|&(c, _)| c).collect();
        let at = text.find("/btw").expect("the command is drawn");
        assert_eq!(cells[at].1, THEME_GREEN, "drawn as: {text:?}");
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
        let (lines, row, col) = wrap_input("hello world", 8, 11, None);
        assert_eq!(lines.len(), 2);
        let row0: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(row0, "hello ");
        // Cursor at end (char 11) lands on the second row after "world".
        assert_eq!((row, col), (1, 5));
    }

    #[test]
    fn word_wrap_hard_breaks_a_too_long_token() {
        // No spaces: a hard break at the width boundary.
        let (lines, _, _) = wrap_input("abcdefgh", 4, 0, None);
        let texts: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert_eq!(texts, vec!["abcd".to_string(), "efgh".to_string()]);
    }

    /// Every cell the selection covers is reversed, and nothing outside it is —
    /// including across a wrap, where the range is expressed over the whole
    /// input but applied per visual row.
    #[test]
    fn a_selection_reverses_exactly_its_own_cells() {
        let reversed = |line: &Line<'static>| -> String {
            line.spans
                .iter()
                .filter(|s| s.style.add_modifier.contains(Modifier::REVERSED))
                .map(|s| s.content.as_ref())
                .collect()
        };
        // "hello world" wraps after "hello " at width 8; select "lo wo".
        let (lines, _, _) = wrap_input("hello world", 8, 0, Some((3, 8)));
        assert_eq!(reversed(&lines[0]), "lo ");
        assert_eq!(reversed(&lines[1]), "wo");
        // Without a selection nothing is reversed at all.
        let (plain, _, _) = wrap_input("hello world", 8, 0, None);
        assert_eq!(reversed(&plain[0]), "");
        assert_eq!(reversed(&plain[1]), "");
    }

    /// The selection highlight rides on top of the command highlighting rather
    /// than replacing it: a selected `/help` stays green *and* reverses.
    #[test]
    fn a_selection_over_a_command_keeps_the_command_colour() {
        let (lines, _, _) = wrap_input("/help", 20, 0, Some((0, 5)));
        for span in &lines[0].spans {
            assert!(span.style.add_modifier.contains(Modifier::REVERSED));
            assert_eq!(span.style.fg, Some(THEME_GREEN));
        }
    }

    #[test]
    fn a_click_maps_to_the_char_under_it() {
        let area = Rect::new(4, 10, 8, 2);
        // Row 0 is "hello ", row 1 is "world".
        assert_eq!(input_hit(area, "hello world", 4, 10), Some(0));
        assert_eq!(input_hit(area, "hello world", 8, 10), Some(4));
        assert_eq!(input_hit(area, "hello world", 6, 11), Some(8));
    }

    #[test]
    fn a_click_past_a_row_lands_on_its_end_and_off_the_prompt_misses() {
        let area = Rect::new(4, 10, 8, 2);
        // Right of the text on row 0: the end of that wrapped segment.
        assert_eq!(input_hit(area, "hello world", 11, 10), Some(6));
        // Left of the text area (on the prompt glyph): the row's start.
        assert_eq!(input_hit(area, "hello world", 0, 11), Some(6));
        // A different row entirely: not the prompt.
        assert_eq!(input_hit(area, "hello world", 6, 3), None);
        assert_eq!(input_hit(area, "hello world", 6, 12), None);
    }

    #[test]
    fn a_click_below_the_last_row_clamps_to_the_end_of_the_text() {
        // A three-row area holding one row of text: the spare rows clamp.
        let area = Rect::new(0, 0, 10, 3);
        assert_eq!(input_hit(area, "abc", 0, 2), Some(3));
    }

    #[test]
    fn a_click_lands_on_a_wide_char_rather_than_between_its_columns() {
        let area = Rect::new(0, 0, 10, 1);
        // "a漢b": 漢 occupies columns 1 and 2.
        assert_eq!(input_hit(area, "a漢b", 1, 0), Some(1));
        assert_eq!(input_hit(area, "a漢b", 2, 0), Some(1));
        assert_eq!(input_hit(area, "a漢b", 3, 0), Some(2));
    }

    #[test]
    fn the_slash_menu_columns_never_starve_the_descriptions() {
        let short = vec!["/new".to_string(), "/help".to_string()];
        let (cmd, desc) = slash_columns(&short, 40);
        assert_eq!(cmd, 5, "sized to the widest label");
        assert_eq!(desc, 33);
        // A label wider than half the row is capped rather than taking it all.
        let long = vec!["/verylongcommand [with args]".to_string()];
        let (cmd, desc) = slash_columns(&long, 40);
        assert_eq!(cmd, 20);
        assert_eq!(desc, 18);
    }

    #[test]
    fn eliding_right_keeps_the_start_and_marks_the_cut() {
        assert_eq!(elide_right("abcdef", 10), "abcdef");
        assert_eq!(elide_right("abcdef", 4), "abc…");
        assert_eq!(elide_right("abcdef", 1), "…");
        assert_eq!(elide_right("abcdef", 0), "");
    }

    #[test]
    fn the_slash_menu_draws_the_command_and_its_description() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let menu = crate::slashmenu::SlashMenu::new(
            vec![crate::slashmenu::Entry {
                name: "/compact".into(),
                args: "[instructions]".into(),
                desc: "summarize the transcript".into(),
                source: crate::slashmenu::Source::Builtin,
            }],
            "",
        );
        let mut term = Terminal::new(TestBackend::new(60, 6)).unwrap();
        term.draw(|f| {
            draw(
                f,
                &OutputLog::new(),
                Some(InputState::new("/comp", 5)),
                "idle",
                &mut OutputView::default(),
                None,
                &TaskView::default(),
                None,
            );
            draw_slash_menu(f, "/comp", &menu);
        })
        .unwrap();
        let text: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(text.contains("/compact"), "{text}");
        assert!(text.contains("summarize the transcript"), "{text}");
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
    fn selection_screen_projects_and_clamps() {
        // Fully visible: rows shift down by `top`, columns untouched.
        assert_eq!(
            selection_screen(((2, 5), (4, 7)), 3, 10),
            Some(((2, 2), (4, 4)))
        );
        // Start above the viewport clamps to the top-left (full first row).
        assert_eq!(
            selection_screen(((1, 1), (3, 8)), 5, 10),
            Some(((0, 0), (3, 3)))
        );
        // End below the viewport clamps to the bottom-right sentinel.
        assert_eq!(
            selection_screen(((2, 5), (9, 30)), 0, 10),
            Some(((2, 5), (u16::MAX, 9)))
        );
        // Endpoints given out of order are normalized.
        assert_eq!(
            selection_screen(((4, 7), (2, 5)), 3, 10),
            Some(((2, 2), (4, 4)))
        );
        // Entirely above or below the viewport yields nothing.
        assert_eq!(selection_screen(((0, 0), (2, 2)), 10, 5), None);
        assert_eq!(selection_screen(((0, 20), (2, 22)), 0, 10), None);
    }

    #[test]
    fn selection_text_content_reads_across_rows() {
        let mut log = OutputLog::new();
        log.push_plain("hello");
        log.push_plain("world");
        log.push_plain("foobar");

        // A multi-row selection: full first/middle rows, partial last row.
        assert_eq!(
            selection_text_content(&log, 20, ((0, 0), (4, 2))),
            "hello\nworld\nfooba"
        );
        // A single-row partial selection.
        assert_eq!(selection_text_content(&log, 20, ((1, 0), (3, 0))), "ell");
        // A row past the end clamps rather than panicking.
        assert_eq!(
            selection_text_content(&log, 20, ((0, 0), (4, 99))),
            "hello\nworld\nfooba"
        );
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
    fn jump_hint_rect_tracks_follow_state() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut log = OutputLog::new();
        for i in 0..50 {
            log.visible_text(&format!("line {i}"));
            log.end_line();
        }
        // Scrolled up (not following): the hint is drawn and its rect recorded
        // at the bottom-right of the area, so the mouse handler can hit-test it.
        let mut view = OutputView {
            top: 0,
            follow: false,
            jump_hint_rect: None,
        };
        let mut term = Terminal::new(TestBackend::new(40, 6)).unwrap();
        term.draw(|f| {
            let area = f.area();
            render_output(f, area, &log, &mut view, None);
        })
        .unwrap();
        let r = view
            .jump_hint_rect
            .expect("hint rect set while scrolled up");
        assert_eq!(r.y, 5, "hint on the bottom row");
        assert_eq!(r.x + r.width, 40, "hint flush to the right edge");
        assert_eq!(r.height, 1);

        // Following the bottom: no hint, rect cleared.
        view.follow = true;
        term.draw(|f| {
            let area = f.area();
            render_output(f, area, &log, &mut view, None);
        })
        .unwrap();
        assert!(
            view.jump_hint_rect.is_none(),
            "hint hidden while following the newest output"
        );
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
    fn streaming_bursts_are_throttled_but_flush_completely() {
        // Regression: a code block used to be re-highlighted on every streamed
        // token — the markdown crate recompiles a tree-sitter query per render,
        // so an N-token block cost N expensive renders and wedged the TUI.
        // A same-segment burst must collapse to few renders, and a boundary
        // (EndLine) must still commit every token.
        let mut log = OutputLog::new();
        let toks = [
            "```rust\n",
            "fn ",
            "main",
            "() ",
            "{\n",
            "    let x = 1;\n",
            "}\n",
            "```\n",
        ];
        for t in toks {
            log.visible_text(t);
        }
        // The whole burst runs in far under MD_RENDER_MIN_GAP, so only the
        // first token renders eagerly; the rest defer behind the throttle.
        assert!(
            log.renders < toks.len(),
            "expected throttled renders, got {} for {} tokens",
            log.renders,
            toks.len()
        );
        assert!(
            log.md_dirty,
            "deferred tokens should leave the buffer dirty"
        );

        // The end-of-segment flush commits the full, highlighted block.
        log.end_line();
        assert!(!log.md_dirty, "flush clears the dirty flag");
        let joined: String = log
            .lines
            .iter()
            .flat_map(|l| &l.spans)
            .map(|s| s.content.as_ref())
            .collect();
        assert!(
            joined.contains("let x = 1;"),
            "final render must contain every streamed token: {joined:?}"
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
    fn code_copy_preserves_long_lines_across_soft_wrap() {
        // A single source line longer than the (test-default 80) render width
        // is soft-wrapped into several rendered rows. Copying must yield the
        // original one line, not the wrapped rows joined by newlines.
        let mut log = OutputLog::new();
        let source = "x".repeat(200);
        log.visible_text(&format!("```bash\n{source}\n```\n"));
        assert_eq!(log.code_blocks.len(), 1, "one block recorded");
        assert_eq!(log.code_blocks[0].code, source);
        assert!(
            !log.code_blocks[0].code.contains('\n'),
            "soft-wrap must not introduce newlines"
        );
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

    /// Both status rows flattened into one span list, for assertions about
    /// content rather than placement.
    fn status_spans(text: &str, tick_ms: u64, base: Style, tasks: &TaskView) -> Vec<Span<'static>> {
        status_bar_lines(text, tick_ms, base, tasks)
            .into_iter()
            .flat_map(|l| l.spans)
            .collect()
    }

    #[test]
    fn status_bar_counter_is_themed_in_flight_and_dim_when_done() {
        use crate::tasks::TaskStatus::{Completed, InProgress};
        let base = Style::default();
        // An empty list adds no task counter to the status bar (the "✓ n/n"
        // segment); rotating tips may still contribute other spans.
        let empty = status_spans("idle", 0, base, &TaskView::default());
        assert!(!empty.iter().any(|s| s.content.contains('✓')));

        // In flight: the counter carries the theme color.
        let theme = Color::Indexed(crate::status::THEME_COLOR);
        let tv = task_view_with(&[("a", Completed), ("b", InProgress)]);
        let line = status_spans("idle", 0, base, &tv);
        let counter = line
            .iter()
            .find(|s| s.content.contains("1/2"))
            .expect("counter span present");
        assert_eq!(counter.style.fg, Some(theme));

        // Fully complete: the counter goes dim gray, not theme.
        let tv = task_view_with(&[("a", Completed)]);
        let line = status_spans("idle", 0, base, &tv);
        let counter = line.iter().find(|s| s.content.contains("1/1")).unwrap();
        assert_eq!(counter.style.fg, Some(Color::Indexed(240)));
    }

    /// With the think segment present, the branch must still end at the branch:
    /// `push_dir_prefix` splits on the *last* " | ", so if the segment were left
    /// inside the prefix slice the branch would read "main | 🧠 medium".
    #[test]
    fn status_bar_keeps_the_think_segment_out_of_the_branch() {
        let base = Style::default();
        let theme = Color::Indexed(crate::status::THEME_COLOR);
        let glyph = crate::status::POWERLINE_BRANCH;
        let mark = crate::status::THINK_MARK;
        let text = format!("~/Code/plank {glyph} main | {mark} max | ctx 12% | idle");
        let line = status_spans(&text, 0, base, &TaskView::default());

        let branch = line
            .iter()
            .find(|s| s.content == "main")
            .expect("branch span ends at the branch");
        assert_eq!(branch.style.fg, Some(theme));

        // The segment renders as its own plain span, and only once.
        let think: Vec<_> = line.iter().filter(|s| s.content.contains(mark)).collect();
        assert_eq!(think.len(), 1, "{line:?}");
        assert_eq!(think[0].content, format!("{mark} max"));
        assert_eq!(think[0].style.fg, None, "plain, like the ctx gauge");

        // Nothing is dropped: the rendered spans still spell the input.
        let joined: String = line.iter().map(|s| s.content.as_ref()).collect();
        assert!(
            joined.contains(&format!("{mark} max | ctx 12%")),
            "{joined}"
        );
    }

    /// The shell-escape marker is colored by *consequence*, which is the one
    /// thing the two forms differ in and the one thing invisible once typed:
    /// red `!` feeds the command and its output to the model, green `!!` keeps
    /// it between the user and the shell. Only the marker is colored.
    #[test]
    fn the_bang_marker_is_colored_by_where_its_output_goes() {
        let colored = |input: &str| -> Vec<(String, Option<Color>)> {
            input_spans(input)
                .into_iter()
                .map(|s| (s.content.to_string(), s.style.fg))
                .collect()
        };

        assert_eq!(
            colored("!ls -la"),
            vec![
                ("!".to_string(), Some(Color::Red)),
                ("ls -la".to_string(), None),
            ],
            "a single bang reaches the model, so it is the loud one"
        );
        assert_eq!(
            colored("!!ls -la"),
            vec![
                ("!!".to_string(), Some(THEME_GREEN)),
                ("ls -la".to_string(), None),
            ],
            "a double bang stays local"
        );
        // Bare markers still color, so the cue appears on the first keystroke.
        assert_eq!(colored("!"), vec![("!".to_string(), Some(Color::Red))]);
        assert_eq!(colored("!!"), vec![("!!".to_string(), Some(THEME_GREEN))]);
    }

    /// The turn footer always shows all three units, so consecutive turns line
    /// up when scrolling back through a session.
    #[test]
    fn the_turn_footer_reads_as_one_shape() {
        use std::time::Duration;
        assert_eq!(fmt_turn_duration(Duration::from_secs(0)), "0h 00m 00s");
        assert_eq!(fmt_turn_duration(Duration::from_secs(9)), "0h 00m 09s");
        assert_eq!(fmt_turn_duration(Duration::from_secs(247)), "0h 04m 07s");
        assert_eq!(fmt_turn_duration(Duration::from_secs(3729)), "1h 02m 09s");
        assert_eq!(fmt_turn_duration(Duration::from_hours(24)), "24h 00m 00s");
        assert_eq!(
            turn_footer(Duration::from_secs(3729)),
            "\u{273b} Planked for 1h 02m 09s"
        );
        // Bold grey: findable at a turn boundary without competing with output.
        let st = turn_footer_style();
        assert_eq!(st.fg, Some(Color::Indexed(245)));
        assert!(st.add_modifier.contains(Modifier::BOLD));
    }

    /// The plug renders through the same path as any other tip: same 💡 prefix,
    /// same yellow-bold styling, same slot. "Shows like one" is the point — an
    /// advertisement that looked different would read as an advertisement.
    #[test]
    fn the_promo_tip_renders_exactly_like_a_tip() {
        let base = Style::default();
        let text = format!("~/x | {} med | ctx 12% | idle", crate::status::THINK_MARK);
        let tip_spans = |rotation: u64| -> Vec<(String, Option<Color>, bool)> {
            let tick = crate::status::TIP_ROTATE_MS * rotation;
            status_bar_lines(&text, tick, base, &TaskView::default())
                .into_iter()
                .flat_map(|l| l.spans)
                .filter(|s| s.content.contains('💡'))
                .map(|s| {
                    (
                        s.content.to_string(),
                        s.style.fg,
                        s.style.add_modifier.contains(Modifier::BOLD),
                    )
                })
                .collect()
        };

        let ordinary = tip_spans(1);
        let promo = tip_spans(crate::status::PROMO_EVERY);
        assert_eq!(ordinary.len(), 1, "an ordinary rotation shows one tip");
        assert_eq!(promo.len(), 1, "so does the promo rotation");
        assert!(promo[0].0.contains("free-tokens"), "{}", promo[0].0);
        assert_eq!(
            (promo[0].1, promo[0].2),
            (ordinary[0].1, ordinary[0].2),
            "same colour and weight as a real tip"
        );
        assert!(promo[0].0.starts_with("💡 "), "same prefix: {}", promo[0].0);
    }

    /// A fenced code block that opens on the line directly after a paragraph —
    /// no blank line between, which `CommonMark` §4.5 allows and which is how
    /// prose normally introduces a code sample — must still render *after* that
    /// paragraph.
    ///
    /// `ratatui-markdown` 0.3.6 parsed the block ahead of the paragraph, so an
    /// assistant message showed its code sample above the line introducing it.
    /// Fixed in the pinned fork; this pins the behaviour plank depends on, so a
    /// future dependency bump that regresses it fails here rather than on
    /// screen.
    #[test]
    fn a_fence_after_a_paragraph_renders_below_it() {
        let mut log = OutputLog::new();
        // Through the real streaming sink, the way a turn feeds it.
        crate::viz::RenderSink::visible_text(&mut log, "Run it with:\n```sh\ncd local\n```\n");
        log.end_line();

        let rows: Vec<String> = log
            .to_text()
            .lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.to_string()).collect())
            .collect();
        let intro = rows
            .iter()
            .position(|r| r.contains("Run it with:"))
            .unwrap_or_else(|| panic!("intro line missing: {rows:?}"));
        let code = rows
            .iter()
            .position(|r| r.contains("cd local"))
            .unwrap_or_else(|| panic!("code line missing: {rows:?}"));
        assert!(
            intro < code,
            "paragraph must precede its code block: {rows:?}"
        );
    }

    /// The brain blinks only while a local pass is in flight, and the dark half
    /// is the glyph's own width in spaces so the bar never reflows. This is the
    /// only signal that says *which* engine is working, so it has to hold still
    /// when nothing local is running and actually alternate when something is.
    ///
    /// Asserts on the glyph's presence and the row's width, not on a color:
    /// `THINK_MARK` is a color emoji, and an earlier version that changed its
    /// foreground passed a color assertion while the screen showed nothing.
    ///
    /// The phase comes from the pass's own elapsed time, not from `tick_ms`, so
    /// the sweep here moves the pass clock rather than the animation clock.
    #[test]
    fn the_brain_blinks_only_while_a_local_pass_runs() {
        let base = Style::default();
        let mark = crate::status::THINK_MARK;
        let text = format!("~/x | {mark} med | ctx 12% | generating");
        let rows = || -> Vec<String> {
            status_bar_lines(&text, 0, base, &TaskView::default())
                .into_iter()
                .map(|l| l.spans.iter().map(|sp| sp.content.to_string()).collect())
                .collect()
        };
        let brain_showing = || rows().iter().any(|r| r.contains(mark));
        // Every rendering must occupy the same columns, blinked or not.
        let widths = |r: &[String]| -> Vec<usize> { r.iter().map(|l| l.width()).collect() };
        let reference = widths(&rows());

        // Idle: showing, whatever the clock is doing.
        assert!(!crate::status::local_pass_active());
        assert!(brain_showing(), "no blink when nothing local is running");

        // A local pass in flight: both phases appear across one cycle, and it
        // starts showing — the point of phasing off the pass's own clock.
        {
            let _guard = crate::status::LocalPass::begin();
            assert!(crate::status::local_pass_active());
            assert!(brain_showing(), "a pass starts with the brain showing");
            let phases: Vec<bool> = (0..8u64)
                .map(|step| {
                    crate::status::set_local_pass_ms(step * crate::status::BRAIN_BLINK_MS / 8);
                    assert_eq!(widths(&rows()), reference, "the bar holds its columns");
                    brain_showing()
                })
                .collect();
            assert!(phases.contains(&false), "{phases:?}");
            assert!(phases.contains(&true), "{phases:?}");

            // Reduced motion collapses the blink to its static form like every
            // other effect: showing, even mid-cycle where it would otherwise be
            // dark. Asserted here rather than in a test of its own — both the
            // reduced-motion toggle and the local-pass flag are process-global,
            // so two tests holding them would race under the default harness.
            crate::status::set_local_pass_ms(crate::status::BRAIN_BLINK_MS * 3 / 4);
            assert!(!brain_showing(), "the phase used for the check");
            crate::anim::set_reduced_motion(true);
            let still_showing = brain_showing();
            crate::anim::set_reduced_motion(false);
            assert!(still_showing, "reduced motion holds the brain showing");

            // And end-to-end through a real terminal buffer, which is the only
            // place the property that matters is visible: the dark half leaves
            // the emoji's cells genuinely blank and everything after them
            // exactly where it was. Folded in here rather than given a test of
            // its own because the local-pass flag is process-global.
            let rendered = |ms: u64| -> String {
                crate::status::set_local_pass_ms(ms);
                let mut term =
                    ratatui::Terminal::new(ratatui::backend::TestBackend::new(60, 2)).unwrap();
                term.draw(|f| {
                    f.render_widget(
                        ratatui::widgets::Paragraph::new(status_bar_lines(
                            &text,
                            0,
                            base,
                            &TaskView::default(),
                        )),
                        f.area(),
                    );
                })
                .unwrap();
                let buf = term.backend().buffer().clone();
                (0..buf.area.height)
                    .map(|y| {
                        (0..buf.area.width)
                            .map(|x| buf[(x, y)].symbol().to_string())
                            .collect::<String>()
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            let on = rendered(0);
            let off = rendered(crate::status::BRAIN_BLINK_MS * 3 / 4);
            assert!(on.contains(mark), "the lit half draws the brain: {on:?}");
            assert!(!off.contains(mark), "the dark half does not: {off:?}");
            assert_eq!(
                on.chars().count(),
                off.chars().count(),
                "only the glyph's own cells changed"
            );
            assert!(off.contains("med | ctx 12%"), "{off:?}");
        }

        // And the guard's drop ends it, so a finished pass cannot leave the bar
        // blinking forever.
        assert!(!crate::status::local_pass_active());
        assert!(brain_showing());
    }

    /// The bar is two rows: row one is the directory and branch and nothing
    /// else, row two opens with the engine origin and carries everything
    /// volatile. The origin moved rows deliberately — row one has to hold still
    /// while the rest churns — so pin where each piece lands.
    #[test]
    fn status_bar_splits_location_from_everything_volatile() {
        let base = Style::default();
        let theme = Color::Indexed(crate::status::THEME_COLOR);
        let glyph = crate::status::POWERLINE_BRANCH;
        let _guard = crate::status::origin_test_guard();
        let origin = crate::status::engine_origin_label();
        let text = format!("~/Code/plank {glyph} main | {origin} | ctx 12% | idle");
        let rows = status_bar_lines(&text, 0, base, &TaskView::default());
        assert_eq!(rows.len(), 2, "two rows");

        let row =
            |i: usize| -> String { rows[i].spans.iter().map(|s| s.content.as_ref()).collect() };

        // Row one: the location, themed, with the glyph plain — and no origin,
        // no gauge, no state.
        assert_eq!(row(0), format!("~/Code/plank {glyph} main"));
        let branch = rows[0]
            .spans
            .iter()
            .find(|s| s.content == "main")
            .expect("branch span");
        assert_eq!(branch.style.fg, Some(theme));
        assert!(
            !row(0).contains(origin.as_str()),
            "origin is not on row one: {}",
            row(0)
        );
        assert!(!row(0).contains("ctx "), "no gauge on row one: {}", row(0));

        // Row two: origin first, then the rest, in the old order.
        assert!(
            row(1).starts_with(&format!("{origin} | ctx 12%")),
            "row two: {}",
            row(1)
        );
        let shown = rows[1]
            .spans
            .iter()
            .find(|s| s.content.contains(origin.as_str()))
            .expect("origin span");
        assert_eq!(shown.style.fg, None, "plain, like the ctx gauge");
    }

    #[test]
    fn status_bar_themes_path_and_branch_but_not_the_powerline_glyph() {
        let base = Style::default();
        let theme = Color::Indexed(crate::status::THEME_COLOR);
        let glyph = crate::status::POWERLINE_BRANCH;
        let text = format!("~/Code/plank {glyph} main | ctx 12% | idle");
        let line = status_spans(&text, 0, base, &TaskView::default());

        let path = line
            .iter()
            .find(|s| s.content == "~/Code/plank")
            .expect("path span");
        assert_eq!(path.style.fg, Some(theme));

        let branch = line
            .iter()
            .find(|s| s.content == "main")
            .expect("branch span");
        assert_eq!(branch.style.fg, Some(theme));

        // The powerline glyph is not themed green.
        let glyph_span = line
            .iter()
            .find(|s| s.content.contains(glyph))
            .expect("glyph span");
        assert_ne!(glyph_span.style.fg, Some(theme));
    }

    /// Renders the two rows into a real buffer, since passing geometry is not
    /// the same as looking right: row one must carry the location and row two
    /// the gauge and state, with nothing bleeding across.
    #[test]
    fn status_rows_render_location_above_state() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let glyph = crate::status::POWERLINE_BRANCH;
        let _guard = crate::status::origin_test_guard();
        let origin = crate::status::engine_origin_label();
        let text = format!("~/Code/plank {glyph} main | {origin} | ctx 12% | idle");
        let mut term = Terminal::new(TestBackend::new(70, 2)).unwrap();
        term.draw(|f| {
            let rows = status_bar_lines(&text, 0, Style::default(), &TaskView::default());
            f.render_widget(ratatui::widgets::Paragraph::new(rows), f.area());
        })
        .unwrap();
        let buf = term.backend().buffer();
        let row = |y: u16| -> String {
            (0..buf.area.width)
                .map(|x| buf[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_string()
        };
        assert_eq!(row(0), format!("~/Code/plank {glyph} main"));
        // Compared in pieces rather than against `origin` verbatim: the origin
        // carries the local engine's ⚡ power badge, and ⚡ (U+26A1) has
        // Emoji_Presentation, so the buffer holds it as one wide cell plus a
        // blank continuation. Reading cells back therefore yields a space the
        // source string does not contain.
        //
        // `starts_with`: the tail notification slot (a rotating tip, or a
        // running tool's label) also lives on row two, after the state.
        let head = origin.split('⚡').next().unwrap_or_default();
        assert!(row(1).starts_with(head), "row two: {}", row(1));
        assert!(row(1).contains("| ctx 12% | idle"), "row two: {}", row(1));
    }

    #[test]
    fn frame_geom_reserves_strip_rows_only_when_present() {
        let area = Rect::new(0, 0, 80, 24);
        // No strip: the top rule sits directly above the input, the bottom
        // rule directly below it (above the status bar).
        let (out0, in0, st0, rule0, rule_bot0, strip0) = frame_geom(area, true, 1, 0);
        assert!(strip0.is_none());
        // The status bar is two rows: location above, everything volatile below.
        // The literal, not `STATUS_ROWS` — comparing the layout against the
        // constant that produced it asserts nothing.
        assert_eq!(st0.height, 2, "status bar is two rows");
        assert_eq!(st0.bottom(), area.bottom(), "and it sits at the bottom");
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
        // Any shade of the ramp counts as highlighted: the window is graded, so
        // no single color covers the whole highlight any more.
        let shades: Vec<Color> = crate::status::SHIMMER_RAMP
            .iter()
            .map(|&i| Color::Indexed(i))
            .collect();
        // Collect the shimmer segment text at each step of one full cycle.
        let text = "◆ Pondering… 3s";
        let mut highlights = Vec::new();
        for step in 0..40u64 {
            let line = status_spans(
                text,
                step * crate::status::SHIMMER_STEP_MS,
                base,
                &TaskView::default(),
            );
            let hit: String = line
                .iter()
                .filter(|s| s.style.fg.is_some_and(|c| shades.contains(&c)))
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

    // The highlight is a gradient, not a flat block: when it sits fully on the
    // word, every shade of the ramp is on screen at once and the brightest one
    // is in the middle. A regression to a single flat color (white or otherwise)
    // fails here even though the sweep would still animate.
    #[test]
    fn verb_shimmer_is_graded_not_a_flat_block() {
        let base = Style::default();
        let ramp = crate::status::SHIMMER_RAMP;
        let text = "◆ Pondering… 3s";
        let brightest = Color::Indexed(ramp[ramp.len() - 1]);
        let dimmest = Color::Indexed(ramp[0]);
        let lines: Vec<_> = (0..40u64)
            .map(|step| {
                status_spans(
                    text,
                    step * crate::status::SHIMMER_STEP_MS,
                    base,
                    &TaskView::default(),
                )
            })
            .collect();

        // Mid-sweep the window sits wholly inside the word, and there the bright
        // center is flanked by the dimmest shade on *both* sides. (At the ends of
        // the travel the window overhangs the word, so only part of the ramp is
        // on screen — which is why this looks for the interior step.)
        let flanked = lines.iter().any(|line| {
            let first = line.iter().position(|s| s.style.fg == Some(dimmest));
            let bright = line.iter().position(|s| s.style.fg == Some(brightest));
            let last = line.iter().rposition(|s| s.style.fg == Some(dimmest));
            match (first, bright, last) {
                (Some(f), Some(b), Some(l)) => f < b && b < l,
                _ => false,
            }
        });
        assert!(
            flanked,
            "no step showed a bright center flanked by dimmer shades; \
             the highlight is not graded"
        );

        // And no pure white at any point in the cycle.
        for line in &lines {
            assert!(
                !line.iter().any(|s| {
                    s.style.fg == Some(Color::Indexed(231)) || s.style.fg == Some(Color::White)
                }),
                "the shimmer uses shades of the theme hue, never pure white"
            );
        }
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
                Some(InputState::new("hi", 2)),
                "idle",
                &mut view,
                None,
                &TaskView::default(),
                None,
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
                Some(InputState::new("hi", 2)),
                "idle",
                &mut view,
                None,
                &TaskView::default(),
                None,
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
                "generating",
                &mut view,
                None,
                &TaskView::default(),
                None,
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
                Some(InputState::new("aa\nbb", 5)),
                "idle",
                &mut view,
                None,
                &TaskView::default(),
                None,
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

    /// An arcade with `cmd` on screen.
    fn opened(cmd: &str) -> crate::arcade::Arcade {
        let mut a = crate::arcade::Arcade::new();
        assert!(a.open(cmd, false, 7, 80, 24), "{cmd} did not open");
        a
    }

    /// An arcade showing the starfield screensaver.
    ///
    /// The face is asked for rather than left to `ui.screensaverFace`: a test
    /// about the starfield that silently drew one of the other faces would be
    /// worse than no test.
    fn screensaver() -> crate::arcade::Arcade {
        let mut a = crate::arcade::Arcade::new();
        a.open_screensaver_as(crate::arcade::ScreensaverFace::Starfield, 7, 80, 24);
        a
    }

    /// An arcade showing the minions screensaver, three seconds in.
    ///
    /// Both numbers matter: they fade up from the dark, so frame zero is
    /// deliberately empty, and the seed decides where along its crossing the
    /// pair is — this one has them well inside an 80-column frame rather than
    /// half off the edge.
    fn minions_screensaver() -> crate::arcade::Arcade {
        let mut a = crate::arcade::Arcade::new();
        a.open_screensaver_as(crate::arcade::ScreensaverFace::Minions, 4, 80, 24);
        for _ in 0..60 {
            a.step(50);
        }
        a
    }

    /// The screensaver paints its own background, and it has to be a real
    /// black. `Color::Black` is ANSI index 0, which terminal themes are free to
    /// remap — most render it as a dark grey — so the starfield came up on grey
    /// instead of black. Only an explicit RGB triple is actually black.
    #[test]
    fn the_screensaver_background_is_true_black() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let arcade = screensaver();
        let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
        term.draw(|f| draw_arcade(f, &arcade)).unwrap();
        let buf = term.backend().buffer().clone();

        // Every cell of the play area, star or gap, sits on the same black.
        for y in 0..23 {
            for x in 0..80 {
                let cell = buf.cell(Position::new(x, y)).unwrap();
                assert_eq!(
                    cell.bg,
                    Color::Rgb(0, 0, 0),
                    "cell ({x},{y}) is not true black: {:?}",
                    cell.bg
                );
            }
        }
    }

    /// Draws one arcade frame and returns it as plain rows of text.
    fn arcade_frame(arcade: &crate::arcade::Arcade, w: u16, h: u16) -> Vec<String> {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        term.draw(|f| draw_arcade(f, arcade)).unwrap();
        let buf = term.backend().buffer().clone();
        (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| buf.cell(Position::new(x, y)).unwrap().symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_owned()
            })
            .collect()
    }

    #[test]
    fn arcade_draws_the_pelota_field_and_scoreboard() {
        let rows = arcade_frame(&opened("/pelota"), 80, 26);
        let field = rows.join("\n");
        assert!(field.contains('█'), "paddles missing:\n{field}");
        assert!(field.contains('●'), "ball missing:\n{field}");
        assert!(field.contains('─'), "walls missing:\n{field}");
        // The scoreboard owns the bottom row, outside the playing field.
        let footer = rows.last().unwrap();
        assert!(footer.contains("level 1/5"), "footer was {footer:?}");
        assert!(footer.contains("you 0 — 0 cpu"), "footer was {footer:?}");
    }

    #[test]
    fn arcade_tells_you_when_the_terminal_is_too_small() {
        let rows = arcade_frame(&opened("/pelota"), 20, 6);
        let field = rows.join("\n");
        assert!(!field.contains('█'), "field drawn anyway:\n{field}");
        assert!(field.contains("terminale"), "no explanation:\n{field}");
    }

    /// The translucent layer is the one piece with no equivalent elsewhere in
    /// the UI, so it is worth pinning: text underneath must survive, and it
    /// must come back dimmed rather than at full brightness.
    #[test]
    fn a_veiled_arcade_leaves_the_ui_visible_underneath() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut log = OutputLog::new();
        for i in 0..40 {
            log.visible_text(&format!("output line {i} still readable"));
            log.end_line();
        }
        let draw = |translucent: bool| {
            let mut arcade = screensaver();
            arcade.translucent = translucent;
            let mut view = OutputView::default();
            let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
            term.draw(|f| {
                render_output(f, f.area(), &log, &mut view, None);
                draw_arcade(f, &arcade);
            })
            .unwrap();
            term.backend().buffer().clone()
        };

        let veiled = draw(true);
        let opaque = draw(false);
        let text_of = |buf: &Buffer| {
            (0..24)
                .map(|y| {
                    (0..80)
                        .map(|x| buf.cell(Position::new(x, y)).unwrap().symbol())
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert!(
            text_of(&veiled).contains("still readable"),
            "the veil erased the output underneath"
        );
        assert!(
            !text_of(&opaque).contains("still readable"),
            "the opaque layer let output through"
        );
        // Nothing underneath keeps the terminal's default foreground: every
        // cell the veil touched was pushed back, or it would outshine the sky.
        let bright = (0..23)
            .flat_map(|y| (0..80).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                let cell = veiled.cell(Position::new(x, y)).unwrap();
                cell.symbol() != " " && cell.fg == Color::Reset
            })
            .count();
        assert_eq!(bright, 0, "{bright} cells shone through the veil");
    }

    /// The exit hint is the one thing on screen that must never be squeezed
    /// out — without it a full-screen takeover has no visible way back.
    #[test]
    fn the_exit_hint_survives_any_terminal_width() {
        let exit = crate::arcade::Arcade::EXIT_HINT;
        for cmd in crate::arcade::Arcade::COMMANDS {
            let mut a = opened(cmd);
            a.translucent = true;
            for width in [8u16, 12, 20, 40, 80, 200] {
                let line = arcade_footer_line(&a, width);
                assert!(
                    line.width() <= usize::from(width),
                    "{cmd} at {width}: line is {} wide",
                    line.width()
                );
                if usize::from(width) > exit.width() + 2 {
                    assert!(
                        line.ends_with(exit),
                        "{cmd} at {width}: lost the exit hint in {line:?}"
                    );
                }
            }
            // With room to spare the status is there too, not just the exit.
            let wide = arcade_footer_line(&a, 200);
            assert!(wide.trim_start().starts_with(|c: char| c != ' '));
            assert_eq!(wide.width(), 200);
        }
    }

    #[test]
    fn arcade_scatters_stars_over_the_whole_frame() {
        let rows = arcade_frame(&screensaver(), 80, 24);
        // The screensaver has no footer, so every row is sky; stars should
        // reach the top and the bottom of it, not clump in a band.
        let painted: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.trim().is_empty())
            .map(|(i, _)| i)
            .collect();
        assert!(painted.len() > 10, "only {} rows had stars", painted.len());
        assert!(painted[0] < 4, "nothing near the top: {painted:?}");
        assert!(painted[painted.len() - 1] > 18, "nothing near the bottom");
        assert!(
            !rows
                .iter()
                .any(|r| r.contains("speed") || r.contains("quit")),
            "the screensaver must not offer the games' controls: {rows:?}"
        );
    }

    /// One of the screensaver's faces: two minions on a shore, standing in the
    /// lower half of the frame with sky above them and water below, and no
    /// footer of its own either.
    #[test]
    fn the_minions_stand_on_their_shore() {
        let rows = arcade_frame(&minions_screensaver(), 80, 24);
        let painted: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.trim().is_empty())
            .map(|(i, _)| i)
            .collect();
        assert!(painted.len() > 10, "only {} rows drawn", painted.len());
        // The pair itself is the widest solid thing on screen: some row in the
        // lower half has to be more than a scattering of stars.
        let body = rows[8..20]
            .iter()
            .map(|r| r.chars().filter(|c| *c == '█').count())
            .max()
            .unwrap_or(0);
        assert!(body > 6, "no minion found on the shore: {rows:?}");
        assert!(
            !rows
                .iter()
                .any(|r| r.contains("pace") || r.contains("quit")),
            "the screensaver must not offer the controls: {rows:?}"
        );
    }
}
