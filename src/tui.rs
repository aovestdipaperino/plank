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

/// One sub-agent run: its own [`OutputLog`] and scroll state (so switching
/// between runs never disturbs another's position), plus the live telemetry the
/// roster row reports — what it is doing, how long it has been at it, and what
/// it has spent.
#[derive(Debug, Default)]
pub struct AgentRun {
    /// The sub-agent's rendered output.
    pub log: OutputLog,
    /// Scroll/follow state for `log`, independent of every other pane's.
    pub view: OutputView,
    /// Display label — the agent definition's name, as the roster shows it.
    pub label: String,
    /// What the agent was asked to do, already reduced to a single line. This is
    /// what the roster row reports beside the name.
    pub task: String,
    /// Whether this run is still in flight.
    pub running: bool,
    /// Monotonic milliseconds ([`crate::anim::clock_ms`]) when the run started.
    pub started_ms: u64,
    /// Monotonic milliseconds when it finished; `None` while it runs. Frozen at
    /// the end so a finished row keeps reporting how long it actually took
    /// instead of counting on forever.
    pub ended_ms: Option<u64>,
    /// Tokens ingested and generated over this run's **completed** passes.
    pub prefill: u64,
    pub generated: u64,
    /// The in-flight pass's counters, tracked from the worker's status snapshots
    /// so a row is not silently blank for the minutes a single long pass takes.
    /// Cleared when the run ends, leaving the completed totals to speak.
    pub live: Option<LiveTokens>,
}

/// A sub-agent's in-flight pass, as the worker's status snapshots report it.
/// Mirrors the fields [`crate::status::progress_segment`] draws for the main
/// turn, so a roster row and the progress line say the same thing about the same
/// pass.
#[derive(Debug, Default, Clone, Copy)]
pub struct LiveTokens {
    /// Whether the pass is still prefilling (as opposed to generating).
    pub prefilling: bool,
    pub prefill_done: i32,
    pub prefill_total: i32,
    pub generated: i32,
}

impl AgentRun {
    /// Wall-clock milliseconds the run has been going, or took: live against
    /// `now` while running, frozen at [`Self::ended_ms`] once finished.
    #[must_use]
    pub fn elapsed_ms(&self, now: u64) -> u64 {
        self.ended_ms.unwrap_or(now).saturating_sub(self.started_ms)
    }

    /// The run's token column: `↑ done/total tokens` while a pass prefills,
    /// otherwise `↓ n tokens` counting everything generated so far — completed
    /// passes plus the one in flight. Empty before anything has been counted.
    ///
    /// Prefill is shown only *during* prefill because that is the phase where
    /// the number is the progress indicator; once tokens start coming out, what
    /// the run has produced is the interesting figure.
    #[must_use]
    pub fn tokens_text(&self) -> String {
        if let Some(live) = self.live.filter(|_| self.running) {
            if live.prefilling && live.prefill_total > 0 {
                return format!(
                    "↑ {}/{} tokens",
                    crate::status::format_ctx_size(live.prefill_done.min(live.prefill_total)),
                    crate::status::format_ctx_size(live.prefill_total)
                );
            }
            let total = self
                .generated
                .saturating_add(u64::try_from(live.generated).unwrap_or(0));
            return fmt_tokens(total);
        }
        fmt_tokens(self.generated)
    }
}

/// How many runs the roster keeps. Older *finished* runs are dropped once the
/// list is full; a running one is never evicted, because its buffer is still
/// being written to. Eight is well past what fits on screen while still letting
/// the user look back at the earlier agents of a fan-out.
const ROSTER_MAX: usize = 8;

/// The sub-agent roster: every run of the session (capped at [`ROSTER_MAX`]),
/// the cursor the user moves over it, and whether the selected run's output is
/// expanded over the main transcript.
///
/// Replaces the former single last-run-only pane: a fan-out puts several agents
/// in flight, and each needs its own buffer for their output to stay separable.
#[derive(Debug, Default)]
pub struct SubPane {
    /// Every run, oldest first — the order the roster draws them in.
    pub runs: Vec<AgentRun>,
    /// Index into `runs` of the run streaming right now, or the most recent one.
    /// Sub-agent output events carry no agent id, and the worker serialises
    /// them (one `SubStart` … `SubEnd` bracket at a time, even within a
    /// lockstep fan-out round), so "the current run" is enough to route them.
    current: usize,
    /// Roster cursor: `0` is the `main` row, `n` is `runs[n - 1]`.
    pub cursor: usize,
    /// Whether the cursor is being moved — set by `←`, cleared by `Esc`. Only
    /// then is the cursor drawn, so the roster is a quiet status readout until
    /// the user reaches for it.
    pub selecting: bool,
    /// Whether the selected run's output is on screen instead of the transcript.
    pub active: bool,
    /// Set while a `/subagent` turn is in flight: the turn's own render events
    /// are applied to the current run's log instead of the main transcript.
    pub adopt_turn: bool,
}

impl SubPane {
    /// Starts a run at `now` (monotonic ms), appending a roster row and making
    /// it the current one. A repeat `begin` for a label that is still running
    /// resumes that row rather than opening a second one for the same agent:
    /// a lockstep fan-out re-brackets each slot once per round.
    pub fn begin(&mut self, label: String, task: &str, now: u64) {
        if let Some(i) = self.runs.iter().position(|r| r.running && r.label == label) {
            self.current = i;
            return;
        }
        // Make room by dropping the oldest finished run. A running one is still
        // being written to, so it is never evicted.
        if self.runs.len() >= ROSTER_MAX
            && let Some(i) = self.runs.iter().position(|r| !r.running)
        {
            self.runs.remove(i);
            // The cursor and the current index address rows by position, so
            // both shift with the removal.
            self.current = self.current.saturating_sub(usize::from(i <= self.current));
            self.cursor = self.cursor.saturating_sub(usize::from(i < self.cursor));
        }
        self.runs.push(AgentRun {
            label,
            task: one_line(task),
            running: true,
            started_ms: now,
            ..AgentRun::default()
        });
        self.current = self.runs.len() - 1;
    }

    /// Ends the current run at `now`, leaving its buffer readable.
    pub fn end(&mut self, now: u64) {
        if let Some(run) = self.runs.get_mut(self.current) {
            run.running = false;
            run.ended_ms = Some(now);
            // Nothing is in flight any more; the completed totals speak for it.
            run.live = None;
        }
    }

    /// Adds one pass's tally to a run's spend: to the row `label` names when the
    /// emitter knew it, otherwise to the current run.
    ///
    /// A named row that has already finished is not credited — the tally would
    /// belong to a later run of the same agent, and a frozen row reporting a
    /// climbing cost reads as a bug.
    pub fn add_tokens(&mut self, label: Option<&str>, prefill: u64, generated: u64) {
        let target = match label {
            Some(label) => self.runs.iter_mut().find(|r| r.running && r.label == label),
            None => self.runs.get_mut(self.current),
        };
        if let Some(run) = target {
            run.prefill = run.prefill.saturating_add(prefill);
            run.generated = run.generated.saturating_add(generated);
            // The pass whose counters `live` was tracking is the one just folded
            // in, so dropping it here is what keeps the two from double-counting.
            run.live = None;
        }
    }

    /// Tracks the in-flight pass from a worker status snapshot.
    ///
    /// Applied only when exactly one run is going: the status line describes
    /// whichever pass the engine is running, and a fan-out has several in flight
    /// at once, so there would be no honest row to attribute it to. A fan-out's
    /// rows are fed by their own per-pass [`Self::add_tokens`] instead.
    pub fn note_status(&mut self, st: &crate::status::Status) {
        let live = match st.state {
            crate::status::WorkerState::Prefill => LiveTokens {
                prefilling: true,
                prefill_done: st.prefill_done,
                prefill_total: st.prefill_total,
                generated: 0,
            },
            crate::status::WorkerState::Generating => LiveTokens {
                prefilling: false,
                prefill_done: st.prefill_done,
                prefill_total: st.prefill_total,
                generated: st.generated,
            },
            // Any other state is between passes: leave the last figures up
            // rather than blanking the column for a frame.
            _ => return,
        };
        let mut running = self.runs.iter_mut().filter(|r| r.running);
        if let (Some(run), None) = (running.next(), running.next()) {
            run.live = Some(live);
        }
    }

    /// The current run's log, for the output events addressed to it. `None`
    /// before any run has started, in which case the caller falls back to the
    /// main transcript rather than dropping the output.
    pub fn current_log_mut(&mut self) -> Option<&mut OutputLog> {
        self.runs.get_mut(self.current).map(|r| &mut r.log)
    }

    /// The run the cursor sits on, or `None` on the `main` row.
    #[must_use]
    pub fn selected(&self) -> Option<&AgentRun> {
        self.runs.get(self.cursor.checked_sub(1)?)
    }

    /// Label of the run the pane would show: the selected one, falling back to
    /// the current one when the cursor rests on `main`.
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        self.selected()
            .or_else(|| self.runs.get(self.current))
            .map(|r| r.label.as_str())
    }

    /// Whether any sub-agent is running right now.
    #[must_use]
    pub fn running(&self) -> bool {
        self.runs.iter().any(|r| r.running)
    }

    /// Handles a `SubStart` from the worker. A nested `agent` tool call made by
    /// the model *inside* a `/subagent` turn also emits `SubStart`; honouring it
    /// would clear the outer run's output, steal its label, and (through the
    /// matching `SubEnd`) mark the still-streaming outer run as finished. While
    /// `adopt_turn` is set the outer run owns the pane, so the nested lifecycle
    /// is ignored and its output simply continues into the same buffer.
    pub fn on_sub_start(&mut self, label: String, task: &str, now: u64) {
        if self.adopt_turn {
            return;
        }
        self.begin(label, task, now);
    }

    /// Handles a `SubEnd` from the worker; the counterpart of
    /// [`Self::on_sub_start`] and ignored for the same reason.
    pub fn on_sub_end(&mut self, now: u64) {
        if self.adopt_turn {
            return;
        }
        self.end(now);
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
        match (self.active, self.selected()) {
            (true, Some(run)) => &run.log,
            _ => main,
        }
    }

    /// The scroll state the visible pane owns — the mutable counterpart of
    /// [`Self::active_log`]. Every scroll/follow gesture goes through this so
    /// the hidden pane's position is never disturbed.
    pub fn active_view<'a>(&'a mut self, main: &'a mut OutputView) -> &'a mut OutputView {
        match self.cursor.checked_sub(1).filter(|_| self.active) {
            Some(i) => match self.runs.get_mut(i) {
                Some(run) => &mut run.view,
                None => main,
            },
            None => main,
        }
    }

    /// Moves the roster cursor by `delta` rows (negative is up, toward `main`),
    /// entering selection mode. Returns `false` (changing nothing) when no
    /// sub-agent has ever run, so there is nothing to select.
    ///
    /// Moving off a row that was expanded collapses back to the transcript: the
    /// cursor and what is on screen never disagree.
    pub fn move_cursor(&mut self, delta: isize) -> bool {
        if self.runs.is_empty() {
            return false;
        }
        let last = isize::try_from(self.runs.len()).unwrap_or(isize::MAX);
        let from = isize::try_from(self.cursor).unwrap_or(0);
        // First `←` only reveals the cursor where it already rests; it does not
        // also jump a row, or the roster would twitch under the user's hand.
        let to = if self.selecting { from + delta } else { from };
        self.selecting = true;
        self.cursor = usize::try_from(to.clamp(0, last)).unwrap_or(0);
        if self.cursor != usize::try_from(from).unwrap_or(0) {
            self.active = false;
        }
        true
    }

    /// Expands the selected run's output over the transcript. Returns `false`
    /// on the `main` row (nothing to expand) — the caller leaves the key to
    /// whatever handles it next.
    pub fn expand(&mut self) -> bool {
        if self.selected().is_none() {
            return false;
        }
        self.active = true;
        true
    }

    /// Leaves the roster: collapses back to the transcript and hides the cursor.
    /// Returns whether anything changed, so `Esc` falls through when it did not.
    pub fn collapse(&mut self) -> bool {
        let changed = self.active || self.selecting;
        self.active = false;
        self.selecting = false;
        changed
    }

    /// Re-pins every run's view to its newest output. Called when the user acts
    /// on the main conversation, so a run they had scrolled back through is not
    /// still frozen mid-buffer the next time they look at it.
    pub fn follow_all(&mut self) {
        for run in &mut self.runs {
            run.view.follow = true;
        }
    }

    /// Snapshots the roster for drawing at `now` (monotonic ms).
    ///
    /// Owned strings rather than borrows: the draw site has already borrowed a
    /// run's `log` and `view` for the output area, so a roster that borrowed the
    /// pane too could not be passed alongside them.
    #[must_use]
    pub fn roster_view(&self, now: u64) -> RosterView {
        // The roster is a live readout, so it goes away once the last agent
        // finishes rather than leaving stale rows pinned under the status bar.
        // The exception is a user who is *in* it — the rows must not vanish from
        // under the cursor mid-read, and an expanded pane needs its row to stay
        // on screen for as long as it is being read.
        if self.runs.is_empty() || !(self.running() || self.selecting || self.active) {
            return RosterView::default();
        }
        let mut rows = vec![RosterRow {
            label: "main".to_owned(),
            cursor: self.selecting && self.cursor == 0,
            ..RosterRow::default()
        }];
        rows.extend(self.runs.iter().enumerate().map(|(i, run)| RosterRow {
            label: run.label.clone(),
            activity: run.task.clone(),
            running: run.running,
            elapsed: fmt_elapsed(run.elapsed_ms(now)),
            tokens: run.tokens_text(),
            cursor: self.selecting && self.cursor == i + 1,
            expanded: self.active && self.cursor == i + 1,
        }));
        RosterView { rows }
    }
}

/// One drawn roster row. `main` is row zero and carries only a label; the
/// sub-agent rows below it carry the live telemetry.
#[derive(Debug, Default, Clone)]
pub struct RosterRow {
    /// Agent name, or `main` for the transcript row.
    pub label: String,
    /// The newest line the agent emitted, shown beside its name.
    pub activity: String,
    /// Whether it is still working — decides the bullet and the emphasis.
    pub running: bool,
    /// Pre-formatted wall clock, e.g. `3m 28s`; empty on the `main` row.
    pub elapsed: String,
    /// Pre-formatted spend, e.g. `↓ 51.9k tokens`; empty on the `main` row.
    pub tokens: String,
    /// Whether the roster cursor rests here.
    pub cursor: bool,
    /// Whether this row's output is the one expanded over the transcript.
    pub expanded: bool,
}

/// The roster as the draw pass sees it: rows, or empty before any sub-agent has
/// run (in which case the roster occupies no screen rows at all).
#[derive(Debug, Default, Clone)]
pub struct RosterView {
    /// `main` first, then every run oldest-first.
    pub rows: Vec<RosterRow>,
}

impl RosterView {
    /// Screen rows the roster needs: a blank separator plus one row each, or
    /// zero when there is nothing to show.
    #[must_use]
    pub fn height(&self) -> u16 {
        if self.rows.is_empty() {
            return 0;
        }
        u16::try_from(self.rows.len())
            .unwrap_or(u16::MAX)
            .saturating_add(1)
    }
}

/// Collapses a delegated task to the single line a roster row can hold: every
/// run of whitespace (newlines included) becomes one space, and the result is
/// trimmed. A task is often a paragraph; the row shows its beginning and the
/// renderer elides the rest to the width available.
#[must_use]
fn one_line(task: &str) -> String {
    task.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The clock the roster times runs against: milliseconds since the shared
/// epoch, ignoring reduced motion. [`crate::anim::clock_ms`] would return `None`
/// there and freeze every row's elapsed time at zero — reduced motion is about
/// animation, not about withholding how long an agent has been working.
#[must_use]
pub fn roster_clock_ms() -> u64 {
    crate::anim::epoch_ms()
}

/// Formats a run's duration the way the roster reports it: `47s`, `3m 28s`,
/// `1h 4m`. Delegates to [`crate::status::format_elapsed`] so a roster row and
/// the status bar never disagree about how to write a duration.
#[must_use]
fn fmt_elapsed(ms: u64) -> String {
    #[allow(clippy::cast_precision_loss)] // Display only, to whole seconds.
    crate::status::format_elapsed(ms as f64 / 1000.0)
}

/// Formats a run's token tally, on the status bar's own scale
/// ([`crate::status::format_ctx_size`]): `999`, `51.9k`, `2M`. Empty when
/// nothing has been spent yet, which reads as nothing rather than as `↓ 0`.
#[must_use]
fn fmt_tokens(tokens: u64) -> String {
    if tokens == 0 {
        return String::new();
    }
    let n = i32::try_from(tokens).unwrap_or(i32::MAX);
    format!("↓ {} tokens", crate::status::format_ctx_size(n))
}

/// The one-shot dim line pushed into the main transcript when a sub-agent run
/// starts. Single source of the wording so the `/subagent` command, the model's
/// `agent` tool, and anything replaying the transcript all say the same thing.
#[must_use]
pub fn subagent_signpost(label: &str) -> String {
    format!("[sub-agent: {label} — ← for agents]")
}

/// The dim line pushed into the main transcript when several sub-agents run
/// concurrently; the plural counterpart of [`subagent_signpost`].
#[must_use]
pub fn subagents_signpost(labels: &[&str]) -> String {
    format!("[sub-agents: {} — ← for agents]", labels.join(", "))
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
    roster: &RosterView,
) -> (Rect, Rect, Rect) {
    // The task strip (issue #35) sits directly above the rule; it appears only
    // at rest (with a prompt) and only when a task is in flight, capped at
    // three rows so it never crowds the scrollback.
    let strip = if has_prompt { tasks.strip_rows() } else { &[] };
    let strip_rows = u16::try_from(strip.len()).unwrap_or(0);
    let FrameGeom {
        output,
        input,
        status,
        rule_top,
        rule_bottom,
        strip: strip_area,
        roster: roster_area,
    } = frame_geom(area, has_prompt, input_rows, strip_rows, roster.height());
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
    if let Some(roster_area) = roster_area {
        if crate::uiremote::recording_enabled() {
            crate::uiremote::region("roster", roster_area, &[]);
        }
        render_agent_roster(frame, roster_area, roster);
    }
    // Both rules bracket the resting prompt (above and below the input).
    for rule in [rule_top, rule_bottom].into_iter().flatten() {
        let text = "─".repeat(rule.width as usize);
        frame.render_widget(
            Paragraph::new(Span::styled(text, Style::default().fg(THEME_GREEN))),
            rule,
        );
    }
    // Drawn over the top rule, so it needs the rule painted first.
    if let Some(rule) = rule_top {
        render_session_name(frame, rule);
    }
    (output, input, status)
}

/// Columns of bare rule kept to the right of the session name, so the label
/// reads as sitting on the line rather than running off its end.
const SESSION_NAME_GAP: u16 = 2;

/// Shortest run of rule left of the session name; below this the label is
/// dropped rather than swallowing the whole line on a narrow terminal.
const SESSION_NAME_MIN_RULE: u16 = 8;

/// Floats the session name at the right end of the rule above the prompt, so the
/// name the transcript will be saved under is visible from the first frame
/// instead of only being announced on exit.
fn render_session_name(frame: &mut Frame, rule: Rect) {
    let name = session_name();
    if name.is_empty() {
        return;
    }
    // Spaces on both sides: the label sits in a gap in the rule, not on top of it.
    let label = format!(" {name} ");
    let w = u16::try_from(label.chars().count()).unwrap_or(u16::MAX);
    if rule.width < w.saturating_add(SESSION_NAME_GAP + SESSION_NAME_MIN_RULE) {
        return;
    }
    let area = Rect::new(rule.right() - SESSION_NAME_GAP - w, rule.y, w, 1);
    crate::uiremote::region(
        "session_name",
        area,
        &[("name", crate::tools::mcp::Json::Str(name.clone()))],
    );
    frame.render_widget(
        Paragraph::new(Span::styled(
            label,
            Style::default().fg(Color::Indexed(245)),
        )),
        area,
    );
}

/// The session name the next frame will float on the rule above the prompt.
///
/// A process global rather than a `draw` parameter: every one of `draw`'s call
/// sites would otherwise have to thread it through, including the ones that
/// repaint from inside a slow command with no session in reach.
static SESSION_NAME: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

/// Publishes the current session's name for later frames to draw.
pub fn set_session_name(name: &str) {
    if let Ok(mut slot) = SESSION_NAME.lock()
        && *slot != name
    {
        slot.clear();
        slot.push_str(name);
    }
}

/// The published session name, empty when none has been set.
#[must_use]
pub fn session_name() -> String {
    SESSION_NAME.lock().map(|s| s.clone()).unwrap_or_default()
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

/// Widest the task column is allowed to get, whatever the terminal's width. Set
/// so a row stays scannable at a glance — long enough to recognise which job is
/// which, short enough that the eye still finds the name and the tally.
const TASK_MAX_COLS: u16 = 44;

/// Draws the sub-agent roster below the status bar: one row per agent, `main`
/// first, each with a state bullet, its name, the task it was given, and —
/// right-aligned — how long it has run and what it has spent.
///
/// `area`'s first row is left blank, separating the roster from the status bar
/// the way the task strip is separated from the rule above it.
fn render_agent_roster(frame: &mut Frame, area: Rect, roster: &RosterView) {
    let dim = Style::default().fg(Color::Indexed(245));
    for (i, row) in roster.rows.iter().enumerate() {
        // +1 for the blank separator row.
        let Some(y) = area
            .y
            .checked_add(u16::try_from(i.saturating_add(1)).unwrap_or(u16::MAX))
        else {
            break;
        };
        if y >= area.bottom() {
            break;
        }
        let line = Rect::new(area.x, y, area.width, 1);
        let mut name = Style::default().fg(Color::Indexed(252));
        if row.running {
            name = name.add_modifier(Modifier::BOLD);
        }
        if row.expanded {
            name = name.fg(THEME_GREEN);
        }
        let mut spans = vec![
            Span::styled(
                if row.cursor { "› " } else { "  " },
                Style::default().fg(THEME_GREEN),
            ),
            Span::styled(
                if row.running { "● " } else { "○ " },
                if row.running { name } else { dim },
            ),
            Span::styled(row.label.clone(), name),
        ];
        // The right-hand tally, and how much room the activity has left once
        // it is reserved. Both are dropped on a narrow terminal rather than
        // wrapped: the roster is strictly one row per agent.
        // Elapsed alone is still worth showing: a local engine reports its token
        // count only once a pass completes, and suppressing the whole tally until
        // then left the row looking like it carried no telemetry at all.
        let tally = match (row.elapsed.as_str(), row.tokens.as_str()) {
            ("", t) => t.to_owned(),
            (e, "") => e.to_owned(),
            (e, t) => format!("{e} · {t}"),
        };
        let tally_width = u16::try_from(tally.chars().count()).unwrap_or(u16::MAX);
        let used = u16::try_from(
            spans
                .iter()
                .map(|s| s.content.chars().count())
                .sum::<usize>(),
        )
        .unwrap_or(u16::MAX);
        // Leave a two-column gutter between the task and the tally, and cap the
        // task at [`TASK_MAX_COLS`] however much room is left: a task is prose,
        // and letting it run to the edge turns the roster into a wall of text
        // rather than a glance.
        let room = area
            .width
            .saturating_sub(used)
            .saturating_sub(tally_width)
            .saturating_sub(4)
            .min(TASK_MAX_COLS);
        if !row.activity.is_empty() && room > 0 {
            let mut activity: String = row.activity.chars().take(room as usize).collect();
            if activity.chars().count() < row.activity.chars().count() {
                activity.pop();
                activity.push('…');
            }
            spans.push(Span::raw("  "));
            spans.push(Span::styled(activity, if row.running { name } else { dim }));
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), line);
        if !tally.is_empty() && area.width > tally_width {
            frame.render_widget(
                Paragraph::new(Span::styled(
                    tally,
                    if row.running {
                        dim.add_modifier(Modifier::BOLD)
                    } else {
                        dim
                    },
                )),
                Rect::new(area.right() - tally_width, y, tally_width, 1),
            );
        }
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
    roster_rows: u16,
) -> FrameGeom {
    // The roster sits below everything, and it never shrinks the scrollback to
    // nothing: on a short terminal it gives its rows back to the output.
    let roster_rows = roster_rows.min(area.height.saturating_sub(STATUS_ROWS + input_rows + 3));
    if has_prompt {
        let r = Layout::vertical([
            Constraint::Min(1),              // output
            Constraint::Length(strip_rows),  // task strip (0 when idle-empty)
            Constraint::Length(1),           // top rule
            Constraint::Length(input_rows),  // input
            Constraint::Length(1),           // bottom rule
            Constraint::Length(STATUS_ROWS), // status (two rows: see status_bar_lines)
            Constraint::Length(roster_rows), // sub-agent roster (0 until one runs)
        ])
        .split(area);
        FrameGeom {
            output: r[0],
            input: r[3],
            status: r[5],
            rule_top: Some(r[2]),
            rule_bottom: Some(r[4]),
            strip: (strip_rows > 0).then(|| r[1]),
            roster: (roster_rows > 0).then(|| r[6]),
        }
    } else {
        let r = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(STATUS_ROWS),
            Constraint::Length(roster_rows),
        ])
        .split(area);
        FrameGeom {
            output: r[0],
            input: r[1],
            status: r[2],
            rule_top: None,
            rule_bottom: None,
            strip: None,
            roster: (roster_rows > 0).then(|| r[3]),
        }
    }
}

/// The rows [`frame_geom`] carves `area` into. A struct rather than a tuple
/// because there are now seven of them and a positional `.4` at the call site
/// says nothing about which row it is.
struct FrameGeom {
    output: Rect,
    input: Rect,
    status: Rect,
    /// Present only with a resting prompt: the rules bracketing it.
    rule_top: Option<Rect>,
    rule_bottom: Option<Rect>,
    /// Present only when non-empty.
    strip: Option<Rect>,
    roster: Option<Rect>,
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
pub fn draw_popup(
    frame: &mut Frame,
    input_text: &str,
    popup: &crate::complete::Popup,
    roster_rows: u16,
) {
    let tw = input_text_width(frame.area().width);
    let g = frame_geom(
        frame.area(),
        true,
        input_height(input_text, tw),
        0,
        roster_rows,
    );
    let rows = u16::try_from(popup.rows().len()).unwrap_or(u16::MAX);
    render_popup(frame, popup_rect(g.output, g.input, rows), popup);
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
pub fn draw_slash_menu(
    frame: &mut Frame,
    input_text: &str,
    menu: &crate::slashmenu::SlashMenu,
    roster_rows: u16,
) {
    let tw = input_text_width(frame.area().width);
    let g = frame_geom(
        frame.area(),
        true,
        input_height(input_text, tw),
        0,
        roster_rows,
    );
    let rows = u16::try_from(menu.rows().len()).unwrap_or(u16::MAX);
    render_slash_menu(frame, popup_rect(g.output, g.input, rows), menu);
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

/// The rectangle a centered modal paints into: `lines` rows plus a border, at
/// most `max_width` wide, clamped so it can never spill outside `area`.
///
/// The clamps run in this order deliberately: the `max(20)`/`max(3)` floors
/// keep the box usable on an ordinary terminal, but the trailing `min` against
/// the frame has the last word — ratatui's `render_widget` does not intersect
/// with the viewport, so a rect wider than the frame panics on the first write.
#[must_use]
fn modal_rect(area: Rect, lines: usize, max_width: u16) -> Rect {
    let width = max_width
        .min(area.width.saturating_sub(4))
        .max(20)
        .min(area.width);
    let height = u16::try_from(lines + 2)
        .unwrap_or(u16::MAX)
        .min(area.height.saturating_sub(2))
        .max(3)
        .min(area.height);
    Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    }
}

/// The modal rectangle [`draw_config`] paints into.
#[must_use]
fn config_rect(area: Rect, lines: usize) -> Rect {
    modal_rect(area, lines, 66)
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

    let rect = config_rect(frame.area(), lines.len());
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" config → ./.plank/settings.json ")
        .title_style(Style::default().add_modifier(Modifier::BOLD));
    frame.render_widget(Clear, rect);
    frame.render_widget(Paragraph::new(Text::from(lines)).block(block), rect);
}

/// The modal rectangle `draw_kvcache` paints into: `lines` rows plus a border,
/// centered in `area` and clamped so it never spills outside a small terminal.
///
/// Pulled out of the draw call so the clamping is testable on its own.
#[must_use]
fn kvcache_rect(area: Rect, lines: usize) -> Rect {
    modal_rect(area, lines, 84)
}

/// Draws the `/kvcache` lineage tree as a centered modal overlay.
///
/// Rows come from [`crate::kvpane::KvPane::rows`]: indentation encodes depth,
/// the selected row is reversed, and pinned and expired markers ride on the
/// right. A footer carries the totals and the key hints.
pub fn draw_kvcache(frame: &mut Frame, pane: &crate::kvpane::KvPane) {
    let rows = pane.rows();
    let mut lines: Vec<Line<'static>> = Vec::with_capacity(rows.len() + 3);
    // Where the selected row's own label landed among the rendered lines. A row
    // can contribute two lines, so the window below has to be measured in lines
    // rather than row indices or it drifts by one per expanded detail.
    let mut selected_line = 0usize;
    for row in &rows {
        if row.selected {
            selected_line = lines.len();
        }
        let indent = "  ".repeat(row.depth);
        let marker = match (row.has_children, row.expanded) {
            (true, true) => "▾ ",
            (true, false) => "▸ ",
            (false, _) => "  ",
        };
        let left = format!("{indent}{marker}{}", row.label);
        let style = if row.selected {
            Style::default().add_modifier(Modifier::REVERSED)
        } else {
            Style::default()
        };
        lines.push(Line::from(vec![
            Span::styled(left, style),
            Span::raw("  "),
            Span::styled(row.right.clone(), Style::default().fg(Color::DarkGray)),
        ]));
        if !row.detail.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("{indent}    {}", row.detail),
                Style::default().fg(Color::DarkGray),
            )));
        }
    }
    // The blank line, the footer and the hints always show; only the tree
    // scrolls, so they are built separately and appended after the window.
    let mut tail: Vec<Line<'static>> = Vec::with_capacity(3);
    tail.push(Line::from(""));
    tail.push(Line::from(Span::styled(
        format!("  {}", pane.footer()),
        Style::default().fg(Color::Cyan),
    )));
    // A pending delete replaces the hints with the question it is waiting on,
    // so `d` never leaves the user guessing what the pane wants next.
    let hints = if pane.pending_delete() {
        "  delete this entry? y to confirm, any other key cancels"
    } else {
        "  ↑↓ move · ←→ fold · p pin · d delete · g sweep · Esc close"
    };
    tail.push(Line::from(Span::styled(
        hints,
        Style::default().fg(Color::DarkGray),
    )));

    let rect = kvcache_rect(frame.area(), lines.len() + tail.len());
    // `Paragraph` clips rather than scrolls, so without a window the cursor
    // could sit on an invisible row — and `d`/`y` would then delete a blob the
    // user cannot see. Slide the tree so the selected row is always drawn.
    let inner = usize::from(rect.height.saturating_sub(2));
    let room = inner.saturating_sub(tail.len());
    let mut shown = if room == 0 {
        // No room for the tree at all: keep the footer and hints, which are the
        // only lines that still say something useful at this size. The blank
        // separator has to go with the tree it was separating, or it spends the
        // one or two lines that are left and `Paragraph`, which clips from the
        // bottom, drops the hints instead.
        tail.remove(0);
        Vec::new()
    } else if lines.len() <= room {
        lines
    } else {
        let start = (selected_line + 1)
            .saturating_sub(room)
            .min(lines.len() - room);
        lines.split_off(start).into_iter().take(room).collect()
    };
    shown.append(&mut tail);
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" kv cache ")
        .title_style(Style::default().add_modifier(Modifier::BOLD));
    frame.render_widget(Clear, rect);
    frame.render_widget(Paragraph::new(Text::from(shown)).block(block), rect);
}

/// Accent for the `/resume` picker: its rule, its header and its selected row.
const RESUME_ACCENT: Color = Color::Indexed(105);
/// Every list line opens with a two-column gutter, so the cursor caret and the
/// scroll arrows can be written into it after the window is chosen without
/// re-laying out the line.
const RESUME_GUTTER: &str = "  ";

/// Renders the picker's sessions into display lines.
///
/// Returns the lines, the index at which each session's block starts, and the
/// line the selected session's own name landed on. The starts are what let the
/// window snap to a session boundary instead of slicing a row in half.
fn resume_list_lines(
    pane: &crate::resumepane::ResumePane,
) -> (Vec<Line<'static>>, Vec<usize>, usize) {
    let rows = pane.rows();
    let mut lines: Vec<Line<'static>> = Vec::with_capacity(rows.len() * 3);
    let mut starts: Vec<usize> = Vec::with_capacity(rows.len());
    let mut selected_line = 0usize;
    for (i, row) in rows.iter().enumerate() {
        // A blank line between sessions, never before the first or after the
        // last: the padding is the separator, so a trailing one would only
        // spend a line the list could have shown a session on.
        if i > 0 {
            lines.push(Line::from(Span::raw(RESUME_GUTTER)));
        }
        starts.push(lines.len());
        if row.selected {
            selected_line = lines.len();
        }
        let name = if row.selected {
            Style::default()
                .fg(RESUME_ACCENT)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        lines.push(Line::from(vec![
            Span::styled(
                if row.selected { "❯ " } else { RESUME_GUTTER },
                Style::default().fg(RESUME_ACCENT),
            ),
            Span::styled(row.label.clone(), name),
        ]));
        let dim = Style::default().fg(Color::DarkGray);
        lines.push(Line::from(vec![
            Span::raw(RESUME_GUTTER),
            Span::styled(row.detail.clone(), dim),
        ]));
        for extra in &row.extra {
            lines.push(Line::from(vec![
                Span::raw(RESUME_GUTTER),
                Span::styled(extra.clone(), dim),
            ]));
        }
    }
    (lines, starts, selected_line)
}

/// Slides `lines` so the selected session is on screen, snapping the top to a
/// session boundary and marking the cut ends with arrows.
///
/// Windowing at a boundary matters for more than looks: the top line of the
/// view carries the "there is more above" arrow, and an arrow on the tail of
/// some other session's detail would point at the wrong thing.
fn resume_window(
    mut lines: Vec<Line<'static>>,
    starts: &[usize],
    selected_line: usize,
    room: usize,
) -> Vec<Line<'static>> {
    let total = lines.len();
    if room == 0 {
        return Vec::new();
    }
    if total <= room {
        return lines;
    }
    // Aim to sit the selection around the middle of the view rather than at its
    // top: pinning it to the top would scroll the entire list on every single
    // Down press, and there would never be a session visible above the cursor.
    let target = selected_line.saturating_sub(room / 2);
    let before = |limit: usize| starts.iter().copied().rfind(|&s| s <= limit).unwrap_or(0);
    let start = before(target).min(total - room);
    // …but visibility wins over centering: at the end of the list the clamp
    // above can push the top past the selection, so fall back to the last
    // boundary that still leaves the selected line inside the window.
    let start = if selected_line.saturating_sub(start) < room && start <= selected_line {
        start
    } else {
        starts
            .iter()
            .copied()
            .rfind(|&s| s <= selected_line && selected_line - s < room)
            .unwrap_or(0)
            .min(total - room)
    };
    let end = start + room;
    let mut shown: Vec<Line<'static>> = lines.split_off(start).into_iter().take(room).collect();
    let arrow = Style::default().fg(Color::DarkGray);
    // The arrows replace the gutter, never the cursor caret: the caret is the
    // one mark that says which session Enter would resume.
    let mut mark = |line: usize, glyph: &'static str| {
        if line != selected_line
            && let Some(span) = shown
                .get_mut(line - start)
                .and_then(|l| l.spans.first_mut())
        {
            *span = Span::styled(glyph, arrow);
        }
    };
    if start > 0 {
        mark(start, "↑ ");
    }
    // The down arrow rides the last *session name* in view rather than the last
    // line, which is usually a dim detail line and reads as a stray glyph.
    if end < total
        && let Some(&last) = starts.iter().rfind(|&&s| s >= start && s < end)
    {
        mark(last, "↓ ");
    }
    shown
}

/// Draws the picker's header: the words in the accent, the count beside them
/// in gray, so `(3 of 48)` reads as a position rather than part of the title.
fn draw_resume_header(frame: &mut Frame, rect: Rect, pane: &crate::resumepane::ResumePane) {
    let header = pane.header();
    let (title, count) = header.split_once(" (").unwrap_or((header.as_str(), ""));
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                title.to_owned(),
                Style::default()
                    .fg(RESUME_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                if count.is_empty() {
                    String::new()
                } else {
                    format!(" ({count}")
                },
                Style::default().fg(Color::DarkGray),
            ),
        ])),
        rect,
    );
}

/// Draws the picker's text field.
///
/// One field, two jobs: it becomes the rename box while a rename is open — and
/// says so, in the accent and with a title — so the pane never shows two places
/// the keys being typed could be going.
fn draw_resume_field(frame: &mut Frame, rect: Rect, pane: &crate::resumepane::ResumePane) {
    let renaming = pane.rename_buffer().is_some();
    let text = pane.rename_buffer().unwrap_or_else(|| pane.query());
    let dim = Style::default().fg(Color::DarkGray);
    let field = if text.is_empty() && !renaming {
        Line::from(Span::styled("⌕ Search…", dim))
    } else {
        Line::from(vec![
            Span::styled(if renaming { "✎ " } else { "⌕ " }, dim),
            Span::raw(text.to_owned()),
            Span::styled("▏", Style::default().fg(RESUME_ACCENT)),
        ])
    };
    let style = if renaming {
        Style::default().fg(RESUME_ACCENT)
    } else {
        dim
    };
    frame.render_widget(
        Paragraph::new(field).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(style)
                .title(if renaming { " rename " } else { "" })
                .title_style(style),
        ),
        rect,
    );
}

/// Draws the `/resume` session picker as a panel anchored above the status bar.
///
/// Chrome, top to bottom: an accent rule, the `Resume session (n of m)` header,
/// a boxed search field (which doubles as the rename field), the project label,
/// the session list, and the key hints. Everything but the list is fixed, so
/// the list is the only thing that scrolls — [`resume_window`] slides it.
pub fn draw_resume(frame: &mut Frame, pane: &crate::resumepane::ResumePane) {
    let area = frame.area();
    let (lines, starts, selected_line) = resume_list_lines(pane);
    let scope_rows = u16::from(!pane.scope().is_empty());
    // Rule, header, gap, the three rows of the search box, the scope label, a
    // gap, and two lines of hints — everything the list does not get.
    let chrome = 1 + 1 + 1 + 3 + scope_rows + 1 + 3;
    // `max(1)`: an empty result set still needs the one line that says so.
    let want = u16::try_from(lines.len().max(1))
        .unwrap_or(u16::MAX)
        .saturating_add(chrome);
    // The status bar owns the last row; the panel takes what it needs of the rest.
    let avail = area.height.saturating_sub(1);
    let height = want.min(avail);
    if height == 0 {
        return;
    }
    let panel = Rect {
        x: area.x,
        y: area.y + avail - height,
        width: area.width,
        height,
    };
    frame.render_widget(Clear, panel);
    frame.render_widget(
        Block::default()
            .borders(Borders::TOP)
            .border_style(Style::default().fg(RESUME_ACCENT)),
        panel,
    );
    // Inset from the rule and the terminal edges; `saturating_sub` keeps a very
    // narrow terminal from producing a wider-than-frame rect, which panics.
    let body = Rect {
        x: panel.x + 2,
        y: panel.y + 1,
        width: panel.width.saturating_sub(4),
        height: panel.height.saturating_sub(1),
    };
    let r = Layout::vertical([
        Constraint::Length(1),          // header
        Constraint::Length(1),          // gap
        Constraint::Length(3),          // search box
        Constraint::Length(scope_rows), // project label
        Constraint::Length(1),          // gap
        Constraint::Min(0),             // session list
        Constraint::Length(3),          // blank line, then two of hints
    ])
    .split(body);

    draw_resume_header(frame, r[0], pane);
    draw_resume_field(frame, r[2], pane);

    if scope_rows == 1 {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!("{RESUME_GUTTER}{}", pane.scope()),
                Style::default().fg(Color::Gray),
            ))),
            r[3],
        );
    }

    // An empty result set says so where the list would have been, rather than
    // leaving a blank panel that looks like a failure to draw.
    if pane.is_empty() {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!("{RESUME_GUTTER}no session matches"),
                Style::default().fg(Color::DarkGray),
            ))),
            r[5],
        );
    } else {
        let shown = resume_window(lines, &starts, selected_line, usize::from(r[5].height));
        frame.render_widget(Paragraph::new(Text::from(shown)), r[5]);
    }

    frame.render_widget(
        Paragraph::new(Text::from(vec![
            Line::from(""),
            Line::from(Span::styled(
                format!("{RESUME_GUTTER}{}", pane.footer()),
                Style::default().fg(Color::DarkGray),
            )),
        ]))
        .wrap(Wrap { trim: false }),
        r[6],
    );
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
/// Maps a crossterm mouse event onto the frame ABI's `MouseEvent`.
///
/// `None` for the kinds a component has no use for (non-left buttons, and
/// button-up of a button it never saw pressed), so an author's handler is not
/// woken for noise. Coordinates are clamped into the frame: a drag that leaves
/// the window still reports a position on its edge, which is what a paddle
/// wants. Kinds are strings for the same reason key names are — the ABI must
/// not pin plank to crossterm's enum ordering.
#[must_use]
pub fn frame_mouse_event(
    m: &ratatui::crossterm::event::MouseEvent,
    w: u16,
    h: u16,
) -> Option<crate::wasmreg::FrameMouse> {
    use ratatui::crossterm::event::{MouseButton, MouseEventKind};
    let kind = match m.kind {
        MouseEventKind::Down(MouseButton::Left) => "down",
        MouseEventKind::Up(MouseButton::Left) => "up",
        MouseEventKind::Drag(MouseButton::Left) => "drag",
        MouseEventKind::Moved => "move",
        MouseEventKind::ScrollUp => "scroll_up",
        MouseEventKind::ScrollDown => "scroll_down",
        _ => return None,
    };
    Some(crate::wasmreg::FrameMouse {
        kind,
        x: m.column.min(w.saturating_sub(1)),
        y: m.row.min(h.saturating_sub(1)),
        w,
        h,
    })
}

/// The name a WASM `frame` component sees for a key press.
///
/// Lower-case, spelled out for anything that is not a bare character, with
/// modifiers prefixed (`ctrl-c`, `shift-tab`). A component matches on strings
/// rather than on a numeric code so the ABI does not pin plank to crossterm's
/// enum ordering, which is exactly the kind of thing that shifts under a
/// dependency bump and silently remaps every game's controls.
#[must_use]
pub fn key_code_name(key: ratatui::crossterm::event::KeyEvent) -> String {
    use ratatui::crossterm::event::{KeyCode, KeyModifiers};

    let base = match key.code {
        KeyCode::Char(c) => c.to_lowercase().to_string(),
        KeyCode::Left => "left".to_string(),
        KeyCode::Right => "right".to_string(),
        KeyCode::Up => "up".to_string(),
        KeyCode::Down => "down".to_string(),
        KeyCode::Enter => "enter".to_string(),
        KeyCode::Esc => "escape".to_string(),
        KeyCode::Backspace => "backspace".to_string(),
        KeyCode::Tab => "tab".to_string(),
        KeyCode::BackTab => "backtab".to_string(),
        KeyCode::Delete => "delete".to_string(),
        KeyCode::Home => "home".to_string(),
        KeyCode::End => "end".to_string(),
        KeyCode::PageUp => "pageup".to_string(),
        KeyCode::PageDown => "pagedown".to_string(),
        KeyCode::F(n) => format!("f{n}"),
        other => format!("{other:?}").to_lowercase(),
    };
    let mut out = String::new();
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        out.push_str("ctrl-");
    }
    if key.modifiers.contains(KeyModifiers::ALT) {
        out.push_str("alt-");
    }
    // Shift is reported only for keys where it is not already in the
    // character: `shift-a` would be a lie when the char arrives as `A`.
    if key.modifiers.contains(KeyModifiers::SHIFT) && !matches!(key.code, KeyCode::Char(_)) {
        out.push_str("shift-");
    }
    out.push_str(&base);
    out
}

/// Draws an open WASM `frame` component, on the same terms as the built-in
/// faces: veiled over the live UI, or on a real black ground.
///
/// Shares [`draw_arcade`]'s ground-painting rather than reimplementing it —
/// the veil and the explicit RGB black are hard-won (see that function), and a
/// component must not get a different-looking screen than a built-in face for
/// no reason. What it adds is the two things the packed wire format carries
/// that `arcade::Glyph` does not: bold, and an optional background color.
pub fn draw_wasm_frame(frame: &mut Frame, open: &crate::wasmreg::OpenFrame) {
    let area = frame.area();
    paint_frame_ground(frame, area, open.veiled);
    if area.height < 2 {
        return;
    }
    // The bottom row stays with the UI, exactly as the arcade leaves it.
    let play = Rect {
        height: area.height - 1,
        ..area
    };

    let buf = frame.buffer_mut();
    for (i, g) in open.last.glyphs.iter().enumerate() {
        // A component draws in its own coordinates and may be a frame behind a
        // resize, so anything outside the current area is dropped rather than
        // clamped: clamping would pile a whole edge of glyphs into one column.
        if g.x >= play.width || g.y >= play.height {
            continue;
        }
        let (x, y) = (play.x + g.x, play.y + g.y);
        let Some(cell) = buf.cell_mut(Position::new(x, y)) else {
            continue;
        };
        let (r, gr, b) = g.color;
        cell.set_char(g.ch).set_fg(Color::Rgb(r, gr, b));
        if open.last.bold.get(i).copied().unwrap_or(false) {
            cell.set_style(cell.style().add_modifier(Modifier::BOLD));
        }
        if let Some(Some((br, bg, bb))) = open.last.bg.get(i) {
            cell.set_bg(Color::Rgb(*br, *bg, *bb));
        }
    }
}

/// Paints the ground an occupying surface draws onto: the veil over the live
/// UI, or a real black.
fn paint_frame_ground(frame: &mut Frame, area: Rect, translucent: bool) {
    if translucent {
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
}

pub fn draw_arcade(frame: &mut Frame, arcade: &crate::arcade::Arcade) {
    let area = frame.area();
    paint_frame_ground(frame, area, arcade.translucent);
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
    roster: &RosterView,
) {
    let area = frame.area();
    let tw = input_text_width(area.width);
    let (output, input_row, status_row) = frame_rows(
        frame,
        area,
        input.is_some(),
        input.map_or(1, |s| input_height(s.text, tw)),
        tasks,
        roster,
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
    roster: &RosterView,
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
        roster,
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
    const HINT: &str = " Esc: back to main ";
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
        spans.push(Span::styled(
            format!("✓ Tasks: {done}/{total}"),
            counter_style,
        ));
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
    fn frame_mouse_events_map_and_clamp() {
        use ratatui::crossterm::event::{KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
        let ev = |kind, column, row| MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        };

        let m = super::frame_mouse_event(&ev(MouseEventKind::Moved, 5, 3), 40, 12).unwrap();
        assert_eq!((m.kind, m.x, m.y, m.w, m.h), ("move", 5, 3, 40, 12));

        for (kind, name) in [
            (MouseEventKind::Down(MouseButton::Left), "down"),
            (MouseEventKind::Up(MouseButton::Left), "up"),
            (MouseEventKind::Drag(MouseButton::Left), "drag"),
            (MouseEventKind::ScrollUp, "scroll_up"),
            (MouseEventKind::ScrollDown, "scroll_down"),
        ] {
            let m = super::frame_mouse_event(&ev(kind, 1, 1), 40, 12).unwrap();
            assert_eq!(m.kind, name);
        }

        // Noise a component has no use for is dropped rather than delivered.
        assert!(
            super::frame_mouse_event(&ev(MouseEventKind::Down(MouseButton::Right), 1, 1), 40, 12)
                .is_none()
        );

        // A drag that left the window reports the edge: a paddle tracking the
        // pointer must not be handed a coordinate outside its own frame.
        let m =
            super::frame_mouse_event(&ev(MouseEventKind::Drag(MouseButton::Left), 99, 99), 40, 12)
                .unwrap();
        assert_eq!((m.x, m.y), (39, 11));
    }

    #[test]
    fn a_run_opens_a_row_and_selection_needs_one() {
        let mut pane = SubPane::default();
        // Nothing has run yet: there is nothing to select.
        assert!(!pane.move_cursor(-1));
        assert!(!pane.active);

        pane.begin("research".to_string(), "", 0);
        assert_eq!(pane.label(), Some("research"));
        assert!(pane.running());
        pane.current_log_mut().unwrap().push_plain("first output");
        let after_first = pane.runs[0].log.line_count();
        assert!(after_first > 0);

        pane.end(1);
        assert!(!pane.running());
        // The finished run stays readable.
        assert_eq!(pane.runs[0].log.line_count(), after_first);

        // A second run gets its own row and its own empty buffer; the first is
        // still there to go back to (the point of the roster).
        pane.begin("other".to_string(), "", 2);
        assert_eq!(pane.runs.len(), 2);
        assert_eq!(pane.runs[1].log.line_count(), 0);
        assert_eq!(pane.label(), Some("other"));
        assert_eq!(pane.runs[0].log.line_count(), after_first);
    }

    #[test]
    fn each_run_gets_a_fresh_scroll_view_of_its_own() {
        let mut pane = SubPane::default();
        pane.begin("first".to_string(), "", 0);
        pane.current_log_mut().unwrap().push_plain("a long run");
        // The user scrolled up through the first run's output.
        pane.runs[0].view.follow = false;
        pane.runs[0].view.top = 42;

        // The new run starts at the top in follow mode, and does not inherit
        // the position the user left the previous one at.
        pane.begin("second".to_string(), "", 1);
        assert_eq!(pane.runs[1].view.top, 0);
        assert!(pane.runs[1].view.follow);
        assert_eq!(pane.runs[0].view.top, 42, "the first run keeps its place");
    }

    #[test]
    fn sub_start_and_end_never_clobber_an_adopted_run() {
        // A `/subagent` turn owns the pane (`adopt_turn`). The model inside it
        // may still call the `agent` tool, which emits SubStart/SubEnd; those
        // must not clear the buffer, steal the label, or mark the outer run
        // finished while it is still streaming.
        let mut pane = SubPane::default();
        pane.begin("reviewer".to_string(), "", 0);
        pane.adopt_turn = true;
        pane.current_log_mut().unwrap().push_plain("outer output");
        let before = pane.runs[0].log.line_count();

        pane.on_sub_start("nested".to_string(), "", 100);
        assert_eq!(pane.runs.len(), 1, "no row opened for the nested call");
        assert_eq!(pane.label(), Some("reviewer"), "label kept");
        assert_eq!(pane.runs[0].log.line_count(), before, "buffer not cleared");
        assert!(pane.running(), "outer run still running");

        pane.current_log_mut().unwrap().push_plain("more output");
        pane.on_sub_end(200);
        assert!(pane.running(), "inner SubEnd must not end the outer run");
        assert_eq!(pane.runs[0].log.line_count(), before + 1);

        // Outside an adopted run the events do their normal work: a *new* agent
        // gets its own row, and its own buffer, beside the first.
        pane.adopt_turn = false;
        pane.on_sub_start("nested".to_string(), "", 300);
        assert_eq!(pane.runs.len(), 2);
        assert_eq!(pane.label(), Some("nested"));
        assert_eq!(pane.runs[1].log.line_count(), 0, "its own empty buffer");
        assert_eq!(
            pane.runs[0].log.line_count(),
            before + 1,
            "the earlier run's output survives a later run"
        );
        pane.on_sub_end(400);
        assert!(!pane.runs[1].running);
    }

    #[test]
    fn a_repeat_sub_start_resumes_the_running_row_it_names() {
        // A lockstep fan-out re-brackets each slot once per round. A second
        // SubStart for a still-running agent must resume its row, not open a
        // duplicate — otherwise a three-round fan-out shows nine agents.
        let mut pane = SubPane::default();
        pane.begin("alpha".to_string(), "", 0);
        pane.begin("beta".to_string(), "", 0);
        pane.begin("alpha".to_string(), "", 500);
        assert_eq!(pane.runs.len(), 2, "no duplicate row for alpha");
        assert_eq!(pane.label(), Some("alpha"), "and it is the current run");
        assert_eq!(pane.runs[0].started_ms, 0, "the original start time stands");

        // Once it has finished, the same name is a genuinely new run.
        pane.end(600);
        pane.begin("alpha".to_string(), "", 700);
        assert_eq!(pane.runs.len(), 3);
    }

    #[test]
    fn the_roster_caps_its_rows_and_never_evicts_a_running_one() {
        let mut pane = SubPane::default();
        // One long-lived run, then enough short ones to overflow the cap.
        pane.begin("keeper".to_string(), "", 0);
        for i in 0..ROSTER_MAX + 3 {
            pane.begin(format!("short{i}"), "", 0);
            pane.end(1);
        }
        assert_eq!(pane.runs.len(), ROSTER_MAX);
        assert_eq!(
            pane.runs.iter().filter(|r| r.label == "keeper").count(),
            1,
            "the still-running row survives every eviction"
        );
        assert!(
            pane.runs
                .iter()
                .any(|r| r.label == format!("short{}", ROSTER_MAX + 2)),
            "the newest run is kept"
        );
    }

    #[test]
    fn tokens_are_credited_to_the_row_they_name() {
        let mut pane = SubPane::default();
        pane.begin("alpha".to_string(), "", 0);
        pane.begin("beta".to_string(), "", 0);
        // `beta` is current, so an unnamed tally lands there.
        pane.add_tokens(None, 0, 10);
        pane.add_tokens(Some("alpha"), 0, 40);
        assert_eq!(pane.runs[0].generated, 40);
        assert_eq!(pane.runs[1].generated, 10);

        // A finished row is not credited: the tally belongs to a later run.
        pane.runs[0].running = false;
        pane.add_tokens(Some("alpha"), 0, 999);
        assert_eq!(pane.runs[0].generated, 40);
    }

    #[test]
    fn a_finished_row_freezes_its_elapsed_time() {
        let mut pane = SubPane::default();
        pane.begin("alpha".to_string(), "", 1_000);
        assert_eq!(pane.runs[0].elapsed_ms(4_000), 3_000, "live while running");
        pane.end(5_000);
        assert_eq!(
            pane.runs[0].elapsed_ms(9_999),
            4_000,
            "frozen at what it took"
        );
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
                    &RosterView::default(),
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
        assert!(with.contains("Esc: back to main"), "{with}");

        // The main transcript is untouched: no title, no hint, and the output
        // text still starts on the very first row.
        let without = render(None);
        assert!(!without.contains("sub-agent"), "{without}");
        assert!(!without.contains("back to main"), "{without}");
        assert!(without.starts_with("sub output"), "{without}");
    }

    #[test]
    fn reset_drops_everything_the_old_session_left_behind() {
        let mut pane = SubPane::default();
        pane.begin("research".to_string(), "", 0);
        pane.current_log_mut().unwrap().push_plain("old output");
        pane.adopt_turn = true;
        assert!(pane.move_cursor(-1));
        assert!(pane.move_cursor(1));
        assert!(pane.expand());
        assert!(pane.active);

        pane.reset();
        assert!(pane.runs.is_empty());
        assert!(pane.label().is_none());
        assert!(
            !pane.active,
            "a hidden pane must not swallow the new session"
        );
        assert!(!pane.selecting);
        assert!(!pane.running());
        assert!(!pane.adopt_turn);
        // Nothing to select again, exactly as at launch.
        assert!(!pane.move_cursor(-1));
        assert!(!pane.expand());
    }

    #[test]
    fn the_roster_cursor_walks_the_rows_and_bounds_at_both_ends() {
        let mut pane = SubPane::default();
        pane.begin("alpha".to_string(), "", 0);
        pane.begin("beta".to_string(), "", 0);

        // The first `←` only reveals the cursor where it rests, without moving.
        assert!(pane.move_cursor(-1));
        assert!(pane.selecting);
        assert_eq!(pane.cursor, 0, "starts on the `main` row");

        pane.move_cursor(1);
        assert_eq!(pane.cursor, 1);
        pane.move_cursor(1);
        assert_eq!(pane.cursor, 2);
        pane.move_cursor(1);
        assert_eq!(pane.cursor, 2, "clamped at the last agent");
        pane.move_cursor(-1);
        pane.move_cursor(-1);
        pane.move_cursor(-1);
        assert_eq!(pane.cursor, 0, "clamped at `main`");
    }

    #[test]
    fn moving_off_an_expanded_row_collapses_it() {
        // The cursor and what is on screen must never disagree: walking away
        // from an expanded agent puts the transcript back.
        let mut pane = SubPane::default();
        pane.begin("alpha".to_string(), "", 0);
        pane.move_cursor(-1);
        pane.move_cursor(1);
        assert!(pane.expand());
        assert!(pane.active);

        pane.move_cursor(-1);
        assert!(!pane.active, "collapsed by moving to another row");

        // `main` has nothing to expand, and Esc leaves the roster entirely.
        assert!(!pane.expand());
        assert!(pane.collapse());
        assert!(!pane.selecting);
        assert!(!pane.collapse(), "already out: Esc falls through");
    }

    #[test]
    fn the_roster_draws_below_the_status_bar_with_bullets_and_a_tally() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut pane = SubPane::default();
        pane.begin(
            "general-purpose".to_string(),
            "Committing malformed_dsml fix",
            0,
        );
        pane.add_tokens(None, 0, 51_900);
        pane.end(208_000);
        pane.begin(
            "general-purpose".to_string(),
            "Discovering existing viz.rs commit",
            208_000,
        );
        pane.add_tokens(None, 0, 39_900);
        let roster = pane.roster_view(280_000);

        let mut log = OutputLog::new();
        log.push_plain("transcript");
        let mut view = OutputView::default();
        let mut term = Terminal::new(TestBackend::new(90, 12)).unwrap();
        term.draw(|f| {
            draw(
                f,
                &log,
                Some(InputState::new("", 0)),
                "idle",
                &mut view,
                None,
                &TaskView::default(),
                None,
                &roster,
            );
        })
        .unwrap();
        let buf = term.backend().buffer().clone();
        let rows: Vec<String> = (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect();
        let screen = rows.join("\n");

        // The roster is the bottom of the frame, in order, below the status bar.
        let row_of = |needle: &str| {
            rows.iter()
                .position(|r| r.contains(needle))
                .unwrap_or_else(|| panic!("{needle:?} not on screen:\n{screen}"))
        };
        assert!(
            row_of("idle") < row_of("main"),
            "roster below the status bar"
        );
        assert!(row_of("main") < row_of("Committing malformed_dsml fix"));
        assert!(row_of("Committing malformed_dsml fix") < row_of("Discovering"));

        // Finished runs take the hollow bullet, the live one the filled bullet.
        assert!(rows[row_of("Committing")].contains('○'), "{screen}");
        assert!(rows[row_of("Discovering")].contains('●'), "{screen}");
        // And each carries its own right-aligned tally.
        assert!(rows[row_of("Committing")].contains("3m 28s · ↓ 51.9k tokens"));
        assert!(rows[row_of("Discovering")].contains("1m 12s · ↓ 39.9k tokens"));
        assert!(
            rows[row_of("Committing")].trim_end().ends_with("tokens"),
            "the tally is flush right: {:?}",
            rows[row_of("Committing")]
        );
    }

    #[test]
    fn a_long_activity_line_is_truncated_rather_than_overrunning_the_tally() {
        let mut pane = SubPane::default();
        pane.begin("agent".to_string(), &"x".repeat(400), 0);
        pane.add_tokens(None, 0, 1_000);
        let roster = pane.roster_view(5_000);

        let mut term = ratatui::Terminal::new(ratatui::backend::TestBackend::new(60, 10)).unwrap();
        let log = OutputLog::new();
        let mut view = OutputView::default();
        term.draw(|f| {
            draw(
                f,
                &log,
                Some(InputState::new("", 0)),
                "idle",
                &mut view,
                None,
                &TaskView::default(),
                None,
                &roster,
            );
        })
        .unwrap();
        let buf = term.backend().buffer().clone();
        let row = (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
            })
            .find(|r| r.contains("agent"))
            .expect("the agent row is drawn");
        assert!(row.contains('…'), "activity elided: {row:?}");
        assert!(row.contains("↓ 1k tokens"), "the tally still fits: {row:?}");
    }

    #[test]
    fn the_roster_is_empty_until_an_agent_runs_and_then_leads_with_main() {
        let mut pane = SubPane::default();
        assert!(pane.roster_view(0).rows.is_empty());
        assert_eq!(pane.roster_view(0).height(), 0, "no rows, no screen space");

        pane.begin("research".to_string(), "Committing the fix", 1_000);
        pane.add_tokens(None, 0, 51_900);
        let roster = pane.roster_view(209_000);
        let labels: Vec<&str> = roster.rows.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(labels, ["main", "research"]);
        assert_eq!(roster.height(), 3, "two rows plus the separator");
        let row = &roster.rows[1];
        assert!(row.running);
        assert_eq!(row.activity, "Committing the fix");
        assert_eq!(row.elapsed, "3m 28s");
        assert_eq!(row.tokens, "↓ 51.9k tokens");
        // The `main` row carries no telemetry of its own.
        assert_eq!(roster.rows[0].elapsed, "");
        assert_eq!(roster.rows[0].tokens, "");
    }

    #[test]
    fn a_row_reports_its_task_and_never_the_output_streaming_into_it() {
        // Regression: the row used to show the newest line of the agent's own
        // output, which lands mid-statement on whatever the model is writing
        // (`vals =`). A row summarises the job it was given.
        let mut pane = SubPane::default();
        pane.begin("sub-agent".to_string(), "convert 38 to Roman numerals", 0);
        pane.current_log_mut()
            .unwrap()
            .push_plain("let vals = [1000, 900, 500];");
        let row = &pane.roster_view(1_000).rows[1];
        assert_eq!(row.activity, "convert 38 to Roman numerals");
        assert!(!row.activity.contains("vals"), "not the streamed output");
    }

    #[test]
    fn a_long_task_is_capped_even_when_the_terminal_has_room_to_spare() {
        // A task is prose. On a wide terminal there is room to print all of it,
        // and doing so turned the roster into a wall of text — so the column is
        // capped at TASK_MAX_COLS regardless of the width available.
        const TASK: &str = "Find the largest (longest) Roman numeral string when \
            counting from 1 to 50 inclusive. Write out your reasoning.";
        let mut pane = SubPane::default();
        pane.begin("sub-agent".to_string(), TASK, 0);
        let roster = pane.roster_view(27_000);

        let mut term = ratatui::Terminal::new(ratatui::backend::TestBackend::new(200, 10)).unwrap();
        let log = OutputLog::new();
        let mut view = OutputView::default();
        term.draw(|f| {
            draw(
                f,
                &log,
                Some(InputState::new("", 0)),
                "idle",
                &mut view,
                None,
                &TaskView::default(),
                None,
                &roster,
            );
        })
        .unwrap();
        let buf = term.backend().buffer().clone();
        let row = (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
            })
            .find(|r| r.contains("sub-agent"))
            .expect("the agent row is drawn");

        assert!(row.contains('…'), "elided: {row:?}");
        assert!(
            !row.contains("counting from"),
            "cut well before the width allows: {row:?}"
        );
        // The task occupies at most TASK_MAX_COLS columns between the name and
        // the tally, however wide the terminal is.
        let task = row
            .split_once("sub-agent")
            .map(|(_, rest)| rest.trim())
            .and_then(|rest| rest.split_once('…'))
            .map(|(task, _)| task.trim())
            .expect("the task column is present");
        assert!(
            task.chars().count() < TASK_MAX_COLS as usize,
            "{} columns is over the cap: {task:?}",
            task.chars().count()
        );
        assert!(row.trim_end().ends_with("27s"), "the tally still lands");
    }

    #[test]
    fn a_multi_line_task_is_flattened_to_the_one_line_a_row_holds() {
        assert_eq!(one_line("  review the diff  "), "review the diff");
        assert_eq!(
            one_line("review the diff\n\nand report\tback"),
            "review the diff and report back",
            "a paragraph task cannot be allowed to break the row"
        );
        assert_eq!(one_line(""), "");
    }

    #[test]
    fn the_roster_goes_away_once_the_last_agent_is_done() {
        let mut pane = SubPane::default();
        pane.begin("alpha".to_string(), "", 0);
        pane.begin("beta".to_string(), "", 0);
        assert_eq!(pane.roster_view(5_000).rows.len(), 3, "main plus two");

        // One finishing is not enough — the other is still working.
        pane.current = 0;
        pane.end(5_000);
        assert_eq!(pane.roster_view(5_000).rows.len(), 3);

        pane.current = 1;
        pane.end(6_000);
        assert!(
            pane.roster_view(9_000).rows.is_empty(),
            "the last agent finishing takes the roster with it"
        );
        assert_eq!(pane.roster_view(9_000).height(), 0, "and its screen rows");
    }

    #[test]
    fn a_finished_roster_stays_on_screen_while_the_user_is_in_it() {
        // Rows must not vanish from under the cursor: `←` brings the finished
        // roster back, and an expanded pane keeps its row visible while read.
        let mut pane = SubPane::default();
        pane.begin("alpha".to_string(), "", 0);
        pane.end(1_000);
        assert!(pane.roster_view(2_000).rows.is_empty());

        assert!(pane.move_cursor(-1), "still reachable after it hid");
        assert_eq!(pane.roster_view(2_000).rows.len(), 2);

        pane.move_cursor(1);
        assert!(pane.expand());
        pane.selecting = false;
        assert_eq!(
            pane.roster_view(2_000).rows.len(),
            2,
            "an expanded row stays on screen even with the cursor hidden"
        );

        pane.collapse();
        assert!(pane.roster_view(2_000).rows.is_empty(), "Esc puts it away");
    }

    #[test]
    fn a_row_counts_the_pass_in_flight_and_hands_over_when_it_completes() {
        use crate::status::{Status, WorkerState};

        // A local pass reports nothing until it finishes, which for a long one
        // is minutes of a blank column. The row tracks the worker's status
        // snapshots meanwhile — the same source the main progress line uses.
        let mut pane = SubPane::default();
        pane.begin("sub-agent".to_string(), "count", 0);
        assert_eq!(pane.runs[0].tokens_text(), "", "nothing counted yet");

        pane.note_status(&Status {
            state: WorkerState::Prefill,
            prefill_done: 8,
            prefill_total: 422,
            ..Status::default()
        });
        assert_eq!(pane.runs[0].tokens_text(), "↑ 8/422 tokens");

        pane.note_status(&Status {
            state: WorkerState::Generating,
            generated: 266,
            ..Status::default()
        });
        assert_eq!(pane.runs[0].tokens_text(), "↓ 266 tokens");

        // The completed pass folds in, and the live figures it was tracking are
        // dropped in the same breath so the two cannot double-count.
        pane.add_tokens(None, 422, 266);
        assert!(pane.runs[0].live.is_none());
        assert_eq!(pane.runs[0].tokens_text(), "↓ 266 tokens");

        // A second pass counts on top of the first.
        pane.note_status(&Status {
            state: WorkerState::Generating,
            generated: 100,
            ..Status::default()
        });
        assert_eq!(pane.runs[0].tokens_text(), "↓ 366 tokens");

        // Once the run ends, only what it actually completed is reported.
        pane.end(1_000);
        assert_eq!(pane.runs[0].tokens_text(), "↓ 266 tokens");
    }

    #[test]
    fn a_fanouts_rows_ignore_the_status_line_they_cannot_own() {
        use crate::status::{Status, WorkerState};

        // The snapshot describes one pass; with several agents in flight there
        // is no honest row to attribute it to, so a fan-out's rows are fed only
        // by their own per-pass tallies.
        let mut pane = SubPane::default();
        pane.begin("alpha".to_string(), "a", 0);
        pane.begin("beta".to_string(), "b", 0);
        pane.note_status(&Status {
            state: WorkerState::Generating,
            generated: 999,
            ..Status::default()
        });
        assert!(pane.runs.iter().all(|r| r.live.is_none()), "not attributed");

        pane.add_tokens(Some("alpha"), 10, 40);
        assert_eq!(pane.runs[0].tokens_text(), "↓ 40 tokens");
        assert_eq!(pane.runs[1].tokens_text(), "");
    }

    #[test]
    fn a_row_shows_its_elapsed_time_before_any_tokens_are_counted() {
        // A local engine reports a pass's tokens only when the pass completes.
        // Until then the row still has something true to say.
        let mut pane = SubPane::default();
        pane.begin("alpha".to_string(), "", 0);
        let row = &pane.roster_view(12_000).rows[1];
        assert_eq!(row.elapsed, "12s");
        assert_eq!(row.tokens, "");

        let mut term = ratatui::Terminal::new(ratatui::backend::TestBackend::new(60, 10)).unwrap();
        let log = OutputLog::new();
        let mut view = OutputView::default();
        let roster = pane.roster_view(12_000);
        term.draw(|f| {
            draw(
                f,
                &log,
                Some(InputState::new("", 0)),
                "idle",
                &mut view,
                None,
                &TaskView::default(),
                None,
                &roster,
            );
        })
        .unwrap();
        let buf = term.backend().buffer().clone();
        let row = (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
            })
            .find(|r| r.contains("alpha"))
            .expect("the agent row is drawn");
        assert!(
            row.trim_end().ends_with("12s"),
            "elapsed is shown on its own: {row:?}"
        );
    }

    #[test]
    fn the_roster_cursor_is_hidden_until_the_user_reaches_for_it() {
        let mut pane = SubPane::default();
        pane.begin("alpha".to_string(), "", 0);
        assert!(
            pane.roster_view(0).rows.iter().all(|r| !r.cursor),
            "a quiet status readout until `←` is pressed"
        );
        pane.move_cursor(-1);
        assert!(pane.roster_view(0).rows[0].cursor, "on `main` first");
    }

    #[test]
    fn elapsed_and_tokens_are_formatted_for_a_glance() {
        assert_eq!(fmt_elapsed(0), "0s");
        assert_eq!(fmt_elapsed(47_400), "47s");
        assert_eq!(fmt_elapsed(208_000), "3m 28s");
        assert_eq!(fmt_elapsed(3_840_000), "1h 4m");
        assert_eq!(fmt_tokens(0), "", "nothing spent yet reads as nothing");
        assert_eq!(fmt_tokens(999), "↓ 999 tokens");
        assert_eq!(fmt_tokens(51_900), "↓ 51.9k tokens");
        assert_eq!(fmt_tokens(2_000_000), "↓ 2M tokens");
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
        pane.begin("research".to_string(), "", 0);
        pane.runs[0].log.push_plain("sub output");
        pane.runs[0].log.push_plain("sub output two");
        pane.runs[0].view.top = 3;

        // Not expanded: everything routes to the main pane, unchanged.
        assert_eq!(pane.active_log(&main_log).line_count(), 1);
        assert_eq!(pane.active_view(&mut main_view).top, 7);

        pane.move_cursor(-1);
        pane.move_cursor(1);
        assert!(pane.expand());
        assert_eq!(pane.active_log(&main_log).line_count(), 2);
        let v = pane.active_view(&mut main_view);
        assert_eq!(v.top, 3);
        // Scrolling the visible pane leaves the hidden one alone.
        v.top = 9;
        assert_eq!(pane.runs[0].view.top, 9);
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
                &RosterView::default(),
            );
            draw_slash_menu(f, "/comp", &menu, 0);
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
        // An empty list adds no task counter to the status bar (the
        // "✓ Tasks: n/n"
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
        let g0 = frame_geom(area, true, 1, 0, 0);
        let (out0, in0, st0, rule0, rule_bot0, strip0) = (
            g0.output,
            g0.input,
            g0.status,
            g0.rule_top,
            g0.rule_bottom,
            g0.strip,
        );
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
        let g3 = frame_geom(area, true, 1, 3, 0);
        let (out3, rule3, strip3) = (g3.output, g3.rule_top, g3.strip);
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
                &RosterView::default(),
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
    fn session_name_floats_right_on_the_rule_above_the_prompt() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let log = OutputLog::new();
        let mut view = OutputView::default();
        let row_at = |buf: &ratatui::buffer::Buffer, y: u16| -> String {
            (0..buf.area.width)
                .map(|x| buf[(x, y)].symbol())
                .collect::<String>()
        };
        let render = |w: u16, view: &mut OutputView| {
            let mut term = Terminal::new(TestBackend::new(w, 8)).unwrap();
            term.draw(|f| {
                draw(
                    f,
                    &log,
                    Some(InputState::new("hi", 2)),
                    "idle",
                    view,
                    None,
                    &TaskView::default(),
                    None,
                    &RosterView::default(),
                );
            })
            .unwrap();
            let buf = term.backend().buffer();
            let y = input_row(buf).expect("prompt row present") - 1;
            row_at(buf, y)
        };

        set_session_name("deadly-einstein");
        let rule = render(40, &mut view);
        assert!(
            rule.ends_with("─ deadly-einstein ──"),
            "name floats right with a two-column tail of rule: {rule:?}"
        );
        assert!(rule.starts_with("──────"), "rule still leads: {rule:?}");

        // Too narrow to keep a readable run of rule: the label is dropped
        // rather than eating the whole line.
        let narrow = render(20, &mut view);
        assert_eq!(narrow, "─".repeat(20), "no room for the name");

        set_session_name("");
        let bare = render(40, &mut view);
        assert_eq!(bare, "─".repeat(40), "no name published, plain rule");
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
                &RosterView::default(),
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
                &RosterView::default(),
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
                &RosterView::default(),
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
            let g = frame_geom(screen, true, input_height(input_text, 78), 0, 0);
            let (output, input, status, rule) = (g.output, g.input, g.status, g.rule_top);
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
        assert!(a.open(cmd, false, 7), "{cmd} did not open");
        a
    }

    /// An arcade showing the screensaver.
    ///
    /// The face is asked for rather than left to `ui.screensaverFace`: a test
    /// that silently drew a different face would be worse than no test. The
    /// rain is the only built-in face left — the others are plugins now — so
    /// it is what these tests draw.
    fn screensaver() -> crate::arcade::Arcade {
        let mut a = crate::arcade::Arcade::new();
        a.open_screensaver_as(crate::arcade::ScreensaverFace::Matrix, 7);
        // The rain fades in per column, so a fresh field is nearly empty and a
        // test about coverage would measure the wrong thing.
        for _ in 0..40 {
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

    /// The generic blitter draws what a component painted, carries the two
    /// attributes the packed format adds over `arcade::Glyph`, and — the part
    /// that matters — drops glyphs outside the current area instead of
    /// clamping them, since a component may be a frame behind a resize.
    #[test]
    fn a_wasm_frame_draws_its_glyphs_and_clips_to_the_area() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let glyph = |x, y, ch| crate::arcade::Glyph {
            x,
            y,
            ch,
            color: (10, 200, 30),
        };
        let open = crate::wasmreg::OpenFrame {
            id: "dev.plank.test".to_string(),
            screensaver: false,
            veiled: false,
            last: crate::wasmglyph::GlyphFrame {
                w: 80,
                h: 23,
                glyphs: vec![glyph(0, 0, 'A'), glyph(3, 1, 'B'), glyph(200, 5, 'X')],
                bold: vec![true, false, false],
                bg: vec![None, Some((5, 6, 7)), None],
            },
        };
        let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
        term.draw(|f| draw_wasm_frame(f, &open)).unwrap();
        let buf = term.backend().buffer().clone();

        let cell = |x, y| buf.cell(Position::new(x, y)).unwrap().clone();
        assert_eq!(cell(0, 0).symbol(), "A");
        assert_eq!(cell(0, 0).fg, Color::Rgb(10, 200, 30));
        assert!(
            cell(0, 0).style().add_modifier.contains(Modifier::BOLD),
            "the bold flag did not reach the cell"
        );
        assert_eq!(cell(3, 1).symbol(), "B");
        assert_eq!(cell(3, 1).bg, Color::Rgb(5, 6, 7), "background colour");
        assert_eq!(
            cell(0, 0).bg,
            Color::Rgb(0, 0, 0),
            "a glyph without a background keeps the frame's own ground"
        );
        // The out-of-area glyph was dropped, not folded onto the last column:
        // clamping would pile a whole edge into one cell after a resize.
        assert_eq!(cell(79, 5).symbol(), " ");
    }

    /// A veiled component leaves the transcript readable underneath, exactly
    /// as a veiled arcade face does — they share the ground painter, and this
    /// is what pins that they keep sharing it.
    #[test]
    fn a_veiled_wasm_frame_leaves_the_ui_visible_underneath() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut log = OutputLog::new();
        for i in 0..40 {
            log.visible_text(&format!("output line {i} still readable"));
            log.end_line();
        }
        let draw = |veiled: bool| {
            let open = crate::wasmreg::OpenFrame {
                id: "dev.plank.test".to_string(),
                screensaver: false,
                veiled,
                last: crate::wasmglyph::GlyphFrame::default(),
            };
            let mut view = OutputView::default();
            let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
            term.draw(|f| {
                render_output(f, f.area(), &log, &mut view, None);
                draw_wasm_frame(f, &open);
            })
            .unwrap();
            term.backend().buffer().clone()
        };

        let text_of = |buf: &ratatui::buffer::Buffer| {
            (0..24)
                .map(|y| {
                    (0..80)
                        .map(|x| buf.cell(Position::new(x, y)).unwrap().symbol().to_string())
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert!(
            text_of(&draw(true)).contains("still readable"),
            "a veiled frame hid the transcript"
        );
        assert!(
            !text_of(&draw(false)).contains("still readable"),
            "an opaque frame left the transcript showing"
        );
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
        // A *game*, not the screensaver: translucency is what happens when a
        // game opens over a running turn, and the rain — the only built-in
        // face left — paints nearly every cell, so it would hide the
        // transcript whatever the veil did. Breakout is sparse, which is the
        // property this test is really about.
        let draw = |translucent: bool| {
            let mut arcade = opened("/breakout");
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
    fn the_screensaver_covers_the_whole_frame() {
        let rows = arcade_frame(&screensaver(), 80, 24);
        // The screensaver has no footer, so every row is drawable; the field
        // should reach the top and the bottom, not clump in a band.
        let painted: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.trim().is_empty())
            .map(|(i, _)| i)
            .collect();
        assert!(painted.len() > 10, "only {} rows drawn", painted.len());
        assert!(painted[0] < 4, "nothing near the top: {painted:?}");
        assert!(painted[painted.len() - 1] > 18, "nothing near the bottom");
        assert!(
            !rows
                .iter()
                .any(|r| r.contains("speed") || r.contains("quit")),
            "the screensaver must not offer the games' controls: {rows:?}"
        );
    }

    /// A day in seconds, for the `/kvcache` modal tests.
    const KV_DAY: u64 = 86_400;

    /// One freshly-used node for the `/kvcache` modal tests.
    fn kv_meta(
        role: crate::kvmeta::KvRole,
        fp: &str,
        parent: Option<&str>,
    ) -> crate::kvmeta::KvMeta {
        crate::kvmeta::KvMeta {
            version: crate::kvmeta::META_VERSION,
            role,
            fingerprint: fp.to_owned(),
            parent: parent.map(ToOwned::to_owned),
            model: "m".into(),
            created: 0,
            last_used: 1_000 * KV_DAY,
            hits: 1,
            bytes: 4096,
            pinned: false,
            label: crate::kvmeta::KvLabel::Unknown,
        }
    }

    /// A pane over a two-node lineage, for the `/kvcache` modal tests.
    fn kv_test_pane() -> crate::kvpane::KvPane {
        const DAY: u64 = KV_DAY;
        crate::kvpane::KvPane::new(
            crate::kvtree::build(vec![
                kv_meta(crate::kvmeta::KvRole::System, "a19f", None),
                kv_meta(crate::kvmeta::KvRole::Session, "7c02", Some("a19f")),
            ]),
            crate::kvgc::SweepPolicy {
                ttl_session_secs: 14 * DAY,
                ttl_tier_secs: 30 * DAY,
                max_bytes: 0,
            },
            Vec::new(),
            1_000 * DAY,
        )
    }

    #[test]
    fn the_kvcache_modal_fits_inside_a_small_terminal() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // The same clamping `draw_config` relies on: on a 24x8 terminal the
        // modal must stay inside the frame rather than panicking in ratatui's
        // Rect math.
        let pane = kv_test_pane();
        let mut term = Terminal::new(TestBackend::new(24, 8)).unwrap();
        term.draw(|f| draw_kvcache(f, &pane)).unwrap();

        // A terminal narrower than the modal's floor used to produce a rect
        // *larger* than the frame — ratatui does not intersect with the
        // viewport, so `Clear` then indexed out of bounds and aborted the TUI.
        let mut tiny = Terminal::new(TestBackend::new(10, 5)).unwrap();
        tiny.draw(|f| draw_kvcache(f, &pane)).unwrap();

        // Too short for even one tree line: the footer and the key hints are the
        // last things to go, so the blank separator must be dropped rather than
        // spending one of the two remaining lines. With the separator kept, this
        // frame drew a blank line and the footer and lost the `Esc close` hint.
        let mut squat = Terminal::new(TestBackend::new(90, 6)).unwrap();
        squat.draw(|f| draw_kvcache(f, &pane)).unwrap();
        let buf = squat.backend().buffer().clone();
        let text: String = (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("reclaimable"), "{text}");
        assert!(text.contains("Esc close"), "{text}");

        // And the geometry itself: whatever the frame size, the rect fits
        // inside it. The 24x8 + 40-lines case also pins the height clamp.
        for (w, h) in [(4u16, 1u16), (10, 5), (24, 8), (120, 40)] {
            let area = Rect::new(0, 0, w, h);
            for lines in [1usize, 40] {
                for rect in [kvcache_rect(area, lines), config_rect(area, lines)] {
                    assert!(rect.width <= area.width, "{w}x{h} lines {lines}: {rect:?}");
                    assert!(
                        rect.height <= area.height,
                        "{w}x{h} lines {lines}: {rect:?}"
                    );
                    assert!(rect.right() <= area.right(), "{w}x{h}: {rect:?}");
                    assert!(rect.bottom() <= area.bottom(), "{w}x{h}: {rect:?}");
                }
            }
        }
    }

    #[test]
    fn the_kvcache_modal_scrolls_the_selected_row_into_view() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        // A chain far taller than the terminal: without windowing the cursor
        // would rest on a clipped row, and `d`/`y` would delete an unseen blob.
        //
        // Every session node carries a `Session` label, so every row emits a
        // second detail line. That is the point: the window is measured in
        // *lines*, not rows, so a fixture whose rows are one line each never
        // exercises the arithmetic this test exists to protect.
        let mut metas = vec![kv_meta(crate::kvmeta::KvRole::System, "root", None)];
        for i in 0..30u32 {
            let fp = format!("blob{i:02}");
            let parent = if i == 0 {
                "root".to_owned()
            } else {
                format!("blob{:02}", i - 1)
            };
            let mut m = kv_meta(crate::kvmeta::KvRole::Session, &fp, Some(&parent));
            m.label = crate::kvmeta::KvLabel::Session {
                name: format!("sess{i:02}"),
                title: format!("a title for row {i:02}"),
            };
            metas.push(m);
        }
        let rows_total = metas.len();
        let build = || {
            crate::kvpane::KvPane::new(
                crate::kvtree::build(metas.clone()),
                crate::kvgc::SweepPolicy {
                    ttl_session_secs: 14 * KV_DAY,
                    ttl_tier_secs: 30 * KV_DAY,
                    max_bytes: 0,
                },
                Vec::new(),
                1_000 * KV_DAY,
            )
        };

        // The cursor at the top, the middle and the bottom, on a roomy frame and
        // on one barely taller than the footer.
        for down in [0usize, rows_total / 2, rows_total - 1] {
            let mut pane = build();
            for _ in 0..down {
                pane.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
            }
            let rows = pane.rows();
            let want = rows
                .iter()
                .find(|r| r.selected)
                .map(|r| r.label.clone())
                .expect("some row is always selected");

            for (w, h) in [(90u16, 14u16), (90, 8)] {
                let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
                term.draw(|f| draw_kvcache(f, &pane)).unwrap();
                let buf = term.backend().buffer().clone();
                let line_at = |y: u16| -> String {
                    (0..buf.area.width)
                        .map(|x| buf[(x, y)].symbol())
                        .collect::<String>()
                };
                // Find the row the *terminal* is highlighting, not the row the
                // pane thinks is selected: asserting the text is merely present
                // would pass while the highlight sat on a different line.
                let reversed: Vec<String> = (0..buf.area.height)
                    .filter(|&y| {
                        (0..buf.area.width)
                            .any(|x| buf[(x, y)].modifier.contains(Modifier::REVERSED))
                    })
                    .map(line_at)
                    .collect();
                assert_eq!(
                    reversed.len(),
                    1,
                    "exactly one drawn row is highlighted at {w}x{h}, down {down}: {reversed:?}"
                );
                assert!(
                    reversed[0].contains(want.trim()),
                    "the highlighted row must be the selected one at {w}x{h}, down {down}: want {want:?}, got {:?}",
                    reversed[0]
                );
            }
        }

        // ...and the window really did scroll, plus the footer and hints survive
        // it, on the frame with room for both.
        let mut pane = build();
        for _ in 0..25 {
            pane.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        }
        let mut term = Terminal::new(TestBackend::new(90, 14)).unwrap();
        term.draw(|f| draw_kvcache(f, &pane)).unwrap();
        let buf = term.backend().buffer().clone();
        let text: String = (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !text.contains(" root"),
            "the top of the tree should have scrolled away: {text}"
        );
        assert!(text.contains("reclaimable"), "{text}");
        assert!(text.contains("Esc close"), "{text}");
    }

    #[test]
    fn the_kvcache_modal_draws_the_tree_and_its_footer() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let pane = kv_test_pane();
        let mut term = Terminal::new(TestBackend::new(90, 14)).unwrap();
        term.draw(|f| draw_kvcache(f, &pane)).unwrap();
        let buf = term.backend().buffer().clone();
        let text: String = (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("kv cache"), "{text}");
        assert!(text.contains("system"), "{text}");
        assert!(text.contains("session"), "{text}");
        assert!(text.contains("reclaimable"), "{text}");
        assert!(text.contains("Esc close"), "{text}");
    }

    /// Saved sessions for the `/resume` picker tests, most-recent first.
    fn resume_test_pane(n: usize) -> crate::resumepane::ResumePane {
        let entries = (0..n)
            .map(|i| crate::session::SessionEntry {
                id: format!("session-{i}"),
                title: format!("Session number {i}"),
                created_at: 900 - i as u64,
                last_used: 900 - i as u64,
                file_size: 1024 * (i as u64 + 1),
                tag: String::new(),
                last_prompt: String::new(),
                payload_bytes: 0,
                path: std::path::PathBuf::from(format!("/tmp/session-{i}.kv")),
            })
            .collect();
        crate::resumepane::ResumePane::new(entries, 1000).with_scope("plank")
    }

    /// Renders `pane` at `w`x`h` and returns the frame as text.
    fn resume_frame(pane: &crate::resumepane::ResumePane, w: u16, h: u16) -> String {
        let mut term = ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
        term.draw(|f| draw_resume(f, pane)).unwrap();
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

    #[test]
    fn the_resume_picker_draws_its_header_search_box_scope_and_hints() {
        let text = resume_frame(&resume_test_pane(3), 90, 24);
        assert!(text.contains("Resume session (1 of 3)"), "{text}");
        assert!(text.contains("Search…"), "{text}");
        assert!(text.contains("plank"), "the project label: {text}");
        assert!(text.contains("Session number 0"), "{text}");
        assert!(text.contains("❯ Session number 0"), "the cursor: {text}");
        assert!(text.contains("Space to preview"), "{text}");
        // The panel is anchored to the bottom, leaving the status row free.
        let last = text.lines().next_back().unwrap_or_default().trim();
        assert!(last.is_empty(), "the status row was painted over: {text}");
    }

    #[test]
    fn typing_shows_the_query_and_an_empty_result_says_so() {
        let mut pane = resume_test_pane(3);
        for c in "zzz".chars() {
            pane.handle_key(ratatui::crossterm::event::KeyEvent::new(
                ratatui::crossterm::event::KeyCode::Char(c),
                ratatui::crossterm::event::KeyModifiers::NONE,
            ));
        }
        let text = resume_frame(&pane, 90, 24);
        assert!(text.contains("zzz"), "{text}");
        assert!(!text.contains("Search…"), "placeholder gone: {text}");
        assert!(text.contains("no session matches"), "{text}");
        assert!(text.contains("Resume session (0 of 0)"), "{text}");
    }

    /// The bug this guards: with more sessions than rows, the selected one has
    /// to stay on screen — Enter resumes it, and it must be the one being read.
    #[test]
    fn a_long_listing_scrolls_to_keep_the_selection_visible() {
        let mut pane = resume_test_pane(40);
        for _ in 0..12 {
            pane.handle_key(ratatui::crossterm::event::KeyEvent::new(
                ratatui::crossterm::event::KeyCode::Down,
                ratatui::crossterm::event::KeyModifiers::NONE,
            ));
        }
        let text = resume_frame(&pane, 90, 24);
        assert!(text.contains("❯ Session number 12"), "{text}");
        assert!(
            !text.contains("Session number 0\n"),
            "scrolled away: {text}"
        );
        assert!(text.contains('↑'), "more above: {text}");
        assert!(text.contains('↓'), "more below: {text}");
    }

    /// Small terminals: ratatui does not clip rects for you, so a panel taller
    /// or wider than the frame aborts the TUI rather than looking wrong.
    #[test]
    fn the_resume_picker_survives_a_tiny_terminal() {
        for (w, h) in [(4u16, 1u16), (10, 3), (24, 8), (200, 60)] {
            resume_frame(&resume_test_pane(6), w, h);
        }
    }

    /// An armed wipe has to say what it is about to take, in the panel, before
    /// the second press: the hint line is the only warning there is.
    #[test]
    fn an_armed_wipe_names_the_damage_in_the_hints() {
        let mut pane = resume_test_pane(6);
        pane.handle_key(ratatui::crossterm::event::KeyEvent::new(
            ratatui::crossterm::event::KeyCode::Char('w'),
            ratatui::crossterm::event::KeyModifiers::CONTROL,
        ));
        let text = resume_frame(&pane, 100, 24);
        assert!(text.contains("ALL 6 saved sessions"), "{text}");
        assert!(text.contains("any other key cancels"), "{text}");
    }

    #[test]
    fn a_rename_takes_over_the_search_box() {
        let mut pane = resume_test_pane(3);
        pane.handle_key(ratatui::crossterm::event::KeyEvent::new(
            ratatui::crossterm::event::KeyCode::Char('r'),
            ratatui::crossterm::event::KeyModifiers::CONTROL,
        ));
        let text = resume_frame(&pane, 90, 24);
        assert!(text.contains("rename"), "{text}");
        assert!(text.contains("session-0"), "prefilled with the id: {text}");
        assert!(text.contains("Enter to rename"), "{text}");
    }
}
