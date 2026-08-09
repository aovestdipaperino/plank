// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Interactive REPL and headless front-ends over the agent turn loop.
//!
//! Port of the "Interactive Runtime Loop" section of `ds4_agent.c`. Like the
//! C, the TUI runs each turn on a worker thread (see `crate::worker`) while
//! the UI thread keeps handling input — the next prompt stays editable and
//! queueable during generation. The plain line REPL (piped stdin) stays a
//! synchronous inline loop: without a live terminal there is no input to
//! multiplex.

use std::io::{BufRead, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ratatui::crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags, MouseButton,
    MouseEventKind, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};

use crate::compact;
use crate::config::{AgentConfig, slash_command_known};
use crate::context::{ContextContent, ContextTokens};
use crate::dsml::ToolCall;
use crate::editor::{History, LineBuffer, default_history_path};
use crate::engine::{Engine, EngineEvent};
use crate::remote::control::RemoteState;
use crate::render::{RenderOptions, TokenRenderer};
use crate::session::{Message, Session, SessionStore};
use crate::status::{self, Status, WorkerState};
use crate::sysprompt::{self, SystemPromptReminder};
use crate::tools::{ToolContext, dispatch, dispatch_all};
use crate::trace::Trace;
use crate::tui::{self, OutputLog};
use crate::viz::{RenderSink, StreamRenderer};
use crate::worker::{self, BroadcastBus, ChannelSink, TurnShared, UiEvent};

/// UI-thread state for `--ui-remote` remote control.
///
/// Owns the listener handle, the queue of keys injected by remote clients,
/// and the `snapshot`/`uitree` requests whose replies are deliberately held
/// back until the screen reflects those keys (see [`UiRemote::drain`]).
///
/// It is wrapped in a `Mutex` and shared by `Arc` rather than passed as
/// `&mut` because the TUI turn loop hands `&mut self` (the whole `Agent`) to
/// a worker closure while the same tick still needs the remote state; the
/// `Mutex` is uncontended in practice — only the UI thread ever locks it,
/// the listener thread talks over channels.
#[derive(Debug)]
pub struct UiRemote {
    /// Listener handle. `None` in unit tests, which exercise the queueing
    /// logic without binding a port.
    handle: Option<crate::uiremote::RemoteHandle>,
    /// Key events queued by `keypress`, consumed by [`next_event`].
    injected: std::collections::VecDeque<Event>,
    /// `snapshot`/`uitree` requests waiting for a post-key frame.
    deferred: Vec<crate::uiremote::Pending>,
    /// The frame captured inside the qualifying draw closure. The terminal's
    /// current buffer is already the *next* frame's once `draw` returns, so
    /// the screen has to be read while the frame is still live.
    captured: Option<CapturedFrame>,
}

/// One rendered frame, recorded for a deferred `snapshot`/`uitree` reply.
#[derive(Debug)]
struct CapturedFrame {
    /// The screen as ANSI text.
    ansi: String,
    /// Pre-rendered `uitree` JSON, spliced into the reply verbatim.
    tree: String,
    /// Frame width in columns.
    cols: u16,
    /// Frame height in rows.
    rows: u16,
    /// Cursor position, or `None` when the cursor is hidden.
    cursor: Option<(u16, u16)>,
}

impl UiRemote {
    /// Wraps a started listener for the TUI loops.
    fn new(handle: crate::uiremote::RemoteHandle) -> Self {
        Self {
            handle: Some(handle),
            injected: std::collections::VecDeque::new(),
            deferred: Vec::new(),
            captured: None,
        }
    }

    /// A detached instance with no listener, for unit tests of the queueing
    /// and deferral rules.
    #[cfg(test)]
    fn detached() -> Self {
        Self {
            handle: None,
            injected: std::collections::VecDeque::new(),
            deferred: Vec::new(),
            captured: None,
        }
    }

    /// Takes every command the listener has queued.
    ///
    /// `keypress` is answered immediately — the client only needs to know the
    /// keys were accepted. `snapshot` and `uitree` are held: answering them
    /// now would describe the screen *before* the keys took effect, which is
    /// exactly the race this feature exists to remove.
    fn drain(&mut self) {
        while let Some(p) = self
            .handle
            .as_ref()
            .and_then(crate::uiremote::RemoteHandle::try_recv)
        {
            match p.cmd {
                crate::uiremote::RemoteCmd::Keypress(keys) => {
                    for k in keys {
                        self.injected.push_back(Event::Key(k));
                    }
                    let _ = p.reply.send(crate::uiremote::ok_reply(&[]));
                }
                _ => self.deferred.push(p),
            }
        }
    }

    /// Called at the end of every draw closure: records the finished frame
    /// when a deferred reply is waiting and every injected key has already
    /// been consumed.
    fn capture(&mut self, frame: &mut ratatui::Frame) {
        if self.deferred.is_empty() || !self.injected.is_empty() {
            return;
        }
        let area = frame.area();
        let cursor = crate::uiremote::frame_cursor();
        self.captured = Some(CapturedFrame {
            ansi: crate::uiremote::buffer_to_ansi(frame.buffer_mut()),
            tree: crate::uiremote::frame_tree(),
            cols: area.width,
            rows: area.height,
            cursor,
        });
    }

    /// Answers every still-deferred request with an error, so a client is
    /// never left waiting out the reply timeout after the UI has gone.
    fn abandon(&mut self) {
        for p in self.deferred.drain(..) {
            let _ = p.reply.send(crate::uiremote::error_reply("ui exiting"));
        }
    }

    /// Called just after `terminal.draw` returns: answers the deferred
    /// requests from the frame [`capture`](Self::capture) recorded, if any.
    fn service(&mut self) {
        let Some(frame) = self.captured.take() else {
            return;
        };
        // `cursor` is a two-element array, or JSON null when hidden — never
        // invented coordinates, so a harness can tell "hidden" from "at 0,0".
        let cursor = frame
            .cursor
            .map_or_else(|| "null".to_string(), |(x, y)| format!("[{x},{y}]"));
        for p in self.deferred.drain(..) {
            let reply = match p.cmd {
                crate::uiremote::RemoteCmd::Snapshot => crate::uiremote::ok_reply_raw(&[
                    ("ansi", &crate::uiremote::json_string(&frame.ansi)),
                    ("cols", &frame.cols.to_string()),
                    ("rows", &frame.rows.to_string()),
                    ("cursor", &cursor),
                ]),
                // Spliced raw so `tree` is a real object, not a string a
                // client would have to decode a second time.
                crate::uiremote::RemoteCmd::Uitree => {
                    crate::uiremote::ok_reply_raw(&[("tree", &frame.tree)])
                }
                // `drain` never defers a keypress.
                crate::uiremote::RemoteCmd::Keypress(_) => {
                    crate::uiremote::error_reply("keypress deferred unexpectedly")
                }
            };
            let _ = p.reply.send(reply);
        }
    }
}

/// Drains remote commands for this tick, if remote control is on.
fn remote_drain(remote: Option<&Mutex<UiRemote>>) {
    if let Some(m) = remote
        && let Ok(mut g) = m.lock()
    {
        g.drain();
    }
}

/// Captures the just-drawn frame for any deferred remote request. Call as the
/// last statement inside a `terminal.draw` closure.
fn remote_capture(remote: Option<&Mutex<UiRemote>>, frame: &mut ratatui::Frame) {
    if let Some(m) = remote
        && let Ok(mut g) = m.lock()
    {
        g.capture(frame);
    }
}

/// Answers deferred remote requests. Call right after `terminal.draw` returns.
fn remote_service(remote: Option<&Mutex<UiRemote>>) {
    if let Some(m) = remote
        && let Ok(mut g) = m.lock()
    {
        g.service();
    }
}

/// Fails any still-deferred remote request. Call when a key loop exits.
///
/// A `snapshot` deferred just before `/quit` or Ctrl-C would otherwise never
/// be answered, leaving the harness blocked for the full reply timeout on
/// every teardown.
fn remote_abandon(remote: Option<&Mutex<UiRemote>>) {
    if let Some(m) = remote
        && let Ok(mut g) = m.lock()
    {
        g.abandon();
    }
}

/// Map a ratatui [`Color`](ratatui::style::Color) to concrete 24-bit RGB for
/// the CRT-off frame image. `Reset` becomes a neutral gray (the assumed
/// default foreground); for the background variant, where `Reset` should be
/// black, see [`bg_to_rgb`].
fn color_to_rgb(c: ratatui::style::Color) -> (u8, u8, u8) {
    use ratatui::style::Color;
    match c {
        Color::Rgb(r, g, b) => (r, g, b),
        Color::Reset => (192, 192, 192),
        Color::Black => (0, 0, 0),
        Color::Red => (205, 0, 0),
        Color::Green => (0, 205, 0),
        Color::Yellow => (205, 205, 0),
        Color::Blue => (0, 0, 238),
        Color::Magenta => (205, 0, 205),
        Color::Cyan => (0, 205, 205),
        Color::Gray => (229, 229, 229),
        Color::DarkGray => (127, 127, 127),
        Color::LightRed => (255, 0, 0),
        Color::LightGreen => (0, 255, 0),
        Color::LightYellow => (255, 255, 0),
        Color::LightBlue => (92, 92, 255),
        Color::LightMagenta => (255, 0, 255),
        Color::LightCyan => (0, 255, 255),
        Color::White => (255, 255, 255),
        Color::Indexed(i) => indexed_to_rgb(i),
    }
}

/// Map a cell background [`Color`](ratatui::style::Color) to RGB. Identical to
/// [`color_to_rgb`] except `Reset` becomes black (the terminal's default
/// background) rather than the default-foreground gray.
fn bg_to_rgb(c: ratatui::style::Color) -> (u8, u8, u8) {
    match c {
        ratatui::style::Color::Reset => (0, 0, 0),
        other => color_to_rgb(other),
    }
}

/// The single RGB colour that represents a cell in the CRT-off frame image.
///
/// A cell with a visible glyph is drawn in its foreground colour; a blank cell
/// (space or empty) is drawn in its background colour, so background fills like
/// the status bar and selection highlight keep their colour instead of
/// collapsing to black. The `REVERSED` modifier swaps foreground and
/// background before the choice, matching how the terminal paints it. Other
/// text styles (bold/dim/italic/underline) are not represented (issue #55).
fn cell_rgb(cell: &ratatui::buffer::Cell) -> (u8, u8, u8) {
    let reversed = cell.modifier.contains(ratatui::style::Modifier::REVERSED);
    let (fg, bg) = if reversed {
        (cell.bg, cell.fg)
    } else {
        (cell.fg, cell.bg)
    };
    if cell.symbol().trim().is_empty() {
        bg_to_rgb(bg)
    } else {
        color_to_rgb(fg)
    }
}

/// Rasterize a rendered ratatui frame into an [`image::RgbaImage`] for the
/// CRT-off effect, reproducing both foreground glyphs and background fills
/// (see [`cell_rgb`]). crt-off packs two vertical image pixels per terminal
/// cell (a half-block per text row), so the image is `width` x `height * 2`
/// and each cell paints both of its pixels the same colour. Feed the result
/// to [`crt_off::animate`].
fn frame_to_image(buf: &ratatui::buffer::Buffer) -> image::RgbaImage {
    let area = buf.area();
    let w = u32::from(area.width).max(1);
    let h = u32::from(area.height).max(1);
    let mut img = image::RgbaImage::from_pixel(w, h * 2, image::Rgba([0, 0, 0, 255]));
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            let (cr, cg, cb) = cell_rgb(&buf[(x, y)]);
            let px = image::Rgba([cr, cg, cb, 255]);
            let px_x = u32::from(x - area.left());
            let px_y = u32::from(y - area.top()) * 2;
            img.put_pixel(px_x, px_y, px);
            img.put_pixel(px_x, px_y + 1, px);
        }
    }
    img
}

/// xterm-256 palette index to RGB: 0-15 base colors, 16-231 the 6x6x6 cube,
/// 232-255 the 24-step grayscale ramp.
#[allow(clippy::many_single_char_names)]
fn indexed_to_rgb(i: u8) -> (u8, u8, u8) {
    const BASE: [(u8, u8, u8); 16] = [
        (0, 0, 0),
        (205, 0, 0),
        (0, 205, 0),
        (205, 205, 0),
        (0, 0, 238),
        (205, 0, 205),
        (0, 205, 205),
        (229, 229, 229),
        (127, 127, 127),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (92, 92, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
    ];
    match i {
        0..=15 => BASE[i as usize],
        16..=231 => {
            let n = i - 16;
            let steps = [0u8, 95, 135, 175, 215, 255];
            let r = steps[(n / 36) as usize];
            let g = steps[((n / 6) % 6) as usize];
            let b = steps[(n % 6) as usize];
            (r, g, b)
        }
        232..=255 => {
            let v = 8 + 10 * (i - 232);
            (v, v, v)
        }
    }
}

/// The single event source both TUI key loops use.
///
/// Injected events (from `--ui-remote`) are drained before the terminal is
/// polled, so a remote keypress is always processed on the tick it arrives
/// and never waits out the poll timeout. Returns `Ok(None)` when the poll
/// timed out with nothing to report.
fn next_event(
    remote: Option<&Mutex<UiRemote>>,
    timeout: Duration,
) -> Result<Option<Event>, String> {
    if let Some(m) = remote
        && let Ok(mut g) = m.lock()
        && let Some(ev) = g.injected.pop_front()
    {
        return Ok(Some(ev));
    }
    if !event::poll(timeout).map_err(|e| e.to_string())? {
        return Ok(None);
    }
    let ev = event::read().map_err(|e| e.to_string())?;
    // Track window focus for the "unfocused" notification mode; the event
    // still flows to the caller (which ignores focus events).
    match ev {
        Event::FocusGained => crate::notify::set_focused(true),
        Event::FocusLost => crate::notify::set_focused(false),
        _ => {}
    }
    Ok(Some(ev))
}

/// Stdout writer that flushes after every write so tokens appear as streamed.
#[derive(Debug)]
struct FlushingStdout;

impl Write for FlushingStdout {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut out = std::io::stdout();
        let n = out.write(buf)?;
        out.flush()?;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        std::io::stdout().flush()
    }
}

/// Routes viz output into the markdown token renderer.
struct TerminalSink<W: Write> {
    renderer: TokenRenderer<W>,
}

impl<W: Write> RenderSink for TerminalSink<W> {
    fn visible_text(&mut self, text: &str) {
        self.renderer.set_in_think(false);
        self.renderer.write(text);
    }
    fn think_text(&mut self, text: &str) {
        self.renderer.set_in_think(true);
        self.renderer.write(text);
    }
    fn tool_text(&mut self, text: &str) {
        // Tool banners carry their own styling and must render verbatim; going
        // through `write` would markdown-process them and eat `*`/`_`/backtick
        // out of param values (e.g. `pattern=**/mod.rs`).
        self.renderer.set_in_think(false);
        self.renderer.plain(text);
    }
    fn error_text(&mut self, text: &str) {
        self.renderer.set_in_think(false);
        self.renderer.color("\x1b[1;31m");
        self.renderer.plain(text);
        self.renderer.color(ANSI_RESET);
    }
}

/// Wraps a `/btw` side question in the reference agent's system-reminder
/// framing: a separate lightweight answer over the shared context, no tools,
/// single response, and nothing enters the main conversation.
fn btw_user_message(question: &str) -> String {
    format!(
        "<system-reminder>This is a side question from the user. You must answer this question directly in a single response.\n\
         \n\
         IMPORTANT CONTEXT:\n\
         - You are a separate, lightweight agent spawned to answer this one question\n\
         - The main conversation is NOT interrupted - this exchange will not become part of it\n\
         - You share the conversation context but are a completely separate instance\n\
         - Do NOT reference being interrupted or what you were \"previously doing\" - that framing is incorrect\n\
         \n\
         CRITICAL CONSTRAINTS:\n\
         - You have NO tools available - you cannot read files, run commands, search, or take any actions\n\
         - This is a one-off response - there will be no follow-up turns\n\
         - You can ONLY provide information based on what you already know from the conversation context\n\
         - NEVER say things like \"Let me try...\", \"I'll now...\", \"Let me check...\", or promise to take any action\n\
         - If you don't know the answer, say so - do not offer to look it up or investigate\n\
         \n\
         Simply answer the question with the information you have.</system-reminder>\n\
         \n\
         {question}"
    )
}

/// A [`RenderSink`] that discards everything. Used by the sub-agent driver
/// (issue #50), whose sidechain generation must run the same [`StreamRenderer`]
/// call/greedy detection as a normal turn but produce no on-screen output.
struct NullSink;

impl RenderSink for NullSink {
    fn visible_text(&mut self, _text: &str) {}
    fn think_text(&mut self, _text: &str) {}
}

/// Where a headless sub-agent's rendered output goes. Sub-agents run
/// synchronously inside a turn, so this is set by whichever front end can
/// display the result: the TUI routes it over the worker→UI channel, the plain
/// REPL prints it inline, and the `--non-interactive` protocol path discards it
/// (its stdout carries a machine protocol that model text would corrupt).
#[derive(Debug, Default)]
pub enum SubSinkTarget {
    /// Discard sub-agent output (the default, and the non-interactive path).
    #[default]
    Null,
    /// Forward over the worker→UI channel as [`crate::worker::UiEvent::Sub`].
    Events(std::sync::mpsc::Sender<crate::worker::UiEvent>),
    /// Print inline on stdout (the plain REPL).
    Stdout,
}

/// Why a generation pass failed, which decides how it is worded to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PassError {
    /// A tool argument was rejected before the call completed.
    Preflight,
    /// The DSML itself was wrong.
    Dsml,
    /// The DSML was fine but sat inside `<think></think>`.
    InThink,
}

/// Builds the model-visible payload for a failed generation pass, matching
/// the C worker loop: a preflight failure is fed back verbatim, a DSML parse
/// failure gets the C's `invalid DSML tool call: ` prefix plus the syntax
/// reminder so the model can correct its markup.
///
/// A call made inside `<think></think>` is the exception, and a deliberate
/// divergence from the C (`ds4_agent.c:7853` routes it through the malformed
/// path): its markup was *valid*, it was in the wrong place. Prefixing it with
/// "invalid DSML tool call" and handing over the syntax reminder tells the
/// model to fix something that was never broken, so it gets the placement rule
/// verbatim — the same sentence the tools prompt already gave it — and no
/// syntax reminder at all.
fn tool_error_payload(kind: PassError, err: &str) -> String {
    match kind {
        PassError::Preflight => format!("Tool error: {err}\n"),
        // Written without a `\`-continued literal on purpose: continuations
        // strip the next line's indentation, which is a silent way to mangle
        // model-facing text (see CLAUDE.md).
        PassError::InThink => format!(
            concat!(
                "Tool error: {}\n",
                "The tool call was not run. Close the thinking block with ",
                "</think>, then emit the same call again.\n",
            ),
            sysprompt::IN_THINK_PROHIBITION
        ),
        PassError::Dsml => format!(
            "Tool error: invalid DSML tool call: {err}\n{}",
            sysprompt::dsml_syntax_reminder()
        ),
    }
}

/// Which [`PassError`] a finished pass represents.
fn pass_error_kind(preflight: bool, in_think_rejected: bool) -> PassError {
    if preflight {
        PassError::Preflight
    } else if in_think_rejected {
        PassError::InThink
    } else {
        PassError::Dsml
    }
}

/// What a compaction attempt actually did.
///
/// Mirrors the C's `err == "interrupted"` signal out of `agent_worker_compact`:
/// a Ctrl-C during the summary pass is not a failure, it means the turn stops
/// with the conversation exactly as it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Compacted {
    /// The transcript was summarized and rebuilt (or needed no compaction).
    Done,
    /// The user interrupted the summary pass; nothing was rebuilt.
    Interrupted,
    /// The summary pass produced nothing usable, so nothing was rebuilt. A
    /// separate outcome from [`Compacted::Interrupted`] because no one asked for
    /// it: the pass ran to completion and came back empty. Callers treat both the
    /// same way — abandon the turn, keep the conversation — but only an
    /// interrupt should consume the interrupt flag.
    NoSummary,
}

impl Compacted {
    /// True when the pass did not rebuild the transcript, whatever the reason.
    /// The turn is abandoned and the conversation is left as it was.
    fn aborted(self) -> bool {
        matches!(self, Compacted::Interrupted | Compacted::NoSummary)
    }
}

/// Shown when a compaction pass is interrupted, byte-for-byte with the C's
/// `agent_worker_compact`.
const COMPACT_INTERRUPTED: &str =
    "Compaction interrupted; keeping the previous conversation state.";

/// Reported when the summary pass produced nothing usable.
///
/// Rebuilding on an empty summary would destroy the transcript and put an empty
/// summary in its place, so a no-summary pass is a **failure**, not a quiet
/// success: the conversation is left as it was (minus whatever microcompact
/// already reclaimed, which is a real gain worth keeping) and the caller decides
/// what to do. The reference agent throws here for the same reason.
const COMPACT_NO_SUMMARY: &str =
    "Compaction produced no summary; keeping the previous conversation state.";

/// Receives a compaction pass's progress notes and drives the caller's redraw.
///
/// One trait rather than two closures for the same reason as
/// [`crate::tools::bash::ImmediateSink`]: on the TUI slash-command path both
/// halves need `&mut` access to the same output log, which two closures cannot
/// share.
pub(crate) trait CompactSink {
    /// One human-facing progress line.
    fn note(&mut self, text: String);
    /// Called as the compaction bar advances, so a caller that owns the
    /// terminal can repaint. Does nothing by default: the worker-thread path's
    /// UI thread is already redrawing on its own clock.
    fn redraw(&mut self) {}
}

/// Adapts a plain note-taking closure to [`CompactSink`], for callers that do
/// not own the terminal and so have nothing to repaint.
pub(crate) struct NoteSink<F: FnMut(String)>(pub F);

impl<F: FnMut(String)> CompactSink for NoteSink<F> {
    fn note(&mut self, text: String) {
        (self.0)(text);
    }
}

/// [`CompactSink`] for a `/compact` typed at the TUI prompt: notes land in the
/// output log and every progress step repaints the frame, so the status bar's
/// compaction bar advances even though the UI thread is the one blocked on the
/// compaction.
struct TuiCompactSink<'a> {
    log: &'a mut OutputLog,
    terminal: &'a mut ratatui::DefaultTerminal,
    view: &'a mut tui::OutputView,
}

impl CompactSink for TuiCompactSink<'_> {
    fn note(&mut self, text: String) {
        self.log.push_dim(text);
        self.redraw();
    }

    fn redraw(&mut self) {
        // Same slot as the worker path uses: the progress line below the output,
        // in place of the throbber and spinner verb.
        if let Some(frac) = status::compact_progress() {
            self.log
                .set_progress(Some(tui::compact_progress_line(frac)));
        }
        let (log, view) = (&*self.log, &mut *self.view);
        let _ = self.terminal.draw(|f| {
            tui::draw(
                f,
                log,
                None,
                "compacting (Esc to stop)",
                view,
                None,
                &tui::TaskView::default(),
                None,
            );
        });
    }
}

impl Drop for TuiCompactSink<'_> {
    /// Takes the compaction bar down however the pass ended — done, interrupted,
    /// or an engine error — so it cannot outlive the compaction on screen.
    fn drop(&mut self) {
        self.log.set_progress(None);
    }
}

/// Closes a `<think>` block the model left open when the turn is about to
/// continue with a `<tool_result>` user message.
///
/// Callers pass `ended_in_think` already gated on whether the pass actually
/// produced tool calls or a tool-error payload to feed back — never on
/// `ended_in_think` alone. An in-think stanza that gets discarded (parity
/// mode) or a stream cut short by user interrupt produces no continuation,
/// so the C reference never appends `</think>` there either; appending one
/// keeps the transcript well-formed only for the continuing case. The chat
/// template re-opens thinking on the next pass, so reasoning resumes. Cheap
/// in KV terms: the divergence is at the very end of the reply, so only that
/// tail reprefills.
fn close_open_think(text: &mut String, ended_in_think: bool) {
    if ended_in_think && !text.ends_with("</think>") {
        text.push_str("</think>");
    }
}

/// Builds the mid-stream edit preflight hook for a [`StreamRenderer`]: it
/// validates an `edit` call's `old` selector against the file on disk the
/// moment that parameter closes (the C's `agent_stream_preflight_closed_param`).
/// Captures only the working directory, so the live `ToolContext` stays free
/// for tool dispatch.
fn edit_preflight(
    ctx: &ToolContext,
) -> impl FnMut(&ToolCall) -> Result<(), String> + 'static + use<> {
    edit_preflight_cwd(&ctx.cwd)
}

/// [`edit_preflight`] from a bare working directory.
///
/// The returned closure is `'static` and captures nothing borrowed, so a
/// parallel fan-out can build one per slot and move each into its own thread —
/// which it must, the closure not being `Clone`.
fn edit_preflight_cwd(
    cwd: &std::path::Path,
) -> impl FnMut(&ToolCall) -> Result<(), String> + 'static + use<> {
    let ctx = ToolContext::new(cwd.to_path_buf());
    move |call| crate::tools::edit::preflight_edit_old(&ctx, call)
}

/// One sub-agent sidechain in a parallel fan-out.
struct FanoutSlot {
    /// Cache key, so the engine goes back where it came from.
    key: EngineKey,
    engine: Box<dyn Engine>,
    /// The slot's own transcript, holding just the framed task — fan-out slots
    /// are always clean-room.
    session: Session,
    label: String,
    /// Model text accumulated for the pane, flushed as one labelled block when
    /// the whole fan-out finishes.
    output: String,
    /// Tool calls carried from the generate phase to the dispatch phase.
    pending_calls: Vec<ToolCall>,
    done: bool,
    error: Option<String>,
}

/// Parses a `/btw <question>` line, returning the question. Accepts a
/// whitespace or `:` separator, mirroring `OpenClaw`'s `isBtwCommand`
/// matcher; returns `None` for other input or an empty question.
fn btw_question(line: &str) -> Option<&str> {
    let rest = line.trim().strip_prefix("/btw")?;
    let rest = if let Some(r) = rest.strip_prefix(':') {
        r
    } else if rest.starts_with(char::is_whitespace) {
        rest
    } else {
        return None; // "/btwfoo" is not a btw command
    };
    let q = rest.trim();
    if q.is_empty() { None } else { Some(q) }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Seeds an easter egg from the wall clock.
///
/// [`crate::arcade`] is deliberately clock-free — that is what makes a rally
/// replayable in a test — so the one moment randomness enters is here, when the
/// modal opens. Sub-second precision keeps two openings in the same second from
/// producing the same sky.
/// Whether `ev` is the user actually being present, for the idle clock behind
/// `ui.screensaver`.
///
/// Keys, mouse and pastes are somebody at the machine. Focus and resize events
/// are not: a window manager cycling focus, or another app resizing the
/// terminal, would otherwise keep the idle timer pinned at zero and the
/// screensaver would never come up on a busy desktop.
fn is_user_activity(ev: &Event) -> bool {
    matches!(ev, Event::Key(_) | Event::Mouse(_) | Event::Paste(_))
}

fn arcade_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(1, |d| d.as_secs() ^ u64::from(d.subsec_nanos()) << 20)
}

/// The arcade command `line` starts with, if any — `/pelota new` included.
/// `None` when `ui.easterEggs` is off, which is what keeps them unreachable
/// rather than merely unlisted.
fn arcade_command(line: &str) -> Option<&'static str> {
    crate::arcade::command_of(line)
}

/// Turns any-motion mouse reporting (DECSET 1003) on or off.
///
/// `EnableMouseCapture` asks for buttons and drags but not free hover, which is
/// what a paddle wants to follow. Crossterm has no command for 1003, so this
/// writes it directly; it is additive over the capture already in place, and is
/// switched back off when the arcade closes so the rest of the UI keeps
/// receiving only the events it expects. Best-effort: a terminal that does not
/// know the mode ignores it, and click-and-drag still steers.
fn arcade_hover_reporting(on: bool) {
    use std::io::Write as _;
    let mut out = std::io::stdout();
    let _ = out.write_all(if on { b"\x1b[?1003h" } else { b"\x1b[?1003l" });
    let _ = out.flush();
}

/// Injects the session-start context as **two** user messages — project-stable
/// first, then session-volatile — so Tier 2 and Tier 3 of the KV cache are
/// distinct, separately-checkpointable spans (issues #60, #64).
///
/// The split is a cache boundary, not a content change: the concatenation is
/// exactly the old `combined()` block, in the same stable-then-volatile order
/// #60 already established, and it stays out of the system prompt (so
/// `tests/c_parity.rs`, which pins the system prompt and the tool wire formats,
/// is untouched). Splitting at a message boundary is what makes the Tier 2
/// snapshot a reproducible token prefix — a mid-message boundary could shift
/// under BPE merges across the seam. An empty half is skipped, so a project
/// with no `AGENTS.md` still injects exactly one message, as before.
///
/// Shared by `new_agent` and both front-ends' `/clear` handlers so the plain
/// REPL and the TUI cannot drift apart.
fn push_session_context(session: &mut Session, content: &ContextContent) {
    let stable = content.stable_context();
    if !stable.is_empty() {
        session.push(Message::user(stable));
    }
    let volatile = content.volatile_context();
    if !volatile.is_empty() {
        session.push(Message::user(volatile));
    }
}

/// Renders the session transcript as plain text for the engine.
fn render_transcript(session: &Session, system: &str) -> String {
    use std::fmt::Write as _;
    let mut out = format!("[system]\n{system}\n");
    // Append-only invariant (matching the C's token transcript): nothing is
    // ever injected between the system prompt and the messages, or the
    // engine's KV common-prefix probe would stop right after the system
    // prompt and re-prefill the whole conversation every turn. The task list
    // (issue #35) reaches the model through appended `task` tool observations
    // and a one-time re-injection after compaction instead.
    for m in &session.transcript {
        let tag = match m.role {
            crate::session::Role::User => "user",
            crate::session::Role::Assistant => "assistant",
        };
        let _ = write!(out, "[{tag}]\n{}\n", m.text);
    }
    out
}

/// Owned buffers backing a [`crate::engine::StructuredTurn`]; kept alive at the
/// call site so the borrowed `StructuredTurn` outlives the `generate` call.
struct StructuredBufs {
    system: String,
    messages: Vec<crate::engine::ChatMessage>,
    tools: Vec<crate::engine::ToolSpec>,
    rendered: String,
}

/// Removes DSML tool-call stanzas from assistant text so a provider engine
/// sees only natural language (the DSML is plank-internal framing, §4.4).
fn strip_dsml(text: &str) -> String {
    const OPEN: &str = "<｜DSML｜tool_calls>";
    const CLOSE: &str = "</｜DSML｜tool_calls>";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(OPEN) {
        out.push_str(&rest[..start]);
        if let Some(end) = rest[start..].find(CLOSE) {
            rest = &rest[start + end + CLOSE.len()..];
        } else {
            rest = "";
            break;
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

/// Reconstructs a JSON-object arguments string for one DSML tool call, so the
/// provider request carries the same arguments the model chose. String args
/// become JSON strings; anything flagged non-string is parsed as raw JSON
/// (falling back to a string when it does not parse).
fn dsml_args_to_json(call: &crate::dsml::ToolCall) -> String {
    let mut map = serde_json::Map::new();
    for arg in &call.args {
        let value = if arg.is_string {
            serde_json::Value::String(arg.value.clone())
        } else {
            serde_json::from_str::<serde_json::Value>(arg.value.trim())
                .unwrap_or_else(|_| serde_json::Value::String(arg.value.clone()))
        };
        map.insert(arg.name.clone(), value);
    }
    serde_json::Value::Object(map).to_string()
}

/// Splits a combined `dispatch_all` tool-result payload into `n` per-call
/// chunks, using the `Tool result K (name):` headers `dispatch_all` writes so
/// each chunk can be paired to the call it answers. Returns exactly `n`
/// chunks (padding with empty strings / folding any overflow into the last)
/// so every assistant `tool_use`/`tool_call` id gets one — and only one —
/// result message, keeping both providers' schemas well-formed.
fn split_tool_results(payload: &str, n: usize) -> Vec<String> {
    if n <= 1 {
        return vec![payload.to_string()];
    }
    // Header line starts (byte offsets) of each `Tool result K (`.
    let mut starts = Vec::new();
    let mut idx = 0;
    for line in payload.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if trimmed.starts_with("Tool result ")
            && trimmed
                .trim_start_matches("Tool result ")
                .starts_with(|c: char| c.is_ascii_digit())
        {
            starts.push(idx);
        }
        idx += line.len();
    }
    if starts.len() != n {
        // Framing did not line up with the id count: put everything on the
        // first result and leave the rest empty, still one-per-id.
        let mut chunks = vec![String::new(); n];
        chunks[0] = payload.to_string();
        return chunks;
    }
    let mut chunks = Vec::with_capacity(n);
    for (k, &start) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(payload.len());
        chunks.push(payload[start..end].to_string());
    }
    chunks
}

/// Maps a session transcript to provider chat messages: tool-result pseudo-user
/// turns become [`ChatRole::Tool`], other user turns stay user, and assistant
/// turns are stripped of DSML framing (empty ones dropped).
///
/// Provider-native tool-call ids are threaded across turns (§4.4): each
/// assistant DSML tool-call is assigned a deterministic id
/// (`call_{turn}_{i}`), carried on the assistant [`ChatMessage`], and echoed
/// onto the [`ChatRole::Tool`] message(s) that answer it — so multi-turn tool
/// conversations are well-formed for both the `OpenAI` and Anthropic schemas.
/// ds4/echo never see these (they read the flat transcript), so parity holds.
/// Runs one round's generations concurrently, at most `width` at a time, and
/// returns each slot's outcome in slot order.
///
/// Appends whatever each pass rendered to its slot's `output` buffer. Split out
/// of [`Agent::run_fanout_rounds`] to keep both under the function-length lint,
/// and because it is the one part that touches threads: nothing borrowed from the
/// `Agent` crosses the boundary, only `&mut` engines and plain data.
fn generate_fanout_round(
    slots: &mut [FanoutSlot],
    prepared: &[Option<(String, Option<StructuredBufs>)>],
    width: usize,
    ctx: &PassCtx<'_>,
    cwd: &std::path::Path,
) -> Vec<Option<Result<QuietPass, String>>> {
    let mut passes: Vec<Option<Result<QuietPass, String>>> = Vec::new();
    for (slot_chunk, prep_chunk) in slots.chunks_mut(width).zip(prepared.chunks(width)) {
        let mut chunk: Vec<Option<Result<QuietPass, String>>> =
            (0..slot_chunk.len()).map(|_| None).collect();
        let mut texts: Vec<(usize, String)> = Vec::new();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (i, (slot, prep)) in slot_chunk.iter_mut().zip(prep_chunk.iter()).enumerate() {
                let Some((prompt, bufs)) = prep else { continue };
                let sink = crate::viz::CollectSink::default();
                let collected = sink.clone();
                // `edit_preflight` is not `Clone`, so build one per slot.
                let preflight = edit_preflight_cwd(cwd);
                let engine = slot.engine.as_mut();
                handles.push((
                    i,
                    collected,
                    scope.spawn(move || {
                        generate_pass(
                            engine,
                            prompt,
                            bufs.as_ref(),
                            ctx,
                            Box::new(sink),
                            preflight,
                        )
                    }),
                ));
            }
            for (i, collected, handle) in handles {
                chunk[i] = Some(
                    handle
                        .join()
                        .unwrap_or_else(|_| Err("sub-agent panicked".to_string())),
                );
                // Buffered here and applied after the scope: the spawned threads
                // hold `&mut` on the slots until it ends.
                texts.push((i, collected.take()));
            }
        });
        for (i, text) in texts.drain(..) {
            slot_chunk[i].output.push_str(&text);
        }
        passes.extend(chunk);
    }
    passes
}

/// The last non-empty assistant message in `messages` — a sub-agent's final
/// report. `None` when it never produced one (interrupted before any output).
fn last_assistant_text(messages: &[Message]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|m| matches!(m.role, crate::session::Role::Assistant) && !m.text.trim().is_empty())
        .map(|m| m.text.trim().to_owned())
}

/// Concatenates tool outputs into the model-facing result block, given
/// `(tool name, output)` pairs in the model's call order.
///
/// Kept separate from dispatch so results can be *collected* out of order — a
/// parallel sub-agent fan-out finishes in completion order — and still render in
/// call order. Byte stability matters to [`crate::repro`], so the numbering must
/// follow the call, never the completion.
fn format_tool_results(results: &[(String, String)]) -> String {
    use std::fmt::Write as _;
    let mut all = String::new();
    for (i, (name, out)) in results.iter().enumerate() {
        let name = if name.is_empty() {
            "unknown"
        } else {
            name.as_str()
        };
        let _ = writeln!(all, "Tool result {} ({}):", i + 1, name);
        all.push_str(out);
        if !out.is_empty() && !out.ends_with('\n') {
            all.push('\n');
        }
    }
    all
}
fn session_to_messages(session: &Session) -> Vec<crate::engine::ChatMessage> {
    use crate::engine::{ChatMessage, ChatRole, ToolCallRef};
    let mut out = Vec::new();
    let mut turn = 0usize;
    // Ids from the most recent assistant tool-call turn awaiting their result.
    let mut pending_ids: Vec<String> = Vec::new();
    for m in &session.transcript {
        match m.role {
            crate::session::Role::User => {
                let t = m.text.trim();
                let is_tool = t.starts_with("<tool_result>")
                    || t.starts_with("Tool:")
                    || t.starts_with("Tool result");
                if is_tool {
                    let payload = t.strip_prefix("<tool_result>").map_or(t, |inner| {
                        inner.strip_suffix("</tool_result>").unwrap_or(inner)
                    });
                    let payload = payload.trim();
                    if pending_ids.is_empty() {
                        // A tool result with no prior tool-call turn (compaction
                        // summary, stop-hook feedback): no id to pair.
                        out.push(ChatMessage::new(ChatRole::Tool, payload));
                    } else {
                        let ids = std::mem::take(&mut pending_ids);
                        let chunks = split_tool_results(payload, ids.len());
                        for (id, chunk) in ids.into_iter().zip(chunks) {
                            let mut msg = ChatMessage::new(ChatRole::Tool, chunk);
                            msg.tool_call_id = Some(id);
                            out.push(msg);
                        }
                    }
                } else {
                    // A genuine user turn ends any pending pairing.
                    pending_ids.clear();
                    out.push(ChatMessage::new(ChatRole::User, m.text.clone()));
                }
            }
            crate::session::Role::Assistant => {
                turn += 1;
                pending_ids.clear();
                let clean = strip_dsml(&m.text);
                // Recover the DSML tool calls this turn issued, assigning
                // deterministic ids paired to the results that follow.
                let mut parser = crate::dsml::DsmlParser::new();
                parser.feed(m.text.as_bytes());
                let mut tool_calls = Vec::new();
                for (i, call) in parser.calls().iter().enumerate() {
                    if call.name.is_empty() {
                        continue;
                    }
                    let id = format!("call_{turn}_{i}");
                    tool_calls.push(ToolCallRef {
                        id: id.clone(),
                        name: call.name.clone(),
                        arguments: dsml_args_to_json(call),
                    });
                    pending_ids.push(id);
                }
                if !clean.is_empty() || !tool_calls.is_empty() {
                    let mut msg = ChatMessage::new(ChatRole::Assistant, clean);
                    msg.tool_calls = tool_calls;
                    out.push(msg);
                }
            }
        }
    }
    out
}

/// ANSI reset used by the slash-command reports.
const ANSI_RESET: &str = "\x1b[0m";

/// Image pasting is feature-gated off until the model's handling of
/// image-file references is understood (`--features images` re-enables it).
/// The code stays compiled either way; this constant kills every runtime
/// path: clipboard probing, paste capture, and attachment injection.
const IMAGES_ENABLED: bool = cfg!(feature = "images");

/// Renders the `/mcp` server report following Claude Code's layout: a header
/// with the server count, then one `name · status · N tools` line each.
pub(crate) fn render_mcp_report(servers: &[crate::tools::mcp::McpServer], color: bool) -> String {
    use std::fmt::Write as _;
    let (green, red, reset) = if color {
        ("\x1b[38;5;42m", "\x1b[38;5;204m", ANSI_RESET)
    } else {
        ("", "", "")
    };
    let mut out = String::from("Manage MCP servers\n");
    if servers.is_empty() {
        out.push_str("no servers configured (checked ./.mcp.json and ~/.plank/.mcp.json)\n");
        return out;
    }
    let plural = if servers.len() == 1 { "" } else { "s" };
    let _ = writeln!(out, "{} server{plural}\n", servers.len());
    for s in servers {
        if s.alive() {
            let plural = if s.tools.len() == 1 { "" } else { "s" };
            let _ = writeln!(
                out,
                "  {} · {green}✔ connected{reset} · {} tool{plural}",
                s.name,
                s.tools.len()
            );
        } else if s.is_offline() {
            let plural = if s.tools.len() == 1 { "" } else { "s" };
            let _ = writeln!(
                out,
                "  {} · {red}✘ not running{reset} · {} cached tool{plural} still advertised · calls will report it as down",
                s.name,
                s.tools.len()
            );
        } else {
            let _ = writeln!(out, "  {} · {red}✘ failed{reset}", s.name);
        }
    }
    out
}

/// Shared turn state for the interactive and headless front-ends.
// The bools are independent UI/turn latches, not a disguised state machine.
#[allow(clippy::struct_excessive_bools)]
struct Agent<'a> {
    engine: Box<dyn Engine>,
    cfg: &'a AgentConfig,
    session: Session,
    store: SessionStore,
    tool_ctx: ToolContext,
    system: String,
    reminder: SystemPromptReminder,
    trace: Trace,
    power_percent: i32,
    /// A session KV payload was restored at startup, so the live KV already
    /// covers the whole transcript — a superset of every warm tier. The tier
    /// walk must then be skipped: it restores a tier checkpoint whose transcript
    /// is empty, which would wipe the restored token transcript and rewind the
    /// KV to the session-context boundary, re-prefilling the conversation.
    payload_restored: bool,
    /// Byte length of `system`'s trusted control-text prefix, handed to the
    /// engine so it tokenizes that span as rendered chat, and folded into the
    /// Tier 1 KV key so a checkpoint written under a different split is never
    /// restored (the prompt *text* is identical either way, so the text alone
    /// cannot tell them apart).
    trusted_system_len: usize,
    /// Live reasoning level, seeded from `cfg.generation.think_mode` and
    /// changed by `/think`. Owned here rather than read from `cfg` on each
    /// turn because `cfg` is shared immutably for the agent's lifetime.
    think: crate::engine::ThinkMode,
    color: bool,
    show_footer: bool,
    /// True when the line editor renders its own resting footer, so the turn
    /// loop must not print a second one after generation.
    editor_owns_footer: bool,
    /// KV position reported by the engine after the last generation; 0 when
    /// no generation has run against the current transcript. Anchors the
    /// `/context` report to the real context usage.
    last_ctx_used: i32,
    /// Whether the most recent turn ended by user interrupt, so the turn-end
    /// notification says "interrupted" instead of "finished".
    last_turn_interrupted: bool,
    /// A framed `/btw` prompt waiting to be answered *alongside* the next main
    /// pass rather than in place of it (`docs/SESSION-CLONE-DESIGN.md` §6.2).
    ///
    /// Set when a `/btw` preempts a pass and the engine can multiplex; the next
    /// `run_pass` takes it and runs both streams interleaved. `None` on every
    /// other path, which is what keeps the ordinary single-stream turn — and
    /// every engine that cannot fork — completely unchanged.
    pending_aside: Option<String>,
    /// Context content collected at session start (git, AGENTS.md, date).
    context_content: ContextContent,
    /// Skills loaded from ~/.plank/skills overlaid by ./.plank/skills.
    skills: Vec<crate::skills::Skill>,
    /// Prompt templates loaded from ~/.plank/templates overlaid by
    /// ./.plank/templates (issue #67).
    templates: Vec<crate::templates::Template>,
    /// Named agent definitions loaded from ~/.plank/agents overlaid by
    /// ./.plank/agents; dispatched via `/subagent <name> <task>`.
    agents: Vec<crate::agents::AgentDef>,
    /// Per-process counter naming each isolated sub-agent's throwaway worktree.
    isolation_seq: u32,
    /// Named in-session rollback points (`/checkpoint`, `/rollback`); dropped
    /// when the session is replaced.
    checkpoints: crate::checkpoint::CheckpointStore,
    /// Live remote-control bridge (issue #25): the shared [`BroadcastBus`] that
    /// this agent's turn output mirrors into, plus the shared [`TurnShared`] that
    /// remote `prompt`/`btw`/`interrupt` frames drive. `None` until `/rc` (or
    /// `/remote-control`) starts a server, in which case the turn loops behave
    /// exactly as before.
    remote: Option<Arc<RemoteState>>,
    /// The remote-control listener backing [`Agent::remote`], owned here so
    /// `/remote-control` can start and stop it mid-session. `Drop` on the
    /// `RemoteServer` joins the accept thread, so dropping the agent tears the
    /// listener down. `None` whenever `remote` is `None`; the two are installed
    /// and cleared together.
    remote_server: Option<crate::remote::RemoteServer>,
    /// TUI remote-control state (`--ui-remote`). `None` (the default) means
    /// no listener thread, no injected keys and no draw-time recording.
    ui_remote: Option<Arc<Mutex<UiRemote>>>,
    /// Cumulative billed token usage for online (provider) turns this session,
    /// surfaced by `/usage`. Stays zero for local engines, which report none.
    usage: SessionUsage,
    /// Engine-agnostic in/out token tally for the end-of-session stats.
    stats: SessionStats,
    /// When the current session began (process start, or the last `/clear`,
    /// `/resume`, or `/switch`), for the end-of-session duration.
    session_start: std::time::Instant,
    /// Destination for headless sub-agent output; see [`SubSinkTarget`].
    sub_sink: SubSinkTarget,
    /// Parent-KV snapshots captured by
    /// [`begin_subagent_fork`](Self::begin_subagent_fork), one per open fork
    /// (a stack: nested forks restore LIFO). The C engine's sync can only
    /// *extend* a checkpoint, so without the restore at fork end the next
    /// turn — parent prefix plus the small report — diverges behind the
    /// sidechain's live end and the whole parent context re-prefills from
    /// token zero. `None` entries are engines without KV support (Echo),
    /// where the restore no-ops.
    fork_kv: Vec<Option<crate::kvcache::KVCache>>,
    /// Engines for definitions that override the parent's (cross-provider
    /// sub-agents). Cached across dispatches so `discover_ctx_size`'s network
    /// probe happens at most once per key per session.
    ///
    /// An engine is *removed* while its sidechain runs and reinserted
    /// afterwards, which is what lets the borrow checker enforce that a swap
    /// cannot leak: the value cannot be in the map and in `self.engine` at once.
    alt_engines: std::collections::HashMap<EngineKey, Box<dyn Engine>>,
    /// Diagnostic from the last [`warm_alt_local`](Agent::warm_alt_local), for
    /// the front end to render. Set on every attempt, hit or miss.
    warm_note: Option<String>,
    /// Whether the alt *local* engine has had its system prompt warmed
    /// ([`Agent::warm_alt_local`]). Once is enough: the engine keeps its session
    /// across sidechains, so every later dispatch already finds the prefix in
    /// its KV. Set before the walk runs, so a failed warm is not retried on
    /// every dispatch.
    local_alt_warmed: bool,
}

/// Identity of an alternate sub-agent engine: provider, resolved base URL,
/// model, context window, and API-key variable.
///
/// The key variable belongs in the identity, not beside it: a cached engine
/// holds the key *value* it was built with, so two definitions agreeing on
/// everything else but reading different variables are different engines.
/// Omitting it would let the second silently reuse the first one's
/// credentials — the wrong account, with no error to notice.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum EngineKey {
    /// The local ds4 engine held for `provider: local` definitions when the
    /// main agent is a provider. There is only ever one.
    Local,
    /// A provider-backed engine: provider, resolved base URL, model, context
    /// window, and API-key variable.
    Provider(
        crate::remote::provider::ProviderKind,
        String,
        String,
        i32,
        String,
    ),
}

/// Formats a non-negative token count with thousands separators (`12345` →
/// `12,345`) for the `/usage` report.
fn fmt_int(n: i32) -> String {
    let s = n.max(0).to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// Renders the joke "invoice" `/usage` prints when inference is running on a
/// local engine, where there is no provider bill to report. Real token counts
/// are accurate; every monetary/physical unit below is deliberately absurd,
/// and the closing line states the real cost plainly so the gag can never be
/// mistaken for a charge.
///
/// Pure: usage numbers in, rendered block out — no engine, no terminal.
fn render_local_invoice(model: &str, input_tokens: u64, output_tokens: u64, color: bool) -> String {
    use std::fmt::Write as _;
    let dim = |s: &str| {
        if color {
            format!("\x1b[38;5;238m{s}{ANSI_RESET}")
        } else {
            s.to_owned()
        }
    };
    let total = input_tokens.saturating_add(output_tokens);
    // Entirely made-up conversion factors. Do not cite these anywhere.
    // Integer math only: milli-watt-hours and thousandths of an espresso.
    let milli_wh = total / 5;
    let milli_espresso = total / 100;
    let fan_revs = total.saturating_mul(11);
    let knee_minutes = total / 900;
    let gpu_tears = output_tokens / 1_000;

    let model = if model.is_empty() {
        "local model"
    } else {
        model
    };
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{}",
        dim(&format!("Local Inference Invoice — {model}"))
    );
    let _ = writeln!(out, "  tokens in     {}", fmt_u64(input_tokens));
    let _ = writeln!(out, "  tokens out    {}", fmt_u64(output_tokens));
    let _ = writeln!(
        out,
        "  electricity   {}.{:03} Wh ({}.{:03} espressos)",
        milli_wh / 1_000,
        milli_wh % 1_000,
        milli_espresso / 1_000,
        milli_espresso % 1_000
    );
    let _ = writeln!(
        out,
        "  fan service   {} revolutions of quiet dignity",
        fmt_u64(fan_revs)
    );
    let _ = writeln!(out, "  lap heat      {knee_minutes} toasty-knee-minutes");
    let _ = writeln!(out, "  GPU tears     {gpu_tears} (shed silently, in Metal)");
    let _ = writeln!(out, "  amount due    0 (zero) dollars");
    let _ = writeln!(
        out,
        "{}",
        dim(
            "This invoice is a joke. Real cost is $0.00 — inference ran locally on your own hardware, so nobody is billing you."
        )
    );
    out
}

/// Formats a `u64` token count with thousands separators, for the run stats.
fn fmt_u64(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// Formats a duration as `H:MM:SS`, dropping the hours field when zero
/// (`4:07`, `1:02:09`), for the end-of-session stats.
fn fmt_duration(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// Running tally of provider token usage across a session's turns.
#[derive(Debug, Clone, Copy, Default)]
struct SessionUsage {
    /// Provider turns counted (passes that reported a `usage` block).
    turns: u32,
    /// Summed token usage across those turns.
    total: crate::engine::TokenUsage,
}

/// Engine-agnostic token tally for the end-of-session stats, in both
/// directions. Unlike [`SessionUsage`] (provider billing only), this counts
/// local turns too: output is the generated tokens, input the prompt tokens
/// ingested (from the provider `usage` block when present, else the
/// context-size delta of the pass).
#[derive(Debug, Clone, Default)]
struct SessionStats {
    /// Tokens the model ingested (prompt / prefill), summed over all passes.
    input_tokens: u64,
    /// Tokens the model generated, summed over all passes.
    output_tokens: u64,
    /// The same tally split by the engine that served the pass, in the order
    /// each engine first contributed.
    ///
    /// A `Vec` rather than a map so the main engine leads the report: it is the
    /// one that served the ordinary turns, and the sub-agent engines below it
    /// read as the exceptions they are. There are never more than a handful.
    by_engine: Vec<(String, u64, u64)>,
}

impl SessionStats {
    /// Adds one pass's tally, to the total and to `engine`'s row.
    fn add(&mut self, engine: &str, input: u64, output: u64) {
        self.input_tokens += input;
        self.output_tokens += output;
        if let Some(row) = self.by_engine.iter_mut().find(|r| r.0 == engine) {
            row.1 += input;
            row.2 += output;
        } else {
            self.by_engine.push((engine.to_owned(), input, output));
        }
    }
}

/// How an engine is named in the end-of-session breakdown: its model, with
/// local ones marked as such.
///
/// The mark is the point of the breakdown — under a provider main agent with a
/// `provider: local` sub-agent, "which of these tokens cost money" is exactly
/// the question the split answers.
fn engine_stats_label(engine: &dyn Engine) -> String {
    let model = engine.model_name();
    if engine.is_local() {
        format!("{model} (local)")
    } else {
        model
    }
}

/// Default number of user turns replayed by `/history`.
const HISTORY_DEFAULT_TURNS: usize = 3;
/// Sessions shown by the /resume picker.
const RESUME_LIST_LIMIT: usize = 10;

/// Outcome of `/insights`: a written report, or the user's decision to stop.
///
/// Cancelling is not a failure, and saying "insights failed" to someone who
/// pressed Esc on purpose would be a small lie.
#[derive(Debug)]
enum Insights {
    /// The report was written.
    Done {
        /// Where the HTML landed.
        path: std::path::PathBuf,
        /// Condensed summary for the terminal.
        summary: Vec<String>,
    },
    /// Stopped before there was anything to write.
    Cancelled,
}

/// Puts the window title back to idle when it drops, so a long-running
/// command cannot leave the window describing work that has finished — on the
/// early-return and error paths as much as the successful one.
struct TitleRestore;

impl Drop for TitleRestore {
    fn drop(&mut self) {
        crate::title::set(crate::title::State::Idle);
    }
}

/// Token budget for one `/insights` narrative section, covering the model's
/// reasoning and the JSON answer that follows it.
///
/// Generous enough to think and then answer, bounded so one wandering reply
/// costs that section rather than the user's afternoon.
const INSIGHTS_SECTION_TOKENS: i32 = 3000;
/// Maximum user turns `/history` accepts.
const HISTORY_MAX_TURNS: usize = 200;
/// Name of the auto-checkpoint saved before a `/rollback`, so a rollback is
/// itself undoable via `/rollback pre-rollback`.
const PRE_ROLLBACK_CHECKPOINT: &str = "pre-rollback";

/// A sub-agent's throwaway worktree, and what it displaced.
///
/// Held only for the duration of one `agent` call: created before the fork,
/// unwound after it, never persisted.
struct AgentIsolation {
    /// The live worktree session, for the removal that may follow.
    session: crate::worktree::WorktreeSession,
    /// The worktree's directory, kept separately so the message written after
    /// `session` is consumed can still name it.
    path: std::path::PathBuf,
    /// The parent's working directory, restored when the sub-agent finishes.
    outer_cwd: std::path::PathBuf,
}

impl Agent<'_> {
    /// Builds owned structured-turn buffers for a provider engine (§4.4). The
    /// provider gets a machine-readable tool registry and its own system prompt
    /// (never the DS4 byte-parity prompt), plus the flat render as a fallback.
    fn build_structured(&self, rendered: &str) -> StructuredBufs {
        self.build_structured_for(&self.session, rendered)
    }

    /// [`build_structured`](Self::build_structured) for an arbitrary session.
    ///
    /// A parallel sub-agent slot owns its own transcript, so it cannot use
    /// `self.session`. This must be used rather than passing no structured
    /// buffers at all: a provider engine given a flat prompt gets an empty tool
    /// list, which would leave a remote sub-agent unable to call anything.
    fn build_structured_for(&self, session: &Session, rendered: &str) -> StructuredBufs {
        StructuredBufs {
            system: sysprompt::provider_system_prompt(&self.cfg.system),
            messages: session_to_messages(session),
            tools: sysprompt::provider_tool_registry(&self.tool_ctx.mcp),
            rendered: rendered.to_string(),
        }
    }

    /// Wraps a debug/status message in the thinking gray on color terminals.
    fn debug_line(&self, text: &str) -> String {
        if self.color {
            format!("\x1b[38;5;238m{text}{ANSI_RESET}")
        } else {
            text.to_owned()
        }
    }

    /// Streams one generation pass: paints the live status bar for prefill and
    /// generation, and routes model text through the viz + markdown pipeline.
    #[allow(clippy::type_complexity)]
    fn stream_generation(
        &mut self,
        prompt_text: &str,
        turn_start: Instant,
    ) -> Result<
        (
            StreamRenderer<TerminalSink<FlushingStdout>>,
            String,
            crate::engine::GenerationStats,
        ),
        String,
    > {
        let sink = TerminalSink {
            renderer: TokenRenderer::new(
                FlushingStdout,
                RenderOptions {
                    use_color: self.color,
                    format_thinking: true,
                    format_markdown: true,
                },
            ),
        };
        // See the matching guard in `worker_generate_kind`: the plain REPL has
        // no blinking brain to drive, but the flag is process-global and a
        // remote client attached to this session renders off it.
        let _local = self.engine.is_local().then(crate::status::LocalPass::begin);
        let mut stream = StreamRenderer::new(sink);
        stream.set_show_tool_calls(crate::settings::active().ui.show_tool_calls);
        stream.set_show_thinking(crate::settings::active().ui.show_thinking);
        stream.set_thinking_tool_calls(crate::settings::active().engine.thinking_tool_calls);
        stream.set_tool_names(sysprompt::tool_names(&self.tool_ctx.mcp));
        stream.set_preflight(edit_preflight(&self.tool_ctx));
        // With thinking enabled, the *local* chat template opens `<think>` in
        // the prefill prefix, so generation streams thinking content first
        // without a leading tag; start the renderer inside the think block so it
        // renders gray until `</think>`. Provider engines are excluded: their
        // translator emits explicit `<think>`/`</think>` tags, so pre-opening
        // here would mis-color any output not preceded by a reasoning delta.
        if !matches!(self.think, crate::engine::ThinkMode::Off) && !self.engine.wants_structured() {
            stream.begin_in_think();
        }
        let mut assistant_text = String::new();
        let ctx_size = self.engine.ctx_size();
        let power = self.power_percent;
        let think = self.think;
        // Bound here rather than inside the event closure, which cannot borrow
        // `self` while `self.engine` is generating.
        let model_name = self.engine.model_name();
        let prompt_tokens = self.engine.count_tokens(prompt_text);
        let mut bar = crate::statusbar::StatusBar::new(self.show_footer && self.color, self.color);
        let verb = status::random_verb_index();
        // Set when a mid-stream preflight fails: stops the engine early, but
        // is not a user interrupt — the caller feeds the error to the model.
        let preflight_stop = AtomicBool::new(false);
        // Mirrors the C's worker greedy flag: argmax sampling while the
        // stream renderer is inside a DSML tool-call stanza.
        let greedy = AtomicBool::new(false);
        // Provider engines take a structured turn; local engines keep the flat
        // rendered transcript (byte parity, §4.4). `bufs`/`st` outlive the call.
        let bufs = self
            .engine
            .wants_structured()
            .then(|| self.build_structured(prompt_text));
        let st;
        let prompt = match &bufs {
            Some(b) => {
                st = crate::engine::StructuredTurn {
                    system: &b.system,
                    messages: &b.messages,
                    tools: &b.tools,
                    rendered: &b.rendered,
                };
                crate::engine::Prompt::Structured(&st)
            }
            None => crate::engine::Prompt::Flat(prompt_text),
        };
        let stats = self
            .engine
            .generate(
                prompt,
                &self.cfg.generation,
                &|| preflight_stop.load(Ordering::Relaxed) || crate::interrupt::pending(),
                &|| greedy.load(Ordering::Relaxed),
                &mut |ev| match ev {
                    EngineEvent::Text(t) => {
                        // Model output has started: drop the prefill bar so the
                        // text streams cleanly from column zero.
                        bar.clear();
                        assistant_text.push_str(&t);
                        stream.push(&t);
                        greedy.store(stream.wants_greedy_sampling(), Ordering::Relaxed);
                        if stream.preflight_error().is_some() {
                            preflight_stop.store(true, Ordering::Relaxed);
                        }
                    }
                    EngineEvent::Prefill(p) => {
                        // Only a finished pass has a meaningful rate: mid-pass
                        // events divide by an elapsed time that has barely
                        // started, so their tok/s runs high.
                        if p.is_complete() {
                            crate::speeds::note_prefill(&model_name, p.tps, p.done);
                        }
                        bar.show(&Status {
                            // A finished prefill means the engine is sampling,
                            // not prefilling. Saying "prefilling" through the
                            // whole time-to-first-token reads as a hang, and a
                            // fully cached turn has no further event coming to
                            // correct it (#64 follow-up).
                            state: if p.is_complete() {
                                WorkerState::Generating
                            } else {
                                WorkerState::Prefill
                            },
                            prefill_done: p.done,
                            prefill_total: p.total,
                            prefill_label: verb,
                            prefill_tps: p.tps,
                            elapsed_secs: turn_start.elapsed().as_secs_f64(),
                            ctx_used: prompt_tokens,
                            ctx_size,
                            power_percent: power,
                            think,
                            ..Status::default()
                        });
                    }
                    // Notices are a warm-up-only signal; never emitted mid-turn.
                    EngineEvent::Notice(_) => {}
                },
            )
            .map_err(|e| e.to_string())?;
        stream.finish();
        bar.clear();
        self.record_usage(&stats);
        self.last_ctx_used = stats.ctx_used;
        Ok((stream, assistant_text, stats))
    }

    /// Executes one DSML block's tool calls, routing any `agent` call through
    /// the sub-agent driver (issue #50) and everything else through the normal
    /// [`dispatch_all`]. Frames results identically to [`dispatch_all`] so the
    /// model sees the same `Tool result K (name):` headers regardless of path.
    ///
    /// The common case (no `agent` call) delegates straight to `dispatch_all`
    /// for zero behavioral change; the special path only engages when the model
    /// actually delegates.
    fn run_tool_calls(&mut self, calls: &[ToolCall]) -> String {
        // Holds the tool label in the status bar for the whole dispatch, then
        // clears it on drop whichever way we return.
        let _running = (!calls.is_empty()).then(|| {
            let names = calls
                .iter()
                .map(|c| {
                    if c.name.is_empty() {
                        "unknown"
                    } else {
                        c.name.as_str()
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            crate::status::ToolActivity::begin(format!("🔧 {names}"))
        });
        if !calls.iter().any(|c| c.name == "agent") {
            return dispatch_all(calls, &mut self.tool_ctx);
        }
        if calls.is_empty() {
            return "Tool error: empty tool call block\n".to_string();
        }
        // A block of nothing but remote-backed agent calls runs concurrently.
        // Anything else falls through to the serial loop below.
        if let Some(results) = self.run_agent_fanout(calls) {
            return format_tool_results(&results);
        }
        // Mirror dispatch_all: clear any undrained previews so cards never leak.
        self.tool_ctx.edit_previews.clear();
        let mut results: Vec<(String, String)> = Vec::with_capacity(calls.len());
        for call in calls {
            let out = if call.name == "agent" {
                self.run_agent_tool(call)
            } else {
                dispatch(call, &mut self.tool_ctx).output
            };
            results.push((call.name.clone(), out));
        }
        format_tool_results(&results)
    }

    /// Sends an event on the worker→UI channel when the front end is listening.
    /// The channel is the same one the parent turn renders through, so an
    /// ordinary variant (e.g. [`UiEvent::Dim`](crate::worker::UiEvent::Dim))
    /// sent here lands in the main log *and* on the remote bus, while the
    /// sub-agent variants stay local to the pane.
    fn emit_sub(&self, ev: crate::worker::UiEvent) {
        if let SubSinkTarget::Events(tx) = &self.sub_sink {
            let _ = tx.send(ev);
        }
    }

    /// Runs the model-invocable `agent` tool: delegates `task` to a fresh scoped
    /// sub-agent (a sidechain fork of the live transcript) and returns only its
    /// final report as the tool observation (issue #50). The sidechain shares
    /// the parent transcript prefix, so the engine reuses the parent KV cache on
    /// the way in and rolls the fork back out afterward.
    fn run_agent_tool(&mut self, call: &ToolCall) -> String {
        let task = call
            .arg_value("task")
            .or_else(|| call.arg_value("prompt"))
            .unwrap_or("")
            .trim()
            .to_owned();
        if task.is_empty() {
            return "Tool error: agent requires a non-empty 'task' to delegate\n".to_string();
        }
        if self.tool_ctx.subagent_depth >= crate::tools::SUBAGENT_DEPTH_CAP {
            return "Tool error: sub-agent nesting limit reached; complete this work directly\n"
                .to_string();
        }
        let requested = call.arg_value("name").unwrap_or("").trim();
        // A name the model could not have known about must not burn a round:
        // fall back to the general-purpose sub-agent and say so in the report.
        // `auto: false` definitions are `/subagent`-only, so they are treated
        // as absent here exactly like a typo — the model was never offered
        // them, so from its side there is no difference.
        let matched = self
            .agents
            .iter()
            .find(|d| d.name == requested && d.auto)
            .cloned();
        let fallback_note = if requested.is_empty() || matched.is_some() {
            None
        } else {
            Some(format!(
                "note: no agent named '{requested}' is available; ran a general-purpose sub-agent instead.\n"
            ))
        };
        let name = matched.as_ref().map_or("", |d| d.name.as_str());
        let instructions = matched.as_ref().map(|d| d.body.clone());
        // Resolve the alternate engine *before* forking, so a missing key or an
        // unbuildable engine leaves the transcript exactly as it was.
        let alt = match self.resolve_subagent_alt(matched.as_ref().and_then(|d| d.engine.clone())) {
            Ok(alt) => alt,
            Err(e) => return format!("Tool error: agent '{name}' engine unavailable: {e}\n"),
        };
        // `isolation: worktree` gives this sub-agent its own checkout, so a
        // fan-out of agents editing the same files cannot overwrite each other.
        // Set up before the fork: a failure here must leave the transcript
        // untouched, exactly like the engine resolution above.
        let isolation = if matched.as_ref().is_some_and(|d| d.isolate) {
            match self.begin_agent_isolation() {
                Ok(iso) => iso,
                Err(e) => return format!("Tool error: agent '{name}' worktree unavailable: {e}\n"),
            }
        } else {
            None
        };
        let instructions = match &isolation {
            Some(iso) => Some(crate::agents::worktree_notice(
                instructions.as_deref(),
                &iso.path,
            )),
            None => instructions,
        };
        if let Some(note) = self.take_warm_note() {
            self.emit_sub(crate::worker::UiEvent::Dim(note));
        }
        let fork_at = self.begin_subagent_fork(instructions.as_deref(), &task, alt.is_none());
        let label = if name.is_empty() {
            "sub-agent".to_string()
        } else {
            name.to_string()
        };
        // The signpost goes out as an ordinary dim line so it lands in the main
        // transcript by the normal route — and therefore reaches remote clients
        // too, which never see the pane-only `Sub*` variants.
        self.emit_sub(crate::worker::UiEvent::Dim(crate::tui::subagent_signpost(
            &label,
        )));
        self.emit_sub(crate::worker::UiEvent::SubStart(label));
        self.tool_ctx.subagent_depth += 1;
        let result = match alt {
            None => self.run_subagent_loop(),
            Some((key, engine)) => self.run_sidechain_on(key, engine, Self::run_subagent_loop),
        };
        self.tool_ctx.subagent_depth -= 1;
        let isolation_note = self.end_agent_isolation(isolation);
        self.emit_sub(crate::worker::UiEvent::SubEnd);
        // Extract the sidechain's final report before truncating it back out.
        let report = last_assistant_text(&self.session.transcript[fork_at..]);
        self.session.transcript.truncate(fork_at);
        self.restore_fork_kv();
        match result {
            Err(e) => format!("Tool error: sub-agent failed: {e}\n"),
            Ok(()) => match report {
                // The note leads, so the model reads why its chosen persona did
                // not apply before it reads the report it produced anyway.
                Some(r) => format!(
                    "{}Sub-agent report:\n{r}\n{}",
                    fallback_note.unwrap_or_default(),
                    isolation_note.unwrap_or_default()
                ),
                None => "Tool error: sub-agent produced no report\n".to_string(),
            },
        }
    }

    /// Creates the throwaway worktree for an `isolation: worktree` sub-agent
    /// and points the tool context at it, returning what is needed to undo
    /// both. `Ok(None)` means the parent is not in a git repository at all,
    /// which is not an error — there is simply nothing to isolate from.
    ///
    /// # Errors
    /// Returns a message when the worktree could not be created.
    fn begin_agent_isolation(&mut self) -> Result<Option<AgentIsolation>, String> {
        if crate::worktree::canonical_git_root(&self.tool_ctx.cwd).is_none() {
            return Ok(None);
        }
        // Unique per run within this process, which is all the slug has to be:
        // a leaked worktree from a *previous* process is the stale sweep's job,
        // and it recognizes any name of this shape.
        let id = self.isolation_seq;
        self.isolation_seq += 1;
        let slug = crate::worktree::agent_slug(u64::from(std::process::id()) << 8 | u64::from(id));
        let session = crate::worktree::create_agent_worktree(&self.tool_ctx.cwd, &slug)?;
        let outer_cwd = std::mem::replace(&mut self.tool_ctx.cwd, session.path.clone());
        Ok(Some(AgentIsolation {
            path: session.path.clone(),
            session,
            outer_cwd,
        }))
    }

    /// Restores the parent's working directory after an isolated sub-agent and
    /// disposes of its worktree.
    ///
    /// A clean worktree is removed; one holding edits or commits is **kept**,
    /// and its path is reported back to the model. Deleting it would silently
    /// throw away the very work the sub-agent was asked to do, so the choice is
    /// always to leave it for the parent to inspect and merge.
    fn end_agent_isolation(&mut self, isolation: Option<AgentIsolation>) -> Option<String> {
        let iso = isolation?;
        self.tool_ctx.cwd = iso.outer_cwd;
        if !crate::worktree::has_changes(&iso.path, iso.session.original_head.as_deref()) {
            let _ = crate::worktree::cleanup(&iso.session, &self.tool_ctx.hooks);
            return None;
        }
        Some(format!(
            "The sub-agent worked in an isolated worktree and left changes there: {}. They are \
             not in your working copy — review and merge them from that directory if you want \
             them.\n",
            iso.path.display()
        ))
    }

    /// Headless generate→dispatch loop for a sub-agent sidechain (issue #50):
    /// like the main turn loop but with no on-screen streaming, footer, hooks,
    /// or compaction. Bounded by a round budget so a stuck sub-agent cannot loop
    /// forever. Nested `agent` calls route through [`run_tool_calls`], so the
    /// [`SUBAGENT_DEPTH_CAP`](crate::tools::SUBAGENT_DEPTH_CAP) guard applies.
    fn run_subagent_loop(&mut self) -> Result<(), String> {
        // The parent turn's status sink writes bare `SystemStatus` events into
        // the MAIN log; leaving it installed would scatter the sub-agent's
        // "Searching Google for …" notices across the parent transcript while
        // its model text goes to the pane. Same treatment as the neighbouring
        // `edit_previews` / `task_completions` / `hook_warnings`: keep them off
        // the parent's screen — routed into the pane when there is one, dropped
        // otherwise — and restore the parent's sink on every exit path.
        let parent_sink = self.tool_ctx.status_sink.take();
        self.tool_ctx.status_sink = match &self.sub_sink {
            SubSinkTarget::Events(tx) => {
                let tx = tx.clone();
                Some(Box::new(move |msg: &str| {
                    let _ = tx.send(crate::worker::UiEvent::Sub(Box::new(
                        crate::worker::UiEvent::SystemStatus(msg.to_owned()),
                    )));
                }))
            }
            SubSinkTarget::Null | SubSinkTarget::Stdout => None,
        };
        let result = self.run_subagent_rounds();
        self.tool_ctx.status_sink = parent_sink;
        result
    }

    /// The bounded generate→dispatch rounds of a sub-agent sidechain; see
    /// [`run_subagent_loop`](Self::run_subagent_loop), which owns the
    /// status-sink swap around it.
    fn run_subagent_rounds(&mut self) -> Result<(), String> {
        const MAX_ROUNDS: usize = 40;
        let turn_start = Instant::now();
        for round in 0..MAX_ROUNDS {
            // On the last permitted round, ask for the report instead of letting
            // the budget simply run out: a sub-agent that calls a tool on every
            // pass would otherwise hand the parent an error and throw away
            // everything it found.
            let last_round = round + 1 == MAX_ROUNDS;
            if last_round {
                self.session
                    .push(Message::user(crate::agents::final_round_reminder()));
            }
            let prompt_text = render_transcript(&self.session, &self.system);
            let (calls, assistant_text, err) = self.generate_quiet(&prompt_text, turn_start)?;
            self.session.push(Message::assistant(assistant_text));
            if last_round {
                // Whatever it asked for, this text is the report.
                return Ok(());
            }
            if let Some(payload) = err {
                self.session.push(Message::user(format!(
                    "<tool_result>{payload}</tool_result>"
                )));
                continue;
            }
            if calls.is_empty() {
                return Ok(());
            }
            let observations = self.run_tool_calls(&calls);
            self.sync_tasks_after_dispatch();
            // The sidechain has no UI to drain these into; discard so they never
            // leak onto the parent turn's screen.
            self.tool_ctx.edit_previews.clear();
            self.tool_ctx.task_completions.clear();
            self.tool_ctx.hook_warnings.clear();
            self.session.push(Message::user(format!(
                "<tool_result>{observations}</tool_result>"
            )));
        }
        // Unreachable: the final iteration always returns above. `MAX_ROUNDS` is
        // a non-zero constant, so the loop cannot fall through without it.
        Ok(())
    }

    /// One quiet generation pass for the sub-agent loop: drives the engine with
    /// a discarding sink (no stdout / TUI output) and returns the parsed tool
    /// calls, the assistant text, and an optional tool-error payload to feed
    /// back (preflight or engine-reported parse error). Mirrors the call/greedy
    /// detection of [`stream_generation`] via the shared [`StreamRenderer`].
    fn generate_quiet(
        &mut self,
        prompt_text: &str,
        _turn_start: Instant,
    ) -> Result<(Vec<ToolCall>, String, Option<String>), String> {
        let sink = self.sub_sink_render_sink();
        let bufs = self
            .engine
            .wants_structured()
            .then(|| self.build_structured(prompt_text));
        let ctx = PassCtx {
            opts: &self.cfg.generation,
            think_off: matches!(self.think, crate::engine::ThinkMode::Off),
            // Read here, not inside the pass: `settings::install_for_test` is
            // thread-local, so a spawned pass would silently see defaults.
            thinking_tool_calls: crate::settings::active().engine.thinking_tool_calls,
            tool_names: sysprompt::tool_names(&self.tool_ctx.mcp),
        };
        let preflight = edit_preflight(&self.tool_ctx);
        let pass = generate_pass(
            self.engine.as_mut(),
            prompt_text,
            bufs.as_ref(),
            &ctx,
            sink,
            preflight,
        )?;
        self.record_usage(&pass.stats);
        self.last_ctx_used = pass.stats.ctx_used;
        Ok((pass.calls, pass.assistant_text, pass.tool_error))
    }

    /// Builds the render sink a sub-agent pass writes through, per the current
    /// [`SubSinkTarget`]. Shared by the serial loop and the parallel fan-out.
    fn sub_sink_render_sink(&self) -> Box<dyn crate::viz::RenderSink + Send> {
        match &self.sub_sink {
            SubSinkTarget::Null => Box::new(NullSink),
            SubSinkTarget::Events(tx) => Box::new(crate::worker::SubAgentSink(tx.clone())),
            SubSinkTarget::Stdout => Box::new(TerminalSink {
                renderer: TokenRenderer::new(
                    FlushingStdout,
                    RenderOptions {
                        use_color: self.color,
                        format_thinking: true,
                        format_markdown: true,
                    },
                ),
            }),
        }
    }
}

/// Outcome of one quiet generation pass.
struct QuietPass {
    calls: Vec<ToolCall>,
    assistant_text: String,
    /// A preflight or parse error to feed back as a tool result.
    tool_error: Option<String>,
    /// Returned rather than recorded, because usage accounting lives on the
    /// `Agent` and a pass may run on a thread that cannot touch it.
    stats: crate::engine::GenerationStats,
}

/// The `Agent`-derived inputs a quiet pass needs, gathered on the main thread so
/// the pass itself borrows nothing from `self` and can run on a spawned thread.
struct PassCtx<'a> {
    opts: &'a crate::engine::GenerationOptions,
    think_off: bool,
    thinking_tool_calls: bool,
    tool_names: Vec<String>,
}

/// Runs one quiet generation against `engine`, with no stdout/TUI output beyond
/// `sink`, and returns the parsed tool calls, assistant text, an optional
/// tool-error payload, and the pass stats.
///
/// A free function over `&mut dyn Engine` rather than a method: the parallel
/// fan-out needs to drive several engines at once from separate threads, which a
/// `&mut self` method cannot express. Mirrors the call/greedy detection of
/// [`Agent::stream_generation`] via the shared [`StreamRenderer`].
fn generate_pass(
    engine: &mut dyn Engine,
    prompt_text: &str,
    bufs: Option<&StructuredBufs>,
    ctx: &PassCtx<'_>,
    sink: Box<dyn crate::viz::RenderSink + Send>,
    preflight: impl FnMut(&ToolCall) -> Result<(), String> + 'static,
) -> Result<QuietPass, String> {
    // Held for the whole pass, so the status bar's brain blinks while *this*
    // engine works — and stops when the pass ends, however it ends. A sidechain
    // on a `provider: local` definition swaps the engine before getting here, so
    // this reports the engine actually generating rather than the session's.
    let _local = engine.is_local().then(crate::status::LocalPass::begin);
    let mut stream = StreamRenderer::new(sink);
    stream.set_preflight(preflight);
    stream.set_thinking_tool_calls(ctx.thinking_tool_calls);
    stream.set_tool_names(ctx.tool_names.clone());
    if !ctx.think_off && !engine.wants_structured() {
        stream.begin_in_think();
    }
    let mut assistant_text = String::new();
    let preflight_stop = AtomicBool::new(false);
    let greedy = AtomicBool::new(false);
    let st;
    let prompt = match bufs {
        Some(b) => {
            st = crate::engine::StructuredTurn {
                system: &b.system,
                messages: &b.messages,
                tools: &b.tools,
                rendered: &b.rendered,
            };
            crate::engine::Prompt::Structured(&st)
        }
        None => crate::engine::Prompt::Flat(prompt_text),
    };
    let stats = engine
        .generate(
            prompt,
            ctx.opts,
            &|| preflight_stop.load(Ordering::Relaxed) || crate::interrupt::pending(),
            &|| greedy.load(Ordering::Relaxed),
            &mut |ev| {
                if let EngineEvent::Text(t) = ev {
                    assistant_text.push_str(&t);
                    stream.push(&t);
                    greedy.store(stream.wants_greedy_sampling(), Ordering::Relaxed);
                    if stream.preflight_error().is_some() {
                        preflight_stop.store(true, Ordering::Relaxed);
                    }
                }
            },
        )
        .map_err(|e| e.to_string())?;
    stream.finish();
    let preflight_error = stream.preflight_error().map(str::to_owned);
    if stats.interrupted && preflight_error.is_none() {
        crate::interrupt::clear();
        return Err("interrupted".to_string());
    }
    let finished = stream.finished();
    let ended_in_think = finished.ended_in_think;
    if let Some(err) = preflight_error.as_deref().or(finished.error) {
        let payload = tool_error_payload(
            pass_error_kind(preflight_error.is_some(), finished.in_think_rejected),
            err,
        );
        close_open_think(&mut assistant_text, ended_in_think);
        return Ok(QuietPass {
            calls: Vec::new(),
            assistant_text,
            tool_error: Some(payload),
            stats,
        });
    }
    let calls = finished.calls.to_vec();
    close_open_think(&mut assistant_text, ended_in_think && !calls.is_empty());
    Ok(QuietPass {
        calls,
        assistant_text,
        tool_error: None,
        stats,
    })
}

impl Agent<'_> {
    /// Runs one model turn: stream text, execute tool calls, repeat until
    /// a turn produces no tool calls. Compacts first when context is tight.
    /// Mirrors the live task list back onto the session after a tool dispatch
    /// may have mutated it, so the persisted/rendered copy stays current and
    /// the session is marked dirty when the list actually changed.
    fn sync_tasks_after_dispatch(&mut self) {
        if self.session.tasks != self.tool_ctx.tasks {
            self.session.tasks.clone_from(&self.tool_ctx.tasks);
            self.session.dirty = true;
        }
    }

    #[allow(clippy::too_many_lines)] // flat generate→tools loop; splitting hurts readability
    fn run_turn(&mut self) -> Result<(), String> {
        crate::title::set(crate::title::State::Busy(self.last_user_prompt()));
        self.last_turn_interrupted = false;
        self.tool_ctx.skill_invocations = 0;
        // The session owns the persisted task list; load it into the live tool
        // context so the `task` tool mutates the copy that renders and saves.
        self.tool_ctx.tasks.clone_from(&self.session.tasks);
        if let Some(reason) = self.fire_user_prompt_submit(&mut |w| println!("{w}")) {
            println!("{}", self.debug_line(&format!("halted by hook: {reason}")));
            return Ok(());
        }
        // A compaction that did not rebuild (interrupted, or no usable summary)
        // ends the turn here, with the conversation untouched — the C goes
        // straight back to IDLE (`worker_run_turn`).
        if self.maybe_compact()?.aborted() {
            return Ok(());
        }
        self.maybe_append_system_prompt_reminder();
        // One clock for the whole turn: elapsed time accumulates across the
        // generate → tools → generate loop instead of restarting per pass.
        let turn_start = Instant::now();
        // Stop hooks run at most once per turn, so a hook that always exits 2
        // cannot loop the model forever.
        let mut stop_hook_ran = false;
        loop {
            let prompt_text = render_transcript(&self.session, &self.system);
            let (stream, assistant_text, stats) =
                self.stream_generation(&prompt_text, turn_start)?;

            let mut assistant_text = assistant_text;
            // A preflight stop reads as an engine interrupt, but it is a tool
            // error to feed back to the model, not a user abort.
            let preflight_error = stream.preflight_error().map(str::to_owned);
            let finished = stream.finished();
            let real_interrupt = stats.interrupted && preflight_error.is_none();
            // A real interrupt lands regardless of what the parser made of
            // the partial stanza it cut off (often an "incomplete DSML tool
            // call" parse error): it never continues with a <tool_result>,
            // so it must win over `finished.error` here.
            let turn_continues = !real_interrupt
                && (!finished.calls.is_empty()
                    || preflight_error.is_some()
                    || finished.error.is_some());
            close_open_think(
                &mut assistant_text,
                finished.ended_in_think && turn_continues,
            );
            self.session.push(Message::assistant(assistant_text));
            let st = Status {
                state: if stats.interrupted {
                    WorkerState::Stopped
                } else {
                    WorkerState::Idle
                },
                ctx_used: stats.ctx_used,
                ctx_size: self.engine.ctx_size(),
                generated: stats.generated,
                gen_tps: stats.tps,
                power_percent: self.power_percent,
                think: self.think,
                ..Status::default()
            };
            if real_interrupt {
                crate::interrupt::clear();
                let mut renderer = stream.into_sink().renderer;
                renderer.finish();
                if !renderer.last_output_newline() {
                    println!();
                }
                if self.show_footer && !self.editor_owns_footer {
                    print_footer(&st, self.color);
                }
                self.last_turn_interrupted = true;
                if crate::notify::should_notify_complete(
                    turn_start.elapsed(),
                    crate::settings::active().ui.notify_after_secs,
                ) {
                    self.notify_task_complete();
                }
                return Ok(());
            }
            if let Some(err) = preflight_error.as_deref().or(finished.error) {
                let payload = tool_error_payload(
                    pass_error_kind(preflight_error.is_some(), finished.in_think_rejected),
                    err,
                );
                self.session.push(Message::user(format!(
                    "<tool_result>{payload}</tool_result>"
                )));
                continue;
            }
            if !finished.calls.is_empty() {
                let calls = finished.calls.to_vec();
                let observations = self.run_tool_calls(&calls);
                self.sync_tasks_after_dispatch();
                let mut renderer = stream.into_sink().renderer;
                renderer.finish();
                for preview in std::mem::take(&mut self.tool_ctx.edit_previews) {
                    print!("{}", preview.to_ansi(self.color));
                }
                for line in std::mem::take(&mut self.tool_ctx.task_completions) {
                    println!("{}", self.debug_line(&format!("✓ {line}")));
                }
                for warning in self.tool_ctx.hook_warnings.drain(..) {
                    let line = if self.color {
                        format!("\x1b[38;5;238m{warning}{ANSI_RESET}")
                    } else {
                        warning
                    };
                    println!("{line}");
                }
                self.session.push(Message::user(format!(
                    "<tool_result>{observations}</tool_result>"
                )));
                // A tool hook's `continue:false` envelope halts the turn.
                if let Some(reason) = self.tool_ctx.hook_stop.take() {
                    println!("{}", self.debug_line(&format!("halted by hook: {reason}")));
                    return Ok(());
                }
                continue;
            }
            let mut renderer = stream.into_sink().renderer;
            renderer.finish();
            if !renderer.last_output_newline() {
                println!();
            }
            // Stop hooks: exit 2 feeds stderr to the model and the turn
            // continues (at most once).
            if !stop_hook_ran && let Some(feedback) = self.run_stop_hooks(&mut |w| println!("{w}"))
            {
                stop_hook_ran = true;
                self.session.push(Message::user(format!(
                    "<tool_result>Stop hook feedback:\n{feedback}</tool_result>"
                )));
                continue;
            }
            if self.show_footer && !self.editor_owns_footer {
                print_footer(&st, self.color);
            }
            if crate::notify::should_notify_complete(
                turn_start.elapsed(),
                crate::settings::active().ui.notify_after_secs,
            ) {
                self.notify_task_complete();
            }
            // Turn over: the front end is back at the prompt.
            crate::title::set(crate::title::State::Idle);
            crate::warp::emit("stop", &self.session.id);
            return Ok(());
        }
    }

    /// Runs the Stop hooks; returns the model-visible feedback of the first
    /// exit-2 hook, `None` when the turn may conclude. `warn` receives
    /// user-only lines from other nonzero exits.
    fn run_stop_hooks(&mut self, warn: &mut dyn FnMut(String)) -> Option<String> {
        if self.tool_ctx.hooks.stop.is_empty() {
            return None;
        }
        let input = crate::hooks::tool_event_input("Stop", "", "{}", None, &self.tool_ctx.cwd);
        let out =
            crate::hooks::run_event(&self.tool_ctx.hooks.stop, "", &input, &self.tool_ctx.cwd);
        for w in out.warnings.into_iter().chain(out.system_messages) {
            warn(w);
        }
        // A `continue:false` envelope wins over an exit-2 feedback loop: the
        // turn concludes rather than being fed back to the model.
        if out.stop_reason.is_some() {
            return None;
        }
        // A Stop `prompt` hook's text is fed to the model just like exit-2
        // feedback, so a prompt hook can steer the model to keep working.
        out.block.or(out.context)
    }

    /// Fires the `UserPromptSubmit` hooks for the turn's triggering prompt (the
    /// last user message). Exit-0 stdout and any exit-2 block feedback inject a
    /// `<hook_context>` user message into this turn; other nonzero exits warn.
    fn fire_user_prompt_submit(&mut self, warn: &mut dyn FnMut(String)) -> Option<String> {
        if self.tool_ctx.hooks.user_prompt_submit.is_empty() {
            return None;
        }
        let prompt = self
            .session
            .transcript
            .iter()
            .rev()
            .find(|m| m.role == crate::session::Role::User)
            .map(|m| m.text.clone())
            .unwrap_or_default();
        let input = crate::hooks::lifecycle_event_input(
            "UserPromptSubmit",
            &[("prompt", &prompt)],
            &self.tool_ctx.cwd,
        );
        let out = crate::hooks::run_event_ctx(
            &self.tool_ctx.hooks.user_prompt_submit,
            "",
            &input,
            &self.tool_ctx.cwd,
        );
        for w in out.warnings.into_iter().chain(out.system_messages) {
            warn(w);
        }
        if let Some(ctx) = out.context.or(out.block) {
            self.session
                .push(Message::user(format!("<hook_context>{ctx}</hook_context>")));
        }
        out.stop_reason
    }

    /// Fires the `SessionStart` hooks with the given source (startup|resume|
    /// clear|compact), injecting any produced context as a `<hook_context>`
    /// user message so it rides along with the session.
    fn fire_session_start(&mut self, source: &str, warn: &mut dyn FnMut(String)) {
        if self.tool_ctx.hooks.session_start.is_empty() {
            return;
        }
        let input = crate::hooks::lifecycle_event_input(
            "SessionStart",
            &[("source", source)],
            &self.tool_ctx.cwd,
        );
        let out = crate::hooks::run_event_ctx(
            &self.tool_ctx.hooks.session_start,
            "",
            &input,
            &self.tool_ctx.cwd,
        );
        for w in out.warnings.into_iter().chain(out.system_messages) {
            warn(w);
        }
        if let Some(ctx) = out.context {
            self.session
                .push(Message::user(format!("<hook_context>{ctx}</hook_context>")));
        }
    }

    /// Fires the `SessionEnd` hooks with the exit `reason`. Terminal event: no
    /// context is injected, only user-visible warnings are surfaced.
    fn fire_session_end(&mut self, reason: &str, warn: &mut dyn FnMut(String)) {
        if self.tool_ctx.hooks.session_end.is_empty() {
            return;
        }
        let input = crate::hooks::lifecycle_event_input(
            "SessionEnd",
            &[("reason", reason)],
            &self.tool_ctx.cwd,
        );
        let out = crate::hooks::run_event(
            &self.tool_ctx.hooks.session_end,
            "",
            &input,
            &self.tool_ctx.cwd,
        );
        for w in out
            .warnings
            .into_iter()
            .chain(out.system_messages)
            .chain(out.block)
        {
            warn(w);
        }
    }

    /// Re-injects the trusted system prompt shape after enough context has
    /// passed since it was last seen, mirroring the C's pressure policy.
    fn maybe_append_system_prompt_reminder(&mut self) {
        let rendered = render_transcript(&self.session, &self.system);
        let pos = self.engine.count_tokens(&rendered);
        if !self.reminder.should_remind(pos) {
            return;
        }
        println!(
            "{}",
            self.debug_line("Re-injecting system prompt reminder...")
        );
        self.trace.line(&format!(
            "system prompt reminder injected at transcript={pos}"
        ));
        let mut text = sysprompt::build_system_prompt_reminder(
            &self.tool_ctx.mcp,
            !crate::settings::active().engine.thinking_tool_calls,
        );
        if !self.cfg.system.is_empty() {
            text.push_str("\nAdditional system instructions reminder:\n");
            text.push_str(&self.cfg.system);
            text.push_str("\n[End additional system instructions reminder.]\n\n");
        }
        self.session.push(Message::user(text));
    }

    /// Compacts the transcript when the rendered context is nearly full.
    fn maybe_compact(&mut self) -> Result<Compacted, String> {
        let rendered = render_transcript(&self.session, &self.system);
        let used = self.engine.count_tokens(&rendered);
        if !compact::should_compact(self.engine.ctx_size(), used) {
            return Ok(Compacted::Done);
        }
        // Cheapest step first: clear old tool-result bodies (no model
        // round-trip) and only fall back to full summarization if still tight.
        if let Some(cleared) = self.try_microcompact() {
            println!(
                "{}",
                self.debug_line(&format!(
                    "microcompacted: cleared {cleared} old tool result(s)"
                ))
            );
            return Ok(Compacted::Done);
        }
        self.compact("low context", "")
    }

    /// Runs microcompact; returns the cleared count when it freed enough
    /// context to skip full compaction, `None` when full compaction is still
    /// needed (any clearing done is kept — it only helps the summary pass).
    fn try_microcompact(&mut self) -> Option<usize> {
        let cleared = compact::microcompact(&mut self.session.transcript);
        if cleared == 0 {
            return None;
        }
        self.last_ctx_used = 0;
        let rendered = render_transcript(&self.session, &self.system);
        let used = self.engine.count_tokens(&rendered);
        (!compact::should_compact(self.engine.ctx_size(), used)).then_some(cleared)
    }

    /// Rebuilds the transcript after a summarization pass: extracted summary
    /// + verbatim tail + budgeted re-injection of recently read files.
    fn rebuild_after_compact(&mut self, raw_summary: &str) {
        let summary = compact::extract_summary(raw_summary);
        let budget = compact::tail_budget(self.engine.ctx_size());
        let mut tail_start = self.session.transcript.len();
        let mut tail_tokens = 0;
        while tail_start > 0 {
            let m = &self.session.transcript[tail_start - 1];
            tail_tokens += self.engine.count_tokens(&m.text);
            if tail_tokens > budget {
                break;
            }
            tail_start -= 1;
        }
        let tail: Vec<Message> = self.session.transcript[tail_start..].to_vec();
        self.session.transcript = Vec::new();
        // Off-path branches index into the transcript being replaced here, so
        // they cannot survive the rewrite; drop them rather than let them
        // point at the wrong messages (issue #65).
        self.session.clear_branches();
        self.session.push(Message::user(format!(
            "<tool_result>Compacted session summary:\n{summary}</tool_result>"
        )));
        self.session.transcript.extend(tail);
        let reinject = compact::build_reinjection(
            &self.tool_ctx.recent_reads,
            compact::reinject_budget(self.engine.ctx_size()),
            &mut |s| self.engine.count_tokens(s),
        );
        if let Some(block) = reinject {
            self.session.push(Message::user(block));
        }
        // The rebuild already invalidated the KV prefix, so re-surfacing the
        // task list here is free — and afterwards the transcript is append-only
        // again, keeping the engine's cached prefix valid (issue #35).
        if let Some(block) = self.session.tasks.inject_block() {
            self.session.push(Message::user(block));
        }
        self.last_ctx_used = 0;
    }

    /// The `trigger` value compaction hooks receive: `manual` for a user-driven
    /// `/compact`, `auto` for a threshold-driven pass.
    fn compact_trigger(reason: &str) -> &'static str {
        if reason == "user request" {
            "manual"
        } else {
            "auto"
        }
    }

    /// Fires `PreCompact`. Any injected context is pinned as a user message so
    /// it survives the rebuild inside the verbatim tail.
    ///
    /// Shared by both orchestrators ([`Agent::compact`] and
    /// [`Agent::do_compact_notify`]) rather than inlined in one of them: the
    /// hooks used to fire only on the plain-REPL path, so a hook configured by a
    /// TUI user — the default front-end — silently never ran.
    fn fire_pre_compact(&mut self, trigger: &str, note: &mut dyn FnMut(String)) {
        if self.tool_ctx.hooks.pre_compact.is_empty() {
            return;
        }
        let input = crate::hooks::lifecycle_event_input(
            "PreCompact",
            &[("trigger", trigger)],
            &self.tool_ctx.cwd,
        );
        let out = crate::hooks::run_event_ctx(
            &self.tool_ctx.hooks.pre_compact,
            "",
            &input,
            &self.tool_ctx.cwd,
        );
        for w in out.warnings.into_iter().chain(out.system_messages) {
            note(w);
        }
        if let Some(ctx) = out.context {
            self.session
                .push(Message::user(format!("<hook_context>{ctx}</hook_context>")));
        }
    }

    /// Fires `PostCompact` with the extracted durable summary. Injected context
    /// is appended after the rebuilt transcript. See [`Agent::fire_pre_compact`]
    /// for why this is shared.
    fn fire_post_compact(&mut self, trigger: &str, summary: &str, note: &mut dyn FnMut(String)) {
        if self.tool_ctx.hooks.post_compact.is_empty() {
            return;
        }
        let input = crate::hooks::lifecycle_event_input(
            "PostCompact",
            &[("trigger", trigger), ("summary", summary)],
            &self.tool_ctx.cwd,
        );
        let out = crate::hooks::run_event_ctx(
            &self.tool_ctx.hooks.post_compact,
            "",
            &input,
            &self.tool_ctx.cwd,
        );
        for w in out.warnings.into_iter().chain(out.system_messages) {
            note(w);
        }
        if let Some(ctx) = out.context {
            self.session
                .push(Message::user(format!("<hook_context>{ctx}</hook_context>")));
        }
    }

    /// Performs the compaction exchange and rebuilds the transcript as
    /// summary + recent verbatim tail.
    fn compact(&mut self, reason: &str, instructions: &str) -> Result<Compacted, String> {
        print!("{}", compact::banner(reason, self.color));
        // Restored on drop, so an interrupted or failed pass hands the window
        // back to whatever it said before (a running turn, or the idle prompt).
        let _title = crate::title::Scoped::set(crate::title::State::Compacting);
        let trigger = Self::compact_trigger(reason);
        self.fire_pre_compact(trigger, &mut |w| println!("{w}"));
        let mut prompt_text = render_transcript(&self.session, &self.system);
        {
            use std::fmt::Write as _;
            let _ = write!(
                prompt_text,
                "[user]\n{}\n",
                compact::make_prompt(reason, instructions)
            );
        }
        // Posted on this path too, for the sake of one behavior rather than two:
        // the plain REPL has no status bar to draw it, so nothing renders here,
        // but the state is then correct for whoever reads it.
        let progress = status::CompactProgress::begin();
        let mut summary = String::new();
        let stats = self
            .engine
            .generate(
                crate::engine::Prompt::Flat(&prompt_text),
                &self.cfg.generation,
                &|| crate::interrupt::pending(),
                &|| false,
                &mut |ev| match ev {
                    EngineEvent::Text(t) => {
                        summary.push_str(&t);
                        progress.summarizing(summary.len());
                    }
                    EngineEvent::Prefill(p) => progress.prefill(p.done, p.total),
                    EngineEvent::Notice(_) => {}
                },
            )
            .map_err(|e| e.to_string())?;
        drop(progress);
        if self.color {
            print!("\x1b[0m");
        }
        if stats.interrupted {
            println!("{}", status::system_line(COMPACT_INTERRUPTED, self.color));
            crate::interrupt::clear();
            return Ok(Compacted::Interrupted);
        }
        let extracted = compact::extract_summary(&summary);
        if extracted.trim().is_empty() {
            println!("{}", status::system_line(COMPACT_NO_SUMMARY, self.color));
            return Ok(Compacted::NoSummary);
        }
        self.rebuild_after_compact(&summary);
        self.fire_post_compact(trigger, &extracted, &mut |w| println!("{w}"));
        println!("{}", self.debug_line("context compacted"));
        Ok(Compacted::Done)
    }

    /// Takes the pending `/btw` question and frames it for a *multiplexed*
    /// answer, or `None` when the aside should freeze the main task instead.
    ///
    /// **One aside at a time, and no queue.** A multiplexed aside occupies the
    /// one side panel and costs one fork's worth of KV for as long as it runs,
    /// so a second concurrent one has nowhere to render and no reason to exist.
    /// Extra questions are therefore *dropped with a notice* rather than
    /// queued: with the aside answering beside the main task instead of
    /// freezing it, a queue would only hold questions back behind an answer the
    /// user can already read.
    ///
    /// `None` also covers the engine being unable to fork or multiplex
    /// (`EchoEngine`, remote engines), which falls back to the freeze path.
    ///
    /// The frozen partial is spliced in for the same reason the freeze path
    /// splices it: it is live in the KV, and a prompt that omits it re-prefills
    /// the whole conversation.
    fn multiplexable_aside(
        &mut self,
        tx: &Sender<UiEvent>,
        shared: &TurnShared,
        frozen_partial: &str,
    ) -> Option<String> {
        if !self.engine.supports_multiplexing() || self.pending_aside.is_some() {
            return None;
        }
        let question = shared.pop_btw()?;
        let dropped = shared.clear_btw();
        if dropped > 0 {
            let _ = tx.send(UiEvent::Dim(format!(
                "[btw — answering one at a time; dropped {dropped} more]"
            )));
        }
        let mut prompt = render_transcript(&self.session, &self.system);
        {
            use std::fmt::Write as _;
            if !frozen_partial.trim().is_empty() {
                let _ = write!(prompt, "[assistant]\n{}\n", frozen_partial.trim_end());
            }
            let _ = write!(prompt, "[user]\n{}\n", btw_user_message(&question));
        }
        Some(prompt)
    }

    /// Runs a tool-free aside, picking the best tier the engine offers.
    ///
    /// Tier 1, preferred whenever the engine offers it, answers on a forked
    /// session, so the live KV is never written to. Tier 2 is the historical
    /// path: answer destructively on the live session and restore it
    /// unconditionally afterwards. Callers that cannot run either fall back to
    /// the boundary queue, which is the third tier and lives at the call site.
    ///
    /// An `unsupported` error from the fork tier is a fall-through, not a
    /// failure — a real backend error is not, and surfaces to the caller.
    fn generate_aside_best(
        &mut self,
        prompt: &str,
        opts: &crate::engine::GenerationOptions,
        interrupt: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(crate::engine::EngineEvent),
    ) -> Result<crate::engine::GenerationStats, crate::engine::EngineError> {
        if self.engine.supports_forked_aside() {
            match self
                .engine
                .generate_aside_forked(prompt, opts, interrupt, on_event)
            {
                Err(e) if e.is_unsupported() => {}
                result => return result,
            }
        }
        self.engine
            .generate_aside(prompt, opts, interrupt, on_event)
    }

    /// Folds a completed pass's provider usage into the session tally. A no-op
    /// for local engines (`stats.usage` is `None`), so `/usage` stays empty
    /// unless an online provider is driving the turns.
    fn record_usage(&mut self, stats: &crate::engine::GenerationStats) {
        // Peak decode rate for this model, reported at exit. Attributed to the
        // engine that actually ran the pass, which during a sidechain is the
        // alt engine — same rule as the token tally below.
        crate::speeds::note_generation(&self.engine.model_name(), stats.tps, stats.generated);
        // Engine-agnostic in/out tally. Must run before `self.last_ctx_used` is
        // updated for this pass, so the local input estimate below sees the
        // previous context size.
        let (input, output) = if let Some(u) = stats.usage {
            // Provider: exact figures from the usage block. `stats.generated`
            // is not populated on the provider path, so read the output there.
            (
                i64::from(u.input_tokens)
                    + i64::from(u.cache_read_tokens)
                    + i64::from(u.cache_write_tokens),
                i64::from(u.output_tokens),
            )
        } else {
            // Local: output is the generated count; input is the growth in
            // context minus what the model itself generated. Clamped so
            // compaction (context shrinking) never subtracts from the tally.
            (
                i64::from(stats.ctx_used)
                    - i64::from(self.last_ctx_used)
                    - i64::from(stats.generated),
                i64::from(stats.generated),
            )
        };
        // Attributed to `self.engine`, which during a sidechain *is* the alt
        // engine — the swap happens before the pass, so this needs no notion of
        // sub-agents to split their tokens out correctly.
        self.stats.add(
            &engine_stats_label(&*self.engine),
            u64::try_from(input.max(0)).unwrap_or(0),
            u64::try_from(output.max(0)).unwrap_or(0),
        );

        if let Some(u) = stats.usage {
            self.usage.total.add(u);
            self.usage.turns += 1;
        }
    }

    /// Renders the `/usage` report: cumulative billed token usage for online
    /// (provider) models this session. Prints a short note when no provider
    /// turn has run (local engine, or nothing generated yet).
    fn render_usage_report(&self, color: bool) -> String {
        use std::fmt::Write as _;
        let dim = |s: &str| {
            if color {
                format!("\x1b[38;5;238m{s}{ANSI_RESET}")
            } else {
                s.to_owned()
            }
        };
        if self.usage.turns == 0 {
            if self.cfg.provider.is_none() {
                // Local engine: there is no bill, so bill the user in nonsense.
                return render_local_invoice(
                    &self.engine.model_name(),
                    self.stats.input_tokens,
                    self.stats.output_tokens,
                    color,
                );
            }
            return format!(
                "{}\n",
                dim("No provider usage yet this session — run a turn first.")
            );
        }
        let t = self.usage.total;
        let model = self
            .cfg
            .provider_model
            .as_deref()
            .unwrap_or("(unknown model)");
        let provider = self.cfg.provider.map_or("provider", |p| p.label());
        let prompt_total = t
            .input_tokens
            .saturating_add(t.cache_read_tokens)
            .saturating_add(t.cache_write_tokens);
        let grand_total = prompt_total.saturating_add(t.output_tokens);
        let mut out = String::new();
        let _ = writeln!(out, "{}", dim(&format!("Usage — {provider}:{model}")));
        let _ = writeln!(out, "  turns          {}", self.usage.turns);
        let _ = writeln!(out, "  input tokens   {}", fmt_int(t.input_tokens));
        let _ = writeln!(out, "  output tokens  {}", fmt_int(t.output_tokens));
        // Cache figures are only reported by providers that support prompt
        // caching (Anthropic); omit the section entirely when both are zero.
        if t.cache_read_tokens > 0 || t.cache_write_tokens > 0 {
            let _ = writeln!(out, "  cache read     {}", fmt_int(t.cache_read_tokens));
            let _ = writeln!(out, "  cache write    {}", fmt_int(t.cache_write_tokens));
            if prompt_total > 0 {
                let pct = i64::from(t.cache_read_tokens) * 100 / i64::from(prompt_total);
                let _ = writeln!(out, "  cache hit rate {pct}% of prompt tokens");
            }
        }
        let _ = writeln!(
            out,
            "  total tokens   {} {}",
            fmt_int(grand_total),
            dim("(prompt + output)")
        );
        out
    }

    /// Renders the `/context` usage breakdown with Claude Code's layout: a
    /// 20-column cell grid (1k tokens per cell, coarser for large contexts
    /// so the grid stays within half a typical screen) beside the model and
    /// totals, then the estimated usage per category.
    #[allow(clippy::too_many_lines)]
    fn render_context_report(&self, color: bool) -> String {
        use std::fmt::Write as _;
        /// Glyph for an unused context cell in the grid.
        const FREE_CELL: char = '⛶';
        /// Grid width in cells.
        const GRID_COLS: usize = 20;
        /// Maximum grid height in rows.
        const MAX_GRID_ROWS: usize = 16;
        /// Category colors matching Claude Code: violet, cyan, purple, gray.
        const COL_SYSTEM: &str = "\x1b[38;5;105m";
        const COL_MCP: &str = "\x1b[38;5;44m";
        const COL_MSG: &str = "\x1b[38;5;134m";
        const COL_CONTEXT: &str = "\x1b[38;5;208m";
        const COL_MEMORY: &str = "\x1b[38;5;114m";
        const COL_FREE: &str = "\x1b[38;5;240m";
        let paint = |col: &'static str| if color { col } else { "" };
        let reset = if color { ANSI_RESET } else { "" };
        let ctx_size = self.engine.ctx_size().max(1);
        let mut schemas = String::new();
        crate::tools::mcp::append_tool_schemas(&mut schemas, &self.tool_ctx.mcp);
        let mcp_tokens = if schemas.is_empty() {
            0
        } else {
            self.engine.count_tokens(&schemas)
        };
        // MCP tool schemas are embedded in the composed system prompt; split
        // them out so the two categories don't double-count.
        // The system prompt includes: tools prompt + user system text
        let mut system_tokens = (self.engine.count_tokens(&self.system) - mcp_tokens).max(0);
        let mut mcp_tokens = mcp_tokens;
        // AGENTS.md tokens from the context collected at session start.
        let context_tokens =
            ContextTokens::count(&self.context_content, |s| self.engine.count_tokens(s));
        // Message tokens: all transcript messages (user and assistant)
        let raw_message_tokens: i32 = self
            .session
            .transcript
            .iter()
            .map(|m| self.engine.count_tokens(&m.text))
            .sum();
        // AGENTS.md gets its own category; git and date context stay grouped
        // under Messages (they are part of the injected first user message).
        let agents_md_tokens = context_tokens.agents_md;
        let memory_tokens = context_tokens.memory;
        let mut message_tokens = raw_message_tokens - agents_md_tokens - memory_tokens;

        let estimated =
            system_tokens + mcp_tokens + message_tokens + agents_md_tokens + memory_tokens;
        if self.last_ctx_used > estimated && estimated > 0 {
            let scale = |t: i32| {
                i32::try_from(i64::from(t) * i64::from(self.last_ctx_used) / i64::from(estimated))
                    .unwrap_or(t)
            };
            system_tokens = scale(system_tokens);
            mcp_tokens = scale(mcp_tokens);
            message_tokens = scale(message_tokens);
        }

        let used = (system_tokens + mcp_tokens + message_tokens + agents_md_tokens + memory_tokens)
            .min(ctx_size);
        let free = ctx_size - used;
        let pct = |n: i32| f64::from(n) * 100.0 / f64::from(ctx_size);

        // Categories are told apart by color; the glyph of each cell shows
        // how full that cell is (see `fill_glyph`).
        let mut categories = vec![
            ("System prompt", system_tokens, COL_SYSTEM),
            ("MCP tools", mcp_tokens, COL_MCP),
        ];

        if agents_md_tokens > 0 {
            categories.push(("AGENTS.md", agents_md_tokens, COL_CONTEXT));
        }

        if memory_tokens > 0 {
            categories.push(("Memory", memory_tokens, COL_MEMORY));
        }

        categories.push(("Messages", message_tokens, COL_MSG));

        // Glyph for a cell by its fill fraction: <25%, <50%, <75%, full.
        let fill_glyph = |frac: f64| -> char {
            if frac < 0.25 {
                '⛀'
            } else if frac < 0.5 {
                '⛂'
            } else if frac < 0.75 {
                '⛁'
            } else {
                '⛃'
            }
        };

        // Adaptive density: 1k tokens per cell, coarsened (in 1k steps) so the
        // grid never exceeds half a typical 24-row screen. Every non-empty
        // category shows at least one cell; free space takes what remains.
        #[allow(clippy::cast_sign_loss)]
        let ctx = ctx_size as usize;
        let tokens_per_cell = ctx
            .div_ceil(GRID_COLS * MAX_GRID_ROWS)
            .div_ceil(1000)
            .max(1)
            * 1000;
        let total_cells = ctx.div_ceil(tokens_per_cell);
        let mut cells: Vec<(char, &'static str)> = Vec::with_capacity(total_cells);
        for &(_, tokens, col) in &categories {
            if tokens <= 0 || cells.len() == total_cells {
                continue;
            }
            // Whole cells render full; the trailing remainder renders with a
            // glyph matching its fill fraction.
            #[allow(clippy::cast_sign_loss)]
            let tokens = tokens as usize;
            let full = (tokens / tokens_per_cell).min(total_cells - cells.len());
            cells.extend(std::iter::repeat_n(('⛃', col), full));
            let rem = tokens % tokens_per_cell;
            if rem > 0 && cells.len() < total_cells {
                #[allow(clippy::cast_precision_loss)]
                cells.push((fill_glyph(rem as f64 / tokens_per_cell as f64), col));
            }
        }
        cells.truncate(total_cells);
        cells.resize(total_cells, (FREE_CELL, COL_FREE));
        let grid_rows = total_cells.div_ceil(GRID_COLS);

        // Right-hand column: model line, totals, then the category legend.
        let model = self.engine.model_name();
        let mut right: Vec<String> = Vec::new();
        if !model.is_empty() {
            right.push(model);
        }
        right.push(format!(
            "{}/{} tokens ({:.0}%)",
            status::format_ctx_size(used),
            status::format_ctx_size(ctx_size),
            pct(used)
        ));
        right.push(String::new());
        right.push("Estimated usage by category".to_owned());
        for &(label, tokens, col) in &categories {
            right.push(format!(
                "{}⛃{reset} {label}: {} tokens ({:.1}%)",
                paint(col),
                status::format_ctx_size(tokens),
                pct(tokens)
            ));
        }
        right.push(format!(
            "{}{FREE_CELL}{reset} Free space: {} ({:.1}%)",
            paint(COL_FREE),
            status::format_ctx_size(free),
            pct(free)
        ));
        right.push(format!(
            "1 cell = {} tokens",
            status::format_ctx_size(i32::try_from(tokens_per_cell).unwrap_or(i32::MAX))
        ));

        let mut out = String::from("Context Usage\n");
        let rows = right.len().max(grid_rows);
        for row in 0..rows {
            out.push_str("  ");
            if row < grid_rows {
                let start = row * GRID_COLS;
                let end = (start + GRID_COLS).min(total_cells);
                for &(glyph, col) in &cells[start..end] {
                    out.push_str(paint(col));
                    out.push(glyph);
                    out.push_str(reset);
                    out.push(' ');
                }
                out.push_str(&" ".repeat(2 * (start + GRID_COLS - end)));
            } else {
                out.push_str(&" ".repeat(2 * GRID_COLS));
            }
            if let Some(text) = right.get(row) {
                let _ = write!(out, "   {text}");
            }
            out.push('\n');
        }
        out
    }

    /// Runs the /init command: prompts the model to create AGENTS.md
    fn run_init(&mut self) {
        println!("Initializing AGENTS.md...");
        println!("The model will now analyze the codebase and generate documentation.\n");

        let prompt = concat!(
            "Analyze this codebase and create an AGENTS.md file for future agent sessions.\n\n",
            "Include:\n",
            "1. Build, lint, and test commands (especially non-standard ones)\n",
            "2. High-level architecture and structure\n",
            "3. Required setup or environment variables\n",
            "4. Non-obvious gotchas or workflow quirks\n\n",
            "Exclude:\n",
            "- File-by-file listings Claude can discover\n",
            "- Standard language conventions\n",
            "- Generic advice\n",
            "- Information from README unless essential\n\n",
            "Preface with:\n",
            "```",
            "# AGENTS.md\n\n",
            "This file provides guidance to the agent when working with code in this repository.",
            "```",
            "\n\n",
            "Write the AGENTS.md file to the current directory."
        );

        self.session.push(Message::user(prompt));
        if let Err(e) = self.run_turn() {
            println!("/init failed: {e}");
        }
    }

    /// Runs the /init command in TUI mode.
    #[allow(clippy::too_many_arguments)]
    fn tui_run_init(
        &mut self,
        log: &mut OutputLog,
        terminal: &mut ratatui::DefaultTerminal,
        view: &mut tui::OutputView,
        input: &mut TuiInput,
        btw: &mut BtwPanel,
        arcade: &mut crate::arcade::Arcade,
        sub: &mut tui::SubPane,
    ) {
        log.push_plain("Initializing AGENTS.md...");
        log.push_plain("The model will now analyze the codebase and generate documentation.\n");

        let prompt = concat!(
            "Analyze this codebase and create an AGENTS.md file for future agent sessions.\n\n",
            "Include:\n",
            "1. Build, lint, and test commands (especially non-standard ones)\n",
            "2. High-level architecture and structure\n",
            "3. Required setup or environment variables\n",
            "4. Non-obvious gotchas or workflow quirks\n\n",
            "Exclude:\n",
            "- File-by-file listings Claude can discover\n",
            "- Standard language conventions\n",
            "- Generic advice\n",
            "- Information from README unless essential\n\n",
            "Preface with:\n",
            "```",
            "# AGENTS.md\n\n",
            "This file provides guidance to the agent when working with code in this repository.",
            "```",
            "\n\n",
            "Write the AGENTS.md file to the current directory."
        );

        log.push_spans(tui::user_echo_spans(prompt));
        self.session.push(Message::user(prompt));
        if let Err(e) = self.tui_turn(terminal, log, view, input, btw, arcade, sub) {
            log.push_plain(format!("/init failed: {e}"));
        }
    }

    /// Handles a slash command; returns false when the REPL should exit.
    #[allow(clippy::too_many_lines)]
    fn slash(&mut self, input: &str) -> Result<bool, String> {
        let mut parts = input.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or(input);
        let arg = parts.next().unwrap_or("").trim();
        match cmd {
            "/init" => {
                self.run_init();
                return Ok(true);
            }
            "/quit" | "/exit" => return Ok(false),
            "/new" | "/clear" => {
                self.session = Session::new();
                self.broadcast_session_reset(None);
                self.reminder = SystemPromptReminder::new();
                self.context_content = ContextContent::new();
                push_session_context(&mut self.session, &self.context_content);
                // Scaffolding only — not activity worth a resume point (see
                // `save_for_exit`); a real turn re-dirties it.
                self.session.dirty = false;
                self.last_ctx_used = 0;
                self.checkpoints.clear();
                self.usage = SessionUsage::default();
                // Reinstate the warm prefix; without it the next turn silently
                // rebuilds the whole system-prompt KV (see `rewarm_after_reset`).
                // The plain REPL has no persistent prompt to replace, so the
                // analogue of the TUI throbber is one transient stderr line,
                // erased once the KV is back (matching `warm_plain`).
                let color = self.color;
                let mut announced = false;
                self.rewarm_after_reset(&mut || {
                    if !announced {
                        announced = true;
                        if color {
                            eprint!("\x1b[33mstarting a new session…{ANSI_RESET}");
                        } else {
                            eprint!("starting a new session…");
                        }
                        let _ = std::io::Write::flush(&mut std::io::stderr());
                    }
                });
                if announced {
                    eprint!("\r\x1b[2K");
                    let _ = std::io::Write::flush(&mut std::io::stderr());
                }
                self.fire_session_start("clear", &mut |w| println!("{w}"));
                println!("started a new session");
            }
            "/help" => print!("{}", crate::config::usage()),
            "/version" => println!("plank {}", crate::logo::version_label()),
            // Easter eggs. `anim.rs` keeps motion in the Ratatui front-end, so
            // the games decline here rather than growing a second game loop
            // against raw stdout.
            _ if crate::arcade::enabled() && crate::arcade::Arcade::COMMANDS.contains(&cmd) => {
                println!("{cmd} needs the full-screen UI — run plank in a terminal");
            }
            "/checkpoint" => {
                if arg.is_empty() {
                    print!(
                        "{}",
                        crate::checkpoint::render_list(&self.checkpoints, now_secs(), self.color)
                    );
                } else {
                    println!("{}", self.checkpoint_create(arg));
                }
            }
            "/rollback" => {
                if arg.is_empty() {
                    println!("usage: /rollback <name> (see /checkpoint for the list)");
                } else {
                    match self.rollback_to(arg) {
                        Ok(msg) => println!("{msg}"),
                        Err(e) => println!("{e}"),
                    }
                }
            }
            "/tree" => print!("{}", self.tree_view(self.color)),
            "/fork" => match self.fork_branch(arg, self.color) {
                Ok(msg) => println!("{msg}"),
                Err(e) => println!("{e}"),
            },
            "/clone" => match self.clone_branch() {
                Ok(msg) => println!("{msg}"),
                Err(e) => println!("{e}"),
            },
            "/save" => match self.save_session() {
                Ok(id) => {
                    println!("saved session {}", &id[..8]);
                    if let Some(note) = self.save_session_payload() {
                        println!("{}", self.debug_line(&note));
                    }
                }
                Err(e) => println!("save failed: {e}"),
            },
            "/list" => match self.store.list() {
                Ok(entries) => print!(
                    "{}",
                    crate::session::render_session_list(&entries, now_secs(), self.color)
                ),
                Err(e) => println!("list failed: {e}"),
            },
            "/switch" => match self.store.load(arg) {
                Ok(s) => {
                    print!(
                        "{}",
                        crate::session::render_history(&s.transcript, 6, self.color)
                    );
                    if let Some(note) = self.load_session_payload(&s) {
                        println!("{}", self.debug_line(&note));
                    }
                    self.session = s;
                    self.broadcast_session_reset(Some(
                        "[session replaced — its history is on the local screen only]",
                    ));
                    self.last_ctx_used = 0;
                    self.checkpoints.clear();
                    self.usage = SessionUsage::default();
                }
                Err(e) => println!("switch failed: {e}"),
            },
            "/del" => match self.store.delete(arg) {
                Ok(id) => println!("deleted session {}", &id[..8]),
                Err(e) => println!("delete failed: {e}"),
            },
            "/resume" => match self.resume_pick(arg) {
                Ok(None) => match self.store.list() {
                    Ok(entries) => print!(
                        "{}",
                        crate::session::render_resume_list(
                            &entries,
                            now_secs(),
                            self.color,
                            RESUME_LIST_LIMIT
                        )
                    ),
                    Err(e) => println!("resume failed: {e}"),
                },
                Ok(Some(s)) => {
                    print!(
                        "{}",
                        crate::session::render_history(&s.transcript, 6, self.color)
                    );
                    if let Some(note) = self.load_session_payload(&s) {
                        println!("{}", self.debug_line(&note));
                    }
                    self.session = s;
                    self.broadcast_session_reset(Some(
                        "[session replaced — its history is on the local screen only]",
                    ));
                    self.last_ctx_used = 0;
                    self.checkpoints.clear();
                    self.usage = SessionUsage::default();
                }
                Err(e) => println!("resume failed: {e}"),
            },
            "/tag" => {
                if arg.is_empty() {
                    if self.session.tag.is_empty() {
                        println!("no tag set; usage: /tag <text> (\"/tag -\" clears)");
                    } else {
                        println!("tag: {}", self.session.tag);
                    }
                } else {
                    match self.set_tag(arg) {
                        Ok(msg) => println!("{msg}"),
                        Err(e) => println!("tag failed: {e}"),
                    }
                }
            }
            "/history" => {
                let turns = if arg.is_empty() {
                    HISTORY_DEFAULT_TURNS
                } else {
                    arg.parse::<usize>()
                        .unwrap_or(HISTORY_DEFAULT_TURNS)
                        .clamp(1, HISTORY_MAX_TURNS)
                };
                print!(
                    "{}",
                    crate::session::render_history(&self.session.transcript, turns, self.color)
                );
            }
            "/power" => match crate::config::parse_power_percent(arg) {
                Some(power) => {
                    // No GPU backend yet: record and show it in the footer,
                    // like the C's deferred worker_request_power.
                    self.power_percent = power;
                    crate::status::set_local_power(power);
                    println!("power limit set to {power}%");
                }
                None => println!("usage: /power <1..100>"),
            },
            "/think" => println!("{}", self.think_command(arg)),
            "/notify" => println!("{}", Self::notify_command(arg)),
            // Non-advertised: re-shows the last desktop notification so it can be
            // screenshotted. Not in `/help` or `slash_command_known`.
            "/renotify" => {
                if crate::notify::renotify() {
                    println!("re-showing last notification");
                } else {
                    println!("no notification to re-show yet");
                }
            }
            "/strip" => {
                if arg.is_empty() {
                    println!("usage: /strip <sha-prefix>");
                } else {
                    match self.strip_session(arg) {
                        Ok((sha, tokens)) => {
                            println!("stripped session {} ({tokens} tokens)", &sha[..8]);
                        }
                        Err(e) => println!("strip failed: {e}"),
                    }
                }
            }
            "/config" => {
                if arg.is_empty() {
                    print!(
                        "{}",
                        crate::configform::render_text_list(crate::settings::active())
                    );
                } else {
                    let mut p = arg.splitn(2, char::is_whitespace);
                    let key = p.next().unwrap_or("");
                    let val = p.next().unwrap_or("").trim();
                    let mut working = crate::settings::active().clone();
                    match crate::configform::set_from_path(&mut working, key, val) {
                        Ok(field) => {
                            let (section, fkey) = (field.section, field.key);
                            match crate::settings::project_path() {
                                Some(path) => match working.save_to(&path) {
                                    Ok(()) => {
                                        crate::settings::reinstall(working);
                                        println!(
                                            "set {section}.{fkey} = {} (saved to {})",
                                            crate::configform::display(
                                                crate::settings::active(),
                                                field.id
                                            ),
                                            path.display()
                                        );
                                    }
                                    Err(e) => println!("config save failed: {e}"),
                                },
                                None => println!("config: no working directory"),
                            }
                        }
                        Err(e) => println!("{e}"),
                    }
                }
            }
            "/mcp" => print!("{}", render_mcp_report(&self.tool_ctx.mcp, self.color)),
            "/context" => print!("{}", self.render_context_report(self.color)),
            "/usage" => print!("{}", self.render_usage_report(self.color)),
            "/compact" => {
                // Any argument is extra summarization instructions for this one
                // pass. The interrupted case already printed its own notice.
                self.compact("user request", arg)?;
            }
            "/skills" => print!("{}", crate::skills::render_list(&self.skills)),
            "/templates" => print!("{}", crate::templates::render_list(&self.templates)),
            "/tasks" => print!("{}", self.session.tasks.render_list()),
            "/agent" => print!("{}", crate::agents::render_list(&self.agents)),
            "/hooks" => print!("{}", crate::hooks::render_list(&self.tool_ctx.hooks)),
            "/remote-control" | "/rc" => {
                println!(
                    "{cmd} needs the full-screen TUI — a piped session can't mirror output or run remote prompts"
                );
            }
            "/btw" => {
                if arg.is_empty() {
                    println!("usage: /btw <question>");
                } else {
                    self.btw_plain(arg)?;
                }
            }
            "/remember" => match remember_from_arg(&self.tool_ctx.cwd, arg) {
                Ok(path) => println!(
                    "{}",
                    self.debug_line(&format!("[saved to {}]", path.display()))
                ),
                Err(e) => println!("{e}\nusage: /remember [user] <text> (default scope: project)"),
            },
            "/export" => match self.write_export(arg) {
                Ok(path) => println!("exported session to {}", path.display()),
                Err(e) => println!("export failed: {e}\nusage: /export [md|html] [path]"),
            },
            "/insights" => {
                let color = self.color;
                let mut note = |line: String| println!("{}", status::system_line(&line, color));
                // Reasoning is dimmed prose, not a status line — the same
                // distinction the streaming renderer draws during a turn.
                let mut tick = |line: String| {
                    if color {
                        println!("\x1b[90m{line}\x1b[0m");
                    } else {
                        println!("{line}");
                    }
                };
                match self.run_insights(arg, &mut note, &mut tick) {
                    Ok(Insights::Done { path, summary }) => {
                        for line in summary {
                            println!("{line}");
                        }
                        println!("report written to {}", path.display());
                    }
                    Ok(Insights::Cancelled) => println!("insights cancelled"),
                    Err(e) => println!("insights failed: {e}\nusage: /insights [fast]"),
                }
            }
            "/repro" => match self.write_repro(arg) {
                Ok(path) => println!(
                    "{}",
                    self.debug_line(&format!("[repro written to {}]", path.display()))
                ),
                Err(e) => println!("repro failed: {e}"),
            },
            c if crate::agents::is_subagent_command(c) => {
                // The name now rides on the command token (`/subagent:name`),
                // so the whole argument is the task — no first-token guessing,
                // and a task whose first word happens to match a definition is
                // no longer silently reinterpreted as a persona.
                let mut def = None;
                if let Some(name) = crate::agents::command_name(c) {
                    // A named definition that is not there is an error, not a
                    // fallback: the user asked for a specific persona, and
                    // running a different one would be worse than saying so.
                    let Some(d) = crate::agents::resolve_named(&self.agents, name) else {
                        println!("{}", crate::agents::unknown_name_error(&self.agents, name));
                        return Ok(true);
                    };
                    def = Some(d);
                }
                let (instructions, spec, task, started) = match def {
                    Some(d) => (
                        Some(d.body.clone()),
                        d.engine.clone(),
                        arg.to_string(),
                        format!("[subagent started: {}]", d.name),
                    ),
                    None => (
                        None,
                        None,
                        arg.to_string(),
                        "[subagent started]".to_string(),
                    ),
                };
                if task.is_empty() {
                    println!("usage: /subagent[:<name>] <task>");
                } else {
                    // Same resolve the `agent` tool does, and for the same reason:
                    // a definition that names an engine must actually run on it
                    // here too, and a missing key must fail before the fork so the
                    // transcript is left exactly as it was.
                    let alt = match self.resolve_subagent_alt(spec) {
                        Ok(alt) => alt,
                        Err(e) => {
                            println!("/subagent: engine unavailable: {e}");
                            return Ok(true);
                        }
                    };
                    if let Some(note) = self.take_warm_note() {
                        println!("{}", self.debug_line(&note));
                    }
                    println!("{}", self.debug_line(&started));
                    let fork_at =
                        self.begin_subagent_fork(instructions.as_deref(), &task, alt.is_none());
                    // Restore the transcript even when the turn errored.
                    let turn = match alt {
                        None => self.run_turn(),
                        Some((key, engine)) => self.run_sidechain_on(key, engine, Self::run_turn),
                    };
                    let reported = self.finish_subagent_fork(fork_at, &task);
                    turn?;
                    if reported {
                        println!(
                            "{}",
                            self.debug_line("[subagent report added to the conversation]")
                        );
                        // The report is delegated work coming back, so the main
                        // loop runs on it — the same continuation the `agent`
                        // tool gets by returning its report as a tool result.
                        // Without this the report lands in the transcript and
                        // nothing acts on it until the user types again.
                        //
                        // The prompt here is the restored parent prefix plus the
                        // report, which is exactly what `restore_fork_kv` just
                        // set the KV up for: only the report re-prefills.
                        self.run_turn()?;
                    } else {
                        println!(
                            "{}",
                            self.debug_line("[subagent produced no report — nothing added]")
                        );
                    }
                }
            }
            _ if slash_command_known(cmd) => println!("{cmd}: not implemented yet"),
            _ => match self.slash_message(cmd, arg) {
                Some(Ok(message)) => {
                    print!("{}", status::format_user_prompt_echo(input, self.color));
                    self.session.push(Message::user(message));
                    self.run_turn()?;
                }
                Some(Err(e)) => println!("{e}"),
                None => println!("unknown command: {cmd}"),
            },
        }
        Ok(true)
    }

    /// Runs a `/btw` side question in the plain REPL: one generation pass
    /// over the shared context plus the framed question, tools denied,
    /// nothing pushed to the session. The next real turn's KV sync reuses
    /// the still-matching prefix and re-prefills past the divergence, so the
    /// side question rolls back automatically.
    /// Resolves a `/resume` argument: `Ok(None)` for an empty argument (show
    /// the picker), otherwise the loaded session — a small number picks from
    /// the recency-sorted listing, anything else is a sha prefix.
    fn resume_pick(&self, arg: &str) -> Result<Option<Session>, String> {
        let arg = arg.trim();
        if arg.is_empty() {
            return Ok(None);
        }
        if let Ok(n) = arg.parse::<usize>() {
            let entries = self.store.list().map_err(|e| e.to_string())?;
            let entry = entries
                .get(n.wrapping_sub(1))
                .ok_or_else(|| format!("no session number {n} (see /resume)"))?;
            return self
                .store
                .load(&entry.id)
                .map(Some)
                .map_err(|e| e.to_string());
        }
        self.store.load(arg).map(Some).map_err(|e| e.to_string())
    }

    /// Resumes a session named on the command line (`plank /resume [prefix]`)
    /// before the interactive loop starts. An empty `arg` resumes the most
    /// recent session; otherwise it is a number from the listing or a sha
    /// prefix. Only loads the session — each front-end renders the recovered
    /// history itself (see [`resumed_history`]), since the TUI's alternate
    /// screen would wipe anything printed here.
    fn resume_from_cli(&mut self, arg: &str) -> Result<(), String> {
        let session = if arg.trim().is_empty() {
            let entries = self.store.list().map_err(|e| e.to_string())?;
            let entry = entries
                .first()
                .ok_or_else(|| "no saved sessions to resume".to_string())?;
            self.store.load(&entry.id).map_err(|e| e.to_string())?
        } else {
            self.resume_pick(arg)?
                .ok_or_else(|| "no such session".to_string())?
        };
        // Restore the KV payload too, mirroring the `/resume` and `/switch`
        // slash commands. Loading only the transcript leaves the next turn to
        // re-prefill every token of it; the payload is exactly the cache that
        // avoids that. Best-effort — a stale or absent one just falls back to
        // prefill, which is the behavior this path had unconditionally.
        let note = self.load_session_payload(&session);
        self.session = session;
        self.last_ctx_used = 0;
        if let Some(note) = note {
            println!("{note}");
        }
        Ok(())
    }

    /// Recent-history text for a just-resumed session (empty when the current
    /// session was not loaded from disk), plus a `[resumed …]` trailer, for a
    /// front-end to display at startup.
    fn resumed_history(&self) -> Option<String> {
        use std::fmt::Write as _;
        if self.session.id.is_empty() {
            return None;
        }
        let mut out = crate::session::render_history(&self.session.transcript, 6, self.color);
        let short = crate::session::display_id(&self.session.id);
        let _ = write!(
            out,
            "{}",
            self.debug_line(&format!("[resumed session {short}]"))
        );
        Some(out)
    }

    /// Replays a just-resumed session's recent history into the TUI output log,
    /// rendering each message the way the live stream does: assistant text
    /// through the markdown renderer (with thinking dimmed and tool-call banners
    /// restored), user turns as prompt echoes, and tool results in gray. The
    /// plain REPL uses [`resumed_history`] instead; the TUI needs structured
    /// spans, not an ANSI string.
    fn replay_history_into_log(&self, log: &mut OutputLog) {
        use crate::session::Role;
        if self.session.id.is_empty() {
            return;
        }
        let transcript = &self.session.transcript;
        let Some((start, _tool_only)) =
            crate::session::history_window(transcript, HISTORY_DEFAULT_TURNS)
        else {
            return;
        };

        log.push_dim("--- session history ---");
        let show_tool_calls = crate::settings::active().ui.show_tool_calls;
        let show_thinking = crate::settings::active().ui.show_thinking;
        let thinking_tool_calls = crate::settings::active().engine.thinking_tool_calls;
        let tool_names = sysprompt::tool_names(&self.tool_ctx.mcp);
        let pre_open_think =
            !matches!(self.think, crate::engine::ThinkMode::Off) && !self.engine.wants_structured();

        for m in &transcript[start..] {
            match m.role {
                Role::User if m.is_tool_user() => {
                    log.push_dim("Tool result:");
                    for line in m.tool_result_payload().lines().take(12) {
                        log.push_dim(line.to_string());
                    }
                }
                Role::User => {
                    let text = m.text.trim();
                    if !text.is_empty() {
                        log.push_spans(tui::user_echo_spans(text));
                    }
                }
                Role::Assistant => {
                    let text = m.text.trim();
                    if text.is_empty() {
                        continue;
                    }
                    // Stream the stored text through the same renderer the live
                    // turn uses, so markdown, thinking gray, and tool-call
                    // banners come back exactly as they were shown.
                    let mut stream = StreamRenderer::new(std::mem::take(log));
                    stream.set_replay(true);
                    stream.set_show_tool_calls(show_tool_calls);
                    stream.set_show_thinking(show_thinking);
                    stream.set_thinking_tool_calls(thinking_tool_calls);
                    stream.set_tool_names(tool_names.clone());
                    if pre_open_think {
                        stream.begin_in_think();
                    }
                    stream.push(text);
                    stream.finish();
                    *log = stream.into_sink();
                    log.end_line();
                }
            }
        }

        let short = crate::session::display_id(&self.session.id);
        log.push_dim(format!("[resumed session {short}]"));
    }

    /// Captures a named checkpoint: the current transcript plus the engine KV
    /// snapshot (when the engine supports it). Returns a status line.
    fn checkpoint_create(&mut self, name: &str) -> String {
        let kv = self.engine.get_kv();
        let had_kv = kv.is_some();
        let replaced = self.checkpoints.save(name, &self.session, kv);
        let verb = if replaced { "updated" } else { "saved" };
        let note = if had_kv {
            " (with engine KV)"
        } else {
            " (transcript only)"
        };
        format!("checkpoint {verb}: {name}{note}")
    }

    /// Rolls back to a named checkpoint: the current tail is saved first as
    /// `pre-rollback` (so the rollback is undoable), then the transcript is
    /// restored verbatim and, when the checkpoint carries engine KV, the
    /// session KV is restored so the next turn skips re-prefill.
    fn rollback_to(&mut self, name: &str) -> Result<String, String> {
        let Some(cp) = self.checkpoints.get(name).cloned() else {
            return Err(format!("no checkpoint named {name} (see /checkpoint)"));
        };
        // Snapshot the current tail before discarding it.
        let tail_kv = self.engine.get_kv();
        self.checkpoints
            .save(PRE_ROLLBACK_CHECKPOINT, &self.session, tail_kv);
        crate::checkpoint::restore_transcript(&mut self.session, &cp);
        self.last_ctx_used = 0;
        let note = match &cp.kv {
            Some(cache) if self.engine.set_kv(cache).is_ok() => {
                " (engine KV restored, zero re-prefill)"
            }
            _ => " (transcript restored, re-prefill on next turn)",
        };
        Ok(format!(
            "rolled back to {name}{note}; tail saved as \"{PRE_ROLLBACK_CHECKPOINT}\""
        ))
    }

    /// Renders the session tree for `/tree` (issue #65).
    fn tree_view(&self, color: bool) -> String {
        crate::branch::render_tree(&self.session.tree(), color)
    }

    /// Forks the session at a previous user message (`/fork [n]`).
    ///
    /// `n` is a 1-based index into the fork points `/tree` lists (the real
    /// user prompts on the active branch); with no argument the tree view is
    /// returned instead, so the user can pick one. Forking rewinds the live
    /// transcript to just before that prompt and keeps everything after it as
    /// a sibling branch, so the next prompt explores a different path without
    /// losing the old one.
    ///
    /// The new transcript is a strict *prefix* of the old one, so the engine's
    /// token common-prefix probe still reuses the cached KV up to the fork
    /// point — no KV bytes are copied or restored here.
    fn fork_branch(&mut self, arg: &str, color: bool) -> Result<String, String> {
        let mut tree = self.session.tree();
        let points = tree.fork_points();
        if points.is_empty() {
            return Err("nothing to fork from yet; send a prompt first".to_owned());
        }
        if arg.is_empty() {
            return Ok(self.tree_view(color).trim_end().to_owned());
        }
        let n: usize = arg
            .parse()
            .ok()
            .filter(|n| (1..=points.len()).contains(n))
            .ok_or_else(|| {
                format!(
                    "usage: /fork <1..{}> (see /tree for the fork points)",
                    points.len()
                )
            })?;
        let before = self.session.transcript.len();
        tree.fork_at(points[n - 1])?;
        self.session.set_tree(&tree);
        self.last_ctx_used = 0;
        let kept = self.session.transcript.len();
        Ok(format!(
            "forked at fork point {n}: {kept} of {before} messages kept (cached prefix reused); \
the previous branch is still in /tree"
        ))
    }

    /// Duplicates the active branch (`/clone`).
    ///
    /// The copy becomes the live transcript and is byte-identical to what it
    /// was, so the engine's cached prefix stays valid in full; the original
    /// branch is frozen where it stands and remains visible in `/tree`.
    fn clone_branch(&mut self) -> Result<String, String> {
        let mut tree = self.session.tree();
        if tree.clone_active().is_none() {
            return Err("nothing to clone yet; send a prompt first".to_owned());
        }
        self.session.set_tree(&tree);
        let n = self.session.transcript.len();
        Ok(format!(
            "cloned the active branch ({n} messages, cached prefix reused in full); \
the original is frozen and listed in /tree"
        ))
    }

    /// Fingerprint tying a session's engine KV payload to this exact model,
    /// system prompt, and the session's rendered transcript — the repo's KV
    /// discipline rule: any drift makes the payload stale, and stale payloads
    /// are re-prefilled, never trusted.
    fn payload_fingerprint_for(&self, session: &Session) -> String {
        crate::session::payload_fingerprint(
            &self.engine.model_name(),
            &self.system,
            &render_transcript(session, &self.system),
            self.think,
            self.trusted_system_len,
        )
    }

    /// After a successful `/save`, snapshots the engine KV state to the
    /// session's payload sidecar. Returns a user-facing note, or `None` when
    /// the backend has no KV to persist (echo stub) — saving is best-effort
    /// and never fails the `/save` itself.
    ///
    /// The KV comes from the shared [`Engine::get_kv`] primitive; this layer
    /// only picks the [`KvKey::Session`] it is stored under — the file is named
    /// by session id but signed with the fingerprint, so a payload captured
    /// under another model or system prompt reads back as a miss.
    fn save_session_payload(&mut self) -> Option<String> {
        if self.session.id.is_empty() {
            return None;
        }
        // No KV support (echo stub) or nothing prefilled yet: nothing to save.
        let cache = self.engine.get_kv()?;
        let key = crate::session::KvKey::Session {
            id: self.session.id.clone(),
            fp: self.payload_fingerprint_for(&self.session),
        };
        match self.store.kv_store(&key, &cache) {
            Ok(()) => Some(format!(
                "saved KV payload ({:.2} MB)",
                crate::session::to_mb(self.store.payload_bytes(&self.session.id))
            )),
            Err(e) => Some(format!("KV payload save failed: {e}")),
        }
    }

    /// On `/switch` / `/resume`, tries to restore the session's KV payload so
    /// the next turn skips re-prefilling the transcript. Returns a note when
    /// there was a payload to consider; a stale, missing-fingerprint, or
    /// unloadable payload just falls back to re-prefill.
    ///
    /// The staleness gate is [`SessionStore::kv_load`], which only returns a
    /// cache when the stored signature equals the fingerprint; a matching cache
    /// is then fed back through the shared [`Engine::set_kv`] primitive
    /// (`SessionSnapshot::restore_bytes`, the non-owning path — see
    /// `FINDINGS.md` on the double-free).
    fn load_session_payload(&mut self, s: &Session) -> Option<String> {
        if s.id.is_empty() {
            return None;
        }
        if !self.store.payload_path(&s.id).exists() {
            return None;
        }
        let key = crate::session::KvKey::Session {
            id: s.id.clone(),
            fp: self.payload_fingerprint_for(s),
        };
        // kv_load returns None for a missing file too; the file exists here, so
        // None means stale/corrupt => re-prefill.
        let Some(cache) = self.store.kv_load(&key) else {
            return Some("KV payload is stale; the transcript will be re-prefilled".to_owned());
        };
        match self.engine.set_kv(&cache) {
            Ok(()) => {
                self.payload_restored = true;
                Some("restored KV payload; resume skips re-prefill".to_owned())
            }
            Err(e) => Some(format!(
                "KV payload load failed: {e}; the transcript will be re-prefilled"
            )),
        }
    }

    /// `/strip`: deletes the session's KV payload sidecar, keeping the
    /// transcript, and reports the transcript's token count — the prefill
    /// cost a later `/switch` pays to rebuild the KV — matching the C's
    /// `agent_worker_strip_session` report shape.
    fn strip_session(&mut self, prefix: &str) -> Result<(String, i32), String> {
        let (id, _had_payload) = self.store.strip(prefix).map_err(|e| e.to_string())?;
        let s = self.store.load(&id).map_err(|e| e.to_string())?;
        let tokens = self
            .engine
            .count_tokens(&render_transcript(&s, &self.system))
            .max(0);
        Ok((id, tokens))
    }

    /// Parses a `/notify [on|off]` argument and applies it; returns the
    /// status line to report to the user.
    fn notify_command(arg: &str) -> String {
        let new_state = match arg.trim() {
            "on" => true,
            "off" => false,
            "" => !crate::notify::enabled(),
            other => return format!("/notify: expected on|off, got `{other}`"),
        };
        crate::notify::set_enabled(new_state);
        format!("notifications {}", if new_state { "on" } else { "off" })
    }

    /// Parses a `/think [off|low|medium|max]` argument and applies it; returns the
    /// status line to report to the user. Shared by both front-ends so the two
    /// dispatchers cannot drift.
    ///
    /// With no argument it reports the current level. `max` is refused below
    /// [`THINK_MAX_MIN_CONTEXT`]: the preamble asks for a reasoning budget a
    /// smaller context is not meant to hold, and a refusal the user can act on
    /// (raise `--ctx-size`) beats the C's silent downgrade to `medium`.
    ///
    /// [`THINK_MAX_MIN_CONTEXT`]: crate::engine::THINK_MAX_MIN_CONTEXT
    fn think_command(&mut self, arg: &str) -> String {
        use crate::engine::{THINK_MAX_MIN_CONTEXT, ThinkMode};

        let current = self.think;
        let arg = arg.trim();
        if arg.is_empty() {
            return format!("thinking: {} (off|low|medium|max)", current.name());
        }
        let Some(level) = ThinkMode::parse(arg) else {
            return format!("/think: expected off|low|medium|max, got `{arg}`");
        };
        let ctx = self.engine.ctx_size();
        if level == ThinkMode::Max && ctx < THINK_MAX_MIN_CONTEXT {
            return format!(
                "/think max needs a context of at least {THINK_MAX_MIN_CONTEXT} tokens \
                 (this session has {ctx}; restart with --ctx {THINK_MAX_MIN_CONTEXT}); \
                 still {}",
                current.name()
            );
        }
        if level == current {
            return format!("thinking already {}", level.name());
        }
        self.think = level;
        // A change of effort preamble changes the prompt prefix, so the engine
        // drops its cached tokens and KV here. Re-warm from the tier
        // checkpoints under the new fingerprint rather than making the next
        // turn re-prefill the system prompt inline.
        let prefix_changed = current.effort_prefix() != level.effort_prefix();
        self.engine.set_think_mode(level);
        // Cached alt engines too: `self.think` keys their Tier 1 checkpoint and
        // frames their sidechains, so an engine left at the old level would
        // build its tokens at one level while being keyed at another — the one
        // disagreement a fingerprint cannot catch, because it is between the key
        // and the tokens rather than between two keys. Idempotent when the level
        // is unchanged, so this costs nothing for engines already at `level`.
        for engine in self.alt_engines.values_mut() {
            engine.set_think_mode(level);
        }
        // Only the live engine is re-warmed: an alt engine's own first take
        // restores its Tier 1 checkpoint under the new fingerprint.
        if prefix_changed {
            self.local_alt_warmed = false;
            self.rewarm_after_reset(&mut || {});
        }
        format!("thinking {}", level.name())
    }

    /// Sets (or with `-` clears) the session tag, re-saving immediately when
    /// the session was already saved so listings pick it up.
    fn set_tag(&mut self, arg: &str) -> Result<String, String> {
        let tag = if arg == "-" { "" } else { arg.trim() };
        tag.clone_into(&mut self.session.tag);
        self.session.dirty = true;
        let mut msg = if tag.is_empty() {
            "tag cleared".to_string()
        } else {
            format!("tag set: {tag}")
        };
        if !self.session.id.is_empty() {
            self.save_session().map_err(|e| e.to_string())?;
            msg.push_str(" (saved)");
        }
        Ok(msg)
    }

    /// Runs `/insights` end to end and returns the report path plus the
    /// summary lines to show. `note` receives progress as it goes; a first
    /// run over a long history parses every session, and the model calls that
    /// follow are not fast either.
    ///
    /// `note` receives status lines; `tick` receives the model's reasoning as
    /// it streams, so a section being written looks like work rather than a
    /// hang. Each front-end decides how to show them.
    ///
    /// `arg` accepts `fast`, which skips the narrative and reports only the
    /// statistics.
    ///
    /// The two halves are deliberately unequal in status: the statistics are
    /// the report, and every model call is allowed to fail without taking
    /// anything else down with it.
    fn run_insights(
        &mut self,
        arg: &str,
        note: &mut dyn FnMut(String),
        tick: &mut dyn FnMut(String),
    ) -> Result<Insights, String> {
        use crate::insights;

        let fast = matches!(arg.trim(), "fast" | "--fast" | "quick");
        let root = insights::usage_dir();
        let tz = insights::local_utc_offset();

        // The report reads back the user's whole history and can take minutes,
        // so the window says so. Restored on every exit path, including the
        // `?` below — a title left saying "introspecting" after the command
        // failed would outlive the thing it describes.
        crate::title::set(crate::title::State::Introspecting);
        let _title = TitleRestore;

        let mut last_pct = usize::MAX;
        let scan = insights::collect_metas(
            &self.store,
            &root,
            &mut |done, total| {
                if total == 0 {
                    return;
                }
                let pct = done * 100 / total;
                // One line per decile, not one per session: a 200-session first
                // run should report progress, not scroll.
                if pct / 10 != last_pct / 10 {
                    last_pct = pct;
                    // Transient: the scan is over in seconds and its deciles are
                    // not worth keeping in the log next to the report.
                    tick(format!("reading sessions… {done}/{total}"));
                }
            },
            &|| crate::interrupt::pending(),
        )?;
        let insights::Scan::Done(metas) = scan else {
            // Stopped during the scan: nothing has been computed worth
            // showing, and the half-filled cache makes the next run shorter.
            crate::interrupt::clear();
            return Ok(Insights::Cancelled);
        };
        let agg = insights::aggregate(&metas, tz);

        let mut narrative = insights::Narrative::new();
        if agg.sessions_counted == 0 {
            note("[no sessions substantial enough to report on]".to_owned());
        } else if fast {
            note("[skipping the written sections (fast)]".to_owned());
        } else if !self.engine.supports_aside() {
            // An aside answers a question against a scratch copy of the KV
            // state; without one, the only way to ask is to overwrite the
            // live conversation's cache, which a slash command must never do.
            note("[this engine cannot answer asides; statistics only]".to_owned());
        } else {
            let context = insights::narrative_context(&agg);
            for spec in insights::SECTIONS {
                if crate::interrupt::pending() {
                    note("[interrupted; writing what is ready]".to_owned());
                    break;
                }
                note(format!("[writing “{}”…]", spec.heading));
                // Deliberately *not* the agent system prompt. That prompt
                // exists to make the model reach for tools and reason at
                // length about a codebase; under it, a request for a bare
                // JSON object comes back as thinking and tool calls. A
                // section is a writing task, so it gets a writing task's
                // framing and nothing else — which also makes the prefill
                // small, since there is no agent prompt to lay down.
                let prompt = format!(
                    "[system]\n{}\n[user]\n{}\n",
                    insights::ANALYST_SYSTEM,
                    insights::section_prompt(spec, &context)
                );
                let mut reply = String::new();
                // Stream the reasoning as it arrives, the way an ordinary
                // turn does, so the wait is legible rather than blank.
                let mut ticker = insights::ThinkTicker::new();
                // A section is a paragraph or a short list, and the section
                // budget is what stops one wandering reply from making the
                // whole report take minutes. The session's own generation
                // limit is meant for a coding turn and is far too generous
                // here.
                let opts = crate::engine::GenerationOptions {
                    n_predict: INSIGHTS_SECTION_TOKENS,
                    // Thinking stays on so there is something to show while
                    // the section is written: a couple of silent minutes per
                    // section reads as a hang. The budget below covers the
                    // reasoning and the answer together, and `extract_json`
                    // takes the answer from after the think block.
                    think_mode: crate::engine::ThinkMode::Medium,
                    ..self.cfg.generation.clone()
                };
                let stop = AtomicBool::new(false);
                let generated = self.generate_aside_best(
                    &prompt,
                    &opts,
                    &|| stop.load(Ordering::Relaxed) || crate::interrupt::pending(),
                    &mut |ev| {
                        if let crate::engine::EngineEvent::Text(t) = ev {
                            reply.push_str(&t);
                            for line in ticker.feed(&t) {
                                tick(line);
                            }
                            // `tick` is where the TUI reads the keyboard, so
                            // polling here is what turns an Esc pressed
                            // mid-sentence into a stopped generation rather
                            // than one noticed at the next section boundary.
                            if crate::interrupt::pending() {
                                stop.store(true, Ordering::Relaxed);
                            }
                        }
                    },
                );
                // A section the user stopped is not a section that failed:
                // say nothing about it and leave the loop, rather than
                // reporting it "unavailable" and trying the next one.
                if generated.as_ref().is_ok_and(|s| s.interrupted) {
                    note("[stopped; writing the report as it stands]".to_owned());
                    break;
                }
                match generated.map_err(|e| e.to_string()).and_then(|_| {
                    insights::extract_json(&reply).ok_or_else(|| "no JSON in reply".to_owned())
                }) {
                    Ok(value) => {
                        narrative.insert(spec.key.to_owned(), value);
                    }
                    // A section that fails is a section the report goes
                    // without; it never costs the statistics.
                    Err(e) => note(format!("[“{}” unavailable: {e}]", spec.heading)),
                }
            }
            crate::interrupt::clear();
        }

        // The statistics are finished by this point even if the prose was
        // interrupted, so the report is still written: throwing away a
        // completed scan because the user stopped the writing would punish
        // them for the part that was slow. The interrupt flag is therefore
        // cleared first — the stop has been honoured, and it must not now be
        // read as "abandon the render too".
        crate::interrupt::clear();
        tick("writing the report…".to_owned());
        let Some(html) = insights::render_html_cancellable(&agg, &narrative, tz, &|| {
            crate::interrupt::pending()
        }) else {
            // Stopped mid-render: nothing is written, so the report the user
            // already had is still on disk, whole.
            crate::interrupt::clear();
            return Ok(Insights::Cancelled);
        };
        let path = insights::write_report(&root, &html)?;
        Ok(Insights::Done {
            path,
            summary: insights::render_summary(&agg, &narrative, tz),
        })
    }

    /// Saves the live session, recording the project directory it ran in.
    ///
    /// The transcript never states which project produced it, so `/insights`
    /// gets the path from here — stamped at every save rather than at session
    /// creation, because a session outlives the several places that replace
    /// it (`/new`, `/resume`, `/clone`).
    fn save_session(&mut self) -> crate::session::Result<String> {
        self.session.cwd = self.tool_ctx.cwd.to_string_lossy().into_owned();
        self.store.save(&mut self.session)
    }

    /// Saves the session at exit and returns `(id, path)` so the caller can
    /// tell the user how to resume it. Returns `None` when there is nothing
    /// worth saving (no user turn) or the save fails.
    fn save_for_exit(&mut self) -> Option<(String, std::path::PathBuf)> {
        // No activity since the session was started or loaded — nothing worth
        // persisting. This skips both a fresh session with no turns and a
        // resumed one exited without any new exchange (which would otherwise
        // be re-written, bumping its timestamp for nothing). `dirty` is set by
        // every transcript push, task update, and tag, and cleared on save and
        // load.
        if !self.session.dirty {
            return None;
        }
        let id = self.save_session().ok()?;
        // Snapshot the KV alongside the transcript. Without this an exit-saved
        // session is transcript-only, so `plank /resume` has nothing to restore
        // and the next turn re-prefills the whole conversation — minutes of it
        // at local prefill speeds. `/save` has always captured the payload; the
        // exit path is where sessions actually get saved.
        let _ = self.save_session_payload();
        let path = self
            .store
            .find(&id)
            .map_or_else(|_| self.store.dir().join(format!("{id}.kv")), |(_, p)| p);
        Some((id, path))
    }

    /// At session end, saves the transcript and prints where it landed and how
    /// to resume it. A session with no activity this run (nothing pushed since
    /// it was started or loaded) is silently skipped.
    fn report_session_on_exit(&mut self) {
        let Some((id, path)) = self.save_for_exit() else {
            return;
        };
        let short = crate::session::display_id(&id);
        let (bold, dim, reset) = if self.color {
            ("\x1b[1m", "\x1b[38;5;238m", ANSI_RESET)
        } else {
            ("", "", "")
        };
        println!();
        println!("{bold}Session saved{reset} {dim}{}{reset}", path.display());
        println!("Resume it later with:  {bold}plank /resume {short}{reset}");
    }

    /// Prints the run's stats at exit: total tokens ingested and generated
    /// across every turn (both directions), and the wall-clock duration of the
    /// whole run. Silent when nothing was generated, so an idle run stays
    /// quiet. Independent of the session save, so it reports even when the
    /// final session was empty (e.g. after `/clear`).
    fn report_run_stats(&self) {
        let s = &self.stats;
        if s.input_tokens == 0 && s.output_tokens == 0 {
            return;
        }
        let (bold, dim, reset) = if self.color {
            ("\x1b[1m", "\x1b[38;5;238m", ANSI_RESET)
        } else {
            ("", "", "")
        };
        let elapsed = fmt_duration(self.session_start.elapsed());
        println!();
        println!(
            "{bold}Session stats{reset}  ↓ {} ↑ {}  {dim}·{reset}  {elapsed}",
            fmt_u64(s.input_tokens),
            fmt_u64(s.output_tokens),
        );
        self.report_peak_speeds(bold, dim, reset);
        // Only when more than one engine served: with a single one the rows
        // would just repeat the totals a line lower.
        if s.by_engine.len() < 2 {
            return;
        }
        let width = s.by_engine.iter().map(|r| r.0.chars().count()).max();
        for (label, input, output) in &s.by_engine {
            println!(
                "  {dim}{label:<w$}{reset}  ↓ {} ↑ {}",
                fmt_u64(*input),
                fmt_u64(*output),
                w = width.unwrap_or(0),
            );
        }
    }

    /// Prints the model's best-ever prefill and generation rates, folding this
    /// session's peaks into the stored record on the way out.
    ///
    /// Silent for engines that never reported a rate — the echo stub, and
    /// online providers, whose throughput is someone else's network — so a
    /// provider-only session's exit message is unchanged.
    fn report_peak_speeds(&self, bold: &str, dim: &str, reset: &str) {
        let model = self.engine.model_name();
        let Some(c) = crate::speeds::commit(&model) else {
            return;
        };
        let mut parts: Vec<String> = Vec::new();
        if c.best.prefill_tps > 0.0 {
            parts.push(format!(
                "prefill {}{:.1}{reset} tok/s{}",
                bold,
                c.best.prefill_tps,
                if c.new_prefill { " (new best)" } else { "" },
            ));
        }
        if c.best.gen_tps > 0.0 {
            parts.push(format!(
                "generation {}{:.1}{reset} tok/s{}",
                bold,
                c.best.gen_tps,
                if c.new_gen { " (new best)" } else { "" },
            ));
        }
        if parts.is_empty() {
            return;
        }
        println!(
            "{dim}peak{reset} {model}  {}",
            parts.join(&format!("  {dim}·{reset}  ")),
        );
    }

    /// Writes a `/repro` diagnostic dump — the exact rendered engine input
    /// plus the runtime knobs that shape generation — to `~/.plank/repro/`.
    /// `note` is an optional free-text description of the bug. Read-only: the
    /// live session is untouched.
    fn write_repro(&self, note: &str) -> Result<std::path::PathBuf, String> {
        let rendered = render_transcript(&self.session, &self.system);
        let version = crate::logo::version_label();
        let date = crate::context::current_local_iso_date();
        let meta = crate::repro::Meta {
            version: &version,
            date: &date,
            ctx_size: self.engine.ctx_size(),
            transcript_tokens: self.engine.count_tokens(&rendered),
            last_ctx_used: self.last_ctx_used,
            power_percent: self.power_percent,
            think: self.think,
            session_id: &self.session.id,
            session_tag: &self.session.tag,
            note: note.trim(),
        };
        let report = crate::repro::build_report(&meta, self.cfg, &rendered);
        crate::repro::save(&self.tool_ctx.cwd, now_secs(), &report)
    }

    /// Writes a `/export` transcript dump (issue #66). `arg` is
    /// `[md|html] [path]`: the format defaults to markdown and the path to an
    /// auto-named file in the working directory. Read-only: the live session
    /// is untouched.
    fn write_export(&self, arg: &str) -> Result<std::path::PathBuf, String> {
        let mut parts = arg.trim().splitn(2, char::is_whitespace);
        let first = parts.next().unwrap_or("").trim();
        let rest = parts.next().unwrap_or("").trim();
        let (format, path_arg) = match crate::export::parse_format(first) {
            Some(f) => (f, rest),
            None => (crate::export::Format::Markdown, arg.trim()),
        };

        let title = if self.session.title.trim().is_empty() {
            crate::session::title_from_transcript(&self.session.transcript, 60)
        } else {
            self.session.title.clone()
        };
        let body = crate::export::render(&self.session.transcript, &title, format);

        let auto = || crate::export::default_filename(&title, now_secs(), format);
        let mut path = if path_arg.is_empty() {
            self.tool_ctx.cwd.join(auto())
        } else {
            let given = std::path::PathBuf::from(path_arg);
            if given.is_absolute() {
                given
            } else {
                self.tool_ctx.cwd.join(given)
            }
        };
        if path.is_dir() {
            path = path.join(auto());
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        std::fs::write(&path, body).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(path)
    }

    /// Starts a `/subagent` fork: appends the framed task to the live
    /// transcript and returns the pre-fork length for later truncation. The
    /// fork inherits the parent transcript prefix, so the engine's per-turn
    /// sync reuses the parent KV cache.
    ///
    /// A sidechain on an alternate engine passes `snapshot_kv` false: the parent
    /// engine is never called, so there is no divergence to roll back. It still
    /// pushes a `None` rather than skipping the push — `restore_fork_kv` pops
    /// unconditionally, so skipping would unbalance the stack and a nested fork
    /// would pop the *parent's* snapshot. `None` already means "nothing to
    /// restore", so the stack stays LIFO-correct.
    fn begin_subagent_fork(
        &mut self,
        instructions: Option<&str>,
        task: &str,
        snapshot_kv: bool,
    ) -> usize {
        let fork_at = self.session.transcript.len();
        // Capture the live KV before the sidechain diverges it; the matching
        // restore is `restore_fork_kv`, called by every fork-end path. `None`
        // on engines without snapshot support — the restore then no-ops and
        // the next turn re-prefills as before this guard existed.
        self.fork_kv.push(if snapshot_kv {
            self.engine.get_kv()
        } else {
            None
        });
        self.session.push(Message::user(crate::agents::task_message(
            instructions,
            task,
        )));
        fork_at
    }

    /// Runs a whole block of remote-backed `agent` calls concurrently, or returns
    /// `None` when the block is not eligible and the caller should use the serial
    /// path.
    ///
    /// Eligible only when *every* call in the block is an `agent` call naming a
    /// definition with its own engine whose capability exceeds 1, there are at
    /// least two, and the effective width exceeds 1. Deliberately conservative:
    /// if a `bash` call sat between two `agent` calls, running them concurrently
    /// would reorder side effects relative to it, so a mixed block stays serial.
    fn run_agent_fanout(&mut self, calls: &[ToolCall]) -> Option<Vec<(String, String)>> {
        // Read the budget here: `settings::install_for_test` is thread-local, so
        // reading it inside a spawned pass would silently see defaults.
        let budget = crate::settings::active().agents.max_parallel;
        if calls.len() < 2 || budget < 2 || !calls.iter().all(|c| c.name == "agent") {
            return None;
        }
        if self.tool_ctx.subagent_depth >= crate::tools::SUBAGENT_DEPTH_CAP {
            return None;
        }
        // Resolve every call before touching the cache, so an ineligible block
        // leaves no engine removed and no transcript disturbed.
        let mut specs = Vec::with_capacity(calls.len());
        for call in calls {
            let task = call
                .arg_value("task")
                .or_else(|| call.arg_value("prompt"))?
                .trim();
            if task.is_empty() {
                return None;
            }
            let name = call.arg_value("name").unwrap_or("").trim();
            let def = self.agents.iter().find(|d| d.name == name && d.auto)?;
            let spec = self.resolve_alt_spec(def.engine.clone())?;
            specs.push((task.to_owned(), def.body.clone(), spec, def.name.clone()));
        }
        // Now take the engines. Any that turns out serial-only sends the whole
        // block back to the serial path with every engine returned.
        let mut slots: Vec<FanoutSlot> = Vec::with_capacity(specs.len());
        for (task, body, spec, label) in &specs {
            match self.take_alt_engine(spec) {
                Ok((key, engine)) if engine.max_parallel() > 1 => {
                    let mut session = Session::new();
                    session.push(Message::user(crate::agents::task_message(
                        Some(body.as_str()),
                        task,
                    )));
                    slots.push(FanoutSlot {
                        key,
                        engine,
                        session,
                        label: label.clone(),
                        output: String::new(),
                        pending_calls: Vec::new(),
                        done: false,
                        error: None,
                    });
                }
                Ok((key, engine)) => {
                    self.alt_engines.insert(key, engine);
                    self.return_slot_engines(slots);
                    return None;
                }
                Err(_) => {
                    self.return_slot_engines(slots);
                    return None;
                }
            }
        }
        let cap = slots
            .iter()
            .map(|s| s.engine.max_parallel())
            .min()
            .unwrap_or(1);
        let width = budget.min(cap);
        if width < 2 {
            self.return_slot_engines(slots);
            return None;
        }

        let labels: Vec<&str> = slots.iter().map(|s| s.label.as_str()).collect();
        self.emit_sub(crate::worker::UiEvent::Dim(crate::tui::subagents_signpost(
            &labels,
        )));
        self.tool_ctx.subagent_depth += 1;
        self.run_fanout_rounds(&mut slots, width);
        self.tool_ctx.subagent_depth -= 1;

        let results = slots
            .iter()
            .map(|s| {
                let out = match (&s.error, last_assistant_text(&s.session.transcript)) {
                    (Some(e), _) => format!("Tool error: sub-agent failed: {e}\n"),
                    (None, Some(r)) => format!("Sub-agent report:\n{r}\n"),
                    (None, None) => "Tool error: sub-agent produced no report\n".to_string(),
                };
                ("agent".to_string(), out)
            })
            .collect();
        self.flush_fanout_panes(&slots);
        self.return_slot_engines(slots);
        Some(results)
    }

    /// The lockstep rounds of a fan-out: concurrent generation, then serial
    /// dispatch.
    ///
    /// Each round runs every live slot's generation in a `std::thread::scope`
    /// (at most `width` at a time), then dispatches all resulting tool calls on
    /// *this* thread. So [`ToolContext`] — MCP clients, async bash jobs, edit
    /// previews, consent state, the plan-mode gate — is only ever touched from
    /// the main thread: no lock, and no two sub-agents mid-edit on the same file.
    ///
    /// The cost is a barrier per round: a fast slot waits for the slowest before
    /// its next generation. Accepted, because the win is still roughly N× on the
    /// network-bound part, which is essentially all of a remote sidechain's cost.
    fn run_fanout_rounds(&mut self, slots: &mut [FanoutSlot], width: usize) {
        const MAX_ROUNDS: usize = 40;
        let system = self.system.clone();
        let opts = self.cfg.generation.clone();
        let ctx = PassCtx {
            opts: &opts,
            think_off: matches!(self.think, crate::engine::ThinkMode::Off),
            thinking_tool_calls: crate::settings::active().engine.thinking_tool_calls,
            tool_names: sysprompt::tool_names(&self.tool_ctx.mcp),
        };
        let cwd = self.tool_ctx.cwd.clone();
        for round in 0..MAX_ROUNDS {
            if slots.iter().all(|s| s.done) || crate::interrupt::pending() {
                break;
            }
            // On the last permitted round ask each live slot to report now, and
            // treat its text as the answer whatever it asks for. Letting the
            // budget simply run out would discard all its work.
            let last_round = round + 1 == MAX_ROUNDS;
            if last_round {
                for slot in slots.iter_mut().filter(|s| !s.done) {
                    slot.session
                        .push(Message::user(crate::agents::final_round_reminder()));
                }
            }
            // Phase 1, main thread: render each live slot's prompt and build its
            // structured buffers from its *own* session.
            let prepared: Vec<Option<(String, Option<StructuredBufs>)>> = slots
                .iter()
                .map(|slot| {
                    if slot.done {
                        return None;
                    }
                    let prompt = render_transcript(&slot.session, &system);
                    let bufs = slot
                        .engine
                        .wants_structured()
                        .then(|| self.build_structured_for(&slot.session, &prompt));
                    Some((prompt, bufs))
                })
                .collect();

            // Phase 2, `width` threads at a time: generate. Only an engine and
            // plain data cross the boundary.
            let passes = generate_fanout_round(slots, &prepared, width, &ctx, &cwd);

            // Phase 3, main thread only: fold results in, then dispatch tools.
            for (slot, pass) in slots.iter_mut().zip(passes) {
                let Some(pass) = pass else { continue };
                match pass {
                    Err(e) => {
                        slot.error = Some(e);
                        slot.done = true;
                    }
                    Ok(pass) => {
                        // A fan-out slot runs on its own engine, so its tokens
                        // belong in that engine's row rather than the main
                        // agent's — and they were never counted at all before
                        // the breakdown made their absence visible. The usage
                        // block is the only valid source here: the local
                        // ctx-delta estimate is keyed to `self.last_ctx_used`,
                        // which describes the main engine's context, not this
                        // slot's. Fan-out is provider-only by construction, so
                        // there is nothing to fall back to.
                        if let Some(u) = pass.stats.usage {
                            self.stats.add(
                                &engine_stats_label(&*slot.engine),
                                u64::try_from(
                                    i64::from(u.input_tokens)
                                        + i64::from(u.cache_read_tokens)
                                        + i64::from(u.cache_write_tokens),
                                )
                                .unwrap_or(0),
                                u64::try_from(u.output_tokens).unwrap_or(0),
                            );
                            self.usage.total.add(u);
                            self.usage.turns += 1;
                        }
                        slot.session.push(Message::assistant(pass.assistant_text));
                        if last_round {
                            slot.done = true;
                        } else if let Some(payload) = pass.tool_error {
                            slot.session.push(Message::user(format!(
                                "<tool_result>{payload}</tool_result>"
                            )));
                        } else if pass.calls.is_empty() {
                            slot.done = true;
                        } else {
                            slot.pending_calls = pass.calls;
                        }
                    }
                }
            }
            // Collect the work first so no slot borrow is held across
            // `run_tool_calls`, which needs `&mut self`.
            let pending: Vec<(usize, Vec<ToolCall>)> = slots
                .iter_mut()
                .enumerate()
                .filter_map(|(i, s)| {
                    let calls = std::mem::take(&mut s.pending_calls);
                    (!calls.is_empty()).then_some((i, calls))
                })
                .collect();
            for (i, calls) in pending {
                let observations = self.run_tool_calls(&calls);
                self.sync_tasks_after_dispatch();
                // The sidechain has no UI to drain these into.
                self.tool_ctx.edit_previews.clear();
                self.tool_ctx.task_completions.clear();
                self.tool_ctx.hook_warnings.clear();
                slots[i].session.push(Message::user(format!(
                    "<tool_result>{observations}</tool_result>"
                )));
            }
        }
        // Only an interrupt leaves a slot unfinished. Its text, if any, still
        // becomes its report — nothing is invented here.
        for slot in slots.iter_mut() {
            slot.done = true;
        }
    }

    /// Returns every slot's engine to the cache. Called on all exit paths,
    /// including the ineligible-block early returns.
    fn return_slot_engines(&mut self, slots: Vec<FanoutSlot>) {
        for slot in slots {
            self.alt_engines.insert(slot.key, slot.engine);
        }
    }

    /// Appends each slot's buffered output to the sub-agent pane as one labelled
    /// block, in call order.
    ///
    /// The pane holds a single label and a single log, so N sidechains streaming
    /// live would interleave into unreadable output. Fan-out therefore buffers
    /// and flushes; the serial path still streams live.
    fn flush_fanout_panes(&mut self, slots: &[FanoutSlot]) {
        for slot in slots {
            self.emit_sub(crate::worker::UiEvent::SubStart(slot.label.clone()));
            if !slot.output.trim().is_empty() {
                self.emit_sub(crate::worker::UiEvent::Sub(Box::new(
                    crate::worker::UiEvent::Visible(slot.output.clone()),
                )));
            }
            self.emit_sub(crate::worker::UiEvent::SubEnd);
        }
    }

    /// Resolves `spec` to an engine, **removing** it from the cache so the
    /// caller owns it for the sidechain's duration.
    ///
    /// The API key is read first, so a definition whose variable has been unset
    /// mid-session fails even on a cache hit — the key is part of the engine's
    /// identity, not a one-time construction detail. Builds and probes only on a
    /// miss, so the probe costs at most one request per key per session.
    ///
    /// # Errors
    /// When the key variable is unset or empty, or the engine cannot be built.
    /// What a definition's engine override resolves to *in this session*.
    ///
    /// `provider: local` under a local main agent is not an override at all —
    /// the parent already *is* the local engine, so running on it is both what
    /// the definition asked for and one fewer engine to hold. It only becomes a
    /// real override when the main agent is a provider.
    fn resolve_alt_spec(
        &self,
        spec: Option<crate::agents::AgentEngine>,
    ) -> Option<crate::agents::AgentEngine> {
        match spec {
            Some(crate::agents::AgentEngine::Local) if self.cfg.provider.is_none() => None,
            other => other,
        }
    }

    /// The engine a `/subagent` definition asks for, already taken from the
    /// cache and ready to run on — or `None` when it runs on the parent's.
    ///
    /// Both `/subagent` front ends go through this, so neither can quietly drop
    /// the engine override the way they both used to.
    ///
    /// # Errors
    /// When the definition names an engine this session cannot provide (an unset
    /// key, or no local engine loaded at startup).
    fn resolve_subagent_alt(
        &mut self,
        spec: Option<crate::agents::AgentEngine>,
    ) -> Result<Option<AltEngine>, String> {
        match self.resolve_alt_spec(spec) {
            None => Ok(None),
            Some(spec) => self.take_alt_engine(&spec).map(Some),
        }
    }

    /// Puts the system prompt into the alt local engine's KV before its first
    /// sidechain — by **restoring** the Tier 1 checkpoint when one exists, which
    /// is a disk read rather than a prefill of the whole system prompt.
    ///
    /// Only the System tier: a sidechain is clean-room (see
    /// [`run_sidechain_on`](Self::run_sidechain_on)), so its prompt is the
    /// system prompt plus the framed task with none of the project/session
    /// context tiers in between. Restoring a deeper tier would seed the KV with
    /// tokens the sidechain's prompt does not contain, and the sync would have
    /// to walk back out of them.
    ///
    /// Restore only, never prefill: on a miss the engine is left cold and the
    /// sidechain's own pass prefills it, with the progress bar and the interrupt
    /// that a pass has and this call does not. Prefilling here froze the front
    /// end for the length of a cold system prompt — `warm_sync` cannot be
    /// interrupted, and on the TUI `/subagent` path this runs on the thread that
    /// draws (see [`kvtier::restore`](crate::kvtier::restore)).
    ///
    /// The consequence is that nothing writes a Tier 1 checkpoint for the local
    /// engine when the main agent is a provider — a provider never warms — so
    /// the hit depends on an ordinary local-main session having written one.
    /// That is the common case, and paying a *visible* prefill when it has not
    /// is strictly better than the freeze.
    /// The outcome is stashed in [`warm_note`](Self::warm_note) rather than
    /// printed here: this runs on the UI thread for `/subagent` and on the
    /// worker thread for the `agent` tool, and only the caller knows which sink
    /// its front end is holding. Always noted, hit or miss — a silent hit is
    /// indistinguishable from a silent miss, which is the position this landed
    /// in twice.
    fn warm_alt_local(&mut self, engine: &mut dyn Engine) {
        let model = engine.model_name();
        let tiers = self.kv_tiers_for(&model);
        let Some(system) = tiers
            .first()
            .filter(|t| t.kind == crate::kvtier::TierKind::System)
        else {
            return;
        };
        let outcome = crate::kvtier::restore(engine, Some(&self.store), system);
        // Enough to act on without a debugger: which engine, which fingerprint,
        // what happened, and the exact path that was or was not there. The
        // fingerprint is the part that usually explains a miss — it covers the
        // system prompt, and the sub-agent roster is *in* the system prompt, so
        // a project with its own `.plank/agents` keys differently from the same
        // model anywhere else.
        let fp: String = system.fingerprint.chars().take(12).collect();
        let path = self.store.kv_path(system.key.as_ref().unwrap_or(
            &crate::session::KvKey::System {
                fp: system.fingerprint.clone(),
            },
        ));
        self.warm_note = Some(match outcome {
            Ok(r) => format!(
                "[{model} sub-agent KV: {} — tier1 {fp} · {}]",
                r.reason(),
                path.display()
            ),
            Err(e) => format!("[{model} sub-agent KV: warm failed: {e} — tier1 {fp}]"),
        });
    }

    /// Takes the pending alt-engine warm diagnostic, if any, for a front end to
    /// render through whichever sink it owns.
    fn take_warm_note(&mut self) -> Option<String> {
        self.warm_note.take()
    }

    fn take_alt_engine(&mut self, spec: &crate::agents::AgentEngine) -> Result<AltEngine, String> {
        use crate::remote::provider::ProviderEngine;
        let spec = match spec {
            // The local engine is loaded at startup, never built here: it needs
            // ~82 GB and a model file, and a mid-turn failure for either would
            // be far worse than refusing before the prompt. If it is absent,
            // this session was started without one.
            crate::agents::AgentEngine::Local => {
                let mut engine = self.alt_engines.remove(&EngineKey::Local).ok_or_else(|| {
                    "no local engine in this session (a `provider: local` definition needs one \
                     at startup; check the roster with /agent)"
                        .to_owned()
                })?;
                if !self.local_alt_warmed {
                    self.local_alt_warmed = true;
                    self.warm_alt_local(&mut *engine);
                }
                return Ok((EngineKey::Local, engine));
            }
            crate::agents::AgentEngine::Provider(p) => p,
        };
        let api_key = std::env::var(&spec.api_key_env)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .ok_or_else(|| format!("{} is not set", spec.api_key_env))?;
        let base_url = spec
            .base_url
            .clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| spec.kind.default_base_url().to_string())
            .trim_end_matches('/')
            .to_string();
        // A definition that states its window is believed. Otherwise ask the
        // provider once: the local default is sized for the ds4 model and says
        // nothing about a provider's, and the parent's window is the last
        // resort rather than a guess dressed up as an answer.
        let ctx = match spec.ctx {
            Some(c) => c,
            None => ProviderEngine::discover_ctx_size(
                spec.kind,
                Some(base_url.as_str()),
                &api_key,
                &spec.model,
            )
            .unwrap_or_else(|| self.engine.ctx_size()),
        };
        let key = EngineKey::Provider(
            spec.kind,
            base_url.clone(),
            spec.model.clone(),
            ctx,
            spec.api_key_env.clone(),
        );
        if let Some(engine) = self.alt_engines.remove(&key) {
            return Ok((key, engine));
        }
        // A definition's host joins the footer's origin list the first time it
        // is actually built, so the bar names every place inference is running
        // this session rather than only where the main agent runs.
        crate::status::set_engine_origin(&crate::status::url_host(&base_url));
        let engine = ProviderEngine::new(
            spec.kind,
            Some(base_url),
            api_key,
            spec.model.clone(),
            ctx,
            true,
        )
        .map_err(|e| e.to_string())?;
        Ok((key, Box::new(engine)))
    }

    /// Runs a sub-agent sidechain on `engine` instead of the parent's, returning
    /// the engine to the cache on **every** exit path.
    ///
    /// `run` drives the rounds: the `agent` tool passes
    /// [`run_subagent_loop`](Self::run_subagent_loop), while `/subagent` passes
    /// its front end's ordinary turn so the work still renders live. The engine
    /// swap is the same either way — which is the point of taking it as a
    /// parameter rather than hardcoding one loop.
    ///
    /// No `fork_kv` snapshot is meaningful here: the parent engine is never
    /// called during the sidechain, so its KV cannot be dirtied. The sidechain
    /// always runs clean-room — the parent transcript is stashed and only the
    /// framed task is visible — so no parent context is sent to the provider and
    /// only the task is billed.
    fn run_sidechain_on(
        &mut self,
        key: EngineKey,
        engine: Box<dyn Engine>,
        run: impl FnOnce(&mut Self) -> Result<(), String>,
    ) -> Result<(), String> {
        let parent_engine = std::mem::replace(&mut self.engine, engine);
        // The framed task is the last message; keep it, hide everything before.
        let stashed = {
            let mut prefix = std::mem::take(&mut self.session.transcript);
            let task = prefix.pop();
            self.session.transcript = task.into_iter().collect();
            prefix
        };
        let result = run(self);
        // Unconditional, and with no `?` between the swap in and the swap out: a
        // leaked swap would leave the whole session pointed at the wrong engine,
        // which is the worst failure this design can produce.
        let alt = std::mem::replace(&mut self.engine, parent_engine);
        self.alt_engines.insert(key, alt);
        let mut restored = stashed;
        restored.append(&mut self.session.transcript);
        self.session.transcript = restored;
        result
    }

    /// Rolls the engine's KV back to the parent prefix captured by
    /// [`begin_subagent_fork`](Self::begin_subagent_fork). Without it, the
    /// post-fork prompt (parent prefix + the small report) diverges behind
    /// the sidechain's live end, and the extend-only C sync re-prefills the
    /// whole parent context from token zero instead of just the report. A
    /// restore failure keeps exactly that status quo, so it is swallowed.
    fn restore_fork_kv(&mut self) {
        let Some(Some(kv)) = self.fork_kv.pop() else {
            return;
        };
        let _ = self.engine.set_kv(&kv);
    }

    /// Ends a `/subagent` fork: truncates the sidechain back out of the
    /// transcript and pushes only the framed final report. Returns false when
    /// the sidechain produced no report (e.g. interrupted before any output);
    /// the transcript is still restored.
    fn finish_subagent_fork(&mut self, fork_at: usize, task: &str) -> bool {
        let report = last_assistant_text(&self.session.transcript[fork_at..]);
        self.session.transcript.truncate(fork_at);
        self.restore_fork_kv();
        match report {
            Some(report) => {
                self.session
                    .push(Message::user(crate::agents::report_message(task, &report)));
                true
            }
            None => false,
        }
    }

    fn btw_plain(&mut self, question: &str) -> Result<(), String> {
        let mut prompt_text = render_transcript(&self.session, &self.system);
        {
            use std::fmt::Write as _;
            let _ = write!(prompt_text, "[user]\n{}\n", btw_user_message(question));
        }
        let saved_ctx = self.last_ctx_used;
        let (stream, _text, _stats) = self.stream_generation(&prompt_text, Instant::now())?;
        let tried_tool = !stream.finished().calls.is_empty() || stream.finished().error.is_some();
        let mut renderer = stream.into_sink().renderer;
        renderer.finish();
        if !renderer.last_output_newline() {
            println!();
        }
        if tried_tool {
            println!(
                "(the model tried to call a tool; tools are disabled during /btw — ask in the main conversation)"
            );
        }
        println!(
            "{}",
            self.debug_line("[btw — not part of the conversation]")
        );
        self.last_ctx_used = saved_ctx;
        Ok(())
    }

    /// Resolves `/name args` against the loaded skills, rendering the
    /// user-turn preamble on a match.
    fn skill_message(&self, cmd: &str, arg: &str) -> Option<String> {
        let name = cmd.strip_prefix('/')?;
        let skill = self.skills.iter().find(|s| s.name == name)?;
        Some(crate::skills::render(skill, arg))
    }

    /// Resolves an unrecognized `/name args` against the user's extensions:
    /// skills first, then prompt templates (issue #67). `None` means no
    /// match — the caller reports an unknown command; `Some(Err)` is a
    /// matched template whose variables could not be bound.
    fn slash_message(&self, cmd: &str, arg: &str) -> Option<Result<String, String>> {
        if let Some(message) = self.skill_message(cmd, arg) {
            return Some(Ok(message));
        }
        let tpl = crate::templates::resolve(&self.templates, cmd)?;
        Some(crate::templates::render(tpl, arg))
    }
}

/// Parses `/remember [user] <text>` and appends to the right memory scope:
/// a leading `user` word selects the user file, everything else lands in the
/// project file.
fn remember_from_arg(cwd: &std::path::Path, arg: &str) -> Result<std::path::PathBuf, String> {
    let arg = arg.trim();
    let (scope, text) = match arg.split_once(char::is_whitespace) {
        Some(("user", rest)) => (crate::memory::Scope::User, rest),
        _ => (crate::memory::Scope::Project, arg),
    };
    crate::memory::remember(scope, cwd, text, &crate::context::current_local_iso_date())
}

/// The `/btw` side panel: `Some` while it splits the screen (main 60% / btw
/// 40%). Owned by [`Agent::tui_loop`] so it persists across turn boundaries —
/// a finished main task never closes it; only Esc does.
type BtwPanel = Option<(OutputLog, tui::OutputView)>;

/// An engine taken out of the alt cache, paired with the key it must be put
/// back under — the two always travel together, so a sidechain cannot lose one.
type AltEngine = (EngineKey, Box<dyn Engine>);

/// Result of one TUI generation pass.
struct TurnOutput {
    interrupted: bool,
    /// A priority `/btw` stopped this main pass; the caller discards the
    /// partial output, answers the side question, and re-runs the pass.
    preempted: bool,
    assistant_text: String,
    /// The pass ended with a `<think>` block still open (a tool call fired
    /// mid-thought); the turn loop closes it before pushing the message.
    ended_in_think: bool,
    calls: Vec<ToolCall>,
    error: Option<String>,
}

/// Interactive input state for the ratatui UI.
struct TuiInput {
    buf: LineBuffer,
    history: History,
    /// Position within [`TuiInput::hist_eligible`], not within the history
    /// itself: in bash mode the two differ.
    hist_idx: Option<usize>,
    /// True when the current history walk started from a `!` line, fixing it
    /// to bash mode for the rest of the walk.
    hist_bang: bool,
    stash: String,
    /// Open `@` suggestion popup, when one is showing.
    popup: Option<crate::complete::Popup>,
    /// Open `/` command menu, when one is showing. Mutually exclusive with
    /// `popup`: `@` needs whitespace before it and `/` only opens at byte 0.
    slash: Option<crate::slashmenu::SlashMenu>,
    /// Every command the `/` menu can offer — built-ins plus the skills and
    /// templates loaded at startup. Snapshotted once because neither set
    /// changes during a session.
    slash_catalog: Vec<crate::slashmenu::Entry>,
    /// Index worker, started lazily on the first `@`.
    worker: Option<crate::complete::IndexWorker>,
    /// MCP resource candidates, refreshed by `tui_loop` and handed to the
    /// worker. Lives here so the free-function busy loop gets identical
    /// behavior without threading the agent through.
    ///
    /// Refreshed by `tui_loop` on every idle tick and pushed to the running
    /// worker, so a server that connects mid-session starts contributing
    /// completions (issue #41).
    mcp_extra: Vec<crate::complete::Candidate>,
}

impl TuiInput {
    fn new() -> Self {
        Self {
            buf: LineBuffer::new(),
            history: History::new(crate::settings::active().ui.history_size),
            hist_idx: None,
            hist_bang: false,
            stash: String::new(),
            popup: None,
            slash: None,
            slash_catalog: crate::slashmenu::catalog(&[], &[]),
            worker: None,
            mcp_extra: Vec::new(),
        }
    }

    /// Replaces the `/` menu's candidate list, folding this session's skills
    /// and templates in beside the built-ins.
    fn set_slash_catalog(
        &mut self,
        skills: &[crate::skills::Skill],
        templates: &[crate::templates::Template],
    ) {
        self.slash_catalog = crate::slashmenu::catalog(skills, templates);
    }

    /// The prompt as the renderer needs it: text, cursor, and selection, all
    /// in char indices.
    fn state(&self) -> tui::InputState<'_> {
        tui::InputState {
            text: self.buf.text(),
            cursor: self.cursor_char(),
            sel: self.selection_chars(),
        }
    }

    /// The selected range as char indices, which is what the renderer wants;
    /// [`crate::editor::LineBuffer`] tracks bytes.
    fn selection_chars(&self) -> Option<(usize, usize)> {
        let (a, b) = self.buf.selection()?;
        let text = self.buf.text();
        Some((text[..a].chars().count(), text[..b].chars().count()))
    }

    /// Text of the current buffer left of the cursor, used for `@` detection.
    fn left_of_cursor(&self) -> &str {
        let text = self.buf.text();
        &text[..self.buf.cursor().min(text.len())]
    }

    /// True when the cursor sits at the end of the `@` token it is inside.
    ///
    /// [`crate::complete::Popup`] replaces the byte range `token.start ..
    /// cursor`, which is only the whole token while nothing of it trails the
    /// cursor. Without this guard, typing `@src`, pressing Left twice and then
    /// Tab would glue the stale tail onto the completion.
    fn cursor_at_token_end(&self) -> bool {
        let text = self.buf.text();
        let cursor = self.buf.cursor().min(text.len());
        text[cursor..]
            .chars()
            .next()
            .is_none_or(char::is_whitespace)
    }

    /// Opens, retargets, or closes both typeahead menus to match the current
    /// input text.
    ///
    /// Called after every key. Starts the index worker lazily on the first `@`
    /// so a session that never completes never shells out to git.
    fn sync_popup(&mut self) {
        self.sync_slash();
        let token = crate::complete::detect_at_token(self.left_of_cursor())
            .filter(|_| self.cursor_at_token_end());
        let Some(token) = token else {
            self.popup = None;
            return;
        };
        if self.worker.is_none() {
            let root = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
            self.worker = Some(crate::complete::IndexWorker::spawn(
                root,
                self.mcp_extra.clone(),
                crate::settings::active().ui.respect_gitignore,
            ));
        }
        let query = token.query.clone();
        let popup = self
            .popup
            .get_or_insert_with(|| crate::complete::Popup::new(token.clone()));
        let generation = popup.bump_generation(token);
        if let Some(w) = &self.worker {
            w.query(generation, &query);
        }
    }

    /// Opens, refilters, or closes the `/` command menu for the current input.
    ///
    /// A query that matches nothing closes it outright rather than showing an
    /// empty box: an unknown `/command` is a legitimate thing to type (it goes
    /// to the model as an ordinary prompt), and a menu hovering over it with no
    /// rows would just be in the way.
    fn sync_slash(&mut self) {
        let token = crate::slashmenu::detect_slash_token(self.left_of_cursor())
            .filter(|_| self.cursor_at_token_end());
        let Some(token) = token else {
            self.slash = None;
            return;
        };
        match &mut self.slash {
            Some(menu) => menu.retarget(&token.query),
            None => {
                self.slash = Some(crate::slashmenu::SlashMenu::new(
                    self.slash_catalog.clone(),
                    &token.query,
                ));
            }
        }
        if self
            .slash
            .as_ref()
            .is_some_and(crate::slashmenu::SlashMenu::is_empty)
        {
            self.slash = None;
        }
    }

    /// Replaces the MCP resource candidates, forwarding them to a running
    /// worker so a server connecting mid-session becomes completable.
    ///
    /// A no-op when the list is unchanged, which is the common case: this runs
    /// on every idle tick.
    fn set_mcp_extra(&mut self, extra: Vec<crate::complete::Candidate>) {
        if extra == self.mcp_extra {
            return;
        }
        self.mcp_extra = extra;
        if let Some(w) = &self.worker {
            w.set_extra(self.mcp_extra.clone());
        }
    }

    /// Drains worker messages into the popup. Call once per event-loop tick.
    fn pump_popup(&mut self) {
        let Some(w) = &self.worker else { return };
        let mut msgs = Vec::new();
        while let Some(msg) = w.try_recv() {
            msgs.push(msg);
        }
        let mut refreshed = false;
        for msg in msgs {
            if matches!(msg, crate::complete::IndexMsg::Refreshed) {
                refreshed = true;
            }
            if let Some(p) = &mut self.popup {
                p.accept_msg(msg);
            }
        }
        // The index changed under an open popup (the untracked fold or a
        // rebuild landed): re-issue the current query so the list is not stale
        // until the user happens to type another character.
        if refreshed && self.popup.is_some() {
            self.sync_popup();
        }
    }

    /// Offers `key` to an open menu (`/` first, then `@`), the single entry
    /// point both TUI key loops share so they cannot drift.
    ///
    /// Returns true when a menu consumed the key and the caller must skip its
    /// own binding for it.
    fn popup_key(&mut self, key: KeyEvent) -> bool {
        use crate::complete::PopupAction;
        if self.slash_key(key) {
            return true;
        }
        if self.popup.is_none() {
            return false;
        }
        let before = self.buf.text().to_owned();
        let Some(popup) = self.popup.as_mut() else {
            return false;
        };
        match popup.handle_key(key, &mut self.buf) {
            PopupAction::Passthrough => false,
            PopupAction::Dismissed => {
                // Esc (and an empty accept) closes without re-syncing, so the
                // popup stays shut until the next edit.
                self.popup = None;
                true
            }
            PopupAction::Consumed => {
                // Re-sync only when the key actually edited the buffer (Tab,
                // Enter-on-directory). Re-syncing after a pure selection key
                // would re-issue the same query, and the worker's reply resets
                // `selected` to 0 — cancelling the user's Up/Down.
                if self.buf.text() != before {
                    self.sync_popup();
                }
                true
            }
        }
    }

    /// Selection and clipboard keys, the second thing both TUI key loops share
    /// (after [`TuiInput::popup_key`]) so the two cannot drift.
    ///
    /// Shift turns every cursor motion into a selecting motion by pinning the
    /// anchor before the caller's own binding runs; an unshifted motion drops
    /// the selection. Ctrl-C copies a selection (falling through to the
    /// caller's "clear the line" meaning when there is none), Ctrl-X cuts,
    /// Ctrl-V pastes, and Ctrl-Shift-A selects everything.
    ///
    /// Shift+Up/Down are the exception that must be *consumed* here: unshifted
    /// they walk the history, so the caller's binding is the wrong one and
    /// cannot be reached by falling through.
    ///
    /// A consumed key re-syncs the menus itself. Both loops reach their
    /// end-of-iteration `sync_popup` by *falling through* the key match, so a
    /// consumed key that skipped ahead would otherwise leave a menu open over
    /// text that no longer justifies it.
    fn selection_key(&mut self, key: KeyEvent) -> bool {
        let consumed = self.selection_key_inner(key);
        if consumed {
            self.sync_popup();
        }
        consumed
    }

    /// The body of [`TuiInput::selection_key`], split out so the re-sync above
    /// covers every consuming arm without each having to remember it.
    fn selection_key_inner(&mut self, key: KeyEvent) -> bool {
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            // Ctrl-C / Ctrl-Shift-C: copy. With nothing selected this falls
            // through so Ctrl-C keeps meaning "clear the line" / "interrupt".
            KeyCode::Char('c' | 'C') if ctrl => self.copy_selection(false),
            KeyCode::Char('x' | 'X') if ctrl => self.copy_selection(true),
            KeyCode::Char('v' | 'V') if ctrl => {
                match crate::tui::paste_from_clipboard() {
                    Some(text) => {
                        self.hist_idx = None;
                        // The prompt is multi-line but a pasted newline would
                        // submit on the next Enter; fold to spaces, matching
                        // the bracketed-paste path.
                        self.buf
                            .insert(text.replace("\r\n", "\n").replace(['\n', '\r'], " "));
                    }
                    None => crate::status::set_flash_tip("clipboard has no text".to_owned()),
                }
                true
            }
            KeyCode::Char('a' | 'A') if ctrl && shift => {
                self.buf.select_all();
                true
            }
            // Shift+Up/Down move by logical line instead of walking history.
            KeyCode::Up if shift => {
                self.buf.anchor_here();
                self.buf.move_line_up();
                true
            }
            KeyCode::Down if shift => {
                self.buf.anchor_here();
                self.buf.move_line_down();
                true
            }
            KeyCode::Left | KeyCode::Right | KeyCode::Home | KeyCode::End => {
                if shift {
                    self.buf.anchor_here();
                } else {
                    self.buf.clear_selection();
                }
                false
            }
            // Unshiftable motions (the readline aliases): they always collapse
            // the selection.
            KeyCode::Up | KeyCode::Down => {
                self.buf.clear_selection();
                false
            }
            KeyCode::Char('a' | 'e') if ctrl => {
                self.buf.clear_selection();
                false
            }
            KeyCode::Char('b' | 'f') if key.modifiers.contains(KeyModifiers::ALT) => {
                self.buf.clear_selection();
                false
            }
            _ => false,
        }
    }

    /// Copies the selection to the clipboard, deleting it too when `cut`.
    /// Returns whether there was anything to copy.
    ///
    /// A plain copy leaves the selection standing, so the copied text stays
    /// visible and can still be cut, replaced by typing, or extended.
    fn copy_selection(&mut self, cut: bool) -> bool {
        let Some(text) = self.buf.selected_text().map(str::to_owned) else {
            return false;
        };
        crate::tui::copy_to_clipboard(&text);
        let chars = text.chars().count();
        let verb = if cut { "Cut" } else { "Copied" };
        crate::status::set_flash_tip(format!("📋 {verb} {chars} chars"));
        if cut {
            self.hist_idx = None;
            self.buf.delete_selection();
        }
        true
    }

    /// Points the cursor at the screen cell `(col, row)` when it lands in the
    /// prompt text rect `rect`, starting (`drag == false`) or extending
    /// (`drag == true`) a selection there. Returns whether the cell was in the
    /// prompt at all.
    ///
    /// Callers pass `rect` from [`crate::tui::last_input_rect`] — the rect the
    /// last frame actually drew, rather than a recomputed one, because the
    /// prompt's position depends on the task strip's height.
    fn mouse_to_cursor(
        &mut self,
        rect: ratatui::layout::Rect,
        col: u16,
        row: u16,
        drag: bool,
    ) -> bool {
        let Some(idx) = crate::tui::input_hit(rect, self.buf.text(), col, row) else {
            return false;
        };
        let byte = self
            .buf
            .text()
            .char_indices()
            .nth(idx)
            .map_or(self.buf.text().len(), |(b, _)| b);
        if !drag {
            self.buf.clear_selection();
        }
        self.buf.set_cursor(byte);
        // Pin the anchor where the press landed, so the drag that may follow
        // has an origin. On the press itself anchor == cursor, which
        // `LineBuffer::selection` reports as no selection — a plain click
        // therefore just moves the caret. On a drag the anchor is already set
        // and this leaves it alone.
        self.buf.anchor_here();
        // The caret moved, so a menu anchored to where it was may no longer
        // apply: clicking into the middle of `@src/foo` closes its popup, the
        // same as arrowing there would.
        self.sync_popup();
        true
    }

    /// Offers `key` to the open `/` menu. Returns true when it consumed it.
    ///
    /// Accepting an entry rewrites the buffer, so the menus are re-synced
    /// afterwards: the new text ends in a space, which closes this menu and
    /// leaves the prompt ready for arguments.
    fn slash_key(&mut self, key: KeyEvent) -> bool {
        use crate::slashmenu::MenuAction;
        let Some(menu) = self.slash.as_mut() else {
            return false;
        };
        match menu.handle_key(key, &mut self.buf) {
            MenuAction::Passthrough => false,
            MenuAction::Consumed => true,
            MenuAction::Dismissed => {
                self.slash = None;
                true
            }
        }
    }

    /// Cursor position as a char index into the input text. The TUI wraps the
    /// prompt itself, so it maps this to a visual `(row, col)` at render time.
    fn cursor_char(&self) -> usize {
        let text = self.buf.text();
        text[..self.buf.cursor().min(text.len())].chars().count()
    }

    /// Moves through history like the line editor (dir -1 = older).
    /// Indices of the history entries this navigation may visit, oldest first.
    ///
    /// In bash mode only past `!` commands are eligible, mirroring the
    /// reference: prompt mode shows everything, bash mode filters to bash.
    ///
    /// Directory scope is an orthogonal, second filter (issue #49): entries
    /// entered in another directory are hidden, keeping untagged/global entries
    /// visible. The two filters compose — a `!` walk still cycles `!` commands
    /// only, now further restricted to the current directory.
    fn hist_eligible(&self) -> Vec<usize> {
        (0..self.history.len())
            .filter(|i| self.history.is_eligible(*i))
            .filter(|i| !self.hist_bang || self.history.get(*i).is_some_and(|e| e.starts_with('!')))
            .collect()
    }

    fn history_move(&mut self, dir: i32) {
        if self.hist_idx.is_none() {
            // Mode is fixed when navigation starts. Re-deriving it per keypress
            // would flip it the moment a non-`!` entry lands in the buffer,
            // stranding the user in the middle of a cycle.
            self.hist_bang = self.buf.text().starts_with('!');
        }
        let eligible = self.hist_eligible();
        if eligible.is_empty() {
            return;
        }
        let len = eligible.len();
        let new_index = match (self.hist_idx, dir) {
            (None, d) if d < 0 => {
                self.stash = self.buf.text().to_owned();
                Some(len - 1)
            }
            (None, _) => None,
            (Some(0), d) if d < 0 => Some(0),
            (Some(i), d) if d < 0 => Some(i - 1),
            (Some(i), _) if i + 1 < len => Some(i + 1),
            (Some(_), _) => {
                self.buf.set_text(std::mem::take(&mut self.stash));
                self.hist_idx = None;
                return;
            }
        };
        self.hist_idx = new_index;
        if let Some(i) = new_index {
            let entry = eligible
                .get(i)
                .and_then(|h| self.history.get(*h))
                .unwrap_or_default()
                .to_owned();
            self.buf.set_text(entry);
        }
    }
}

impl Agent<'_> {
    /// Runs the full-screen ratatui interactive session.
    ///
    /// # Errors
    /// Returns an error string on unrecoverable terminal or engine failure.
    fn run_tui(&mut self) -> Result<(), String> {
        // Install the `ask` rendezvous (issue #34): the worker's asker parks a
        // question on the shared bridge and the event loop renders it. Both
        // halves share one Arc-backed bridge.
        let ask_bridge = crate::tools::ask::AskBridge::new();
        self.tool_ctx.asker = Some(Box::new(crate::tools::ask::BridgeAsker(ask_bridge.clone())));
        self.tool_ctx.ask_bridge = Some(ask_bridge);
        // stdout is the alternate screen from here on: drop the REPL's
        // print-to-stdout status sink. `worker_turn` installs a channel-backed
        // one for the duration of each turn.
        self.tool_ctx.status_sink = None;
        // `--ui-remote`: bind the loopback listener *before* the alternate
        // screen is entered, so the port line lands on a clean stderr (stdout
        // belongs to the UI). Started here rather than in `main` because this
        // is the only front end the feature applies to.
        if let Some(port) = self.cfg.ui_remote {
            let handle = crate::uiremote::start(port)?;
            eprintln!("ui-remote listening on 127.0.0.1:{}", handle.port);
            crate::uiremote::set_recording(true);
            self.ui_remote = Some(Arc::new(Mutex::new(UiRemote::new(handle))));
        }
        let mut terminal = ratatui::init();
        // Capture the mouse so wheel events scroll the output buffer instead
        // of being translated by the terminal into arrow keys (history moves),
        // and drags select text for copying. Bracketed paste makes Cmd-V
        // arrive as a single Paste event instead of a burst of key presses.
        let _ = ratatui::crossterm::execute!(
            std::io::stdout(),
            EnableMouseCapture,
            EnableBracketedPaste,
            event::EnableFocusChange,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        );
        // The window the user just launched plank in is focused; focus events
        // track changes from here for the "unfocused" notification mode.
        crate::notify::set_focused(true);
        // Still launching: `tui_loop` warms the KV cache before the real UI
        // appears, and only flips the title to Idle once it accepts input.
        crate::title::set(crate::title::State::Loading);
        let result = self.tui_loop(&mut terminal);
        // Retro CRT power-off of the final frame on a clean exit. Best-effort:
        // any error is swallowed so the terminal is always restored and the
        // real turn outcome (`result`) is what we return. `tui_loop` hands back
        // the last frame it drew (already captured pre-buffer-swap); skipped on
        // error exits (keep error text readable), non-TTY stdout, or when
        // disabled (in which case the image is `None`).
        if let Ok(Some(img)) = &result {
            let cfg = crt_off::Config {
                hold_secs: 0.0,
                vstretch_secs: 0.35,
                hstretch_secs: 0.25,
                // Long enough that the phosphor dot visibly dims away instead
                // of blinking out: crt-off ramps its brightness linearly to
                // black across this window.
                dot_fade_secs: 0.9,
                black_secs: 0.1,
                fps: 60.0,
            };
            // `use_alt_screen: false` — we already own the alternate screen; the
            // effect repaints the final frame in place and restores raw mode on
            // drop before our own teardown runs.
            let _ = crt_off::animate(img, false, &cfg);
        }
        let _ = ratatui::crossterm::execute!(
            std::io::stdout(),
            PopKeyboardEnhancementFlags,
            event::DisableFocusChange,
            DisableBracketedPaste,
            DisableMouseCapture
        );
        ratatui::restore();
        result.map(|_| ())
    }

    /// Lays the version line out to the *right* of the logo, on the logo's
    /// middle row, so the two read as one masthead instead of stacking.
    ///
    /// Art with no rows has no middle to hang the text off, so the version
    /// falls back to a line of its own and nothing is lost.
    fn masthead(
        mut art: Vec<ratatui::text::Line<'static>>,
        version: String,
    ) -> Vec<ratatui::text::Line<'static>> {
        if art.is_empty() {
            return vec![ratatui::text::Line::from(version)];
        }
        let middle = (art.len() - 1) / 2;
        art[middle]
            .spans
            .push(ratatui::text::Span::raw(format!("  {version}")));
        art
    }

    /// Writes the startup banner (logo art, version/context line, hints) into
    /// `log`. Used both at launch and after `/clear` and `/new`, so a cleared
    /// screen looks exactly like a fresh start.
    fn tui_write_banner(&self, log: &mut OutputLog) {
        let version = format!(
            "plank {} 🪵 Agent, context {} tokens",
            crate::logo::version_label(),
            status::format_ctx_size(self.engine.ctx_size())
        );
        let art = tui::ansi_to_lines(&crate::logo::art(crate::logo::DEFAULT_WIDTH * 3 / 4));
        for line in Self::masthead(art, version) {
            log.push_spans(line.spans);
        }
        log.push_plain("Type a message, or /help for commands. Ctrl-D to quit.");
        // Non-intrusive one-time update hint (issue #56), shown in yellow just
        // below the welcome line; absent when up to date or the check is off.
        if let Some(notice) = crate::upgrade::update_notice() {
            log.push_spans(vec![ratatui::text::Span::styled(
                notice.to_string(),
                ratatui::style::Style::default().fg(ratatui::style::Color::Yellow),
            )]);
        }
        // Only the echo stub has no model name, and it answers nothing useful.
        if self.engine.model_name().is_empty() {
            log.push_plain(String::new());
            for (i, line) in status::no_model_lines().into_iter().enumerate() {
                if i == 0 {
                    log.push_spans(vec![ratatui::text::Span::styled(
                        line,
                        ratatui::style::Style::default().fg(ratatui::style::Color::Yellow),
                    )]);
                } else {
                    log.push_dim(line);
                }
            }
        }
        log.push_plain(String::new());
    }

    #[allow(clippy::too_many_lines)]
    /// Returns the final rendered frame as an image (when the CRT-off effect is
    /// enabled and stdout is a TTY) so the caller can animate a power-off of the
    /// last visible screen. `None` otherwise. The image is captured from the
    /// live draw each tick — ratatui swaps buffers after `draw`, so by the time
    /// the loop returns `current_buffer_mut` is the blank incoming buffer, not
    /// what the user last saw.
    fn tui_loop(
        &mut self,
        terminal: &mut ratatui::DefaultTerminal,
    ) -> Result<Option<image::RgbaImage>, String> {
        // Cloned out of `self` so the remote state stays reachable while the
        // loop hands `&mut self` to a turn.
        let ui_remote = self.ui_remote.clone();
        let rem = ui_remote.as_deref();
        let mut input = TuiInput::new();
        input.set_mcp_extra(crate::tools::mcp::resource_candidates(&self.tool_ctx.mcp));
        input.set_slash_catalog(&self.skills, &self.templates);
        let hist_path = default_history_path();
        input.history.load(&hist_path).ok();

        // Rebuild the system-prompt cache first, behind a simple progress bar,
        // so the full UI appears only once the one slow launch step is done.
        self.tui_warm(terminal)?;
        // Warming is done: the full-screen UI is about to be up and accepting
        // input, so the window title stops saying "launching".
        crate::title::set(crate::title::State::Idle);

        let mut log = OutputLog::new();
        self.tui_write_banner(&mut log);

        // A `plank /resume` startup shows the recovered conversation so far,
        // rendered like the live stream (markdown + thinking gray).
        self.replay_history_into_log(&mut log);

        let mut view = tui::OutputView::default();
        // The sub-agent output pane, owned here for the same reason as the
        // `/btw` panel: a finished sub-agent run stays readable after the turn
        // that produced it returns to this idle loop.
        let mut sub_pane = tui::SubPane::default();
        // The `/btw` side panel, owned here so it outlives any single turn:
        // once opened it stays until the user presses Esc, even after the main
        // task finishes and control returns to this idle loop.
        let mut btw_panel: BtwPanel = None;
        // An open easter egg (`/pelota`, …) or the screensaver, same modal
        // contract as the
        // `/config` form: while it is Some it owns the screen and every key.
        // Owned here, like the `/btw` panel, so it survives the transition into
        // and out of a turn — that is what lets it keep running while the model
        // works. `arcade_last` measures the real frame delta to feed its
        // simulation; the poll timeout alone would stall it whenever keys
        // arrive.
        let mut arcade = crate::arcade::Arcade::new();
        let mut arcade_last = Instant::now();
        // When the user was last heard from, for `ui.screensaver`. Only real
        // input counts: a poll that times out is exactly the idleness the
        // screensaver is waiting for. A running turn never reaches this loop,
        // so a long generation cannot be mistaken for an idle user.
        let mut last_activity = Instant::now();
        if let Some(initial) = self.cfg.prompt.as_deref().filter(|p| !p.is_empty()) {
            log.push_spans(tui::user_echo_spans(initial));
            self.session.push(Message::user(initial));
            self.tui_turn(
                terminal,
                &mut log,
                &mut view,
                &mut input,
                &mut btw_panel,
                &mut arcade,
                &mut sub_pane,
            )?;
            // `--prompt` runs before the loop is ever idle: restart the clock
            // so the screensaver waits out a full idle stretch afterwards.
            last_activity = Instant::now();
        }

        // Endpoints of a mouse drag selection over the output area, in content
        // space: `(column, absolute-wrapped-row)`. Anchoring the row to content
        // (not the screen) lets the selection survive scrolling. Copied to the
        // clipboard on button release.
        let mut selection: Option<tui::ContentSelection> = None;
        // True between press and release of a drag that started inside the
        // prompt: that drag selects input text (tracked on the `LineBuffer`)
        // rather than transcript text, so the two never fight over one gesture.
        let mut input_drag = false;
        // The interactive `/config` modal, when open; it intercepts all keys
        // and renders over the frame until Esc (save) or q/Ctrl-C (cancel).
        let mut config_form: Option<crate::configform::ConfigForm> = None;
        // Images pasted (clipboard or file path) awaiting the next submit;
        // attached to the message as file references the model's tools can
        // read. Always empty while IMAGES_ENABLED is off.
        let mut attachments: Vec<crate::imagepaste::PastedImage> = Vec::new();
        // Clipboard-image hint, re-probed every few seconds (the probe shells
        // out to osascript, so it must not run on every 200ms poll tick).
        let mut clip_has_image = IMAGES_ENABLED && crate::imagepaste::clipboard_has_image();
        let mut clip_checked = Instant::now();
        // Only rasterize the frame when the CRT-off power-down will actually run;
        // otherwise the per-tick capture is pure waste.
        let capture_crt = crate::settings::active().ui.crt_off && std::io::stdout().is_terminal();
        let mut crt_frame: Option<image::RgbaImage> = None;
        loop {
            if IMAGES_ENABLED && clip_checked.elapsed() >= Duration::from_secs(3) {
                clip_has_image = crate::imagepaste::clipboard_has_image();
                clip_checked = Instant::now();
            }
            remote_drain(rem);
            // Advance an open easter egg by the real elapsed time. `step`
            // clamps a long delta itself, so a modal that just opened (or a
            // suspended terminal) resumes smoothly instead of jumping.
            if arcade.is_open() {
                let dt = arcade_last.elapsed();
                arcade_last = Instant::now();
                arcade.step(u64::try_from(dt.as_millis()).unwrap_or(u64::MAX));
            }
            input.set_mcp_extra(crate::tools::mcp::resource_candidates(&self.tool_ctx.mcp));
            input.pump_popup();
            let mut status = self.idle_status_text();
            if clip_has_image {
                status.push_str(" | 📷 image in clipboard (Cmd-V attaches)");
            }
            let task_view = tui::TaskView::from(&self.session.tasks);
            // Same pane selection as the busy loop, hoisted out of the draw
            // closure: a Ctrl-O pressed while idle has to be visible here too.
            let sub_active = sub_pane.active;
            // Owned so the draw closure does not hold a borrow of `sub_pane`
            // alongside the mutable borrow of its view.
            let sub_title: Option<String> = if sub_active {
                sub_pane.label.clone()
            } else {
                None
            };
            let idle_status = if let (true, Some(label)) = (sub_active, sub_pane.label.as_deref()) {
                format!("[sub-agent: {label}] {status}")
            } else {
                status
            };
            let (draw_log, draw_view): (&OutputLog, &mut tui::OutputView) = if sub_active {
                (&sub_pane.log, &mut sub_pane.view)
            } else {
                (&log, &mut view)
            };
            let completed = terminal
                .draw(|f| {
                    // A `/btw` panel left open from an earlier turn keeps the
                    // split view even while idle; text selection falls back to
                    // the single-column path (no panel). The sub-agent pane
                    // takes precedence: the split is about the main task.
                    if let (false, Some((btw_log, btw_view))) = (sub_active, btw_panel.as_mut()) {
                        tui::draw_btw_split(
                            f,
                            draw_log,
                            btw_log,
                            btw_view,
                            Some(input.state()),
                            &idle_status,
                            draw_view,
                            &task_view,
                        );
                    } else {
                        tui::draw(
                            f,
                            draw_log,
                            Some(input.state()),
                            &idle_status,
                            draw_view,
                            selection,
                            &task_view,
                            sub_title.as_deref(),
                        );
                    }
                    if let Some(m) = &input.slash {
                        tui::draw_slash_menu(f, input.buf.text(), m);
                    }
                    if let Some(p) = &input.popup {
                        tui::draw_popup(f, input.buf.text(), p);
                    }
                    if let Some(form) = &config_form {
                        tui::draw_config(f, form);
                    }
                    // Drawn last: the arcade covers the whole frame.
                    if arcade.is_open() {
                        tui::draw_arcade(f, &arcade);
                    }
                    remote_capture(rem, f);
                })
                .map_err(|e| e.to_string())?;
            // Snapshot the just-drawn buffer (ratatui has already swapped, so
            // `completed.buffer` is the frame the user sees, not the blank one).
            if capture_crt {
                crt_frame = Some(frame_to_image(completed.buffer));
            }
            remote_service(rem);

            // 200 ms is five frames a second — fine for an idle prompt, far too
            // slow for a game. An open easter egg polls at the shared 20 Hz
            // animation tick instead.
            // Idle long enough? Put the stars up. Skipped while anything
            // modal is on screen — a screensaver over a dialog would hide a
            // question the user still has to answer.
            if !arcade.is_open()
                && config_form.is_none()
                && let Some(after) = crate::settings::active().ui.screensaver.duration()
                && last_activity.elapsed() >= after
            {
                let (w, h) = terminal
                    .size()
                    .map_or((80, 23), |sz| (sz.width, sz.height.saturating_sub(1)));
                arcade.open_screensaver(arcade_seed(), w, h);
                arcade_last = Instant::now();
            }

            let poll = if arcade.is_open() {
                Duration::from_millis(crate::anim::TICK_MS)
            } else {
                Duration::from_millis(200)
            };
            let Some(ev) = next_event(rem, poll)? else {
                // Remote-driven input (issue #25): a remote controller's
                // `prompt`/`command` frames start a local turn just as if typed
                // here, so the local screen and the remote mirror stay in sync.
                if let Some(r) = self.remote.clone() {
                    let queued = r.shared.take_queued();
                    let mut run = false;
                    for line in queued {
                        let line = line.trim().to_owned();
                        if line.is_empty() {
                            continue;
                        }
                        if line.starts_with('/') {
                            if !self.tui_slash(
                                &line,
                                &mut log,
                                terminal,
                                &mut view,
                                &mut input,
                                &mut btw_panel,
                                &mut config_form,
                                &mut arcade,
                                &mut sub_pane,
                            ) {
                                input.history.save(&hist_path).ok();
                                remote_abandon(rem);
                                return Ok(crt_frame);
                            }
                        } else {
                            r.bus.broadcast(UiEvent::UserEcho(line.clone()));
                            log.push_spans(tui::user_echo_spans(&line));
                            self.session.push(Message::user(line));
                            run = true;
                        }
                    }
                    if run {
                        self.tui_turn(
                            terminal,
                            &mut log,
                            &mut view,
                            &mut input,
                            &mut btw_panel,
                            &mut arcade,
                            &mut sub_pane,
                        )?;
                        // A remote-driven turn is time the user was not idle
                        // at the prompt: start the screensaver clock from the
                        // moment the UI comes back to idle, not from before
                        // the turn.
                        last_activity = Instant::now();
                    }
                }
                continue;
            };
            // What counts as the user being here: keys, mouse, and pastes.
            // Focus and resize events deliberately do not — a window manager
            // moving focus around, or another app resizing the terminal, is
            // not somebody at the keyboard, and treating it as activity means
            // the screensaver never comes up on a busy desktop.
            let from_user = is_user_activity(&ev);
            if from_user {
                last_activity = Instant::now();
            }
            // The screensaver is dismissed by any of those, not just a key:
            // moving the mouse is a person coming back. The event that wakes
            // it is consumed rather than acted on — waking a screensaver
            // should not leave a stray character in the prompt or click a
            // button the user could not see.
            if arcade.is_screensaver() && from_user {
                arcade.close();
                continue;
            }
            // An open easter egg takes the mouse (wheel, click and drag steer
            // the paddle) and swallows everything else that is not a key, so
            // nothing underneath it scrolls or accepts text while it is up.
            if arcade.is_open() && !matches!(ev, Event::Key(_)) {
                if let Event::Mouse(m) = ev {
                    let (w, h) = terminal
                        .size()
                        .map_or((80, 23), |s| (s.width, s.height.saturating_sub(1)));
                    arcade.handle_mouse(m, w, h);
                }
                continue;
            }
            if let Event::Mouse(m) = &ev {
                match m.kind {
                    MouseEventKind::ScrollUp => {
                        // Selection endpoints are content-anchored, so scrolling
                        // leaves them alone — the highlight tracks the text.
                        let v = sub_pane.active_view(&mut view);
                        v.follow = false;
                        v.top = v.top.saturating_sub(3);
                    }
                    MouseEventKind::ScrollDown => {
                        // Clamped by draw, which re-enters follow mode at the bottom.
                        let v = sub_pane.active_view(&mut view);
                        v.top = v.top.saturating_add(3);
                    }
                    MouseEventKind::Down(MouseButton::Left) => {
                        // Every press decides afresh which surface the gesture
                        // belongs to, so a release lost off-window cannot leave
                        // the next drag stuck on the prompt.
                        input_drag = false;
                        // A click on the jump-to-bottom hint resumes follow mode
                        // (same as End) instead of starting a text selection.
                        let v = sub_pane.active_view(&mut view);
                        if v.jump_hint_rect.is_some_and(|r| {
                            r.contains(ratatui::layout::Position::new(m.column, m.row))
                        }) {
                            v.follow = true;
                            selection = None;
                        } else if tui::last_input_rect()
                            .is_some_and(|r| input.mouse_to_cursor(r, m.column, m.row, false))
                        {
                            // A press inside the prompt places the caret and
                            // arms an input-text drag, leaving the output
                            // pane's own selection alone.
                            input_drag = true;
                            selection = None;
                        } else {
                            let row = v.top.saturating_add(usize::from(m.row));
                            selection = Some(((m.column, row), (m.column, row)));
                        }
                    }
                    MouseEventKind::Drag(MouseButton::Left) if input_drag => {
                        if let Some(r) = tui::last_input_rect() {
                            input.mouse_to_cursor(r, m.column, m.row, true);
                        }
                    }
                    MouseEventKind::Drag(MouseButton::Left) => {
                        let top = sub_pane.active_view(&mut view).top;
                        if let Some((_, end)) = &mut selection {
                            *end = (m.column, top.saturating_add(usize::from(m.row)));
                        }
                    }
                    MouseEventKind::Up(MouseButton::Left) if input_drag => {
                        // Releasing an input drag copies what it selected, the
                        // same bargain the output pane makes.
                        input_drag = false;
                        input.copy_selection(false);
                    }
                    MouseEventKind::Up(MouseButton::Left) => {
                        let size = terminal.size().unwrap_or_default();
                        let top = sub_pane.active_view(&mut view).top;
                        if let Some(sel) = selection.filter(|(a, b)| a != b) {
                            // A drag: extract from the content model (not the
                            // screen buffer) so a selection larger than the
                            // viewport still copies in full — from whichever
                            // pane is on screen, so the copy matches the pixels.
                            let text = tui::selection_text_content(
                                sub_pane.active_log(&log),
                                size.width,
                                sel,
                            );
                            if !text.trim().is_empty() {
                                tui::copy_to_clipboard(&text);
                                let chars = text.chars().count();
                                crate::status::set_flash_tip(format!("📋 Copied {chars} chars"));
                            }
                        } else if let Some((col, row)) = selection.map(|(a, _)| a) {
                            // A plain click (no drag): copy a fenced code block
                            // when its header `⧉ copy` control was clicked. The
                            // stored row is absolute, so map it back to a screen
                            // row for the click test.
                            let out_h = size.height.saturating_sub(2);
                            let screen_row =
                                u16::try_from(row.saturating_sub(top)).unwrap_or(u16::MAX);
                            if screen_row < out_h
                                && let Some(code) = sub_pane
                                    .active_log(&log)
                                    .code_copy_at(size.width, top, col, screen_row)
                            {
                                tui::copy_to_clipboard(&code);
                                let chars = code.chars().count();
                                crate::status::set_flash_tip(format!("📋 Copied {chars} chars"));
                            }
                            selection = None;
                        } else {
                            selection = None;
                        }
                    }
                    _ => {}
                }
                continue;
            }
            if let Event::Paste(pasted) = &ev {
                input.hist_idx = None;
                // An empty bracketed paste means the clipboard holds an image
                // (macOS pastes no text for image content); pasted text that is
                // an image file path attaches that file.
                if IMAGES_ENABLED {
                    if pasted.trim().is_empty() {
                        match crate::imagepaste::from_clipboard() {
                            Some(img) => {
                                log.push_dim(format!(
                                    "[image #{} attached: {}]",
                                    attachments.len() + 1,
                                    img.describe()
                                ));
                                attachments.push(img);
                            }
                            None => log.push_dim("[clipboard has no image to paste]"),
                        }
                        continue;
                    }
                    if let Some(img) = crate::imagepaste::from_path_text(pasted) {
                        log.push_dim(format!(
                            "[image #{} attached: {}]",
                            attachments.len() + 1,
                            img.describe()
                        ));
                        attachments.push(img);
                        continue;
                    }
                }
                // The line editor is single-line; fold pasted newlines into
                // spaces so the paste stays editable.
                input
                    .buf
                    .insert(pasted.replace("\r\n", "\n").replace(['\n', '\r'], " "));
                input.sync_popup();
                continue;
            }
            let Event::Key(key) = ev else {
                continue;
            };
            if key.kind != KeyEventKind::Press {
                continue;
            }
            // An open easter egg owns every key until Esc/q/Ctrl-C closes it.
            if arcade.is_open() {
                if let crate::arcade::Outcome::Close(line) = arcade.handle_key(key) {
                    arcade_hover_reporting(false);
                    if let Some(line) = line {
                        log.push_dim(line);
                    }
                }
                continue;
            }
            // The `/config` modal, when open, owns every key until it closes.
            if let Some(form) = config_form.as_mut() {
                match form.handle_key(key) {
                    crate::configform::Outcome::Stay => {}
                    crate::configform::Outcome::Cancel => {
                        config_form = None;
                        log.push_dim("config: cancelled (no changes saved)");
                    }
                    crate::configform::Outcome::Save(settings) => {
                        config_form = None;
                        match crate::settings::project_path() {
                            Some(path) => match settings.save_to(&path) {
                                Ok(()) => {
                                    crate::settings::reinstall(*settings);
                                    log.push_plain(format!("config saved to {}", path.display()));
                                }
                                Err(e) => log.push_plain(format!("config save failed: {e}")),
                            },
                            None => log.push_plain("config: no working directory"),
                        }
                    }
                }
                continue;
            }
            // Any keystroke dismisses the mouse selection highlight (the text
            // was already copied on mouse release).
            selection = None;
            // The popup sees keys first: Esc closes it before the `/btw`
            // panel, and Tab/Enter/Up/Down drive the suggestion list.
            if input.popup_key(key) {
                continue;
            }
            // Then the shared selection keymap: it consumes the clipboard keys
            // and pins the anchor for Shift+motions before the motion bindings
            // below run.
            if input.selection_key(key) {
                continue;
            }
            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
            // Alt (Option on macOS) or Ctrl turns arrows and Backspace/Delete
            // into word-wise operations.
            let word_mod = ctrl || key.modifiers.contains(KeyModifiers::ALT);
            match key.code {
                // Ctrl-O switches between the main transcript and the most
                // recent sub-agent's output, while idle as well as mid-turn.
                KeyCode::Char('o') if ctrl => {
                    if !sub_pane.toggle() {
                        log.push_dim("[no sub-agent has run yet]");
                    }
                    // A selection belongs to the pane it was dragged over, so it
                    // does not survive a pane switch — otherwise its highlight
                    // would be painted over the other pane's text. (Every key
                    // already clears it above; kept here so the invariant is
                    // stated where the switch happens.)
                    selection = None;
                }
                KeyCode::Char('c') if ctrl => {
                    if !input.buf.text().is_empty() {
                        input.buf.clear();
                    } else if attachments.is_empty() {
                        log.push_spans(quit_hint_spans());
                    } else {
                        attachments.clear();
                        log.push_dim("[image attachments removed]");
                    }
                }
                KeyCode::Char('d') if ctrl => {
                    if input.buf.text().is_empty() {
                        break;
                    }
                    input.buf.delete();
                }
                KeyCode::Char('u') if ctrl => input.buf.kill_to_start(),
                KeyCode::Char('k') if ctrl => input.buf.kill_to_end(),
                KeyCode::Char('w') if ctrl => input.buf.delete_prev_word(),
                KeyCode::Char('a') if ctrl => input.buf.move_home(),
                KeyCode::Char('e') if ctrl => input.buf.move_end(),
                KeyCode::Char(c) if !ctrl && !key.modifiers.contains(KeyModifiers::ALT) => {
                    input.hist_idx = None;
                    input.buf.insert(c.to_string());
                }
                // Alt/Ctrl+Backspace deletes the previous word. Terminals that
                // cannot report the modifier send Ctrl-W, handled above.
                KeyCode::Backspace if word_mod => input.buf.delete_prev_word(),
                KeyCode::Backspace => {
                    input.buf.backspace();
                }
                KeyCode::Delete if word_mod => input.buf.delete_next_word(),
                KeyCode::Delete => {
                    input.buf.delete();
                }
                KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::ALT) => {
                    input.buf.delete_next_word();
                }
                KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::ALT) => {
                    input.buf.move_prev_word();
                }
                KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::ALT) => {
                    input.buf.move_next_word();
                }
                KeyCode::Left if word_mod => {
                    input.buf.move_prev_word();
                }
                KeyCode::Left => {
                    input.buf.move_left();
                }
                KeyCode::Right if word_mod => {
                    input.buf.move_next_word();
                }
                KeyCode::Right => {
                    input.buf.move_right();
                }
                KeyCode::Home => input.buf.move_home(),
                KeyCode::End => input.buf.move_end(),
                KeyCode::Up => input.history_move(-1),
                KeyCode::Down => input.history_move(1),
                // Esc while idle dismisses a `/btw` panel left open from an
                // earlier turn (the only way it ever closes).
                KeyCode::Esc if btw_panel.is_some() => btw_panel = None,
                // Shift+Enter inserts a newline instead of submitting.
                // Terminals without the kitty keyboard protocol cannot
                // report it, so Alt+Enter and Ctrl-J work everywhere.
                KeyCode::Enter
                    if key.modifiers.contains(KeyModifiers::SHIFT)
                        || key.modifiers.contains(KeyModifiers::ALT) =>
                {
                    input.hist_idx = None;
                    input.buf.insert("\n");
                }
                KeyCode::Char('j') if ctrl => {
                    input.hist_idx = None;
                    input.buf.insert("\n");
                }
                // Ctrl-G hands the half-typed prompt to the built-in editor or
                // `$EDITOR` — the escape hatch for prompts too long to
                // comfortably edit inline.
                KeyCode::Char('g') if ctrl => {
                    let current = input.buf.text().to_owned();
                    let edited = with_tui_suspended(terminal, || open_editor(&current));
                    match edited {
                        Ok(Some(text)) => {
                            input.hist_idx = None;
                            input.buf.set_text(text);
                        }
                        // Non-zero exit or cancelled: keep what was typed.
                        Ok(None) => {}
                        Err(e) => log.push_dim(format!("[editor failed: {e}]")),
                    }
                    input.sync_popup();
                }
                KeyCode::Enter => {
                    let line = input.buf.text().trim().to_owned();
                    input.buf.clear();
                    input.popup = None;
                    input.slash = None;
                    input.hist_idx = None;
                    view.follow = true;
                    sub_pane.view.follow = true;
                    if line.is_empty() && attachments.is_empty() {
                        continue;
                    }
                    if !line.is_empty() && !line.contains('\n') {
                        input.history.add(&line);
                        input.history.save(&hist_path).ok();
                    }
                    if let Some(rest) = line.strip_prefix('!') {
                        // `!!` is user-only shell execution: output goes to the TUI log
                        // but NOT into the session transcript (issue #20). A single `!`
                        // runs the same way but also records the command and its output
                        // as one user message, so the model has it as history — still
                        // without triggering a turn.
                        let (feedback, cmd) = match rest.strip_prefix('!') {
                            Some(rest) => (false, rest.trim().to_owned()),
                            None => (true, rest.trim().to_owned()),
                        };
                        if cmd.is_empty() {
                            log.push_dim(
                                "usage: !<shell command> (feeds the result to the model) or !!<shell command>",
                            );
                            continue;
                        }
                        log.push_spans(tui::user_echo_spans(&line));
                        let result = Self::tui_bang(
                            &self.tool_ctx.cwd.clone(),
                            &cmd,
                            &mut log,
                            terminal,
                            &mut view,
                        );
                        if feedback {
                            self.session
                                .push(Message::user(bang_transcript_entry(&cmd, &result)));
                            log.push_dim(
                                "[recorded for the model — ask about it in your next message]",
                            );
                        }
                    } else if line.starts_with('/') {
                        if !self.tui_slash(
                            &line,
                            &mut log,
                            terminal,
                            &mut view,
                            &mut input,
                            &mut btw_panel,
                            &mut config_form,
                            &mut arcade,
                            &mut sub_pane,
                        ) {
                            break;
                        }
                    } else {
                        // The engine is text-only: attach pasted images as
                        // cached-file references the model can open with its
                        // read/bash tools instead of inline content blocks.
                        let mut message = line.clone();
                        for (i, img) in attachments.drain(..).enumerate() {
                            use std::fmt::Write as _;
                            let _ = write!(
                                message,
                                "\n[Attached image #{}: {}{}. Use your tools to view it.]",
                                i + 1,
                                img.describe(),
                                img.source_path.as_deref().map_or(String::new(), |p| {
                                    format!(", original: {}", p.display())
                                })
                            );
                        }
                        let echo = if line.is_empty() { &message } else { &line };
                        log.push_spans(tui::user_echo_spans(echo));
                        self.session.push(Message::user(&message));
                        self.tui_turn(
                            terminal,
                            &mut log,
                            &mut view,
                            &mut input,
                            &mut btw_panel,
                            &mut arcade,
                            &mut sub_pane,
                        )?;
                    }
                }
                _ => {}
            }
            // Re-stamp the idle clock now that the event has been fully
            // handled. The stamp above covers the short paths that `continue`;
            // this one covers the long ones — a key that starts a turn may not
            // return here for minutes, and timing the screensaver from the
            // keystroke would put the stars up the instant the turn finishes.
            // Going back to idle is what starts the countdown.
            if from_user {
                last_activity = Instant::now();
            }
            // Retarget (or close) the popup after every edit and cursor move.
            input.sync_popup();
        }
        input.history.save(&hist_path).ok();
        Ok(crt_frame)
    }

    /// Runs an immediate shell command for the user, streaming its output into
    /// the TUI log. The model is never consulted, so no turn happens either way.
    /// The frame keeps redrawing while the command runs so Esc/Ctrl-C can kill
    /// it. The captured result is returned so the caller can decide whether it
    /// also lands in the transcript.
    ///
    /// # Behavior is intentional
    ///
    /// `!!` is **user-only** shell execution: output is displayed but never
    /// enters the conversation. That is by design, not a bug — see
    /// <https://github.com/aovestdipaperino/plank/issues/20>. `!!` commands are
    /// for the operator's convenience (checking status, running diagnostics,
    /// manual file operations), and the model should not fold that into its
    /// reasoning uninvited.
    ///
    /// A single `!` is the opt-in variant: the caller records the command and
    /// its output as one user message (see [`bang_transcript_entry`]) so the
    /// model has it as history on the next real prompt. For output the model
    /// should act on *now*, use a regular turn and let it call the `bash` tool.
    fn tui_bang(
        cwd: &std::path::Path,
        cmd: &str,
        log: &mut OutputLog,
        terminal: &mut ratatui::DefaultTerminal,
        view: &mut tui::OutputView,
    ) -> Result<crate::tools::bash::ImmediateOutput, String> {
        // Output streams into the log as it arrives (issue #22): the sink's
        // `line` appends and `tick` redraws, so a long-running command shows
        // progress instead of dumping everything at exit. Both halves need
        // `&mut log`, which is why this is one sink and not two closures.
        struct Sink<'a, 'b> {
            log: &'a mut OutputLog,
            terminal: &'a mut ratatui::DefaultTerminal,
            view: &'a mut tui::OutputView,
            cmd: &'b str,
            start: Instant,
            dirty: bool,
        }
        impl crate::tools::bash::ImmediateSink for Sink<'_, '_> {
            fn line(&mut self, _stream: crate::tools::bash::Stream, text: &str) {
                self.log.push_dim(text.to_owned());
                self.dirty = true;
            }
            fn tick(&mut self) -> bool {
                let status = format!(
                    "! {} ({}s, Esc to stop)",
                    self.cmd,
                    self.start.elapsed().as_secs()
                );
                let (log, view) = (&*self.log, &mut *self.view);
                let _ = self.terminal.draw(|f| {
                    tui::draw(
                        f,
                        log,
                        None,
                        &status,
                        view,
                        None,
                        &tui::TaskView::default(),
                        None,
                    );
                });
                self.dirty = false;
                while event::poll(Duration::ZERO).unwrap_or(false) {
                    if let Ok(Event::Key(k)) = event::read()
                        && k.kind == KeyEventKind::Press
                        && (matches!(k.code, KeyCode::Esc)
                            || (matches!(k.code, KeyCode::Char('c'))
                                && k.modifiers.contains(KeyModifiers::CONTROL)))
                    {
                        return true;
                    }
                }
                false
            }
        }
        let start = Instant::now();
        let mut sink = Sink {
            log,
            terminal,
            view,
            cmd,
            start,
            dirty: false,
        };
        let result = crate::tools::bash::run_immediate(cwd, cmd, &mut sink);
        match &result {
            Ok(out) => {
                if out.interrupted {
                    log.push_dim("[interrupted]");
                } else {
                    if out.exit_code != 0 {
                        log.push_dim(format!("[exit code: {}]", out.exit_code));
                    }
                    // A command that prints nothing is otherwise indis-
                    // tinguishable from one still running, so say it finished.
                    // Only when it did: an interrupted command did not.
                    log.push_spans(vec![ratatui::text::Span::styled(
                        "done.",
                        crate::tui::done_style(),
                    )]);
                }
            }
            Err(e) => log.push_dim(format!("!{cmd}: {e}")),
        }
        result
    }

    /// Plans the KV cache tiers below the system prompt for this launch
    /// (issue #64): Tier 2 (project-stable context, checkpointed per project at
    /// `kvcache/<project-key>/project-<fp2>.kv`) and Tier 3 (session-volatile
    /// context, prefill-only).
    ///
    /// Tier 2's key folds in the **project-local** MCP tool definitions but not
    /// the global ones — those already live inside the system prompt that keys
    /// Tier 1, and moving them down would needlessly fork Tier 2 while moving
    /// local ones up would fork the model-global Tier 1 per project.
    fn kv_tiers(&self) -> Vec<crate::kvtier::TierSpec> {
        self.kv_tiers_for(&self.engine.model_name())
    }

    /// [`kv_tiers`](Self::kv_tiers) for an engine other than the live one.
    ///
    /// The model name is a parameter rather than read off `self.engine` because
    /// a checkpoint belongs to the *model whose KV it holds*: warming the local
    /// alt engine under a provider main agent must key on `ds4`, not on the
    /// provider's model, or it would look up a checkpoint that cannot describe
    /// its KV. Keying it correctly is also what lets it share Tier 1 with an
    /// ordinary local-main session — which is where most of those checkpoints
    /// get written.
    fn kv_tiers_for(&self, model: &str) -> Vec<crate::kvtier::TierSpec> {
        let fp1 = crate::kvtier::system_fingerprint(
            model,
            &self.system,
            self.think,
            self.trusted_system_len,
        );
        let local_names = crate::tools::mcp::local_server_names(None);
        let local_defs = crate::tools::mcp::local_tool_defs(&self.tool_ctx.mcp, &local_names);
        let local_material = crate::kvtier::tool_defs_material(&local_defs);
        crate::kvtier::plan(
            &fp1,
            &self.system,
            &self.context_content.stable_context(),
            &self.context_content.volatile_context(),
            &local_material,
            Some(&self.tool_ctx.cwd),
        )
    }

    /// Drops superseded Tier 2 checkpoints for this project once the current
    /// one is known good; they can never be read again and are large.
    ///
    /// Every *live* Tier 1 fingerprint is kept, not just the main engine's. A
    /// `provider: local` sub-agent has its own — keyed on the local model, not
    /// on the provider's — and collecting against the main engine's alone
    /// deleted it on every launch, which is why the sub-agent re-prefilled its
    /// system prompt every single run. Under a provider it was worse than that:
    /// the provider's fingerprint never has a file, so *nothing* matched the
    /// keep and the whole directory of system checkpoints went, including ones
    /// belonging to ordinary local sessions.
    fn gc_kv_tiers(&self, tiers: &[crate::kvtier::TierSpec]) {
        let mut keep: Vec<String> = tiers
            .iter()
            .filter(|t| t.kind == crate::kvtier::TierKind::System)
            .map(|t| t.fingerprint.clone())
            .collect();
        if let Some(alt) = self.alt_engines.get(&EngineKey::Local) {
            keep.extend(
                self.kv_tiers_for(&alt.model_name())
                    .into_iter()
                    .filter(|t| t.kind == crate::kvtier::TierKind::System)
                    .map(|t| t.fingerprint),
            );
        }
        if !keep.is_empty() {
            let keep: Vec<&str> = keep.iter().map(String::as_str).collect();
            let _removed = self.store.gc_system_checkpoints(&keep);
        }
        if let Some(t) = tiers
            .iter()
            .find(|t| t.kind == crate::kvtier::TierKind::ProjectStable)
        {
            let _removed = self
                .store
                .gc_project_checkpoints(&self.tool_ctx.cwd, &t.fingerprint);
        }
    }

    /// Re-establishes the warm KV prefix after the transcript is reset by
    /// `/new` or `/clear`.
    ///
    /// A reset makes the next prompt a strict *prefix* of the live KV (the fresh
    /// transcript is the old one's head: same system prompt, same session
    /// context). `ds4_session_sync` cannot rewrite behind its live end, so it
    /// would silently discard the whole KV and re-prefill the system prompt from
    /// scratch — thousands of tokens, with the progress bar primed as complete
    /// because every token "matched". Restoring the tier checkpoint instead
    /// installs a genuine frontier snapshot at the warm boundary, so the next
    /// turn extends it and prefills only the new question.
    ///
    /// Best-effort and silent: on any failure the next turn just pays the
    /// rebuild it would have paid anyway.
    ///
    /// `on_progress` is called before the walk starts and again for every engine
    /// event, so a front-end can keep an indicator alive. It is *not* instant:
    /// restoring the tier checkpoint reads a snapshot that runs to tens of
    /// megabytes and loads it into the backend.
    fn rewarm_after_reset(&mut self, on_progress: &mut dyn FnMut()) {
        let tiers = self.kv_tiers();
        // Paint once up front: the restore leg is a single blocking read+load
        // that emits no events, so without this the indicator would never
        // appear in the common (checkpoint present) case.
        on_progress();
        let _ = crate::kvtier::warm(
            &mut *self.engine,
            Some(&self.store),
            &tiers,
            &mut |_| on_progress(),
            &mut |_| {},
        );
    }

    /// Warms the KV cache — system prompt and session-start context tiers — in
    /// one walk before the full TUI is shown. When the cache is already current
    /// nothing prefills and nothing is drawn; otherwise a minimal centered
    /// progress bar covers the rebuild, and the caller renders the real UI over
    /// it.
    /// Warms the alt local engine's Tier 1 at startup: prefilled and persisted
    /// when no checkpoint exists, restored from disk when one does.
    ///
    /// Here rather than at first use because this is the one place a prefill is
    /// already expected, drawn, and paid for — mid-turn it froze the front end,
    /// `warm_sync` being uninterruptible.
    ///
    /// **Only Tier 1**, which is also what makes the checkpoint appear at all.
    /// `warm` restores the deepest tier that loads and skips every tier above
    /// it, so on a machine whose Tier 2 checkpoint is valid, Tier 1 is never
    /// prefilled and therefore never persisted — which is why `sysprompt-*.kv`
    /// can be absent on a session that has warmed happily for months. A tier
    /// list of one has nothing deeper to short-circuit it, so Tier 1 gets built
    /// and written, and every later session (this engine's sidechains and any
    /// local-main session on the same fingerprint) restores instead.
    fn warm_alt_local_tier1(
        &mut self,
        on_event: &mut dyn FnMut(EngineEvent),
        on_stage: &mut dyn FnMut(crate::kvtier::TierKind),
    ) {
        let Some(mut engine) = self.alt_engines.remove(&EngineKey::Local) else {
            return;
        };
        let tiers = self.kv_tiers_for(&engine.model_name());
        if let Some(system) = tiers
            .first()
            .filter(|t| t.kind == crate::kvtier::TierKind::System)
        {
            let _ = crate::kvtier::warm(
                &mut *engine,
                Some(&self.store),
                std::slice::from_ref(system),
                on_event,
                on_stage,
            );
            // Set even on failure: the on-demand restore would fail the same
            // way, and retrying it per dispatch buys nothing.
            self.local_alt_warmed = true;
        }
        self.alt_engines.insert(EngineKey::Local, engine);
    }

    fn tui_warm(&mut self, terminal: &mut ratatui::DefaultTerminal) -> Result<(), String> {
        if self.skip_warm_after_restore() {
            return Ok(());
        }
        let tiers = self.kv_tiers();
        // The rebuild reason arrives as a Notice before prefill; keep it and
        // render it below the bar.
        let mut notice: Option<String> = None;
        // Which tier the bar is covering right now. A `Cell` because the stage
        // and event callbacks both need it and only one of them may hold the
        // terminal mutably; the stage callback only records the label, and the
        // prefill events that follow immediately paint it.
        let stage = std::cell::Cell::new(crate::kvtier::TierKind::System.warm_label());
        crate::kvtier::warm(
            &mut *self.engine,
            Some(&self.store),
            &tiers,
            &mut |ev| match ev {
                EngineEvent::Notice(msg) => {
                    notice = Some(msg);
                    let _ = terminal
                        .draw(|f| tui::draw_warm(f, 0, 1, 0.0, stage.get(), notice.as_deref()));
                }
                EngineEvent::Prefill(p) => {
                    let _ = terminal.draw(|f| {
                        tui::draw_warm(f, p.done, p.total, p.tps, stage.get(), notice.as_deref());
                    });
                }
                EngineEvent::Text(_) => {}
            },
            &mut |kind| stage.set(kind.warm_label()),
        )
        .map_err(|e| e.to_string())?;
        // The sub-agent's local engine gets the same treatment on the same
        // screen; it is a second engine, so it needs its own walk.
        let alt_stage = "Caching the system prompt for the local sub-agent";
        self.warm_alt_local_tier1(
            &mut |ev| {
                if let EngineEvent::Prefill(p) = ev {
                    let _ = terminal.draw(|f| {
                        tui::draw_warm(f, p.done, p.total, p.tps, alt_stage, None);
                    });
                }
            },
            &mut |_| {},
        );
        self.gc_kv_tiers(&tiers);
        Ok(())
    }

    /// Whether the startup warm walk must be skipped because a session payload
    /// was already restored.
    ///
    /// The walk's last act for each cacheable tier is `set_kv` on that tier's
    /// checkpoint — whose transcript is empty by construction, since a tier has
    /// no conversation in it. Running that *after* a session restore is strictly
    /// destructive: it rewinds the live KV from the end of the transcript back to
    /// the session-context boundary and clears the token transcript, so the next
    /// turn re-prefills every conversation token. Measured at 165 tokens on a
    /// two-turn session; it scales with the whole conversation.
    ///
    /// The payload is a superset of every tier prefix — it was captured from a
    /// session that had already been warmed — so there is nothing left to warm.
    fn skip_warm_after_restore(&self) -> bool {
        self.payload_restored
    }

    /// Warms the KV cache for non-TUI runs, announcing a rebuild on stderr.
    fn warm_plain(&mut self) -> Result<(), String> {
        if self.skip_warm_after_restore() {
            return Ok(());
        }
        let tiers = self.kv_tiers();
        let color = self.color;
        // Which tier is prefilling, and the label last printed for it: each
        // stage announces itself once, so the note names the work actually
        // running rather than always the system prompt.
        let stage = std::cell::Cell::new(crate::kvtier::TierKind::System.warm_label());
        let mut shown: Option<&'static str> = None;
        let mut announced = 0usize;
        let mut notice: Option<String> = None;
        crate::kvtier::warm(
            &mut *self.engine,
            Some(&self.store),
            &tiers,
            &mut |ev| match ev {
                EngineEvent::Notice(msg) => notice = Some(msg),
                EngineEvent::Prefill(_) if shown != Some(stage.get()) => {
                    let label = stage.get();
                    shown = Some(label);
                    announced += 1;
                    if color {
                        eprintln!("\x1b[33m{label}...{ANSI_RESET}");
                    } else {
                        eprintln!("{label}...");
                    }
                    // The rebuild reason belongs to the first (Tier 1) note
                    // only; later stages must not reprint it.
                    if let Some(msg) = notice.as_ref().filter(|_| announced == 1) {
                        for line in msg.lines() {
                            // Match the code-diff card colors on the -/+ rows.
                            let colored = match (color, line.as_bytes().first()) {
                                (true, Some(b'-')) => {
                                    format!("\x1b[48;5;52m\x1b[38;5;224m{line}{ANSI_RESET}")
                                }
                                (true, Some(b'+')) => {
                                    format!("\x1b[48;5;22m\x1b[38;5;194m{line}{ANSI_RESET}")
                                }
                                _ => line.to_owned(),
                            };
                            eprintln!("  {colored}");
                        }
                    }
                }
                EngineEvent::Prefill(_) | EngineEvent::Text(_) => {}
            },
            &mut |kind| stage.set(kind.warm_label()),
        )
        .map_err(|e| e.to_string())?;
        // Erase the transient "Updating…" note. Only when a single stage
        // announced itself and printed nothing under it; reason lines, if any,
        // stay in the scrollback on purpose.
        if announced == 1 && color && notice.is_none() {
            eprint!("\x1b[A\x1b[2K\r");
        }
        let mut alt_announced = false;
        self.warm_alt_local_tier1(
            &mut |ev| {
                if matches!(ev, EngineEvent::Prefill(_)) && !alt_announced {
                    alt_announced = true;
                    eprintln!("Caching the system prompt for the local sub-agent...");
                }
            },
            &mut |_| {},
        );
        self.gc_kv_tiers(&tiers);
        Ok(())
    }

    fn idle_status_text(&mut self) -> String {
        let st = self.idle_status();
        status::build_status_text(&st, false, true)
    }

    /// The between-turns status snapshot: idle, with the context gauge for the
    /// transcript as it now stands.
    fn idle_status(&mut self) -> Status {
        let rendered = render_transcript(&self.session, &self.system);
        Status {
            state: WorkerState::Idle,
            ctx_used: self.engine.count_tokens(&rendered),
            ctx_size: self.engine.ctx_size(),
            power_percent: self.power_percent,
            think: self.think,
            ..Status::default()
        }
    }

    /// Publishes the idle snapshot to attached remote clients at the end of a
    /// turn. Status frames otherwise come only from engine callbacks *during* a
    /// turn, so without this the last thing a remote ever sees is
    /// `generating` — its context gauge freezes and anything keyed off "a turn
    /// is running" (the page's stop button) stays stuck on. Skipped with no
    /// bridge, since building it re-renders the transcript to count tokens.
    fn broadcast_idle_status(&mut self) {
        if self.remote.is_none() {
            return;
        }
        let st = self.idle_status();
        if let Some(r) = &self.remote {
            r.bus.broadcast(UiEvent::Status(st));
        }
    }

    /// One TUI turn: runs the generate → tools loop on a worker thread while
    /// the UI thread keeps the terminal live (typing, scrolling, interrupts),
    /// then feeds user lines queued during the turn into follow-up turns.
    #[allow(clippy::too_many_arguments)]
    fn tui_turn(
        &mut self,
        terminal: &mut ratatui::DefaultTerminal,
        log: &mut OutputLog,
        view: &mut tui::OutputView,
        input: &mut TuiInput,
        btw: &mut BtwPanel,
        arcade: &mut crate::arcade::Arcade,
        sub: &mut tui::SubPane,
    ) -> Result<(), String> {
        // The first iteration runs the main turn; later iterations run either
        // a follow-up turn (leftover queued user lines) or a btw-only drain
        // (side questions queued after the worker's final boundary).
        // With a remote bridge, share its persistent `TurnShared` so remote
        // `prompt`/`btw`/`interrupt` frames land in the same queues the local
        // editor uses, and mirror every event onto its bus (issue #25).
        // Stamped once at the outermost per-user-turn boundary for the TUI
        // front end: `tui_turn` is called exactly once per user submission
        // (see call sites in `tui_loop`/`busy_ui_loop`), even though its own
        // inner loop may run extra rounds for leftover queued lines or a
        // btw-only drain.
        let turn_started = Instant::now();
        let remote = self.remote.clone();
        let bus = remote.as_ref().map(|r| Arc::clone(&r.bus));
        // Same remote-control state the idle loop uses: a turn started by an
        // injected Enter must keep servicing the deferred snapshot/uitree.
        let ui_remote = self.ui_remote.clone();
        let rem = ui_remote.as_deref();
        let mut run_main = true;
        let mut carry_btw: Vec<String> = Vec::new();
        loop {
            let local_shared = TurnShared::default();
            let shared: &TurnShared = remote
                .as_deref()
                .map_or(&local_shared, |r| r.shared.as_ref());
            for q in carry_btw.drain(..) {
                let _ = shared.push_btw(q);
            }
            let bus_ref = bus.as_deref();
            // UI-side handle to the `ask` rendezvous (issue #34), cloned out of
            // the tool context before the closure borrows `self`. Only the main
            // turn dispatches tools (and thus `ask`); the btw drain never does.
            let ask_bridge = self.tool_ctx.ask_bridge.clone();
            // Snapshot the read-only reports so `/context` & co. stay usable
            // while the worker owns the engine for this turn.
            let live = LiveCommands::capture(self);
            if run_main {
                run_worker_ui(
                    terminal,
                    log,
                    view,
                    input,
                    btw,
                    arcade,
                    sub,
                    shared,
                    bus_ref,
                    rem,
                    ask_bridge.as_ref(),
                    &live,
                    |tx| self.worker_turn(&tx, shared),
                )??;
            } else {
                run_worker_ui(
                    terminal,
                    log,
                    view,
                    input,
                    btw,
                    arcade,
                    sub,
                    shared,
                    bus_ref,
                    rem,
                    None,
                    &live,
                    |tx| {
                        self.drain_btw(&tx, shared);
                    },
                )?;
            }
            // Lines typed while busy that no tool round drained become the
            // next turn's user message(s), as if resubmitted by hand.
            let leftover = shared.take_queued();
            carry_btw = shared.take_btw();
            if leftover.is_empty() && carry_btw.is_empty() {
                // Closing line, before anything switches to idle: `turn_started`
                // covers the whole user turn — every generate/tools round, plus
                // any leftover-queued follow-ups this loop absorbed — not the
                // last pass, which is what the status bar's elapsed shows while
                // the turn runs.
                //
                // Blank line first: the footer is a boundary marker, and butted
                // against the reply's last line it reads as part of it.
                log.push_plain("");
                log.push_spans(vec![ratatui::text::Span::styled(
                    tui::turn_footer(turn_started.elapsed()),
                    tui::turn_footer_style(),
                )]);
                self.broadcast_idle_status();
                if crate::notify::should_notify_complete(
                    turn_started.elapsed(),
                    crate::settings::active().ui.notify_after_secs,
                ) {
                    self.notify_task_complete();
                }
                // Turn over: the front end is back at the prompt.
                crate::title::set(crate::title::State::Idle);
                crate::warp::emit("stop", &self.session.id);
                return Ok(());
            }
            run_main = !leftover.is_empty();
            for line in leftover {
                self.session.push(Message::user(line));
            }
        }
    }

    /// Tells attached remote clients the transcript was replaced, so they clear
    /// their log instead of appending a new session under an old one. Also
    /// drops the bus scrollback, so a client attaching *after* the reset is not
    /// replayed the transcript that was just cleared. A no-op with no bridge.
    ///
    /// Call it wherever `self.session` is replaced wholesale — `/clear`,
    /// `/new`, `/switch`, `/resume` — right after the swap.
    fn broadcast_session_reset(&self, note: Option<&str>) {
        let Some(r) = &self.remote else { return };
        r.bus.broadcast(UiEvent::SessionReset);
        // `/switch` and `/resume` replay the loaded transcript into the *local*
        // log directly, not through the bus, so a remote client would be left
        // looking at an empty page. Say why rather than leave it blank.
        if let Some(note) = note {
            r.bus.broadcast(UiEvent::Dim(note.to_owned()));
        }
    }

    /// Whether a remote-control bridge is currently live.
    fn remote_is_on(&self) -> bool {
        self.remote_server.is_some()
    }

    /// The one-click browser link for a bound server: the token rides in the
    /// query string, which `serve_http` strips before routing, so the page is
    /// served normally and reads the token from `location.search`. On a
    /// loopback-only listener whose lifetime is one toggle this is an accepted
    /// trade for one-click attach (spec §6).
    ///
    /// The host is written out rather than taken from `addr`, which is sound
    /// only because the bind is always loopback — see [`Agent::remote_on`].
    fn remote_link(addr: std::net::SocketAddr, token: &str) -> String {
        format!("http://127.0.0.1:{}/?t={token}", addr.port())
    }

    /// Starts the remote-control server and installs it on this agent. Returns
    /// the bound address and the token clients must present. Idempotent: with a
    /// server already live the existing address and token come back unchanged.
    ///
    /// `allow_control` seeds the control policy: `true` lets an attaching client
    /// take control without a local `/grant`, which is what makes the
    /// `/remote-control` link usable while the local TUI holds the slot.
    ///
    /// # Errors
    /// Returns the bind error as a string; `self` is left untouched on failure.
    fn remote_on(
        &mut self,
        addr: &str,
        token: Option<String>,
        allow_control: bool,
    ) -> Result<(std::net::SocketAddr, String), String> {
        if let Some(server) = &self.remote_server {
            return Ok((server.local_addr, server.state.token.clone()));
        }
        // Loopback is not merely the default here, it is load-bearing:
        // `remote_link` writes `127.0.0.1` into the printed URL rather than
        // reading it back from the bound address, so a non-loopback bind would
        // hand out a link pointing somewhere else entirely.
        debug_assert!(
            addr.starts_with("127.0.0.1:") || addr.starts_with("[::1]:"),
            "remote control binds loopback only, got {addr}"
        );
        let token = token
            .filter(|t| !t.is_empty())
            .unwrap_or_else(crate::remote::generate_token);
        // Loopback-only, so no browser Origin allow-list is needed: a missing or
        // loopback Origin is always accepted. The queue cap keeps its default.
        let server_cfg = crate::remote::control::ServerConfig {
            token: token.clone(),
            // `local_present: true` is unconditional because `/rc` can only be typed
            // in the TUI: a headless session has no slash dispatch and no way to
            // install a bridge. Restoring a headless path means computing this from
            // whether a local front-end actually exists.
            local_present: true,
            allow_control,
            allowed_origins: Vec::new(),
            queue_max: crate::config::DEFAULT_CONTROL_QUEUE_MAX,
        };
        let server = crate::remote::RemoteServer::start(
            addr,
            server_cfg,
            Arc::new(crate::worker::BroadcastBus::new()),
            Arc::new(TurnShared::default()),
        )
        .map_err(|e| e.to_string())?;
        let bound = server.local_addr;
        self.remote = Some(Arc::clone(&server.state));
        self.remote_server = Some(server);
        Ok((bound, token))
    }

    /// Stops the remote-control server and clears the bridge. Connected clients
    /// get a `bye` first. Returns whether a server was running. The token dies
    /// with the server, so an old link is refused by the next one.
    fn remote_off(&mut self) -> bool {
        let Some(mut server) = self.remote_server.take() else {
            return false;
        };
        self.remote = None;
        server.state.say_bye("remote control turned off");
        server.shutdown();
        true
    }

    /// Applies a `/remote-control` toggle and returns the lines to show. `cmd`
    /// is the invoked command name (`/remote-control` or `/rc`), used to name
    /// the command in error messages. `arg` is `""` (toggle), `"on"`, or
    /// `"off"` (case-insensitive); anything else reports usage.
    ///
    /// Starting from here always uses an ephemeral loopback port, so the command
    /// never collides with another plank or a stale listener, and always sets
    /// `allow_control`: the operator typing the command is the consent that a
    /// remote-side allow flag would otherwise encode.
    fn remote_toggle_lines(&mut self, cmd: &str, arg: &str) -> Vec<String> {
        let want_on = if arg.is_empty() {
            !self.remote_is_on()
        } else if arg.eq_ignore_ascii_case("on") {
            true
        } else if arg.eq_ignore_ascii_case("off") {
            false
        } else {
            return vec![format!(
                "{cmd}: unknown argument {arg:?} (use on, off, or no argument)"
            )];
        };
        if !want_on {
            return if self.remote_off() {
                vec!["remote control off".to_owned()]
            } else {
                vec!["remote control is already off".to_owned()]
            };
        }
        match self.remote_on(crate::remote::LOOPBACK_EPHEMERAL, None, true) {
            Ok((addr, token)) => {
                let port = addr.port();
                vec![
                    format!("remote control on — {}", Self::remote_link(addr, &token)),
                    format!("tunnel:  ssh -L {port}:localhost:{port} user@thishost"),
                ]
            }
            Err(e) => vec![format!("{cmd}: could not start: {e}")],
        }
    }

    /// Worker-side turn loop (the C's `worker_run_turn`): generate, dispatch
    /// tools, drain queued user lines between rounds, repeat until settled.
    /// Runs on the worker thread and talks to the UI only through `tx`.
    #[allow(clippy::too_many_lines)] // flat generate→tools loop; splitting hurts readability
    fn worker_turn(&mut self, tx: &Sender<UiEvent>, shared: &TurnShared) -> Result<(), String> {
        // TUI: a headless sub-agent's output is forwarded over the same
        // worker→UI channel as the parent turn's own render events.
        self.sub_sink = SubSinkTarget::Events(tx.clone());
        crate::title::set(crate::title::State::Busy(self.last_user_prompt()));
        self.last_turn_interrupted = false;
        self.tool_ctx.skill_invocations = 0;
        self.tool_ctx.tasks.clone_from(&self.session.tasks);
        let mut note = |s: String| {
            let _ = tx.send(UiEvent::Dim(s));
        };
        // Tools publish system status through the same channel as render
        // events, so a "Searching Google for ..." notice lands in the log in
        // the order it happened. Reinstalled per turn: `tx` is per-turn.
        self.tool_ctx.status_sink = Some({
            let tx = tx.clone();
            Box::new(move |msg: &str| {
                let _ = tx.send(UiEvent::SystemStatus(msg.to_owned()));
            })
        });
        if let Some(reason) = self.fire_user_prompt_submit(&mut |w| {
            let _ = tx.send(UiEvent::Dim(w));
        }) {
            let _ = tx.send(UiEvent::Dim(format!("halted by hook: {reason}")));
            return Ok(());
        }
        let compact_interrupt =
            || shared.interrupt.load(Ordering::Relaxed) || crate::interrupt::pending();
        // No redraw hook: the UI thread paints the compaction bar off its own
        // clock while this worker thread compacts.
        if self
            .maybe_compact_notify(&mut NoteSink(&mut note), &compact_interrupt)?
            .aborted()
        {
            // Consume the interrupt so the next turn starts clean, then go
            // back to idle with the conversation untouched (`worker_run_turn`).
            shared.interrupt.store(false, Ordering::Relaxed);
            return Ok(());
        }
        self.maybe_reminder_notify(&mut note);
        // One clock for the whole turn: elapsed time accumulates across the
        // generate → tools → generate loop instead of restarting per pass.
        let turn_start = Instant::now();
        // Stop hooks run at most once per turn, so a hook that always exits 2
        // cannot loop the model forever.
        let mut stop_hook_ran = false;
        loop {
            let base_prompt = render_transcript(&self.session, &self.system);
            // Text already streamed for this pass and preserved across in-pass
            // `/btw` suspensions (BTW-SUSPEND-DESIGN §4.3). Empty unless the
            // pass was frozen and resumed at least once.
            let mut resumed_prefix = String::new();
            let suspend_enabled = self.cfg.btw.suspend && self.engine.supports_aside();
            let out = loop {
                // On resume, re-open the assistant turn with the partial reply
                // so the engine splices its exact tokens (zero re-prefill) and
                // continues from where it froze; otherwise the plain prompt.
                let prompt = if resumed_prefix.is_empty() {
                    base_prompt.clone()
                } else {
                    format!("{base_prompt}[assistant]\n{resumed_prefix}")
                };
                let out = self.worker_generate(tx, shared, &prompt, turn_start, true)?;
                if out.preempted && suspend_enabled {
                    let _ = tx.send(UiEvent::EndLine);
                    resumed_prefix.push_str(&out.assistant_text);
                    // The aside is asked *over* the paused reply, which is what
                    // keeps its prompt an extension of the live KV.
                    if let Some(framed) = self.multiplexable_aside(tx, shared, &resumed_prefix) {
                        // Nothing pauses: the next pass carries this aside
                        // alongside the main continuation, so the suspend and
                        // resume markers would be lying about a freeze that
                        // never happens.
                        //
                        // Open the panel and immediately leave its routing
                        // mode. The mode would swallow the main task's events
                        // too — the two streams interleave, so the aside's
                        // events address the panel individually instead
                        // (`UiEvent::Btw`).
                        let _ = tx.send(UiEvent::BtwBegin);
                        let _ = tx.send(UiEvent::BtwEnd);
                        self.pending_aside = Some(framed);
                    } else {
                        // Freeze: keep the partial on screen, answer the queued
                        // aside(s) on the paused session, then resume the pass.
                        let _ = tx.send(UiEvent::Dim(worker::BTW_SUSPEND_MARKER.to_owned()));
                        self.drain_aside(tx, shared, &resumed_prefix);
                        let _ = tx.send(UiEvent::Dim(worker::BTW_RESUME_MARKER.to_owned()));
                    }
                    continue;
                }
                break out;
            };
            // A priority `/btw` stopped this pass without suspend support:
            // nothing was committed, so roll back the partial output, answer
            // the side question(s) at the boundary, and re-run the same step.
            if out.preempted {
                let _ = tx.send(UiEvent::MainRollback);
                self.drain_btw(tx, shared);
                continue;
            }
            // Splice any suspended-and-resumed prefix back onto the final
            // continuation so the transcript holds the whole reply.
            let mut assistant_text = if resumed_prefix.is_empty() {
                out.assistant_text
            } else {
                format!("{resumed_prefix}{}", out.assistant_text)
            };
            // A real interrupt never continues with a <tool_result>, even
            // when the partial stanza it cut off reads as a parse error.
            let turn_continues = !out.interrupted && (!out.calls.is_empty() || out.error.is_some());
            close_open_think(&mut assistant_text, out.ended_in_think && turn_continues);
            self.session.push(Message::assistant(assistant_text));
            let _ = tx.send(UiEvent::EndLine);
            if out.interrupted {
                crate::interrupt::clear();
                self.last_turn_interrupted = true;
                let _ = tx.send(UiEvent::Dim("[interrupted]".to_owned()));
                // Drain point 3 (BTW-DESIGN §4.4): the user asked mid-turn;
                // answer even though the main turn was interrupted.
                self.drain_btw(tx, shared);
                return Ok(());
            }
            // Side questions answer at every generation boundary, before the
            // next tool dispatch (BTW-DESIGN §4.4 drain points 1 and 2).
            self.drain_btw(tx, shared);
            if let Some(payload) = out.error {
                self.session.push(Message::user(format!(
                    "<tool_result>{payload}</tool_result>"
                )));
                self.drain_queued(shared, tx);
                continue;
            }
            if !out.calls.is_empty() {
                let observations = self.run_tool_calls(&out.calls);
                self.sync_tasks_after_dispatch();
                for preview in std::mem::take(&mut self.tool_ctx.edit_previews) {
                    let _ = tx.send(UiEvent::EditCard(preview));
                }
                for line in std::mem::take(&mut self.tool_ctx.task_completions) {
                    let _ = tx.send(UiEvent::Dim(format!("✓ {line}")));
                }
                let _ = tx.send(UiEvent::Tasks(tui::TaskView::from(&self.session.tasks)));
                for warning in self.tool_ctx.hook_warnings.drain(..) {
                    let _ = tx.send(UiEvent::Dim(warning));
                }
                self.session.push(Message::user(format!(
                    "<tool_result>{observations}</tool_result>"
                )));
                if crate::settings::active().ui.show_tool_results {
                    for line in observations.lines() {
                        let _ = tx.send(UiEvent::Dim(line.to_owned()));
                    }
                }
                // A tool hook's `continue:false` envelope halts the turn.
                if let Some(reason) = self.tool_ctx.hook_stop.take() {
                    let _ = tx.send(UiEvent::Dim(format!("halted by hook: {reason}")));
                    return Ok(());
                }
                self.drain_queued(shared, tx);
                continue;
            }
            // Stop hooks: exit 2 feeds stderr to the model and the turn
            // continues (at most once).
            if !stop_hook_ran {
                let mut warnings = Vec::new();
                let feedback = self.run_stop_hooks(&mut |w| warnings.push(w));
                for w in warnings {
                    let _ = tx.send(UiEvent::Dim(w));
                }
                if let Some(feedback) = feedback {
                    stop_hook_ran = true;
                    let _ = tx.send(UiEvent::Dim("[Stop hook] continuing the turn".to_owned()));
                    self.session.push(Message::user(format!(
                        "<tool_result>Stop hook feedback:\n{feedback}</tool_result>"
                    )));
                    continue;
                }
            }
            return Ok(());
        }
    }

    /// Answers queued `/btw` side questions FIFO at a generation boundary
    /// (worker thread). Each answer is one tool-free pass over the live
    /// transcript plus the framed question; nothing enters the session and
    /// `last_ctx_used` is restored, so the side exchange is rolled back by
    /// the next real pass's prefix sync. An interrupt during an answer
    /// flushes the rest of the queue (the user is saying "stop the asides");
    /// a failed answer is logged and the queue continues — side questions
    /// must never abort the main turn.
    ///
    /// While answering, `BtwBegin`/`BtwEnd` bracket the render events so the
    /// UI opens a side panel (main conversation 60%, `/btw` 40%). The drain
    /// answers every queued question FIFO and then **returns** — it does not
    /// wait for the panel to be dismissed, so the main task resumes as soon as
    /// the answer is done. `BtwEnd` only stops routing to the panel; the UI
    /// keeps it on screen (frozen, readable) until the user presses Esc.
    fn drain_btw(&mut self, tx: &Sender<UiEvent>, shared: &TurnShared) {
        self.drain_btw_inner(tx, shared, false, "");
    }

    /// Suspend-mode drain: answers the queued `/btw` question(s) on a session
    /// that still holds the frozen main-task KV (BTW-SUSPEND-DESIGN §4.3).
    /// Used only from the in-pass suspend path, where the main pass is paused
    /// mid-reply and the partial reply must survive the aside.
    ///
    /// `frozen_partial` is the text the paused pass had already produced. It
    /// must be spliced into the aside's prompt, exactly as the resume splices
    /// it: those tokens are live in the KV, and `ds4_session_sync` reuses the
    /// KV *only* for a prompt that extends its end (see
    /// [`crate::engine::reusable_prefix`]). Leaving it out makes the prompt
    /// diverge behind the live end, which silently re-prefills the entire
    /// conversation — BTW-SUSPEND-DESIGN §4.3 step 2 assumed a cursor rollback
    /// the engine cannot perform.
    fn drain_aside(&mut self, tx: &Sender<UiEvent>, shared: &TurnShared, frozen_partial: &str) {
        self.drain_btw_inner(tx, shared, true, frozen_partial);
    }

    fn drain_btw_inner(
        &mut self,
        tx: &Sender<UiEvent>,
        shared: &TurnShared,
        aside: bool,
        frozen_partial: &str,
    ) {
        let Some(mut question) = shared.pop_btw() else {
            return;
        };
        let _ = tx.send(UiEvent::BtwBegin);
        // A stale interrupt (e.g. the preempt path) must not cancel the answer
        // before the user has seen anything.
        shared.interrupt.store(false, Ordering::Relaxed);
        loop {
            let _ = tx.send(UiEvent::UserEcho(format!("/btw {question}")));
            let _ = tx.send(UiEvent::Dim("[btw]".to_owned()));
            let saved_ctx = self.last_ctx_used;
            let mut prompt = render_transcript(&self.session, &self.system);
            {
                use std::fmt::Write as _;
                // Close the paused assistant turn with what it had produced, so
                // the prompt extends the live KV instead of diverging behind it.
                if !frozen_partial.trim().is_empty() {
                    let _ = write!(prompt, "[assistant]\n{}\n", frozen_partial.trim_end());
                }
                let _ = write!(prompt, "[user]\n{}\n", btw_user_message(&question));
            }
            match self.worker_generate_kind(tx, shared, &prompt, Instant::now(), false, aside) {
                Ok(out) => {
                    let _ = tx.send(UiEvent::EndLine);
                    self.last_ctx_used = saved_ctx;
                    if out.interrupted {
                        // Esc during a streaming answer: cancel it and flush
                        // the rest of the queue.
                        crate::interrupt::clear();
                        let _ = tx.send(UiEvent::Dim("[interrupted]".to_owned()));
                        let cleared = shared.clear_btw();
                        if cleared > 0 {
                            let _ =
                                tx.send(UiEvent::Dim(format!("[btw queue cleared: {cleared}]")));
                        }
                        break;
                    }
                    if !out.calls.is_empty() || out.error.is_some() {
                        let _ = tx.send(UiEvent::Dim(
                            "(the model tried to call a tool; tools are disabled during /btw — ask in the main conversation)"
                                .to_owned(),
                        ));
                    }
                    let _ = tx.send(UiEvent::Dim(
                        "[btw — not part of the conversation]".to_owned(),
                    ));
                }
                Err(e) => {
                    let _ = tx.send(UiEvent::Dim(format!("/btw failed: {e}")));
                    self.last_ctx_used = saved_ctx;
                }
            }
            let Some(next) = shared.pop_btw() else {
                break;
            };
            question = next;
        }
        // Consume any cancelling Esc so the resumed main task is not itself
        // interrupted by it.
        shared.interrupt.store(false, Ordering::Relaxed);
        crate::interrupt::clear();
        let _ = tx.send(UiEvent::BtwEnd);
    }

    /// Moves user lines queued during the turn into the transcript between
    /// tool rounds, mirroring the C's `queued_user_drain`.
    fn drain_queued(&mut self, shared: &TurnShared, tx: &Sender<UiEvent>) {
        for line in shared.take_queued() {
            let _ = tx.send(UiEvent::Dim("[queued message joined the turn]".to_owned()));
            self.session.push(Message::user(line));
        }
    }

    /// Streams one generation pass on the worker thread, forwarding rendered
    /// output and status snapshots to the UI over `tx`.
    ///
    /// `turn_start` is when the user submitted the prompt: the status bar's
    /// elapsed time counts the whole turn (all generation passes and tool
    /// runs), not just this pass. Tokens/s stays per-pass.
    ///
    /// `is_main` marks a main-task pass (vs. a `/btw` side answer): only main
    /// passes send a `MainCheckpoint` and honor the priority-`/btw` preempt
    /// flag, so a side answer is never interrupted by a queued side question.
    fn worker_generate(
        &mut self,
        tx: &Sender<UiEvent>,
        shared: &TurnShared,
        prompt: &str,
        turn_start: Instant,
        is_main: bool,
    ) -> Result<TurnOutput, String> {
        self.worker_generate_kind(tx, shared, prompt, turn_start, is_main, false)
    }

    /// As [`worker_generate`](Self::worker_generate), but `aside` selects
    /// [`Engine::generate_aside`] instead of `generate` so a mid-pass `/btw`
    /// answer snapshots and restores the frozen main-task KV around itself
    /// (BTW-SUSPEND-DESIGN §4.2). An aside is never a main pass, so `is_main`
    /// must be `false` when `aside` is `true`.
    #[allow(clippy::too_many_lines)]
    fn worker_generate_kind(
        &mut self,
        tx: &Sender<UiEvent>,
        shared: &TurnShared,
        prompt: &str,
        turn_start: Instant,
        is_main: bool,
        aside: bool,
    ) -> Result<TurnOutput, String> {
        // Snapshot the main log before streaming so a preempt can roll back
        // this pass's partial output before it re-runs.
        if is_main {
            let _ = tx.send(UiEvent::MainCheckpoint);
        }
        // Held for the whole pass — prefill included, which is most of the wait
        // — so the status bar's brain blinks while *this* engine works. Taken
        // here rather than only in `generate_pass`: that one covers the quiet
        // and fan-out passes, and this is the path every ordinary TUI turn (and
        // every `/subagent` sidechain) actually runs through.
        let _local = self.engine.is_local().then(crate::status::LocalPass::begin);
        let mut stream = StreamRenderer::new(ChannelSink(tx.clone()));
        stream.set_show_tool_calls(crate::settings::active().ui.show_tool_calls);
        stream.set_show_thinking(crate::settings::active().ui.show_thinking);
        stream.set_thinking_tool_calls(crate::settings::active().engine.thinking_tool_calls);
        stream.set_tool_names(sysprompt::tool_names(&self.tool_ctx.mcp));
        stream.set_preflight(edit_preflight(&self.tool_ctx));
        // Local engines open `<think>` implicitly in the prefill; provider
        // engines emit explicit tags, so only pre-open for local ones (see the
        // matching note in the plain-REPL path).
        if !matches!(self.think, crate::engine::ThinkMode::Off) && !self.engine.wants_structured() {
            stream.begin_in_think();
        }
        // Set when a mid-stream preflight fails: stops the engine early, but
        // is not a user interrupt — the turn loop feeds the error to the model.
        let preflight_stop = AtomicBool::new(false);
        // Mirrors the C's worker greedy flag: argmax sampling while the
        // stream renderer is inside a DSML tool-call stanza.
        let greedy = AtomicBool::new(false);
        let ctx_size = self.engine.ctx_size();
        let power = self.power_percent;
        let think = self.think;
        // Bound before the event closure, which cannot borrow `self` while
        // `self.engine` is generating.
        let model_name = self.engine.model_name();
        // Prompt tokens already in context; generated tokens add onto this so
        // the ctx gauge moves while the model streams.
        let prompt_tokens = self.engine.count_tokens(prompt);
        let mut assistant_text = String::new();
        let mut gen_count = 0;
        let verb = status::random_verb_index();
        let start = Instant::now();

        let interrupt = || {
            shared.interrupt.load(Ordering::Relaxed)
                || (is_main && shared.preempt.load(Ordering::Relaxed))
                || preflight_stop.load(Ordering::Relaxed)
                || crate::interrupt::pending()
        };
        let greedy_fn = || greedy.load(Ordering::Relaxed);
        let mut on_event = |ev| {
            let status = match ev {
                EngineEvent::Text(t) => {
                    assistant_text.push_str(&t);
                    stream.push(&t);
                    greedy.store(stream.wants_greedy_sampling(), Ordering::Relaxed);
                    if stream.preflight_error().is_some() {
                        preflight_stop.store(true, Ordering::Relaxed);
                    }
                    gen_count += 1;
                    let secs = start.elapsed().as_secs_f64();
                    Status {
                        state: WorkerState::Generating,
                        generated: gen_count,
                        prefill_label: verb,
                        gen_tps: if secs > 0.0 {
                            f64::from(gen_count) / secs
                        } else {
                            0.0
                        },
                        elapsed_secs: turn_start.elapsed().as_secs_f64(),
                        ctx_used: prompt_tokens + gen_count,
                        ctx_size,
                        power_percent: power,
                        think,
                        greedy_sampling: greedy.load(Ordering::Relaxed),
                        ..Status::default()
                    }
                }
                EngineEvent::Prefill(p) => Status {
                    // See the plain-REPL path: a completed prefill is the
                    // sampling wait, not prefilling (#64 follow-up). A finished
                    // pass is also the only one whose tok/s means anything, so
                    // that is where the peak is sampled.
                    state: if p.is_complete() {
                        crate::speeds::note_prefill(&model_name, p.tps, p.done);
                        WorkerState::Generating
                    } else {
                        WorkerState::Prefill
                    },
                    prefill_done: p.done,
                    prefill_total: p.total,
                    prefill_label: verb,
                    prefill_tps: p.tps,
                    elapsed_secs: turn_start.elapsed().as_secs_f64(),
                    ctx_used: prompt_tokens,
                    ctx_size,
                    power_percent: power,
                    think,
                    ..Status::default()
                },
                // Warm-up-only signal; never emitted mid-turn.
                EngineEvent::Notice(_) => return,
            };
            let _ = tx.send(UiEvent::Status(status));
        };
        // Provider engines take a structured turn; local engines keep the flat
        // rendered transcript (byte parity, §4.4). `bufs`/`st` outlive the call.
        let bufs =
            (!aside && self.engine.wants_structured()).then(|| self.build_structured(prompt));
        let st;
        let engine_prompt = match &bufs {
            Some(b) => {
                st = crate::engine::StructuredTurn {
                    system: &b.system,
                    messages: &b.messages,
                    tools: &b.tools,
                    rendered: &b.rendered,
                };
                crate::engine::Prompt::Structured(&st)
            }
            None => crate::engine::Prompt::Flat(prompt),
        };
        let result = if aside {
            // The aside keeps the main KV intact itself — by forking it, or by
            // snapshot/restore — and forces greedy off internally, so no
            // greedy sampler is passed.
            self.generate_aside_best(prompt, &self.cfg.generation, &interrupt, &mut on_event)
        } else if let Some(aside_prompt) = self.pending_aside.take() {
            // A `/btw` arrived during the previous pass. Rather than freezing
            // the main task for the whole answer, run both: this continuation
            // on the live session and the aside on a fork, interleaved. The
            // main stream keeps its ordinary renderer — only the aside's events
            // are routed away, to the side panel.
            //
            // The aside gets a renderer of its own, so its thinking is split
            // from its answer the way any other generation's is. Tool calls are
            // denied for an aside, so it needs none of the dispatch machinery.
            let mut aside_renderer = StreamRenderer::new(crate::worker::BtwSink(tx.clone()));
            aside_renderer.set_show_thinking(crate::settings::active().ui.show_thinking);
            if !matches!(self.think, crate::engine::ThinkMode::Off) {
                aside_renderer.begin_in_think();
            }
            // Shared between the token sink and the completion hook; the borrow
            // checker will not let both closures hold it mutably.
            let aside_stream = std::cell::RefCell::new(aside_renderer);
            self.engine
                .generate_multiplexed(
                    prompt,
                    &aside_prompt,
                    &self.cfg.generation,
                    &interrupt,
                    &mut |which, ev| match which {
                        crate::engine::AsideStream::Main => on_event(ev),
                        crate::engine::AsideStream::Aside => {
                            if let EngineEvent::Text(t) = ev {
                                aside_stream.borrow_mut().push(&t);
                            }
                        }
                    },
                    // Close the answer out the moment the aside itself is done,
                    // not when the main task catches up. A one-line aside
                    // finishes in a slice or two; waiting would leave its last
                    // line uncommitted for the rest of the turn.
                    &mut |which| {
                        if which == crate::engine::AsideStream::Aside {
                            aside_stream.borrow_mut().finish();
                            let _ = tx.send(UiEvent::Btw(Box::new(UiEvent::EndLine)));
                        }
                    },
                )
                .map(|(main, _aside)| main)
        } else {
            self.engine.generate(
                engine_prompt,
                &self.cfg.generation,
                &interrupt,
                &greedy_fn,
                &mut on_event,
            )
        };

        let stats = result.map_err(|e| e.to_string())?;
        self.record_usage(&stats);
        self.last_ctx_used = stats.ctx_used;
        stream.finish();
        let finished = stream.finished();
        let calls = finished.calls.to_vec();
        // A preflight stop reads as an engine interrupt, but it is a tool
        // error to feed back to the model, not a user abort.
        let preflight_error = stream.preflight_error();
        let error = preflight_error
            .map(|e| tool_error_payload(PassError::Preflight, e))
            .or_else(|| {
                finished.error.map(|e| {
                    tool_error_payload(pass_error_kind(false, finished.in_think_rejected), e)
                })
            });
        let user_interrupt = shared.interrupt.load(Ordering::Relaxed);
        // A real interrupt (Esc) takes precedence; only otherwise is a stopped
        // main pass a priority-`/btw` preempt. Preempt is not an error, so a
        // preflight failure never counts as one.
        let preempted = is_main
            && !user_interrupt
            && shared.preempt.load(Ordering::Relaxed)
            && preflight_error.is_none();
        if preempted {
            shared.preempt.store(false, Ordering::Relaxed);
        }
        let interrupted =
            (stats.interrupted || user_interrupt) && !preempted && preflight_error.is_none();
        // Consume the interrupt so a queued follow-up turn starts clean.
        shared.interrupt.store(false, Ordering::Relaxed);
        Ok(TurnOutput {
            interrupted,
            preempted,
            assistant_text,
            ended_in_think: finished.ended_in_think,
            calls,
            error,
        })
    }

    /// The turn's triggering prompt: the last real user message, skipping the
    /// tool results that are stored as user turns.
    fn last_user_prompt(&self) -> &str {
        self.session
            .transcript
            .iter()
            .rev()
            .find(|m| m.role == crate::session::Role::User && !m.is_tool_user())
            .map_or("", |m| m.text.as_str())
    }

    /// Fires the turn-end desktop notification: `'<prompt...>' finished` (or
    /// `interrupted`, per [`Self::last_turn_interrupted`]) as the (bold)
    /// headline and the tail of the assistant's final output as the body.
    fn notify_task_complete(&self) {
        let interrupted = self.last_turn_interrupted;
        let prompt = self.last_user_prompt();
        let output = self
            .session
            .transcript
            .iter()
            .rev()
            .find(|m| m.role == crate::session::Role::Assistant)
            .map_or("", |m| m.text.as_str());
        let title = crate::notify::finished_title(prompt, interrupted);
        let body = crate::notify::latest_output_body(output, interrupted);
        // Attached remote front-ends raise their own notification from this:
        // the local desktop one only reaches whoever is at this machine, which
        // is exactly the person a remote session is not.
        if let Some(r) = &self.remote {
            r.bus.broadcast(UiEvent::Notify {
                title: title.clone(),
                body: body.clone(),
            });
        }
        crate::notify::notify_sticky(&title, None, &body);
    }

    /// Compacts before a TUI turn when context is tight; progress goes to
    /// `sink` (the TUI log, or the worker→UI channel during a turn).
    fn maybe_compact_notify(
        &mut self,
        sink: &mut dyn CompactSink,
        interrupt: &dyn Fn() -> bool,
    ) -> Result<Compacted, String> {
        let rendered = render_transcript(&self.session, &self.system);
        let used = self.engine.count_tokens(&rendered);
        if !compact::should_compact(self.engine.ctx_size(), used) {
            return Ok(Compacted::Done);
        }
        // Cheapest step first: clear old tool-result bodies (no model
        // round-trip) and only fall back to full summarization if still tight.
        if let Some(cleared) = self.try_microcompact() {
            sink.note(format!(
                "microcompacted: cleared {cleared} old tool result(s)"
            ));
            return Ok(Compacted::Done);
        }
        self.do_compact_notify("low context", "", sink, interrupt)
    }

    /// Performs a compaction pass and rebuilds the transcript.
    ///
    /// `interrupt` is polled by the engine between tokens; when it fires the
    /// summary is discarded and the transcript is left exactly as it was.
    fn do_compact_notify(
        &mut self,
        reason: &str,
        instructions: &str,
        sink: &mut dyn CompactSink,
        interrupt: &dyn Fn() -> bool,
    ) -> Result<Compacted, String> {
        sink.note(format!(
            "COMPACTING {reason}: summarizing durable task state..."
        ));
        if !instructions.is_empty() {
            sink.note(format!("with your instructions: {instructions}"));
        }
        // Restored on drop, so an interrupted or failed pass hands the window
        // back to whatever it said before (a running turn, or the idle prompt).
        let _title = crate::title::Scoped::set(crate::title::State::Compacting);
        let trigger = Self::compact_trigger(reason);
        self.fire_pre_compact(trigger, &mut |w| sink.note(w));
        let mut prompt = render_transcript(&self.session, &self.system);
        {
            use std::fmt::Write as _;
            let _ = write!(
                prompt,
                "[user]\n{}\n",
                compact::make_prompt(reason, instructions)
            );
        }
        // Drives the status bar's compaction bar; cleared on drop, including on
        // the interrupt and engine-error paths below.
        let progress = status::CompactProgress::begin();
        let mut summary = String::new();
        let stats = self
            .engine
            .generate(
                crate::engine::Prompt::Flat(&prompt),
                &self.cfg.generation,
                interrupt,
                &|| false,
                &mut |ev| {
                    match ev {
                        EngineEvent::Text(t) => {
                            summary.push_str(&t);
                            progress.summarizing(summary.len());
                        }
                        EngineEvent::Prefill(p) => progress.prefill(p.done, p.total),
                        EngineEvent::Notice(_) => {}
                    }
                    sink.redraw();
                },
            )
            .map_err(|e| e.to_string())?;
        drop(progress);
        if stats.interrupted {
            sink.note(COMPACT_INTERRUPTED.to_owned());
            crate::interrupt::clear();
            return Ok(Compacted::Interrupted);
        }
        let extracted = compact::extract_summary(&summary);
        if extracted.trim().is_empty() {
            sink.note(COMPACT_NO_SUMMARY.to_owned());
            return Ok(Compacted::NoSummary);
        }
        self.rebuild_after_compact(&summary);
        self.fire_post_compact(trigger, &extracted, &mut |w| sink.note(w));
        sink.note("context compacted".to_owned());
        Ok(Compacted::Done)
    }

    /// Re-injects the system-prompt reminder in the TUI when due.
    fn maybe_reminder_notify(&mut self, note: &mut dyn FnMut(String)) {
        let rendered = render_transcript(&self.session, &self.system);
        let pos = self.engine.count_tokens(&rendered);
        if !self.reminder.should_remind(pos) {
            return;
        }
        note("Re-injecting system prompt reminder...".to_owned());
        self.trace.line(&format!(
            "system prompt reminder injected at transcript={pos}"
        ));
        let mut text = sysprompt::build_system_prompt_reminder(
            &self.tool_ctx.mcp,
            !crate::settings::active().engine.thinking_tool_calls,
        );
        if !self.cfg.system.is_empty() {
            text.push_str("\nAdditional system instructions reminder:\n");
            text.push_str(&self.cfg.system);
            text.push_str("\n[End additional system instructions reminder.]\n\n");
        }
        self.session.push(Message::user(text));
    }

    /// Runs a `/btw` side question typed while the agent is idle. It reuses
    /// the same `drain_btw` path as a mid-turn `/btw`, so the answer streams
    /// into the (persistent) side panel and the panel stays open afterwards —
    /// dismissed only by Esc — exactly like the busy-time case.
    #[allow(clippy::too_many_arguments)]
    fn tui_btw(
        &mut self,
        question: &str,
        log: &mut OutputLog,
        terminal: &mut ratatui::DefaultTerminal,
        view: &mut tui::OutputView,
        input: &mut TuiInput,
        btw: &mut BtwPanel,
        arcade: &mut crate::arcade::Arcade,
        sub: &mut tui::SubPane,
    ) -> Result<(), String> {
        let remote = self.remote.clone();
        let bus = remote.as_ref().map(|r| Arc::clone(&r.bus));
        let ui_remote = self.ui_remote.clone();
        let shared = TurnShared::default();
        shared.push_btw(question.to_owned());
        let live = LiveCommands::capture(self);
        run_worker_ui(
            terminal,
            log,
            view,
            input,
            btw,
            arcade,
            sub,
            &shared,
            bus.as_deref(),
            ui_remote.as_deref(),
            None,
            &live,
            |tx| {
                self.drain_btw(&tx, &shared);
            },
        )?;
        Ok(())
    }

    /// Handles a slash command in the TUI; returns false to quit.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    fn tui_slash(
        &mut self,
        line: &str,
        log: &mut OutputLog,
        terminal: &mut ratatui::DefaultTerminal,
        view: &mut tui::OutputView,
        input: &mut TuiInput,
        btw: &mut BtwPanel,
        config_form: &mut Option<crate::configform::ConfigForm>,
        arcade: &mut crate::arcade::Arcade,
        sub: &mut tui::SubPane,
    ) -> bool {
        let mut parts = line.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or(line);
        let arg = parts.next().unwrap_or("").trim();
        match cmd {
            "/quit" | "/exit" => return false,
            // Easter eggs: deliberately absent from `/help` and the completion
            // popup, but known commands, so they run instead of being sent to
            // the model. They take over the screen until Esc, and resume where
            // they were left unless the argument asks for a new game.
            _ if crate::arcade::enabled() && crate::arcade::Arcade::COMMANDS.contains(&cmd) => {
                let (w, h) = terminal.size().map_or((80, 24), |s| (s.width, s.height));
                let fresh = crate::arcade::Arcade::wants_new(arg);
                let resuming = !fresh && arcade.has_parked(cmd);
                arcade_hover_reporting(true);
                arcade.open(cmd, fresh, arcade_seed(), w, h);
                arcade.sound.set(crate::arcade::Sound::wanted(arg));
                if resuming {
                    log.push_dim(format!("{cmd}: resumed where you left off"));
                }
            }
            "/config" => {
                // Open the interactive modal; the run loop drives it and
                // persists on close. `arg` is ignored (the form edits everything).
                *config_form = Some(crate::configform::ConfigForm::new(
                    crate::settings::active().clone(),
                ));
            }
            "/new" | "/clear" => {
                self.session = Session::new();
                self.broadcast_session_reset(None);
                self.reminder = SystemPromptReminder::new();
                self.context_content = ContextContent::new();
                push_session_context(&mut self.session, &self.context_content);
                // Scaffolding only — not activity worth a resume point (see
                // `save_for_exit`); a real turn re-dirties it.
                self.session.dirty = false;
                self.last_ctx_used = 0;
                self.checkpoints.clear();
                self.usage = SessionUsage::default();
                // Issue #72: the screen must reflect the fresh session, so drop
                // the old conversation and re-render what a launch shows.
                log.clear();
                *view = tui::OutputView::default();
                // The pane belongs to the old session. Left alone it would keep
                // an obsolete sub-agent transcript on offer under Ctrl-O — and,
                // while it is the active pane, would swallow the cleared log and
                // every later turn behind the still-displayed old output.
                sub.reset();
                self.tui_write_banner(log);
                // Reinstate the warm prefix; without it the next turn silently
                // rebuilds the whole system-prompt KV (see `rewarm_after_reset`).
                // It takes long enough to notice, so hide the prompt and pin a
                // throbber in its place — the same "agent is busy" shape a turn
                // uses (`draw` with `input: None`), so the fresh banner stays
                // visible and no input can be typed into a session whose KV is
                // still being restored.
                self.rewarm_after_reset(&mut || {
                    log.set_progress(Some(tui::progress_line(&format!(
                        "{} starting a new session",
                        crate::status::throbber()
                    ))));
                    let (l, v) = (&*log, &mut *view);
                    let _ = terminal.draw(|f| {
                        tui::draw(f, l, None, "", v, None, &tui::TaskView::default(), None);
                    });
                });
                log.set_progress(None);
                self.fire_session_start("clear", &mut |w| log.push_plain(w));
                log.push_plain("started a new session");
            }
            "/checkpoint" => {
                if arg.is_empty() {
                    log.push_ansi(&crate::checkpoint::render_list(
                        &self.checkpoints,
                        now_secs(),
                        true,
                    ));
                } else {
                    log.push_plain(self.checkpoint_create(arg));
                }
            }
            "/rollback" => {
                if arg.is_empty() {
                    log.push_plain("usage: /rollback <name> (see /checkpoint for the list)");
                } else {
                    match self.rollback_to(arg) {
                        Ok(msg) => log.push_plain(msg),
                        Err(e) => log.push_plain(e),
                    }
                }
            }
            "/tree" => log.push_ansi(&self.tree_view(true)),
            "/fork" => match self.fork_branch(arg, true) {
                Ok(msg) => log.push_ansi(&msg),
                Err(e) => log.push_plain(e),
            },
            "/clone" => match self.clone_branch() {
                Ok(msg) => log.push_plain(msg),
                Err(e) => log.push_plain(e),
            },
            "/help" => {
                for line in crate::config::usage().lines() {
                    log.push_plain(line.to_owned());
                }
            }
            "/version" => log.push_plain(format!("plank {}", crate::logo::version_label())),
            "/mcp" => log.push_ansi(&render_mcp_report(&self.tool_ctx.mcp, true)),
            "/context" => log.push_ansi(&self.render_context_report(true)),
            "/usage" => log.push_ansi(&self.render_usage_report(true)),
            "/init" => self.tui_run_init(log, terminal, view, input, btw, arcade, sub),
            "/compact" => {
                let result = {
                    // A slash command runs on the UI thread, so nothing else is
                    // repainting: the sink has to draw each frame itself for the
                    // compaction bar to advance on screen at all.
                    let mut sink = TuiCompactSink {
                        log,
                        terminal,
                        view,
                    };
                    // No worker is running for a slash command, so the only
                    // interrupt source is a real SIGINT.
                    // Any argument is extra summarization instructions for this
                    // one pass.
                    self.do_compact_notify(
                        "user request",
                        arg,
                        &mut sink,
                        &crate::interrupt::pending,
                    )
                };
                if let Err(e) = result {
                    log.push_plain(format!("compact failed: {e}"));
                }
            }
            "/save" => match self.save_session() {
                Ok(id) => {
                    log.push_plain(format!("saved session {}", &id[..8]));
                    if let Some(note) = self.save_session_payload() {
                        log.push_dim(note);
                    }
                }
                Err(e) => log.push_plain(format!("save failed: {e}")),
            },
            "/list" => match self.store.list() {
                Ok(entries) => {
                    for line in
                        crate::session::render_session_list(&entries, now_secs(), false).lines()
                    {
                        log.push_plain(line.to_owned());
                    }
                }
                Err(e) => log.push_plain(format!("list failed: {e}")),
            },
            "/switch" => match self.store.load(arg) {
                Ok(s) => {
                    let note = self.load_session_payload(&s);
                    self.session = s;
                    self.broadcast_session_reset(Some(
                        "[session replaced — its history is on the local screen only]",
                    ));
                    self.last_ctx_used = 0;
                    self.checkpoints.clear();
                    self.usage = SessionUsage::default();
                    // Same reason as `/clear`: the pane is the old session's.
                    sub.reset();
                    self.replay_history_into_log(log);
                    if let Some(note) = note {
                        log.push_dim(note);
                    }
                }
                Err(e) => log.push_plain(format!("switch failed: {e}")),
            },
            "/del" => match self.store.delete(arg) {
                Ok(id) => log.push_plain(format!("deleted session {}", &id[..8])),
                Err(e) => log.push_plain(format!("delete failed: {e}")),
            },
            "/resume" => match self.resume_pick(arg) {
                Ok(None) => match self.store.list() {
                    Ok(entries) => log.push_ansi(&crate::session::render_resume_list(
                        &entries,
                        now_secs(),
                        true,
                        RESUME_LIST_LIMIT,
                    )),
                    Err(e) => log.push_plain(format!("resume failed: {e}")),
                },
                Ok(Some(s)) => {
                    let note = self.load_session_payload(&s);
                    self.session = s;
                    self.broadcast_session_reset(Some(
                        "[session replaced — its history is on the local screen only]",
                    ));
                    self.last_ctx_used = 0;
                    self.checkpoints.clear();
                    self.usage = SessionUsage::default();
                    // Same reason as `/clear`: the pane is the old session's.
                    sub.reset();
                    self.replay_history_into_log(log);
                    if let Some(note) = note {
                        log.push_dim(note);
                    }
                }
                Err(e) => log.push_plain(format!("resume failed: {e}")),
            },
            "/tag" => {
                if arg.is_empty() {
                    if self.session.tag.is_empty() {
                        log.push_plain("no tag set; usage: /tag <text> (\"/tag -\" clears)");
                    } else {
                        log.push_plain(format!("tag: {}", self.session.tag));
                    }
                } else {
                    match self.set_tag(arg) {
                        Ok(msg) => log.push_plain(msg),
                        Err(e) => log.push_plain(format!("tag failed: {e}")),
                    }
                }
            }
            "/history" => {
                let turns = if arg.is_empty() {
                    HISTORY_DEFAULT_TURNS
                } else {
                    arg.parse::<usize>()
                        .unwrap_or(HISTORY_DEFAULT_TURNS)
                        .clamp(1, HISTORY_MAX_TURNS)
                };
                for line in
                    crate::session::render_history(&self.session.transcript, turns, false).lines()
                {
                    log.push_plain(line.to_owned());
                }
            }
            "/power" => match crate::config::parse_power_percent(arg) {
                Some(power) => {
                    self.power_percent = power;
                    crate::status::set_local_power(power);
                    log.push_plain(format!("power limit set to {power}%"));
                }
                None => log.push_plain("usage: /power <1..100>"),
            },
            "/think" => {
                let msg = self.think_command(arg);
                log.push_plain(msg);
            }
            "/notify" => log.push_plain(Self::notify_command(arg)),
            // Non-advertised: re-shows the last desktop notification so it can be
            // screenshotted. Not in `/help` or `slash_command_known`.
            "/renotify" => {
                if crate::notify::renotify() {
                    log.push_plain("re-showing last notification");
                } else {
                    log.push_plain("no notification to re-show yet");
                }
            }
            "/strip" => {
                if arg.is_empty() {
                    log.push_plain("usage: /strip <sha-prefix>");
                } else {
                    match self.strip_session(arg) {
                        Ok((sha, tokens)) => {
                            log.push_plain(format!(
                                "stripped session {} ({tokens} tokens)",
                                &sha[..8]
                            ));
                        }
                        Err(e) => log.push_plain(format!("strip failed: {e}")),
                    }
                }
            }
            "/skills" => {
                for line in crate::skills::render_list(&self.skills).lines() {
                    log.push_plain(line.to_owned());
                }
            }
            "/templates" => {
                for line in crate::templates::render_list(&self.templates).lines() {
                    log.push_plain(line.to_owned());
                }
            }
            "/tasks" => {
                for line in self.session.tasks.render_list().lines() {
                    log.push_plain(line.to_owned());
                }
            }
            "/agent" => {
                for line in crate::agents::render_list(&self.agents).lines() {
                    log.push_plain(line.to_owned());
                }
            }
            "/hooks" => {
                for line in crate::hooks::render_list(&self.tool_ctx.hooks).lines() {
                    log.push_plain(line.to_owned());
                }
            }
            "/remote-control" | "/rc" => {
                for line in self.remote_toggle_lines(cmd, arg) {
                    log.push_plain(line);
                }
            }
            "/btw" => {
                if arg.is_empty() {
                    log.push_plain("usage: /btw <question>");
                } else if let Err(e) =
                    self.tui_btw(arg, log, terminal, view, input, btw, arcade, sub)
                {
                    log.push_plain(format!("/btw failed: {e}"));
                }
            }
            "/remember" => match remember_from_arg(&self.tool_ctx.cwd, arg) {
                Ok(path) => log.push_dim(format!("[saved to {}]", path.display())),
                Err(e) => {
                    log.push_plain(e);
                    log.push_plain("usage: /remember [user] <text> (default scope: project)");
                }
            },
            "/export" => match self.write_export(arg) {
                Ok(path) => log.push_plain(format!("exported session to {}", path.display())),
                Err(e) => {
                    log.push_plain(format!("export failed: {e}"));
                    log.push_dim("usage: /export [md|html] [path]".to_owned());
                }
            },
            "/insights" => {
                // The scan and the model calls both take long enough to look
                // like a hang, so each progress note is pinned in the prompt's
                // place and the frame is redrawn — the same "agent is busy"
                // shape `/clear`'s re-warm uses.
                let result = {
                    let log = std::cell::RefCell::new(&mut *log);
                    let view = std::cell::RefCell::new(&mut *view);
                    let terminal = std::cell::RefCell::new(&mut *terminal);
                    let repaint = |line: &str| {
                        log.borrow_mut()
                            .set_progress(Some(tui::progress_line(&format!(
                                "{} {line}",
                                crate::status::throbber()
                            ))));
                        {
                            let l = log.borrow();
                            let mut v = view.borrow_mut();
                            let _ = terminal.borrow_mut().draw(|f| {
                                tui::draw(
                                    f,
                                    *l,
                                    None,
                                    "",
                                    *v,
                                    None,
                                    &tui::TaskView::default(),
                                    None,
                                );
                            });
                        }
                        // Nothing else is reading the keyboard while the
                        // command runs, so without this drain an Esc is not
                        // merely slow to take effect — it is never seen at
                        // all. Raising the shared interrupt flag lets both the
                        // generation loop and the session scan stop by the
                        // same route a Ctrl-C uses.
                        while event::poll(Duration::ZERO).unwrap_or(false) {
                            if let Ok(Event::Key(k)) = event::read()
                                && k.kind == KeyEventKind::Press
                                && (matches!(k.code, KeyCode::Esc)
                                    || (matches!(k.code, KeyCode::Char('c'))
                                        && k.modifiers.contains(KeyModifiers::CONTROL)))
                            {
                                crate::interrupt::request();
                            }
                        }
                    };
                    // Status lines are kept in the log; streamed reasoning
                    // only ever occupies the progress line, so a section's
                    // thinking ticks past without burying the report.
                    let mut note = |line: String| {
                        repaint(&line);
                        log.borrow_mut().push_dim(line);
                    };
                    let mut tick = |line: String| repaint(&line);
                    self.run_insights(arg, &mut note, &mut tick)
                };
                log.set_progress(None);
                match result {
                    Ok(Insights::Done { path, summary }) => {
                        for line in summary {
                            log.push_plain(line);
                        }
                        log.push_dim(format!("report written to {}", path.display()));
                    }
                    Ok(Insights::Cancelled) => log.push_dim("insights cancelled".to_owned()),
                    Err(e) => {
                        log.push_plain(format!("insights failed: {e}"));
                        log.push_dim("usage: /insights [fast]".to_owned());
                    }
                }
            }
            "/repro" => match self.write_repro(arg) {
                Ok(path) => log.push_dim(format!("[repro written to {}]", path.display())),
                Err(e) => log.push_plain(format!("repro failed: {e}")),
            },
            c if crate::agents::is_subagent_command(c) => {
                // See the plain-REPL arm: the name is part of the command
                // token, so the whole argument is the task.
                let mut def = None;
                if let Some(name) = crate::agents::command_name(c) {
                    let Some(d) = crate::agents::resolve_named(&self.agents, name) else {
                        log.push_plain(crate::agents::unknown_name_error(&self.agents, name));
                        return true;
                    };
                    def = Some(d);
                }
                let (instructions, spec, label, task, started) = match def {
                    Some(d) => (
                        Some(d.body.clone()),
                        d.engine.clone(),
                        d.name.clone(),
                        arg.to_string(),
                        format!("[subagent started: {}]", d.name),
                    ),
                    None => (
                        None,
                        None,
                        "sub-agent".to_string(),
                        arg.to_string(),
                        "[subagent started]".to_string(),
                    ),
                };
                if task.is_empty() {
                    log.push_plain("usage: /subagent[:<name>] <task>");
                } else {
                    // Resolved before the fork, exactly as in `run_agent_tool`: an
                    // engine this session cannot provide must not leave a framed
                    // task behind in the transcript.
                    let alt = match self.resolve_subagent_alt(spec) {
                        Ok(alt) => alt,
                        Err(e) => {
                            log.push_plain(format!("/subagent: engine unavailable: {e}"));
                            return true;
                        }
                    };
                    if let Some(note) = self.take_warm_note() {
                        log.push_dim(note);
                    }
                    log.push_dim(started);
                    let fork_at =
                        self.begin_subagent_fork(instructions.as_deref(), &task, alt.is_none());
                    log.push_dim(tui::subagent_signpost(&label));
                    sub.begin(label);
                    sub.adopt_turn = true;
                    let outcome = match alt {
                        None => self.tui_turn(terminal, log, view, input, btw, arcade, sub),
                        Some((key, engine)) => self.run_sidechain_on(key, engine, |s| {
                            s.tui_turn(terminal, log, view, input, btw, arcade, sub)
                        }),
                    };
                    sub.adopt_turn = false;
                    sub.end();
                    if let Err(e) = outcome {
                        // Restore the transcript even when the turn errored.
                        self.finish_subagent_fork(fork_at, &task);
                        log.push_plain(format!("/subagent failed: {e}"));
                    } else if self.finish_subagent_fork(fork_at, &task) {
                        log.push_dim("[subagent report added to the conversation]");
                        // See the plain-REPL arm: the main loop runs on the
                        // report, so delegated work is acted on rather than
                        // parked in the transcript.
                        if let Err(e) = self.tui_turn(terminal, log, view, input, btw, arcade, sub)
                        {
                            log.push_plain(format!("/subagent follow-up failed: {e}"));
                        }
                    } else {
                        log.push_dim("[subagent produced no report — nothing added]");
                    }
                }
            }
            _ if slash_command_known(cmd) => {
                log.push_plain(format!("{cmd}: not implemented yet"));
            }
            _ => match self.slash_message(cmd, arg) {
                Some(Ok(message)) => {
                    log.push_spans(tui::user_echo_spans(line));
                    self.session.push(Message::user(message));
                    if let Err(e) = self.tui_turn(terminal, log, view, input, btw, arcade, sub) {
                        log.push_plain(format!("{cmd} failed: {e}"));
                    }
                }
                Some(Err(e)) => log.push_plain(e),
                None => log.push_plain(format!("unknown command: {cmd}")),
            },
        }
        true
    }
}

/// Drives an interactive `ask` question (issue #34): renders the option panel
/// into the input region and reads keys until the user answers, declines
/// (Escape), or interrupts (Ctrl-C). Blocks the UI loop while up — the worker
/// is already blocked on the [`AskBridge`], so nothing else needs servicing —
/// and posts the outcome back through the bridge to unblock the worker.
///
/// Escape returns a distinct declined result and the turn continues; Ctrl-C
/// both interrupts the turn and unblocks the worker so no partial state lingers.
#[allow(clippy::too_many_arguments)]
fn run_ask_panel(
    terminal: &mut ratatui::DefaultTerminal,
    log: &OutputLog,
    view: &mut tui::OutputView,
    status: &str,
    tasks: &tui::TaskView,
    shared: &TurnShared,
    bridge: &crate::tools::ask::AskBridge,
) -> Result<(), String> {
    use crate::tools::ask::{AskOutcome, AskState};
    let Some(req) = bridge.take_request() else {
        return Ok(());
    };
    let mut state = AskState::new(req.options.len(), req.multi);
    loop {
        terminal
            .draw(|f| tui::draw_ask(f, log, &req, &state, status, view, tasks))
            .map_err(|e| e.to_string())?;
        let Some(ev) = next_event(None, Duration::from_millis(100))? else {
            continue;
        };
        let Event::Key(key) = ev else { continue };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Up => state.move_up(),
            KeyCode::Down => state.move_down(),
            KeyCode::Char(' ') if req.multi => state.toggle(),
            KeyCode::Enter => {
                bridge.respond(AskOutcome::Answered(state.accept(&req.options)));
                return Ok(());
            }
            KeyCode::Esc => {
                bridge.respond(AskOutcome::Declined);
                return Ok(());
            }
            KeyCode::Char('c') if ctrl => {
                shared.interrupt.store(true, Ordering::Relaxed);
                bridge.respond(AskOutcome::Interrupted);
                return Ok(());
            }
            _ => {}
        }
    }
}

/// Opens the prompt editor: the built-in one when it is compiled in and
/// enabled, `$EDITOR` otherwise. Returns `None` when the user cancels.
///
/// The caller must have suspended the TUI — both editors take the raw
/// terminal.
fn open_editor(current: &str) -> std::io::Result<Option<String>> {
    #[cfg(feature = "builtin_editor")]
    if crate::settings::active().ui.builtin_editor {
        return crate::miniedit::edit_text(current);
    }
    crate::editor::edit_text_externally(current)
}

/// Runs `f` with the TUI fully torn down: raw mode off, alternate screen left,
/// and every input mode we pushed at startup popped, so a child process gets a
/// pristine terminal. Everything is put back before returning and the screen is
/// cleared so the next `draw` repaints from scratch.
///
/// The restore runs from a `Drop` guard, so a panic inside `f` still leaves the
/// user's terminal usable rather than stuck in raw mode on the alternate
/// screen. Every terminal call is best-effort: a failure to, say, pop the
/// keyboard flags must not stop the rest of the restore from running.
fn with_tui_suspended<T>(terminal: &mut ratatui::DefaultTerminal, f: impl FnOnce() -> T) -> T {
    use ratatui::crossterm::terminal::{
        EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
    };

    /// Puts the TUI back on drop — mirrors the setup in `Agent::run_tui`.
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = enable_raw_mode();
            let _ = ratatui::crossterm::execute!(
                std::io::stdout(),
                EnterAlternateScreen,
                EnableMouseCapture,
                EnableBracketedPaste,
                event::EnableFocusChange,
                PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
            );
        }
    }

    let _ = ratatui::crossterm::execute!(
        std::io::stdout(),
        PopKeyboardEnhancementFlags,
        event::DisableFocusChange,
        DisableBracketedPaste,
        DisableMouseCapture,
        LeaveAlternateScreen
    );
    let _ = disable_raw_mode();
    let out = {
        let _restore = Restore;
        f()
    };
    let _ = terminal.clear();
    out
}

/// Pre-rendered output for the read-only slash commands that stay usable while
/// the worker owns the engine (`/context`, `/usage`, `/mcp`, `/help`).
///
/// The worker holds `self` for the whole turn, so the UI thread cannot call
/// back into the agent; these reports are captured once at turn start instead.
/// The cost is a tokenize pass over the transcript for `/context`, which is
/// cheap next to the prefill/decoding the turn is about to do. Commands not
/// listed here still tell the user to wait for the turn to finish.
struct LiveCommands {
    context: String,
    usage: String,
    mcp: String,
}

impl LiveCommands {
    /// Captures the read-only reports before the worker takes the engine.
    fn capture(agent: &Agent<'_>) -> Self {
        Self {
            context: agent.render_context_report(true),
            usage: agent.render_usage_report(true),
            mcp: render_mcp_report(&agent.tool_ctx.mcp, true),
        }
    }

    /// ANSI output for a read-only command runnable mid-turn, or `None` when
    /// the command must wait for the turn to finish. `/help` is static, so it
    /// is rendered on demand rather than snapshotted.
    fn output(&self, cmd: &str) -> Option<std::borrow::Cow<'_, str>> {
        use std::borrow::Cow;
        match cmd {
            "/context" => Some(Cow::Borrowed(self.context.as_str())),
            "/usage" => Some(Cow::Borrowed(self.usage.as_str())),
            "/mcp" => Some(Cow::Borrowed(self.mcp.as_str())),
            "/help" => Some(Cow::Owned(crate::config::usage())),
            "/version" => Some(Cow::Owned(format!(
                "plank {}",
                crate::logo::version_label()
            ))),
            "/config" => Some(Cow::Owned(crate::configform::render_text_list(
                crate::settings::active(),
            ))),
            _ => None,
        }
    }
}

/// Runs `job` on a scoped worker thread while the UI thread keeps the
/// terminal live (the C's worker/UI split). The worker owns the agent for
/// the duration of the job and reports through the channel; the UI applies
/// events to the log, redraws, and keeps the prompt editable.
#[allow(clippy::too_many_arguments)]
fn run_worker_ui<T: Send>(
    terminal: &mut ratatui::DefaultTerminal,
    log: &mut OutputLog,
    view: &mut tui::OutputView,
    input: &mut TuiInput,
    btw: &mut BtwPanel,
    arcade: &mut crate::arcade::Arcade,
    sub: &mut tui::SubPane,
    shared: &TurnShared,
    bus: Option<&BroadcastBus>,
    remote: Option<&Mutex<UiRemote>>,
    ask: Option<&crate::tools::ask::AskBridge>,
    live: &LiveCommands,
    job: impl FnOnce(Sender<UiEvent>) -> T + Send,
) -> Result<T, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::scope(|s| {
        let handle = s.spawn(move || job(tx));
        let ui = busy_ui_loop(
            terminal,
            log,
            view,
            input,
            btw,
            arcade,
            sub,
            &rx,
            shared,
            bus,
            remote,
            ask,
            live,
            || handle.is_finished(),
        );
        // On a UI error (terminal gone) the worker must still be stopped and
        // joined before the scope can end.
        if ui.is_err() {
            shared.interrupt.store(true, Ordering::Relaxed);
        }
        let out = handle
            .join()
            .map_err(|_| "worker thread panicked".to_owned());
        ui?;
        out
    })
}

/// How long an unacknowledged interrupt waits before a second Ctrl-C is taken
/// as a force quit.
///
/// Long enough that the two presses of an ordinary double-tap cannot trigger
/// it, short enough that a genuinely wedged turn is escapable without the user
/// reaching for `kill`.
const FORCE_QUIT_GRACE: Duration = Duration::from_secs(2);

/// Last resort when the worker will not stop: restore the terminal and leave.
///
/// This exits the *process*, not the turn. The worker runs on a scoped thread
/// borrowing the agent, so it cannot be abandoned and the scope cannot be left
/// while it lives — meaning no destructor here can run and the in-flight turn
/// is lost. That is the deal a force quit makes; it beats a wedged terminal,
/// and with the stream idle timeout in [`crate::remote`] it should never be
/// reached in the network-drop case that motivated it.
fn force_quit() -> ! {
    ratatui::restore();
    eprintln!("plank: force quit — the turn was abandoned and not saved.");
    std::process::exit(130);
}

/// Handles Esc / Ctrl-C during a worker job, with meaning that depends on the
/// `/btw` panel state:
/// - a side answer is **streaming** (`btw_active`): cancel it (interrupt) and
///   flag the panel to close when its `BtwEnd` arrives;
/// - the panel is **visible but frozen** (main task running behind it): just
///   dismiss the panel, leaving the main task running;
/// - **no panel**: interrupt the main task, as before.
fn close_or_interrupt(
    shared: &TurnShared,
    btw: &mut Option<(OutputLog, tui::OutputView)>,
    btw_active: bool,
    close_panel_on_end: &mut bool,
) {
    if btw_active {
        shared.interrupt.store(true, Ordering::Relaxed);
        *close_panel_on_end = true;
    } else if btw.is_some() {
        *btw = None;
    } else {
        shared.interrupt.store(true, Ordering::Relaxed);
    }
}

/// UI-thread event loop while a worker job runs: applies streamed render
/// events to the log, keeps the prompt editable (Enter queues the line for
/// the worker), scrolls, and maps Esc/Ctrl-C to a worker interrupt.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn busy_ui_loop(
    terminal: &mut ratatui::DefaultTerminal,
    log: &mut OutputLog,
    view: &mut tui::OutputView,
    input: &mut TuiInput,
    btw: &mut BtwPanel,
    arcade: &mut crate::arcade::Arcade,
    sub: &mut tui::SubPane,
    rx: &Receiver<UiEvent>,
    shared: &TurnShared,
    bus: Option<&BroadcastBus>,
    remote: Option<&Mutex<UiRemote>>,
    ask: Option<&crate::tools::ask::AskBridge>,
    live_cmds: &LiveCommands,
    done: impl Fn() -> bool,
) -> Result<(), String> {
    let mut status_line = String::new();
    // Latest task-list snapshot (issue #35), updated on every `UiEvent::Tasks`
    // and passed to `draw` for the status-bar counter and the contextual strip.
    let mut task_view = tui::TaskView::default();
    // The `/btw` side panel (`btw`) is owned by `tui_loop`, so it survives
    // across turns: it opens on the first BtwBegin and stays up — even after
    // the answer finishes, the main task resumes, and the whole turn ends —
    // until the user dismisses it with Esc. `btw_active` is true only while a
    // side answer is actually streaming: it gates whether render events go to
    // the panel or to the main log, which is what lets the main task keep
    // rendering on the left while a finished answer stays frozen on the right.
    let mut btw_active = false;
    // Set when Esc is pressed mid-answer: the panel is torn down once the
    // cancelled answer's BtwEnd arrives (so late btw tokens don't leak).
    let mut close_panel_on_end = false;
    // Main-log length at the start of the current main pass; a preempting
    // `/btw` truncates back to it so the discarded partial output does not
    // duplicate when the pass re-runs.
    let mut main_checkpoint = 0usize;
    // Latches so a given pending `ask` question notifies at most once: set
    // when we notify, reset as soon as the bridge is observed not pending.
    // Guards against `run_ask_panel` returning while still pending, which
    // would otherwise re-notify for the same question on the next iteration.
    let mut ask_notified = false;
    // True between press and release of a drag that started inside the prompt;
    // the twin of `tui_loop`'s flag, for the mid-turn prompt.
    let mut input_drag = false;
    // True while the progress line is showing the compaction bar, so it is
    // cleared exactly once when the pass ends.
    let mut compacting_line = false;
    // When the main-task interrupt was raised, so an interrupt the worker never
    // acknowledges can escalate to a force quit. `None` whenever no interrupt
    // is outstanding.
    let mut interrupt_at: Option<Instant> = None;
    // Wall-clock pacing for an easter egg opened mid-turn, same as the idle
    // loop: render events arrive irregularly, so the frame delta has to be
    // measured rather than inferred from the poll timeout.
    let mut arcade_last = Instant::now();
    loop {
        if arcade.is_open() {
            let dt = arcade_last.elapsed();
            arcade_last = Instant::now();
            arcade.step(u64::try_from(dt.as_millis()).unwrap_or(u64::MAX));
        }
        // An `ask` question parked by the worker takes over the input region
        // until answered; the worker is blocked meanwhile, so no render events
        // arrive and the takeover is self-contained (issue #34).
        if let Some(bridge) = ask
            && bridge.is_pending()
        {
            if !ask_notified {
                crate::notify::notify_sticky("plank", None, "Waiting for your input");
                ask_notified = true;
            }
            run_ask_panel(
                terminal,
                log,
                view,
                &status_line,
                &task_view,
                shared,
                bridge,
            )?;
            continue;
        }
        ask_notified = false;
        while let Ok(ev) = rx.try_recv() {
            // Mirror every worker event onto the remote bus so remote clients
            // see the same stream as the local TUI (issue #25, dual-path).
            if let Some(bus) = bus
                && !ev.is_local_pane_only()
            {
                bus.broadcast(ev.clone());
            }
            match ev {
                UiEvent::Status(st) => {
                    // The animated progress (throbber + verb + stats) always
                    // lives on a line pinned below the output, not in the
                    // footer — independent of showThinking.
                    status_line = status::build_status_text(&st, false, false);
                    log.set_progress(
                        status::progress_segment(&st, false).map(|p| tui::progress_line(&p)),
                    );
                    // While the sub-agent pane is on screen the status line
                    // must say so, or the user cannot tell which transcript
                    // they are reading.
                    if let (true, Some(label)) = (sub.active, sub.label.as_deref()) {
                        status_line = format!("[sub-agent: {label}] {status_line}");
                    }
                }
                UiEvent::Tasks(tv) => task_view = tv,
                UiEvent::BtwBegin => {
                    if btw.is_none() {
                        *btw = Some((OutputLog::new(), tui::OutputView::default()));
                    }
                    btw_active = true;
                }
                UiEvent::BtwEnd => {
                    btw_active = false;
                    if close_panel_on_end {
                        *btw = None;
                        close_panel_on_end = false;
                    }
                }
                // Checkpoint/rollback always act on the main log, regardless
                // of a live side panel (a preempt fires only in a main pass).
                UiEvent::MainCheckpoint => main_checkpoint = log.checkpoint(),
                UiEvent::MainRollback => {
                    log.truncate_to(main_checkpoint);
                    view.follow = true;
                }
                // The signpost line is emitted by the worker as an ordinary
                // `Dim`, so it reaches remote clients too; these arms only
                // move the pane's state (and stand aside for an adopted run).
                UiEvent::SubStart(label) => sub.on_sub_start(label),
                UiEvent::SubEnd => sub.on_sub_end(),
                UiEvent::Sub(inner) => worker::apply(&mut sub.log, *inner),
                // Addressed to the panel per event, so it lands there even
                // while the main task streams into the main log beside it.
                UiEvent::Btw(inner) => {
                    if let Some((btw_log, _)) = btw.as_mut() {
                        worker::apply(btw_log, *inner);
                    }
                }
                // Route to the panel only while an answer is streaming; once
                // it finishes the main task's output goes to the main log even
                // though the (frozen) panel is still visible.
                ev => {
                    if let (true, Some((btw_log, _))) = (btw_active, btw.as_mut()) {
                        worker::apply(btw_log, ev);
                    } else if sub.adopt_turn {
                        worker::apply(&mut sub.log, ev);
                    } else {
                        worker::apply(log, ev);
                    }
                }
            }
        }
        // Check before drawing: anything sent after this point survives in
        // the channel and is drained below (the sender is gone once the
        // worker returns).
        let finished = done();
        remote_drain(remote);
        input.pump_popup();
        // Pane selection is hoisted out of the draw closure: `sub` is also
        // borrowed by the event drain, so taking the two disjoint field
        // borrows here keeps the closure's capture to just those references.
        // Compaction takes over the progress line — the throbber/verb segment
        // below the output — because that line says what the turn is doing, and
        // during a compaction pass that is the compaction. Refreshed per frame
        // rather than off a `UiEvent`: the worker sends no `Status` events while
        // it compacts, so there is nothing else to hang the animation on. The
        // line is dropped on the frame after the pass ends, which also covers an
        // interrupted pass that never reports a final status.
        match crate::status::compact_progress() {
            Some(frac) => {
                log.set_progress(Some(tui::compact_progress_line(frac)));
                compacting_line = true;
            }
            None if compacting_line => {
                log.set_progress(None);
                compacting_line = false;
            }
            None => {}
        }
        // An interrupt the worker has not acknowledged within the grace period
        // means it is wedged somewhere that cannot poll the flag. Say so, and
        // name the way out — otherwise the UI looks identical to a hang.
        let stuck = interrupt_at.is_some_and(|t: Instant| t.elapsed() >= FORCE_QUIT_GRACE);
        let status_line: std::borrow::Cow<'_, str> = if stuck {
            std::borrow::Cow::Owned(format!(
                "{status_line}  [interrupt pending — Ctrl-C again to force quit]"
            ))
        } else {
            std::borrow::Cow::Borrowed(status_line.as_str())
        };
        let sub_active = sub.active;
        // Owned for the same reason: `sub.view` is borrowed mutably below.
        let sub_title: Option<String> = if sub_active { sub.label.clone() } else { None };
        let (draw_log, draw_view): (&OutputLog, &mut tui::OutputView) = if sub_active {
            (&sub.log, &mut sub.view)
        } else {
            (log, view)
        };
        terminal
            .draw(|f| {
                // The `/btw` split is about the main task, so it steps aside
                // while the sub-agent pane is the one being followed.
                if let (false, Some((btw_log, btw_view))) = (sub_active, btw.as_mut()) {
                    tui::draw_btw_split(
                        f,
                        draw_log,
                        btw_log,
                        btw_view,
                        Some(input.state()),
                        &status_line,
                        draw_view,
                        &task_view,
                    );
                } else {
                    tui::draw(
                        f,
                        draw_log,
                        Some(input.state()),
                        &status_line,
                        draw_view,
                        None,
                        &task_view,
                        sub_title.as_deref(),
                    );
                }
                if let Some(m) = &input.slash {
                    tui::draw_slash_menu(f, input.buf.text(), m);
                }
                if let Some(p) = &input.popup {
                    tui::draw_popup(f, input.buf.text(), p);
                }
                // Drawn last, over the live turn. Translucent by default here,
                // so the model's output keeps streaming legibly underneath.
                if arcade.is_open() {
                    tui::draw_arcade(f, arcade);
                }
                remote_capture(remote, f);
            })
            .map_err(|e| e.to_string())?;
        remote_service(remote);
        if finished {
            // The worker is done (turn over); drain the tail in order. The
            // panel is discarded when this function returns, so late btw
            // events just stop mattering.
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    UiEvent::Status(_) | UiEvent::MainCheckpoint | UiEvent::Tasks(_) => {}
                    UiEvent::BtwBegin => btw_active = true,
                    UiEvent::BtwEnd => btw_active = false,
                    UiEvent::MainRollback => log.truncate_to(main_checkpoint),
                    UiEvent::SubStart(label) => sub.on_sub_start(label),
                    UiEvent::SubEnd => sub.on_sub_end(),
                    UiEvent::Sub(inner) => worker::apply(&mut sub.log, *inner),
                    // Addressed to the panel per event, so a multiplexed aside
                    // lands there while the main task streams into the log.
                    UiEvent::Btw(inner) => {
                        if let Some((btw_log, _)) = btw.as_mut() {
                            worker::apply(btw_log, *inner);
                        }
                    }
                    ev => {
                        if let (true, Some((btw_log, _))) = (btw_active, btw.as_mut()) {
                            worker::apply(btw_log, ev);
                        } else if sub.adopt_turn {
                            worker::apply(&mut sub.log, ev);
                        } else {
                            worker::apply(log, ev);
                        }
                    }
                }
            }
            // The turn is over: drop the pinned progress line so it does not
            // linger into the idle view.
            log.set_progress(None);
            return Ok(());
        }
        // An open easter egg wants the shared 20 Hz animation tick; the idle
        // 100 ms poll would render it at ten frames a second.
        let poll = if arcade.is_open() {
            Duration::from_millis(crate::anim::TICK_MS)
        } else {
            Duration::from_millis(100)
        };
        let Some(ev) = next_event(remote, poll)? else {
            continue;
        };
        // While it is up, the easter egg owns input — including the mouse.
        // Ctrl-C closes it rather than interrupting the turn, so the first
        // Ctrl-C puts the screen back and a second one stops the model.
        if arcade.is_open() {
            match ev {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if let crate::arcade::Outcome::Close(line) = arcade.handle_key(key) {
                        arcade_hover_reporting(false);
                        if let Some(line) = line {
                            log.push_dim(line);
                        }
                    }
                }
                Event::Mouse(m) => {
                    let (w, h) = terminal
                        .size()
                        .map_or((80, 23), |s| (s.width, s.height.saturating_sub(1)));
                    arcade.handle_mouse(m, w, h);
                }
                _ => {}
            }
            continue;
        }
        match ev {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                // Same precedence as `tui_loop`: the popup sees keys first, so
                // Esc closes it before it can interrupt the worker, then the
                // shared selection keymap.
                if input.popup_key(key) || input.selection_key(key) {
                    continue;
                }
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                // Alt (Option on macOS) or Ctrl turns arrows and
                // Backspace/Delete into word-wise operations.
                let word_mod = ctrl || key.modifiers.contains(KeyModifiers::ALT);
                match key.code {
                    // Ctrl-O switches between the main transcript and the
                    // sub-agent pane; mid-turn is exactly when it matters.
                    KeyCode::Char('o') if ctrl => {
                        if !sub.toggle() {
                            log.push_dim("[no sub-agent has run yet]");
                        }
                    }
                    KeyCode::Esc => {
                        close_or_interrupt(shared, btw, btw_active, &mut close_panel_on_end);
                        // Arms the escalation the same way Ctrl-C does, so an
                        // interrupt raised with Esc can still be escaped from.
                        if interrupt_at.is_none() && shared.interrupt.load(Ordering::Relaxed) {
                            interrupt_at = Some(Instant::now());
                        }
                    }
                    KeyCode::Char('c') if ctrl => {
                        // Ctrl-C clears a partly-typed line first; on an empty
                        // line it acts like Esc (cancel answer / close panel /
                        // interrupt the model).
                        if input.buf.text().is_empty() {
                            // Escalation: a second Ctrl-C, once the first has
                            // gone unacknowledged past the grace period, is the
                            // only way out of a worker wedged somewhere it
                            // cannot poll the interrupt flag. It cannot be a
                            // graceful shutdown — the worker is a *scoped*
                            // thread holding `&mut Agent`, so the scope cannot
                            // be left while it runs and there is no safe way to
                            // abandon it.
                            if interrupt_at
                                .is_some_and(|t: Instant| t.elapsed() >= FORCE_QUIT_GRACE)
                            {
                                force_quit();
                            }
                            close_or_interrupt(shared, btw, btw_active, &mut close_panel_on_end);
                            if interrupt_at.is_none() && shared.interrupt.load(Ordering::Relaxed) {
                                interrupt_at = Some(Instant::now());
                            }
                        } else {
                            input.buf.clear();
                        }
                    }
                    // Shift+Enter inserts a newline instead of submitting.
                    // Terminals without the kitty keyboard protocol cannot
                    // report it, so Alt+Enter and Ctrl-J work everywhere.
                    KeyCode::Enter
                        if key.modifiers.contains(KeyModifiers::SHIFT)
                            || key.modifiers.contains(KeyModifiers::ALT) =>
                    {
                        input.hist_idx = None;
                        input.buf.insert("\n");
                    }
                    KeyCode::Char('j') if ctrl => {
                        input.hist_idx = None;
                        input.buf.insert("\n");
                    }
                    KeyCode::Enter => {
                        let line = input.buf.text().trim().to_owned();
                        input.buf.clear();
                        input.hist_idx = None;
                        if line.is_empty() {
                        } else if btw_question(&line).is_some() {
                            // A `/btw` gets priority: it preempts the running
                            // main pass so the side question is answered now,
                            // then the pass re-runs. Only when a side answer is
                            // already streaming (btw_active) is the main task
                            // paused already, so the question just joins the
                            // FIFO queue; a merely-visible frozen panel does
                            // not — the main task is running behind it, so a
                            // new `/btw` preempts it.
                            let question = btw_question(&line).unwrap_or_default().to_owned();
                            input.history.add(&line);
                            if let Some(dropped) = shared.push_btw(question) {
                                log.push_dim(format!(
                                    "[btw queue full — dropped oldest: {dropped}]"
                                ));
                            }
                            if btw_active {
                                log.push_dim("[/btw — answers next in the panel]");
                            } else {
                                shared.preempt.store(true, Ordering::Relaxed);
                                // Whether the main task actually pauses is the
                                // worker's call — it multiplexes when the engine
                                // can fork, and emits BTW_SUSPEND_MARKER only
                                // when it really does freeze. Stay neutral here.
                                log.push_dim("[/btw — answering now]");
                            }
                            view.follow = true;
                            sub.view.follow = true;
                        } else if let Some(out) = line
                            .starts_with('/')
                            .then(|| line.split_whitespace().next().unwrap_or(&line))
                            .and_then(|cmd| live_cmds.output(cmd))
                        {
                            // Read-only reports (`/context`, `/usage`, `/mcp`,
                            // `/help`) run against a turn-start snapshot, so they
                            // stay available while the model streams.
                            input.history.add(&line);
                            log.push_spans(tui::user_echo_spans(&line));
                            log.push_ansi(&out);
                            view.follow = true;
                            sub.view.follow = true;
                        } else if let Some(cmd) = arcade_command(&line) {
                            // The whole point of these is the waiting, so they
                            // are the commands that *do* run mid-turn.
                            // Translucent, so the stream stays readable behind.
                            input.history.add(&line);
                            let arg = line[cmd.len()..].trim();
                            let fresh = crate::arcade::Arcade::wants_new(arg);
                            let resuming = !fresh && arcade.has_parked(cmd);
                            let (w, h) = terminal.size().map_or((80, 24), |s| (s.width, s.height));
                            arcade_hover_reporting(true);
                            arcade.open(cmd, fresh, arcade_seed(), w, h);
                            arcade.sound.set(crate::arcade::Sound::wanted(arg));
                            arcade.veil();
                            if resuming {
                                log.push_dim(format!("{cmd}: resumed where you left off"));
                            }
                            arcade_last = Instant::now();
                        } else if line.starts_with('/') || line.starts_with('!') {
                            log.push_dim(
                                "[that command can't run mid-turn — wait for the model to finish]",
                            );
                        } else {
                            input.history.add(&line);
                            log.push_spans(tui::user_echo_spans(&line));
                            log.push_dim("[queued — joins the conversation at the next step]");
                            shared.push_queued(line);
                            view.follow = true;
                            sub.view.follow = true;
                        }
                    }
                    KeyCode::Char('u') if ctrl => input.buf.kill_to_start(),
                    KeyCode::Char('k') if ctrl => input.buf.kill_to_end(),
                    KeyCode::Char('w') if ctrl => input.buf.delete_prev_word(),
                    KeyCode::Char('a') if ctrl => input.buf.move_home(),
                    KeyCode::Char('e') if ctrl => input.buf.move_end(),
                    KeyCode::Char(c) if !ctrl && !key.modifiers.contains(KeyModifiers::ALT) => {
                        input.hist_idx = None;
                        input.buf.insert(c.to_string());
                    }
                    KeyCode::Backspace if word_mod => input.buf.delete_prev_word(),
                    KeyCode::Backspace => {
                        input.buf.backspace();
                    }
                    KeyCode::Delete if word_mod => input.buf.delete_next_word(),
                    KeyCode::Delete => {
                        input.buf.delete();
                    }
                    KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::ALT) => {
                        input.buf.delete_next_word();
                    }
                    KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::ALT) => {
                        input.buf.move_prev_word();
                    }
                    KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::ALT) => {
                        input.buf.move_next_word();
                    }
                    KeyCode::Left if word_mod => {
                        input.buf.move_prev_word();
                    }
                    KeyCode::Left => {
                        input.buf.move_left();
                    }
                    KeyCode::Right if word_mod => {
                        input.buf.move_next_word();
                    }
                    KeyCode::Right => {
                        input.buf.move_right();
                    }
                    KeyCode::Home => input.buf.move_home(),
                    // End resumes scroll-follow on an empty line, otherwise
                    // it is the usual end-of-line motion.
                    KeyCode::End => {
                        if input.buf.text().is_empty() {
                            sub.active_view(view).follow = true;
                        } else {
                            input.buf.move_end();
                        }
                    }
                    KeyCode::Up => input.history_move(-1),
                    KeyCode::Down => input.history_move(1),
                    _ => {}
                }
                // Retarget (or close) the popup after every edit and cursor move.
                input.sync_popup();
            }
            Event::Paste(pasted) => {
                input.hist_idx = None;
                // The line editor is single-line; fold pasted newlines into
                // spaces so the paste stays editable.
                input
                    .buf
                    .insert(pasted.replace("\r\n", "\n").replace(['\n', '\r'], " "));
                input.sync_popup();
            }
            Event::Mouse(m) => match m.kind {
                MouseEventKind::ScrollUp => {
                    let v = sub.active_view(view);
                    v.follow = false;
                    v.top = v.top.saturating_sub(3);
                }
                MouseEventKind::ScrollDown => {
                    // Clamped by draw, which re-enters follow mode at the bottom.
                    let v = sub.active_view(view);
                    v.top = v.top.saturating_add(3);
                }
                // Clicking the jump-to-bottom hint resumes follow mode.
                MouseEventKind::Down(MouseButton::Left)
                    if sub.active_view(view).jump_hint_rect.is_some_and(|r| {
                        r.contains(ratatui::layout::Position::new(m.column, m.row))
                    }) =>
                {
                    sub.active_view(view).follow = true;
                }
                // The prompt stays live mid-turn (queued lines, `/btw`), so
                // click-and-drag has to place and select in it here too.
                // Every press decides afresh which surface the gesture belongs
                // to, so a release lost off-window cannot strand the next drag.
                MouseEventKind::Down(MouseButton::Left) => {
                    input_drag = tui::last_input_rect()
                        .is_some_and(|r| input.mouse_to_cursor(r, m.column, m.row, false));
                }
                MouseEventKind::Drag(MouseButton::Left) if input_drag => {
                    if let Some(r) = tui::last_input_rect() {
                        input.mouse_to_cursor(r, m.column, m.row, true);
                    }
                }
                MouseEventKind::Up(MouseButton::Left) if input_drag => {
                    input_drag = false;
                    input.copy_selection(false);
                }
                _ => {}
            },
            _ => {}
        }
    }
}

fn print_footer(st: &Status, color: bool) {
    let line = status::build_status_text(st, color, true);
    if color {
        println!(
            "{}{line}{}",
            status::STATUS_STYLE_START,
            status::STATUS_STYLE_END
        );
    } else {
        println!("{line}");
    }
}

fn new_agent(
    mut engine: Box<dyn Engine>,
    cfg: &AgentConfig,
    show_footer: bool,
    local_engine: Option<Box<dyn Engine>>,
) -> Result<Agent<'_>, String> {
    let store = SessionStore::open(SessionStore::default_dir()).map_err(|e| e.to_string())?;
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let mut trace = Trace::open(cfg.trace_path.as_deref()).map_err(|e| e.to_string())?;
    let mut session = Session::new();
    // Collect context at session start
    let context_content = ContextContent::new();
    // Inject context into the session transcript
    trace.text("context", &context_content.combined());
    push_session_context(&mut session, &context_content);
    // Session-start context is scaffolding, not user activity: a session that
    // only ever holds it (no ds4_engine invocation) is not worth a resume point,
    // so it must not count as dirty. A real turn re-dirties it. (See
    // `save_for_exit`.)
    session.dirty = false;
    let mut tool_ctx = ToolContext::new(cwd);
    // `--worktree` already created the worktree and moved the process into it,
    // so `cwd` above is the worktree; adopting the session it left behind is
    // what lets `ExitWorktree` find its way back out.
    tool_ctx.worktree = crate::worktree::take_startup_session();
    // Start MCP servers before composing the system prompt so their tool
    // schemas land in it, like agent_worker_init.
    tool_ctx.mcp = crate::tools::mcp::load_and_start(cfg.mcp_config_path.as_deref());
    tool_ctx.hooks = crate::hooks::load_default(&tool_ctx.cwd);
    for w in &tool_ctx.hooks.warnings {
        eprintln!("{w}");
    }
    tool_ctx.sandbox = crate::sandbox::load_default(&tool_ctx.cwd);
    if let Some(enabled) = cfg.sandbox_override {
        tool_ctx.sandbox.enabled = enabled;
    }
    if show_footer {
        // System status lines go straight to stdout in the REPL; the TUI
        // replaces this sink with one that forwards to the worker channel.
        let color = std::io::stdout().is_terminal();
        tool_ctx.status_sink = Some(Box::new(move |msg: &str| {
            println!("{}", crate::status::system_line(msg, color));
        }));
        // Interactive approval for web access, like agent_web_confirm;
        // headless runs keep the auto-deny default.
        tool_ctx.web_confirm = Some(Box::new(|message: &str| {
            print!("{message}");
            let _ = std::io::stdout().flush();
            let mut answer = String::new();
            if std::io::stdin().read_line(&mut answer).is_err() {
                return false;
            }
            matches!(answer.trim(), "y" | "Y" | "yes")
        }));
    }
    // Loaded before the system prompt because the roster is part of it: the
    // `agent` tool's `name` enum advertises which definitions the model may
    // select. Definitions are on-disk files, stable across a session, so they
    // belong inside the fingerprinted prefix — editing one correctly
    // invalidates `sysprompt.kv` rather than being silently ignored.
    let agents = crate::agents::load_default(&tool_ctx.cwd);
    // Publishes the names so the input line can colour `/subagent:<name>` by
    // whether the name resolves, three call layers below anything that holds
    // the definitions themselves.
    crate::agents::set_roster(&agents);
    let system = sysprompt::build_system_prompt_parts(
        &cfg.system,
        &tool_ctx.mcp,
        !crate::settings::active().engine.thinking_tool_calls,
    );
    // Tell the engine where the trusted control text ends before it tokenizes
    // anything, so `｜DSML｜` in the prompt's examples prefills as the model's
    // own token rather than as spelled-out BPE pieces.
    engine.set_trusted_system_prefix(system.trusted_len);
    // And which reasoning level to build prompts at. Without this the engine
    // stays at its default while `cfg` says otherwise, so `--think-max` would
    // key the KV as `max` and prefill no reasoning-effort preamble — the one
    // combination the fingerprint cannot detect, because it is a disagreement
    // between the key and the tokens rather than between two keys.
    engine.set_think_mode(cfg.generation.think_mode);
    crate::status::set_local_power(cfg.power_percent);
    // The alt local engine needs both for the same reasons, and it cannot be
    // skipped as an optimization: `warm_reset` builds its system tokens from
    // these two fields, so an unconfigured engine tokenizes the *same* system
    // text differently from the one that wrote `sysprompt.kv`. Its Tier 1
    // checkpoint would then restore into a session whose token buffer does not
    // describe it, the first common-prefix probe would truncate back, and the
    // sidechain would pay the full prefill anyway — the restore would look like
    // it worked and buy nothing. It also makes a local sidechain generate at the
    // session's reasoning level rather than the engine's default.
    let mut local_engine = local_engine;
    if let Some(local) = local_engine.as_mut() {
        local.set_trusted_system_prefix(system.trusted_len);
        local.set_think_mode(cfg.generation.think_mode);
    }
    let trusted_system_len = system.trusted_len;
    let system = system.text;
    let skills = crate::skills::load_default(&tool_ctx.cwd);
    let templates = crate::templates::load_default(&tool_ctx.cwd);
    // The `skill` tool resolves names against the same set the slash command
    // uses; hand the dispatch context its own copy.
    tool_ctx.skills.clone_from(&skills);
    Ok(Agent {
        engine,
        cfg,
        session,
        store,
        pending_aside: None,
        tool_ctx,
        system,
        reminder: SystemPromptReminder::new(),
        power_percent: 0,
        payload_restored: false,
        trusted_system_len,
        think: cfg.generation.think_mode,
        trace,
        color: std::io::stdout().is_terminal(),
        show_footer,
        editor_owns_footer: false,
        last_ctx_used: 0,
        last_turn_interrupted: false,
        context_content,
        skills,
        templates,
        agents,
        isolation_seq: 0,
        checkpoints: crate::checkpoint::CheckpointStore::new(),
        remote: None,
        remote_server: None,
        ui_remote: None,
        usage: SessionUsage::default(),
        stats: SessionStats::default(),
        session_start: std::time::Instant::now(),
        sub_sink: SubSinkTarget::default(),
        fork_kv: Vec::new(),
        // A local engine handed in alongside a provider main agent lives in the
        // same cache as any other alternate: `provider: local` definitions take
        // it out for their sidechain and put it back afterwards, so the
        // borrow-checked "cannot be in the map and in self.engine at once"
        // guarantee covers it too.
        alt_engines: local_engine
            .map(|e| (EngineKey::Local, e))
            .into_iter()
            .collect(),
        local_alt_warmed: false,
        warm_note: None,
    })
}

/// Runs the interactive REPL until the user exits.
///
/// # Errors
/// Returns an error string on unrecoverable I/O or engine failure.
pub fn run_interactive(
    engine: Box<dyn Engine>,
    cfg: &AgentConfig,
    local_engine: Option<Box<dyn Engine>>,
) -> Result<(), String> {
    let mut agent = new_agent(engine, cfg, true, local_engine)?;

    // Seed the notification enable flag once, before either front-end loop
    // starts (CLAUDE.md: TUI and plain REPL are parallel paths sharing this
    // one entry point, so this covers both).
    crate::notify::set_mode(crate::settings::active().ui.notifications);

    // Plain REPL: a headless sub-agent's output has nowhere else to go, so
    // print it inline on stdout alongside the parent turn's own stream. The
    // TUI overwrites this with `Events` per turn in `worker_turn`, so this
    // value only ever survives on the plain-REPL path.
    agent.sub_sink = SubSinkTarget::Stdout;

    // `plank /resume [prefix]` loads a prior session before the loop starts.
    let resumed = cfg.resume.is_some();
    if let Some(arg) = &cfg.resume {
        agent.resume_from_cli(arg)?;
    }
    // SessionStart fires once the session identity is settled: `resume` when a
    // prior session was loaded, `startup` otherwise.
    agent.fire_session_start(if resumed { "resume" } else { "startup" }, &mut |w| {
        println!("{w}");
    });

    // A real terminal gets the full-screen ratatui UI (works cleanly in Warp
    // and other block terminals via the alternate screen). Piped input falls
    // back to the plain line REPL.
    let result = if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
        agent.run_tui()
    } else {
        run_plain_flow(&mut agent, cfg)
    };
    // Whatever happened, fire SessionEnd, save the session, and tell the user
    // how to resume it.
    agent.fire_session_end("exit", &mut |w| println!("{w}"));
    agent.report_session_on_exit();
    agent.report_run_stats();
    result
}

/// Plain-REPL [`Asker`](crate::tools::ask::Asker): prints the header, question,
/// and numbered options, then reads one stdin line and resolves it to a choice
/// (a number or a label prefix; an empty line declines). The degraded form of
/// the TUI panel for the piped/non-fullscreen path (issue #34).
struct StdinAsker {
    color: bool,
}

impl crate::tools::ask::Asker for StdinAsker {
    fn ask(&mut self, req: crate::tools::ask::AskRequest) -> crate::tools::ask::AskOutcome {
        use crate::tools::ask::{AskOutcome, parse_repl_answer};
        let mut out = std::io::stdout();
        let _ = writeln!(out, "\n[{}] {}", req.header, req.question);
        for (i, opt) in req.options.iter().enumerate() {
            if opt.description.is_empty() {
                let _ = writeln!(out, "  {}. {}", i + 1, opt.label);
            } else {
                let _ = writeln!(out, "  {}. {} — {}", i + 1, opt.label, opt.description);
            }
        }
        let prompt = if req.multi {
            "Choose (numbers/labels, comma-separated; blank to decline): "
        } else {
            "Choose (number or label; blank to decline): "
        };
        let _ = write!(out, "{prompt}");
        let _ = out.flush();
        let _ = self.color; // reserved for future styled prompts
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_err() {
            return AskOutcome::Declined;
        }
        parse_repl_answer(&req, &line)
    }
}

/// Plain-REPL session flow: warm the cache, run the one-shot `-p` prompt if
/// any, then read lines until EOF.
fn run_plain_flow(agent: &mut Agent<'_>, cfg: &AgentConfig) -> Result<(), String> {
    // The plain REPL answers `ask` questions by printing numbered options and
    // reading a line from stdin (issue #34).
    agent.tool_ctx.asker = Some(Box::new(StdinAsker { color: agent.color }));
    agent.warm_plain()?;
    // The plain REPL is up and accepting input.
    crate::title::set(crate::title::State::Idle);
    if let Some(history) = agent.resumed_history() {
        print!("{history}");
    }
    if let Some(initial) = cfg.prompt.as_deref().filter(|p| !p.is_empty()) {
        print!("{}", status::format_user_prompt_echo(initial, agent.color));
        agent.session.push(Message::user(initial));
        agent.run_turn()?;
    }
    run_repl_plain_local(agent)
}

/// Yellow hint shown when Ctrl-C is pressed on an empty idle prompt.
fn quit_hint_spans() -> Vec<ratatui::text::Span<'static>> {
    vec![ratatui::text::Span::styled(
        "Press Ctrl-D to quit.",
        ratatui::style::Style::default().fg(ratatui::style::Color::Yellow),
    )]
}

/// Streams a plain-REPL `!` command's output to the console as it arrives
/// rather than at exit (issue #22), keeping stdout and stderr on their own
/// console streams.
struct BangConsoleSink;

impl crate::tools::bash::ImmediateSink for BangConsoleSink {
    fn line(&mut self, stream: crate::tools::bash::Stream, text: &str) {
        match stream {
            crate::tools::bash::Stream::Stdout => println!("{text}"),
            crate::tools::bash::Stream::Stderr => eprintln!("{text}"),
        }
    }
    fn tick(&mut self) -> bool {
        crate::interrupt::pending()
    }
}

/// Warning that opens a `!` command's transcript entry, so the model treats the
/// recorded command and its output as background the user happened to run and
/// not as a request addressed to it.
const BANG_CAVEAT: &str = "Caveat: The messages below were generated by the user while running local commands. DO NOT respond to these messages or otherwise consider them in your response unless the user explicitly asks you to.";

/// Head limits for the output a `!` command contributes to the transcript. A
/// `!` line is a convenience, not a turn the user is paying context for, so the
/// entry is capped well below what the `bash` tool would spill to disk.
const BANG_FEEDBACK_LINES: usize = 200;
/// Byte cap paired with [`BANG_FEEDBACK_LINES`].
const BANG_FEEDBACK_BYTES: usize = 16 * 1024;

/// Escapes the two characters that could forge the `<bash-…>` framing around
/// captured output. `>` and the quotes are left alone: shell output is usually
/// code or diffs, and escaping those would make it harder for the model to read
/// without buying any extra safety.
fn bang_escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;")
}

/// Keeps the first [`BANG_FEEDBACK_LINES`] lines / [`BANG_FEEDBACK_BYTES`] bytes
/// of `text`, appending a marker when anything was dropped.
fn bang_head(text: &str) -> String {
    let mut end = text.len();
    let mut lines = 0;
    for (i, b) in text.bytes().enumerate() {
        if i >= BANG_FEEDBACK_BYTES {
            end = i;
            break;
        }
        if b == b'\n' {
            lines += 1;
            if lines >= BANG_FEEDBACK_LINES {
                end = i + 1;
                break;
            }
        }
    }
    // A byte cap can land mid-codepoint; back off to a boundary.
    while end < text.len() && !text.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = bang_escape(&text[..end]);
    if end < text.len() {
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("[output truncated]\n");
    }
    out
}

/// Builds the single user message a `!` command appends to the transcript:
/// the caveat, the command as typed, and its captured output. No model turn is
/// triggered by it — the model sees it as history on the next real prompt.
fn bang_transcript_entry(
    cmd: &str,
    result: &Result<crate::tools::bash::ImmediateOutput, String>,
) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "<local-command-caveat>{BANG_CAVEAT}</local-command-caveat>"
    );
    let _ = writeln!(out, "<bash-input>{}</bash-input>", bang_escape(cmd));
    match result {
        Ok(o) => {
            let _ = writeln!(out, "<bash-stdout>{}</bash-stdout>", bang_head(&o.stdout));
            let _ = writeln!(out, "<bash-stderr>{}</bash-stderr>", bang_head(&o.stderr));
            if o.interrupted {
                let _ = writeln!(out, "<bash-interrupted>true</bash-interrupted>");
            } else if o.exit_code != 0 {
                let _ = writeln!(out, "<bash-exit-code>{}</bash-exit-code>", o.exit_code);
            }
        }
        Err(e) => {
            let _ = writeln!(
                out,
                "<bash-stderr>Command failed: {}</bash-stderr>",
                bang_escape(e)
            );
        }
    }
    out
}

/// Handles one line of plain-REPL input. Returns `Ok(false)` to quit the REPL
/// (a `/quit`-style slash command); `Ok(true)` to keep looping. Shared by the
/// local and remote-aware REPL loops so both paths treat slashes, `!`-shell
/// escapes, and prompts identically (CLAUDE.md: mirror both UI paths).
fn handle_plain_line(agent: &mut Agent<'_>, line: &str) -> Result<bool, String> {
    let input = line.trim();
    if input.is_empty() {
        return Ok(true);
    }
    if input.starts_with('/') {
        return agent.slash(input);
    }
    if let Some(rest) = input.strip_prefix('!') {
        // `!!` is user-only shell execution: output goes to the console but NOT
        // into the session transcript (issue #20). A single `!` runs the same
        // way but also records the command and its output as one user message,
        // so the model has it as history — still without triggering a turn.
        let (feedback, cmd) = match rest.strip_prefix('!') {
            Some(rest) => (false, rest.trim()),
            None => (true, rest.trim()),
        };
        if cmd.is_empty() {
            println!(
                "usage: !<shell command> (feeds the result to the model) or !!<shell command>"
            );
            return Ok(true);
        }
        let result =
            crate::tools::bash::run_immediate(&agent.tool_ctx.cwd, cmd, &mut BangConsoleSink);
        match &result {
            Ok(out) => {
                if out.interrupted {
                    crate::interrupt::clear();
                    println!("[interrupted]");
                } else if out.exit_code != 0 {
                    println!("[exit code: {}]", out.exit_code);
                }
            }
            Err(e) => println!("!{cmd}: {e}"),
        }
        if feedback {
            agent
                .session
                .push(Message::user(bang_transcript_entry(cmd, &result)));
        }
        return Ok(true);
    }
    print!("{}", status::format_user_prompt_echo(input, agent.color));
    agent.session.push(Message::user(input));
    agent.run_turn()?;
    Ok(true)
}

/// The classic blocking plain REPL (no remote bridge): read a line, handle it,
/// repeat until EOF.
fn run_repl_plain_local(agent: &mut Agent<'_>) -> Result<(), String> {
    let stdin = std::io::stdin();
    loop {
        print!("{}", status::prompt_text());
        std::io::stdout().flush().map_err(|e| e.to_string())?;
        let mut line = String::new();
        let n = stdin
            .lock()
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        if n == 0 {
            return Ok(()); // EOF
        }
        if !handle_plain_line(agent, &line)? {
            return Ok(());
        }
    }
}

/// Runs headless mode: one-shot with `-p`, else a stdin-driven protocol.
///
/// # Errors
/// Returns an error string on unrecoverable I/O or engine failure.
pub fn run_non_interactive(
    engine: Box<dyn Engine>,
    cfg: &AgentConfig,
    local_engine: Option<Box<dyn Engine>>,
) -> Result<(), String> {
    let mut agent = new_agent(engine, cfg, false, local_engine)?;
    // Seed the notification enable flag once, mirroring `run_interactive`, so
    // headless/non-interactive runs also honor `ui.notifications`.
    crate::notify::set_mode(crate::settings::active().ui.notifications);
    agent.warm_plain()?;
    agent.fire_session_start("startup", &mut |w| eprintln!("{w}"));
    if let Some(prompt) = cfg.prompt.as_deref() {
        agent.session.push(Message::user(prompt));
        let r = agent.run_turn();
        agent.fire_session_end("exit", &mut |w| eprintln!("{w}"));
        return r;
    }
    // Stdin protocol, like the C: announce readiness on stderr, collect bytes
    // until stdin has been quiet for 200 ms, submit that buffer as one prompt,
    // repeat until EOF. (The C also queues input arriving mid-generation; the
    // synchronous port reads between turns instead.)
    let mut eof = false;
    while !eof {
        eprintln!("+DWARFSTAR_WAITING");
        let Some(prompt) = read_quiet_batched(&mut eof).map_err(|e| e.to_string())? else {
            break;
        };
        if prompt.trim().is_empty() {
            continue;
        }
        agent.session.push(Message::user(prompt.trim_end()));
        agent.run_turn()?;
    }
    agent.fire_session_end("exit", &mut |w| eprintln!("{w}"));
    Ok(())
}

/// Reads one stdin batch: bytes accumulated until a 200 ms quiet window.
///
/// Returns `None` at EOF with nothing buffered; sets `eof` once stdin closes.
fn read_quiet_batched(eof: &mut bool) -> std::io::Result<Option<String>> {
    use std::io::Read as _;
    const QUIET_MS: i32 = 200;
    let mut buf = Vec::new();
    loop {
        let timeout = if buf.is_empty() { -1 } else { QUIET_MS };
        let mut pfd = libc::pollfd {
            fd: libc::STDIN_FILENO,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: pfd points to a valid pollfd for the duration of the call.
        let rc = unsafe { libc::poll(&raw mut pfd, 1, timeout) };
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if rc == 0 {
            // Quiet window elapsed with data buffered: submit it.
            break;
        }
        let mut chunk = [0_u8; 4096];
        let n = std::io::stdin().read(&mut chunk)?;
        if n == 0 {
            *eof = true;
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    if buf.is_empty() {
        return Ok(None);
    }
    Ok(Some(String::from_utf8_lossy(&buf).into_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineError, EngineEvent, GenerationStats, ThinkMode};
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Flattened text of a line, for asserting on layout.
    fn text_of(line: &ratatui::text::Line<'_>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn the_version_sits_beside_the_logo_not_below_it() {
        let art: Vec<ratatui::text::Line<'static>> = (0..7)
            .map(|i| ratatui::text::Line::from(format!("art{i}")))
            .collect();
        let out = Agent::masthead(art, "plank v9 🪵 Agent".to_string());

        // No extra row: the version rides an existing art line.
        assert_eq!(out.len(), 7, "the banner must not grow a row");
        let joined: Vec<String> = out.iter().map(text_of).collect();
        let carrying: Vec<usize> = joined
            .iter()
            .enumerate()
            .filter(|(_, t)| t.contains("plank v9"))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(carrying, vec![3], "on the middle row, exactly once");
        // The art on that row survives, and the version follows it.
        assert_eq!(joined[3], "art3  plank v9 🪵 Agent");
    }

    #[test]
    fn masthead_without_art_still_shows_the_version() {
        let out = Agent::masthead(Vec::new(), "plank v9".to_string());
        assert_eq!(out.len(), 1);
        assert_eq!(text_of(&out[0]), "plank v9");
    }

    #[test]
    fn masthead_middle_row_is_stable_for_even_and_odd_art() {
        for rows in 1..12usize {
            let art: Vec<ratatui::text::Line<'static>> = (0..rows)
                .map(|i| ratatui::text::Line::from(format!("a{i}")))
                .collect();
            let out = Agent::masthead(art, "V".to_string());
            assert_eq!(out.len(), rows, "{rows} rows in, {rows} rows out");
            let hits = out.iter().filter(|l| text_of(l).contains('V')).count();
            assert_eq!(
                hits, 1,
                "exactly one row carries the version at {rows} rows"
            );
        }
    }

    #[test]
    fn render_transcript_never_injects_the_task_list_mid_transcript() {
        // Append-only invariant: a task-list change must not rewrite the
        // rendered prompt between the system block and the messages, or the
        // engine's KV common-prefix reuse dies at the top of the transcript
        // and every turn re-prefills the whole conversation.
        let mut s = Session::new();
        s.push(Message::user("hello"));
        let before = render_transcript(&s, "SYS");
        s.tasks.add("do the thing", None);
        s.tasks
            .update(1, Some(crate::tasks::TaskStatus::InProgress), None, None)
            .unwrap();
        let after = render_transcript(&s, "SYS");
        assert_eq!(before, after, "task state must not perturb the rendering");
        assert!(after.starts_with("[system]\nSYS\n[user]\nhello\n"));
        assert!(!after.contains("# Task list"), "{after}");
    }

    #[test]
    fn task_list_survives_transcript_compaction() {
        // Compaction rewrites the transcript but never the task list state
        // (issue #35 acceptance); the model re-sees it via the one-time
        // re-injection in `rebuild_after_compact`, not via render_transcript.
        let mut s = Session::new();
        for i in 0..40 {
            s.push(Message::user(format!(
                "<tool_result>{}</tool_result>",
                "x".repeat(500)
            )));
            s.push(Message::assistant(format!("reply {i}")));
        }
        s.tasks.add("keep me across compaction", None);
        let before = s.tasks.clone();
        let cleared = crate::compact::microcompact(&mut s.transcript);
        assert!(cleared > 0, "compaction should clear some large results");
        assert_eq!(s.tasks, before, "compaction leaves the task list untouched");
        let block = s.tasks.inject_block().expect("non-empty list has a block");
        assert!(block.contains("keep me across compaction"));
    }

    /// Regression: `run_turn` must not force `sub_sink` to `Stdout`. It is
    /// called by both the plain REPL and `run_non_interactive` (the `-p`
    /// one-shot path and the stdin-protocol loop), and the headless path's
    /// stdout carries the `+DWARFSTAR_WAITING` / one-shot machine protocol
    /// that interleaved sub-agent model text would corrupt. An `Agent` built
    /// the way `run_non_interactive` builds it (default `sub_sink`, i.e.
    /// `Null`) must still have `sub_sink == Null` after a turn runs, and any
    /// sub-agent output emitted through that sink must be silently dropped
    /// rather than printed.
    #[test]
    fn run_turn_does_not_force_sub_sink_to_stdout() {
        let dir = scratch_dir("sub-sink-headless");
        let engine = ScriptedEngine {
            replies: vec!["headless reply\n".to_string()],
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);

        // Sanity: this is the same default `new_agent`/`run_non_interactive`
        // leave in place (no assignment on the non-interactive path).
        assert!(matches!(agent.sub_sink, SubSinkTarget::Null));

        agent.session.push(Message::user("hi"));
        agent.run_turn().expect("turn runs");

        assert!(
            matches!(agent.sub_sink, SubSinkTarget::Null),
            "run_turn must leave a headless agent's sub_sink as Null, got {:?}",
            agent.sub_sink
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rebuild_after_compact_reinjects_the_task_list_once() {
        let dir = scratch_dir("compact-tasks");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent.session.push(Message::user("old context"));
        agent.session.tasks.add("still pending after compact", None);
        agent.rebuild_after_compact("<summary>did things</summary>");
        let rendered = render_transcript(&agent.session, "SYS");
        assert!(
            rendered.contains("# Task list") && rendered.contains("still pending after compact"),
            "{rendered}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A `Write` sink backed by a shared buffer so a test can inspect the exact
    /// bytes the terminal renderer emits.
    #[derive(Clone)]
    struct SharedBuf(Rc<RefCell<Vec<u8>>>);

    impl Write for SharedBuf {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.borrow_mut().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Regression for #48: a tool-call banner param value containing markdown
    /// metacharacters (`*`, `_`, backtick) must render verbatim on the plain /
    /// non-interactive stdout path — the model sent `pattern=**/x.rs` but the
    /// banner used to drop the `*`s because `tool_text` fell through to the
    /// markdown-processing `visible_text`.
    #[test]
    fn tool_banner_param_value_renders_metachars_verbatim() {
        let buf = Rc::new(RefCell::new(Vec::new()));
        let sink = TerminalSink {
            renderer: TokenRenderer::new(
                SharedBuf(buf.clone()),
                RenderOptions {
                    use_color: false,
                    format_thinking: true,
                    format_markdown: true,
                },
            ),
        };
        let mut stream = StreamRenderer::new(sink);
        let stanza = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"search\">",
            "<｜DSML｜parameter name=\"pattern\">**/x.rs a_b `c`</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        stream.push(stanza);
        stream.finish();
        drop(stream);
        let out = String::from_utf8(buf.borrow().clone()).unwrap();
        assert!(
            out.contains("**/x.rs"),
            "star stripped from banner: {out:?}"
        );
        assert!(out.contains("a_b"), "underscore mangled: {out:?}");
        assert!(out.contains("`c`"), "backtick eaten: {out:?}");
    }

    /// A `TuiInput` whose history holds an interleaved mix of prompts and
    /// `!` commands, for the mode-aware navigation tests.
    fn input_with_mixed_history() -> TuiInput {
        let mut input = TuiInput::new();
        for e in ["write a test", "!ls -la", "explain this", "!git status"] {
            input.history.add(e);
        }
        input
    }

    #[test]
    fn history_in_prompt_mode_visits_every_entry() {
        let mut input = input_with_mixed_history();
        let mut seen = Vec::new();
        for _ in 0..4 {
            input.history_move(-1);
            seen.push(input.buf.text().to_string());
        }
        assert_eq!(
            seen,
            ["!git status", "explain this", "!ls -la", "write a test"]
        );
    }

    #[test]
    fn history_on_a_bang_line_visits_only_bang_entries() {
        let mut input = input_with_mixed_history();
        input.buf.set_text("!");
        input.buf.move_end();
        let mut seen = Vec::new();
        for _ in 0..2 {
            input.history_move(-1);
            seen.push(input.buf.text().to_string());
        }
        assert_eq!(seen, ["!git status", "!ls -la"]);
    }

    #[test]
    fn bash_mode_is_fixed_when_the_walk_starts() {
        // Loading a `!` entry makes the buffer start with `!`. If mode were
        // re-derived per keypress, a walk begun in prompt mode would switch to
        // bash mode mid-cycle and strand the user.
        let mut input = input_with_mixed_history();
        input.history_move(-1);
        assert_eq!(input.buf.text(), "!git status");
        input.history_move(-1);
        assert_eq!(input.buf.text(), "explain this", "mode flipped mid-walk");
    }

    #[test]
    fn bash_mode_with_no_bang_entries_leaves_the_line_alone() {
        let mut input = TuiInput::new();
        input.history.add("write a test");
        input.buf.set_text("!gi");
        input.buf.move_end();
        input.history_move(-1);
        assert_eq!(input.buf.text(), "!gi");
    }

    #[test]
    fn history_walk_restores_the_stashed_line_on_the_way_back() {
        let mut input = input_with_mixed_history();
        input.buf.set_text("!half typed");
        input.buf.move_end();
        input.history_move(-1);
        assert_eq!(input.buf.text(), "!git status");
        input.history_move(1);
        assert_eq!(input.buf.text(), "!half typed");
    }

    /// A `TuiInput` with history spread across two directories: prompts and
    /// `!` commands tagged `/proj/a`, and one of each tagged `/proj/b`. The
    /// current directory is pinned to `/proj/a`.
    fn input_with_dir_scoped_history() -> TuiInput {
        let mut input = TuiInput::new();
        let h = &mut input.history;
        h.add_in_dir("build a", Some("/proj/a".into()));
        h.add_in_dir("!ls a", Some("/proj/a".into()));
        h.add_in_dir("build b", Some("/proj/b".into()));
        h.add_in_dir("!ls b", Some("/proj/b".into()));
        h.set_cwd(Some("/proj/a".into()));
        input
    }

    #[test]
    fn history_hides_entries_from_other_directories() {
        let mut input = input_with_dir_scoped_history();
        let mut seen = Vec::new();
        for _ in 0..4 {
            input.history_move(-1);
            seen.push(input.buf.text().to_string());
        }
        // Only /proj/a entries appear; the /proj/b ones never surface and the
        // walk clamps at the oldest eligible entry.
        assert_eq!(seen, ["!ls a", "build a", "build a", "build a"]);
    }

    #[test]
    fn dir_filter_composes_with_bash_mode_filter() {
        // A `!` walk in /proj/a cycles `!` commands only, and only those tagged
        // for the current directory: `!ls b` (from /proj/b) must not appear.
        let mut input = input_with_dir_scoped_history();
        input.buf.set_text("!");
        input.buf.move_end();
        let mut seen = Vec::new();
        for _ in 0..2 {
            input.history_move(-1);
            seen.push(input.buf.text().to_string());
        }
        assert_eq!(seen, ["!ls a", "!ls a"]);
    }

    #[test]
    fn legacy_untagged_history_visits_from_any_directory() {
        // Untagged (pre-#49) entries behave globally: still navigable even when
        // the current directory has no scoped history of its own.
        let mut input = TuiInput::new();
        input.history.add_in_dir("legacy one", None);
        input.history.add_in_dir("legacy two", None);
        input.history.set_cwd(Some("/unrelated/dir".into()));
        input.history_move(-1);
        assert_eq!(input.buf.text(), "legacy two");
        input.history_move(-1);
        assert_eq!(input.buf.text(), "legacy one");
    }

    /// Types `text` into a fresh `TuiInput` one character at a time, re-syncing
    /// the menus after each one exactly as the key loops do.
    fn typed(text: &str) -> TuiInput {
        let mut input = TuiInput::new();
        for c in text.chars() {
            input.buf.insert(c.to_string());
            input.sync_popup();
        }
        input
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn shift(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::SHIFT)
    }

    #[test]
    fn typing_a_slash_opens_the_command_menu_and_a_space_closes_it() {
        let input = typed("/");
        let menu = input.slash.as_ref().expect("menu opens on a bare slash");
        assert!(!menu.is_empty());
        assert!(
            menu.rows().iter().any(|e| e.name == "/help"),
            "the bare menu lists everything"
        );
        assert!(typed("/compact ").slash.is_none(), "arguments close it");
    }

    #[test]
    fn the_command_menu_narrows_as_the_name_is_typed() {
        let input = typed("/comp");
        let menu = input.slash.as_ref().expect("menu");
        assert_eq!(menu.selected_entry().expect("row").name, "/compact");
    }

    #[test]
    fn an_unknown_command_shows_no_menu_rather_than_an_empty_box() {
        assert!(typed("/zzzznotacommand").slash.is_none());
    }

    #[test]
    fn a_slash_mid_sentence_does_not_open_the_menu() {
        assert!(typed("look at /help").slash.is_none());
    }

    #[test]
    fn accepting_a_command_from_the_menu_rewrites_the_line_and_closes_it() {
        let mut input = typed("/comp");
        assert!(input.popup_key(key(KeyCode::Enter)), "menu takes the Enter");
        assert_eq!(input.buf.text(), "/compact ");
        assert!(input.slash.is_none());
    }

    /// A fully typed command must run on the first Enter: the menu hands the
    /// key back rather than "completing" what is already complete.
    #[test]
    fn enter_on_a_fully_typed_command_submits_instead_of_completing() {
        let mut input = typed("/help");
        assert!(input.slash.is_some(), "the menu is still up");
        assert!(
            !input.popup_key(key(KeyCode::Enter)),
            "Enter passes through"
        );
        assert_eq!(input.buf.text(), "/help", "the line is untouched");
    }

    /// A fuzzy highlight must never displace a command the user finished
    /// typing. `mat` is a subsequence of `compact`, so a menu that completed
    /// on Enter regardless would turn `/matrix` into `/compact` — swapping an
    /// easter egg for a context-destroying summarization on one keystroke.
    #[test]
    fn a_hidden_command_is_never_swapped_for_a_fuzzy_match() {
        let mut input = typed("/matrix");
        assert!(
            !input.popup_key(key(KeyCode::Enter)),
            "Enter passes through"
        );
        assert_eq!(input.buf.text(), "/matrix");
        // Mid-word the menu may well be showing something else; what matters is
        // that finishing the name takes the line back.
        let mut half = typed("/mat");
        assert!(half.slash.is_some(), "a fuzzy menu is up mid-word");
        for c in "rix".chars() {
            half.buf.insert(c.to_string());
            half.sync_popup();
        }
        assert!(!half.popup_key(key(KeyCode::Enter)));
        assert_eq!(half.buf.text(), "/matrix");
    }

    #[test]
    fn tab_on_a_fully_typed_command_still_completes_it() {
        let mut input = typed("/help");
        assert!(input.popup_key(key(KeyCode::Tab)));
        assert_eq!(input.buf.text(), "/help ");
    }

    #[test]
    fn esc_closes_the_command_menu_without_touching_the_line() {
        let mut input = typed("/comp");
        assert!(input.popup_key(key(KeyCode::Esc)));
        assert!(input.slash.is_none());
        assert_eq!(input.buf.text(), "/comp");
    }

    #[test]
    fn the_command_menu_and_the_at_popup_are_never_open_at_once() {
        let at = typed("@src");
        assert!(at.slash.is_none());
        let slash = typed("/co");
        assert!(slash.popup.is_none());
    }

    #[test]
    fn shift_arrows_select_and_plain_arrows_drop_the_selection() {
        let mut input = typed("hello world");
        // Three Shift+Lefts: the loop runs `selection_key` (anchoring) then its
        // own motion binding.
        for _ in 0..3 {
            assert!(!input.selection_key(shift(KeyCode::Left)));
            input.buf.move_left();
        }
        assert_eq!(input.buf.selected_text(), Some("rld"));
        assert_eq!(input.selection_chars(), Some((8, 11)));
        // A plain Left collapses it.
        assert!(!input.selection_key(key(KeyCode::Left)));
        input.buf.move_left();
        assert_eq!(input.buf.selection(), None);
    }

    #[test]
    fn shift_home_selects_back_to_the_start() {
        let mut input = typed("hello");
        assert!(!input.selection_key(shift(KeyCode::Home)));
        input.buf.move_home();
        assert_eq!(input.buf.selected_text(), Some("hello"));
    }

    /// Unshifted Up/Down keep walking history, so the selection keymap must
    /// only claim them when Shift is held.
    #[test]
    fn shift_up_moves_by_line_while_plain_up_is_left_for_history() {
        let mut input = TuiInput::new();
        input.buf.set_text("one\ntwo");
        assert!(input.selection_key(shift(KeyCode::Up)), "consumed");
        assert_eq!(input.buf.selected_text(), Some("\ntwo"));
        let mut plain = TuiInput::new();
        plain.buf.set_text("one\ntwo");
        assert!(!plain.selection_key(key(KeyCode::Up)), "passed through");
        assert_eq!(plain.buf.cursor(), plain.buf.text().len());
    }

    /// A consumed selection key skips the loop's end-of-iteration re-sync, so
    /// it has to do the re-sync itself or a menu is left standing over text
    /// that no longer justifies it.
    #[test]
    fn a_consumed_selection_key_re_syncs_the_menus() {
        let mut input = typed("/comp");
        assert!(input.slash.is_some());
        // Shift+Up walks to the start of the line: the caret is no longer at
        // the end of the command token, so the menu must have closed.
        assert!(input.selection_key(shift(KeyCode::Up)));
        assert_eq!(input.buf.cursor(), 0);
        assert!(input.slash.is_none(), "the menu must not outlive the caret");
    }

    #[test]
    fn ctrl_shift_a_selects_the_whole_line() {
        let mut input = typed("select me");
        let ctrl_shift_a = KeyEvent::new(
            KeyCode::Char('a'),
            KeyModifiers::CONTROL | KeyModifiers::SHIFT,
        );
        assert!(input.selection_key(ctrl_shift_a));
        assert_eq!(input.buf.selected_text(), Some("select me"));
    }

    /// Ctrl-C only claims the key when there is something to copy; with an
    /// empty selection it must still mean "clear the line" / "interrupt".
    #[test]
    fn ctrl_c_copies_a_selection_and_otherwise_passes_through() {
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        let mut empty = typed("nothing selected");
        assert!(!empty.selection_key(ctrl_c));
        let mut input = typed("copy me");
        input.buf.select_all();
        assert!(input.selection_key(ctrl_c));
        assert_eq!(
            input.buf.selected_text(),
            Some("copy me"),
            "a copy leaves the selection standing"
        );
    }

    #[test]
    fn ctrl_x_cuts_the_selection_out_of_the_line() {
        let ctrl_x = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL);
        let mut input = typed("keep cut");
        input.buf.set_cursor(4);
        input.buf.anchor_here();
        input.buf.set_cursor(8);
        assert!(input.selection_key(ctrl_x));
        assert_eq!(input.buf.text(), "keep");
        assert_eq!(input.buf.selection(), None);
    }

    #[test]
    fn a_readline_motion_collapses_the_selection() {
        let ctrl_a = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL);
        let mut input = typed("hello");
        input.buf.select_all();
        assert!(!input.selection_key(ctrl_a), "Ctrl-A is still Home");
        assert_eq!(input.buf.selection(), None);
    }

    /// A press places the caret and arms a drag without selecting anything;
    /// the drag that follows selects from the press point.
    #[test]
    fn a_press_places_the_caret_and_the_drag_after_it_selects() {
        let rect = ratatui::layout::Rect::new(4, 10, 20, 1);
        let mut input = typed("hello world");
        // Press on column 4+6 → char 6 ('w').
        assert!(input.mouse_to_cursor(rect, 10, 10, false));
        assert_eq!(input.buf.cursor(), 6);
        assert_eq!(input.buf.selection(), None, "a plain click selects nothing");
        // Drag to the end of the text.
        assert!(input.mouse_to_cursor(rect, 15, 10, true));
        assert_eq!(input.buf.selected_text(), Some("world"));
    }

    #[test]
    fn a_click_outside_the_prompt_is_left_to_the_output_pane() {
        let rect = ratatui::layout::Rect::new(4, 10, 20, 1);
        let mut input = typed("hello");
        assert!(
            !input.mouse_to_cursor(rect, 10, 3, false),
            "a different row"
        );
        assert_eq!(input.buf.cursor(), 5, "the caret did not move");
    }

    /// The selection is reported to the renderer in char indices, not bytes.
    #[test]
    fn the_selection_handed_to_the_renderer_is_measured_in_chars() {
        let mut input = typed("aé漢b");
        input.buf.select_all();
        assert_eq!(input.selection_chars(), Some((0, 4)));
        assert_eq!(input.state().sel, Some((0, 4)));
        assert_eq!(input.state().cursor, 4);
    }

    /// Builds a `TuiInput` whose popup is open with one canned row.
    fn input_with_popup(text: &str, cursor_back: usize) -> TuiInput {
        use crate::complete::{IndexMsg, Kind, Match, Popup, detect_at_token};
        let mut input = TuiInput::new();
        input.buf.set_text(text);
        input.buf.move_end();
        for _ in 0..cursor_back {
            input.buf.move_left();
        }
        let token = detect_at_token(text).expect("token");
        let mut p = Popup::new(token);
        let generation = p.generation();
        p.accept_msg(IndexMsg::Results {
            generation,
            rows: vec![Match {
                text: "source.rs".to_owned(),
                kind: Kind::File,
                score: 0,
            }],
        });
        input.popup = Some(p);
        input
    }

    /// Builds a `TuiInput` whose popup is open with several canned rows.
    fn input_with_rows(text: &str, rows: &[&str]) -> TuiInput {
        use crate::complete::{IndexMsg, Kind, Match, Popup, detect_at_token};
        let mut input = TuiInput::new();
        input.buf.set_text(text);
        input.buf.move_end();
        let mut p = Popup::new(detect_at_token(text).expect("token"));
        let generation = p.generation();
        p.accept_msg(IndexMsg::Results {
            generation,
            rows: rows
                .iter()
                .map(|t| Match {
                    text: (*t).to_owned(),
                    kind: Kind::File,
                    score: 0,
                })
                .collect(),
        });
        input.popup = Some(p);
        input
    }

    #[test]
    fn arrow_selection_is_not_cancelled_by_a_re_query() {
        // Regression: `popup_key` used to call `sync_popup()` for every
        // Consumed key, re-issuing an identical query whose reply reset
        // `selected` to 0 within one tick.
        let mut input = input_with_rows("@a", &["a1.rs", "a2.rs", "a3.rs"]);
        let before_gen = input.popup.as_ref().unwrap().generation();
        assert!(input.popup_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE)));
        assert_eq!(input.popup.as_ref().unwrap().selected(), 1);
        assert_eq!(
            input.popup.as_ref().unwrap().generation(),
            before_gen,
            "a pure selection key must not re-query"
        );
        assert!(input.worker.is_none(), "no worker started for Down");
        input.pump_popup();
        assert_eq!(
            input.popup.as_ref().unwrap().selected(),
            1,
            "selection must survive a pump"
        );
    }

    #[test]
    fn a_buffer_mutating_popup_key_still_re_queries() {
        // Tab rewrites the token, so the query genuinely changed.
        let mut input = input_with_rows("@a", &["a1.rs", "a1x.rs"]);
        let before_gen = input.popup.as_ref().unwrap().generation();
        assert!(input.popup_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)));
        assert_eq!(input.buf.text(), "@a1");
        assert!(input.popup.as_ref().unwrap().generation() > before_gen);
        input.worker = None;
    }

    #[test]
    fn a_refreshed_message_re_queries_the_open_popup() {
        use crate::complete::IndexWorker;
        let dir = std::env::temp_dir().join(format!("plank-ui-refresh-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("ui.rs"), b"x").unwrap();
        let mut input = input_with_rows("@ui", &["stale.rs"]);
        let before_gen = input.popup.as_ref().unwrap().generation();
        // The worker emits `Refreshed` once its untracked fold completes.
        input.worker = Some(IndexWorker::spawn(dir.clone(), Vec::new(), true));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            input.pump_popup();
            if input.popup.as_ref().unwrap().generation() > before_gen {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "Refreshed never triggered a re-query"
            );
            std::thread::yield_now();
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn popup_closes_when_the_cursor_moves_off_the_token_end() {
        // `@src` with the cursor moved two left sits mid-token: accepting there
        // would splice the completion in front of the stale `rc` tail.
        let mut input = input_with_popup("@src", 2);
        input.sync_popup();
        assert!(
            input.popup.is_none(),
            "popup must not survive a cursor move into the token"
        );
        // And the key is no longer consumed, so no mangled text can be written.
        let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
        assert!(!input.popup_key(enter));
        assert_eq!(input.buf.text(), "@src");
    }

    #[test]
    fn popup_survives_while_the_cursor_stays_at_the_token_end() {
        let mut input = input_with_popup("@src", 0);
        assert!(input.popup.is_some());
        let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
        assert!(input.popup_key(enter));
        assert_eq!(input.buf.text(), "source.rs ");
    }

    /// Engine that plays back canned replies in order. Records the prompt of
    /// every generate call in `prompts` (shared, so tests can inspect it
    /// after the engine moves into the Agent) and reports the pass at index
    /// `interrupt_at` as user-interrupted.
    #[derive(Debug, Default)]
    // Independent capability knobs on a test double, not a state machine: each
    // flag turns one real engine behaviour on or off so a test can stand exactly
    // where it needs to.
    #[allow(clippy::struct_excessive_bools)]
    struct ScriptedEngine {
        replies: Vec<String>,
        next: usize,
        prompts: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        interrupt_at: Option<usize>,
        /// When true, `generate_aside` is implemented (mirrors a real engine's
        /// snapshot/restore support) so the in-pass `/btw` suspend path runs
        /// instead of falling back to the boundary queue.
        aside_support: bool,
        /// When true, `generate_aside_forked` is implemented too, so the
        /// fork tier is available above the destructive one.
        fork_aside_support: bool,
        /// When true, the engine reports it can run a main pass and an aside
        /// interleaved, so the `/btw` path multiplexes instead of freezing.
        multiplex_support: bool,
        /// Records which aside tier each call took ("forked" / "destructive"),
        /// so tier-selection tests can assert on the choice rather than on the
        /// reply text, which is identical either way.
        aside_tiers: Option<std::sync::Arc<std::sync::Mutex<Vec<String>>>>,
        /// When set, the engine pretends to have KV snapshot support:
        /// `get_kv` logs `capture` and returns a dummy payload tagged with an
        /// incrementing counter (so nested captures are distinguishable),
        /// `set_kv` logs `restore:<tag>`, and each generate logs `generate` —
        /// letting fork tests assert the capture → sidechain → restore order.
        kv_events: Option<std::sync::Arc<std::sync::Mutex<Vec<String>>>>,
        kv_captures: u8,
        /// Overrides the reported context size, so tests can stand on either
        /// side of `THINK_MAX_MIN_CONTEXT`. `None` reports the usual `100_000`.
        ctx_override: Option<i32>,
        /// When true the engine claims to run on this machine's weights, which is
        /// what `generate_pass` keys the status bar's blinking brain off.
        local: bool,
        /// Records `status::local_pass_active()` as observed from *inside*
        /// `generate`, so a test can assert the pass marked itself while it was
        /// actually generating rather than merely before or after.
        saw_local_pass: Option<std::sync::Arc<AtomicBool>>,
        /// Records every `set_think_mode` call, so a test can assert the level
        /// change reached the engine (where it drops cached tokens and KV).
        think_modes: Option<std::sync::Arc<std::sync::Mutex<Vec<ThinkMode>>>>,
        /// Records every `set_trusted_system_prefix` call, the other half of
        /// the configuration an engine needs before it tokenizes anything.
        trusted_lens: Option<std::sync::Arc<std::sync::Mutex<Vec<usize>>>>,
        /// Records each `warm_reset` system text, so a test can assert which
        /// tiers a warm walk actually covered.
        warm_tiers: Option<std::sync::Arc<std::sync::Mutex<Vec<String>>>>,
        /// Overrides the reported model name, so two spy engines can key
        /// different Tier 1 checkpoints.
        model: Option<String>,
        /// When set, `generate` fails with this message instead of replying, so
        /// tests can exercise the error paths (e.g. that a swapped-in sub-agent
        /// engine is still returned to its cache when the sidechain dies).
        fail_with: Option<String>,
    }

    impl ScriptedEngine {
        /// Notes which aside tier a call took, when the test is watching.
        fn record_tier(&self, tier: &str) {
            if let Some(tiers) = &self.aside_tiers {
                tiers.lock().unwrap().push(tier.to_owned());
            }
        }

        /// Records the prompt and streams the next scripted reply in chunks.
        fn play_next(
            &mut self,
            transcript: &str,
            on_event: &mut dyn FnMut(EngineEvent),
        ) -> GenerationStats {
            self.prompts.lock().unwrap().push(transcript.to_owned());
            if let Some(events) = &self.kv_events {
                events.lock().unwrap().push("generate".to_string());
            }
            let interrupted = self.interrupt_at == Some(self.next);
            let reply = self.replies.get(self.next).cloned().unwrap_or_default();
            self.next += 1;
            // Stream in small chunks to exercise partial-marker handling.
            let mut i = 0;
            while i < reply.len() {
                let mut end = (i + 7).min(reply.len());
                while !reply.is_char_boundary(end) {
                    end += 1;
                }
                on_event(EngineEvent::Text(reply[i..end].to_string()));
                i = end;
            }
            GenerationStats {
                interrupted,
                ..GenerationStats::default()
            }
        }
    }

    impl Engine for ScriptedEngine {
        fn is_local(&self) -> bool {
            self.local
        }
        fn generate(
            &mut self,
            prompt: crate::engine::Prompt<'_>,
            _opts: &crate::engine::GenerationOptions,
            _interrupt: &dyn Fn() -> bool,
            _greedy: &dyn Fn() -> bool,
            on_event: &mut dyn FnMut(EngineEvent),
        ) -> Result<GenerationStats, EngineError> {
            if let Some(seen) = &self.saw_local_pass {
                seen.store(crate::status::local_pass_active(), Ordering::Relaxed);
            }
            if let Some(msg) = &self.fail_with {
                return Err(EngineError::new(msg.clone()));
            }
            Ok(self.play_next(prompt.flat(), on_event))
        }
        fn generate_aside(
            &mut self,
            prompt: &str,
            _opts: &crate::engine::GenerationOptions,
            _interrupt: &dyn Fn() -> bool,
            on_event: &mut dyn FnMut(EngineEvent),
        ) -> Result<GenerationStats, EngineError> {
            if !self.aside_support {
                return Err(EngineError::unsupported());
            }
            self.record_tier("destructive");
            Ok(self.play_next(prompt, on_event))
        }
        fn supports_aside(&self) -> bool {
            self.aside_support
        }
        fn generate_aside_forked(
            &mut self,
            prompt: &str,
            _opts: &crate::engine::GenerationOptions,
            _interrupt: &dyn Fn() -> bool,
            on_event: &mut dyn FnMut(EngineEvent),
        ) -> Result<GenerationStats, EngineError> {
            if !self.fork_aside_support {
                return Err(EngineError::unsupported());
            }
            self.record_tier("forked");
            Ok(self.play_next(prompt, on_event))
        }
        fn supports_forked_aside(&self) -> bool {
            self.fork_aside_support
        }
        fn supports_multiplexing(&self) -> bool {
            self.multiplex_support
        }
        fn get_kv(&mut self) -> Option<crate::kvcache::KVCache> {
            let events = self.kv_events.as_ref()?;
            self.kv_captures += 1;
            events.lock().unwrap().push("capture".to_string());
            Some(crate::kvcache::KVCache::new(
                vec![self.kv_captures],
                crate::ds4tokens::TokenTranscript::new(),
            ))
        }
        fn set_kv(&mut self, cache: &crate::kvcache::KVCache) -> Result<(), EngineError> {
            if let Some(events) = &self.kv_events {
                let tag = cache.kv().first().copied().unwrap_or(0);
                events.lock().unwrap().push(format!("restore:{tag}"));
            }
            Ok(())
        }
        fn ctx_size(&self) -> i32 {
            self.ctx_override.unwrap_or(100_000)
        }

        fn set_think_mode(&mut self, mode: ThinkMode) {
            if let Some(seen) = &self.think_modes {
                seen.lock().unwrap().push(mode);
            }
        }

        fn set_trusted_system_prefix(&mut self, len: usize) {
            if let Some(seen) = &self.trusted_lens {
                seen.lock().unwrap().push(len);
            }
        }

        fn warm_reset(&mut self, system: &str) -> Result<(), crate::engine::EngineError> {
            if let Some(seen) = &self.warm_tiers {
                seen.lock().unwrap().push(system.to_owned());
            }
            Ok(())
        }

        fn model_name(&self) -> String {
            self.model.clone().unwrap_or_default()
        }
    }

    /// Builds an Agent over a scripted engine with the standard test fields.
    fn test_agent<'a>(
        dir: &std::path::Path,
        engine: ScriptedEngine,
        cfg: &'a crate::config::AgentConfig,
    ) -> Agent<'a> {
        Agent {
            engine: Box::new(engine),
            cfg,
            session: Session::new(),
            store: SessionStore::open(dir).unwrap(),
            pending_aside: None,
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            isolation_seq: 0,
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            payload_restored: false,
            trusted_system_len: 0,
            think: cfg.generation.think_mode,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
            last_ctx_used: 0,
            last_turn_interrupted: false,
            context_content: crate::context::ContextContent::new(),
            skills: Vec::new(),
            templates: Vec::new(),
            agents: Vec::new(),
            checkpoints: crate::checkpoint::CheckpointStore::new(),
            remote: None,
            remote_server: None,
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
            sub_sink: SubSinkTarget::default(),
            fork_kv: Vec::new(),
            alt_engines: std::collections::HashMap::new(),
            local_alt_warmed: false,
            warm_note: None,
        }
    }

    fn test_cfg() -> crate::config::AgentConfig {
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.think_mode = crate::engine::ThinkMode::Off;
        cfg
    }

    /// Regression: the screensaver's idle clock must not be reset by focus or
    /// resize events. A window manager cycling focus (or, as it happens, an
    /// agent driving the terminal) fires those constantly, and treating them
    /// as activity pins the timer at zero so the screensaver never appears.
    #[test]
    fn only_keys_mouse_and_paste_count_as_user_activity() {
        use ratatui::crossterm::event::{
            KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
        };

        let key = Event::Key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE));
        let mouse = Event::Mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 1,
            row: 1,
            modifiers: KeyModifiers::NONE,
        });
        let paste = Event::Paste("text".to_string());
        for ev in [key, mouse, paste] {
            assert!(is_user_activity(&ev), "{ev:?} is the user");
        }

        for ev in [Event::FocusGained, Event::FocusLost, Event::Resize(80, 24)] {
            assert!(!is_user_activity(&ev), "{ev:?} is not the user");
        }
    }

    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("plank-ui-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// `remote_on` installs a live bridge on an already-running agent and
    /// `remote_off` tears it down; the link carries the bound port and token.
    #[test]
    fn remote_on_installs_a_bridge_and_remote_off_tears_it_down() {
        let dir = scratch_dir("remote-toggle");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        assert!(!agent.remote_is_on());

        let (addr, token) = agent
            .remote_on("127.0.0.1:0", Some("tok".to_owned()), true)
            .expect("binds an ephemeral loopback port");
        assert!(agent.remote_is_on());
        assert!(agent.remote.is_some());
        assert_ne!(addr.port(), 0, "an ephemeral bind resolves to a real port");
        assert_eq!(
            Agent::remote_link(addr, &token),
            format!("http://127.0.0.1:{}/?t=tok", addr.port())
        );
        // The bound port really accepts connections while on.
        std::net::TcpStream::connect(addr).expect("listener is live");

        // Idempotent: a second on returns the same address and token.
        let (again, same_token) = agent
            .remote_on("127.0.0.1:0", Some("other".to_owned()), true)
            .expect("second on is a no-op");
        assert_eq!(again, addr);
        assert_eq!(same_token, token);

        assert!(agent.remote_off(), "reports that a server was running");
        assert!(!agent.remote_is_on());
        assert!(agent.remote.is_none());
        assert!(!agent.remote_off(), "second off is a no-op");
    }

    /// `remote_off` really drops the server rather than leaking it: the test's
    /// own clone of the shared state must be the last one standing. A leak
    /// (`mem::forget`-ing the server instead of shutting it down) keeps the
    /// server's clone alive and the count stays above 1.
    ///
    /// This deliberately never connects to the listener. A connection makes the
    /// accept loop spawn a handler thread holding its own clone of the state,
    /// which lingers past `remote_off` and inflates the count — so the live-port
    /// probe lives in `remote_on_installs_a_bridge_and_remote_off_tears_it_down`
    /// and the refcount check lives here, never in the same test. Probing the
    /// port *after* `remote_off` is not an option either: another test in the
    /// parallel suite can rebind a released ephemeral port immediately.
    #[test]
    fn remote_off_drops_the_server_rather_than_leaking_it() {
        let dir = scratch_dir("remote-off-drops");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent
            .remote_on("127.0.0.1:0", Some("tok".to_owned()), true)
            .expect("binds an ephemeral loopback port");
        let state = std::sync::Arc::clone(agent.remote.as_ref().expect("bridge installed"));

        assert!(agent.remote_off());
        assert_eq!(
            std::sync::Arc::strong_count(&state),
            1,
            "remote_off dropped the server, so the test holds the last RemoteState reference"
        );
    }

    /// A generated token is used when none is supplied, and it is not empty.
    #[test]
    fn remote_on_generates_a_token_when_none_is_given() {
        let dir = scratch_dir("remote-toggle-token");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        let (_addr, token) = agent.remote_on("127.0.0.1:0", None, true).expect("binds");
        assert!(!token.is_empty());
        agent.remote_off();
    }

    /// A remote's context gauge and its stop button both key off status
    /// frames, which otherwise only arrive from engine callbacks *during* a
    /// turn. Without an idle frame at the end, the last thing a remote ever
    /// sees is `generating`.
    #[test]
    fn turn_end_publishes_an_idle_status_to_the_bus() {
        let dir = scratch_dir("idle-status-remote");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent
            .remote_on("127.0.0.1:0", Some("tok".to_owned()), true)
            .expect("binds");
        let state = agent.remote.clone().expect("bridge installed");
        let sub = state.bus.subscribe();

        agent.broadcast_idle_status();

        let seen: Vec<_> = sub.try_iter().map(|s| s.event).collect();
        let found = seen.iter().any(
            |e| matches!(e, UiEvent::Status(st) if st.state == crate::status::WorkerState::Idle),
        );
        assert!(found, "no idle status reached the bus: {seen:?}");
        agent.remote_off();
    }

    /// `/clear` has to tell attached clients, or the page keeps showing a
    /// session that no longer exists. The bug this pins was exactly here: the
    /// arm reset `self.session` and cleared the *local* log, and nothing
    /// reached the bus.
    #[test]
    fn clear_broadcasts_a_session_reset_to_attached_clients() {
        let dir = scratch_dir("clear-resets-remote");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent
            .remote_on("127.0.0.1:0", Some("tok".to_owned()), true)
            .expect("binds");
        let state = agent.remote.clone().expect("bridge installed");
        let sub = state.bus.subscribe();
        state.bus.broadcast(UiEvent::Visible("old output".into()));

        agent.slash("/clear").expect("clear runs");

        let seen: Vec<_> = sub.try_iter().map(|s| s.event).collect();
        assert!(
            seen.iter().any(|e| matches!(e, UiEvent::SessionReset)),
            "no reset reached the bus: {seen:?}"
        );
        agent.remote_off();
    }

    /// The shared toggle helper drives the bridge and yields the lines both
    /// front-ends print, so the plain REPL and the TUI cannot drift.
    #[test]
    fn rc_toggle_helper_turns_the_bridge_on_and_off() {
        let dir = scratch_dir("rc-toggle-helper");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);

        let on = agent.remote_toggle_lines("/rc", "");
        assert!(agent.remote_is_on());
        assert!(
            on.iter()
                .any(|l| l.contains("http://127.0.0.1:") && l.contains("/?t=")),
            "prints the tokenized link: {on:?}"
        );
        assert!(
            on.iter().any(|l| l.contains("ssh -L")),
            "prints the tunnel hint: {on:?}"
        );

        // `on` again is idempotent and re-prints the same link.
        let again = agent.remote_toggle_lines("/rc", "on");
        assert!(agent.remote_is_on());
        assert_eq!(
            again.iter().find(|l| l.contains("/?t=")),
            on.iter().find(|l| l.contains("/?t=")),
            "the same link comes back"
        );

        // Bare toggle while ON turns it off — the command's headline behaviour.
        let toggled_off = agent.remote_toggle_lines("/rc", "");
        assert!(!agent.remote_is_on(), "a bare /rc turns a live bridge off");
        assert!(
            toggled_off.iter().any(|l| l.contains("off")),
            "{toggled_off:?}"
        );

        // Re-establish an ON bridge for the explicit-"off" transition below.
        agent.remote_toggle_lines("/rc", "on");
        assert!(agent.remote_is_on());

        let off = agent.remote_toggle_lines("/rc", "off");
        assert!(!agent.remote_is_on());
        assert!(off.iter().any(|l| l.contains("off")), "{off:?}");

        // `off` when already off says so rather than erroring.
        let noop = agent.remote_toggle_lines("/rc", "off");
        assert!(!agent.remote_is_on());
        assert!(!noop.is_empty());

        // "ON" (uppercase) works the same as "on" — case-insensitive argument.
        let upper = agent.remote_toggle_lines("/rc", "ON");
        assert!(
            agent.remote_is_on(),
            "ON should turn the bridge on: {upper:?}"
        );

        // Back to off so the final bare-toggle check below observes off->on.
        agent.remote_toggle_lines("/rc", "off");
        assert!(!agent.remote_is_on());

        // A bare toggle from off turns it back on with a *new* token. Compare
        // the tokens themselves, not the whole line: the line also carries the
        // ephemeral port, which differs on every activation, so a line-level
        // `assert_ne!` would pass even if the token were reused.
        let token_of = |lines: &[String]| {
            lines
                .iter()
                .find_map(|l| l.split_once("/?t=").map(|(_, t)| t.to_owned()))
                .expect("the on-line carries a token")
        };
        let back = agent.remote_toggle_lines("/rc", "");
        assert!(agent.remote_is_on());
        assert_ne!(
            token_of(&back),
            token_of(&on),
            "a fresh activation mints a new token"
        );
        agent.remote_off();
    }

    /// The plain-REPL `/rc` arm must refuse rather than start a bridge nothing
    /// can drive: a piped session has no `tui_turn`/`tui_btw` to read `self.remote`
    /// or the bus, so a server started there would sit unattended forever.
    #[test]
    fn plain_repl_rc_refuses_to_start_a_server() {
        let dir = scratch_dir("rc-plain-repl-refuses");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);

        let keep_running = agent.slash("/rc").expect("slash handles /rc");
        assert!(keep_running, "/rc must not end the REPL session");
        assert!(
            !agent.remote_is_on(),
            "the plain REPL must not start a remote-control server"
        );
    }

    /// The `/remote-control` path end to end: a client that authenticates and
    /// requests control may submit a prompt even though a local front-end holds
    /// the slot, and the turn's output reaches the bus.
    #[test]
    fn tokenized_attach_takes_control_and_drives_a_turn() {
        use crate::remote::control::{ClientFrame, ClientMsg};
        use tungstenite::Message;

        let dir = scratch_dir("rc-e2e");
        let engine = ScriptedEngine {
            replies: vec!["hello from echo\n".to_string()],
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        let (addr, token) = agent
            .remote_on("127.0.0.1:0", Some("tok".to_owned()), true)
            .expect("binds");
        assert_eq!(token, "tok");

        let state = agent.remote.clone().expect("bridge installed");
        let sub = state.bus.subscribe();

        let stream = std::net::TcpStream::connect(addr).unwrap();
        let (mut ws, _) = tungstenite::client(
            format!("ws://{addr}/")
                .parse::<tungstenite::http::Uri>()
                .unwrap(),
            stream,
        )
        .expect("ws handshake");
        for m in [
            ClientMsg::Auth {
                token: "tok".into(),
                resume_from: None,
            },
            ClientMsg::RequestControl,
            ClientMsg::Prompt { text: "hi".into() },
        ] {
            ws.send(Message::Text(ClientFrame::new(m).to_json().unwrap()))
                .unwrap();
        }

        // The prompt was accepted (not denied), so it lands in TurnShared for
        // the turn loop to pick up.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut queued = Vec::new();
        while queued.is_empty() && std::time::Instant::now() < deadline {
            queued = state.shared.take_queued();
        }
        assert_eq!(queued, vec!["hi".to_string()], "the prompt was not denied");

        // Nothing on the wire said denied.
        drop(sub);
        agent.remote_off();
    }

    #[test]
    fn session_to_messages_threads_tool_ids_across_turns() {
        use crate::engine::ChatRole;
        use crate::session::Message;
        let mut session = Session::new();
        session.push(Message::user("read a.rs and b.rs"));
        // An assistant turn that issued two tool calls via a DSML stanza.
        session.push(Message::assistant(concat!(
            "Sure.\n",
            "<｜DSML｜tool_calls>\n",
            "<｜DSML｜invoke name=\"read\">\n",
            "<｜DSML｜parameter name=\"path\" string=\"true\">a.rs</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "<｜DSML｜invoke name=\"read\">\n",
            "<｜DSML｜parameter name=\"path\" string=\"true\">b.rs</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜tool_calls>\n",
        )));
        // The combined tool_result dispatch_all produces for that batch.
        session.push(Message::user(concat!(
            "<tool_result>",
            "Tool result 1 (read):\nAAA\n",
            "Tool result 2 (read):\nBBB\n",
            "</tool_result>",
        )));

        let msgs = session_to_messages(&session);
        // user, assistant(2 tool_calls), tool(id0), tool(id1).
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[1].role, ChatRole::Assistant);
        assert_eq!(msgs[1].tool_calls.len(), 2);
        let (id0, id1) = (
            msgs[1].tool_calls[0].id.clone(),
            msgs[1].tool_calls[1].id.clone(),
        );
        assert_ne!(id0, id1);
        // Each tool result pairs to its assistant tool-call id, in order.
        assert_eq!(msgs[2].role, ChatRole::Tool);
        assert_eq!(msgs[2].tool_call_id.as_deref(), Some(id0.as_str()));
        assert!(msgs[2].content.contains("AAA"));
        assert_eq!(msgs[3].tool_call_id.as_deref(), Some(id1.as_str()));
        assert!(msgs[3].content.contains("BBB"));
        // Arguments round-trip as JSON the provider can parse.
        assert_eq!(msgs[1].tool_calls[0].arguments, r#"{"path":"a.rs"}"#);
    }

    #[test]
    fn btw_question_parses_with_boundaries() {
        assert_eq!(btw_question("/btw what is x?"), Some("what is x?"));
        assert_eq!(btw_question("/btw  why?"), Some("why?"));
        assert_eq!(btw_question("/btw: colon form"), Some("colon form"));
        assert_eq!(btw_question("/btwfoo nope"), None);
        assert_eq!(btw_question("/side why?"), None);
        assert_eq!(btw_question("/btw"), None);
        assert_eq!(btw_question("/btw   "), None);
        assert_eq!(btw_question("plain text"), None);
    }

    #[test]
    fn btw_drain_leaves_transcript_untouched() {
        let dir = scratch_dir("btw-clean");
        let prompts: std::sync::Arc<std::sync::Mutex<Vec<String>>> = std::sync::Arc::default();
        let engine = ScriptedEngine {
            replies: vec!["It is 42.\n".to_string()],
            prompts: prompts.clone(),
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("main question"));
        agent.session.push(Message::assistant("main answer"));
        agent.last_ctx_used = 1234;
        let before = agent.session.transcript.clone();

        let shared = TurnShared::default();
        shared.push_btw("what was the answer?".to_owned());
        let (tx, rx) = std::sync::mpsc::channel();
        agent.drain_btw(&tx, &shared);
        drop(tx);

        assert_eq!(agent.session.transcript.len(), before.len());
        assert_eq!(agent.last_ctx_used, 1234);
        let recorded = prompts.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        assert!(recorded[0].contains("side question"), "framing missing");
        assert!(recorded[0].contains("what was the answer?"));
        let events: Vec<UiEvent> = rx.try_iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, UiEvent::UserEcho(t) if t == "/btw what was the answer?"))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, UiEvent::Dim(t) if t == "[btw]"))
        );
        assert!(
            events.iter().any(
                |e| matches!(e, UiEvent::Dim(t) if t.contains("not part of the conversation"))
            )
        );
        // The panel is bracketed exactly once, BtwBegin before the echo and
        // BtwEnd after the trailer, so the UI splits and tears down cleanly.
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, UiEvent::BtwBegin))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, UiEvent::BtwEnd))
                .count(),
            1
        );
        let begin = events
            .iter()
            .position(|e| matches!(e, UiEvent::BtwBegin))
            .unwrap();
        let end = events
            .iter()
            .position(|e| matches!(e, UiEvent::BtwEnd))
            .unwrap();
        let echo = events
            .iter()
            .position(|e| matches!(e, UiEvent::UserEcho(_)))
            .unwrap();
        assert!(begin < echo && echo < end, "panel must bracket the answer");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn btw_denies_tools() {
        let dir = scratch_dir("btw-tools");
        let stanza = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\" string=\"true\">echo nope</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        let engine = ScriptedEngine {
            replies: vec![stanza.to_string()],
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("main"));
        let before = agent.session.transcript.len();

        let shared = TurnShared::default();
        shared.push_btw("run something".to_owned());
        let (tx, rx) = std::sync::mpsc::channel();
        agent.drain_btw(&tx, &shared);
        drop(tx);

        // No dispatch and no tool result: transcript untouched.
        assert_eq!(agent.session.transcript.len(), before);
        let events: Vec<UiEvent> = rx.try_iter().collect();
        assert!(
            events.iter().any(
                |e| matches!(e, UiEvent::Dim(t) if t.contains("tools are disabled during /btw"))
            )
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn btw_answers_at_mid_turn_boundary_and_stays_out_of_main_prompts() {
        let dir = scratch_dir("btw-boundary");
        let stanza = concat!(
            "Working.\n",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\" string=\"true\">echo hi</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        let prompts: std::sync::Arc<std::sync::Mutex<Vec<String>>> = std::sync::Arc::default();
        let engine = ScriptedEngine {
            replies: vec![
                stanza.to_string(),
                "The answer is 7.\n".to_string(), // side answer at the boundary
                "Done.\n".to_string(),            // main continuation
            ],
            prompts: prompts.clone(),
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("do the task"));

        let shared = TurnShared::default();
        shared.push_btw("what is 3+4?".to_owned());
        let (tx, _rx) = std::sync::mpsc::channel();
        agent.worker_turn(&tx, &shared).unwrap();
        drop(tx);

        let recorded = prompts.lock().unwrap();
        assert_eq!(recorded.len(), 3, "main, side, main continuation");
        // The side prompt sees the completed pass (stanza already in the
        // transcript) but runs before the tool dispatch.
        assert!(recorded[1].contains("what is 3+4?"));
        assert!(recorded[1].contains("Working."));
        assert!(!recorded[1].contains("<tool_result>"));
        // The main continuation never sees the side exchange.
        assert!(recorded[2].contains("<tool_result>"));
        assert!(!recorded[2].contains("what is 3+4?"));
        assert!(!recorded[2].contains("The answer is 7."));
        // Nothing side-channel entered the session.
        let flat: String = agent
            .session
            .transcript
            .iter()
            .map(|m| m.text.as_str())
            .collect();
        assert!(!flat.contains("what is 3+4?"));
        assert!(!flat.contains("The answer is 7."));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn preempting_btw_rolls_back_the_pass_and_reruns_after_answering() {
        let dir = scratch_dir("btw-preempt");
        let prompts: std::sync::Arc<std::sync::Mutex<Vec<String>>> = std::sync::Arc::default();
        let engine = ScriptedEngine {
            replies: vec![
                "PARTIAL main output that gets discarded\n".to_string(), // preempted pass
                "The answer is Rust.\n".to_string(),                     // side answer
                "Final main answer.\n".to_string(),                      // re-run pass
            ],
            prompts: prompts.clone(),
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("do the task"));

        // A /btw queued with the preempt flag set (as the UI does mid-pass).
        let shared = TurnShared::default();
        shared.push_btw("what language?".to_owned());
        shared.preempt.store(true, Ordering::Relaxed);
        let (tx, rx) = std::sync::mpsc::channel();
        agent.worker_turn(&tx, &shared).unwrap();
        drop(tx);

        // Three passes ran: preempted main, side answer, re-run main.
        let recorded = prompts.lock().unwrap();
        assert_eq!(recorded.len(), 3);
        // Nothing was committed before the preempt, so the re-run's prompt is
        // byte-identical to the preempted pass's prompt.
        assert_eq!(recorded[0], recorded[2]);
        assert!(recorded[1].contains("side question"));
        assert!(recorded[1].contains("what language?"));

        // The discarded partial never reached the transcript; only the re-run
        // answer did, and the side exchange stayed out entirely.
        assert_eq!(agent.session.transcript.len(), 2);
        assert_eq!(
            agent.session.transcript[1].text.trim(),
            "Final main answer."
        );
        let flat: String = agent
            .session
            .transcript
            .iter()
            .map(|m| m.text.as_str())
            .collect();
        assert!(!flat.contains("PARTIAL"));
        assert!(!flat.contains("The answer is Rust."));
        assert!(!flat.contains("what language?"));

        // The UI was told to roll back the main log, and a panel bracketed
        // the priority answer. The preempt flag is consumed, so the re-run
        // (and future turns) are not stuck preempting.
        let events: Vec<UiEvent> = rx.try_iter().collect();
        assert!(events.iter().any(|e| matches!(e, UiEvent::MainRollback)));
        assert!(events.iter().any(|e| matches!(e, UiEvent::BtwBegin)));
        assert!(!shared.preempt.load(Ordering::Relaxed));
        std::fs::remove_dir_all(&dir).ok();
    }

    // BTW-SUSPEND-DESIGN §4.3: with `btw.suspend` on and an aside-capable
    // engine, an in-pass /btw freezes the main pass, answers the aside via
    // `generate_aside`, and resumes the *same* reply — the partial is kept on
    // screen and spliced back into the transcript, and the main log is never
    // rolled back (unlike the preempt fallback above).
    #[test]
    fn suspend_freezes_answers_aside_and_resumes_the_same_reply() {
        let dir = scratch_dir("btw-suspend");
        let prompts: std::sync::Arc<std::sync::Mutex<Vec<String>>> = std::sync::Arc::default();
        let engine = ScriptedEngine {
            replies: vec![
                "Partial reply so far".to_string(),          // frozen main pass
                "The answer is Rust.\n".to_string(),         // aside answer
                " and the rest of the reply.\n".to_string(), // resumed continuation
            ],
            prompts: prompts.clone(),
            aside_support: true,
            ..ScriptedEngine::default()
        };
        let mut cfg = test_cfg();
        cfg.btw.suspend = true;
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("do the task"));

        let shared = TurnShared::default();
        shared.push_btw("what language?".to_owned());
        shared.preempt.store(true, Ordering::Relaxed);
        let (tx, rx) = std::sync::mpsc::channel();
        agent.worker_turn(&tx, &shared).unwrap();
        drop(tx);

        // Three passes: frozen main, aside answer, resumed continuation.
        let recorded = prompts.lock().unwrap();
        assert_eq!(recorded.len(), 3);
        // The aside sees the framed question but not the partial (nothing is
        // committed to the transcript).
        assert!(recorded[1].contains("what language?"));
        // The resume re-opens the assistant turn with the partial so the
        // engine can splice its tokens and continue from the freeze point.
        assert!(recorded[2].contains("[assistant]\nPartial reply so far"));

        // The transcript holds the whole reply (partial + continuation) as one
        // assistant message; the aside stayed out entirely.
        assert_eq!(agent.session.transcript.len(), 2);
        assert_eq!(
            agent.session.transcript[1].text.trim(),
            "Partial reply so far and the rest of the reply."
        );
        let flat: String = agent
            .session
            .transcript
            .iter()
            .map(|m| m.text.as_str())
            .collect();
        assert!(!flat.contains("what language?"));
        assert!(!flat.contains("The answer is Rust."));

        // Suspend markers bracket the aside; the main log is NOT rolled back
        // (the partial stays on screen), and the preempt flag is consumed.
        let events: Vec<UiEvent> = rx.try_iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, UiEvent::Dim(t) if t == worker::BTW_SUSPEND_MARKER))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, UiEvent::Dim(t) if t == worker::BTW_RESUME_MARKER))
        );
        assert!(events.iter().any(|e| matches!(e, UiEvent::BtwBegin)));
        assert!(
            !events.iter().any(|e| matches!(e, UiEvent::MainRollback)),
            "suspend keeps the partial on screen; no rollback"
        );
        assert!(!shared.preempt.load(Ordering::Relaxed));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A multiplexed aside is one-at-a-time and unqueued: it takes the next
    /// question, drops any others with a notice, and refuses to start a second
    /// while one is still in flight.
    #[test]
    fn multiplexed_aside_is_one_at_a_time_and_unqueued() {
        let dir = scratch_dir("btw-mux-one");
        let engine = ScriptedEngine {
            replies: vec!["answered\n".to_string()],
            aside_support: true,
            fork_aside_support: true,
            multiplex_support: true,
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);

        let shared = TurnShared::default();
        shared.push_btw("first".to_owned());
        shared.push_btw("second".to_owned());
        shared.push_btw("third".to_owned());
        let (tx, rx) = std::sync::mpsc::channel();

        let framed = agent
            .multiplexable_aside(&tx, &shared, "partial reply")
            .expect("a fork-capable engine multiplexes");
        assert!(framed.contains("first"), "it takes the next question");
        assert!(
            !framed.contains("second") && !framed.contains("third"),
            "the others are not merged into one prompt"
        );
        assert!(
            framed.contains("partial reply"),
            "the frozen partial is spliced in so the prompt extends the live KV"
        );

        // No queue: the extras are gone, and the user is told.
        assert_eq!(shared.pop_btw(), None, "nothing is left queued");
        drop(tx);
        let notices: Vec<String> = rx
            .iter()
            .filter_map(|ev| match ev {
                UiEvent::Dim(t) => Some(t),
                _ => None,
            })
            .collect();
        assert!(
            notices.iter().any(|t| t.contains("dropped 2")),
            "the dropped questions are reported: {notices:?}"
        );

        // One at a time: with an aside already pending, a new question does not
        // start a second one — it falls back to the freeze path.
        agent.pending_aside = Some("in flight".to_owned());
        let (tx2, _rx2) = std::sync::mpsc::channel();
        shared.push_btw("fourth".to_owned());
        assert!(
            agent.multiplexable_aside(&tx2, &shared, "").is_none(),
            "a second concurrent aside is refused while one is in flight"
        );
    }

    /// The three aside tiers degrade cleanly into one another
    /// (`docs/SESSION-CLONE-DESIGN.md` §6.4). Runs without Metal, which is the
    /// point: it is the only tier check CI can execute.
    #[test]
    fn aside_tier_selection_degrades_cleanly() {
        // An engine that can fork does; one that cannot (EchoEngine, remote
        // engines) falls through to the destructive path rather than failing.
        let cases = [(true, "forked"), (false, "destructive")];
        for (engine_forks, expected) in cases {
            let dir = scratch_dir("btw-aside-tier");
            let tiers: std::sync::Arc<std::sync::Mutex<Vec<String>>> = std::sync::Arc::default();
            let engine = ScriptedEngine {
                replies: vec!["answered\n".to_string()],
                aside_support: true,
                fork_aside_support: engine_forks,
                aside_tiers: Some(tiers.clone()),
                ..ScriptedEngine::default()
            };
            let cfg = test_cfg();
            let mut agent = test_agent(&dir, engine, &cfg);

            let mut reply = String::new();
            agent
                .generate_aside_best(
                    "[user]\nwhat language?\n",
                    &cfg.generation.clone(),
                    &|| false,
                    &mut |ev| {
                        if let EngineEvent::Text(t) = ev {
                            reply.push_str(&t);
                        }
                    },
                )
                .unwrap();

            assert_eq!(
                tiers.lock().unwrap().as_slice(),
                [expected.to_string()],
                "engine_forks={engine_forks}"
            );
            assert!(
                reply.contains("answered"),
                "every tier still answers the question"
            );
        }
    }

    /// An engine with no aside support at all cannot serve either tier, so the
    /// caller is told to fall back to the boundary queue — the third tier.
    #[test]
    fn aside_without_engine_support_is_unsupported() {
        let dir = scratch_dir("btw-aside-none");
        let engine = ScriptedEngine {
            replies: vec!["unused\n".to_string()],
            aside_support: false,
            fork_aside_support: false,
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);

        let err = agent
            .generate_aside_best(
                "[user]\nwhat language?\n",
                &cfg.generation.clone(),
                &|| false,
                &mut |_| {},
            )
            .expect_err("no tier can serve this engine");
        assert!(
            err.is_unsupported(),
            "the caller must see a fall-through signal, not a hard failure: {err}"
        );
    }

    // BTW-SUSPEND-DESIGN §6 `aside_fifo_cap`: more than the cap of in-pass /btw
    // questions drop the oldest with a notice, and the suspend drain answers
    // the survivors FIFO via `generate_aside`.
    #[test]
    fn aside_fifo_cap() {
        // The queue caps at BTW_QUEUE_CAP, dropping the oldest beyond it and
        // returning it so the caller can surface a visible drop notice.
        let shared = TurnShared::default();
        let mut dropped = Vec::new();
        for i in 0..(crate::worker::BTW_QUEUE_CAP + 2) {
            if let Some(old) = shared.push_btw(format!("q{i}")) {
                dropped.push(old);
            }
        }
        // The two oldest (q0, q1) were dropped; q2..=q21 survive.
        assert_eq!(dropped, vec!["q0".to_string(), "q1".to_string()]);

        // The suspend drain answers every survivor FIFO through generate_aside.
        let dir = scratch_dir("btw-aside-cap");
        let prompts: std::sync::Arc<std::sync::Mutex<Vec<String>>> = std::sync::Arc::default();
        let engine = ScriptedEngine {
            replies: vec!["ok\n".to_string(); crate::worker::BTW_QUEUE_CAP],
            prompts: prompts.clone(),
            aside_support: true,
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("main"));
        let (tx, _rx) = std::sync::mpsc::channel();
        agent.drain_aside(&tx, &shared, "");
        drop(tx);

        let recorded = prompts.lock().unwrap();
        assert_eq!(recorded.len(), crate::worker::BTW_QUEUE_CAP);
        // FIFO: the first answered aside is q2 (oldest survivor), the last q21.
        assert!(recorded[0].contains("q2"));
        assert!(recorded[crate::worker::BTW_QUEUE_CAP - 1].contains("q21"));
        assert!(shared.pop_btw().is_none(), "queue fully drained");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn drain_btw_returns_promptly_so_the_main_task_resumes() {
        let dir = scratch_dir("btw-nonblock");
        let engine = ScriptedEngine {
            replies: vec!["It is Rust.\n".to_string()],
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("main"));

        let shared = TurnShared::default();
        shared.push_btw("what language?".to_owned());
        let (tx, rx) = std::sync::mpsc::channel();
        // No external signal is provided: if the drain parked waiting for the
        // panel to be dismissed, this would hang. It must return on its own.
        agent.drain_btw(&tx, &shared);
        drop(tx);

        // BtwEnd only ends the active answer (the UI keeps the panel visible);
        // the drain still returns, letting the main task resume.
        let events: Vec<UiEvent> = rx.try_iter().collect();
        assert!(events.iter().any(|e| matches!(e, UiEvent::BtwBegin)));
        assert!(events.iter().any(|e| matches!(e, UiEvent::BtwEnd)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resume_from_cli_loads_a_saved_session_by_prefix_and_most_recent() {
        let dir = scratch_dir("resume-cli");
        let cfg = test_cfg();

        // Save a session, capture its id.
        let mut a = test_agent(&dir, ScriptedEngine::default(), &cfg);
        a.session.push(Message::user("remember the alamo"));
        let id = a.store.save(&mut a.session).unwrap();

        // A fresh agent resumes it by sha prefix.
        let mut b = test_agent(&dir, ScriptedEngine::default(), &cfg);
        assert!(b.resumed_history().is_none(), "fresh session: no history");
        b.resume_from_cli(&id[..8]).unwrap();
        assert_eq!(b.session.id, id);
        assert!(
            b.session
                .transcript
                .iter()
                .any(|m| m.text == "remember the alamo")
        );
        let history = b.resumed_history().expect("resumed session shows history");
        assert!(history.contains("resumed session"));

        // A fresh agent with an empty arg resumes the most recent session.
        let mut c = test_agent(&dir, ScriptedEngine::default(), &cfg);
        c.resume_from_cli("").unwrap();
        assert_eq!(c.session.id, id);

        // An unknown prefix is a clean error, not a panic.
        let mut d = test_agent(&dir, ScriptedEngine::default(), &cfg);
        assert!(d.resume_from_cli("nonexistent0").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    // `/think` with no argument reports rather than changes, and names the
    // levels so the user learns the vocabulary from the answer.
    #[test]
    fn think_command_reports_the_current_level() {
        let dir = scratch_dir("think-report");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        let out = agent.think_command("");
        assert!(out.contains("off"), "got: {out}");
        assert!(out.contains("medium") && out.contains("max"), "got: {out}");
        assert_eq!(agent.think, ThinkMode::Off);
        std::fs::remove_dir_all(&dir).ok();
    }

    // Setting a level changes what the next turn generates with *and* tells the
    // engine, which is where the cached prefix decision is made.
    #[test]
    fn think_command_sets_the_level_and_tells_the_engine() {
        let dir = scratch_dir("think-set");
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let engine = ScriptedEngine {
            think_modes: Some(std::sync::Arc::clone(&seen)),
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);

        let out = agent.think_command("medium");
        assert!(out.contains("medium"), "got: {out}");
        assert_eq!(agent.think, ThinkMode::Medium);
        assert_eq!(*seen.lock().unwrap(), vec![ThinkMode::Medium]);

        // A no-op change says so and does not disturb the engine.
        let out = agent.think_command("medium");
        assert!(out.contains("already"), "got: {out}");
        assert_eq!(seen.lock().unwrap().len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The alt local engine is configured exactly like the main one at startup.
    /// Without this its `warm_reset` tokenizes the same system text differently
    /// from the engine that wrote `sysprompt.kv` — trusted-prefix length and
    /// reasoning level are both inputs to `build_system_tokens` — so its Tier 1
    /// checkpoint would restore into a session whose token buffer does not
    /// describe it and the first sync would prefill anyway.
    #[test]
    fn the_alt_local_engine_is_configured_like_the_main_one() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let trusted = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.think_mode = ThinkMode::Medium;
        let local = ScriptedEngine {
            think_modes: Some(std::sync::Arc::clone(&seen)),
            trusted_lens: Some(std::sync::Arc::clone(&trusted)),
            ..ScriptedEngine::default()
        };

        let agent = new_agent(
            Box::new(ScriptedEngine::default()),
            &cfg,
            false,
            Some(Box::new(local)),
        )
        .expect("an agent");

        assert_eq!(*seen.lock().unwrap(), vec![ThinkMode::Medium]);
        assert_eq!(
            *trusted.lock().unwrap(),
            vec![agent.trusted_system_len],
            "the same boundary the main engine was given"
        );
    }

    /// The level keys an alt engine's Tier 1 checkpoint and frames its
    /// sidechains, so a cached engine left at the old level would build its
    /// tokens at one level while being keyed at another — the disagreement a
    /// fingerprint cannot catch. It also re-arms the alt local warm, since the
    /// checkpoint to restore is now a different one.
    #[test]
    fn think_command_reaches_cached_alt_engines() {
        let dir = scratch_dir("think-alt");
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let cfg = test_cfg();
        // `max` is the level that moves the effort preamble, so it is the one
        // that both re-keys the checkpoint and invalidates the engine's tokens.
        // It needs the context to match, or the command refuses before it acts.
        let mut agent = test_agent(
            &dir,
            ScriptedEngine {
                ctx_override: Some(crate::engine::THINK_MAX_MIN_CONTEXT),
                ..ScriptedEngine::default()
            },
            &cfg,
        );
        agent.alt_engines.insert(
            EngineKey::Local,
            Box::new(ScriptedEngine {
                think_modes: Some(std::sync::Arc::clone(&seen)),
                ..ScriptedEngine::default()
            }),
        );
        agent.local_alt_warmed = true;

        let out = agent.think_command("max");
        assert!(out.contains("max"), "got: {out}");

        assert_eq!(*seen.lock().unwrap(), vec![ThinkMode::Max]);
        assert!(
            !agent.local_alt_warmed,
            "a new fingerprint means a new checkpoint to restore"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    // `low` is selectable at any context — unlike `max` it has no floor — and
    // reaches the engine like any other level.
    #[test]
    fn think_command_selects_low_at_any_context() {
        let dir = scratch_dir("think-low");
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let engine = ScriptedEngine {
            think_modes: Some(std::sync::Arc::clone(&seen)),
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);

        let out = agent.think_command("low");
        assert!(out.contains("low"), "got: {out}");
        assert_eq!(agent.think, ThinkMode::Low);
        assert_eq!(*seen.lock().unwrap(), vec![ThinkMode::Low]);

        // And the level listing offers it, so it is discoverable.
        let out = agent.think_command("");
        assert!(out.contains("low"), "got: {out}");
        std::fs::remove_dir_all(&dir).ok();
    }

    // The context guard: `max` is refused below the minimum rather than
    // silently downgraded, and the refusal leaves the level alone.
    #[test]
    fn think_max_is_refused_on_a_small_context() {
        let dir = scratch_dir("think-max-small");
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let engine = ScriptedEngine {
            think_modes: Some(std::sync::Arc::clone(&seen)),
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);

        let out = agent.think_command("max");
        assert!(
            out.contains(&crate::engine::THINK_MAX_MIN_CONTEXT.to_string()),
            "the refusal must name the context it needs; got: {out}"
        );
        assert_eq!(agent.think, ThinkMode::Off, "level unchanged");
        assert!(seen.lock().unwrap().is_empty(), "engine untouched");
        std::fs::remove_dir_all(&dir).ok();
    }

    // With enough context it goes through.
    #[test]
    fn think_max_is_accepted_on_a_large_context() {
        let dir = scratch_dir("think-max-large");
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let engine = ScriptedEngine {
            ctx_override: Some(crate::engine::THINK_MAX_MIN_CONTEXT),
            think_modes: Some(std::sync::Arc::clone(&seen)),
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);

        let out = agent.think_command("max");
        assert!(out.contains("max"), "got: {out}");
        assert_eq!(agent.think, ThinkMode::Max);
        assert_eq!(*seen.lock().unwrap(), vec![ThinkMode::Max]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn think_command_rejects_an_unknown_level() {
        let dir = scratch_dir("think-bad");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        let out = agent.think_command("turbo");
        assert!(out.contains("turbo"), "got: {out}");
        assert_eq!(agent.think, ThinkMode::Off, "level unchanged");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn notify_command_toggles_and_reports() {
        let _g = crate::notify::TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::notify::set_enabled(true);
        let off = Agent::notify_command("off");
        assert!(off.to_lowercase().contains("off"));
        assert!(!crate::notify::enabled());
        let on = Agent::notify_command("on");
        assert!(on.to_lowercase().contains("on"));
        assert!(crate::notify::enabled());
        // bare toggle flips
        Agent::notify_command("");
        assert!(!crate::notify::enabled());
        // unknown arg reports an error and doesn't change state
        let err = Agent::notify_command("bogus");
        assert!(err.contains("bogus"));
        assert!(!crate::notify::enabled());
    }

    #[test]
    fn slash_message_prefers_skills_then_templates_and_never_shadows_builtins() {
        let dir = scratch_dir("slash-templates");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent.skills.push(crate::skills::Skill {
            name: "dual".into(),
            description: String::new(),
            argument_hint: String::new(),
            body: "skill body $ARGUMENTS".into(),
            dir: std::path::PathBuf::new(),
        });
        let template = |name: &str, body: &str| crate::templates::Template {
            name: name.to_string(),
            description: String::new(),
            argument_hint: String::new(),
            body: body.to_string(),
            path: std::path::PathBuf::new(),
        };
        agent.templates.push(template("dual", "template body"));
        agent.templates.push(template("review", "Review {{path}}."));
        agent.templates.push(template("help", "shadow attempt"));

        // A skill of the same name wins over a template.
        assert_eq!(
            agent.slash_message("/dual", "x").unwrap().unwrap(),
            "skill body x"
        );
        assert_eq!(
            agent
                .slash_message("/review", "src/ui.rs")
                .unwrap()
                .unwrap(),
            "Review src/ui.rs."
        );
        // Missing variables surface as an error, not an empty substitution.
        let err = agent.slash_message("/review", "").unwrap().unwrap_err();
        assert!(err.contains("missing value for path"), "{err}");
        // Built-ins win: /help never reaches the template.
        assert!(agent.slash_message("/help", "").is_none());
        assert!(agent.slash_message("/nope", "").is_none());
    }

    #[test]
    fn live_commands_allow_read_only_reports_and_reject_the_rest() {
        let live = LiveCommands {
            context: "CTX".to_owned(),
            usage: "USE".to_owned(),
            mcp: "MCP".to_owned(),
        };
        assert_eq!(live.output("/context").as_deref(), Some("CTX"));
        assert_eq!(live.output("/usage").as_deref(), Some("USE"));
        assert_eq!(live.output("/mcp").as_deref(), Some("MCP"));
        // /help is static, rendered on demand — just present.
        assert!(live.output("/help").is_some());
        // Mutating / stateful commands must not run mid-turn.
        assert!(live.output("/compact").is_none());
        assert!(live.output("/save").is_none());
        assert!(live.output("/resume").is_none());
        assert!(live.output("/context-ish").is_none());
    }

    #[test]
    fn replay_history_renders_markdown_and_thinking_not_plain() {
        let dir = scratch_dir("resume-replay");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent.session.id = "deadbeef".repeat(5);
        agent.session.push(Message::user("hi"));
        agent.session.push(Message::assistant(
            "<think>pondering</think>Here is **bold** text.\n",
        ));

        let mut log = OutputLog::new();
        agent.replay_history_into_log(&mut log);
        let text = log.to_text();
        // Concatenated text per line, for whole-word assertions (think text is
        // emitted one char per span).
        let line_text = |l: &ratatui::text::Line| -> String {
            l.spans.iter().map(|s| s.content.as_ref()).collect()
        };

        // The thinking text renders in the dim gray, not the default style: the
        // line containing "pondering" is entirely dim.
        let dim = ratatui::style::Color::Indexed(238);
        let think_line = text
            .lines
            .iter()
            .find(|l| line_text(l).contains("pondering"))
            .expect("thinking text present");
        assert!(
            think_line
                .spans
                .iter()
                .all(|s| s.content.trim().is_empty() || s.style.fg == Some(dim)),
            "thinking text should be dimmed"
        );

        // The `<think>` tags themselves are consumed, never shown literally.
        assert!(
            !text.lines.iter().any(|l| line_text(l).contains("<think>")),
            "think tags must not appear literally"
        );

        // The visible markdown is styled (bold), i.e. it went through the
        // markdown renderer rather than being pushed as plain text.
        let has_bold = text.lines.iter().any(|l| {
            l.spans.iter().any(|s| {
                s.content.contains("bold")
                    && s.style
                        .add_modifier
                        .contains(ratatui::style::Modifier::BOLD)
            })
        });
        assert!(has_bold, "visible markdown should be rendered (bold)");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Regression: `/resume` replays a stored assistant message whose tool
    /// call was dispatched from inside `<think>`. The replay renderer must
    /// pick up `thinking_tool_calls` from settings the same way the live
    /// turn paths do, or the stanza is flagged with an "ignored" notice
    /// immediately followed by that call's own stored tool result. Under
    /// default settings `show_tool_calls` is off, so this asserts what is
    /// actually observable rather than the banner text (see
    /// `bash_stanza_hides_dsml_and_shows_banner` in `viz.rs` for that case).
    #[test]
    fn replay_history_consumes_in_think_tool_call_instead_of_ignoring_it() {
        // Opt this thread into in-think dispatch; the shipped default is off.
        let mut settings = crate::settings::Settings::default();
        settings.engine.thinking_tool_calls = true;
        crate::settings::install_for_test(settings);
        let dir = scratch_dir("resume-replay-in-think");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent.session.id = "deadbeef".repeat(5);
        agent.session.push(Message::user("hi"));
        agent.session.push(Message::assistant(concat!(
            "<think>I should list the directory",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\">echo hello</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
            "</think>",
        )));

        let mut log = OutputLog::new();
        agent.replay_history_into_log(&mut log);
        let text = log.to_text();
        let line_text = |l: &ratatui::text::Line| -> String {
            l.spans.iter().map(|s| s.content.as_ref()).collect()
        };
        let rendered: Vec<String> = text.lines.iter().map(line_text).collect();
        let joined = rendered.join("\n");

        // Under default settings `ui.show_tool_calls` is false, so the banner
        // itself legitimately does not render here — this pins down what
        // *is* observable: the stanza was parsed and consumed as a tool call
        // (thinking text survives, no raw DSML markup leaks), not printed
        // verbatim or flagged as ignored.
        assert!(
            joined.contains("I should list the directory"),
            "the thinking text preceding the call should still render: {joined:?}"
        );
        assert!(
            !joined.contains("DSML"),
            "the DSML stanza should be parsed away, not leaked verbatim: {joined:?}"
        );
        assert!(
            !joined.contains("tool call ignored"),
            "the replay must not report the in-think call as ignored: {joined:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_for_exit_persists_a_used_session_and_skips_an_empty_one() {
        let dir = scratch_dir("exit-save");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);

        // An empty session (no user turn) has nothing worth saving.
        assert!(agent.save_for_exit().is_none());

        // After a real user turn it saves, returns the id + existing path, and
        // stamps the session id so a resume can find it.
        agent.session.push(Message::user("hello there"));
        let (id, path) = agent.save_for_exit().expect("used session should save");
        assert!(!id.is_empty());
        assert!(path.exists(), "session file written: {}", path.display());
        assert_eq!(agent.session.id, id);
        // The id resolves through the store, which is what `/resume <id>` uses.
        assert!(agent.store.find(&id[..8]).is_ok());

        // Re-exiting with no new activity does not re-save: `save` cleared
        // `dirty`, so there is nothing to persist.
        assert!(
            agent.save_for_exit().is_none(),
            "unchanged session re-saves"
        );
        // A new turn makes it dirty again, and it saves.
        agent.session.push(Message::user("another"));
        assert!(agent.save_for_exit().is_some());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The Ctrl+D → `plank /resume` round trip must not re-prefill.
    ///
    /// Both halves were missing: `save_for_exit` wrote the transcript only, and
    /// `resume_from_cli` loaded the transcript only — so the exit path, which is
    /// how sessions actually get saved, produced a session whose whole context
    /// had to be prefilled again on resume. Minutes of it, at local speeds.
    #[test]
    fn exit_save_and_cli_resume_carry_the_kv_payload() {
        let dir = scratch_dir("exit-kv-roundtrip");
        let cfg = test_cfg();

        // Exit with Ctrl+D: transcript *and* KV payload land on disk.
        let saved_id = {
            let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let engine = ScriptedEngine {
                kv_events: Some(std::sync::Arc::clone(&events)),
                ..ScriptedEngine::default()
            };
            let mut agent = test_agent(&dir, engine, &cfg);
            agent.session.push(Message::user("hello there"));
            agent.session.push(Message::assistant("hi"));

            let (id, _path) = agent.save_for_exit().expect("used session saves");
            assert!(
                events.lock().unwrap().iter().any(|e| e == "capture"),
                "exit must snapshot the KV: {:?}",
                events.lock().unwrap()
            );
            assert!(
                agent.store.payload_bytes(&id) > 0,
                "payload sidecar written"
            );
            id
        };

        // Next launch: `plank /resume <prefix>` restores that payload, so the
        // first turn extends the cached KV instead of rebuilding it.
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let engine = ScriptedEngine {
            kv_events: Some(std::sync::Arc::clone(&events)),
            ..ScriptedEngine::default()
        };
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.resume_from_cli(&saved_id[..8]).expect("resume works");

        assert_eq!(agent.session.id, saved_id);
        let log = events.lock().unwrap().clone();
        assert!(
            log.iter().any(|e| e.starts_with("restore:")),
            "CLI resume must restore the KV payload: {log:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The startup warm walk must not run after a session payload is restored.
    ///
    /// Measured on a real model before this: restoring the payload and then
    /// warming rewound the KV from 13845 tokens to 13696 — the session-context
    /// boundary — and cleared the token transcript (`0 spans held`), so 165
    /// conversation tokens re-prefilled. Warming a restored session can only
    /// destroy cache, because the payload already covers every tier.
    #[test]
    fn a_restored_payload_suppresses_the_startup_warm() {
        let dir = scratch_dir("warm-after-restore");
        let cfg = test_cfg();
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let engine = ScriptedEngine {
            kv_events: Some(std::sync::Arc::clone(&events)),
            ..ScriptedEngine::default()
        };
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("hello"));
        let (id, _) = agent.save_for_exit().expect("saves");

        // Nothing restored yet: the warm walk is expected to run.
        assert!(!agent.skip_warm_after_restore());

        let loaded = agent.store.load(&id[..8]).unwrap();
        assert_eq!(
            agent.load_session_payload(&loaded).as_deref(),
            Some("restored KV payload; resume skips re-prefill")
        );
        assert!(
            agent.skip_warm_after_restore(),
            "a restored payload must suppress the warm walk"
        );

        // And the walk really is skipped: it would `set_kv` each cacheable tier,
        // logging another restore over the one we just made.
        let before = events.lock().unwrap().len();
        agent.warm_plain().expect("warm is a no-op here");
        assert_eq!(
            events.lock().unwrap().len(),
            before,
            "warm touched the engine after a restore: {:?}",
            events.lock().unwrap()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A payload is only reusable for the token stream it was captured from, and
    /// the reasoning level changes that stream without changing a single byte of
    /// the transcript or system prompt. Resuming under a different level must
    /// fall back to prefill rather than restore a KV that does not match.
    #[test]
    fn a_payload_from_another_think_level_is_stale() {
        let dir = scratch_dir("exit-kv-think");
        let cfg = test_cfg();
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let engine = ScriptedEngine {
            kv_events: Some(std::sync::Arc::clone(&events)),
            ..ScriptedEngine::default()
        };
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("hello"));
        let (id, _) = agent.save_for_exit().expect("saves");
        let loaded = agent.store.load(&id[..8]).unwrap();

        // Same level: restored.
        assert_eq!(
            agent.load_session_payload(&loaded).as_deref(),
            Some("restored KV payload; resume skips re-prefill")
        );

        // Level moved since the capture: the payload is not for these tokens.
        agent.think = crate::engine::ThinkMode::Max;
        assert_eq!(
            agent.load_session_payload(&loaded).as_deref(),
            Some("KV payload is stale; the transcript will be re-prefilled")
        );

        // Same for the tokenization split, which is likewise invisible in the
        // transcript text (an upgrade that changes it must not reuse a payload).
        agent.think = crate::engine::ThinkMode::Medium;
        agent.trusted_system_len = 4096;
        assert_eq!(
            agent.load_session_payload(&loaded).as_deref(),
            Some("KV payload is stale; the transcript will be re-prefilled")
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    fn branch_agent<'a>(dir: &std::path::Path, cfg: &'a crate::config::AgentConfig) -> Agent<'a> {
        let mut agent = test_agent(dir, ScriptedEngine::default(), cfg);
        agent.session.push(Message::user("first"));
        agent.session.push(Message::assistant("a1"));
        agent.session.push(Message::user("second"));
        agent.session.push(Message::assistant("a2"));
        agent
    }

    #[test]
    fn fork_rewinds_to_a_user_message_and_keeps_the_old_branch() {
        let dir = scratch_dir("fork-branch");
        let cfg = test_cfg();
        let mut agent = branch_agent(&dir, &cfg);

        // Fork point 2 is "second"; forking there drops it and everything
        // after from the live transcript.
        let msg = agent.fork_branch("2", false).unwrap();
        assert!(msg.contains("2 of 4 messages kept"), "{msg}");
        assert_eq!(
            agent
                .session
                .transcript
                .iter()
                .map(|m| m.text.as_str())
                .collect::<Vec<_>>(),
            ["first", "a1"]
        );
        // The forked-away turns survive as a branch, and the tree shows both.
        assert_eq!(agent.session.branches.len(), 2);
        let tree = agent.session.tree();
        assert_eq!(tree.branch_count(), 2);
        assert_eq!(tree.len(), 4);
        let view = agent.tree_view(false);
        assert!(view.contains("2 branches"), "{view}");

        // Continuing writes into the new branch only.
        agent.session.push(Message::user("second-prime"));
        assert_eq!(agent.session.tree().branch_count(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn fork_rejects_out_of_range_and_missing_points() {
        let dir = scratch_dir("fork-range");
        let cfg = test_cfg();
        let mut agent = branch_agent(&dir, &cfg);
        assert!(agent.fork_branch("0", false).is_err());
        assert!(agent.fork_branch("3", false).is_err());
        assert!(agent.fork_branch("x", false).is_err());
        // Nothing changed after a rejected fork.
        assert_eq!(agent.session.transcript.len(), 4);
        // No argument shows the tree instead of forking.
        let shown = agent.fork_branch("", false).unwrap();
        assert!(shown.contains("fork points"), "{shown}");
        assert_eq!(agent.session.transcript.len(), 4);

        let mut empty = test_agent(&dir, ScriptedEngine::default(), &cfg);
        assert!(empty.fork_branch("1", false).is_err());
        assert!(empty.clone_branch().is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn clone_duplicates_the_branch_and_leaves_the_transcript_identical() {
        let dir = scratch_dir("clone-branch");
        let cfg = test_cfg();
        let mut agent = branch_agent(&dir, &cfg);
        let before = agent.session.transcript.clone();

        let msg = agent.clone_branch().unwrap();
        assert!(msg.contains("4 messages"), "{msg}");
        // Identical transcript: the engine's cached prefix stays valid in full.
        assert_eq!(agent.session.transcript, before);
        assert_eq!(agent.session.branches.len(), 4);
        assert_eq!(agent.session.tree().branch_count(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn branch_commands_round_trip_through_save_and_load() {
        let dir = scratch_dir("branch-persist");
        let cfg = test_cfg();
        let mut agent = branch_agent(&dir, &cfg);
        agent.fork_branch("2", false).unwrap();
        agent.session.push(Message::user("second-prime"));
        let id = agent.store.save(&mut agent.session).unwrap();

        let loaded = agent.store.load(&id).unwrap();
        assert_eq!(loaded.transcript, agent.session.transcript);
        assert_eq!(loaded.branches, agent.session.branches);
        assert_eq!(loaded.tree().branch_count(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compaction_drops_branches_rather_than_leaving_them_dangling() {
        let dir = scratch_dir("branch-compact");
        let cfg = test_cfg();
        let mut agent = branch_agent(&dir, &cfg);
        agent.fork_branch("2", false).unwrap();
        assert!(!agent.session.branches.is_empty());
        agent.rebuild_after_compact("summary");
        assert!(
            agent.session.branches.is_empty(),
            "branch parents index the transcript compaction just replaced"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn readonly_slash_commands_do_not_dirty_or_log() {
        let dir = scratch_dir("readonly-slash");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        // Session-start scaffolding, not activity.
        let combined = agent.context_content.combined();
        agent.session.push(Message::user(combined));
        agent.session.dirty = false;
        let before = agent.session.transcript.len();

        // Commands that only report state or edit settings must neither log
        // into the transcript nor mark the session dirty, so a session that
        // only ran them gets no resume point.
        for cmd in ["/usage", "/context", "/help", "/stats", "/config", "/mcp"] {
            agent.slash(cmd).unwrap_or_else(|e| panic!("{cmd}: {e}"));
            assert_eq!(
                agent.session.transcript.len(),
                before,
                "{cmd} must not log into the conversation"
            );
            assert!(!agent.session.dirty, "{cmd} must not dirty the session");
        }
        assert!(
            agent.save_for_exit().is_none(),
            "a report-only session gets no resume point"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    // Tier 2 and Tier 3 must be distinct KV spans, so the session-start context
    // enters as two user messages — but the concatenation must still be exactly
    // the old single block, in the same stable-then-volatile order (#60, #64).
    #[test]
    fn session_context_splits_into_stable_then_volatile_messages() {
        let content = ContextContent {
            git_content: Some("[git]\nbranch main".to_owned()),
            agents_md_content: Some("do the thing".to_owned()),
            memory_content: None,
            agents_content: None,
            date_content: "[date]\n2026-07-25".to_owned(),
        };
        let mut session = Session::new();
        push_session_context(&mut session, &content);
        assert_eq!(session.transcript.len(), 2, "one message per tier");
        assert!(
            session
                .transcript
                .iter()
                .all(|m| m.role == crate::session::Role::User)
        );
        assert_eq!(session.transcript[0].text, content.stable_context());
        assert_eq!(session.transcript[1].text, content.volatile_context());
        // No bytes added, dropped, or reordered by the split.
        assert_eq!(
            format!(
                "{}{}",
                session.transcript[0].text, session.transcript[1].text
            ),
            content.combined()
        );

        // No AGENTS.md: no Tier 2, so exactly one message — the pre-split shape.
        let mut only_volatile = ContextContent {
            agents_md_content: None,
            ..content.clone()
        };
        only_volatile.memory_content = None;
        let mut session = Session::new();
        push_session_context(&mut session, &only_volatile);
        assert_eq!(session.transcript.len(), 1);
        assert_eq!(session.transcript[0].text, only_volatile.combined());

        // Nothing at all: no scaffolding message is invented.
        let mut session = Session::new();
        push_session_context(&mut session, &ContextContent::default());
        assert!(session.transcript.is_empty());
    }

    #[test]
    fn context_scaffolding_alone_is_not_worth_a_resume_point() {
        let dir = scratch_dir("scaffold-only");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        // Reproduce what new_agent / `/clear` do at session start: inject the
        // session-start context, then mark it non-dirty (it is scaffolding, not
        // user activity — the fix).
        let combined = agent.context_content.combined();
        agent.session.push(Message::user(combined));
        agent.session.dirty = false;
        assert!(
            agent.save_for_exit().is_none(),
            "a session holding only session-start context gets no resume point"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bang_line_is_not_logged_and_leaves_no_resume_point() {
        let dir = scratch_dir("bang-no-log");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        // Session-start scaffolding, not activity.
        let combined = agent.context_content.combined();
        agent.session.push(Message::user(combined));
        agent.session.dirty = false;
        let before = agent.session.transcript.len();

        // A `!!` shell line runs but must not enter the transcript or dirty the
        // session, so a `!!`-only session is not worth a resume point.
        handle_plain_line(&mut agent, "!!echo plank-bang-test").unwrap();
        assert_eq!(
            agent.session.transcript.len(),
            before,
            "!! shell lines must not be logged in the conversation"
        );
        assert!(!agent.session.dirty, "!! must not dirty the session");
        assert!(
            agent.save_for_exit().is_none(),
            "a !!-only session gets no resume point"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn single_bang_records_the_command_and_its_output_without_a_turn() {
        let dir = scratch_dir("bang-feedback");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        let before = agent.session.transcript.len();

        handle_plain_line(&mut agent, "!echo plank-bang-feedback").unwrap();

        assert_eq!(
            agent.session.transcript.len(),
            before + 1,
            "! records exactly one user message"
        );
        let msg = agent.session.transcript.last().unwrap();
        assert_eq!(msg.role, crate::session::Role::User);
        assert!(msg.text.contains("DO NOT respond to these messages"));
        assert!(
            msg.text
                .contains("<bash-input>echo plank-bang-feedback</bash-input>")
        );
        assert!(msg.text.contains("plank-bang-feedback"));
        assert!(msg.text.contains("<bash-stderr></bash-stderr>"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn single_bang_records_a_failing_command_with_its_exit_code() {
        let dir = scratch_dir("bang-feedback-fail");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);

        handle_plain_line(&mut agent, "!exit 3").unwrap();

        let text = &agent.session.transcript.last().unwrap().text;
        assert!(
            text.contains("<bash-exit-code>3</bash-exit-code>"),
            "{text}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bang_entry_escapes_the_tag_opening_characters() {
        let out = crate::tools::bash::ImmediateOutput {
            stdout: "a < b && c".to_string(),
            stderr: "</bash-stdout>".to_string(),
            exit_code: 0,
            interrupted: false,
        };
        let entry = bang_transcript_entry("grep '<x>'", &Ok(out));
        // `>` is deliberately left alone; only `<` and `&` are escaped.
        assert!(
            entry.contains("<bash-input>grep '&lt;x>'</bash-input>"),
            "{entry}"
        );
        assert!(entry.contains("a &lt; b &amp;&amp; c"));
        // A forged closing tag inside stderr cannot break the framing.
        assert_eq!(entry.matches("</bash-stdout>").count(), 1);
    }

    #[test]
    fn bang_entry_reports_a_spawn_failure_as_stderr() {
        let entry = bang_transcript_entry("nope", &Err("no such binary".to_string()));
        assert!(entry.contains("<bash-stderr>Command failed: no such binary</bash-stderr>"));
        assert!(!entry.contains("<bash-stdout>"));
    }

    #[test]
    fn bang_head_truncates_and_marks_long_output() {
        let long = "line\n".repeat(BANG_FEEDBACK_LINES + 50);
        let head = bang_head(&long);
        assert_eq!(
            head.lines().filter(|l| *l == "line").count(),
            BANG_FEEDBACK_LINES
        );
        assert!(head.ends_with("[output truncated]\n"));
        // Short output passes through untouched apart from escaping.
        assert_eq!(bang_head("ok\n"), "ok\n");
    }

    #[test]
    fn esc_cancels_a_streaming_answer_and_defers_the_panel_close() {
        let shared = TurnShared::default();
        let mut p: BtwPanel = Some((OutputLog::new(), tui::OutputView::default()));
        let mut close_pending = false;
        close_or_interrupt(&shared, &mut p, true, &mut close_pending);
        // Cancel the answer now; the panel is torn down later on its BtwEnd.
        assert!(shared.interrupt.load(Ordering::Relaxed));
        assert!(close_pending);
        assert!(p.is_some());
    }

    #[test]
    fn esc_on_a_frozen_panel_dismisses_it_without_interrupting_the_task() {
        let shared = TurnShared::default();
        let mut p: BtwPanel = Some((OutputLog::new(), tui::OutputView::default()));
        let mut close_pending = false;
        close_or_interrupt(&shared, &mut p, false, &mut close_pending);
        assert!(
            !shared.interrupt.load(Ordering::Relaxed),
            "task keeps running"
        );
        assert!(p.is_none(), "panel dismissed");
        assert!(!close_pending);
    }

    #[test]
    fn esc_with_no_panel_interrupts_the_task() {
        let shared = TurnShared::default();
        let mut p: BtwPanel = None;
        let mut close_pending = false;
        close_or_interrupt(&shared, &mut p, false, &mut close_pending);
        assert!(shared.interrupt.load(Ordering::Relaxed));
        assert!(p.is_none());
    }

    // Ctrl-C during the summary pass must leave the conversation exactly as
    // it was: no summary, no rebuilt transcript, and the turn ends (the C's
    // "Compaction interrupted; keeping the previous conversation state.").
    #[test]
    fn interrupted_compaction_keeps_the_previous_transcript() {
        let dir = scratch_dir("compact-interrupt");
        let engine = ScriptedEngine {
            replies: vec!["a partial summ".to_string()],
            interrupt_at: Some(0),
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("first"));
        agent.session.push(Message::assistant("reply"));
        agent.session.push(Message::user("second"));
        let before: Vec<String> = agent
            .session
            .transcript
            .iter()
            .map(|m| m.text.clone())
            .collect();

        let mut notes = Vec::new();
        let mut note = |s: String| notes.push(s);
        let outcome = agent
            .do_compact_notify("low context", "", &mut NoteSink(&mut note), &|| true)
            .unwrap();

        assert_eq!(outcome, Compacted::Interrupted);
        let after: Vec<String> = agent
            .session
            .transcript
            .iter()
            .map(|m| m.text.clone())
            .collect();
        assert_eq!(after, before, "the transcript must be untouched");
        assert!(
            notes.iter().any(|n| n == COMPACT_INTERRUPTED),
            "got: {notes:?}"
        );
        assert!(
            !notes.iter().any(|n| n == "context compacted"),
            "compaction must not claim success: {notes:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    // The uninterrupted path still rebuilds, so the interrupt branch is not
    // swallowing normal compaction.
    #[test]
    fn uninterrupted_compaction_still_rebuilds() {
        let dir = scratch_dir("compact-ok");
        let engine = ScriptedEngine {
            replies: vec!["durable state: the user asked about X".to_string()],
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("first"));
        agent.session.push(Message::assistant("reply"));

        let mut notes = Vec::new();
        let mut note = |s: String| notes.push(s);
        let outcome = agent
            .do_compact_notify("low context", "", &mut NoteSink(&mut note), &|| false)
            .unwrap();

        assert_eq!(outcome, Compacted::Done);
        assert!(notes.iter().any(|n| n == "context compacted"), "{notes:?}");
    }

    /// A prompt-type hook group, which injects context without running a
    /// process — hermetic enough to assert hook dispatch in a unit test.
    fn prompt_hook(text: &str) -> Vec<crate::hooks::HookMatcher> {
        vec![crate::hooks::HookMatcher {
            matcher: String::new(),
            hooks: vec![crate::hooks::HookDef {
                command: String::new(),
                timeout_sec: 5,
                is_async: false,
                prompt: Some(text.to_string()),
            }],
        }]
    }

    // Compaction hooks used to fire only on the plain-REPL path, so a hook
    // configured by a TUI user — the default front-end — silently never ran.
    // Both orchestrators must dispatch both events.
    #[test]
    fn compaction_hooks_fire_on_the_tui_path() {
        let dir = scratch_dir("compact-hooks-tui");
        let engine = ScriptedEngine {
            replies: vec!["<summary>durable state</summary>".to_string()],
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.tool_ctx.hooks.pre_compact = prompt_hook("PRE-COMPACT-HOOK-RAN");
        agent.tool_ctx.hooks.post_compact = prompt_hook("POST-COMPACT-HOOK-RAN");
        agent.session.push(Message::user("first"));
        agent.session.push(Message::assistant("reply"));

        let mut notes = Vec::new();
        let mut note = |s: String| notes.push(s);
        let outcome = agent
            .do_compact_notify("user request", "", &mut NoteSink(&mut note), &|| false)
            .unwrap();
        assert_eq!(outcome, Compacted::Done);

        let rendered = render_transcript(&agent.session, "SYS");
        assert!(
            rendered.contains("PRE-COMPACT-HOOK-RAN"),
            "PreCompact context must reach the transcript: {rendered}"
        );
        assert!(
            rendered.contains("POST-COMPACT-HOOK-RAN"),
            "PostCompact context must reach the transcript: {rendered}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    // `manual` for a user-driven `/compact`, `auto` for a threshold-driven pass;
    // both orchestrators derive it the same way.
    #[test]
    fn compact_trigger_distinguishes_manual_from_auto() {
        assert_eq!(Agent::compact_trigger("user request"), "manual");
        assert_eq!(Agent::compact_trigger("low context"), "auto");
    }

    // Rebuilding on an empty summary would destroy the transcript and put an
    // empty summary in its place. A pass that comes back with nothing usable is
    // a failure that leaves the conversation alone.
    #[test]
    fn a_compaction_with_no_usable_summary_keeps_the_transcript() {
        let dir = scratch_dir("compact-empty");
        // A reply that is *only* a discarded <analysis> block extracts to
        // nothing — the realistic shape of this failure, not just an empty
        // string.
        let engine = ScriptedEngine {
            replies: vec!["<analysis>thinking about it</analysis>".to_string()],
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.tool_ctx.hooks.post_compact = prompt_hook("POST-COMPACT-HOOK-RAN");
        agent.session.push(Message::user("first"));
        agent.session.push(Message::assistant("reply"));
        let before: Vec<String> = agent
            .session
            .transcript
            .iter()
            .map(|m| m.text.clone())
            .collect();

        let mut notes = Vec::new();
        let mut note = |s: String| notes.push(s);
        let outcome = agent
            .do_compact_notify("low context", "", &mut NoteSink(&mut note), &|| false)
            .unwrap();

        assert_eq!(outcome, Compacted::NoSummary);
        assert!(outcome.aborted(), "the turn must be abandoned");
        let after: Vec<String> = agent
            .session
            .transcript
            .iter()
            .map(|m| m.text.clone())
            .collect();
        assert_eq!(after, before, "the transcript must be untouched");
        assert!(notes.iter().any(|n| n == COMPACT_NO_SUMMARY), "{notes:?}");
        assert!(
            !notes.iter().any(|n| n == "context compacted"),
            "compaction must not claim success: {notes:?}"
        );
        // PostCompact describes a completed compaction; there wasn't one.
        assert!(
            !render_transcript(&agent.session, "SYS").contains("POST-COMPACT-HOOK-RAN"),
            "PostCompact must not fire for a failed pass"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn btw_interrupt_flushes_remaining_queue() {
        let dir = scratch_dir("btw-flush");
        let engine = ScriptedEngine {
            replies: vec!["partial".to_string()],
            interrupt_at: Some(0),
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("main"));
        agent.last_ctx_used = 77;

        let shared = TurnShared::default();
        shared.push_btw("first".to_owned());
        shared.push_btw("second".to_owned());
        shared.push_btw("third".to_owned());
        let (tx, rx) = std::sync::mpsc::channel();
        agent.drain_btw(&tx, &shared);
        drop(tx);

        assert!(shared.pop_btw().is_none(), "queue must be flushed");
        assert_eq!(agent.last_ctx_used, 77);
        let events: Vec<UiEvent> = rx.try_iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, UiEvent::Dim(t) if t == "[btw queue cleared: 2]"))
        );
        // The panel is torn down even on the interrupt path, so the split
        // never lingers after the user cancels.
        assert!(events.iter().any(|e| matches!(e, UiEvent::BtwEnd)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn fmt_int_groups_thousands() {
        assert_eq!(fmt_int(0), "0");
        assert_eq!(fmt_int(42), "42");
        assert_eq!(fmt_int(1_000), "1,000");
        assert_eq!(fmt_int(17_859), "17,859");
        assert_eq!(fmt_int(-5), "0");
    }

    #[test]
    fn local_invoice_states_real_cost_is_zero() {
        let out = super::render_local_invoice("deepseek-v4-flash", 12_345, 6_789, false);
        // The gag must never be mistakable for a real charge.
        assert!(
            out.contains("Real cost is $0.00"),
            "missing real-cost disclaimer:\n{out}"
        );
        assert!(out.contains("This invoice is a joke"), "{out}");
        assert!(out.contains("ran locally"), "{out}");
        // No other dollar figure may appear anywhere in the block.
        assert_eq!(out.matches('$').count(), 1, "{out}");
    }

    #[test]
    fn local_invoice_reports_real_token_counts() {
        let out = super::render_local_invoice("m", 12_345, 6_789, false);
        assert!(out.contains("tokens in     12,345"), "{out}");
        assert!(out.contains("tokens out    6,789"), "{out}");
        assert!(out.starts_with("Local Inference Invoice — m\n"), "{out}");
    }

    #[test]
    fn local_invoice_units_are_absurd_and_deterministic() {
        let out = super::render_local_invoice("m", 4_000, 1_000, false);
        // 5,000 tokens: 1.000 Wh, 0.050 espressos, 55,000 revs, 5 knee-min.
        assert!(
            out.contains("electricity   1.000 Wh (0.050 espressos)"),
            "{out}"
        );
        assert!(out.contains("fan service   55,000 revolutions"), "{out}");
        assert!(out.contains("lap heat      5 toasty-knee-minutes"), "{out}");
        assert!(out.contains("GPU tears     1 "), "{out}");
    }

    #[test]
    fn local_invoice_handles_empty_model_and_zero_usage() {
        let out = super::render_local_invoice("", 0, 0, false);
        assert!(
            out.contains("Local Inference Invoice — local model"),
            "{out}"
        );
        assert!(out.contains("0.000 Wh"), "{out}");
        assert!(out.contains("Real cost is $0.00"), "{out}");
    }

    #[test]
    fn token_usage_add_saturates() {
        let mut a = crate::engine::TokenUsage {
            input_tokens: 10,
            output_tokens: 2,
            cache_read_tokens: 100,
            cache_write_tokens: 0,
        };
        a.add(crate::engine::TokenUsage {
            input_tokens: i32::MAX,
            output_tokens: 3,
            cache_read_tokens: 0,
            cache_write_tokens: 7,
        });
        assert_eq!(a.input_tokens, i32::MAX);
        assert_eq!(a.output_tokens, 5);
        assert_eq!(a.cache_read_tokens, 100);
        assert_eq!(a.cache_write_tokens, 7);
    }

    // The re-warm after `/new` restores a tier checkpoint: one blocking read
    // plus a backend load that emits NO engine events. So the indicator can only
    // appear if `rewarm_after_reset` ticks once on its own, before the walk.
    // Without that guarantee the prompt would simply freeze for the duration.
    #[test]
    fn rewarm_after_reset_ticks_even_with_no_engine_events() {
        let dir = std::env::temp_dir().join(format!("plank-rewarm-tick-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = crate::config::AgentConfig::default();
        let store = SessionStore::open(&dir).unwrap();
        let mut agent = Agent {
            engine: Box::new(crate::engine::EchoEngine::new(64)),
            cfg: &cfg,
            session: Session::new(),
            store,
            pending_aside: None,
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            isolation_seq: 0,
            system: "system prompt".to_string(),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            payload_restored: false,
            trusted_system_len: 0,
            think: cfg.generation.think_mode,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
            last_ctx_used: 0,
            last_turn_interrupted: false,
            context_content: crate::context::ContextContent::new(),
            skills: Vec::new(),
            templates: Vec::new(),
            agents: Vec::new(),
            checkpoints: crate::checkpoint::CheckpointStore::new(),
            remote: None,
            remote_server: None,
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
            sub_sink: SubSinkTarget::default(),
            fork_kv: Vec::new(),
            alt_engines: std::collections::HashMap::new(),
            local_alt_warmed: false,
            warm_note: None,
        };

        // The stub engine has no KV support, so `kvtier::warm` emits nothing at
        // all — exactly the shape of a checkpoint restore.
        let mut ticks = 0_usize;
        agent.rewarm_after_reset(&mut || ticks += 1);
        assert!(
            ticks >= 1,
            "the indicator must be painted before the blocking restore, not only \
             from engine events (got {ticks} ticks)"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn usage_report_tallies_provider_turns() {
        let dir = std::env::temp_dir().join(format!("plank-usage-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = crate::config::AgentConfig {
            provider: Some(crate::config::ProviderSelector::OpenAi),
            provider_model: Some("deepseek-v4-flash:cloud".to_string()),
            ..crate::config::AgentConfig::default()
        };
        let store = SessionStore::open(&dir).unwrap();
        let mut agent = Agent {
            engine: Box::new(crate::engine::EchoEngine::new(64)),
            cfg: &cfg,
            session: Session::new(),
            store,
            pending_aside: None,
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            isolation_seq: 0,
            system: String::new(),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            payload_restored: false,
            trusted_system_len: 0,
            think: cfg.generation.think_mode,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
            last_ctx_used: 0,
            last_turn_interrupted: false,
            context_content: crate::context::ContextContent::new(),
            skills: Vec::new(),
            templates: Vec::new(),
            agents: Vec::new(),
            checkpoints: crate::checkpoint::CheckpointStore::new(),
            remote: None,
            remote_server: None,
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
            sub_sink: SubSinkTarget::default(),
            fork_kv: Vec::new(),
            alt_engines: std::collections::HashMap::new(),
            local_alt_warmed: false,
            warm_note: None,
        };

        // Empty state: no provider turn recorded yet.
        assert!(
            agent
                .render_usage_report(false)
                .contains("No provider usage yet")
        );

        let mk = |input, output| crate::engine::GenerationStats {
            usage: Some(crate::engine::TokenUsage {
                input_tokens: input,
                output_tokens: output,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
            }),
            ..Default::default()
        };
        agent.record_usage(&mk(100, 20));
        agent.record_usage(&mk(50, 5));
        // A local pass (no usage) must not bump the turn count.
        agent.record_usage(&crate::engine::GenerationStats::default());

        let report = agent.render_usage_report(false);
        assert!(report.contains("deepseek-v4-flash:cloud"), "got: {report}");
        assert!(report.contains("turns          2"), "got: {report}");
        assert!(report.contains("input tokens   150"), "got: {report}");
        assert!(report.contains("output tokens  25"), "got: {report}");
        // No cache traffic on the OpenAI path: the section is omitted.
        assert!(!report.contains("cache read"), "got: {report}");
        assert!(report.contains("total tokens   175"), "got: {report}");

        // The engine-agnostic run stats tally both directions across every
        // pass, from provider usage here: in = 100+50, out = 20+5.
        assert_eq!(agent.stats.input_tokens, 150);
        assert_eq!(agent.stats.output_tokens, 25);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_stats_count_local_passes_from_the_context_delta() {
        // No provider `usage`: input is the growth in context minus what the
        // pass generated, and compaction (context shrinking) never subtracts.
        let dir = std::env::temp_dir().join(format!("plank-runstats-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        let local = |generated, ctx_used| crate::engine::GenerationStats {
            generated,
            ctx_used,
            ..Default::default()
        };
        // Pass 1: ctx 0 -> 130, generated 30  => input 100, output 30.
        agent.record_usage(&local(30, 130));
        agent.last_ctx_used = 130;
        // Pass 2: ctx 130 -> 175, generated 15 => input 30, output 15.
        agent.record_usage(&local(15, 175));
        agent.last_ctx_used = 175;
        // Pass 3: compaction shrank ctx to 40, generated 5 => input clamps to 0.
        agent.record_usage(&local(5, 40));
        assert_eq!(agent.stats.input_tokens, 130);
        assert_eq!(agent.stats.output_tokens, 50);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A checkpoint belongs to the model whose KV it holds, so Tier 1's key must
    /// follow the engine being warmed — not the live one. Warming the alt local
    /// engine under a provider main agent with the *provider's* model name would
    /// look up a checkpoint that cannot describe its KV.
    #[test]
    fn tier_one_is_keyed_by_the_engine_being_warmed() {
        let dir = scratch_dir("kv-tiers-model");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent.system = "SYSTEM".to_string();

        let live = agent.kv_tiers();
        let other = agent.kv_tiers_for("some-other-model");

        assert_eq!(
            agent.kv_tiers(),
            live,
            "keying is a pure function of inputs"
        );
        assert_ne!(
            live[0].fingerprint, other[0].fingerprint,
            "the model name is part of Tier 1's identity"
        );
        // And the live engine's own name reproduces the live plan, so
        // `kv_tiers` is genuinely just this call with one argument filled in.
        assert_eq!(agent.kv_tiers_for(&agent.engine.model_name()), live);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The whole launch cycle, in the order a launch runs it: warm the alt
    /// engine, then GC. These two fought — the warm wrote a checkpoint and the
    /// GC deleted it before the next launch could read it — and neither in
    /// isolation was wrong. Only the pair is.
    #[test]
    fn a_launch_keeps_the_checkpoint_it_just_wrote() {
        let dir = scratch_dir("launch-cycle");
        let cfg = test_cfg();
        // A provider main agent: its Tier 1 fingerprint never has a file, which
        // is what turned the GC into a full sweep.
        let mut agent = test_agent(
            &dir,
            ScriptedEngine {
                model: Some("provider-model".to_string()),
                ..ScriptedEngine::default()
            },
            &cfg,
        );
        agent.system = "SYSTEM".to_string();
        agent.alt_engines.insert(
            EngineKey::Local,
            Box::new(ScriptedEngine {
                local: true,
                model: Some("ds4-local".to_string()),
                kv_events: Some(std::sync::Arc::new(std::sync::Mutex::new(Vec::new()))),
                ..ScriptedEngine::default()
            }),
        );
        let alt_key = crate::session::KvKey::System {
            fp: agent
                .kv_tiers_for("ds4-local")
                .into_iter()
                .find(|t| t.kind == crate::kvtier::TierKind::System)
                .expect("a system tier")
                .fingerprint,
        };

        agent.warm_alt_local_tier1(&mut |_| {}, &mut |_| {});
        assert!(
            agent.store.kv_path(&alt_key).exists(),
            "the warm wrote it: {}",
            agent.store.kv_path(&alt_key).display()
        );

        agent.gc_kv_tiers(&agent.kv_tiers());

        assert!(
            agent.store.kv_path(&alt_key).exists(),
            "and the GC in the same launch did not take it away again"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Three launches in a row: the toll is paid once, not once per launch.
    /// Each launch gets fresh engines, because carrying one engine across all
    /// three would hide a checkpoint that never survives to disk.
    ///
    /// Measured as "was the checkpoint already there when the launch started" —
    /// absent once, then present. Counting `warm_reset` would not do: it runs
    /// on a restore too, so it reads the same whether the launch prefilled or
    /// not, which is exactly the distinction under test.
    #[test]
    fn repeated_launches_warm_the_sub_agent_engine_only_once() {
        let dir = scratch_dir("launch-repeat");
        let cfg = test_cfg();
        let mut found_on_entry = Vec::new();

        for _ in 0..3 {
            let mut agent = test_agent(
                &dir,
                ScriptedEngine {
                    model: Some("provider-model".to_string()),
                    ..ScriptedEngine::default()
                },
                &cfg,
            );
            agent.system = "SYSTEM".to_string();
            let tiers = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            agent.alt_engines.insert(
                EngineKey::Local,
                Box::new(ScriptedEngine {
                    local: true,
                    model: Some("ds4-local".to_string()),
                    warm_tiers: Some(std::sync::Arc::clone(&tiers)),
                    kv_events: Some(std::sync::Arc::new(std::sync::Mutex::new(Vec::new()))),
                    ..ScriptedEngine::default()
                }),
            );

            let key = crate::session::KvKey::System {
                fp: agent
                    .kv_tiers_for("ds4-local")
                    .into_iter()
                    .find(|t| t.kind == crate::kvtier::TierKind::System)
                    .expect("a system tier")
                    .fingerprint,
            };
            found_on_entry.push(agent.store.kv_path(&key).exists());

            agent.warm_alt_local_tier1(&mut |_| {}, &mut |_| {});
            agent.gc_kv_tiers(&agent.kv_tiers());
            // Warmed exactly once per launch either way — the question is
            // whether that warm had a checkpoint to restore from.
            assert_eq!(tiers.lock().unwrap().len(), 1);
        }

        assert_eq!(
            found_on_entry,
            vec![false, true, true],
            "cold once, then restored — not re-cached every launch"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The GC keeps every live Tier 1, not just the main engine's. Collecting
    /// against one fingerprint deleted the alt local engine's checkpoint on
    /// every launch — so it re-prefilled its system prompt every single run —
    /// and under a provider main agent it swept the directory clean, the
    /// provider's own fingerprint never having a file to match.
    #[test]
    fn gc_keeps_the_alt_local_engines_system_checkpoint() {
        let dir = scratch_dir("gc-alt-tier1");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent.system = "SYSTEM".to_string();
        agent.alt_engines.insert(
            EngineKey::Local,
            Box::new(ScriptedEngine {
                local: true,
                model: Some("ds4-local".to_string()),
                ..ScriptedEngine::default()
            }),
        );

        let main_tiers = agent.kv_tiers();
        let alt_tiers = agent.kv_tiers_for("ds4-local");
        let fp = |t: &[crate::kvtier::TierSpec]| {
            t.iter()
                .find(|t| t.kind == crate::kvtier::TierKind::System)
                .expect("a system tier")
                .fingerprint
                .clone()
        };
        let (main_fp, alt_fp) = (fp(&main_tiers), fp(&alt_tiers));
        assert_ne!(main_fp, alt_fp, "different models, different Tier 1");

        // Both engines' checkpoints on disk, plus a stale third.
        let key = |fp: &str| crate::session::KvKey::System { fp: fp.to_owned() };
        for f in [&main_fp, &alt_fp, &"stale".to_string()] {
            std::fs::write(agent.store.kv_path(&key(f)), b"x").unwrap();
        }

        agent.gc_kv_tiers(&main_tiers);

        assert!(
            agent.store.kv_path(&key(&alt_fp)).exists(),
            "the sub-agent engine's checkpoint survives its own launch"
        );
        assert!(agent.store.kv_path(&key(&main_fp)).exists());
        assert!(!agent.store.kv_path(&key("stale")).exists(), "stale swept");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Startup warms only Tier 1 for the alt engine, and that narrowness is
    /// load-bearing: `warm` restores the deepest tier that loads and skips
    /// every tier above it, so a valid Tier 2 checkpoint means Tier 1 is never
    /// prefilled and never written. A one-tier list has nothing deeper to
    /// short-circuit it, so the checkpoint the sub-agent needs actually
    /// appears.
    #[test]
    fn startup_warms_only_tier_one_for_the_alt_engine() {
        let dir = scratch_dir("alt-warm-tier1");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent.system = "SYSTEM".to_string();
        agent.context_content = crate::context::ContextContent::new();
        let stages = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        agent.alt_engines.insert(
            EngineKey::Local,
            Box::new(ScriptedEngine {
                local: true,
                warm_tiers: Some(std::sync::Arc::clone(&stages)),
                ..ScriptedEngine::default()
            }),
        );

        agent.warm_alt_local_tier1(&mut |_| {}, &mut |_| {});

        assert!(agent.local_alt_warmed, "and it is not redone on first take");
        let seen = stages.lock().unwrap().clone();
        assert_eq!(seen, vec!["SYSTEM"], "system tier only: {seen:?}");
        assert!(
            agent.alt_engines.contains_key(&EngineKey::Local),
            "the engine goes back in the cache"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The alt local engine is warmed on its first take and never again: it
    /// keeps its session across sidechains, so every later dispatch already
    /// finds the system prefix in its KV.
    #[test]
    fn the_alt_local_engine_is_warmed_once() {
        let dir = scratch_dir("alt-local-warm");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent.alt_engines.insert(
            EngineKey::Local,
            Box::new(ScriptedEngine {
                local: true,
                ..ScriptedEngine::default()
            }),
        );
        assert!(!agent.local_alt_warmed);

        let (key, engine) = agent
            .take_alt_engine(&crate::agents::AgentEngine::Local)
            .expect("the local engine is available");
        assert_eq!(key, EngineKey::Local);
        assert!(agent.local_alt_warmed, "the first take warms it");
        agent.alt_engines.insert(key, engine);

        // Second take: still warmed, and the flag is not a per-take toggle.
        let (key, engine) = agent
            .take_alt_engine(&crate::agents::AgentEngine::Local)
            .expect("still available");
        assert!(agent.local_alt_warmed);
        agent.alt_engines.insert(key, engine);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Every pass is attributed to the engine that served it, so a sidechain on
    /// an alternate engine lands in its own row rather than the main agent's.
    /// That split is the whole point under a provider main agent: it answers
    /// which of the session's tokens were billed and which ran on this machine.
    #[test]
    fn run_stats_split_by_the_engine_that_served_the_pass() {
        let dir = scratch_dir("runstats-by-engine");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        let local = |generated, ctx_used| crate::engine::GenerationStats {
            generated,
            ctx_used,
            ..Default::default()
        };
        // Two passes on the main engine: in 100 + 30, out 30 + 15.
        agent.record_usage(&local(30, 130));
        agent.last_ctx_used = 130;
        agent.record_usage(&local(15, 175));

        // A sidechain swaps `self.engine` before the pass, which is exactly
        // what `record_usage` reads — so stand a second engine up the same way.
        let parent = std::mem::replace(
            &mut agent.engine,
            Box::new(ScriptedEngine {
                local: true,
                ..ScriptedEngine::default()
            }),
        );
        agent.last_ctx_used = 0;
        agent.record_usage(&local(7, 57));
        agent.engine = parent;

        assert_eq!(agent.stats.input_tokens, 180, "totals cover both engines");
        assert_eq!(agent.stats.output_tokens, 52);
        assert_eq!(
            agent.stats.by_engine.len(),
            2,
            "{:?}",
            agent.stats.by_engine
        );
        // The main engine leads: it served first.
        let (_, main_in, main_out) = agent.stats.by_engine[0].clone();
        assert_eq!((main_in, main_out), (130, 45));
        let (alt_label, alt_in, alt_out) = agent.stats.by_engine[1].clone();
        assert_eq!((alt_in, alt_out), (50, 7));
        assert!(
            alt_label.contains("(local)"),
            "a local engine is marked as such: {alt_label}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn duration_formats_with_and_without_hours() {
        use std::time::Duration;
        assert_eq!(fmt_duration(Duration::from_secs(0)), "0:00");
        assert_eq!(fmt_duration(Duration::from_secs(247)), "4:07");
        assert_eq!(fmt_duration(Duration::from_secs(3729)), "1:02:09");
        assert_eq!(fmt_u64(1_234_567), "1,234,567");
    }

    #[test]
    fn malformed_stanza_feeds_c_format_tool_error() {
        let dir = std::env::temp_dir().join(format!("plank-ui-err-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptedEngine {
            replies: vec![
                // Legal opener, then a bogus tag the strict parser rejects.
                "<｜DSML｜tool_calls><b>".to_string(),
                "Understood.\n".to_string(),
            ],
            ..ScriptedEngine::default()
        };
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.think_mode = crate::engine::ThinkMode::Off;
        let store = SessionStore::open(&dir).unwrap();
        let mut agent = Agent {
            engine: Box::new(engine),
            cfg: &cfg,
            session: Session::new(),
            store,
            pending_aside: None,
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            isolation_seq: 0,
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            payload_restored: false,
            trusted_system_len: 0,
            think: cfg.generation.think_mode,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
            last_ctx_used: 0,
            last_turn_interrupted: false,
            context_content: crate::context::ContextContent::new(),
            skills: Vec::new(),
            templates: Vec::new(),
            agents: Vec::new(),
            checkpoints: crate::checkpoint::CheckpointStore::new(),
            remote: None,
            remote_server: None,
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
            sub_sink: SubSinkTarget::default(),
            fork_kv: Vec::new(),
            alt_engines: std::collections::HashMap::new(),
            local_alt_warmed: false,
            warm_note: None,
        };
        agent.session.push(Message::user("go"));
        agent.run_turn().unwrap();

        // user, assistant(bad stanza), user(tool error), assistant(final)
        let tool_result = &agent.session.transcript[2].text;
        assert!(
            tool_result.contains("Tool error: invalid DSML tool call: unexpected DSML tag: <b>\n"),
            "got: {tool_result}"
        );
        assert!(
            tool_result.contains("DSML syntax reminder:"),
            "got: {tool_result}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn close_open_think_appends_only_when_needed() {
        let mut open = "let me check".to_string();
        close_open_think(&mut open, true);
        assert_eq!(open, "let me check</think>");

        let mut closed = "done</think>answer".to_string();
        close_open_think(&mut closed, false);
        assert_eq!(closed, "done</think>answer");

        // Already closed by the model on the final byte: no second tag.
        let mut exact = "done</think>".to_string();
        close_open_think(&mut exact, true);
        assert_eq!(exact, "done</think>");
    }

    #[test]
    fn a_discarded_in_think_stanza_tells_the_model_it_was_misplaced() {
        // Parity mode (`StreamRenderer` default: thinking tool calls not
        // allowed): a stanza fired mid-thought is discarded, so the pass
        // produces no calls. It must still report *why*, and the reason is
        // placement, not syntax — the markup here is perfectly well formed.
        const BASH_STANZA: &str = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\">ls -la</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        let mut stream = StreamRenderer::new(NullSink);
        stream.push(format!("<think>let me look{BASH_STANZA}"));
        stream.finish();
        let finished = stream.finished();
        assert!(finished.ended_in_think, "block should still be open");
        assert!(finished.calls.is_empty(), "call must be discarded");
        assert!(finished.in_think_rejected, "rejected for its placement");
        assert_eq!(
            finished.error,
            Some(crate::sysprompt::IN_THINK_PROHIBITION),
            "the model must be told about placement, not syntax"
        );

        // The pass continues (an error is fed back), so the open think block
        // is closed before the tool_result — an unterminated <think> must
        // never sit in front of a user message.
        let turn_continues = !finished.calls.is_empty() || finished.error.is_some();
        assert!(turn_continues);
        let mut assistant_text = format!("let me look{BASH_STANZA}");
        close_open_think(
            &mut assistant_text,
            finished.ended_in_think && turn_continues,
        );
        assert!(
            assistant_text.ends_with("</think>"),
            "got: {assistant_text}"
        );
    }

    /// Regression: an in-think call used to be reported to the model as
    /// invalid DSML — sometimes as "incomplete DSML tool call", the parser's
    /// verdict on a stanza it never got to finish. Both send the model
    /// rewriting markup that was correct. It gets the placement rule instead,
    /// with no syntax reminder attached.
    #[test]
    fn the_in_think_payload_talks_about_placement_not_syntax() {
        let payload = tool_error_payload(PassError::InThink, "incomplete DSML tool call");
        assert!(
            payload.contains(crate::sysprompt::IN_THINK_PROHIBITION),
            "{payload:?}"
        );
        assert_eq!(
            payload,
            concat!(
                "Tool error: Tool calls are not allowed inside <think></think>;",
                " finish thinking before emitting DSML.\n",
                "The tool call was not run. Close the thinking block with ",
                "</think>, then emit the same call again.\n",
            ),
            "exact model-facing wording"
        );
        assert!(!payload.contains("invalid DSML"), "{payload:?}");
        assert!(!payload.contains("incomplete DSML"), "{payload:?}");
        assert!(
            !payload.contains("DSML syntax reminder"),
            "the syntax was fine: {payload:?}"
        );

        // A genuine syntax failure is untouched: prefix and reminder both.
        let dsml = tool_error_payload(PassError::Dsml, "unclosed parameter");
        assert!(dsml.contains("invalid DSML tool call: unclosed parameter"));
        assert!(dsml.contains("DSML syntax reminder"));

        // A preflight failure is still fed back verbatim.
        assert_eq!(
            tool_error_payload(PassError::Preflight, "old not found"),
            "Tool error: old not found\n"
        );
    }

    #[test]
    fn tool_call_inside_think_is_dispatched_and_the_block_is_closed() {
        // Opt this thread into in-think dispatch; the shipped default is off.
        let mut settings = crate::settings::Settings::default();
        settings.engine.thinking_tool_calls = true;
        crate::settings::install_for_test(settings);
        let dir = std::env::temp_dir().join(format!("plank-ui-think-tool-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptedEngine {
            replies: vec![
                // A call fired mid-thought: no </think> before the stanza.
                concat!(
                    "<think>I should list the directory",
                    "<｜DSML｜tool_calls>",
                    "<｜DSML｜invoke name=\"bash\">",
                    "<｜DSML｜parameter name=\"command\">echo hello</｜DSML｜parameter｜>",
                    "</｜DSML｜invoke｜>",
                    "</｜DSML｜tool_calls｜>",
                )
                .to_string(),
                "Done.\n".to_string(),
            ],
            ..ScriptedEngine::default()
        };
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.think_mode = crate::engine::ThinkMode::Off;
        let store = SessionStore::open(&dir).unwrap();
        let mut agent = Agent {
            engine: Box::new(engine),
            cfg: &cfg,
            session: Session::new(),
            store,
            pending_aside: None,
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            isolation_seq: 0,
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            payload_restored: false,
            trusted_system_len: 0,
            think: cfg.generation.think_mode,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
            last_ctx_used: 0,
            last_turn_interrupted: false,
            context_content: crate::context::ContextContent::new(),
            skills: Vec::new(),
            templates: Vec::new(),
            agents: Vec::new(),
            checkpoints: crate::checkpoint::CheckpointStore::new(),
            remote: None,
            remote_server: None,
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
            sub_sink: SubSinkTarget::default(),
            fork_kv: Vec::new(),
            alt_engines: std::collections::HashMap::new(),
            local_alt_warmed: false,
            warm_note: None,
        };
        agent.session.push(Message::user("go"));
        agent.run_turn().unwrap();

        // user, assistant(in-think call), user(tool result), assistant(final)
        assert_eq!(agent.session.transcript.len(), 4);
        let assistant = &agent.session.transcript[1].text;
        assert!(
            assistant.ends_with("</think>"),
            "the open think block is closed before the tool result: {assistant:?}"
        );
        let result = &agent.session.transcript[2].text;
        assert!(
            result.contains("Tool result 1 (bash)"),
            "the in-think call was dispatched: {result:?}"
        );
        assert!(result.contains("hello"), "{result:?}");
        assert!(!result.contains("tool call ignored"), "{result:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn interrupted_mid_think_stanza_leaves_block_open() {
        // A user interrupt lands while a DSML stanza is being streamed inside
        // <think>, cutting it off incomplete. The stream reports both
        // `stats.interrupted` and a parse error for the truncated stanza —
        // but a real interrupt never continues with a <tool_result>, so
        // `close_open_think` must not append a synthetic `</think>` here
        // (finding 1's second named case, distinct from parity-mode discard).
        let dir =
            std::env::temp_dir().join(format!("plank-ui-think-interrupt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptedEngine {
            replies: vec![
                concat!(
                    "<think>let me check",
                    "<｜DSML｜tool_calls>",
                    "<｜DSML｜invoke name=\"bash\">",
                    "<｜DSML｜parameter name=\"command\">echo hi",
                )
                .to_string(),
            ],
            interrupt_at: Some(0),
            ..ScriptedEngine::default()
        };
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.think_mode = crate::engine::ThinkMode::Off;
        let store = SessionStore::open(&dir).unwrap();
        let mut agent = Agent {
            engine: Box::new(engine),
            cfg: &cfg,
            session: Session::new(),
            store,
            pending_aside: None,
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            isolation_seq: 0,
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            payload_restored: false,
            trusted_system_len: 0,
            think: cfg.generation.think_mode,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
            last_ctx_used: 0,
            last_turn_interrupted: false,
            context_content: crate::context::ContextContent::new(),
            skills: Vec::new(),
            templates: Vec::new(),
            agents: Vec::new(),
            checkpoints: crate::checkpoint::CheckpointStore::new(),
            remote: None,
            remote_server: None,
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
            sub_sink: SubSinkTarget::default(),
            fork_kv: Vec::new(),
            alt_engines: std::collections::HashMap::new(),
            local_alt_warmed: false,
            warm_note: None,
        };
        agent.session.push(Message::user("go"));
        agent.run_turn().unwrap();

        assert!(agent.last_turn_interrupted);
        // user, assistant(cut-off reply) — the turn stopped, no <tool_result>
        // ever follows, so no synthetic </think> should have been appended.
        assert_eq!(agent.session.transcript.len(), 2);
        let assistant = &agent.session.transcript[1].text;
        assert!(
            !assistant.ends_with("</think>"),
            "an interrupted stanza must not gain a synthetic close: {assistant:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_think_with_no_call_or_error_is_not_synthetically_closed() {
        // The counterpart to `tool_call_inside_think_is_dispatched_and_the_
        // block_is_closed`: the model stops with <think> still open but
        // produces no DSML at all (no call, no parse error, no interrupt) —
        // there is nothing to continue with, so `run_turn` must not append a
        // synthetic </think> here either. This exercises the production gate
        // in `run_turn` directly (not a re-derived copy), and would fail the
        // same way a regression to unconditional `ended_in_think` would.
        let dir =
            std::env::temp_dir().join(format!("plank-ui-think-no-call-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptedEngine {
            replies: vec!["<think>nothing to do here, done for now".to_string()],
            ..ScriptedEngine::default()
        };
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.think_mode = crate::engine::ThinkMode::Off;
        let store = SessionStore::open(&dir).unwrap();
        let mut agent = Agent {
            engine: Box::new(engine),
            cfg: &cfg,
            session: Session::new(),
            store,
            pending_aside: None,
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            isolation_seq: 0,
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            payload_restored: false,
            trusted_system_len: 0,
            think: cfg.generation.think_mode,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
            last_ctx_used: 0,
            last_turn_interrupted: false,
            context_content: crate::context::ContextContent::new(),
            skills: Vec::new(),
            templates: Vec::new(),
            agents: Vec::new(),
            checkpoints: crate::checkpoint::CheckpointStore::new(),
            remote: None,
            remote_server: None,
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
            sub_sink: SubSinkTarget::default(),
            fork_kv: Vec::new(),
            alt_engines: std::collections::HashMap::new(),
            local_alt_warmed: false,
            warm_note: None,
        };
        agent.session.push(Message::user("go"));
        agent.run_turn().unwrap();

        // user, assistant(final) — the turn ended normally with no call and
        // no tool result to follow.
        assert_eq!(agent.session.transcript.len(), 2);
        let assistant = &agent.session.transcript[1].text;
        assert!(
            !assistant.ends_with("</think>"),
            "an open think block with nothing to continue must stay open: {assistant:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Engine stub with canned KV snapshot support, standing in for
    /// `Ds4Engine`'s snapshot/restore paths in payload bookkeeping tests.
    /// The payload sidecar + fingerprint staleness wrapping lives in the
    /// `Agent` layer (`save_session_payload`/`load_session_payload`), so this
    /// mock only needs to yield and accept raw KV bytes.
    #[derive(Debug)]
    struct KvEngine;

    impl Engine for KvEngine {
        fn generate(
            &mut self,
            _prompt: crate::engine::Prompt<'_>,
            _opts: &crate::engine::GenerationOptions,
            _interrupt: &dyn Fn() -> bool,
            _greedy: &dyn Fn() -> bool,
            _on_event: &mut dyn FnMut(EngineEvent),
        ) -> Result<GenerationStats, EngineError> {
            Ok(GenerationStats::default())
        }
        fn ctx_size(&self) -> i32 {
            100_000
        }
        fn model_name(&self) -> String {
            "kv-test-model".to_owned()
        }
        fn get_kv(&mut self) -> Option<crate::kvcache::KVCache> {
            Some(crate::kvcache::KVCache::new(
                b"fake-kv-bytes".to_vec(),
                crate::ds4tokens::TokenTranscript::new(),
            ))
        }
        fn set_kv(&mut self, _cache: &crate::kvcache::KVCache) -> Result<(), EngineError> {
            Ok(())
        }
    }

    #[test]
    fn payload_save_resume_strip_flow() {
        let dir = std::env::temp_dir().join(format!("plank-ui-kv-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = crate::config::AgentConfig::default();
        let store = SessionStore::open(&dir).unwrap();
        let mut agent = Agent {
            engine: Box::new(KvEngine),
            cfg: &cfg,
            session: Session::new(),
            store,
            pending_aside: None,
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            isolation_seq: 0,
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            payload_restored: false,
            trusted_system_len: 0,
            think: cfg.generation.think_mode,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
            last_ctx_used: 0,
            last_turn_interrupted: false,
            context_content: crate::context::ContextContent::new(),
            skills: Vec::new(),
            templates: Vec::new(),
            agents: Vec::new(),
            checkpoints: crate::checkpoint::CheckpointStore::new(),
            remote: None,
            remote_server: None,
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
            sub_sink: SubSinkTarget::default(),
            fork_kv: Vec::new(),
            alt_engines: std::collections::HashMap::new(),
            local_alt_warmed: false,
            warm_note: None,
        };
        agent.session.push(Message::user("kv payload flow"));
        agent.session.push(Message::assistant("ack"));
        let id = agent.store.save(&mut agent.session).unwrap();

        // /save writes a fingerprinted payload sidecar next to the transcript.
        let note = agent.save_session_payload().unwrap();
        assert!(note.starts_with("saved KV payload ("), "got: {note}");
        assert!(agent.store.payload_bytes(&id) > 0);

        // /switch on an unchanged session restores the payload.
        let loaded = agent.store.load(&id[..8]).unwrap();
        assert_eq!(
            agent.load_session_payload(&loaded).as_deref(),
            Some("restored KV payload; resume skips re-prefill")
        );

        // A transcript that grew since the save makes the payload stale:
        // it is ignored (re-prefill), never trusted.
        let mut grown = loaded.clone();
        grown.push(Message::user("one more turn"));
        assert_eq!(
            agent.load_session_payload(&grown).as_deref(),
            Some("KV payload is stale; the transcript will be re-prefilled")
        );

        // /strip removes the payload and reports the transcript token cost.
        let (sha, tokens) = agent.strip_session(&id[..8]).unwrap();
        assert_eq!(sha, id);
        assert!(tokens > 0, "strip must report the re-prefill token count");
        assert_eq!(agent.store.payload_bytes(&id), 0);
        // Without a payload there is nothing to note on resume.
        assert_eq!(agent.load_session_payload(&loaded), None);
        // Stripping again still succeeds, like the C's rewrite.
        assert!(agent.strip_session(&id[..8]).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Builds an `agent` tool call, optionally naming a definition.
    fn agent_call(task: &str, name: Option<&str>) -> ToolCall {
        let mut args = vec![crate::dsml::ToolArg {
            name: "task".to_string(),
            value: task.to_string(),
            is_string: true,
        }];
        if let Some(n) = name {
            args.push(crate::dsml::ToolArg {
                name: "name".to_string(),
                value: n.to_string(),
                is_string: true,
            });
        }
        ToolCall {
            name: "agent".to_string(),
            args,
        }
    }

    fn named_def(name: &str, auto: bool) -> crate::agents::AgentDef {
        crate::agents::AgentDef {
            name: name.to_string(),
            description: String::new(),
            body: "Persona.".to_string(),
            path: std::path::PathBuf::from(format!("/tmp/{name}.md")),
            engine: None,
            auto,
            isolate: false,
        }
    }

    #[test]
    fn unknown_agent_name_falls_back_to_general_purpose() {
        let dir = std::env::temp_dir().join(format!("plank-ui-agent-fb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptedEngine {
            replies: vec!["Done.\n".to_string(), "Done again.\n".to_string()],
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.agents = vec![named_def("hidden", false)];

        // A name the model could not have known about must not burn a round.
        let out = agent.run_agent_tool(&agent_call("do a thing", Some("nonesuch")));
        assert!(!out.contains("unknown agent"), "no hard error: {out}");
        assert!(
            out.contains("note: no agent named 'nonesuch'"),
            "the report says what happened: {out}"
        );
        assert!(out.contains("Sub-agent report:"), "it still ran: {out}");

        // An `auto: false` definition is not model-selectable either, and is
        // treated exactly like a typo rather than as a distinct error.
        let out = agent.run_agent_tool(&agent_call("do a thing", Some("hidden")));
        assert!(
            out.contains("note: no agent named 'hidden'"),
            "auto:false is not model-selectable: {out}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A remote-backed definition pinned to a test-only key variable, with an
    /// explicit `ctx` so `take_alt_engine` never probes the network.
    fn remote_def(name: &str, key_env: &str) -> crate::agents::AgentDef {
        crate::agents::AgentDef {
            name: name.to_string(),
            description: String::new(),
            body: "Persona.".to_string(),
            path: std::path::PathBuf::from(format!("/tmp/{name}.md")),
            engine: Some(crate::agents::AgentEngine::Provider(
                crate::agents::ProviderSpec {
                    kind: crate::remote::provider::ProviderKind::Anthropic,
                    model: "test-model".to_string(),
                    base_url: Some("https://example.invalid/v1".to_string()),
                    ctx: Some(8192),
                    api_key_env: key_env.to_string(),
                },
            )),
            auto: true,
            isolate: false,
        }
    }

    /// A `provider: local` definition under a *local* main agent is not an
    /// override: the parent already is the local engine, so nothing is taken out
    /// of the cache and no second engine is held.
    #[test]
    fn provider_local_runs_on_the_parent_when_the_main_agent_is_local() {
        let dir = scratch_dir("alt-local-parent");
        let cfg = test_cfg();
        assert!(cfg.provider.is_none(), "test_cfg is a local main agent");
        let agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        assert!(
            agent
                .resolve_alt_spec(Some(crate::agents::AgentEngine::Local))
                .is_none(),
            "local under local resolves to the parent engine"
        );
    }

    /// Under a provider main agent the same definition *is* an override, and it
    /// runs on the local engine handed in at startup.
    #[test]
    fn provider_local_takes_the_startup_engine_under_a_provider_main() {
        let dir = scratch_dir("alt-local-provider");
        let mut cfg = test_cfg();
        cfg.provider = Some(crate::config::ProviderSelector::Anthropic);
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        let spec = crate::agents::AgentEngine::Local;

        // Without one, the dispatch must refuse and say why — never silently run
        // the sidechain on the remote model the definition declined.
        let err = agent.take_alt_engine(&spec).expect_err("no local engine");
        assert!(err.contains("no local engine"), "{err}");

        // Seeded at startup (what `new_agent` does with `--provider` plus a
        // `provider: local` definition), it is taken out for the sidechain.
        agent.alt_engines.insert(
            EngineKey::Local,
            Box::new(ScriptedEngine {
                replies: vec!["from the local model\n".to_string()],
                ..ScriptedEngine::default()
            }),
        );
        let (key, engine) = agent.take_alt_engine(&spec).expect("local engine");
        assert_eq!(key, EngineKey::Local);
        assert!(
            agent.alt_engines.is_empty(),
            "removed while its sidechain runs, so it cannot be in two places"
        );
        // And the resolver keeps it an override in this configuration.
        assert!(
            agent
                .resolve_alt_spec(Some(crate::agents::AgentEngine::Local))
                .is_some(),
            "local under a provider main stays an override"
        );
        drop(engine);
    }

    /// The cache key `take_alt_engine` derives from [`remote_def`], so a test can
    /// pre-seed `alt_engines` and keep the dispatch entirely offline.
    fn remote_key(key_env: &str) -> EngineKey {
        EngineKey::Provider(
            crate::remote::provider::ProviderKind::Anthropic,
            "https://example.invalid/v1".to_string(),
            "test-model".to_string(),
            8192,
            key_env.to_string(),
        )
    }

    #[test]
    fn generate_pass_runs_without_an_agent() {
        // The point of the extraction: a pass needs only an engine and plain
        // data, so several can run on separate threads. If this ever needs an
        // `Agent`, the fan-out is no longer possible.
        let mut engine = ScriptedEngine {
            replies: vec!["hello from the pass\n".to_string()],
            ..ScriptedEngine::default()
        };
        let opts = crate::engine::GenerationOptions::default();
        let ctx = PassCtx {
            opts: &opts,
            think_off: true,
            thinking_tool_calls: false,
            tool_names: Vec::new(),
        };
        let pass = generate_pass(
            &mut engine,
            "[user]\nhi\n",
            None,
            &ctx,
            Box::new(NullSink),
            |_call| Ok(()),
        )
        .expect("a pass");
        assert!(
            pass.assistant_text.contains("hello from the pass"),
            "{}",
            pass.assistant_text
        );
        assert!(pass.calls.is_empty());
        assert!(pass.tool_error.is_none());
    }

    /// The wiring the blink depends on: a pass on a local engine must mark itself
    /// local *while generating*, and must clear it afterwards. Without this the
    /// renderer's blink logic would be correct and never triggered.
    #[test]
    fn a_local_pass_marks_itself_local_while_it_generates() {
        let seen = std::sync::Arc::new(AtomicBool::new(false));
        let mut engine = ScriptedEngine {
            replies: vec!["done\n".to_string()],
            local: true,
            saw_local_pass: Some(std::sync::Arc::clone(&seen)),
            ..ScriptedEngine::default()
        };
        assert!(
            !crate::status::local_pass_active(),
            "nothing running before the pass"
        );
        let opts = crate::engine::GenerationOptions::default();
        let ctx = PassCtx {
            opts: &opts,
            think_off: true,
            thinking_tool_calls: false,
            tool_names: Vec::new(),
        };
        generate_pass(
            &mut engine,
            "[user]\nhi\n",
            None,
            &ctx,
            Box::new(NullSink),
            |_call| Ok(()),
        )
        .expect("a pass");

        assert!(
            seen.load(Ordering::Relaxed),
            "the pass was not marked local while generating"
        );
        assert!(
            !crate::status::local_pass_active(),
            "and the guard cleared it on the way out"
        );

        // A non-local engine must not mark it at all.
        let seen_remote = std::sync::Arc::new(AtomicBool::new(false));
        let mut remote = ScriptedEngine {
            replies: vec!["done\n".to_string()],
            saw_local_pass: Some(std::sync::Arc::clone(&seen_remote)),
            ..ScriptedEngine::default()
        };
        generate_pass(
            &mut remote,
            "[user]\nhi\n",
            None,
            &ctx,
            Box::new(NullSink),
            |_call| Ok(()),
        )
        .expect("a pass");
        assert!(
            !seen_remote.load(Ordering::Relaxed),
            "a provider pass must not claim the local engine is working"
        );
    }

    /// A fan-out-capable stub: reports `max_parallel() > 1` and optionally sleeps
    /// before replying so completion order can be made to differ from call order.
    #[derive(Debug, Default)]
    struct ParallelEngine {
        reply: String,
        delay_ms: u64,
        fail: bool,
        /// Panics inside `generate`, to prove the fan-out's `join` catches it
        /// rather than unwinding the whole turn.
        panics: bool,
        /// Counts generations served, so a test can prove a cached engine is
        /// reused across dispatches instead of rebuilt.
        served: Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>,
    }

    impl Engine for ParallelEngine {
        fn max_parallel(&self) -> usize {
            8
        }
        fn ctx_size(&self) -> i32 {
            100_000
        }
        fn generate(
            &mut self,
            _prompt: crate::engine::Prompt<'_>,
            _opts: &crate::engine::GenerationOptions,
            _interrupt: &dyn Fn() -> bool,
            _greedy: &dyn Fn() -> bool,
            on_event: &mut dyn FnMut(EngineEvent),
        ) -> Result<GenerationStats, EngineError> {
            if let Some(c) = &self.served {
                c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if self.delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(self.delay_ms));
            }
            assert!(!self.panics, "deliberate sub-agent panic");
            if self.fail {
                return Err(EngineError::new("provider exploded".to_string()));
            }
            on_event(EngineEvent::Text(self.reply.clone()));
            Ok(GenerationStats::default())
        }
    }

    /// Installs `n` remote definitions named `a0..`, each with its own key
    /// variable and a pre-seeded fan-out-capable engine.
    fn install_parallel_defs(agent: &mut Agent<'_>, engines: &[(&str, u64, bool)]) -> Vec<String> {
        let mut vars = Vec::new();
        for (i, (reply, delay_ms, fail)) in engines.iter().enumerate() {
            let name = format!("a{i}");
            let var = format!("PLANK_TEST_FANOUT_{i}");
            unsafe { std::env::set_var(&var, "sk-test") };
            let mut def = remote_def(&name, &var);
            // Distinct models so each slot gets its own cache key.
            if let Some(crate::agents::AgentEngine::Provider(p)) = def.engine.as_mut() {
                p.model = format!("model-{i}");
            }
            agent.agents.push(def);
            agent.alt_engines.insert(
                EngineKey::Provider(
                    crate::remote::provider::ProviderKind::Anthropic,
                    "https://example.invalid/v1".to_string(),
                    format!("model-{i}"),
                    8192,
                    var.clone(),
                ),
                Box::new(ParallelEngine {
                    reply: (*reply).to_string(),
                    delay_ms: *delay_ms,
                    fail: *fail,
                    ..ParallelEngine::default()
                }),
            );
            vars.push(var);
        }
        vars
    }

    fn clear_vars(vars: &[String]) {
        for v in vars {
            unsafe { std::env::remove_var(v) };
        }
    }

    #[test]
    fn fanout_returns_results_in_call_order_despite_reversed_completion() {
        let dir = scratch_dir("fanout-order");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        // Slot 0 finishes last, so completion order is the reverse of call order.
        let vars = install_parallel_defs(
            &mut agent,
            &[("slow done\n", 120, false), ("fast done\n", 0, false)],
        );
        let calls = vec![
            agent_call("slow work", Some("a0")),
            agent_call("fast work", Some("a1")),
        ];
        let results = agent.run_agent_fanout(&calls).expect("fanned out");
        assert_eq!(results.len(), 2);
        assert!(results[0].1.contains("slow done"), "{:?}", results[0]);
        assert!(results[1].1.contains("fast done"), "{:?}", results[1]);
        assert_eq!(agent.alt_engines.len(), 2, "both engines back in the cache");
        clear_vars(&vars);
    }

    #[test]
    fn a_panicking_sidechain_is_reported_and_siblings_still_land() {
        let dir = scratch_dir("fanout-panic");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        let vars = install_parallel_defs(&mut agent, &[("", 0, false), ("ok text\n", 0, false)]);
        // Turn slot 0 into a panicking engine, keeping its cache key.
        let key = EngineKey::Provider(
            crate::remote::provider::ProviderKind::Anthropic,
            "https://example.invalid/v1".to_string(),
            "model-0".to_string(),
            8192,
            vars[0].clone(),
        );
        agent.alt_engines.insert(
            key,
            Box::new(ParallelEngine {
                panics: true,
                ..ParallelEngine::default()
            }),
        );
        let calls = vec![
            agent_call("boom", Some("a0")),
            agent_call("fine", Some("a1")),
        ];
        let results = agent.run_agent_fanout(&calls).expect("fanned out");
        assert!(
            results[0].1.contains("panicked"),
            "the panic is reported in its own slot: {:?}",
            results[0]
        );
        assert!(
            results[1].1.contains("ok text"),
            "and the sibling still lands: {:?}",
            results[1]
        );
        assert_eq!(agent.alt_engines.len(), 2, "both engines returned");
        clear_vars(&vars);
    }

    #[test]
    fn a_cached_alt_engine_is_reused_across_dispatches() {
        const KEY: &str = "PLANK_TEST_REUSE_KEY";
        unsafe { std::env::set_var(KEY, "sk-test") };
        let dir = scratch_dir("alt-engine-reuse");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent.agents = vec![remote_def("remote", KEY)];
        let served = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        agent.alt_engines.insert(
            remote_key(KEY),
            Box::new(ParallelEngine {
                reply: "report\n".to_string(),
                served: Some(std::sync::Arc::clone(&served)),
                ..ParallelEngine::default()
            }),
        );

        for _ in 0..3 {
            let out = agent.run_agent_tool(&agent_call("work", Some("remote")));
            assert!(!out.starts_with("Tool error"), "{out}");
        }

        // One cache entry, and the *same* engine served all three — so the
        // context-window probe cannot be repeating per dispatch.
        assert_eq!(agent.alt_engines.len(), 1);
        assert_eq!(
            served.load(std::sync::atomic::Ordering::Relaxed),
            3,
            "the cached engine served every dispatch"
        );
        unsafe { std::env::remove_var(KEY) };
    }

    #[test]
    fn fanout_flushes_one_labelled_pane_block_per_slot_in_call_order() {
        let dir = scratch_dir("fanout-pane");
        let cfg = test_cfg();
        let (tx, rx) = std::sync::mpsc::channel();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent.sub_sink = SubSinkTarget::Events(tx);
        // Slot 0 finishes last, so a naive implementation would flush out of order.
        let vars = install_parallel_defs(
            &mut agent,
            &[("slow text\n", 120, false), ("fast text\n", 0, false)],
        );
        let calls = vec![
            agent_call("slow", Some("a0")),
            agent_call("fast", Some("a1")),
        ];
        agent.run_agent_fanout(&calls).expect("fanned out");

        let events: Vec<crate::worker::UiEvent> = rx.try_iter().collect();
        // One plural signpost, as an ordinary Dim so it also reaches remote
        // clients, which never see the pane-only Sub* variants.
        let signposts: Vec<&String> = events
            .iter()
            .filter_map(|e| match e {
                crate::worker::UiEvent::Dim(t) if t.contains("sub-agent") => Some(t),
                _ => None,
            })
            .collect();
        assert_eq!(signposts.len(), 1, "exactly one signpost: {signposts:?}");
        assert!(
            signposts[0].starts_with("[sub-agents: a0, a1"),
            "plural, in call order: {}",
            signposts[0]
        );

        let labels: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                crate::worker::UiEvent::SubStart(l) => Some(l.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            labels,
            vec!["a0", "a1"],
            "one block per slot, in call order rather than completion order"
        );
        let ends = events
            .iter()
            .filter(|e| matches!(e, crate::worker::UiEvent::SubEnd))
            .count();
        assert_eq!(ends, labels.len(), "every block is closed");
        clear_vars(&vars);
    }

    #[test]
    fn one_failing_sidechain_does_not_abort_its_siblings() {
        let dir = scratch_dir("fanout-fail");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        let vars =
            install_parallel_defs(&mut agent, &[("", 0, true), ("sibling done\n", 0, false)]);
        let calls = vec![
            agent_call("broken", Some("a0")),
            agent_call("working", Some("a1")),
        ];
        let results = agent.run_agent_fanout(&calls).expect("fanned out");
        assert!(results[0].1.starts_with("Tool error"), "{:?}", results[0]);
        assert!(
            results[1].1.contains("sibling done"),
            "the sibling still completed: {:?}",
            results[1]
        );
        assert_eq!(agent.alt_engines.len(), 2);
        clear_vars(&vars);
    }

    #[test]
    fn a_mixed_block_does_not_fan_out() {
        let dir = scratch_dir("fanout-mixed");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        let vars = install_parallel_defs(&mut agent, &[("x\n", 0, false), ("y\n", 0, false)]);
        let calls = vec![
            agent_call("work", Some("a0")),
            ToolCall {
                name: "read".to_string(),
                args: vec![crate::dsml::ToolArg {
                    name: "path".to_string(),
                    value: "Cargo.toml".to_string(),
                    is_string: true,
                }],
            },
        ];
        assert!(
            agent.run_agent_fanout(&calls).is_none(),
            "any non-agent tool in the block forces the serial path"
        );
        assert_eq!(agent.alt_engines.len(), 2, "no engine left removed");
        clear_vars(&vars);
    }

    #[test]
    fn a_local_definition_in_the_block_forces_serial() {
        let dir = scratch_dir("fanout-local");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        let vars = install_parallel_defs(&mut agent, &[("x\n", 0, false)]);
        agent.agents.push(named_def("local", true));
        let calls = vec![agent_call("a", Some("a0")), agent_call("b", Some("local"))];
        assert!(
            agent.run_agent_fanout(&calls).is_none(),
            "a local definition cannot serve a concurrent sidechain"
        );
        assert_eq!(agent.alt_engines.len(), 1, "the remote engine was returned");
        clear_vars(&vars);
    }

    #[test]
    fn max_parallel_one_forces_serial() {
        let dir = scratch_dir("fanout-width1");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        let vars = install_parallel_defs(&mut agent, &[("x\n", 0, false), ("y\n", 0, false)]);
        let mut settings = crate::settings::Settings::default();
        settings.agents.max_parallel = 1;
        crate::settings::install_for_test(settings);
        let calls = vec![agent_call("a", Some("a0")), agent_call("b", Some("a1"))];
        assert!(
            agent.run_agent_fanout(&calls).is_none(),
            "width 1 is the serial path"
        );
        assert_eq!(agent.alt_engines.len(), 2, "no engine left removed");
        crate::settings::install_for_test(crate::settings::Settings::default());
        clear_vars(&vars);
    }

    #[test]
    fn fanout_slots_are_clean_room_and_carry_their_own_persona() {
        let dir = scratch_dir("fanout-cleanroom");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent.session.push(Message::user("parent question"));
        let before = agent.session.transcript.len();
        let vars = install_parallel_defs(&mut agent, &[("r0\n", 0, false), ("r1\n", 0, false)]);
        let calls = vec![
            agent_call("task zero", Some("a0")),
            agent_call("task one", Some("a1")),
        ];
        let results = agent.run_agent_fanout(&calls).expect("fanned out");
        assert_eq!(results.len(), 2);
        assert_eq!(
            agent.session.transcript.len(),
            before,
            "the fan-out never touches the parent transcript"
        );
        clear_vars(&vars);
    }

    #[test]
    fn tool_results_are_formatted_in_call_order() {
        // Encodes today's exact framing: 1-based numbering, `unknown` for an
        // empty name, and a trailing newline added only when one is missing.
        let results = vec![
            ("read".to_string(), "first\n".to_string()),
            ("agent".to_string(), "second".to_string()),
            (String::new(), "third\n".to_string()),
        ];
        assert_eq!(
            format_tool_results(&results),
            "Tool result 1 (read):\nfirst\n\
             Tool result 2 (agent):\nsecond\n\
             Tool result 3 (unknown):\nthird\n"
        );
        assert_eq!(format_tool_results(&[]), "");
    }

    #[test]
    fn remote_definition_runs_on_its_own_engine_and_restores_the_parent() {
        const KEY: &str = "PLANK_TEST_ALT_KEY";
        unsafe { std::env::set_var(KEY, "sk-test") };
        let dir = scratch_dir("alt-engine-runs");
        let cfg = test_cfg();
        // The parent would say "parent reply"; the alt engine says something
        // else, so the report text alone proves which engine served the run.
        let mut agent = test_agent(
            &dir,
            ScriptedEngine {
                replies: vec!["parent reply\n".to_string()],
                ..ScriptedEngine::default()
            },
            &cfg,
        );
        agent.agents = vec![remote_def("remote", KEY)];
        agent.alt_engines.insert(
            remote_key(KEY),
            Box::new(ScriptedEngine {
                replies: vec!["remote report\n".to_string()],
                ..ScriptedEngine::default()
            }),
        );

        let out = agent.run_agent_tool(&agent_call("do a thing", Some("remote")));

        assert!(!out.starts_with("Tool error"), "{out}");
        assert!(
            out.contains("remote report"),
            "the alt engine served the sidechain: {out}"
        );
        assert!(
            !out.contains("parent reply"),
            "the parent engine was not used: {out}"
        );
        assert!(
            agent.alt_engines.contains_key(&remote_key(KEY)),
            "alt engine returned to the cache"
        );
        unsafe { std::env::remove_var(KEY) };
    }

    /// Regression: `/subagent` used to resolve only the definition's *persona*
    /// and then run the fork on the parent engine, silently dropping the engine
    /// override that the `agent` tool honours. A remote-backed definition looked
    /// like it ran remotely while the local model was doing all the work.
    #[test]
    fn slash_subagent_honours_the_definitions_engine() {
        const KEY: &str = "PLANK_TEST_SLASH_ALT_KEY";
        unsafe { std::env::set_var(KEY, "sk-test") };
        let dir = scratch_dir("slash-subagent-alt");
        let cfg = test_cfg();
        let parent_prompts = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut agent = test_agent(
            &dir,
            ScriptedEngine {
                replies: vec!["parent reply\n".to_string()],
                prompts: std::sync::Arc::clone(&parent_prompts),
                ..ScriptedEngine::default()
            },
            &cfg,
        );
        agent.agents = vec![remote_def("remote", KEY)];
        agent.alt_engines.insert(
            remote_key(KEY),
            Box::new(ScriptedEngine {
                replies: vec!["remote report\n".to_string()],
                ..ScriptedEngine::default()
            }),
        );

        agent
            .slash("/subagent:remote do a thing")
            .expect("the command ran");

        let transcript = agent
            .session
            .transcript
            .iter()
            .map(|m| m.text.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            transcript.contains("remote report"),
            "the alt engine served the fork: {transcript}"
        );
        // The parent engine is used exactly once, and not for the sidechain:
        // its single prompt is the follow-up turn, which sees the report and
        // not the framed task the sidechain ran on (that was truncated out).
        let prompts = parent_prompts.lock().unwrap().clone();
        assert_eq!(prompts.len(), 1, "{prompts:?}");
        assert!(
            prompts[0].contains("remote report"),
            "the follow-up turn acts on the report: {}",
            prompts[0]
        );
        assert!(
            !prompts[0].contains("Task: do a thing"),
            "the sidechain itself never reached the parent: {}",
            prompts[0]
        );
        assert!(
            agent.alt_engines.contains_key(&remote_key(KEY)),
            "alt engine returned to the cache"
        );
        unsafe { std::env::remove_var(KEY) };
    }

    /// The main loop acts on the report: `/subagent` runs a parent turn as soon
    /// as the report lands, so delegated work comes back into the conversation
    /// instead of parking in the transcript until the user types again. This is
    /// the same continuation the `agent` tool gets for free by returning its
    /// report as a tool result.
    #[test]
    fn slash_subagent_runs_a_turn_on_the_report() {
        let dir = scratch_dir("slash-subagent-followup");
        let cfg = test_cfg();
        let prompts = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut agent = test_agent(
            &dir,
            ScriptedEngine {
                // Reply 1 is the sidechain's report; reply 2 is the follow-up
                // turn the report triggers.
                replies: vec!["the report\n".to_string(), "acting on it\n".to_string()],
                prompts: std::sync::Arc::clone(&prompts),
                ..ScriptedEngine::default()
            },
            &cfg,
        );

        agent
            .slash("/subagent do a thing")
            .expect("the command ran");

        let seen = prompts.lock().unwrap().clone();
        assert_eq!(seen.len(), 2, "sidechain, then a turn on its report");
        assert!(
            seen[1].contains("the report"),
            "the follow-up prompt carries the report: {}",
            seen[1]
        );
        let last = agent
            .session
            .transcript
            .last()
            .expect("a transcript")
            .text
            .clone();
        assert!(last.contains("acting on it"), "{last}");
    }

    /// A definition whose engine this session cannot provide must fail *before*
    /// the fork, leaving no framed task stranded in the transcript.
    #[test]
    fn slash_subagent_reports_an_unavailable_engine_without_forking() {
        const KEY: &str = "PLANK_TEST_SLASH_MISSING_KEY";
        unsafe { std::env::remove_var(KEY) };
        let dir = scratch_dir("slash-subagent-missing");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent.agents = vec![remote_def("remote", KEY)];
        let before = agent.session.transcript.len();

        agent
            .slash("/subagent:remote do a thing")
            .expect("the command ran");

        assert_eq!(
            agent.session.transcript.len(),
            before,
            "the transcript is untouched"
        );
    }

    #[test]
    fn clean_room_sidechain_hides_the_parent_transcript() {
        const KEY: &str = "PLANK_TEST_CLEANROOM_KEY";
        unsafe { std::env::set_var(KEY, "sk-test") };
        let dir = scratch_dir("alt-engine-cleanroom");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent.session.push(Message::user("parent question"));
        agent.session.push(Message::assistant("parent answer"));
        let before = agent.session.transcript.len();

        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        agent.agents = vec![remote_def("remote", KEY)];
        agent.alt_engines.insert(
            remote_key(KEY),
            Box::new(ScriptedEngine {
                replies: vec!["remote report\n".to_string()],
                prompts: std::sync::Arc::clone(&seen),
                ..ScriptedEngine::default()
            }),
        );

        let out = agent.run_agent_tool(&agent_call("delegated work", Some("remote")));
        assert!(!out.starts_with("Tool error"), "{out}");

        let prompt = seen.lock().unwrap().concat();
        assert!(prompt.contains("delegated work"), "got the task: {prompt}");
        assert!(
            !prompt.contains("parent question"),
            "clean room — no parent context reached the provider: {prompt}"
        );
        // The `agent` tool hands its report back as the tool *result* string, so
        // the sidechain leaves the parent transcript exactly as it found it —
        // unlike `/subagent`, which pushes a framed report message.
        assert_eq!(
            agent.session.transcript.len(),
            before,
            "parent transcript fully restored"
        );
        assert_eq!(
            agent.session.transcript[0].text, "parent question",
            "and in the original order"
        );
        unsafe { std::env::remove_var(KEY) };
    }

    #[test]
    fn the_alt_engine_returns_to_its_cache_when_the_sidechain_fails() {
        const KEY: &str = "PLANK_TEST_FAIL_KEY";
        unsafe { std::env::set_var(KEY, "sk-test") };
        let dir = scratch_dir("alt-engine-fails");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent.session.push(Message::user("parent question"));
        let before = agent.session.transcript.len();
        agent.agents = vec![remote_def("remote", KEY)];
        agent.alt_engines.insert(
            remote_key(KEY),
            Box::new(ScriptedEngine {
                fail_with: Some("provider exploded".to_string()),
                ..ScriptedEngine::default()
            }),
        );

        let out = agent.run_agent_tool(&agent_call("do a thing", Some("remote")));

        assert!(out.starts_with("Tool error"), "failure surfaces: {out}");
        assert!(
            agent.alt_engines.contains_key(&remote_key(KEY)),
            "a leaked swap would leave the session on the wrong engine"
        );
        assert_eq!(
            agent.session.transcript.len(),
            before,
            "the sidechain left no trace on the parent transcript"
        );
        unsafe { std::env::remove_var(KEY) };
    }

    #[test]
    fn a_missing_api_key_fails_before_the_fork() {
        const KEY: &str = "PLANK_TEST_ABSENT_KEY";
        unsafe { std::env::remove_var(KEY) };
        let dir = scratch_dir("alt-engine-nokey");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        agent.agents = vec![remote_def("remote", KEY)];
        let before = agent.session.transcript.len();

        let out = agent.run_agent_tool(&agent_call("work", Some("remote")));

        assert!(out.starts_with("Tool error"), "{out}");
        assert!(out.contains(KEY), "names the missing variable: {out}");
        assert_eq!(
            agent.session.transcript.len(),
            before,
            "no fork started, transcript untouched"
        );
    }

    #[test]
    fn definitions_with_different_key_vars_get_separate_engines() {
        const A: &str = "PLANK_TEST_KEY_A";
        const B: &str = "PLANK_TEST_KEY_B";
        for var in [A, B] {
            unsafe { std::env::set_var(var, "sk-test") };
        }
        let dir = scratch_dir("alt-engine-keys");
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, ScriptedEngine::default(), &cfg);
        // Identical provider, model, base URL and ctx — only the key differs.
        let spec_a = remote_def("work", A).engine.expect("spec a");
        let spec_b = remote_def("home", B).engine.expect("spec b");

        let (key_a, _) = agent.take_alt_engine(&spec_a).expect("engine a");
        let (key_b, _) = agent.take_alt_engine(&spec_b).expect("engine b");

        assert_ne!(key_a, key_b, "distinct cache keys");
        // The key-variable is part of the identity: two definitions differing
        // only in credential must not share a cached engine.
        let env_of = |k: &EngineKey| match k {
            EngineKey::Provider(_, _, _, _, env) => env.clone(),
            EngineKey::Local => panic!("expected a provider key"),
        };
        assert_eq!(env_of(&key_a), A);
        assert_eq!(env_of(&key_b), B);
        for var in [A, B] {
            unsafe { std::env::remove_var(var) };
        }
    }

    #[test]
    fn an_exhausted_round_budget_still_returns_a_report() {
        // An engine that calls a tool on every single pass would otherwise run
        // out of rounds and hand the parent nothing at all.
        let dir = scratch_dir("round-budget");
        let cfg = test_cfg();
        let stanza = concat!(
            "Still working.\n",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\" string=\"true\">echo hi</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        let mut agent = test_agent(
            &dir,
            ScriptedEngine {
                replies: vec![stanza.to_string(); 64],
                ..ScriptedEngine::default()
            },
            &cfg,
        );
        let out = agent.run_agent_tool(&agent_call("never finish", None));
        assert!(
            out.contains("Sub-agent report:"),
            "exhaustion still yields a report: {out}"
        );
        assert!(!out.contains("produced no report"), "{out}");
    }

    /// The roster is no longer in the tool schema at all: it moved to the
    /// session context so that editing a definition rebuilds the small
    /// project-tier cache instead of invalidating the fingerprinted system
    /// prompt. `name` is a plain string in both prompt shapes.
    #[test]
    fn the_tool_schema_carries_no_roster() {
        let defs = vec![named_def("reviewer", true)];
        crate::settings::install_for_test(crate::settings::Settings::default());

        let specs = crate::sysprompt::provider_tool_registry(&[]);
        let agent_spec = specs.iter().find(|s| s.name == "agent").expect("agent");
        assert!(
            agent_spec.parameters["properties"]["name"]
                .get("enum")
                .is_none(),
            "no enum: {}",
            agent_spec.parameters
        );
        let text_prompt = crate::sysprompt::build_system_prompt("", &[], true);
        assert!(
            !text_prompt.contains("reviewer"),
            "a definition name must not reach the system prompt"
        );

        // It reaches the model through the session context instead, and
        // `autoRoute off` withholds it there — the gate moved with the roster.
        let roster = crate::context::agent_roster_context(&defs).expect("roster");
        assert!(roster.contains("reviewer"), "{roster}");
        let mut settings = crate::settings::Settings::default();
        settings.agents.auto_route = false;
        crate::settings::install_for_test(settings);
        assert!(
            crate::context::agent_roster_context(&defs).is_none(),
            "autoRoute off withholds the roster"
        );
        crate::settings::install_for_test(crate::settings::Settings::default());
    }

    #[test]
    fn a_known_auto_agent_runs_without_a_fallback_note() {
        let dir = std::env::temp_dir().join(format!("plank-ui-agent-ok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptedEngine {
            replies: vec!["Reviewed.\n".to_string()],
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.agents = vec![named_def("reviewer", true)];
        let out = agent.run_agent_tool(&agent_call("review it", Some("reviewer")));
        assert!(
            !out.contains("note:"),
            "no note when the name resolves: {out}"
        );
        assert!(out.contains("Sub-agent report:"), "{out}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn agent_tool_streams_its_sub_agent_output_to_the_event_sink() {
        let dir =
            std::env::temp_dir().join(format!("plank-ui-agenttool-sink-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptedEngine {
            replies: vec!["hi there\n".to_string()],
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let (tx, rx) = std::sync::mpsc::channel();
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.sub_sink = SubSinkTarget::Events(tx);
        let call = ToolCall {
            name: "agent".to_string(),
            args: vec![crate::dsml::ToolArg {
                name: "task".to_string(),
                value: "say hi".to_string(),
                is_string: true,
            }],
        };
        let out = agent.run_agent_tool(&call);
        assert!(!out.starts_with("Tool error"), "{out}");
        let got: Vec<crate::worker::UiEvent> = rx.try_iter().collect();
        // The signpost is an ordinary `Dim` (so it reaches the main log and
        // remote clients alike), immediately followed by the pane-only
        // lifecycle event.
        assert!(
            matches!(
                got.first(),
                Some(crate::worker::UiEvent::Dim(d))
                    if d == &crate::tui::subagent_signpost("sub-agent")
            ),
            "first event should be the Dim signpost: {got:?}"
        );
        assert!(
            matches!(got.get(1), Some(crate::worker::UiEvent::SubStart(_))),
            "second event should be SubStart: {got:?}"
        );
        assert!(
            matches!(got.last(), Some(crate::worker::UiEvent::SubEnd)),
            "last event should be SubEnd: {got:?}"
        );
        assert!(
            got.iter()
                .any(|e| matches!(e, crate::worker::UiEvent::Sub(_))),
            "no sub-agent render output was forwarded: {got:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn agent_tool_delegates_and_returns_only_the_report() {
        let dir = std::env::temp_dir().join(format!("plank-ui-agenttool-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Main turn delegates via the `agent` tool.
        let delegate = concat!(
            "Delegating.\n",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"agent\">",
            "<｜DSML｜parameter name=\"task\" string=\"true\">count the tests</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        // The sub-agent runs a bash tool, then reports.
        let sub_tool = concat!(
            "Counting.\n",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\" string=\"true\">echo 42</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        let engine = ScriptedEngine {
            replies: vec![
                delegate.to_string(),
                sub_tool.to_string(),
                "There are 42 tests.\n".to_string(),
                "Done: the sub-agent counted 42.\n".to_string(),
            ],
            ..ScriptedEngine::default()
        };
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.think_mode = crate::engine::ThinkMode::Off;
        let store = SessionStore::open(&dir).unwrap();
        let mut agent = Agent {
            engine: Box::new(engine),
            cfg: &cfg,
            session: Session::new(),
            store,
            pending_aside: None,
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            isolation_seq: 0,
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            payload_restored: false,
            trusted_system_len: 0,
            think: cfg.generation.think_mode,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
            last_ctx_used: 0,
            last_turn_interrupted: false,
            context_content: crate::context::ContextContent::new(),
            skills: Vec::new(),
            templates: Vec::new(),
            agents: Vec::new(),
            checkpoints: crate::checkpoint::CheckpointStore::new(),
            remote: None,
            remote_server: None,
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
            sub_sink: SubSinkTarget::default(),
            fork_kv: Vec::new(),
            alt_engines: std::collections::HashMap::new(),
            local_alt_warmed: false,
            warm_note: None,
        };
        agent.session.push(Message::user("please count the tests"));
        agent.run_turn().unwrap();

        // Find the tool_result carrying the sub-agent's report.
        let tool_result = agent
            .session
            .transcript
            .iter()
            .find(|m| m.text.contains("Tool result 1 (agent):"))
            .expect("agent tool result present");
        assert!(
            tool_result.text.contains("Sub-agent report:"),
            "missing report framing: {}",
            tool_result.text
        );
        assert!(
            tool_result.text.contains("There are 42 tests."),
            "missing report body: {}",
            tool_result.text
        );
        // The sidechain's internal bash call must not leak into the parent.
        assert!(
            !tool_result.text.contains("echo 42"),
            "sidechain leaked: {}",
            tool_result.text
        );
        // The final assistant message concludes the main turn.
        let last = agent.session.transcript.last().unwrap();
        assert!(last.text.contains("Done: the sub-agent counted 42."));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn subagent_fork_truncates_and_carries_only_the_report() {
        let dir = std::env::temp_dir().join(format!("plank-ui-sub-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let stanza = concat!(
            "Counting.\n",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\" string=\"true\">echo 42</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        let engine = ScriptedEngine {
            replies: vec![stanza.to_string(), "There are 42 tests.\n".to_string()],
            ..ScriptedEngine::default()
        };
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.think_mode = crate::engine::ThinkMode::Off;
        let store = SessionStore::open(&dir).unwrap();
        let mut agent = Agent {
            engine: Box::new(engine),
            cfg: &cfg,
            session: Session::new(),
            store,
            pending_aside: None,
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            isolation_seq: 0,
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            payload_restored: false,
            trusted_system_len: 0,
            think: cfg.generation.think_mode,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
            last_ctx_used: 0,
            last_turn_interrupted: false,
            context_content: crate::context::ContextContent::new(),
            skills: Vec::new(),
            templates: Vec::new(),
            agents: Vec::new(),
            checkpoints: crate::checkpoint::CheckpointStore::new(),
            remote: None,
            remote_server: None,
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
            sub_sink: SubSinkTarget::default(),
            fork_kv: Vec::new(),
            alt_engines: std::collections::HashMap::new(),
            local_alt_warmed: false,
            warm_note: None,
        };
        agent.session.push(Message::user("hi"));
        agent.session.push(Message::assistant("hello"));

        let fork_at = agent.begin_subagent_fork(None, "count the tests", true);
        assert_eq!(fork_at, 2);
        agent.run_turn().unwrap();
        // Fork grew: task, assistant(tool call), tool result, final report.
        assert!(agent.session.transcript.len() > 4);

        assert!(agent.finish_subagent_fork(fork_at, "count the tests"));
        // Parent keeps its two messages plus only the framed report.
        assert_eq!(agent.session.transcript.len(), 3);
        let report = &agent.session.transcript[2].text;
        assert!(report.contains("Subagent report:"), "got: {report}");
        assert!(report.contains("There are 42 tests."), "got: {report}");
        assert!(!report.contains("echo 42"), "sidechain leaked: {report}");

        // A fork with no assistant output restores the transcript untouched.
        let fork_at = agent.begin_subagent_fork(None, "noop", true);
        assert!(!agent.finish_subagent_fork(fork_at, "noop"));
        assert_eq!(agent.session.transcript.len(), 3);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The C engine's sync can only *extend* a KV checkpoint — a prompt that
    /// diverges behind the live end (exactly what the fork's truncation
    /// produces) re-prefills the whole context from token zero. The fork must
    /// therefore capture the parent KV at begin and restore it at finish, so
    /// the post-fork turn prefills only the report.
    #[test]
    fn subagent_fork_restores_parent_kv_after_the_sidechain() {
        let dir = std::env::temp_dir().join(format!("plank-ui-kv-fork-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let engine = ScriptedEngine {
            replies: vec!["There are 42 tests.\n".to_string()],
            kv_events: Some(std::sync::Arc::clone(&events)),
            ..ScriptedEngine::default()
        };
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.think_mode = crate::engine::ThinkMode::Off;
        let mut agent = test_agent(&dir, engine, &cfg);
        agent.session.push(Message::user("hi"));
        agent.session.push(Message::assistant("hello"));

        let fork_at = agent.begin_subagent_fork(None, "count the tests", true);
        agent.run_turn().unwrap();
        assert!(agent.finish_subagent_fork(fork_at, "count the tests"));

        assert_eq!(
            events.lock().unwrap().as_slice(),
            ["capture", "generate", "restore:1"]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Nested forks restore LIFO: each finish rolls the KV back to the state
    /// its own begin captured. Also covers the no-report path — the restore
    /// must fire even when the sidechain produced nothing.
    #[test]
    fn nested_subagent_forks_restore_kv_lifo() {
        let dir = std::env::temp_dir().join(format!("plank-ui-kv-nest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let engine = ScriptedEngine {
            kv_events: Some(std::sync::Arc::clone(&events)),
            ..ScriptedEngine::default()
        };
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.think_mode = crate::engine::ThinkMode::Off;
        let mut agent = test_agent(&dir, engine, &cfg);

        let outer = agent.begin_subagent_fork(None, "outer", true);
        let inner = agent.begin_subagent_fork(None, "inner", true);
        assert!(!agent.finish_subagent_fork(inner, "inner"));
        assert!(!agent.finish_subagent_fork(outer, "outer"));

        assert_eq!(
            events.lock().unwrap().as_slice(),
            ["capture", "capture", "restore:2", "restore:1"]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn turn_loop_executes_tool_calls_and_finishes() {
        let dir = std::env::temp_dir().join(format!("plank-ui-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let stanza = concat!(
            "I'll run a command.\n",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\" string=\"true\">echo plank-e2e</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        let engine = ScriptedEngine {
            replies: vec![
                stanza.to_string(),
                "The command printed plank-e2e.\n".to_string(),
            ],
            ..ScriptedEngine::default()
        };
        // Plain (non-thinking) turn: with think on, the prefill opens `<think>`
        // and the stanza would count as an in-think call, which the shipped
        // `engine.thinkingToolCalls` default discards. That path has its own
        // test (`tool_call_inside_think_is_dispatched_and_the_block_is_closed`).
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.think_mode = crate::engine::ThinkMode::Off;
        let store = SessionStore::open(&dir).unwrap();
        let mut agent = Agent {
            engine: Box::new(engine),
            cfg: &cfg,
            session: Session::new(),
            store,
            pending_aside: None,
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            isolation_seq: 0,
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            payload_restored: false,
            trusted_system_len: 0,
            think: cfg.generation.think_mode,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
            last_ctx_used: 0,
            last_turn_interrupted: false,
            context_content: crate::context::ContextContent::new(),
            skills: Vec::new(),
            templates: Vec::new(),
            agents: Vec::new(),
            checkpoints: crate::checkpoint::CheckpointStore::new(),
            remote: None,
            remote_server: None,
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
            sub_sink: SubSinkTarget::default(),
            fork_kv: Vec::new(),
            alt_engines: std::collections::HashMap::new(),
            local_alt_warmed: false,
            warm_note: None,
        };
        agent.session.push(Message::user("run echo"));
        agent.run_turn().unwrap();

        // user, assistant(tool call), user(tool result), assistant(final)
        assert_eq!(agent.session.transcript.len(), 4);
        let tool_result = &agent.session.transcript[2].text;
        assert!(tool_result.contains("plank-e2e"), "got: {tool_result}");
        assert!(tool_result.starts_with("<tool_result>"));
        let last = &agent.session.transcript[3].text;
        assert!(last.contains("The command printed"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn worker_turn_drains_queued_user_between_tool_rounds() {
        let dir = std::env::temp_dir().join(format!("plank-ui-queue-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let stanza = concat!(
            "Checking.\n",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\" string=\"true\">echo hi</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        let engine = ScriptedEngine {
            replies: vec![stanza.to_string(), "Done.\n".to_string()],
            ..ScriptedEngine::default()
        };
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.think_mode = crate::engine::ThinkMode::Off;
        let store = SessionStore::open(&dir).unwrap();
        let mut agent = Agent {
            engine: Box::new(engine),
            cfg: &cfg,
            session: Session::new(),
            store,
            pending_aside: None,
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            isolation_seq: 0,
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
            payload_restored: false,
            trusted_system_len: 0,
            think: cfg.generation.think_mode,
            trace: Trace::open(None).unwrap(),
            color: false,
            show_footer: false,
            editor_owns_footer: false,
            last_ctx_used: 0,
            last_turn_interrupted: false,
            context_content: crate::context::ContextContent::new(),
            skills: Vec::new(),
            templates: Vec::new(),
            agents: Vec::new(),
            checkpoints: crate::checkpoint::CheckpointStore::new(),
            remote: None,
            remote_server: None,
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
            sub_sink: SubSinkTarget::default(),
            fork_kv: Vec::new(),
            alt_engines: std::collections::HashMap::new(),
            local_alt_warmed: false,
            warm_note: None,
        };
        agent.session.push(Message::user("run echo"));

        // A line "typed while busy": queued before the turn, so the first
        // tool round must drain it into the transcript.
        let shared = TurnShared::default();
        shared.push_queued("also check the docs".to_owned());
        let (tx, rx) = std::sync::mpsc::channel();
        agent.worker_turn(&tx, &shared).unwrap();
        drop(tx);

        // user, assistant(tool call), user(tool result), user(queued),
        // assistant(final)
        assert_eq!(agent.session.transcript.len(), 5);
        assert!(
            agent.session.transcript[2]
                .text
                .starts_with("<tool_result>")
        );
        assert_eq!(agent.session.transcript[3].text, "also check the docs");
        assert!(agent.session.transcript[4].text.contains("Done."));
        assert!(shared.take_queued().is_empty());

        // The UI channel saw rendered text, the drain notice, and status
        // snapshots from generation.
        let events: Vec<UiEvent> = rx.try_iter().collect();
        let visible: String = events
            .iter()
            .filter_map(|e| match e {
                UiEvent::Visible(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(visible.contains("Checking"), "got: {visible}");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, UiEvent::Dim(t) if t.contains("queued message joined")))
        );
        assert!(events.iter().any(|e| matches!(e, UiEvent::Status(_))));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A `Pending` wired to a channel the test can read the reply from.
    fn pending(cmd: crate::uiremote::RemoteCmd) -> (crate::uiremote::Pending, Receiver<String>) {
        let (tx, rx) = std::sync::mpsc::channel();
        (crate::uiremote::Pending { cmd, reply: tx }, rx)
    }

    #[test]
    fn injected_events_are_returned_before_polling_the_terminal() {
        let remote = Mutex::new(UiRemote::detached());
        {
            let mut g = remote.lock().unwrap();
            g.injected
                .push_back(Event::Key(KeyEvent::from(KeyCode::Char('@'))));
            g.injected
                .push_back(Event::Key(KeyEvent::from(KeyCode::Down)));
        }
        // No terminal is attached in tests, so a real poll would fail or
        // block; returning the queued events proves they are checked first.
        let a = next_event(Some(&remote), Duration::ZERO).unwrap();
        assert!(matches!(
            a,
            Some(Event::Key(KeyEvent {
                code: KeyCode::Char('@'),
                ..
            }))
        ));
        let b = next_event(Some(&remote), Duration::ZERO).unwrap();
        assert!(matches!(
            b,
            Some(Event::Key(KeyEvent {
                code: KeyCode::Down,
                ..
            }))
        ));
        assert!(remote.lock().unwrap().injected.is_empty());
    }

    #[test]
    fn keypress_answers_at_once_and_service_replies_once_captured_is_set() {
        // This test drives the queueing/reply plumbing (`drain`'s
        // classification, `service`'s reply wiring) with `captured` set by
        // hand. The early-return gate inside `capture()` itself — which is
        // what actually decides *whether* a frame gets captured — is
        // exercised for real by the `capture_*` tests below using a
        // `TestBackend` frame.
        let mut r = UiRemote::detached();
        let (keys, keys_rx) = pending(crate::uiremote::RemoteCmd::Keypress(vec![
            KeyEvent::from(KeyCode::Char('h')),
            KeyEvent::from(KeyCode::Char('i')),
        ]));
        let (snap, snap_rx) = pending(crate::uiremote::RemoteCmd::Snapshot);
        // Stand in for `drain`'s classification (no listener is attached).
        for p in [keys, snap] {
            match p.cmd {
                crate::uiremote::RemoteCmd::Keypress(ref k) => {
                    for key in k.clone() {
                        r.injected.push_back(Event::Key(key));
                    }
                    p.reply.send(crate::uiremote::ok_reply(&[])).unwrap();
                }
                _ => r.deferred.push(p),
            }
        }
        // The keypress is acknowledged immediately...
        assert_eq!(keys_rx.try_recv().unwrap(), r#"{"ok":true}"#);
        // ...but the snapshot is not answered by a frame drawn while keys
        // are still queued: no capture, so `service` has nothing to send.
        assert_eq!(r.injected.len(), 2);
        r.service();
        assert!(snap_rx.try_recv().is_err());

        // Once every key has been consumed, the next frame answers it.
        r.injected.clear();
        r.captured = Some(CapturedFrame {
            ansi: "SCREEN".to_string(),
            tree: "{}".to_string(),
            cols: 80,
            rows: 24,
            cursor: Some((12, 3)),
        });
        r.service();
        let reply = snap_rx.try_recv().unwrap();
        assert!(reply.contains(r#""ansi":"SCREEN""#), "{reply}");
        assert!(reply.contains(r#""cols":80"#), "{reply}");
        assert!(reply.contains(r#""rows":24"#), "{reply}");
        assert!(reply.contains(r#""cursor":[12,3]"#), "{reply}");
        assert!(r.deferred.is_empty());
    }

    /// Draws one `TestBackend` frame and runs `r.capture(frame)` on it,
    /// inside the closure passed to `Terminal::draw` (mirroring how the real
    /// TUI loops call `capture` as the last statement of the draw closure).
    fn capture_one_frame(r: &mut UiRemote) {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let mut term = Terminal::new(TestBackend::new(10, 3)).unwrap();
        term.draw(|f| r.capture(f)).unwrap();
    }

    #[test]
    fn capture_records_nothing_while_keys_are_still_queued() {
        // This is the first of `capture`'s two early returns: a draw that
        // happens mid-key-sequence (`injected` non-empty) must not satisfy a
        // deferred snapshot/uitree request, even though one is pending —
        // otherwise a harness could read a screen that hasn't seen all the
        // keys it just sent.
        let mut r = UiRemote::detached();
        let (snap, _snap_rx) = pending(crate::uiremote::RemoteCmd::Snapshot);
        r.deferred.push(snap);
        r.injected
            .push_back(Event::Key(KeyEvent::from(KeyCode::Char('x'))));

        capture_one_frame(&mut r);

        assert!(
            r.captured.is_none(),
            "capture() must not record a frame while injected keys remain queued"
        );
    }

    #[test]
    fn capture_records_the_frame_once_injected_keys_are_drained() {
        // Second half of the same gate: once every injected key has been
        // consumed (by `next_event`, in the real loop) and a deferred request
        // is still waiting, the very next draw must be captured — this is
        // what lets a harness send `keypress` then `snapshot` with no sleep
        // and get the post-key screen.
        let mut r = UiRemote::detached();
        let (snap, _snap_rx) = pending(crate::uiremote::RemoteCmd::Snapshot);
        r.deferred.push(snap);
        assert!(r.injected.is_empty());

        capture_one_frame(&mut r);

        assert!(
            r.captured.is_some(),
            "capture() must record the frame once injected is empty and a request is deferred"
        );
        let f = r.captured.as_ref().unwrap();
        assert_eq!(f.cols, 10);
        assert_eq!(f.rows, 3);
        assert!(!f.ansi.is_empty());
        assert!(!f.tree.is_empty());
    }

    #[test]
    fn capture_records_nothing_when_nothing_is_deferred() {
        // Second early return: with no deferred request at all, a draw is
        // inert regardless of `injected` — there's nothing for it to answer,
        // and it must not leave a stale `captured` behind for a request that
        // arrives later (which would then race the *next* real capture).
        let mut r = UiRemote::detached();
        assert!(r.deferred.is_empty());

        capture_one_frame(&mut r);

        assert!(
            r.captured.is_none(),
            "capture() must not record a frame when nothing is deferred"
        );
    }

    #[test]
    fn uitree_reply_carries_the_frame_tree_as_a_json_object() {
        let mut r = UiRemote::detached();
        let (p, rx) = pending(crate::uiremote::RemoteCmd::Uitree);
        r.deferred.push(p);
        r.captured = Some(CapturedFrame {
            ansi: String::new(),
            tree: r#"{"name":"root"}"#.to_string(),
            cols: 10,
            rows: 4,
            cursor: None,
        });
        r.service();
        let reply = rx.try_recv().unwrap();
        // Spliced, not escaped: a harness reads reply["tree"]["name"] with a
        // single decode, as the docs promise.
        assert_eq!(reply, r#"{"ok":true,"tree":{"name":"root"}}"#);
    }

    #[test]
    fn snapshot_reports_a_hidden_cursor_as_null() {
        let mut r = UiRemote::detached();
        let (p, rx) = pending(crate::uiremote::RemoteCmd::Snapshot);
        r.deferred.push(p);
        r.captured = Some(CapturedFrame {
            ansi: "x".to_string(),
            tree: "{}".to_string(),
            cols: 10,
            rows: 4,
            cursor: None,
        });
        r.service();
        let reply = rx.try_recv().unwrap();
        // Null, not (0,0) — a harness must be able to tell "hidden" from
        // "parked in the top-left corner".
        assert!(reply.contains(r#""cursor":null"#), "{reply}");
    }

    #[test]
    fn abandoning_answers_deferred_requests_instead_of_stranding_them() {
        let mut r = UiRemote::detached();
        let (p, rx) = pending(crate::uiremote::RemoteCmd::Snapshot);
        r.deferred.push(p);
        r.abandon();
        let reply = rx.try_recv().expect("a reply, not a 10s timeout");
        assert!(reply.contains(r#""ok":false"#), "{reply}");
        assert!(reply.contains("ui exiting"), "{reply}");
        assert!(r.deferred.is_empty());
    }

    #[test]
    fn color_to_rgb_maps_variants() {
        use ratatui::style::Color;
        // Passthrough
        assert_eq!(color_to_rgb(Color::Rgb(10, 20, 30)), (10, 20, 30));
        // Named
        assert_eq!(color_to_rgb(Color::Black), (0, 0, 0));
        assert_eq!(color_to_rgb(Color::White), (255, 255, 255));
        assert_eq!(color_to_rgb(Color::Red), (205, 0, 0));
        // Reset -> neutral gray
        assert_eq!(color_to_rgb(Color::Reset), (192, 192, 192));
        // Indexed: 16 base colors, cube, grayscale ramp
        assert_eq!(color_to_rgb(Color::Indexed(0)), (0, 0, 0));
        assert_eq!(color_to_rgb(Color::Indexed(15)), (255, 255, 255));
        assert_eq!(color_to_rgb(Color::Indexed(16)), (0, 0, 0)); // cube origin
        assert_eq!(color_to_rgb(Color::Indexed(231)), (255, 255, 255)); // cube max
        assert_eq!(color_to_rgb(Color::Indexed(232)), (8, 8, 8)); // ramp start
        assert_eq!(color_to_rgb(Color::Indexed(255)), (238, 238, 238)); // ramp end
    }

    #[test]
    fn frame_to_image_reproduces_dimensions_glyphs_and_backgrounds() {
        use ratatui::buffer::Buffer;
        use ratatui::layout::Rect;
        use ratatui::style::{Color, Modifier};

        let mut buf = Buffer::empty(Rect::new(0, 0, 3, 2));
        // Row 0, col 0: a green glyph -> foreground colour.
        buf[(0, 0)].set_symbol("h").set_fg(Color::Green);
        // Row 0, col 1: a blank cell with a blue background -> background colour
        // (this is the case #55 fixes; the old transcription dropped it to black).
        buf[(1, 0)].set_symbol(" ").set_bg(Color::Blue);
        // Row 1, col 0: reversed blank cell -> fg/bg swap, so the (blank) cell
        // paints in its foreground red rather than its background.
        buf[(0, 1)].set_symbol(" ").set_fg(Color::Red);
        buf[(0, 1)].modifier = Modifier::REVERSED;

        let img = frame_to_image(&buf);

        // width = columns, height = 2 * rows (half-block packing).
        assert_eq!(img.dimensions(), (3, 4));
        // Green glyph, both stacked pixels of the cell.
        assert_eq!(img.get_pixel(0, 0).0, [0, 205, 0, 255]);
        assert_eq!(img.get_pixel(0, 1).0, [0, 205, 0, 255]);
        // Blank cell keeps its blue background.
        assert_eq!(img.get_pixel(1, 0).0, [0, 0, 238, 255]);
        // Reversed blank cell paints in the swapped-in foreground red.
        assert_eq!(img.get_pixel(0, 2).0, [205, 0, 0, 255]);
        // An untouched blank cell (Reset bg) stays black.
        assert_eq!(img.get_pixel(2, 0).0, [0, 0, 0, 255]);
    }
}
