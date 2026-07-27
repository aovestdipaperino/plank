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

/// Builds the model-visible payload for a failed generation pass, matching
/// the C worker loop: a preflight failure is fed back verbatim, a DSML parse
/// failure gets the C's `invalid DSML tool call: ` prefix plus the syntax
/// reminder so the model can correct its markup.
fn tool_error_payload(preflight: bool, err: &str) -> String {
    if preflight {
        format!("Tool error: {err}\n")
    } else {
        format!(
            "Tool error: invalid DSML tool call: {err}\n{}",
            sysprompt::dsml_syntax_reminder()
        )
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
    let ctx = ToolContext::new(ctx.cwd.clone());
    move |call| crate::tools::edit::preflight_edit_old(&ctx, call)
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
    /// Named in-session rollback points (`/checkpoint`, `/rollback`); dropped
    /// when the session is replaced.
    checkpoints: crate::checkpoint::CheckpointStore,
    /// Live remote-control bridge (issue #25): the shared [`BroadcastBus`] that
    /// this agent's turn output mirrors into, plus the shared [`TurnShared`] that
    /// remote `prompt`/`btw`/`interrupt` frames drive. `None` when `--control`
    /// was not given, in which case the turn loops behave exactly as before.
    remote: Option<Arc<RemoteState>>,
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
#[derive(Debug, Clone, Copy, Default)]
struct SessionStats {
    /// Tokens the model ingested (prompt / prefill), summed over all passes.
    input_tokens: u64,
    /// Tokens the model generated, summed over all passes.
    output_tokens: u64,
}

/// Default number of user turns replayed by `/history`.
const HISTORY_DEFAULT_TURNS: usize = 3;
/// Sessions shown by the /resume picker.
const RESUME_LIST_LIMIT: usize = 10;
/// Maximum user turns `/history` accepts.
const HISTORY_MAX_TURNS: usize = 200;
/// Name of the auto-checkpoint saved before a `/rollback`, so a rollback is
/// itself undoable via `/rollback pre-rollback`.
const PRE_ROLLBACK_CHECKPOINT: &str = "pre-rollback";

impl Agent<'_> {
    /// Builds owned structured-turn buffers for a provider engine (§4.4). The
    /// provider gets a machine-readable tool registry and its own system prompt
    /// (never the DS4 byte-parity prompt), plus the flat render as a fallback.
    fn build_structured(&self, rendered: &str) -> StructuredBufs {
        StructuredBufs {
            system: sysprompt::provider_system_prompt(&self.cfg.system),
            messages: session_to_messages(&self.session),
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
        let mut stream = StreamRenderer::new(sink);
        stream.set_show_tool_calls(crate::settings::active().ui.show_tool_calls);
        stream.set_show_thinking(crate::settings::active().ui.show_thinking);
        stream.set_thinking_tool_calls(crate::settings::active().engine.thinking_tool_calls);
        stream.set_preflight(edit_preflight(&self.tool_ctx));
        // With thinking enabled, the *local* chat template opens `<think>` in
        // the prefill prefix, so generation streams thinking content first
        // without a leading tag; start the renderer inside the think block so it
        // renders gray until `</think>`. Provider engines are excluded: their
        // translator emits explicit `<think>`/`</think>` tags, so pre-opening
        // here would mis-color any output not preceded by a reasoning delta.
        if !matches!(
            self.cfg.generation.think_mode,
            crate::engine::ThinkMode::Off
        ) && !self.engine.wants_structured()
        {
            stream.begin_in_think();
        }
        let mut assistant_text = String::new();
        let ctx_size = self.engine.ctx_size();
        let power = self.power_percent;
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
        use std::fmt::Write as _;
        if !calls.is_empty() {
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
            crate::status::set_flash_tip_for(format!("🔧 {names}"), crate::status::TOOL_FLASH_MS);
        }
        if !calls.iter().any(|c| c.name == "agent") {
            return dispatch_all(calls, &mut self.tool_ctx);
        }
        if calls.is_empty() {
            return "Tool error: empty tool call block\n".to_string();
        }
        // Mirror dispatch_all: clear any undrained previews so cards never leak.
        self.tool_ctx.edit_previews.clear();
        let mut all = String::new();
        for (i, call) in calls.iter().enumerate() {
            let out = if call.name == "agent" {
                self.run_agent_tool(call)
            } else {
                dispatch(call, &mut self.tool_ctx).output
            };
            let name = if call.name.is_empty() {
                "unknown"
            } else {
                call.name.as_str()
            };
            let _ = writeln!(all, "Tool result {} ({}):", i + 1, name);
            all.push_str(&out);
            if !out.is_empty() && !out.ends_with('\n') {
                all.push('\n');
            }
        }
        all
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
        if !self.tool_ctx.tools.agent {
            return "Tool error: the agent tool is not enabled\n".to_string();
        }
        if task.is_empty() {
            return "Tool error: agent requires a non-empty 'task' to delegate\n".to_string();
        }
        if self.tool_ctx.subagent_depth >= crate::tools::SUBAGENT_DEPTH_CAP {
            return "Tool error: sub-agent nesting limit reached; complete this work directly\n"
                .to_string();
        }
        let name = call.arg_value("name").unwrap_or("").trim();
        let instructions = if name.is_empty() {
            None
        } else {
            match self.agents.iter().find(|d| d.name == name) {
                Some(d) => Some(d.body.clone()),
                None => return format!("Tool error: unknown agent '{name}'\n"),
            }
        };
        let fork_at = self.begin_subagent_fork(instructions.as_deref(), &task);
        self.tool_ctx.subagent_depth += 1;
        let result = self.run_subagent_loop();
        self.tool_ctx.subagent_depth -= 1;
        // Extract the sidechain's final report before truncating it back out.
        let report = self.session.transcript[fork_at..]
            .iter()
            .rev()
            .find(|m| {
                matches!(m.role, crate::session::Role::Assistant) && !m.text.trim().is_empty()
            })
            .map(|m| m.text.trim().to_owned());
        self.session.transcript.truncate(fork_at);
        match result {
            Err(e) => format!("Tool error: sub-agent failed: {e}\n"),
            Ok(()) => match report {
                Some(r) => format!("Sub-agent report:\n{r}\n"),
                None => "Tool error: sub-agent produced no report\n".to_string(),
            },
        }
    }

    /// Headless generate→dispatch loop for a sub-agent sidechain (issue #50):
    /// like the main turn loop but with no on-screen streaming, footer, hooks,
    /// or compaction. Bounded by a round budget so a stuck sub-agent cannot loop
    /// forever. Nested `agent` calls route through [`run_tool_calls`], so the
    /// [`SUBAGENT_DEPTH_CAP`](crate::tools::SUBAGENT_DEPTH_CAP) guard applies.
    fn run_subagent_loop(&mut self) -> Result<(), String> {
        const MAX_ROUNDS: usize = 40;
        let turn_start = Instant::now();
        for _ in 0..MAX_ROUNDS {
            let prompt_text = render_transcript(&self.session, &self.system);
            let (calls, assistant_text, err) = self.generate_quiet(&prompt_text, turn_start)?;
            self.session.push(Message::assistant(assistant_text));
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
        Err("sub-agent exceeded its round budget".to_string())
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
        let mut stream = StreamRenderer::new(NullSink);
        stream.set_preflight(edit_preflight(&self.tool_ctx));
        stream.set_thinking_tool_calls(crate::settings::active().engine.thinking_tool_calls);
        if !matches!(
            self.cfg.generation.think_mode,
            crate::engine::ThinkMode::Off
        ) && !self.engine.wants_structured()
        {
            stream.begin_in_think();
        }
        let mut assistant_text = String::new();
        let preflight_stop = AtomicBool::new(false);
        let greedy = AtomicBool::new(false);
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
        self.record_usage(&stats);
        self.last_ctx_used = stats.ctx_used;
        let preflight_error = stream.preflight_error().map(str::to_owned);
        if stats.interrupted && preflight_error.is_none() {
            crate::interrupt::clear();
            return Err("interrupted".to_string());
        }
        let finished = stream.finished();
        let ended_in_think = finished.ended_in_think;
        if let Some(err) = preflight_error.as_deref().or(finished.error) {
            let payload = tool_error_payload(preflight_error.is_some(), err);
            close_open_think(&mut assistant_text, ended_in_think);
            return Ok((Vec::new(), assistant_text, Some(payload)));
        }
        let calls = finished.calls.to_vec();
        close_open_think(&mut assistant_text, ended_in_think && !calls.is_empty());
        Ok((calls, assistant_text, None))
    }

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
        self.maybe_compact()?;
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
                let payload = tool_error_payload(preflight_error.is_some(), err);
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
    fn maybe_compact(&mut self) -> Result<(), String> {
        let rendered = render_transcript(&self.session, &self.system);
        let used = self.engine.count_tokens(&rendered);
        if !compact::should_compact(self.engine.ctx_size(), used) {
            return Ok(());
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
            return Ok(());
        }
        self.compact("low context")
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

    /// Performs the compaction exchange and rebuilds the transcript as
    /// summary + recent verbatim tail.
    fn compact(&mut self, reason: &str) -> Result<(), String> {
        print!("{}", compact::banner(reason, self.color));
        // PreCompact: `manual` for a user-driven `/compact`, `auto` otherwise.
        // Injected context is pinned as a user message so it survives the
        // rebuild in the verbatim tail.
        let trigger = if reason == "user request" {
            "manual"
        } else {
            "auto"
        };
        if !self.tool_ctx.hooks.pre_compact.is_empty() {
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
                println!("{w}");
            }
            if let Some(ctx) = out.context {
                self.session
                    .push(Message::user(format!("<hook_context>{ctx}</hook_context>")));
            }
        }
        let mut prompt_text = render_transcript(&self.session, &self.system);
        {
            use std::fmt::Write as _;
            let _ = write!(prompt_text, "[user]\n{}\n", compact::make_prompt(reason));
        }
        let mut summary = String::new();
        self.engine
            .generate(
                crate::engine::Prompt::Flat(&prompt_text),
                &self.cfg.generation,
                &|| false,
                &|| false,
                &mut |ev| {
                    if let EngineEvent::Text(t) = ev {
                        summary.push_str(&t);
                    }
                },
            )
            .map_err(|e| e.to_string())?;
        if self.color {
            print!("\x1b[0m");
        }

        self.rebuild_after_compact(&summary);
        // PostCompact: carries the extracted durable summary; injected context
        // is appended after the rebuilt transcript.
        if !self.tool_ctx.hooks.post_compact.is_empty() {
            let extracted = compact::extract_summary(&summary);
            let input = crate::hooks::lifecycle_event_input(
                "PostCompact",
                &[("trigger", trigger), ("summary", &extracted)],
                &self.tool_ctx.cwd,
            );
            let out = crate::hooks::run_event_ctx(
                &self.tool_ctx.hooks.post_compact,
                "",
                &input,
                &self.tool_ctx.cwd,
            );
            for w in out.warnings.into_iter().chain(out.system_messages) {
                println!("{w}");
            }
            if let Some(ctx) = out.context {
                self.session
                    .push(Message::user(format!("<hook_context>{ctx}</hook_context>")));
            }
        }
        println!("{}", self.debug_line("context compacted"));
        Ok(())
    }

    /// Folds a completed pass's provider usage into the session tally. A no-op
    /// for local engines (`stats.usage` is `None`), so `/usage` stays empty
    /// unless an online provider is driving the turns.
    fn record_usage(&mut self, stats: &crate::engine::GenerationStats) {
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
        self.stats.input_tokens += u64::try_from(input.max(0)).unwrap_or(0);
        self.stats.output_tokens += u64::try_from(output.max(0)).unwrap_or(0);

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
        if let Err(e) = self.tui_turn(terminal, log, view, input, btw, arcade) {
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
            // here the sky is a single still frame and pelota declines rather
            // than growing a second game loop against raw stdout.
            "/stars" if crate::arcade::enabled() => print!(
                "{}",
                crate::arcade::still_sky(arcade_seed(), 78, 20, self.color)
            ),
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
            "/save" => match self.store.save(&mut self.session) {
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
                    println!("power limit set to {power}%");
                }
                None => println!("usage: /power <1..100>"),
            },
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
            "/compact" => self.compact("user request")?,
            "/skills" => print!("{}", crate::skills::render_list(&self.skills)),
            "/templates" => print!("{}", crate::templates::render_list(&self.templates)),
            "/tasks" => print!("{}", self.session.tasks.render_list()),
            "/agent" => print!("{}", crate::agents::render_list(&self.agents)),
            "/hooks" => print!("{}", crate::hooks::render_list(&self.tool_ctx.hooks)),
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
            "/repro" => match self.write_repro(arg) {
                Ok(path) => println!(
                    "{}",
                    self.debug_line(&format!("[repro written to {}]", path.display()))
                ),
                Err(e) => println!("repro failed: {e}"),
            },
            "/subagent" => {
                let (def, task) = crate::agents::resolve(&self.agents, arg);
                let (instructions, task, started) = match def {
                    Some(d) => (
                        Some(d.body.clone()),
                        task.to_string(),
                        format!("[subagent started: {}]", d.name),
                    ),
                    None => (None, task.to_string(), "[subagent started]".to_string()),
                };
                if task.is_empty() {
                    println!("usage: /subagent [<name>] <task>");
                } else {
                    println!("{}", self.debug_line(&started));
                    let fork_at = self.begin_subagent_fork(instructions.as_deref(), &task);
                    // Restore the transcript even when the turn errored.
                    let turn = self.run_turn();
                    let reported = self.finish_subagent_fork(fork_at, &task);
                    turn?;
                    let trailer = if reported {
                        "[subagent report added to the conversation]"
                    } else {
                        "[subagent produced no report — nothing added]"
                    };
                    println!("{}", self.debug_line(trailer));
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
        self.session = session;
        self.last_ctx_used = 0;
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
        let pre_open_think = !matches!(
            self.cfg.generation.think_mode,
            crate::engine::ThinkMode::Off
        ) && !self.engine.wants_structured();

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
                    stream.set_show_tool_calls(show_tool_calls);
                    stream.set_show_thinking(show_thinking);
                    stream.set_thinking_tool_calls(thinking_tool_calls);
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
            Ok(()) => Some("restored KV payload; resume skips re-prefill".to_owned()),
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
            self.store
                .save(&mut self.session)
                .map_err(|e| e.to_string())?;
            msg.push_str(" (saved)");
        }
        Ok(msg)
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
        let id = self.store.save(&mut self.session).ok()?;
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
        let s = self.stats;
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
    fn begin_subagent_fork(&mut self, instructions: Option<&str>, task: &str) -> usize {
        let fork_at = self.session.transcript.len();
        self.session.push(Message::user(crate::agents::task_message(
            instructions,
            task,
        )));
        fork_at
    }

    /// Ends a `/subagent` fork: truncates the sidechain back out of the
    /// transcript and pushes only the framed final report. Returns false when
    /// the sidechain produced no report (e.g. interrupted before any output);
    /// the transcript is still restored.
    fn finish_subagent_fork(&mut self, fork_at: usize, task: &str) -> bool {
        let report = self.session.transcript[fork_at..]
            .iter()
            .rev()
            .find(|m| {
                matches!(m.role, crate::session::Role::Assistant) && !m.text.trim().is_empty()
            })
            .map(|m| m.text.clone());
        self.session.transcript.truncate(fork_at);
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
            worker: None,
            mcp_extra: Vec::new(),
        }
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

    /// Opens, retargets, or closes the popup to match the current input text.
    ///
    /// Called after every key. Starts the index worker lazily on the first `@`
    /// so a session that never completes never shells out to git.
    fn sync_popup(&mut self) {
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

    /// Offers `key` to an open popup, the single entry point both TUI key
    /// loops share so they cannot drift.
    ///
    /// Returns true when the popup consumed the key and the caller must skip
    /// its own binding for it.
    fn popup_key(&mut self, key: KeyEvent) -> bool {
        use crate::complete::PopupAction;
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

    /// Writes the startup banner (logo art, version/context line, hints) into
    /// `log`. Used both at launch and after `/clear` and `/new`, so a cleared
    /// screen looks exactly like a fresh start.
    fn tui_write_banner(&self, log: &mut OutputLog) {
        for line in tui::ansi_to_lines(&crate::logo::art(crate::logo::DEFAULT_WIDTH * 144 / 100)) {
            log.push_spans(line.spans);
        }
        log.push_plain(format!(
            "plank {} 🪵 Agent, context {} tokens",
            crate::logo::version_label(),
            status::format_ctx_size(self.engine.ctx_size())
        ));
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
        // The `/btw` side panel, owned here so it outlives any single turn:
        // once opened it stays until the user presses Esc, even after the main
        // task finishes and control returns to this idle loop.
        let mut btw_panel: BtwPanel = None;
        // An open easter egg (`/stars`, `/pelota`), same modal contract as the
        // `/config` form: while it is Some it owns the screen and every key.
        // Owned here, like the `/btw` panel, so it survives the transition into
        // and out of a turn — that is what lets it keep running while the model
        // works. `arcade_last` measures the real frame delta to feed its
        // simulation; the poll timeout alone would stall it whenever keys
        // arrive.
        let mut arcade = crate::arcade::Arcade::new();
        let mut arcade_last = Instant::now();
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
            )?;
        }

        // Endpoints of a mouse drag selection over the output area, in content
        // space: `(column, absolute-wrapped-row)`. Anchoring the row to content
        // (not the screen) lets the selection survive scrolling. Copied to the
        // clipboard on button release.
        let mut selection: Option<tui::ContentSelection> = None;
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
            let completed = terminal
                .draw(|f| {
                    // A `/btw` panel left open from an earlier turn keeps the
                    // split view even while idle; text selection falls back to
                    // the single-column path (no panel).
                    if let Some((btw_log, btw_view)) = btw_panel.as_mut() {
                        tui::draw_btw_split(
                            f,
                            &log,
                            btw_log,
                            btw_view,
                            Some(input.buf.text()),
                            input.cursor_char(),
                            &status,
                            &mut view,
                            &task_view,
                        );
                    } else {
                        tui::draw(
                            f,
                            &log,
                            Some(input.buf.text()),
                            input.cursor_char(),
                            &status,
                            &mut view,
                            selection,
                            &task_view,
                        );
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
                        )?;
                    }
                }
                continue;
            };
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
                        view.follow = false;
                        view.top = view.top.saturating_sub(3);
                    }
                    MouseEventKind::ScrollDown => {
                        // Clamped by draw, which re-enters follow mode at the bottom.
                        view.top = view.top.saturating_add(3);
                    }
                    MouseEventKind::Down(MouseButton::Left) => {
                        // A click on the jump-to-bottom hint resumes follow mode
                        // (same as End) instead of starting a text selection.
                        if view.jump_hint_rect.is_some_and(|r| {
                            r.contains(ratatui::layout::Position::new(m.column, m.row))
                        }) {
                            view.follow = true;
                            selection = None;
                        } else {
                            let row = view.top.saturating_add(usize::from(m.row));
                            selection = Some(((m.column, row), (m.column, row)));
                        }
                    }
                    MouseEventKind::Drag(MouseButton::Left) => {
                        if let Some((_, end)) = &mut selection {
                            *end = (m.column, view.top.saturating_add(usize::from(m.row)));
                        }
                    }
                    MouseEventKind::Up(MouseButton::Left) => {
                        let size = terminal.size().unwrap_or_default();
                        if let Some(sel) = selection.filter(|(a, b)| a != b) {
                            // A drag: extract from the content model (not the
                            // screen buffer) so a selection larger than the
                            // viewport still copies in full.
                            let text = tui::selection_text_content(&log, size.width, sel);
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
                                u16::try_from(row.saturating_sub(view.top)).unwrap_or(u16::MAX);
                            if screen_row < out_h
                                && let Some(code) =
                                    log.code_copy_at(size.width, view.top, col, screen_row)
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
                                    crate::settings::reinstall(settings);
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
            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
            // Alt (Option on macOS) or Ctrl turns arrows and Backspace/Delete
            // into word-wise operations.
            let word_mod = ctrl || key.modifiers.contains(KeyModifiers::ALT);
            match key.code {
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
                // Ctrl-G hands the half-typed prompt to `$EDITOR` — the escape
                // hatch for prompts too long to comfortably edit inline.
                KeyCode::Char('g') if ctrl => {
                    let current = input.buf.text().to_owned();
                    let edited = with_tui_suspended(terminal, || {
                        crate::editor::edit_text_externally(&current)
                    });
                    match edited {
                        Ok(Some(text)) => {
                            input.hist_idx = None;
                            input.buf.set_text(text);
                        }
                        // Non-zero exit or an emptied file: keep what was typed.
                        Ok(None) => {}
                        Err(e) => log.push_dim(format!("[editor failed: {e}]")),
                    }
                    input.sync_popup();
                }
                KeyCode::Enter => {
                    let line = input.buf.text().trim().to_owned();
                    input.buf.clear();
                    input.popup = None;
                    input.hist_idx = None;
                    view.follow = true;
                    if line.is_empty() && attachments.is_empty() {
                        continue;
                    }
                    if !line.is_empty() && !line.contains('\n') {
                        input.history.add(&line);
                        input.history.save(&hist_path).ok();
                    }
                    if let Some(cmd) = line.strip_prefix('!') {
                        // ! prefix is for user-only shell execution — output goes to TUI log
                        // but NOT into the session transcript. This is intentional and matches
                        // Claude Code's behavior. See issue #20 for discussion.
                        let cmd = cmd.trim().to_owned();
                        if cmd.is_empty() {
                            log.push_dim("usage: !<shell command>");
                            continue;
                        }
                        log.push_spans(tui::user_echo_spans(&line));
                        Self::tui_bang(
                            &self.tool_ctx.cwd.clone(),
                            &cmd,
                            &mut log,
                            terminal,
                            &mut view,
                        );
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
                        )?;
                    }
                }
                _ => {}
            }
            // Retarget (or close) the popup after every edit and cursor move.
            input.sync_popup();
        }
        input.history.save(&hist_path).ok();
        Ok(crt_frame)
    }

    /// Runs a `!` immediate shell command: output lands only in the TUI log,
    /// never in the conversation, and the model is not consulted. The frame
    /// keeps redrawing while the command runs so Esc/Ctrl-C can kill it.
    ///
    /// # Behavior is intentional
    ///
    /// The `!` prefix is for **user-only** shell execution — output is displayed
    /// but NOT fed to the model. This matches Claude Code's behavior and is
    /// by design, not a bug.
    ///
    /// See: <https://github.com/aovestdipaperino/plank/issues/20>
    ///
    /// ## Why output should not go to the model
    ///
    /// - `!` commands are for the human operator's convenience (checking status,
    ///   running diagnostics, manual file operations)
    /// - The model should not incorporate this output into its reasoning unless
    ///   the user explicitly shares it
    /// - If you want the model to see command output, use a regular turn with
    ///   the `bash` tool instead
    fn tui_bang(
        cwd: &std::path::Path,
        cmd: &str,
        log: &mut OutputLog,
        terminal: &mut ratatui::DefaultTerminal,
        view: &mut tui::OutputView,
    ) {
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
                        0,
                        &status,
                        view,
                        None,
                        &tui::TaskView::default(),
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
        match result {
            Ok(out) => {
                if out.interrupted {
                    log.push_dim("[interrupted]");
                } else if out.exit_code != 0 {
                    log.push_dim(format!("[exit code: {}]", out.exit_code));
                }
            }
            Err(e) => log.push_dim(format!("!{cmd}: {e}")),
        }
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
        let fp1 = crate::kvtier::system_fingerprint(&self.engine.model_name(), &self.system);
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
    fn gc_kv_tiers(&self, tiers: &[crate::kvtier::TierSpec]) {
        if let Some(t) = tiers
            .iter()
            .find(|t| t.kind == crate::kvtier::TierKind::System)
        {
            let _removed = self.store.gc_system_checkpoints(&t.fingerprint);
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
    fn tui_warm(&mut self, terminal: &mut ratatui::DefaultTerminal) -> Result<(), String> {
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
        self.gc_kv_tiers(&tiers);
        Ok(())
    }

    /// Warms the KV cache for non-TUI runs, announcing a rebuild on stderr.
    fn warm_plain(&mut self) -> Result<(), String> {
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
        self.gc_kv_tiers(&tiers);
        Ok(())
    }

    fn idle_status_text(&mut self) -> String {
        let rendered = render_transcript(&self.session, &self.system);
        let st = Status {
            state: WorkerState::Idle,
            ctx_used: self.engine.count_tokens(&rendered),
            ctx_size: self.engine.ctx_size(),
            power_percent: self.power_percent,
            ..Status::default()
        };
        status::build_status_text(&st, false, true)
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

    /// Runs one worker turn while mirroring every render/output event onto the
    /// remote [`BroadcastBus`], driving it from the shared [`TurnShared`] so
    /// remote `interrupt` / `btw` / queued frames steer this turn directly.
    /// Falls back to the plain [`run_turn`](Self::run_turn) when no remote
    /// bridge is present. Used by the headless remote-serve loop; the TUI path
    /// mirrors inline in [`busy_ui_loop`] so the local screen stays live too.
    fn run_turn_mirrored(&mut self) -> Result<(), String> {
        let Some(remote) = self.remote.clone() else {
            return self.run_turn();
        };
        self.tool_ctx.skill_invocations = 0;
        // Outermost per-user-turn boundary for the headless remote path: when
        // a remote bridge is present this function (not `worker_turn`, which
        // it drives on a scoped thread) is the single call per queued line in
        // `pump_remote`, so the stamp/notify belongs here, not in the delegate.
        let turn_started = Instant::now();
        let bus = Arc::clone(&remote.bus);
        let shared = &remote.shared;
        let (tx, rx) = std::sync::mpsc::channel::<UiEvent>();
        let result = std::thread::scope(|s| {
            let worker = s.spawn(|| self.worker_turn(&tx, shared));
            loop {
                while let Ok(ev) = rx.try_recv() {
                    bus.broadcast(ev);
                }
                if worker.is_finished() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            // Drain anything sent between the last poll and the worker returning.
            while let Ok(ev) = rx.try_recv() {
                bus.broadcast(ev);
            }
            worker
                .join()
                .map_err(|_| "worker thread panicked".to_owned())?
        });
        if result.is_ok() {
            if crate::notify::should_notify_complete(
                turn_started.elapsed(),
                crate::settings::active().ui.notify_after_secs,
            ) {
                self.notify_task_complete();
            }
            // Turn over: the front end is back at the prompt.
            crate::title::set(crate::title::State::Idle);
            crate::warp::emit("stop", &self.session.id);
        }
        result
    }

    /// Drains the remote controller's pending input once: a mirrored turn for
    /// each queued `prompt`, and `/`-prefixed lines (remote `command` frames)
    /// routed through the shared slash dispatcher exactly like the local REPL.
    /// Returns whether any input was processed. No-op without a remote bridge.
    fn pump_remote(&mut self) -> Result<bool, String> {
        let Some(remote) = self.remote.clone() else {
            return Ok(false);
        };
        let queued = remote.shared.take_queued();
        if queued.is_empty() {
            return Ok(false);
        }
        for line in queued {
            let line = line.trim().to_owned();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('/') {
                // Remote `command` routing: the same slash path the local REPL
                // uses. Its textual report goes to stdout (the headless sink).
                let _ = self.slash(&line)?;
                continue;
            }
            remote.bus.broadcast(UiEvent::UserEcho(line.clone()));
            self.session.push(Message::user(line));
            self.run_turn_mirrored()?;
        }
        Ok(true)
    }

    /// Headless remote-serve loop: block until a remote controller sends input,
    /// process it (mirrored onto the bus), and repeat. Exits on a process-level
    /// interrupt (Ctrl-C); there is no local stdin to read in this mode.
    fn run_remote_headless(&mut self) -> Result<(), String> {
        loop {
            if crate::interrupt::pending() {
                crate::interrupt::clear();
                return Ok(());
            }
            if !self.pump_remote()? {
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }

    /// Worker-side turn loop (the C's `worker_run_turn`): generate, dispatch
    /// tools, drain queued user lines between rounds, repeat until settled.
    /// Runs on the worker thread and talks to the UI only through `tx`.
    #[allow(clippy::too_many_lines)] // flat generate→tools loop; splitting hurts readability
    fn worker_turn(&mut self, tx: &Sender<UiEvent>, shared: &TurnShared) -> Result<(), String> {
        crate::title::set(crate::title::State::Busy(self.last_user_prompt()));
        self.last_turn_interrupted = false;
        self.tool_ctx.skill_invocations = 0;
        self.tool_ctx.tasks.clone_from(&self.session.tasks);
        let mut note = |s: String| {
            let _ = tx.send(UiEvent::Dim(s));
        };
        if let Some(reason) = self.fire_user_prompt_submit(&mut |w| {
            let _ = tx.send(UiEvent::Dim(w));
        }) {
            let _ = tx.send(UiEvent::Dim(format!("halted by hook: {reason}")));
            return Ok(());
        }
        self.maybe_compact_notify(&mut note)?;
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
                    // Freeze: keep the partial on screen, answer the queued
                    // aside(s) via `generate_aside` (which snapshots/restores
                    // the main KV), then resume the same pass.
                    let _ = tx.send(UiEvent::EndLine);
                    resumed_prefix.push_str(&out.assistant_text);
                    let _ = tx.send(UiEvent::Dim(worker::BTW_SUSPEND_MARKER.to_owned()));
                    self.drain_aside(tx, shared);
                    let _ = tx.send(UiEvent::Dim(worker::BTW_RESUME_MARKER.to_owned()));
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
        self.drain_btw_inner(tx, shared, false);
    }

    /// Suspend-mode drain: answers the queued `/btw` question(s) with
    /// [`Engine::generate_aside`] so the frozen main-task KV is snapshotted and
    /// restored around each answer (BTW-SUSPEND-DESIGN §4.3). Used only from
    /// the in-pass suspend path, where the main pass is paused mid-reply and
    /// the partial reply must survive the aside.
    fn drain_aside(&mut self, tx: &Sender<UiEvent>, shared: &TurnShared) {
        self.drain_btw_inner(tx, shared, true);
    }

    fn drain_btw_inner(&mut self, tx: &Sender<UiEvent>, shared: &TurnShared, aside: bool) {
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
        let mut stream = StreamRenderer::new(ChannelSink(tx.clone()));
        stream.set_show_tool_calls(crate::settings::active().ui.show_tool_calls);
        stream.set_show_thinking(crate::settings::active().ui.show_thinking);
        stream.set_thinking_tool_calls(crate::settings::active().engine.thinking_tool_calls);
        stream.set_preflight(edit_preflight(&self.tool_ctx));
        // Local engines open `<think>` implicitly in the prefill; provider
        // engines emit explicit tags, so only pre-open for local ones (see the
        // matching note in the plain-REPL path).
        if !matches!(
            self.cfg.generation.think_mode,
            crate::engine::ThinkMode::Off
        ) && !self.engine.wants_structured()
        {
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
                        greedy_sampling: greedy.load(Ordering::Relaxed),
                        ..Status::default()
                    }
                }
                EngineEvent::Prefill(p) => Status {
                    // See the plain-REPL path: a completed prefill is the
                    // sampling wait, not prefilling (#64 follow-up).
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
            // The aside snapshots/restores the main KV itself and forces greedy
            // off internally, so no greedy sampler is passed.
            self.engine
                .generate_aside(prompt, &self.cfg.generation, &interrupt, &mut on_event)
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
            .map(|e| tool_error_payload(true, e))
            .or_else(|| finished.error.map(|e| tool_error_payload(false, e)));
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
        crate::notify::notify_sticky(
            &crate::notify::finished_title(prompt, interrupted),
            None,
            &crate::notify::latest_output_body(output, interrupted),
        );
    }

    /// Compacts before a TUI turn when context is tight; progress lines go to
    /// `note` (the TUI log, or the worker→UI channel during a turn).
    fn maybe_compact_notify(&mut self, note: &mut dyn FnMut(String)) -> Result<(), String> {
        let rendered = render_transcript(&self.session, &self.system);
        let used = self.engine.count_tokens(&rendered);
        if !compact::should_compact(self.engine.ctx_size(), used) {
            return Ok(());
        }
        // Cheapest step first: clear old tool-result bodies (no model
        // round-trip) and only fall back to full summarization if still tight.
        if let Some(cleared) = self.try_microcompact() {
            note(format!(
                "microcompacted: cleared {cleared} old tool result(s)"
            ));
            return Ok(());
        }
        self.do_compact_notify("low context", note)
    }

    /// Performs a compaction pass and rebuilds the transcript.
    fn do_compact_notify(
        &mut self,
        reason: &str,
        note: &mut dyn FnMut(String),
    ) -> Result<(), String> {
        note(format!(
            "COMPACTING {reason}: summarizing durable task state..."
        ));
        let mut prompt = render_transcript(&self.session, &self.system);
        {
            use std::fmt::Write as _;
            let _ = write!(prompt, "[user]\n{}\n", compact::make_prompt(reason));
        }
        let mut summary = String::new();
        self.engine
            .generate(
                crate::engine::Prompt::Flat(&prompt),
                &self.cfg.generation,
                &|| false,
                &|| false,
                &mut |ev| {
                    if let EngineEvent::Text(t) = ev {
                        summary.push_str(&t);
                    }
                },
            )
            .map_err(|e| e.to_string())?;
        self.rebuild_after_compact(&summary);
        note("context compacted".to_owned());
        Ok(())
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
                        tui::draw(f, l, None, 0, "", v, None, &tui::TaskView::default());
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
            "/init" => self.tui_run_init(log, terminal, view, input, btw, arcade),
            "/compact" => {
                let result = {
                    let mut note = |s: String| log.push_dim(s);
                    self.do_compact_notify("user request", &mut note)
                };
                if let Err(e) = result {
                    log.push_plain(format!("compact failed: {e}"));
                }
            }
            "/save" => match self.store.save(&mut self.session) {
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
                    self.last_ctx_used = 0;
                    self.checkpoints.clear();
                    self.usage = SessionUsage::default();
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
                    self.last_ctx_used = 0;
                    self.checkpoints.clear();
                    self.usage = SessionUsage::default();
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
                    log.push_plain(format!("power limit set to {power}%"));
                }
                None => log.push_plain("usage: /power <1..100>"),
            },
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
            "/btw" => {
                if arg.is_empty() {
                    log.push_plain("usage: /btw <question>");
                } else if let Err(e) = self.tui_btw(arg, log, terminal, view, input, btw, arcade) {
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
            "/repro" => match self.write_repro(arg) {
                Ok(path) => log.push_dim(format!("[repro written to {}]", path.display())),
                Err(e) => log.push_plain(format!("repro failed: {e}")),
            },
            "/subagent" => {
                let (def, task) = crate::agents::resolve(&self.agents, arg);
                let (instructions, task, started) = match def {
                    Some(d) => (
                        Some(d.body.clone()),
                        task.to_string(),
                        format!("[subagent started: {}]", d.name),
                    ),
                    None => (None, task.to_string(), "[subagent started]".to_string()),
                };
                if task.is_empty() {
                    log.push_plain("usage: /subagent [<name>] <task>");
                } else {
                    log.push_dim(started);
                    let fork_at = self.begin_subagent_fork(instructions.as_deref(), &task);
                    if let Err(e) = self.tui_turn(terminal, log, view, input, btw, arcade) {
                        // Restore the transcript even when the turn errored.
                        self.finish_subagent_fork(fork_at, &task);
                        log.push_plain(format!("/subagent failed: {e}"));
                    } else if self.finish_subagent_fork(fork_at, &task) {
                        log.push_dim("[subagent report added to the conversation]");
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
                    if let Err(e) = self.tui_turn(terminal, log, view, input, btw, arcade) {
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
            if let Some(bus) = bus {
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
                // Route to the panel only while an answer is streaming; once
                // it finishes the main task's output goes to the main log even
                // though the (frozen) panel is still visible.
                ev => {
                    if let (true, Some((btw_log, _))) = (btw_active, btw.as_mut()) {
                        worker::apply(btw_log, ev);
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
        terminal
            .draw(|f| {
                if let Some((btw_log, btw_view)) = btw.as_mut() {
                    tui::draw_btw_split(
                        f,
                        log,
                        btw_log,
                        btw_view,
                        Some(input.buf.text()),
                        input.cursor_char(),
                        &status_line,
                        view,
                        &task_view,
                    );
                } else {
                    tui::draw(
                        f,
                        log,
                        Some(input.buf.text()),
                        input.cursor_char(),
                        &status_line,
                        view,
                        None,
                        &task_view,
                    );
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
                    ev => {
                        if let (true, Some((btw_log, _))) = (btw_active, btw.as_mut()) {
                            worker::apply(btw_log, ev);
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
                // Esc closes it before it can interrupt the worker.
                if input.popup_key(key) {
                    continue;
                }
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                // Alt (Option on macOS) or Ctrl turns arrows and
                // Backspace/Delete into word-wise operations.
                let word_mod = ctrl || key.modifiers.contains(KeyModifiers::ALT);
                match key.code {
                    KeyCode::Esc => {
                        close_or_interrupt(shared, btw, btw_active, &mut close_panel_on_end);
                    }
                    KeyCode::Char('c') if ctrl => {
                        // Ctrl-C clears a partly-typed line first; on an empty
                        // line it acts like Esc (cancel answer / close panel /
                        // interrupt the model).
                        if input.buf.text().is_empty() {
                            close_or_interrupt(shared, btw, btw_active, &mut close_panel_on_end);
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
                                log.push_dim("[/btw — pausing the task to answer now]");
                            }
                            view.follow = true;
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
                            view.follow = true;
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
                    view.follow = false;
                    view.top = view.top.saturating_sub(3);
                }
                MouseEventKind::ScrollDown => {
                    // Clamped by draw, which re-enters follow mode at the bottom.
                    view.top = view.top.saturating_add(3);
                }
                MouseEventKind::Down(MouseButton::Left) => {
                    // Clicking the jump-to-bottom hint resumes follow mode.
                    if view.jump_hint_rect.is_some_and(|r| {
                        r.contains(ratatui::layout::Position::new(m.column, m.row))
                    }) {
                        view.follow = true;
                    }
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
    engine: Box<dyn Engine>,
    cfg: &AgentConfig,
    show_footer: bool,
    remote: Option<Arc<RemoteState>>,
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
    let system = sysprompt::build_system_prompt(
        &cfg.system,
        &tool_ctx.mcp,
        !crate::settings::active().engine.thinking_tool_calls,
    );
    let skills = crate::skills::load_default(&tool_ctx.cwd);
    let templates = crate::templates::load_default(&tool_ctx.cwd);
    // The `skill` tool resolves names against the same set the slash command
    // uses; hand the dispatch context its own copy.
    tool_ctx.skills.clone_from(&skills);
    let agents = crate::agents::load_default(&tool_ctx.cwd);
    Ok(Agent {
        engine,
        cfg,
        session,
        store,
        tool_ctx,
        system,
        reminder: SystemPromptReminder::new(),
        power_percent: 0,
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
        checkpoints: crate::checkpoint::CheckpointStore::new(),
        remote,
        ui_remote: None,
        usage: SessionUsage::default(),
        stats: SessionStats::default(),
        session_start: std::time::Instant::now(),
    })
}

/// Runs the interactive REPL until the user exits.
///
/// # Errors
/// Returns an error string on unrecoverable I/O or engine failure.
pub fn run_interactive(
    engine: Box<dyn Engine>,
    cfg: &AgentConfig,
    remote: Option<Arc<RemoteState>>,
) -> Result<(), String> {
    let mut agent = new_agent(engine, cfg, true, remote)?;

    // Seed the notification enable flag once, before either front-end loop
    // starts (CLAUDE.md: TUI and plain REPL are parallel paths sharing this
    // one entry point, so this covers both).
    crate::notify::set_mode(crate::settings::active().ui.notifications);

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
    run_repl_plain(agent)
}

/// Yellow hint shown when Ctrl-C is pressed on an empty idle prompt.
fn quit_hint_spans() -> Vec<ratatui::text::Span<'static>> {
    vec![ratatui::text::Span::styled(
        "Press Ctrl-D to quit.",
        ratatui::style::Style::default().fg(ratatui::style::Color::Yellow),
    )]
}

/// Plain line-based REPL used when stdin is not a terminal. With a remote
/// bridge attached it also interleaves remote-driven input (issue #25); without
/// one it keeps the classic blocking read loop, behavior-identical to before.
fn run_repl_plain(agent: &mut Agent<'_>) -> Result<(), String> {
    if agent.remote.is_some() {
        run_repl_plain_remote(agent)
    } else {
        run_repl_plain_local(agent)
    }
}

/// Handles one line of plain-REPL input. Returns `Ok(false)` to quit the REPL
/// (a `/quit`-style slash command); `Ok(true)` to keep looping. Shared by the
/// local and remote-aware REPL loops so both paths treat slashes, `!`-shell
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

/// escapes, and prompts identically (CLAUDE.md: mirror both UI paths).
fn handle_plain_line(agent: &mut Agent<'_>, line: &str) -> Result<bool, String> {
    let input = line.trim();
    if input.is_empty() {
        return Ok(true);
    }
    if input.starts_with('/') {
        return agent.slash(input);
    }
    if let Some(cmd) = input.strip_prefix('!') {
        // ! prefix is for user-only shell execution — output goes to console
        // but NOT into the session transcript. This is intentional and matches
        // Claude Code's behavior. See issue #20 for discussion.
        let cmd = cmd.trim();
        if cmd.is_empty() {
            println!("usage: !<shell command>");
            return Ok(true);
        }
        match crate::tools::bash::run_immediate(&agent.tool_ctx.cwd, cmd, &mut BangConsoleSink) {
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

/// Interval at which the remote-aware plain REPL wakes to drain the remote
/// input queue while waiting on stdin.
const PLAIN_REMOTE_POLL: Duration = Duration::from_millis(50);

/// Echoes one mirrored bus event to local stdout so a plain-REPL operator sees
/// remote-driven turns. Only text-bearing events are printed; status footers
/// and structural markers are skipped (the plain REPL has no live footer).
fn echo_bus_event(ev: &UiEvent) {
    let mut out = std::io::stdout();
    match ev {
        UiEvent::Visible(t)
        | UiEvent::Think(t)
        | UiEvent::Tool(t)
        | UiEvent::Error(t)
        | UiEvent::Dim(t)
        | UiEvent::Plain(t)
        | UiEvent::UserEcho(t) => {
            let _ = write!(out, "{t}");
        }
        UiEvent::EndLine => {
            let _ = writeln!(out);
        }
        _ => {}
    }
    let _ = out.flush();
}

/// Plain REPL with a remote bridge attached. A dedicated reader thread turns the
/// blocking `read_line` into channel sends so the main loop can `select` between
/// local stdin and the remote input queue: on each idle tick it drains
/// `pump_remote` (mirroring how the TUI idle loop drives remote turns) and
/// echoes the shared bus to stdout so the local operator sees remote output.
///
/// Trade-off: because `read_line` cannot itself be woken, stdin is read on a
/// helper thread rather than in a true `select`. Remote-driven turns run to
/// completion inside `pump_remote` before their (batched) output is echoed,
/// rather than streaming token-by-token as in the TUI; this keeps turn
/// execution single-threaded and the loop simple.
fn run_repl_plain_remote(agent: &mut Agent<'_>) -> Result<(), String> {
    use std::sync::mpsc::{RecvTimeoutError, channel};

    // Reader thread: stdin lines → channel. EOF or error drops the sender,
    // which surfaces as `Disconnected` on the main side.
    let (line_tx, line_rx) = channel::<String>();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut lock = stdin.lock();
        loop {
            let mut line = String::new();
            match lock.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if line_tx.send(line).is_err() {
                        break;
                    }
                }
            }
        }
    });

    let sub = agent.remote.as_ref().map(|r| r.bus.subscribe());
    let drain_bus = |sub: &std::sync::mpsc::Receiver<crate::worker::SeqEvent>| {
        while let Ok(seq) = sub.try_recv() {
            echo_bus_event(&seq.event);
        }
    };

    loop {
        print!("{}", status::prompt_text());
        std::io::stdout().flush().map_err(|e| e.to_string())?;
        // Wait for a typed line or a remote-driven turn; reprint the prompt
        // once either produces output.
        loop {
            if let Some(sub) = &sub {
                drain_bus(sub);
            }
            match line_rx.recv_timeout(PLAIN_REMOTE_POLL) {
                Ok(line) => {
                    if !handle_plain_line(agent, &line)? {
                        return Ok(());
                    }
                    break;
                }
                Err(RecvTimeoutError::Timeout) => {
                    if agent.pump_remote()? {
                        if let Some(sub) = &sub {
                            drain_bus(sub);
                        }
                        break;
                    }
                }
                Err(RecvTimeoutError::Disconnected) => return Ok(()), // stdin EOF
            }
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
    remote: Option<Arc<RemoteState>>,
) -> Result<(), String> {
    let mut agent = new_agent(engine, cfg, false, remote)?;
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
    // Headless with a remote bridge and no `-p`: instead of the stdin protocol,
    // serve remote controllers — drive turns from their `prompt` frames and
    // mirror all output onto the bus (design §5 step 4, headless path).
    if agent.remote.is_some() {
        return agent.run_remote_headless();
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
    use crate::engine::{EngineError, EngineEvent, GenerationStats};
    use std::cell::RefCell;
    use std::rc::Rc;

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
    struct ScriptedEngine {
        replies: Vec<String>,
        next: usize,
        prompts: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        interrupt_at: Option<usize>,
        /// When true, `generate_aside` is implemented (mirrors a real engine's
        /// snapshot/restore support) so the in-pass `/btw` suspend path runs
        /// instead of falling back to the boundary queue.
        aside_support: bool,
    }

    impl ScriptedEngine {
        /// Records the prompt and streams the next scripted reply in chunks.
        fn play_next(
            &mut self,
            transcript: &str,
            on_event: &mut dyn FnMut(EngineEvent),
        ) -> GenerationStats {
            self.prompts.lock().unwrap().push(transcript.to_owned());
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
        fn generate(
            &mut self,
            prompt: crate::engine::Prompt<'_>,
            _opts: &crate::engine::GenerationOptions,
            _interrupt: &dyn Fn() -> bool,
            _greedy: &dyn Fn() -> bool,
            on_event: &mut dyn FnMut(EngineEvent),
        ) -> Result<GenerationStats, EngineError> {
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
            Ok(self.play_next(prompt, on_event))
        }
        fn supports_aside(&self) -> bool {
            self.aside_support
        }
        fn ctx_size(&self) -> i32 {
            100_000
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
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
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
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
        }
    }

    fn test_cfg() -> crate::config::AgentConfig {
        let mut cfg = crate::config::AgentConfig::default();
        cfg.generation.think_mode = crate::engine::ThinkMode::Off;
        cfg
    }

    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("plank-ui-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// End-to-end (issue #25): a real loopback `RemoteServer` sharing the
    /// agent's bus + turn state. A remote `prompt` frame lands in the shared
    /// queue, `pump_remote` drives a real Echo/scripted turn, and the turn's
    /// output is observed mirrored back onto the bus.
    #[test]
    fn remote_prompt_drives_turn_and_mirrors_to_bus() {
        use crate::remote::control::{ClientFrame, ClientMsg, RemoteServer, ServerConfig};
        use tungstenite::Message;

        let dir = scratch_dir("remote-live");
        let engine = ScriptedEngine {
            replies: vec!["hello from echo\n".to_string()],
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);

        // Headless server; the agent adopts its shared bus + turn state.
        let server = RemoteServer::start(
            "127.0.0.1:0",
            ServerConfig {
                token: "tok".to_owned(),
                local_present: false,
                allow_control: false,
                allowed_origins: Vec::new(),
                queue_max: 1 << 20,
            },
            Arc::new(BroadcastBus::new()),
            Arc::new(TurnShared::default()),
        )
        .expect("server binds");
        agent.remote = Some(Arc::clone(&server.state));

        // Observe the bus directly (subscribe before the turn runs).
        let sub = server.state.bus.subscribe();

        // Connect a controller and send auth + prompt.
        let addr = server.local_addr;
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
            ClientMsg::Prompt { text: "hi".into() },
        ] {
            ws.send(Message::Text(ClientFrame::new(m).to_json().unwrap()))
                .unwrap();
            ws.flush().unwrap();
        }

        // Wait for the prompt to reach the shared queue, then drive the turn.
        let deadline = Instant::now() + Duration::from_secs(3);
        while !agent.pump_remote().unwrap() {
            assert!(Instant::now() < deadline, "remote prompt never arrived");
            std::thread::sleep(Duration::from_millis(10));
        }

        // The turn's assistant text was mirrored onto the bus, and the user
        // echo was mirrored too.
        let mut visible = String::new();
        let mut echoed = false;
        while let Ok(seq) = sub.try_recv() {
            match seq.event {
                UiEvent::Visible(t) => visible.push_str(&t),
                UiEvent::UserEcho(t) if t == "hi" => echoed = true,
                _ => {}
            }
        }
        assert!(echoed, "user echo not mirrored");
        assert!(
            visible.contains("hello from echo"),
            "mirrored assistant output missing: {visible:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Plain-REPL path (issue #25 follow-up): the shared `handle_plain_line`
    /// drives a local turn, and a remote `prompt` frame delivered over a real
    /// loopback `RemoteServer` drives a turn via `pump_remote` (the same call
    /// the plain-REPL remote loop makes on each idle tick) with its output
    /// observed mirrored back onto the bus.
    #[test]
    fn plain_repl_handles_local_line_and_remote_drive() {
        use crate::remote::control::{ClientFrame, ClientMsg, RemoteServer, ServerConfig};
        use tungstenite::Message;

        let dir = scratch_dir("plain-remote");
        let engine = ScriptedEngine {
            replies: vec!["local reply\n".to_string(), "remote reply\n".to_string()],
            ..ScriptedEngine::default()
        };
        let cfg = test_cfg();
        let mut agent = test_agent(&dir, engine, &cfg);

        let server = RemoteServer::start(
            "127.0.0.1:0",
            ServerConfig {
                token: "tok".to_owned(),
                local_present: false,
                allow_control: false,
                allowed_origins: Vec::new(),
                queue_max: 1 << 20,
            },
            Arc::new(BroadcastBus::new()),
            Arc::new(TurnShared::default()),
        )
        .expect("server binds");
        agent.remote = Some(Arc::clone(&server.state));
        let sub = server.state.bus.subscribe();

        // A locally-typed prompt runs a turn through the shared handler.
        let before = agent.session.transcript.len();
        assert!(handle_plain_line(&mut agent, "local ask").unwrap());
        assert!(
            agent.session.transcript.len() > before,
            "local line did not advance the session"
        );

        // A remote controller's prompt drives a mirrored turn via pump_remote.
        let addr = server.local_addr;
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
            ClientMsg::Prompt {
                text: "remote ask".into(),
            },
        ] {
            ws.send(Message::Text(ClientFrame::new(m).to_json().unwrap()))
                .unwrap();
            ws.flush().unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(3);
        while !agent.pump_remote().unwrap() {
            assert!(Instant::now() < deadline, "remote prompt never arrived");
            std::thread::sleep(Duration::from_millis(10));
        }

        let mut visible = String::new();
        let mut echoed = false;
        while let Ok(seq) = sub.try_recv() {
            match seq.event {
                UiEvent::Visible(t) => visible.push_str(&t),
                UiEvent::UserEcho(t) if t == "remote ask" => echoed = true,
                _ => {}
            }
        }
        assert!(echoed, "remote user echo not mirrored");
        assert!(
            visible.contains("remote reply"),
            "mirrored remote output missing: {visible:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
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
        agent.drain_aside(&tx, &shared);
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

        // A `!` shell line runs but must not enter the transcript or dirty the
        // session, so a `!`-only session is not worth a resume point.
        handle_plain_line(&mut agent, "!echo plank-bang-test").unwrap();
        assert_eq!(
            agent.session.transcript.len(),
            before,
            "! shell lines must not be logged in the conversation"
        );
        assert!(!agent.session.dirty, "! must not dirty the session");
        assert!(
            agent.save_for_exit().is_none(),
            "a !-only session gets no resume point"
        );
        std::fs::remove_dir_all(&dir).ok();
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
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: "system prompt".to_string(),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
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
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
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
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: String::new(),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
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
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
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
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
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
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
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
    fn close_open_think_skips_discarded_in_think_stanza() {
        // Parity mode (`StreamRenderer` default: thinking tool calls not
        // allowed): a stanza fired mid-thought is discarded, so the pass
        // produces no calls and no continuation follows. The gate the three
        // turn paths apply (`ended_in_think && turn_continues`) must not
        // append a synthetic `</think>` here, or a parity-mode transcript
        // gains a byte the C reference never produces.
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

        let turn_continues = !finished.calls.is_empty() || finished.error.is_some();
        let mut assistant_text = format!("let me look{BASH_STANZA}");
        close_open_think(
            &mut assistant_text,
            finished.ended_in_think && turn_continues,
        );
        assert!(
            !assistant_text.ends_with("</think>"),
            "got: {assistant_text}"
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
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
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
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
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
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
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
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
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
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
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
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
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
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
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
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
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
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
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
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
        };
        agent.tool_ctx.tools.agent = true; // opt-in tool (default off)
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
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
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
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
        };
        agent.session.push(Message::user("hi"));
        agent.session.push(Message::assistant("hello"));

        let fork_at = agent.begin_subagent_fork(None, "count the tests");
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
        let fork_at = agent.begin_subagent_fork(None, "noop");
        assert!(!agent.finish_subagent_fork(fork_at, "noop"));
        assert_eq!(agent.session.transcript.len(), 3);
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
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
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
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
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
            tool_ctx: ToolContext::new(std::env::current_dir().unwrap()),
            system: crate::sysprompt::build_system_prompt("", &[], true),
            reminder: SystemPromptReminder::new(),
            power_percent: 0,
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
            ui_remote: None,
            usage: SessionUsage::default(),
            stats: SessionStats::default(),
            session_start: std::time::Instant::now(),
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
