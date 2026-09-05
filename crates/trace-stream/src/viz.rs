// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Streaming tool-call visualization.
//!
//! Sits between raw model output and the terminal: it detects the DSML
//! tool-call marker in plain text and inside `<think>` blocks, suppresses the
//! raw DSML markup from display, and instead paints compact, human-friendly
//! tool banners ("$ command", "Reading file 1:500...", streamed diff lines
//! with `- `/`+ ` prefixes). Executable tool calls are still parsed by
//! [`crate::dsml::DsmlParser`]; this module only rewrites the terminal
//! projection, never the transcript.
//!
//! Port of `agent_tool_visualizer`, `agent_dsml_marker_detector`, and
//! `agent_stream_renderer` from `ds4_agent.c`.

use crate::dsml::{
    DsmlParser, DsmlState, MARKER_NAMES, ToolCall, tag_prefix_len, tag_prefix_partial,
};

/// Told to the model when it emitted a tool call inside `<think>`.
///
/// Lives here rather than in plank's system prompt module because the stream
/// renderer is what detects the violation. `plank::sysprompt` re-exports it,
/// and `tests/c_parity.rs` locks its text against `refs/ds4`.
pub const IN_THINK_PROHIBITION: &str =
    "Tool calls are not allowed inside <think></think>; finish thinking before emitting DSML.";

/// The canonical DSML tool-call opening marker.
const DSML_START: &[u8] = "<｜DSML｜tool_calls>".as_bytes();
/// Canonical invoke opener, seeded when the model skips the outer wrapper.
const CANONICAL_INVOKE: &[u8] = "<｜DSML｜invoke".as_bytes();
const DSML_BAR: &[u8] = "｜".as_bytes();

const THINK_OPEN: &[u8] = b"<think>";
const THINK_CLOSE: &[u8] = b"</think>";

/// Whether the opt-out env var permits logging. Split out from
/// [`tool_error_logging_enabled`] so the decision is testable: the
/// `cfg!(test)` half of that gate is always true inside a test binary.
fn logging_enabled_for(opt_out: Option<&std::ffi::OsStr>) -> bool {
    opt_out.is_none_or(|v| v != "1")
}

/// Whether tool-call errors are appended to `~/.plank/tool-call-errors.log`.
///
/// Off under `cargo test`, and off in a spawned binary when
/// `PLANK_NO_TOOL_ERROR_LOG=1` is set — the e2e harness sets it so fixture
/// stanzas (`echo plank-e2e`) never enter the developer's real log.
fn tool_error_logging_enabled() -> bool {
    if cfg!(test) {
        return false;
    }
    logging_enabled_for(std::env::var_os("PLANK_NO_TOOL_ERROR_LOG").as_deref())
}

/// Best-effort append of a rejected tool call to `$HOME/.plank/tool-call-errors.log`.
///
/// Records the rejection reason plus the raw DSML stanza the model emitted so a
/// rejected tool call can be inspected after the fact. Always on; any IO error
/// (missing HOME, unwritable dir, …) is silently swallowed so the parse path is
/// never affected.
fn log_tool_error(reason: &str, raw: &[u8]) {
    use std::io::Write;

    if !tool_error_logging_enabled() {
        return;
    }

    let Some(home) = std::env::var_os("HOME").filter(|h| !h.is_empty()) else {
        return;
    };
    let dir = std::path::PathBuf::from(home).join(".plank");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let snippet = String::from_utf8_lossy(raw);
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("tool-call-errors.log"))
    {
        // Ignore write failures; logging must never break parsing.
        let record = format!("[{secs}] {reason}\n---\n{snippet}\n===\n");
        let _ = f.write_all(record.as_bytes());
    }
}

/// Destination for rendered output; the UI layer routes this to its renderer.
///
/// The stream renderer never emits raw DSML through this trait: DSML bytes are
/// replaced by tool banners, which always arrive via
/// [`visible_text`](Self::visible_text).
///
/// This trait is also the animation boundary. Motion is Ratatui-only: the
/// Ratatui sink drives `plank`'s animation module effects (throbber, shimmer, pulse,
/// flash, stall-fade) off the shared 20 Hz clock, while the plain-stdout and
/// `--non-interactive` sinks render the static/reduced-motion form. The stream
/// renderer feeds bytes through here without knowing which sink animates, so the
/// plain path stays untouched.
pub trait RenderSink {
    /// Receives ordinary visible output.
    fn visible_text(&mut self, text: &str);
    /// Receives text produced inside a `<think>` block.
    fn think_text(&mut self, text: &str);
    /// Receives tool banner output (never markdown); defaults to visible.
    fn tool_text(&mut self, text: &str) {
        self.visible_text(text);
    }
    /// Receives error banners (e.g. `[invalid tool call: ...]`); sinks should
    /// style these red. Defaults to tool text.
    fn error_text(&mut self, text: &str) {
        self.tool_text(text);
    }
}

impl RenderSink for Box<dyn RenderSink> {
    fn visible_text(&mut self, text: &str) {
        (**self).visible_text(text);
    }
    fn think_text(&mut self, text: &str) {
        (**self).think_text(text);
    }
    fn tool_text(&mut self, text: &str) {
        (**self).tool_text(text);
    }
    fn error_text(&mut self, text: &str) {
        (**self).error_text(text);
    }
}

/// The `Send` flavour, for a sink that has to cross into a scoped thread — a
/// parallel sub-agent fan-out drives several generations at once, each writing
/// through its own sink.
impl RenderSink for Box<dyn RenderSink + Send> {
    fn visible_text(&mut self, text: &str) {
        (**self).visible_text(text);
    }
    fn think_text(&mut self, text: &str) {
        (**self).think_text(text);
    }
    fn tool_text(&mut self, text: &str) {
        (**self).tool_text(text);
    }
    fn error_text(&mut self, text: &str) {
        (**self).error_text(text);
    }
}

/// A sink that accumulates everything it is given into a shared buffer.
///
/// Used by the parallel fan-out: the sub-agent pane holds one label and one log,
/// so N concurrent sidechains streaming live would interleave into unreadable
/// output. Each slot collects here instead and is flushed as one labelled block
/// when it finishes.
#[derive(Debug, Clone, Default)]
pub struct CollectSink(pub std::sync::Arc<std::sync::Mutex<String>>);

impl CollectSink {
    /// Takes the collected text, leaving the buffer empty.
    #[must_use]
    pub fn take(&self) -> String {
        let mut guard = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::mem::take(&mut *guard)
    }

    fn push(&mut self, text: &str) {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push_str(text);
    }
}

impl RenderSink for CollectSink {
    fn visible_text(&mut self, text: &str) {
        self.push(text);
    }
    // Thinking is deliberately dropped: the pane shows a finished report, and a
    // sub-agent's reasoning is not what the reader came for.
    fn think_text(&mut self, _text: &str) {}
    fn tool_text(&mut self, text: &str) {
        self.push(text);
    }
}

/// Kind of tool parameter, used to select the streaming display style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ParamKind {
    #[default]
    Normal,
    Path,
    Content,
    DiffOld,
    DiffNew,
    BashCommand,
}

fn param_kind_for(tool: &str, param: &str) -> ParamKind {
    match (tool, param) {
        ("bash", "command") => ParamKind::BashCommand,
        ("edit", "old") => ParamKind::DiffOld,
        ("edit", "new") => ParamKind::DiffNew,
        (_, "path" | "file" | "filename") => ParamKind::Path,
        (_, "content" | "text") => ParamKind::Content,
        _ => ParamKind::Normal,
    }
}

/// Display prefix for known tools; `None` falls back to the tool name.
fn tool_prefix(name: &str) -> Option<&'static str> {
    match name {
        "bash" => Some("$ "),
        "read" => Some("read "),
        "write" => Some("write "),
        "edit" => Some("edit "),
        "search" => Some("search "),
        "google_search" => Some("google "),
        "visit_page" => Some("visit "),
        name if name.starts_with("mcp_") => Some("mcp "),
        _ => None,
    }
}

fn diff_prefix(kind: ParamKind) -> Option<&'static str> {
    match kind {
        ParamKind::DiffOld => Some("- "),
        ParamKind::DiffNew => Some("+ "),
        _ => None,
    }
}

fn parse_bool_default(s: &str, default: bool) -> bool {
    if s.is_empty() {
        return default;
    }
    if s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("yes") || s == "1" {
        return true;
    }
    if s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("no") || s == "0" {
        return false;
    }
    default
}

/// Extracts a `name="value"` attribute from a tag, if present.
fn parse_attr(tag: &str, name: &str) -> Option<String> {
    let pat = format!("{name}=\"");
    let start = tag.find(&pat)? + pat.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_string())
}

/// Recognizes a streamed parameter close tag prefix.
///
/// Returns true while `tail` could still become (or already is) a full
/// `</｜DSML｜parameter...>` close tag; sets `complete` when the tail is a
/// full close tag ending exactly at the last byte.
fn parameter_close_tail(tail: &[u8], complete: &mut bool) -> bool {
    *complete = false;
    if tag_prefix_partial(tail, true, "parameter") {
        return true;
    }
    let Some(mut i) = tag_prefix_len(tail, true, "parameter") else {
        return false;
    };
    while i < tail.len() && tail[i].is_ascii_whitespace() {
        i += 1;
    }
    if i < tail.len() && tail.len() - i <= DSML_BAR.len() && DSML_BAR.starts_with(&tail[i..]) {
        return true;
    }
    if tail[i..].starts_with(DSML_BAR) {
        i += DSML_BAR.len();
    }
    while i < tail.len() {
        if tail[i] == b'>' {
            *complete = i == tail.len() - 1;
            return *complete;
        }
        if !tail[i].is_ascii_whitespace() {
            return false;
        }
        i += 1;
    }
    true
}

/// Matches a growing tail against the accepted DSML start forms.
///
/// Returns true while `tail` is a prefix of any accepted opening form; sets
/// `complete` when a form matched fully and `implicit_invoke` when the form
/// was a direct invoke opener without the outer `tool_calls` wrapper.
fn dsml_start_match(tail: &[u8], complete: &mut bool, implicit_invoke: &mut bool) -> bool {
    *complete = false;
    *implicit_invoke = false;
    // Each marker contributes the canonical form and the dropped-leading-bar
    // typo, for both the wrapper and the bare invoke opener. The wrapper also
    // accepts a trailing `｜` before `>` — the closing tags always tolerated it,
    // and post-update weights emit it on the opener too.
    let forms = MARKER_NAMES.iter().flat_map(|m| {
        [
            (format!("<｜{m}｜tool_calls>"), false),
            (format!("<｜{m}｜tool_calls｜>"), false),
            (format!("<{m}｜tool_calls>"), false),
            (format!("<{m}｜tool_calls｜>"), false),
            (format!("<｜{m}｜invoke"), true),
            (format!("<{m}｜invoke"), true),
        ]
    });
    for (form, implicit) in forms {
        let form = form.as_bytes();
        if tail.len() <= form.len() && form[..tail.len()] == *tail {
            *complete = tail.len() == form.len();
            *implicit_invoke = implicit;
            return true;
        }
    }
    false
}

/// Sliding-tail detector for DSML-looking control markers in loose text.
///
/// This helper intentionally has no policy: inside `<think>` a hit means
/// "tool call attempted too early", while in normal output it means malformed
/// DSML the model should see as a tool error.
#[derive(Debug, Default)]
struct MarkerDetector {
    tail: Vec<u8>,
}

impl MarkerDetector {
    const CAP: usize = 32;
    fn feed(&mut self, c: u8) -> bool {
        if self.tail.len() == Self::CAP {
            self.tail.remove(0);
        }
        self.tail.push(c);
        MARKER_NAMES.iter().any(|m| {
            [
                format!("｜{m}｜"),
                format!("|{m}|"),
                format!("<{m}｜"),
                format!("</{m}｜"),
            ]
            .iter()
            .any(|n| self.tail.ends_with(n.as_bytes()))
        })
    }
}

/// Generic tool-call wrappers models fall back to when a tool name is
/// unfamiliar. Matched regardless of the configured tool list.
const GENERIC_PSEUDO_OPENERS: [&str; 3] = ["<tool_call>", "<function_call>", "<invoke "];

/// Detects invented, non-DSML tool-call markup: a bare `<name>` opening a line
/// where `name` is a registered tool, or one of the generic wrappers above.
///
/// Anchored at line start (the tail resets on newline) so prose mentioning
/// `<task>` mid-sentence never matches, and disarmed inside fenced code
/// blocks, where the model is showing markup rather than emitting it. Every
/// byte of the stream is fed to the fence tracker, including thinking, so a
/// fence's open/close parity survives a `<think>`/`</think>` boundary; only
/// tag matching (and thus reporting) is skipped while thinking.
#[derive(Debug, Default)]
struct PseudoToolDetector {
    /// Bytes since the last newline, capped.
    line: Vec<u8>,
    /// Inside a triple-backtick fenced block, where markup is shown rather
    /// than called.
    in_fence: bool,
    /// Names of tools that are actually registered this session.
    tool_names: Vec<String>,
    /// One report per stream is enough; the turn ends after it.
    fired: bool,
}

impl PseudoToolDetector {
    const CAP: usize = 96;

    fn set_tool_names(&mut self, names: Vec<String>) {
        self.tool_names = names;
    }

    /// Clears the line-start anchor. `</think>` is a hard boundary for the
    /// answer region even though no `\n` byte crosses it, so without this the
    /// tail of a thinking line (never cleared, since tag matching — not line
    /// tracking — is what's skipped while thinking) would still be glued
    /// onto the first line of the answer and defeat the line-start anchor.
    fn reset_line(&mut self) {
        self.line.clear();
    }

    /// Feeds one byte; returns the matched line when the byte *completes* a
    /// line that is nothing but a pseudo-tool-call opening.
    ///
    /// Matching deliberately waits for the end of the line. Firing at the `>`
    /// would misjudge `<read> is how you spell it`, and since a stream error
    /// freezes output that fabricated diagnosis would also discard the rest of
    /// a legitimate answer. A stream that ends mid-line is matched by
    /// [`Self::finish_line`].
    ///
    /// `in_think` still runs the byte through the fence tracker — a fence
    /// opened inside `<think>` must flip `in_fence` there so that its
    /// matching closer in the answer region is recognized as a *closer*,
    /// not mistaken for a fresh opener that would silence the detector for
    /// the rest of the stream (issue #51's reproduction: a fence opened in
    /// thinking and closed just after `</think>`, right before a pseudo-tool
    /// block). Only the tag-matching/report path is skipped while thinking.
    fn feed(&mut self, c: u8, in_think: bool) -> Option<String> {
        if c == b'\n' {
            let hit = self.match_line(in_think);
            if self.line.starts_with(b"```") {
                // Parity is tracked unconditionally, even while `in_think`,
                // but the trade-off cuts both ways. It's what makes the
                // thinking-then-answer fence in the doc comment above work.
                // The flip side: a fence opened in `<think>` and never closed
                // there leaves `in_fence` true for the rest of the stream,
                // silently disarming detection for the whole answer region.
                // Resetting `in_fence` at `</think>` would fix that case but
                // reopen the one this exists for, so the choice stands.
                self.in_fence = !self.in_fence;
            }
            self.line.clear();
            return hit;
        }
        if self.line.len() < Self::CAP {
            self.line.push(c);
        }
        None
    }

    /// Matches the line held at end of stream, where no `\n` will arrive.
    fn finish_line(&mut self, in_think: bool) -> Option<String> {
        let hit = self.match_line(in_think);
        self.line.clear();
        hit
    }

    /// Tests the completed line, returning it when it is nothing but an
    /// invented tool-call opening.
    fn match_line(&mut self, in_think: bool) -> Option<String> {
        if in_think || self.fired || self.in_fence {
            return None;
        }
        // Only a tag *alone* on the line counts; surrounding whitespace is
        // allowed, prose after the tag is not.
        let trimmed = self.line.trim_ascii();
        if !trimmed.starts_with(b"<") {
            return None;
        }
        let text = std::str::from_utf8(trimmed).ok()?;
        let matched = GENERIC_PSEUDO_OPENERS.iter().any(|opener| {
            // `<invoke ` carries attributes, so it matches any single tag
            // opening with it; the others must be the whole line.
            if opener.ends_with(' ') {
                text.starts_with(*opener) && is_lone_tag(text)
            } else {
                text == *opener
            }
        }) || self.tool_names.iter().any(|name| {
            text.strip_prefix('<')
                .and_then(|t| t.strip_suffix('>'))
                .is_some_and(|inner| inner == name)
        });
        if !matched {
            return None;
        }
        self.fired = true;
        Some(text.to_string())
    }
}

/// True when `text` is a single `<...>` tag and nothing else.
fn is_lone_tag(text: &str) -> bool {
    text.ends_with('>') && !text[..text.len() - 1].contains('>')
}

/// Scanner state mirroring the DSML parser for display purposes only.
#[derive(Debug, Default)]
enum DsmlScan {
    /// Between tags; whitespace is skipped, `<` opens a tag.
    #[default]
    Between,
    /// Accumulating a structural tag until `>`.
    Tag(Vec<u8>),
    /// Inside a parameter value.
    Value,
}

/// Per-tool-call display state (port of `agent_tool_visualizer`).
// The bool flags mirror the C state machine one-to-one; collapsing them into
// enums would obscure the correspondence with the reference implementation.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Default)]
struct ToolViz {
    active: bool,
    tool_announced: bool,
    param_active: bool,
    at_line_start: bool,
    param_kind: ParamKind,
    tool_name: String,
    param_name: String,
    param_end_tail: Vec<u8>,
    read_style: bool,
    read_prefix_rendered: bool,
    read_line_rendered: bool,
    read_path: String,
    read_start: String,
    read_max: String,
    read_whole: String,
    code_param_active: bool,
    /// Destination captured from a `write` call's path param, for the dim
    /// content-preview header.
    write_path: String,
    /// True when the current `write` targets a file that does not yet exist:
    /// only then does the content stream as a dim preview (an overwrite is left
    /// to the post-edit diff card).
    write_is_create: bool,
}

impl ToolViz {
    const END_TAIL_CAP: usize = 64;
}

/// Snapshot of stream results after generation ends.
///
/// Returned by [`StreamRenderer::finished`].
#[derive(Debug, Clone, Copy)]
pub struct Finished<'a> {
    /// Executable tool calls completed by the DSML parser, in stream order.
    pub calls: &'a [ToolCall],
    /// Error message from malformed or misplaced DSML, if any.
    pub error: Option<&'a str>,
    /// True when a DSML marker was seen inside a `<think>` block.
    pub dsml_in_think: bool,
    /// True when a tool call was rejected for being inside `<think>`, so
    /// [`Self::error`] is about placement rather than syntax. Callers word the
    /// model-facing message from this — telling it the DSML was invalid when
    /// it was merely misplaced is a wild goose chase.
    pub in_think_rejected: bool,
    /// True when the stream ended with a `<think>` block still open — the model
    /// emitted a tool call mid-thought and stopped for the dispatch. The caller
    /// closes the block in the transcript before appending the tool result.
    pub ended_in_think: bool,
}

/// Streaming display state machine for assistant output.
///
/// Feed model text with [`push`](Self::push) and call
/// [`finish`](Self::finish) once when the stream ends. Ordinary prose passes
/// through to the sink; raw DSML is hidden and replaced by tool banners.
/// Partial `<｜DSML｜` prefixes are held back until disambiguated, then either
/// consumed (real tool call) or flushed verbatim (false alarm).
///
/// # Examples
///
/// ```no_run
/// use trace_stream::viz::{RenderSink, StreamRenderer};
///
/// struct Stdout;
/// impl RenderSink for Stdout {
///     fn visible_text(&mut self, t: &str) { print!("{t}"); }
///     fn think_text(&mut self, t: &str) { eprint!("{t}"); }
/// }
///
/// let mut sr = StreamRenderer::new(Stdout);
/// sr.push("Hello ");
/// sr.push("world");
/// sr.finish();
/// assert!(sr.finished().calls.is_empty());
/// ```
// See ToolViz: the flags deliberately mirror the C state machine.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug)]
pub struct StreamRenderer<S> {
    sink: S,
    parser: DsmlParser,
    viz: ToolViz,
    scan: DsmlScan,
    in_think: bool,
    dsml_active: bool,
    dsml_ignored: bool,
    /// Held-back bytes that may begin `<think>` / `</think>`.
    pending: Vec<u8>,
    /// Held-back bytes that may begin a DSML opening marker.
    dsml_start_tail: Vec<u8>,
    plain_dsml: MarkerDetector,
    think_dsml: MarkerDetector,
    /// Detects invented pseudo-tool markup (issue #51) in the answer region.
    pseudo_tool: PseudoToolDetector,
    dsml_in_think: bool,
    dsml_in_think_reported: bool,
    /// A `</think>` was consumed as parameter-value *content* while a stanza
    /// was open (see [`Self::think_close_is_control`]). Cleared when that
    /// stanza parses clean; if it instead dies malformed or is cut off, the
    /// token was almost certainly the real control token and
    /// [`Self::resolve_swallowed_think_close`] un-sticks `in_think`.
    think_close_swallowed: bool,
    /// A tool call was discarded *because* it sat inside `<think>`.
    ///
    /// Distinct from [`Self::dsml_in_think`], which only means the marker was
    /// seen there — that is also true when `engine.thinkingToolCalls` is on
    /// and the call was dispatched. Only this one means the model asked for a
    /// tool and got nothing, so only this one is worth telling it about.
    in_think_rejected: bool,
    /// The stream error came from the pseudo-tool detector, which by
    /// construction only fires in the answer region. It must outrank the
    /// in-think prohibition in [`Self::finished`]: the model's mistake was
    /// inventing markup after `</think>`, not calling a tool inside it.
    pseudo_tool_fired: bool,
    post_think_gap: bool,
    /// Error from DSML markup outside a valid stanza. Freezes further output
    /// only when [`Self::set_freeze_on_error`] opted in.
    stream_error: Option<String>,
    /// Whether an error freezes all further output (default false). See
    /// [`Self::set_freeze_on_error`].
    freeze_on_error: bool,
    last_output_newline: bool,
    /// Calls snapshotted at parser `Done`, surviving later parser resets.
    calls: Vec<ToolCall>,
    /// UTF-8 carry buffers so multi-byte characters split across pushes are
    /// never emitted partially.
    vis_carry: Vec<u8>,
    think_carry: Vec<u8>,
    /// Mid-stream tool preflight hook and its first failure, mirroring the
    /// C's `agent_stream_preflight_closed_param`: an `edit` call's `old`
    /// selector is validated the moment that parameter closes, so a doomed
    /// edit stops generation before `new` is streamed.
    preflight: Preflight,
    preflight_error: Option<String>,
    /// When false, tool-call visualization (banners, params, read/diff lines)
    /// is dropped — the DSML is still parsed and hidden, just not shown. Gated
    /// by `ui.showToolCalls`; defaults true so tests and callers that don't set
    /// it keep the banners. See [`StreamRenderer::set_show_tool_calls`].
    show_tool_calls: bool,
    /// When false, thinking text is parsed (so `<think>`/`</think>` still drive
    /// state) but never emitted to the sink. Gated by `ui.showThinking`;
    /// defaults true. See [`StreamRenderer::set_show_thinking`].
    show_thinking: bool,
    /// When false, the dim content preview a `write` of a new file streams is
    /// dropped too. Independent of `show_tool_calls` (the preview is normally
    /// kept even with banners off); defaults true. See
    /// [`StreamRenderer::set_show_write_preview`].
    show_write_preview: bool,
    /// When true, a DSML stanza opened inside `<think>` is parsed and dispatched
    /// like any other. Defaults **false**, which is strict `refs/ds4` parity:
    /// the stanza is discarded with a `[tool call ignored: ...]` notice.
    /// Production wires this from `engine.thinkingToolCalls`, which defaults
    /// off too, so callers and tests keep the C behavior unless they opt in.
    thinking_tool_calls: bool,
    /// When true, this renderer is replaying text that was already streamed
    /// (and diagnosed) once before, from a stored transcript. The pseudo-tool
    /// detector is a new addition on top of an existing replay path that
    /// discards `finished()`, so any error it raises can never reach the
    /// model; it would only double-log to disk and truncate the replayed
    /// text. Defaults false. See [`StreamRenderer::set_replay`].
    replay: bool,
}

/// Hook validating a partially-parsed tool call mid-stream.
type PreflightFn = Box<dyn FnMut(&ToolCall) -> Result<(), String>>;

#[derive(Default)]
struct Preflight(Option<PreflightFn>);

impl std::fmt::Debug for Preflight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() {
            "Preflight(set)"
        } else {
            "Preflight(unset)"
        })
    }
}

const DSML_START_TAIL_CAP: usize = 64;

impl<S: RenderSink> StreamRenderer<S> {
    /// Creates a renderer that writes rendered output to `sink`.
    pub fn new(sink: S) -> Self {
        Self {
            sink,
            parser: DsmlParser::new(),
            viz: ToolViz::default(),
            scan: DsmlScan::Between,
            in_think: false,
            freeze_on_error: false,
            dsml_active: false,
            dsml_ignored: false,
            pending: Vec::new(),
            dsml_start_tail: Vec::new(),
            plain_dsml: MarkerDetector::default(),
            think_dsml: MarkerDetector::default(),
            pseudo_tool: PseudoToolDetector::default(),
            dsml_in_think: false,
            dsml_in_think_reported: false,
            think_close_swallowed: false,
            in_think_rejected: false,
            pseudo_tool_fired: false,
            post_think_gap: false,
            stream_error: None,
            last_output_newline: true,
            calls: Vec::new(),
            vis_carry: Vec::new(),
            think_carry: Vec::new(),
            preflight: Preflight(None),
            preflight_error: None,
            show_tool_calls: true,
            show_thinking: true,
            show_write_preview: true,
            thinking_tool_calls: false,
            replay: false,
        }
    }

    /// Sets whether this renderer is replaying already-diagnosed text from a
    /// stored transcript (default false). Replayed text was already streamed
    /// and diagnosed the first time it was produced, so re-reporting it here
    /// would both double-log to `~/.plank/tool-call-errors.log` and truncate
    /// the rest of the replayed message via `stream_error`. Wire this at the
    /// replay construction site only; live generation must leave it false.
    pub fn set_replay(&mut self, replay: bool) {
        self.replay = replay;
    }

    /// Sets the tool names the pseudo-tool detector recognizes (default none).
    ///
    /// Production wires this from the live registry; with an empty list only
    /// the generic wrappers (`<tool_call>`, `<function_call>`, `<invoke `) are
    /// matched, which is what unit tests and echo paths get.
    pub fn set_tool_names(&mut self, names: Vec<String>) {
        self.pseudo_tool.set_tool_names(names);
    }

    /// Sets whether tool-call visualization is shown (default true). Production
    /// wires this from `ui.showToolCalls`; when false the DSML is still parsed
    /// and hidden but no banner, param, read, or diff line is emitted.
    pub fn set_show_tool_calls(&mut self, show: bool) {
        self.show_tool_calls = show;
    }

    /// Sets whether thinking text is displayed (default true). Production wires
    /// this from `ui.showThinking`; when false the model still produces its
    /// thinking (and `<think>` tags still drive parser state) but none of it is
    /// emitted to the sink.
    pub fn set_show_thinking(&mut self, show: bool) {
        self.show_thinking = show;
    }

    /// Sets whether a new-file `write` streams its content as a dim preview
    /// (default true). `/init` turns it off together with the banners so the
    /// generated AGENTS.md draft never scrolls past.
    pub fn set_show_write_preview(&mut self, show: bool) {
        self.show_write_preview = show;
    }

    /// Sets whether a DSML error freezes all further output (default false).
    ///
    /// Opt-in, and deliberately so: freezing is only correct for a renderer
    /// whose life is one generation pass. plank builds one per pass, so the
    /// freeze keeps raw tool-call markup off the user's screen for the rest of
    /// a doomed stanza and ends with the pass. A renderer that outlives a pass
    /// -- the debug-console mirror keeps one per *connection*, fed a raw byte
    /// tee with no pass boundaries in it -- would instead discard every byte of
    /// every later pass, and the console window went dead after the first bad
    /// stanza while plank itself recovered normally. Defaulting to false means
    /// a consumer gets the safe behaviour without knowing this flag exists; the
    /// one consumer that wants the freeze asks for it.
    ///
    /// Independent of error *reporting*: [`Self::finished`]'s `error` is set
    /// either way.
    pub fn set_freeze_on_error(&mut self, freeze: bool) {
        self.freeze_on_error = freeze;
    }

    /// Sets whether tool calls emitted inside `<think></think>` are dispatched
    /// (default false = strict C parity). Production wires this from
    /// `engine.thinkingToolCalls`. When false an in-think stanza is discarded
    /// with a `[tool call ignored: ...]` notice, exactly as the C agent does.
    pub fn set_thinking_tool_calls(&mut self, allow: bool) {
        self.thinking_tool_calls = allow;
    }

    /// Installs the mid-stream preflight hook: called with the pending call
    /// each time an `edit` invoke's `old` parameter closes. A returned error
    /// records [`preflight_error`](Self::preflight_error); the caller is
    /// expected to stop generation and feed the error back to the model.
    pub fn set_preflight(&mut self, f: impl FnMut(&ToolCall) -> Result<(), String> + 'static) {
        self.preflight = Preflight(Some(Box::new(f)));
    }

    /// The first mid-stream preflight failure, if any (model-facing text).
    #[must_use]
    pub fn preflight_error(&self) -> Option<&str> {
        self.preflight_error.as_deref()
    }

    /// Records a failure raised outside the renderer, on the same channel a
    /// preflight failure uses, so the caller's existing "stop the engine and
    /// feed the error back to the model" path handles it. The first failure
    /// wins; later ones are ignored. Used by the turn loop's repetition guard.
    pub fn fail_preflight(&mut self, err: impl Into<String>) {
        if self.preflight_error.is_none() {
            self.preflight_error = Some(err.into());
        }
    }

    /// Starts the stream already inside a `<think>` block.
    ///
    /// Use when the chat template opened thinking in the prefill prefix, so the
    /// model streams thinking content before any `</think>` and without an
    /// opening tag of its own.
    pub fn begin_in_think(&mut self) {
        self.in_think = true;
    }

    /// Whether the stream is currently inside a `<think>` block.
    ///
    /// Read live, mid-generation, so the status bar can say the model is
    /// reasoning rather than answering; it is not a parse result and says
    /// nothing about whether the block will close cleanly.
    #[must_use]
    pub fn in_think(&self) -> bool {
        self.in_think
    }

    /// Feeds one streamed chunk of model output.
    pub fn push(&mut self, text: impl AsRef<str>) {
        self.stream_text(text.as_ref().as_bytes(), false);
    }

    /// Signals end of stream, flushing held-back bytes and open banners.
    ///
    /// An interrupted tool call is closed with a `[tool call interrupted]`
    /// status line; DSML seen inside thinking is reported as ignored.
    pub fn finish(&mut self) {
        self.stream_text(b"", true);
        self.flush_carry();
        self.flush_pseudo_tool();
    }

    /// Results after the stream ends: completed calls and error state.
    ///
    /// A stream that ends mid-stanza (parser still structural or inside a
    /// parameter value) reports `incomplete DSML tool call`, like the C's
    /// worker loop; callers must check user interruption first, since an
    /// interrupted stanza is not a model error.
    #[must_use]
    pub fn finished(&self) -> Finished<'_> {
        // A call inside `<think>` is *misplaced*, not malformed, and that is
        // what the model needs to hear: the parser's own verdict on the
        // discarded stanza ("incomplete DSML tool call", say) would send it
        // rewriting syntax that was never wrong. `dsml_ignored` covers a
        // stanza still open when the stream ended; `in_think_rejected` covers
        // one that completed and was thrown away. Mirrors the C worker loop
        // (`ds4_agent.c:7853`), which likewise overwrites the parse error.
        //
        // The one exception is a pseudo-tool hit: it can only be raised in the
        // answer region, after `</think>`, so overwriting it with the
        // prohibition would tell the model to move a call it never made there.
        // A stanza leaked from thinking *and* invented markup in the answer is
        // exactly the reported case, and the answer-region mistake is the one
        // the model must fix.
        let in_think = (self.in_think_rejected || self.dsml_ignored) && !self.pseudo_tool_fired;
        let error = self
            .stream_error
            .as_deref()
            .filter(|_| !in_think)
            .or_else(|| in_think.then_some(IN_THINK_PROHIBITION))
            .or_else(|| (self.parser.state() == DsmlState::Error).then(|| self.parser.error()))
            .or_else(|| {
                matches!(
                    self.parser.state(),
                    DsmlState::Structural | DsmlState::ParamValue
                )
                .then_some("incomplete DSML tool call")
            });
        Finished {
            calls: &self.calls,
            error,
            dsml_in_think: self.dsml_in_think,
            in_think_rejected: in_think,
            ended_in_think: self.in_think,
        }
    }

    /// True while the sampler should run greedy (argmax), mirroring
    /// `agent_stream_wants_greedy_sampling`: inside a tool-call stanza's
    /// structural markup, while a parameter close tag is streaming, or once
    /// the start detector holds a DSML-shaped prefix longer than one byte.
    /// Derived purely from per-round parser state, so EOS, errors, or the
    /// next turn can never leave sampling stuck greedy.
    #[must_use]
    pub fn wants_greedy_sampling(&self) -> bool {
        if matches!(self.parser.state(), DsmlState::Error | DsmlState::Done) {
            return false;
        }
        // A single '<' is too common in prose/code to justify forcing argmax;
        // a longer held-back prefix is specifically DSML-shaped.
        if self.dsml_start_tail.len() > 1 {
            return true;
        }
        if !self.dsml_active {
            return false;
        }
        match self.parser.state() {
            DsmlState::Structural => true,
            DsmlState::ParamValue => self.parser.param_close_prefix(),
            _ => false,
        }
    }

    /// Borrows the underlying sink.
    pub fn sink(&self) -> &S {
        &self.sink
    }

    /// Mutable access to the sink, for callers that must drain it mid-stream.
    pub fn sink_mut(&mut self) -> &mut S {
        &mut self.sink
    }

    /// Consumes the renderer, returning the sink.
    pub fn into_sink(self) -> S {
        self.sink
    }

    // ---- output helpers -------------------------------------------------

    fn flush_stream(sink_write: impl FnOnce(&mut S, &str), sink: &mut S, carry: &mut Vec<u8>) {
        if carry.is_empty() {
            return;
        }
        match std::str::from_utf8(carry) {
            Ok(s) => {
                sink_write(sink, s);
                carry.clear();
            }
            Err(e) if e.error_len().is_none() && e.valid_up_to() > 0 => {
                let tail = carry.split_off(e.valid_up_to());
                // The prefix is valid UTF-8 by construction.
                sink_write(sink, std::str::from_utf8(carry).unwrap_or_default());
                *carry = tail;
            }
            Err(e) if e.error_len().is_none() => {}
            Err(_) => {
                let s = String::from_utf8_lossy(carry).into_owned();
                sink_write(sink, &s);
                carry.clear();
            }
        }
    }

    fn emit_visible_bytes(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        // Tool-call visualization goes through the tool_text channel (that is
        // what `self.viz.active` selects below). When banners are off, drop it
        // entirely — parsing and DSML hiding are unaffected.
        if self.viz.active && !self.show_tool_calls {
            return;
        }
        self.last_output_newline = bytes.last() == Some(&b'\n');
        self.vis_carry.extend_from_slice(bytes);
        let write = if self.viz.active {
            S::tool_text as fn(&mut S, &str)
        } else {
            S::visible_text
        };
        Self::flush_stream(write, &mut self.sink, &mut self.vis_carry);
    }

    fn emit_think_bytes(&mut self, bytes: &[u8]) {
        if !self.show_thinking {
            return;
        }
        self.think_carry.extend_from_slice(bytes);
        Self::flush_stream(S::think_text, &mut self.sink, &mut self.think_carry);
    }

    /// Emits `write`-preview bytes in the thinking (dim) color. Independent of
    /// `show_tool_calls` and `show_thinking`, so a write's content is always
    /// visible as a dim preview of what is being saved — unless
    /// `show_write_preview` is off.
    fn emit_preview_bytes(&mut self, bytes: &[u8]) {
        if bytes.is_empty() || !self.show_write_preview {
            return;
        }
        self.last_output_newline = bytes.last() == Some(&b'\n');
        self.think_carry.extend_from_slice(bytes);
        Self::flush_stream(S::think_text, &mut self.sink, &mut self.think_carry);
    }

    fn viz_preview_puts(&mut self, s: &str) {
        self.emit_preview_bytes(s.as_bytes());
    }

    /// True while streaming the `write` tool's content body, which renders as a
    /// dim preview rather than through the normal (banner-gated) channel.
    fn viz_is_write_preview(&self) -> bool {
        self.viz.tool_name == "write" && self.viz.param_kind == ParamKind::Content
    }

    fn flush_carry(&mut self) {
        for (write, carry) in [
            (S::visible_text as fn(&mut S, &str), &mut self.vis_carry),
            (S::think_text, &mut self.think_carry),
        ] {
            if !carry.is_empty() {
                let s = String::from_utf8_lossy(carry).into_owned();
                write(&mut self.sink, &s);
                carry.clear();
            }
        }
    }

    /// Routes one ordinary output byte to the visible or think stream.
    fn write_char(&mut self, c: u8) {
        if self.in_think {
            self.emit_think_bytes(&[c]);
        } else {
            self.emit_visible_bytes(&[c]);
        }
    }

    fn viz_puts(&mut self, s: &str) {
        self.emit_visible_bytes(s.as_bytes());
    }

    /// Emits an error banner through the sink's red-styled channel.
    fn viz_error_puts(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        self.last_output_newline = s.ends_with('\n');
        self.sink.error_text(s);
    }

    fn viz_newline_if_open(&mut self) {
        if !self.last_output_newline {
            self.viz_puts("\n");
        }
    }

    // ---- tool visualizer -------------------------------------------------

    fn viz_start(&mut self) {
        let line_open = !self.last_output_newline;
        self.viz = ToolViz {
            active: true,
            at_line_start: true,
            ..ToolViz::default()
        };
        self.scan = DsmlScan::Between;
        if line_open {
            self.viz_puts("\n");
        }
    }

    /// Starts a tool banner line: "🛠️ ".
    fn viz_line_prefix(&mut self) {
        self.viz_newline_if_open();
        self.viz_puts("🛠️ ");
        self.viz.at_line_start = false;
    }

    fn viz_tool(&mut self, name: &str) {
        if self.viz.tool_announced && self.viz.tool_name == name {
            return;
        }
        if self.viz.tool_announced {
            self.viz_newline_if_open();
        }
        self.viz.tool_name = name.to_string();
        self.viz.tool_announced = true;
        self.viz.read_style = name == "read";
        self.viz_line_prefix();
        if self.viz.read_style {
            self.viz_puts("Reading ");
            self.viz.read_prefix_rendered = true;
            return;
        }
        if let Some(prefix) = tool_prefix(name) {
            self.viz_puts(prefix);
        } else {
            let owned = name.to_string();
            self.viz_puts(&owned);
            self.viz_puts(" ");
        }
    }

    fn viz_read_value_byte(&mut self, c: u8) {
        let field = match self.viz.param_name.as_str() {
            "path" => &mut self.viz.read_path,
            "start_line" => &mut self.viz.read_start,
            "max_lines" => &mut self.viz.read_max,
            "whole" => &mut self.viz.read_whole,
            _ => return,
        };
        field.push(c as char);
        if self.viz.param_name == "path" && self.viz.read_prefix_rendered {
            self.emit_visible_bytes(&[c]);
        }
    }

    /// Renders the one-line read banner, e.g. "Reading src/x.rs 1:500...".
    fn viz_render_read(&mut self) {
        if !self.viz.read_style || self.viz.read_line_rendered {
            return;
        }
        if !self.viz.read_prefix_rendered {
            self.viz_line_prefix();
            self.viz_puts("Reading ");
            let path = if self.viz.read_path.is_empty() {
                "<unknown>".to_string()
            } else {
                self.viz.read_path.clone()
            };
            self.viz_puts(&path);
        } else if self.viz.read_path.is_empty() {
            self.viz_puts("<unknown>");
        }
        let whole = parse_bool_default(&self.viz.read_whole, false);
        let range = if whole && (self.viz.read_start.is_empty() || self.viz.read_start == "1") {
            " (whole file)".to_string()
        } else if whole {
            format!(" {}:EOF", self.viz.read_start)
        } else {
            let start = if self.viz.read_start.is_empty() {
                "1"
            } else {
                &self.viz.read_start
            };
            let max = if self.viz.read_max.is_empty() {
                "500"
            } else {
                &self.viz.read_max
            };
            format!(" {start}:{max}")
        };
        self.viz_puts(&range);
        self.viz_puts("...\n");
        self.viz.read_line_rendered = true;
    }

    fn viz_param_is_code_body(&self) -> bool {
        match self.viz.tool_name.as_str() {
            "write" => self.viz.param_kind == ParamKind::Content,
            "edit" => matches!(
                self.viz.param_kind,
                ParamKind::DiffOld | ParamKind::DiffNew | ParamKind::Content
            ),
            _ => false,
        }
    }

    /// Emits the diff line prefix ("- " / "+ ") at the start of a code line.
    fn viz_code_prefix(&mut self) {
        if !self.viz.at_line_start {
            return;
        }
        if let Some(prefix) = diff_prefix(self.viz.param_kind) {
            self.viz_puts(prefix);
            self.viz.at_line_start = false;
        }
    }

    fn viz_code_begin(&mut self) {
        self.viz.code_param_active = true;
        if matches!(self.viz.param_kind, ParamKind::DiffOld | ParamKind::DiffNew) {
            self.viz_code_prefix();
        }
    }

    fn viz_code_end(&mut self) {
        if !self.viz.code_param_active {
            return;
        }
        self.viz.code_param_active = false;
        self.viz.at_line_start = true;
    }

    fn viz_code_byte(&mut self, c: u8) {
        // A new file's body streams as a dim preview; an overwrite's body is
        // dropped here (the post-edit diff card shows it).
        if self.viz_is_write_preview() {
            if self.viz.write_is_create {
                self.emit_preview_bytes(&[c]);
            }
            self.viz.at_line_start = c == b'\n';
            return;
        }
        self.viz_code_prefix();
        self.emit_visible_bytes(&[c]);
        self.viz.at_line_start = c == b'\n';
    }

    fn viz_param_begin(&mut self, name: &str) {
        self.viz.param_name = name.to_string();
        self.viz.param_kind = param_kind_for(&self.viz.tool_name, name);
        self.viz.param_active = true;
        self.viz.param_end_tail.clear();

        if self.viz.read_style {
            return;
        }
        match self.viz.param_kind {
            ParamKind::DiffOld | ParamKind::DiffNew => {
                self.viz_newline_if_open();
                self.viz.at_line_start = true;
                self.viz_code_begin();
            }
            ParamKind::Content => {
                self.viz_newline_if_open();
                if self.viz.tool_name == "write" {
                    // Stream the content as a dim preview only for a new file;
                    // an overwrite is shown by the post-edit diff card instead.
                    self.viz.write_is_create = !std::path::Path::new(&self.viz.write_path).exists();
                    // A dim header names the file for the content preview. Only
                    // when the banner is off, else the banner already shows it.
                    if self.viz.write_is_create && !self.show_tool_calls {
                        let path = if self.viz.write_path.is_empty() {
                            "<file>".to_string()
                        } else {
                            self.viz.write_path.clone()
                        };
                        self.viz_preview_puts(&format!("write {path}\n"));
                    }
                } else {
                    let label = format!("{name}:\n");
                    self.viz_puts(&label);
                }
                self.viz.at_line_start = true;
                if self.viz_param_is_code_body() {
                    self.viz_code_begin();
                }
            }
            ParamKind::BashCommand => {}
            ParamKind::Normal | ParamKind::Path => {
                if !self.viz.at_line_start {
                    self.viz_puts(" ");
                }
                let label = format!("{name}=");
                self.viz_puts(&label);
            }
        }
    }

    fn viz_param_end(&mut self) {
        self.viz.param_end_tail.clear();
        if self.viz.code_param_active {
            self.viz_code_end();
        }
        self.viz.param_active = false;
        self.viz.param_name.clear();
        self.scan = DsmlScan::Between;
    }

    fn viz_param_raw_byte(&mut self, c: u8) {
        if self.viz.read_style {
            self.viz_read_value_byte(c);
            return;
        }
        if self.viz.code_param_active {
            self.viz_code_byte(c);
            return;
        }
        if matches!(self.viz.param_kind, ParamKind::DiffOld | ParamKind::DiffNew) {
            self.viz_code_begin();
            self.viz_code_byte(c);
            return;
        }
        // Capture the write destination for the content-preview header.
        if self.viz.tool_name == "write" && self.viz.param_kind == ParamKind::Path {
            self.viz.write_path.push(c as char);
        }
        self.emit_visible_bytes(&[c]);
        self.viz.at_line_start = c == b'\n';
    }

    /// Streams one parameter value byte, hiding partial close-tag tails.
    ///
    /// The visualizer must not wait for the whole parameter: large write/edit
    /// contents should show progress while still detecting the closing tag.
    fn viz_param_value_byte(&mut self, c: u8) {
        if !self.viz.param_end_tail.is_empty() || c == b'<' {
            if self.viz.param_end_tail.len() == ToolViz::END_TAIL_CAP {
                let held = std::mem::take(&mut self.viz.param_end_tail);
                for b in held {
                    self.viz_param_raw_byte(b);
                }
                if c != b'<' {
                    self.viz_param_raw_byte(c);
                    return;
                }
            }
            self.viz.param_end_tail.push(c);
            let mut complete = false;
            if parameter_close_tail(&self.viz.param_end_tail, &mut complete) {
                if complete {
                    self.viz_param_end();
                }
                return;
            }
            let held = std::mem::take(&mut self.viz.param_end_tail);
            for b in held {
                self.viz_param_raw_byte(b);
            }
            return;
        }
        self.viz_param_raw_byte(c);
    }

    /// Called when an invoke closes: flush the read banner, reset announce.
    fn viz_invoke_end(&mut self) {
        if !self.viz.tool_announced || self.viz.param_active {
            return;
        }
        self.viz_render_read();
        self.viz_newline_if_open();
        self.viz.read_style = false;
        self.viz.read_prefix_rendered = false;
        self.viz.read_line_rendered = false;
        self.viz.read_path.clear();
        self.viz.read_start.clear();
        self.viz.read_max.clear();
        self.viz.read_whole.clear();
        self.viz.tool_announced = false;
    }

    fn viz_finish(&mut self, status: Option<&str>) {
        if !self.viz.active {
            return;
        }
        if self.viz.param_active {
            self.viz_param_end();
        }
        if status.is_none() {
            self.viz_render_read();
        }
        if let Some(status) = status {
            self.viz_newline_if_open();
            let owned = status.to_string();
            self.viz_error_puts(&owned);
        }
        self.viz_newline_if_open();
        self.viz.active = false;
    }

    /// Suppresses the rejected stanza's on-screen rendering: only the red
    /// `[invalid tool call: ...]` banner (which names the offending tag) is
    /// shown, never the raw DSML bytes. The full raw output still reaches the
    /// transcript, so failures stay debuggable there.
    fn viz_drop_invalid_dsml(&mut self) {
        if !self.viz.active {
            return;
        }
        if self.viz.param_active {
            self.viz.param_active = false;
            self.viz.param_end_tail.clear();
            self.viz.param_name.clear();
        }
        self.viz_newline_if_open();
    }

    // ---- DSML scanning ----------------------------------------------------

    /// Mirrors parser progress into the visualizer from the raw byte stream.
    fn scan_dsml_byte(&mut self, c: u8) {
        match &mut self.scan {
            DsmlScan::Between => {
                if c == b'<' {
                    self.scan = DsmlScan::Tag(vec![c]);
                }
            }
            DsmlScan::Tag(tag) => {
                tag.push(c);
                if c == b'>' {
                    let tag = std::mem::take(tag);
                    self.scan = DsmlScan::Between;
                    self.scan_dsml_tag(&tag);
                }
            }
            DsmlScan::Value => self.viz_param_value_byte(c),
        }
    }

    fn scan_dsml_tag(&mut self, tag: &[u8]) {
        let tag = String::from_utf8_lossy(tag).into_owned();
        let b = tag.as_bytes();
        if tag_prefix_len(b, true, "invoke").is_some() {
            self.viz_invoke_end();
        } else if tag_prefix_len(b, false, "invoke").is_some() {
            let name = parse_attr(&tag, "name").unwrap_or_else(|| "tool".to_string());
            self.viz_tool(&name);
        } else if tag_prefix_len(b, false, "parameter").is_some()
            && let Some(name) = parse_attr(&tag, "name")
        {
            self.viz_param_begin(&name);
            self.scan = DsmlScan::Value;
        } else if parse_attr(&tag, "name").is_none()
            && let Some(elem) = crate::dsml::element_name(&tag)
        {
            // The two shorthand forms the parser accepts, told apart the same
            // way it tells them apart: before an invoke is open a bare element
            // names the tool, inside one it names a parameter. `tool_announced`
            // is this side's copy of the parser's "an invoke is open".
            // Without mirroring it here a shorthand call runs with no banner,
            // or with its tool name rendered as a parameter.
            if self.viz.tool_announced {
                self.viz_param_begin(&elem);
                self.scan = DsmlScan::Value;
            } else {
                self.viz_tool(&elem);
            }
        }
        // Anything else is malformed; the strict parser reports it.
    }

    fn feed_dsml_byte(&mut self, c: u8) {
        let was_param = self.parser.state() == DsmlState::ParamValue;
        self.parser.feed([c]);
        if !self.dsml_ignored {
            self.scan_dsml_byte(c);
            if was_param && self.parser.state() != DsmlState::ParamValue {
                self.preflight_closed_param();
            }
        }
        match self.parser.state() {
            DsmlState::Done => {
                // The stanza parsed clean, so a `</think>` inside it really was
                // payload text: leave `in_think` as it stands (the placement
                // verdict below depends on it) and drop the pending question.
                self.think_close_swallowed = false;
                if self.rejects_in_think() {
                    // A discarded in-think stanza must never surface as an
                    // executable call: this parser instance is shared across
                    // the whole stream, so leaving `self.calls` unset here
                    // (rather than syncing it from the parser, which parsed
                    // the ignored stanza too) keeps a prior real call intact
                    // and never lets the ignored one leak into dispatch.
                    self.reject_in_think_stanza(
                        "tool calling is not allowed inside <think></think>",
                    );
                } else {
                    self.calls = self.parser.calls().to_vec();
                    self.viz_finish(None);
                    self.dsml_active = false;
                }
            }
            DsmlState::Error => {
                // Before the placement verdict: a stanza that swallowed a
                // `</think>` and then failed to parse was not inside thinking
                // when it died, so it must not be reported as if it were.
                self.resolve_swallowed_think_close();
                if self.rejects_in_think() {
                    self.reject_in_think_stanza("malformed tool call inside <think></think>");
                } else {
                    let err = if self.parser.error().is_empty() {
                        "parse error"
                    } else {
                        self.parser.error()
                    };
                    log_tool_error(err, self.parser.raw());
                    let status = format!("[invalid tool call: {err}]\n");
                    self.viz_drop_invalid_dsml();
                    self.viz_finish(Some(&status));
                    self.dsml_active = false;
                }
            }
            _ => {}
        }
    }

    /// Preflights an `edit` call as soon as its `old` parameter closes,
    /// mirroring `agent_stream_preflight_closed_param`: a selector that
    /// already fails to match means the pass is doomed, so record the error
    /// before the model wastes tokens streaming `new`.
    fn preflight_closed_param(&mut self) {
        if self.preflight_error.is_some() {
            return;
        }
        let Preflight(Some(check)) = &mut self.preflight else {
            return;
        };
        let Some(call) = self.parser.pending_call() else {
            return;
        };
        if call.name != "edit" || call.args.last().is_none_or(|a| a.name != "old") {
            return;
        }
        if let Err(err) = check(&call) {
            self.preflight_error = Some(format!(
                "edit old selector failed before new was generated: {err}"
            ));
        }
    }

    /// Starts a DSML block; the parser is seeded with canonical bytes so all
    /// later parsing stays strict even when a typo form was accepted.
    ///
    /// A stanza opening inside `<think>` is always *tracked* — whether it is
    /// allowed is decided at its stop token by [`Self::rejects_in_think`], not
    /// here. It is not *rendered* yet, though: until it leaves the thinking
    /// block it may still turn out to be the model quoting syntax, and a
    /// banner for a call that never happens is worse than a late one. Rendering
    /// starts if and when `</think>` arrives with the stanza still open.
    fn start_dsml(&mut self) {
        self.dsml_active = true;
        self.dsml_ignored = self.rejects_in_think();
        if self.in_think {
            self.dsml_in_think = true;
        }
        self.dsml_start_tail.clear();
        self.post_think_gap = false;
        self.parser.feed(DSML_START);
        self.scan = DsmlScan::Between;
        if !self.dsml_ignored {
            self.viz_start();
        }
    }

    /// Whether the stanza reaching its stop token right now must be discarded
    /// for sitting inside `<think>`.
    ///
    /// Judged at the **stop token**, against the think state at that moment —
    /// never at the opening marker. A model reasoning about DSML syntax writes
    /// an opening marker mid-thought all the time, and often goes on to close
    /// the thinking block and emit the real call (`repro-1785754509.md`).
    /// Deciding at the opening threw that correct call away and told the model
    /// to stop doing something it had not done, which reliably sent it into a
    /// rewrite loop. An opening marker is only ever a *candidate*; the model has
    /// not called a tool until the stanza closes.
    fn rejects_in_think(&self) -> bool {
        self.in_think && !self.thinking_tool_calls
    }

    /// Rejects the stanza that just closed (or was cut off) inside thinking.
    /// The banner it already streamed is closed off first, the way an invalid
    /// stanza's is: rendering is optimistic because acceptance is not known
    /// until the stop token.
    fn reject_in_think_stanza(&mut self, msg: &str) {
        self.dsml_ignored = true;
        self.viz_drop_invalid_dsml();
        self.finish_ignored_dsml(msg);
    }

    /// Whether a `</think>` at the current position is a control token rather
    /// than stanza content.
    ///
    /// Outside a stanza it always is. Inside one it is, *except* within a
    /// parameter value: a `write` or `edit` payload may legitimately contain
    /// the literal text `</think>` (this repo's own sources and docs do), and
    /// swallowing it there would silently corrupt what gets written.
    fn think_close_is_control(&self) -> bool {
        !self.dsml_active || self.parser.state() != DsmlState::ParamValue
    }

    /// Settles a `</think>` that was swallowed as parameter-value content by a
    /// stanza that then failed to parse (or never finished).
    ///
    /// Treating it as content is right for a stanza that *completes* — a
    /// `write` payload may legitimately contain the literal text. But when the
    /// stanza dies malformed or the stream ends mid-flight, there is no payload
    /// to protect and the far likelier reading is that it was the control token
    /// all along. Without this the swallowed token left `in_think` stuck true,
    /// so every remaining byte of the answer was rendered as thinking, hidden
    /// from the visible transcript, and `ended_in_think` was reported wrongly.
    fn resolve_swallowed_think_close(&mut self) {
        if !std::mem::take(&mut self.think_close_swallowed) || !self.in_think {
            return;
        }
        self.in_think = false;
        // Same boundary bookkeeping as the control path in `stream_text`: the
        // answer region starts here even though no `\n` byte crossed it.
        self.pseudo_tool.reset_line();
        self.plain_dsml = MarkerDetector::default();
        // And the same un-holding of the stanza: it opened inside thinking, so
        // its banner was held back pending the placement verdict. Thinking is
        // over, so the call is real — render what there is of it rather than
        // reporting it as a call made inside a thought.
        if self.dsml_active && self.dsml_ignored {
            self.dsml_ignored = false;
            self.viz_start();
        }
    }

    fn finish_ignored_dsml(&mut self, msg: &str) {
        // The parser buffer is often already drained by the time a rejection
        // lands, which left the most frequent failure in the log recorded with
        // an empty payload. Fall back to the held opener bytes so every entry
        // carries some evidence of what was rejected.
        let raw = if self.parser.raw().is_empty() {
            self.dsml_start_tail.clone()
        } else {
            self.parser.raw().to_vec()
        };
        log_tool_error(msg, &raw);
        self.resolve_swallowed_think_close();
        self.dsml_in_think = true;
        self.in_think_rejected = true;
        self.dsml_in_think_reported = true;
        self.viz_newline_if_open();
        let line = format!("[tool call ignored: {msg}]\n");
        self.viz_error_puts(&line);
        self.parser.reset();
        self.dsml_active = false;
        self.dsml_ignored = false;
    }

    fn malformed_dsml(&mut self, msg: &str) {
        if self.stream_error.is_some() {
            return;
        }
        self.stream_error = Some(msg.to_string());
        log_tool_error(msg, self.parser.raw());
        self.viz_newline_if_open();
        let line = format!("[invalid tool call: {msg}]\n");
        self.viz_error_puts(&line);
    }

    fn pseudo_tool_call(&mut self, opener: &str) {
        self.pseudo_tool_fired = true;
        self.malformed_dsml(&format!(
            "{opener} is not a tool call; tools are invoked with the DSML syntax in the system prompt"
        ));
    }

    fn output_frozen(&self) -> bool {
        self.freeze_on_error
            && (self.stream_error.is_some() || self.parser.state() == DsmlState::Error)
    }

    fn note_thinking_dsml_byte(&mut self, c: u8) {
        if !self.in_think || self.dsml_in_think {
            return;
        }
        if self.think_dsml.feed(c) {
            self.dsml_in_think = true;
        }
    }

    fn note_plain_dsml_byte(&mut self, c: u8) {
        // Gated on `in_think_rejected`, NOT on `dsml_in_think`. Both are sticky,
        // but they mean different things: `dsml_in_think` is set the moment
        // DSML-*shaped* bytes appear inside `<think>`, which is what a model
        // reasoning about the syntax does, and letting that disable the plain
        // validator for the rest of the stream meant one quoted marker in
        // thinking bought the model silent impunity for genuinely malformed
        // markup in its answer. `in_think_rejected` means a stanza was actually
        // rejected and the model already has a tool error in hand, so a second
        // report would be noise.
        if self.output_frozen() || self.dsml_active || self.in_think || self.in_think_rejected {
            return;
        }
        if self.plain_dsml.feed(c) {
            self.malformed_dsml("DSML markup outside a valid tool_calls block");
        }
    }

    fn note_pseudo_tool_byte(&mut self, c: u8) {
        // Deliberately NOT gated on `dsml_in_think`: that flag is sticky for
        // the whole stream, so a generation in which the model tried DSML
        // inside <think> and then fell back to `<task>` XML in its answer —
        // the exact issue #51 scenario — would go unreported.
        //
        // NOT gated on `in_think` either: the byte still has to reach the
        // detector's fence tracker so a fence opened inside thinking keeps
        // correct open/close parity across the `</think>` boundary (see
        // `PseudoToolDetector::feed`). `in_think` is passed through instead
        // so tag matching, and thus reporting, is still skipped while
        // thinking.
        if self.output_frozen() || self.dsml_active || self.replay {
            return;
        }
        if let Some(opener) = self.pseudo_tool.feed(c, self.in_think) {
            self.pseudo_tool_call(&opener);
        }
    }

    /// Matches the last line at end of stream: a hallucinated tag as the final
    /// line, with no trailing newline, is still a hallucinated call.
    fn flush_pseudo_tool(&mut self) {
        if self.output_frozen() || self.dsml_active || self.replay {
            return;
        }
        if let Some(opener) = self.pseudo_tool.finish_line(self.in_think) {
            self.pseudo_tool_call(&opener);
        }
    }

    fn flush_start_tail(&mut self) {
        if self.dsml_start_tail.is_empty() {
            return;
        }
        self.post_think_gap = false;
        let held = std::mem::take(&mut self.dsml_start_tail);
        for b in held {
            self.write_char(b);
            self.note_plain_dsml_byte(b);
            self.note_pseudo_tool_byte(b);
            if self.output_frozen() {
                break;
            }
        }
    }

    /// Routes an ordinary byte to rendering or into the DSML start detector.
    ///
    /// The detector must hold short prefixes because the model can split
    /// `<｜DSML｜tool_calls>` across arbitrary tokens.
    fn normal_byte(&mut self, c: u8) {
        if self.output_frozen() {
            return;
        }
        self.note_thinking_dsml_byte(c);

        // Swallow the visual whitespace gap the model emits right after
        // `</think>`; normal rendering resumes at the first non-space byte.
        if self.post_think_gap && matches!(c, b' ' | b'\t' | b'\r' | b'\n') {
            return;
        }

        if !self.dsml_start_tail.is_empty() || c == b'<' {
            if self.dsml_start_tail.len() < DSML_START_TAIL_CAP {
                self.dsml_start_tail.push(c);
            }
            let (mut complete, mut implicit_invoke) = (false, false);
            if dsml_start_match(&self.dsml_start_tail, &mut complete, &mut implicit_invoke) {
                if complete {
                    // Parity mode discards an in-think stanza; otherwise it is
                    // an ordinary tool call that happens to sit inside a thought.
                    self.start_dsml();
                    if implicit_invoke {
                        for &b in CANONICAL_INVOKE {
                            self.feed_dsml_byte(b);
                        }
                    }
                }
                return;
            }
            // The mismatching byte may itself start a new marker: flush all
            // but the trailing '<' and keep matching from there.
            if self.dsml_start_tail.len() > 1 && self.dsml_start_tail.last() == Some(&b'<') {
                self.post_think_gap = false;
                let held = std::mem::take(&mut self.dsml_start_tail);
                for &b in &held[..held.len() - 1] {
                    self.write_char(b);
                    self.note_plain_dsml_byte(b);
                    self.note_pseudo_tool_byte(b);
                    if self.output_frozen() {
                        return;
                    }
                }
                self.dsml_start_tail.push(b'<');
                return;
            }
            self.flush_start_tail();
            return;
        }

        self.post_think_gap = false;
        self.write_char(c);
        self.note_plain_dsml_byte(c);
        self.note_pseudo_tool_byte(c);
    }

    /// The single streaming display state machine for assistant output.
    fn stream_text(&mut self, text: &[u8], finish: bool) {
        let mut buf = std::mem::take(&mut self.pending);
        buf.extend_from_slice(text);

        let mut i = 0;
        while i < buf.len() {
            let rem = &buf[i..];
            if !self.dsml_active && rem.starts_with(THINK_OPEN) {
                self.flush_start_tail();
                self.post_think_gap = false;
                self.in_think = true;
                // The plain-DSML tail is not fed while thinking, so bytes from
                // before the block must not glue onto bytes from after it and
                // spell a marker that was never written.
                self.plain_dsml = MarkerDetector::default();
                i += THINK_OPEN.len();
                continue;
            }
            if rem.starts_with(THINK_CLOSE) && self.think_close_is_control() {
                if self.dsml_active {
                    // The model closed its thought part-way through a stanza.
                    // `</think>` is a control token, never stanza content, so
                    // it is consumed without reaching the parser and the call
                    // carries on — now outside thinking, so its stop token
                    // will accept it. The call is real after all, so start
                    // rendering it if the opener was held back.
                    self.in_think = false;
                    if self.dsml_ignored {
                        self.dsml_ignored = false;
                        self.viz_start();
                    }
                } else {
                    self.flush_start_tail();
                    self.in_think = false;
                    self.pseudo_tool.reset_line();
                    self.viz_newline_if_open();
                    self.emit_visible_bytes(b"\n");
                    self.post_think_gap = true;
                }
                self.plain_dsml = MarkerDetector::default();
                i += THINK_CLOSE.len();
                continue;
            }
            if rem.starts_with(THINK_CLOSE) {
                // Not a control token here (the branch above owns that case):
                // it is inside a parameter value and is about to be fed to the
                // parser as content. Remember it, in case the stanza never
                // completes — see `resolve_swallowed_think_close`.
                self.think_close_swallowed = true;
            }
            if !finish
                && rem[0] == b'<'
                && self.think_close_is_control()
                && (is_partial_prefix(rem, THINK_OPEN) || is_partial_prefix(rem, THINK_CLOSE))
            {
                self.pending = rem.to_vec();
                break;
            }

            let c = rem[0];
            if self.dsml_active {
                self.feed_dsml_byte(c);
            } else {
                // In-think bytes still flow through the DSML start detector so
                // an accidental in-think tool stanza is suppressed cleanly.
                self.normal_byte(c);
            }
            i += 1;
        }

        if finish {
            self.flush_start_tail();
            self.post_think_gap = false;
            if self.dsml_active {
                // A stanza cut off after swallowing a `</think>` never got to
                // prove the token was payload text, so it is settled as the
                // control token it most likely was — before the placement
                // verdict below reads `in_think`.
                self.resolve_swallowed_think_close();
                if self.rejects_in_think() {
                    // Cut off mid-stanza and still inside thinking: the model
                    // never left the block, so the placement rule applies.
                    self.reject_in_think_stanza(
                        "tool calling is not allowed inside <think></think>",
                    );
                } else {
                    self.viz_finish(Some(if self.preflight_error.is_some() {
                        "[tool call stopped: edit old selector failed]\n"
                    } else {
                        "[tool call interrupted]\n"
                    }));
                    self.dsml_active = false;
                }
            }
            // Deliberately nothing here for `dsml_in_think` alone. That flag is
            // set by `note_thinking_dsml_byte` the moment DSML-shaped bytes
            // appear inside `<think>`, which happens whenever the model reasons
            // *about* the syntax — quoting a marker is not calling a tool. It
            // used to raise the prohibition at finish, so a model explaining its
            // own tool-call format got a tool error back and rewrote correct
            // syntax. Only a stanza that reaches its stop token (or is cut off
            // mid-flight) is a call, and both are handled above.
        }
    }
}

fn is_partial_prefix(bytes: &[u8], prefix: &[u8]) -> bool {
    bytes.len() < prefix.len() && prefix[..bytes.len()] == *bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default)]
    struct Cap {
        visible: String,
        think: String,
        errors: String,
    }

    impl RenderSink for Cap {
        fn visible_text(&mut self, text: &str) {
            self.visible.push_str(text);
        }
        fn error_text(&mut self, text: &str) {
            self.errors.push_str(text);
            self.visible.push_str(text);
        }
        fn think_text(&mut self, text: &str) {
            self.think.push_str(text);
        }
    }

    fn run_chunked(text: &str) -> StreamRenderer<Cap> {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.push(text);
        sr.finish();
        sr
    }

    fn pseudo_tool_renderer() -> StreamRenderer<Cap> {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.set_tool_names(vec!["task".to_string(), "read".to_string()]);
        sr
    }

    // Issue #51: the model invents <task> XML for tools it was not trained on.
    // Nothing recognized it, so the turn ended with no tool call and no error and
    // the model retried forever.
    #[test]
    fn pseudo_tool_block_after_think_is_reported() {
        let mut sr = pseudo_tool_renderer();
        sr.push("<think>planning</think>");
        sr.push("<task>\ntask_context: \"add headers\"\n</task>");
        sr.finish();
        assert!(
            sr.finished().error.is_some(),
            "hallucinated call produced no error for the model to correct from"
        );
    }

    // While thinking, the model muses about markup; firing there would punish it
    // for reasoning. Only the answer region counts.
    #[test]
    fn pseudo_tool_block_inside_think_is_ignored() {
        let mut sr = pseudo_tool_renderer();
        sr.push("<think>\n<task>\n</task>\n</think>");
        sr.push("done");
        sr.finish();
        assert!(sr.finished().error.is_none());
    }

    // A fence opened inside <think> IS observed by the detector: every byte
    // reaches the fence tracker, and only tag matching is gated on `!in_think`.
    // That is what makes its matching close in the answer region read as a
    // close rather than a fresh opener that would silence the detector for the
    // rest of the stream. Issue #51 case: the model tries DSML-like
    // markup inside thinking, then falls back to `<task>` XML in its answer.
    #[test]
    fn stray_fence_close_leaking_from_think_does_not_disarm_the_detector() {
        let mut sr = pseudo_tool_renderer();
        sr.push("<think>```rust\nfoo\n</think>```\n<task>\nx\n</task>");
        sr.finish();
        assert!(
            sr.finished().error.is_some(),
            "a fence opened inside <think> disarmed the detector for the rest of the stream"
        );
    }

    // Discussing the markup in prose or code is not attempting to call it.
    #[test]
    fn pseudo_tool_name_in_prose_or_fence_is_ignored() {
        let mut sr = pseudo_tool_renderer();
        sr.push("the <task> element is XML, not DSML\n");
        sr.push("```xml\n<task>\n</task>\n```\n");
        sr.finish();
        assert!(sr.finished().error.is_none());
    }

    // A pseudo-tool hit happens by construction in the ANSWER region, after
    // `</think>`. Reporting the in-think prohibition instead tells the model to
    // move a call it never made, which is the misdiagnosis this detector exists
    // to end.
    #[test]
    fn pseudo_tool_error_wins_over_the_in_think_prohibition() {
        let mut sr = pseudo_tool_renderer();
        sr.push(concat!(
            "<think><｜DSML｜tool_calls><｜DSML｜invoke name=\"bash\">",
            "</｜DSML｜invoke｜></｜DSML｜tool_calls｜></think>\n",
            "<task>\nx\n</task>",
        ));
        sr.finish();
        let fin = sr.finished();
        assert_eq!(
            fin.error,
            Some(
                "<task> is not a tool call; tools are invoked with the DSML syntax in the system prompt"
            ),
            "pseudo-tool hit must not be masked by the in-think prohibition"
        );
        assert!(
            !fin.in_think_rejected,
            "in-think framing would wrap the answer-region error in the wrong advice"
        );
    }

    // The other direction: a genuinely leaked in-think stanza with no
    // pseudo-tool hit still reports the prohibition.
    #[test]
    fn leaked_in_think_stanza_alone_still_reports_the_prohibition() {
        let mut sr = pseudo_tool_renderer();
        sr.push(concat!(
            "<think><｜DSML｜tool_calls><｜DSML｜invoke name=\"bash\">",
            "</｜DSML｜invoke｜></｜DSML｜tool_calls｜></think>\n",
            "all done\n",
        ));
        sr.finish();
        let fin = sr.finished();
        assert_eq!(fin.error, Some(IN_THINK_PROHIBITION));
        assert!(fin.in_think_rejected);
    }

    // A line that opens with a tool tag and then continues in prose is prose:
    // firing there both fabricates a tool error and truncates a legitimate
    // answer, because a stream error freezes output.
    #[test]
    fn prose_continuing_after_a_tool_tag_does_not_fire() {
        let mut sr = pseudo_tool_renderer();
        sr.push("<read> is how you spell it, unlike the others.\nand more text\n");
        sr.finish();
        assert!(sr.finished().error.is_none(), "{:?}", sr.finished().error);
        assert!(
            sr.sink().visible.contains("and more text"),
            "answer truncated: {:?}",
            sr.sink().visible
        );
    }

    // A line that is nothing but the tag is still a hallucinated call.
    #[test]
    fn bare_tool_tag_on_its_own_line_fires() {
        let mut sr = pseudo_tool_renderer();
        sr.push("here goes\n<read>\npath: /tmp/x\n</read>\n");
        sr.finish();
        assert!(sr.finished().error.is_some());
    }

    // ... including when the stream ends without a trailing newline.
    #[test]
    fn bare_tool_tag_as_final_line_without_newline_fires() {
        let mut sr = pseudo_tool_renderer();
        sr.push("here goes\n<task>");
        sr.finish();
        assert_eq!(
            sr.finished().error,
            Some(
                "<task> is not a tool call; tools are invoked with the DSML syntax in the system prompt"
            )
        );
    }

    // The model-facing text names the line it actually saw.
    #[test]
    fn pseudo_tool_message_quotes_the_matched_line() {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.push("<invoke name=\"bash\">\n");
        sr.finish();
        assert_eq!(
            sr.finished().error,
            Some(
                "<invoke name=\"bash\"> is not a tool call; tools are invoked with the DSML syntax in the system prompt"
            )
        );
    }

    // Generic wrappers are matched with no tool list configured at all.
    #[test]
    fn generic_tool_call_wrapper_is_reported() {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.push("<tool_call>\n{\"name\": \"read\"}\n</tool_call>");
        sr.finish();
        assert!(sr.finished().error.is_some());
    }

    // Replaying stored transcript text must not produce new diagnostics: the
    // text was already diagnosed when it was first produced.
    #[test]
    fn pseudo_tool_block_is_not_reported_during_replay() {
        let mut sr = pseudo_tool_renderer();
        sr.set_replay(true);
        sr.push("<task>\ntask_context: \"add headers\"\n</task>");
        sr.finish();
        assert!(sr.finished().error.is_none());
    }

    // A real DSML call must never be mistaken for a hallucination.
    #[test]
    fn real_dsml_call_is_untouched_by_the_pseudo_detector() {
        let mut sr = pseudo_tool_renderer();
        sr.push(
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"read\">\
             <｜DSML｜parameter name=\"path\" string=\"true\">/tmp/x</｜DSML｜parameter｜>\
             </｜DSML｜invoke｜></｜DSML｜tool_calls｜>",
        );
        sr.finish();
        let fin = sr.finished();
        assert!(fin.error.is_none(), "error: {:?}", fin.error);
        assert_eq!(fin.calls.len(), 1);
        assert_eq!(fin.calls[0].name, "read");
    }

    #[test]
    fn write_content_previews_dim_even_with_banners_off() {
        let stanza = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"write\">",
            "<｜DSML｜parameter name=\"path\">src/foo.rs</｜DSML｜parameter>",
            "<｜DSML｜parameter name=\"content\">fn main() {}\n</｜DSML｜parameter>",
            "</｜DSML｜invoke>",
            "</｜DSML｜tool_calls>",
        );
        // Banners off (default): the content still previews, on the think
        // (dim) channel, with a header naming the file — nothing in visible.
        let mut sr = StreamRenderer::new(Cap::default());
        sr.set_show_tool_calls(false);
        sr.push(stanza);
        sr.finish();
        let think = &sr.sink().think;
        assert!(think.contains("write src/foo.rs"), "header: {think:?}");
        assert!(think.contains("fn main() {}"), "content preview: {think:?}");
        assert!(
            !sr.sink().visible.contains("fn main()"),
            "content not on the visible channel: {:?}",
            sr.sink().visible
        );
        assert_eq!(sr.finished().calls.len(), 1, "call still parsed");
    }

    #[test]
    fn show_write_preview_false_drops_the_content_preview_entirely() {
        let stanza = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"write\">",
            "<｜DSML｜parameter name=\"path\">src/foo.rs</｜DSML｜parameter>",
            "<｜DSML｜parameter name=\"content\">fn main() {}\n</｜DSML｜parameter>",
            "</｜DSML｜invoke>",
            "</｜DSML｜tool_calls>",
        );
        // The /init wiring: banners and preview both off. Nothing reaches any
        // channel, but the call is still parsed so the file still gets written.
        let mut sr = StreamRenderer::new(Cap::default());
        sr.set_show_tool_calls(false);
        sr.set_show_write_preview(false);
        sr.push(stanza);
        sr.finish();
        assert!(sr.sink().think.is_empty(), "think: {:?}", sr.sink().think);
        assert!(
            !sr.sink().visible.contains("fn main()"),
            "visible: {:?}",
            sr.sink().visible
        );
        assert_eq!(sr.finished().calls.len(), 1, "call still parsed");
    }

    #[test]
    fn show_tool_calls_false_suppresses_the_banner_but_keeps_visible_text() {
        let stanza = concat!(
            "answer before. ",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\">ls -la</｜DSML｜parameter>",
            "</｜DSML｜invoke>",
            "</｜DSML｜tool_calls>",
        );
        // Default: banner shows.
        let shown = run_chunked(stanza);
        assert!(
            shown.sink().visible.contains("🛠️"),
            "{:?}",
            shown.sink().visible
        );

        // Gated off: no banner, no `ls -la`, but the model's prose survives and
        // the tool call is still parsed (so it would still execute).
        let mut sr = StreamRenderer::new(Cap::default());
        sr.set_show_tool_calls(false);
        sr.push(stanza);
        sr.finish();
        let vis = &sr.sink().visible;
        assert!(vis.contains("answer before."), "prose kept: {vis:?}");
        assert!(!vis.contains("🛠️"), "banner suppressed: {vis:?}");
        assert!(!vis.contains("ls -la"), "params suppressed: {vis:?}");
        assert_eq!(sr.finished().calls.len(), 1, "call still parsed");
    }

    #[test]
    fn show_thinking_false_suppresses_thinking_but_keeps_visible_text() {
        let text = "<think>secret reasoning</think>visible answer";

        // Default: thinking is emitted to the think channel.
        let shown = run_chunked(text);
        assert_eq!(shown.sink().think, "secret reasoning");
        assert_eq!(shown.sink().visible.trim(), "visible answer");

        // Gated off: nothing on the think channel, prose unaffected (a leading
        // post-think separator newline may remain).
        let mut sr = StreamRenderer::new(Cap::default());
        sr.set_show_thinking(false);
        sr.push(text);
        sr.finish();
        assert_eq!(sr.sink().think, "", "thinking suppressed");
        assert_eq!(sr.sink().visible.trim(), "visible answer", "prose kept");
    }

    fn run_charwise(text: &str) -> StreamRenderer<Cap> {
        let mut sr = StreamRenderer::new(Cap::default());
        for ch in text.chars() {
            sr.push(ch.to_string());
        }
        sr.finish();
        sr
    }

    const BASH_STANZA: &str = concat!(
        "<｜DSML｜tool_calls>",
        "<｜DSML｜invoke name=\"bash\">",
        "<｜DSML｜parameter name=\"command\">ls -la</｜DSML｜parameter｜>",
        "</｜DSML｜invoke｜>",
        "</｜DSML｜tool_calls｜>",
    );

    #[test]
    fn begin_in_think_routes_thinking_then_answer() {
        // The chat template opens <think> in the prefill prefix, so generation
        // streams thinking first and closes with a real </think> token.
        let mut sr = StreamRenderer::new(Cap::default());
        sr.begin_in_think();
        sr.push("weighing options</think>Final answer.");
        sr.finish();
        assert!(sr.sink().think.contains("weighing options"));
        assert!(sr.sink().visible.contains("Final answer."));
        assert!(!sr.sink().visible.contains("weighing options"));
        assert!(!sr.sink().visible.contains("think"));
    }

    #[test]
    fn provider_explicit_think_tags_route_correctly() {
        // Provider engines do NOT begin_in_think: their translator emits its own
        // <think>/</think> tags around reasoning. A turn that opens, reasons, and
        // closes must route cleanly, and — the regression that turned visible
        // output gray — a turn with NO reasoning must stay fully visible.
        let mut sr = StreamRenderer::new(Cap::default());
        sr.push("<think>reasoning here</think>");
        sr.push("visible answer");
        sr.finish();
        assert!(sr.sink().think.contains("reasoning here"));
        assert!(!sr.sink().think.contains("visible answer"));
        assert!(sr.sink().visible.contains("visible answer"));

        // No reasoning at all: content is emitted directly and stays visible
        // (with begin_in_think this would have been misclassified as thinking).
        let mut sr = StreamRenderer::new(Cap::default());
        sr.push("just an answer, no thinking");
        sr.finish();
        assert_eq!(sr.sink().visible, "just an answer, no thinking");
        assert_eq!(sr.sink().think, "");
    }

    /// Repro `~/.plank/repro/repro-1785161356.md`: mid-session the model wrote
    /// `<｜SSML｜…>` for the whole stanza. Every other byte was correct, but the
    /// call parsed as nothing, printed raw, and the turn ended with no tool
    /// error — so the model could not even retry. `MARKER_NAMES` accepts SSML
    /// as an alias; the stanza must dispatch exactly like the DSML spelling.
    #[test]
    fn ssml_misspelling_is_accepted_as_an_alias() {
        let text = concat!(
            "Let me look at the documents module.\n",
            "<｜SSML｜tool_calls>\n",
            "<｜SSML｜invoke name=\"bash\">\n",
            "<｜SSML｜parameter name=\"command\" string=\"true\">cat documents.rs",
            "</｜SSML｜parameter>\n",
            "</｜SSML｜invoke>\n",
            "</｜SSML｜tool_calls>",
        );
        for sr in [run_chunked(text), run_charwise(text)] {
            let vis = &sr.sink().visible;
            assert!(vis.contains("🛠️ $ cat documents.rs"), "{vis:?}");
            assert!(!vis.contains("SSML"), "{vis:?}");
            let fin = sr.finished();
            assert_eq!(fin.calls.len(), 1);
            assert_eq!(fin.calls[0].name, "bash");
            assert_eq!(fin.calls[0].arg_value("command"), Some("cat documents.rs"));
            assert!(fin.error.is_none(), "{:?}", fin.error);
        }
    }

    /// The alias is per-tag, not per-stanza: the drift is a sampling slip on a
    /// spelled-out marker, so it can hit one tag and not the next.
    #[test]
    fn dsml_and_ssml_tags_mix_within_one_stanza() {
        let text = concat!(
            "<｜DSML｜tool_calls>",
            "<｜SSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\">ls -la</｜SSML｜parameter>",
            "</｜DSML｜invoke>",
            "</｜SSML｜tool_calls>",
        );
        for sr in [run_chunked(text), run_charwise(text)] {
            let fin = sr.finished();
            assert_eq!(fin.calls.len(), 1);
            assert_eq!(fin.calls[0].arg_value("command"), Some("ls -la"));
            assert!(fin.error.is_none(), "{:?}", fin.error);
        }
    }

    /// The alias must not widen to any four letters: only the one misspelling
    /// the model actually produces is recovered, so unrelated markup in prose
    /// still passes through as text rather than becoming a tool call.
    #[test]
    fn unrelated_marker_names_are_still_plain_text() {
        let text = "<｜XSML｜tool_calls><｜XSML｜invoke name=\"bash\">";
        for sr in [run_chunked(text), run_charwise(text)] {
            assert!(sr.finished().calls.is_empty());
            assert_eq!(sr.sink().visible, text);
        }
    }

    #[test]
    fn prose_passes_through() {
        for sr in [run_chunked("Hello, world."), run_charwise("Hello, world.")] {
            assert_eq!(sr.sink().visible, "Hello, world.");
            assert_eq!(sr.sink().think, "");
            assert!(sr.finished().calls.is_empty());
            assert!(sr.finished().error.is_none());
        }
    }

    #[test]
    fn bash_stanza_hides_dsml_and_shows_banner() {
        let text = format!("Let me look.\n{BASH_STANZA}");
        for sr in [run_chunked(&text), run_charwise(&text)] {
            let vis = &sr.sink().visible;
            assert!(vis.starts_with("Let me look.\n"), "{vis:?}");
            assert!(vis.contains("🛠️ $ ls -la"), "{vis:?}");
            assert!(!vis.contains("DSML"), "{vis:?}");
            let fin = sr.finished();
            assert_eq!(fin.calls.len(), 1);
            assert_eq!(fin.calls[0].name, "bash");
            assert_eq!(fin.calls[0].arg_value("command"), Some("ls -la"));
            assert!(fin.error.is_none());
        }
    }

    #[test]
    fn read_banner_shows_path_and_range() {
        let stanza = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"read\">",
            "<｜DSML｜parameter name=\"path\" string=\"true\">src/main.rs</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        for sr in [run_chunked(stanza), run_charwise(stanza)] {
            let vis = &sr.sink().visible;
            assert!(vis.contains("🛠️ Reading src/main.rs 1:500...\n"), "{vis:?}");
            assert!(!vis.contains("DSML"), "{vis:?}");
        }
    }

    #[test]
    fn read_banner_whole_file() {
        let stanza = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"read\">",
            "<｜DSML｜parameter name=\"path\" string=\"true\">a.c</｜DSML｜parameter｜>",
            "<｜DSML｜parameter name=\"whole\">true</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        let sr = run_chunked(stanza);
        assert!(
            sr.sink()
                .visible
                .contains("🛠️ Reading a.c (whole file)...\n"),
            "{:?}",
            sr.sink().visible
        );
    }

    #[test]
    fn edit_diff_uses_minus_plus_prefixes() {
        let stanza = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"edit\">",
            "<｜DSML｜parameter name=\"path\" string=\"true\">a.rs</｜DSML｜parameter｜>",
            "<｜DSML｜parameter name=\"old\">let a = 1;</｜DSML｜parameter｜>",
            "<｜DSML｜parameter name=\"new\">let a = 2;</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        for sr in [run_chunked(stanza), run_charwise(stanza)] {
            let vis = &sr.sink().visible;
            assert!(vis.contains("🛠️ edit  path=a.rs"), "{vis:?}");
            assert!(vis.contains("- let a = 1;"), "{vis:?}");
            assert!(vis.contains("+ let a = 2;"), "{vis:?}");
            assert!(!vis.contains("DSML"), "{vis:?}");
            assert_eq!(sr.finished().calls[0].arg_value("new"), Some("let a = 2;"));
        }
    }

    #[test]
    fn partial_marker_false_alarm_is_flushed() {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.push("<｜DSM");
        // Nothing shown while the prefix is still ambiguous.
        assert_eq!(sr.sink().visible, "");
        sr.push("ok");
        sr.finish();
        assert_eq!(sr.sink().visible, "<｜DSMok");
        assert!(sr.finished().error.is_none());
    }

    #[test]
    fn partial_marker_flushed_at_stream_end() {
        // A held-back prefix containing a complete loose marker is flushed at
        // end of stream and then flagged by the plain-marker detector, exactly
        // as in the C reference.
        let mut sr = StreamRenderer::new(Cap::default());
        sr.push("done <｜DSML｜tool_c");
        sr.finish();
        assert!(
            sr.sink().visible.starts_with("done <｜DSML｜"),
            "{:?}",
            sr.sink().visible
        );
        assert!(
            sr.sink().visible.contains("[invalid tool call: "),
            "{:?}",
            sr.sink().visible
        );
    }

    #[test]
    fn think_text_routes_to_think_sink() {
        let sr = run_chunked("<think>pondering</think>Answer.");
        assert_eq!(sr.sink().think, "pondering");
        assert!(
            sr.sink().visible.ends_with("Answer."),
            "{:?}",
            sr.sink().visible
        );
        assert!(!sr.sink().visible.contains("pondering"));
    }

    #[test]
    fn think_tag_split_across_pushes() {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.push("<th");
        sr.push("ink>hidden</th");
        sr.push("ink>shown");
        sr.finish();
        assert_eq!(sr.sink().think, "hidden");
        assert!(
            sr.sink().visible.ends_with("shown"),
            "{:?}",
            sr.sink().visible
        );
    }

    #[test]
    fn dsml_inside_think_is_ignored_and_reported() {
        let text = format!("<think>{BASH_STANZA}</think>ok");
        for sr in [run_chunked(&text), run_charwise(&text)] {
            let fin = sr.finished();
            assert!(fin.dsml_in_think);
            // An ignored stanza must never surface as an executable call:
            // if `feed_dsml_byte`'s `Done` arm ever syncs `self.calls` before
            // checking `dsml_ignored` again, this call would leak through and
            // the turn loop would dispatch it despite the "ignored" notice.
            assert!(fin.calls.is_empty(), "{:?}", fin.calls);
            assert!(
                sr.sink().visible.contains(
                    "[tool call ignored: tool calling is not allowed inside <think></think>]"
                ),
                "{:?}",
                sr.sink().visible
            );
            assert!(!sr.sink().think.contains("DSML"), "{:?}", sr.sink().think);
        }
    }

    fn run_allowing_in_think(text: &str) -> StreamRenderer<Cap> {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.set_thinking_tool_calls(true);
        sr.push(text);
        sr.finish();
        sr
    }

    fn run_allowing_in_think_charwise(text: &str) -> StreamRenderer<Cap> {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.set_thinking_tool_calls(true);
        for ch in text.chars() {
            sr.push(ch.to_string());
        }
        sr.finish();
        sr
    }

    /// The model stopped mid-stanza inside `<think>`: the parser's own verdict
    /// would be "incomplete DSML tool call", which is true and useless — the
    /// markup was cut off only because the call had no business being there.
    /// The reported error is the placement rule.
    #[test]
    fn an_unfinished_in_think_stanza_reports_placement_not_syntax() {
        let mut sr = StreamRenderer::new(Cap::default());
        // Opens a stanza inside thinking and stops: no closing tags.
        sr.push("<think>let me look<｜DSML｜tool_calls><｜DSML｜invoke name=\"bash\">");
        sr.finish();
        let fin = sr.finished();
        assert!(fin.calls.is_empty(), "{:?}", fin.calls);
        assert!(fin.in_think_rejected, "rejected for placement");
        assert_eq!(fin.error, Some(IN_THINK_PROHIBITION));
        assert!(fin.ended_in_think);
    }

    /// The verdict belongs at the stanza's *stop* token, not its opening.
    ///
    /// From `repro-1785754509.md`: the model was reasoning about the correct
    /// syntax, wrote an opening `<｜DSML｜tool_calls>` as part of that thought,
    /// then closed the thinking block and emitted the real call. Judging at the
    /// opening threw the whole (correct, post-`</think>`) call away and told
    /// the model to stop calling tools inside thinking — which it had not done.
    #[test]
    fn a_stanza_opened_in_think_but_closed_after_it_is_dispatched() {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.push("<think>the correct format is:\n\n<｜DSML｜tool_calls>\n</think>\n\n");
        sr.push("<｜DSML｜invoke name=\"bash\">");
        sr.push("<｜DSML｜parameter name=\"command\">ls -la</｜DSML｜parameter｜>");
        sr.push("</｜DSML｜invoke｜></｜DSML｜tool_calls｜>");
        sr.finish();
        let fin = sr.finished();
        assert_eq!(fin.calls.len(), 1, "{:?}", fin.calls);
        assert_eq!(fin.calls[0].arg_value("command"), Some("ls -la"));
        assert!(!fin.in_think_rejected, "the call closed outside thinking");
        assert!(fin.error.is_none(), "{:?}", fin.error);
        assert!(
            !sr.sink().visible.contains("[tool call ignored:"),
            "{:?}",
            sr.sink().visible
        );
    }

    /// Merely *mentioning* the markup while reasoning is not a tool call and
    /// must not poison the turn: with no stanza and no stop token there is
    /// nothing to block. The marker detector alone used to raise the
    /// prohibition at finish, so a model recalling its own syntax mid-thought
    /// got a tool error back and went off rewriting correct markup.
    #[test]
    fn dsml_mentioned_while_thinking_without_a_stop_token_is_not_rejected() {
        for text in [
            // A parameter line quoted mid-thought: DSML-shaped, but not an
            // opener, so no stanza is ever tracked.
            "<think>each arg is <｜DSML｜parameter name=\"x\" string=\"true\">v</｜DSML｜parameter｜>              inside the invoke</think>the answer",
            // A bare closing tag, the other half of the same recollection.
            "<think>and it ends with </｜DSML｜tool_calls｜> of course</think>the answer",
        ] {
            let sr = run_chunked(text);
            let fin = sr.finished();
            assert!(fin.calls.is_empty(), "{:?}", fin.calls);
            assert!(!fin.in_think_rejected, "nothing completed; {text}");
            assert_eq!(fin.error, None, "{text}");
            // The marker was still *seen* in thinking, which is reported as
            // information; it just is not an error any more.
            assert!(fin.dsml_in_think, "{text}");
            assert!(sr.sink().visible.contains("the answer"), "{text}");
        }
    }

    /// `</think>` is only a control token where it cannot be data. Inside a
    /// parameter value it is payload — this repo's own sources and docs contain
    /// the literal text — so it must reach the parser untouched. Swallowing it
    /// there would silently corrupt every file written about thinking blocks.
    #[test]
    fn think_close_inside_a_parameter_value_is_payload_not_a_control_token() {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.push("<think>writing it up</think>");
        sr.push("<｜DSML｜tool_calls><｜DSML｜invoke name=\"write\">");
        sr.push("<｜DSML｜parameter name=\"content\">close it with </think> when done");
        sr.push("</｜DSML｜parameter｜></｜DSML｜invoke｜></｜DSML｜tool_calls｜>");
        sr.finish();
        let fin = sr.finished();
        assert_eq!(fin.calls.len(), 1, "{:?}", fin.calls);
        assert_eq!(
            fin.calls[0].arg_value("content"),
            Some("close it with </think> when done")
        );
        assert!(fin.error.is_none(), "{:?}", fin.error);
    }

    /// A stanza that both opens and closes inside thinking is still rejected —
    /// that is the trained rule — and it renders no banner on the way, since it
    /// never became a real call.
    #[test]
    fn a_stanza_wholly_inside_think_is_rejected_without_a_banner() {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.push(format!("<think>thinking{BASH_STANZA}</think>done"));
        sr.finish();
        let fin = sr.finished();
        assert!(fin.calls.is_empty(), "{:?}", fin.calls);
        assert!(fin.in_think_rejected);
        assert_eq!(fin.error, Some(IN_THINK_PROHIBITION));
        assert!(
            !sr.sink().visible.contains("🛠️"),
            "no banner for a call that never happened: {:?}",
            sr.sink().visible
        );
    }

    /// A shorthand invoke (`<｜DSML｜edit>`) dispatches, so it must also draw a
    /// banner naming the tool — a call that runs invisibly is worse than one
    /// that is refused.
    #[test]
    fn shorthand_invoke_renders_a_banner() {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.push("<｜DSML｜tool_calls><｜DSML｜bash>");
        sr.push("<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter｜>");
        sr.push("</｜DSML｜invoke｜></｜DSML｜tool_calls｜>");
        sr.finish();
        let fin = sr.finished();
        assert_eq!(fin.calls.len(), 1, "{:?}", fin.calls);
        assert_eq!(fin.calls[0].name, "bash");
        assert!(
            sr.sink().visible.contains("🛠️ $ ls -la"),
            "{:?}",
            sr.sink().visible
        );
        assert!(
            !sr.sink().visible.contains("DSML"),
            "{:?}",
            sr.sink().visible
        );
    }

    /// The parameter shorthand renders under its own name too — the same
    /// mirroring, one level down.
    #[test]
    fn shorthand_parameter_renders_under_its_element_name() {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.push("<｜DSML｜tool_calls><｜DSML｜invoke name=\"bash\">");
        sr.push("<｜DSML｜command string=\"true\">ls -la</｜DSML｜invoke>");
        sr.push("</｜DSML｜invoke｜></｜DSML｜tool_calls｜>");
        sr.finish();
        let fin = sr.finished();
        assert_eq!(fin.calls.len(), 1, "{:?}", fin.calls);
        assert_eq!(fin.calls[0].arg_value("command"), Some("ls -la"));
        assert!(
            sr.sink().visible.contains("🛠️ $ ls -la"),
            "{:?}",
            sr.sink().visible
        );
    }

    /// A completed stanza inside `<think>` reports the same placement error,
    /// rather than falling through to whatever the parser concluded.
    #[test]
    fn a_completed_in_think_stanza_reports_placement() {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.push(format!("<think>thinking{BASH_STANZA}"));
        sr.finish();
        let fin = sr.finished();
        assert!(fin.calls.is_empty(), "the call must not be dispatched");
        assert!(fin.in_think_rejected);
        assert_eq!(fin.error, Some(IN_THINK_PROHIBITION));
    }

    /// With `engine.thinkingToolCalls` on, an in-think call is dispatched and
    /// there is nothing to report: the placement error must not leak into the
    /// allow path just because the marker was seen inside thinking.
    #[test]
    fn allowing_in_think_calls_reports_no_placement_error() {
        let text = format!("<think>{BASH_STANZA}</think>ok");
        let sr = run_allowing_in_think(&text);
        let fin = sr.finished();
        assert_eq!(fin.calls.len(), 1);
        assert!(!fin.in_think_rejected, "nothing was rejected");
        assert!(fin.error.is_none(), "{:?}", fin.error);
        assert!(fin.dsml_in_think, "the marker was still seen in thinking");
    }

    #[test]
    fn dsml_inside_think_is_executed_when_allowed() {
        let text = format!("<think>{BASH_STANZA}</think>ok");
        for sr in [
            run_allowing_in_think(&text),
            run_allowing_in_think_charwise(&text),
        ] {
            let fin = sr.finished();
            assert_eq!(fin.calls.len(), 1, "{:?}", fin.calls);
            assert_eq!(fin.calls[0].name, "bash");
            assert_eq!(fin.calls[0].arg_value("command"), Some("ls -la"));
            assert!(fin.error.is_none(), "{:?}", fin.error);
            assert!(
                !sr.sink().visible.contains("[tool call ignored:"),
                "{:?}",
                sr.sink().visible
            );
            // The banner renders like any other tool call, and raw DSML never
            // reaches either sink.
            assert!(
                sr.sink().visible.contains("🛠️ $ ls -la"),
                "{:?}",
                sr.sink().visible
            );
            assert!(
                !sr.sink().visible.contains("DSML"),
                "{:?}",
                sr.sink().visible
            );
            assert!(!sr.sink().think.contains("DSML"), "{:?}", sr.sink().think);
        }
    }

    #[test]
    fn ended_in_think_reports_an_open_block() {
        // A stanza fired mid-thought: the stream ends with <think> still open.
        let sr = run_allowing_in_think(&format!("<think>let me look{BASH_STANZA}"));
        assert!(sr.finished().ended_in_think);

        // A closed block, and a stream that never thought at all, both report
        // false.
        let closed = run_allowing_in_think(&format!("<think>done</think>{BASH_STANZA}"));
        assert!(!closed.finished().ended_in_think);
        assert!(!run_chunked("plain answer").finished().ended_in_think);

        // A stanza inside a think block that then closes: the DSML marker
        // sets `dsml_active` mid-block, but `</think>` still arrives before
        // the stream ends, so the block is not open at finish.
        let stanza_then_close =
            run_allowing_in_think(&format!("<think>a{BASH_STANZA}</think>tail"));
        assert!(!stanza_then_close.finished().ended_in_think);
    }

    #[test]
    fn interrupted_stanza_reports_status() {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.push("<｜DSML｜tool_calls><｜DSML｜invoke name=\"bash\">");
        sr.push("<｜DSML｜parameter name=\"command\">sleep 1");
        sr.finish();
        let vis = &sr.sink().visible;
        assert!(vis.contains("🛠️ $ sleep 1"), "{vis:?}");
        assert!(vis.contains("[tool call interrupted]\n"), "{vis:?}");
        assert!(sr.finished().calls.is_empty());
    }

    #[test]
    fn incomplete_stanza_reports_incomplete_error() {
        let sr = run_chunked(concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\">ls",
        ));
        assert_eq!(sr.finished().error, Some("incomplete DSML tool call"));
        assert!(sr.finished().calls.is_empty());
    }

    #[test]
    fn greedy_sampling_tracks_dsml_state() {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.push("hello ");
        assert!(!sr.wants_greedy_sampling(), "prose");
        sr.push("<｜DS");
        assert!(sr.wants_greedy_sampling(), "DSML-shaped held prefix");
        sr.push("ML｜tool_calls><｜DSML｜invoke name=\"bash\">");
        assert!(sr.wants_greedy_sampling(), "structural markup");
        sr.push("<｜DSML｜parameter name=\"command\">ls -la");
        assert!(!sr.wants_greedy_sampling(), "free-form parameter value");
        sr.push("</｜DSML｜parameter");
        assert!(sr.wants_greedy_sampling(), "close tag streaming");
        sr.push("｜></｜DSML｜invoke｜></｜DSML｜tool_calls｜>");
        assert!(!sr.wants_greedy_sampling(), "stanza done");
    }

    #[test]
    fn edit_old_preflight_failure_is_reported_midstream() {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.set_preflight(|call| {
            assert_eq!(call.name, "edit");
            assert_eq!(call.arg_value("path"), Some("src/a.rs"));
            Err("old text is not a unique match".to_string())
        });
        sr.push(concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"edit\">",
            "<｜DSML｜parameter name=\"path\">src/a.rs</｜DSML｜parameter｜>",
            "<｜DSML｜parameter name=\"old\">nope</｜DSML｜parameter｜>",
        ));
        // The failure is recorded the moment `old` closes, before `new`.
        assert_eq!(
            sr.preflight_error(),
            Some(
                "edit old selector failed before new was generated: \
                 old text is not a unique match"
            )
        );
        sr.finish();
        assert!(
            sr.sink()
                .errors
                .contains("[tool call stopped: edit old selector failed]"),
            "{:?}",
            sr.sink().errors
        );
    }

    #[test]
    fn edit_old_preflight_pass_leaves_stream_clean() {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.set_preflight(|_| Ok(()));
        sr.push(concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"edit\">",
            "<｜DSML｜parameter name=\"path\">src/a.rs</｜DSML｜parameter｜>",
            "<｜DSML｜parameter name=\"old\">a</｜DSML｜parameter｜>",
            "<｜DSML｜parameter name=\"new\">b</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        ));
        sr.finish();
        assert!(sr.preflight_error().is_none());
        assert_eq!(sr.finished().calls.len(), 1);
        assert!(sr.finished().error.is_none());
    }

    #[test]
    fn malformed_stanza_suppresses_raw_and_reports_error() {
        let sr = run_chunked("<｜DSML｜tool_calls><b>");
        let vis = &sr.sink().visible;
        // The banner names the offending tag through the error channel...
        assert!(
            sr.sink()
                .errors
                .contains("[invalid tool call: unexpected DSML tag: <b>]"),
            "{:?}",
            sr.sink().errors
        );
        // ...but the raw stanza bytes never reach the screen.
        assert!(!vis.contains("tool_calls"), "{vis:?}");
        assert!(sr.finished().error.is_some());
    }

    /// A long-lived renderer -- the debug-console mirror, which keeps one
    /// renderer per connection rather than one per generation pass -- must not
    /// be killed by a DSML error. plank's own renderer is built fresh per pass,
    /// so freezing output there is scoped and correct; the console's spans the
    /// whole session, so the same freeze silently discarded every byte of every
    /// later pass and the window went dead after the first bad stanza. Freezing
    /// is therefore opt-in: the consumer that wants it asks for it.
    #[test]
    fn a_dsml_error_does_not_freeze_a_renderer_that_did_not_opt_in() {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.push("junk \u{ff5c}DSML\u{ff5c} junk");
        // What the next pass over the same connection looks like to the console.
        sr.push("<think>reconsidering</think>Here is the corrected answer.");
        sr.finish();
        let vis = &sr.sink().visible;
        assert!(
            vis.contains("[invalid tool call: DSML markup outside a valid tool_calls block]"),
            "the error line must still render: {vis:?}"
        );
        assert!(
            vis.contains("Here is the corrected answer."),
            "output after the error must still render: {vis:?}"
        );
        // The error is still reported -- only the output freeze is opt-in.
        assert_eq!(
            sr.finished().error,
            Some("DSML markup outside a valid tool_calls block")
        );
    }

    /// The freeze itself, opted in, still works: this is what plank relies on
    /// so raw tool-call markup never spills to the user after a bad stanza.
    #[test]
    fn an_opted_in_renderer_still_freezes_on_a_dsml_error() {
        let mut sr = StreamRenderer::new(Cap::default());
        sr.set_freeze_on_error(true);
        sr.push("junk \u{ff5c}DSML\u{ff5c} junk");
        sr.push("<think>reconsidering</think>Here is the corrected answer.");
        sr.finish();
        let vis = &sr.sink().visible;
        assert!(
            !vis.contains("Here is the corrected answer."),
            "frozen output must stay frozen: {vis:?}"
        );
    }

    #[test]
    fn loose_dsml_marker_is_flagged() {
        let sr = run_chunked("junk ｜DSML｜ junk");
        assert!(
            sr.sink()
                .visible
                .contains("[invalid tool call: DSML markup outside a valid tool_calls block]"),
            "{:?}",
            sr.sink().visible
        );
        assert_eq!(
            sr.finished().error,
            Some("DSML markup outside a valid tool_calls block")
        );
    }

    /// The model quoting a DSML marker while thinking used to set the sticky
    /// `dsml_in_think` flag, which disabled the loose-marker validator for the
    /// whole rest of the stream — so genuinely malformed markup in the answer
    /// went unreported and the turn ended with nothing for the model to correct.
    #[test]
    fn a_quoted_marker_in_think_does_not_disarm_the_loose_marker_validator() {
        let sr = run_chunked("<think>the ｜DSML｜ marker opens a call</think>junk ｜DSML｜ junk");
        assert_eq!(
            sr.finished().error,
            Some("DSML markup outside a valid tool_calls block"),
            "{:?}",
            sr.sink().visible
        );
    }

    /// A `</think>` inside a parameter value is consumed as content so a
    /// payload containing that literal text is not corrupted. When the stanza
    /// never completes, the token was the real control token: `in_think` must
    /// not stay stuck, or the stream is reported as having ended mid-thought
    /// and the cut-off call is misdiagnosed as an in-think placement error.
    #[test]
    fn a_think_close_swallowed_by_an_unfinished_stanza_still_ends_thinking() {
        let sr = run_chunked(concat!(
            "<think>",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"write\">",
            "<｜DSML｜parameter name=\"content\">x</think>",
        ));
        let fin = sr.finished();
        assert!(!fin.ended_in_think, "in_think stayed stuck after </think>");
        assert!(
            !fin.in_think_rejected,
            "a call cut off after thinking closed is not an in-think call"
        );
        assert!(
            sr.sink().visible.contains("[tool call interrupted]"),
            "{:?}",
            sr.sink()
        );
    }

    /// The mirror case: a stanza that *completes* proves the `</think>` in its
    /// payload was content, so thinking is still open and the payload keeps the
    /// literal text.
    #[test]
    fn a_think_close_inside_a_valid_payload_stays_payload_text() {
        let sr = run_chunked(concat!(
            "<think>",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\">echo </think></｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        ));
        let fin = sr.finished();
        assert!(fin.ended_in_think, "the thinking block never closed");
        assert_eq!(fin.calls.len(), 0, "an in-think stanza is not dispatched");
    }

    #[test]
    fn implicit_invoke_opener_is_accepted() {
        let stanza = concat!(
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\">pwd</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        for sr in [run_chunked(stanza), run_charwise(stanza)] {
            assert!(
                sr.sink().visible.contains("🛠️ $ pwd"),
                "{:?}",
                sr.sink().visible
            );
            let fin = sr.finished();
            assert_eq!(fin.calls.len(), 1);
            assert_eq!(fin.calls[0].arg_value("command"), Some("pwd"));
        }
    }

    /// Regression for the repro captured after a weights update: every turn
    /// opened with `<｜DSML｜tool_calls｜>`, which matched no opener form, so the
    /// stanza streamed as prose and the inner `<｜DSML｜invoke` tripped the
    /// loose-marker detector instead of dispatching the tool.
    #[test]
    fn opener_with_trailing_bar_is_accepted() {
        let stanza = concat!(
            "<｜DSML｜tool_calls｜>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        for sr in [run_chunked(stanza), run_charwise(stanza)] {
            let fin = sr.finished();
            assert_eq!(fin.error, None);
            assert_eq!(fin.calls.len(), 1);
            assert_eq!(fin.calls[0].name, "bash");
            assert_eq!(fin.calls[0].arg_value("command"), Some("pwd"));
        }
    }

    /// Second recorded repro: the parameter written as its own element, closed
    /// with `</｜DSML｜invoke>`. Rejecting it cost three turns and ended with the
    /// model breaking the think gate, so the shorthand dispatches instead.
    #[test]
    fn shorthand_parameter_element_dispatches() {
        let stanza = concat!(
            "<｜DSML｜tool_calls｜>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜command string=\"true\">ls</｜DSML｜invoke>",
            "</｜DSML｜invoke>",
            "</｜DSML｜tool_calls｜>",
        );
        for sr in [run_chunked(stanza), run_charwise(stanza)] {
            let fin = sr.finished();
            assert_eq!(fin.error, None);
            assert_eq!(fin.calls.len(), 1);
            assert_eq!(fin.calls[0].name, "bash");
            assert_eq!(fin.calls[0].arg_value("command"), Some("ls"));
        }
    }

    #[test]
    fn write_content_streams_without_label() {
        let stanza = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"write\">",
            "<｜DSML｜parameter name=\"path\" string=\"true\">x.txt</｜DSML｜parameter｜>",
            "<｜DSML｜parameter name=\"content\">line one\nline two</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        for sr in [run_chunked(stanza), run_charwise(stanza)] {
            let vis = &sr.sink().visible;
            assert!(vis.contains("🛠️ write  path=x.txt"), "{vis:?}");
            // The content now previews on the dim (think) channel, not visible.
            assert!(!vis.contains("line one"), "{vis:?}");
            assert!(
                sr.sink().think.contains("line one\nline two"),
                "{:?}",
                sr.sink().think
            );
            assert!(!vis.contains("content:"), "{vis:?}");
            assert!(!vis.contains("DSML"), "{vis:?}");
        }
    }

    #[test]
    fn post_think_whitespace_gap_is_swallowed() {
        let sr = run_chunked("<think>x</think>\n\n  Answer");
        assert!(
            sr.sink().visible.ends_with("Answer"),
            "{:?}",
            sr.sink().visible
        );
        assert!(!sr.sink().visible.contains("\n\n  Answer"));
    }

    #[test]
    fn charwise_and_chunked_agree() {
        let text = format!("hi <not dsml> there\n{BASH_STANZA}");
        let a = run_chunked(&text);
        let b = run_charwise(&text);
        assert_eq!(a.sink().visible, b.sink().visible);
        assert_eq!(a.finished().calls, b.finished().calls);
    }

    #[test]
    fn tool_error_logging_honors_the_opt_out_env_var() {
        use std::ffi::OsStr;
        // The e2e harness sets this when spawning the binary so fixture stanzas
        // never enter the developer's real ~/.plank log, where they previously
        // outnumbered genuine model failures four to one.
        assert!(!super::logging_enabled_for(Some(OsStr::new("1"))));
        assert!(super::logging_enabled_for(None));
        // Only an exact "1" disables it; anything else is not an opt-out.
        assert!(super::logging_enabled_for(Some(OsStr::new("0"))));
    }
}
