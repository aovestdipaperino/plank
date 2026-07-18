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

use crate::dsml::{DsmlParser, DsmlState, ToolCall};

/// The canonical DSML tool-call opening marker.
const DSML_START: &[u8] = "<｜DSML｜tool_calls>".as_bytes();
/// Canonical invoke opener, seeded when the model skips the outer wrapper.
const CANONICAL_INVOKE: &[u8] = "<｜DSML｜invoke".as_bytes();
/// Prefix of a parameter close tag, for online tail suppression.
const PARAM_CLOSE_PREFIX: &[u8] = "</｜DSML｜parameter".as_bytes();
const DSML_BAR: &[u8] = "｜".as_bytes();

const THINK_OPEN: &[u8] = b"<think>";
const THINK_CLOSE: &[u8] = b"</think>";

/// Destination for rendered output; the UI layer routes this to its renderer.
///
/// The stream renderer never emits raw DSML through this trait: DSML bytes are
/// replaced by tool banners, which always arrive via
/// [`visible_text`](Self::visible_text).
pub trait RenderSink {
    /// Receives ordinary visible output, including tool banners.
    fn visible_text(&mut self, text: &str);
    /// Receives text produced inside a `<think>` block.
    fn think_text(&mut self, text: &str);
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
    if tail.len() <= PARAM_CLOSE_PREFIX.len() {
        return PARAM_CLOSE_PREFIX.starts_with(tail);
    }
    if !tail.starts_with(PARAM_CLOSE_PREFIX) {
        return false;
    }
    let mut i = PARAM_CLOSE_PREFIX.len();
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
    const FORMS: [(&str, bool); 4] = [
        ("<｜DSML｜tool_calls>", false),
        ("<DSML｜tool_calls>", false),
        ("<｜DSML｜invoke", true),
        ("<DSML｜invoke", true),
    ];
    *complete = false;
    *implicit_invoke = false;
    for (form, implicit) in FORMS {
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
    const NEEDLES: [&'static str; 4] = ["｜DSML｜", "|DSML|", "<DSML｜", "</DSML｜"];

    fn feed(&mut self, c: u8) -> bool {
        if self.tail.len() == Self::CAP {
            self.tail.remove(0);
        }
        self.tail.push(c);
        Self::NEEDLES
            .iter()
            .any(|n| self.tail.ends_with(n.as_bytes()))
    }
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
/// use plank::viz::{RenderSink, StreamRenderer};
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
    dsml_in_think: bool,
    dsml_in_think_reported: bool,
    post_think_gap: bool,
    /// Error from DSML markup outside a valid stanza; freezes further output.
    stream_error: Option<String>,
    last_output_newline: bool,
    /// Calls snapshotted at parser `Done`, surviving later parser resets.
    calls: Vec<ToolCall>,
    /// UTF-8 carry buffers so multi-byte characters split across pushes are
    /// never emitted partially.
    vis_carry: Vec<u8>,
    think_carry: Vec<u8>,
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
            dsml_active: false,
            dsml_ignored: false,
            pending: Vec::new(),
            dsml_start_tail: Vec::new(),
            plain_dsml: MarkerDetector::default(),
            think_dsml: MarkerDetector::default(),
            dsml_in_think: false,
            dsml_in_think_reported: false,
            post_think_gap: false,
            stream_error: None,
            last_output_newline: true,
            calls: Vec::new(),
            vis_carry: Vec::new(),
            think_carry: Vec::new(),
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
    }

    /// Results after the stream ends: completed calls and error state.
    #[must_use]
    pub fn finished(&self) -> Finished<'_> {
        let error = self
            .stream_error
            .as_deref()
            .or_else(|| (self.parser.state() == DsmlState::Error).then(|| self.parser.error()));
        Finished {
            calls: &self.calls,
            error,
            dsml_in_think: self.dsml_in_think,
        }
    }

    /// Borrows the underlying sink.
    pub fn sink(&self) -> &S {
        &self.sink
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
        self.last_output_newline = bytes.last() == Some(&b'\n');
        self.vis_carry.extend_from_slice(bytes);
        Self::flush_stream(S::visible_text, &mut self.sink, &mut self.vis_carry);
    }

    fn emit_think_bytes(&mut self, bytes: &[u8]) {
        self.think_carry.extend_from_slice(bytes);
        Self::flush_stream(S::think_text, &mut self.sink, &mut self.think_carry);
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
                if self.viz.tool_name != "write" {
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
            self.viz_puts(&owned);
        }
        self.viz_newline_if_open();
        self.viz.active = false;
    }

    /// Shows the exact rejected DSML bytes so failures are debuggable.
    fn viz_dump_invalid_dsml(&mut self) {
        if !self.viz.active {
            return;
        }
        if self.viz.param_active {
            self.viz.param_active = false;
            self.viz.param_end_tail.clear();
            self.viz.param_name.clear();
        }
        self.viz_newline_if_open();
        if self.parser.raw().is_empty() {
            self.viz_puts("<empty DSML>");
        } else {
            let raw = self.parser.raw().to_vec();
            self.emit_visible_bytes(&raw);
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
        if tag.starts_with("</｜DSML｜invoke") {
            self.viz_invoke_end();
        } else if tag.starts_with("<｜DSML｜invoke") {
            let name = parse_attr(&tag, "name").unwrap_or_else(|| "tool".to_string());
            self.viz_tool(&name);
        } else if tag.starts_with("<｜DSML｜parameter")
            && let Some(name) = parse_attr(&tag, "name")
        {
            self.viz_param_begin(&name);
            self.scan = DsmlScan::Value;
        }
        // Anything else is malformed; the strict parser reports it.
    }

    fn feed_dsml_byte(&mut self, c: u8) {
        self.parser.feed([c]);
        if !self.dsml_ignored {
            self.scan_dsml_byte(c);
        }
        match self.parser.state() {
            DsmlState::Done => {
                self.calls = self.parser.calls().to_vec();
                if self.dsml_ignored {
                    self.finish_ignored_dsml("tool calling is not allowed inside <think></think>");
                } else {
                    self.viz_finish(None);
                    self.dsml_active = false;
                }
            }
            DsmlState::Error => {
                if self.dsml_ignored {
                    self.finish_ignored_dsml("malformed tool call inside <think></think>");
                } else {
                    let err = if self.parser.error().is_empty() {
                        "parse error"
                    } else {
                        self.parser.error()
                    };
                    let status = format!("[invalid tool call: {err}]\n");
                    self.viz_dump_invalid_dsml();
                    self.viz_finish(Some(&status));
                    self.dsml_active = false;
                }
            }
            _ => {}
        }
    }

    /// Starts a DSML block; the parser is seeded with canonical bytes so all
    /// later parsing stays strict even when a typo form was accepted.
    fn start_dsml(&mut self, ignored: bool) {
        self.dsml_active = true;
        self.dsml_ignored = ignored;
        if ignored {
            self.dsml_in_think = true;
        }
        self.dsml_start_tail.clear();
        self.post_think_gap = false;
        self.parser.feed(DSML_START);
        self.scan = DsmlScan::Between;
        if !ignored {
            self.viz_start();
        }
    }

    fn finish_ignored_dsml(&mut self, msg: &str) {
        self.dsml_in_think = true;
        self.dsml_in_think_reported = true;
        self.viz_newline_if_open();
        let line = format!("[tool call ignored: {msg}]\n");
        self.viz_puts(&line);
        self.parser.reset();
        self.dsml_active = false;
        self.dsml_ignored = false;
    }

    fn malformed_dsml(&mut self, msg: &str) {
        if self.stream_error.is_some() {
            return;
        }
        self.stream_error = Some(msg.to_string());
        self.viz_newline_if_open();
        let line = format!("[invalid tool call: {msg}]\n");
        self.viz_puts(&line);
    }

    fn output_frozen(&self) -> bool {
        self.stream_error.is_some() || self.parser.state() == DsmlState::Error
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
        if self.output_frozen() || self.dsml_active || self.in_think || self.dsml_in_think {
            return;
        }
        if self.plain_dsml.feed(c) {
            self.malformed_dsml("DSML markup outside a valid tool_calls block");
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
                    self.start_dsml(self.in_think);
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
                i += THINK_OPEN.len();
                continue;
            }
            if !self.dsml_active && rem.starts_with(THINK_CLOSE) {
                self.flush_start_tail();
                self.in_think = false;
                self.viz_newline_if_open();
                self.emit_visible_bytes(b"\n");
                self.post_think_gap = true;
                i += THINK_CLOSE.len();
                continue;
            }
            if !finish
                && !self.dsml_active
                && rem[0] == b'<'
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
                if self.dsml_ignored {
                    self.finish_ignored_dsml("tool calling is not allowed inside <think></think>");
                } else {
                    self.viz_finish(Some("[tool call interrupted]\n"));
                    self.dsml_active = false;
                }
            }
            if self.dsml_in_think && !self.dsml_in_think_reported {
                self.finish_ignored_dsml("tool calling is not allowed inside <think></think>");
            }
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
    }

    impl RenderSink for Cap {
        fn visible_text(&mut self, text: &str) {
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
    fn malformed_stanza_dumps_raw_and_reports_error() {
        let sr = run_chunked("<｜DSML｜tool_calls><b>");
        let vis = &sr.sink().visible;
        assert!(vis.contains("[invalid tool call: "), "{vis:?}");
        // The rejected raw bytes are shown for debugging.
        assert!(vis.contains("<b>"), "{vis:?}");
        assert!(sr.finished().error.is_some());
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
            assert!(vis.contains("line one\nline two"), "{vis:?}");
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
}
