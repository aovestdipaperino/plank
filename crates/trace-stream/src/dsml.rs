// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Streaming DSML tool-call parser.
//!
//! The model streams raw text tokens. This parser recognizes completed DSML
//! tool stanzas (`<｜DSML｜tool_calls>` ... `</｜DSML｜tool_calls｜>`) and keeps
//! a copy of the raw stanza for diagnostics. Inner tags tolerate the one
//! observed typo (a dropped leading `｜`, e.g. `<DSML｜invoke ...>`), matching
//! the tolerance the stanza opener already had; beyond that the parser stays
//! strict, so the actual tool parser stays small and predictable.
//!
//! Port of the `agent_dsml_*` family from `ds4_agent.c`.

const DSML_START: &[u8] = "<｜DSML｜tool_calls>".as_bytes();
const SSML_START: &[u8] = "<｜SSML｜tool_calls>".as_bytes();
/// The same openers with a trailing `｜` before `>`. Closing tags have always
/// tolerated that bar; post-update weights emit it on the opener too, and
/// without these forms the stanza never opens and the model only sees the
/// downstream "DSML markup outside a valid `tool_calls` block" error.
const DSML_START_BAR: &[u8] = "<｜DSML｜tool_calls｜>".as_bytes();
const SSML_START_BAR: &[u8] = "<｜SSML｜tool_calls｜>".as_bytes();
/// Cheap scan filter used to locate candidate closing tags: any `</` byte
/// pair, not just a validated close marker. Real validation happens in
/// [`close_tag_at`], which requires a full [`tag_prefix_len`] match against
/// the accepted marker/name spellings — so a bare `</` inside a parameter
/// value (e.g. HTML written through a `write` or `edit` call) never
/// terminates the parameter on its own.
const CLOSE_SCAN_HEAD: &[u8] = "</".as_bytes();
const DSML_BAR: &[u8] = "｜".as_bytes();

/// Marker names accepted inside a tag: `<｜NAME｜invoke ...>`.
///
/// `DSML` is canonical and the only form the system prompt teaches. `SSML` is
/// an alias for a misspelling the model actually emits: `｜DSML｜` is a
/// dedicated vocab token, but plank composes the tools prompt as an ordinary
/// system message, so the marker arrives as ordinary BPE pieces and the model
/// spells it back out — where the far more common pretraining string "SSML"
/// occasionally wins the "D". Without the alias the stanza parses as nothing,
/// prints raw, and the turn ends with no tool error for the model to retry
/// from. The prompt tells the model SSML is unsupported so this stays a
/// recovery path rather than a second syntax.
pub(crate) const MARKER_NAMES: [&str; 2] = ["DSML", "SSML"];

/// Matches an opening or closing tag prefix for `name` under any accepted
/// marker, returning the matched length.
///
/// Both the canonical `<｜NAME｜tag` and the dropped-leading-bar `<NAME｜tag`
/// typo are accepted, mirroring the tolerance `dsml_start_match` has always
/// had on the stanza opener. The two forms differ in length, so the matched
/// length is taken from the form that actually matched.
pub(crate) fn tag_prefix_len(s: &[u8], closing: bool, name: &str) -> Option<usize> {
    MARKER_NAMES.iter().find_map(|marker| {
        tag_prefix_forms(marker, closing, name)
            .into_iter()
            .find_map(|form| segments_prefix_of(&form, s))
    })
}

/// True when `s` is a (possibly incomplete) prefix of a tag opener for `name`
/// under any accepted marker, in either the canonical or dropped-bar form.
pub(crate) fn tag_prefix_partial(s: &[u8], closing: bool, name: &str) -> bool {
    MARKER_NAMES.iter().any(|marker| {
        tag_prefix_forms(marker, closing, name)
            .iter()
            .any(|form| is_prefix_of_segments(s, form))
    })
}

/// The accepted spellings of a tag prefix, as segment lists: canonical first,
/// then the dropped-leading-bar typo the model actually emits.
///
/// Segments rather than assembled `String`s because this sits on the hot path:
/// [`find_close_tag_any`] re-scans the accumulated parameter value on every
/// `feed`, and `CLOSE_SCAN_HEAD` is a bare `</`, which occurs on nearly every
/// line of HTML written through a `write` or `edit` parameter. Building the
/// spellings with `format!` cost four transient allocations per candidate,
/// which is order 10^6 for a few hundred lines of markup.
fn tag_prefix_forms<'a>(marker: &'a str, closing: bool, name: &'a str) -> [[&'a [u8]; 6]; 2] {
    let slash: &[u8] = if closing { b"/" } else { b"" };
    let (marker, name) = (marker.as_bytes(), name.as_bytes());
    [
        [b"<", slash, DSML_BAR, marker, DSML_BAR, name],
        [b"<", slash, marker, DSML_BAR, name, b""],
    ]
}

/// Length of the concatenated `segments` when they are a prefix of `s`.
fn segments_prefix_of(segments: &[&[u8]], s: &[u8]) -> Option<usize> {
    let mut at = 0;
    for seg in segments {
        if !s[at..].starts_with(seg) {
            return None;
        }
        at += seg.len();
    }
    Some(at)
}

/// True when `s` is a prefix of the concatenated `segments`, `s` possibly
/// stopping part-way through one of them.
fn is_prefix_of_segments(s: &[u8], segments: &[&[u8]]) -> bool {
    let mut rest = s;
    for seg in segments {
        if rest.len() < seg.len() {
            return seg.starts_with(rest);
        }
        if !rest.starts_with(seg) {
            return false;
        }
        rest = &rest[seg.len()..];
    }
    rest.is_empty()
}

/// Byte offset of the earliest complete tool-call stanza opening in `s`, if any.
///
/// Port of the C server's `find_any_tool_start`: the wrapper opener under any
/// accepted marker, its dropped-leading-bar typo, and the bare `<tool_calls>`
/// the model sometimes emits. Deliberately *not* the bare `invoke` opener the
/// streaming detector also accepts — this feeds mid-generation recovery, where
/// acting on a weaker signal costs a forced injection.
///
/// Matching is on accumulated text, so how the marker was tokenized does not
/// matter; an incomplete opening does not match, and the caller is expected to
/// re-scan from far enough back that one split across tokens is still seen.
#[must_use]
pub fn find_tool_start(s: &str) -> Option<usize> {
    let mut forms: Vec<String> = vec!["<tool_calls>".to_owned()];
    for m in MARKER_NAMES {
        forms.push(format!("<｜{m}｜tool_calls>"));
        forms.push(format!("<｜{m}｜tool_calls｜>"));
        forms.push(format!("<{m}｜tool_calls>"));
        forms.push(format!("<{m}｜tool_calls｜>"));
    }
    forms.iter().filter_map(|f| s.find(f.as_str())).min()
}

/// Bytes held back when re-scanning a stream for [`find_tool_start`]: longer
/// than the longest opening, so one split across future tokens is still seen
/// from its first byte.
pub const TOOL_START_SCAN_HOLD: usize = 80;

/// One named argument of a parsed tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolArg {
    /// Argument name from the `name="..."` attribute.
    pub name: String,
    /// Raw argument value (bytes between the parameter tags).
    pub value: String,
    /// True when the parameter carried `string="true"`.
    pub is_string: bool,
}

/// A parsed tool invocation: tool name plus its arguments in stream order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolCall {
    /// Tool name from the invoke tag's `name="..."` attribute.
    pub name: String,
    /// Arguments in the order they were streamed.
    pub args: Vec<ToolArg>,
}

impl ToolCall {
    /// Returns the value of the named argument, if present.
    pub fn arg_value(&self, name: impl AsRef<str>) -> Option<&str> {
        let name = name.as_ref();
        self.args
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.value.as_str())
    }
}

/// Parser progress; terminal states are `Done` and `Error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DsmlState {
    /// Scanning free text for the opening `<｜DSML｜tool_calls>` marker.
    #[default]
    Search,
    /// Between tags: expecting invoke/parameter open tags or close tags.
    Structural,
    /// Accumulating a parameter value until its close tag arrives.
    ParamValue,
    /// A full `tool_calls` stanza was parsed.
    Done,
    /// The stanza was malformed; see [`DsmlParser::error`].
    Error,
}

/// Incremental parser for one DSML tool-call stanza.
///
/// Feed streamed bytes with [`feed`](Self::feed); it can be called after every
/// byte. Incomplete input leaves state unchanged until enough bytes arrive,
/// while malformed completed input switches to [`DsmlState::Error`] so the
/// model gets a retryable tool error.
#[derive(Debug, Default)]
pub struct DsmlParser {
    state: DsmlState,
    search_tail: Vec<u8>,
    raw: Vec<u8>,
    parse_pos: usize,
    current: Option<PendingCall>,
    param_name: Option<String>,
    param_is_string: bool,
    param_value_start: usize,
    /// Element name when the open parameter used the shorthand form
    /// (`<｜DSML｜command …>` instead of `<｜DSML｜parameter name="command" …>`),
    /// which widens the accepted terminators for *this value only*. `None` for
    /// canonical parameters, so their strict `parameter`-only terminator — the
    /// thing that keeps a `</` inside a `write` payload from ending the value —
    /// is never relaxed.
    param_elem: Option<String>,
    /// True while the raw tail looks like a partial parameter close tag, so
    /// online rendering can hide it before the full tag arrives.
    param_close_prefix: bool,
    calls: Vec<ToolCall>,
    error: String,
}

#[derive(Debug, Default)]
struct PendingCall {
    name: String,
    args: Vec<ToolArg>,
}

/// True when a name is an unsubstituted placeholder copied from the tools
/// prompt (`$TOOL_NAME`, `$PARAMETER_NAME`, `$PARAMETER_VALUE`).
///
/// Exactly those three: a tool or parameter genuinely named `$path` is a
/// different mistake and must not be told it copied the prompt.
fn is_prompt_placeholder(name: &str) -> bool {
    matches!(name, "$TOOL_NAME" | "$PARAMETER_NAME" | "$PARAMETER_VALUE")
}

impl DsmlParser {
    /// Creates a parser in the `Search` state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Current parser state.
    #[must_use]
    pub fn state(&self) -> DsmlState {
        self.state
    }

    /// Tool calls completed so far, in stream order.
    #[must_use]
    pub fn calls(&self) -> &[ToolCall] {
        &self.calls
    }

    /// Error message; empty unless the state is [`DsmlState::Error`].
    #[must_use]
    pub fn error(&self) -> &str {
        &self.error
    }

    /// Snapshot of the invoke currently being parsed (name plus the
    /// arguments whose close tags have arrived), for mid-stream preflight.
    #[must_use]
    pub fn pending_call(&self) -> Option<ToolCall> {
        self.current.as_ref().map(|c| ToolCall {
            name: c.name.clone(),
            args: c.args.clone(),
        })
    }

    /// Raw bytes of the stanza accumulated so far, for diagnostics.
    #[must_use]
    pub fn raw(&self) -> &[u8] {
        &self.raw
    }

    /// True while the raw tail is a partial parameter close tag.
    #[must_use]
    pub fn param_close_prefix(&self) -> bool {
        self.param_close_prefix
    }

    /// Resets the parser to a fresh `Search` state, discarding all results.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Feeds streamed bytes; no-op once the parser is `Done` or `Error`.
    pub fn feed(&mut self, s: impl AsRef<[u8]>) {
        let s = s.as_ref();
        if matches!(self.state, DsmlState::Done | DsmlState::Error) {
            return;
        }
        for &c in s {
            if self.state == DsmlState::Search {
                if self.search_tail.len() == 64 {
                    self.search_tail.remove(0);
                }
                self.search_tail.push(c);
                if [DSML_START, SSML_START, DSML_START_BAR, SSML_START_BAR]
                    .iter()
                    .any(|f| self.search_tail.ends_with(f))
                {
                    self.start();
                }
                continue;
            }

            self.raw.push(c);
            self.parse();
            if self.state == DsmlState::ParamValue {
                self.update_param_close_prefix();
            } else {
                self.param_close_prefix = false;
            }
        }
    }

    fn start(&mut self) {
        self.state = DsmlState::Structural;
        self.search_tail.clear();
        self.raw.extend_from_slice(DSML_START);
        self.parse_pos = DSML_START.len();
    }

    fn set_error(&mut self, msg: impl Into<String>) {
        self.state = DsmlState::Error;
        self.error = msg.into();
    }

    fn push_current(&mut self) {
        if let Some(call) = self.current.take() {
            self.calls.push(ToolCall {
                name: call.name,
                args: call.args,
            });
        }
    }

    /// Parses as much of the accumulated buffer as possible.
    fn parse(&mut self) {
        loop {
            match self.state {
                DsmlState::ParamValue => {
                    // A shorthand parameter is closed inconsistently: the
                    // recorded repro ends `<｜DSML｜command …>` with
                    // `</｜DSML｜invoke>`, not `</｜DSML｜command>`. Accept either,
                    // plus `parameter`, and take whichever lands first.
                    let mut names: Vec<&str> = vec!["parameter"];
                    if let Some(elem) = self.param_elem.as_deref() {
                        names.push(elem);
                        names.push("invoke");
                    }
                    let Some((end, tag_len)) =
                        find_close_tag_any(&self.raw[self.param_value_start..], &names)
                    else {
                        return;
                    };
                    let value_bytes =
                        &self.raw[self.param_value_start..self.param_value_start + end];
                    let arg = ToolArg {
                        name: self.param_name.take().unwrap_or_default(),
                        value: String::from_utf8_lossy(value_bytes).into_owned(),
                        is_string: self.param_is_string,
                    };
                    self.current
                        .get_or_insert_with(Default::default)
                        .args
                        .push(arg);
                    self.param_close_prefix = false;
                    self.param_elem = None;
                    self.parse_pos = self.param_value_start + end + tag_len;
                    self.state = DsmlState::Structural;
                }
                DsmlState::Structural => {
                    while self.parse_pos < self.raw.len()
                        && self.raw[self.parse_pos].is_ascii_whitespace()
                    {
                        self.parse_pos += 1;
                    }
                    if self.parse_pos >= self.raw.len() {
                        return;
                    }

                    let rest = &self.raw[self.parse_pos..];
                    if let Some(close_len) = close_tag_at(rest, "tool_calls") {
                        self.push_current();
                        self.parse_pos += close_len;
                        self.state = DsmlState::Done;
                        return;
                    }
                    if let Some(close_len) = close_tag_at(rest, "invoke") {
                        self.push_current();
                        self.parse_pos += close_len;
                        continue;
                    }

                    let Some(gt) = rest.iter().position(|&b| b == b'>') else {
                        return;
                    };
                    let tag_len = gt + 1;
                    let tag = String::from_utf8_lossy(&rest[..tag_len]).into_owned();

                    if !self.open_tag(&tag, tag_len) {
                        return;
                    }
                }
                _ => return,
            }
        }
    }

    /// Handles one opening tag inside a stanza. Returns false when parsing must
    /// stop, having set the error.
    ///
    /// The four accepted spellings, in precedence order: a canonical `invoke`,
    /// a canonical `parameter`, and the two bare-element shorthands — parameter
    /// first, since it is the one that applies while an invoke is open (see
    /// [`Self::shorthand_invoke_name`] for why the open invoke is the whole
    /// distinction).
    fn open_tag(&mut self, tag: &str, tag_len: usize) -> bool {
        // A repeated wrapper opener is the model restating itself, not a second
        // stanza — `repro-1785770781.md` does it 37 times. The stanza is already
        // open, so consuming the tag and moving on is idempotent, and strictly
        // better than the alternatives: erroring costs the turn, and treating it
        // as a name invents a `tool_calls` call that swallows the parameters.
        if open_tag_is(tag, "tool_calls") {
            self.parse_pos += tag_len;
            return true;
        }
        if open_tag_is(tag, "invoke") {
            let Some(name) = parse_attr(tag, "name") else {
                self.set_error("tool invoke without name");
                return false;
            };
            if is_prompt_placeholder(&name) {
                self.set_error(format!(
                    "tool name is the prompt's placeholder {name}, not a real tool; substitute the actual tool name"
                ));
                return false;
            }
            self.open_invoke(name, tag_len);
        } else if open_tag_is(tag, "parameter") {
            let Some(name) = parse_attr(tag, "name") else {
                self.set_error("tool parameter without name");
                return false;
            };
            if is_prompt_placeholder(&name) {
                self.set_error(format!(
                    "parameter name is the prompt's placeholder {name}, not a real parameter; substitute the actual parameter name"
                ));
                return false;
            }
            self.param_elem = None;
            self.open_param(name, tag, tag_len);
        } else if let Some(elem) = self.shorthand_param_name(tag) {
            self.param_elem = Some(elem.clone());
            self.open_param(elem, tag, tag_len);
        } else if let Some(elem) = self.shorthand_invoke_name(tag) {
            self.open_invoke(elem, tag_len);
        } else {
            let shown: String = tag.chars().take(80).collect();
            self.set_error(format!("unexpected DSML tag: {shown}"));
            return false;
        }
        true
    }

    /// Opens a tool call, however its name was spelled.
    fn open_invoke(&mut self, name: String, tag_len: usize) {
        self.current = Some(PendingCall {
            name,
            args: Vec::new(),
        });
        self.parse_pos += tag_len;
    }

    /// Enters `ParamValue` for a parameter named `name` opened by `tag`.
    fn open_param(&mut self, name: String, tag: &str, tag_len: usize) {
        self.param_name = Some(name);
        self.param_is_string = parse_attr(tag, "string").as_deref() == Some("true");
        self.parse_pos += tag_len;
        self.param_value_start = self.parse_pos;
        self.param_close_prefix = false;
        self.state = DsmlState::ParamValue;
    }

    /// The parameter name for the shorthand form the post-update weights emit:
    /// the parameter name written as the element name,
    /// `<｜DSML｜command string="true">ls</｜DSML｜invoke>` in place of
    /// `<｜DSML｜parameter name="command" string="true">ls</｜DSML｜parameter>`.
    ///
    /// Deliberately narrow, because accepting it means *running* a tool call
    /// that was not written the way the prompt teaches. All of these must hold:
    /// an invoke is already open, the tag carries the DSML marker, its element
    /// name is a plain identifier, and it has no `name` attribute — a tag with
    /// one is some other malformation and still errors. Rejecting instead is
    /// not free: the recorded repro shows the model unable to find its way back
    /// from the error, re-emitting the same shape and then breaking the think
    /// gate, so the turn is lost either way.
    fn shorthand_param_name(&self, tag: &str) -> Option<String> {
        if self.current.is_none() || parse_attr(tag, "name").is_some() {
            return None;
        }
        let elem = element_name(tag)?;
        (!is_prompt_placeholder(&elem) && !Self::STRUCTURAL_ELEMS.contains(&elem.as_str()))
            .then_some(elem)
    }

    /// Structural element names, which can never be a tool or a parameter name.
    ///
    /// Without this the shorthands read the model restating a wrapper tag as a
    /// name: `repro-1785770781.md` opens 37 stanzas with `<｜DSML｜tool_calls>`
    /// twice in a row, and the second one parsed as a *successful* call named
    /// `tool_calls` that swallowed the real parameters. Erroring would be better
    /// than that; skipping it, as [`Self::open_tag`] now does, is better still.
    const STRUCTURAL_ELEMS: [&'static str; 3] = ["tool_calls", "invoke", "parameter"];

    /// The same shorthand one level up: the *tool* name written as the element
    /// name, `<｜DSML｜edit>…</｜DSML｜invoke>` in place of
    /// `<｜DSML｜invoke name="edit">…</｜DSML｜invoke>`.
    ///
    /// Which of the two shorthands a bare element is depends on one thing:
    /// whether an invoke is open. Before one, the model is naming the tool it
    /// wants; inside one, a parameter. That is why this is checked *after*
    /// [`Self::shorthand_param_name`] — the parameter reading wins whenever
    /// both could apply, which is exactly when an invoke is already open.
    ///
    /// Same narrowness as the parameter form, and for the same reason:
    /// accepting it means *running* a call written the way the prompt does not
    /// teach. The tag must carry the DSML marker, its element name must be a
    /// plain identifier, and it must have no `name` attribute — a tag with one
    /// is some other malformation and still errors. An element name that is not
    /// a real tool reaches dispatch and fails there by name, which is a clear
    /// error the model can act on; rejecting the stanza outright is not free.
    /// `repro-1785754509.md` is the recorded cost: five rejections of this
    /// shape, no recovery, and the model finally breaking the think gate while
    /// trying to restate the syntax back to itself.
    fn shorthand_invoke_name(&self, tag: &str) -> Option<String> {
        if self.current.is_some() || parse_attr(tag, "name").is_some() {
            return None;
        }
        let elem = element_name(tag)?;
        (!is_prompt_placeholder(&elem) && !Self::STRUCTURAL_ELEMS.contains(&elem.as_str()))
            .then_some(elem)
    }

    /// Tracks whether the raw tail is a partial parameter close tag, so the
    /// terminal renderer can hide it without waiting for the whole parameter.
    fn update_param_close_prefix(&mut self) {
        self.param_close_prefix = false;
        if self.state != DsmlState::ParamValue || self.raw.len() <= self.param_value_start {
            return;
        }
        let value = &self.raw[self.param_value_start..];
        let Some(lt) = value.iter().rposition(|&b| b == b'<') else {
            return;
        };
        let tail = &value[lt..];
        if tail.len() > 64 || tag_prefix_len(tail, true, "").is_none() {
            return;
        }
        let mut complete = false;
        self.param_close_prefix = parameter_close_tail(tail, &mut complete) && !complete;
    }
}

/// The element name of a DSML-marked opening tag, e.g. `command` for
/// `<｜DSML｜command string="true">`.
///
/// Used only to sharpen the "unexpected DSML tag" error. Post-update weights
/// write the *parameter name* as the element name; echoing the tag back taught
/// the model nothing, and the recorded repro shows it guessing at the marker
/// spelling for three turns and then emitting DSML inside `<think>`. Naming the
/// rewrite gives it something to act on.
pub(crate) fn element_name(tag: &str) -> Option<String> {
    let len = tag_prefix_len(tag.as_bytes(), false, "")?;
    let name: String = tag[len..]
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// Checks whether `tag` is an opening DSML tag with the given element name.
fn open_tag_is(tag: &str, name: &str) -> bool {
    let Some(len) = tag_prefix_len(tag.as_bytes(), false, name) else {
        return false;
    };
    tag.as_bytes()
        .get(len)
        .is_some_and(|&c| c == b'>' || c.is_ascii_whitespace())
}

/// Recognizes a DSML closing tag at the start of `s`, returning its length.
///
/// Accepts the few harmless closing-tag variants the model has been observed
/// to emit (whitespace and an optional trailing `｜` before `>`). Opening tags
/// stay strict so accidental prose does not become a tool call.
fn close_tag_at(s: &[u8], name: &str) -> Option<usize> {
    let mut i = tag_prefix_len(s, true, name)?;
    while i < s.len() && s[i].is_ascii_whitespace() {
        i += 1;
    }
    if s[i..].starts_with(DSML_BAR) {
        i += DSML_BAR.len();
    }
    while i < s.len() && s[i].is_ascii_whitespace() {
        i += 1;
    }
    if s.get(i) != Some(&b'>') {
        return None;
    }
    Some(i + 1)
}

/// Finds the earliest DSML closing tag for any of `names`; returns
/// (offset, tag length).
///
/// Scanning position-first rather than name-first matters: the winner must be
/// the tag that appears earliest in the value, not the one whose name happens
/// to come first in the list.
fn find_close_tag_any(s: &[u8], names: &[&str]) -> Option<(usize, usize)> {
    let mut from = 0;
    while let Some(pos) = find_bytes(&s[from..], CLOSE_SCAN_HEAD) {
        let at = from + pos;
        if let Some(tag_len) = names.iter().find_map(|n| close_tag_at(&s[at..], n)) {
            return Some((at, tag_len));
        }
        from = at + 1;
    }
    None
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Recognizes a streamed parameter close tag prefix.
///
/// Full close detection is handled by [`close_tag_at`]; this exists for online
/// behavior: terminal rendering must hide partial close tags without waiting
/// for the whole parameter to finish. Sets `complete` when the tail is a full
/// close tag ending exactly at the last byte.
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

/// Extracts a `name="value"` attribute from a tag, if present.
fn parse_attr(tag: &str, name: &str) -> Option<String> {
    let pat = format!("{name}=\"");
    let start = tag.find(&pat)? + pat.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const STANZA: &str = concat!(
        "<｜DSML｜tool_calls>",
        "<｜DSML｜invoke name=\"read_file\">",
        "<｜DSML｜parameter name=\"path\" string=\"true\">src/main.rs</｜DSML｜parameter｜>",
        "<｜DSML｜parameter name=\"offset\">42</｜DSML｜parameter｜>",
        "</｜DSML｜invoke｜>",
        "</｜DSML｜tool_calls｜>",
    );

    fn feed_all(p: &mut DsmlParser, s: &str) {
        p.feed(s.as_bytes());
    }

    fn feed_bytewise(p: &mut DsmlParser, s: &str) {
        for b in s.as_bytes() {
            p.feed([*b]);
        }
    }

    /// Post-update weights close the stanza opener with `｜>` rather than `>`.
    /// Before this was accepted the stanza never opened at all, and the model
    /// saw only "DSML markup outside a valid `tool_calls` block" — with no way to
    /// tell which part of its syntax was rejected — turn after turn.
    #[test]
    fn opener_tolerates_trailing_bar() {
        let stanza = STANZA.replacen("<｜DSML｜tool_calls>", "<｜DSML｜tool_calls｜>", 1);
        for feed in [feed_all as fn(&mut DsmlParser, &str), feed_bytewise] {
            let mut p = super::DsmlParser::new();
            feed(&mut p, &stanza);
            assert_eq!(p.state(), super::DsmlState::Done);
            assert_eq!(p.calls().len(), 1);
            assert_eq!(p.calls()[0].name, "read_file");
            assert_eq!(p.calls()[0].arg_value("path"), Some("src/main.rs"));
        }
        assert_eq!(super::find_tool_start(&stanza), Some(0));
    }

    /// Post-update weights write the parameter name as the element name, and
    /// close it with `</｜DSML｜invoke>`. Verbatim from the recorded repro, in
    /// which rejecting it cost the whole turn.
    #[test]
    fn shorthand_parameter_element_is_executed() {
        let stanza = concat!(
            "<｜DSML｜tool_calls｜>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜command string=\"true\">cd /tmp && ls</｜DSML｜invoke>",
            "</｜DSML｜invoke>",
            "</｜DSML｜tool_calls｜>",
        );
        for feed in [feed_all as fn(&mut DsmlParser, &str), feed_bytewise] {
            let mut p = super::DsmlParser::new();
            feed(&mut p, stanza);
            assert_eq!(p.state(), super::DsmlState::Done, "{}", p.error());
            assert_eq!(p.calls().len(), 1);
            assert_eq!(p.calls()[0].name, "bash");
            assert_eq!(p.calls()[0].arg_value("command"), Some("cd /tmp && ls"));
            assert!(p.calls()[0].args[0].is_string);
        }
    }

    /// The same shorthand one level up: the *tool* name written as the element
    /// name, with no `invoke` wrapper. Verbatim from `repro-1785754509.md`,
    /// where the model emitted this shape five times for `write` and `edit`,
    /// never recovered from the rejection, and finally broke the think gate
    /// trying to restate the syntax.
    #[test]
    fn shorthand_invoke_element_is_executed() {
        let stanza = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜edit>",
            "<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/a.rs</｜DSML｜parameter>",
            "<｜DSML｜parameter name=\"old\" string=\"true\">one</｜DSML｜parameter>",
            "<｜DSML｜parameter name=\"new\" string=\"true\">two</｜DSML｜parameter>",
            "</｜DSML｜invoke>",
            "</｜DSML｜tool_calls>",
        );
        for feed in [feed_all as fn(&mut DsmlParser, &str), feed_bytewise] {
            let mut p = super::DsmlParser::new();
            feed(&mut p, stanza);
            assert_eq!(p.state(), super::DsmlState::Done, "{}", p.error());
            assert_eq!(p.calls().len(), 1);
            assert_eq!(p.calls()[0].name, "edit");
            assert_eq!(p.calls()[0].arg_value("path"), Some("/tmp/a.rs"));
            assert_eq!(p.calls()[0].arg_value("old"), Some("one"));
            assert_eq!(p.calls()[0].arg_value("new"), Some("two"));
        }
    }

    /// A repeated `<｜DSML｜tool_calls>` opener is the model restating itself and
    /// must not become a call. Verbatim from `repro-1785770781.md`, which opens
    /// 37 stanzas that way; the invoke shorthand read the second one as a tool
    /// named `tool_calls` and reported Done, so a nonsense call reached dispatch
    /// carrying the real parameters.
    #[test]
    fn a_repeated_wrapper_opener_is_skipped_not_named() {
        // The benign case: the repeat is absorbed and the real call is intact.
        let stanza = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter>",
            "</｜DSML｜invoke>",
            "</｜DSML｜tool_calls>",
        );
        for feed in [feed_all as fn(&mut DsmlParser, &str), feed_bytewise] {
            let mut p = super::DsmlParser::new();
            feed(&mut p, stanza);
            assert_eq!(p.state(), super::DsmlState::Done, "{}", p.error());
            assert_eq!(p.calls().len(), 1, "{:?}", p.calls());
            assert_eq!(p.calls()[0].name, "bash");
            assert_eq!(p.calls()[0].arg_value("command"), Some("ls -la"));
        }

        // The dangerous case, and the one that regressed: with no real invoke
        // after the repeat, the shorthand read `tool_calls` as the tool name and
        // reported Done, so a call literally named `tool_calls` — carrying the
        // parameters meant for the real tool — reached dispatch. Whatever else
        // this shape yields, that name must never appear.
        let no_invoke = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜tool_calls>",
            "<｜DSML｜parameter name=\"command\" string=\"true\">ls</｜DSML｜parameter>",
            "</｜DSML｜invoke>",
            "</｜DSML｜tool_calls>",
        );
        for feed in [feed_all as fn(&mut DsmlParser, &str), feed_bytewise] {
            let mut p = super::DsmlParser::new();
            feed(&mut p, no_invoke);
            assert!(
                p.calls().iter().all(|c| c.name != "tool_calls"),
                "the wrapper name must never become a tool: {:?}",
                p.calls()
            );
        }
    }

    /// The full malformed stanza from `repro-1785770781.md`, verbatim: a repeated
    /// wrapper opener followed by `<｜DSML｜tool ATTR="...">`, where the model
    /// folded the element name and the attribute name together.
    ///
    /// The repeated opener is tolerated, but the rest is *not* guessed at. The
    /// shape is self-inconsistent — `path="/tmp/lib.rs"` puts the value in the
    /// attribute while `old="true"` uses it as the `string=` flag with the value
    /// as element text — so any reading would be wrong half the time, and this is
    /// an `edit`. Erroring is the correct outcome; fabricating a call is not.
    #[test]
    fn the_folded_attribute_shape_errors_without_fabricating_a_call() {
        let stanza = concat!(
            "<｜DSML｜tool_calls>\n",
            "<｜DSML｜tool_calls>\n",
            "<｜DSML｜tool name=\"edit\">\n",
            "<｜DSML｜tool path=\"/tmp/lib.rs\">\n",
            "<｜DSML｜tool old=\"true\">OLD TEXT</｜DSML｜tool>\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜tool_calls>",
        );
        for feed in [feed_all as fn(&mut DsmlParser, &str), feed_bytewise] {
            let mut p = super::DsmlParser::new();
            feed(&mut p, stanza);
            assert_eq!(p.state(), super::DsmlState::Error, "{:?}", p.calls());
            assert!(
                p.error().starts_with("unexpected DSML tag:"),
                "{}",
                p.error()
            );
            // Nothing executable may survive under the wrapper's name, and no
            // half-built `edit` carrying a bogus `old`.
            assert!(
                p.calls().iter().all(|c| c.name != "tool_calls"),
                "{:?}",
                p.calls()
            );
            assert!(
                p.calls().iter().all(|c| c.arg_value("old") != Some("true")),
                "`old=\"true\"` is a string flag, never the old text: {:?}",
                p.calls()
            );
        }
    }

    /// The shorthands must never read a structural element as a name, whichever
    /// level they are at. `tool_calls` is the wrapper; a bare `invoke` or
    /// `parameter` is a missing-name error, which is its own clearer message.
    #[test]
    fn structural_elements_are_never_names() {
        // At invoke level: no `tool_calls` call is fabricated.
        let mut p = super::DsmlParser::new();
        feed_all(
            &mut p,
            "<｜DSML｜tool_calls><｜DSML｜tool_calls>\
             <｜DSML｜parameter name=\"command\">ls</｜DSML｜parameter>",
        );
        assert!(
            p.calls().iter().all(|c| c.name != "tool_calls"),
            "{:?}",
            p.calls()
        );

        // At parameter level: an invoke is open, and a repeated wrapper tag
        // still must not become a parameter called `tool_calls`.
        let mut p = super::DsmlParser::new();
        feed_all(
            &mut p,
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"bash\"><｜DSML｜tool_calls>\
             <｜DSML｜parameter name=\"command\" string=\"true\">ls</｜DSML｜parameter>\
             </｜DSML｜invoke></｜DSML｜tool_calls>",
        );
        assert_eq!(p.state(), super::DsmlState::Done, "{}", p.error());
        assert_eq!(p.calls().len(), 1);
        assert_eq!(p.calls()[0].name, "bash");
        assert!(
            p.calls()[0].args.iter().all(|a| a.name != "tool_calls"),
            "{:?}",
            p.calls()[0].args
        );
        assert_eq!(p.calls()[0].arg_value("command"), Some("ls"));

        // A bare `invoke` / `parameter` keeps its own missing-name error rather
        // than being silently accepted as a shorthand name.
        for (text, want) in [
            (
                "<｜DSML｜tool_calls><｜DSML｜invoke>",
                "tool invoke without name",
            ),
            (
                "<｜DSML｜tool_calls><｜DSML｜invoke name=\"bash\"><｜DSML｜parameter>",
                "tool parameter without name",
            ),
        ] {
            let mut p = super::DsmlParser::new();
            feed_all(&mut p, text);
            assert_eq!(p.state(), super::DsmlState::Error, "{text}");
            assert_eq!(p.error(), want, "{text}");
        }
    }

    /// Which shorthand a bare element is depends only on whether an invoke is
    /// open: before one it names the tool, inside one it names a parameter.
    /// Both spellings in a single stanza must land in the right slots.
    #[test]
    fn bare_elements_are_tool_then_parameter_names() {
        let mut p = super::DsmlParser::new();
        feed_all(
            &mut p,
            "<｜DSML｜tool_calls>\
             <｜DSML｜bash>\
             <｜DSML｜command string=\"true\">ls -la</｜DSML｜command>\
             </｜DSML｜invoke>\
             </｜DSML｜tool_calls>",
        );
        assert_eq!(p.state(), super::DsmlState::Done, "{}", p.error());
        assert_eq!(p.calls().len(), 1);
        assert_eq!(p.calls()[0].name, "bash");
        assert_eq!(p.calls()[0].arg_value("command"), Some("ls -la"));
    }

    /// The self-consistent spelling of the shorthand closes with its own
    /// element name and a single `</｜DSML｜invoke>`.
    #[test]
    fn shorthand_parameter_closed_by_its_own_element() {
        let mut p = super::DsmlParser::new();
        feed_all(
            &mut p,
            "<｜DSML｜tool_calls>\
             <｜DSML｜invoke name=\"read\">\
             <｜DSML｜path string=\"true\">src/main.rs</｜DSML｜path>\
             </｜DSML｜invoke>\
             </｜DSML｜tool_calls>",
        );
        assert_eq!(p.state(), super::DsmlState::Done, "{}", p.error());
        assert_eq!(p.calls()[0].arg_value("path"), Some("src/main.rs"));
    }

    /// The tolerance must not reach canonical parameters: a `write` payload
    /// that itself contains `</｜DSML｜invoke>` (this repo's own sources and
    /// docs do) still runs to its real `</｜DSML｜parameter>` terminator.
    #[test]
    fn canonical_parameter_value_is_not_truncated_by_a_foreign_close_tag() {
        let content = "docs mentioning </｜DSML｜invoke> and </｜DSML｜command> inline";
        let mut p = super::DsmlParser::new();
        feed_all(
            &mut p,
            &format!(
                "<｜DSML｜tool_calls>\
                 <｜DSML｜invoke name=\"write\">\
                 <｜DSML｜parameter name=\"content\" string=\"true\">{content}</｜DSML｜parameter｜>\
                 </｜DSML｜invoke｜>\
                 </｜DSML｜tool_calls｜>"
            ),
        );
        assert_eq!(p.state(), super::DsmlState::Done, "{}", p.error());
        assert_eq!(p.calls()[0].arg_value("content"), Some(content));
    }

    /// A tag carrying `name=` is a different malformation, not the shorthand,
    /// so it must not be silently turned into a parameter and run.
    #[test]
    fn unknown_element_with_a_name_attribute_still_errors() {
        let mut p = super::DsmlParser::new();
        feed_all(
            &mut p,
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"bash\">\
             <｜DSML｜argument name=\"command\">ls</｜DSML｜argument>",
        );
        assert_eq!(p.state(), super::DsmlState::Error);
        assert!(
            p.error().starts_with("unexpected DSML tag:"),
            "{}",
            p.error()
        );
    }

    /// The hint is only meaningful inside an open invoke; stray markup before
    /// one keeps the plain echo rather than inventing a parameter name.
    #[test]
    fn unexpected_tag_outside_an_invoke_keeps_the_plain_error() {
        let mut p = super::DsmlParser::new();
        feed_all(&mut p, "<｜DSML｜tool_calls><b>");
        assert_eq!(p.state(), super::DsmlState::Error);
        assert_eq!(p.error(), "unexpected DSML tag: <b>");
    }

    // The model copies TOOLS_PROMPT_INTRO verbatim (4 recorded occurrences).
    // Telling it "not allowed inside <think>" sends it fixing placement when the
    // real mistake is that it never substituted anything.
    #[test]
    fn placeholder_tool_name_is_named_as_such() {
        let mut p = super::DsmlParser::new();
        p.feed("<｜DSML｜tool_calls><｜DSML｜invoke name=\"$TOOL_NAME\">".as_bytes());
        assert_eq!(p.state(), super::DsmlState::Error);
        assert_eq!(
            p.error(),
            "tool name is the prompt's placeholder $TOOL_NAME, not a real tool; substitute the actual tool name"
        );
    }

    #[test]
    fn placeholder_parameter_name_is_named_as_such() {
        let mut p = super::DsmlParser::new();
        p.feed(
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"bash\">\
             <｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true\">x"
                .as_bytes(),
        );
        assert_eq!(p.state(), super::DsmlState::Error);
        assert_eq!(
            p.error(),
            "parameter name is the prompt's placeholder $PARAMETER_NAME, not a real parameter; substitute the actual parameter name"
        );
    }

    // A name with a dollar sign in it is untouched: only the three literal
    // placeholders from the tools prompt count, so a tool named `$path` is not
    // told it copied the prompt.
    #[test]
    fn dollar_inside_a_name_is_not_a_placeholder() {
        for name in ["we$rd", "$path"] {
            let mut p = super::DsmlParser::new();
            p.feed(
                format!(
                    "<｜DSML｜tool_calls><｜DSML｜invoke name=\"{name}\">\
                     </｜DSML｜invoke｜></｜DSML｜tool_calls｜>"
                )
                .as_bytes(),
            );
            assert_eq!(p.state(), super::DsmlState::Done, "error: {}", p.error());
            assert_eq!(p.calls()[0].name, name);
        }
    }

    /// The SSML alias (see [`MARKER_NAMES`]) must parse identically to the
    /// canonical spelling, including when only some tags drifted, and `raw()`
    /// must stay usable for the diagnostics that quote it.
    #[test]
    fn ssml_alias_parses_like_dsml() {
        let ssml = STANZA.replace("DSML", "SSML");
        let mixed = STANZA.replacen("DSML", "SSML", 2);
        for text in [ssml.as_str(), mixed.as_str()] {
            for mut p in [DsmlParser::new(), DsmlParser::new()] {
                feed_all(&mut p, text);
                assert_eq!(p.state(), DsmlState::Done, "{text:?}");
                assert_eq!(p.calls().len(), 1);
                assert_eq!(p.calls()[0].name, "read_file");
                assert_eq!(p.calls()[0].arg_value("path"), Some("src/main.rs"));
                assert_eq!(p.calls()[0].arg_value("offset"), Some("42"));
                assert!(!p.raw().is_empty());
            }
            let mut p = DsmlParser::new();
            feed_bytewise(&mut p, text);
            assert_eq!(p.state(), DsmlState::Done, "bytewise {text:?}");
            assert_eq!(p.calls()[0].arg_value("path"), Some("src/main.rs"));
        }
    }

    /// Only the one observed misspelling is an alias; other marker names stay
    /// unrecognized so prose cannot open a stanza.
    #[test]
    fn other_marker_names_do_not_open_a_stanza() {
        let mut p = DsmlParser::new();
        feed_all(&mut p, &STANZA.replace("DSML", "XSML"));
        assert_eq!(p.state(), DsmlState::Search);
        assert!(p.calls().is_empty());
    }

    #[test]
    fn parses_full_stanza() {
        let mut p = DsmlParser::new();
        feed_all(&mut p, STANZA);
        assert_eq!(p.state(), DsmlState::Done);
        assert_eq!(p.calls().len(), 1);
        let call = &p.calls()[0];
        assert_eq!(call.name, "read_file");
        assert_eq!(call.arg_value("path"), Some("src/main.rs"));
        assert_eq!(call.arg_value("offset"), Some("42"));
        assert_eq!(call.arg_value("missing"), None);
        assert!(call.args[0].is_string);
        assert!(!call.args[1].is_string);
    }

    #[test]
    fn parses_bytewise_identically() {
        let mut p = DsmlParser::new();
        feed_bytewise(&mut p, STANZA);
        assert_eq!(p.state(), DsmlState::Done);
        assert_eq!(p.calls().len(), 1);
        assert_eq!(p.calls()[0].arg_value("path"), Some("src/main.rs"));
    }

    // `find_tool_start` reports the *earliest* opening under any accepted
    // form, so recovery reacts to the first one the model wrote.
    #[test]
    fn find_tool_start_matches_every_accepted_wrapper_form() {
        for form in [
            "<｜DSML｜tool_calls>",
            "<DSML｜tool_calls>",
            "<｜SSML｜tool_calls>",
            "<tool_calls>",
        ] {
            let text = format!("prose {form} rest");
            assert_eq!(
                super::find_tool_start(&text),
                Some("prose ".len()),
                "{form}"
            );
        }
    }

    // Incomplete openings and the bare invoke opener are deliberately not
    // matched: acting on a weaker signal costs a forced injection.
    #[test]
    fn find_tool_start_ignores_partial_and_bare_invoke() {
        assert_eq!(super::find_tool_start("<"), None);
        assert_eq!(super::find_tool_start("<｜DSML｜tool_call"), None);
        assert_eq!(super::find_tool_start("<｜DSML｜invoke name=\"a\">"), None);
    }

    #[test]
    fn skips_leading_prose_before_marker() {
        let mut p = DsmlParser::new();
        feed_all(&mut p, "Some thinking text first. ");
        assert_eq!(p.state(), DsmlState::Search);
        feed_all(&mut p, STANZA);
        assert_eq!(p.state(), DsmlState::Done);
    }

    #[test]
    fn incomplete_input_stays_pending() {
        let mut p = DsmlParser::new();
        feed_all(
            &mut p,
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"bash\"><｜DSML｜parameter name=\"command\">ls -la",
        );
        assert_eq!(p.state(), DsmlState::ParamValue);
        assert!(p.calls().is_empty());
    }

    #[test]
    fn close_tag_variants_accepted() {
        // Whitespace and missing trailing bar in close tags are tolerated.
        let s = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"t\">",
            "<｜DSML｜parameter name=\"a\">v</｜DSML｜parameter >",
            "</｜DSML｜invoke ｜ >",
            "</｜DSML｜tool_calls>",
        );
        let mut p = DsmlParser::new();
        feed_all(&mut p, s);
        assert_eq!(p.state(), DsmlState::Done);
        assert_eq!(p.calls()[0].arg_value("a"), Some("v"));
    }

    /// A literal `</` in a parameter value (e.g. HTML written through a
    /// `write` call's `content` param) must not terminate the parameter: the
    /// cheap `</` scan in `find_close_tag_any` is only a candidate filter, and
    /// `close_tag_at` requires the full `</｜DSML｜parameter` prefix (or its
    /// dropped-bar variant) before accepting a close. This pins the safety
    /// that let `CLOSE_SCAN_HEAD` widen from `"</｜"` to `"</"`.
    #[test]
    fn literal_close_bytes_in_param_value_do_not_terminate_it() {
        // Includes a bare `</parameter>` (no DSML marker) so a validator
        // that dropped the marker check would truncate the value here.
        let html = "<div>hi</div></p> see </parameter> too";
        let s = format!(
            concat!(
                "<｜DSML｜tool_calls>",
                "<｜DSML｜invoke name=\"write\">",
                "<｜DSML｜parameter name=\"content\" string=\"true\">{html}</｜DSML｜parameter｜>",
                "</｜DSML｜invoke｜>",
                "</｜DSML｜tool_calls｜>",
            ),
            html = html
        );
        let mut p = DsmlParser::new();
        feed_all(&mut p, &s);
        assert_eq!(p.state(), DsmlState::Done);
        assert_eq!(p.calls().len(), 1);
        assert_eq!(p.calls()[0].arg_value("content"), Some(html));
    }

    #[test]
    fn multiple_invokes() {
        let s = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"a\"></｜DSML｜invoke｜>",
            "<｜DSML｜invoke name=\"b\"></｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        let mut p = DsmlParser::new();
        feed_all(&mut p, s);
        assert_eq!(p.state(), DsmlState::Done);
        let names: Vec<_> = p.calls().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["a", "b"]);
    }

    #[test]
    fn invoke_without_name_errors() {
        let mut p = DsmlParser::new();
        feed_all(&mut p, "<｜DSML｜tool_calls><｜DSML｜invoke>");
        assert_eq!(p.state(), DsmlState::Error);
        assert_eq!(p.error(), "tool invoke without name");
    }

    #[test]
    fn unexpected_tag_errors() {
        let mut p = DsmlParser::new();
        feed_all(&mut p, "<｜DSML｜tool_calls><b>");
        assert_eq!(p.state(), DsmlState::Error);
        assert!(p.error().starts_with("unexpected DSML tag:"));
    }

    #[test]
    fn param_value_may_contain_angle_brackets() {
        let s = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"write\">",
            "<｜DSML｜parameter name=\"content\">if a < b { x > y }</｜DSML｜parameter｜>",
            "</｜DSML｜invoke｜>",
            "</｜DSML｜tool_calls｜>",
        );
        let mut p = DsmlParser::new();
        feed_all(&mut p, s);
        assert_eq!(p.state(), DsmlState::Done);
        assert_eq!(
            p.calls()[0].arg_value("content"),
            Some("if a < b { x > y }")
        );
    }

    #[test]
    fn param_close_prefix_tracks_partial_close_tag() {
        let mut p = DsmlParser::new();
        feed_all(
            &mut p,
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"t\"><｜DSML｜parameter name=\"a\">v",
        );
        assert!(!p.param_close_prefix());
        feed_all(&mut p, "</｜DSML｜parameter");
        assert!(p.param_close_prefix());
        feed_all(&mut p, "｜>");
        assert!(!p.param_close_prefix());
        assert_eq!(p.state(), DsmlState::Structural);
    }

    #[test]
    fn reset_returns_to_search() {
        let mut p = DsmlParser::new();
        feed_all(&mut p, STANZA);
        p.reset();
        assert_eq!(p.state(), DsmlState::Search);
        assert!(p.calls().is_empty());
        feed_all(&mut p, STANZA);
        assert_eq!(p.state(), DsmlState::Done);
    }

    #[test]
    fn ignores_input_after_done() {
        let mut p = DsmlParser::new();
        feed_all(&mut p, STANZA);
        feed_all(&mut p, "trailing garbage <b>");
        assert_eq!(p.state(), DsmlState::Done);
        assert_eq!(p.calls().len(), 1);
    }

    // The model drops the leading fullwidth bar on inner tags (~35 recorded
    // occurrences). The opener matcher already tolerates it; without the same
    // tolerance here the stanza opens and dies on its first inner tag, and the
    // model reads "unexpected DSML tag" as a claim that its `｜` was wrong.
    #[test]
    fn inner_tags_tolerate_the_dropped_leading_bar() {
        let mut p = super::DsmlParser::new();
        p.feed(
            "<｜DSML｜tool_calls><DSML｜invoke name=\"bash\">\
             <DSML｜parameter name=\"command\" string=\"true\">ls</DSML｜parameter｜>\
             </DSML｜invoke｜></｜DSML｜tool_calls｜>"
                .as_bytes(),
        );
        assert_eq!(p.state(), super::DsmlState::Done, "error: {}", p.error());
        let calls = p.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arg_value("command"), Some("ls"));
    }

    // The canonical form must keep parsing identically.
    #[test]
    fn canonical_inner_tags_still_parse() {
        let mut p = super::DsmlParser::new();
        p.feed(
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"bash\">\
             <｜DSML｜parameter name=\"command\" string=\"true\">ls</｜DSML｜parameter｜>\
             </｜DSML｜invoke｜></｜DSML｜tool_calls｜>"
                .as_bytes(),
        );
        assert_eq!(p.state(), super::DsmlState::Done, "error: {}", p.error());
        assert_eq!(p.calls()[0].arg_value("command"), Some("ls"));
    }
}
