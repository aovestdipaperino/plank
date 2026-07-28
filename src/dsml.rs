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
        tag_prefixes(marker, closing, name)
            .into_iter()
            .find(|prefix| s.starts_with(prefix.as_bytes()))
            .map(|prefix| prefix.len())
    })
}

/// True when `s` is a (possibly incomplete) prefix of a tag opener for `name`
/// under any accepted marker, in either the canonical or dropped-bar form.
pub(crate) fn tag_prefix_partial(s: &[u8], closing: bool, name: &str) -> bool {
    MARKER_NAMES.iter().any(|marker| {
        tag_prefixes(marker, closing, name)
            .iter()
            .any(|prefix| prefix.as_bytes().starts_with(s))
    })
}

/// The accepted spellings of a tag prefix: canonical first, then the
/// dropped-leading-bar typo the model actually emits.
fn tag_prefixes(marker: &str, closing: bool, name: &str) -> [String; 2] {
    let slash = if closing { "/" } else { "" };
    [
        format!("<{slash}｜{marker}｜{name}"),
        format!("<{slash}{marker}｜{name}"),
    ]
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
        forms.push(format!("<{m}｜tool_calls>"));
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
                if self.search_tail.ends_with(DSML_START) || self.search_tail.ends_with(SSML_START)
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
                    let Some((end, tag_len)) =
                        find_close_tag(&self.raw[self.param_value_start..], "parameter")
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

                    if open_tag_is(&tag, "invoke") {
                        let Some(name) = parse_attr(&tag, "name") else {
                            self.set_error("tool invoke without name");
                            return;
                        };
                        self.current = Some(PendingCall {
                            name,
                            args: Vec::new(),
                        });
                        self.parse_pos += tag_len;
                    } else if open_tag_is(&tag, "parameter") {
                        let Some(name) = parse_attr(&tag, "name") else {
                            self.set_error("tool parameter without name");
                            return;
                        };
                        self.param_name = Some(name);
                        self.param_is_string =
                            parse_attr(&tag, "string").as_deref() == Some("true");
                        self.parse_pos += tag_len;
                        self.param_value_start = self.parse_pos;
                        self.param_close_prefix = false;
                        self.state = DsmlState::ParamValue;
                    } else {
                        let shown: String = tag.chars().take(80).collect();
                        self.set_error(format!("unexpected DSML tag: {shown}"));
                        return;
                    }
                }
                _ => return,
            }
        }
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

/// Finds a DSML closing tag for `name` in `s`; returns (offset, tag length).
fn find_close_tag(s: &[u8], name: &str) -> Option<(usize, usize)> {
    let mut from = 0;
    while let Some(pos) = find_bytes(&s[from..], CLOSE_SCAN_HEAD) {
        let at = from + pos;
        if let Some(tag_len) = close_tag_at(&s[at..], name) {
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
    /// cheap `</` scan in `find_close_tag` is only a candidate filter, and
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
