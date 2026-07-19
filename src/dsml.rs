//! Streaming DSML tool-call parser.
//!
//! The model streams raw text tokens. This parser recognizes completed DSML
//! tool stanzas (`<｜DSML｜tool_calls>` ... `</｜DSML｜tool_calls｜>`) and keeps
//! a copy of the raw stanza for diagnostics. It is deliberately strict after
//! the opening marker: typo recovery belongs to the streaming detector so the
//! actual tool parser stays small and predictable.
//!
//! Port of the `agent_dsml_*` family from `ds4_agent.c`.

const DSML_START: &[u8] = "<｜DSML｜tool_calls>".as_bytes();
const DSML_OPEN_MARKER: &str = "<｜DSML｜";
const DSML_CLOSE_MARKER: &[u8] = "</｜DSML｜".as_bytes();
const DSML_BAR: &[u8] = "｜".as_bytes();

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
                if self.search_tail.ends_with(DSML_START) {
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
        if tail.len() > 64 || !tail.starts_with(DSML_CLOSE_MARKER) {
            return;
        }
        let mut complete = false;
        self.param_close_prefix = parameter_close_tail(tail, &mut complete) && !complete;
    }
}

/// Checks whether `tag` is an opening DSML tag with the given element name.
fn open_tag_is(tag: &str, name: &str) -> bool {
    let prefix = format!("{DSML_OPEN_MARKER}{name}");
    tag.strip_prefix(&prefix)
        .and_then(|rest| rest.bytes().next())
        .is_some_and(|c| c == b'>' || (c as char).is_ascii_whitespace())
}

/// Recognizes a DSML closing tag at the start of `s`, returning its length.
///
/// Accepts the few harmless closing-tag variants the model has been observed
/// to emit (whitespace and an optional trailing `｜` before `>`). Opening tags
/// stay strict so accidental prose does not become a tool call.
fn close_tag_at(s: &[u8], name: &str) -> Option<usize> {
    let prefix = format!("</｜DSML｜{name}");
    let prefix = prefix.as_bytes();
    if !s.starts_with(prefix) {
        return None;
    }
    let mut i = prefix.len();
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
    while let Some(pos) = find_bytes(&s[from..], DSML_CLOSE_MARKER) {
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
    const PREFIX: &[u8] = "</｜DSML｜parameter".as_bytes();
    *complete = false;
    if tail.len() <= PREFIX.len() {
        return PREFIX.starts_with(tail);
    }
    if !tail.starts_with(PREFIX) {
        return false;
    }
    let mut i = PREFIX.len();
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
}
