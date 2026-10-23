//! Incremental parser for Flint's native tool-call syntax.
//!
//! Flint writes
//! `<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value></tool_call>`,
//! with one key/value pair per argument. Values are written raw, with no
//! escaping, so a multi-line code body survives verbatim; a value that is
//! itself JSON (an object, array, number, boolean or null) is typed as
//! non-string by the Qwen parser's rule. Shares [`ToolCall`]/[`ToolArg`] and
//! the state enum with [`crate::dsml`], as the Qwen and Gemma parsers do.

use crate::dsml::{DsmlState, ToolArg, ToolCall};

pub const START: &[u8] = b"<tool_call>";
pub const CLOSE: &[u8] = b"</tool_call>";
pub const KEY_OPEN: &[u8] = b"<arg_key>";
pub const KEY_CLOSE: &[u8] = b"</arg_key>";
pub const VALUE_OPEN: &[u8] = b"<arg_value>";
pub const VALUE_CLOSE: &[u8] = b"</arg_value>";

/// Incremental parser for one or more Flint tool-call stanzas.
///
/// Fed bytes *starting at* `<tool_call>` (the renderer detects the opener,
/// as for [`crate::gemma::GemmaParser`]) and re-derives its result from the
/// accumulated buffer on every [`feed`](Self::feed), so a stream split at any
/// byte boundary parses identically to one fed whole.
#[derive(Debug, Default)]
pub struct FlintParser {
    state: DsmlState,
    raw: Vec<u8>,
    /// Offset in `raw` where the stanza not yet folded into `calls` starts.
    stanza_start: usize,
    calls: Vec<ToolCall>,
    error: Option<String>,
    /// Set once a stanza closes, while it is still open whether another
    /// `<tool_call>` follows.
    after_call: bool,
    /// Set once the run can never produce more calls: `finish()` was called,
    /// or trailing content after a stanza ruled a second one out.
    finished: bool,
}

/// Outcome of trying to parse one stanza from the start of a byte slice.
enum Stanza {
    /// More bytes are needed.
    Incomplete(Partial),
    /// A whole stanza; the `usize` is the number of bytes it consumed.
    Complete(ToolCall, usize),
    Malformed(String),
}

/// What an incomplete stanza has settled so far.
#[derive(Default)]
struct Partial {
    /// The call, once its name is final (an `<arg_key>` followed it), with
    /// every argument whose `</arg_value>` has arrived.
    call: Option<ToolCall>,
    /// Offset of the open value's first byte, while inside `<arg_value>`.
    value_start: Option<usize>,
}

impl FlintParser {
    /// A parser waiting for the first byte.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parser progress. [`DsmlState::ParamValue`] while inside an
    /// `<arg_value>`, so a literal `</think>` there stays payload.
    #[must_use]
    pub fn state(&self) -> DsmlState {
        self.state
    }

    /// The parsed calls, in stream order.
    #[must_use]
    pub fn calls(&self) -> &[ToolCall] {
        &self.calls
    }

    /// The error message, once [`DsmlState::Error`] is reached.
    #[must_use]
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Raw bytes fed so far, for diagnostics.
    #[must_use]
    pub fn raw(&self) -> &[u8] {
        &self.raw
    }

    /// Snapshot of the call being streamed: its name once an `<arg_key>`
    /// follows it, and every argument completed so far.
    #[must_use]
    pub fn pending_call(&self) -> Option<ToolCall> {
        if self.after_call || self.finished {
            return None;
        }
        match parse_stanza(&self.raw[self.stanza_start..]) {
            Stanza::Incomplete(p) => p.call,
            _ => None,
        }
    }

    /// True while inside a value whose tail is a partial `</arg_value>`, so
    /// the renderer holds those bytes back and sampling stays greedy.
    #[must_use]
    pub fn param_close_prefix(&self) -> bool {
        if self.state != DsmlState::ParamValue {
            return false;
        }
        let rest = &self.raw[self.stanza_start..];
        let Stanza::Incomplete(Partial {
            value_start: Some(at),
            ..
        }) = parse_stanza(rest)
        else {
            return false;
        };
        let value = &rest[at..];
        let Some(lt) = value.iter().rposition(|&b| b == b'<') else {
            return false;
        };
        is_partial_prefix(&value[lt..], VALUE_CLOSE)
    }

    /// True after a stanza closed, while another may still follow.
    #[must_use]
    pub fn awaits_another_stanza(&self) -> bool {
        self.after_call && !self.finished
    }

    /// Resets to a fresh parser, discarding all results.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Feeds streamed bytes. Safe to call a byte at a time; a no-op once
    /// [`DsmlState::Error`] or the run has finished.
    pub fn feed(&mut self, bytes: impl AsRef<[u8]>) {
        if matches!(self.state, DsmlState::Error) || self.finished {
            return;
        }
        self.raw.extend_from_slice(bytes.as_ref());
        self.advance();
    }

    /// Marks the generation finished: a stanza still open at this point can
    /// never complete, so it is reported as an error rather than left waiting.
    pub fn finish(&mut self) {
        if self.finished || matches!(self.state, DsmlState::Error) {
            return;
        }
        self.finished = true;
        let rest = &self.raw[self.stanza_start..];
        let pending = !rest.iter().all(u8::is_ascii_whitespace);
        if pending && !self.after_call {
            let msg = match parse_stanza(rest) {
                Stanza::Malformed(m) => m,
                _ => "unclosed tool call".to_owned(),
            };
            self.fail(msg);
        } else if !self.calls.is_empty() {
            self.state = DsmlState::Done;
        }
    }

    fn fail(&mut self, msg: String) {
        self.state = DsmlState::Error;
        self.error = Some(msg);
    }

    /// Parses as much of `raw[stanza_start..]` as is available, looping over
    /// complete stanzas and the whitespace between them.
    fn advance(&mut self) {
        loop {
            let rest = &self.raw[self.stanza_start..];
            if self.after_call {
                let skip = rest.iter().take_while(|b| b.is_ascii_whitespace()).count();
                let tail = &rest[skip..];
                if tail.is_empty() || is_partial_prefix(tail, START) {
                    return; // more bytes could still start another stanza
                }
                if !tail.starts_with(START) {
                    // Visible text after the calls ends the run; the renderer
                    // owns that text, not this parser.
                    self.after_call = false;
                    self.finished = true;
                    self.state = DsmlState::Done;
                    return;
                }
                self.stanza_start += skip;
                self.after_call = false;
                continue;
            }
            match parse_stanza(&self.raw[self.stanza_start..]) {
                Stanza::Incomplete(p) => {
                    self.state = if p.value_start.is_some() {
                        DsmlState::ParamValue
                    } else {
                        DsmlState::Structural
                    };
                    return;
                }
                Stanza::Malformed(m) => {
                    self.fail(m);
                    return;
                }
                Stanza::Complete(call, used) => {
                    self.calls.push(call);
                    self.stanza_start += used;
                    self.after_call = true;
                    self.state = DsmlState::Done;
                }
            }
        }
    }
}

/// Whether `s` is a non-empty proper prefix of `pat`, so more bytes could
/// still complete it.
fn is_partial_prefix(s: &[u8], pat: &[u8]) -> bool {
    !s.is_empty() && s.len() < pat.len() && pat.starts_with(s)
}

/// Byte offset of the first `pat` in `s`, if any.
fn find(s: &[u8], pat: &[u8]) -> Option<usize> {
    s.windows(pat.len()).position(|w| w == pat)
}

/// Index of the first non-whitespace byte at or after `i`.
fn skip_ws(s: &[u8], i: usize) -> usize {
    i + s[i..]
        .iter()
        .take_while(|b| b.is_ascii_whitespace())
        .count()
}

/// Parses one stanza from the start of `s`, which begins at `<tool_call>`.
fn parse_stanza(s: &[u8]) -> Stanza {
    let incomplete = |call: Option<ToolCall>| {
        Stanza::Incomplete(Partial {
            call,
            value_start: None,
        })
    };
    if !s.starts_with(START) {
        return if START.starts_with(s) {
            incomplete(None)
        } else {
            Stanza::Malformed("expected <tool_call>".to_owned())
        };
    }
    let Some(lt) = s[START.len()..].iter().position(|&b| b == b'<') else {
        return incomplete(None);
    };
    let name_end = START.len() + lt;
    let name = String::from_utf8_lossy(&s[START.len()..name_end])
        .trim()
        .to_owned();
    if name.is_empty() {
        return Stanza::Malformed("empty tool name".to_owned());
    }
    if name.chars().any(char::is_whitespace) {
        return Stanza::Malformed(format!("invalid tool name {name:?}"));
    }
    let mut call = ToolCall {
        name,
        args: Vec::new(),
    };
    // The name is final only once a whole `<arg_key>` follows it.
    let mut named = false;
    let mut i = name_end;
    loop {
        i = skip_ws(s, i);
        let rest = &s[i..];
        if rest.starts_with(CLOSE) {
            return Stanza::Complete(call, i + CLOSE.len());
        }
        if rest.starts_with(VALUE_OPEN) {
            return Stanza::Malformed("value without a key".to_owned());
        }
        if !rest.starts_with(KEY_OPEN) {
            if rest.is_empty()
                || [CLOSE, KEY_OPEN, VALUE_OPEN]
                    .iter()
                    .any(|t| is_partial_prefix(rest, t))
            {
                return incomplete(named.then_some(call));
            }
            return Stanza::Malformed("junk between tool-call tags".to_owned());
        }
        named = true;
        let key_at = i + KEY_OPEN.len();
        let Some(rel) = find(&s[key_at..], KEY_CLOSE) else {
            return incomplete(Some(call));
        };
        let key = String::from_utf8_lossy(&s[key_at..key_at + rel])
            .trim()
            .to_owned();
        if key.is_empty() {
            return Stanza::Malformed("empty argument key".to_owned());
        }
        // A key may repeat with the very same value (Flint ends some `edit`
        // calls with their `path` again); the repeat is dropped once its
        // value is read. A repeat with another value is an error.
        let seen = call.args.iter().position(|a| a.name == key);
        i = skip_ws(s, key_at + rel + KEY_CLOSE.len());
        let rest = &s[i..];
        if !rest.starts_with(VALUE_OPEN) {
            if rest.is_empty() || is_partial_prefix(rest, VALUE_OPEN) {
                return incomplete(Some(call));
            }
            return Stanza::Malformed(format!("junk after </arg_key> for key {key}"));
        }
        let value_at = i + VALUE_OPEN.len();
        let Some(rel) = find(&s[value_at..], VALUE_CLOSE) else {
            return Stanza::Incomplete(Partial {
                call: Some(call),
                value_start: Some(value_at),
            });
        };
        let raw_value = &s[value_at..value_at + rel];
        let value = String::from_utf8_lossy(raw_value).into_owned();
        match seen {
            Some(at) if call.args[at].value == value => {}
            Some(_) => {
                return Stanza::Malformed(format!(
                    "argument {key} given twice with different values; give each argument once"
                ));
            }
            None => call.args.push(ToolArg {
                name: key,
                value,
                is_string: !crate::qwen::value_is_json(raw_value),
            }),
        }
        i = value_at + rel + VALUE_CLOSE.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_all(s: &str) -> FlintParser {
        let mut p = FlintParser::new();
        p.feed(s.as_bytes());
        p.finish();
        p
    }

    #[test]
    fn one_call_with_string_and_json_values() {
        let p = parse_all(
            "<tool_call>bash<arg_key>command</arg_key><arg_value>ls -la | head</arg_value><arg_key>timeout</arg_key><arg_value>30</arg_value></tool_call>",
        );
        assert_eq!(p.state(), DsmlState::Done);
        let c = &p.calls()[0];
        assert_eq!(c.name, "bash");
        assert_eq!(c.args[0].name, "command");
        assert_eq!(c.args[0].value, "ls -la | head");
        assert!(c.args[0].is_string);
        assert_eq!(c.args[1].value, "30");
        assert!(!c.args[1].is_string, "30 is JSON");
    }

    #[test]
    fn multi_line_values_and_marker_lookalikes_survive() {
        let body = "fn a() {\n    let s = \"</arg\";\n}\n";
        let p = parse_all(&format!(
            "<tool_call>write<arg_key>path</arg_key><arg_value>a.rs</arg_value><arg_key>content</arg_key><arg_value>{body}</arg_value></tool_call>"
        ));
        assert_eq!(p.state(), DsmlState::Done);
        assert_eq!(p.calls()[0].arg_value("content"), Some(body));
    }

    #[test]
    fn several_calls_back_to_back() {
        let p = parse_all(
            "<tool_call>read<arg_key>path</arg_key><arg_value>a</arg_value></tool_call><tool_call>read<arg_key>path</arg_key><arg_value>b</arg_value></tool_call>",
        );
        assert_eq!(p.calls().len(), 2);
        assert_eq!(p.calls()[1].arg_value("path"), Some("b"));
    }

    #[test]
    fn a_call_without_arguments() {
        let p = parse_all("<tool_call>list_jobs</tool_call>");
        assert_eq!(p.calls()[0].name, "list_jobs");
        assert!(p.calls()[0].args.is_empty());
    }

    #[test]
    fn split_at_every_byte_parses_the_same() {
        let s = "<tool_call>edit<arg_key>old</arg_key><arg_value>a\nb</arg_value><arg_key>n</arg_key><arg_value>{\"x\": 1}</arg_value></tool_call>";
        let whole = parse_all(s);
        for cut in 1..s.len() {
            let mut p = FlintParser::new();
            p.feed(&s.as_bytes()[..cut]);
            p.feed(&s.as_bytes()[cut..]);
            p.finish();
            assert_eq!(p.calls(), whole.calls(), "cut at {cut}");
        }
    }

    /// Flint sometimes ends an `edit` call with its `path` again, the same
    /// value as the first: the call is unambiguous and parses as if the
    /// repeat were not there.
    #[test]
    fn a_key_repeated_with_the_same_value_is_dropped() {
        let p = parse_all(
            "<tool_call>edit<arg_key>path</arg_key><arg_value>/a.rs</arg_value>\
             <arg_key>old</arg_key><arg_value>x</arg_value>\
             <arg_key>new</arg_key><arg_value>y</arg_value>\
             <arg_key>path</arg_key><arg_value>/a.rs</arg_value></tool_call>",
        );
        assert_eq!(p.state(), DsmlState::Done);
        let calls = p.calls();
        assert_eq!(calls.len(), 1);
        let names: Vec<&str> = calls[0].args.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["path", "old", "new"]);
        assert_eq!(calls[0].args[0].value, "/a.rs");
    }

    #[test]
    fn malformed_stanzas_are_errors() {
        for (s, why) in [
            (
                "<tool_call><arg_key>a</arg_key><arg_value>1</arg_value></tool_call>",
                "empty tool name",
            ),
            (
                "<tool_call>x<arg_key>a</arg_key><arg_value>1</arg_value><arg_key>a</arg_key><arg_value>2</arg_value></tool_call>",
                "argument a given twice with different values",
            ),
            (
                "<tool_call>x<arg_key>a</arg_key>junk<arg_value>1</arg_value></tool_call>",
                "junk",
            ),
            (
                "<tool_call>x<arg_value>1</arg_value></tool_call>",
                "value without a key",
            ),
        ] {
            let p = parse_all(s);
            assert_eq!(p.state(), DsmlState::Error, "{s}");
            assert!(
                p.error().unwrap_or_default().contains(why),
                "{s}: {:?}",
                p.error()
            );
        }
    }

    #[test]
    fn an_unclosed_stanza_at_finish_is_an_error() {
        let p = parse_all("<tool_call>bash<arg_key>command</arg_key><arg_value>ls");
        assert_eq!(p.state(), DsmlState::Error);
    }

    #[test]
    fn pending_call_names_the_tool_once_a_key_follows() {
        let mut p = FlintParser::new();
        p.feed("<tool_call>write");
        assert_eq!(p.pending_call(), None, "the name may still grow");
        p.feed("<arg_key>path</arg_key><arg_value>a.rs</arg_value><arg_key>content");
        let call = p.pending_call().expect("named");
        assert_eq!(call.name, "write");
        assert_eq!(call.arg_value("path"), Some("a.rs"));
        assert_eq!(call.args.len(), 1, "only completed arguments");
    }

    #[test]
    fn param_close_prefix_tracks_a_partial_close_tag() {
        let mut p = FlintParser::new();
        p.feed("<tool_call>write<arg_key>content</arg_key><arg_value>a < b");
        assert_eq!(p.state(), DsmlState::ParamValue);
        assert!(!p.param_close_prefix(), "plain value text");
        p.feed("</arg_");
        assert!(p.param_close_prefix(), "partial close tag");
        p.feed("value>");
        assert!(!p.param_close_prefix(), "close tag consumed");
        assert_eq!(p.state(), DsmlState::Structural);
    }

    #[test]
    fn text_after_a_stanza_ends_the_run() {
        let mut p = FlintParser::new();
        p.feed("<tool_call>list_jobs</tool_call>\n");
        assert!(p.awaits_another_stanza());
        p.feed("done");
        assert!(!p.awaits_another_stanza());
        assert_eq!(p.state(), DsmlState::Done);
        assert_eq!(p.calls().len(), 1);
    }
}
