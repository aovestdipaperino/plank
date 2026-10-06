//! Incremental parser for Gemma 4's native tool-call syntax.
//!
//! Gemma 4 writes `<|tool_call>call:NAME{key:<|"|>text<|"|>,n:3}<tool_call|>`.
//! Strings sit between `<|"|>` delimiters with no escaping inside, which is
//! what lets a multi-line code body survive; every other value is a bare
//! scalar or a balanced `[…]`/`{…}`. Shares [`ToolCall`]/[`ToolArg`] and the
//! state enum with [`crate::dsml`], as the Qwen parser does.

use crate::dsml::{DsmlState, ToolArg, ToolCall};

pub const START: &[u8] = b"<|tool_call>";
pub const CLOSE: &[u8] = b"<tool_call|>";
pub const STR_DELIM: &[u8] = b"<|\"|>";
/// `START` followed by the literal `call:` that always follows it.
const HEAD: &[u8] = b"<|tool_call>call:";

/// Incremental parser for one or more Gemma tool-call stanzas.
///
/// Fed bytes *starting at* `<|tool_call>` — the renderer detects the opener,
/// as for [`crate::qwen::QwenParser`] — and re-derives its result from the
/// accumulated buffer on every [`feed`](Self::feed) rather than threading
/// incremental state through the grammar, so a stream split at any byte
/// boundary parses identically to one fed whole.
#[derive(Debug, Default)]
pub struct GemmaParser {
    state: DsmlState,
    raw: Vec<u8>,
    /// Offset in `raw` where the stanza not yet folded into `calls` starts.
    stanza_start: usize,
    calls: Vec<ToolCall>,
    error: Option<String>,
    /// Set once a stanza closes, while it is still open whether another
    /// `<|tool_call>` follows.
    after_call: bool,
    /// Set once the run can never produce more calls: `finish()` was called,
    /// or trailing content after a stanza ruled a second one out.
    finished: bool,
}

/// Outcome of trying to parse one stanza from the start of a byte slice.
enum Stanza {
    /// More bytes are needed. Carries the tool name once it is known, for
    /// [`GemmaParser::pending_call`].
    Incomplete(Option<String>),
    /// A whole stanza; the `usize` is the number of bytes it consumed.
    Complete(ToolCall, usize),
    Malformed(String),
}

impl GemmaParser {
    /// A parser waiting for the first byte.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parser progress.
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

    /// Snapshot of the call being streamed, with whatever name has arrived.
    #[must_use]
    pub fn pending_call(&self) -> Option<ToolCall> {
        match parse_stanza(&self.raw[self.stanza_start..]) {
            Stanza::Incomplete(Some(name)) => Some(ToolCall {
                name,
                args: Vec::new(),
            }),
            _ => None,
        }
    }

    /// Gemma has no parameter-close tag for the renderer to hold back.
    #[must_use]
    pub fn param_close_prefix(&self) -> bool {
        false
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
                _ if unterminated_string(rest) => "unterminated string".to_owned(),
                _ => "unexpected end of tool call".to_owned(),
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
            self.state = DsmlState::Structural;
            match parse_stanza(&self.raw[self.stanza_start..]) {
                Stanza::Incomplete(_) => return,
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

/// True when `rest` holds an odd number of `STR_DELIM` occurrences, meaning a
/// string was opened and never closed.
fn unterminated_string(rest: &[u8]) -> bool {
    let mut n = 0usize;
    let mut i = 0;
    while i + STR_DELIM.len() <= rest.len() {
        if rest[i..i + STR_DELIM.len()] == *STR_DELIM {
            n += 1;
            i += STR_DELIM.len();
        } else {
            i += 1;
        }
    }
    n % 2 == 1
}

/// First index `>= from` where `s[j]` is one of `:`, `}`, `,`, or a string
/// delimiter begins — the terminators a bare key can be followed by.
fn key_end_at(s: &[u8], from: usize) -> Option<usize> {
    (from..s.len()).find(|&j| matches!(s[j], b':' | b'}' | b',') || s[j..].starts_with(STR_DELIM))
}

/// Byte offset of the next `STR_DELIM` in `s`, if any.
fn find_delim(s: &[u8]) -> Option<usize> {
    s.windows(STR_DELIM.len()).position(|w| w == STR_DELIM)
}

/// Parses one stanza from the start of `s`, which begins at `<|tool_call>`.
fn parse_stanza(s: &[u8]) -> Stanza {
    if s.len() < HEAD.len() {
        return if HEAD.starts_with(s) {
            Stanza::Incomplete(None)
        } else {
            Stanza::Malformed("expected call:".to_owned())
        };
    }
    if !s.starts_with(HEAD) {
        return Stanza::Malformed("expected call:".to_owned());
    }
    let i = HEAD.len();
    let Some(brace_rel) = s[i..].iter().position(|&b| b == b'{') else {
        let partial = String::from_utf8_lossy(&s[i..]).into_owned();
        return Stanza::Incomplete((!partial.is_empty()).then_some(partial));
    };
    let name_end = i + brace_rel;
    let name = String::from_utf8_lossy(&s[i..name_end]).trim().to_owned();
    if name.is_empty() {
        return Stanza::Malformed("empty tool name".to_owned());
    }
    let mut args: Vec<ToolArg> = Vec::new();
    let end = match parse_pairs(s, name_end, &mut args) {
        Ok(Some(end)) => end,
        Ok(None) => return Stanza::Incomplete(Some(name)),
        Err(m) => return Stanza::Malformed(m),
    };
    let tail = &s[end..];
    if tail.len() < CLOSE.len() {
        return if CLOSE.starts_with(tail) {
            Stanza::Incomplete(Some(name))
        } else {
            Stanza::Malformed("junk after '}'".to_owned())
        };
    }
    if !tail.starts_with(CLOSE) {
        return Stanza::Malformed("junk after '}'".to_owned());
    }
    Stanza::Complete(ToolCall { name, args }, end + CLOSE.len())
}

/// Parses `{k:v,…}` starting at `s[at] == b'{'`, collecting each top-level
/// pair into `args`. Returns the offset just past the matching `}`, or `None`
/// if the object is not complete yet.
fn parse_pairs(s: &[u8], at: usize, args: &mut Vec<ToolArg>) -> Result<Option<usize>, String> {
    let mut i = at + 1;
    loop {
        if i >= s.len() {
            return Ok(None);
        }
        if s[i] == b'}' {
            return Ok(Some(i + 1));
        }
        let Some(key_end) = key_end_at(s, i) else {
            return Ok(None);
        };
        if s[key_end] != b':' {
            return Err("expected ':' after key".to_owned());
        }
        let key = String::from_utf8_lossy(&s[i..key_end]).trim().to_owned();
        let Some((value, is_string, next)) = parse_top_value(s, key_end + 1)? else {
            return Ok(None);
        };
        if args.iter().any(|a| a.name == key) {
            return Err(format!("duplicate key {key}"));
        }
        args.push(ToolArg {
            name: key,
            value,
            is_string,
        });
        i = next;
        if i >= s.len() {
            return Ok(None);
        }
        match s[i] {
            b',' => i += 1,
            b'}' => return Ok(Some(i + 1)),
            _ => return Err("junk after '}'".to_owned()),
        }
    }
}

/// Parses one top-level argument value at `s[i]`. Returns
/// `(value, is_string, index just past it)`, or `None` if incomplete.
///
/// A string keeps its raw bytes (`is_string: true`); a scalar keeps its raw
/// text; a list or object is converted to JSON text by [`json_value`].
fn parse_top_value(s: &[u8], i: usize) -> Result<Option<(String, bool, usize)>, String> {
    if i >= s.len() {
        return Ok(None);
    }
    if s[i..].starts_with(STR_DELIM) {
        let body = i + STR_DELIM.len();
        let Some(rel) = find_delim(&s[body..]) else {
            return Ok(None);
        };
        let text = String::from_utf8_lossy(&s[body..body + rel]).into_owned();
        return Ok(Some((text, true, body + rel + STR_DELIM.len())));
    }
    if STR_DELIM.starts_with(&s[i..]) {
        return Ok(None); // a delimiter is still arriving
    }
    if matches!(s[i], b'[' | b'{') {
        return match json_value(s, i)? {
            None => Ok(None),
            Some((json, end)) => Ok(Some((json, false, end))),
        };
    }
    let Some(end) = (i..s.len()).find(|&j| matches!(s[j], b',' | b'}')) else {
        return Ok(None);
    };
    let scalar = String::from_utf8_lossy(&s[i..end]).trim().to_owned();
    Ok(Some((scalar, false, end)))
}

/// Parses one Gemma-syntax value — string, scalar, list or object — at
/// `s[i]`, converting it to JSON text. Returns `(json, index just past it)`,
/// or `None` if incomplete.
fn json_value(s: &[u8], i: usize) -> Result<Option<(String, usize)>, String> {
    if i >= s.len() {
        return Ok(None);
    }
    if s[i..].starts_with(STR_DELIM) {
        let body = i + STR_DELIM.len();
        let Some(rel) = find_delim(&s[body..]) else {
            return Ok(None);
        };
        let text = String::from_utf8_lossy(&s[body..body + rel]);
        return Ok(Some((json_escape(&text), body + rel + STR_DELIM.len())));
    }
    if STR_DELIM.starts_with(&s[i..]) {
        return Ok(None);
    }
    match s[i] {
        b'{' => json_object(s, i),
        b'[' => json_array(s, i),
        _ => {
            let Some(end) = (i..s.len()).find(|&j| matches!(s[j], b',' | b']' | b'}')) else {
                return Ok(None);
            };
            let scalar = String::from_utf8_lossy(&s[i..end]).trim().to_owned();
            Ok(Some((scalar, end)))
        }
    }
}

/// Parses a Gemma-syntax object into JSON object text, from `s[at] == b'{'`.
fn json_object(s: &[u8], at: usize) -> Result<Option<(String, usize)>, String> {
    let mut i = at + 1;
    let mut out = String::from("{");
    let mut first = true;
    loop {
        if i >= s.len() {
            return Ok(None);
        }
        if s[i] == b'}' {
            out.push('}');
            return Ok(Some((out, i + 1)));
        }
        let Some(key_end) = key_end_at(s, i) else {
            return Ok(None);
        };
        if s[key_end] != b':' {
            return Err("expected ':' after key".to_owned());
        }
        let key = String::from_utf8_lossy(&s[i..key_end]).trim().to_owned();
        let Some((value, next)) = json_value(s, key_end + 1)? else {
            return Ok(None);
        };
        if !first {
            out.push(',');
        }
        out.push_str(&json_escape(&key));
        out.push(':');
        out.push_str(&value);
        i = next;
        first = false;
        if i >= s.len() {
            return Ok(None);
        }
        match s[i] {
            b',' => i += 1,
            b'}' => {
                out.push('}');
                return Ok(Some((out, i + 1)));
            }
            _ => return Err("junk after '}'".to_owned()),
        }
    }
}

/// Parses a Gemma-syntax list into JSON array text, from `s[at] == b'['`.
fn json_array(s: &[u8], at: usize) -> Result<Option<(String, usize)>, String> {
    let mut i = at + 1;
    let mut out = String::from("[");
    let mut first = true;
    loop {
        if i >= s.len() {
            return Ok(None);
        }
        if s[i] == b']' {
            out.push(']');
            return Ok(Some((out, i + 1)));
        }
        let Some((value, next)) = json_value(s, i)? else {
            return Ok(None);
        };
        if !first {
            out.push(',');
        }
        out.push_str(&value);
        i = next;
        first = false;
        if i >= s.len() {
            return Ok(None);
        }
        match s[i] {
            b',' => i += 1,
            b']' => {
                out.push(']');
                return Ok(Some((out, i + 1)));
            }
            _ => return Err("junk after ']'".to_owned()),
        }
    }
}

/// JSON-escapes `s` into a quoted JSON string literal.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_all(input: &str) -> GemmaParser {
        let mut p = GemmaParser::new();
        p.feed(input);
        p.finish();
        p
    }

    #[test]
    fn single_string_argument() {
        let p = parse_all("<|tool_call>call:read{path:<|\"|>src/main.rs<|\"|>}<tool_call|>");
        assert_eq!(p.state(), DsmlState::Done, "{:?}", p.error());
        assert_eq!(p.calls().len(), 1);
        let c = &p.calls()[0];
        assert_eq!(c.name, "read");
        assert_eq!(
            c.args,
            vec![ToolArg {
                name: "path".into(),
                value: "src/main.rs".into(),
                is_string: true
            }]
        );
    }

    #[test]
    fn every_scalar_kind_and_multiline_string() {
        let p = parse_all(concat!(
            "<|tool_call>call:edit{path:<|\"|>a.rs<|\"|>,old:<|\"|>fn a() {\n    x, y\n}<|\"|>,",
            "count:3,ratio:0.5,replace_all:false,extra:null}<tool_call|>"
        ));
        assert_eq!(p.state(), DsmlState::Done, "{:?}", p.error());
        let c = &p.calls()[0];
        assert_eq!(c.arg_value("old"), Some("fn a() {\n    x, y\n}"));
        let count = c.args.iter().find(|a| a.name == "count").unwrap();
        assert_eq!((count.value.as_str(), count.is_string), ("3", false));
        assert_eq!(c.arg_value("ratio"), Some("0.5"));
        assert_eq!(c.arg_value("replace_all"), Some("false"));
        assert_eq!(c.arg_value("extra"), Some("null"));
    }

    #[test]
    fn lists_and_objects_become_json() {
        let p = parse_all(concat!(
            "<|tool_call>call:fanout{subtasks:[{name:<|\"|>a<|\"|>,task:<|\"|>say \"hi\"<|\"|>}],",
            "tags:[<|\"|>x<|\"|>,<|\"|>y<|\"|>]}<tool_call|>"
        ));
        assert_eq!(p.state(), DsmlState::Done, "{:?}", p.error());
        let c = &p.calls()[0];
        let sub: serde_json::Value =
            serde_json::from_str(c.arg_value("subtasks").unwrap()).unwrap();
        assert_eq!(
            sub,
            serde_json::json!([{"name": "a", "task": "say \"hi\""}])
        );
        let tags: serde_json::Value = serde_json::from_str(c.arg_value("tags").unwrap()).unwrap();
        assert_eq!(tags, serde_json::json!(["x", "y"]));
    }

    #[test]
    fn two_stanzas_in_one_pass() {
        let p = parse_all(concat!(
            "<|tool_call>call:read{path:<|\"|>a<|\"|>}<tool_call|>\n",
            "<|tool_call>call:read{path:<|\"|>b<|\"|>}<tool_call|>"
        ));
        assert_eq!(p.state(), DsmlState::Done);
        let paths: Vec<_> = p
            .calls()
            .iter()
            .map(|c| c.arg_value("path").unwrap().to_owned())
            .collect();
        assert_eq!(paths, ["a", "b"]);
    }

    #[test]
    fn every_byte_split_gives_the_same_result() {
        let input =
            "<|tool_call>call:bash{command:<|\"|>ls -la | head<|\"|>,timeout:30}<tool_call|>";
        let whole = parse_all(input);
        for cut in 1..input.len() {
            let mut p = GemmaParser::new();
            p.feed(&input.as_bytes()[..cut]);
            p.feed(&input.as_bytes()[cut..]);
            p.finish();
            assert_eq!(p.calls(), whole.calls(), "split at {cut}");
            assert_eq!(
                p.state(),
                DsmlState::Done,
                "split at {cut}: {:?}",
                p.error()
            );
        }
    }

    #[test]
    fn strict_rejections() {
        for (input, needle) in [
            ("<|tool_call>call:{a:1}<tool_call|>", "empty tool name"),
            ("<|tool_call>call:x{a:1,a:2}<tool_call|>", "duplicate key a"),
            (
                "<|tool_call>call:x{a<|\"|>v<|\"|>}<tool_call|>",
                "expected ':' after key",
            ),
            ("<|tool_call>call:x{a:1} junk<tool_call|>", "junk after '}'"),
        ] {
            let p = parse_all(input);
            assert_eq!(p.state(), DsmlState::Error, "{input}");
            assert!(
                p.error().unwrap().contains(needle),
                "{input}: {:?}",
                p.error()
            );
        }
        let p = parse_all("<|tool_call>call:x{a:<|\"|>never closed");
        assert_eq!(p.state(), DsmlState::Error);
        assert!(p.error().unwrap().contains("unterminated string"));
    }

    #[test]
    fn pending_call_exposes_the_name_mid_stream() {
        let mut p = GemmaParser::new();
        p.feed("<|tool_call>call:write{path:<|\"|>x");
        assert_eq!(p.pending_call().map(|c| c.name), Some("write".to_owned()));
    }
}
