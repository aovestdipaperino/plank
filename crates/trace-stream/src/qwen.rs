//! Incremental parser for the Qwen3.8-Flash-Next tool-call syntax.
//!
//! Qwen3.8 does not emit DSML. It writes its own dialect, which the C engine
//! parses in `agent_qwen_tool_parse` (`refs/ds4/ds4_agent.c`):
//!
//! ```text
//! <tool_call>
//! <function=write>
//! <parameter=path>
//! src/main.rs
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! This is a port of that function, sharing [`ToolCall`]/[`ToolArg`] and the
//! state enum with [`crate::dsml`] exactly as the C shares one
//! `agent_dsml_parser` across all three dialects.

use crate::dsml::{DsmlState, ToolArg, ToolCall, unescape_close_delimiter};

const START: &[u8] = b"<tool_call>";
const CLOSE: &[u8] = b"</tool_call>";
const FN_OPEN: &[u8] = b"<function=";
const FN_CLOSE: &[u8] = b"</function>";
const PARAM_OPEN: &[u8] = b"<parameter=";
const PARAM_CLOSE: &[u8] = b"</parameter>";

/// Incremental parser for one or more `<tool_call>` stanzas.
#[derive(Debug, Default)]
pub struct QwenParser {
    state: DsmlState,
    raw: Vec<u8>,
    parse_pos: usize,
    current: Option<ToolCall>,
    param_name: Option<String>,
    param_value_start: usize,
    /// Set after a stanza closes, while deciding whether another follows.
    after_call: bool,
    calls: Vec<ToolCall>,
    error: Option<String>,
}

impl QwenParser {
    /// A parser waiting for the first byte.
    ///
    /// Starts in `Search`, like `DsmlParser`, and *must*: the renderer reads
    /// `Structural` as "a stanza is open and unfinished" and reports an
    /// incomplete tool call for it. Starting there made every plain-prose
    /// generation report a phantom incomplete call, which fed an error back to
    /// a model that had done nothing wrong — and it answered, was told again,
    /// and looped.
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

    /// Snapshot of the call being parsed, for mid-stream preflight.
    #[must_use]
    pub fn pending_call(&self) -> Option<ToolCall> {
        self.current.clone()
    }

    /// Raw bytes of the stanza accumulated so far, for diagnostics.
    #[must_use]
    pub fn raw(&self) -> &[u8] {
        &self.raw
    }

    /// True while the tail of the open value is a partial `</parameter>`.
    ///
    /// Derived rather than tracked: the renderer uses it to force greedy
    /// sampling through a close tag, and the tail is cheap to re-check from
    /// the last `<` in the value.
    #[must_use]
    pub fn param_close_prefix(&self) -> bool {
        if self.state != DsmlState::ParamValue {
            return false;
        }
        let value = &self.raw[self.param_value_start.min(self.raw.len())..];
        let Some(lt) = value.iter().rposition(|&b| b == b'<') else {
            return false;
        };
        is_partial_prefix(&value[lt..], PARAM_CLOSE)
    }

    /// Resets to a fresh parser, discarding all results.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Feeds streamed bytes. Safe to call a byte at a time.
    ///
    /// Incomplete input leaves the state alone until enough bytes arrive;
    /// input that cannot become a valid stanza switches to
    /// [`DsmlState::Error`] so the model gets a retryable tool error. A no-op
    /// once terminal, matching `DsmlParser::feed`.
    pub fn feed(&mut self, bytes: impl AsRef<[u8]>) {
        if matches!(self.state, DsmlState::Done | DsmlState::Error) {
            return;
        }
        self.raw.extend_from_slice(bytes.as_ref());
        self.parse();
    }

    /// Marks the generation finished, which is what completes a stanza here.
    ///
    /// Unlike DSML, this dialect has no terminator that ends the *run*: after
    /// `</tool_call>` another `<tool_call>` may follow, so the C sits in its
    /// structural state waiting to find out and only reaches `DONE` when
    /// trailing content rules a second call out. At end of generation there is
    /// no trailing content to read, so the caller has to say so.
    pub fn finish(&mut self) {
        if self.state == DsmlState::Structural && !self.calls.is_empty() {
            self.state = DsmlState::Done;
        } else if self.state == DsmlState::ParamValue {
            self.set_error("unterminated <parameter> in Qwen tool call");
        }
    }

    fn set_error(&mut self, msg: &str) {
        self.state = DsmlState::Error;
        self.error = Some(msg.to_owned());
    }

    /// Port of `agent_qwen_tool_parse`.
    fn parse(&mut self) {
        // The C refuses to parse until the buffer opens with the marker, which
        // is also what keeps ordinary prose from being read as a call.
        if !self.raw.starts_with(START) {
            // Still short enough to become the opener: keep waiting. Anything
            // else is prose, and stays prose.
            if !START.starts_with(&self.raw) {
                self.state = DsmlState::Search;
            }
            return;
        }
        if self.state == DsmlState::Search {
            self.state = DsmlState::Structural;
        }
        if self.parse_pos == 0 {
            self.parse_pos = START.len();
        }
        while matches!(self.state, DsmlState::Structural | DsmlState::ParamValue) {
            if self.state == DsmlState::ParamValue {
                if !self.parse_param_value() {
                    return;
                }
                continue;
            }
            if !self.parse_structural() {
                return;
            }
        }
    }

    /// Closes the open parameter. Returns false to wait for more bytes (or on
    /// error), true when the value was taken and parsing should continue.
    fn parse_param_value(&mut self) -> bool {
        let from = self.param_value_start;
        let Some(rel) = find(&self.raw[from..], PARAM_CLOSE) else {
            // A stanza that ends while a value is still open can never
            // complete: say so now rather than waiting forever.
            if find(&self.raw[from..], CLOSE).is_some() {
                self.set_error("unterminated <parameter> in Qwen tool call");
            }
            return false;
        };
        let value_end = from + rel;
        // The syntax puts a newline on each side of the value; exactly one
        // comes off each end, so an intentionally blank line survives.
        let mut vs = from;
        let mut ve = value_end;
        if vs < ve && self.raw[vs] == b'\n' {
            vs += 1;
        }
        if ve > vs && self.raw[ve - 1] == b'\n' {
            ve -= 1;
        }
        let raw_value = &self.raw[vs..ve];
        let is_string = !value_is_json(raw_value);
        let value_bytes = if is_string {
            unescape_close_delimiter(raw_value, &PARAM_CLOSE[1..])
        } else {
            raw_value.to_vec()
        };
        let arg = ToolArg {
            name: self.param_name.take().unwrap_or_default(),
            value: String::from_utf8_lossy(&value_bytes).into_owned(),
            is_string,
        };
        self.current
            .get_or_insert_with(Default::default)
            .args
            .push(arg);
        self.parse_pos = value_end + PARAM_CLOSE.len();
        self.state = DsmlState::Structural;
        true
    }

    /// Consumes one structural token. Returns false to wait or stop.
    fn parse_structural(&mut self) -> bool {
        while self
            .raw
            .get(self.parse_pos)
            .is_some_and(|b| matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
        {
            self.parse_pos += 1;
        }
        if self.parse_pos >= self.raw.len() {
            return false;
        }
        let cur = &self.raw[self.parse_pos..];

        // Just closed a stanza: another may follow, and anything else ends the
        // run rather than being an error — the model is allowed to stop here.
        if self.after_call {
            if cur.starts_with(START) {
                self.parse_pos += START.len();
                self.after_call = false;
                return true;
            }
            if is_partial_prefix(cur, START) {
                return false;
            }
            self.after_call = false;
            self.state = DsmlState::Done;
            return false;
        }

        if self.current.is_none() {
            if !cur.starts_with(FN_OPEN) {
                if is_partial_prefix(cur, FN_OPEN) {
                    return false;
                }
                self.set_error("expected <function=...> in Qwen tool call");
                return false;
            }
            let name = match tag_operand(cur, FN_OPEN) {
                Operand::Pending => return false,
                Operand::Blank => {
                    self.set_error("Qwen tool call without function name");
                    return false;
                }
                Operand::Found(n) => n,
            };
            self.parse_pos += FN_OPEN.len() + name.consumed;
            self.current = Some(ToolCall {
                name: name.text,
                args: Vec::new(),
            });
            return true;
        }

        if cur.starts_with(FN_CLOSE) {
            let after = skip_ascii_space(&cur[FN_CLOSE.len()..]);
            let rest = &cur[FN_CLOSE.len() + after..];
            if rest.is_empty() || is_partial_prefix(rest, CLOSE) {
                return false;
            }
            if !rest.starts_with(CLOSE) {
                self.set_error("expected </tool_call> after </function>");
                return false;
            }
            self.parse_pos += FN_CLOSE.len() + after + CLOSE.len();
            if let Some(call) = self.current.take() {
                self.calls.push(call);
            }
            self.after_call = true;
            return true;
        }
        if is_partial_prefix(cur, FN_CLOSE) {
            return false;
        }

        if !cur.starts_with(PARAM_OPEN) {
            if is_partial_prefix(cur, PARAM_OPEN) {
                return false;
            }
            self.set_error("expected <parameter=...> or </function> in Qwen tool call");
            return false;
        }
        let key = match tag_operand(cur, PARAM_OPEN) {
            Operand::Pending => return false,
            Operand::Blank => {
                self.set_error("empty <parameter=> name in Qwen tool call");
                return false;
            }
            Operand::Found(k) => k,
        };
        self.param_name = Some(key.text);
        self.param_value_start = self.parse_pos + PARAM_OPEN.len() + key.consumed;
        self.parse_pos = self.param_value_start;
        self.state = DsmlState::ParamValue;
        true
    }
}

/// The trimmed text between an opening tag's `=` and its `>`.
struct TagOperand {
    text: String,
    /// Bytes after the tag prefix up to and including the `>`.
    consumed: usize,
}

/// Outcome of reading an opening tag's operand. The three cases are genuinely
/// different actions: keep waiting, fail, or proceed.
enum Operand {
    /// The `>` has not arrived yet.
    Pending,
    /// The `>` arrived but the operand is blank — an error, not a wait.
    Blank,
    Found(TagOperand),
}

/// Reads the operand of `<function=NAME>` / `<parameter=KEY>`.
fn tag_operand(cur: &[u8], prefix: &[u8]) -> Operand {
    let after = &cur[prefix.len()..];
    let Some(gt) = find(after, b">") else {
        return Operand::Pending;
    };
    let text = String::from_utf8_lossy(&after[..gt]).trim().to_owned();
    if text.is_empty() {
        return Operand::Blank;
    }
    Operand::Found(TagOperand {
        text,
        consumed: gt + 1,
    })
}

/// Count of leading ASCII whitespace, mirroring `agent_skip_ascii_space` —
/// which despite the name also skips `\r` and `\n`, and has to: the syntax
/// puts a newline between `</function>` and `</tool_call>`.
fn skip_ascii_space(s: &[u8]) -> usize {
    s.iter()
        .take_while(|b| matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
        .count()
}

/// Whether `s` is a non-empty proper prefix of `pat`, so more bytes could
/// still complete it. Mirrors `agent_bytes_partial_prefix_at`.
fn is_partial_prefix(s: &[u8], pat: &[u8]) -> bool {
    !s.is_empty() && s.len() < pat.len() && pat.starts_with(s)
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Whether a value reads as JSON, so it is not a string parameter.
///
/// Port of `agent_qwen_value_is_json`. Leading spaces and tabs are trimmed but
/// not newlines; trailing newlines are. The C finishes with `strtod` over a
/// 64-byte buffer, so anything longer is never a number.
fn value_is_json(v: &[u8]) -> bool {
    let start = v.iter().take_while(|b| matches!(b, b' ' | b'\t')).count();
    let v = &v[start..];
    let mut n = v.len();
    while n > 0 && matches!(v[n - 1], b' ' | b'\t' | b'\n') {
        n -= 1;
    }
    let v = &v[..n];
    if v.is_empty() {
        return false;
    }
    let (first, last) = (v[0], v[n - 1]);
    if (first == b'{' && last == b'}') || (first == b'[' && last == b']') {
        return true;
    }
    if matches!(v, b"true" | b"false" | b"null") {
        return true;
    }
    if n >= 64 {
        return false;
    }
    let Ok(text) = std::str::from_utf8(v) else {
        return false;
    };
    // `strtod` also takes C99 hex floats, which Rust's parser rejects; keeping
    // that case matches the C on values like `0x10`.
    text.parse::<f64>().is_ok()
        || text
            .strip_prefix("0x")
            .or_else(|| text.strip_prefix("0X"))
            .is_some_and(|h| !h.is_empty() && i64::from_str_radix(h, 16).is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_all(p: &mut QwenParser, s: &str) {
        p.feed(s.as_bytes());
    }

    fn feed_bytewise(p: &mut QwenParser, s: &str) {
        for b in s.as_bytes() {
            p.feed([*b]);
        }
    }

    /// The C's `test_agent_tool_argument_literal_markup` fixture for this
    /// dialect (the `qwen`/`expected` arrays), byte for byte. Also pins that
    /// the value keeps its inner newlines while losing exactly the one
    /// newline each side of it that the syntax contributes.
    const C_LITERAL_MARKUP: &str = concat!(
        "<tool_call>\n<function=write>\n<parameter=content>\n<p>&amp; &lt;</p> ",
        "&lt;/parameter> &amp;lt;/parameter>\n</parameter>\n</function>\n</tool_call>",
    );
    const C_LITERAL_MARKUP_EXPECTED: &str = "<p>&amp; &lt;</p> </parameter> &lt;/parameter>";

    #[test]
    fn parses_the_c_literal_markup_fixture() {
        let mut p = QwenParser::new();
        feed_all(&mut p, C_LITERAL_MARKUP);
        p.finish();
        assert_eq!(p.state(), DsmlState::Done, "error: {:?}", p.error());
        assert_eq!(p.calls().len(), 1);
        assert_eq!(p.calls()[0].name, "write");
        assert_eq!(
            p.calls()[0].arg_value("content"),
            Some(C_LITERAL_MARKUP_EXPECTED)
        );
    }

    #[test]
    fn parses_the_c_fixture_bytewise() {
        let mut p = QwenParser::new();
        feed_bytewise(&mut p, C_LITERAL_MARKUP);
        p.finish();
        assert_eq!(p.state(), DsmlState::Done, "error: {:?}", p.error());
        assert_eq!(
            p.calls()[0].arg_value("content"),
            Some(C_LITERAL_MARKUP_EXPECTED)
        );
    }

    #[test]
    fn parses_multiple_parameters_in_order() {
        let mut p = QwenParser::new();
        feed_all(
            &mut p,
            "<tool_call>\n<function=edit>\n<parameter=path>\nsrc/a.rs\n</parameter>\n\
             <parameter=old>\none\n</parameter>\n<parameter=new>\ntwo\n</parameter>\n\
             </function>\n</tool_call>",
        );
        p.finish();
        assert_eq!(p.state(), DsmlState::Done, "error: {:?}", p.error());
        let names: Vec<&str> = p.calls()[0].args.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["path", "old", "new"]);
        assert_eq!(p.calls()[0].arg_value("new"), Some("two"));
    }

    /// The C consumes a following `<tool_call>` and keeps going, so one
    /// generation can carry several calls.
    #[test]
    fn parses_two_consecutive_calls() {
        let mut p = QwenParser::new();
        feed_all(
            &mut p,
            "<tool_call>\n<function=list>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>\n\
             <tool_call>\n<function=read>\n<parameter=path>\na\n</parameter>\n</function>\n</tool_call>",
        );
        p.finish();
        assert_eq!(p.state(), DsmlState::Done, "error: {:?}", p.error());
        assert_eq!(p.calls().len(), 2);
        assert_eq!(p.calls()[1].name, "read");
    }

    /// A value that reads as JSON is not a string, so it is not unescaped —
    /// `agent_qwen_value_is_json` decides, and the arg carries the flag on.
    #[test]
    fn json_shaped_values_are_not_strings() {
        let mut p = QwenParser::new();
        feed_all(
            &mut p,
            "<tool_call>\n<function=bash>\n<parameter=timeout_sec>\n30\n</parameter>\n\
             <parameter=command>\nls\n</parameter>\n</function>\n</tool_call>",
        );
        p.finish();
        assert_eq!(p.state(), DsmlState::Done, "error: {:?}", p.error());
        let args = &p.calls()[0].args;
        assert!(!args[0].is_string, "30 is JSON");
        assert!(args[1].is_string, "ls is not JSON");
    }

    #[test]
    fn multiline_values_keep_their_inner_newlines() {
        let mut p = QwenParser::new();
        feed_all(
            &mut p,
            "<tool_call>\n<function=write>\n<parameter=content>\nline one\nline two\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(
            p.calls()[0].arg_value("content"),
            Some("line one\nline two")
        );
    }

    #[test]
    fn incomplete_input_waits_rather_than_erroring() {
        for cut in 1..C_LITERAL_MARKUP.len() {
            let mut p = QwenParser::new();
            feed_all(&mut p, &C_LITERAL_MARKUP[..cut]);
            assert_ne!(
                p.state(),
                DsmlState::Error,
                "errored on a {cut}-byte prefix: {:?}",
                p.error()
            );
        }
    }

    #[test]
    fn a_missing_function_tag_errors() {
        let mut p = QwenParser::new();
        feed_all(&mut p, "<tool_call>\n<parameter=path>\na\n</parameter>\n");
        assert_eq!(p.state(), DsmlState::Error);
        assert!(p.error().unwrap().contains("<function="));
    }

    #[test]
    fn an_empty_parameter_name_errors() {
        let mut p = QwenParser::new();
        feed_all(
            &mut p,
            "<tool_call>\n<function=write>\n<parameter=>\nv\n</parameter>\n",
        );
        assert_eq!(p.state(), DsmlState::Error);
        assert!(p.error().unwrap().contains("empty"));
    }

    #[test]
    fn a_parameter_closed_by_the_stanza_errors() {
        let mut p = QwenParser::new();
        feed_all(
            &mut p,
            "<tool_call>\n<function=write>\n<parameter=path>\na\n</tool_call>",
        );
        assert_eq!(p.state(), DsmlState::Error);
        assert!(p.error().unwrap().contains("unterminated"));
    }

    #[test]
    fn param_close_prefix_tracks_a_partial_close_tag() {
        let mut p = QwenParser::new();
        feed_all(
            &mut p,
            "<tool_call>\n<function=write>\n<parameter=path>\nsrc/a",
        );
        assert!(!p.param_close_prefix(), "plain value text");
        feed_all(&mut p, "\n</para");
        assert!(p.param_close_prefix(), "partial close tag");
        feed_all(&mut p, "meter>");
        assert!(!p.param_close_prefix(), "close tag consumed");
    }

    /// An angle bracket in the value is not a close-tag prefix, or greedy
    /// sampling would latch on ordinary code.
    #[test]
    fn a_lone_angle_bracket_is_not_a_close_prefix() {
        let mut p = QwenParser::new();
        feed_all(
            &mut p,
            "<tool_call>\n<function=write>\n<parameter=c>\nif a < b",
        );
        assert!(!p.param_close_prefix());
    }

    #[test]
    fn feeding_after_a_terminal_state_is_a_no_op() {
        let mut p = QwenParser::new();
        feed_all(&mut p, C_LITERAL_MARKUP);
        p.finish();
        assert_eq!(p.state(), DsmlState::Done);
        feed_all(
            &mut p,
            "<tool_call>\n<function=rm>\n</function>\n</tool_call>",
        );
        assert_eq!(p.calls().len(), 1, "a second stanza must not sneak in");
    }

    #[test]
    fn pending_call_exposes_the_call_mid_stream() {
        let mut p = QwenParser::new();
        feed_all(
            &mut p,
            "<tool_call>\n<function=read>\n<parameter=path>\na\n</parameter>\n",
        );
        let pending = p.pending_call().expect("a call is open");
        assert_eq!(pending.name, "read");
        assert_eq!(pending.arg_value("path"), Some("a"));
        assert!(p.calls().is_empty(), "not pushed until the stanza closes");
    }

    /// The C's own route to `DONE`: content after a stanza that cannot start
    /// another call settles the question without waiting for end of stream.
    #[test]
    fn trailing_prose_after_a_call_ends_the_run() {
        let mut p = QwenParser::new();
        feed_all(
            &mut p,
            "<tool_call>\n<function=list>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>\nDone.",
        );
        assert_eq!(p.state(), DsmlState::Done, "error: {:?}", p.error());
        assert_eq!(p.calls().len(), 1);
    }

    /// A generation that stops mid-value cannot be dispatched, and the model
    /// needs to hear that rather than have a truncated argument run.
    #[test]
    fn finishing_with_an_open_parameter_errors() {
        let mut p = QwenParser::new();
        feed_all(
            &mut p,
            "<tool_call>\n<function=write>\n<parameter=path>\nsrc/a.rs",
        );
        assert_eq!(p.state(), DsmlState::ParamValue);
        p.finish();
        assert_eq!(p.state(), DsmlState::Error);
        assert!(p.error().unwrap().contains("unterminated"));
    }

    /// The regression that made a working port loop: a generation with no
    /// tool call at all must not look like an open stanza.
    #[test]
    fn a_fresh_parser_is_searching_not_mid_stanza() {
        let p = QwenParser::new();
        assert_eq!(p.state(), DsmlState::Search);
        let mut p = QwenParser::new();
        feed_all(&mut p, "Here is my answer, no tools needed.");
        assert_eq!(p.state(), DsmlState::Search);
        p.finish();
        assert_eq!(
            p.state(),
            DsmlState::Search,
            "finish must not invent a call"
        );
        assert!(p.calls().is_empty());
    }

    #[test]
    fn free_text_before_the_stanza_is_not_a_tool_call() {
        let mut p = QwenParser::new();
        feed_all(&mut p, "Let me read that file.\n");
        assert_ne!(p.state(), DsmlState::Done);
        assert!(p.calls().is_empty());
    }
}
