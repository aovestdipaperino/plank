//! Flint's chat format, following the GGUF's own `tokenizer.chat_template`
//! as Hugging Face transformers renders it.
//!
//! ```text
//! <|system|>SYSTEM<|system|>\n# Tools ... <tools>{fn}{fn}</tools> ...\n
//! <|user|>USER<|assistant|><think>…</think>TEXT<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>
//! <|observation|><tool_response>RESULT</tool_response><|assistant|>…
//! ```
//!
//! No turn has a closing marker, so a section's text never depends on what
//! follows it: [`render`] needs only the section itself. System, developer,
//! user and tool text pass through [`escape`], the template's `flint_text`;
//! assistant text is the model's own output and is kept verbatim. The tools
//! preamble goes before the first non-system message.

use gemma_engine::template::split_tool_results;
pub use gemma_engine::template::{Kind, Piece, classify};

/// System turn marker.
pub const SYSTEM: &str = "<|system|>";
/// Developer turn marker.
pub const DEVELOPER: &str = "<|developer|>";
/// User turn marker.
pub const USER: &str = "<|user|>";
/// Assistant turn marker.
pub const ASSISTANT: &str = "<|assistant|>";
/// Tool observation turn marker.
pub const OBSERVATION: &str = "<|observation|>";
/// Opens a reasoning span.
pub const THINK_OPEN: &str = "<think>";
/// Closes a reasoning span.
pub const THINK_CLOSE: &str = "</think>";
/// Opens a tool call.
pub const TOOL_CALL_OPEN: &str = "<tool_call>";
/// Closes a tool call.
pub const TOOL_CALL_CLOSE: &str = "</tool_call>";
/// Opens an argument name.
pub const ARG_KEY_OPEN: &str = "<arg_key>";
/// Closes an argument name.
pub const ARG_KEY_CLOSE: &str = "</arg_key>";
/// Opens an argument value.
pub const ARG_VALUE_OPEN: &str = "<arg_value>";
/// Closes an argument value.
pub const ARG_VALUE_CLOSE: &str = "</arg_value>";
/// Opens a tool result body.
pub const TOOL_RESPONSE_OPEN: &str = "<tool_response>";
/// Closes a tool result body.
pub const TOOL_RESPONSE_CLOSE: &str = "</tool_response>";

/// The marker replacements of `flint_text`, in the template's order.
const ESCAPES: [(&str, &str); 7] = [
    ("<|begin_of_text|>", "&lt;|begin_of_text|&gt;"),
    ("<|end_of_text|>", "&lt;|end_of_text|&gt;"),
    ("<|system|>", "&lt;|system|&gt;"),
    ("<|developer|>", "&lt;|developer|&gt;"),
    ("<|user|>", "&lt;|user|&gt;"),
    ("<|assistant|>", "&lt;|assistant|&gt;"),
    ("<|observation|>", "&lt;|observation|&gt;"),
];

/// The template's `flint_text`: drops NUL, `<think>` and `</think>`, then
/// spells each turn marker as `&lt;|…|&gt;`, each replacement over the
/// result of the one before.
#[must_use]
pub fn escape(text: &str) -> String {
    let mut s = text
        .replace('\0', "")
        .replace(THINK_OPEN, "")
        .replace(THINK_CLOSE, "");
    for (from, to) in ESCAPES {
        s = s.replace(from, to);
    }
    s
}

/// What opens the tools block, up to and including `<tools>`.
const TOOLS_HEAD: &str = concat!(
    "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n",
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
);

/// What closes the tools block, from `</tools>` on.
const TOOLS_TAIL: &str = concat!(
    "</tools>\n\nFor each function call, output the function name and arguments within the ",
    "following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>",
    "<arg_value>{arg-value-1}</arg_value>...</tool_call>\n",
);

/// The tools block the template puts before the first non-system message.
/// `functions` are the exact JSON texts of each function (transformers'
/// `tojson`), written back to back with no separator, as the template does.
/// [`render_tools`] is the same text as pieces.
#[must_use]
pub fn tools_preamble(functions: &[String]) -> String {
    let mut s = String::from(TOOLS_HEAD);
    for f in functions {
        s.push_str(f);
    }
    s.push_str(TOOLS_TAIL);
    s
}

/// [`tools_preamble`] as template pieces: the scaffold is trusted, and each
/// function's JSON is plain text, so a schema (an MCP server's, say) that
/// spells `<tool_call>` or `<|user|>` never yields a control id. The text is
/// not escaped, since the template writes `tojson` verbatim.
#[must_use]
pub fn render_tools(functions: &[String]) -> Vec<Piece> {
    let mut out = vec![Piece::Trusted(TOOLS_HEAD.into())];
    for f in functions {
        push(&mut out, Piece::Plain(f.clone()));
    }
    out.push(Piece::Trusted(TOOLS_TAIL.into()));
    out
}

fn push(out: &mut Vec<Piece>, p: Piece) {
    match &p {
        Piece::Trusted(s) | Piece::Plain(s) if s.is_empty() => {}
        _ => out.push(p),
    }
}

/// One plank section as template pieces.
///
/// A system section trusts its first `trusted_len` bytes (rounded down to a
/// char boundary; plank's own prompt), so control spellings there become
/// control ids; the rest is plain. A tool-result section is plank's
/// `<tool_result>` message, split into one observation per result. An
/// assistant section is the model's own reply, trusted verbatim.
#[must_use]
pub fn render(kind: Kind, text: &str, trusted_len: usize) -> Vec<Piece> {
    let mut out = Vec::new();
    match kind {
        Kind::System => {
            push(&mut out, Piece::Trusted(SYSTEM.into()));
            let split = (0..=trusted_len.min(text.len()))
                .rev()
                .find(|&i| text.is_char_boundary(i))
                .unwrap_or(0);
            push(&mut out, Piece::Trusted(escape(&text[..split])));
            push(&mut out, Piece::Plain(escape(&text[split..])));
        }
        Kind::User => {
            push(&mut out, Piece::Trusted(USER.into()));
            push(&mut out, Piece::Plain(escape(text)));
        }
        Kind::ToolResult => {
            for (_, body) in split_tool_results(text) {
                push(
                    &mut out,
                    Piece::Trusted(format!("{OBSERVATION}{TOOL_RESPONSE_OPEN}")),
                );
                push(&mut out, Piece::Plain(escape(&body)));
                push(&mut out, Piece::Trusted(TOOL_RESPONSE_CLOSE.into()));
            }
        }
        Kind::Assistant => {
            push(&mut out, Piece::Trusted(ASSISTANT.into()));
            push(&mut out, Piece::Trusted(text.to_owned()));
        }
    }
    out
}

/// What opens the model's reply: `<|assistant|>`, whatever came before.
#[must_use]
pub fn generation_prefix() -> Vec<Piece> {
    vec![Piece::Trusted(ASSISTANT.into())]
}

/// A message role, as the template knows them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// `system`.
    System,
    /// `developer`.
    Developer,
    /// `user`.
    User,
    /// `assistant`.
    Assistant,
    /// `tool`.
    Tool,
}

/// One tool call in an assistant message; each value is the exact text the
/// template writes for it (a string as is, anything else as its JSON).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Function name.
    pub name: String,
    /// Argument names and their rendered values, in order.
    pub args: Vec<(String, String)>,
}

/// One message as the template sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    /// Who spoke.
    pub role: Role,
    /// The message text.
    pub content: String,
    /// `reasoning_content`.
    pub reasoning: Option<String>,
    /// Calls an assistant message makes.
    pub tool_calls: Vec<ToolCall>,
}

/// The whole template over `messages`, as jinja renders it: the reference
/// [`render`]'s sections are tested against.
#[must_use]
pub fn render_messages(
    messages: &[Message],
    tools: &[String],
    add_generation_prompt: bool,
) -> String {
    let preamble = if tools.is_empty() {
        String::new()
    } else {
        tools_preamble(tools)
    };
    let mut pending = true;
    let mut out = String::new();
    for m in messages {
        if pending && m.role != Role::System {
            out.push_str(&preamble);
            pending = false;
        }
        match m.role {
            Role::System => {
                out.push_str(SYSTEM);
                out.push_str(&escape(&m.content));
            }
            Role::Developer => {
                out.push_str(DEVELOPER);
                out.push_str(&escape(&m.content));
            }
            Role::User => {
                out.push_str(USER);
                out.push_str(&escape(&m.content));
            }
            Role::Assistant => {
                out.push_str(ASSISTANT);
                if let Some(r) = m.reasoning.as_deref().filter(|r| !r.is_empty()) {
                    out.push_str(THINK_OPEN);
                    out.push_str(r);
                    out.push_str(THINK_CLOSE);
                }
                out.push_str(&m.content);
                for call in &m.tool_calls {
                    out.push_str(TOOL_CALL_OPEN);
                    out.push_str(&call.name);
                    for (k, v) in &call.args {
                        out.push_str(ARG_KEY_OPEN);
                        out.push_str(k);
                        out.push_str(ARG_KEY_CLOSE);
                        out.push_str(ARG_VALUE_OPEN);
                        out.push_str(v);
                        out.push_str(ARG_VALUE_CLOSE);
                    }
                    out.push_str(TOOL_CALL_CLOSE);
                }
            }
            Role::Tool => {
                out.push_str(OBSERVATION);
                out.push_str(TOOL_RESPONSE_OPEN);
                out.push_str(&escape(&m.content));
                out.push_str(TOOL_RESPONSE_CLOSE);
            }
        }
    }
    if pending {
        out.push_str(&preamble);
    }
    if add_generation_prompt {
        out.push_str(ASSISTANT);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_strips_think_and_neutralises_markers_in_template_order() {
        assert_eq!(escape("a<think>b</think>c"), "abc");
        assert_eq!(escape("x\0y"), "xy");
        assert_eq!(escape("<|user|>"), "&lt;|user|&gt;");
        assert_eq!(
            escape("<|end_of_text|><|observation|>"),
            "&lt;|end_of_text|&gt;&lt;|observation|&gt;"
        );
        assert_eq!(escape("<tool_call> stays"), "<tool_call> stays");
    }

    #[test]
    fn a_user_section_is_trusted_marker_then_plain_escaped_text() {
        assert_eq!(
            render(Kind::User, "hi <|user|>", 0),
            vec![
                Piece::Trusted(USER.into()),
                Piece::Plain("hi &lt;|user|&gt;".into())
            ]
        );
    }

    #[test]
    fn the_trusted_prefix_of_a_system_section_is_split_on_a_char_boundary() {
        let p = render(Kind::System, "héllo", 2);
        assert_eq!(
            p,
            vec![
                Piece::Trusted(SYSTEM.into()),
                Piece::Trusted("h".into()),
                Piece::Plain("éllo".into()),
            ]
        );
    }

    #[test]
    fn a_function_spelling_a_control_token_stays_plain() {
        let f = r#"{"name": "x", "description": "emits <tool_call> and <|user|>"}"#.to_string();
        let p = render_tools(std::slice::from_ref(&f));
        assert_eq!(p.len(), 3);
        assert!(matches!(&p[0], Piece::Trusted(s) if s.ends_with("<tools>")));
        assert_eq!(p[1], Piece::Plain(f.clone()));
        assert!(matches!(&p[2], Piece::Trusted(s) if s.starts_with("</tools>")));
        let flat: String = p
            .iter()
            .map(|p| match p {
                Piece::Trusted(s) | Piece::Plain(s) => s.as_str(),
            })
            .collect();
        assert_eq!(flat, tools_preamble(&[f]));
    }

    #[test]
    fn tools_preamble_concatenates_function_lines() {
        let p = tools_preamble(&["{\"a\": 1}".into(), "{\"b\": 2}".into()]);
        assert!(p.contains("<tools>{\"a\": 1}{\"b\": 2}</tools>"));
        assert!(p.starts_with("<|system|>\n# Tools\n"));
        assert!(p.ends_with("</tool_call>\n"));
    }
}
