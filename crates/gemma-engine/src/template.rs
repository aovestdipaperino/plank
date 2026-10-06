//! Gemma 4's chat format, as pieces of trusted control text and plain content.
//!
//! Pure: no tokenizer, no tensors. Rendering is per section and depends only
//! on the previous section's kind, so a section's tokens never change once
//! the next one arrives — which is what lets the engine keep a recorded span's
//! tokens verbatim. The one context-dependent byte, the `<turn|>` that closes
//! a model turn, is therefore emitted by the *following* user section.

pub const TURN_OPEN: &str = "<|turn>";
pub const TURN_CLOSE: &str = "<turn|>";
pub const THINK: &str = "<|think|>";
pub const CHANNEL_OPEN: &str = "<|channel>";
pub const CHANNEL_CLOSE: &str = "<channel|>";
pub const TOOL_DECL_OPEN: &str = "<|tool>";
pub const TOOL_DECL_CLOSE: &str = "<tool|>";
pub const CALL_OPEN: &str = "<|tool_call>";
pub const CALL_CLOSE: &str = "<tool_call|>";
pub const RESP_OPEN: &str = "<|tool_response>";
pub const RESP_CLOSE: &str = "<tool_response|>";
pub const STR: &str = "<|\"|>";

pub const CONTROL_SPELLINGS: [&str; 12] = [
    TURN_OPEN,
    TURN_CLOSE,
    THINK,
    CHANNEL_OPEN,
    CHANNEL_CLOSE,
    TOOL_DECL_OPEN,
    TOOL_DECL_CLOSE,
    CALL_OPEN,
    CALL_CLOSE,
    RESP_OPEN,
    RESP_CLOSE,
    STR,
];

/// A run of text and how it must be tokenized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece {
    /// Control spellings inside map to their atomic ids.
    Trusted(String),
    /// Never yields a control id, whatever it spells.
    Plain(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    System,
    User,
    ToolResult,
    Assistant,
}

const TOOL_RESULT_OPEN: &str = "<tool_result>";
const TOOL_RESULT_CLOSE: &str = "</tool_result>";

#[must_use]
pub fn classify(role: &str, text: &str, prev: Option<Kind>) -> Kind {
    match role {
        "system" => Kind::System,
        "assistant" => Kind::Assistant,
        _ if text.starts_with(TOOL_RESULT_OPEN)
            && matches!(prev, Some(Kind::Assistant | Kind::ToolResult)) =>
        {
            Kind::ToolResult
        }
        _ => Kind::User,
    }
}

fn push(out: &mut Vec<Piece>, p: Piece) {
    match &p {
        Piece::Trusted(s) | Piece::Plain(s) if s.is_empty() => {}
        _ => out.push(p),
    }
}

fn trusted(s: impl Into<String>) -> Piece {
    Piece::Trusted(s.into())
}

#[must_use]
pub fn render(
    kind: Kind,
    text: &str,
    prev: Option<Kind>,
    think: bool,
    trusted_len: usize,
) -> Vec<Piece> {
    let mut out = Vec::new();
    let in_model_turn = matches!(prev, Some(Kind::Assistant | Kind::ToolResult));
    match kind {
        Kind::System => {
            push(&mut out, trusted(format!("{TURN_OPEN}system\n")));
            if think {
                push(&mut out, trusted(format!("{THINK}\n")));
            }
            let split = (0..=trusted_len.min(text.len()))
                .rev()
                .find(|&i| text.is_char_boundary(i))
                .unwrap_or(0);
            push(&mut out, trusted(&text[..split]));
            push(&mut out, Piece::Plain(text[split..].to_owned()));
            push(&mut out, trusted(format!("{TURN_CLOSE}\n")));
        }
        Kind::User => {
            if in_model_turn {
                push(&mut out, trusted(format!("{TURN_CLOSE}\n")));
            }
            push(&mut out, trusted(format!("{TURN_OPEN}user\n")));
            push(&mut out, Piece::Plain(text.to_owned()));
            push(&mut out, trusted(format!("{TURN_CLOSE}\n")));
        }
        Kind::ToolResult => {
            for (name, body) in split_tool_results(text) {
                push(&mut out, trusted(format!("{RESP_OPEN}response:")));
                push(&mut out, Piece::Plain(name));
                push(&mut out, trusted(format!("{{value:{STR}")));
                push(&mut out, Piece::Plain(body));
                push(&mut out, trusted(format!("{STR}}}{RESP_CLOSE}")));
            }
        }
        Kind::Assistant => {
            if prev != Some(Kind::ToolResult) {
                push(&mut out, trusted(format!("{TURN_OPEN}model\n")));
            }
            push(&mut out, trusted(think_to_channel(text)));
        }
    }
    out
}

#[must_use]
pub fn generation_prefix(last: Option<Kind>) -> Vec<Piece> {
    match last {
        Some(Kind::ToolResult | Kind::Assistant) => Vec::new(),
        _ => vec![trusted(format!("{TURN_OPEN}model\n"))],
    }
}

#[must_use]
pub fn split_tool_results(text: &str) -> Vec<(String, String)> {
    let inner = text.strip_prefix(TOOL_RESULT_OPEN).unwrap_or(text);
    let inner = inner.strip_suffix('\n').unwrap_or(inner);
    let inner = inner.strip_suffix(TOOL_RESULT_CLOSE).unwrap_or(inner);
    let mut out: Vec<(String, String)> = Vec::new();
    let mut body = String::new();
    let mut next = 1usize;
    for line in inner.split_inclusive('\n') {
        let header = line.trim_end_matches('\n');
        let prefix = format!("Tool result {next} (");
        if let Some(rest) = header
            .strip_prefix(&prefix)
            .and_then(|r| r.strip_suffix("):"))
        {
            if let Some(last) = out.last_mut() {
                std::mem::take(&mut body)
                    .trim_end_matches('\n')
                    .clone_into(&mut last.1);
            }
            out.push((rest.to_owned(), String::new()));
            next += 1;
            continue;
        }
        body.push_str(line);
    }
    match out.last_mut() {
        Some(last) => body.trim_end_matches('\n').clone_into(&mut last.1),
        None => out.push(("tool".to_owned(), body.trim_end_matches('\n').to_owned())),
    }
    out
}

#[must_use]
pub fn think_to_channel(reply: &str) -> String {
    reply
        .replace("<think>", &format!("{CHANNEL_OPEN}thought\n"))
        .replace("</think>", CHANNEL_CLOSE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(pieces: &[Piece]) -> String {
        pieces
            .iter()
            .map(|p| match p {
                Piece::Trusted(s) | Piece::Plain(s) => s.as_str(),
            })
            .collect()
    }

    #[test]
    fn a_whole_tool_turn_renders_in_gemma_order() {
        let sections = [
            ("system", "You are plank."),
            ("user", "list files"),
            (
                "assistant",
                "<|tool_call>call:bash{command:<|\"|>ls<|\"|>}<tool_call|>",
            ),
            (
                "user",
                "<tool_result>Tool result 1 (bash):\na.rs\nb.rs\n</tool_result>",
            ),
            ("assistant", "Two files."),
            ("user", "thanks"),
        ];
        let mut prev = None;
        let mut out = String::new();
        for (role, text) in sections {
            let kind = classify(role, text, prev);
            out.push_str(&flat(&render(kind, text, prev, false, usize::MAX)));
            prev = Some(kind);
        }
        out.push_str(&flat(&generation_prefix(prev)));
        assert_eq!(
            out,
            concat!(
                "<|turn>system\nYou are plank.<turn|>\n",
                "<|turn>user\nlist files<turn|>\n",
                "<|turn>model\n<|tool_call>call:bash{command:<|\"|>ls<|\"|>}<tool_call|>",
                "<|tool_response>response:bash{value:<|\"|>a.rs\nb.rs<|\"|>}<tool_response|>",
                "Two files.",
                "<turn|>\n<|turn>user\nthanks<turn|>\n",
                "<|turn>model\n",
            )
        );
    }

    #[test]
    fn think_flag_and_trusted_split_in_the_system_turn() {
        let p = render(
            Kind::System,
            "TRUSTED<|tool>x<tool|>MCP <turn|>",
            None,
            true,
            22,
        );
        assert_eq!(
            p,
            vec![
                Piece::Trusted("<|turn>system\n".into()),
                Piece::Trusted("<|think|>\n".into()),
                Piece::Trusted("TRUSTED<|tool>x<tool|>".into()),
                Piece::Plain("MCP <turn|>".into()),
                Piece::Trusted("<turn|>\n".into()),
            ]
        );
    }

    #[test]
    fn tool_output_with_delimiter_spelling_stays_plain_piece() {
        let p = render(
            Kind::ToolResult,
            "<tool_result>Tool result 1 (read):\nx <|\"|> <turn|>\n</tool_result>",
            Some(Kind::Assistant),
            false,
            0,
        );
        assert!(
            p.contains(&Piece::Plain("x <|\"|> <turn|>".into())),
            "{p:?}"
        );
    }

    #[test]
    fn several_results_and_a_fake_header_inside_output() {
        let r = split_tool_results(concat!(
            "<tool_result>Tool result 1 (read):\nTool result 3 (bash):\nstill body\n",
            "Tool result 2 (bash):\nok\n</tool_result>"
        ));
        assert_eq!(
            r,
            vec![
                (
                    "read".to_owned(),
                    "Tool result 3 (bash):\nstill body".to_owned()
                ),
                ("bash".to_owned(), "ok".to_owned()),
            ]
        );
        assert_eq!(
            split_tool_results("<tool_result>Tool error: bad\n</tool_result>"),
            vec![("tool".to_owned(), "Tool error: bad".to_owned())]
        );
    }

    #[test]
    fn a_tool_result_without_a_preceding_assistant_is_a_user_turn() {
        assert_eq!(
            classify("user", "<tool_result>x</tool_result>", Some(Kind::User)),
            Kind::User
        );
        assert_eq!(
            classify(
                "user",
                "<tool_result>x</tool_result>",
                Some(Kind::Assistant)
            ),
            Kind::ToolResult
        );
    }

    #[test]
    fn think_tags_map_to_the_thought_channel() {
        assert_eq!(
            think_to_channel("<think>plan</think>Answer"),
            "<|channel>thought\nplan<channel|>Answer"
        );
    }
}
