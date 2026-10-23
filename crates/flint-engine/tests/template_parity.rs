//! The chat format against the GGUF's own template, as rendered by
//! `tools/flint_fixtures.py` (Hugging Face transformers' jinja settings).

use flint_engine::template::{self, Kind, Message, Piece, Role, ToolCall, render_messages};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/flint_template.json")).unwrap()
}

fn message(v: &Value) -> Message {
    let role = match v["role"].as_str().unwrap() {
        "system" => Role::System,
        "developer" => Role::Developer,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        r => panic!("unknown role {r}"),
    };
    let tool_calls = v["tool_calls"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| ToolCall {
            name: c["name"].as_str().unwrap().to_string(),
            args: c["args"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| {
                    (
                        a[0].as_str().unwrap().to_string(),
                        a[1].as_str().unwrap().to_string(),
                    )
                })
                .collect(),
        })
        .collect();
    Message {
        role,
        content: v["content"].as_str().unwrap_or("").to_string(),
        reasoning: v["reasoning"].as_str().map(str::to_string),
        tool_calls,
    }
}

#[test]
fn every_fixture_case_renders_byte_for_byte() {
    let f = fixture();
    let cases = f["cases"].as_array().unwrap();
    assert!(cases.len() >= 10);
    for case in cases {
        let messages: Vec<Message> = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(message)
            .collect();
        let tools: Vec<String> = case["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap().to_string())
            .collect();
        let got = render_messages(
            &messages,
            &tools,
            case["add_generation_prompt"].as_bool().unwrap(),
        );
        assert_eq!(
            got,
            case["rendered"].as_str().unwrap(),
            "case {}",
            case["name"]
        );
    }
}

fn flat(pieces: &[Piece]) -> String {
    pieces
        .iter()
        .map(|p| match p {
            Piece::Trusted(s) | Piece::Plain(s) => s.as_str(),
        })
        .collect()
}

#[test]
fn plank_sections_compose_to_the_same_text() {
    // How plank records a tool turn: the assistant's own raw reply, then the
    // results as one `<tool_result>` user message. Rendered section by
    // section, it must equal the jinja render of the same conversation
    // without the tools preamble, which the system prompt carries.
    let f = fixture();
    let case = f["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "multi_tool_results")
        .unwrap();
    let mut text = String::new();
    text += &flat(&template::render(
        Kind::System,
        "You are plank, a coding agent.",
        usize::MAX,
    ));
    text += &flat(&template::render(
        Kind::User,
        "How many files are in this directory?",
        0,
    ));
    text += &flat(&template::render(
        Kind::Assistant,
        concat!(
            "<think>I should list the directory.</think><tool_call>bash<arg_key>command</arg_key>",
            "<arg_value>ls -1 | wc -l</arg_value><arg_key>timeout</arg_key><arg_value>30</arg_value></tool_call>",
        ),
        0,
    ));
    text += &flat(&template::render(
        Kind::ToolResult,
        "<tool_result>Tool result 1 (bash):\n7\nTool result 2 (bash):\nsecond\nresult\n</tool_result>",
        0,
    ));
    text += &flat(&template::render(Kind::Assistant, "There are 7.", 0));
    text += &flat(&template::render(Kind::User, "thanks", 0));
    text += &flat(&template::generation_prefix());
    let messages: Vec<Message> = case["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(message)
        .collect();
    assert_eq!(text, render_messages(&messages, &[], true));
}

#[test]
fn the_tools_case_composes_from_sections() {
    // The tools preamble as plank sends it: a trusted scaffold around each
    // function's JSON as plain text, between the system and user sections.
    let f = fixture();
    let case = f["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "tools")
        .unwrap();
    let tools: Vec<String> = case["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap().to_string())
        .collect();
    let pieces = template::render_tools(&tools);
    let plain: Vec<&str> = pieces
        .iter()
        .filter_map(|p| match p {
            Piece::Plain(s) => Some(s.as_str()),
            Piece::Trusted(_) => None,
        })
        .collect();
    assert_eq!(plain, tools.iter().map(String::as_str).collect::<Vec<_>>());
    let mut text = String::new();
    text += &flat(&template::render(
        Kind::System,
        "You are plank, a coding agent.",
        usize::MAX,
    ));
    text += &flat(&pieces);
    text += &flat(&template::render(
        Kind::User,
        "List the files in the current directory.",
        0,
    ));
    text += &flat(&template::generation_prefix());
    assert_eq!(text, case["rendered"].as_str().unwrap());
}
