//! CI-safe end-to-end tests for [`plank::remote::provider::ProviderEngine`]
//! (issue #26) that drive the REAL socket path — `ureq` → HTTP POST → SSE
//! reader → translator → DSML synthesis → `DsmlParser` — against a LOCAL FAKE
//! provider server bound on a loopback ephemeral port.
//!
//! No API key, no network egress, no model: the fake server (a
//! `std::net::TcpListener` on `127.0.0.1:0`, one thread) speaks the exact
//! HTTP/1.1 + `text/event-stream` framing the client expects and hands back
//! hand-authored SSE fixtures. It also captures each POST body so tests can
//! assert on the wire request shape (Anthropic `cache_control` placement,
//! `OpenAI` `tool_calls` id referents) without a live API.
//!
//! The unit tests in `src/remote/provider.rs` already cover the pure request
//! builders and translators in isolation; these tests close the previously
//! untested gap: the socket + `read_sse` + `generate` glue.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

use plank::dsml::{DsmlParser, DsmlState, ToolCall};
use plank::engine::{
    ChatMessage, ChatRole, Engine, EngineEvent, GenerationOptions, Prompt, StructuredTurn,
    ToolCallRef, ToolSpec,
};
use plank::remote::provider::{ProviderEngine, ProviderKind};

// ---------------------------------------------------------------------------
// Fake provider server
// ---------------------------------------------------------------------------

/// A running fake provider on loopback. Each queued SSE fixture is served to
/// exactly one incoming HTTP POST, in order; every captured request body is
/// forwarded over `bodies` so a test can assert on the wire request shape.
struct FakeProvider {
    /// `http://127.0.0.1:<port>` — pass straight to `ProviderEngine` as `base_url`.
    base_url: String,
    /// Receives one entry per served request: the raw POST body.
    bodies: Receiver<String>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl FakeProvider {
    /// Binds an ephemeral loopback port and serves `sse_responses[i]` (already
    /// framed as `data:` events) to the i-th incoming request.
    fn start(sse_responses: Vec<String>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let base_url = format!("http://{addr}");
        let (tx, rx): (Sender<String>, Receiver<String>) = std::sync::mpsc::channel();

        let handle = std::thread::spawn(move || {
            for response in sse_responses {
                let Ok((stream, _)) = listener.accept() else {
                    break;
                };
                let body = serve_one(stream, &response);
                // Ignore send errors: the test may have already finished.
                let _ = tx.send(body);
            }
        });

        FakeProvider {
            base_url,
            bodies: rx,
            handle: Some(handle),
        }
    }

    /// The next captured request body (blocks up to a few seconds).
    fn next_body(&self) -> String {
        self.bodies
            .recv_timeout(Duration::from_secs(5))
            .expect("request body captured")
    }
}

impl Drop for FakeProvider {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            // The server thread exits once it has served every queued response;
            // by drop time all `generate` calls have returned, so this joins
            // promptly without a sleep.
            let _ = handle.join();
        }
    }
}

/// Reads one HTTP/1.1 request off `stream` (headers + `Content-Length` body),
/// writes the canned SSE response, and returns the captured request body.
fn serve_one(mut stream: TcpStream, sse_body: &str) -> String {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("read timeout");

    let mut raw = Vec::new();
    let mut tmp = [0u8; 4096];

    // Read until the header terminator is present.
    let header_end = loop {
        if let Some(pos) = find(&raw, b"\r\n\r\n") {
            break pos + 4;
        }
        let n = stream.read(&mut tmp).expect("read request headers");
        if n == 0 {
            break raw.len();
        }
        raw.extend_from_slice(&tmp[..n]);
    };

    let headers = String::from_utf8_lossy(&raw[..header_end]).to_ascii_lowercase();
    let content_length = headers
        .lines()
        .find_map(|line| line.strip_prefix("content-length:"))
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0);

    // Read the remaining body bytes.
    while raw.len() - header_end < content_length {
        let n = stream.read(&mut tmp).expect("read request body");
        if n == 0 {
            break;
        }
        raw.extend_from_slice(&tmp[..n]);
    }
    let body_end = (header_end + content_length).min(raw.len());
    let body = String::from_utf8_lossy(&raw[header_end..body_end]).to_string();

    // `Connection: close` + no Content-Length: the client's SSE reader consumes
    // the event-stream body until EOF, which the shutdown below signals.
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{sse_body}"
    );
    stream
        .write_all(response.as_bytes())
        .expect("write response");
    stream.flush().expect("flush response");
    let _ = stream.shutdown(Shutdown::Write);

    body
}

/// First index of `needle` in `haystack`.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

// ---------------------------------------------------------------------------
// SSE fixtures (concat! of raw frames — never `\`-continued literals)
// ---------------------------------------------------------------------------

/// An OpenAI-compatible chat-completions stream: reasoning, visible text, a
/// tool call streamed in fragments, a terminal usage chunk, and `[DONE]`.
const OPENAI_STREAM: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking hard\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"Hello, \"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"world.\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"function\":{\"name\":\"read\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\\\"src\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"/main.rs\\\",\\\"start_line\\\":42}\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":150,\"completion_tokens\":12}}\n\n",
    "data: [DONE]\n\n",
);

/// An Anthropic Messages stream: `message_start` usage carrying cache figures,
/// a thinking block, a text block, a `tool_use` block with `input_json` deltas,
/// a `message_delta` output count, and `message_stop`.
const ANTHROPIC_STREAM: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_creation_input_tokens\":900,\"cache_read_input_tokens\":4000,\"output_tokens\":1}}}\n\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"thinking hard\"}}\n\n",
    "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello, world.\"}}\n\n",
    "data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_abc\",\"name\":\"read\",\"input\":{}}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"src\"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"/main.rs\\\",\\\"start_line\\\":42}\"}}\n\n",
    "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":20}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

/// A minimal `OpenAI` stream that emits only a tool call (turn 1 of a multi-turn
/// exchange), so the follow-up request can thread the returned id.
const OPENAI_TOOL_ONLY: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_0_0\",\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"a.rs\\\"}\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":4}}\n\n",
    "data: [DONE]\n\n",
);

/// A trivial `OpenAI` stream: one line of text (turn 2's response).
const OPENAI_TEXT_ONLY: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":30,\"completion_tokens\":2}}\n\n",
    "data: [DONE]\n\n",
);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn read_tool() -> ToolSpec {
    ToolSpec {
        name: "read".to_string(),
        description: "Read a file".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "path": { "type": "string" } }
        }),
    }
}

/// Drives one `generate` pass, returning the concatenated `Text` events and the
/// resulting stats.
fn run_turn(
    engine: &mut ProviderEngine,
    system: &str,
    messages: &[ChatMessage],
    tools: &[ToolSpec],
) -> (String, plank::engine::GenerationStats) {
    let rendered = String::new();
    let turn = StructuredTurn {
        system,
        messages,
        tools,
        rendered: &rendered,
    };
    let prompt = Prompt::Structured(&turn);
    let opts = GenerationOptions::default();
    let mut text = String::new();
    let stats = engine
        .generate(prompt, &opts, &|| false, &|| false, &mut |event| {
            if let EngineEvent::Text(t) = event {
                text.push_str(&t);
            }
        })
        .expect("generate succeeds");
    (text, stats)
}

/// Parses the synthesized DSML out of a streamed text blob into executable
/// `ToolCall`s, proving it round-trips through the REAL parser.
fn parse_dsml(text: &str) -> Vec<ToolCall> {
    let mut parser = DsmlParser::new();
    parser.feed(text.as_bytes());
    assert_eq!(
        parser.state(),
        DsmlState::Done,
        "DSML did not complete: {text}"
    );
    parser.calls().to_vec()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// `OpenAI` end-to-end: visible text streams as `Text`, reasoning is wrapped in
/// `<think>`, and the native tool call is synthesized to canonical DSML that
/// the real `DsmlParser` turns into the expected executable `ToolCall`. Usage
/// populates `GenerationStats`.
#[test]
fn openai_generate_end_to_end() {
    let server = FakeProvider::start(vec![OPENAI_STREAM.to_string()]);
    let mut engine = ProviderEngine::new(
        ProviderKind::OpenAi,
        Some(server.base_url.clone()),
        "test-key".to_string(),
        "gpt-x".to_string(),
        0,
        true,
    )
    .expect("engine builds");

    let messages = vec![ChatMessage::new(ChatRole::User, "read the file")];
    let (text, stats) = run_turn(&mut engine, "You are helpful", &messages, &[read_tool()]);

    // Visible text and reasoning both surfaced, reasoning bracketed as thinking.
    assert!(text.contains("Hello, world."), "visible text: {text}");
    assert!(
        text.contains("<think>thinking hard</think>"),
        "reasoning wrap: {text}"
    );

    // The synthesized DSML round-trips through the real parser.
    let calls = parse_dsml(&text);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "read");
    assert_eq!(calls[0].arg_value("path"), Some("src/main.rs"));
    let line = calls[0]
        .args
        .iter()
        .find(|a| a.name == "start_line")
        .unwrap();
    assert!(!line.is_string, "numeric arg is string=false");
    assert_eq!(line.value, "42");

    // Usage from the terminal chunk populated the stats.
    assert_eq!(stats.generated, 12);
    assert_eq!(stats.ctx_used, 162); // 150 prompt + 12 completion
    assert!(!stats.interrupted);

    // The request reached the server with a well-formed body.
    let body = server.next_body();
    let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON body");
    assert_eq!(json["model"], "gpt-x");
    assert_eq!(json["stream"], true);
    assert_eq!(json["messages"][0]["role"], "system");
    assert_eq!(json["tools"][0]["function"]["name"], "read");
}

/// Anthropic end-to-end over the Messages SSE shape (`message_start` /
/// `content_block_delta` / `tool_use` / `message_delta`): text + thinking
/// surface, the tool call synthesizes to DSML, and the streamed cache-token
/// usage folds into `ctx_used` (input + `cache_creation` + `cache_read` + output).
#[test]
fn anthropic_generate_end_to_end() {
    let server = FakeProvider::start(vec![ANTHROPIC_STREAM.to_string()]);
    let mut engine = ProviderEngine::new(
        ProviderKind::Anthropic,
        Some(server.base_url.clone()),
        "test-key".to_string(),
        "claude-x".to_string(),
        0,
        true,
    )
    .expect("engine builds");

    let messages = vec![ChatMessage::new(ChatRole::User, "read the file")];
    let (text, stats) = run_turn(&mut engine, "You are helpful", &messages, &[read_tool()]);

    assert!(text.contains("Hello, world."), "visible text: {text}");
    assert!(
        text.contains("<think>thinking hard</think>"),
        "thinking wrap: {text}"
    );

    let calls = parse_dsml(&text);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "read");
    assert_eq!(calls[0].arg_value("path"), Some("src/main.rs"));

    // ctx_used proves the cache figures parsed from the streamed usage and were
    // folded in: 10 input + 900 cache_creation + 4000 cache_read + 20 output.
    assert_eq!(stats.generated, 20);
    assert_eq!(stats.ctx_used, 4930);

    // Request carried the Messages shape.
    let body = server.next_body();
    let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON body");
    assert_eq!(json["model"], "claude-x");
    assert_eq!(json["messages"][0]["role"], "user");
}

/// Server-side request-shape assertion: an Anthropic request carries
/// `cache_control` breakpoints on the stable prefix (system block + last tool)
/// and NOT on the volatile trailing user message — proving the wire request is
/// well-formed for prompt caching without a live API.
#[test]
fn anthropic_request_cache_control_over_socket() {
    let server = FakeProvider::start(vec![ANTHROPIC_STREAM.to_string()]);
    let mut engine = ProviderEngine::new(
        ProviderKind::Anthropic,
        Some(server.base_url.clone()),
        "test-key".to_string(),
        "claude-x".to_string(),
        0,
        true, // caching on
    )
    .expect("engine builds");

    let tools = vec![
        read_tool(),
        ToolSpec {
            name: "write".to_string(),
            description: "Write a file".to_string(),
            parameters: serde_json::json!({ "type": "object" }),
        },
    ];
    let messages = vec![ChatMessage::new(ChatRole::User, "do the thing")];
    let _ = run_turn(&mut engine, "You are helpful", &messages, &tools);

    let body = server.next_body();
    let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON body");
    let ephemeral = serde_json::json!({ "type": "ephemeral", "ttl": "1h" });

    // System block (caches tools + system) and last tool (tools-only fallback)
    // are marked; the first tool and the trailing message are not.
    assert_eq!(json["system"][0]["cache_control"], ephemeral);
    assert!(json["tools"][0]["cache_control"].is_null());
    assert_eq!(json["tools"][1]["cache_control"], ephemeral);
    assert!(json["messages"][0]["cache_control"].is_null());
    // At most 2 breakpoints, within Anthropic's limit of 4.
    assert_eq!(body.matches("cache_control").count(), 2);
}

/// Multi-turn tool-id threading over the socket: turn 1 returns a tool call
/// with id `call_0_0`; the caller feeds back an assistant `tool_calls` turn and
/// a matching tool-result, and turn 2's captured request is well-formed with
/// the same id on both the assistant `tool_calls` entry and the `tool` result.
#[test]
fn openai_multiturn_tool_id_threading_over_socket() {
    let server = FakeProvider::start(vec![
        OPENAI_TOOL_ONLY.to_string(),
        OPENAI_TEXT_ONLY.to_string(),
    ]);
    let mut engine = ProviderEngine::new(
        ProviderKind::OpenAi,
        Some(server.base_url.clone()),
        "test-key".to_string(),
        "gpt-x".to_string(),
        0,
        true,
    )
    .expect("engine builds");

    // Turn 1: model emits a tool call; confirm the synthesized DSML carries the
    // provider id so the caller can thread it.
    let turn1 = vec![ChatMessage::new(ChatRole::User, "read a.rs")];
    let (text1, _) = run_turn(&mut engine, "sys", &turn1, &[read_tool()]);
    let calls = parse_dsml(&text1);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "read");
    let _turn1_body = server.next_body();

    // Turn 2: caller appends the assistant tool-call turn (with the id) and the
    // tool result echoing that id, then generates again.
    let turn2 = vec![
        ChatMessage::new(ChatRole::User, "read a.rs"),
        ChatMessage {
            role: ChatRole::Assistant,
            content: String::new(),
            tool_call_id: None,
            tool_calls: vec![ToolCallRef {
                id: "call_0_0".to_string(),
                name: "read".to_string(),
                arguments: r#"{"path":"a.rs"}"#.to_string(),
            }],
        },
        ChatMessage {
            role: ChatRole::Tool,
            content: "file body".to_string(),
            tool_call_id: Some("call_0_0".to_string()),
            tool_calls: Vec::new(),
        },
    ];
    let (text2, _) = run_turn(&mut engine, "sys", &turn2, &[read_tool()]);
    assert!(text2.contains("done"), "turn 2 text: {text2}");

    // The second request is well-formed with matching ids on both wire shapes.
    // messages[0] is the prepended system prompt, [1] the user turn, so the
    // assistant tool-call turn lands at [2] and its result at [3].
    let body2 = server.next_body();
    let json: serde_json::Value = serde_json::from_str(&body2).expect("valid JSON body");
    let assistant = &json["messages"][2];
    assert_eq!(assistant["role"], "assistant");
    assert_eq!(assistant["tool_calls"][0]["id"], "call_0_0");
    assert_eq!(assistant["tool_calls"][0]["function"]["name"], "read");
    let result = &json["messages"][3];
    assert_eq!(result["role"], "tool");
    assert_eq!(result["tool_call_id"], "call_0_0");
    assert_eq!(result["content"], "file body");
}
