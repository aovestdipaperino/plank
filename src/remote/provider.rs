//! Flavor (b): [`ProviderEngine`] over third-party LLM APIs (issue #26, §4.2).
//!
//! v1 wires the **OpenAI-compatible** chat-completions API (which also covers
//! `vLLM`, `Ollama`, `OpenRouter`, `Together` and any gateway speaking that shape). The
//! Anthropic Messages API is sequenced next; the translation core here is
//! written so a second provider reuses the DSML-synthesis and structured-input
//! machinery.
//!
//! The design's "no second tool-call source" rule (§2.1) is honored: native
//! provider tool calls are **re-emitted as DSML text** into the
//! [`EngineEvent::Text`] stream, so everything downstream of `generate`
//! ([`crate::viz::StreamRenderer`] → [`crate::dsml::DsmlParser`] →
//! `dispatch_all`) is byte-identical to the local path. One tool dispatch path,
//! one renderer, regardless of backend.
//!
//! Transport is the already-vendored blocking `ureq` client (matching flavor
//! a): the `OpenAI` SSE stream arrives per chunk and the `interrupt` closure is
//! polled between frames, so the synchronous `Engine::generate` contract holds
//! with no async runtime.

use crate::engine::{
    ChatMessage, ChatRole, Engine, EngineError, EngineEvent, GenerationOptions, GenerationStats,
    PrefillProgress, Prompt, ToolSpec,
};
use crate::remote::read_sse;

/// Which provider API family a [`ProviderEngine`] speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    /// OpenAI-compatible `/chat/completions` (also `vLLM`, `Ollama`, `OpenRouter`...).
    OpenAi,
    /// Anthropic `/v1/messages` (not yet wired; reserved).
    Anthropic,
}

impl ProviderKind {
    /// Parses the `--provider` flag value.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "openai" => Some(Self::OpenAi),
            "anthropic" => Some(Self::Anthropic),
            _ => None,
        }
    }

    /// Environment variable holding the API key for this provider.
    #[must_use]
    pub fn api_key_env(self) -> &'static str {
        match self {
            Self::OpenAi => "OPENAI_API_KEY",
            Self::Anthropic => "ANTHROPIC_API_KEY",
        }
    }

    /// Default base URL when `--base-url` is not given.
    #[must_use]
    pub fn default_base_url(self) -> &'static str {
        match self {
            Self::OpenAi => "https://api.openai.com/v1",
            Self::Anthropic => "https://api.anthropic.com/v1",
        }
    }

    /// Short lowercase label (`openai` / `anthropic`).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
        }
    }
}

// ---------------------------------------------------------------------------
// DSML synthesis (the crux: native tool call -> DSML the dispatcher expects)
// ---------------------------------------------------------------------------

/// A finalized native tool call from a provider: its name and JSON arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeToolCall {
    /// Tool name as chosen by the model.
    pub name: String,
    /// The provider tool-call id (retained for tool-result pairing, §4.4).
    pub id: String,
    /// Raw JSON arguments string as streamed by the provider.
    pub arguments: String,
}

/// Synthesizes the canonical DSML `tool_calls` stanza for a batch of native
/// provider tool calls, so [`crate::dsml::DsmlParser`] produces the same
/// executable `ToolCall`s a local model would emit (design §4.2/§4.3).
///
/// Each JSON argument becomes a `<｜DSML｜parameter>`; string values carry
/// `string="true"` with the raw text, all other JSON values carry
/// `string="false"` with compact JSON text — matching the syntax the DS4 tools
/// prompt documents.
#[must_use]
pub fn synthesize_dsml(calls: &[NativeToolCall]) -> String {
    use std::fmt::Write as _;
    let mut out = String::from("<｜DSML｜tool_calls>\n");
    for call in calls {
        let _ = writeln!(out, "<｜DSML｜invoke name=\"{}\">", call.name);
        // Arguments arrive as a JSON object string; degrade to no parameters if
        // the provider emitted something unparseable rather than aborting.
        if let Ok(serde_json::Value::Object(map)) =
            serde_json::from_str::<serde_json::Value>(call.arguments.trim())
        {
            for (key, value) in &map {
                let (is_string, rendered) = match value {
                    serde_json::Value::String(s) => (true, s.clone()),
                    other => (false, other.to_string()),
                };
                let _ = writeln!(
                    out,
                    "<｜DSML｜parameter name=\"{key}\" string=\"{is_string}\">{rendered}</｜DSML｜parameter>"
                );
            }
        }
        out.push_str("</｜DSML｜invoke>\n");
    }
    out.push_str("</｜DSML｜tool_calls>\n");
    out
}

// ---------------------------------------------------------------------------
// OpenAI streaming translation (SSE payload -> EngineEvent)
// ---------------------------------------------------------------------------

/// Token usage reported by the provider on the terminal chunk.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProviderUsage {
    /// Prompt tokens consumed.
    pub input_tokens: i32,
    /// Completion tokens generated.
    pub output_tokens: i32,
}

/// Accumulator that turns an OpenAI-compatible SSE stream into the
/// [`EngineEvent`] shape the renderer expects, with native tool calls
/// re-emitted as synthesized DSML at finalization.
///
/// Feed each SSE `data:` payload with [`feed`](Self::feed); call
/// [`finish`](Self::finish) once the stream ends (either a `[DONE]` frame or
/// end of body) to flush any open thinking block and the DSML tool stanza.
#[derive(Debug, Default)]
pub struct OpenAiTranslator {
    /// Tool calls accumulated by streamed `index`.
    tool_calls: Vec<NativeToolCall>,
    /// True while a `<think>` block is open (reasoning deltas).
    thinking_open: bool,
    /// Usage from the terminal chunk, if any.
    usage: Option<ProviderUsage>,
    /// True once a `[DONE]` frame or `finish_reason` was seen.
    done: bool,
    /// True once the DSML tool stanza has been flushed by `finish`.
    flushed: bool,
}

impl OpenAiTranslator {
    /// Creates an empty translator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Usage reported so far.
    #[must_use]
    pub fn usage(&self) -> Option<ProviderUsage> {
        self.usage
    }

    /// Feeds one SSE `data:` payload, emitting any resulting events. Returns
    /// `false` when the stream is complete (`[DONE]`), so the caller can stop.
    pub fn feed(&mut self, payload: &str, on_event: &mut dyn FnMut(EngineEvent)) -> bool {
        let payload = payload.trim();
        if payload.is_empty() {
            return true;
        }
        if payload == "[DONE]" {
            self.done = true;
            return false;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
            return true;
        };
        if let Some(usage) = value.get("usage").and_then(parse_usage) {
            self.usage = Some(usage);
        }
        let Some(choice) = value.get("choices").and_then(|c| c.get(0)) else {
            return true;
        };
        if let Some(delta) = choice.get("delta") {
            self.handle_delta(delta, on_event);
        }
        if choice.get("finish_reason").is_some_and(|r| !r.is_null()) {
            self.done = true;
        }
        true
    }

    fn handle_delta(&mut self, delta: &serde_json::Value, on_event: &mut dyn FnMut(EngineEvent)) {
        // Reasoning content (deepseek/openai-compatible) is wrapped in a single
        // synthetic <think>…</think> so the renderer routes it to think_text.
        if let Some(reasoning) = delta
            .get("reasoning_content")
            .or_else(|| delta.get("reasoning"))
            .and_then(|v| v.as_str())
            && !reasoning.is_empty()
        {
            if !self.thinking_open {
                on_event(EngineEvent::Text("<think>".to_string()));
                self.thinking_open = true;
            }
            on_event(EngineEvent::Text(reasoning.to_string()));
        }
        if let Some(content) = delta.get("content").and_then(|v| v.as_str())
            && !content.is_empty()
        {
            if self.thinking_open {
                on_event(EngineEvent::Text("</think>".to_string()));
                self.thinking_open = false;
            }
            on_event(EngineEvent::Text(content.to_string()));
        }
        if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tool_calls {
                self.accumulate_tool_call(tc);
            }
        }
    }

    fn accumulate_tool_call(&mut self, tc: &serde_json::Value) {
        let index = usize::try_from(
            tc.get("index")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        )
        .unwrap_or(0);
        while self.tool_calls.len() <= index {
            self.tool_calls.push(NativeToolCall {
                name: String::new(),
                id: String::new(),
                arguments: String::new(),
            });
        }
        let slot = &mut self.tool_calls[index];
        if let Some(id) = tc.get("id").and_then(|v| v.as_str())
            && !id.is_empty()
        {
            slot.id = id.to_string();
        }
        if let Some(func) = tc.get("function") {
            if let Some(name) = func.get("name").and_then(|v| v.as_str())
                && !name.is_empty()
            {
                slot.name.push_str(name);
            }
            if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                slot.arguments.push_str(args);
            }
        }
    }

    /// Flushes an open thinking block and the synthesized DSML tool stanza.
    /// Idempotent: safe to call once at end of stream.
    pub fn finish(&mut self, on_event: &mut dyn FnMut(EngineEvent)) {
        if self.flushed {
            return;
        }
        self.flushed = true;
        if self.thinking_open {
            on_event(EngineEvent::Text("</think>".to_string()));
            self.thinking_open = false;
        }
        let calls: Vec<NativeToolCall> = self
            .tool_calls
            .iter()
            .filter(|c| !c.name.is_empty())
            .cloned()
            .collect();
        if !calls.is_empty() {
            on_event(EngineEvent::Text(synthesize_dsml(&calls)));
        }
    }

    /// The finalized native tool calls (names non-empty), for the id side-map.
    #[must_use]
    pub fn finalized_calls(&self) -> Vec<NativeToolCall> {
        self.tool_calls
            .iter()
            .filter(|c| !c.name.is_empty())
            .cloned()
            .collect()
    }
}

fn parse_usage(value: &serde_json::Value) -> Option<ProviderUsage> {
    if value.is_null() {
        return None;
    }
    let input = value
        .get("prompt_tokens")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0);
    let output = value
        .get("completion_tokens")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0);
    Some(ProviderUsage {
        input_tokens: i32::try_from(input).unwrap_or(i32::MAX),
        output_tokens: i32::try_from(output).unwrap_or(i32::MAX),
    })
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

/// Builds the OpenAI-compatible `/chat/completions` request body.
///
/// Pure and unit-testable: no network, no engine state.
#[must_use]
pub fn build_openai_request(
    model: &str,
    system: &str,
    messages: &[ChatMessage],
    tools: &[ToolSpec],
    opts: &GenerationOptions,
) -> serde_json::Value {
    let mut wire_messages = Vec::new();
    if !system.is_empty() {
        wire_messages.push(serde_json::json!({ "role": "system", "content": system }));
    }
    for m in messages {
        match m.role {
            ChatRole::System => {
                wire_messages.push(serde_json::json!({ "role": "system", "content": m.content }));
            }
            ChatRole::User => {
                wire_messages.push(serde_json::json!({ "role": "user", "content": m.content }));
            }
            ChatRole::Assistant => {
                wire_messages
                    .push(serde_json::json!({ "role": "assistant", "content": m.content }));
            }
            ChatRole::Tool => {
                // A tool result with a retained id uses the native `tool` role;
                // without one, degrade to a user message so any gateway accepts
                // it (design §4.4 / constraint 8).
                if let Some(id) = &m.tool_call_id {
                    wire_messages.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": id,
                        "content": m.content,
                    }));
                } else {
                    wire_messages.push(serde_json::json!({
                        "role": "user",
                        "content": format!("Tool result:\n{}", m.content),
                    }));
                }
            }
        }
    }

    let mut body = serde_json::json!({
        "model": model,
        "messages": wire_messages,
        "stream": true,
        "stream_options": { "include_usage": true },
        "temperature": opts.temperature,
        "top_p": opts.top_p,
    });
    if opts.n_predict > 0 {
        body["max_tokens"] = serde_json::json!(opts.n_predict);
    }
    if !tools.is_empty() {
        let wire_tools: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })
            })
            .collect();
        body["tools"] = serde_json::json!(wire_tools);
        body["tool_choice"] = serde_json::json!("auto");
        // plank dispatches serially and re-feeds; parallel batches would
        // complicate the single-transcript reconciliation (§4.3).
        body["parallel_tool_calls"] = serde_json::json!(false);
    }
    body
}

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

/// Third-party provider engine (flavor b). Speaks the OpenAI-compatible API in
/// v1; selectable via `--provider openai --model NAME`.
#[derive(Debug)]
pub struct ProviderEngine {
    kind: ProviderKind,
    base_url: String,
    api_key: String,
    model: String,
    ctx_size: i32,
}

impl ProviderEngine {
    /// Constructs a provider engine. `base_url` defaults per provider when
    /// empty; `api_key` must be resolved by the caller (env or flag).
    ///
    /// # Errors
    /// Returns [`EngineError`] when the provider is not yet supported or the
    /// API key is empty.
    pub fn new(
        kind: ProviderKind,
        base_url: Option<String>,
        api_key: String,
        model: String,
        ctx_size: i32,
    ) -> Result<Self, EngineError> {
        if kind != ProviderKind::OpenAi {
            return Err(EngineError::new(format!(
                "provider '{}' is not wired yet; use --provider openai",
                kind.label()
            )));
        }
        if api_key.trim().is_empty() {
            return Err(EngineError::new(format!(
                "no API key: set ${} or pass --api-key",
                kind.api_key_env()
            )));
        }
        let base_url = base_url
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| kind.default_base_url().to_string())
            .trim_end_matches('/')
            .to_string();
        Ok(Self {
            kind,
            base_url,
            api_key,
            model,
            ctx_size: if ctx_size > 0 { ctx_size } else { 128_000 },
        })
    }

    /// Builds the request for whatever `Prompt` variant arrives. A `Flat`
    /// prompt (e.g. compaction) becomes a single user message with no tools.
    fn request_for(&self, prompt: Prompt<'_>, opts: &GenerationOptions) -> serde_json::Value {
        match prompt {
            Prompt::Structured(turn) => {
                build_openai_request(&self.model, turn.system, turn.messages, turn.tools, opts)
            }
            Prompt::Flat(text) => {
                let messages = [ChatMessage::new(ChatRole::User, text)];
                build_openai_request(&self.model, "", &messages, &[], opts)
            }
        }
    }
}

impl Engine for ProviderEngine {
    fn wants_structured(&self) -> bool {
        true
    }

    fn generate(
        &mut self,
        prompt: Prompt<'_>,
        opts: &GenerationOptions,
        interrupt: &dyn Fn() -> bool,
        _greedy: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError> {
        let body = self.request_for(prompt, opts);
        let payload = serde_json::to_string(&body)
            .map_err(|e| EngineError::new(format!("serialize provider request: {e}")))?;

        // Providers report no prefill; emit one honest done-event so the
        // progress bar completes instead of hanging (§4.2).
        let total = self.count_tokens(prompt.flat());
        on_event(EngineEvent::Prefill(PrefillProgress {
            done: total,
            total,
            tps: 0.0,
        }));

        let url = format!("{}/chat/completions", self.base_url);
        let mut resp = ureq::post(&url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send(payload.as_str())
            .map_err(|e| EngineError::new(format!("provider request: {e}")))?;

        let mut translator = OpenAiTranslator::new();
        let mut interrupted = false;
        let reader = resp.body_mut().as_reader();
        read_sse(reader, |data| {
            if interrupt() {
                interrupted = true;
                return false;
            }
            translator.feed(data, on_event)
        })
        .map_err(|e| EngineError::new(format!("provider stream read: {e}")))?;

        if interrupted {
            drop(resp);
            return Ok(GenerationStats {
                interrupted: true,
                ..GenerationStats::default()
            });
        }

        translator.finish(on_event);
        let usage = translator.usage().unwrap_or(ProviderUsage {
            input_tokens: total,
            output_tokens: 0,
        });
        Ok(GenerationStats {
            generated: usage.output_tokens,
            tps: 0.0,
            ctx_used: usage.input_tokens + usage.output_tokens,
            interrupted: false,
        })
    }

    fn ctx_size(&self) -> i32 {
        self.ctx_size
    }

    fn model_name(&self) -> String {
        format!("{}:{}", self.kind.label(), self.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsml::{DsmlParser, DsmlState};

    fn collect_text(events: &[EngineEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                EngineEvent::Text(t) => Some(t.as_str()),
                EngineEvent::Prefill(_) => None,
            })
            .collect()
    }

    #[test]
    fn provider_text_passthrough() {
        let mut t = OpenAiTranslator::new();
        let mut events = Vec::new();
        for s in ["Hel", "lo ", "world"] {
            let frame = format!("{{\"choices\":[{{\"delta\":{{\"content\":\"{s}\"}}}}]}}");
            t.feed(&frame, &mut |e| events.push(e));
        }
        t.finish(&mut |e| events.push(e));
        assert_eq!(collect_text(&events), "Hello world");
    }

    #[test]
    fn provider_thinking_wrap() {
        let mut t = OpenAiTranslator::new();
        let mut events = Vec::new();
        t.feed(
            r#"{"choices":[{"delta":{"reasoning_content":"pondering"}}]}"#,
            &mut |e| events.push(e),
        );
        t.feed(
            r#"{"choices":[{"delta":{"content":"answer"}}]}"#,
            &mut |e| events.push(e),
        );
        t.finish(&mut |e| events.push(e));
        // Reasoning is bracketed and closed before visible content starts.
        assert_eq!(collect_text(&events), "<think>pondering</think>answer");
    }

    #[test]
    fn provider_toolcall_to_dsml() {
        // A tool call streamed in fragments across chunks (name once, arguments
        // in pieces), the OpenAI streaming shape.
        let mut t = OpenAiTranslator::new();
        let mut events = Vec::new();
        let frames = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":\"src"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"/main.rs\",\"start_line\":42}"}}]}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        ];
        for f in frames {
            t.feed(f, &mut |e| events.push(e));
        }
        t.finish(&mut |e| events.push(e));

        let dsml = collect_text(&events);
        // The synthesized stanza parses into the exact executable ToolCall.
        let mut parser = DsmlParser::new();
        parser.feed(dsml.as_bytes());
        assert_eq!(parser.state(), DsmlState::Done, "raw: {dsml}");
        let calls = parser.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arg_value("path"), Some("src/main.rs"));
        // A string arg carries string="true"; a number carries string="false".
        let path_arg = calls[0].args.iter().find(|a| a.name == "path").unwrap();
        assert!(path_arg.is_string);
        let line_arg = calls[0]
            .args
            .iter()
            .find(|a| a.name == "start_line")
            .unwrap();
        assert!(!line_arg.is_string);
        assert_eq!(line_arg.value, "42");
    }

    #[test]
    fn provider_usage_accounting() {
        let mut t = OpenAiTranslator::new();
        let mut events = Vec::new();
        t.feed(
            r#"{"choices":[{"delta":{"content":"hi"}}],"usage":{"prompt_tokens":120,"completion_tokens":8}}"#,
            &mut |e| events.push(e),
        );
        assert_eq!(
            t.usage(),
            Some(ProviderUsage {
                input_tokens: 120,
                output_tokens: 8
            })
        );
    }

    #[test]
    fn done_frame_stops_stream() {
        let mut t = OpenAiTranslator::new();
        let mut events = Vec::new();
        assert!(
            t.feed(r#"{"choices":[{"delta":{"content":"x"}}]}"#, &mut |e| {
                events.push(e);
            })
        );
        assert!(!t.feed("[DONE]", &mut |e| events.push(e)));
    }

    #[test]
    fn request_includes_tools_and_system() {
        let tools = vec![ToolSpec {
            name: "read".to_string(),
            description: "Read a file".to_string(),
            parameters: serde_json::json!({"type":"object","properties":{"path":{"type":"string"}}}),
        }];
        let messages = vec![ChatMessage::new(ChatRole::User, "hello")];
        let body = build_openai_request(
            "gpt-x",
            "You are helpful",
            &messages,
            &tools,
            &GenerationOptions::default(),
        );
        assert_eq!(body["model"], "gpt-x");
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["tools"][0]["function"]["name"], "read");
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["parallel_tool_calls"], false);
    }

    #[test]
    fn tool_result_pairs_by_id_or_degrades() {
        let with_id = ChatMessage {
            role: ChatRole::Tool,
            content: "output".to_string(),
            tool_call_id: Some("call_1".to_string()),
        };
        let no_id = ChatMessage {
            role: ChatRole::Tool,
            content: "output".to_string(),
            tool_call_id: None,
        };
        let body = build_openai_request(
            "m",
            "",
            &[with_id, no_id],
            &[],
            &GenerationOptions::default(),
        );
        assert_eq!(body["messages"][0]["role"], "tool");
        assert_eq!(body["messages"][0]["tool_call_id"], "call_1");
        assert_eq!(body["messages"][1]["role"], "user");
    }

    #[test]
    fn unsupported_provider_and_missing_key() {
        assert!(
            ProviderEngine::new(ProviderKind::Anthropic, None, "k".into(), "m".into(), 0).is_err()
        );
        assert!(
            ProviderEngine::new(ProviderKind::OpenAi, None, String::new(), "m".into(), 0).is_err()
        );
        let e =
            ProviderEngine::new(ProviderKind::OpenAi, None, "k".into(), "gpt".into(), 0).unwrap();
        assert_eq!(e.model_name(), "openai:gpt");
    }
}
