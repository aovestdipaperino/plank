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
    /// Anthropic Messages API (`/v1/messages`).
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
    /// Prompt tokens consumed (Anthropic: the *uncached* remainder only —
    /// cache-write and cache-read tokens are reported separately below).
    pub input_tokens: i32,
    /// Completion tokens generated.
    pub output_tokens: i32,
    /// Anthropic prompt-cache tokens written this request (`message_start`
    /// `cache_creation_input_tokens`). Zero for `OpenAI`.
    pub cache_creation_input_tokens: i32,
    /// Anthropic prompt-cache tokens served from cache this request
    /// (`message_start` `cache_read_input_tokens`). Zero for `OpenAI`.
    pub cache_read_input_tokens: i32,
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
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
    })
}

// ---------------------------------------------------------------------------
// Shared streaming-translator surface
// ---------------------------------------------------------------------------

/// The SSE→[`EngineEvent`] surface shared by every provider translator, so
/// [`ProviderEngine::generate`] drives any backend through one code path.
pub trait SseTranslator {
    /// Feeds one SSE `data:` payload; returns `false` when the stream is
    /// complete so the reader can stop.
    fn feed(&mut self, payload: &str, on_event: &mut dyn FnMut(EngineEvent)) -> bool;
    /// Flushes an open thinking block and the synthesized DSML tool stanza.
    fn finish(&mut self, on_event: &mut dyn FnMut(EngineEvent));
    /// Usage reported so far, if any.
    fn usage(&self) -> Option<ProviderUsage>;
}

impl SseTranslator for OpenAiTranslator {
    fn feed(&mut self, payload: &str, on_event: &mut dyn FnMut(EngineEvent)) -> bool {
        OpenAiTranslator::feed(self, payload, on_event)
    }
    fn finish(&mut self, on_event: &mut dyn FnMut(EngineEvent)) {
        OpenAiTranslator::finish(self, on_event);
    }
    fn usage(&self) -> Option<ProviderUsage> {
        OpenAiTranslator::usage(self)
    }
}

// ---------------------------------------------------------------------------
// Anthropic streaming translation (Messages SSE -> EngineEvent)
// ---------------------------------------------------------------------------

/// Accumulator that turns an Anthropic Messages SSE stream into the
/// [`EngineEvent`] shape the renderer expects, with native `tool_use` blocks
/// re-emitted as synthesized DSML at finalization — the SAME canonical stanza
/// the `OpenAI` path emits, so `viz`/`dsml`/`dispatch` stay backend-agnostic.
///
/// Events dispatch on the JSON `type` field (`content_block_start`,
/// `content_block_delta`, `message_delta`, …), so the shared [`read_sse`]
/// reader — which forwards only `data:` payloads — suffices; `event:` lines are
/// redundant and ignored.
#[derive(Debug, Default)]
pub struct AnthropicTranslator {
    /// Tool calls accumulated, in content-block order.
    tool_calls: Vec<NativeToolCall>,
    /// Maps a streamed content-block `index` to its slot in `tool_calls`.
    block_to_call: std::collections::HashMap<u64, usize>,
    /// True while a `<think>` block is open (thinking deltas).
    thinking_open: bool,
    /// Prompt tokens from `message_start`.
    input_tokens: i32,
    /// Cumulative completion tokens from `message_delta`.
    output_tokens: i32,
    /// Prompt-cache tokens written this request (`cache_creation_input_tokens`).
    cache_creation_input_tokens: i32,
    /// Prompt-cache tokens read this request (`cache_read_input_tokens`).
    cache_read_input_tokens: i32,
    /// True once a usage figure has been seen.
    saw_usage: bool,
    /// True once the DSML tool stanza has been flushed.
    flushed: bool,
}

impl AnthropicTranslator {
    /// Creates an empty translator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Usage reported so far.
    #[must_use]
    pub fn usage(&self) -> Option<ProviderUsage> {
        self.saw_usage.then_some(ProviderUsage {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            cache_creation_input_tokens: self.cache_creation_input_tokens,
            cache_read_input_tokens: self.cache_read_input_tokens,
        })
    }

    /// Feeds one SSE `data:` payload. Returns `false` on `message_stop`.
    pub fn feed(&mut self, payload: &str, on_event: &mut dyn FnMut(EngineEvent)) -> bool {
        let payload = payload.trim();
        if payload.is_empty() {
            return true;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
            return true;
        };
        match value.get("type").and_then(|v| v.as_str()) {
            Some("message_start") => {
                if let Some(u) = value.pointer("/message/usage") {
                    self.note_usage(u);
                }
            }
            Some("content_block_start") => self.handle_block_start(&value),
            Some("content_block_delta") => self.handle_block_delta(&value, on_event),
            Some("message_delta") => {
                if let Some(u) = value.get("usage") {
                    self.note_usage(u);
                }
            }
            Some("message_stop") => return false,
            _ => {}
        }
        true
    }

    fn note_usage(&mut self, usage: &serde_json::Value) {
        if let Some(input) = usage
            .get("input_tokens")
            .and_then(serde_json::Value::as_i64)
        {
            self.input_tokens = i32::try_from(input).unwrap_or(i32::MAX);
            self.saw_usage = true;
        }
        if let Some(output) = usage
            .get("output_tokens")
            .and_then(serde_json::Value::as_i64)
        {
            self.output_tokens = i32::try_from(output).unwrap_or(i32::MAX);
            self.saw_usage = true;
        }
        // Cache tokens appear on `message_start`; keep the last non-null figure.
        if let Some(created) = usage
            .get("cache_creation_input_tokens")
            .and_then(serde_json::Value::as_i64)
        {
            self.cache_creation_input_tokens = i32::try_from(created).unwrap_or(i32::MAX);
            self.saw_usage = true;
        }
        if let Some(read) = usage
            .get("cache_read_input_tokens")
            .and_then(serde_json::Value::as_i64)
        {
            self.cache_read_input_tokens = i32::try_from(read).unwrap_or(i32::MAX);
            self.saw_usage = true;
        }
    }

    fn handle_block_start(&mut self, value: &serde_json::Value) {
        let Some(index) = value.get("index").and_then(serde_json::Value::as_u64) else {
            return;
        };
        let block = value.get("content_block");
        if block.and_then(|b| b.get("type")).and_then(|t| t.as_str()) == Some("tool_use") {
            let name = block
                .and_then(|b| b.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let id = block
                .and_then(|b| b.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            self.tool_calls.push(NativeToolCall {
                name,
                id,
                arguments: String::new(),
            });
            self.block_to_call.insert(index, self.tool_calls.len() - 1);
        }
    }

    fn handle_block_delta(
        &mut self,
        value: &serde_json::Value,
        on_event: &mut dyn FnMut(EngineEvent),
    ) {
        let Some(delta) = value.get("delta") else {
            return;
        };
        match delta.get("type").and_then(|t| t.as_str()) {
            Some("text_delta") => {
                if let Some(text) = delta.get("text").and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    if self.thinking_open {
                        on_event(EngineEvent::Text("</think>".to_string()));
                        self.thinking_open = false;
                    }
                    on_event(EngineEvent::Text(text.to_string()));
                }
            }
            Some("thinking_delta") => {
                if let Some(text) = delta.get("thinking").and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    if !self.thinking_open {
                        on_event(EngineEvent::Text("<think>".to_string()));
                        self.thinking_open = true;
                    }
                    on_event(EngineEvent::Text(text.to_string()));
                }
            }
            Some("input_json_delta") => {
                if let Some(index) = value.get("index").and_then(serde_json::Value::as_u64)
                    && let Some(&slot) = self.block_to_call.get(&index)
                    && let Some(fragment) = delta.get("partial_json").and_then(|v| v.as_str())
                {
                    self.tool_calls[slot].arguments.push_str(fragment);
                }
            }
            // signature_delta and any future delta kinds carry no visible text.
            _ => {}
        }
    }

    /// Flushes an open thinking block and the synthesized DSML tool stanza.
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
}

impl SseTranslator for AnthropicTranslator {
    fn feed(&mut self, payload: &str, on_event: &mut dyn FnMut(EngineEvent)) -> bool {
        AnthropicTranslator::feed(self, payload, on_event)
    }
    fn finish(&mut self, on_event: &mut dyn FnMut(EngineEvent)) {
        AnthropicTranslator::finish(self, on_event);
    }
    fn usage(&self) -> Option<ProviderUsage> {
        AnthropicTranslator::usage(self)
    }
}

/// Builds the Anthropic Messages API request body.
///
/// The system prompt is a top-level `system` block array (not a bare string, so
/// a `cache_control` breakpoint can attach); tool results are coalesced into a
/// single `user` turn of `tool_result` blocks paired to the assistant's
/// `tool_use` ids (§4.4). Pure and unit-testable.
///
/// # Prompt caching (`cache`)
/// When `cache` is true, `cache_control: {type: "ephemeral"}` breakpoints are
/// placed on the **largest stable prefix** — the last tool definition and the
/// (single) system block. Anthropic renders `tools` → `system` → `messages`, so
/// a breakpoint on the system block caches tools+system together, and the
/// last-tool breakpoint is a second, tools-only fallback that still hits when
/// only the system text changes. The volatile trailing `messages` are never
/// marked. This stays within Anthropic's 4-breakpoint limit (at most 2 here) and
/// makes the FIRST real request establish the cache so every later turn reads
/// it. Caching is off for a `Flat` prompt (no tools, no reused system).
#[must_use]
#[allow(clippy::too_many_lines)]
/// Rounds a sampling parameter to two decimals as a clean JSON number.
///
/// An `f32` like `0.6` widens to the noisy `f64` `0.6000000238…`, which
/// `serde_json` prints in full. Some Anthropic-compatible gateways (e.g. z.ai)
/// reject more than two decimal places, so we round and emit a tidy value.
/// Two decimals is ample precision for `temperature`/`top_p`.
/// Whether a `ureq` send error is a transient connection-setup failure worth
/// retrying (a stale pooled socket dropped by the server before any response).
/// A real HTTP status (`Error::StatusCode`) or any other class is not retried.
fn is_transient_send_error(e: &ureq::Error) -> bool {
    match e {
        ureq::Error::Io(io) => matches!(
            io.kind(),
            std::io::ErrorKind::BrokenPipe
                | std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::ConnectionAborted
                | std::io::ErrorKind::NotConnected
                | std::io::ErrorKind::UnexpectedEof
        ),
        _ => false,
    }
}

fn round2(x: f32) -> serde_json::Value {
    let v = (f64::from(x) * 100.0).round() / 100.0;
    serde_json::json!(v)
}

pub fn build_anthropic_request(
    model: &str,
    system: &str,
    messages: &[ChatMessage],
    tools: &[ToolSpec],
    opts: &GenerationOptions,
    cache: bool,
) -> serde_json::Value {
    let mut sys = system.to_string();
    let mut wire_messages = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        let m = &messages[i];
        match m.role {
            ChatRole::System => {
                if !sys.is_empty() {
                    sys.push('\n');
                }
                sys.push_str(&m.content);
                i += 1;
            }
            ChatRole::User => {
                wire_messages.push(serde_json::json!({ "role": "user", "content": m.content }));
                i += 1;
            }
            ChatRole::Assistant => {
                let mut content = Vec::new();
                if !m.content.is_empty() {
                    content.push(serde_json::json!({ "type": "text", "text": m.content }));
                }
                for tc in &m.tool_calls {
                    let input = serde_json::from_str::<serde_json::Value>(tc.arguments.trim())
                        .ok()
                        .filter(serde_json::Value::is_object)
                        .unwrap_or_else(|| serde_json::json!({}));
                    content.push(serde_json::json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": input,
                    }));
                }
                if content.is_empty() {
                    content.push(serde_json::json!({ "type": "text", "text": "" }));
                }
                wire_messages.push(serde_json::json!({ "role": "assistant", "content": content }));
                i += 1;
            }
            ChatRole::Tool => {
                // Coalesce a run of tool results into one user turn of blocks;
                // Anthropic pairs each `tool_result` to a prior `tool_use` id.
                let mut blocks = Vec::new();
                while i < messages.len() && messages[i].role == ChatRole::Tool {
                    let tm = &messages[i];
                    if let Some(id) = &tm.tool_call_id {
                        blocks.push(serde_json::json!({
                            "type": "tool_result",
                            "tool_use_id": id,
                            "content": tm.content,
                        }));
                    } else {
                        // No retained id: degrade to plain text so the request
                        // is still valid (constraint 8 / §4.4).
                        blocks.push(serde_json::json!({
                            "type": "text",
                            "text": format!("Tool result:\n{}", tm.content),
                        }));
                    }
                    i += 1;
                }
                wire_messages.push(serde_json::json!({ "role": "user", "content": blocks }));
            }
        }
    }

    let mut body = serde_json::json!({
        "model": model,
        "messages": wire_messages,
        "stream": true,
        "max_tokens": if opts.n_predict > 0 { opts.n_predict } else { 4096 },
        "temperature": round2(opts.temperature),
        "top_p": round2(opts.top_p),
    });
    if !sys.is_empty() {
        // System as a one-element block array so a cache breakpoint can attach.
        let mut sys_block = serde_json::json!({ "type": "text", "text": sys });
        if cache {
            sys_block["cache_control"] = serde_json::json!({ "type": "ephemeral" });
        }
        body["system"] = serde_json::json!([sys_block]);
    }
    if !tools.is_empty() {
        let mut wire_tools: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                })
            })
            .collect();
        // Breakpoint on the last (stable) tool: caches the whole tool prefix.
        if cache && let Some(last) = wire_tools.last_mut() {
            last["cache_control"] = serde_json::json!({ "type": "ephemeral" });
        }
        body["tools"] = serde_json::json!(wire_tools);
        // Serial dispatch mirrors the OpenAI path: disable parallel tool use.
        body["tool_choice"] =
            serde_json::json!({ "type": "auto", "disable_parallel_tool_use": true });
    }
    body
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
                // An assistant turn that issued tool calls carries them as the
                // OpenAI `tool_calls` array; the matching `tool` messages echo
                // each id (§4.4). `content` stays present (null when empty) as
                // the API requires alongside `tool_calls`.
                let mut msg = serde_json::json!({ "role": "assistant" });
                msg["content"] = if m.content.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::json!(m.content)
                };
                if !m.tool_calls.is_empty() {
                    let calls: Vec<serde_json::Value> = m
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "id": tc.id,
                                "type": "function",
                                "function": { "name": tc.name, "arguments": tc.arguments },
                            })
                        })
                        .collect();
                    msg["tool_calls"] = serde_json::json!(calls);
                }
                wire_messages.push(msg);
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
        "temperature": round2(opts.temperature),
        "top_p": round2(opts.top_p),
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
    /// Anthropic prompt caching over the stable prefix (tools + system). On by
    /// default; ignored by the `OpenAi` path (server-side prefix caching there
    /// is automatic). See [`build_anthropic_request`].
    cache: bool,
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
        cache: bool,
    ) -> Result<Self, EngineError> {
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
            cache,
        })
    }

    /// Builds the request for whatever `Prompt` variant arrives. A `Flat`
    /// prompt (e.g. compaction) becomes a single user message with no tools.
    fn request_for(&self, prompt: Prompt<'_>, opts: &GenerationOptions) -> serde_json::Value {
        match (self.kind, prompt) {
            (ProviderKind::OpenAi, Prompt::Structured(turn)) => {
                build_openai_request(&self.model, turn.system, turn.messages, turn.tools, opts)
            }
            (ProviderKind::OpenAi, Prompt::Flat(text)) => {
                let messages = [ChatMessage::new(ChatRole::User, text)];
                build_openai_request(&self.model, "", &messages, &[], opts)
            }
            (ProviderKind::Anthropic, Prompt::Structured(turn)) => build_anthropic_request(
                &self.model,
                turn.system,
                turn.messages,
                turn.tools,
                opts,
                self.cache,
            ),
            (ProviderKind::Anthropic, Prompt::Flat(text)) => {
                // A flat prompt has no reusable prefix — no caching.
                let messages = [ChatMessage::new(ChatRole::User, text)];
                build_anthropic_request(&self.model, "", &messages, &[], opts, false)
            }
        }
    }

    /// The API endpoint path for this provider's streaming completion.
    fn endpoint(&self) -> &'static str {
        match self.kind {
            ProviderKind::OpenAi => "/chat/completions",
            ProviderKind::Anthropic => "/messages",
        }
    }

    /// A fresh streaming translator for this provider.
    fn translator(&self) -> Box<dyn SseTranslator> {
        match self.kind {
            ProviderKind::OpenAi => Box::new(OpenAiTranslator::new()),
            ProviderKind::Anthropic => Box::new(AnthropicTranslator::new()),
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

        let url = format!("{}{}", self.base_url, self.endpoint());
        // A pooled keep-alive connection can be closed by the server between
        // turns; the reuse then fails on write with a broken pipe / reset before
        // any response is read. The write never reached the server, so retrying
        // on a fresh connection is safe (and only these connection-setup errors
        // are retried — never a real HTTP status or a mid-stream failure).
        let mut resp = None;
        let mut last_err = None;
        for attempt in 0..=2 {
            let request = ureq::post(&url).header("Content-Type", "application/json");
            let request = match self.kind {
                ProviderKind::OpenAi => {
                    request.header("Authorization", format!("Bearer {}", self.api_key))
                }
                ProviderKind::Anthropic => request
                    .header("x-api-key", self.api_key.as_str())
                    .header("anthropic-version", "2023-06-01"),
            };
            match request.send(payload.as_str()) {
                Ok(r) => {
                    resp = Some(r);
                    break;
                }
                Err(e) if attempt < 2 && is_transient_send_error(&e) => {
                    last_err = Some(e);
                    std::thread::sleep(std::time::Duration::from_millis(150));
                }
                Err(e) => return Err(EngineError::new(format!("provider request: {e}"))),
            }
        }
        let Some(mut resp) = resp else {
            let msg = last_err.map_or_else(
                || "provider request: connection failed".to_string(),
                |e| format!("provider request: {e}"),
            );
            return Err(EngineError::new(msg));
        };

        let mut translator = self.translator();
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
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        });
        // Anthropic reports `input_tokens` as the *uncached* remainder, so the
        // true prompt size is input + cache-write + cache-read; fold all of them
        // into ctx_used so cached turns aren't under-counted (OpenAI leaves the
        // cache figures at 0, so this reduces to input + output there).
        let prompt_total = usage
            .input_tokens
            .saturating_add(usage.cache_creation_input_tokens)
            .saturating_add(usage.cache_read_input_tokens);
        Ok(GenerationStats {
            generated: usage.output_tokens,
            tps: 0.0,
            ctx_used: prompt_total.saturating_add(usage.output_tokens),
            interrupted: false,
        })
    }

    /// No-op for providers: there is no client-side KV to prefill (§4.5). The
    /// Anthropic path relies on **server-side** prompt caching instead — the
    /// FIRST real [`generate`](Self::generate) request already carries the
    /// `cache_control` breakpoints (see [`build_anthropic_request`]), so it
    /// establishes the cache and every subsequent turn reads it. Returning
    /// `false` (no prefill happened) is correct and matches the trait default;
    /// this override exists to document the behavior.
    fn warm_system_prompt(
        &mut self,
        _system: &str,
        _checkpoint: Option<&std::path::Path>,
        _on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<bool, EngineError> {
        Ok(false)
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
                output_tokens: 8,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
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
            tool_calls: Vec::new(),
        };
        let no_id = ChatMessage {
            role: ChatRole::Tool,
            content: "output".to_string(),
            tool_call_id: None,
            tool_calls: Vec::new(),
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
    fn missing_key_errors_both_providers() {
        assert!(
            ProviderEngine::new(
                ProviderKind::OpenAi,
                None,
                String::new(),
                "m".into(),
                0,
                true
            )
            .is_err()
        );
        assert!(
            ProviderEngine::new(
                ProviderKind::Anthropic,
                None,
                String::new(),
                "m".into(),
                0,
                true
            )
            .is_err()
        );
        let e = ProviderEngine::new(
            ProviderKind::OpenAi,
            None,
            "k".into(),
            "gpt".into(),
            0,
            true,
        )
        .unwrap();
        assert_eq!(e.model_name(), "openai:gpt");
        // Anthropic is now wired end-to-end.
        let a = ProviderEngine::new(
            ProviderKind::Anthropic,
            None,
            "k".into(),
            "claude".into(),
            0,
            true,
        )
        .unwrap();
        assert_eq!(a.model_name(), "anthropic:claude");
        assert_eq!(a.endpoint(), "/messages");
    }

    fn collect_anthropic(frames: &[&str]) -> String {
        let mut t = AnthropicTranslator::new();
        let mut events = Vec::new();
        for f in frames {
            t.feed(f, &mut |e| events.push(e));
        }
        t.finish(&mut |e| events.push(e));
        collect_text(&events)
    }

    #[test]
    fn anthropic_text_and_thinking() {
        let text = collect_anthropic(&[
            r#"{"type":"message_start","message":{"usage":{"input_tokens":50,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"pondering"}}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}"#,
            r#"{"type":"message_stop"}"#,
        ]);
        assert_eq!(text, "<think>pondering</think>answer");
    }

    #[test]
    fn anthropic_tooluse_to_dsml() {
        // A tool_use block with input_json streamed in fragments.
        let frames = [
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"read","input":{}}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"src"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"/main.rs\",\"start_line\":42}"}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let dsml = collect_anthropic(&frames);
        // The synthesized stanza parses into the exact executable ToolCall.
        let mut parser = DsmlParser::new();
        parser.feed(dsml.as_bytes());
        assert_eq!(parser.state(), DsmlState::Done, "raw: {dsml}");
        let calls = parser.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arg_value("path"), Some("src/main.rs"));
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
    fn anthropic_usage_accounting() {
        let mut t = AnthropicTranslator::new();
        t.feed(
            r#"{"type":"message_start","message":{"usage":{"input_tokens":120,"output_tokens":1}}}"#,
            &mut |_| {},
        );
        t.feed(
            r#"{"type":"message_delta","usage":{"output_tokens":8}}"#,
            &mut |_| {},
        );
        assert_eq!(
            t.usage(),
            Some(ProviderUsage {
                input_tokens: 120,
                output_tokens: 8,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            })
        );
    }

    #[test]
    fn anthropic_cache_token_usage_parses() {
        // A hand-written message_start frame carrying both cache figures, plus a
        // message_delta with the running output count — as Anthropic streams it.
        let mut t = AnthropicTranslator::new();
        t.feed(
            r#"{"type":"message_start","message":{"usage":{"input_tokens":12,"cache_creation_input_tokens":900,"cache_read_input_tokens":4096,"output_tokens":1}}}"#,
            &mut |_| {},
        );
        t.feed(
            r#"{"type":"message_delta","usage":{"output_tokens":20}}"#,
            &mut |_| {},
        );
        assert_eq!(
            t.usage(),
            Some(ProviderUsage {
                input_tokens: 12,
                output_tokens: 20,
                cache_creation_input_tokens: 900,
                cache_read_input_tokens: 4096,
            })
        );
    }

    #[test]
    fn anthropic_request_shape() {
        let tools = vec![ToolSpec {
            name: "read".to_string(),
            description: "Read a file".to_string(),
            parameters: serde_json::json!({"type":"object","properties":{"path":{"type":"string"}}}),
        }];
        let messages = vec![ChatMessage::new(ChatRole::User, "hello")];
        let body = build_anthropic_request(
            "claude-x",
            "You are helpful",
            &messages,
            &tools,
            &GenerationOptions::default(),
            true,
        );
        assert_eq!(body["model"], "claude-x");
        assert_eq!(body["stream"], true);
        // System is a top-level block array (not a bare string) so a cache
        // breakpoint can attach; the text is the first block.
        assert_eq!(body["system"][0]["type"], "text");
        assert_eq!(body["system"][0]["text"], "You are helpful");
        assert!(body["max_tokens"].as_i64().unwrap() > 0);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["tools"][0]["name"], "read");
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(body["tool_choice"]["type"], "auto");
        assert_eq!(body["tool_choice"]["disable_parallel_tool_use"], true);
    }

    #[test]
    fn anthropic_cache_control_on_stable_prefix_only() {
        let tools = vec![
            ToolSpec {
                name: "read".to_string(),
                description: "Read a file".to_string(),
                parameters: serde_json::json!({"type":"object"}),
            },
            ToolSpec {
                name: "write".to_string(),
                description: "Write a file".to_string(),
                parameters: serde_json::json!({"type":"object"}),
            },
        ];
        // A volatile trailing user turn — it must NOT be marked.
        let messages = vec![ChatMessage::new(ChatRole::User, "do the thing")];
        let body = build_anthropic_request(
            "claude-x",
            "You are helpful",
            &messages,
            &tools,
            &GenerationOptions::default(),
            true,
        );
        let eph = serde_json::json!({ "type": "ephemeral" });
        // End of system prompt (caches tools + system).
        assert_eq!(body["system"][0]["cache_control"], eph);
        // Last tool definition (tools-only fallback breakpoint); earlier tools
        // are unmarked.
        assert!(body["tools"][0]["cache_control"].is_null());
        assert_eq!(body["tools"][1]["cache_control"], eph);
        // At most 2 breakpoints, within Anthropic's limit of 4.
        let count = serde_json::to_string(&body)
            .unwrap()
            .matches("cache_control")
            .count();
        assert_eq!(count, 2);
        // Volatile trailing message carries no breakpoint.
        assert!(body["messages"][0]["cache_control"].is_null());
    }

    #[test]
    fn anthropic_cache_off_omits_control() {
        let tools = vec![ToolSpec {
            name: "read".to_string(),
            description: "Read a file".to_string(),
            parameters: serde_json::json!({"type":"object"}),
        }];
        let messages = vec![ChatMessage::new(ChatRole::User, "hi")];
        let body = build_anthropic_request(
            "claude-x",
            "You are helpful",
            &messages,
            &tools,
            &GenerationOptions::default(),
            false,
        );
        // System is still a block array (needed regardless), but no breakpoints.
        assert_eq!(body["system"][0]["text"], "You are helpful");
        assert!(
            !serde_json::to_string(&body)
                .unwrap()
                .contains("cache_control")
        );
    }

    #[test]
    fn anthropic_threads_tool_use_and_result_ids() {
        // A prior assistant turn issued a tool call with id "call_0_0"; its
        // result echoes that id. Both wire shapes must carry the same id.
        let messages = vec![
            ChatMessage::new(ChatRole::User, "read the file"),
            ChatMessage {
                role: ChatRole::Assistant,
                content: "sure".to_string(),
                tool_call_id: None,
                tool_calls: vec![crate::engine::ToolCallRef {
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
        let body = build_anthropic_request(
            "claude-x",
            "",
            &messages,
            &[],
            &GenerationOptions::default(),
            true,
        );
        let assistant = &body["messages"][1];
        assert_eq!(assistant["role"], "assistant");
        let tool_use = &assistant["content"][1];
        assert_eq!(tool_use["type"], "tool_use");
        assert_eq!(tool_use["id"], "call_0_0");
        assert_eq!(tool_use["name"], "read");
        assert_eq!(tool_use["input"]["path"], "a.rs");
        let result = &body["messages"][2];
        assert_eq!(result["role"], "user");
        assert_eq!(result["content"][0]["type"], "tool_result");
        assert_eq!(result["content"][0]["tool_use_id"], "call_0_0");

        // The OpenAI shape threads the same id.
        let oa = build_openai_request("gpt-x", "", &messages, &[], &GenerationOptions::default());
        assert_eq!(oa["messages"][1]["tool_calls"][0]["id"], "call_0_0");
        assert_eq!(oa["messages"][2]["role"], "tool");
        assert_eq!(oa["messages"][2]["tool_call_id"], "call_0_0");
    }
}
