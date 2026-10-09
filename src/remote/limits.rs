// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Context window and output cap of a provider's model, asked of the server.
//!
//! There is no standard field. `OpenAI`'s own `/v1/models` carries neither
//! number, but most OpenAI-compatible servers add one, each in its own place:
//!
//! | server      | endpoint                  | window                                  | output cap                            |
//! |-------------|---------------------------|-----------------------------------------|---------------------------------------|
//! | vLLM        | `/v1/models`              | `max_model_len`                         | —                                     |
//! | llama.cpp   | `/v1/models`              | `meta.n_ctx` (per slot; `n_ctx_train` is the model's trained maximum) | — |
//! | llama.cpp   | `/props`                  | `default_generation_settings.n_ctx`     | —                                     |
//! | `OpenRouter` | `/api/v1/models`          | `top_provider.context_length`, `context_length` | `top_provider.max_completion_tokens` |
//! | LM Studio   | `/api/v1/models`          | `loaded_instances[].config.context_length` | —                                  |
//! | Ollama      | `/api/ps`                 | `models[].context_length`               | —                                     |
//! | Anthropic   | `/v1/models/{id}`         | `max_input_tokens`                      | `max_tokens`                          |
//!
//! The *running* window is preferred over the model's trained maximum wherever
//! a server reports both: llama.cpp, LM Studio and Ollama all load a model with
//! a window far smaller than it was trained for, and a gauge reading 128K on a
//! server that rejects the request at 4K is worse than no gauge. That is also
//! why LM Studio and Ollama only answer for a model that is already loaded:
//! their trained maximum (`max_context_length`, `/api/show`) overstates what a
//! fresh load will get.
//!
//! Every probe is best-effort with a short timeout. A transport failure on the
//! first request ends the probe, so an unreachable server costs one timeout,
//! not five; a 404 just moves on to the next server's endpoint.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use crate::remote::provider::ProviderKind;

/// Per-request timeout. Short on purpose: this runs at startup, and the
/// configured window is a usable fallback.
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// What a server said about a model. Either half may be unknown.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProviderLimits {
    /// Tokens the model can attend to: prompt plus output.
    pub ctx: Option<i32>,
    /// Most tokens one response may generate.
    pub max_output: Option<i32>,
}

impl ProviderLimits {
    /// Fills whichever half is still unknown from `other`.
    fn or(self, other: Self) -> Self {
        Self {
            ctx: self.ctx.or(other.ctx),
            max_output: self.max_output.or(other.max_output),
        }
    }
}

/// What the `/models` lookup said about the key and the model name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelCheck {
    /// The server listed (or described) the model, so key and name both work.
    Confirmed,
    /// Nothing could be concluded: the server was unreachable, has no models
    /// endpoint, or the key is a placeholder plank does not send to a real API.
    /// The reason is for a warning, not an error.
    Unverified(String),
    /// The server refused the key (401/403), with its own message.
    BadKey(String),
    /// The key works but the server has no such model. `available` is what it
    /// does list, empty when it only answers for one model at a time.
    UnknownModel { available: Vec<String> },
}

/// One startup probe: the model's limits and whether key and name are good.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    pub limits: ProviderLimits,
    pub check: ModelCheck,
}

impl Probe {
    fn unverified(reason: impl Into<String>) -> Self {
        Self {
            limits: ProviderLimits::default(),
            check: ModelCheck::Unverified(reason.into()),
        }
    }

    /// The startup error for a key or model the server refused, if it did.
    #[must_use]
    pub fn error(&self, kind: ProviderKind, base_url: &str, model: &str) -> Option<String> {
        match &self.check {
            ModelCheck::Confirmed | ModelCheck::Unverified(_) => None,
            ModelCheck::BadKey(msg) => Some(format!(
                "{} at {base_url} rejected the API key ({msg}); check {}",
                kind.label(),
                kind.api_key_env()
            )),
            ModelCheck::UnknownModel { available } => {
                let mut err = format!("{} at {base_url} has no model `{model}`", kind.label());
                if !available.is_empty() {
                    err.push_str("; available: ");
                    err.push_str(&model_menu(available, model));
                }
                Some(err)
            }
        }
    }
}

/// At most [`MENU_LEN`] of `available`, the ones resembling `model` first, so a
/// typo next to a 300-model list (`OpenRouter`) still shows the right names.
fn model_menu(available: &[String], model: &str) -> String {
    const MENU_LEN: usize = 20;
    let needle = model.to_ascii_lowercase();
    let stem = needle.rsplit('/').next().unwrap_or(&needle);
    let resembles = |id: &&String| {
        let id = id.to_ascii_lowercase();
        id.contains(stem) || stem.contains(id.rsplit('/').next().unwrap_or(&id))
    };
    let mut picked: Vec<&String> = available.iter().filter(resembles).collect();
    picked.extend(available.iter().filter(|id| !resembles(id)));
    let mut menu = picked
        .iter()
        .take(MENU_LEN)
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    if available.len() > MENU_LEN {
        let _ = write!(menu, " and {} more", available.len() - MENU_LEN);
    }
    menu
}

/// Asks the server behind `base_url` what `model` can hold and emit, and
/// whether it accepts the key and knows the model.
///
/// Only a refused key or an unlisted model is decisive; every other failure —
/// network error, unknown server, unexpected shape — leaves the limits unknown
/// and the check [`ModelCheck::Unverified`]: a missing number must never fail
/// startup.
#[must_use]
pub fn discover(kind: ProviderKind, base_url: &str, api_key: &str, model: &str) -> Probe {
    if model.is_empty() {
        return Probe::unverified("no model name");
    }
    let base = base_url.trim_end_matches('/');
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(PROBE_TIMEOUT))
        .http_status_as_error(false)
        .build()
        .into();
    match kind {
        ProviderKind::Anthropic => discover_anthropic(&agent, base, api_key, model),
        ProviderKind::OpenAi | ProviderKind::OpenAiResponses => {
            discover_openai_compatible(&agent, base, api_key, model)
        }
    }
}

/// [`discover`], asked at most once per provider, URL and model per process.
///
/// For sub-agent definitions, which resolve their engine at every dispatch:
/// without this, each `agent` call on an OpenAI-compatible server would pay up
/// to four probe round-trips before its first token. An unknown answer is
/// cached too — a server that did not say the first time will not the next.
#[must_use]
pub fn discover_cached(kind: ProviderKind, base_url: &str, api_key: &str, model: &str) -> Probe {
    type Key = (ProviderKind, String, String);
    static CACHE: LazyLock<Mutex<HashMap<Key, Probe>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    let key = (kind, base_url.to_string(), model.to_string());
    if let Some(hit) = CACHE.lock().ok().and_then(|c| c.get(&key).cloned()) {
        return hit;
    }
    let probe = discover(kind, base_url, api_key, model);
    if let Ok(mut c) = CACHE.lock() {
        c.insert(key, probe.clone());
    }
    probe
}

fn discover_anthropic(agent: &ureq::Agent, base: &str, api_key: &str, model: &str) -> Probe {
    // The real API rejects a placeholder key; probing it only buys a timeout.
    if is_placeholder_key(api_key) {
        return Probe::unverified("placeholder API key");
    }
    let headers = [
        ("x-api-key", api_key.to_string()),
        ("anthropic-version", "2023-06-01".to_string()),
    ];
    match get_json(agent, &format!("{base}/models/{model}"), &headers) {
        Fetch::Json(v) => Probe {
            limits: parse_anthropic_model(&v),
            check: ModelCheck::Confirmed,
        },
        Fetch::Status(401 | 403, msg) => Probe {
            limits: ProviderLimits::default(),
            check: ModelCheck::BadKey(msg),
        },
        // The endpoint answers per model, so there is no list to offer.
        Fetch::Status(404, _) => Probe {
            limits: ProviderLimits::default(),
            check: ModelCheck::UnknownModel {
                available: Vec::new(),
            },
        },
        other => Probe::unverified(other.describe()),
    }
}

fn discover_openai_compatible(
    agent: &ureq::Agent,
    base: &str,
    api_key: &str,
    model: &str,
) -> Probe {
    // No placeholder-key shortcut here: a key-less local server (vLLM,
    // llama.cpp, Ollama) is exactly the case that needs probing.
    let headers = [("Authorization", format!("Bearer {api_key}"))];
    let (mut limits, check) = match get_json(agent, &format!("{base}/models"), &headers) {
        Fetch::Json(v) => (
            parse_openai_models(&v, model),
            check_openai_models(&v, model),
        ),
        Fetch::Status(401 | 403, msg) => (ProviderLimits::default(), ModelCheck::BadKey(msg)),
        // Unreachable: every other endpoint lives on the same host.
        fetch @ Fetch::Unreachable(_) => return Probe::unverified(fetch.describe()),
        other => (
            ProviderLimits::default(),
            ModelCheck::Unverified(other.describe()),
        ),
    };
    // OpenAI itself serves none of the endpoints below, and after a refusal
    // the startup error is all that matters.
    if limits.ctx.is_some()
        || base == ProviderKind::OpenAi.default_base_url()
        || matches!(
            check,
            ModelCheck::BadKey(_) | ModelCheck::UnknownModel { .. }
        )
    {
        return Probe { limits, check };
    }
    // The server-specific endpoints sit beside `/v1`, not under it.
    let root = base.strip_suffix("/v1").unwrap_or(base);
    let probes: [(&str, ProbeParse); 3] = [
        ("/props", |v, _| parse_llamacpp_props(v)),
        ("/api/v1/models", parse_lmstudio_models),
        ("/api/ps", parse_ollama_ps),
    ];
    for (path, parse) in probes {
        match get_json(agent, &format!("{root}{path}"), &headers) {
            Fetch::Json(v) => limits = limits.or(parse(&v, model)),
            Fetch::Unreachable(_) => break,
            Fetch::Status(..) | Fetch::NotJson => {}
        }
        if limits.ctx.is_some() {
            break;
        }
    }
    Probe { limits, check }
}

/// Reads one server-specific limits endpoint for a model.
type ProbeParse = fn(&serde_json::Value, &str) -> ProviderLimits;

/// Whether an OpenAI-shaped `GET /models` list knows `model`.
///
/// Matched the way [`find_entry`] matches, lone entry included: llama.cpp
/// lists the one model it serves under its file name and answers to any name,
/// so a mismatch there is not an error. An empty or unexpected list proves
/// nothing either way.
fn check_openai_models(v: &serde_json::Value, model: &str) -> ModelCheck {
    let list = v.get("data").and_then(|d| d.as_array());
    let ids: Vec<String> = list
        .into_iter()
        .flatten()
        .filter_map(|e| e.get("id")?.as_str().map(str::to_string))
        .collect();
    if ids.is_empty() {
        return ModelCheck::Unverified("empty model list".to_string());
    }
    if find_entry(list, &["id"], model).is_some() {
        ModelCheck::Confirmed
    } else {
        ModelCheck::UnknownModel { available: ids }
    }
}

/// What one probe `GET` came back with.
enum Fetch {
    Json(serde_json::Value),
    /// A non-2xx status with the server's own error message.
    Status(u16, String),
    /// A 2xx whose body is not JSON.
    NotJson,
    /// No HTTP answer at all: the probe should stop asking this host.
    Unreachable(String),
}

impl Fetch {
    fn describe(&self) -> String {
        match self {
            Self::Json(_) => "ok".to_string(),
            Self::Status(code, msg) => format!("HTTP {code}: {msg}"),
            Self::NotJson => "the models endpoint did not answer JSON".to_string(),
            Self::Unreachable(e) => e.clone(),
        }
    }
}

fn get_json(agent: &ureq::Agent, url: &str, headers: &[(&str, String)]) -> Fetch {
    let mut req = agent.get(url);
    for (name, value) in headers {
        req = req.header(*name, value);
    }
    let mut resp = match req.call() {
        Ok(resp) => resp,
        Err(e) => return Fetch::Unreachable(e.to_string()),
    };
    let status = resp.status().as_u16();
    let body = resp.body_mut().read_to_string().unwrap_or_default();
    if (200..300).contains(&status) {
        serde_json::from_str(&body).map_or(Fetch::NotJson, Fetch::Json)
    } else {
        Fetch::Status(status, error_message(&body))
    }
}

/// The human part of an error body: `error.message` (`OpenAI`, Anthropic, vLLM,
/// llama.cpp), a bare string `error` (Ollama), else the body itself, clipped.
fn error_message(body: &str) -> String {
    let v: Option<serde_json::Value> = serde_json::from_str(body).ok();
    let msg = v.as_ref().and_then(|v| {
        v.pointer("/error/message")
            .or_else(|| v.get("error"))
            .or_else(|| v.get("message"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    });
    let msg = msg.unwrap_or_else(|| body.trim().to_string());
    match msg.char_indices().nth(200) {
        Some((i, _)) => format!("{}...", &msg[..i]),
        None => msg,
    }
}

/// A key that marks a key-less or mock endpoint rather than a real account.
pub(crate) fn is_placeholder_key(api_key: &str) -> bool {
    let k = api_key.trim();
    k.is_empty() || k == "DUMMY" || k.contains("DEADBEEF")
}

/// A positive token count at `pointer` (a JSON pointer), if there is one.
fn tokens_at(v: &serde_json::Value, pointer: &str) -> Option<i32> {
    let n = v.pointer(pointer)?.as_i64()?;
    i32::try_from(n).ok().filter(|n| *n > 0)
}

/// The first of `pointers` that holds a positive token count.
fn first_tokens(v: &serde_json::Value, pointers: &[&str]) -> Option<i32> {
    pointers.iter().find_map(|p| tokens_at(v, p))
}

/// The entry of `list` that names `model` under one of `keys`, or the only
/// entry when there is just one: llama.cpp serves one model and answers to any
/// name, so its id need not match what the user typed.
fn find_entry<'a>(
    list: Option<&'a Vec<serde_json::Value>>,
    keys: &[&str],
    model: &str,
) -> Option<&'a serde_json::Value> {
    let list = list?;
    let names = |e: &serde_json::Value| -> bool {
        keys.iter().any(|k| {
            e.get(*k)
                .and_then(serde_json::Value::as_str)
                .is_some_and(|name| {
                    // Ollama tags an untagged pull `:latest`.
                    name == model || name.strip_suffix(":latest") == Some(model)
                })
        })
    };
    list.iter()
        .find(|e| names(e))
        .or_else(|| (list.len() == 1).then(|| &list[0]))
}

/// Anthropic `GET /v1/models/{id}`: `max_input_tokens` is the window and
/// `max_tokens` the output cap — not to be confused with each other.
fn parse_anthropic_model(v: &serde_json::Value) -> ProviderLimits {
    ProviderLimits {
        ctx: tokens_at(v, "/max_input_tokens"),
        max_output: tokens_at(v, "/max_tokens"),
    }
}

/// An OpenAI-shaped `GET /models` list (`{"data": [...]}`), as served by vLLM,
/// llama.cpp and `OpenRouter`.
fn parse_openai_models(v: &serde_json::Value, model: &str) -> ProviderLimits {
    let Some(entry) = find_entry(v.get("data").and_then(|d| d.as_array()), &["id"], model) else {
        return ProviderLimits::default();
    };
    ProviderLimits {
        ctx: first_tokens(
            entry,
            &[
                "/max_model_len",               // vLLM
                "/meta/n_ctx",                  // llama.cpp, per slot
                "/top_provider/context_length", // OpenRouter, what is served
                "/context_length",              // OpenRouter, the model's
                "/meta/n_ctx_train",            // llama.cpp builds without `n_ctx`
            ],
        ),
        max_output: first_tokens(
            entry,
            &[
                "/top_provider/max_completion_tokens",
                "/max_completion_tokens",
            ],
        ),
    }
}

/// llama.cpp `GET /props`: the per-slot window of its one model.
fn parse_llamacpp_props(v: &serde_json::Value) -> ProviderLimits {
    ProviderLimits {
        ctx: tokens_at(v, "/default_generation_settings/n_ctx"),
        max_output: None,
    }
}

/// LM Studio `GET /api/v1/models`: the window of a loaded instance only.
fn parse_lmstudio_models(v: &serde_json::Value, model: &str) -> ProviderLimits {
    let ctx = find_entry(v.get("models").and_then(|m| m.as_array()), &["key"], model)
        .and_then(|e| e.get("loaded_instances")?.as_array()?.first())
        .and_then(|i| tokens_at(i, "/config/context_length"));
    ProviderLimits {
        ctx,
        max_output: None,
    }
}

/// Ollama `GET /api/ps`: the window of a running model only.
fn parse_ollama_ps(v: &serde_json::Value, model: &str) -> ProviderLimits {
    let ctx = v
        .get("models")
        .and_then(|m| m.as_array())
        .and_then(|list| {
            list.iter().find(|e| {
                ["name", "model"].iter().any(|k| {
                    e.get(*k)
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|name| {
                            name == model || name.strip_suffix(":latest") == Some(model)
                        })
                })
            })
        })
        .and_then(|e| tokens_at(e, "/context_length"));
    ProviderLimits {
        ctx,
        max_output: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(s: &str) -> serde_json::Value {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn anthropic_keeps_the_window_and_the_output_cap_apart() {
        let v = json(r#"{"id":"claude-haiku-4-5","max_input_tokens":200000,"max_tokens":64000}"#);
        assert_eq!(
            parse_anthropic_model(&v),
            ProviderLimits {
                ctx: Some(200_000),
                max_output: Some(64_000)
            }
        );
        // `max_tokens` alone is the output cap, never the window.
        let v = json(r#"{"max_tokens":64000}"#);
        assert_eq!(parse_anthropic_model(&v).ctx, None);
        assert_eq!(
            parse_anthropic_model(&json(r#"{"max_input_tokens":0}"#)).ctx,
            None
        );
    }

    #[test]
    fn vllm_reports_max_model_len_for_the_named_model() {
        let v = json(
            r#"{"object":"list","data":[
                {"id":"other","object":"model","max_model_len":8192},
                {"id":"Qwen/Qwen3-32B","object":"model","owned_by":"vllm","max_model_len":32768}
            ]}"#,
        );
        assert_eq!(parse_openai_models(&v, "Qwen/Qwen3-32B").ctx, Some(32_768));
        // Two models and neither named: no guessing.
        assert_eq!(parse_openai_models(&v, "missing").ctx, None);
    }

    #[test]
    fn llamacpp_prefers_the_slot_window_over_the_trained_one() {
        let v = json(
            r#"{"object":"list","data":[{"id":"model.gguf","object":"model",
                "meta":{"n_ctx_train":131072,"n_ctx":4096}}]}"#,
        );
        // The lone entry answers to any name.
        assert_eq!(parse_openai_models(&v, "whatever").ctx, Some(4096));
        let old = json(r#"{"data":[{"id":"m","meta":{"n_ctx_train":131072}}]}"#);
        assert_eq!(parse_openai_models(&old, "m").ctx, Some(131_072));
        let props = json(r#"{"default_generation_settings":{"n_ctx":8192}}"#);
        assert_eq!(parse_llamacpp_props(&props).ctx, Some(8192));
    }

    #[test]
    fn openrouter_reports_both_the_window_and_the_output_cap() {
        let v = json(
            r#"{"data":[{"id":"deepseek/deepseek-v4","context_length":163840,
                "top_provider":{"context_length":131072,"max_completion_tokens":32768}}]}"#,
        );
        assert_eq!(
            parse_openai_models(&v, "deepseek/deepseek-v4"),
            ProviderLimits {
                ctx: Some(131_072),
                max_output: Some(32_768)
            }
        );
    }

    #[test]
    fn openai_itself_reports_nothing() {
        let v = json(r#"{"data":[{"id":"gpt-5","object":"model","owned_by":"openai"}]}"#);
        assert_eq!(parse_openai_models(&v, "gpt-5"), ProviderLimits::default());
    }

    #[test]
    fn lmstudio_answers_only_for_a_loaded_model() {
        let v = json(
            r#"{"models":[
                {"key":"google/gemma-4-26b-a4b","max_context_length":262144,
                 "loaded_instances":[{"id":"x","config":{"context_length":4096}}]},
                {"key":"deepseek-r1","max_context_length":131072,"loaded_instances":[]}
            ]}"#,
        );
        assert_eq!(
            parse_lmstudio_models(&v, "google/gemma-4-26b-a4b").ctx,
            Some(4096)
        );
        // Not loaded: the trained maximum would overstate a fresh load.
        assert_eq!(parse_lmstudio_models(&v, "deepseek-r1").ctx, None);
    }

    #[test]
    fn ollama_matches_an_untagged_name_to_latest() {
        let v = json(
            r#"{"models":[{"name":"qwen3:latest","model":"qwen3:latest","context_length":8192}]}"#,
        );
        assert_eq!(parse_ollama_ps(&v, "qwen3").ctx, Some(8192));
        assert_eq!(parse_ollama_ps(&v, "qwen3:latest").ctx, Some(8192));
        assert_eq!(parse_ollama_ps(&v, "llama3").ctx, None);
    }

    #[test]
    fn the_first_answer_wins_each_half() {
        let a = ProviderLimits {
            ctx: Some(1),
            max_output: None,
        };
        let b = ProviderLimits {
            ctx: Some(2),
            max_output: Some(3),
        };
        assert_eq!(
            a.or(b),
            ProviderLimits {
                ctx: Some(1),
                max_output: Some(3)
            }
        );
    }

    #[test]
    fn placeholder_keys_are_recognised() {
        for key in ["", "  ", "DUMMY", "sk-DEADBEEF"] {
            assert!(is_placeholder_key(key), "{key:?} should be a placeholder");
        }
        assert!(!is_placeholder_key("sk-live-01"));
    }

    #[test]
    fn nothing_to_look_up_without_a_model_or_a_real_anthropic_key() {
        let p = discover(
            ProviderKind::Anthropic,
            "http://127.0.0.1:1",
            "DUMMY",
            "claude",
        );
        assert_eq!(p.limits, ProviderLimits::default());
        assert!(matches!(p.check, ModelCheck::Unverified(_)));
        let p = discover(ProviderKind::OpenAi, "http://127.0.0.1:1", "k", "");
        assert!(matches!(p.check, ModelCheck::Unverified(_)));
    }

    #[test]
    fn an_unreachable_server_is_unknown_not_an_error() {
        // Port 1 refuses at once; the probe must stop after the first request.
        let p = discover(ProviderKind::OpenAi, "http://127.0.0.1:1/v1", "DUMMY", "m");
        assert_eq!(p.limits, ProviderLimits::default());
        assert!(matches!(p.check, ModelCheck::Unverified(_)));
        assert_eq!(p.error(ProviderKind::OpenAi, "x", "m"), None);
    }

    #[test]
    fn a_listed_model_is_confirmed_and_an_unlisted_one_is_not() {
        let v = json(r#"{"data":[{"id":"gpt-5"},{"id":"gpt-5-mini"}]}"#);
        assert_eq!(check_openai_models(&v, "gpt-5"), ModelCheck::Confirmed);
        assert_eq!(
            check_openai_models(&v, "gpt-6"),
            ModelCheck::UnknownModel {
                available: vec!["gpt-5".into(), "gpt-5-mini".into()]
            }
        );
        // llama.cpp's lone model answers to any name.
        let v = json(r#"{"data":[{"id":"model.gguf"}]}"#);
        assert_eq!(check_openai_models(&v, "anything"), ModelCheck::Confirmed);
        // Ollama lists untagged pulls as `:latest`.
        let v = json(r#"{"data":[{"id":"qwen3:latest"},{"id":"llama3:8b"}]}"#);
        assert_eq!(check_openai_models(&v, "qwen3"), ModelCheck::Confirmed);
        // An empty list proves nothing.
        let v = json(r#"{"data":[]}"#);
        assert!(matches!(
            check_openai_models(&v, "m"),
            ModelCheck::Unverified(_)
        ));
    }

    #[test]
    fn refusals_become_startup_errors_naming_the_fix() {
        let bad_key = Probe {
            limits: ProviderLimits::default(),
            check: ModelCheck::BadKey("Incorrect API key provided".into()),
        };
        let err = bad_key
            .error(ProviderKind::OpenAi, "https://api.openai.com/v1", "gpt-5")
            .unwrap();
        assert!(err.contains("Incorrect API key provided"), "{err}");
        assert!(err.contains(ProviderKind::OpenAi.api_key_env()), "{err}");

        let unknown = Probe {
            limits: ProviderLimits::default(),
            check: ModelCheck::UnknownModel {
                available: vec!["gpt-5".into(), "gpt-5-mini".into()],
            },
        };
        let err = unknown.error(ProviderKind::OpenAi, "u", "gpt5").unwrap();
        assert!(err.contains("no model `gpt5`"), "{err}");
        assert!(err.contains("available: gpt-5, gpt-5-mini"), "{err}");
    }

    #[test]
    fn the_model_menu_puts_lookalikes_first_and_clips_long_lists() {
        let mut ids: Vec<String> = (0..30).map(|i| format!("vendor/m{i}")).collect();
        ids.push("deepseek/deepseek-v4-flash".into());
        let menu = model_menu(&ids, "deepseek-v4-flash");
        assert!(menu.starts_with("deepseek/deepseek-v4-flash, "), "{menu}");
        assert!(menu.ends_with(" and 11 more"), "{menu}");
    }

    #[test]
    fn error_bodies_are_read_in_each_servers_shape() {
        assert_eq!(
            error_message(r#"{"error":{"message":"bad key","type":"invalid_request_error"}}"#),
            "bad key"
        );
        assert_eq!(error_message(r#"{"error":"unauthorized"}"#), "unauthorized");
        assert_eq!(error_message("plain text"), "plain text");
    }
}
