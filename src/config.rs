// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Agent configuration and command-line parsing.
//!
//! Ports the "Small Utilities And Command-Line Parsing" section of the C
//! reference (`refs/ds4/ds4_agent.c`): the `agent_config` struct, option
//! parsing, numeric parsing helpers, and slash-command recognition. Unlike
//! the C code, parse failures return `Err` instead of exiting the process.
//! Distributed-mode options are not ported (plank is single-machine).

use std::path::PathBuf;

use crate::engine::{GenerationOptions, ThinkMode};

/// Default system prompt, mirroring the C agent's default.
pub const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a helpful coding assistant running inside ds4-agent.";

/// Default maximum tokens to generate.
pub const DEFAULT_N_PREDICT: i32 = 50_000;

/// Default context window size in tokens (1M = 1024 * 1024).
pub const DEFAULT_CTX_SIZE: i32 = 1_048_576;

/// Parsed agent configuration, mirroring the C `agent_config`.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // flat CLI flags, not a state machine
pub struct AgentConfig {
    /// Sampling and length options for generation.
    pub generation: GenerationOptions,
    /// One-shot prompt supplied with `-p`/`--prompt`.
    pub prompt: Option<String>,
    /// Session to resume at startup, from `plank /resume [prefix]`. `Some("")`
    /// resumes the most recent session; `Some(prefix)` a specific one.
    pub resume: Option<String>,
    /// System prompt; defaults to [`DEFAULT_SYSTEM_PROMPT`].
    pub system: String,
    /// Trace log path supplied with `--trace`.
    pub trace_path: Option<PathBuf>,
    /// Working directory supplied with `--chdir`.
    pub chdir_path: Option<PathBuf>,
    /// Worktree name supplied with `--worktree NAME`: the session starts inside
    /// an isolated checkout of the repository instead of the current one.
    pub worktree: Option<String>,
    /// Pull request supplied with `--worktree-pr N`: the worktree is based on
    /// that PR's head rather than the default branch. Implies `--worktree`.
    pub worktree_pr: Option<u32>,
    /// MCP server config supplied with `--mcp-config`; `None` = `./.mcp.json`.
    pub mcp_config_path: Option<PathBuf>,
    /// Directories named by `--plugin-dir`, loaded as session-only plugins.
    pub plugin_dirs: Vec<PathBuf>,
    /// True when `--non-interactive` was given.
    pub non_interactive: bool,
    /// True when `--minimal-prompt` was given: start with the smallest prompt
    /// this build can produce — no MCP servers, skills, templates, plugin
    /// agents, WASM components, or session-start context. For measuring
    /// throughput against a bare `llama-cli`-style prompt, where a full
    /// session's tool schemas and context dominate prefill.
    pub minimal_prompt: bool,
    /// Loopback port for `--ui-remote` TUI remote control; `Some(0)` asks for
    /// an ephemeral port. `None` (the default) leaves the feature off entirely.
    pub ui_remote: Option<u16>,
    /// True when `-h`/`--help` was given; caller should print [`usage`] and exit.
    pub show_help: bool,
    /// True when `-V`/`--version` was given; caller should print
    /// [`crate::logo::version_line`] and exit.
    pub show_version: bool,
    /// Optional help topic following `-h`/`--help`.
    pub help_topic: Option<String>,
    /// Model file supplied with `-m`/`--model`; enables the real ds4 engine.
    pub model_path: Option<PathBuf>,
    /// Backend selector from `--metal`/`--cuda`/`--cpu`; `None` = platform default.
    pub backend: Option<Backend>,
    /// Worker thread count from `-t`/`--threads`; 0 = engine default.
    pub n_threads: i32,
    /// GPU power cap percent from `--power`; 0 = unset.
    pub power_percent: i32,
    /// `--sandbox`/`--no-sandbox` override for the bash write sandbox;
    /// `None` defers to sandbox.json (default on where sandbox-exec exists).
    pub sandbox_override: Option<bool>,
    /// Native-engine tuning knobs (MTP, SSD streaming, steering, ...).
    pub engine: EngineTuning,
    /// `/btw` side-question behavior (mid-generation suspend).
    pub btw: BtwConfig,
    /// Remote plank host from `--remote URL` (flavor a, issue #26); selects
    /// [`crate::remote::ds4_client::RemoteDs4Engine`] instead of a local engine.
    pub remote_url: Option<String>,
    /// Bearer token for `--remote`, from `--remote-token` or `$PLANK_REMOTE_TOKEN`.
    pub remote_token: Option<String>,
    /// `--insecure`: allow plaintext `http://` to a non-loopback remote host.
    pub insecure: bool,
    /// `--shared-engine`: route `plank serve` (and the in-process host) through
    /// the shared reference-counted engine (issue #28) — one model, many
    /// concurrent sessions on a single GPU thread. Default off; when off, plank
    /// behaves exactly as before (single owner, no host, no scheduler thread).
    pub shared_engine: bool,
    /// `--max-sessions`: admission cap on concurrently attached sessions when
    /// `shared_engine` is on (design §7). Ignored otherwise.
    pub max_sessions: i32,
    /// `--idle-reclaim-secs`: when `shared_engine` is on, snapshot a session's
    /// live KV to disk and reclaim its context after it has been idle this many
    /// seconds, restoring transparently on the next request (design §7). `0`
    /// (default) disables reclamation entirely — a strict no-op.
    pub idle_reclaim_secs: u64,
    /// `--session-ctx-size`: default per-session context window in tokens for
    /// `--shared-engine` (design §7, v2). `0` (default) means each session gets
    /// the model's full `ctx_size`; a smaller value fits more clients. A
    /// per-request `ctx_size` from a client overrides this. Clamped to the model
    /// maximum.
    pub session_ctx_size: i32,
    /// `--kv-budget-bytes`: aggregate KV-bytes admission budget for
    /// `--shared-engine` (design §7, v2). `0` (default) keeps admission
    /// count-only; a positive value rejects an `attach` that would push the
    /// host's estimated resident KV past the budget, bounding RAM instead of
    /// OOM-ing.
    pub kv_budget_bytes: u64,
    /// Third-party provider from `--provider openai|anthropic` (flavor b, issue
    /// #26); selects [`crate::remote::provider::ProviderEngine`]. `None` unless
    /// `--provider` was given.
    pub provider: Option<ProviderSelector>,
    /// Provider model name from `--model NAME` when `--provider` is set.
    pub provider_model: Option<String>,
    /// Provider base URL override from `--base-url` (for OpenAI-compatible
    /// gateways); `None` uses the provider default.
    pub provider_base_url: Option<String>,
    /// Provider API key from `--api-key` or the provider's key env var.
    pub provider_api_key: Option<String>,
    /// Anthropic prompt caching via `cache_control` breakpoints (`--provider-cache
    /// on|off`). On by default (issue #26 flavor (b)): low-risk, saves cost and
    /// latency across multi-turn conversations by reusing the cached stable
    /// prefix (tools + system). Only consulted for `ProviderKind::Anthropic`.
    pub provider_cache: bool,
    /// Whether the context size came from the user (`-c/--ctx` or the settings
    /// file) rather than [`DEFAULT_CTX_SIZE`]. Only consulted by the provider
    /// path: an untouched default is a guess sized for the local ds4 model, so
    /// it may be replaced by the provider's reported window; an explicit value
    /// is the user's decision and is never overridden.
    pub ctx_size_explicit: bool,
    /// Settings keys (`section.key`) a CLI flag overrode, for `/config
    /// --resolved`. Populated by [`parse_options_with`]; empty when no flag
    /// shadowed a settings key.
    pub cli_provenance: std::collections::BTreeMap<String, crate::provenance::Origin>,
    /// True when `--dump-config` was given: print the resolved configuration
    /// (every effective key with its origin) and exit, without starting a
    /// session. Works under `--non-interactive`.
    pub dump_config: bool,
}

/// Third-party provider family selector (`--provider`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderSelector {
    /// OpenAI-compatible chat completions.
    OpenAi,
    /// Anthropic Messages.
    Anthropic,
}

impl ProviderSelector {
    /// Short lowercase label (`openai` / `anthropic`) for reports.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
        }
    }
}

/// `/btw` side-question configuration (BTW-SUSPEND-DESIGN §4.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BtwConfig {
    /// Answer an in-pass `/btw` by genuinely freezing the running generation,
    /// answering the aside via [`crate::engine::Engine::generate_aside`], and
    /// resuming — instead of preempt-and-rerun. **On by default**; disable with
    /// `--disable-btw-suspend`. When off (or the engine has no aside support,
    /// e.g. `EchoEngine`) an in-pass `/btw` falls back to the boundary queue.
    pub suspend: bool,
}

impl Default for BtwConfig {
    fn default() -> Self {
        Self { suspend: true }
    }
}

/// Default per-client outbound queue cap (bytes) for the remote-control
/// server's per-client backpressure eviction. Generous enough for a healthy
/// client's burst, small enough to evict a stalled one promptly.
pub const DEFAULT_CONTROL_QUEUE_MAX: usize = 1 << 20; // 1 MiB

/// Default prefill chunk size (tokens). Non-zero so a long prompt is prefilled
/// in bounded chunks: the engine checks the cancel callback at each chunk
/// boundary, so Ctrl-C during prefill is observed within ~one chunk instead of
/// only after the whole prompt is prefilled. Small enough to feel responsive,
/// large enough that the extra command-buffer overhead stays negligible.
pub const DEFAULT_PREFILL_CHUNK: u32 = 256;

/// Engine tuning options forwarded to the native ds4 engine, mirroring the
/// engine-relevant fields of the C `agent_config.engine`. Zero/`None` values
/// keep the engine defaults; the whole struct is ignored by `EchoEngine`.
// The bools deliberately mirror the C options struct one-to-one.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq)]
pub struct EngineTuning {
    /// Multi-token-prediction draft model from `--mtp`.
    pub mtp_path: Option<PathBuf>,
    /// Draft tokens per MTP step from `--mtp-draft` (C default: 1).
    pub mtp_draft_tokens: i32,
    /// MTP acceptance margin from `--mtp-margin` (C default: 3.0).
    pub mtp_margin: f32,
    /// Use a `DSpark` draft model. On by default; `--dspark-off` turns it off.
    ///
    /// The support GGUF comes from `--mtp` when given; otherwise it is
    /// resolved to `~/.plank/ds4flash.dspark.gguf` at startup and downloaded
    /// if absent (`download::ensure_dspark_support`).
    ///
    /// `DSpark` is `DeepSeek`'s auxiliary draft checkpoint for V4 Flash; it
    /// replaces the legacy one-stage MTP path. `--dspark-confidence` and
    /// `--dspark-strict` also imply it, mirroring the C CLI.
    pub dspark: bool,
    /// Load `DSpark` support but keep target-only decode, from `--dspark-strict`.
    pub dspark_strict: bool,
    /// Confidence-pruning threshold from `--dspark-confidence F`, `0..=1`.
    ///
    /// `None` leaves the engine's own default in force, which is
    /// backend-dependent (Metal 0.6, CUDA/ROCm 0.7) and has changed with
    /// tuning — so plank does not keep a copy of the number to go stale.
    /// The engine is told explicitly whether the flag was set, because `0`
    /// means "fixed five-token blocks", not "unset".
    pub dspark_confidence: Option<f32>,
    /// Prefill chunk size in tokens, fixed at [`DEFAULT_PREFILL_CHUNK`]. Chunked
    /// so Ctrl-C is observed at chunk boundaries instead of only after the whole
    /// prompt is prefilled. Not user-configurable (no CLI flag).
    pub prefill_chunk: u32,
    /// Quality mode from `--quality`.
    pub quality: bool,
    /// Touch all weights at load from `--warm-weights`.
    pub warm_weights: bool,
    /// Stream experts from SSD from `--ssd-streaming`.
    pub ssd_streaming: bool,
    /// Cold-cache SSD streaming from `--ssd-streaming-cold`.
    pub ssd_streaming_cold: bool,
    /// Expert-count cache bound from `--ssd-streaming-cache-experts N`.
    pub ssd_streaming_cache_experts: u32,
    /// Byte cache bound from `--ssd-streaming-cache-experts <N>GB`.
    pub ssd_streaming_cache_bytes: u64,
    /// Experts preloaded at startup from `--ssd-streaming-preload-experts`.
    pub ssd_streaming_preload_experts: u32,
    /// Pretend this much memory is already used, from `--simulate-used-memory`.
    pub simulate_used_memory_bytes: u64,
    /// Directional-steering vector file from `--dir-steering-file`.
    pub dir_steering_file: Option<PathBuf>,
    /// Attention steering scale from `--dir-steering-attn`.
    pub dir_steering_attn: f32,
    /// FFN steering scale from `--dir-steering-ffn`; defaults to 1.0 when a
    /// steering file is given without an explicit scale, like the C.
    pub dir_steering_ffn: f32,
}

impl Default for EngineTuning {
    fn default() -> Self {
        Self {
            mtp_path: None,
            mtp_draft_tokens: 1,
            mtp_margin: 3.0,
            dspark: true,
            dspark_strict: false,
            dspark_confidence: None,
            prefill_chunk: DEFAULT_PREFILL_CHUNK,
            quality: false,
            warm_weights: false,
            ssd_streaming: false,
            ssd_streaming_cold: false,
            ssd_streaming_cache_experts: 0,
            ssd_streaming_cache_bytes: 0,
            ssd_streaming_preload_experts: 0,
            simulate_used_memory_bytes: 0,
            dir_steering_file: None,
            dir_steering_attn: 0.0,
            dir_steering_ffn: 0.0,
        }
    }
}

/// Inference backend selector, mirroring `ds4_backend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Apple Metal GPU backend.
    Metal,
    /// NVIDIA CUDA backend.
    Cuda,
    /// CPU reference backend.
    Cpu,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            generation: GenerationOptions {
                n_predict: DEFAULT_N_PREDICT,
                ctx_size: DEFAULT_CTX_SIZE,
                think_mode: ThinkMode::Medium,
                ..GenerationOptions::default()
            },
            prompt: None,
            resume: None,
            system: DEFAULT_SYSTEM_PROMPT.to_owned(),
            trace_path: None,
            chdir_path: None,
            worktree: None,
            worktree_pr: None,
            mcp_config_path: None,
            plugin_dirs: Vec::new(),
            non_interactive: false,
            minimal_prompt: false,
            ui_remote: None,
            show_help: false,
            show_version: false,
            help_topic: None,
            model_path: None,
            backend: None,
            n_threads: 0,
            power_percent: 0,
            sandbox_override: None,
            engine: EngineTuning::default(),
            btw: BtwConfig::default(),
            remote_url: None,
            remote_token: None,
            insecure: false,
            shared_engine: false,
            max_sessions: i32::try_from(crate::host::DEFAULT_MAX_SESSIONS).unwrap_or(8),
            idle_reclaim_secs: 0,
            session_ctx_size: 0,
            kv_budget_bytes: 0,
            provider: None,
            provider_model: None,
            provider_base_url: None,
            provider_api_key: None,
            provider_cache: true,
            ctx_size_explicit: false,
            cli_provenance: std::collections::BTreeMap::new(),
            dump_config: false,
        }
    }
}

impl AgentConfig {
    /// Builds a config whose defaults come from `settings.json`.
    ///
    /// Only the keys the file is allowed to hold are consulted; an
    /// unrecognised `engine.backend` is ignored rather than rejected, since a
    /// settings file must never stop plank from starting.
    #[must_use]
    pub fn from_settings(s: &crate::settings::Settings) -> Self {
        let mut c = Self::default();
        c.model_path.clone_from(&s.engine.model);
        if let Some(t) = s.engine.threads {
            c.n_threads = t;
        }
        if let Some(b) = s.engine.backend.as_deref().and_then(parse_backend) {
            c.backend = Some(b);
        }
        if let Some(p) = s.engine.power {
            c.power_percent = p;
        }
        if let Some(ctx) = s.engine.ctx {
            c.generation.ctx_size = ctx;
            c.ctx_size_explicit = true;
        }
        // Forward recovery from an in-think tool call only pays where such a
        // call is otherwise wasted. With `thinkingToolCalls` on the stanza is
        // dispatched as-is, so there is nothing to recover from and forcing a
        // `</think>` would only cut reasoning short.
        c.generation.think_tool_recovery = !s.engine.thinking_tool_calls;
        c.sandbox_override = s.safety.sandbox;
        if let Some(b) = s.safety.btw_suspend {
            c.btw.suspend = b;
        }
        c
    }

    /// Records that a CLI flag set the settings key `key` (`section.key`), so
    /// `/config --resolved` can show the flag beating the file provenance.
    fn cli_set(&mut self, key: &str) {
        self.cli_provenance
            .insert(key.to_string(), crate::provenance::Origin::Cli);
    }
}

/// Parses a backend name; `None` for anything unrecognised.
#[must_use]
pub fn parse_backend(name: &str) -> Option<Backend> {
    match name {
        "metal" => Some(Backend::Metal),
        "cuda" => Some(Backend::Cuda),
        "cpu" => Some(Backend::Cpu),
        _ => None,
    }
}

/// Returns the usage help text, close to the C agent's `-h` output.
#[must_use]
#[allow(clippy::too_many_lines)] // one long string literal
pub fn usage() -> String {
    "\
Usage: plank [options]

Options:
  -h, --help [topic]       show this help and exit
  -V, --version            show the version and commit id, then exit
  -m, --model PATH         load a ds4 GGUF model (real inference)
  -t, --threads N          worker thread count (backend default when unset)
      --backend NAME       select backend by name: metal, cuda, cpu
      --metal              use the Metal backend
      --cuda               use the CUDA backend
      --cpu                use the CPU backend
      --power N            GPU power cap percent (1..100)
      --mtp PATH           multi-token-prediction draft model (GGUF)
      --mtp-draft N        draft tokens per MTP step (default 1)
      --mtp-margin F       MTP acceptance margin (default 3.0)
      --dspark             DSpark speculative decoding (on by default); downloads
                           the support model to ~/.plank/ds4flash.dspark.gguf unless
                           --mtp names one (defaults --temp to 0 unless --temp is given)
      --dspark-off         disable DSpark speculative decoding (target-only decode)
      --dspark-confidence F  DSpark confidence pruning threshold 0..1
                           (engine default: Metal 0.6, CUDA/ROCm 0.7; 0 = fixed blocks)
      --dspark-strict      load DSpark support but keep target-only decode
      --quality            enable quality mode
      --warm-weights       touch all weights at load
      --ssd-streaming      stream experts from SSD instead of loading resident
      --ssd-streaming-cold          assume a cold SSD cache
      --ssd-streaming-cache-experts N|<N>GB   bound the expert cache
      --ssd-streaming-preload-experts N       preload N experts at startup
      --simulate-used-memory <N>GB  pretend N GiB of memory is already used
      --dir-steering-file PATH      directional steering vectors
      --dir-steering-ffn F          FFN steering scale (-100..100)
      --dir-steering-attn F         attention steering scale (-100..100)
      --remote URL         drive a remote `plank serve` host instead of a local
                           engine (https://, or http:// to localhost); token via
                           --remote-token or $PLANK_REMOTE_TOKEN
      --remote-token TOK   bearer token for --remote
      --insecure           allow plaintext http:// to a non-loopback --remote host
      --shared-engine      host one model for many concurrent sessions (issue
                           #28); applies to `plank serve`. Default off — plank
                           otherwise runs a single owned engine as before
      --max-sessions N     admission cap for --shared-engine (default 8)
      --idle-reclaim-secs S  snapshot & reclaim a session's KV after it is idle S
                           seconds, restoring on next request (--shared-engine;
                           default 0 = off)
      --session-ctx-size N default per-session context window in tokens for
                           --shared-engine; fits more clients (default 0 = model
                           max; a client's own ctx_size request overrides)
      --kv-budget-bytes B  aggregate KV-bytes admission budget for --shared-engine;
                           reject an attach past B rather than OOM (default 0 = off,
                           count-only admission)
      --provider NAME      drive a third-party LLM API: openai (OpenAI-compatible,
                           also vLLM/Ollama/OpenRouter) or anthropic. Use with
                           --model NAME; key from --api-key or $OPENAI_API_KEY /
                           $ANTHROPIC_API_KEY
      --base-url URL       base URL for --provider (OpenAI-compatible gateways)
      --api-key KEY        API key for --provider (prefer the env var)
      --provider-cache on|off  Anthropic prompt caching over the stable prefix
                           (tools + system). Default on: low-risk, cuts cost and
                           latency across turns. Ignored for --provider openai
  -p, --prompt TEXT        run one prompt and exit after the reply
  /resume [prefix]         resume a saved session at startup (a sha prefix or
                           list number; omit to resume the most recent)
  /export [md|html] [path] write the current transcript to a shareable file
                           (default: markdown, auto-named in the working
                           directory); HTML output is standalone
  /open [path]             edit a file in the built-in editor; with no path,
                           reopens the last file a tool call edited or plank
                           generated (e.g. a /repro dump)
  /insights [fast]         report on how you have been using plank, from every
                           saved session; writes ~/.plank/usage-data/report.html
                           (\"fast\" skips the written sections)
      --non-interactive    disable the interactive UI
      --dump-config        print every effective setting with the layer it came
                           from (default, plugin, ~/.plank, ./.plank, CLI) and exit
      --minimal-prompt     start with the smallest prompt this build can make:
                           no MCP servers, skills, templates, plugin agents,
                           WASM components or session-start context. For
                           measuring throughput against a bare prompt.
      --ui-remote[=PORT]   accept TUI remote control on 127.0.0.1:PORT
                           (omit PORT for an ephemeral one, printed to stderr)
  -sys, --system TEXT      override the system prompt
      --trace PATH         append a trace log to PATH
  -c, --ctx N              context window size in tokens (default 1048576)
  -n, --tokens N           maximum tokens to generate (default 50000)
      --temp F             sampling temperature (0..100)
      --top-p F            nucleus sampling threshold (0..1)
      --min-p F            minimum-probability threshold (0..1)
      --seed N             RNG seed (positive integer)
      --think              ordinary thinking (default); same as /think medium
      --think-low          ask for brief reasoning (experimental; prompt-only)
      --think-max          maximum reasoning effort; needs --ctx 393216 or more
      --nothink            disable thinking
      --chdir PATH         change working directory before starting
      --worktree NAME      start inside an isolated git worktree of this repo
      --worktree-pr N      base that worktree on pull request N (implies --worktree)
      --mcp-config FILE    local MCP server config (default: ./.mcp.json);
                           overlays the global ~/.plank/.mcp.json by name
      --plugin-dir PATH    load a plugin directory for this session (repeatable)
      --sandbox            run model bash commands under sandbox-exec
                           (writes limited to cwd/temp; see sandbox.json).
                           On by default on macOS
      --no-sandbox         disable the bash sandbox
      --disable-btw-suspend
                           answer an in-pass /btw by queuing at the next
                           generation boundary instead of freezing/resuming the
                           running generation (freeze/resume is the default)

Settings file:
      ~/.plank/settings.json, then ./.plank/settings.json (later wins), holds
      defaults for preferences rather than per-run choices. Flags override it.

        {
          \"engine\": { \"model\": \"~/models/ds4.gguf\", \"threads\": 8,
                      \"backend\": \"metal\", \"power\": 80, \"ctx\": 262144 },
          \"ui\":     { \"respectGitignore\": true, \"popupRows\": 15,
                      \"indexRefreshSecs\": 5, \"historySize\": 512 },
          \"safety\": { \"sandbox\": true, \"btwSuspend\": true },
          \"mcp\":    { \"timeoutSecs\": 30 }
        }

      Edit these interactively with /config (a TUI form), or set one from the
      prompt: /config <section>.<key> <value> (e.g. /config ui.showThinking
      false). Changes save to ./.plank/settings.json.

      No secrets: keep the provider API key on --api-key or the environment,
      since ./.plank/settings.json is inside the working tree.
"
    .to_owned()
}

/// Parses a positive `i32`, naming `opt` in the error message.
///
/// # Errors
/// Returns an error when `s` is not an integer in `1..=i32::MAX`.
pub fn parse_int(s: &str, opt: &str) -> Result<i32, String> {
    s.parse::<i32>()
        .ok()
        .filter(|v| *v > 0)
        .ok_or_else(|| format!("invalid value for {opt}: {s}"))
}

/// Parses a positive `u64`, naming `opt` in the error message.
///
/// # Errors
/// Returns an error when `s` is not a nonzero unsigned integer.
pub fn parse_u64(s: &str, opt: &str) -> Result<u64, String> {
    s.parse::<u64>()
        .ok()
        .filter(|v| *v != 0)
        .ok_or_else(|| format!("invalid value for {opt}: {s}"))
}

/// Parses a finite `f32` within `[min, max]`, naming `opt` in the error message.
///
/// # Errors
/// Returns an error when `s` is not a finite float within the range.
pub fn parse_float_range(s: &str, opt: &str, min: f32, max: f32) -> Result<f32, String> {
    s.parse::<f32>()
        .ok()
        .filter(|v| v.is_finite() && *v >= min && *v <= max)
        .ok_or_else(|| format!("invalid value for {opt}: {s}"))
}

/// Parses a positive GiB size like `64` or `64GB` into bytes, mirroring
/// `ds4_parse_gib_arg`: digits with an optional case-insensitive `gb` suffix.
#[must_use]
pub fn parse_gib_arg(s: &str) -> Option<u64> {
    let digits = s
        .strip_suffix("gb")
        .or_else(|| s.strip_suffix("GB"))
        .or_else(|| s.strip_suffix("Gb"))
        .or_else(|| s.strip_suffix("gB"))
        .unwrap_or(s);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let v = digits.parse::<u64>().ok().filter(|v| *v != 0)?;
    v.checked_mul(1024 * 1024 * 1024)
}

/// Parses `--ssd-streaming-cache-experts`: a positive expert count, or a
/// `<N>GB` byte bound. Mirrors `ds4_parse_streaming_cache_experts_arg`;
/// exactly one of the returned pair is nonzero.
#[must_use]
pub fn parse_streaming_cache_experts_arg(s: &str) -> Option<(u32, u64)> {
    if s.len() > 2 && s[s.len() - 2..].eq_ignore_ascii_case("gb") {
        return parse_gib_arg(s).map(|bytes| (0, bytes));
    }
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse::<u32>().ok().filter(|v| *v != 0).map(|v| (v, 0))
}

/// Parses a power percentage in `1..=100`; returns `None` when invalid.
#[must_use]
pub fn parse_power_percent(arg: &str) -> Option<i32> {
    arg.parse::<i32>().ok().filter(|v| (1..=100).contains(v))
}

/// The leading whitespace-delimited token of `cmd`.
#[must_use]
fn first_token(cmd: &str) -> &str {
    cmd.split_once(char::is_whitespace).map_or(cmd, |(t, _)| t)
}

/// One built-in slash command as the `/` menu shows it: the command itself, a
/// hint for what it takes, and a one-line explanation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlashCommand {
    /// Command including the leading slash, e.g. `/compact`.
    pub name: &'static str,
    /// Argument hint shown after the name; empty when it takes none.
    pub args: &'static str,
    /// One-line help, shown beside the name in the menu.
    pub desc: &'static str,
}

/// Every built-in slash command the `/` menu offers, in the order it lists them
/// — roughly most-reached-for first rather than alphabetical.
///
/// Deliberately excluded, matching what [`slash_command_known_with`] treats as
/// hidden: the `crate::arcade` easter eggs and `/renotify` (a screenshot aid).
/// Skills and templates are *not* here — they are discovered at runtime and
/// appended by `crate::slashmenu::catalog`.
///
/// A test in this module asserts every entry is a command dispatch actually
/// knows, so the menu can never advertise something that would be forwarded to
/// the model as an ordinary prompt.
pub const SLASH_COMMANDS: &[SlashCommand] = &[
    SlashCommand {
        name: "/help",
        args: "",
        desc: "show the full option and command reference",
    },
    SlashCommand {
        name: "/new",
        args: "",
        desc: "start a fresh conversation, dropping the transcript",
    },
    SlashCommand {
        name: "/clear",
        args: "",
        desc: "alias for /new",
    },
    SlashCommand {
        name: "/compact",
        args: "[instructions]",
        desc: "summarize the transcript to reclaim context",
    },
    SlashCommand {
        name: "/context",
        args: "",
        desc: "report what is filling the context window",
    },
    SlashCommand {
        name: "/usage",
        args: "",
        desc: "report token usage and timings for this session",
    },
    SlashCommand {
        name: "/config",
        args: "[section.key value]",
        desc: "edit settings in a form, or set one inline",
    },
    SlashCommand {
        name: "/save",
        args: "",
        desc: "save the current session to ~/.plank",
    },
    SlashCommand {
        name: "/rename",
        args: "<name>",
        desc: "rename this session (the saved copy keeps its old name)",
    },
    SlashCommand {
        name: "/rate",
        args: "[+|-] [note]",
        desc: "rate the last turn (thumbs up by default) with an optional note",
    },
    SlashCommand {
        name: "/list",
        args: "",
        desc: "list saved sessions",
    },
    SlashCommand {
        name: "/resume",
        args: "[prefix]",
        desc: "resume a saved session (name or list number)",
    },
    SlashCommand {
        name: "/switch",
        args: "<name>",
        desc: "switch to another saved session by name",
    },
    SlashCommand {
        name: "/del",
        args: "<prefix>",
        desc: "delete a saved session",
    },
    SlashCommand {
        name: "/retitle",
        args: "",
        desc: "re-derive every saved session's title from its transcript",
    },
    SlashCommand {
        name: "/tag",
        args: "<text>",
        desc: "label this session (\"-\" clears the label)",
    },
    SlashCommand {
        name: "/fork",
        args: "[name]",
        desc: "branch the conversation at the current turn",
    },
    SlashCommand {
        name: "/clone",
        args: "",
        desc: "copy the current branch into a new session",
    },
    SlashCommand {
        name: "/tree",
        args: "",
        desc: "show the branch tree of this session",
    },
    SlashCommand {
        name: "/checkpoint",
        args: "[name]",
        desc: "snapshot the working tree so a turn can be undone",
    },
    SlashCommand {
        name: "/rollback",
        args: "[name]",
        desc: "restore the working tree from a checkpoint",
    },
    SlashCommand {
        name: "/history",
        args: "",
        desc: "show the prompt history for this directory",
    },
    SlashCommand {
        name: "/btw",
        args: "<question>",
        desc: "ask a side question without disturbing the main task",
    },
    SlashCommand {
        name: "/subagent",
        args: "<task>",
        desc: "run a task in a sub-agent with its own context",
    },
    SlashCommand {
        name: "/agent",
        args: "",
        desc: "list the sub-agents available to the model",
    },
    SlashCommand {
        name: "/tasks",
        args: "",
        desc: "show the model's task list for this session",
    },
    SlashCommand {
        name: "/goal",
        args: "[--max n] <objective>",
        desc: "work autonomously until the objective is settled",
    },
    SlashCommand {
        name: "/skills",
        args: "",
        desc: "list the skills loaded from SKILL.md files",
    },
    SlashCommand {
        name: "/plugins",
        args: "[install <dir|url>|remove|trust|info|disable|enable|publisher]",
        desc: "list, install, remove, inspect, disable, approve, or trust a key",
    },
    SlashCommand {
        name: "/install-claude-plugin",
        args: "<url|owner/repo> [plugin-name] [--force]",
        desc: "install a Claude Code plugin from a repository or a tarball",
    },
    SlashCommand {
        name: "/frame",
        args: "[id]",
        desc: "open a wasm frame component, or list the openable ones",
    },
    SlashCommand {
        name: "/templates",
        args: "",
        desc: "list the prompt templates available as commands",
    },
    SlashCommand {
        name: "/hooks",
        args: "",
        desc: "list the configured hooks and what triggers them",
    },
    SlashCommand {
        name: "/mcp",
        args: "",
        desc: "report connected MCP servers and their tools",
    },
    SlashCommand {
        name: "/init",
        args: "",
        desc: "write an AGENTS.md describing this repository",
    },
    SlashCommand {
        name: "/remember",
        args: "[user] <text>",
        desc: "append a durable note to project or user memory",
    },
    SlashCommand {
        name: "/think",
        args: "[low|medium|max|off]",
        desc: "set how much reasoning to ask the model for",
    },
    SlashCommand {
        name: "/power",
        args: "<1-100>",
        desc: "set the GPU power cap percentage",
    },
    SlashCommand {
        name: "/notify",
        args: "[mode]",
        desc: "choose when a finished turn notifies you",
    },
    SlashCommand {
        name: "/strip",
        args: "[on|off]",
        desc: "toggle stripping thinking blocks from the transcript",
    },
    SlashCommand {
        name: "/kvcache",
        args: "[gc|pin|unpin|rm]",
        desc: "show the KV cache tree; pin, delete or sweep entries",
    },
    SlashCommand {
        name: "/export",
        args: "[md|html] [path]",
        desc: "write the transcript to a shareable file",
    },
    SlashCommand {
        name: "/open",
        args: "[path]",
        desc: "edit a file in the built-in editor (default: last edited)",
    },
    SlashCommand {
        name: "/insights",
        args: "[fast]",
        desc: "report on how you have been using plank",
    },
    SlashCommand {
        name: "/repro",
        args: "[path]",
        desc: "write a reproducer bundle for the current session",
    },
    SlashCommand {
        name: "/remote-control",
        args: "[on|ask|off]",
        desc: "show the TUI remote-control endpoint",
    },
    SlashCommand {
        name: "/rc",
        args: "[on|ask|off]",
        desc: "alias for /remote-control",
    },
    SlashCommand {
        name: "/remote",
        args: "",
        desc: "report the remote engine this session is driving",
    },
    SlashCommand {
        name: "/grant",
        args: "[session]",
        desc: "hand remote control to a waiting client",
    },
    SlashCommand {
        name: "/version",
        args: "",
        desc: "print the plank version",
    },
    SlashCommand {
        name: "/quit",
        args: "",
        desc: "leave plank",
    },
    SlashCommand {
        name: "/exit",
        args: "",
        desc: "alias for /quit",
    },
];

/// True when `cmd` is `name` alone or `name` followed by whitespace.
#[must_use]
pub fn slash_command_with_args(cmd: &str, name: &str) -> bool {
    cmd.strip_prefix(name)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with(char::is_whitespace))
}

/// True when `cmd` is one of the agent's known slash commands.
///
/// Reads `ui.easterEggs` for the arcade commands; see
/// [`slash_command_known_with`] for the pure form the tests drive.
#[must_use]
pub fn slash_command_known(cmd: &str) -> bool {
    slash_command_known_with(cmd, crate::settings::active().ui.easter_eggs)
}

/// [`slash_command_known`] with the `ui.easterEggs` decision passed in, so both
/// answers are testable without touching the process-wide settings.
#[must_use]
pub fn slash_command_known_with(cmd: &str, easter_eggs: bool) -> bool {
    matches!(
        cmd,
        "/help"
            | "/save"
            | "/list"
            | "/retitle"
            | "/quit"
            | "/exit"
            | "/new"
            | "/clear"
            | "/mcp"
            | "/context"
            | "/usage"
            | "/init"
            | "/skills"
            | "/plugins"
            | "/install-claude-plugin"
            | "/frame"
            | "/templates"
            | "/tasks"
            | "/agent"
            | "/hooks"
            | "/remote-control"
            | "/rc"
            | "/remote"
            | "/grant"
            | "/version"
            | "/tree"
            | "/clone"
    )
        // Easter eggs (`crate::arcade`). Known so the dispatcher runs them
        // instead of forwarding the line to the model, but deliberately left
        // out of `usage()` and the completion popup. They take an argument
        // (`new`), so they go through the with-args form. Kept in sync with
        // `Arcade::COMMANDS` by a test in that module. With `ui.easterEggs`
        // off they stop being known at all, so the line reaches the model as
        // an ordinary prompt rather than being recognized and refused.
        || (easter_eggs
            && crate::arcade::Arcade::COMMANDS
                .iter()
                .any(|egg| slash_command_with_args(cmd, egg)))
        // `/compact` takes optional extra summarization instructions, so it goes
        // through the with-args form: bare `/compact` and
        // `/compact focus on the parser bug` are both valid invocations.
        || slash_command_with_args(cmd, "/compact")
        || slash_command_with_args(cmd, "/config")
        || slash_command_with_args(cmd, "/fork")
        || slash_command_with_args(cmd, "/btw")
        // `/subagent` also takes a `:<name>` suffix selecting a definition, so
        // it needs its own recognizer rather than the plain with-args form.
        // Any name is *known* here; whether it exists is dispatch's business,
        // and reporting a bad name beats silently forwarding the line to the
        // model as an ordinary prompt.
        || crate::agents::is_subagent_command(first_token(cmd))
        || slash_command_with_args(cmd, "/remember")
        || slash_command_with_args(cmd, "/repro")
        || slash_command_with_args(cmd, "/export")
        || slash_command_with_args(cmd, "/open")
        || slash_command_with_args(cmd, "/insights")
        || slash_command_with_args(cmd, "/resume")
        || slash_command_with_args(cmd, "/tag")
        || slash_command_with_args(cmd, "/rename")
        || slash_command_with_args(cmd, "/rate")
        || slash_command_with_args(cmd, "/power")
        || slash_command_with_args(cmd, "/think")
        || slash_command_with_args(cmd, "/switch")
        || slash_command_with_args(cmd, "/del")
        || slash_command_with_args(cmd, "/strip")
        || slash_command_with_args(cmd, "/kvcache")
        || slash_command_with_args(cmd, "/history")
        || slash_command_with_args(cmd, "/checkpoint")
        || slash_command_with_args(cmd, "/rollback")
        || slash_command_with_args(cmd, "/notify")
        || slash_command_with_args(cmd, "/goal")
        || slash_command_with_args(cmd, "/remote-control")
        || slash_command_with_args(cmd, "/rc")
        // `/grant` takes an optional session id (`/grant 3`) as well as the bare
        // form that answers the oldest waiting request.
        || slash_command_with_args(cmd, "/grant")
}

/// Why a slash command cannot be run from a remote client, or `None` if it can.
///
/// A remote `command` frame is queued for the same dispatcher the local user
/// types into, so most commands work unchanged. Two families do not, and
/// queueing them silently is worse than refusing them: commands that take over
/// the *local* terminal (the operator sees a pane open with nobody driving it),
/// and commands that would saw off the branch the client is sitting on.
///
/// Pane commands are refused only in their bare, interactive form —
/// `/kvcache gc` and `/resume 3` are ordinary non-interactive commands and stay
/// available. `line` is the whole command line, leading slash included.
#[must_use]
pub fn slash_command_remote_refusal(line: &str) -> Option<&'static str> {
    let mut it = line.split_whitespace();
    let cmd = it.next().unwrap_or("");
    let bare = it.next().is_none();
    match cmd {
        "/grant" => Some(
            "/grant answers a remote request and is local-only — asking for it from the client that wants control would be self-granting",
        ),
        "/remote-control" | "/rc" => {
            Some("toggling remote control would disconnect this client; run it locally")
        }
        "/open" => Some("/open takes over the local terminal with the built-in editor"),
        "/quit" | "/exit" => Some("quitting is local-only; use the local terminal to stop plank"),
        "/kvcache" | "/resume" if bare => {
            Some("opens an interactive local pane; pass an argument for the non-interactive form")
        }
        _ => None,
    }
}

/// Parses one engine-tuning option that takes a value (already extracted as
/// `v`). `steering_scale_set` tracks explicit steering scales so a steering
/// file alone can default the FFN scale to 1.0, like the C.
fn parse_engine_option(
    e: &mut EngineTuning,
    arg: &str,
    v: &str,
    steering_scale_set: &mut bool,
) -> Result<(), String> {
    match arg {
        "--mtp" => e.mtp_path = Some(PathBuf::from(v)),
        "--mtp-draft" => e.mtp_draft_tokens = parse_int(v, arg)?,
        "--mtp-margin" => e.mtp_margin = parse_float_range(v, arg, 0.0, 1000.0)?,
        // The C turns DSpark on for any of its three flags, so the threshold
        // flag alone is enough to select the DSpark runtime.
        "--dspark-confidence" => {
            e.dspark = true;
            e.dspark_confidence = Some(parse_float_range(v, arg, 0.0, 1.0)?);
        }
        "--ssd-streaming-cache-experts" => {
            let (experts, bytes) = parse_streaming_cache_experts_arg(v)
                .ok_or_else(|| format!("{arg} must be a positive count or <number>GB: {v}"))?;
            e.ssd_streaming_cache_experts = experts;
            e.ssd_streaming_cache_bytes = bytes;
        }
        "--ssd-streaming-preload-experts" => {
            e.ssd_streaming_preload_experts = u32::try_from(parse_int(v, arg)?).unwrap_or(0);
        }
        "--simulate-used-memory" => {
            e.simulate_used_memory_bytes = parse_gib_arg(v)
                .ok_or_else(|| format!("{arg} must be a positive GiB value, e.g. 64GB: {v}"))?;
        }
        "--dir-steering-file" => e.dir_steering_file = Some(PathBuf::from(v)),
        "--dir-steering-ffn" => {
            e.dir_steering_ffn = parse_float_range(v, arg, -100.0, 100.0)?;
            *steering_scale_set = true;
        }
        "--dir-steering-attn" => {
            e.dir_steering_attn = parse_float_range(v, arg, -100.0, 100.0)?;
            *steering_scale_set = true;
        }
        _ => return Err(format!("unknown option: {arg}")),
    }
    Ok(())
}

/// Parses command-line arguments (without the program name) into a config.
///
/// # Errors
/// Returns an error naming the offending option when a value is missing,
/// out of range, or an option is unknown.
#[allow(clippy::too_many_lines)] // flat flag-dispatch match; splitting hurts readability.
pub fn parse_options(args: &[String]) -> Result<AgentConfig, String> {
    parse_options_with(&crate::settings::Settings::default(), args)
}

/// Parses command-line options over `settings` as the starting defaults.
///
/// A flag always wins over the settings file, because `settings` only seeds
/// the initial config and every flag assigns over it. Values the file cannot
/// hold (`--prompt`, `--trace`, the serve options) are unaffected.
///
/// # Errors
/// Returns an error naming the offending option when a value is missing,
/// out of range, or an option is unknown.
#[allow(clippy::too_many_lines)] // flat flag-dispatch match; splitting hurts readability.
pub fn parse_options_with(
    settings: &crate::settings::Settings,
    args: &[String],
) -> Result<AgentConfig, String> {
    let mut c = AgentConfig::from_settings(settings);
    // Tracks whether a steering scale was given explicitly; a steering file
    // without one defaults the FFN scale to 1.0, like the C.
    let mut steering_scale_set = false;
    let mut temp_set = false;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let need_arg = |i: &mut usize| -> Result<&str, String> {
            if *i + 1 >= args.len() {
                return Err(format!("missing value for {arg}"));
            }
            *i += 1;
            Ok(args[*i].as_str())
        };
        match arg {
            "-V" | "--version" => {
                c.show_version = true;
                return Ok(c);
            }
            "-h" | "--help" => {
                c.show_help = true;
                if let Some(topic) = args.get(i + 1)
                    && !topic.starts_with('-')
                {
                    c.help_topic = Some(topic.clone());
                }
                return Ok(c);
            }
            "-m" | "--model" => {
                c.model_path = Some(PathBuf::from(need_arg(&mut i)?));
                c.cli_set("engine.model");
            }
            "-t" | "--threads" => {
                c.n_threads = parse_int(need_arg(&mut i)?, arg)?;
                c.cli_set("engine.threads");
            }
            "--backend" => {
                let v = need_arg(&mut i)?;
                c.backend = Some(parse_backend(v).ok_or_else(|| format!("invalid backend: {v}"))?);
                c.cli_set("engine.backend");
            }
            "--metal" => {
                c.backend = Some(Backend::Metal);
                c.cli_set("engine.backend");
            }
            "--cuda" => {
                c.backend = Some(Backend::Cuda);
                c.cli_set("engine.backend");
            }
            "--cpu" => {
                c.backend = Some(Backend::Cpu);
                c.cli_set("engine.backend");
            }
            "--power" => {
                let v = need_arg(&mut i)?;
                c.power_percent = parse_power_percent(v)
                    .ok_or_else(|| format!("invalid value for {arg}: {v}"))?;
                c.cli_set("engine.power");
            }
            "-p" | "--prompt" => c.prompt = Some(need_arg(&mut i)?.to_owned()),
            "/resume" => {
                // Optional following token is the session prefix/number; a
                // flag (or nothing) means "most recent".
                let prefix = match args.get(i + 1) {
                    Some(next) if !next.starts_with('-') => {
                        i += 1;
                        next.clone()
                    }
                    _ => String::new(),
                };
                c.resume = Some(prefix);
            }
            "--remote" => c.remote_url = Some(need_arg(&mut i)?.to_owned()),
            "--remote-token" => c.remote_token = Some(need_arg(&mut i)?.to_owned()),
            "--insecure" => c.insecure = true,
            "--shared-engine" => c.shared_engine = true,
            "--max-sessions" => c.max_sessions = parse_int(need_arg(&mut i)?, arg)?,
            "--idle-reclaim-secs" => {
                c.idle_reclaim_secs =
                    u64::try_from(parse_int(need_arg(&mut i)?, arg)?.max(0)).unwrap_or(0);
            }
            "--session-ctx-size" => c.session_ctx_size = parse_int(need_arg(&mut i)?, arg)?.max(0),
            "--kv-budget-bytes" => {
                c.kv_budget_bytes = parse_u64(need_arg(&mut i)?, arg)?;
            }
            "--provider" => {
                c.provider = Some(match need_arg(&mut i)? {
                    "openai" => ProviderSelector::OpenAi,
                    "anthropic" => ProviderSelector::Anthropic,
                    other => return Err(format!("invalid provider: {other}")),
                });
            }
            "--base-url" => c.provider_base_url = Some(need_arg(&mut i)?.to_owned()),
            "--api-key" => c.provider_api_key = Some(need_arg(&mut i)?.to_owned()),
            "--provider-cache" => {
                c.provider_cache = match need_arg(&mut i)? {
                    "on" | "true" | "1" => true,
                    "off" | "false" | "0" => false,
                    other => {
                        return Err(format!(
                            "invalid --provider-cache value: {other} (use on|off)"
                        ));
                    }
                };
            }
            "--non-interactive" => c.non_interactive = true,
            "--dump-config" => c.dump_config = true,
            "--minimal-prompt" => c.minimal_prompt = true,
            // Bare `--ui-remote` means an ephemeral port. A following bare
            // number is almost certainly someone meaning to pin one, so
            // reject it rather than silently binding an ephemeral port and
            // leaving the number to fall through as an unknown argument.
            "--ui-remote" => {
                if args.get(i + 1).is_some_and(|n| n.parse::<u16>().is_ok()) {
                    return Err(format!(
                        "--ui-remote takes no separate argument; use --ui-remote={}",
                        args[i + 1]
                    ));
                }
                c.ui_remote = Some(0);
            }
            a if a.starts_with("--ui-remote=") => {
                let v = &a["--ui-remote=".len()..];
                c.ui_remote = Some(
                    v.parse::<u16>()
                        .map_err(|_| format!("--ui-remote: bad port {v:?}"))?,
                );
            }
            "-sys" | "--system" => need_arg(&mut i)?.clone_into(&mut c.system),
            "--trace" => c.trace_path = Some(PathBuf::from(need_arg(&mut i)?)),
            "-c" | "--ctx" => {
                c.generation.ctx_size = parse_int(need_arg(&mut i)?, arg)?;
                c.ctx_size_explicit = true;
                c.cli_set("engine.ctx");
            }
            "-n" | "--tokens" => c.generation.n_predict = parse_int(need_arg(&mut i)?, arg)?,
            "--temp" => {
                c.generation.temperature = parse_float_range(need_arg(&mut i)?, arg, 0.0, 100.0)?;
                temp_set = true;
            }
            "--top-p" => c.generation.top_p = parse_float_range(need_arg(&mut i)?, arg, 0.0, 1.0)?,
            "--min-p" => c.generation.min_p = parse_float_range(need_arg(&mut i)?, arg, 0.0, 1.0)?,
            "--seed" => c.generation.seed = parse_u64(need_arg(&mut i)?, arg)?,
            // `--think-max` used to be an alias for ordinary thinking; it now
            // selects the level it names, which the engine has always had.
            "--think" => c.generation.think_mode = ThinkMode::Medium,
            "--think-low" => c.generation.think_mode = ThinkMode::Low,
            "--think-max" => c.generation.think_mode = ThinkMode::Max,
            "--nothink" => c.generation.think_mode = ThinkMode::Off,
            "--chdir" => c.chdir_path = Some(PathBuf::from(need_arg(&mut i)?)),
            "--worktree" => c.worktree = Some(need_arg(&mut i)?.to_string()),
            "--worktree-pr" => {
                let n: i32 = parse_int(need_arg(&mut i)?, arg)?;
                let n = u32::try_from(n)
                    .map_err(|_| format!("{arg} must be a positive pull request number"))?;
                c.worktree_pr = Some(n);
            }
            "--mcp-config" => c.mcp_config_path = Some(PathBuf::from(need_arg(&mut i)?)),
            "--plugin-dir" => c.plugin_dirs.push(PathBuf::from(need_arg(&mut i)?)),
            "--sandbox" => {
                c.sandbox_override = Some(true);
                c.cli_set("safety.sandbox");
            }
            "--no-sandbox" => {
                c.sandbox_override = Some(false);
                c.cli_set("safety.sandbox");
            }
            "--btw-suspend" => {
                c.btw.suspend = true;
                c.cli_set("safety.btwSuspend");
            }
            "--disable-btw-suspend" => {
                c.btw.suspend = false;
                c.cli_set("safety.btwSuspend");
            }
            "--quality" => c.engine.quality = true,
            "--warm-weights" => c.engine.warm_weights = true,
            "--ssd-streaming" => c.engine.ssd_streaming = true,
            "--ssd-streaming-cold" => c.engine.ssd_streaming_cold = true,
            "--dspark" => c.engine.dspark = true,
            "--dspark-off" => c.engine.dspark = false,
            "--dspark-strict" => {
                c.engine.dspark = true;
                c.engine.dspark_strict = true;
            }
            "--mtp"
            | "--mtp-draft"
            | "--mtp-margin"
            | "--dspark-confidence"
            | "--ssd-streaming-cache-experts"
            | "--ssd-streaming-preload-experts"
            | "--simulate-used-memory"
            | "--dir-steering-file"
            | "--dir-steering-ffn"
            | "--dir-steering-attn" => {
                parse_engine_option(
                    &mut c.engine,
                    arg,
                    need_arg(&mut i)?,
                    &mut steering_scale_set,
                )?;
            }
            _ => return Err(format!("unknown option: {arg}")),
        }
        i += 1;
    }
    finalize(&mut c, steering_scale_set, temp_set)?;
    Ok(c)
}

/// Post-parse fixups: the steering-scale default, the `--dspark` temperature
/// default, and `--remote` validation.
fn finalize(c: &mut AgentConfig, steering_scale_set: bool, temp_set: bool) -> Result<(), String> {
    if c.engine.dir_steering_file.is_some() && !steering_scale_set {
        c.engine.dir_steering_ffn = 1.0;
    }
    // Speculative decoding only engages at temperature 0 (see `ds4engine`'s
    // draft gate), so DSpark defaults the temperature to 0. Done here rather
    // than at the flag because `--temp` may follow it; an explicit `--temp` in
    // either order still wins. `--dspark-off` leaves the 0.6 default in force.
    if c.engine.dspark && !temp_set {
        c.generation.temperature = 0.0;
    }
    // The same context floor `/think max` enforces, applied to `--think-max`.
    // Checked here rather than at the flag because `--ctx` may follow it.
    if c.generation.think_mode == ThinkMode::Max
        && c.generation.ctx_size < crate::engine::THINK_MAX_MIN_CONTEXT
    {
        return Err(format!(
            "--think-max needs --ctx {} or more (got {})",
            crate::engine::THINK_MAX_MIN_CONTEXT,
            c.generation.ctx_size
        ));
    }
    // --provider (flavor b) selects a third-party API engine. Mutually
    // exclusive with the local selectors and with --remote (§4.7).
    if let Some(provider) = c.provider {
        if c.remote_url.is_some() {
            return Err("--provider cannot be combined with --remote".to_string());
        }
        if c.backend.is_some() {
            return Err("--provider cannot be combined with --metal/--cuda/--cpu".to_string());
        }
        // `--model NAME` names the provider model (not a local GGUF path).
        if let Some(path) = c.model_path.take() {
            c.provider_model = Some(path.to_string_lossy().into_owned());
        }
        if c.provider_model.is_none() {
            return Err("--provider requires --model NAME".to_string());
        }
        // Keys never live in config; read from the provider's env var when not
        // given on the command line (§4.7, constraint 6).
        if c.provider_api_key.is_none() {
            let env = match provider {
                ProviderSelector::OpenAi => "OPENAI_API_KEY",
                ProviderSelector::Anthropic => "ANTHROPIC_API_KEY",
            };
            c.provider_api_key = std::env::var(env).ok().filter(|k| !k.is_empty());
        }
        return Ok(());
    }
    let Some(url) = c.remote_url.clone() else {
        return Ok(());
    };
    // --remote is mutually exclusive with the local model/backend selectors (§4.7).
    if c.model_path.is_some() {
        return Err("--remote cannot be combined with -m/--model".to_string());
    }
    if c.backend.is_some() {
        return Err("--remote cannot be combined with --metal/--cuda/--cpu".to_string());
    }
    crate::remote::validate_remote_url(&url, c.insecure)?;
    if c.remote_token.is_none() {
        c.remote_token = std::env::var("PLANK_REMOTE_TOKEN")
            .ok()
            .filter(|t| !t.is_empty());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(ToString::to_string).collect()
    }

    /// Settings with every engine and safety key set, for override tests.
    fn full_settings() -> crate::settings::Settings {
        crate::settings::Settings {
            engine: crate::settings::EngineSettings {
                model: Some(PathBuf::from("/from/settings.gguf")),
                threads: Some(4),
                backend: Some("cpu".to_string()),
                power: Some(50),
                ctx: Some(4096),
                thinking_tool_calls: true,
            },
            safety: crate::settings::SafetySettings {
                sandbox: Some(true),
                btw_suspend: Some(true),
            },
            ..crate::settings::Settings::default()
        }
    }

    #[test]
    fn settings_seed_the_engine_and_safety_defaults() {
        let c = parse_options_with(&full_settings(), &[]).unwrap();
        assert_eq!(c.model_path, Some(PathBuf::from("/from/settings.gguf")));
        assert_eq!(c.n_threads, 4);
        assert_eq!(c.backend, Some(Backend::Cpu));
        assert_eq!(c.power_percent, 50);
        assert_eq!(c.generation.ctx_size, 4096);
        assert!(c.ctx_size_explicit, "a settings `ctx` is the user's choice");
        assert_eq!(c.sandbox_override, Some(true));
        assert!(c.btw.suspend);
    }

    #[test]
    fn every_flag_overrides_the_settings_file() {
        let c = parse_options_with(
            &full_settings(),
            &args(&[
                "-m",
                "/from/flag.gguf",
                "-t",
                "16",
                "--metal",
                "--power",
                "90",
                "-c",
                "8192",
                "--no-sandbox",
                "--disable-btw-suspend",
            ]),
        )
        .unwrap();
        assert_eq!(c.model_path, Some(PathBuf::from("/from/flag.gguf")));
        assert_eq!(c.n_threads, 16);
        assert_eq!(c.backend, Some(Backend::Metal));
        assert_eq!(c.power_percent, 90);
        assert_eq!(c.generation.ctx_size, 8192);
        assert_eq!(c.sandbox_override, Some(false));
        assert!(!c.btw.suspend);
    }

    #[test]
    fn cli_flags_record_provenance_for_the_keys_they_override() {
        // `/config --resolved` needs to show a CLI flag beating the file
        // provenance, so every flag that shadows a settings key must record it.
        use crate::provenance::Origin;
        let c = parse_options_with(
            &full_settings(),
            &args(&[
                "-m",
                "/from/flag.gguf",
                "-t",
                "16",
                "--metal",
                "--power",
                "90",
                "-c",
                "8192",
                "--no-sandbox",
                "--disable-btw-suspend",
            ]),
        )
        .unwrap();
        assert_eq!(c.cli_provenance.get("engine.model"), Some(&Origin::Cli));
        assert_eq!(c.cli_provenance.get("engine.threads"), Some(&Origin::Cli));
        assert_eq!(c.cli_provenance.get("engine.backend"), Some(&Origin::Cli));
        assert_eq!(c.cli_provenance.get("engine.power"), Some(&Origin::Cli));
        assert_eq!(c.cli_provenance.get("engine.ctx"), Some(&Origin::Cli));
        assert_eq!(c.cli_provenance.get("safety.sandbox"), Some(&Origin::Cli));
        assert_eq!(
            c.cli_provenance.get("safety.btwSuspend"),
            Some(&Origin::Cli)
        );
        // A flag that shadows no settings key records nothing.
        let c = parse_options(&args(&["--non-interactive"])).unwrap();
        assert!(c.cli_provenance.is_empty());
    }

    #[test]
    fn dump_config_flag_is_parsed() {
        let c = parse_options(&args(&["--dump-config"])).unwrap();
        assert!(c.dump_config);
    }

    #[test]
    fn an_unrecognised_backend_in_settings_is_ignored_not_fatal() {
        // A typo in settings.json must never stop plank from starting; the
        // same text on `--backend` is still an error.
        let s = crate::settings::Settings {
            engine: crate::settings::EngineSettings {
                backend: Some("quantum".to_string()),
                ..crate::settings::EngineSettings::default()
            },
            ..crate::settings::Settings::default()
        };
        assert_eq!(parse_options_with(&s, &[]).unwrap().backend, None);
        assert!(parse_options(&args(&["--backend", "quantum"])).is_err());
    }

    #[test]
    fn plugin_dir_is_repeatable() {
        let c =
            parse_options(&args(&["--plugin-dir", "/a", "--plugin-dir", "/b"])).expect("parses");
        assert_eq!(
            c.plugin_dirs,
            vec![PathBuf::from("/a"), PathBuf::from("/b")]
        );
    }

    #[test]
    fn parse_options_ignores_the_users_real_settings_file() {
        // `parse_options` must stay hermetic: it is what the tests and library
        // consumers call, and it may not read whatever is on this machine.
        assert_eq!(
            parse_options(&[]).unwrap().model_path,
            AgentConfig::default().model_path
        );
    }

    #[test]
    fn defaults() {
        let c = parse_options(&[]).unwrap();
        assert_eq!(c.generation.n_predict, DEFAULT_N_PREDICT);
        assert_eq!(c.generation.ctx_size, DEFAULT_CTX_SIZE);
        assert!(!c.ctx_size_explicit);
        assert!(
            parse_options(&["-c".into(), "8192".into()])
                .unwrap()
                .ctx_size_explicit
        );
        assert_eq!(c.system, DEFAULT_SYSTEM_PROMPT);
        assert_eq!(c.generation.think_mode, ThinkMode::Medium);
        assert!(c.prompt.is_none());
        // In-pass /btw suspend is on by default; --disable-btw-suspend opts out.
        assert!(c.btw.suspend);
        assert!(!c.non_interactive);
        assert!(!c.show_help);
        // Shared engine is opt-in (issue #28); off by default.
        assert!(!c.shared_engine);
    }

    #[test]
    fn parses_shared_engine_flags() {
        let c = parse_options(&args(&[
            "--shared-engine",
            "--max-sessions",
            "4",
            "--idle-reclaim-secs",
            "30",
        ]))
        .unwrap();
        assert!(c.shared_engine);
        assert_eq!(c.max_sessions, 4);
        assert_eq!(c.idle_reclaim_secs, 30);
        // Reclamation is off by default (strict no-op).
        let d = parse_options(&args(&["--shared-engine"])).unwrap();
        assert_eq!(d.idle_reclaim_secs, 0);
        // Per-session sizing + KV budget default to off (today's behavior).
        assert_eq!(d.session_ctx_size, 0);
        assert_eq!(d.kv_budget_bytes, 0);
    }

    #[test]
    fn parses_session_ctx_size_and_kv_budget() {
        let c = parse_options(&args(&[
            "--shared-engine",
            "--session-ctx-size",
            "2048",
            "--kv-budget-bytes",
            "1073741824",
        ]))
        .unwrap();
        assert_eq!(c.session_ctx_size, 2048);
        assert_eq!(c.kv_budget_bytes, 1_073_741_824);
    }

    #[test]
    fn provider_flag_takes_model_name_and_key() {
        let c = parse_options(&args(&[
            "--provider",
            "openai",
            "--model",
            "gpt-4o",
            "--base-url",
            "http://localhost:8000/v1",
            "--api-key",
            "sk-test",
        ]))
        .unwrap();
        assert_eq!(c.provider, Some(ProviderSelector::OpenAi));
        // `--model NAME` is the provider model, not a local GGUF path.
        assert_eq!(c.provider_model.as_deref(), Some("gpt-4o"));
        assert!(c.model_path.is_none());
        assert_eq!(
            c.provider_base_url.as_deref(),
            Some("http://localhost:8000/v1")
        );
        assert_eq!(c.provider_api_key.as_deref(), Some("sk-test"));
    }

    #[test]
    fn provider_requires_model_and_is_exclusive() {
        // Missing --model.
        assert!(parse_options(&args(&["--provider", "openai", "--api-key", "k"])).is_err());
        // Mutually exclusive with --remote.
        let err = parse_options(&args(&[
            "--provider",
            "openai",
            "--model",
            "m",
            "--api-key",
            "k",
            "--remote",
            "https://box:8080",
        ]))
        .unwrap_err();
        assert!(err.contains("cannot be combined"), "got: {err}");
        // Invalid provider name.
        assert!(parse_options(&args(&["--provider", "bogus"])).is_err());
    }

    #[test]
    fn parses_ui_remote_in_both_forms() {
        assert_eq!(parse_options(&args(&[])).unwrap().ui_remote, None);
        assert_eq!(
            parse_options(&args(&["--ui-remote"])).unwrap().ui_remote,
            Some(0)
        );
        assert_eq!(
            parse_options(&args(&["--ui-remote=4321"]))
                .unwrap()
                .ui_remote,
            Some(4321)
        );
        assert!(parse_options(&args(&["--ui-remote=nope"])).is_err());
    }

    #[test]
    fn space_separated_ui_remote_port_is_rejected_not_silently_ignored() {
        // `--ui-remote 7777` reads as "pin port 7777" but would otherwise
        // bind an ephemeral port and leave 7777 as a stray argument.
        let err = parse_options(&args(&["--ui-remote", "7777"])).unwrap_err();
        assert!(err.contains("--ui-remote=7777"), "{err}");
        // A non-numeric follower is someone else's argument, not a port.
        assert_eq!(
            parse_options(&args(&["--ui-remote", "--non-interactive"]))
                .unwrap()
                .ui_remote,
            Some(0)
        );
    }

    #[test]
    fn parses_common_flags() {
        let c = parse_options(&args(&[
            "-p",
            "hi",
            "--non-interactive",
            "-sys",
            "sys",
            "--trace",
            "/tmp/t.log",
            "-c",
            "4096",
            "-n",
            "128",
            "--temp",
            "0.7",
            "--top-p",
            "0.9",
            "--min-p",
            "0.05",
            "--seed",
            "42",
            "--nothink",
            "--chdir",
            "/tmp",
        ]))
        .unwrap();
        assert_eq!(c.prompt.as_deref(), Some("hi"));
        assert!(c.non_interactive);
        assert_eq!(c.system, "sys");
        assert_eq!(c.trace_path, Some(PathBuf::from("/tmp/t.log")));
        assert_eq!(c.generation.ctx_size, 4096);
        assert_eq!(c.generation.n_predict, 128);
        assert!((c.generation.temperature - 0.7).abs() < 1e-6);
        assert!((c.generation.top_p - 0.9).abs() < 1e-6);
        assert!((c.generation.min_p - 0.05).abs() < 1e-6);
        assert_eq!(c.generation.seed, 42);
        assert_eq!(c.generation.think_mode, ThinkMode::Off);
        assert_eq!(c.chdir_path, Some(PathBuf::from("/tmp")));
    }

    #[test]
    fn btw_suspend_on_by_default_and_disable_opts_out() {
        // Default is on.
        assert!(parse_options(&args(&[])).unwrap().btw.suspend);
        // Explicit --btw-suspend keeps it on (accepted for compatibility).
        assert!(
            parse_options(&args(&["--btw-suspend"]))
                .unwrap()
                .btw
                .suspend
        );
        // --disable-btw-suspend opts out.
        assert!(
            !parse_options(&args(&["--disable-btw-suspend"]))
                .unwrap()
                .btw
                .suspend
        );
    }

    #[test]
    fn think_flags() {
        assert_eq!(
            parse_options(&args(&["--think"]))
                .unwrap()
                .generation
                .think_mode,
            ThinkMode::Medium
        );
        // `--think-max` selects the max level now, not the ordinary one.
        assert_eq!(
            parse_options(&args(&["--think-max"]))
                .unwrap()
                .generation
                .think_mode,
            ThinkMode::Max
        );
        assert_eq!(
            parse_options(&args(&["--nothink"]))
                .unwrap()
                .generation
                .think_mode,
            ThinkMode::Off
        );
        // `--think-low` has no context floor: its preamble asks for *less*
        // reasoning, so there is no budget to fall short of.
        assert_eq!(
            parse_options(&args(&["--think-low"]))
                .unwrap()
                .generation
                .think_mode,
            ThinkMode::Low
        );
    }

    // The max level's context floor is enforced at launch too, not only by
    // `/think max` — and the flag order must not matter.
    #[test]
    fn think_max_requires_enough_context() {
        // The default context is well above the floor, so the flag alone works.
        assert!(parse_options(&args(&["--think-max"])).is_ok());
        for flags in [
            vec!["--think-max", "--ctx", "8192"],
            vec!["--ctx", "8192", "--think-max"],
        ] {
            let err = parse_options(&args(&flags)).unwrap_err();
            assert!(err.contains("--think-max"), "got: {err}");
            assert!(
                err.contains(&crate::engine::THINK_MAX_MIN_CONTEXT.to_string()),
                "the error must name the context it needs; got: {err}"
            );
        }
        // The floor exactly is enough.
        assert!(
            parse_options(&args(&[
                "--think-max",
                "--ctx",
                &crate::engine::THINK_MAX_MIN_CONTEXT.to_string(),
            ]))
            .is_ok()
        );
    }

    #[test]
    fn minimal_prompt_flag() {
        let c = parse_options(&args(&["--minimal-prompt"])).unwrap();
        assert!(c.minimal_prompt);
        // Off unless asked: every other run must keep MCP, skills and context.
        let d = parse_options(&args(&[])).unwrap();
        assert!(!d.minimal_prompt);
        // Composes with the flags a benchmark actually uses.
        let e = parse_options(&args(&[
            "--minimal-prompt",
            "--non-interactive",
            "--provider",
            "openai",
            "--model",
            "m",
        ]))
        .unwrap();
        assert!(e.minimal_prompt && e.non_interactive);
    }

    #[test]
    fn version_flag() {
        for a in ["--version", "-V"] {
            let c = parse_options(&args(&[a])).unwrap();
            assert!(c.show_version, "{a}");
            assert!(!c.show_help, "{a}");
        }
        assert!(!parse_options(&args(&[])).unwrap().show_version);
    }

    #[test]
    fn help_flag_and_topic() {
        let c = parse_options(&args(&["--help", "sampling"])).unwrap();
        assert!(c.show_help);
        assert_eq!(c.help_topic.as_deref(), Some("sampling"));
        let c = parse_options(&args(&["-h"])).unwrap();
        assert!(c.show_help);
        assert!(c.help_topic.is_none());
    }

    #[test]
    fn missing_value_errors() {
        let err = parse_options(&args(&["--prompt"])).unwrap_err();
        assert!(err.contains("missing value for --prompt"));
        let err = parse_options(&args(&["--seed"])).unwrap_err();
        assert!(err.contains("--seed"));
    }

    #[test]
    fn invalid_values_error_with_option_name() {
        let err = parse_options(&args(&["-c", "zero"])).unwrap_err();
        assert!(err.contains("invalid value for -c"));
        let err = parse_options(&args(&["-n", "-3"])).unwrap_err();
        assert!(err.contains("invalid value for -n"));
        let err = parse_options(&args(&["--top-p", "1.5"])).unwrap_err();
        assert!(err.contains("invalid value for --top-p"));
        let err = parse_options(&args(&["--seed", "0"])).unwrap_err();
        assert!(err.contains("invalid value for --seed"));
        let err = parse_options(&args(&["--temp", "nan"])).unwrap_err();
        assert!(err.contains("invalid value for --temp"));
    }

    #[test]
    fn dspark_flags_select_the_runtime() {
        let c = parse_options(&args(&["--dspark"])).unwrap();
        assert!(c.engine.dspark);
        assert!(!c.engine.dspark_strict);
        // Left unset, so the engine keeps its own default rather than 0.
        assert_eq!(c.engine.dspark_confidence, None);

        // Either of the other two flags implies --dspark, like the C CLI.
        let c = parse_options(&args(&["--dspark-strict"])).unwrap();
        assert!(c.engine.dspark);
        assert!(c.engine.dspark_strict);

        let c = parse_options(&args(&["--dspark-confidence", "0.35"])).unwrap();
        assert!(c.engine.dspark);
        assert!((c.engine.dspark_confidence.unwrap() - 0.35).abs() < 1e-6);
    }

    #[test]
    fn dspark_confidence_zero_is_set_not_absent() {
        // 0 means "fixed draft length", which is why the C carries a separate
        // `_set` bool; it must not be confused with the flag being omitted.
        let c = parse_options(&args(&["--dspark-confidence", "0"])).unwrap();
        assert_eq!(c.engine.dspark_confidence, Some(0.0));
    }

    #[test]
    fn dspark_confidence_is_bounded_to_a_probability() {
        let err = parse_options(&args(&["--dspark-confidence", "1.5"])).unwrap_err();
        assert!(err.contains("--dspark-confidence"));
    }

    #[test]
    fn dspark_defaults_the_temperature_to_zero() {
        // DSpark is on by default, so a bare run samples argmax.
        let c = parse_options(&args(&[])).unwrap();
        assert!((c.generation.temperature - 0.0).abs() < 1e-6);

        // The implying flags carry the same default.
        let c = parse_options(&args(&["--dspark-strict"])).unwrap();
        assert!((c.generation.temperature - 0.0).abs() < 1e-6);
        let c = parse_options(&args(&["--dspark-confidence", "0.3"])).unwrap();
        assert!((c.generation.temperature - 0.0).abs() < 1e-6);
    }

    #[test]
    fn dspark_off_keeps_the_sampling_default() {
        // --dspark-off turns speculation off, so the 0.6 temperature default
        // stays in force rather than being forced to 0.
        let c = parse_options(&args(&["--dspark-off"])).unwrap();
        assert!(!c.engine.dspark);
        assert!((c.generation.temperature - 0.6).abs() < 1e-6);
    }

    #[test]
    fn an_explicit_temp_beats_the_dspark_default_in_either_order() {
        let c = parse_options(&args(&["--dspark", "--temp", "0.9"])).unwrap();
        assert!((c.generation.temperature - 0.9).abs() < 1e-6);
        let c = parse_options(&args(&["--temp", "0.9", "--dspark"])).unwrap();
        assert!((c.generation.temperature - 0.9).abs() < 1e-6);
        // --dspark-off with an explicit --temp also keeps the explicit value.
        let c = parse_options(&args(&["--dspark-off", "--temp", "0.9"])).unwrap();
        assert!((c.generation.temperature - 0.9).abs() < 1e-6);
    }

    #[test]
    fn dspark_is_on_by_default() {
        let c = parse_options(&args(&[])).unwrap();
        assert!(c.engine.dspark);
        assert!(!c.engine.dspark_strict);
        assert_eq!(c.engine.dspark_confidence, None);
    }

    #[test]
    fn unknown_option_errors() {
        let err = parse_options(&args(&["--bogus"])).unwrap_err();
        assert!(err.contains("unknown option: --bogus"));
    }

    #[test]
    fn engine_tuning_flags() {
        let c = parse_options(&args(&[
            "--mtp",
            "draft.gguf",
            "--mtp-draft",
            "2",
            "--mtp-margin",
            "5.5",
            "--quality",
            "--warm-weights",
            "--ssd-streaming",
            "--ssd-streaming-cold",
            "--ssd-streaming-preload-experts",
            "8",
            "--simulate-used-memory",
            "64GB",
        ]))
        .unwrap();
        assert_eq!(c.engine.mtp_path, Some(PathBuf::from("draft.gguf")));
        assert_eq!(c.engine.mtp_draft_tokens, 2);
        assert!((c.engine.mtp_margin - 5.5).abs() < 1e-6);
        assert_eq!(c.engine.prefill_chunk, DEFAULT_PREFILL_CHUNK);
        assert!(c.engine.quality);
        assert!(c.engine.warm_weights);
        assert!(c.engine.ssd_streaming);
        assert!(c.engine.ssd_streaming_cold);
        assert_eq!(c.engine.ssd_streaming_preload_experts, 8);
        assert_eq!(c.engine.simulate_used_memory_bytes, 64 << 30);
    }

    #[test]
    fn engine_tuning_defaults_mirror_c() {
        let c = parse_options(&[]).unwrap();
        assert_eq!(c.engine.mtp_draft_tokens, 1);
        assert!((c.engine.mtp_margin - 3.0).abs() < 1e-6);
        assert_eq!(c.engine, EngineTuning::default());
    }

    #[test]
    fn backend_by_name() {
        for (name, want) in [
            ("metal", Backend::Metal),
            ("cuda", Backend::Cuda),
            ("cpu", Backend::Cpu),
        ] {
            let c = parse_options(&args(&["--backend", name])).unwrap();
            assert_eq!(c.backend, Some(want));
        }
        let err = parse_options(&args(&["--backend", "tpu"])).unwrap_err();
        assert!(err.contains("invalid backend: tpu"));
    }

    #[test]
    fn steering_file_defaults_ffn_scale() {
        let c = parse_options(&args(&["--dir-steering-file", "v.bin"])).unwrap();
        assert!((c.engine.dir_steering_ffn - 1.0).abs() < 1e-6);
        assert!((c.engine.dir_steering_attn - 0.0).abs() < 1e-6);
        // An explicit scale suppresses the 1.0 default.
        let c = parse_options(&args(&[
            "--dir-steering-file",
            "v.bin",
            "--dir-steering-attn",
            "0.5",
        ]))
        .unwrap();
        assert!((c.engine.dir_steering_ffn - 0.0).abs() < 1e-6);
        assert!((c.engine.dir_steering_attn - 0.5).abs() < 1e-6);
    }

    #[test]
    fn gib_and_cache_experts_args() {
        assert_eq!(parse_gib_arg("64"), Some(64 << 30));
        assert_eq!(parse_gib_arg("64GB"), Some(64 << 30));
        assert_eq!(parse_gib_arg("64gb"), Some(64 << 30));
        assert_eq!(parse_gib_arg("0"), None);
        assert_eq!(parse_gib_arg(""), None);
        assert_eq!(parse_gib_arg("GB"), None);
        assert_eq!(parse_gib_arg("6x4"), None);
        assert_eq!(parse_streaming_cache_experts_arg("16"), Some((16, 0)));
        assert_eq!(parse_streaming_cache_experts_arg("4GB"), Some((0, 4 << 30)));
        assert_eq!(parse_streaming_cache_experts_arg("0"), None);
        assert_eq!(parse_streaming_cache_experts_arg("x"), None);
        let err = parse_options(&args(&["--ssd-streaming-cache-experts", "no"])).unwrap_err();
        assert!(err.contains("positive count or <number>GB"));
        let err = parse_options(&args(&["--simulate-used-memory", "no"])).unwrap_err();
        assert!(err.contains("positive GiB value"));
    }

    #[test]
    fn remote_flags_and_mutual_exclusion() {
        let c = parse_options(&args(&[
            "--remote",
            "https://box:8080",
            "--remote-token",
            "s3cr",
        ]))
        .unwrap();
        assert_eq!(c.remote_url.as_deref(), Some("https://box:8080"));
        assert_eq!(c.remote_token.as_deref(), Some("s3cr"));
        // localhost http is allowed without --insecure.
        assert!(parse_options(&args(&["--remote", "http://localhost:9000"])).is_ok());
        // non-loopback http requires --insecure.
        let err = parse_options(&args(&["--remote", "http://box.example.com"])).unwrap_err();
        assert!(
            err.contains("--insecure") || err.contains("plaintext"),
            "{err}"
        );
        assert!(
            parse_options(&args(&["--remote", "http://box.example.com", "--insecure"])).is_ok()
        );
        // mutually exclusive with -m.
        let err = parse_options(&args(&["--remote", "https://box", "-m", "x.gguf"])).unwrap_err();
        assert!(err.contains("-m"), "{err}");
        // mutually exclusive with a backend selector.
        let err = parse_options(&args(&["--remote", "https://box", "--cpu"])).unwrap_err();
        assert!(err.contains("--metal"), "{err}");
    }

    #[test]
    fn power_percent() {
        assert_eq!(parse_power_percent("50"), Some(50));
        assert_eq!(parse_power_percent("1"), Some(1));
        assert_eq!(parse_power_percent("100"), Some(100));
        assert_eq!(parse_power_percent("0"), None);
        assert_eq!(parse_power_percent("101"), None);
        assert_eq!(parse_power_percent(""), None);
        assert_eq!(parse_power_percent("50x"), None);
    }

    #[test]
    fn slash_commands() {
        for cmd in [
            "/help",
            "/save",
            "/compact",
            "/list",
            "/quit",
            "/exit",
            "/new",
            "/clear",
            "/mcp",
            "/context",
            "/usage",
            "/init",
            "/skills",
            "/templates",
            "/tasks",
            "/agent",
            "/hooks",
            "/remote-control",
            "/rc",
            "/version",
        ] {
            assert!(slash_command_known(cmd), "{cmd}");
        }
        assert!(slash_command_known("/rc on"));
        assert!(slash_command_known("/rc off"));
        assert!(slash_command_known("/remote-control on"));
        assert!(!slash_command_known("/rcx"));
        assert!(slash_command_known("/btw what is this?"));
        assert!(!slash_command_known("/btwx"));
    }

    /// The `/` menu offers exactly what dispatch will accept: advertising a
    /// name dispatch does not know would send the line to the model as an
    /// ordinary prompt the moment the user picked it.
    #[test]
    fn every_menu_command_is_known_to_the_dispatcher() {
        for c in SLASH_COMMANDS {
            assert!(
                slash_command_known_with(c.name, true),
                "{} is offered by the menu but unknown to dispatch",
                c.name
            );
            assert!(c.name.starts_with('/'), "{}", c.name);
            assert!(!c.desc.is_empty(), "{} has no description", c.name);
        }
    }

    #[test]
    fn plugins_is_a_known_slash_command() {
        assert!(slash_command_known("/plugins"));
        assert!(slash_command_known("/install-claude-plugin"));
    }

    #[test]
    fn the_menu_catalog_has_no_duplicate_commands() {
        let mut names: Vec<&str> = SLASH_COMMANDS.iter().map(|c| c.name).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate entry in SLASH_COMMANDS");
    }

    /// The arcade easter eggs and `/renotify` are deliberately undiscoverable;
    /// the menu is the place that would leak them.
    #[test]
    fn hidden_commands_stay_out_of_the_menu() {
        for hidden in crate::arcade::Arcade::COMMANDS {
            assert!(
                !SLASH_COMMANDS.iter().any(|c| c.name == hidden),
                "{hidden} must stay hidden"
            );
        }
        assert!(!SLASH_COMMANDS.iter().any(|c| c.name == "/renotify"));
        assert!(slash_command_known("/subagent count the tests"));
        assert!(!slash_command_known("/subagentx"));
        // The `:<name>` form is known whatever the name — an unknown name is
        // reported by dispatch, not silently sent to the model as a prompt.
        assert!(slash_command_known("/subagent:reviewer count the tests"));
        assert!(slash_command_known("/subagent:nope count the tests"));
        // ...but the name must actually be there: a bare colon selects nothing.
        assert!(!slash_command_known("/subagent:"));
        assert!(slash_command_known("/power 50"));
        assert!(slash_command_known("/power"));
        assert!(slash_command_known("/switch 2"));
        assert!(slash_command_known("/del 1"));
        assert!(slash_command_known("/strip"));
        assert!(slash_command_known("/kvcache"));
        assert!(slash_command_known("/kvcache gc"));
        assert!(!slash_command_known("/kvcaches"));
        assert!(slash_command_known("/history 10"));
        assert!(slash_command_known("/repro"));
        assert!(slash_command_known("/repro looping bug"));
        assert!(!slash_command_known("/reprox"));
        assert!(slash_command_known("/export"));
        assert!(slash_command_known("/export html notes.html"));
        assert!(slash_command_known("/insights"));
        assert!(slash_command_known("/insights fast"));
        assert!(!slash_command_known("/insightsx"));
        assert!(slash_command_known("/open"));
        assert!(slash_command_known("/open src/ui.rs"));
        assert!(!slash_command_known("/opened"));
        assert!(!slash_command_known("/exports"));
        assert!(!slash_command_known("/powerful"));
        assert!(!slash_command_known("/unknown"));
        assert!(!slash_command_known("/helpme"));
        // `/compact` takes optional summarization instructions.
        assert!(slash_command_known("/compact"));
        assert!(slash_command_known("/compact focus on the parser bug"));
        assert!(!slash_command_known("/compacting"));
    }

    #[test]
    fn resume_cli_arg() {
        // `plank /resume deadbeef` selects a specific session.
        let c = parse_options(&args(&["/resume", "deadbeef"])).unwrap();
        assert_eq!(c.resume.as_deref(), Some("deadbeef"));
        // Bare `/resume` means "most recent" (empty prefix).
        let c = parse_options(&args(&["/resume"])).unwrap();
        assert_eq!(c.resume.as_deref(), Some(""));
        // A following flag is not consumed as the prefix.
        let c = parse_options(&args(&["/resume", "--nothink"])).unwrap();
        assert_eq!(c.resume.as_deref(), Some(""));
        assert_eq!(c.generation.think_mode, ThinkMode::Off);
        // Not given at all.
        assert!(parse_options(&args(&[])).unwrap().resume.is_none());
    }

    #[test]
    fn rate_is_a_known_slash_command_with_and_without_args() {
        assert!(slash_command_known("/rate"));
        assert!(slash_command_known("/rate + good"));
        assert!(slash_command_known("/rate - wrong answer"));
    }

    #[test]
    fn remote_slash_commands() {
        assert!(slash_command_known("/remote"));
        assert!(slash_command_known("/grant"));
        // `/grant 3` names the waiting session explicitly.
        assert!(slash_command_known("/grant 3"));
    }

    #[test]
    fn ordinary_commands_and_prompts_are_remote_safe() {
        for line in ["/help", "/context", "/compact focus on the parser", "hello"] {
            assert_eq!(slash_command_remote_refusal(line), None, "{line}");
        }
    }

    #[test]
    fn commands_needing_the_local_terminal_are_refused_remotely() {
        for line in [
            "/grant",
            "/grant 3",
            "/rc",
            "/rc off",
            "/remote-control",
            "/open",
            "/open src/ui.rs",
            "/quit",
            "/exit",
        ] {
            assert!(
                slash_command_remote_refusal(line).is_some(),
                "{line} should be refused"
            );
        }
    }

    /// A pane command is refused only in its interactive form; the same command
    /// with an argument does its work without a terminal and stays available.
    #[test]
    fn pane_commands_are_refused_bare_but_allowed_with_an_argument() {
        assert!(slash_command_remote_refusal("/kvcache").is_some());
        assert_eq!(slash_command_remote_refusal("/kvcache gc"), None);
        assert!(slash_command_remote_refusal("/resume").is_some());
        assert_eq!(slash_command_remote_refusal("/resume 3"), None);
    }

    /// Surrounding whitespace must not smuggle a refused command past the gate.
    #[test]
    fn the_remote_gate_ignores_surrounding_whitespace() {
        assert!(slash_command_remote_refusal("  /quit  ").is_some());
        assert!(slash_command_remote_refusal("/kvcache   ").is_some());
    }

    #[test]
    fn slash_command_with_args_boundaries() {
        assert!(slash_command_with_args("/power", "/power"));
        assert!(slash_command_with_args("/power 10", "/power"));
        assert!(!slash_command_with_args("/powerx", "/power"));
    }
}
