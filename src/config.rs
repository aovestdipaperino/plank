//! Agent configuration and command-line parsing.
//!
//! Ports the "Small Utilities And Command-Line Parsing" section of the C
//! reference (`ds4-ref/ds4_agent.c`): the `agent_config` struct, option
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
pub struct AgentConfig {
    /// Sampling and length options for generation.
    pub generation: GenerationOptions,
    /// One-shot prompt supplied with `-p`/`--prompt`.
    pub prompt: Option<String>,
    /// System prompt; defaults to [`DEFAULT_SYSTEM_PROMPT`].
    pub system: String,
    /// Trace log path supplied with `--trace`.
    pub trace_path: Option<PathBuf>,
    /// Working directory supplied with `--chdir`.
    pub chdir_path: Option<PathBuf>,
    /// MCP server config supplied with `--mcp-config`; `None` = `./.mcp.json`.
    pub mcp_config_path: Option<PathBuf>,
    /// True when `--non-interactive` was given.
    pub non_interactive: bool,
    /// True when `-h`/`--help` was given; caller should print [`usage`] and exit.
    pub show_help: bool,
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
    /// Native-engine tuning knobs (MTP, SSD streaming, steering, ...).
    pub engine: EngineTuning,
}

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
    /// Prefill chunk size from `--prefill-chunk`; 0 = engine default.
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
            prefill_chunk: 0,
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
                think_mode: ThinkMode::On,
                ..GenerationOptions::default()
            },
            prompt: None,
            system: DEFAULT_SYSTEM_PROMPT.to_owned(),
            trace_path: None,
            chdir_path: None,
            mcp_config_path: None,
            non_interactive: false,
            show_help: false,
            help_topic: None,
            model_path: None,
            backend: None,
            n_threads: 0,
            power_percent: 0,
            engine: EngineTuning::default(),
        }
    }
}

/// Returns the usage help text, close to the C agent's `-h` output.
#[must_use]
pub fn usage() -> String {
    "\
Usage: plank [options]

Options:
  -h, --help [topic]       show this help and exit
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
      --prefill-chunk N    prefill chunk size in tokens (engine default when unset)
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
  -p, --prompt TEXT        run one prompt and exit after the reply
      --non-interactive    disable the interactive UI
  -sys, --system TEXT      override the system prompt
      --trace PATH         append a trace log to PATH
  -c, --ctx N              context window size in tokens (default 1048576)
  -n, --tokens N           maximum tokens to generate (default 50000)
      --temp F             sampling temperature (0..100)
      --top-p F            nucleus sampling threshold (0..1)
      --min-p F            minimum-probability threshold (0..1)
      --seed N             RNG seed (positive integer)
      --think              enable thinking (default)
      --think-max          enable maximum thinking effort
      --nothink            disable thinking
      --chdir PATH         change working directory before starting
      --mcp-config FILE    local MCP server config (default: ./.mcp.json);
                           overlays the global ~/.plank/.mcp.json by name
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

/// True when `cmd` is `name` alone or `name` followed by whitespace.
#[must_use]
pub fn slash_command_with_args(cmd: &str, name: &str) -> bool {
    cmd.strip_prefix(name)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with(char::is_whitespace))
}

/// True when `cmd` is one of the agent's known slash commands.
#[must_use]
pub fn slash_command_known(cmd: &str) -> bool {
    matches!(
        cmd,
        "/help"
            | "/save"
            | "/compact"
            | "/list"
            | "/quit"
            | "/exit"
            | "/new"
            | "/clear"
            | "/mcp"
            | "/context"
            | "/init"
            | "/skills"
            | "/hooks"
    ) || slash_command_with_args(cmd, "/btw")
        || slash_command_with_args(cmd, "/power")
        || slash_command_with_args(cmd, "/switch")
        || slash_command_with_args(cmd, "/del")
        || slash_command_with_args(cmd, "/strip")
        || slash_command_with_args(cmd, "/history")
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
        "--prefill-chunk" => {
            e.prefill_chunk = u32::try_from(parse_int(v, arg)?).unwrap_or(0);
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
pub fn parse_options(args: &[String]) -> Result<AgentConfig, String> {
    let mut c = AgentConfig::default();
    // Tracks whether a steering scale was given explicitly; a steering file
    // without one defaults the FFN scale to 1.0, like the C.
    let mut steering_scale_set = false;
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
            "-h" | "--help" => {
                c.show_help = true;
                if let Some(topic) = args.get(i + 1)
                    && !topic.starts_with('-')
                {
                    c.help_topic = Some(topic.clone());
                }
                return Ok(c);
            }
            "-m" | "--model" => c.model_path = Some(PathBuf::from(need_arg(&mut i)?)),
            "-t" | "--threads" => c.n_threads = parse_int(need_arg(&mut i)?, arg)?,
            "--backend" => {
                c.backend = Some(match need_arg(&mut i)? {
                    "metal" => Backend::Metal,
                    "cuda" => Backend::Cuda,
                    "cpu" => Backend::Cpu,
                    other => return Err(format!("invalid backend: {other}")),
                });
            }
            "--metal" => c.backend = Some(Backend::Metal),
            "--cuda" => c.backend = Some(Backend::Cuda),
            "--cpu" => c.backend = Some(Backend::Cpu),
            "--power" => {
                let v = need_arg(&mut i)?;
                c.power_percent = parse_power_percent(v)
                    .ok_or_else(|| format!("invalid value for {arg}: {v}"))?;
            }
            "-p" | "--prompt" => c.prompt = Some(need_arg(&mut i)?.to_owned()),
            "--non-interactive" => c.non_interactive = true,
            "-sys" | "--system" => need_arg(&mut i)?.clone_into(&mut c.system),
            "--trace" => c.trace_path = Some(PathBuf::from(need_arg(&mut i)?)),
            "-c" | "--ctx" => c.generation.ctx_size = parse_int(need_arg(&mut i)?, arg)?,
            "-n" | "--tokens" => c.generation.n_predict = parse_int(need_arg(&mut i)?, arg)?,
            "--temp" => {
                c.generation.temperature = parse_float_range(need_arg(&mut i)?, arg, 0.0, 100.0)?;
            }
            "--top-p" => c.generation.top_p = parse_float_range(need_arg(&mut i)?, arg, 0.0, 1.0)?,
            "--min-p" => c.generation.min_p = parse_float_range(need_arg(&mut i)?, arg, 0.0, 1.0)?,
            "--seed" => c.generation.seed = parse_u64(need_arg(&mut i)?, arg)?,
            "--think" | "--think-max" => c.generation.think_mode = ThinkMode::On,
            "--nothink" => c.generation.think_mode = ThinkMode::Off,
            "--chdir" => c.chdir_path = Some(PathBuf::from(need_arg(&mut i)?)),
            "--mcp-config" => c.mcp_config_path = Some(PathBuf::from(need_arg(&mut i)?)),
            "--quality" => c.engine.quality = true,
            "--warm-weights" => c.engine.warm_weights = true,
            "--ssd-streaming" => c.engine.ssd_streaming = true,
            "--ssd-streaming-cold" => c.engine.ssd_streaming_cold = true,
            "--mtp"
            | "--mtp-draft"
            | "--mtp-margin"
            | "--prefill-chunk"
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
    if c.engine.dir_steering_file.is_some() && !steering_scale_set {
        c.engine.dir_steering_ffn = 1.0;
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn defaults() {
        let c = parse_options(&[]).unwrap();
        assert_eq!(c.generation.n_predict, DEFAULT_N_PREDICT);
        assert_eq!(c.generation.ctx_size, DEFAULT_CTX_SIZE);
        assert_eq!(c.system, DEFAULT_SYSTEM_PROMPT);
        assert_eq!(c.generation.think_mode, ThinkMode::On);
        assert!(c.prompt.is_none());
        assert!(!c.non_interactive);
        assert!(!c.show_help);
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
    fn think_flags() {
        assert_eq!(
            parse_options(&args(&["--think"]))
                .unwrap()
                .generation
                .think_mode,
            ThinkMode::On
        );
        assert_eq!(
            parse_options(&args(&["--think-max"]))
                .unwrap()
                .generation
                .think_mode,
            ThinkMode::On
        );
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
            "--prefill-chunk",
            "512",
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
        assert_eq!(c.engine.prefill_chunk, 512);
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
            "/help", "/save", "/compact", "/list", "/quit", "/exit", "/new", "/clear", "/mcp",
            "/context", "/init", "/skills", "/hooks",
        ] {
            assert!(slash_command_known(cmd), "{cmd}");
        }
        assert!(slash_command_known("/btw what is this?"));
        assert!(slash_command_known("/power 50"));
        assert!(slash_command_known("/power"));
        assert!(slash_command_known("/switch 2"));
        assert!(slash_command_known("/del 1"));
        assert!(slash_command_known("/strip"));
        assert!(slash_command_known("/history 10"));
        assert!(!slash_command_known("/powerful"));
        assert!(!slash_command_known("/unknown"));
        assert!(!slash_command_known("/helpme"));
    }

    #[test]
    fn slash_command_with_args_boundaries() {
        assert!(slash_command_with_args("/power", "/power"));
        assert!(slash_command_with_args("/power 10", "/power"));
        assert!(!slash_command_with_args("/powerx", "/power"));
    }
}
