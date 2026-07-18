//! Agent configuration and command-line parsing.
//!
//! Ports the "Small Utilities And Command-Line Parsing" section of the C
//! reference (`ds4-ref/ds4_agent.c`): the `agent_config` struct, option
//! parsing, numeric parsing helpers, and slash-command recognition. Unlike
//! the C code, parse failures return `Err` instead of exiting the process.
//! Engine-backend options (model path, backend, threads, MTP, SSD streaming,
//! distributed mode) are not ported yet because plank has no native engine.

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
            non_interactive: false,
            show_help: false,
            help_topic: None,
            model_path: None,
            backend: None,
            n_threads: 0,
            power_percent: 0,
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
      --metal              use the Metal backend
      --cuda               use the CUDA backend
      --cpu                use the CPU backend
      --power N            GPU power cap percent (1..100)
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
        "/help" | "/save" | "/compact" | "/list" | "/quit" | "/exit" | "/new"
    ) || slash_command_with_args(cmd, "/power")
        || slash_command_with_args(cmd, "/switch")
        || slash_command_with_args(cmd, "/del")
        || slash_command_with_args(cmd, "/strip")
        || slash_command_with_args(cmd, "/history")
}

/// Parses command-line arguments (without the program name) into a config.
///
/// # Errors
/// Returns an error naming the offending option when a value is missing,
/// out of range, or an option is unknown.
pub fn parse_options(args: &[String]) -> Result<AgentConfig, String> {
    let mut c = AgentConfig::default();
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
            _ => return Err(format!("unknown option: {arg}")),
        }
        i += 1;
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
            "/help", "/save", "/compact", "/list", "/quit", "/exit", "/new",
        ] {
            assert!(slash_command_known(cmd), "{cmd}");
        }
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
