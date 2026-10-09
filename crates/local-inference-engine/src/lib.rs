// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Local GGUF inference behind one API, over two backends:
//!
//! - **ds4**: the ds4 C engine (`DeepSeek` V4 and V4.1 Flash, Qwen3.8-Flash-Next,
//!   GLM), built from source and linked statically. Metal only, so macOS with
//!   the C sources present; [`DS4_AVAILABLE`] says whether this build has it.
//! - **Gemma**: Gemma 4 through `crates/gemma-engine` (candle; Metal on macOS,
//!   the CPU elsewhere), behind the default `gemma` feature.
//!
//! [`Model::open`] reads the GGUF's `general.architecture` and picks the
//! backend: `gemma4` goes to Gemma, everything else to ds4, which is also the
//! C's own fallthrough for an architecture it does not name. [`Model`] and
//! [`Session`] then work the same for both: render a one-turn chat, prefill
//! it reusing the cached prefix, evaluate, sample, read log-probabilities.
//!
//! Below that:
//!
//! - [`ffi`]: the ds4 C engine's raw `extern "C"` declarations and
//!   `#[repr(C)]` mirrors. plank's `Ds4Engine` drives these directly.
//! - [`metal`]: finding the Metal kernel sources the C engine compiles at open.
//! - [`ds4`]: the ds4 backend's own safe types, for ds4-only capabilities.
//! - `gemma_engine`: the Gemma crate itself, re-exported.
//!
//! A backend that is not built makes [`Model::open`] fail for the models it
//! would serve, never the build, so a dependent needs no `cfg` of its own.
//!
//! # Activation dumps (ds4)
//!
//! The C writes per-layer activations to `<prefix>_<name>-<layer>_pos<pos>.bin`
//! when `DS4_METAL_GRAPH_DUMP_PREFIX` (and optionally `_NAME`, `_LAYER`, `_POS`)
//! is set. It reads those variables **once per process**, at the first dump
//! check, so set them before the first [`Session::sync`] and keep the prefix
//! fixed; each capture then overwrites the previous one's files. `pos` is the
//! position the prefill chunk starts at, so call [`Session::invalidate`] before
//! each prompt — otherwise a shared prefix (the system prompt) is reused, the
//! prefill starts past 0, and nothing is written under `pos0`. Size the
//! [`Options::prefill_chunk`] to hold the whole prompt so each layer gets one
//! file whose last row is the last prompt token. Gemma has no dump hook.
//!
//! ```no_run
//! use std::sync::Arc;
//! use local_inference_engine::{Model, Options, Session, Think};
//!
//! let model = Arc::new(Model::open(&Options::new("gemma4-e4b.gguf").ctx_size(2048))?);
//! let mut session = Session::new(&model, 2048)?;
//! session.sync(&model.encode_chat("Answer briefly.", "Why is the sky blue?", Think::Off)?)?;
//! let mut rng = 42;
//! let mut reply = Vec::new();
//! for _ in 0..64 {
//!     let token = session.sample(0.0, 0, 1.0, 0.0, &mut rng);
//!     if token == model.eos() {
//!         break;
//!     }
//!     reply.extend(model.token_bytes(token));
//!     session.eval(token)?;
//! }
//! println!("{}", String::from_utf8_lossy(&reply));
//! # Ok::<(), local_inference_engine::Error>(())
//! ```

pub mod ds4;
pub mod ffi;
#[cfg(feature = "gemma")]
mod gemma;
pub mod gguf;
pub mod metal;

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub use ffi::{Ds4Backend as Backend, Ds4TokenScore as TokenScore};
#[cfg(feature = "gemma")]
pub use gemma_engine;

/// Whether this build linked the ds4 C engine.
pub const DS4_AVAILABLE: bool = cfg!(ds4_engine);

/// Whether this build carries the Gemma backend.
pub const GEMMA_AVAILABLE: bool = cfg!(feature = "gemma");

/// A failure reported by the engine, or the engine being absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(String);

impl Error {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// What [`Model::open`] loads and how.
///
/// The companion, steering, power and kernel knobs apply to the ds4 backend
/// only; Gemma reads the model path, the context size and [`Options::cpu`].
/// Every ds4 knob not listed here is left at the C's zero value, which is its
/// default: single machine, no distributed split, no tensor parallelism.
#[derive(Debug, Clone)]
pub struct Options {
    model: PathBuf,
    mtp: Option<PathBuf>,
    vision: Option<PathBuf>,
    backend: Backend,
    threads: i32,
    ctx_size: i32,
    prefill_chunk: u32,
    power_percent: i32,
    steering_file: Option<PathBuf>,
    steering_attn: f32,
    steering_ffn: f32,
    warm_weights: bool,
    quality: bool,
    cpu: bool,
}

impl Options {
    /// Options for the GGUF at `model` on Metal, with a 4096-token context
    /// and every other setting at the engine's default.
    #[must_use]
    pub fn new(model: impl AsRef<Path>) -> Self {
        Self {
            model: model.as_ref().to_path_buf(),
            mtp: None,
            vision: None,
            backend: Backend::Metal,
            threads: 0,
            ctx_size: 4096,
            prefill_chunk: 0,
            power_percent: 0,
            steering_file: None,
            steering_attn: 0.0,
            steering_ffn: 0.0,
            warm_weights: false,
            quality: false,
            cpu: false,
        }
    }

    /// The model file these options load.
    #[must_use]
    pub fn model(&self) -> &Path {
        &self.model
    }

    /// Sets the largest context a session on this model may use (clamped to
    /// the model's own for Gemma).
    #[must_use]
    pub fn ctx_size(mut self, ctx_size: i32) -> Self {
        self.ctx_size = ctx_size;
        self
    }

    /// Sets how many prompt tokens one prefill step processes; 0 lets the
    /// engine choose. A prompt that fits one chunk is prefilled in one graph
    /// pass, which is what keeps activation dumps to a single file per layer.
    #[must_use]
    pub fn prefill_chunk(mut self, tokens: u32) -> Self {
        self.prefill_chunk = tokens;
        self
    }

    /// Selects the ds4 engine's compute backend; Metal is the only one built.
    #[must_use]
    pub fn backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Sets the CPU thread count; 0 lets the engine choose.
    #[must_use]
    pub fn threads(mut self, threads: i32) -> Self {
        self.threads = threads;
        self
    }

    /// Caps GPU power, as a percentage in `1..=100`; 0 (the default) runs
    /// at full power.
    #[must_use]
    pub fn power_percent(mut self, percent: i32) -> Self {
        self.power_percent = percent;
        self
    }

    /// Loads a drafter (`DSpark` or MTP) companion GGUF.
    #[must_use]
    pub fn mtp(mut self, path: impl AsRef<Path>) -> Self {
        self.mtp = Some(path.as_ref().to_path_buf());
        self
    }

    /// Loads a vision-encoder companion GGUF.
    #[must_use]
    pub fn vision(mut self, path: impl AsRef<Path>) -> Self {
        self.vision = Some(path.as_ref().to_path_buf());
        self
    }

    /// Loads a directional-steering vector applied at `attn` and `ffn` scale.
    #[must_use]
    pub fn steering(mut self, file: impl AsRef<Path>, attn: f32, ffn: f32) -> Self {
        self.steering_file = Some(file.as_ref().to_path_buf());
        self.steering_attn = attn;
        self.steering_ffn = ffn;
        self
    }

    /// Touches every weight page at open instead of on first use.
    #[must_use]
    pub fn warm_weights(mut self, on: bool) -> Self {
        self.warm_weights = on;
        self
    }

    /// Runs Gemma on the CPU even where Metal is available (tests, or a
    /// machine whose GPU is busy). The ds4 engine has no CPU path here.
    #[must_use]
    pub fn cpu(mut self, on: bool) -> Self {
        self.cpu = on;
        self
    }

    /// Prefers exact kernels over the fast approximate ones.
    #[must_use]
    pub fn quality(mut self, on: bool) -> Self {
        self.quality = on;
        self
    }
}

/// How much the model reasons before answering, for [`Model::encode_chat`].
///
/// Gemma's template has one switch, so [`Think::High`] and [`Think::Max`]
/// both turn its `<|think|>` on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Think {
    /// Answer directly.
    #[default]
    Off,
    /// The ds4 `--think` mode.
    High,
    /// The ds4 `--think-max` mode; the engine falls back to `High` when the
    /// context is too small for it.
    Max,
}

impl Think {
    /// The ds4 engine's value for this mode.
    #[must_use]
    pub fn ds4(self) -> ffi::Ds4ThinkMode {
        match self {
            Self::Off => ffi::Ds4ThinkMode::NONE,
            Self::High => ffi::Ds4ThinkMode::HIGH,
            Self::Max => ffi::Ds4ThinkMode::MAX,
        }
    }
}

/// Which backend serves a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// The ds4 C engine.
    Ds4,
    /// Gemma 4 on candle.
    Gemma,
}

impl Family {
    /// The backend for the GGUF at `path`: [`Family::Gemma`] for a `gemma4`
    /// architecture, [`Family::Ds4`] for anything else, unreadable files
    /// included (the C's own fallthrough; it then reports the real problem).
    #[must_use]
    pub fn of(path: &Path) -> Self {
        match gguf::architecture(path).as_deref() {
            Some(gguf::GEMMA4_ARCH) => Self::Gemma,
            _ => Self::Ds4,
        }
    }
}

/// A loaded model of either family: the weights every [`Session`] on it
/// shares.
#[derive(Debug)]
pub struct Model {
    inner: ModelInner,
}

#[derive(Debug)]
enum ModelInner {
    Ds4(Arc<ds4::Model>),
    #[cfg(feature = "gemma")]
    Gemma {
        model: Arc<gemma_engine::model::Model>,
        ctx: usize,
    },
}

impl Model {
    /// Opens the model `options` names on the backend its architecture
    /// calls for.
    ///
    /// For ds4, the C exits the process outright, rather than returning, when
    /// the model file is missing or another process holds `/tmp/ds4.lock`.
    ///
    /// # Errors
    /// Fails when the backend the model needs is not in this build, or the
    /// backend refuses the model.
    pub fn open(options: &Options) -> Result<Self, Error> {
        let inner = match Family::of(&options.model) {
            Family::Ds4 => ModelInner::Ds4(Arc::new(ds4::Model::open(options)?)),
            #[cfg(feature = "gemma")]
            Family::Gemma => {
                let ctx = usize::try_from(options.ctx_size).unwrap_or(1).max(1);
                let model = gemma::open(&options.model, ctx, options.cpu)?;
                let ctx = model.context_length();
                ModelInner::Gemma { model, ctx }
            }
            #[cfg(not(feature = "gemma"))]
            Family::Gemma => {
                return Err(Error::new(format!(
                    "{} is a Gemma model; this build has no Gemma backend (enable the `gemma` feature)",
                    options.model.display()
                )));
            }
        };
        Ok(Self { inner })
    }

    /// The backend serving this model.
    #[must_use]
    pub fn family(&self) -> Family {
        match &self.inner {
            ModelInner::Ds4(_) => Family::Ds4,
            #[cfg(feature = "gemma")]
            ModelInner::Gemma { .. } => Family::Gemma,
        }
    }

    /// The ds4 backend's model, when ds4 serves this one.
    #[must_use]
    pub fn as_ds4(&self) -> Option<&ds4::Model> {
        match &self.inner {
            ModelInner::Ds4(m) => Some(m),
            #[cfg(feature = "gemma")]
            ModelInner::Gemma { .. } => None,
        }
    }

    /// The Gemma model, when Gemma serves this one.
    #[cfg(feature = "gemma")]
    #[must_use]
    pub fn as_gemma(&self) -> Option<&Arc<gemma_engine::model::Model>> {
        match &self.inner {
            ModelInner::Gemma { model, .. } => Some(model),
            ModelInner::Ds4(_) => None,
        }
    }

    /// The largest context a session on this model may use.
    #[must_use]
    pub fn ctx_size(&self) -> i32 {
        match &self.inner {
            ModelInner::Ds4(m) => m.ctx_size(),
            #[cfg(feature = "gemma")]
            ModelInner::Gemma { ctx, .. } => i32::try_from(*ctx).unwrap_or(i32::MAX),
        }
    }

    /// The model name the backend reports.
    #[must_use]
    pub fn name(&self) -> String {
        match &self.inner {
            ModelInner::Ds4(m) => m.name(),
            #[cfg(feature = "gemma")]
            ModelInner::Gemma { model, .. } => model.name.clone(),
        }
    }

    /// The end-of-sequence token.
    #[must_use]
    pub fn eos(&self) -> i32 {
        match &self.inner {
            ModelInner::Ds4(m) => m.eos(),
            #[cfg(feature = "gemma")]
            ModelInner::Gemma { model, .. } => gemma::eos(model),
        }
    }

    /// Renders and tokenizes a one-turn chat under the model's own template:
    /// the system turn when `system` is non-empty (or, for Gemma, when
    /// thinking), `prompt` as the user turn, and the assistant opener.
    ///
    /// # Errors
    /// Fails, on ds4, when `system` or `prompt` contains a NUL byte.
    pub fn encode_chat(&self, system: &str, prompt: &str, think: Think) -> Result<Vec<i32>, Error> {
        match &self.inner {
            ModelInner::Ds4(m) => m.encode_chat(system, prompt, think.ds4()),
            #[cfg(feature = "gemma")]
            ModelInner::Gemma { model, .. } => Ok(gemma::encode_chat(model, system, prompt, think)),
        }
    }

    /// Tokenizes already-rendered chat text, so control strings become their
    /// special tokens rather than literal pieces.
    ///
    /// # Errors
    /// Fails, on ds4, when `text` contains a NUL byte.
    pub fn tokenize_rendered(&self, text: &str) -> Result<Vec<i32>, Error> {
        match &self.inner {
            ModelInner::Ds4(m) => m.tokenize_rendered(text),
            #[cfg(feature = "gemma")]
            ModelInner::Gemma { model, .. } => Ok(gemma::tokenize_rendered(model, text)),
        }
    }

    /// The raw bytes of one token. Byte-level tokenizers split multi-byte
    /// characters across tokens, so decode a run of tokens, never one alone.
    #[must_use]
    pub fn token_bytes(&self, token: i32) -> Vec<u8> {
        match &self.inner {
            ModelInner::Ds4(m) => m.token_bytes(token),
            #[cfg(feature = "gemma")]
            ModelInner::Gemma { model, .. } => gemma::token_bytes(model, token),
        }
    }
}

/// One inference stream over a [`Model`]: its own KV cache and position.
#[derive(Debug)]
pub struct Session {
    inner: SessionInner,
    model: Arc<Model>,
}

#[derive(Debug)]
enum SessionInner {
    Ds4(ds4::Session),
    #[cfg(feature = "gemma")]
    Gemma(gemma::Session),
}

impl Session {
    /// Creates a session of `ctx_size` tokens, clamped to the model's.
    ///
    /// # Errors
    /// Fails when the ds4 engine cannot allocate the session.
    pub fn new(model: &Arc<Model>, ctx_size: i32) -> Result<Self, Error> {
        let inner = match &model.inner {
            ModelInner::Ds4(ds4) => SessionInner::Ds4(ds4::Session::new(ds4, ctx_size)?),
            #[cfg(feature = "gemma")]
            ModelInner::Gemma { model: gemma, ctx } => {
                let want = usize::try_from(ctx_size).unwrap_or(1).clamp(1, *ctx);
                SessionInner::Gemma(gemma::Session::new(gemma, want))
            }
        };
        Ok(Self {
            inner,
            model: Arc::clone(model),
        })
    }

    /// The model this session runs on.
    #[must_use]
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    /// The ds4 backend's session, when ds4 serves this one.
    #[must_use]
    pub fn as_ds4(&self) -> Option<&ds4::Session> {
        match &self.inner {
            SessionInner::Ds4(s) => Some(s),
            #[cfg(feature = "gemma")]
            SessionInner::Gemma(_) => None,
        }
    }

    /// The Gemma session, when Gemma serves this one.
    #[cfg(feature = "gemma")]
    #[must_use]
    pub fn as_gemma(&self) -> Option<&gemma_engine::session::Session> {
        match &self.inner {
            SessionInner::Gemma(s) => Some(s.inner()),
            SessionInner::Ds4(_) => None,
        }
    }

    /// The session's context size.
    #[must_use]
    pub fn ctx(&self) -> i32 {
        match &self.inner {
            SessionInner::Ds4(s) => s.ctx(),
            #[cfg(feature = "gemma")]
            SessionInner::Gemma(s) => i32::try_from(s.ctx()).unwrap_or(i32::MAX),
        }
    }

    /// How many tokens the KV cache holds.
    #[must_use]
    pub fn pos(&self) -> i32 {
        match &self.inner {
            SessionInner::Ds4(s) => s.pos(),
            #[cfg(feature = "gemma")]
            SessionInner::Gemma(s) => i32::try_from(s.pos()).unwrap_or(i32::MAX),
        }
    }

    /// Drops the KV cache so the next [`sync`](Self::sync) prefills from
    /// position 0 instead of reusing a common prefix.
    pub fn invalidate(&mut self) {
        match &mut self.inner {
            SessionInner::Ds4(s) => s.invalidate(),
            #[cfg(feature = "gemma")]
            SessionInner::Gemma(s) => s.invalidate(),
        }
    }

    /// Makes the KV cache hold exactly `tokens`, prefilling only what differs
    /// from the prefix already cached.
    ///
    /// # Errors
    /// Fails when the prompt does not fit the context or the prefill fails.
    pub fn sync(&mut self, tokens: &[i32]) -> Result<(), Error> {
        self.sync_with_progress(tokens, &mut |_, _, _| {})
    }

    /// Like [`sync`](Self::sync), calling `on_progress(event, current, total)`
    /// as the backend reports prefill progress. `current` is an absolute
    /// prompt position; Gemma reports once per 512-token chunk.
    ///
    /// # Errors
    /// Fails when the prompt does not fit the context or the prefill fails.
    pub fn sync_with_progress(
        &mut self,
        tokens: &[i32],
        on_progress: &mut dyn FnMut(&str, i32, i32),
    ) -> Result<(), Error> {
        match &mut self.inner {
            SessionInner::Ds4(s) => s.sync_with_progress(tokens, on_progress),
            #[cfg(feature = "gemma")]
            SessionInner::Gemma(s) => s.sync(tokens, on_progress),
        }
    }

    /// Evaluates `token` at the current position, appending it to the cache.
    ///
    /// # Errors
    /// Fails when the context is full or the evaluation fails.
    pub fn eval(&mut self, token: i32) -> Result<(), Error> {
        match &mut self.inner {
            SessionInner::Ds4(s) => s.eval(token),
            #[cfg(feature = "gemma")]
            SessionInner::Gemma(s) => s.eval(token),
        }
    }

    /// Samples the next token from the logits at the current position.
    /// `temperature <= 0` is greedy; `top_k` 0 keeps every token. `rng` is
    /// the sampler state, advanced in place; seed it once per generation.
    /// Gemma returns `-1` when nothing has been evaluated yet.
    #[must_use]
    pub fn sample(
        &mut self,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        rng: &mut u64,
    ) -> i32 {
        match &mut self.inner {
            SessionInner::Ds4(s) => s.sample(temperature, top_k, top_p, min_p, rng),
            #[cfg(feature = "gemma")]
            SessionInner::Gemma(s) => s.sample(temperature, top_k, top_p, min_p, rng),
        }
    }

    /// The `k` most likely next tokens, best first, with log-probabilities
    /// over the full vocabulary. Empty when the logits are unusable.
    #[must_use]
    pub fn top_logprobs(&self, k: usize) -> Vec<TokenScore> {
        match &self.inner {
            SessionInner::Ds4(s) => s.top_logprobs(k),
            #[cfg(feature = "gemma")]
            SessionInner::Gemma(s) => s.top_logprobs(k),
        }
    }
}
