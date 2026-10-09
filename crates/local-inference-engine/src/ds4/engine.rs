// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The ds4 backend's [`Model`] and [`Session`] over the linked C engine.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::ffi::Ds4ThinkMode as ThinkMode;
use crate::{Error, Options, TokenScore, ffi, metal};

/// A loaded model: the weights every [`Session`] on it shares.
///
/// Dropping the last handle closes the engine, which frees its Metal buffers
/// and the machine-wide `/tmp/ds4.lock`.
#[derive(Debug)]
pub struct Model {
    raw: NonNull<ffi::Ds4Engine>,
    ctx_size: i32,
}

// SAFETY: the C engine's weights are read-only once open and every session
// keeps its own state; plank shares one engine across threads the same way.
unsafe impl Send for Model {}
// SAFETY: as above.
unsafe impl Sync for Model {}

impl Model {
    /// Opens the model `options` names, pointing the Metal kernel variables
    /// at the bundled sources first.
    ///
    /// The C exits the process outright, rather than returning, when the model
    /// file is missing or another process holds `/tmp/ds4.lock`; check both
    /// beforehand if that matters.
    ///
    /// # Errors
    /// Fails when a path holds a NUL byte or the engine refuses the model.
    pub fn open(options: &Options) -> Result<Self, Error> {
        metal::set_source_env();
        let c_path = |p: &Path, what: &str| {
            CString::new(p.to_string_lossy().as_bytes())
                .map_err(|_| Error::new(format!("{what} path contains a NUL byte")))
        };
        let c_opt = |p: Option<&Path>, what: &str| p.map(|p| c_path(p, what)).transpose();
        let model = c_path(&options.model, "model")?;
        let mtp = c_opt(options.mtp.as_deref(), "mtp model")?;
        let vision = c_opt(options.vision.as_deref(), "vision encoder")?;
        let steering = c_opt(options.steering_file.as_deref(), "steering file")?;
        let as_ptr = |c: &Option<CString>| c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let opts = ffi::Ds4EngineOptions {
            model_path: model.as_ptr(),
            mtp_path: as_ptr(&mtp),
            vision_path: as_ptr(&vision),
            backend: options.backend,
            n_threads: options.threads,
            context_size: options.ctx_size,
            prefill_chunk: options.prefill_chunk,
            mtp_draft_tokens: 0,
            mtp_margin: 0.0,
            dspark_confidence_threshold: 0.0,
            directional_steering_file: as_ptr(&steering),
            expert_profile_path: std::ptr::null(),
            directional_steering_attn: options.steering_attn,
            directional_steering_ffn: options.steering_ffn,
            power_percent: options.power_percent,
            ssd_streaming_cache_experts: 0,
            ssd_streaming_cache_bytes: 0,
            ssd_streaming_full_layers: 0,
            ssd_streaming_preload_experts: 0,
            simulate_used_memory_bytes: 0,
            warm_weights: options.warm_weights,
            quality: options.quality,
            glm_mtp: false,
            glm_mtp_timing: false,
            dspark: false,
            dspark_strict: false,
            dspark_exact_sampling: false,
            dspark_confidence_threshold_set: false,
            cuda_tensor_parallel: false,
            ssd_streaming: false,
            ssd_streaming_cold: false,
            ssd_streaming_full_layers_set: false,
            inspect_only: false,
            placement_ctx_hint: options.ctx_size,
            placement_session_count_hint: 0,
            share_session_prefill_workspace: false,
            first_token_test: false,
            metal_graph_test: false,
            load_slice: false,
            load_layer_start: 0,
            load_layer_end: 0,
            load_output: false,
            distributed: ffi::Ds4DistributedOptions::default(),
            tp: ffi::Ds4TpOptions::default(),
        };
        let mut raw: *mut ffi::Ds4Engine = std::ptr::null_mut();
        // SAFETY: `opts` and the CStrings it points into outlive the call;
        // `raw` is a valid out-pointer.
        let rc = unsafe { ffi::ds4_engine_open(&raw mut raw, &raw const opts) };
        match NonNull::new(raw) {
            Some(raw) if rc == 0 => Ok(Self {
                raw,
                ctx_size: options.ctx_size,
            }),
            _ => {
                let kernels = if metal::kernels_missing() {
                    format!(
                        "; Metal kernel sources not found (set DS4_METAL_DIR, looked in {})",
                        metal::source_dir().display()
                    )
                } else {
                    String::new()
                };
                Err(Error::new(format!(
                    "ds4 could not open {} (rc {rc}){kernels}",
                    options.model.display()
                )))
            }
        }
    }

    /// The raw engine handle, for calls the safe API does not cover.
    #[must_use]
    pub fn as_raw(&self) -> *mut ffi::Ds4Engine {
        self.raw.as_ptr()
    }

    /// The context size the model was opened with.
    #[must_use]
    pub fn ctx_size(&self) -> i32 {
        self.ctx_size
    }

    /// The model name the engine reports.
    #[must_use]
    pub fn name(&self) -> String {
        // SAFETY: the engine is open; the result is a static C string or null.
        let p = unsafe { ffi::ds4_engine_model_name(self.as_raw()) };
        if p.is_null() {
            return String::new();
        }
        // SAFETY: `p` is a NUL-terminated string owned by the engine.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }

    /// The end-of-sequence token.
    #[must_use]
    pub fn eos(&self) -> i32 {
        // SAFETY: the engine is open.
        unsafe { ffi::ds4_token_eos(self.as_raw()) }
    }

    /// Whether a vision encoder is loaded.
    #[must_use]
    pub fn has_vision(&self) -> bool {
        // SAFETY: the engine is open.
        unsafe { ffi::ds4_engine_has_vision(self.as_raw()) }
    }

    /// Renders and tokenizes a one-turn chat under the model's template: the
    /// reasoning prefix, `system` when non-empty, `prompt` as the user
    /// message, and the assistant prefix. The same tokens the ds4 CLI builds
    /// for `--system SYSTEM -p PROMPT`.
    ///
    /// # Errors
    /// Fails when `system` or `prompt` contains a NUL byte.
    pub fn encode_chat(
        &self,
        system: &str,
        prompt: &str,
        think: ThinkMode,
    ) -> Result<Vec<i32>, Error> {
        let system = CString::new(system).map_err(|_| Error::new("system prompt contains NUL"))?;
        let prompt = CString::new(prompt).map_err(|_| Error::new("prompt contains NUL"))?;
        let mut tokens = Tokens::new();
        // SAFETY: the engine is open, the strings outlive the call, and
        // `tokens` is a valid token vector.
        unsafe {
            ffi::ds4_encode_chat_prompt(
                self.as_raw(),
                system.as_ptr(),
                prompt.as_ptr(),
                think,
                tokens.as_mut_ptr(),
            );
        }
        Ok(tokens.to_vec())
    }

    /// Tokenizes already-rendered chat text, so control strings such as
    /// `</think>` become their special tokens rather than literal pieces.
    ///
    /// # Errors
    /// Fails when `text` contains a NUL byte.
    pub fn tokenize_rendered(&self, text: &str) -> Result<Vec<i32>, Error> {
        let text = CString::new(text).map_err(|_| Error::new("text contains NUL"))?;
        let mut tokens = Tokens::new();
        // SAFETY: the engine is open and `text` outlives the call.
        unsafe {
            ffi::ds4_tokenize_rendered_chat(self.as_raw(), text.as_ptr(), tokens.as_mut_ptr());
        }
        Ok(tokens.to_vec())
    }

    /// The raw bytes of one token. Byte-level BPE splits multi-byte characters
    /// across tokens, so decode a run of tokens, never one at a time.
    #[must_use]
    pub fn token_bytes(&self, token: i32) -> Vec<u8> {
        let mut len: usize = 0;
        // SAFETY: the engine is open; `len` is a valid out-pointer.
        let p = unsafe { ffi::ds4_token_text(self.as_raw(), token, &raw mut len) };
        if p.is_null() {
            return Vec::new();
        }
        // SAFETY: `p` holds `len` bytes the caller owns.
        let bytes = unsafe { std::slice::from_raw_parts(p.cast::<u8>(), len) }.to_vec();
        // SAFETY: `p` was malloc'd by ds4_token_text for the caller to free.
        unsafe { libc::free(p.cast()) };
        bytes
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: the engine was opened by `open` and is closed only here.
        unsafe { ffi::ds4_engine_close(self.as_raw()) };
    }
}

/// One inference stream over a [`Model`]: its own KV cache and position.
#[derive(Debug)]
pub struct Session {
    raw: NonNull<ffi::Ds4Session>,
    // Keeps the weights alive for as long as the session points into them.
    model: Arc<Model>,
}

// SAFETY: a session is used from one thread at a time (`&mut self`), and the
// C keeps no thread-affine state in it.
unsafe impl Send for Session {}

impl Session {
    /// Creates a session of `ctx_size` tokens, clamped to the model's.
    ///
    /// # Errors
    /// Fails when the engine cannot allocate the session.
    pub fn new(model: &Arc<Model>, ctx_size: i32) -> Result<Self, Error> {
        let ctx_size = ctx_size.clamp(1, model.ctx_size.max(1));
        let mut raw: *mut ffi::Ds4Session = std::ptr::null_mut();
        // SAFETY: the engine is open; `raw` is a valid out-pointer.
        let rc = unsafe { ffi::ds4_session_create(&raw mut raw, model.as_raw(), ctx_size) };
        match NonNull::new(raw) {
            Some(raw) if rc == 0 => Ok(Self {
                raw,
                model: Arc::clone(model),
            }),
            _ => Err(Error::new(format!(
                "ds4 could not create a {ctx_size}-token session (rc {rc})"
            ))),
        }
    }

    /// The raw session handle, for calls the safe API does not cover.
    #[must_use]
    pub fn as_raw(&self) -> *mut ffi::Ds4Session {
        self.raw.as_ptr()
    }

    /// The model this session runs on.
    #[must_use]
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    /// The session's context size.
    #[must_use]
    pub fn ctx(&self) -> i32 {
        // SAFETY: the session is live.
        unsafe { ffi::ds4_session_ctx(self.as_raw()) }
    }

    /// How many tokens the KV cache holds.
    #[must_use]
    pub fn pos(&self) -> i32 {
        // SAFETY: the session is live.
        unsafe { ffi::ds4_session_pos(self.as_raw()) }
    }

    /// Drops the KV cache so the next [`sync`](Self::sync) prefills from
    /// position 0 instead of reusing a common prefix.
    pub fn invalidate(&mut self) {
        // SAFETY: the session is live.
        unsafe { ffi::ds4_session_invalidate(self.as_raw()) };
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
    /// as the engine reports prefill progress. `current` is an absolute
    /// prompt position.
    ///
    /// # Errors
    /// Fails when the prompt does not fit the context or the prefill fails.
    pub fn sync_with_progress(
        &mut self,
        tokens: &[i32],
        on_progress: &mut dyn FnMut(&str, i32, i32),
    ) -> Result<(), Error> {
        let mut prompt = Tokens::new();
        prompt.push_all(tokens);
        let mut cb: &mut dyn FnMut(&str, i32, i32) = on_progress;
        let ud = (&raw mut cb).cast::<c_void>();
        let mut err = [0 as c_char; 512];
        // SAFETY: the session is live; `ud` points at `cb`, which outlives the
        // sync, and both hooks are cleared again before it goes out of scope.
        let rc = unsafe {
            ffi::ds4_session_set_progress(self.as_raw(), Some(progress_trampoline), ud);
            ffi::ds4_session_set_display_progress(self.as_raw(), Some(progress_trampoline), ud);
            let rc =
                ffi::ds4_session_sync(self.as_raw(), prompt.as_ptr(), err.as_mut_ptr(), err.len());
            ffi::ds4_session_set_progress(self.as_raw(), None, std::ptr::null_mut());
            ffi::ds4_session_set_display_progress(self.as_raw(), None, std::ptr::null_mut());
            rc
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(c_error(&err, &format!("prefill failed (rc {rc})")))
        }
    }

    /// Evaluates `token` at the current position, appending it to the cache.
    ///
    /// # Errors
    /// Fails when the context is full or the evaluation fails.
    pub fn eval(&mut self, token: i32) -> Result<(), Error> {
        let mut err = [0 as c_char; 512];
        // SAFETY: the session is live; `err` is a writable buffer of its length.
        let rc =
            unsafe { ffi::ds4_session_eval(self.as_raw(), token, err.as_mut_ptr(), err.len()) };
        if rc == 0 {
            Ok(())
        } else {
            Err(c_error(&err, &format!("eval failed (rc {rc})")))
        }
    }

    /// Samples the next token from the logits at the current position.
    /// `temperature <= 0` is greedy. `rng` is the sampler state, advanced in
    /// place; seed it once per generation.
    #[must_use]
    pub fn sample(
        &mut self,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        rng: &mut u64,
    ) -> i32 {
        // SAFETY: the session is live; `rng` is a valid out-pointer.
        unsafe { ffi::ds4_session_sample(self.as_raw(), temperature, top_k, top_p, min_p, rng) }
    }

    /// The `k` most likely next tokens, best first, with their log-probabilities
    /// over the full vocabulary. Empty when the logits are unusable.
    #[must_use]
    pub fn top_logprobs(&self, k: usize) -> Vec<TokenScore> {
        let k_c = c_int::try_from(k).unwrap_or(c_int::MAX);
        let mut out = vec![TokenScore::default(); k];
        // SAFETY: the session is live; `out` has room for `k` entries.
        let n = unsafe { ffi::ds4_session_top_logprobs(self.as_raw(), out.as_mut_ptr(), k_c) };
        if n <= 0 {
            return Vec::new();
        }
        // Unfilled slots keep `id = -1` whatever the return says.
        out.truncate(out.iter().position(|s| s.id < 0).unwrap_or(out.len()));
        out
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: the session was created by `new` and is freed only here,
        // before `model` (and so the engine) can drop.
        unsafe { ffi::ds4_session_free(self.as_raw()) };
    }
}

/// Forwards a C progress event to the `&mut dyn FnMut` behind `ud`.
unsafe extern "C" fn progress_trampoline(
    ud: *mut c_void,
    event: *const c_char,
    cur: c_int,
    total: c_int,
) {
    if ud.is_null() {
        return;
    }
    // SAFETY: `ud` is the `&mut &mut dyn FnMut` set by `sync_with_progress`,
    // alive for the duration of the sync that invokes this.
    let cb = unsafe { &mut *ud.cast::<&mut dyn FnMut(&str, i32, i32)>() };
    let event = if event.is_null() {
        std::borrow::Cow::Borrowed("")
    } else {
        // SAFETY: the C passes a NUL-terminated event name.
        unsafe { CStr::from_ptr(event) }.to_string_lossy()
    };
    cb(&event, cur, total);
}

/// The NUL-terminated message the C wrote into `buf`, or `fallback`.
fn c_error(buf: &[c_char], fallback: &str) -> Error {
    if buf.first().copied().unwrap_or(0) == 0 {
        return Error::new(fallback);
    }
    // SAFETY: the C NUL-terminates its message within the buffer.
    Error::new(unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy())
}

/// An owned `ds4_tokens` vector, freed on drop.
struct Tokens(ffi::Ds4Tokens);

impl Tokens {
    fn new() -> Self {
        Self(ffi::Ds4Tokens::default())
    }

    fn as_mut_ptr(&mut self) -> *mut ffi::Ds4Tokens {
        &raw mut self.0
    }

    fn as_ptr(&self) -> *const ffi::Ds4Tokens {
        &raw const self.0
    }

    fn push_all(&mut self, tokens: &[i32]) {
        for &t in tokens {
            // SAFETY: `self.0` is a valid token vector.
            unsafe { ffi::ds4_tokens_push(self.as_mut_ptr(), t) };
        }
    }

    fn to_vec(&self) -> Vec<i32> {
        let Ok(len) = usize::try_from(self.0.len) else {
            return Vec::new();
        };
        if self.0.v.is_null() || len == 0 {
            return Vec::new();
        }
        // SAFETY: `v` points at `len` initialised ids owned by the vector.
        unsafe { std::slice::from_raw_parts(self.0.v, len) }.to_vec()
    }
}

impl Drop for Tokens {
    fn drop(&mut self) {
        // SAFETY: the buffer was allocated by ds4 and is freed only here.
        unsafe { ffi::ds4_tokens_free(&raw mut self.0) };
    }
}
