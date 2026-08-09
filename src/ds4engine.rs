// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Safe wrapper around the ds4 C engine implementing [`Engine`].
//!
//! Present only under the `ds4_engine` cfg (macOS + built `refs/ds4` submodule).
//! The transcript arriving from the UI is plain role-tagged text; this wrapper
//! reparses it into ds4 chat-template tokens, prefills a session, and samples.
//!
//! The engine is split into two types (issue #28, `docs/SHARED-ENGINE-DESIGN.md`
//! §3): [`Ds4Model`] owns the immutable weights/tokenizer/Metal queue behind an
//! `Arc`, and [`Ds4Session`] owns one live FFI session (its private KV suffix +
//! cursor) and implements [`Engine`]. Today's single-owner path is a
//! `Ds4Session` over a solely-owned `Ds4Model` — behavior-identical to before
//! the split. The [`crate::host`] shared engine hands out many `Ds4Session`s
//! over one shared `Ds4Model`.

use std::ffi::{CStr, CString};
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

use crate::ds4tokens::{self, SectionKey, SpanRole, TokenTranscript};
use crate::engine::{
    Engine, EngineError, EngineEvent, GenerationOptions, GenerationStats, PrefillProgress,
    ThinkMode,
};
use crate::ffi;
use crate::host::{HostSession, ModelHandle};
use crate::snapshot::{RestoreOnDrop, SessionSnapshot};

/// Tokens a single speculative step may commit: the sampled token plus the
/// longest draft block the engine proposes. Matches the C CLI's buffer, which
/// is the ceiling the entry point itself is written against.
const SPEC_ACCEPT_CAP: usize = 17;

/// The immutable, shareable half of the ds4 engine: weights, tokenizer, and the
/// Metal command queue. Cheap to share read-only, expensive to build, so it
/// lives behind an `Arc` (design §3, §4). Frees the engine on drop; the last
/// `Arc` to drop tears down the Metal context.
#[derive(Debug)]
pub struct Ds4Model {
    engine: *mut ffi::Ds4Engine,
    ctx_size: i32,
    /// Chat-template token overhead of an empty message (`-1` until measured),
    /// subtracted by `count_tokens` so it returns just the text's tokens.
    count_overhead: AtomicI32,
    /// Warm system-prompt KV captured once at host bootstrap; restored into
    /// each freshly spawned shared session so attach never cold-prefills the
    /// system prompt (design §6). `None` for the single-owner path.
    warm: Mutex<Option<SessionSnapshot>>,
}

// SAFETY: the engine pointer owns read-only weights + the Metal queue and is
// freed only on drop. `Ds4Model` is used single-threaded for `ds4_session_*`
// (all such calls run on the host's one GPU thread, design §5); the tokenizer
// reads (`count_tokens`, `message_tokens`) touch immutable state. The warm
// snapshot is guarded by a `Mutex`. Send+Sync let it live in an `Arc`.
unsafe impl Send for Ds4Model {}
unsafe impl Sync for Ds4Model {}

thread_local! {
    static INTERRUPT: AtomicBool = const { AtomicBool::new(false) };
}

unsafe extern "C" fn cancel_cb(_ud: *mut std::os::raw::c_void) -> bool {
    INTERRUPT.with(|f| f.load(Ordering::SeqCst))
}

/// Bridges ds4's C display-progress callback to the Rust event sink.
///
/// `base` is the count of cached tokens reused from the live KV prefix. The C
/// backend already reports absolute prompt positions, so `base` is the bar's
/// floor and the subtrahend for per-prefill throughput — never an offset added
/// to the reported position (see `PrefillProgress::from_absolute`).
struct ProgressCtx<'a> {
    on_event: &'a mut dyn FnMut(EngineEvent),
    interrupt: &'a dyn Fn() -> bool,
    start: std::time::Instant,
    base: i32,
    total: i32,
}

unsafe extern "C" fn progress_cb(
    ud: *mut std::os::raw::c_void,
    _event: *const std::os::raw::c_char,
    cur: std::os::raw::c_int,
    _total: std::os::raw::c_int,
) {
    if ud.is_null() {
        return;
    }
    // SAFETY: ud is the ProgressCtx pointer we installed for this sync call.
    let ctx = unsafe { &mut *ud.cast::<ProgressCtx>() };
    // Prefill runs inside ds4_session_sync, which only observes the cancel
    // callback's flag; relay the caller's interrupt here so Esc/Ctrl-C can
    // abort prefill, not just token generation.
    if (ctx.interrupt)() {
        INTERRUPT.with(|f| f.store(true, Ordering::SeqCst));
    }
    let secs = ctx.start.elapsed().as_secs_f64();
    let progress = PrefillProgress::from_absolute(ctx.base, cur, &mut ctx.total, secs);
    (ctx.on_event)(EngineEvent::Prefill(progress));
}

/// Decode rate measured from the warmup mark to now, or 0 when the pass never
/// reached the mark or produced too little after it to mean anything.
///
/// `mark` is the instant and token count at which the pass crossed
/// [`crate::engine::STEADY_WARMUP_SECS`]; `generated` is the count now.
fn steady_rate(mark: Option<(std::time::Instant, i32)>, generated: i32) -> f64 {
    let Some((at, tokens_at)) = mark else {
        return 0.0;
    };
    let secs = at.elapsed().as_secs_f64();
    let tokens = generated - tokens_at;
    if secs <= 0.0 || tokens < STEADY_MIN_TOKENS {
        return 0.0;
    }
    f64::from(tokens) / secs
}

/// Tokens a pass must produce *after* the warmup before its steady rate is
/// reported. A couple of tokens divided by a sliver of a second is noise.
const STEADY_MIN_TOKENS: i32 = 8;

impl Ds4Model {
    /// Opens a model file with the given backend, context size, and tuning
    /// knobs (`--mtp`, `--ssd-streaming`, steering, ...).
    ///
    /// # Errors
    /// Returns [`EngineError`] if the model fails to load.
    pub fn open(
        model_path: impl AsRef<Path>,
        backend: ffi::Ds4Backend,
        ctx_size: i32,
        n_threads: i32,
        power_percent: i32,
        tuning: &crate::config::EngineTuning,
    ) -> Result<Self, EngineError> {
        set_metal_source_env();
        let path = model_path.as_ref();
        let c_path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| EngineError::new("model path contains a NUL byte"))?;
        let c_opt_path = |p: Option<&Path>, what: &str| -> Result<Option<CString>, EngineError> {
            p.map(|p| {
                CString::new(p.to_string_lossy().as_bytes())
                    .map_err(|_| EngineError::new(format!("{what} path contains a NUL byte")))
            })
            .transpose()
        };
        let c_mtp = c_opt_path(tuning.mtp_path.as_deref(), "mtp model")?;
        let c_steering = c_opt_path(tuning.dir_steering_file.as_deref(), "dir-steering file")?;
        let as_ptr = |c: &Option<CString>| c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let opts = ffi::Ds4EngineOptions {
            model_path: c_path.as_ptr(),
            mtp_path: as_ptr(&c_mtp),
            backend,
            n_threads,
            context_size: ctx_size,
            prefill_chunk: tuning.prefill_chunk,
            mtp_draft_tokens: tuning.mtp_draft_tokens,
            mtp_margin: tuning.mtp_margin,
            // Ignored by the engine unless `_set` below is true, in which case
            // it picks its own backend-dependent default. Passing a placeholder
            // beats mirroring a number that upstream retunes.
            dspark_confidence_threshold: tuning.dspark_confidence.unwrap_or(0.0),
            directional_steering_file: as_ptr(&c_steering),
            expert_profile_path: std::ptr::null(),
            directional_steering_attn: tuning.dir_steering_attn,
            directional_steering_ffn: tuning.dir_steering_ffn,
            power_percent,
            ssd_streaming_cache_experts: tuning.ssd_streaming_cache_experts,
            ssd_streaming_cache_bytes: tuning.ssd_streaming_cache_bytes,
            ssd_streaming_full_layers: 0,
            ssd_streaming_preload_experts: tuning.ssd_streaming_preload_experts,
            simulate_used_memory_bytes: tuning.simulate_used_memory_bytes,
            warm_weights: tuning.warm_weights,
            quality: tuning.quality,
            glm_mtp: false,
            glm_mtp_timing: false,
            dspark: tuning.dspark,
            dspark_strict: tuning.dspark_strict,
            dspark_confidence_threshold_set: tuning.dspark_confidence.is_some(),
            cuda_tensor_parallel: false,
            ssd_streaming: tuning.ssd_streaming,
            ssd_streaming_cold: tuning.ssd_streaming_cold,
            ssd_streaming_full_layers_set: false,
            inspect_only: false,
            placement_ctx_hint: ctx_size,
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
        let mut engine: *mut ffi::Ds4Engine = std::ptr::null_mut();
        // SAFETY: opts and its CStrings outlive the call; engine is a valid out-ptr.
        let rc = unsafe { ffi::ds4_engine_open(&raw mut engine, &raw const opts) };
        if rc != 0 || engine.is_null() {
            let mut msg = format!("failed to open model {}", path.display());
            let kernels_missing = std::env::var_os("DS4_METAL_FLASH_ATTN_SOURCE")
                .is_none_or(|p| !Path::new(&p).exists());
            if kernels_missing {
                msg.push_str(
                    " (Metal kernel sources not found; set DS4_METAL_DIR to a \
                     directory containing the .metal files)",
                );
            }
            return Err(EngineError::new(msg));
        }
        Ok(Self {
            engine,
            ctx_size,
            count_overhead: AtomicI32::new(-1),
            warm: Mutex::new(None),
        })
    }

    /// Opens a shared model and warms the system-prompt prefix once, capturing
    /// it so [`ModelHandle::spawn`] can restore it into each fresh session
    /// without a cold prefill (design §6). Returns the `Arc` the host holds.
    ///
    /// # Errors
    /// Returns [`EngineError`] if the model fails to load or warm.
    ///
    /// # Panics
    /// Panics if the warm-snapshot mutex is poisoned (a prior panic while it was
    /// held) — impossible during single-threaded bootstrap.
    pub fn open_shared(
        model_path: impl AsRef<Path>,
        backend: ffi::Ds4Backend,
        ctx_size: i32,
        n_threads: i32,
        power_percent: i32,
        tuning: &crate::config::EngineTuning,
        system: &str,
    ) -> Result<Arc<Self>, EngineError> {
        let model = Arc::new(Self::open(
            model_path,
            backend,
            ctx_size,
            n_threads,
            power_percent,
            tuning,
        )?);
        // Warm on a throwaway bootstrap session, then capture its KV.
        let mut boot = Ds4Session::from_model(Arc::clone(&model));
        boot.warm_reset(system)?;
        boot.warm_sync(&mut |_| {})?;
        let session = boot.ensure_session()?;
        let snap = SessionSnapshot::capture(session)?;
        *model.warm.lock().unwrap() = Some(snap);
        Ok(model)
    }

    /// Creates a fresh FFI session over the shared weights with a context
    /// window of `ctx_size` tokens (design §7, v2 per-client sizing). `ctx_size`
    /// is clamped to `[1, self.ctx_size]` so a session never over-books the
    /// model's configured maximum.
    fn create_session(&self, ctx_size: i32) -> Result<*mut ffi::Ds4Session, EngineError> {
        let ctx_size = ctx_size.clamp(1, self.ctx_size.max(1));
        let mut session: *mut ffi::Ds4Session = std::ptr::null_mut();
        // SAFETY: engine valid; session is a valid out-ptr.
        let rc = unsafe { ffi::ds4_session_create(&raw mut session, self.engine, ctx_size) };
        if rc != 0 || session.is_null() {
            return Err(EngineError::new("failed to create session"));
        }
        Ok(session)
    }

    /// Model name reported by the engine.
    #[must_use]
    pub fn model_name(&self) -> String {
        // SAFETY: engine is valid; the returned pointer is a static C string.
        let p = unsafe { ffi::ds4_engine_model_name(self.engine) };
        if p.is_null() {
            return String::new();
        }
        // SAFETY: p is a valid NUL-terminated string owned by the engine.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }

    /// Tokenizes `text` as the content of a single user message, returning the
    /// full templated length (chat begin + role wrapper + content).
    fn templated_len(&self, text: &str) -> Option<i32> {
        let content = CString::new(text).ok()?;
        let mut tokens = Ds4TokensGuard::new();
        // SAFETY: engine and tokens are valid; role/content outlive the calls.
        unsafe {
            ffi::ds4_chat_begin(self.engine, tokens.as_mut_ptr());
            ffi::ds4_chat_append_message(
                self.engine,
                tokens.as_mut_ptr(),
                c"user".as_ptr(),
                content.as_ptr(),
            );
        }
        Some(tokens.len())
    }

    /// Builds chat-template tokens for the system prompt alone (no assistant
    /// prefix), used to warm and checkpoint the KV cache.
    fn build_system_tokens(
        &self,
        system: &str,
        trusted_len: usize,
        think: ThinkMode,
    ) -> Ds4TokensGuard {
        let mut tokens = Ds4TokensGuard::new();
        // SAFETY: engine and tokens are valid for the whole build.
        unsafe { ffi::ds4_chat_begin(self.engine, tokens.as_mut_ptr()) };
        self.append_effort_prefix(&mut tokens, think);
        self.append_system_text(&mut tokens, system, trusted_len);
        tokens
    }

    /// Appends a system prompt, split at `trusted_len` into a control-text
    /// prefix and plain content — the C's `agent_append_system_prompt`.
    ///
    /// The prefix goes through `ds4_tokenize_rendered_chat`, which maps the
    /// literal `｜DSML｜` in the prompt's examples (and `<think>`, and the turn
    /// markers) onto their real vocabulary tokens. The model was trained with
    /// `｜DSML｜` as one atomic token; spelling it out as BPE pieces is what the
    /// model then reproduces, letter by letter, and occasionally corrupts —
    /// `<｜DSinvoke name="bash">` in `repro-1785754509.md`.
    ///
    /// The remainder is deliberately *not* given that treatment. It holds MCP
    /// schemas and `-sys` text, and a `<｜User｜>` in either must stay inert
    /// prose rather than become a real turn boundary. `trusted_len` is clamped
    /// and snapped to a character boundary, so a stale or wrong value can only
    /// ever shrink the trusted span, never leak untrusted bytes into it.
    fn append_system_text(&self, tokens: &mut Ds4TokensGuard, system: &str, trusted_len: usize) {
        let split = (0..=trusted_len.min(system.len()))
            .rev()
            .find(|&i| system.is_char_boundary(i))
            .unwrap_or(0);
        let (trusted, plain) = system.split_at(split);
        if !trusted.is_empty()
            && let Ok(text) = CString::new(trusted)
        {
            // SAFETY: engine and tokens valid; text outlives the call.
            unsafe {
                ffi::ds4_tokenize_rendered_chat(self.engine, text.as_ptr(), tokens.as_mut_ptr());
            }
        }
        if !plain.is_empty()
            && let (Ok(role), Ok(content)) = (CString::new("system"), CString::new(plain))
        {
            // SAFETY: role/content strings outlive the call.
            unsafe {
                ffi::ds4_chat_append_message(
                    self.engine,
                    tokens.as_mut_ptr(),
                    role.as_ptr(),
                    content.as_ptr(),
                );
            }
        }
    }

    /// Chat-template preamble tokens emitted by `ds4_chat_begin` (BOS / opening
    /// template), prepended once ahead of the first message. Folded into the
    /// first span of the token transcript so the buffer is self-contained.
    fn begin_tokens(&self, think: ThinkMode) -> Vec<i32> {
        let mut tokens = Ds4TokensGuard::new();
        // SAFETY: engine and tokens are valid for the call.
        unsafe { ffi::ds4_chat_begin(self.engine, tokens.as_mut_ptr()) };
        self.append_effort_prefix(&mut tokens, think);
        tokens.to_vec()
    }

    /// Appends the level's effort preamble ([`ThinkMode::effort_prefix`]), or
    /// nothing when the level has none.
    ///
    /// The preamble is plain text with no role wrapper, so its position is
    /// load-bearing: immediately after the `ds4_chat_begin` preamble and ahead
    /// of the system message, which is where the C puts it — both in
    /// `encode_chat_prompt` (tokenized straight before the system text) and in
    /// the REPL's `repl_chat_apply_max_prefix` (inserted at transcript index 1,
    /// past the BOS). Folding it into the system *string* instead would place
    /// it after the system role marker and diverge.
    ///
    /// `Max` goes through the C's own `ds4_chat_append_max_effort_prefix` so its
    /// tokens stay byte-identical to the reference; `Low`, which the C does not
    /// have, is tokenized here from [`crate::engine::THINK_LOW_PREFIX`] through
    /// the same rendered-chat tokenizer the C symbol uses internally.
    fn append_effort_prefix(&self, tokens: &mut Ds4TokensGuard, think: ThinkMode) {
        match think {
            ThinkMode::Max => {
                // SAFETY: engine and tokens are valid for the call.
                unsafe { ffi::ds4_chat_append_max_effort_prefix(self.engine, tokens.as_mut_ptr()) };
            }
            ThinkMode::Low => {
                tokens.push_all(&self.tokenize_rendered(crate::engine::THINK_LOW_PREFIX));
            }
            ThinkMode::Off | ThinkMode::Medium => {}
        }
    }

    /// The chat-template tokens for one message (role wrapper + content),
    /// independent of surrounding context — the ds4 template concatenates
    /// per-message, so a message's tokens are the delta `ds4_chat_append_message`
    /// adds. Isolated here by appending after a `ds4_chat_begin` and returning
    /// only the bytes past the preamble.
    fn message_tokens(&self, role: &str, content: &str) -> Vec<i32> {
        let (Ok(c_role), Ok(c_content)) = (CString::new(role), CString::new(content)) else {
            return Vec::new();
        };
        let mut tokens = Ds4TokensGuard::new();
        // SAFETY: engine/tokens valid; strings outlive the calls.
        unsafe {
            ffi::ds4_chat_begin(self.engine, tokens.as_mut_ptr());
            let base = usize::try_from(tokens.len()).unwrap_or(0);
            ffi::ds4_chat_append_message(
                self.engine,
                tokens.as_mut_ptr(),
                c_role.as_ptr(),
                c_content.as_ptr(),
            );
            tokens.slice_from(base).to_vec()
        }
    }

    /// The tokens for a system message, split at `trusted_len` — the delta
    /// [`Self::append_system_text`] contributes, isolated from any preamble the
    /// same way [`Self::message_tokens`] isolates a message's.
    ///
    /// A system message carries no role wrapper (`ds4_chat_append_message`
    /// tokenizes its content directly), so this is exactly the content tokens.
    fn system_message_tokens(&self, system: &str, trusted_len: usize) -> Vec<i32> {
        let mut tokens = Ds4TokensGuard::new();
        self.append_system_text(&mut tokens, system, trusted_len);
        tokens.to_vec()
    }

    /// The assistant-prefix tokens for the given think mode (the `<assistant>`
    /// opening the model generates after). Used both to open a live generation
    /// prompt and, captured, as the head of a stored reply span.
    fn assistant_prefix_tokens(&self, think: ThinkMode) -> Vec<i32> {
        let mut tokens = Ds4TokensGuard::new();
        // SAFETY: engine and tokens valid.
        unsafe {
            ffi::ds4_chat_append_assistant_prefix(
                self.engine,
                tokens.as_mut_ptr(),
                ds4_think(think),
            );
        }
        tokens.to_vec()
    }

    /// Raw detokenized bytes for one token. Byte-level BPE splits multi-byte
    /// characters across tokens, so a single token's bytes may not be valid
    /// UTF-8 — decode across tokens with [`Utf8Stream`], never per token.
    fn token_bytes(&self, token: i32) -> Vec<u8> {
        let mut len: usize = 0;
        // SAFETY: engine valid; len is a valid out-ptr.
        let p = unsafe { ffi::ds4_token_text(self.engine, token, &raw mut len) };
        if p.is_null() {
            return Vec::new();
        }
        // SAFETY: p points to len bytes owned by us; we copy then free.
        let bytes = unsafe { std::slice::from_raw_parts(p.cast::<u8>(), len) }.to_vec();
        // SAFETY: p was allocated by ds4_token_text for the caller to free.
        unsafe { libc::free(p.cast()) };
        bytes
    }

    /// Tokenizes already-rendered chat text, so control strings map to their
    /// special tokens (`</think>` becomes one token, not seven pieces).
    /// Empty when `text` contains a NUL.
    fn tokenize_rendered(&self, text: &str) -> Vec<i32> {
        let Ok(c) = CString::new(text) else {
            return Vec::new();
        };
        let mut tokens = Ds4TokensGuard::new();
        // SAFETY: engine and tokens are valid; `c` outlives the call.
        unsafe { ffi::ds4_tokenize_rendered_chat(self.engine, c.as_ptr(), tokens.as_mut_ptr()) };
        tokens.to_vec()
    }

    /// Approximate token count of `text`, excluding chat-template overhead.
    #[must_use]
    pub fn count_tokens(&self, text: &str) -> i32 {
        if self.count_overhead.load(Ordering::Relaxed) < 0 {
            self.count_overhead.store(
                self.templated_len("").unwrap_or(-1).max(-1),
                Ordering::Relaxed,
            );
        }
        let overhead = self.count_overhead.load(Ordering::Relaxed);
        match (overhead, self.templated_len(text)) {
            (o, Some(len)) if o >= 0 => (len - o).max(0),
            // NUL bytes (or a failed overhead probe) fall back to the estimate.
            _ => i32::try_from(text.len() / 4).unwrap_or(i32::MAX),
        }
    }
}

impl Drop for Ds4Model {
    fn drop(&mut self) {
        // SAFETY: engine was opened by us and not yet closed.
        unsafe { ffi::ds4_engine_close(self.engine) };
    }
}

impl ModelHandle for Ds4Model {
    fn spawn(self: Arc<Self>, ctx_size: i32) -> Result<Box<dyn HostSession>, EngineError> {
        // Fresh session over the shared weights, warmed from the captured
        // system-prompt snapshot so attach never cold-prefills it (§6). The
        // host clamps `ctx_size` to the model range; `create_session` re-clamps.
        let session = self.create_session(ctx_size)?;
        if let Some(warm) = self.warm.lock().unwrap().as_ref()
            && let Err(e) = warm.restore(session)
        {
            // SAFETY: session was just created by us and not yet freed.
            unsafe { ffi::ds4_session_free(session) };
            return Err(e);
        }
        let inner = Ds4Session {
            model: Arc::clone(&self),
            session,
            transcript: TokenTranscript::new(),
            warm_tokens: Ds4TokensGuard::new(),
            think: ThinkMode::default(),
            trusted_system_len: 0,
        };
        Ok(Box::new(Ds4HostSession {
            inner,
            pending: None,
            active: None,
        }))
    }

    fn model_name(&self) -> String {
        Ds4Model::model_name(self)
    }

    fn ctx_size(&self) -> i32 {
        self.ctx_size
    }

    fn count_tokens(&self, text: &str) -> i32 {
        Ds4Model::count_tokens(self, text)
    }
}

/// Loaded ds4 session: one live FFI session (its private KV suffix + cursor)
/// over a shared [`Ds4Model`]. Implements [`Engine`] for the single-owner path.
///
/// The session is kept alive across turns so `ds4_session_sync` reuses the
/// cached KV prefix (the constant system prompt and any unchanged earlier
/// turns) and only evaluates the new suffix each turn.
#[derive(Debug)]
pub struct Ds4Session {
    model: Arc<Ds4Model>,
    session: *mut ffi::Ds4Session,
    /// Append-only token transcript — the source of truth for this session's
    /// prompt prefix, mirroring the C's `w->transcript` (issue #58). Each turn
    /// reconciles the UI's rendered transcript against it structurally (keep the
    /// matching span prefix verbatim, retokenize the divergent tail) instead of
    /// re-templating from text. Persisted with the KV in `get_kv`/`set_kv` so a
    /// resumed session keeps full prefix reuse.
    transcript: TokenTranscript,
    /// Cumulative token buffer for an in-progress warm walk (`warm_reset` /
    /// `warm_sync`). Empty outside a walk.
    warm_tokens: Ds4TokensGuard,
    /// Reasoning level for prompts built from here on. Only the levels carrying
    /// an effort preamble ([`ThinkMode::effort_prefix`]) change the prompt
    /// *prefix*, so `set_think_mode` drops the token transcript exactly when the
    /// preamble changes.
    think: ThinkMode,
    /// Byte length of the system prompt's trusted, control-text prefix; see
    /// [`Engine::set_trusted_system_prefix`]. Zero means "tokenize all of it as
    /// plain content", the safe default.
    trusted_system_len: usize,
}

/// Back-compatible alias: the single-owner engine callers used before the split
/// (design §3 — today's path is a `Ds4Session` over a solely-owned model).
pub type Ds4Engine = Ds4Session;

// SAFETY: the session is used single-threaded (the agent turn loop, or the
// host's one GPU thread). It owns the FFI session and frees it on drop; the
// model is shared read-only behind an `Arc`. Send lets it move into the boxed
// trait object.
unsafe impl Send for Ds4Session {}

impl Ds4Session {
    /// Opens a solely-owned model and returns a session over it — the
    /// single-owner path, behavior-identical to the pre-split engine.
    ///
    /// # Errors
    /// Returns [`EngineError`] if the model fails to load.
    pub fn open(
        model_path: impl AsRef<Path>,
        backend: ffi::Ds4Backend,
        ctx_size: i32,
        n_threads: i32,
        power_percent: i32,
        tuning: &crate::config::EngineTuning,
    ) -> Result<Self, EngineError> {
        let model = Ds4Model::open(
            model_path,
            backend,
            ctx_size,
            n_threads,
            power_percent,
            tuning,
        )?;
        Ok(Self::from_model(Arc::new(model)))
    }

    /// Wraps a shared model in a session whose FFI session is created lazily.
    #[must_use]
    pub fn from_model(model: Arc<Ds4Model>) -> Self {
        Self {
            model,
            session: std::ptr::null_mut(),
            transcript: TokenTranscript::new(),
            warm_tokens: Ds4TokensGuard::new(),
            think: ThinkMode::default(),
            trusted_system_len: 0,
        }
    }

    /// Forks this session: a sibling over the *same* weights, pre-loaded with
    /// this session's KV and token transcript, which then diverges
    /// independently (`docs/SESSION-CLONE-DESIGN.md` §5).
    ///
    /// This is the local form of the clone primitive. It needs no
    /// [`EngineHost`](crate::host::EngineHost): a `Ds4Session` already holds an
    /// `Arc<Ds4Model>`, so the sibling shares the loaded weights and Metal
    /// queue and costs only its own KV. The interactive agent can therefore
    /// fork its live session directly, without being host-backed.
    ///
    /// Capture is non-destructive — `self` is left exactly as it was, cursor
    /// included — so the caller keeps generating on it. The two sessions share
    /// nothing mutable afterwards.
    ///
    /// Cost is one full KV allocation for the fork's lifetime, plus the
    /// serialize/restore round trip. Callers that fork per aside should drop
    /// the sibling as soon as it has answered.
    ///
    /// # Errors
    /// Returns [`EngineError`] if this session has no live KV to capture, or if
    /// the sibling cannot be created or loaded.
    pub fn fork(&mut self) -> Result<Self, EngineError> {
        let bytes = self
            .capture_bytes()
            .ok_or_else(|| EngineError::new("cannot fork a session with no live KV"))?;
        let mut sibling = Self::from_model(Arc::clone(&self.model));
        sibling.load_bytes(&bytes)?;
        Ok(sibling)
    }

    /// Forks this session into a *sliceable* one: a [`HostSession`] carrying
    /// this session's KV and transcript, ready to hand to
    /// [`SliceRunner`](crate::slice::SliceRunner).
    ///
    /// This is the bridge that lets a plain interactive `Ds4Session` fan out.
    /// [`fork`](Self::fork) gives a sibling that generates all-or-nothing;
    /// wrapping it in [`Ds4HostSession`] gives one that generates in resumable
    /// K-token slices, which is what interleaving several of them requires.
    ///
    /// # Errors
    /// Returns [`EngineError`] if this session has no live KV to capture, or if
    /// the sibling cannot be created or loaded.
    pub fn fork_sliceable(&mut self) -> Result<Box<dyn HostSession>, EngineError> {
        Ok(Box::new(Ds4HostSession {
            inner: self.fork()?,
            pending: None,
            active: None,
        }))
    }

    /// Current absolute cursor position in the live KV, in tokens; 0 when no
    /// FFI session exists yet.
    #[must_use]
    pub fn ctx_tokens(&self) -> i32 {
        if self.session.is_null() {
            return 0;
        }
        // SAFETY: session is non-null here and freed only on drop.
        unsafe { ffi::ds4_session_pos(self.session) }
    }

    /// This session's token transcript — the source of truth for its prompt
    /// prefix. Exposed so callers can assert an operation left it untouched.
    #[must_use]
    pub fn transcript_tokens(&self) -> &[i32] {
        self.transcript.tokens()
    }

    /// The transcript's span index, whose shape must mirror the rendered
    /// transcript's sections (`docs/DOUBLE-BTW.md`).
    #[must_use]
    pub fn transcript_spans(&self) -> &[crate::ds4tokens::Span] {
        self.transcript.spans()
    }

    /// The shared model behind this session, for creating siblings over the
    /// same weights without reloading them.
    #[must_use]
    pub fn model_arc(&self) -> Arc<Ds4Model> {
        Arc::clone(&self.model)
    }

    /// Serializes this session's live KV and token transcript to owned bytes,
    /// or `None` when it has no live session to capture. Non-destructive.
    ///
    /// The transcript rides in front of the KV bytes (the same wrap as
    /// [`get_kv`](Engine::get_kv)) so a session loaded from these bytes keeps
    /// its exact token buffer and prefills only genuinely new tokens.
    fn capture_bytes(&mut self) -> Option<Vec<u8>> {
        if self.session.is_null() {
            return None;
        }
        // Copy out an owned buffer so the engine-owned snapshot is freed here.
        // The returned Vec is Rust's, which is why `load_bytes` must restore it
        // through the non-owning FFI path (FINDINGS double-free).
        let snap = SessionSnapshot::capture(self.session).ok()?;
        let mut out = ds4tokens::encode(&self.transcript);
        out.extend_from_slice(snap.as_bytes());
        Some(out)
    }

    /// Loads bytes produced by [`capture_bytes`](Self::capture_bytes) into this
    /// session, creating its FFI session if needed. The restored KV/cursor and
    /// its captured token transcript supersede whatever this session held.
    ///
    /// # Errors
    /// Returns [`EngineError`] if the session cannot be created or the backend
    /// rejects the payload.
    fn load_bytes(&mut self, bytes: &[u8]) -> Result<(), EngineError> {
        let session = self.ensure_session()?;
        let (transcript, kv) = match ds4tokens::decode(bytes) {
            Some((transcript, off)) => (transcript, &bytes[off..]),
            None => (TokenTranscript::new(), strip_legacy(bytes)),
        };
        // Rust owns `bytes`, so this is the non-owning restore path: the engine
        // must never free our buffer (FINDINGS double-free).
        SessionSnapshot::restore_bytes(session, kv)?;
        self.transcript = transcript;
        Ok(())
    }

    /// Model name reported by the engine.
    #[must_use]
    pub fn model_name(&self) -> String {
        self.model.model_name()
    }

    /// Ensures a live session exists, creating it lazily.
    fn ensure_session(&mut self) -> Result<*mut ffi::Ds4Session, EngineError> {
        if self.session.is_null() {
            // Single-owner path: the session gets the model's full context.
            self.session = self.model.create_session(self.model.ctx_size)?;
        }
        Ok(self.session)
    }

    /// Structurally reconciles the token transcript to the UI's rendered
    /// `transcript` (issue #58): keep the leading spans whose (role, text) key
    /// still matches, drop everything from the first divergence, then retokenize
    /// and append the divergent tail. Deterministic user/system/tool messages
    /// retokenize to the same ids; a re-appended assistant section (compaction,
    /// legacy resume) retokenizes from text and pays one re-prefill at that
    /// point — which the KV was already going to force. This replaces the old
    /// text-splice matching entirely.
    fn reconcile(&mut self, transcript: &str) {
        let sections = parse_sections(transcript);
        let keys: Vec<SectionKey> = sections
            .iter()
            .filter_map(|(role, text)| {
                SpanRole::from_tag(role).map(|role| SectionKey {
                    role,
                    text: text.clone(),
                })
            })
            .collect();
        let keep = self.transcript.common_prefix(&keys);
        kv_debug(|| {
            let mut s = format!(
                "reconcile: {} spans held, {} sections in, kept {}",
                self.transcript.spans().len(),
                keys.len(),
                keep
            );
            // The first divergent span is the whole story: everything from here
            // is retokenized and re-prefilled. Show both sides so a mismatch in
            // invisible characters (trailing whitespace, #64) is diagnosable.
            if keep < keys.len() {
                let k = &keys[keep];
                let _ = write!(
                    s,
                    "\n  diverged at {keep}: incoming {:?} len={} head={:?}",
                    k.role,
                    k.text.len(),
                    k.text.chars().take(60).collect::<String>()
                );
                if let Some(held) = self.transcript.spans().get(keep) {
                    let _ = write!(
                        s,
                        "\n              held     {:?} len={} head={:?}",
                        held.role,
                        held.text.len(),
                        held.text.chars().take(60).collect::<String>()
                    );
                }
            }
            s
        });
        self.transcript.truncate_spans(keep);
        for (role, text) in sections.iter().skip(keep) {
            let Some(span_role) = SpanRole::from_tag(role) else {
                continue;
            };
            // The system section is tokenized with the trusted/plain split, the
            // same way the warm walk builds it. Both paths must produce byte-
            // identical tokens or the KV common-prefix probe stops right after
            // the system prompt and every turn re-prefills the conversation.
            let mut toks = if span_role == SpanRole::System {
                self.model
                    .system_message_tokens(text, self.trusted_system_len)
            } else {
                self.model.message_tokens(role, text)
            };
            if self.transcript.is_empty() {
                // First message carries the chat-template preamble (BOS/open).
                let mut begin = self.model.begin_tokens(self.think);
                begin.extend_from_slice(&toks);
                toks = begin;
            }
            self.transcript.push_span(span_role, 0, text.clone(), &toks);
        }
    }

    /// Reconciles the transcript and builds this turn's live prompt: the token
    /// buffer followed by a fresh assistant prefix for `think` (transient — the
    /// prefix is re-derived every turn and only enters the buffer once its reply
    /// is recorded).
    fn build_prompt(&mut self, transcript: &str, think: ThinkMode) -> Ds4TokensGuard {
        self.reconcile(transcript);
        let mut prompt = Ds4TokensGuard::new();
        prompt.push_all(self.transcript.tokens());
        prompt.push_all(&self.model.assistant_prefix_tokens(think));
        prompt
    }

    /// Appends a just-sampled reply to the token transcript as one assistant
    /// span: the assistant prefix it followed, the sampled ids, and the EOS
    /// (sampled but never evaluated) — exactly the token sequence the next
    /// turn's KV common-prefix probe expects. `text` is the trimmed reply text
    /// used as the span's reconciliation key.
    fn record_reply(&mut self, text: String, reply_tokens: &[i32], think: ThinkMode) {
        if reply_tokens.is_empty() {
            return;
        }
        let mut span = self.model.assistant_prefix_tokens(think);
        span.extend_from_slice(reply_tokens);
        // SAFETY: engine valid.
        let eos = unsafe { ffi::ds4_token_eos(self.model.engine) };
        span.push(eos);
        // A reply that follows another assistant span *continues* it: the pass
        // was suspended for an in-pass `/btw` and resumed, and the UI splices
        // both halves into one assistant message. Extend rather than append, so
        // the span index matches the single section the next turn reconciles
        // against — appending instead diverges there and re-prefills the whole
        // conversation (`docs/DOUBLE-BTW.md`). The ids appended are the same
        // either way, so the KV is untouched by the choice.
        if self.transcript.last_is_assistant() {
            self.transcript.extend_last_span(&text, &span);
        } else {
            self.transcript
                .push_span(SpanRole::Assistant, ds4_think(think) as u8, text, &span);
        }
    }
}

impl Engine for Ds4Session {
    /// This machine's own weights, which is what the status bar's blinking brain
    /// reports (see [`crate::status::LocalPass`]).
    fn is_local(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_lines)]
    fn generate(
        &mut self,
        prompt: crate::engine::Prompt<'_>,
        opts: &GenerationOptions,
        interrupt: &dyn Fn() -> bool,
        greedy: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError> {
        let transcript = prompt.flat();
        // Reconcile the append-only token transcript to the rendered transcript
        // and build this turn's prompt (buffer + a fresh assistant prefix). The
        // token buffer is the source of truth; spans that left the transcript
        // (compaction, rollback, sidechain truncation) are dropped by the
        // common-prefix reconciliation in `build_prompt`.
        let tokens = self.build_prompt(transcript, opts.think_mode);
        let prompt_len = tokens.len();

        // Reuse the live session across turns so its cached KV prefix (the
        // constant system prompt and unchanged earlier turns) is not
        // recomputed. Create it lazily on the first turn.
        let session = self.ensure_session()?;

        INTERRUPT.with(|f| f.store(false, Ordering::SeqCst));
        // SAFETY: session valid; cancel_cb reads a thread-local flag.
        unsafe { ffi::ds4_session_set_cancel(session, Some(cancel_cb), std::ptr::null_mut()) };

        // How many prompt tokens match the live KV, and how many the sync below
        // will actually *keep*. These differ: the engine reuses the live KV only
        // when the prompt extends its end, so a prompt that is a strict prefix
        // of the checkpoint (what `/new` produces) matches fully yet is rebuilt
        // from zero. Reporting `common` there primes the bar as complete and
        // hides the whole rebuild — see `engine::reusable_prefix`.
        // SAFETY: session and tokens are valid.
        let common = unsafe { ffi::ds4_session_common_prefix(session, tokens.as_ptr()) };
        // SAFETY: session valid.
        let pos = unsafe { ffi::ds4_session_pos(session) };
        let cached = crate::engine::reusable_prefix(pos, common);
        kv_debug(|| {
            let mut s = format!(
                "generate: prompt={prompt_len} cached={cached} prefill={} ({:.1}% reused)",
                prompt_len - cached,
                f64::from(cached) * 100.0 / f64::from(prompt_len.max(1))
            );
            if cached != common {
                let _ = write!(
                    s,
                    "\n  full rebuild: prompt is a {common}-token prefix of a {pos}-token \
                     live KV; sync cannot rewrite behind the live end"
                );
            }
            s
        });
        // Prime the progress bar so it reflects the cached prefix immediately.
        on_event(EngineEvent::Prefill(PrefillProgress::primed(
            cached, prompt_len,
        )));

        // Prefill: sync the session to the prompt tokens, streaming progress
        // events so the caller can paint a live progress bar.
        let mut err = [0_i8; 512];
        let mut progress = ProgressCtx {
            on_event,
            interrupt,
            start: std::time::Instant::now(),
            base: cached.clamp(0, prompt_len),
            total: prompt_len,
        };
        let progress_ptr = (&raw mut progress).cast::<std::os::raw::c_void>();
        // SAFETY: session valid; progress_cb reads the ProgressCtx we pass,
        // which outlives the sync call, and the callback is cleared right after.
        unsafe {
            ffi::ds4_session_set_display_progress(session, Some(progress_cb), progress_ptr);
        }
        // SAFETY: session, tokens, and err buffer are valid for the call.
        let sync_rc =
            unsafe { ffi::ds4_session_sync(session, tokens.as_ptr(), err.as_mut_ptr(), err.len()) };
        // SAFETY: session valid; clearing the callback before ProgressCtx dies.
        unsafe { ffi::ds4_session_set_display_progress(session, None, std::ptr::null_mut()) };
        let on_event = progress.on_event;
        if sync_rc != 0 {
            if interrupt() || INTERRUPT.with(|f| f.load(Ordering::SeqCst)) {
                return Ok(GenerationStats {
                    interrupted: true,
                    ctx_used: prompt_len,
                    ..GenerationStats::default()
                });
            }
            return Err(EngineError::new(cstr_message(
                &err,
                "prompt processing failed",
            )));
        }

        // SAFETY: session valid.
        let ctx = unsafe { ffi::ds4_session_ctx(session) };
        let pos = unsafe { ffi::ds4_session_pos(session) };
        let room = ctx - pos;
        let mut max_tokens = if opts.n_predict < 0 {
            room - 1
        } else {
            opts.n_predict.min(room - 1)
        };
        max_tokens = max_tokens.max(0);

        // SAFETY: engine valid.
        let eos = unsafe { ffi::ds4_token_eos(self.model.engine) };
        let mut rng: u64 = if opts.seed != 0 {
            opts.seed
        } else {
            0x2545_f491_4f6c_dd1d
        };
        let mut generated = 0;
        let mut reply_tokens: Vec<i32> = Vec::new();
        let mut reply_text = String::new();
        let mut utf8 = crate::engine::Utf8Stream::default();
        // The local chat template opens `<think>` in the prefill prefix unless
        // thinking is off, so generation starts inside the block with no tag of
        // its own — same rule the stream renderer uses.
        let mut recovery = opts.think_tool_recovery.then(|| {
            crate::engine::ThinkToolRecovery::new(!matches!(
                opts.think_mode,
                crate::engine::ThinkMode::Off
            ))
        });
        let start = std::time::Instant::now();
        // Token count and instant at which this pass crossed the warmup, used
        // to report a decode rate that excludes the unrepresentative opening.
        let mut steady_mark: Option<(std::time::Instant, i32)> = None;

        // Speculative decode verifies drafts by argmax, so it reproduces the
        // sampled stream only when the whole generation is greedy anyway.
        // Same gate the C CLI uses: temperature at or below zero, and a
        // support model that proposes blocks rather than single tokens.
        // `greedy()` flipping per token inside a DSML stanza is irrelevant
        // here — at this temperature both branches sample argmax.
        let draft_block = if opts.temperature <= 0.0 {
            // SAFETY: engine valid for the life of the model.
            unsafe { ffi::ds4_engine_mtp_draft_tokens(self.model.engine) }
        } else {
            0
        };
        let speculative = draft_block > 1;

        while generated < max_tokens {
            if steady_mark.is_none()
                && start.elapsed().as_secs_f64() >= crate::engine::STEADY_WARMUP_SECS
            {
                steady_mark = Some((std::time::Instant::now(), generated));
            }
            if interrupt() {
                INTERRUPT.with(|f| f.store(true, Ordering::SeqCst));
                break;
            }
            // Greedy (argmax) sampling while the caller says so — inside a
            // DSML stanza — mirroring the C's `worker_sample_with_mode`.
            let g = greedy();
            // SAFETY: session valid; rng is a valid out-ptr.
            let token = unsafe {
                ffi::ds4_session_sample(
                    session,
                    if g { 0.0 } else { opts.temperature },
                    0,
                    if g { 1.0 } else { opts.top_p },
                    if g { 0.0 } else { opts.min_p },
                    &raw mut rng,
                )
            };
            if token == eos {
                break;
            }
            if speculative {
                let mut accepted = [0_i32; SPEC_ACCEPT_CAP];
                let cap = i32::try_from(accepted.len()).unwrap_or(i32::MAX);
                // SAFETY: session valid; `accepted` is a valid out-buffer of
                // `cap` ints; err buffer valid.
                let n = unsafe {
                    ffi::ds4_session_eval_speculative_argmax(
                        session,
                        token,
                        max_tokens - generated,
                        eos,
                        accepted.as_mut_ptr(),
                        cap,
                        err.as_mut_ptr(),
                        err.len(),
                    )
                };
                if n < 0 {
                    return Err(EngineError::new(cstr_message(&err, "decode failed")));
                }
                // Nothing committed — not even the sampled token — so there is
                // no progress to make and looping again would spin forever.
                if n == 0 {
                    break;
                }
                // `n` is positive here: negative returned above, zero broke.
                let run = &accepted[..usize::try_from(n).unwrap_or(0).min(accepted.len())];
                let mut hit_eos = false;
                for &t in run {
                    if t == eos {
                        hit_eos = true;
                        break;
                    }
                    let text = utf8.push(self.model.token_bytes(t));
                    reply_tokens.push(t);
                    if !text.is_empty() {
                        reply_text.push_str(&text);
                        on_event(EngineEvent::Text(text));
                    }
                    generated += 1;
                    if generated >= max_tokens {
                        break;
                    }
                }
                if hit_eos {
                    break;
                }
            } else {
                // SAFETY: session valid; err buffer valid.
                let eval_rc =
                    unsafe { ffi::ds4_session_eval(session, token, err.as_mut_ptr(), err.len()) };
                if eval_rc != 0 {
                    return Err(EngineError::new(cstr_message(&err, "decode failed")));
                }
                let text = utf8.push(self.model.token_bytes(token));
                reply_tokens.push(token);
                if !text.is_empty() {
                    reply_text.push_str(&text);
                    on_event(EngineEvent::Text(text));
                }
                generated += 1;
            }

            // Live recovery for a tool call opened inside an unclosed
            // `<think>`: force-feed the close and let the model restart the
            // call on the executable side of it.
            //
            // Under speculation this runs once per accepted block rather than
            // once per token, so it can notice the open a few tokens later
            // than the serial path would. It cannot be checked mid-block: the
            // whole run is already committed to the KV cache by the time it
            // returns, so there is nothing to rewind and the injection has to
            // follow the last accepted token either way.
            if let Some(rec) = recovery.as_mut()
                && rec.should_recover(&reply_text)
            {
                let inject = self
                    .model
                    .tokenize_rendered(crate::engine::THINK_RECOVERY_TEXT);
                // SAFETY: session valid.
                let room = unsafe { ffi::ds4_session_ctx(session) - ffi::ds4_session_pos(session) };
                let n = i32::try_from(inject.len()).unwrap_or(i32::MAX);
                if inject.is_empty() || n >= room || generated + n >= max_tokens {
                    // No budget to recover; leave the stream as generated and
                    // let the turn's parse-time fallback deal with it.
                    rec.skipped(reply_text.len());
                } else {
                    for t in &inject {
                        // SAFETY: session valid; err buffer valid.
                        let rc = unsafe {
                            ffi::ds4_session_eval(session, *t, err.as_mut_ptr(), err.len())
                        };
                        if rc != 0 {
                            return Err(EngineError::new(cstr_message(&err, "decode failed")));
                        }
                        reply_tokens.push(*t);
                        generated += 1;
                    }
                    reply_text.push_str(crate::engine::THINK_RECOVERY_TEXT);
                    on_event(EngineEvent::Text(
                        crate::engine::THINK_RECOVERY_TEXT.to_owned(),
                    ));
                    rec.injected(reply_text.len());
                }
            }
        }
        let tail = utf8.flush();
        if !tail.is_empty() {
            reply_text.push_str(&tail);
            on_event(EngineEvent::Text(tail));
        }

        // Append the sampled reply to the token transcript so the next turn's
        // reconciliation keeps these exact tokens verbatim and the live KV
        // prefix reaches this turn's end. The text is recorded *raw*: a
        // suspended pass's partial may later be extended by its continuation,
        // and the two halves have to concatenate into exactly what the UI
        // renders (`docs/DOUBLE-BTW.md` §4.1). `common_prefix` trims when it
        // compares, so a completed reply still matches its rendered section.
        self.record_reply(reply_text, &reply_tokens, opts.think_mode);

        let interrupted = interrupt() || INTERRUPT.with(|f| f.load(Ordering::SeqCst));
        let secs = start.elapsed().as_secs_f64();
        // SAFETY: session valid.
        let ctx_used = unsafe { ffi::ds4_session_pos(session) };
        Ok(GenerationStats {
            generated,
            tps: if secs > 0.0 {
                f64::from(generated) / secs
            } else {
                0.0
            },
            steady_tps: steady_rate(steady_mark, generated),
            ctx_used,
            interrupted,
            usage: None,
        })
    }

    fn generate_aside(
        &mut self,
        prompt: &str,
        opts: &GenerationOptions,
        interrupt: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError> {
        // Step 1 (§4.2): freeze the live main-task KV — transcript, partial
        // reply, and cursor — in a snapshot before touching the session.
        let session = self.ensure_session()?;
        let snapshot = SessionSnapshot::capture(session)?;

        // Step 3: restore is unconditional. The guard reloads the snapshot on
        // every exit path (success, `?` error, interrupt) so an aside can
        // never leave the main task's KV corrupted. Declared before the
        // destructive run and after `snapshot` so it drops first, while the
        // snapshot buffer is still alive.
        let _restore = RestoreOnDrop::new(|| {
            let _ = snapshot.restore(session);
        });

        // Step 4: the aside's tokens must not perturb the main context
        // accounting. The token transcript is the only mutable state `generate`
        // mutates; keep it in place so the aside's reconciliation still reuses
        // the shared transcript prefix, and restore this copy afterwards so the
        // aside's own reply (and the truncation its divergent prompt caused)
        // never leaks into the main task's transcript.
        let saved_transcript = self.transcript.clone();

        // Step 2: answer destructively on the same session. `ds4_session_sync`
        // rolls the cursor back to the common prefix with the frozen KV (the
        // transcript; the partial reply diverges) and prefills only the framed
        // question, then samples the answer. Greedy is forced off (a constant
        // `false` sampler mode) and tool-call denial is the caller's concern
        // (it drops `finished().calls`); the engine simply streams Text.
        let result = self.generate(
            crate::engine::Prompt::Flat(prompt),
            opts,
            interrupt,
            &|| false,
            on_event,
        );

        // Restore the main task's transcript regardless of the aside's
        // outcome; the KV itself is restored when `_restore` drops next.
        self.transcript = saved_transcript;
        result
    }

    fn supports_aside(&self) -> bool {
        true
    }

    fn generate_aside_forked(
        &mut self,
        prompt: &str,
        opts: &GenerationOptions,
        interrupt: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError> {
        // The whole aside runs on a sibling session. Compare `generate_aside`
        // above: no snapshot, no `RestoreOnDrop`, no transcript save/restore —
        // all three exist there only because it answers destructively on the
        // live session. Here nothing of `self` is written, so there is nothing
        // to undo, on any exit path.
        //
        // The fork's KV is freed when it drops at the end of this call.
        let mut aside = self.fork()?;
        aside.generate(
            crate::engine::Prompt::Flat(prompt),
            opts,
            interrupt,
            &|| false,
            on_event,
        )
    }

    fn supports_forked_aside(&self) -> bool {
        true
    }

    fn generate_multiplexed(
        &mut self,
        main_prompt: &str,
        aside_prompt: &str,
        opts: &GenerationOptions,
        interrupt: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(crate::engine::AsideStream, EngineEvent),
        on_stream_end: &mut dyn FnMut(crate::engine::AsideStream),
    ) -> Result<(GenerationStats, GenerationStats), EngineError> {
        use crate::engine::AsideStream;
        use crate::slice::{JobId, SliceRunner};

        // Fork first: if the fork is refused there is nothing to undo, and the
        // caller can still fall back to the freeze/answer/resume path.
        let mut aside = Ds4HostSession::from_session(self.fork()?);

        // Lend our own live session to the rotation. It has to be owned by a
        // `Ds4HostSession` to be sliceable, so it moves out and back; the
        // placeholder is never generated on.
        let placeholder = Self::from_model(Arc::clone(&self.model));
        let mut main = Ds4HostSession::from_session(std::mem::replace(self, placeholder));

        // The runner polls an `AtomicBool` per job; bridge the caller's
        // closure onto one shared flag, since an interrupt means "stop the
        // whole thing" here.
        let stop = Arc::new(AtomicBool::new(false));
        let outcomes = {
            let mut runner = SliceRunner::new(crate::host::DEFAULT_SLICE_TOKENS);
            runner.add(
                &mut main,
                main_prompt.to_owned(),
                opts.clone(),
                Arc::clone(&stop),
            );
            // The aside gets the larger share: it is a short question with the
            // user waiting on it, while the main task is background progress.
            runner.add_weighted(
                &mut aside,
                aside_prompt.to_owned(),
                opts.clone(),
                Arc::clone(&stop),
                crate::slice::ASIDE_SLICE_WEIGHT,
            );
            let stream_of = |id: JobId| {
                if id == JobId(0) {
                    AsideStream::Main
                } else {
                    AsideStream::Aside
                }
            };
            runner.run_with(
                &mut |id, ev| {
                    if interrupt() {
                        stop.store(true, Ordering::Relaxed);
                    }
                    on_event(stream_of(id), ev);
                },
                &mut |id| on_stream_end(stream_of(id)),
            )
        };

        // Take the live session back before surfacing any error, so a failed
        // aside never strands the main task in the placeholder.
        *self = main.into_session();

        let mut it = outcomes.into_iter();
        let main_stats = it.next().expect("job 0 was added").result?;
        let aside_stats = it.next().expect("job 1 was added").result?;
        Ok((main_stats, aside_stats))
    }

    fn supports_multiplexing(&self) -> bool {
        true
    }

    fn warm_reset(&mut self, system: &str) -> Result<(), EngineError> {
        self.warm_tokens =
            self.model
                .build_system_tokens(system, self.trusted_system_len, self.think);
        Ok(())
    }

    fn set_trusted_system_prefix(&mut self, len: usize) {
        if self.trusted_system_len == len {
            return;
        }
        // The system prompt's tokens change, so everything cached below them
        // does too. Same reset as a `max` toggle: drop the token transcript and
        // the live KV rather than letting a common-prefix probe walk into
        // tokens that no longer match.
        self.trusted_system_len = len;
        self.transcript = TokenTranscript::new();
        self.warm_tokens = Ds4TokensGuard::new();
        if !self.session.is_null() {
            // SAFETY: session is non-null and owned by us.
            unsafe { ffi::ds4_session_invalidate(self.session) };
        }
    }

    fn set_think_mode(&mut self, mode: ThinkMode) {
        if self.think == mode {
            return;
        }
        // Only a change of effort preamble changes the prompt prefix; `Off` vs
        // `Medium` lives entirely in the per-turn assistant prefix, which is
        // re-derived every turn and never cached. So a level change that keeps
        // the preamble where it is costs nothing.
        let prefix_changed = self.think.effort_prefix() != mode.effort_prefix();
        self.think = mode;
        if prefix_changed {
            // The whole token buffer now starts with the wrong prefix. Drop it
            // so the next turn retokenizes from text, and invalidate the live
            // KV — the C's `ds4_session_invalidate` in `repl_chat_apply_max_prefix`.
            self.transcript = TokenTranscript::new();
            self.warm_tokens = Ds4TokensGuard::new();
            if !self.session.is_null() {
                // SAFETY: session is non-null and owned by us.
                unsafe { ffi::ds4_session_invalidate(self.session) };
            }
        }
    }

    fn warm_append(&mut self, text: Option<&str>) -> Result<(), EngineError> {
        if let Some(text) = text {
            let msg = self.model.message_tokens("user", text);
            self.warm_tokens.push_all(&msg);
        }
        Ok(())
    }

    fn warm_sync(&mut self, on_event: &mut dyn FnMut(EngineEvent)) -> Result<bool, EngineError> {
        let total = self.warm_tokens.len();
        let session = self.ensure_session()?;
        // SAFETY: session and tokens are valid.
        let cached = unsafe { ffi::ds4_session_common_prefix(session, self.warm_tokens.as_ptr()) };
        if cached >= total {
            return Ok(false);
        }
        on_event(EngineEvent::Prefill(PrefillProgress::primed(cached, total)));
        let mut progress = ProgressCtx {
            on_event: &mut *on_event,
            interrupt: &|| false,
            start: std::time::Instant::now(),
            base: cached.clamp(0, total),
            total,
        };
        let progress_ptr = (&raw mut progress).cast::<std::os::raw::c_void>();
        // SAFETY: session valid; progress outlives the sync and the callback is
        // cleared right after.
        unsafe {
            ffi::ds4_session_set_display_progress(session, Some(progress_cb), progress_ptr);
        }
        let mut err = [0_i8; 512];
        // SAFETY: session, tokens, and err buffer are valid.
        let rc = unsafe {
            ffi::ds4_session_sync(
                session,
                self.warm_tokens.as_ptr(),
                err.as_mut_ptr(),
                err.len(),
            )
        };
        // SAFETY: session valid; clearing before ProgressCtx drops.
        unsafe { ffi::ds4_session_set_display_progress(session, None, std::ptr::null_mut()) };
        if rc != 0 {
            return Err(EngineError::new(cstr_message(
                &err,
                "context prefill failed",
            )));
        }
        Ok(true)
    }

    fn count_tokens(&self, text: &str) -> i32 {
        self.model.count_tokens(text)
    }

    fn ctx_size(&self) -> i32 {
        self.model.ctx_size
    }

    fn model_name(&self) -> String {
        self.model.model_name()
    }

    fn get_kv(&mut self) -> Option<crate::kvcache::KVCache> {
        if self.session.is_null() {
            return None;
        }
        // Reuse the single snapshot primitive (src/snapshot.rs) rather than
        // hand-rolling the FFI: capture owns an engine buffer and frees it on
        // drop; we copy the payload out for the caller to persist. The token
        // transcript rides in the value so a restore reconstructs the exact
        // token buffer the KV was captured with.
        let snap = SessionSnapshot::capture(self.session).ok()?;
        Some(crate::kvcache::KVCache::new(
            snap.as_bytes().to_vec(),
            self.transcript.clone(),
        ))
    }

    fn set_kv(&mut self, cache: &crate::kvcache::KVCache) -> Result<(), EngineError> {
        let session = self.ensure_session()?;
        // Persisted bytes go through the non-owning restore path: the engine
        // copies from a transient struct and never frees it (snapshot.rs,
        // FINDINGS.md).
        SessionSnapshot::restore_bytes(session, cache.kv())?;
        self.transcript = cache.transcript().clone();
        Ok(())
    }
}

impl Drop for Ds4Session {
    fn drop(&mut self) {
        if !self.session.is_null() {
            // SAFETY: the session was created by us and not yet freed. The
            // model (weights + Metal context) is dropped separately when its
            // Arc refcount reaches zero (design §4).
            unsafe { ffi::ds4_session_free(self.session) };
        }
    }
}

/// Resumable per-generation state for the cooperative scheduler (design §5):
/// captured once the suffix prefill completes, then advanced K tokens at a time.
#[derive(Debug)]
struct GenState {
    opts: GenerationOptions,
    rng: u64,
    eos: i32,
    generated: i32,
    max_tokens: i32,
    reply_tokens: Vec<i32>,
    reply_text: String,
    /// Cross-token UTF-8 carry — see [`Ds4Model::token_bytes`].
    utf8: crate::engine::Utf8Stream,
    start: std::time::Instant,
    /// Token count and instant at which this pass crossed the warmup, for the
    /// steady decode rate. `None` until it does.
    steady_mark: Option<(std::time::Instant, i32)>,
}

/// A [`HostSession`] wrapping a [`Ds4Session`] for the shared engine: it runs
/// generation in resumable K-token slices so the scheduler can interleave many
/// sessions on the one GPU thread (design §5). All calls run on that thread.
#[derive(Debug)]
pub struct Ds4HostSession {
    inner: Ds4Session,
    /// The pending request captured by `begin`; prefilled on the first advance.
    pending: Option<(String, GenerationOptions)>,
    /// Active resumable generation state; `None` before prefill / after finish.
    active: Option<GenState>,
}

impl Ds4HostSession {
    /// Wraps an owned session so it can be driven in resumable K-token slices.
    #[must_use]
    pub fn from_session(inner: Ds4Session) -> Self {
        Self {
            inner,
            pending: None,
            active: None,
        }
    }

    /// Unwraps the session, discarding any slicing state. Used to hand a live
    /// session back to its owner after it has been lent to a
    /// [`SliceRunner`](crate::slice::SliceRunner).
    #[must_use]
    pub fn into_session(self) -> Ds4Session {
        self.inner
    }

    /// Non-preemptible suffix prefill (design §5, §11.8). Builds the prompt,
    /// syncs the session, and returns the initial [`GenState`], or `Err(stats)`
    /// if interrupted during prefill.
    fn prefill(
        &mut self,
        transcript: &str,
        opts: &GenerationOptions,
        interrupt: &AtomicBool,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<Result<GenState, GenerationStats>, EngineError> {
        // Reconcile the token transcript and build the prompt (see
        // `Ds4Session::build_prompt`) — the token buffer is the source of truth.
        let tokens = self.inner.build_prompt(transcript, opts.think_mode);
        let prompt_len = tokens.len();
        let session = self.inner.ensure_session()?;

        INTERRUPT.with(|f| f.store(false, Ordering::SeqCst));
        // SAFETY: session valid; cancel_cb reads a thread-local flag.
        unsafe { ffi::ds4_session_set_cancel(session, Some(cancel_cb), std::ptr::null_mut()) };

        // SAFETY: session and tokens are valid.
        let cached = unsafe { ffi::ds4_session_common_prefix(session, tokens.as_ptr()) };
        on_event(EngineEvent::Prefill(PrefillProgress::primed(
            cached, prompt_len,
        )));

        let poll = || interrupt.load(Ordering::SeqCst);
        let mut err = [0_i8; 512];
        let mut progress = ProgressCtx {
            on_event,
            interrupt: &poll,
            start: std::time::Instant::now(),
            base: cached.clamp(0, prompt_len),
            total: prompt_len,
        };
        let progress_ptr = (&raw mut progress).cast::<std::os::raw::c_void>();
        // SAFETY: session valid; progress outlives the sync; cleared right after.
        unsafe {
            ffi::ds4_session_set_display_progress(session, Some(progress_cb), progress_ptr);
        }
        // SAFETY: session, tokens, and err buffer valid.
        let sync_rc =
            unsafe { ffi::ds4_session_sync(session, tokens.as_ptr(), err.as_mut_ptr(), err.len()) };
        // SAFETY: session valid; clearing before ProgressCtx drops.
        unsafe { ffi::ds4_session_set_display_progress(session, None, std::ptr::null_mut()) };
        if sync_rc != 0 {
            if interrupt.load(Ordering::SeqCst) || INTERRUPT.with(|f| f.load(Ordering::SeqCst)) {
                return Ok(Err(GenerationStats {
                    interrupted: true,
                    ctx_used: prompt_len,
                    ..GenerationStats::default()
                }));
            }
            return Err(EngineError::new(cstr_message(
                &err,
                "prompt processing failed",
            )));
        }

        // SAFETY: session valid.
        let ctx = unsafe { ffi::ds4_session_ctx(session) };
        let pos = unsafe { ffi::ds4_session_pos(session) };
        let room = ctx - pos;
        let mut max_tokens = if opts.n_predict < 0 {
            room - 1
        } else {
            opts.n_predict.min(room - 1)
        };
        max_tokens = max_tokens.max(0);
        // SAFETY: engine valid.
        let eos = unsafe { ffi::ds4_token_eos(self.inner.model.engine) };
        Ok(Ok(GenState {
            opts: opts.clone(),
            rng: if opts.seed != 0 {
                opts.seed
            } else {
                0x2545_f491_4f6c_dd1d
            },
            eos,
            generated: 0,
            max_tokens,
            reply_tokens: Vec::new(),
            reply_text: String::new(),
            utf8: crate::engine::Utf8Stream::default(),
            start: std::time::Instant::now(),
            steady_mark: None,
        }))
    }

    /// Builds the terminal stats and appends the sampled reply to the token
    /// transcript (see [`Ds4Session::record_reply`]).
    fn finalize(&mut self, interrupted: bool) -> GenerationStats {
        let mut st = self
            .active
            .take()
            .expect("finalize called without an active generation");
        let mut reply_text = st.reply_text;
        // An incomplete trailing sequence at end of stream decodes lossily;
        // mid-stream characters were already reassembled by the carry.
        reply_text.push_str(&st.utf8.flush());
        self.inner
            .record_reply(reply_text, &st.reply_tokens, st.opts.think_mode);
        let secs = st.start.elapsed().as_secs_f64();
        // SAFETY: session valid (created during prefill).
        let ctx_used = unsafe { ffi::ds4_session_pos(self.inner.session) };
        GenerationStats {
            generated: st.generated,
            tps: if secs > 0.0 {
                f64::from(st.generated) / secs
            } else {
                0.0
            },
            steady_tps: steady_rate(st.steady_mark, st.generated),
            ctx_used,
            interrupted,
            usage: None,
        }
    }
}

impl HostSession for Ds4HostSession {
    fn begin(&mut self, transcript: String, opts: GenerationOptions) {
        // Stash the request; prefill runs on the first advance so it happens on
        // the GPU thread (non-preemptible, §5).
        self.pending = Some((transcript, opts));
        self.active = None;
    }

    fn advance(
        &mut self,
        k: usize,
        interrupt: &AtomicBool,
        sink: &mut dyn FnMut(EngineEvent),
    ) -> Result<Option<GenerationStats>, EngineError> {
        // First slice: run the non-preemptible prefill.
        if self.active.is_none() {
            let Some((transcript, opts)) = self.pending.take() else {
                return Ok(Some(GenerationStats::default()));
            };
            match self.prefill(&transcript, &opts, interrupt, sink)? {
                Ok(state) => self.active = Some(state),
                Err(stats) => return Ok(Some(stats)),
            }
        }

        // Produce up to k tokens, ceding after the slice (§5).
        let mut produced = 0usize;
        loop {
            let st = self.active.as_mut().expect("gen present after prefill");
            if interrupt.load(Ordering::SeqCst) {
                INTERRUPT.with(|f| f.store(true, Ordering::SeqCst));
                return Ok(Some(self.finalize(true)));
            }
            if st.generated >= st.max_tokens {
                return Ok(Some(self.finalize(false)));
            }
            if produced >= k {
                return Ok(None);
            }
            if st.steady_mark.is_none()
                && st.start.elapsed().as_secs_f64() >= crate::engine::STEADY_WARMUP_SECS
            {
                st.steady_mark = Some((std::time::Instant::now(), st.generated));
            }
            let session = self.inner.session;
            // SAFETY: session valid; rng is a valid out-ptr. Greedy is off in
            // the shared/serve path (server samples per opts).
            let token = unsafe {
                ffi::ds4_session_sample(
                    session,
                    st.opts.temperature,
                    0,
                    st.opts.top_p,
                    st.opts.min_p,
                    &raw mut st.rng,
                )
            };
            if token == st.eos {
                return Ok(Some(self.finalize(false)));
            }
            let mut err = [0_i8; 512];
            // SAFETY: session valid; err buffer valid.
            let eval_rc =
                unsafe { ffi::ds4_session_eval(session, token, err.as_mut_ptr(), err.len()) };
            if eval_rc != 0 {
                self.active = None;
                return Err(EngineError::new(cstr_message(&err, "decode failed")));
            }
            let text = st.utf8.push(self.inner.model.token_bytes(token));
            st.reply_tokens.push(token);
            if !text.is_empty() {
                st.reply_text.push_str(&text);
                sink(EngineEvent::Text(text));
            }
            st.generated += 1;
            produced += 1;
        }
    }

    fn ctx_tokens(&self) -> i32 {
        if self.inner.session.is_null() {
            return 0;
        }
        // SAFETY: session valid (created in spawn, freed only on drop).
        unsafe { ffi::ds4_session_pos(self.inner.session) }
    }

    fn snapshot_bytes(&mut self) -> Option<Vec<u8>> {
        self.inner.capture_bytes()
    }

    fn restore_bytes(&mut self, bytes: &[u8]) -> Result<(), EngineError> {
        self.inner.load_bytes(bytes)?;
        // The restored KV supersedes any generation this slot had queued.
        self.pending = None;
        self.active = None;
        Ok(())
    }
}

/// Owns a `Ds4Tokens` value and frees its buffer on drop.
#[derive(Debug)]
struct Ds4TokensGuard(ffi::Ds4Tokens);

impl Ds4TokensGuard {
    fn new() -> Self {
        Self(ffi::Ds4Tokens::default())
    }
    fn as_mut_ptr(&mut self) -> *mut ffi::Ds4Tokens {
        &raw mut self.0
    }
    fn as_ptr(&self) -> *const ffi::Ds4Tokens {
        &raw const self.0
    }
    fn len(&self) -> i32 {
        self.0.len
    }

    /// The token ids as a borrowed slice.
    fn as_slice(&self) -> &[i32] {
        let Ok(len) = usize::try_from(self.0.len) else {
            return &[];
        };
        if self.0.v.is_null() || len == 0 {
            return &[];
        }
        // SAFETY: v points to len valid i32s owned by the ds4 token vector.
        unsafe { std::slice::from_raw_parts(self.0.v, len) }
    }

    /// The token ids from `base` onward — used to isolate one message's tokens
    /// past the chat-template preamble.
    fn slice_from(&self, base: usize) -> &[i32] {
        self.as_slice().get(base..).unwrap_or(&[])
    }

    /// Copies the token ids into an owned `Vec`.
    fn to_vec(&self) -> Vec<i32> {
        self.as_slice().to_vec()
    }

    /// Appends every id in `tokens` to the buffer.
    fn push_all(&mut self, tokens: &[i32]) {
        for &t in tokens {
            // SAFETY: self.0 is a valid ds4 token vector.
            unsafe { ffi::ds4_tokens_push(self.as_mut_ptr(), t) };
        }
    }
}

impl Drop for Ds4TokensGuard {
    fn drop(&mut self) {
        // SAFETY: the token buffer was allocated by ds4 and not yet freed.
        unsafe { ffi::ds4_tokens_free(&raw mut self.0) };
    }
}

/// Points the ds4 Metal kernel loader at the `.metal` files bundled with the
/// build, unless the caller already set the overrides. Without this the loader
/// only searches the current directory and aborts.
fn set_metal_source_env() {
    const KERNELS: [(&str, &str); 19] = [
        ("DS4_METAL_FLASH_ATTN_SOURCE", "flash_attn.metal"),
        ("DS4_METAL_DENSE_SOURCE", "dense.metal"),
        ("DS4_METAL_MOE_SOURCE", "moe.metal"),
        ("DS4_METAL_DSV4_HC_SOURCE", "dsv4_hc.metal"),
        ("DS4_METAL_UNARY_SOURCE", "unary.metal"),
        ("DS4_METAL_DSV4_KV_SOURCE", "dsv4_kv.metal"),
        ("DS4_METAL_DSV4_ROPE_SOURCE", "dsv4_rope.metal"),
        ("DS4_METAL_DSV4_MISC_SOURCE", "dsv4_misc.metal"),
        ("DS4_METAL_ARGSORT_SOURCE", "argsort.metal"),
        ("DS4_METAL_CPY_SOURCE", "cpy.metal"),
        ("DS4_METAL_CONCAT_SOURCE", "concat.metal"),
        ("DS4_METAL_GET_ROWS_SOURCE", "get_rows.metal"),
        ("DS4_METAL_SUM_ROWS_SOURCE", "sum_rows.metal"),
        ("DS4_METAL_SOFTMAX_SOURCE", "softmax.metal"),
        ("DS4_METAL_REPEAT_SOURCE", "repeat.metal"),
        ("DS4_METAL_GLU_SOURCE", "glu.metal"),
        ("DS4_METAL_NORM_SOURCE", "norm.metal"),
        ("DS4_METAL_BIN_SOURCE", "bin.metal"),
        ("DS4_METAL_SET_ROWS_SOURCE", "set_rows.metal"),
    ];
    let dir = metal_source_dir();
    for (var, file) in KERNELS {
        if std::env::var_os(var).is_none() {
            // SAFETY: called once at startup before any threads are spawned.
            unsafe { std::env::set_var(var, dir.join(file)) };
        }
    }
}

/// Resolves the directory holding the bundled `.metal` kernel sources.
///
/// Tried in order: the `DS4_METAL_DIR` environment variable, the path baked
/// in at compile time (valid for local builds), and `../share/plank/metal`
/// relative to the executable (where Homebrew bottles install the kernels —
/// the compile-time path is the CI runner's checkout and doesn't exist on
/// user machines).
fn metal_source_dir() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("DS4_METAL_DIR") {
        return dir.into();
    }
    let built = Path::new(env!("DS4_METAL_DIR"));
    if built.is_dir() {
        return built.to_path_buf();
    }
    if let Ok(exe) = std::env::current_exe().and_then(std::fs::canonicalize)
        && let Some(prefix) = exe.parent().and_then(Path::parent)
    {
        let shared = prefix.join("share").join("plank").join("metal");
        if shared.is_dir() {
            return shared;
        }
    }
    built.to_path_buf()
}

fn cstr_message(buf: &[i8], fallback: &str) -> String {
    if buf.first().copied().unwrap_or(0) == 0 {
        return fallback.to_string();
    }
    // SAFETY: buf is NUL-terminated within its length by the C callee.
    let s = unsafe { CStr::from_ptr(buf.as_ptr()) };
    s.to_string_lossy().into_owned()
}

/// Maps the engine-agnostic think mode to ds4's.
fn ds4_think(think: ThinkMode) -> ffi::Ds4ThinkMode {
    match think {
        ThinkMode::Off => ffi::Ds4ThinkMode::None,
        // `Low` is `HIGH` to the engine — the brevity request lives entirely in
        // the prompt preamble, since the engine has no level below `HIGH`.
        ThinkMode::Low | ThinkMode::Medium => ffi::Ds4ThinkMode::High,
        ThinkMode::Max => ffi::Ds4ThinkMode::Max,
    }
}

/// Skips a legacy `plank-replies-v1` splice-cache header, returning the raw KV
/// snapshot bytes that follow so an old session still restores its KV (the new
/// token transcript starts empty, so the next turn rebuilds it from text — one
/// re-prefill, the C's "rebuilt from text" fallback). Payloads with no magic are
/// already raw KV and pass through unchanged. Kept for one release.
///
/// The legacy layout is: magic, `u32` reply count, then per reply
/// `think:u8 ntok:u32 ntok×i32 text_len:u32 text`. We parse it only to find
/// where the KV begins; the replies themselves are discarded.
fn strip_legacy(bytes: &[u8]) -> &[u8] {
    let Some(rest) = bytes.strip_prefix(ds4tokens::LEGACY_REPLIES_MAGIC) else {
        return bytes;
    };
    let mut pos = 0usize;
    let read_u32 = |pos: &mut usize| -> Option<usize> {
        let b = rest.get(*pos..pos.checked_add(4)?)?;
        *pos += 4;
        Some(u32::from_le_bytes(b.try_into().unwrap()) as usize)
    };
    let skip = |pos: &mut usize, n: usize| -> Option<()> {
        let end = pos.checked_add(n)?;
        if end > rest.len() {
            return None;
        }
        *pos = end;
        Some(())
    };
    let mut parse = || -> Option<usize> {
        let count = read_u32(&mut pos)?;
        for _ in 0..count {
            skip(&mut pos, 1)?; // think
            let ntok = read_u32(&mut pos)?;
            skip(&mut pos, ntok.checked_mul(4)?)?; // token ids
            let text_len = read_u32(&mut pos)?;
            skip(&mut pos, text_len)?; // text
        }
        Some(pos)
    };
    match parse() {
        Some(off) => &rest[off..],
        // Malformed legacy header: fall back to treating the whole payload as
        // raw KV (best effort; a load failure just forces a cold prefill).
        None => bytes,
    }
}

/// Appends a line to the file named by `PLANK_KV_DEBUG`, if set.
///
/// KV prefix reuse is the one part of the engine whose failures are silent and
/// expensive: a mismatch costs a full re-prefill and looks exactly like "the
/// model is slow today". The closure is only called when the variable is set,
/// so this costs an env lookup per turn otherwise. Diagnostic only — nothing
/// reads these files back.
fn kv_debug(f: impl FnOnce() -> String) {
    use std::io::Write as _;
    let Ok(path) = std::env::var("PLANK_KV_DEBUG") else {
        return;
    };
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "{}", f());
    }
}

fn parse_sections(transcript: &str) -> Vec<(&str, String)> {
    let mut out: Vec<(&str, String)> = Vec::new();
    let mut current: Option<&str> = None;
    let mut buf = String::new();
    for line in transcript.split_inclusive('\n') {
        let trimmed = line.trim_end_matches('\n');
        let tag = match trimmed {
            "[system]" => Some("system"),
            "[user]" | "[tool]" => Some("user"),
            "[assistant]" => Some("assistant"),
            _ => None,
        };
        if let Some(role) = tag {
            if let Some(prev) = current.take() {
                out.push((prev, buf.trim_end().to_string()));
                buf.clear();
            }
            current = Some(role);
        } else if current.is_some() {
            buf.push_str(line);
        }
    }
    if let Some(prev) = current {
        out.push((prev, buf.trim_end().to_string()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{parse_sections, strip_legacy};

    #[test]
    fn strip_legacy_skips_replies_header_to_kv() {
        // Build a legacy `plank-replies-v1` payload by hand: magic, count=1,
        // then one reply (think, ntok=2, 2×i32, text_len, text), then raw KV.
        let kv = b"raw-engine-kv-bytes";
        let mut legacy = Vec::new();
        legacy.extend_from_slice(crate::ds4tokens::LEGACY_REPLIES_MAGIC);
        legacy.extend_from_slice(&1u32.to_le_bytes()); // count
        legacy.push(1u8); // think
        legacy.extend_from_slice(&2u32.to_le_bytes()); // ntok
        legacy.extend_from_slice(&7i32.to_le_bytes());
        legacy.extend_from_slice(&9i32.to_le_bytes());
        legacy.extend_from_slice(&3u32.to_le_bytes()); // text_len
        legacy.extend_from_slice(b"abc");
        legacy.extend_from_slice(kv);
        assert_eq!(strip_legacy(&legacy), kv);
        // A payload with no legacy magic is already raw KV: pass-through.
        assert_eq!(strip_legacy(kv), kv);
    }

    /// A resumed pass generates twice into one assistant message, and the
    /// rendered transcript holds it as one section — so the transcript must
    /// hold one span, or reconciliation diverges there and the next prompt
    /// stops extending the live KV (`docs/DOUBLE-BTW.md` §3).
    ///
    /// Exercises the span bookkeeping directly, without a model: what matters
    /// is that a second reply extends rather than appends, and that the merged
    /// key is what `parse_sections` will produce from the merged message.
    #[test]
    fn a_resumed_reply_extends_one_assistant_span() {
        use crate::ds4tokens::{SectionKey, SpanRole, TokenTranscript};

        // The suspended pass records its partial, raw (trailing newline and
        // all), then the resumed pass records the continuation.
        let mut t = TokenTranscript::new();
        t.push_span(SpanRole::User, 0, "count to three".into(), &[1, 2]);
        t.push_span(SpanRole::Assistant, 0, "one\ntwo\n".into(), &[3, 4]);
        assert!(t.last_is_assistant(), "the partial is the last span");
        t.extend_last_span("three\n", &[5, 6]);

        assert_eq!(t.spans().len(), 2, "one reply is one span");

        // What the UI renders next turn: the two halves spliced into one
        // message, whose section key `parse_sections` trims at the end.
        let rendered = "[user]\ncount to three\n[assistant]\none\ntwo\nthree\n";
        let sections: Vec<SectionKey> = parse_sections(rendered)
            .into_iter()
            .filter_map(|(role, text)| {
                SpanRole::from_tag(role).map(|role| SectionKey { role, text })
            })
            .collect();
        assert_eq!(
            t.common_prefix(&sections),
            2,
            "the merged span matches the merged section, so nothing is stale"
        );
    }

    #[test]
    fn splits_role_sections() {
        let t = "[system]\nyou are helpful\n[user]\nhi\n[assistant]\nhello\n";
        let s = parse_sections(t);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], ("system", "you are helpful".to_string()));
        assert_eq!(s[1], ("user", "hi".to_string()));
        assert_eq!(s[2], ("assistant", "hello".to_string()));
    }

    #[test]
    fn tool_maps_to_user() {
        let s = parse_sections("[tool]\nresult text\n");
        assert_eq!(s, vec![("user", "result text".to_string())]);
    }

    /// Contract `kvtier::plan` depends on (#64): a message's trailing
    /// whitespace does not survive the transcript round-trip, so the tiers the
    /// warm path tokenizes are canonicalized to match. Two adjacent user
    /// sections are the session-start shape (stable then volatile) and must
    /// stay two sections, not merge. If this trimming ever changes, the tier
    /// canonicalization in `kvtier::plan` has to change with it or the KV
    /// common-prefix probe silently diverges and re-prefills every turn.
    #[test]
    fn adjacent_user_sections_stay_split_and_lose_trailing_whitespace() {
        let s = parse_sections("[system]\nsys\n[user]\nstable\n\n[user]\nvolatile\n");
        assert_eq!(
            s,
            vec![
                ("system", "sys".to_string()),
                ("user", "stable".to_string()),
                ("user", "volatile".to_string()),
            ]
        );
    }

    // Real-model round-trip proof (BTW-SUSPEND-DESIGN §5.3): snapshot the
    // session mid-reply, run an aside, restore, and assert the main task's
    // continuation is byte-identical to an uninterrupted seeded run — i.e. the
    // snapshot restored the KV losslessly and the aside left it untouched.
    //
    // Requires a loaded model, so it is gated on `ds4_engine` and skips unless
    // PLANK_TEST_MODEL points at a GGUF. It will only run on a Metal box with
    // the `refs/ds4` submodule built.
    // The trusted span of the system prompt must tokenize as *rendered chat*,
    // so the literal `｜DSML｜` in its examples becomes the model's dedicated
    // vocabulary token rather than a spelled-out BPE sequence. Asserted by
    // detokenizing: exactly one token whose text is the whole marker.
    //
    // Requires a loaded model, so it is gated on `ds4_engine` and skips unless
    // PLANK_TEST_MODEL points at a GGUF.
    #[cfg(ds4_engine)]
    #[test]
    fn trusted_system_prefix_emits_the_native_dsml_token() {
        use crate::ffi::Ds4Backend;

        let Some(model) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let engine =
            super::Ds4Session::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap();
        let parts = crate::sysprompt::build_system_prompt_parts("", &[], true);

        let marker = "｜DSML｜".as_bytes();
        let count_marker_tokens = |toks: &[i32]| {
            toks.iter()
                .filter(|&&t| engine.model.token_bytes(t) == marker)
                .count()
        };

        let split = engine
            .model
            .system_message_tokens(&parts.text, parts.trusted_len);
        let plain = engine.model.system_message_tokens(&parts.text, 0);

        assert!(
            count_marker_tokens(&split) > 0,
            "the trusted span must yield native ｜DSML｜ tokens"
        );
        assert_eq!(
            count_marker_tokens(&plain),
            0,
            "plain BPE spells the marker out; that is the bug this fixes"
        );
        eprintln!(
            "system prompt: {} tokens split vs {} plain ({} native ｜DSML｜ tokens)",
            split.len(),
            plain.len(),
            count_marker_tokens(&split)
        );
        // Collapsing each spelled-out marker into one token can only shorten it.
        assert!(
            split.len() < plain.len(),
            "split={} plain={}",
            split.len(),
            plain.len()
        );
    }

    // A session interrupted *mid-thinking* and resumed must reuse its whole KV.
    //
    // The interrupted assistant span is left with an unclosed `<think>` (the
    // turn did not continue, so no `</think>` is appended), and on resume that
    // span is retokenized from text rather than replayed from sampled ids. If
    // the retokenization diverges by one token, the common-prefix probe stops
    // there and the conversation silently re-prefills — the exact symptom
    // reported against 2.7.9, where the exit path saved no payload at all.
    //
    // Requires a loaded model; skips unless PLANK_TEST_MODEL points at a GGUF.
    #[cfg(ds4_engine)]
    #[test]
    fn resume_mid_thinking_reuses_the_whole_kv() {
        use crate::engine::{Engine, ThinkMode};
        use crate::ffi::{self, Ds4Backend};

        let Some(model) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let parts = crate::sysprompt::build_system_prompt_parts("", &[], true);
        // The shape an interrupt mid-thinking leaves behind: the last assistant
        // span opens `<think>` and never closes it.
        let transcript = format!(
            "[system]\n{}\n[user]\nrewrite the list\n[assistant]\n<think>Let me look at the \
             current implementation before I change anything",
            parts.text
        );

        // One model, two sessions: the second launch is modelled by a sibling
        // session over the same weights, which is also what keeps this test from
        // mmapping ~87 GB twice.
        let weights = std::sync::Arc::new(
            super::Ds4Model::open(&model, Ds4Backend::Metal, 8192, 0, 100, &tuning).unwrap(),
        );
        let mut a = super::Ds4Session::from_model(std::sync::Arc::clone(&weights));
        a.set_trusted_system_prefix(parts.trusted_len);

        // Prefill the transcript, then snapshot as the exit path does.
        let toks = a.build_prompt(&transcript, ThinkMode::Medium);
        let prompt_len = toks.len();
        let session = a.ensure_session().unwrap();
        let mut err = [0_i8; 512];
        let rc =
            unsafe { ffi::ds4_session_sync(session, toks.as_ptr(), err.as_mut_ptr(), err.len()) };
        assert_eq!(
            rc,
            0,
            "prefill failed: {}",
            super::cstr_message(&err, "sync")
        );
        let cache = a.get_kv().expect("KV snapshot");

        // A fresh launch restores it and rebuilds the same prompt.
        let mut b = super::Ds4Session::from_model(std::sync::Arc::clone(&weights));
        b.set_trusted_system_prefix(parts.trusted_len);
        b.set_kv(&cache).expect("KV restore");

        let toks2 = b.build_prompt(&transcript, ThinkMode::Medium);
        assert_eq!(
            toks2.as_slice(),
            toks.as_slice(),
            "the resumed prompt must retokenize to the same ids"
        );
        let session_b = b.ensure_session().unwrap();
        let common = unsafe { ffi::ds4_session_common_prefix(session_b, toks2.as_ptr()) };
        eprintln!("restored: {common}/{prompt_len} tokens reused");
        assert_eq!(
            common, prompt_len,
            "resume must reuse the whole KV: {common}/{prompt_len} tokens matched"
        );

        // Control: without the restore the same prompt reuses nothing, so the
        // assertion above is measuring the payload and not something incidental.
        let mut c = super::Ds4Session::from_model(std::sync::Arc::clone(&weights));
        c.set_trusted_system_prefix(parts.trusted_len);
        let toks3 = c.build_prompt(&transcript, ThinkMode::Medium);
        let session_c = c.ensure_session().unwrap();
        let cold = unsafe { ffi::ds4_session_common_prefix(session_c, toks3.as_ptr()) };
        eprintln!("cold:     {cold}/{prompt_len} tokens reused");
        assert_eq!(cold, 0, "a session with no payload starts cold");
    }

    // The warm walk and the per-turn reconcile both tokenize the system prompt.
    // If they ever disagree the KV common-prefix probe stops right after it and
    // every turn re-prefills the whole conversation, silently.
    #[cfg(ds4_engine)]
    #[test]
    fn warm_and_reconcile_tokenize_the_system_prompt_identically() {
        use crate::ffi::Ds4Backend;

        let Some(model) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let engine =
            super::Ds4Session::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap();
        let parts = crate::sysprompt::build_system_prompt_parts("Be terse.", &[], true);

        for think in [
            crate::engine::ThinkMode::Off,
            crate::engine::ThinkMode::Medium,
        ] {
            let warm = engine
                .model
                .build_system_tokens(&parts.text, parts.trusted_len, think)
                .to_vec();
            // What `reconcile` builds for the first (system) section.
            let mut turn = engine.model.begin_tokens(think);
            turn.extend_from_slice(
                &engine
                    .model
                    .system_message_tokens(&parts.text, parts.trusted_len),
            );
            assert_eq!(warm, turn, "{think:?}");
        }
    }

    #[cfg(ds4_engine)]
    #[test]
    fn aside_snapshot_roundtrip_lossless() {
        use crate::engine::{Engine, EngineEvent, GenerationOptions};
        use crate::ffi::Ds4Backend;
        use crate::snapshot::SessionSnapshot;

        let Some(model) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let mut engine =
            super::Ds4Session::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap();

        let opts = GenerationOptions {
            seed: 42,
            n_predict: 24,
            ..GenerationOptions::default()
        };
        let transcript = "[user]\nCount slowly from one to twenty.\n";

        // Establish a real conversation KV first — a valid checkpoint. This is
        // the only state in which an aside can actually fire: `generate_aside`
        // is invoked from the worker *mid-pass*, after the (non-preemptible)
        // prefill, so the session always has a committed KV to snapshot.
        engine
            .generate(
                crate::engine::Prompt::Flat(transcript),
                &opts,
                &|| false,
                &|| false,
                &mut |_| {},
            )
            .unwrap();

        // The lossless invariant this method guarantees (§4.5): after an aside,
        // the session's KV is byte-identical to before it, so resume does zero
        // re-prefill. Capture the frozen state, run an aside, capture again.
        let session = engine.ensure_session().unwrap();
        let before = SessionSnapshot::capture(session)
            .unwrap()
            .as_bytes()
            .to_vec();

        let mut aside = String::new();
        engine
            .generate_aside("[user]\nWhat is 2 plus 2?\n", &opts, &|| false, &mut |e| {
                if let EngineEvent::Text(t) = e {
                    aside.push_str(&t);
                }
            })
            .unwrap();
        assert!(!aside.is_empty(), "aside should have produced text");

        let session = engine.ensure_session().unwrap();
        let after = SessionSnapshot::capture(session)
            .unwrap()
            .as_bytes()
            .to_vec();

        assert_eq!(
            before, after,
            "aside must restore the main-task KV byte-for-byte (zero-drift resume)"
        );
    }

    // Shared-engine attach proofs (design §10). Gated on `ds4_engine` + a real
    // model; inspection-only where no Metal box is available. Verify a freshly
    // attached session generates over the warm prefix, and that two sessions
    // generate independent conversations without KV/output leakage.
    #[cfg(ds4_engine)]
    #[test]
    fn attach_restores_warm_prefix() {
        use crate::ffi::Ds4Backend;
        use crate::host::{EngineHost, HostConfig};
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        let Some(model_path) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let model = super::Ds4Model::open_shared(
            &model_path,
            Ds4Backend::Metal,
            4096,
            0,
            100,
            &tuning,
            "you are a helpful assistant",
        )
        .unwrap();
        let host = EngineHost::new(model, HostConfig::default());
        let s = host.attach().unwrap();
        // A freshly attached session should generate normally over the warm
        // prefix; the runtime proof (prefill count ≈ suffix only) requires
        // instrumenting prefill token counts on a Metal box.
        let stats = s
            .generate(
                "[user]\nSay hi.\n",
                &crate::engine::GenerationOptions {
                    n_predict: 8,
                    ..Default::default()
                },
                Arc::new(AtomicBool::new(false)),
                &mut |_| {},
            )
            .unwrap();
        assert!(stats.generated > 0);
    }

    // Per-client ctx sizing over the real engine (design §7, v2). Inspection-
    // only without a model: a smaller requested ctx_size must be honored by
    // `ds4_session_create`, surface in host status, and still generate.
    #[cfg(ds4_engine)]
    #[test]
    fn attach_sized_honors_smaller_ctx() {
        use crate::ffi::Ds4Backend;
        use crate::host::{EngineHost, HostConfig};
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        let Some(model_path) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let model = super::Ds4Model::open_shared(
            &model_path,
            Ds4Backend::Metal,
            4096,
            0,
            100,
            &tuning,
            "you are a helpful assistant",
        )
        .unwrap();
        let host = EngineHost::new(model, HostConfig::default());
        let s = host.attach_sized(Some(1024)).unwrap();
        // The session's configured ctx is surfaced in status as requested.
        let mut sized = false;
        for _ in 0..200 {
            if host.status().sessions.iter().any(|a| a.ctx_size == 1024) {
                sized = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(
            sized,
            "requested per-session ctx_size is honored + reported"
        );
        let stats = s
            .generate(
                "[user]\nSay hi.\n",
                &crate::engine::GenerationOptions {
                    n_predict: 8,
                    ..Default::default()
                },
                Arc::new(AtomicBool::new(false)),
                &mut |_| {},
            )
            .unwrap();
        assert!(stats.generated > 0);
    }

    #[cfg(ds4_engine)]
    #[test]
    fn two_sessions_no_cross_contamination() {
        use crate::ffi::Ds4Backend;
        use crate::host::{EngineHost, HostConfig};
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        let Some(model_path) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let model = super::Ds4Model::open_shared(
            &model_path,
            Ds4Backend::Metal,
            4096,
            0,
            100,
            &tuning,
            "you are a helpful assistant",
        )
        .unwrap();
        let host = EngineHost::new(model, HostConfig::default());
        let a = host.attach().unwrap();
        let b = host.attach().unwrap();
        let opts = crate::engine::GenerationOptions {
            seed: 1,
            n_predict: 16,
            ..Default::default()
        };
        let mut out_a = String::new();
        a.generate(
            "[user]\nName a color.\n",
            &opts,
            Arc::new(AtomicBool::new(false)),
            &mut |e| {
                if let crate::engine::EngineEvent::Text(t) = e {
                    out_a.push_str(&t);
                }
            },
        )
        .unwrap();
        let mut out_b = String::new();
        b.generate(
            "[user]\nName a country.\n",
            &opts,
            Arc::new(AtomicBool::new(false)),
            &mut |e| {
                if let crate::engine::EngineEvent::Text(t) = e {
                    out_b.push_str(&t);
                }
            },
        )
        .unwrap();
        assert!(!out_a.is_empty() && !out_b.is_empty());
    }

    // Idle reclamation on real Metal (design §7): a session generates, goes idle
    // past the threshold so its KV is snapshotted to disk and reclaimed, then
    // restores transparently on the next request and keeps its context.
    #[cfg(ds4_engine)]
    #[test]
    fn idle_reclaim_restores_on_metal() {
        use crate::ffi::Ds4Backend;
        use crate::host::{EngineHost, HostConfig};
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        use std::time::Duration;

        let Some(model_path) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let model = super::Ds4Model::open_shared(
            &model_path,
            Ds4Backend::Metal,
            4096,
            0,
            100,
            &tuning,
            "you are a helpful assistant",
        )
        .unwrap();
        let host = EngineHost::new(
            model,
            HostConfig {
                max_sessions: 2,
                slice_tokens: crate::host::DEFAULT_SLICE_TOKENS,
                idle_reclaim: Some(Duration::from_millis(200)),
                ..HostConfig::default()
            },
        );
        let s = host.attach().unwrap();
        let opts = crate::engine::GenerationOptions {
            n_predict: 8,
            ..Default::default()
        };
        s.generate(
            "[user]\nSay hi.\n",
            &opts,
            Arc::new(AtomicBool::new(false)),
            &mut |_| {},
        )
        .unwrap();

        // Wait for the scheduler to reclaim the idle session to disk.
        let reclaimed = (0..500).any(|_| {
            if host.status().sessions.iter().any(|x| x.reclaimed) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
            false
        });
        assert!(reclaimed, "idle session must be reclaimed to disk");

        // Next request restores the persisted KV and generates normally.
        let stats = s
            .generate(
                "[user]\nSay bye.\n",
                &opts,
                Arc::new(AtomicBool::new(false)),
                &mut |_| {},
            )
            .unwrap();
        assert!(stats.generated > 0, "restored session generates normally");
        assert!(
            host.status().sessions.iter().all(|x| !x.reclaimed),
            "session is resident again after restore"
        );
    }

    // Test-support: run one generation over a solely-owned `Ds4Session` and
    // return how many prompt tokens this turn still had to evaluate — the
    // priming `Prefill` event's `total`, which `PrefillProgress::primed`
    // defines as `prompt_len - cached`.
    //
    // Reuse is therefore visible as a *small* return value, and a full rebuild
    // as one proportional to the prompt. There is no "reused" count to read
    // directly: `primed` reports `done == 0` by construction (a bar must not
    // read 100% before any work has happened — #64 follow-up), so the reuse
    // assertions below are written as comparisons against a control run.
    #[cfg(ds4_engine)]
    fn gen_capture_prefill(
        engine: &mut super::Ds4Session,
        transcript: &str,
        opts: &crate::engine::GenerationOptions,
    ) -> i32 {
        gen_capture_reply(engine, transcript, opts).0
    }

    /// As [`gen_capture_prefill`], plus the generated reply text — needed to
    /// build the *next* turn's transcript the way the agent does, which is the
    /// only construction under which a restored KV is genuinely reusable (see
    /// `rollback_checkpoint_zero_reprefill`).
    fn gen_capture_reply(
        engine: &mut super::Ds4Session,
        transcript: &str,
        opts: &crate::engine::GenerationOptions,
    ) -> (i32, String) {
        use crate::engine::{Engine, EngineEvent, Prompt};
        let mut remaining: Option<i32> = None;
        let mut reply = String::new();
        engine
            .generate(
                Prompt::Flat(transcript),
                opts,
                &|| false,
                &|| false,
                &mut |e| match e {
                    EngineEvent::Prefill(p) if remaining.is_none() => remaining = Some(p.total),
                    EngineEvent::Text(t) => reply.push_str(&t),
                    EngineEvent::Prefill(_) | EngineEvent::Notice(_) => {}
                },
            )
            .unwrap();
        (
            remaining.expect("generate always primes a Prefill event"),
            reply,
        )
    }

    // #29 — /checkpoint + /rollback reuses the checkpoint KV. Establish a
    // conversation, capture its KV via the snapshot path (the /checkpoint
    // mechanism), diverge onto a different turn, then restore the checkpoint
    // via `set_kv` (the /rollback mechanism) and prove the next generation
    // reuses the whole restored prefix. Runtime signal: the primed Prefill
    // event's reused count.
    //
    // The resumed turn must continue the conversation the way `rollback_to`
    // does — restoring the transcript *with* its assistant turn alongside the
    // KV — because that is the only shape the engine can reuse. Re-running the
    // bare user transcript instead drops the reply span, leaving a prompt that
    // sits *behind* the restored KV's end; `ds4_session_sync` cannot rewrite
    // behind the live end, so it silently rebuilds (see
    // `engine::reusable_prefix` and the strict-prefix assertion below). This
    // test asserted `np <= 2` on that construction for a long time and passed
    // only because the reused count was computed from
    // `ds4_session_common_prefix`, which reports what *matches*, not what will
    // be *kept*.
    #[cfg(ds4_engine)]
    #[test]
    fn rollback_checkpoint_zero_reprefill() {
        use crate::engine::{Engine, GenerationOptions};
        use crate::ffi::Ds4Backend;

        let Some(model) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let mut engine =
            super::Ds4Session::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap();
        let opts = GenerationOptions {
            seed: 42,
            n_predict: 16,
            ..GenerationOptions::default()
        };

        let transcript1 = "[user]\nName a fruit.\n";
        let transcript2 = "[user]\nName a planet.\n";

        // Establish the checkpoint conversation, then capture its KV.
        let (_, reply1) = gen_capture_reply(&mut engine, transcript1, &opts);
        let checkpoint = engine.get_kv().expect("get_kv yields a checkpoint payload");
        let restored_len =
            i32::try_from(checkpoint.transcript().tokens().len()).expect("token count fits i32");
        assert!(
            restored_len > 0,
            "a checkpoint captured after a turn carries that turn's token buffer"
        );

        // Diverge onto a different single-user turn: only the system prompt is
        // reusable, so this turn does real re-prefill (the control).
        let divergent = gen_capture_prefill(&mut engine, transcript2, &opts);
        assert!(
            divergent > 0,
            "a divergent turn must actually prefill its new suffix (remaining={divergent})"
        );

        // Roll back, then continue as `rollback_to` does: the restored
        // transcript carries the assistant turn, so the next prompt *extends*
        // the restored KV and everything before the new question is reused.
        engine.set_kv(&checkpoint).unwrap();
        let resumed = format!(
            "{transcript1}[assistant]\n{}\n[user]\nName a planet.\n",
            reply1.trim()
        );
        let resumed_remaining = gen_capture_prefill(&mut engine, &resumed, &opts);
        // The resumed prompt is *longer* than the checkpoint — it appends the
        // assistant turn and a new question — yet it evaluates only a handful of
        // tokens. That gap is the reuse: the restored prefix was kept, not
        // rebuilt. (The KV is one token shorter than the token transcript: the
        // reply's EOS is sampled and recorded but never evaluated in
        // `record_reply`, so it is among the tokens this turn pays for.)
        assert!(
            resumed_remaining < restored_len,
            "rollback must reuse the restored prefix: the resumed turn evaluates \
             its new suffix ({resumed_remaining}), not the whole {restored_len}-token \
             checkpoint"
        );

        // The sharp edge, asserted so it cannot regress into a silent rebuild:
        // a prompt that sits *behind* the live KV's end reuses nothing at all,
        // so it pays for every one of its own tokens.
        engine.set_kv(&checkpoint).unwrap();
        let behind = gen_capture_prefill(&mut engine, transcript1, &opts);
        assert!(
            behind > 0,
            "a prompt behind the live KV's end is rebuilt from zero, not reused"
        );
    }

    // #12 — /switch payload resume prefills only the new suffix. Capture a
    // session payload (`get_kv`), drop the engine, open a *fresh* engine,
    // restore the payload, and prove the follow-up generation reuses the whole
    // restored prefix (prior transcript) rather than re-prefilling it.
    //
    // As in `rollback_checkpoint_zero_reprefill`, the resumed turn must continue
    // the conversation — a resumed session's transcript carries its assistant
    // turns — so that the prompt *extends* the restored KV. Re-running the bare
    // user transcript leaves the prompt behind the KV's end, which rebuilds.
    #[cfg(ds4_engine)]
    #[test]
    fn switch_payload_resume_suffix_only() {
        use crate::engine::{Engine, GenerationOptions};
        use crate::ffi::Ds4Backend;

        let Some(model) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let opts = GenerationOptions {
            seed: 42,
            n_predict: 16,
            ..GenerationOptions::default()
        };
        let transcript = "[user]\nName a fruit.\n";

        // Only ONE live engine per process: each engine lives in its own scope
        // so its Metal model is fully dropped before the next one opens.
        let (payload, reply) = {
            let mut a =
                super::Ds4Session::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap();
            let (_, reply) = gen_capture_reply(&mut a, transcript, &opts);
            (a.get_kv().expect("get_kv yields a payload"), reply)
        };
        let restored_len =
            i32::try_from(payload.transcript().tokens().len()).expect("token count fits i32");
        // The turn a resumed session actually runs: prior turn plus a new question.
        let resumed = format!(
            "{transcript}[assistant]\n{}\n[user]\nName a planet.\n",
            reply.trim()
        );

        // Cold-baseline control: a fresh engine with an empty KV must prefill
        // the whole prompt (nothing reused). This calibrates "full re-prefill".
        let cold = {
            let mut c =
                super::Ds4Session::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap();
            gen_capture_prefill(&mut c, &resumed, &opts)
        };

        // Fresh engine: restore the payload and continue. It must reuse the
        // restored prefix instead of re-prefilling it.
        let mut b =
            super::Ds4Session::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap();
        b.set_kv(&payload).unwrap();
        let resumed_remaining = gen_capture_prefill(&mut b, &resumed, &opts);
        // Same prompt, two starting states: the resumed session evaluates only
        // the suffix past the payload, the cold one evaluates the whole thing.
        assert!(
            resumed_remaining < cold,
            "resume ({resumed_remaining}) must re-prefill far less than a cold \
             start ({cold}) over the identical prompt"
        );
        assert!(
            resumed_remaining < restored_len,
            "resume reused the payload prefix rather than rebuilding its \
             {restored_len} tokens (remaining={resumed_remaining})"
        );
    }

    // Session cloning on real Metal (docs/SESSION-CLONE-DESIGN.md §5.2). These
    // are the tests that actually decide whether cloning is sound: everything
    // below `EngineHost::attach_clone` is a no-op on `EchoEngine`, so the
    // host-level tests in `host.rs` prove only the plumbing.
    //
    // None of this runs in CI. It compiles there behind `cfg(ds4_engine)` and
    // skips without `PLANK_TEST_MODEL`, so a green CI run is not evidence.
    // Run locally on macOS with a GGUF before trusting the primitive.

    /// P1 + P2 — capture is non-destructive and the clone is faithful. The
    /// source keeps generating normally after being cloned, and the clone
    /// starts from the source's cursor rather than an empty one.
    #[cfg(ds4_engine)]
    #[test]
    fn clone_carries_source_transcript_and_spares_source() {
        use crate::engine::{EngineEvent, GenerationOptions};
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        let Some(host) = clone_test_host(2) else {
            return;
        };
        let a = host.attach().unwrap();
        let opts = GenerationOptions {
            seed: 3,
            n_predict: 12,
            ..GenerationOptions::default()
        };
        let mut reply = String::new();
        a.generate(
            "[user]\nName a fruit.\n",
            &opts,
            Arc::new(AtomicBool::new(false)),
            &mut |e| {
                if let EngineEvent::Text(t) = e {
                    reply.push_str(&t);
                }
            },
        )
        .unwrap();
        assert!(!reply.is_empty(), "the source generates before the clone");

        let b = host.attach_clone(&a).unwrap();

        // Read both cursors from the *same* status snapshot: the scheduler
        // publishes after replying, so comparing across two reads races it.
        let st = poll_host(&host, |s| (s.live_sessions == 2).then(|| s.clone()))
            .expect("the clone attaches alongside A");
        let (source, cloned) = (
            st.sessions.first().expect("A has the older id"),
            st.sessions.last().expect("the clone has the newer id"),
        );
        assert!(source.ctx_tokens > 0, "A holds a real cursor to clone from");
        assert_eq!(
            cloned.ctx_tokens, source.ctx_tokens,
            "the clone holds A's cursor, not a fresh one"
        );
        assert_eq!(
            cloned.ctx_size, source.ctx_size,
            "the clone inherits A's context window"
        );

        // P1: A is undamaged by the capture and keeps generating.
        let mut after = String::new();
        a.generate(
            &format!("[user]\nName a fruit.\n[assistant]\n{reply}\n[user]\nName a planet.\n"),
            &opts,
            Arc::new(AtomicBool::new(false)),
            &mut |e| {
                if let EngineEvent::Text(t) = e {
                    after.push_str(&t);
                }
            },
        )
        .unwrap();
        assert!(!after.is_empty(), "A survives being cloned");
        drop(b);
    }

    /// P3 — the headline. A clone must continue from the source's KV, not
    /// re-prefill it.
    ///
    /// The measurement is `PrefillProgress::total`: the tokens *this pass* has
    /// to evaluate, cached prefix excluded. Reuse therefore shows up as a small
    /// total, and the proof is a direct comparison — the clone and a cold
    /// sibling are given the identical continuation, and the clone must
    /// evaluate far fewer tokens. (`done` cannot serve here: it is rebased to
    /// the cached prefix and starts at 0 by construction.)
    ///
    /// This test also records what the capture cost. Correctness is not enough:
    /// §8 flags capture latency as the risk that decides whether cloning is
    /// usable interactively, and `HostStatus::last_capture` is where that lands.
    #[cfg(ds4_engine)]
    #[test]
    fn clone_avoids_cold_reprefill() {
        use crate::engine::{EngineEvent, GenerationOptions};
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        let Some(host) = clone_test_host(3) else {
            return;
        };
        let a = host.attach().unwrap();
        let opts = GenerationOptions {
            seed: 5,
            n_predict: 12,
            ..GenerationOptions::default()
        };

        let mut reply = String::new();
        a.generate(
            "[user]\nName a fruit.\n",
            &opts,
            Arc::new(AtomicBool::new(false)),
            &mut |e| {
                if let EngineEvent::Text(t) = e {
                    reply.push_str(&t);
                }
            },
        )
        .unwrap();
        assert!(!reply.is_empty(), "the source generates before the clone");
        let continuation =
            format!("[user]\nName a fruit.\n[assistant]\n{reply}\n[user]\nName a planet.\n");

        let b = host.attach_clone(&a).unwrap();
        // `attach_clone` replies from the GPU thread before that thread
        // republishes status, so poll rather than racing it.
        let capture =
            poll_host(&host, |s| s.last_capture).expect("cloning records what the capture cost");
        eprintln!(
            "clone capture: {} bytes in {:?}",
            capture.bytes, capture.elapsed
        );

        let (stats, cloned_total) = prefill_total(&b, &continuation, &opts);
        assert!(stats.generated > 0, "the clone generates normally");

        // Control: a fresh session over the same warm system prompt, given the
        // identical transcript. It has to evaluate the whole conversation.
        let cold = host.attach().unwrap();
        let (_, cold_total) = prefill_total(&cold, &continuation, &opts);

        eprintln!("prefill totals: clone={cloned_total} cold={cold_total}");
        assert!(
            cold_total > 0,
            "the control must prefill something for the comparison to mean anything"
        );
        assert!(
            cloned_total < cold_total,
            "the clone continued A's context instead of re-prefilling it \
             (clone={cloned_total} < cold={cold_total})"
        );
    }

    /// Runs one generation and reports its terminal stats plus the largest
    /// prefill `total` seen — the tokens that pass had to evaluate.
    #[cfg(ds4_engine)]
    fn prefill_total(
        session: &crate::host::SessionHandle,
        transcript: &str,
        opts: &crate::engine::GenerationOptions,
    ) -> (crate::engine::GenerationStats, i32) {
        use crate::engine::EngineEvent;
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        let mut total = 0;
        let stats = session
            .generate(
                transcript,
                opts,
                Arc::new(AtomicBool::new(false)),
                &mut |e| {
                    if let EngineEvent::Prefill(p) = e {
                        total = total.max(p.total);
                    }
                },
            )
            .unwrap();
        (stats, total)
    }

    /// The local fork path, which is the one the interactive agent can actually
    /// use: no `EngineHost`, no scheduler thread, just a `Ds4Session` forking
    /// itself over its own `Arc<Ds4Model>`.
    ///
    /// Proves the same properties as the host-backed clone: the source is
    /// undisturbed (P1), the fork carries its transcript (P2), the fork
    /// continues rather than re-prefilling (P3), and the two diverge without
    /// contaminating each other (P4).
    #[cfg(ds4_engine)]
    #[test]
    fn fork_is_a_local_clone_without_a_host() {
        use crate::engine::GenerationOptions;
        use crate::ffi::Ds4Backend;

        let Some(model) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let mut a =
            super::Ds4Session::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap();
        let opts = GenerationOptions {
            seed: 21,
            n_predict: 16,
            ..GenerationOptions::default()
        };

        let transcript = "[user]\nName a fruit.\n";
        let (_, reply) = gen_capture_reply(&mut a, transcript, &opts);
        assert!(!reply.is_empty(), "the source generates before forking");
        let source_pos = a.ctx_tokens();
        assert!(
            source_pos > 0,
            "the source holds a real cursor to fork from"
        );

        let mut b = a.fork().expect("a live session forks");

        // P1: capture left A's cursor exactly where it was.
        assert_eq!(
            a.ctx_tokens(),
            source_pos,
            "forking must not disturb the source's cursor"
        );
        // P2: B came across at the same position, over the same weights.
        assert_eq!(
            b.ctx_tokens(),
            source_pos,
            "the fork holds A's cursor, not a fresh one"
        );

        let continued = format!("{transcript}[assistant]\n{}\n[user]\n", reply.trim());

        // P3: B continues A's conversation instead of rebuilding it. The cold
        // control is a fresh session over the same model, given the same prompt.
        let b_prompt = format!("{continued}What is the capital of France?\n");
        let b_remaining = gen_capture_prefill(&mut b, &b_prompt, &opts);
        let cold_remaining = {
            let mut c = super::Ds4Session::from_model(a.model_arc());
            gen_capture_prefill(&mut c, &b_prompt, &opts)
        };
        eprintln!("fork prefill: fork={b_remaining} cold={cold_remaining}");
        assert!(
            cold_remaining > 0,
            "the control must prefill something for the comparison to mean anything"
        );
        assert!(
            b_remaining < cold_remaining,
            "the fork continued A's context instead of re-prefilling it \
             (fork={b_remaining} < cold={cold_remaining})"
        );

        // P4: A is unaffected by what B was asked.
        let (_, out_a) = gen_capture_reply(&mut a, &format!("{continued}Name a planet.\n"), &opts);
        assert!(!out_a.is_empty(), "A keeps generating after being forked");
        assert!(
            !out_a.to_lowercase().contains("paris"),
            "B's question must not leak into A: {out_a}"
        );
    }

    /// The aside must *reuse* the main task's KV, not just avoid damaging it.
    /// `drain_btw_inner` frames the question onto the whole rendered
    /// transcript, so the fork's prompt shares everything but the last few
    /// tokens with the KV it was forked from: the aside should prefill the
    /// question and nothing else.
    #[cfg(ds4_engine)]
    #[test]
    fn forked_aside_reuses_the_main_prefix() {
        use crate::engine::{Engine, EngineEvent, GenerationOptions};
        use crate::ffi::Ds4Backend;
        use std::fmt::Write as _;

        let Some(model) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let mut engine =
            super::Ds4Session::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap();
        let opts = GenerationOptions {
            seed: 41,
            n_predict: 16,
            ..GenerationOptions::default()
        };

        // Build a conversation of a few turns, so "reuse the prefix" and
        // "re-prefill everything" are far apart.
        let mut convo = String::new();
        for q in ["Name a fruit.", "Name a color.", "Name a country."] {
            let _ = write!(convo, "[user]\n{q}\n");
            let (_, reply) = gen_capture_reply(&mut engine, &convo, &opts);
            let _ = write!(convo, "[assistant]\n{}\n", reply.trim());
        }

        // Exactly what `drain_btw_inner` builds: the rendered transcript plus
        // the framed question.
        let aside_prompt = format!("{convo}[user]\nWhat is the capital of France?\n");

        let mut aside_remaining = 0;
        engine
            .generate_aside_forked(&aside_prompt, &opts, &|| false, &mut |e| {
                if let EngineEvent::Prefill(p) = e {
                    aside_remaining = aside_remaining.max(p.total);
                }
            })
            .unwrap();

        // Control: the same prompt with no KV to reuse.
        let cold = {
            let mut c = super::Ds4Session::from_model(engine.model_arc());
            gen_capture_prefill(&mut c, &aside_prompt, &opts)
        };
        eprintln!("aside prefill: fork={aside_remaining} cold={cold}");
        assert!(cold > 0, "the control must actually prefill the prompt");
        assert!(
            aside_remaining < cold,
            "the aside must continue the forked KV, not re-prefill the whole \
             conversation (aside={aside_remaining} cold={cold})"
        );
    }

    /// `/btw` without freezing: the main task keeps producing tokens *while*
    /// the aside is answered on a fork, and neither leaks into the other.
    ///
    /// The freeze/answer/resume path emits the main task's tokens strictly
    /// before or after the aside's; this one must interleave them.
    #[cfg(ds4_engine)]
    #[test]
    fn multiplexed_aside_keeps_the_main_task_moving() {
        use crate::engine::{AsideStream, Engine, EngineEvent, GenerationOptions};
        use crate::ffi::Ds4Backend;

        let Some(model) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let mut engine =
            super::Ds4Session::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap();
        let opts = GenerationOptions {
            seed: 81,
            n_predict: 40,
            ..GenerationOptions::default()
        };

        let base = "[user]\nName a fruit.\n";
        let (_, reply) = gen_capture_reply(&mut engine, base, &opts);
        let shared = format!("{base}[assistant]\n{}\n", reply.trim());
        let pos_before = engine.ctx_tokens();

        let mut main_text = String::new();
        let mut aside_text = String::new();
        let mut order: Vec<AsideStream> = Vec::new();
        let (main_stats, aside_stats) = engine
            .generate_multiplexed(
                &format!("{shared}[user]\nCount slowly from one to twenty.\n"),
                &format!("{shared}[user]\nWhat is the capital of France?\n"),
                &opts,
                &|| false,
                &mut |which, ev| {
                    if let EngineEvent::Text(t) = ev {
                        if order.last() != Some(&which) {
                            order.push(which);
                        }
                        match which {
                            AsideStream::Main => main_text.push_str(&t),
                            AsideStream::Aside => aside_text.push_str(&t),
                        }
                    }
                },
                &mut |_| {},
            )
            .unwrap();

        assert!(main_stats.generated > 0, "the main task generated");
        assert!(aside_stats.generated > 0, "the aside generated");
        assert!(!main_text.trim().is_empty() && !aside_text.trim().is_empty());

        // The whole point: the streams hand off to each other repeatedly. A
        // freeze/resume would produce at most two runs.
        eprintln!("stream hand-offs: {}", order.len());
        assert!(
            order.len() > 2,
            "the aside froze the main task instead of interleaving with it \
             (only {} hand-offs)",
            order.len()
        );

        // The aside ran on a fork, so it never entered the main transcript.
        assert!(
            !main_text.to_lowercase().contains("paris"),
            "the aside leaked into the main stream: {main_text}"
        );
        // And the live session advanced past where it started, i.e. the main
        // continuation really ran on it rather than on a copy.
        assert!(
            engine.ctx_tokens() > pos_before,
            "the main task's own KV advanced"
        );
    }

    /// The fan-out this whole line of work is for: one live session forked
    /// several ways, all generating *together* on one thread via
    /// [`SliceRunner`](crate::slice::SliceRunner), each continuing the shared
    /// conversation without re-prefilling it.
    ///
    /// One Metal queue means this is time-slicing, so it does not finish sooner
    /// than running them in series — what it proves is that the streams
    /// genuinely interleave and stay separate.
    #[cfg(ds4_engine)]
    #[test]
    fn forks_interleave_through_the_slice_runner() {
        use crate::engine::{EngineEvent, GenerationOptions};
        use crate::ffi::Ds4Backend;
        use crate::slice::{JobId, SliceRunner};
        use std::collections::HashMap;
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        let Some(model) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let mut engine =
            super::Ds4Session::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap();
        let opts = GenerationOptions {
            seed: 71,
            n_predict: 24,
            ..GenerationOptions::default()
        };

        // A shared conversation for every fork to continue from.
        let base = "[user]\nName a fruit.\n";
        let (_, reply) = gen_capture_reply(&mut engine, base, &opts);
        let shared = format!("{base}[assistant]\n{}\n", reply.trim());

        let questions = [
            "What is the capital of France?",
            "What is the capital of Japan?",
            "What is the capital of Peru?",
        ];
        // The runner borrows, so the forks are owned here for its lifetime.
        let mut forks: Vec<Box<dyn crate::host::HostSession>> = questions
            .iter()
            .map(|_| engine.fork_sliceable().expect("a live session forks"))
            .collect();
        let mut runner = SliceRunner::new(4);
        for (fork, q) in forks.iter_mut().zip(questions) {
            runner.add(
                fork.as_mut(),
                format!("{shared}[user]\n{q}\n"),
                opts.clone(),
                Arc::new(AtomicBool::new(false)),
            );
        }

        let mut text: HashMap<JobId, String> = HashMap::new();
        let mut order: Vec<JobId> = Vec::new();
        let mut prefill: HashMap<JobId, i32> = HashMap::new();
        let outcomes = runner.run(&mut |id, ev| match ev {
            EngineEvent::Text(t) => {
                if order.last() != Some(&id) {
                    order.push(id);
                }
                text.entry(id).or_default().push_str(&t);
            }
            EngineEvent::Prefill(p) => {
                let e = prefill.entry(id).or_default();
                *e = (*e).max(p.total);
            }
            EngineEvent::Notice(_) => {}
        });

        assert_eq!(outcomes.len(), 3);
        for o in &outcomes {
            assert!(
                o.result.as_ref().unwrap().generated > 0,
                "job {:?} generated nothing",
                o.id
            );
            assert!(
                !text[&o.id].trim().is_empty(),
                "job {:?} produced no text",
                o.id
            );
            // Each fork continues the shared conversation, so it pays for its
            // own question and not for the transcript.
            eprintln!("job {:?}: prefill={} ", o.id, prefill[&o.id]);
        }

        // Interleaving: the streams hand off to one another repeatedly rather
        // than each running to completion in turn (which would be 3 runs).
        eprintln!("hand-offs: {}", order.len());
        assert!(
            order.len() > questions.len(),
            "the forks ran serially, not interleaved (only {} hand-offs)",
            order.len()
        );

        // And they stayed separate: no fork answered another's question.
        let paris = text
            .values()
            .filter(|t| t.to_lowercase().contains("paris"))
            .count();
        assert!(
            paris <= 1,
            "the forks contaminated each other: {paris} mention Paris"
        );
    }

    /// The headline for `docs/DOUBLE-BTW.md`: a *second* in-pass aside must
    /// still reuse the prefix. The first one worked all along; the second one
    /// re-prefilled the whole conversation, because the resumed pass appended a
    /// second assistant span while the UI renders one merged section.
    ///
    /// Drives the real shape twice over: freeze, aside, resume, freeze again,
    /// aside again.
    #[cfg(ds4_engine)]
    #[test]
    fn a_second_aside_still_reuses_the_prefix() {
        use crate::engine::{Engine, EngineEvent, GenerationOptions};
        use crate::ffi::Ds4Backend;
        use std::cell::Cell;
        use std::fmt::Write as _;

        let Some(model) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let mut engine =
            super::Ds4Session::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap();
        let opts = GenerationOptions {
            seed: 91,
            n_predict: 64,
            ..GenerationOptions::default()
        };

        let base = "[user]\nCount slowly from one to forty.\n";
        // Text the main task has produced so far, spliced into every prompt the
        // way `run_turn` splices it.
        let mut produced = String::new();
        let mut asides = Vec::new();

        for round in 0..2 {
            // Freeze a pass partway, as a preempting `/btw` does.
            let seen = Cell::new(0);
            let mut prompt = base.to_owned();
            if !produced.is_empty() {
                let _ = write!(prompt, "[assistant]\n{produced}");
            }
            engine
                .generate(
                    crate::engine::Prompt::Flat(&prompt),
                    &opts,
                    &|| seen.get() > 6,
                    &|| false,
                    &mut |e| {
                        if let EngineEvent::Text(t) = e {
                            seen.set(seen.get() + 1);
                            produced.push_str(&t);
                        }
                    },
                )
                .unwrap();
            assert!(!produced.is_empty(), "round {round} produced a partial");

            // The aside, framed over the paused reply.
            let aside_prompt =
                format!("{base}[assistant]\n{produced}\n[user]\nWhat is the capital of France?\n");
            let mut remaining = 0;
            engine
                .generate_aside_forked(&aside_prompt, &opts, &|| false, &mut |e| {
                    if let EngineEvent::Prefill(p) = e {
                        remaining = remaining.max(p.total);
                    }
                })
                .unwrap();
            asides.push(remaining);
        }

        eprintln!("aside prefill by round: {asides:?}");

        // The structural invariant, and the one that actually discriminates:
        // two generations into one assistant message are one span, because the
        // rendered transcript holds them as one section. Appending instead is
        // what makes the next reconciliation diverge.
        let assistant_spans = engine
            .transcript_spans()
            .iter()
            .filter(|s| s.role == crate::ds4tokens::SpanRole::Assistant)
            .count();
        assert_eq!(
            assistant_spans, 1,
            "two resumed generations must be one assistant span, not {assistant_spans}"
        );

        // And the cost follows from it: the second aside prefills its question,
        // not the conversation. Held tight deliberately — with the fix both
        // rounds cost the same, and without it the second grew threefold even
        // on this short transcript.
        assert!(
            asides[1] <= asides[0] + 4,
            "the second aside re-prefilled the conversation instead of reusing \
             it (first={}, second={})",
            asides[0],
            asides[1]
        );
    }

    /// An aside answered while a pass is frozen must not rebuild the whole
    /// prompt. The frozen partial reply is live in the KV *and* recorded in the
    /// token transcript, but the aside's prompt does not contain it, so the
    /// aside's prompt is a strict *prefix* of the live KV — and
    /// `ds4_session_sync` cannot rewrite behind the live end, so a naive run
    /// rebuilds from zero (see `rollback_checkpoint_zero_reprefill`).
    ///
    /// Both tiers are measured here, because the fix has to hold for both.
    #[cfg(ds4_engine)]
    #[test]
    fn aside_during_a_frozen_pass_reuses_the_prefix() {
        use crate::engine::{Engine, EngineEvent, GenerationOptions};
        use crate::ffi::Ds4Backend;
        use std::cell::Cell;

        let Some(model) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let opts = GenerationOptions {
            seed: 61,
            n_predict: 64,
            ..GenerationOptions::default()
        };
        let base = "[user]\nCount slowly from one to thirty.\n";

        // One model, two fresh sessions over it — opening a second Ds4Model in
        // the same process is not supported.
        let shared = std::sync::Arc::new(
            super::Ds4Model::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap(),
        );

        // `forked` selects the tier. `freeze` decides whether the preceding
        // pass is interrupted (leaving a partial reply live in the KV) or runs
        // to completion — the only difference between the two, so any gap in
        // prefill cost is attributable to the freeze.
        let run = |forked: bool, freeze: bool| -> i32 {
            let mut engine = super::Ds4Session::from_model(std::sync::Arc::clone(&shared));
            let seen = Cell::new(0);
            let mut partial = String::new();
            engine
                .generate(
                    crate::engine::Prompt::Flat(base),
                    &opts,
                    &|| freeze && seen.get() > 6,
                    &|| false,
                    &mut |e| {
                        if let EngineEvent::Text(t) = e {
                            seen.set(seen.get() + 1);
                            partial.push_str(&t);
                        }
                    },
                )
                .unwrap();
            assert!(!partial.is_empty(), "the pass produced a reply");

            // `drain_btw_inner` frames the question onto the rendered
            // transcript *plus* the paused pass's partial reply, so the prompt
            // extends the live KV in both cases. Dropping that splice is the
            // bug this test guards: the partial is live in the KV, and a prompt
            // that omits it lands behind the live end and rebuilds from zero.
            let aside_prompt = format!(
                "{base}[assistant]\n{}\n[user]\nWhat is the capital of France?\n",
                partial.trim()
            );

            let mut remaining = 0;
            let sink = &mut |e: EngineEvent| {
                if let EngineEvent::Prefill(p) = e {
                    remaining = remaining.max(p.total);
                }
            };
            if forked {
                engine
                    .generate_aside_forked(&aside_prompt, &opts, &|| false, sink)
                    .unwrap();
            } else {
                engine
                    .generate_aside(&aside_prompt, &opts, &|| false, sink)
                    .unwrap();
            }
            remaining
        };

        for forked in [false, true] {
            let tier = if forked { "forked" } else { "destructive" };
            let clean = run(forked, false);
            let frozen = run(forked, true);
            eprintln!("{tier} aside: after-clean-pass={clean} after-frozen-pass={frozen}");
            assert!(
                frozen <= clean,
                "{tier}: freezing the pass cost the aside its cache — it \
                 prefilled {frozen} tokens where an unfrozen pass costs \
                 {clean}. The partial reply is live in the KV but absent from \
                 the aside's prompt, so the prompt no longer extends the live \
                 end and ds4_session_sync rebuilds from zero."
            );
        }
    }

    /// The suspend/resume shape `run_turn` actually drives: a pass is frozen
    /// mid-generation (partial reply in the KV, *not* in the token transcript),
    /// an aside is answered, then the pass resumes with the partial spliced
    /// back onto the prompt. The resume must cost nothing — that is the whole
    /// point of freezing rather than re-running.
    #[cfg(ds4_engine)]
    #[test]
    fn forked_aside_does_not_cost_the_resume_its_cache() {
        use crate::engine::{Engine, EngineEvent, GenerationOptions};
        use crate::ffi::Ds4Backend;
        use std::cell::Cell;

        let Some(model) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let mut engine =
            super::Ds4Session::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap();
        let opts = GenerationOptions {
            seed: 51,
            n_predict: 64,
            ..GenerationOptions::default()
        };

        let base = "[user]\nCount slowly from one to twenty.\n";

        // Freeze a pass partway: interrupt after a few text events, exactly as
        // a preempting `/btw` does. The partial stays in the KV; nothing is
        // recorded to the transcript.
        let seen = Cell::new(0);
        let mut partial = String::new();
        engine
            .generate(
                crate::engine::Prompt::Flat(base),
                &opts,
                &|| seen.get() > 6,
                &|| false,
                &mut |e| {
                    if let EngineEvent::Text(t) = e {
                        seen.set(seen.get() + 1);
                        partial.push_str(&t);
                    }
                },
            )
            .unwrap();
        assert!(!partial.is_empty(), "the frozen pass produced a partial");

        // The aside runs while the pass is frozen.
        let mut aside_text = String::new();
        engine
            .generate_aside_forked(
                &format!("{base}[user]\nWhat is the capital of France?\n"),
                &opts,
                &|| false,
                &mut |e| {
                    if let EngineEvent::Text(t) = e {
                        aside_text.push_str(&t);
                    }
                },
            )
            .unwrap();
        assert!(!aside_text.is_empty(), "the aside answers");

        // Resume: `run_turn` re-opens the assistant turn with the partial so
        // the engine splices its exact tokens and continues.
        let resume_prompt = format!("{base}[assistant]\n{partial}");
        let remaining = gen_capture_prefill(&mut engine, &resume_prompt, &opts);
        eprintln!(
            "resume after aside: remaining={remaining} partial_chars={}",
            partial.len()
        );

        // A cold session over the identical resume prompt is the calibration:
        // it has to evaluate the whole thing.
        let cold = {
            let mut c = super::Ds4Session::from_model(engine.model_arc());
            gen_capture_prefill(&mut c, &resume_prompt, &opts)
        };
        eprintln!("resume cold control: {cold}");
        assert!(
            remaining < cold,
            "the aside cost the main pass its cache: resuming re-prefilled \
             {remaining} tokens where a cold start pays {cold}"
        );
    }

    /// §6.4's strongest regression: a forked aside must leave the main session
    /// exactly as it found it — the direct analogue of the invariant
    /// `generate_aside` protects with `RestoreOnDrop`, but held structurally
    /// here because the aside never touches the live session at all.
    #[cfg(ds4_engine)]
    #[test]
    fn forked_aside_leaves_the_main_session_clean() {
        use crate::engine::{Engine, GenerationOptions};
        use crate::ffi::Ds4Backend;

        let Some(model) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let mut engine =
            super::Ds4Session::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap();
        let opts = GenerationOptions {
            seed: 31,
            n_predict: 16,
            ..GenerationOptions::default()
        };

        let transcript = "[user]\nName a fruit.\n";
        let (_, reply) = gen_capture_reply(&mut engine, transcript, &opts);
        let pos_before = engine.ctx_tokens();
        let tokens_before = engine.transcript_tokens().to_vec();

        let mut aside_text = String::new();
        engine
            .generate_aside_forked(
                "[user]\nWhat is the capital of France?\n",
                &opts,
                &|| false,
                &mut |e| {
                    if let crate::engine::EngineEvent::Text(t) = e {
                        aside_text.push_str(&t);
                    }
                },
            )
            .unwrap();
        assert!(!aside_text.is_empty(), "the aside produces an answer");

        assert_eq!(
            engine.ctx_tokens(),
            pos_before,
            "the aside must not move the main cursor"
        );
        assert_eq!(
            engine.transcript_tokens(),
            tokens_before,
            "the aside must not touch the main token transcript"
        );

        // And the main task continues as if nothing happened: its next turn
        // still reuses its own prefix rather than rebuilding.
        let continuation = format!(
            "{transcript}[assistant]\n{}\n[user]\nName a planet.\n",
            reply.trim()
        );
        let remaining = gen_capture_prefill(&mut engine, &continuation, &opts);
        let cold = {
            let mut c = super::Ds4Session::from_model(engine.model_arc());
            gen_capture_prefill(&mut c, &continuation, &opts)
        };
        assert!(
            remaining < cold,
            "the main session's KV survived the aside intact \
             (remaining={remaining} < cold={cold})"
        );
    }

    /// P4 — the two sessions diverge without contaminating each other. B is
    /// asked something A never saw; A's own continuation must not mention it,
    /// and both must keep producing output.
    #[cfg(ds4_engine)]
    #[test]
    fn clone_diverges_independently() {
        use crate::engine::{EngineEvent, GenerationOptions};
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        let Some(host) = clone_test_host(2) else {
            return;
        };
        let a = host.attach().unwrap();
        let opts = GenerationOptions {
            seed: 11,
            n_predict: 16,
            ..GenerationOptions::default()
        };
        let shared = "[user]\nName a fruit.\n";
        let mut reply = String::new();
        a.generate(shared, &opts, Arc::new(AtomicBool::new(false)), &mut |e| {
            if let EngineEvent::Text(t) = e {
                reply.push_str(&t);
            }
        })
        .unwrap();

        let b = host.attach_clone(&a).unwrap();
        let prefix = format!("{shared}[assistant]\n{reply}\n");

        let mut out_b = String::new();
        b.generate(
            &format!("{prefix}[user]\nWhat is the capital of France?\n"),
            &opts,
            Arc::new(AtomicBool::new(false)),
            &mut |e| {
                if let EngineEvent::Text(t) = e {
                    out_b.push_str(&t);
                }
            },
        )
        .unwrap();

        let mut out_a = String::new();
        a.generate(
            &format!("{prefix}[user]\nName a planet.\n"),
            &opts,
            Arc::new(AtomicBool::new(false)),
            &mut |e| {
                if let EngineEvent::Text(t) = e {
                    out_a.push_str(&t);
                }
            },
        )
        .unwrap();

        assert!(!out_a.is_empty() && !out_b.is_empty());
        assert!(
            !out_a.to_lowercase().contains("paris"),
            "B's aside must not leak into A's continuation: {out_a}"
        );
    }

    /// Polls host status until `pick` yields a value, or the budget runs out.
    /// Host commands reply from the GPU thread *before* it republishes status,
    /// so a status read right after `attach_clone` races the publish.
    #[cfg(ds4_engine)]
    fn poll_host<T>(
        host: &crate::host::EngineHost,
        pick: impl Fn(&crate::host::HostStatus) -> Option<T>,
    ) -> Option<T> {
        for _ in 0..500 {
            if let Some(v) = pick(&host.status()) {
                return Some(v);
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        None
    }

    /// Opens a shared model and host for the clone tests, or `None` when no
    /// test model is configured. Metal-only; see the module note above.
    #[cfg(ds4_engine)]
    fn clone_test_host(max_sessions: usize) -> Option<crate::host::EngineHost> {
        use crate::ffi::Ds4Backend;
        use crate::host::{EngineHost, HostConfig};

        let model_path = std::env::var_os("PLANK_TEST_MODEL").or_else(|| {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            None
        })?;
        let tuning = crate::config::EngineTuning::default();
        let model = super::Ds4Model::open_shared(
            &model_path,
            Ds4Backend::Metal,
            4096,
            0,
            100,
            &tuning,
            "you are a helpful assistant",
        )
        .unwrap();
        Some(EngineHost::new(
            model,
            HostConfig {
                max_sessions,
                slice_tokens: crate::host::DEFAULT_SLICE_TOKENS,
                idle_reclaim: None,
                ..HostConfig::default()
            },
        ))
    }

    // #28 — idle-reclaim restore keeps context without a cold re-prefill. Via
    // the shared EngineHost: attach, generate (recording the system-prompt
    // reuse baseline), force idle reclamation (KV snapshotted to disk + dropped)
    // with a short idle window, then a follow-up *continuation* restores from
    // disk. Proof: the first Prefill event's reused count exceeds the
    // system-prompt baseline, i.e. the session's own context survived the
    // reclaim and was not re-prefilled.
    #[cfg(ds4_engine)]
    #[test]
    fn idle_reclaim_restore_no_cold_reprefill() {
        use crate::engine::{EngineEvent, GenerationOptions};
        use crate::ffi::Ds4Backend;
        use crate::host::{EngineHost, HostConfig};
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        use std::time::Duration;

        let Some(model_path) = std::env::var_os("PLANK_TEST_MODEL") else {
            eprintln!("skipping: set PLANK_TEST_MODEL to a GGUF to run");
            return;
        };
        let tuning = crate::config::EngineTuning::default();
        let model = super::Ds4Model::open_shared(
            &model_path,
            Ds4Backend::Metal,
            4096,
            0,
            100,
            &tuning,
            "you are a helpful assistant",
        )
        .unwrap();
        let host = EngineHost::new(
            model,
            HostConfig {
                max_sessions: 2,
                slice_tokens: crate::host::DEFAULT_SLICE_TOKENS,
                idle_reclaim: Some(Duration::from_millis(150)),
                ..HostConfig::default()
            },
        );
        let s = host.attach().unwrap();
        let opts = GenerationOptions {
            seed: 7,
            n_predict: 12,
            ..GenerationOptions::default()
        };

        // First turn: just the reply. The reuse measurement comes later, from
        // `total` — see the cold control below.
        let mut reply1 = String::new();
        s.generate(
            "[user]\nName a fruit.\n",
            &opts,
            Arc::new(AtomicBool::new(false)),
            &mut |e| {
                if let EngineEvent::Text(t) = e {
                    reply1.push_str(&t);
                }
            },
        )
        .unwrap();
        assert!(!reply1.is_empty(), "first turn must produce a reply");

        // Force reclamation to disk (short idle window; wait for the scheduler).
        let reclaimed = (0..600).any(|_| {
            if host.status().sessions.iter().any(|x| x.reclaimed) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
            false
        });
        assert!(reclaimed, "idle session must be reclaimed to disk");

        // Follow-up continuation: after restore from disk, the reused prefix
        // must reach past the system prompt into the prior turn + reply.
        let transcript2 =
            format!("[user]\nName a fruit.\n[assistant]\n{reply1}\n[user]\nName a planet.\n");
        let (stats, restored_total) = prefill_total(&s, &transcript2, &opts);
        assert!(stats.generated > 0, "restored session generates normally");

        // Control: a fresh session over the same warm system prompt, given the
        // identical transcript, has to evaluate the whole conversation.
        let cold = host.attach().unwrap();
        let (_, cold_total) = prefill_total(&cold, &transcript2, &opts);
        assert!(
            cold_total > 0,
            "the control must prefill something for the comparison to mean anything"
        );
        assert!(
            restored_total < cold_total,
            "restore kept context beyond the system prompt: no cold re-prefill \
             (restored={restored_total} < cold={cold_total})"
        );
        drop(cold);
        assert!(
            host.status().sessions.iter().all(|x| !x.reclaimed),
            "session is resident again after restore"
        );
    }
}
