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
            prefill_chunk: tuning.prefill_chunk,
            mtp_draft_tokens: tuning.mtp_draft_tokens,
            mtp_margin: tuning.mtp_margin,
            directional_steering_file: as_ptr(&c_steering),
            expert_profile_path: std::ptr::null(),
            directional_steering_attn: tuning.dir_steering_attn,
            directional_steering_ffn: tuning.dir_steering_ffn,
            power_percent,
            ssd_streaming_cache_experts: tuning.ssd_streaming_cache_experts,
            ssd_streaming_cache_bytes: tuning.ssd_streaming_cache_bytes,
            ssd_streaming_preload_experts: tuning.ssd_streaming_preload_experts,
            simulate_used_memory_bytes: tuning.simulate_used_memory_bytes,
            warm_weights: tuning.warm_weights,
            quality: tuning.quality,
            ssd_streaming: tuning.ssd_streaming,
            ssd_streaming_cold: tuning.ssd_streaming_cold,
            inspect_only: false,
            load_slice: false,
            load_layer_start: 0,
            load_layer_end: 0,
            load_output: false,
            distributed: ffi::Ds4DistributedOptions {
                role: 0,
                layers_start: 0,
                layers_end: 0,
                layers_has_output: false,
                layers_set: false,
                listen_host: std::ptr::null(),
                listen_port: 0,
                coordinator_host: std::ptr::null(),
                coordinator_port: 0,
                prefill_chunk: 0,
                prefill_window: 0,
                activation_bits: 0,
                replay_check: false,
                debug: false,
            },
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
    #[allow(clippy::too_many_arguments)]
    pub fn open_shared(
        model_path: impl AsRef<Path>,
        backend: ffi::Ds4Backend,
        ctx_size: i32,
        n_threads: i32,
        power_percent: i32,
        tuning: &crate::config::EngineTuning,
        system: &str,
        checkpoint: Option<&Path>,
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
        // Restore a matching on-disk checkpoint when there is one; otherwise
        // prefill the system prompt and persist a fresh checkpoint. Best-effort
        // on both ends: a cache miss only costs one prefill.
        let fingerprint = model.checkpoint_fingerprint(system);
        let restored = checkpoint
            .and_then(|path| crate::kvcache::KVCache::from_file(path, &fingerprint))
            .is_some_and(|cache| boot.set_kv(&cache).is_ok());
        if !restored {
            boot.warm_reset(system)?;
            boot.warm_sync(None, &mut |_| {})?;
            if let Some(path) = checkpoint
                && let Some(cache) = boot.get_kv()
            {
                let _ = cache.persist(path, &fingerprint);
            }
        }
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
    fn build_system_tokens(&self, system: &str) -> Ds4TokensGuard {
        let mut tokens = Ds4TokensGuard::new();
        // SAFETY: engine and tokens are valid for the whole build.
        unsafe { ffi::ds4_chat_begin(self.engine, tokens.as_mut_ptr()) };
        if let (Ok(role), Ok(content)) = (CString::new("system"), CString::new(system)) {
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
        tokens
    }

    /// Chat-template preamble tokens emitted by `ds4_chat_begin` (BOS / opening
    /// template), prepended once ahead of the first message. Folded into the
    /// first span of the token transcript so the buffer is self-contained.
    fn begin_tokens(&self) -> Vec<i32> {
        let mut tokens = Ds4TokensGuard::new();
        // SAFETY: engine and tokens are valid for the call.
        unsafe { ffi::ds4_chat_begin(self.engine, tokens.as_mut_ptr()) };
        tokens.to_vec()
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

    /// Fingerprint tying a checkpoint to this exact model and system prompt —
    /// Tier 1 of the KV cache chain, and the parent every deeper tier's key
    /// hangs off. Shared with the tier planner so the two cannot disagree.
    fn checkpoint_fingerprint(&self, system: &str) -> String {
        crate::kvtier::system_fingerprint(&self.model_name(), system)
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
        }
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
            let mut toks = self.model.message_tokens(role, text);
            if self.transcript.is_empty() {
                // First message carries the chat-template preamble (BOS/open).
                let mut begin = self.model.begin_tokens();
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
        self.transcript
            .push_span(SpanRole::Assistant, ds4_think(think) as u8, text, &span);
    }
}

impl Engine for Ds4Session {
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

        // How many prompt tokens the live KV already holds; the sync below only
        // evaluates the suffix beyond this cached prefix.
        // SAFETY: session and tokens are valid.
        let cached = unsafe { ffi::ds4_session_common_prefix(session, tokens.as_ptr()) };
        kv_debug(|| {
            format!(
                "generate: prompt={prompt_len} cached={cached} prefill={} ({:.1}% reused)",
                prompt_len - cached,
                f64::from(cached) * 100.0 / f64::from(prompt_len.max(1))
            )
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
        let start = std::time::Instant::now();

        while generated < max_tokens {
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
        let tail = utf8.flush();
        if !tail.is_empty() {
            reply_text.push_str(&tail);
            on_event(EngineEvent::Text(tail));
        }

        // Append the sampled reply to the token transcript so the next turn's
        // reconciliation keeps these exact tokens verbatim (its span key is the
        // trimmed reply text) and the live KV prefix reaches this turn's end.
        reply_text.truncate(reply_text.trim_end().len());
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

    fn warm_reset(&mut self, system: &str) -> Result<(), EngineError> {
        self.warm_tokens = self.model.build_system_tokens(system);
        Ok(())
    }

    fn warm_sync(
        &mut self,
        text: Option<&str>,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<bool, EngineError> {
        if let Some(text) = text {
            let msg = self.model.message_tokens("user", text);
            self.warm_tokens.push_all(&msg);
        }
        let total = self.warm_tokens.len();
        let session = self.ensure_session()?;
        // SAFETY: session and tokens are valid.
        let cached = unsafe { ffi::ds4_session_common_prefix(session, self.warm_tokens.as_ptr()) };
        if cached >= total {
            return Ok(false);
        }
        on_event(EngineEvent::Prefill(PrefillProgress {
            done: cached.clamp(0, (total - 1).max(0)),
            total,
            tps: 0.0,
        }));
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
        on_event(EngineEvent::Prefill(PrefillProgress {
            done: cached.clamp(0, (prompt_len - 1).max(0)),
            total: prompt_len,
            tps: 0.0,
        }));

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
        reply_text.truncate(reply_text.trim_end().len());
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
        if self.inner.session.is_null() {
            return None;
        }
        // Reuse the single snapshot primitive; copy out an owned buffer so the
        // engine-owned snapshot is freed here (the returned Vec is Rust's, and
        // the disk-read restore path must never free it — FINDINGS double-free).
        // The token transcript rides in front of the KV bytes (same wrap as
        // `get_kv`) so an idle-reclaimed session resumes with its exact
        // token buffer intact and only prefills genuinely new tokens.
        let snap = SessionSnapshot::capture(self.inner.session).ok()?;
        let mut out = ds4tokens::encode(&self.inner.transcript);
        out.extend_from_slice(snap.as_bytes());
        Some(out)
    }

    fn restore_bytes(&mut self, bytes: &[u8]) -> Result<(), EngineError> {
        // Restore into the freshly spawned session (already warmed with the
        // shared system-prompt prefix). Disk-read bytes use the non-owning FFI
        // path so the engine never frees Rust's buffer (FINDINGS double-free).
        let session = self.inner.ensure_session()?;
        let (transcript, kv) = match ds4tokens::decode(bytes) {
            Some((transcript, off)) => (transcript, &bytes[off..]),
            None => (TokenTranscript::new(), strip_legacy(bytes)),
        };
        SessionSnapshot::restore_bytes(session, kv)?;
        // The restored KV/cursor and its captured token transcript supersede any
        // prior Rust-side transcript.
        self.inner.transcript = transcript;
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
        ThinkMode::Auto | ThinkMode::On => ffi::Ds4ThinkMode::High,
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
            None,
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
            None,
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
            None,
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
            None,
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
    // return `(new_prefill, cached)` derived from the *first* `Prefill` event.
    // `generate` always primes one such event up front with `done == cached`
    // (tokens reused from the live KV prefix) and `total == prompt_len`, so
    // `new_prefill = total - done` is the count of tokens actually evaluated
    // this turn (§KV-cache discipline). This is the reuse signal the
    // zero-re-prefill assertions below key off of; no production change needed.
    #[cfg(ds4_engine)]
    fn gen_capture_prefill(
        engine: &mut super::Ds4Session,
        transcript: &str,
        opts: &crate::engine::GenerationOptions,
    ) -> (i32, i32) {
        use crate::engine::{Engine, EngineEvent, Prompt};
        let mut first: Option<(i32, i32)> = None;
        engine
            .generate(
                Prompt::Flat(transcript),
                opts,
                &|| false,
                &|| false,
                &mut |e| {
                    if let EngineEvent::Prefill(p) = e
                        && first.is_none()
                    {
                        first = Some((p.done, p.total));
                    }
                },
            )
            .unwrap();
        let (done, total) = first.expect("generate always primes a Prefill event");
        (total - done, done)
    }

    // #29 — /checkpoint + /rollback does zero re-prefill. Establish a
    // conversation, capture its KV via the snapshot path (the /checkpoint
    // mechanism), diverge onto a different turn, then restore the checkpoint
    // via `set_kv` (the /rollback mechanism) and prove the next generation
    // reuses the checkpoint KV wholesale instead of re-prefilling the
    // transcript. Runtime signal: the first Prefill event's reused count.
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
        let _ = gen_capture_prefill(&mut engine, transcript1, &opts);
        let checkpoint = engine.get_kv().expect("get_kv yields a checkpoint payload");

        // Diverge onto a different single-user turn: only the system prompt is
        // reusable, so this turn does real re-prefill (the control).
        let (np_divergent, _cached_div) = gen_capture_prefill(&mut engine, transcript2, &opts);
        assert!(
            np_divergent > 0,
            "a divergent turn must actually prefill its new suffix (np={np_divergent})"
        );

        // Roll back to the checkpoint, then re-run the checkpoint's own turn.
        // The live KV already holds these exact tokens, so re-prefill ~ 0.
        engine.set_kv(&checkpoint).unwrap();
        let (np_restored, _cached_r) = gen_capture_prefill(&mut engine, transcript1, &opts);
        assert!(
            np_restored <= 2,
            "rollback must reuse the checkpoint KV (new prefill ~0, got {np_restored})"
        );
        assert!(
            np_restored < np_divergent,
            "rollback re-prefill ({np_restored}) must be far below a real prefill ({np_divergent})"
        );
    }

    // #12 — /switch payload resume prefills only the new suffix. Capture a
    // session payload (get_kv bytes), drop the engine, open a *fresh*
    // engine, restore the bytes, and prove the follow-up generation reuses the
    // whole restored prefix (system prompt + prior transcript) rather than
    // re-prefilling it: reused ~ prompt_len, new prefill ~ 0.
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
        let payload = {
            let mut a =
                super::Ds4Session::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap();
            let _ = gen_capture_prefill(&mut a, transcript, &opts);
            a.get_kv().expect("get_kv yields a payload")
        };

        // Cold-baseline control: a fresh engine with an empty KV must prefill
        // the whole prompt (nothing reused). This calibrates "full re-prefill".
        let (np_cold, cached_cold) = {
            let mut c =
                super::Ds4Session::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap();
            gen_capture_prefill(&mut c, transcript, &opts)
        };

        // Fresh engine: restore the payload and re-run the same turn. It must
        // reuse the restored prefix instead of re-prefilling it.
        let mut b =
            super::Ds4Session::open(&model, Ds4Backend::Metal, 4096, 0, 100, &tuning).unwrap();
        b.set_kv(&payload).unwrap();
        let (np, cached) = gen_capture_prefill(&mut b, transcript, &opts);
        assert!(
            np <= 2,
            "resumed session prefills only the suffix, not the transcript (np={np})"
        );
        // Against the cold baseline: the resumed run re-prefills far less and
        // reuses far more KV, proving the payload's prefix was NOT recomputed.
        assert!(
            np < np_cold,
            "resume ({np}) must re-prefill far less than a cold start ({np_cold})"
        );
        assert!(
            cached > cached_cold,
            "resume reuses the payload KV; cold start reuses none \
             (cached={cached} > cold={cached_cold})"
        );
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
            None,
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

        // First turn: record the system-prompt reuse baseline + the reply.
        let mut reply1 = String::new();
        let mut first1: Option<(i32, i32)> = None;
        s.generate(
            "[user]\nName a fruit.\n",
            &opts,
            Arc::new(AtomicBool::new(false)),
            &mut |e| match e {
                EngineEvent::Prefill(p) => {
                    if first1.is_none() {
                        first1 = Some((p.done, p.total));
                    }
                }
                EngineEvent::Text(t) => reply1.push_str(&t),
                EngineEvent::Notice(_) => {}
            },
        )
        .unwrap();
        let (sys_cached, _t1) = first1.expect("first turn primes a Prefill event");
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
        let mut first2: Option<(i32, i32)> = None;
        let stats = s
            .generate(
                &transcript2,
                &opts,
                Arc::new(AtomicBool::new(false)),
                &mut |e| {
                    if let EngineEvent::Prefill(p) = e
                        && first2.is_none()
                    {
                        first2 = Some((p.done, p.total));
                    }
                },
            )
            .unwrap();
        let (cached2, _t2) = first2.expect("restored turn primes a Prefill event");
        assert!(stats.generated > 0, "restored session generates normally");
        assert!(
            cached2 > sys_cached,
            "restore kept context beyond the system prompt: no cold re-prefill \
             (cached2={cached2} > sys_baseline={sys_cached})"
        );
        assert!(
            host.status().sessions.iter().all(|x| !x.reclaimed),
            "session is resident again after restore"
        );
    }
}
