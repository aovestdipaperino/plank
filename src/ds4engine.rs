//! Safe wrapper around the ds4 C engine implementing [`Engine`].
//!
//! Present only under the `ds4_engine` cfg (macOS + built `ds4-ref` submodule).
//! The transcript arriving from the UI is plain role-tagged text; this wrapper
//! reparses it into ds4 chat-template tokens, prefills a session, and samples.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::engine::{
    Engine, EngineError, EngineEvent, GenerationOptions, GenerationStats, PrefillProgress,
    ThinkMode,
};
use crate::ffi;

/// Loaded ds4 model plus a live session reused across turns.
///
/// The session is kept alive so `ds4_session_sync` reuses the cached KV prefix
/// (the constant system prompt and any unchanged earlier turns) and only
/// evaluates the new suffix each turn.
#[derive(Debug)]
pub struct Ds4Engine {
    engine: *mut ffi::Ds4Engine,
    session: *mut ffi::Ds4Session,
    ctx_size: i32,
    last_reply: Option<LastReply>,
    /// Chat-template token overhead of an empty message (`-1` until measured),
    /// subtracted by `count_tokens` so it returns just the text's tokens.
    count_overhead: std::cell::Cell<i32>,
}

/// The most recent generated reply, kept so the next prompt can splice the
/// exact sampled token sequence for that assistant turn instead of
/// re-templating its text. Retokenized text is not guaranteed to reproduce
/// the sampled tokens, and a mismatch invalidates the live KV prefix at the
/// start of the reply, forcing it to be re-prefilled on the follow-up turn.
#[derive(Debug)]
struct LastReply {
    /// Concatenated token text, trailing-whitespace-trimmed to compare
    /// against `parse_sections` output.
    text: String,
    /// The sampled token IDs, exactly as evaluated into the KV.
    tokens: Vec<i32>,
    /// Think mode of the assistant prefix the tokens followed.
    think: ffi::Ds4ThinkMode,
}

// SAFETY: the engine is used single-threaded by the agent turn loop; the
// pointer owns the model and is freed on drop. Send lets it move into the
// boxed trait object.
unsafe impl Send for Ds4Engine {}

thread_local! {
    static INTERRUPT: AtomicBool = const { AtomicBool::new(false) };
}

unsafe extern "C" fn cancel_cb(_ud: *mut std::os::raw::c_void) -> bool {
    INTERRUPT.with(|f| f.load(Ordering::SeqCst))
}

/// Bridges ds4's C display-progress callback to the Rust event sink.
///
/// `base` is the count of cached tokens reused from the live KV prefix, so the
/// progress bar starts partially filled and reflects the whole prompt.
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
    // `base + cur` can overshoot `total` when the backend re-evaluates tokens
    // the common-prefix probe counted as cached. Grow the estimated total with
    // ~5% headroom on overshoot so the bar keeps advancing smoothly instead of
    // parking at 100% while prefill is still going.
    let done = (ctx.base + cur).max(0);
    if done >= ctx.total {
        ctx.total = done + (ctx.total / 20).max(1);
    }
    let secs = ctx.start.elapsed().as_secs_f64();
    let tps = if secs > 0.0 {
        f64::from(cur) / secs
    } else {
        0.0
    };
    (ctx.on_event)(EngineEvent::Prefill(PrefillProgress {
        done,
        total: ctx.total,
        tps,
    }));
}

impl Ds4Engine {
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
            session: std::ptr::null_mut(),
            ctx_size,
            last_reply: None,
            count_overhead: std::cell::Cell::new(-1),
        })
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

    /// Ensures a live session exists, creating it lazily.
    fn ensure_session(&mut self) -> Result<*mut ffi::Ds4Session, EngineError> {
        if self.session.is_null() {
            let mut session: *mut ffi::Ds4Session = std::ptr::null_mut();
            // SAFETY: engine valid; session is a valid out-ptr.
            let rc =
                unsafe { ffi::ds4_session_create(&raw mut session, self.engine, self.ctx_size) };
            if rc != 0 || session.is_null() {
                return Err(EngineError::new("failed to create session"));
            }
            self.session = session;
        }
        Ok(self.session)
    }

    /// Serializes the session KV to `path`, prefixed by its fingerprint line.
    ///
    /// Best-effort: a failure to save just means the next launch re-prefills.
    fn save_checkpoint(session: *mut ffi::Ds4Session, path: &Path, fingerprint: &str) {
        let mut snap = ffi::Ds4SessionSnapshot::default();
        let mut err = [0_i8; 512];
        // SAFETY: session valid; snap is a valid out-ptr the engine fills.
        let rc = unsafe {
            ffi::ds4_session_save_snapshot(session, &raw mut snap, err.as_mut_ptr(), err.len())
        };
        if rc == 0 && !snap.ptr.is_null() {
            let len = usize::try_from(snap.len).unwrap_or(0);
            // SAFETY: snap.ptr points to snap.len bytes owned by the engine.
            let bytes = unsafe { std::slice::from_raw_parts(snap.ptr, len) };
            let mut file = Vec::with_capacity(bytes.len() + 41);
            file.extend_from_slice(fingerprint.as_bytes());
            file.push(b'\n');
            file.extend_from_slice(bytes);
            let _ = std::fs::write(path, &file);
        }
        // SAFETY: snap was filled by ds4_session_save_snapshot.
        unsafe { ffi::ds4_session_snapshot_free(&raw mut snap) };
    }

    /// Fingerprint tying a checkpoint to this exact model and system prompt.
    fn checkpoint_fingerprint(&self, system: &str) -> String {
        let mut data = self.model_name().into_bytes();
        data.push(0);
        data.extend_from_slice(system.as_bytes());
        crate::session::sha1_hex(&data)
    }

    /// Builds chat-template tokens from a role-tagged transcript.
    ///
    /// The transcript uses `[system]`/`[user]`/`[assistant]` section markers
    /// produced by the UI; each section becomes a ds4 chat message.
    fn build_tokens(&self, transcript: &str, think: ThinkMode) -> Ds4TokensGuard {
        let mut tokens = Ds4TokensGuard::new();
        // SAFETY: engine and tokens are valid for the whole build.
        unsafe { ffi::ds4_chat_begin(self.engine, tokens.as_mut_ptr()) };
        let sections = parse_sections(transcript);
        // Splice the exact sampled tokens for the reply generated last turn
        // (the final assistant section, when its text matches) so the KV
        // common-prefix probe reaches through it instead of diverging on a
        // retokenization of its text. Earlier assistant turns were prefilled
        // from re-templated text already, so they stay text-rendered.
        let splice_idx = self.last_reply.as_ref().and_then(|last| {
            sections
                .iter()
                .rposition(|(role, _)| *role == "assistant")
                .filter(|&i| sections[i].1 == last.text)
        });
        for (i, (role, content)) in sections.iter().enumerate() {
            if Some(i) == splice_idx
                && let Some(last) = &self.last_reply
            {
                Self::append_reply_tokens(self.engine, &mut tokens, last);
                continue;
            }
            let (Ok(c_role), Ok(c_content)) = (CString::new(*role), CString::new(content.clone()))
            else {
                continue;
            };
            // SAFETY: role/content strings outlive the call.
            unsafe {
                ffi::ds4_chat_append_message(
                    self.engine,
                    tokens.as_mut_ptr(),
                    c_role.as_ptr(),
                    c_content.as_ptr(),
                );
            }
        }
        // SAFETY: engine and tokens valid.
        unsafe {
            ffi::ds4_chat_append_assistant_prefix(
                self.engine,
                tokens.as_mut_ptr(),
                ds4_think(think),
            );
        }
        tokens
    }

    /// Appends the last generated reply as its exact sampled token sequence:
    /// the assistant prefix it followed, the sampled tokens, and the closing
    /// EOS (which generation sampled but never evaluated).
    fn append_reply_tokens(
        engine: *mut ffi::Ds4Engine,
        tokens: &mut Ds4TokensGuard,
        last: &LastReply,
    ) {
        // SAFETY: engine and tokens valid; think matches generation.
        unsafe { ffi::ds4_chat_append_assistant_prefix(engine, tokens.as_mut_ptr(), last.think) };
        for &t in &last.tokens {
            // SAFETY: tokens is a valid ds4 token vector.
            unsafe { ffi::ds4_tokens_push(tokens.as_mut_ptr(), t) };
        }
        // SAFETY: engine valid; tokens is a valid ds4 token vector.
        unsafe {
            let eos = ffi::ds4_token_eos(engine);
            ffi::ds4_tokens_push(tokens.as_mut_ptr(), eos);
        }
    }

    fn token_text(&self, token: i32) -> String {
        let mut len: usize = 0;
        // SAFETY: engine valid; len is a valid out-ptr.
        let p = unsafe { ffi::ds4_token_text(self.engine, token, &raw mut len) };
        if p.is_null() {
            return String::new();
        }
        // SAFETY: p points to len bytes owned by us; we copy then free.
        let bytes = unsafe { std::slice::from_raw_parts(p.cast::<u8>(), len) };
        let text = String::from_utf8_lossy(bytes).into_owned();
        // SAFETY: p was allocated by ds4_token_text for the caller to free.
        unsafe { libc::free(p.cast()) };
        text
    }
}

impl Engine for Ds4Engine {
    #[allow(clippy::too_many_lines)]
    fn generate(
        &mut self,
        transcript: &str,
        opts: &GenerationOptions,
        interrupt: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError> {
        let tokens = self.build_tokens(transcript, opts.think_mode);
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
        // Prime the progress bar so it reflects the cached prefix immediately.
        on_event(EngineEvent::Prefill(PrefillProgress {
            done: cached.clamp(0, (prompt_len - 1).max(0)),
            total: prompt_len,
            tps: 0.0,
        }));

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
        let eos = unsafe { ffi::ds4_token_eos(self.engine) };
        let mut rng: u64 = if opts.seed != 0 {
            opts.seed
        } else {
            0x2545_f491_4f6c_dd1d
        };
        let mut generated = 0;
        let mut reply_tokens: Vec<i32> = Vec::new();
        let mut reply_text = String::new();
        let start = std::time::Instant::now();

        while generated < max_tokens {
            if interrupt() {
                INTERRUPT.with(|f| f.store(true, Ordering::SeqCst));
                break;
            }
            // SAFETY: session valid; rng is a valid out-ptr.
            let token = unsafe {
                ffi::ds4_session_sample(
                    session,
                    opts.temperature,
                    0,
                    opts.top_p,
                    opts.min_p,
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
            let text = self.token_text(token);
            reply_tokens.push(token);
            reply_text.push_str(&text);
            on_event(EngineEvent::Text(text));
            generated += 1;
        }

        // Remember the sampled reply so the next prompt build can splice these
        // exact tokens for its assistant section, keeping the live KV prefix
        // valid through the whole reply (see `build_tokens`).
        reply_text.truncate(reply_text.trim_end().len());
        self.last_reply = if reply_tokens.is_empty() {
            None
        } else {
            Some(LastReply {
                text: reply_text,
                tokens: reply_tokens,
                think: ds4_think(opts.think_mode),
            })
        };

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
        })
    }

    fn warm_system_prompt(
        &mut self,
        system: &str,
        checkpoint: Option<&Path>,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<bool, EngineError> {
        let tokens = self.build_system_tokens(system);
        let fingerprint = self.checkpoint_fingerprint(system);
        let session = self.ensure_session()?;

        // Fast path: restore a matching on-disk checkpoint, skipping prefill.
        if let Some(path) = checkpoint
            && let Ok(mut bytes) = std::fs::read(path)
            && let Some(nl) = bytes.iter().position(|&b| b == b'\n')
            && bytes[..nl] == *fingerprint.as_bytes()
        {
            let mut payload = bytes.split_off(nl + 1);
            let snap = ffi::Ds4SessionSnapshot {
                ptr: payload.as_mut_ptr(),
                len: payload.len() as u64,
                cap: payload.capacity() as u64,
            };
            let mut err = [0_i8; 512];
            // SAFETY: session valid; snap borrows `payload` which outlives the call.
            let rc = unsafe {
                ffi::ds4_session_load_snapshot(
                    session,
                    &raw const snap,
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            drop(payload);
            if rc == 0 {
                return Ok(false);
            }
            // A stale/incompatible snapshot: fall through and rebuild.
        }

        // Cache miss: prefill the system prompt, streaming progress.
        let mut progress = ProgressCtx {
            on_event,
            interrupt: &|| false,
            start: std::time::Instant::now(),
            base: 0,
            total: tokens.len(),
        };
        let progress_ptr = (&raw mut progress).cast::<std::os::raw::c_void>();
        // SAFETY: session valid; progress outlives the sync and the callback is cleared after.
        unsafe {
            ffi::ds4_session_set_display_progress(session, Some(progress_cb), progress_ptr);
        }
        let mut err = [0_i8; 512];
        // SAFETY: session, tokens, and err buffer are valid.
        let rc =
            unsafe { ffi::ds4_session_sync(session, tokens.as_ptr(), err.as_mut_ptr(), err.len()) };
        // SAFETY: session valid; clearing before ProgressCtx drops.
        unsafe { ffi::ds4_session_set_display_progress(session, None, std::ptr::null_mut()) };
        if rc != 0 {
            return Err(EngineError::new(cstr_message(
                &err,
                "system prompt prefill failed",
            )));
        }

        // Persist a fresh checkpoint for the next launch.
        if let Some(path) = checkpoint {
            Self::save_checkpoint(session, path, &fingerprint);
        }
        Ok(true)
    }

    fn count_tokens(&self, text: &str) -> i32 {
        if self.count_overhead.get() < 0 {
            self.count_overhead
                .set(self.templated_len("").unwrap_or(-1).max(-1));
        }
        let overhead = self.count_overhead.get();
        match (overhead, self.templated_len(text)) {
            (o, Some(len)) if o >= 0 => (len - o).max(0),
            // NUL bytes (or a failed overhead probe) fall back to the estimate.
            _ => i32::try_from(text.len() / 4).unwrap_or(i32::MAX),
        }
    }

    fn ctx_size(&self) -> i32 {
        self.ctx_size
    }

    fn model_name(&self) -> String {
        Ds4Engine::model_name(self)
    }
}

impl Drop for Ds4Engine {
    fn drop(&mut self) {
        if !self.session.is_null() {
            // SAFETY: the session was created by us and not yet freed.
            unsafe { ffi::ds4_session_free(self.session) };
        }
        // SAFETY: engine was opened by us and not yet closed.
        unsafe { ffi::ds4_engine_close(self.engine) };
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

/// Splits a role-tagged transcript into `(role, content)` chat messages.
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
    use super::parse_sections;

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
}
