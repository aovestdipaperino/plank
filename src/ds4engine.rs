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
    /// Opens a model file with the given backend and context size.
    ///
    /// # Errors
    /// Returns [`EngineError`] if the model fails to load.
    pub fn open(
        model_path: impl AsRef<Path>,
        backend: ffi::Ds4Backend,
        ctx_size: i32,
        n_threads: i32,
        power_percent: i32,
    ) -> Result<Self, EngineError> {
        set_metal_source_env();
        let path = model_path.as_ref();
        let c_path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| EngineError::new("model path contains a NUL byte"))?;
        let opts = ffi::Ds4EngineOptions {
            model_path: c_path.as_ptr(),
            mtp_path: std::ptr::null(),
            backend,
            n_threads,
            prefill_chunk: 0,
            mtp_draft_tokens: 0,
            mtp_margin: 0.0,
            directional_steering_file: std::ptr::null(),
            expert_profile_path: std::ptr::null(),
            directional_steering_attn: 0.0,
            directional_steering_ffn: 0.0,
            power_percent,
            ssd_streaming_cache_experts: 0,
            ssd_streaming_cache_bytes: 0,
            ssd_streaming_preload_experts: 0,
            simulate_used_memory_bytes: 0,
            warm_weights: false,
            quality: false,
            ssd_streaming: false,
            ssd_streaming_cold: false,
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
        // SAFETY: opts and its CString outlive the call; engine is a valid out-ptr.
        let rc = unsafe { ffi::ds4_engine_open(&raw mut engine, &raw const opts) };
        if rc != 0 || engine.is_null() {
            return Err(EngineError::new(format!(
                "failed to open model {}",
                path.display()
            )));
        }
        Ok(Self {
            engine,
            session: std::ptr::null_mut(),
            ctx_size,
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
        for (role, content) in parse_sections(transcript) {
            let (Ok(c_role), Ok(c_content)) = (CString::new(role), CString::new(content)) else {
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
        let think_mode = match think {
            ThinkMode::Off => ffi::Ds4ThinkMode::None,
            ThinkMode::Auto | ThinkMode::On => ffi::Ds4ThinkMode::High,
        };
        // SAFETY: engine and tokens valid.
        unsafe {
            ffi::ds4_chat_append_assistant_prefix(self.engine, tokens.as_mut_ptr(), think_mode);
        }
        tokens
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
            on_event(EngineEvent::Text(self.token_text(token)));
            generated += 1;
        }

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
    const DIR: &str = env!("DS4_METAL_DIR");
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
    for (var, file) in KERNELS {
        if std::env::var_os(var).is_none() {
            // SAFETY: called once at startup before any threads are spawned.
            unsafe { std::env::set_var(var, Path::new(DIR).join(file)) };
        }
    }
}

fn cstr_message(buf: &[i8], fallback: &str) -> String {
    if buf.first().copied().unwrap_or(0) == 0 {
        return fallback.to_string();
    }
    // SAFETY: buf is NUL-terminated within its length by the C callee.
    let s = unsafe { CStr::from_ptr(buf.as_ptr()) };
    s.to_string_lossy().into_owned()
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
