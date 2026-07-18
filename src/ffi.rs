//! Raw FFI declarations for the ds4 C inference engine.
//!
//! Only the subset plank needs is declared: engine open/close, chat-template
//! tokenization, session create/sync/sample/eval, and token text lookup.
//! Present only when the `ds4_engine` cfg is set (macOS + built submodule).
#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_void};

/// Backend selector, mirroring `ds4_backend`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum Ds4Backend {
    Metal = 0,
    Cuda = 1,
    Cpu = 2,
}

/// Reasoning mode, mirroring `ds4_think_mode`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum Ds4ThinkMode {
    None = 0,
    High = 1,
    Max = 2,
}

/// Growable token vector, mirroring `ds4_tokens`.
#[repr(C)]
#[derive(Debug)]
pub struct Ds4Tokens {
    pub v: *mut c_int,
    pub len: c_int,
    pub cap: c_int,
}

impl Default for Ds4Tokens {
    fn default() -> Self {
        Self {
            v: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        }
    }
}

/// Distributed options, mirroring `ds4_distributed_options` (unused defaults).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Ds4DistributedOptions {
    pub role: c_int,
    pub layers_start: u32,
    pub layers_end: u32,
    pub layers_has_output: bool,
    pub layers_set: bool,
    pub listen_host: *const c_char,
    pub listen_port: c_int,
    pub coordinator_host: *const c_char,
    pub coordinator_port: c_int,
    pub prefill_chunk: u32,
    pub prefill_window: u32,
    pub activation_bits: u32,
    pub replay_check: bool,
    pub debug: bool,
}

/// Engine open options, mirroring `ds4_engine_options` field-for-field.
#[repr(C)]
#[derive(Debug)]
pub struct Ds4EngineOptions {
    pub model_path: *const c_char,
    pub mtp_path: *const c_char,
    pub backend: Ds4Backend,
    pub n_threads: c_int,
    pub prefill_chunk: u32,
    pub mtp_draft_tokens: c_int,
    pub mtp_margin: f32,
    pub directional_steering_file: *const c_char,
    pub expert_profile_path: *const c_char,
    pub directional_steering_attn: f32,
    pub directional_steering_ffn: f32,
    pub power_percent: c_int,
    pub ssd_streaming_cache_experts: u32,
    pub ssd_streaming_cache_bytes: u64,
    pub ssd_streaming_preload_experts: u32,
    pub simulate_used_memory_bytes: u64,
    pub warm_weights: bool,
    pub quality: bool,
    pub ssd_streaming: bool,
    pub ssd_streaming_cold: bool,
    pub inspect_only: bool,
    pub load_slice: bool,
    pub load_layer_start: u32,
    pub load_layer_end: u32,
    pub load_output: bool,
    pub distributed: Ds4DistributedOptions,
}

/// Opaque handle to a loaded ds4 model.
#[repr(C)]
#[allow(missing_debug_implementations)]
pub struct Ds4Engine {
    _private: [u8; 0],
}

/// Opaque handle to a ds4 inference session.
#[repr(C)]
#[allow(missing_debug_implementations)]
pub struct Ds4Session {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn ds4_engine_open(out: *mut *mut Ds4Engine, opt: *const Ds4EngineOptions) -> c_int;
    pub fn ds4_engine_close(e: *mut Ds4Engine);
    pub fn ds4_engine_summary(e: *mut Ds4Engine);
    pub fn ds4_engine_model_name(e: *mut Ds4Engine) -> *const c_char;

    pub fn ds4_tokens_push(tv: *mut Ds4Tokens, token: c_int);
    pub fn ds4_tokens_free(tv: *mut Ds4Tokens);

    pub fn ds4_chat_begin(e: *mut Ds4Engine, tokens: *mut Ds4Tokens);
    pub fn ds4_chat_append_message(
        e: *mut Ds4Engine,
        tokens: *mut Ds4Tokens,
        role: *const c_char,
        content: *const c_char,
    );
    pub fn ds4_chat_append_assistant_prefix(
        e: *mut Ds4Engine,
        tokens: *mut Ds4Tokens,
        think_mode: Ds4ThinkMode,
    );

    pub fn ds4_token_text(e: *mut Ds4Engine, token: c_int, len: *mut usize) -> *mut c_char;
    pub fn ds4_token_eos(e: *mut Ds4Engine) -> c_int;

    pub fn ds4_session_create(
        out: *mut *mut Ds4Session,
        e: *mut Ds4Engine,
        ctx_size: c_int,
    ) -> c_int;
    pub fn ds4_session_free(s: *mut Ds4Session);
    pub fn ds4_session_set_display_progress(
        s: *mut Ds4Session,
        f: Option<
            unsafe extern "C" fn(ud: *mut c_void, event: *const c_char, cur: c_int, total: c_int),
        >,
        ud: *mut c_void,
    );
    pub fn ds4_session_set_cancel(
        s: *mut Ds4Session,
        f: Option<unsafe extern "C" fn(ud: *mut c_void) -> bool>,
        ud: *mut c_void,
    );
    pub fn ds4_session_sync(
        s: *mut Ds4Session,
        prompt: *const Ds4Tokens,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn ds4_session_sample(
        s: *mut Ds4Session,
        temperature: f32,
        top_k: c_int,
        top_p: f32,
        min_p: f32,
        rng: *mut u64,
    ) -> c_int;
    pub fn ds4_session_eval(
        s: *mut Ds4Session,
        token: c_int,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn ds4_session_ctx(s: *mut Ds4Session) -> c_int;
    pub fn ds4_session_pos(s: *mut Ds4Session) -> c_int;
    pub fn ds4_session_common_prefix(s: *mut Ds4Session, prompt: *const Ds4Tokens) -> c_int;
}
