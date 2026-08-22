// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Serialized session KV state, mirroring `ds4_session_snapshot`.
#[repr(C)]
#[derive(Debug)]
pub struct Ds4SessionSnapshot {
    pub ptr: *mut u8,
    pub len: u64,
    pub cap: u64,
}

impl Default for Ds4SessionSnapshot {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
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

impl Default for Ds4DistributedOptions {
    /// All-zero: plank never runs the distributed coordinator/worker split, so
    /// the struct is only here to keep `Ds4EngineOptions` the right shape.
    fn default() -> Self {
        Self {
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
        }
    }
}

/// Tensor-parallel options, mirroring `ds4_tp_options` (unused defaults).
///
/// plank drives a single machine, so every field here stays at its zero
/// value; the struct exists only so `Ds4EngineOptions` keeps the same shape
/// and size as the C `ds4_engine_options` it is passed to.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Ds4TpOptions {
    pub role: c_int,
    pub requested: bool,
    pub listen_host: *const c_char,
    pub listen_port: c_int,
    pub leader_host: *const c_char,
    pub leader_port: c_int,
    pub transport: c_int,
    pub rdma_device: *const c_char,
    pub rdma_gid_index: c_int,
    pub rdma_gid_index_set: bool,
    pub glm_token_prefill: bool,
    pub debug_hash: c_int,
}

impl Default for Ds4TpOptions {
    /// All-zero, which is what `DS4_TP_NONE` and an unrequested role mean.
    fn default() -> Self {
        Self {
            role: 0,
            requested: false,
            listen_host: std::ptr::null(),
            listen_port: 0,
            leader_host: std::ptr::null(),
            leader_port: 0,
            transport: 0,
            rdma_device: std::ptr::null(),
            rdma_gid_index: 0,
            rdma_gid_index_set: false,
            glm_token_prefill: false,
            debug_hash: 0,
        }
    }
}

/// Engine open options, mirroring `ds4_engine_options` field-for-field.
#[repr(C)]
#[derive(Debug)]
pub struct Ds4EngineOptions {
    pub model_path: *const c_char,
    pub mtp_path: *const c_char,
    pub backend: Ds4Backend,
    pub n_threads: c_int,
    pub context_size: c_int,
    pub prefill_chunk: u32,
    pub mtp_draft_tokens: c_int,
    pub mtp_margin: f32,
    /// `DSpark` confidence-pruning threshold, 0..1 (C default: 0.7).
    pub dspark_confidence_threshold: f32,
    pub directional_steering_file: *const c_char,
    pub expert_profile_path: *const c_char,
    pub directional_steering_attn: f32,
    pub directional_steering_ffn: f32,
    pub power_percent: c_int,
    pub ssd_streaming_cache_experts: u32,
    pub ssd_streaming_cache_bytes: u64,
    pub ssd_streaming_full_layers: u32,
    pub ssd_streaming_preload_experts: u32,
    pub simulate_used_memory_bytes: u64,
    pub warm_weights: bool,
    pub quality: bool,
    pub glm_mtp: bool,
    pub glm_mtp_timing: bool,
    /// Use the `mtp_path` support GGUF as a `DSpark` draft model.
    pub dspark: bool,
    /// Load `DSpark` support but keep target-only decode.
    pub dspark_strict: bool,
    /// Whether `dspark_confidence_threshold` was set explicitly.
    pub dspark_confidence_threshold_set: bool,
    pub cuda_tensor_parallel: bool,
    pub ssd_streaming: bool,
    pub ssd_streaming_cold: bool,
    pub ssd_streaming_full_layers_set: bool,
    pub inspect_only: bool,
    pub placement_ctx_hint: c_int,
    pub placement_session_count_hint: c_int,
    pub share_session_prefill_workspace: bool,
    pub first_token_test: bool,
    pub metal_graph_test: bool,
    pub load_slice: bool,
    pub load_layer_start: u32,
    pub load_layer_end: u32,
    pub load_output: bool,
    pub distributed: Ds4DistributedOptions,
    pub tp: Ds4TpOptions,
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

    /// Appends the reasoning-effort preamble `DS4_THINK_MAX` prepends ahead of
    /// the system prompt. Tokenized as plain text, so it must sit immediately
    /// after `ds4_chat_begin` and before the first message — the position the
    /// C's `repl_chat_apply_max_prefix` inserts it at.
    ///
    /// The C also exposes the text and its context floor as
    /// `ds4_think_max_prefix` / `ds4_think_max_min_context`; plank mirrors those
    /// as `engine::THINK_MAX_PREFIX` / `engine::THINK_MAX_MIN_CONTEXT` rather
    /// than calling them, so both are available without a model loaded.
    /// `tests/c_parity.rs` holds the Rust copies equal to the C source.
    pub fn ds4_chat_append_max_effort_prefix(e: *mut Ds4Engine, tokens: *mut Ds4Tokens);

    /// Tokenizes already-rendered chat text, so control strings like
    /// `</think>` map to their special tokens rather than to literal pieces.
    pub fn ds4_tokenize_rendered_chat(e: *mut Ds4Engine, text: *const c_char, out: *mut Ds4Tokens);

    pub fn ds4_token_text(e: *mut Ds4Engine, token: c_int, len: *mut usize) -> *mut c_char;
    pub fn ds4_token_eos(e: *mut Ds4Engine) -> c_int;

    pub fn ds4_session_create(
        out: *mut *mut Ds4Session,
        e: *mut Ds4Engine,
        ctx_size: c_int,
    ) -> c_int;
    pub fn ds4_session_free(s: *mut Ds4Session);
    /// Drops the session's cached KV so the next sync prefills from zero. Used
    /// when the prompt *prefix* changes under it (the reasoning-effort preamble
    /// moving in or out), which a common-prefix probe alone cannot recover from.
    pub fn ds4_session_invalidate(s: *mut Ds4Session);
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
    /// Whether this session can take the Metal greedy-chain decode path.
    ///
    /// The C declines for anything the chain cannot reproduce bit-identically:
    /// a non-Metal backend, GLM, distributed/TP, SSD streaming, quality mode,
    /// directional steering, a CPU-routed layer — and, notably, **any loaded
    /// support model**, so `--dspark` and the chain are mutually exclusive.
    /// Must be called on a prefilled session: it also requires a valid
    /// checkpoint.
    /// Non-zero when the Metal device is an Apple M5. Reads the device name
    /// captured at Metal init, so it is only meaningful once a model is loaded.
    pub fn ds4_gpu_device_is_m5_apple_silicon() -> c_int;
    pub fn ds4_session_chain_greedy_supported(s: *const Ds4Session) -> bool;
    /// Decodes a burst of argmax tokens with the next id kept on-device,
    /// removing the per-token host round-trip (sync + logits readback + CPU
    /// argmax) that [`ds4_session_eval`] pays.
    ///
    /// The seed is this session's current argmax — the caller does not sample
    /// it. `on_token` receives the seed first, then every chained id in order;
    /// returning `false` stops the burst *without* committing that token, so
    /// EOS and interrupt are both expressed that way. Only approved tokens
    /// enter the session checkpoint.
    ///
    /// Returns the number of approved tokens, or -1 with `err` set. `max_tokens`
    /// must be at least 2 — a one-token burst never runs a GPU eval and would
    /// leave the logits stale. When `completed` comes back false the burst
    /// stopped early and the session's logits are stale; when true they hold
    /// the logits after the last approved token, as after `ds4_session_eval`.
    pub fn ds4_session_eval_chain_greedy(
        s: *mut Ds4Session,
        max_tokens: c_int,
        on_token: Option<unsafe extern "C" fn(*mut c_void, c_int) -> bool>,
        on_token_ctx: *mut c_void,
        completed: *mut bool,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    /// Draft-block size the loaded support model can propose per step, or 0
    /// when there is none. `> 1` is the C's own test for "speculation is worth
    /// attempting" — under `DSpark` it is the checkpoint's block size.
    pub fn ds4_engine_mtp_draft_tokens(e: *mut Ds4Engine) -> c_int;

    /// Evaluates `first_token`, then verifies a drafted continuation against
    /// the target model, committing only the prefix the target agrees with.
    ///
    /// This is the only entry point that consumes `DSpark`/MTP drafts;
    /// [`ds4_session_eval`] advances one token and can never accept a block.
    /// Writes the committed run — **including `first_token` at index 0** —
    /// into `accepted`, and returns how many it wrote, or a negative value on
    /// error. A return of 0 means nothing was committed, `first_token`
    /// included. Every committed token is already in the session's KV cache,
    /// so the run cannot be partially rejected after the fact.
    ///
    /// Verification is argmax, so this is only equivalent to sampling when
    /// the generation is greedy — hence the C's `temperature <= 0` gate.
    /// Degrades to a plain single-token eval on CPU and distributed backends.
    pub fn ds4_session_eval_speculative_argmax(
        s: *mut Ds4Session,
        first_token: c_int,
        max_tokens: c_int,
        eos_token: c_int,
        accepted: *mut c_int,
        accepted_cap: c_int,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn ds4_session_ctx(s: *mut Ds4Session) -> c_int;
    pub fn ds4_session_pos(s: *mut Ds4Session) -> c_int;
    pub fn ds4_session_common_prefix(s: *mut Ds4Session, prompt: *const Ds4Tokens) -> c_int;
    pub fn ds4_session_save_snapshot(
        s: *mut Ds4Session,
        snap: *mut Ds4SessionSnapshot,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn ds4_session_load_snapshot(
        s: *mut Ds4Session,
        snap: *const Ds4SessionSnapshot,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn ds4_session_snapshot_free(snap: *mut Ds4SessionSnapshot);
}

/// Confirm callback type, mirroring `ds4_web_confirm_fn`.
pub type Ds4WebConfirmFn = unsafe extern "C" fn(
    privdata: *mut c_void,
    message: *const c_char,
    err: *mut c_char,
    err_len: usize,
) -> c_int;
/// Log callback type, mirroring `ds4_web_log_fn`.
pub type Ds4WebLogFn = unsafe extern "C" fn(privdata: *mut c_void, message: *const c_char);
/// Cancel callback type, mirroring `ds4_web_cancel_fn`.
pub type Ds4WebCancelFn = unsafe extern "C" fn(privdata: *mut c_void) -> bool;

/// Web subsystem config, mirroring `ds4_web_config` in `ds4_web.h`.
#[repr(C)]
#[allow(missing_debug_implementations)]
pub struct Ds4WebConfig {
    pub home_dir: *const c_char,
    pub port: c_int,
    pub confirm: Option<Ds4WebConfirmFn>,
    pub confirm_privdata: *mut c_void,
    pub log: Option<Ds4WebLogFn>,
    pub log_privdata: *mut c_void,
    pub cancel: Option<Ds4WebCancelFn>,
    pub cancel_privdata: *mut c_void,
}

/// Opaque browser handle, mirroring `struct ds4_web`.
#[repr(C)]
#[allow(missing_debug_implementations)]
pub struct Ds4Web {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn ds4_web_create(cfg: *const Ds4WebConfig) -> *mut Ds4Web;
    pub fn ds4_web_free(web: *mut Ds4Web);
    pub fn ds4_web_google_search(
        web: *mut Ds4Web,
        query: *const c_char,
        err: *mut c_char,
        err_len: usize,
    ) -> *mut c_char;
    pub fn ds4_web_visit_page(
        web: *mut Ds4Web,
        url: *const c_char,
        err: *mut c_char,
        err_len: usize,
    ) -> *mut c_char;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// Pins every field of [`Ds4EngineOptions`] to the byte offset the C
    /// `ds4_engine_options` puts it at.
    ///
    /// The mirror is positional, so a field inserted mid-struct on the C side
    /// shifts everything after it — and nothing about that fails to compile.
    /// The engine would simply read its configuration from the wrong bytes:
    /// `quality` from `warm_weights`, a path pointer from an integer. That is
    /// a crash at best and a silently mis-tuned engine at worst.
    ///
    /// The numbers come from `offsetof` on the checked-out submodule header.
    /// When a `refs/ds4` bump breaks this test, re-derive them rather than
    /// nudging the constants — the whole point is that they are the C's.
    /// Asserts a `(field name, actual offset, expected offset)` table.
    fn assert_offsets(fields: &[(&str, usize, usize)]) {
        for &(name, got, want) in fields {
            assert_eq!(got, want, "offset of {name}");
        }
    }

    #[test]
    fn engine_options_size_matches_the_c_layout() {
        assert_eq!(size_of::<Ds4EngineOptions>(), 272, "struct size");
    }

    /// The scalars and pointers ahead of the option flags.
    #[test]
    fn engine_options_head_fields_match_the_c_layout() {
        assert_offsets(&[
            ("model_path", offset_of!(Ds4EngineOptions, model_path), 0),
            ("mtp_path", offset_of!(Ds4EngineOptions, mtp_path), 8),
            ("backend", offset_of!(Ds4EngineOptions, backend), 16),
            ("n_threads", offset_of!(Ds4EngineOptions, n_threads), 20),
            (
                "context_size",
                offset_of!(Ds4EngineOptions, context_size),
                24,
            ),
            (
                "prefill_chunk",
                offset_of!(Ds4EngineOptions, prefill_chunk),
                28,
            ),
            (
                "mtp_draft_tokens",
                offset_of!(Ds4EngineOptions, mtp_draft_tokens),
                32,
            ),
            ("mtp_margin", offset_of!(Ds4EngineOptions, mtp_margin), 36),
            (
                "dspark_confidence_threshold",
                offset_of!(Ds4EngineOptions, dspark_confidence_threshold),
                40,
            ),
            (
                "directional_steering_file",
                offset_of!(Ds4EngineOptions, directional_steering_file),
                48,
            ),
            (
                "expert_profile_path",
                offset_of!(Ds4EngineOptions, expert_profile_path),
                56,
            ),
            (
                "directional_steering_attn",
                offset_of!(Ds4EngineOptions, directional_steering_attn),
                64,
            ),
            (
                "directional_steering_ffn",
                offset_of!(Ds4EngineOptions, directional_steering_ffn),
                68,
            ),
            (
                "power_percent",
                offset_of!(Ds4EngineOptions, power_percent),
                72,
            ),
            (
                "ssd_streaming_cache_experts",
                offset_of!(Ds4EngineOptions, ssd_streaming_cache_experts),
                76,
            ),
            (
                "ssd_streaming_cache_bytes",
                offset_of!(Ds4EngineOptions, ssd_streaming_cache_bytes),
                80,
            ),
            (
                "ssd_streaming_full_layers",
                offset_of!(Ds4EngineOptions, ssd_streaming_full_layers),
                88,
            ),
            (
                "ssd_streaming_preload_experts",
                offset_of!(Ds4EngineOptions, ssd_streaming_preload_experts),
                92,
            ),
            (
                "simulate_used_memory_bytes",
                offset_of!(Ds4EngineOptions, simulate_used_memory_bytes),
                96,
            ),
        ]);
    }

    /// The packed option flags and the trailing sub-structs. This is the run
    /// most likely to shift: single bytes with no padding between them, so an
    /// inserted C `bool` slides every flag after it by one.
    #[test]
    fn engine_options_flag_fields_match_the_c_layout() {
        assert_offsets(&[
            (
                "warm_weights",
                offset_of!(Ds4EngineOptions, warm_weights),
                104,
            ),
            ("quality", offset_of!(Ds4EngineOptions, quality), 105),
            ("glm_mtp", offset_of!(Ds4EngineOptions, glm_mtp), 106),
            (
                "glm_mtp_timing",
                offset_of!(Ds4EngineOptions, glm_mtp_timing),
                107,
            ),
            ("dspark", offset_of!(Ds4EngineOptions, dspark), 108),
            (
                "dspark_strict",
                offset_of!(Ds4EngineOptions, dspark_strict),
                109,
            ),
            (
                "dspark_confidence_threshold_set",
                offset_of!(Ds4EngineOptions, dspark_confidence_threshold_set),
                110,
            ),
            (
                "cuda_tensor_parallel",
                offset_of!(Ds4EngineOptions, cuda_tensor_parallel),
                111,
            ),
            (
                "ssd_streaming",
                offset_of!(Ds4EngineOptions, ssd_streaming),
                112,
            ),
            (
                "ssd_streaming_cold",
                offset_of!(Ds4EngineOptions, ssd_streaming_cold),
                113,
            ),
            (
                "ssd_streaming_full_layers_set",
                offset_of!(Ds4EngineOptions, ssd_streaming_full_layers_set),
                114,
            ),
            (
                "inspect_only",
                offset_of!(Ds4EngineOptions, inspect_only),
                115,
            ),
            (
                "placement_ctx_hint",
                offset_of!(Ds4EngineOptions, placement_ctx_hint),
                116,
            ),
            (
                "placement_session_count_hint",
                offset_of!(Ds4EngineOptions, placement_session_count_hint),
                120,
            ),
            (
                "share_session_prefill_workspace",
                offset_of!(Ds4EngineOptions, share_session_prefill_workspace),
                124,
            ),
            (
                "first_token_test",
                offset_of!(Ds4EngineOptions, first_token_test),
                125,
            ),
            (
                "metal_graph_test",
                offset_of!(Ds4EngineOptions, metal_graph_test),
                126,
            ),
            ("load_slice", offset_of!(Ds4EngineOptions, load_slice), 127),
            (
                "load_layer_start",
                offset_of!(Ds4EngineOptions, load_layer_start),
                128,
            ),
            (
                "load_layer_end",
                offset_of!(Ds4EngineOptions, load_layer_end),
                132,
            ),
            (
                "load_output",
                offset_of!(Ds4EngineOptions, load_output),
                136,
            ),
            (
                "distributed",
                offset_of!(Ds4EngineOptions, distributed),
                144,
            ),
            ("tp", offset_of!(Ds4EngineOptions, tp), 208),
        ]);
    }
}
