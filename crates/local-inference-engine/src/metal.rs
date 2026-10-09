// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Locating the Metal kernel sources the C engine compiles at startup.
//!
//! The C looks for each kernel through a `DS4_METAL_<NAME>_SOURCE` variable,
//! falling back to `metal/...` relative to the working directory, which only
//! resolves from the ds4 source root. A program linking the engine therefore
//! has to point every variable at the kernels itself, before the first
//! [`crate::Model::open`]; [`set_source_env`] does that.

use std::path::{Path, PathBuf};

/// Metal kernel sources the C engine requires, as `(env var, file name)`.
///
/// Must stay in lockstep with the `required_sources` table in
/// `refs/ds4/ds4_metal.m` (`ds4_gpu_full_source`): every entry there is
/// mandatory, so a kernel shipped upstream but missing here aborts startup
/// with "metal backend unavailable". The C's own fallback search paths
/// (relative `metal/...` and `./metal/...`) only resolve from the submodule
/// root, so the host has to point at each one explicitly.
///
/// `metal_kernels_match_the_c_reference` in plank's `tests/c_parity.rs` parses
/// that table out of the C and fails on any drift — the comment alone was not
/// enough, and a submodule bump that added two Qwen kernels shipped a build
/// that could not open a model at all.
pub const KERNEL_SOURCES: &[(&str, &str)] = &[
    // The GLM 5.3 and vision kernels landed in the antirez/main sync and must
    // be pointed at explicitly. `DS4_METAL_DSV41_SOURCE` (`dsv41.metal`)
    // arrived with the V4.1 bump for the same reason: the combined Metal
    // source is compiled once for every model, so a missing V4.1 kernel would
    // abort startup even for a plain V4 run.
    ("DS4_METAL_FLASH_ATTN_SOURCE", "flash_attn.metal"),
    ("DS4_METAL_DENSE_SOURCE", "dense.metal"),
    ("DS4_METAL_GLM53_BF16_SOURCE", "glm53_bf16.metal"),
    ("DS4_METAL_GLM53_VISION_SOURCE", "glm53_vision.metal"),
    (
        "DS4_METAL_DEEPSEEK4_VISION_SOURCE",
        "deepseek4_vision.metal",
    ),
    ("DS4_METAL_GLM53_KDA_SOURCE", "glm53_kda.metal"),
    ("DS4_METAL_MOE_SOURCE", "moe.metal"),
    ("DS4_METAL_DSV4_HC_SOURCE", "dsv4_hc.metal"),
    ("DS4_METAL_UNARY_SOURCE", "unary.metal"),
    ("DS4_METAL_DSV4_KV_SOURCE", "dsv4_kv.metal"),
    ("DS4_METAL_DSV41_SOURCE", "dsv41.metal"),
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
    // Added by the Qwen3.8-Flash-Next bump. Required unconditionally,
    // not only for a Qwen run: the C compiles one combined Metal source
    // for every model, so a missing Qwen kernel aborts a DeepSeek
    // startup too.
    ("DS4_METAL_QWEN4_SOURCE", "qwen4.metal"),
    ("DS4_METAL_QWEN4_VISION_SOURCE", "qwen4_vision.metal"),
];

/// The kernel directory baked in when the engine was built, if it was.
const BUILT_DIR: Option<&str> = option_env!("DS4_METAL_DIR");

/// Points every unset `DS4_METAL_<NAME>_SOURCE` at [`source_dir`].
///
/// A variable already set is left alone, so a caller can override any single
/// kernel. [`crate::Model::open`] calls this itself.
pub fn set_source_env() {
    let dir = source_dir();
    for &(var, file) in KERNEL_SOURCES {
        if std::env::var_os(var).is_none() {
            // SAFETY: the engine is opened before the host spawns threads that
            // read the environment; plank and pt both open it at startup.
            unsafe { std::env::set_var(var, dir.join(file)) };
        }
    }
}

/// Resolves the directory holding the `.metal` kernel sources.
///
/// Tried in order: the `DS4_METAL_DIR` environment variable, the path baked
/// in at compile time (valid for local builds), and `../share/plank/metal`
/// relative to the executable (where Homebrew bottles install the kernels —
/// the compile-time path is the CI runner's checkout and doesn't exist on
/// user machines).
#[must_use]
pub fn source_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("DS4_METAL_DIR") {
        return dir.into();
    }
    let built = BUILT_DIR.map(PathBuf::from);
    if let Some(built) = built.as_ref().filter(|d| d.is_dir()) {
        return built.clone();
    }
    if let Ok(exe) = std::env::current_exe().and_then(std::fs::canonicalize)
        && let Some(prefix) = exe.parent().and_then(Path::parent)
    {
        let shared = prefix.join("share").join("plank").join("metal");
        if shared.is_dir() {
            return shared;
        }
    }
    built.unwrap_or_else(|| PathBuf::from("metal"))
}

/// Whether the Metal kernel sources are absent from where the engine looks.
///
/// [`set_source_env`] points `DS4_METAL_FLASH_ATTN_SOURCE` at them when it can
/// find them; if the variable is unset or names a path that is gone, the
/// engine cannot build its kernels and an open fails with nothing on the model
/// file's own account.
#[must_use]
pub fn kernels_missing() -> bool {
    std::env::var_os("DS4_METAL_FLASH_ATTN_SOURCE").is_none_or(|p| !Path::new(&p).exists())
}
