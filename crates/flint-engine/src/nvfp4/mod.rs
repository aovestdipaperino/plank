//! NVFP4 weights: the block format, and the kernels that multiply by them.

pub mod block;
#[cfg(feature = "candle")]
pub mod cpu;
#[cfg(all(feature = "candle", target_os = "macos"))]
pub(crate) mod metal;
#[cfg(feature = "candle")]
pub mod ops;
#[cfg(feature = "candle")]
pub mod tensor;
