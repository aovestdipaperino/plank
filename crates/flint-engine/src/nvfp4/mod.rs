//! NVFP4 weights: the block format, and the kernels that multiply by them.

pub mod block;
#[cfg(feature = "candle")]
pub mod cpu;
#[cfg(feature = "candle")]
pub mod tensor;
