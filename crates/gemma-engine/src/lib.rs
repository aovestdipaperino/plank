//! Gemma 4 inference in native Rust.
//!
//! [`template`] is pure and always compiled, so plank's tests cover the chat
//! format without a model. Everything that touches tensors sits behind the
//! `candle` feature.

pub mod template;
#[cfg(all(feature = "candle", any(test, feature = "testgguf")))]
pub mod testgguf;
#[cfg(feature = "candle")]
pub mod tokenizer;

#[cfg(feature = "candle")]
pub mod kv;

#[cfg(feature = "candle")]
pub mod model;
#[cfg(feature = "candle")]
pub mod sample;
#[cfg(feature = "candle")]
pub mod session;

/// Every failure this crate reports. A message, never a panic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

#[cfg(feature = "candle")]
impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Self(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
