//! Flint inference in native Rust.
//!
//! [`nvfp4::block`] is pure and always compiled, so the weight format is
//! tested without a model. Everything that touches tensors sits behind the
//! `candle` feature.

#[cfg(feature = "candle")]
pub mod file;
pub mod gguf;
pub mod nvfp4;
#[cfg(any(test, feature = "testgguf"))]
pub mod testgguf;
#[cfg(all(test, feature = "candle"))]
mod testutil;

/// Every failure this crate reports. A message, never a panic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self(e.to_string())
    }
}

#[cfg(feature = "candle")]
impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Self(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn err(msg: impl Into<String>) -> Error {
    Error(msg.into())
}
