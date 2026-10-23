//! Flint inference in native Rust.
//!
//! [`nvfp4::block`] is pure and always compiled, so the weight format is
//! tested without a model. Everything that touches tensors sits behind the
//! `candle` feature.

#[cfg(feature = "candle")]
mod attention;
pub mod config;
#[cfg(feature = "candle")]
mod ffn;
#[cfg(feature = "candle")]
pub mod file;
pub mod gguf;
#[cfg(feature = "candle")]
pub mod model;
pub mod nvfp4;
#[cfg(feature = "candle")]
mod rope;
pub mod template;
#[cfg(test)]
mod testalloc;
#[cfg(any(test, feature = "testgguf"))]
pub mod testgguf;
#[cfg(all(test, feature = "candle"))]
mod testutil;
#[cfg(feature = "candle")]
pub mod tokenizer;

#[cfg(feature = "candle")]
pub use gemma_engine::sample::Sampler;
/// The session, cache and snapshot format shared with gemma-engine.
/// `Session<model::Model>` reports its failures as [`gemma_engine::Error`];
/// `?` turns one into this crate's [`Error`].
#[cfg(feature = "candle")]
pub use gemma_engine::session::{Session, SessionModel};

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

/// A failure from the shared [`Session`] (or anything else in
/// gemma-engine), carried over with its message.
impl From<gemma_engine::Error> for Error {
    fn from(e: gemma_engine::Error) -> Self {
        Self(e.0)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gemma_engine_error_keeps_its_message() {
        fn session_op() -> std::result::Result<(), gemma_engine::Error> {
            Err(gemma_engine::Error("context full: 9 tokens > 8".into()))
        }
        fn ours() -> Result<()> {
            session_op()?;
            Ok(())
        }
        assert_eq!(ours(), Err(Error("context full: 9 tokens > 8".into())));
    }
}
