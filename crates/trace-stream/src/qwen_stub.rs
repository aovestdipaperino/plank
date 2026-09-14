//! No-op stand-in for [`crate::qwen`] when the `qwen` feature is off.
//!
//! The same shape as the real parser, the same reason as `WasmHost`'s no-op
//! implementation in plank: [`crate::viz`] dispatches over a `DialectParser`
//! enum with one arm per dialect, and gating that arm would put a `#[cfg]` on
//! a dozen match arms in the hottest code in the crate — the code every DSML
//! generation runs through. Swapping the module instead leaves `viz.rs`
//! byte-for-byte the file it is with the feature on.
//!
//! The stub never leaves [`DsmlState::Search`], so a build without Qwen
//! support treats a `<tool_call>` stanza as ordinary prose rather than
//! half-recognising it. Nothing should reach here regardless: plank refuses to
//! open a Qwen model at all without the feature, and this is the second line of
//! that same decision.

use crate::dsml::{DsmlState, ToolCall};

/// A parser that recognises nothing. See the module docs.
#[derive(Debug, Default)]
pub struct QwenParser {
    /// Never read. Present so the stub owns storage for [`Self::raw`] to
    /// borrow, exactly as the real parser does.
    raw: Vec<u8>,
    /// Never read, for the same reason [`Self::raw`] exists.
    calls: Vec<ToolCall>,
}

impl QwenParser {
    /// A parser that will stay in `Search` for its whole life.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Always `Search`: no stanza is ever open, so the renderer never reports
    /// an incomplete tool call against prose.
    #[must_use]
    pub fn state(&self) -> DsmlState {
        DsmlState::Search
    }

    /// Always empty.
    #[must_use]
    pub fn calls(&self) -> &[ToolCall] {
        &self.calls
    }

    /// Always `None`: recognising nothing is not an error here.
    #[must_use]
    pub fn error(&self) -> Option<&str> {
        None
    }

    /// Always `None`.
    #[must_use]
    pub fn pending_call(&self) -> Option<ToolCall> {
        None
    }

    /// Always empty: nothing is buffered because nothing is parsed.
    #[must_use]
    pub fn raw(&self) -> &[u8] {
        &self.raw
    }

    /// Always `false`.
    #[must_use]
    pub fn param_close_prefix(&self) -> bool {
        false
    }

    /// Nothing to reset.
    pub fn reset(&mut self) {}

    /// Discards the bytes.
    pub fn feed(&mut self, _bytes: impl AsRef<[u8]>) {}

    /// Nothing to finish.
    pub fn finish(&mut self) {}
}
