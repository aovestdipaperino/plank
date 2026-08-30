// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Routes [`crate::viz`] stream output into the markdown token renderer.

use std::io::Write;

use crate::render::TokenRenderer;
use crate::viz::RenderSink;

/// ANSI reset, emitted around error banners.
const ANSI_RESET: &str = "\x1b[0m";

/// Routes viz output into the markdown token renderer.
#[derive(Debug)]
pub struct TerminalSink<W: Write> {
    renderer: TokenRenderer<W>,
}

impl<W: Write> TerminalSink<W> {
    /// Wraps a token renderer as a [`RenderSink`].
    pub fn new(renderer: TokenRenderer<W>) -> Self {
        Self { renderer }
    }

    /// Borrows the underlying renderer, for callers that need to drive it
    /// directly (fence handling, capture, `finish`).
    pub fn renderer_mut(&mut self) -> &mut TokenRenderer<W> {
        &mut self.renderer
    }

    /// Consumes the sink and returns the renderer.
    pub fn into_renderer(self) -> TokenRenderer<W> {
        self.renderer
    }
}

impl<W: Write> RenderSink for TerminalSink<W> {
    fn visible_text(&mut self, text: &str) {
        self.renderer.set_in_think(false);
        self.renderer.write(text);
    }
    fn think_text(&mut self, text: &str) {
        self.renderer.set_in_think(true);
        self.renderer.write(text);
    }
    fn tool_text(&mut self, text: &str) {
        // Tool banners carry their own styling and must render verbatim; going
        // through `write` would markdown-process them and eat `*`/`_`/backtick
        // out of param values (e.g. `pattern=**/mod.rs`).
        self.renderer.set_in_think(false);
        self.renderer.plain(text);
    }
    fn error_text(&mut self, text: &str) {
        self.renderer.set_in_think(false);
        self.renderer.color("\x1b[1;31m");
        self.renderer.plain(text);
        self.renderer.color(ANSI_RESET);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::RenderOptions;

    #[test]
    fn tool_text_is_not_markdown_processed() {
        let opts = RenderOptions {
            use_color: false,
            format_thinking: true,
            format_markdown: true,
        };
        let mut sink = TerminalSink::new(TokenRenderer::new(Vec::new(), opts));
        sink.tool_text("pattern=**/mod.rs");
        let out = String::from_utf8(sink.into_renderer().into_sink()).unwrap();
        assert!(
            out.contains("**/mod.rs"),
            "tool banners must render verbatim, got {out:?}"
        );
    }
}
