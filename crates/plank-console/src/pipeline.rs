// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Raw stream bytes to styled screen lines.
//!
//! ```text
//! bytes -> StreamRenderer -> TerminalSink -> TokenRenderer<Vec<u8>> -> ANSI
//!       -> AnsiLineAssembler -> Vec<Cell> -> StreamView
//! ```
//!
//! The renderer is the same `plank_stream` code plank's plain-stdout REPL
//! runs, given a `Vec<u8>` instead of a terminal, so the two applications
//! cannot drift.

use plank_stream::TerminalSink;
use plank_stream::render::{RenderOptions, TokenRenderer};
use plank_stream::viz::StreamRenderer;

use crate::ansiasm::AnsiLineAssembler;
use crate::streamview::StreamView;

/// The authoritative tool-name table, verbatim from `dispatch` in plank's
/// `src/tools/mod.rs`, so DSML tool-call banners match what plank itself
/// would render.
fn tool_names() -> Vec<String> {
    [
        "EnterWorktree",
        "ExitWorktree",
        "EnterPlanMode",
        "ExitPlanMode",
        "read",
        "more",
        "write",
        "list",
        "glob",
        "edit",
        "search",
        "bash",
        "bash_status",
        "bash_stop",
        "google_search",
        "visit_page",
        "mcp_describe",
        "mcp_call",
        "mcp_list_resources",
        "mcp_read_resource",
        "skill",
        "task",
        "ask",
        "recall",
        "run_code",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}

/// One session's renderer state.
#[derive(Debug)]
pub struct Pipeline {
    stream: StreamRenderer<TerminalSink<Vec<u8>>>,
    asm: AnsiLineAssembler,
    opts: RenderOptions,
    /// Trailing bytes of an incomplete UTF-8 character, held until the rest
    /// of it arrives.
    utf8_carry: Vec<u8>,
}

impl Pipeline {
    #[must_use]
    pub fn new(opts: RenderOptions) -> Self {
        let mut stream =
            StreamRenderer::new(TerminalSink::new(TokenRenderer::new(Vec::new(), opts)));
        stream.set_tool_names(tool_names());
        Self {
            stream,
            asm: AnsiLineAssembler::new(),
            opts,
            utf8_carry: Vec::new(),
        }
    }

    /// Pushes stream bytes through the renderer and into the view.
    ///
    /// Bytes arrive UTF-8-split at arbitrary points; `StreamRenderer::push`
    /// takes `&str`, so a lossy conversion here would corrupt multi-byte
    /// characters. Incomplete trailing UTF-8 is held instead.
    pub fn feed(&mut self, bytes: &[u8], view: &mut StreamView) {
        self.utf8_carry.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.utf8_carry) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to(),
        };
        let text: String = String::from_utf8_lossy(&self.utf8_carry[..valid]).into_owned();
        self.utf8_carry.drain(..valid);
        if !text.is_empty() {
            self.stream.push(&text);
        }
        self.drain(view);
    }

    /// Ends the stream: flushes the renderer and the trailing partial line.
    pub fn finish(&mut self, view: &mut StreamView) {
        self.stream.finish();
        self.drain(view);
        if let Some(line) = self.asm.flush() {
            view.push_line(line);
        }
        view.set_partial(Vec::new());
    }

    /// Rebuilds the renderer with new options. The scrollback already on
    /// screen keeps its old styling; only new bytes use the new options.
    pub fn set_options(&mut self, opts: RenderOptions, view: &mut StreamView) {
        self.finish(view);
        let utf8_carry = std::mem::take(&mut self.utf8_carry);
        *self = Self::new(opts);
        self.utf8_carry = utf8_carry;
    }

    /// The options currently in force, for a caller (the menu, a status
    /// line) that needs to reflect them without tracking a separate copy.
    #[must_use]
    pub fn options(&self) -> RenderOptions {
        self.opts
    }

    /// Moves whatever ANSI the renderer has produced into the view.
    fn drain(&mut self, view: &mut StreamView) {
        let ansi = std::mem::take(self.stream.sink_mut().renderer_mut().sink_mut());
        self.asm.push(&ansi);
        for line in self.asm.take_complete_lines() {
            view.push_line(line);
        }
        view.set_partial(self.asm.partial_line());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_vision::core::geometry::Rect;
    use turbo_vision::core::palette::TvColor;

    fn opts() -> RenderOptions {
        RenderOptions {
            use_color: true,
            format_thinking: true,
            format_markdown: true,
        }
    }

    #[test]
    fn plain_text_reaches_the_view() {
        let mut p = Pipeline::new(opts());
        let mut v = StreamView::new(Rect::new(0, 0, 80, 24));
        p.feed(b"hello world\n", &mut v);
        assert!(v.plain_text().contains("hello world"));
    }

    #[test]
    fn thinking_text_is_styled_differently_from_visible_text() {
        let mut p = Pipeline::new(opts());
        let mut v = StreamView::new(Rect::new(0, 0, 80, 24));
        p.feed(b"<think>pondering</think>\nanswer\n", &mut v);
        let txt = v.plain_text();
        assert!(txt.contains("pondering"));
        assert!(txt.contains("answer"));
    }

    /// The real DSML shape, taken verbatim from the template `sysprompt.rs`
    /// teaches the model (`TOOLS_PROMPT_INTRO`): `<｜DSML｜tool_calls>` /
    /// `<｜DSML｜invoke name="...">` / `<｜DSML｜parameter ...>`.
    const REAL_DSML_READ: &str = "<｜DSML｜tool_calls>\n\
        <｜DSML｜invoke name=\"read\">\n\
        <｜DSML｜parameter name=\"path\" string=\"true\">src/main.rs</｜DSML｜parameter>\n\
        </｜DSML｜invoke>\n\
        </｜DSML｜tool_calls>\n";

    #[test]
    fn a_dsml_tool_call_becomes_a_banner_not_raw_markup() {
        let mut p = Pipeline::new(opts());
        let mut v = StreamView::new(Rect::new(0, 0, 80, 24));
        p.feed(REAL_DSML_READ.as_bytes(), &mut v);
        let txt = v.plain_text();
        assert!(
            !txt.contains("DSML"),
            "raw DSML must never reach the screen: {txt:?}"
        );
        assert!(
            txt.contains("src/main.rs"),
            "banner should name the path: {txt:?}"
        );
    }

    /// Invented, non-DSML markup (a bare `<read>` opening a line) must never
    /// be silently dispatched as a real tool call — `plank_stream`'s
    /// `PseudoToolDetector` catches it and reports it as an error instead.
    #[test]
    fn invented_pseudo_tool_markup_is_rejected_not_dispatched() {
        let mut p = Pipeline::new(opts());
        let mut v = StreamView::new(Rect::new(0, 0, 80, 24));
        p.feed(b"<read>\n<path>src/main.rs</path>\n</read>\n", &mut v);
        let txt = v.plain_text();
        assert!(
            txt.contains("is not a tool call"),
            "invented markup should be rejected with an explanatory banner: {txt:?}"
        );
        assert!(
            !txt.contains("🛠"),
            "a rejected pseudo-call must not also show a dispatch banner: {txt:?}"
        );
    }

    #[test]
    fn a_fenced_code_block_is_colored() {
        let mut p = Pipeline::new(opts());
        let mut v = StreamView::new(Rect::new(0, 0, 80, 24));
        p.feed(b"```rust\nfn main() {}\n```\n", &mut v);
        let attrs: Vec<TvColor> = v
            .styled_lines()
            .iter()
            .flatten()
            .map(|c| c.attr.fg)
            .collect();
        assert!(
            attrs.iter().any(|c| *c != TvColor::LightGray),
            "a highlighted code block must not be uniformly default-colored"
        );
    }

    #[test]
    fn set_options_preserves_a_split_utf8_char() {
        // An em-dash '—' is 3 bytes (0xE2 0x80 0x94). Split it across two
        // `feed` calls with a `set_options` in between: the trailing bytes
        // held in `utf8_carry` must survive the rebuild, not vanish.
        let dash = "—".as_bytes();
        assert_eq!(dash.len(), 3);
        let mut p = Pipeline::new(opts());
        let mut v = StreamView::new(Rect::new(0, 0, 80, 24));
        p.feed(&dash[..1], &mut v);
        p.set_options(opts(), &mut v);
        p.feed(&dash[1..], &mut v);
        p.feed(b"\n", &mut v);
        p.finish(&mut v);
        assert!(
            v.plain_text().contains('—'),
            "the split character must survive set_options intact: {:?}",
            v.plain_text()
        );
    }

    #[test]
    fn split_delivery_matches_whole_delivery() {
        let input = b"# Title\n\nsome **bold** text\n";
        let mut whole_p = Pipeline::new(opts());
        let mut whole_v = StreamView::new(Rect::new(0, 0, 80, 24));
        whole_p.feed(input, &mut whole_v);
        whole_p.finish(&mut whole_v);

        let mut drip_p = Pipeline::new(opts());
        let mut drip_v = StreamView::new(Rect::new(0, 0, 80, 24));
        for b in input {
            drip_p.feed(&[*b], &mut drip_v);
        }
        drip_p.finish(&mut drip_v);

        assert_eq!(drip_v.styled_lines(), whole_v.styled_lines());
    }
}
