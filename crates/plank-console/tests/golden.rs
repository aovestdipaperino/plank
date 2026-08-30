// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! End-to-end: a recorded stream through the whole pipeline, asserted as a
//! rendered cell grid. Regenerate with `PLANK_REGEN_GOLDEN=1 cargo test -p plank-console`.

use plank_console::pipeline::Pipeline;
use plank_console::streamview::StreamView;
use plank_stream::render::RenderOptions;
use turbo_vision::core::geometry::Rect;

/// The DSML shape here is taken verbatim from the template `sysprompt.rs`
/// teaches the model (`TOOLS_PROMPT_INTRO`): `<｜DSML｜tool_calls>` /
/// `<｜DSML｜invoke name="...">` / `<｜DSML｜parameter ...>`, so this fixture
/// exercises the real dispatch path rather than invented markup.
const CAPTURE: &str = "\
<think>The user wants a guard on the parser.</think>
Here is the change:

```rust
fn feed(&mut self, b: u8) {
    self.buf.push(b);
}
```

Applying it now.
<｜DSML｜tool_calls>
<｜DSML｜invoke name=\"read\">
<｜DSML｜parameter name=\"path\" string=\"true\">src/dsml.rs</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>
";

/// Renders the grid as `char` plus a hex attribute per cell, one line per row.
fn render() -> String {
    let mut p = Pipeline::new(RenderOptions {
        use_color: true,
        format_thinking: true,
        format_markdown: true,
    });
    let mut v = StreamView::new(Rect::new(0, 0, 80, 24));
    p.feed(CAPTURE.as_bytes(), &mut v);
    p.finish(&mut v);

    v.styled_lines()
        .iter()
        .map(|line| {
            let text: String = line.iter().map(|c| c.ch).collect();
            let attrs = line.iter().fold(String::new(), |mut acc, c| {
                use std::fmt::Write as _;
                let _ = write!(acc, "{:02x}", c.attr.to_u8());
                acc
            });
            format!("{text}\n  {attrs}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn golden_stream_render() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/stream.golden");
    let got = render();
    if std::env::var("PLANK_REGEN_GOLDEN").is_ok() {
        std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap()).unwrap();
        std::fs::write(path, &got).unwrap();
        return;
    }
    let want = std::fs::read_to_string(path).expect("run with PLANK_REGEN_GOLDEN=1 to create it");
    assert_eq!(got, want, "rendered stream changed");
}
