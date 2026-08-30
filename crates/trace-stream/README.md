# trace-stream

The streaming renderer behind the [plank](https://github.com/aovestdipaperino/plank)
coding agent, published on its own so other tools can render a model's token
stream the same way.

A language model streams bytes. Somewhere in them are markdown, fenced code,
`<think>` blocks the user should see differently from the answer, and tool
calls that should appear as banners rather than raw markup. This crate turns
that byte stream into styled terminal output, incrementally, without ever
needing the whole message first.

Three layers, each usable alone:

- **`dsml`** — a strict parser for the DSML tool-call syntax, recognising
  complete stanzas in a stream that arrives a few bytes at a time.
- **`viz`** — [`StreamRenderer`], which splits raw bytes into visible text,
  thinking text, tool banners and error banners, and drives a [`RenderSink`].
- **`render`** — [`TokenRenderer`], a byte-at-a-time renderer emitting ANSI:
  bold, italic, inline code, fenced blocks with keyword highlighting, and
  dimmed thinking text. UTF-8 safe across arbitrary chunk boundaries.

```rust
use trace_stream::render::{RenderOptions, TokenRenderer};
use trace_stream::viz::StreamRenderer;
use trace_stream::TerminalSink;

let opts = RenderOptions { use_color: true, format_thinking: true, format_markdown: true };
let mut stream = StreamRenderer::new(TerminalSink::new(TokenRenderer::new(Vec::new(), opts)));

stream.push("Here is **bold** text\n");
stream.finish();
```

Give `TokenRenderer` a `std::io::Stdout` to write to a terminal, or a `Vec<u8>`
to capture the ANSI and convert it to something else — which is what
[turbo-debug-console](https://github.com/aovestdipaperino/turbo-debug-console)
does to render a stream into a text-mode UI.

Zero dependencies beyond `std`.

## License

MIT
