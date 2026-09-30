# Changelog

## 0.1.7

- A `</think>` quoted in backticks inside a thought no longer ends it. Both
  `StreamRenderer` and `TokenRenderer` took the quoted tag for the control
  token, so it vanished and the rest of the thought rendered as answer text.
  Inline code spans are now tracked while thinking; a span ends at its line,
  so an unmatched backtick cannot hold the thought open past the real close.

## 0.1.6

- A `<think>` right after a Qwen `</tool_call>` is no longer split in two.
  The run stays open after a stanza in case another `<tool_call>` follows, and
  the `<th` it shares with that opener was swallowed into the closed stanza,
  leaving `ink>` as visible text in front of the thought. The renderer now
  settles the run before the first byte that cannot open another stanza.

## 0.1.5

- Qwen3.8's tool-call dialect (`<tool_call>` / `<function=…>` /
  `<parameter=…>`) is parsed and rendered as a banner, adopted from the
  stanza opener like the two DSML spellings, so a stream with no model name
  attached still renders it. Over-escaped HTML entities in Qwen arguments are
  decoded, and selecting the Qwen dialect no longer aborts the render.
- A write whose content streams before its path no longer prints `<file>`.

## 0.1.2

- A DSML error no longer freezes output permanently. `StreamRenderer` used to
  gate every later byte behind a `stream_error` that nothing ever cleared,
  which is correct only for a renderer scoped to a single generation pass. A
  renderer that outlives a pass -- one per connection, as a debug console
  keeps -- went dead at the first bad stanza and rendered nothing afterwards.
  Freezing is now opt-in via `StreamRenderer::set_freeze_on_error` and
  defaults to off; error reporting through `finished().error` is unchanged.

## 0.1.1

- Syntax highlighting now carries text styles in addition to color: keywords
  render **bold** and comments *italic*. Strings, numbers, and normal text are
  unchanged. Each highlighted run still resets with `\x1b[0m`, so terminals that
  ignore a style code fall back to color-only.

## 0.1.0

- Initial release: streaming renderer for model token streams with tool-call
  parsing, thinking-text split, and markdown/syntax highlighting to ANSI.
