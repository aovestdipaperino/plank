# Changelog

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
