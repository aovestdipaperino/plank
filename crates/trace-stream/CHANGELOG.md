# Changelog

## 0.1.1

- Syntax highlighting now carries text styles in addition to color: keywords
  render **bold** and comments *italic*. Strings, numbers, and normal text are
  unchanged. Each highlighted run still resets with `\x1b[0m`, so terminals that
  ignore a style code fall back to color-only.

## 0.1.0

- Initial release: streaming renderer for model token streams with tool-call
  parsing, thinking-text split, and markdown/syntax highlighting to ANSI.
