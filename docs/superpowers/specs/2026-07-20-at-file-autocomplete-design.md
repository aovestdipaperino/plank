# `@` file autocomplete in the TUI prompt

Design for [issue #31](https://github.com/aovestdipaperino/plank/issues/31).

## Problem

The Ratatui TUI prompt has no completion. `editor.rs` has a linenoise-style
`CompletionState` with Tab cycling, but that belongs to the plain-REPL
`LineEditor`; the TUI key loops never bind Tab and never install a callback.
Naming a file in a prompt means typing the whole path from memory or spending a
turn asking the model to find it.

The fix is a unified `@` typeahead: one popup, fed from several sources, ranked
together.

## Scope

Files/directories **and** MCP resources ship together in one change. Agent-name
suggestions are out of scope (they depend on #19). The plain REPL in
`editor.rs` is untouched; whether its Tab cycling later moves onto this engine
is a separate decision.

## Module boundary

All logic lands in a new `src/complete.rs`, self-contained and unit-testable
without a terminal. `ui.rs` is already 5164 lines and gains only glue.

```rust
pub struct AtToken { pub start: usize, pub query: String, pub quoted: bool }
pub fn detect_at_token(text_left_of_cursor: &str) -> Option<AtToken>;

pub struct FileIndex { /* paths, signature, last_refresh */ }
impl FileIndex {
    pub fn build(root: &Path, respect_gitignore: bool) -> Self;
    pub fn signature(&self) -> u64;
    pub fn needs_refresh(&self, now: Instant, git_index_mtime: Option<SystemTime>) -> bool;
}

pub fn rank(query: &str, cands: &[Candidate], limit: usize) -> Vec<Match>;
pub fn longest_common_prefix(matches: &[Match]) -> String;

pub struct Popup { /* candidates, selected, generation */ }
pub enum PopupAction { Consumed, Dismissed, Accept(String), Passthrough }
impl Popup { pub fn handle_key(&mut self, key: KeyEvent, buf: &mut LineBuffer) -> PopupAction; }
```

`ui.rs` gains: an `Option<Popup>` field on `TuiInput`, one dispatch block at the
top of each key loop, and a render call. `tui.rs` gains a `render_popup`
overlay.

### Both key loops

The TUI has two duplicated key loops — `tui_loop` (ui.rs:2028) and
`run_worker_ui` (ui.rs:3369). Both call the same `Popup::handle_key` helper so
they cannot drift.

### Sources

Both sources sit behind one trait and are ranked in a single pass:

```rust
trait Source { fn candidates(&self, q: &str) -> Vec<Candidate>; }
```

- `FileSource` — files and directories.
- `McpResourceSource` — connected servers, addressed `{server}:{uri}`.

On equal score, files outrank MCP resources.

## Trigger and editing behavior

- Trigger: `@` at start of input or after whitespace, matching
  `(^|\s)@([\w-]*)$` against the text left of the cursor. Never on a line
  starting with `!` (shell escape).
- Quoted paths: `@"my file with spaces"`, closing quote optional. Accepting
  yields a correctly quoted token.
- Path expansion: `~` → home, `./` → cwd. `.git/` and friends excluded.
- Directories keep a trailing `/` and leave the popup open for drill-down.
- **Tab** inserts the longest common prefix of surviving candidates, popup
  stays open. **Enter** accepts the selection, replaces the whole token
  including `@`, appends exactly one trailing space. **Esc** dismisses and
  leaves typed text untouched. **Up/Down** move the selection.
- Cap at 15 rows.

## Matching

No new dependencies. `rank()` is a hand-rolled case-insensitive subsequence
matcher (~80 lines) with bonuses for consecutive runs, matches at a
path-segment boundary, and basename hits; ties break toward the shorter path.

A follow-up issue tracks swapping in the `nucleo` crate behind `rank()`. The
signature above is chosen so that is a one-file change.

## Indexing

Shell out rather than adding a walker crate:

- Foreground build: `git ls-files --recurse-submodules`.
- Background pass folds in `git ls-files --others`, plus `--exclude-standard`
  when `respectGitignore` is true.
- Fallback outside a git tree: `rg --files`.
- Every parent directory is synthesized as its own entry with a trailing `/`.
- Signature: FNV hash of (path count, every 16th path). An equal signature
  skips the rebuild.
- Refresh throttled to 5s, bypassed when `.git/index` mtime has moved.

## Concurrency

One worker thread owns the index. The UI sends `Query { generation, text }`
over an mpsc channel and receives `Results { generation, rows }`. The popup
discards any result whose generation is below its current one, so a query
issued while an earlier one is in flight never renders the earlier one's
results. The untracked background pass sends a `Refreshed` message, which tests
await instead of sleeping.

## Rendering

`render_popup` computes `min(15, candidates)` rows and draws `Clear` plus a
bordered list at `input_area.y - height`, clamped to the top of the output
pane. If fewer than 3 rows fit above, it shrinks rather than moving down, so
the status bar is never overlapped and the input area still grows with
multi-line input (the layout added in 0dd8ab5).

## Config

`respectGitignore` lands in `config.rs` alongside existing settings, default
`true`. No CLI flag.

## Errors

A failing `git`/`rg` invocation yields an empty index and no popup rather than
an error dialog; the prompt stays usable. An MCP server that is disconnected or
slow contributes no candidates and never blocks the file results.

## Testing

Every acceptance-criterion bullet from #31 becomes a `#[test]` in
`complete.rs`, run against a `tempfile` git repository:

- `@` after whitespace opens; `@` mid-word (`user@host`) does not; `@` on a
  `!`-prefixed line does not.
- `@src/` lists entries under `src/`, directories carrying a trailing `/`.
- `@~/` and `@./` expand; no result is ever under `.git/`.
- `@"two words` completes a path containing a space; accepting quotes it.
- Tab with candidates `src/utils` and `src/utilities` and input `@src/uti`
  yields `@src/util`, popup still open.
- Enter replaces the token including `@` and leaves exactly one trailing space.
- Esc closes and leaves typed text untouched.
- An untracked file appears after the background pass, awaited via the
  `Refreshed` message.
- Outside a git repository, suggestions come back via the `rg` fallback.
- `respectGitignore: true` hides a gitignored file; `false` shows it.
- Touching `.git/index` forces a refresh before the 5s throttle allows one.
- Two builds over an unchanged file list produce the same signature and the
  second is skipped.
- An MCP resource is offered as `{server}:{uri}` and accepting inserts that
  exact token.
- Popup placement is a pure-geometry test on the rect math: never overlaps the
  status bar, shrinks when space above is tight.
- A stale-generation result is dropped, driven directly through the channel.
