# Harness Tool Concurrency Bugs

Status: observed. Root cause unresolved in the agent tool harness.
Discovered: 2026-07-28, while drafting a Medium post via `medium_create_draft`.

## Summary

The agent's tool harness dispatches all tools in a single frame without
ordering guarantees or write-locks around file mutations. When two tools
that operate on the same file fire in the same frame, their output streams
interleave at the OS level, corrupting the file and producing cascading
failures in subsequent tool calls.

## Failure Modes

### 1. Write + bash race

`write` creates or overwrites a file while `bash` reports on it in the same
frame. The bash output stream lands as literal bytes inside the file content,
including any DSML tool-call syntax that invoked the bash command. The
resulting file contains the intended text plus an embedded tool-call stanza
and its output.

**Symptoms:**
- `edit` fails with "old text anchor not found" — the anchor string was
  written correctly, but the file now contains extra bytes from the bash
  output, so the anchor no longer matches.
- `read` returns a file containing raw DSML markers (`<｜DSML｜tool_calls>`,
  `</think>`) as content rather than Markdown.
- The corrupted file is valid text but structurally wrong — later tools
  operate on the corrupted version.

### 2. Bash heredoc captures DSML syntax as literal stdin

Using `cat << 'POSTEOF'` heredocs to append content to a file, the bash
process receives DSML markers as literal stdin bytes. The system then tries
to parse those stdin bytes as tool calls instead of file content.

**Symptoms:**
- Parse errors from the DSML parser, which sees a tool-call opening inside
  what should be a heredoc.
- The heredoc body is truncated at the point where the DSML marker appears,
  because the parser consumes those bytes as a tool invocation.
- The file ends up with partial content and an embedded tool-call stanza.

### 3. Write-tool content truncated by concurrent bash

The `write` tool's content parameter is a single string. When a bash job in
the same frame produces output before the write completes, the write's byte
stream is interleaved with the bash output at the OS level. The written
content is truncated at the point of interleaving — the tail of the intended
content is replaced by the bash output.

**Symptoms:**
- The file ends abruptly mid-sentence.
- The missing tail is replaced by bash output (e.g. `wc -w` results).
- Subsequent `edit` calls fail because the old text anchor (which included
  the tail) no longer exists in the file.

## Root Cause

The tool harness has no write-lock around file mutations. All tools in a
frame are dispatched concurrently, and the harness provides no ordering
guarantee between a tool that writes a file and a tool that reads or reports
on that file. The OS does not serialize the output streams of concurrent
processes writing to the same file descriptor, so bytes from different tools
interleave at arbitrary boundaries.

## Affected Tool Pairs

Any pair where one tool produces output that another consumes as input,
when both fire in the same frame:

| Tool A (writer) | Tool B (reader/reporter) | Effect |
|---|---|---|
| `write` | `bash` (e.g. `wc -w`, `cat`) | Bash output embedded in file |
| `write` | `read` | File contains bash output, not intended content |
| `edit` | `bash` (e.g. `wc -w`, `cat`) | Bash output embedded in file after edit |
| `write` | `edit` | Edit's old anchor matches corrupted content |
| MCP tool that writes a file | built-in `read` | MCP output interleaved with file content |
| MCP tool that writes a file | built-in `bash` | MCP output interleaved with bash output |

## Workaround

Use Python for file writes instead of the `write` tool or bash heredocs.
Python runs as a single bash command (`python3 -c "..."`) and produces no
stdout unless explicitly printed, so there is no concurrent output stream to
interleave with the file. Example:

```sh
python3 -c "
content = '''...'''
with open('/path/to/file', 'w') as f:
    f.write(content)
"
```

This is single-threaded within the bash process and avoids the race entirely.
The cost is that the file content must be embedded in the Python command,
which is fragile for large files and requires escaping single quotes and
backslashes.

## Unresolved

The root cause — no write-lock around file mutations in the tool harness —
is not fixed in the agent itself. The harness dispatches all tools in a
single frame, and the concurrency model provides no mechanism for a tool to
declare "I am writing file X, do not dispatch any tool that reads file X in
this frame." Fixing this requires either:

1. **Serializing writes**: all file-writing tools run sequentially on a
   single writer thread, with file-reading tools queued behind them.
2. **Frame-level dependency tracking**: the harness infers write-read
   dependencies from tool parameters and serializes only the dependent pair.
3. **Write-lock declaration**: tools declare their write targets in the
   tool schema, and the harness refuses to dispatch a reader of the same
   target in the same frame.

None of these are implemented. The bug is reproducible on demand by
dispatching `write` and `bash` in the same frame with the same file path.
