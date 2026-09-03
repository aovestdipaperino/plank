[← Advanced workflows](12-advanced-workflows.md) · [Index](README.md)

# 13. Troubleshooting

## plank feels slow, or forgetful

Check the startup line first:

```
plank: settings in effect (/path/to/.plank/settings.json): threads=3, backend=cpu, ctx=65536
```

A `settings.json` that moved you off Metal or shrank the context is invisible once the UI is up and shows only as "plank got slow". plank prints that line precisely so you can catch it, and it lists only settings actually in effect — with no settings file, or one that changes nothing, there is no line at all.

Then check `/context`. A window that is nearly full compacts often, and a conversation that has been compacted several times has lost detail by design. `/compact` deliberately, or start a fresh session for a new task.

## "Compaction produced no summary"

The pass ran but the model returned nothing usable, so plank kept your conversation instead of rebuilding it on a bad summary — nothing was lost, but nothing was reclaimed either, and the turn was abandoned rather than continuing on a full window.

Run `/compact` again; it usually succeeds on a second attempt. If it keeps failing, give it something to aim at (`/compact summarize the parser work so far`), or start a fresh session and carry over what matters by hand.

Interrupting a pass with `Esc` is the same outcome by a different route, and says so separately: `Compaction interrupted; keeping the previous conversation state.`

## "the model isn't using my instructions"

- Is `AGENTS.md` where plank will find it? It is discovered from the **working directory**, and `--chdir` does not carry project settings with it.
- Is the instruction something the model can already read from the repo? Those get diluted. Memory and `AGENTS.md` are for what it *cannot* re-derive.
- Did the conversation get compacted? Session-start context is re-injected, but a specific instruction you gave in turn three may not have survived as a verbatim sentence. Put durable ones in a file, not in a message.

## A tool did something I did not expect

Turn the machinery on and watch it:

```
/config ui.showToolCalls true
/config ui.showToolResults true
```

Neither changes what the model receives — only what you see.

If it is a class of action you want prevented rather than observed, that is a `PreToolUse` hook exiting 2, or plan mode. See [Advanced workflows](12-advanced-workflows.md).

## A bash command failed with a permission error

The sandbox limits model-initiated writes to the working directory and temp. A build that writes to a shared cache outside the tree will fail, and so will a `write` or `edit` aimed there, with `Tool error: write path escapes workspace: <path>`.

Name the path rather than switching the sandbox off:

```json
{ "writablePaths": ["/Users/me/.cache/my-build"] }
```

in `~/.plank/sandbox.json`. It has to be the user-level file: a project's `./.plank/sandbox.json` can only tighten the sandbox, so `writablePaths` there is ignored. Commands you type with `!` or `!!` are never sandboxed.

## An interactive command hangs

The model's `bash` tool cannot answer a prompt, page through output, or drive an editor. Run those yourself with `!` (or `!!` to keep the output out of the conversation).

## The two log files

The messages the model and the UI see are terse one-liners (`Tool error: visit_page failed: …`). The detail lives on disk:

| File | What is in it |
|---|---|
| `~/.plank/errors.log` | which subsystem failed, for which input, with the full error text |
| `~/.plank/tool-call-errors.log` | tool calls the model emitted that plank could not parse or would not run |

Check the first one whenever a tool failed for a reason the on-screen message did not explain. Logging is best-effort by design — it never turns a recoverable tool error into a crash — so the absence of a line is not proof that nothing happened.

The second one is mostly of interest when the model seems to be *trying* to use a tool and nothing happens. Two entries dominate in practice:

- **`tool calling is not allowed inside <think></think>`** — the model tried to act mid-reasoning. Whether that is dispatched or refused is `engine.thinkingToolCalls` (default `true`, which dispatches it). If you have set it to `false` for ds4 parity, this is the expected consequence.
- **`DSML markup outside a valid tool_calls block`** / **`unexpected DSML tag`** — a malformed call. Usually transient; the model recovers on the next attempt.

## An MCP server's tools vanished — or worse, didn't

`/mcp` shows what is connected. A server that misses the response deadline is dropped **along with all of its tools**, so a slow-starting server looks like a server that has no tools. Raise `mcp.timeoutSecs`.

But a **global** server that fails to start does the opposite of vanishing. Its tool schemas are part of the cached system prompt, so plank substitutes the server's last-known-good advertisement from `~/.plank/mcp-advert/` rather than change that prompt and force a full reload. The model therefore still believes the tools exist and calls fail at dispatch. `~/.plank/errors.log` names it explicitly:

```
mcp: server "x" failed to start (…); substituted its cached advertisement so the
system prompt is byte-identical and the Tier 1 system-prompt KV cache stays valid
```

If a server's tools are failing in a way that makes no sense, that line is the answer. Project-local servers have no cached record and simply are not advertised.

If a server is connected but the model does not seem to know its tools in detail, check `primaryTools`: non-primary tools appear only as a compact directory entry until the model calls `mcp_describe`.

## Which model am I actually talking to?

Do not ask the model. "Which model are you?" is answered from training data, not from your configuration, and it is wrong often enough to be useless — a local model will happily claim to be something else.

Check the configuration instead:

- the **startup line**, which names the backend and context actually in force
- **`/version`** for plank itself
- **`/usage`**, which reports billed tokens only when you are on a hosted provider — a local run has nothing to bill

To confirm a provider or gateway is really wired up end to end, send a prompt whose answer proves the round trip happened rather than one you have to judge:

```
reply with exactly: OLLAMA-OK
```

Run it once per configuration you are testing, with a different token each time. It fails loudly and unambiguously when the request never left, went to the wrong endpoint, or came back from a cache.

## A PDF came back as an error

`read` converts PDFs through a parser before the model sees them. `Tool error: convert <file>: …` means the conversion failed, not that the model refused — the file is corrupt, encrypted, or not actually a PDF.

A successfully converted document is cached under `~/.plank/doc-cache/` by content, so re-reading it is free. PDF is the only format converted this way; other binary files read as bytes.

## A pasted image did nothing

Pasting attaches the image and gives the model its **path**, not its contents. The local ds4 engine is text-only — there is no vision model wired in yet — so a screenshot reaches the model as a filename. Transcribe the part that matters into your prompt.

## The model downloads stopped, or won't start

The model download streams to a `.part` file next to the destination and resumes on the next launch, so an interrupted transfer is not lost.

It refuses outright below 96 GB of RAM, because the default quant needs about 82 GB resident. That is a hard guard, not a warning — you find out before spending hours on the transfer rather than after.

With stdin not attached to a terminal there is nobody to answer the download prompt, so plank exits with instructions rather than hanging your script.

## "no API key" from a provider

`--api-key` wins if given, otherwise `$OPENAI_API_KEY` or `$ANTHROPIC_API_KEY`. With neither set, startup fails with a clear message rather than a confusing API error. The key is deliberately not a `settings.json` key — that file lives inside your working tree.

Also: `--provider` cannot be combined with `--remote` or the local backend selectors. It *is* the engine for that run.

## A remote connection is refused

- Plain `http://` is allowed to localhost only. For anything else, use TLS or pass `--insecure` knowingly.
- `plank serve` itself refuses to start on a non-loopback address without a `--token`. Bind to loopback, give it a token, or pass `plank serve --insecure` if you really mean to serve unauthenticated; this is a different flag from the client's `--insecure`.
- The token comes from `--remote-token` or `$PLANK_REMOTE_TOKEN`.
- A browser client needs its Origin allowed with `--control-origin`; missing and loopback Origins are always allowed, other browser Origins are refused by default.
- A remote client that mirrors but cannot *drive* is a bridge started with `/rc ask`: approve its request locally with `/grant` (or `/grant <session>`, the id in the notice). Plain `/rc` pre-authorizes control and has no such step.
- A remote command that comes back refused rather than running is the remote-safety gate, not a bug: `/open`, bare `/kvcache`, bare `/resume`, `/rc`, `/quit`, `/exit` and `/grant` cannot work over the wire, and the refusal says which case applies.
- A client whose unsent output exceeds `--control-queue-max` is evicted. A very slow link disconnecting mid-turn is that.

## A hosted turn froze after the network dropped

Wi-Fi off, a sleep, a NAT rebind: a silently dropped connection sends no reset, so the socket sits established and black-holed. plank no longer waits on it forever. Ninety seconds of silence is reported as a stalled stream rather than a hang, and the interrupt flag is polled on a clock rather than on arriving bytes, so `Ctrl-C` lands within a quarter second even against a dead socket. If an interrupt is not acknowledged within two seconds, a second `Ctrl-C` force-quits and the status bar says so.

## Ctrl-C is not stopping anything

- During a turn, `Ctrl-C` and `Esc` both interrupt the generation.
- With an arcade game open, the **first** `Ctrl-C` closes the game and the second interrupts the model.
- At an idle prompt, `Ctrl-C` clears the input line; `Ctrl-D` on an empty line quits.

## Shift+Enter inserts nothing

Your terminal does not report it — that needs the kitty keyboard protocol. `Alt+Enter` and `Ctrl-J` insert a newline everywhere.

## The screen keeps going to a starfield

That is the screensaver. `ui.screensaver` takes `1m`, `2m`, `5m`, or `never`. It never comes up mid-turn or over a dialog, and any key dismisses it.

## Animations are distracting

```
/config ui.reducedMotion true
```

collapses every animation — throbber, shimmer, pulse, flash, stall-fade — to a static fallback. `ui.crtOff false` removes the power-off animation on exit.

## I typed a slash command and the model answered it

Unrecognized slash lines go to the model as ordinary text. Check the spelling against `/help`.

One deliberate case: with `ui.easterEggs` set to `false`, the arcade commands stop being commands at all, so `/pelota` reaches the model like any other prompt.

## My settings file has a mistake in it

Nothing breaks. Malformed JSON, a wrongly-typed value, an unknown key, or an unrecognised backend name each fall back to that key's default, because a settings file must never stop plank from starting. The same bad name passed to `--backend` *is* an error — a flag is an explicit instruction, a config file is a preference.

## Reporting a bug

```
/repro
```

Run it before you change anything. It writes `~/.plank/repro/repro-<timestamp>.md` containing the exact prompt the engine would see plus the model, backend, context size, sampling settings, think mode and engine tuning — self-contained, read-only, and enough for a maintainer to reproduce the state that triggered the problem without your session.

Attach that to an issue at [github.com/aovestdipaperino/plank](https://github.com/aovestdipaperino/plank/issues). Strip anything proprietary first — a repro carries your actual prompt and transcript.

---

[← Back to the index](README.md)
