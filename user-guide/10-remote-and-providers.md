[← Extending plank](09-extending.md) · [Index](README.md) · Next: [The arcade →](11-arcade.md)

# 10. Remote and hosted engines

All of this is off by default. A plain `plank` runs the local Metal engine and nothing else.

## Hosted providers

plank can drive a hosted model instead of the local one. The provider sits behind the same engine boundary as the Metal backend, so tools, sessions, `/btw`, compaction, and the rest of the agent loop behave identically — native provider tool calls are translated back into plank's own tool protocol on the way through.

```sh
export OPENAI_API_KEY=sk-...
plank --provider openai --model gpt-4o

export ANTHROPIC_API_KEY=sk-ant-...
plank --provider anthropic --model claude-sonnet-4-5
```

One-shot works too: `plank --provider anthropic --model <name> -p "…"`.

| Flag | What |
|---|---|
| `--provider openai\|anthropic` | provider family. `openai` speaks OpenAI-compatible Chat Completions; `anthropic` speaks the Messages API. |
| `--model NAME` | the provider's model name, not a GGUF path. Required with `--provider`. |
| `--api-key KEY` | the key, if you would rather not use the env var. Prefer the env var — a key on the command line lands in your shell history. |
| `--base-url URL` | override the endpoint. Defaults to `https://api.openai.com/v1` and `https://api.anthropic.com/v1`. |
| `--provider-cache on\|off` | Anthropic prompt caching over the stable prefix (tools + system). On by default; ignored for OpenAI. |

**Key resolution** — `--api-key` wins, otherwise `$OPENAI_API_KEY` or `$ANTHROPIC_API_KEY`. With neither, startup fails with a clear message rather than a confusing API error.

**OpenAI-compatible gateways** — `--provider openai` plus `--base-url` reaches anything speaking that shape: vLLM, Ollama, OpenRouter, Together, LM Studio.

```sh
plank --provider openai --model llama3.3 \
      --base-url http://localhost:11434/v1 --api-key ollama
```

**What stays the same** — every tool, MCP tools, `@` completion, sessions and `/resume`, `/btw`, compaction. The one difference is the system prompt: a provider gets plank's own prompt with native tool definitions, never the byte-parity DeepSeek prompt, which is meant only for the local model it was trained on.

**Two hosted models on one key** — a subagent definition can point at a different model on the *same* endpoint with the *same* credential. Only the model line differs from the parent:

```yaml
provider: openai
model: qwen3-coder-next                 # the only difference from the parent
base-url: https://api.regolo.ai/v1
api-key-env: REGOLO_API_KEY             # the same variable the parent uses
```

Since only the variable's name is in the file, the definition is safe to commit. To confirm the sidechain really reached the second model, check `/usage`: two model rows against one key rather than a single total. Asking the model to name itself is weaker evidence — models are unreliable about their own identity, and billing is not.

**Notes** — `--provider` cannot be combined with `--remote` or the local backend selectors; it *is* the engine for that run. `/usage` reports billed tokens including cache reads, writes, and hit rate. The key is never written to `settings.json`.

## Serve and connect

Run the model on the machine that has the GPU, work from the machine you are sitting at.

```sh
# on the Metal box
plank serve

# on the laptop
plank --remote https://metal-box:PORT
```

The transport is synchronous, adds no async runtime, and streams tokens as they generate.

| Flag | What |
|---|---|
| `--remote URL` | drive a remote `plank serve` host instead of a local engine |
| `--remote-token TOK` | bearer token (or `$PLANK_REMOTE_TOKEN`) |
| `--insecure` | allow plaintext `http://` to a non-loopback host |

Plain `http://` is allowed to localhost and refused elsewhere unless you pass `--insecure`. Keep a real deployment behind an SSH tunnel or a TLS reverse proxy.

The server side has its own guard, and its own `--insecure`, which means something different. `plank serve --listen ADDR` on a non-loopback address with no `--token` refuses to start, because that hands the model to anyone who can reach the port; bind to `127.0.0.1`, pass a token, or pass `plank serve --insecure` to serve unauthenticated anyway. So the client flag waives TLS, the server flag waives authentication, and each is checked only by the side it belongs to. The server also caps a request body at 16 MiB and drops a connection that sits silent for 30 s mid-request.

## Shared engine

```sh
plank serve --shared-engine
```

loads the weights once and serves many concurrent sessions from a single cooperative GPU thread. The scheduler round-robins at token granularity, so sessions are **time-sliced, not parallel** — there is one Metal queue. A freshly attached session restores the warm system-prompt prefix instead of cold-prefilling it.

| Flag | What |
|---|---|
| `--max-sessions N` | admission cap (default 8) |
| `--kv-budget-bytes B` | aggregate KV-bytes budget; reject an attach past it rather than OOM |
| `--session-ctx-size N` | per-session context window (0 = model max; a client's own request overrides) |
| `--idle-reclaim-secs S` | snapshot an idle session's KV to disk and restore it on the next request |

`/info` reports live-session and KV accounting.

A host session belongs to a client, not to a turn. The client sends a stable `X-Plank-Client-Id` header, so many turns from one laptop reuse one warm session instead of attaching a new one each time; a session idle for 30 minutes is swept. An older client that does not send the header still works and is keyed per turn as before.

## Remote control

```sh
plank --control            # loopback WebSocket on 127.0.0.1:31415
```

Another process, a browser, or a terminal client can attach to a **running** plank instance: it mirrors the output, sends prompts, commands, `/btw` questions and interrupts, and can take or hand back control. One controller at a time, many mirrors, with a reconnect grace window. A self-contained web client is served at `/`.

```sh
plank remote ws://127.0.0.1:31415/
```

is the terminal client: typed lines become prompts, slash lines become commands, `/btw <q>` becomes a side question, and Ctrl-C interrupts. The token defaults to `$PLANK_REMOTE_TOKEN`.

| Flag | What |
|---|---|
| `--control[=ADDR]` | start the server (default `127.0.0.1:31415`, loopback only) |
| `--control-token TOKEN` | shared bearer token; otherwise `$PLANK_REMOTE_TOKEN`, otherwise one is generated and printed once to stderr |
| `--control-allow` | let a remote client take control without a local grant (implied in headless server mode) |
| `--control-origin ORIGIN` | allow a browser Origin on the WebSocket upgrade (repeatable or comma-separated) |
| `--control-queue-max BYTES` | per-client outbound queue cap; a client that exceeds it is evicted (default 1048576) |

Missing and loopback Origins are always allowed; other browser Origins are refused by default.

`/rc` (and `/rc on`) pre-authorizes control: typing the command is your consent, so a client opening the link can drive immediately. `/rc ask` starts the same bridge without that consent. An attaching client then mirrors output but cannot drive; its request surfaces locally as `[remote session 3 wants control — /grant or /grant 3 to allow]` and waits. Bare `/grant` approves the oldest waiting request, `/grant 3` approves that session. Approving one request declines any others, since only one client can drive at a time.

Some commands are refused over the wire rather than queued: `/open`, bare `/kvcache` and bare `/resume` need the local terminal, and `/rc`, `/quit`, `/exit` and `/grant` would cut off the client running them. The refusal comes back with its reason. The same commands with an argument (`/kvcache gc`, `/resume 3`) are not interactive and work normally.

## `--ui-remote`

For driving the TUI from a test harness:

```sh
plank --ui-remote=7777
```

Opens a `127.0.0.1`-only listener (bare `--ui-remote` picks an ephemeral port and prints it to stderr) accepting line-delimited JSON `keypress`, `snapshot`, and `uitree` commands. `snapshot` and `uitree` replies are held until the screen reflects any keys sent first, so a harness can assert without sleeping. One client at a time; a second queues.

---

Next: [The arcade →](11-arcade.md)
