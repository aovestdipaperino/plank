# remote main → remote sub-agent, one key

The main agent runs on `glm5.2` at regolo.ai; `remote-coder` sends its sidechain
to `qwen3-coder-next` at the same endpoint, authenticated with the same
`REGOLO_API_KEY`. No local model is loaded, so this is the only one of the three
that starts instantly and runs on any machine.

What it isolates is the case the other two cannot reach: two *different* remote
engines that differ only in the model name. Everything else about them — wire
format, base URL, credential — is identical, so a bug that collapses the two
into one engine (a cached client keyed on the base URL, a key lookup that
memoises the first hit, a swap that restores the wrong model) shows up here as
the sub-agent quietly answering as glm5.2 and nowhere else.

## Run

```sh
export REGOLO_API_KEY=...
./run.sh
```

Nothing else to fill in. The definition is complete:

```yaml
provider: openai                        # regolo.ai is OpenAI-compatible
model: qwen3-coder-next                 # the ONLY difference from the parent
base-url: https://api.regolo.ai/v1      # plank appends /chat/completions
api-key-env: REGOLO_API_KEY             # the same variable the parent uses
```

Only the variable's name lives in the file, so it stays committable; plank reads
the key from the environment when the sidechain starts. One key serving both
engines is the point here — `docs/MULTI-PROVIDER-SMOKE-TESTS.md` covers the
two-key separation case separately.

`sample/` is a small Rust crate to give the sub-agent something real to read.

## What to check

1. **`/agent`** — `remote-coder` shows `[openai qwen3-coder-next]`, not the
   parent's `glm5.2`; `inherits-parent` shows no engine at all.
2. **`/subagent remote-coder which function in sample/ can overflow, and on what
   input?`** — a `[sub-agent: remote-coder — ctrl+o to follow]` line, and Ctrl+O
   showing a sidechain that read the file. `is_prime` in
   `sample/src/main.rs` is the answer, via the `d <= n / d` guard and the
   `large_prime_no_overflow` test that documents the old `d * d <= n`.
3. **It is really the other model** — ask both definitions the same question and
   compare. `/subagent remote-coder name your model` and `/subagent
   inherits-parent name your model` should not agree. Self-reported identity is
   weak evidence on its own, so also check `/usage`: two model rows against one
   key, not one row with the whole total.
4. **Tools inside the sidechain** — `/subagent remote-coder count the files under
   sample/ by listing the directory`. It must call a tool and cite a real number.
   "I cannot access files" means the structured prompt and tool registry are not
   reaching the second provider.
5. **The swap is restored** — after each dispatch the footer's engine-origin
   segment must still show `glm5.2`. This is the failure this direction is most
   likely to expose: with both engines sharing a base URL and a key, a restore
   that compares the wrong field looks like success right up until the main agent
   starts answering as qwen3-coder-next.
6. **Parallel fan-out** — dispatch `remote-coder` and `inherits-parent` on the
   same question in one turn. Both should run concurrently against one key
   (`agents.maxParallel` is 4 in `.plank/settings.json`) and land as two separate
   framed reports, with neither's output appearing in the other's.
7. **The missing-key path** — unset `REGOLO_API_KEY` after startup (`run.sh`
   refuses before it). `/agent` gains `(no REGOLO_API_KEY)` on `remote-coder`
   and the model stops being offered it, while `inherits-parent` keeps working:
   the parent already holds its key from the command line, so only the
   env-sourced definition degrades.

## If a model name is rejected

```sh
./test-regolo.sh --models
```

Then `REGOLO_MODEL=<id> ./run.sh` for the parent. For the sub-agent, override
`REGOLO_SUB_MODEL` *and* edit `model:` in `.plank/agents/remote-coder.md` — the
variable is only what `run.sh` prints, while the definition is what plank reads.
Keep the two ids different, or the test stops testing anything.
