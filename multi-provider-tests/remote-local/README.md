# remote main → local sub-agent

The main agent runs on `glm5.2` at regolo.ai; `cheap-local` carries
`provider: local`, so plank loads the local ds4 model **alongside** the provider
and runs that sidechain on it.

This direction did not work until the `provider: local` marker existed: omitting
`provider:` means "whatever the parent is", which under `--provider` is the remote
model, so a definition meant to be cheap was quietly billed to the expensive one.

## Read this before running

**Startup loads the local model.** The default quant needs ~82 GB resident and
only one plank process can hold it, so this session is as heavy to start as a
purely local one — you get regolo.ai *and* the full local load. plank prints a
line saying a definition asked for the local engine before it loads. Quit any
other plank holding the model first, or startup refuses with the single-instance
error.

If you do not want that, delete `.plank/agents/cheap-local.md` and this becomes an
ordinary provider session with no local load at all — which is also test 6 below.

## Run

```sh
export REGOLO_API_KEY=...
./run.sh
```

Nothing else to fill in. `cheap-local.md` is just:

```yaml
provider: local
```

No model, no URL, no key — it is this process's own engine.

## What to check

1. **Startup order** — a line saying a sub-agent definition asked for the local
   engine, then the model load, then `provider engine ready`.
2. **`/agent`** — `cheap-local` shows `[local]` with **no** key marker (it has no
   credential to be missing); `inherits-parent` shows no engine at all.
3. **`/subagent cheap-local summarise the files in this directory`** — runs on the
   local model: recognisably the local voice, and slow to first token on a cold
   KV. `/usage` should attribute nothing to regolo.ai for it.
4. **`/subagent inherits-parent same question`** — runs on **regolo.ai**, because
   "inherits the parent" means the remote engine here, and it is billed to your
   key. The contrast with (3) is exactly the distinction the two definitions exist
   to make visible.
5. **The swap is restored** — after both, the footer's engine-origin segment still
   shows regolo.ai.
6. **The load is opt-in** — delete `cheap-local.md` and restart: no local load, no
   startup line. Put it back, then dispatch it in a session that started *without*
   it: expect `engine unavailable: no local engine in this session`, never a
   silent run on the remote model the definition declined.

## Why `--provider openai` for regolo.ai

regolo.ai is OpenAI-compatible, so the protocol is `openai` and the endpoint comes
from `--base-url https://api.regolo.ai/v1`, to which plank appends
`/chat/completions`. `--provider` names the wire format, not the vendor.
