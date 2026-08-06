# Multi-provider sub-agent tests

Two runnable plank sessions that exercise cross-engine sub-agents in both
directions, checked into the repo so anyone with a provider key can run them.
They hold no credentials: the key is read from the environment, and a sub-agent
definition names the *variable* rather than the secret. The remote side is [regolo.ai](https://regolo.ai) — OpenAI-compatible,
so plank reaches it with `--provider openai` and a base URL.

| Directory | Main agent | Sub-agent |
|---|---|---|
| [`local-remote/`](local-remote) | local ds4 model | `glm5.2` on regolo.ai |
| [`remote-local/`](remote-local) | `glm5.2` on regolo.ai | local ds4 model (`provider: local`) |

## One thing to set

```sh
export REGOLO_API_KEY=...        # https://regolo.ai
./test-regolo.sh                 # is the remote side working at all?
cd local-remote && ./run.sh      # or: cd remote-local && ./run.sh
```

Start with `./test-regolo.sh`: one chat completion straight over curl, using the
same variables the plank tests use. A failure there is the key or the endpoint,
not plank. `./test-regolo.sh --models` lists what your key can reach, which is
what you want when a model name is rejected, and `./test-regolo.sh "your prompt"`
sends something other than the default one-liner.

Everything else is derived, so these are reusable as-is by anyone with a key.
Overridable if you want something other than the defaults:

| Variable | Default | Notes |
|---|---|---|
| `REGOLO_API_KEY` | — | Required. The only thing you must set. |
| `REGOLO_MODEL` | `glm5.2` | List what your key can reach with `./test-regolo.sh --models`. |
| `REGOLO_BASE_URL` | `https://api.regolo.ai/v1` | plank appends `/chat/completions`. |
| `PLANK` | — | An explicit plank binary. |
| `PLANK_REPO` | the repo root | Where to look for `target/{release,debug}/plank`. Defaults to this directory's parent, so a plain `cargo build --release` is enough. |

Without `REGOLO_API_KEY`, `run.sh` says so and stops rather than starting a
session that will fail on first dispatch.

## Which one to run

**`local-remote/`** is the cheap one to try first: the main agent is local, and
only the sub-agent costs tokens. It answers "does a hosted sub-agent work at all"
— tools inside the sidechain, the roster's engine label, the key-variable
handling, and that the engine swap is restored afterwards.

**`remote-local/`** is the inverse and the expensive one to start: because a
definition asks for `provider: local`, plank loads the local model *alongside*
regolo.ai, so startup pays the full local residency (~82 GB, and only one plank
process can hold it). Read that directory's README before running it.

Each directory's README lists what to check. `docs/MULTI-PROVIDER-SMOKE-TESTS.md`
in the plank repo is the fuller manual matrix, including two-key separation,
parallel fan-out timing, and the failure cases.

## Shared plumbing

`lib.sh` holds the endpoint defaults, the key check, and the binary lookup; both
`run.sh` scripts and `test-regolo.sh` source it. Nothing here writes to the repo,
and no key is ever passed to the sub-agent on a command line — a definition names
the *variable*
(`api-key-env: REGOLO_API_KEY`) and plank reads it from the environment, which is
why these directories are safe to commit.
