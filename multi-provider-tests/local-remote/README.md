# local main → remote sub-agent

The main agent runs on the local ds4 model; `remote-reviewer` sends its sidechain
to `glm5.2` on regolo.ai. This is the configuration cross-engine sub-agents were
built for, and the cheaper of the two to try: only the sidechain costs tokens.

## Run

```sh
export REGOLO_API_KEY=...
./run.sh
```

Nothing else to fill in. The definition is complete:

```yaml
provider: openai                        # regolo.ai is OpenAI-compatible
model: glm5.2
base-url: https://api.regolo.ai/v1      # plank appends /chat/completions
api-key-env: REGOLO_API_KEY             # the variable's NAME, never the key
```

Only the variable's name lives in the file, so it stays committable; plank reads
the key from the environment when the sidechain starts.

## What to check

1. **`/agent`** — `remote-reviewer` shows `[openai glm5.2]`; `inherits-parent`
   shows no engine, because it runs on the parent. Run once with the key unset
   (`env -u REGOLO_API_KEY ./run.sh` will refuse, so unset it *after* starting):
   the line gains `(no REGOLO_API_KEY)` and the model stops being offered that
   definition, while `/subagent:remote-reviewer …` still works and fails clearly.
2. **`/subagent:remote-reviewer what does this directory contain?`** — a
   `[sub-agent: remote-reviewer — ctrl+o to follow]` line, and Ctrl+O shows text
   that reads like glm5.2 rather than the local model. Only the framed report
   enters the main transcript.
3. **Tools inside the sidechain** —
   `/subagent:remote-reviewer count the files here by listing the directory`.
   It must actually call a tool and cite a real number. "I cannot access files"
   means the structured prompt and tool registry are not reaching the provider,
   which is the failure most likely to pass unnoticed.
4. **`/subagent:inherits-parent same question`** — runs on the **local** model,
   since it has no `provider:`. The contrast with (2) is the point.
5. **The swap is restored** — after each dispatch the footer's engine-origin
   segment must still show the local engine. A leaked swap would leave the whole
   session pointed at regolo.ai, which is the worst failure this design can
   produce.
6. **Model-initiated routing** — ask for something that invites delegation
   without naming an agent ("review the run.sh in this directory"), and see
   whether the model reaches for `remote-reviewer` on its own. `/config
   agents.autoRoute false` should stop that while `/subagent` keeps working.

If `glm5.2` is not a model your key can reach, plank reports a provider error and
the session keeps working. List what is available with:

```sh
curl -s -H "Authorization: Bearer $REGOLO_API_KEY" https://api.regolo.ai/v1/models
```

then `REGOLO_MODEL=<id> ./run.sh` — note that overrides the *script's* default,
so also update `model:` in `.plank/agents/remote-reviewer.md`, which is what the
sub-agent actually reads.
