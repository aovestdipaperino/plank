# WASM plugins: remaining work

What is left to call the plugin system finished, as of the `wasm` branch.
`docs/WASM-PLUGINS.md` is the design and records what landed; this is the
inventory of what did not, why it matters, and what it would cost.

Ordered by what blocks a user, not by what is interesting to build.

## Blocking: nobody can get a plugin

The system works and is unreachable. A user who downloads plank has no way to
end up with a plugin short of cloning the repository, installing a wasm
toolchain, and building the guests by hand. Everything below this heading is
downstream of that.

**The release ships no guest artifacts.** `release.yml` builds the binary,
tars it, and packages a Homebrew bottle. It never builds `guests/`, so the
screensavers and arcades components exist only on a developer's disk. The fix
is a CI step that runs `guests/build.sh` and attaches the two `.wasm` files, or
better, two `.tar.gz` plugin directories complete with manifests, to the
release. Small job, high value: without it `/plugins install` has nothing to
point at.

**`/plugins install` takes a local directory only.** No URL, no archive. Given
release artifacts, the natural next step is `install <url>`, which means
fetching over the network into a temporary directory, verifying it looks like a
plugin, and copying it in. plank already has an HTTP client for the remote
engine, so this is plumbing rather than design.

**No signing.** The design describes optional minisign signatures with a
publisher key in the trust store, so an update from a known publisher installs
without re-prompting. None of it exists. Trust today is entirely first-use
approval on a SHA-256, which is sound but means every update re-prompts. This
is only worth building once plugins come from somewhere other than a local
path, so it sits behind the two items above.

## The ABI is narrower than the design

Both cuts below were deliberate and are recorded in the design doc. They are
listed here because "cut from v1" and "never happening" are different states
and the difference should not be lost.

**`panel`** was cut as the one surface with no consumer. It is the reason the
layout-arbitration question exists, so adding it means answering that question
first: who wins when three components want a sidebar.

**`token_batch`** was cut because it is the only event that would put a WASM
call inside `viz::StreamRenderer`, which is the path under a byte-parity
contract with the C reference. If it ever returns it should be sampled rather
than per-batch.

**Fifteen events are described and five exist.** Implemented:
`session_start`, `user_prompt_submit`, `pre_tool_use`, `post_tool_use`,
`turn_end`. Missing: `session_end`, `turn_start`, `generation_start`,
`generation_end`, `stop`, `pre_compact`, `post_compact`, `context_pressure`,
`idle`, `activity`, `resize`, `theme_change`, `key`, `focus`, `job_start`,
`job_end`, `subagent_start`, `subagent_end`, `worktree_create`,
`worktree_remove`, `file_edit`, `file_read`. Each is a variant, a firing site,
and a line in the manifest parser. The bus already handles classes and chaining,
so these are cheap individually. Add them when something needs them: an event
nothing fires is a promise, and this project has already shipped three of those
by accident.

**Four capabilities of ten are wired.** `log`, `print`, `state` and `sound`
work. `notify`, `agent` and `session` are declared, approvable, and reach
nothing. `fs`, `net` and `exec` are the three that undo the sandbox and were
left out on purpose; each needs its own decision about what the grant means
before it needs code. A component asking for an unwired capability today gets
approval for something that does not exist, which is worse than refusing it.
The cheap fix is to warn at load when a granted capability has no host
function behind it.

**`frame_mouse` is never delivered.** All four ported games had mouse handlers
and each was removed whole rather than left half-wired. The host has the events;
it does not route them to an open frame. Until it does, a `frame` component is
keyboard-only.

**No `frame_step_text`.** The design offers a JSON fallback for guests that
only draw text, so an author's first afternoon does not involve packing binary
buffers. The host decodes only `PGLY` today.

**No `[config.*]` manifest section.** A plugin cannot declare user-settable
options, so anything configurable has to be baked in or stored through the
`state` capability. The design has these surfacing in the config form and
arriving in `OpenParams`.

## Host-side gaps

**Budgets are wall-clock only.** The design calls for fuel metering plus epoch
interruption plus strike-out. Strike-out exists (three failures disables a
component for the session) and every call has a one second deadline, but there
is no fuel accounting and no per-surface budget. A `frame_step` and a
`tool_call` currently get the same allowance, which is wrong in both directions:
50ms is generous for a frame and a second is stingy for a tool the user is
already waiting on.

**Status bar segments ignore their own priority.** `segment_render` returns a
priority and the registry sorts by it, but the bar joins the rendered cells and
never uses priority for elision when the line overflows. Foreground and
background colours are parsed and dropped too.

**The in-turn loop has no frame support.** `tui_turn_inner` drives the built-in
arcade during a generation but knows nothing about WASM frames. Unreachable
today, since a frame can only be opened while idle, and it is the same poll gate
that caused the five-frames-per-second bug, so it will need the same treatment
the moment a frame should survive a turn.

**No `/plugins disable`, `reload`, or `info`.** Disabling means editing the
directory. Reload does not exist, which is right for `tool` components (their
schemas are in the fingerprinted system prompt) and merely missing for the rest.
`info` would show a component's surfaces, grants, strikes and hash.

## Testing and CI

**CI never compiles the feature.** `ci.yml` runs `cargo clippy --all-targets`
and `cargo test --release` with default features, so the entire `plugins`
feature, every Extism call site, and all 31 integration tests are unbuilt and
unrun on every push. They pass locally and that is the only place. This is the
highest-value item in this document after the release artifacts: a second job
with `--features plugins` plus a `guests/build.sh` step would make the whole
system continuously verified instead of verified when someone remembers.

**Guest artifacts are not reproducible.** Nothing checks that the `.wasm` a
release ships matches what the source in `guests/` builds to. Once artifacts
are published this becomes a real gap, because the trust store keys on the hash
of a file nobody can independently reproduce.

**No performance guard on the hot paths.** There is one frame-budget test, kept
deliberately loose because a wall-clock assertion in a parallel suite measures
the machine. Nothing watches the segment or tool paths at all.

## Structural debt

**`support.rs` exists twice.** The screensavers and arcades guests each carry a
copy, and they must stay behaviour-identical: the RNG in particular, since a
drifting `next_f32` breaks reproducibility silently. A shared crate would fix it
and would also be the first piece of a plugin SDK, which is the thing that makes
third-party authoring plausible.

**Both guests live in this repository.** That is convenient and slightly
dishonest: a real plugin is developed outside the host's tree, and nothing has
proven that path works. Moving one guest out, or building it from a checkout
elsewhere, would test the authoring story rather than assuming it.

**No authoring documentation in the repo.** The design document explains the
system to someone changing plank. There is nothing that explains it to someone
writing a plugin. The blog post drafted in `local/PLUGIN-POST.md` is the closest
thing and is not a repository document.

## Merging

The branch is not merged. It carries twelve commits on top of the five bug
fixes that stayed on `main`, and the two decisions worth making before it lands
are whether the `plugins` feature stays off by default (it should, at +18 MiB)
and whether the arcade removal ships in the same release as the plugins that
replace it. Shipping the removal first would leave a release where the games
simply vanished, which is the one sequencing mistake available here.
