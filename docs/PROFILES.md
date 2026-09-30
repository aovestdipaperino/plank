# Profiles

A profile is a plugin that also declares a `profile` block in its manifest.
Nothing about the plugin format changes: the same directory, the same
`plugin.json`, the same scan roots. What the `profile` block adds is a way
for `--profile <name>` to launch plank as a different agent entirely, with
its own system prompt, its own set of usable builtin tools, its own settings,
and its own logo, display name and accent color in the interactive TUI (see
"What actually changes on screen" below for the exact reach of the last two).
Without the flag the plugin
contributes exactly as it would otherwise, so a plugin author can ship a
profile and a set of ordinary hooks or skills from the same directory without
one affecting the other.

## The manifest block

The block lives at `profile` inside `plugin.json` (or `.claude-plugin/plugin.json`
for the Claude Code layout), alongside the plugin's existing `name`,
`description` and `version`.

| Field | Type | Required | Default when absent or malformed |
|---|---|---|---|
| `systemPrompt` | string, path to a file | yes | none; without it the manifest is not treated as a profile at all |
| `displayName` | string | no | the plugin's own `name` |
| `logo` | string, path to a PNG | no | plank's own logo |
| `accent` | string | no | plank's built-in green |
| `secondary` | string | no | the word's resting colour; derived from `accent`, or the built-in ramp when neither is set |
| `tools.builtin` | array of strings | no | every builtin tool is offered |
| `settings` | object | no | no extra settings layer |
| `folderContext` | boolean | no | `false`: no launch-folder context |
| `agentsMd` | boolean | no | `false`: `AGENTS.md` is neither read nor offered |
| `recommendedModel` | string, an engine name | no | none: the model is chosen as without a profile |
| `grids` | object, MCP server name to component id | no | empty: no MCP server is routed to a grid |

`systemPrompt` is the only required field, and deliberately so: a `profile`
block without a prompt is a skin over plank's own identity, and activating a
skin as though it were a full agent would misrepresent what is running. A
manifest with a `profile` object but no `systemPrompt` is not recognized as a
profile at all, and `--profile` treats the plugin as though it had none.

Every other field degrades on its own rather than failing the whole profile.
An `accent` that is neither a bare ANSI index (`"160"`) nor a `#rrggbb` hex
triple is ignored with a warning and plank's default green is used instead.
A `secondary` that does not parse warns the same way and falls back to the
derived value, so a typo costs the declared shade but never the profile.
A `logo` that does not point at a readable, decodable PNG falls back to
plank's own art, also with a warning; the sample profile in this repository
ships without a `logo` key specifically to exercise that fallback. A
`displayName` that is not a non-empty string falls back to the plugin's name.
A `settings` value that is not an object is dropped.

### Folder context and AGENTS.md

A session normally starts with context about the folder plank was launched
from. Most profiles are not about that folder, so a profile gets none of it
unless its manifest asks:

- `folderContext: true` restores the git status block and the project memory
  file, `<folder>/.plank/MEMORY.md`. Without it the session still gets the
  user's own memory (`~/.plank/MEMORY.md`), the date and the sub-agent
  roster, and tools still work in the folder; the model is just not told
  about it up front.
- `agentsMd: true` restores reading `AGENTS.md` and `AGENTS.local.md`, the
  launch offer to generate an `AGENTS.md` when there is none, and linking one
  to a lone `CLAUDE.md`.

Both default to `false`, and a value that is not `true` or `false` warns and
counts as `false`, so a typo never turns a context source on. A coding-style
profile sets both to `true`. Without `--profile`, plank behaves as it always
has: both are on.

### Recommended model

`recommendedModel` names an engine from the catalog (`engines.json`, layered
with `~/.plank/engines.local.json`) that suits the profile, and plank uses it
only if it is locally available: the engine's main model file, and every
companion it declares (`mtp`, `vision`), must already exist on disk, at their
derived managed paths under `~/.plank` or at a local engine's `path`. A
recommendation is never downloaded and never asked about, so a missing
companion falls through exactly as a missing main would.
The model is chosen in this order: `--model`, `-m` or `--model:` on the
command line; the recommendation, when its file is on disk; `engine.model`
from settings; the `engines.local.json` default; the catalog default. A used
recommendation prints `plank: using qwen, recommended by profile HAL`. An
engine that is not installed prints one line naming the model used instead,
and a name that is no engine prints one line saying it is ignored; either way
the launch goes on with the next rule. A value that is not a non-empty string
of lowercase letters, digits and dashes warns and is ignored, like any other
malformed field. Without `--profile` the field has no effect.

`tools.builtin`, if present, is expected to be an array of tool names: any
other shape is a mistake, not a restriction the author meant, but a
restriction is exactly what an allow-list is for, so it cannot be allowed to
fail toward "everything is available" the way the other fields do. A
malformed `tools.builtin` therefore fails closed: it produces an *empty*
allow-list, not the unrestricted default. A profile that gets this field
wrong loses every builtin tool rather than gaining tools its author never
listed. This is a deliberate safety property, not an oversight, and it is
worth designing sample and real profiles to notice quickly if it fires
(an agent that suddenly cannot read a file is a loud failure; one that
quietly gained `bash` would not be).

### Grids

`grids` maps an MCP server's name to the id of a WASM frame component that
can render the tables that server returns, such as `dev.plank.csvedit` for
the Turbo Vision CSV editor. It exists so a profile can opt a specific server
into that handling without changing anything about servers it does not
mention: only the pairs a manifest actually lists are routed. A server's
results are otherwise untouched, with one visible difference: a
`plank-frame://` resource from a server the profile does not route is still
taken out of the result, and the model sees a `grid not opened:` note in its
place (`this profile does not route <server>'s grids`) rather than the item
vanishing silently. A malformed entry, a non-string
value, an empty key or an empty value, is dropped with a warning rather than
routed, and the rest of the map is kept; a `grids` value that is not an
object at all warns once and yields no routes. A route is keyed by the MCP
server's final name after every `.mcp.json` layer has merged, so a server of
the same name configured in `./.mcp.json` or `~/.plank/.mcp.json` inherits the
profile's route. How plank actually turns a
tool result into a grid handed to the frame, and back, is described in
[the Grid bridge section of WASM-PLUGINS.md](WASM-PLUGINS.md#grid-bridge).

The component a route names can ship inside the profile itself: a `wasm`
section in the profile's `plugin.json` and the module under its `wasm/`
directory, exactly as any plugin bundles one. A bundled component still needs
the user's approval, but it is asked for when the profile launches rather than
left for `/plugins trust`: the TUI shows a Trust / Not now panel (Not now
first, so a stray Enter changes nothing), the plain REPL asks `[y/N]`, and a
piped stdin declines. Only a component that is new, changed or asking for more
is offered; one with a bad signature or one the user disabled is not, and
declining leaves it held until `/plugins trust <id>`.
If a separately installed plugin declares the same component id, the running
profile's copy is the one kept (with a warning naming both), so a stale
standalone install cannot shadow what the profile bundles. Everywhere else the
first plugin to declare an id keeps it.

## The tool-protocol token

A profile's `systemPrompt` file is composed as-is except for one
substitution: the literal text `{{plank:tool-protocol}}`, wherever it
appears, expands to the trained DSML call-syntax text the model was actually
trained against (on Qwen, to the Qwen call format: see [On Qwen](#on-qwen)). A profile prompt should almost always include this token,
because without it the model has no idea how to format a tool call at all.
A prompt without it still loads, since a chat-only profile may want exactly
that, but startup prints a warning naming the profile so the omission is
never silent.

On a DeepSeek V4.1 model the expanded text, like plank's own prompt, has its
DSML tag names respelled to the V4.1 dialect before any MCP, WASM or `-sys`
text is appended. The respelling touches only the tag names immediately after
the DSML marker, so a profile's own prose is unaffected.

The expansion is the whole prefix of the C reference prompt that precedes the
tool-schema block, and that prefix is not generic. It includes prose that
names specific plank tools by name — `bash_status`, `bash_stop`,
`google_search`, `visit_page`, guidance about reading before editing, and so
on. A profile that restricts `tools.builtin` to a small set, say just `bash`,
still gets this prose in full: the model will read sentences describing
`google_search` or `visit_page` even though neither tool is available to it.
This is not a bug in the substitution; it is a property of reusing the
trained text verbatim, which is the whole reason the token exists rather than
composing the protocol text field-by-field. Authors restricting the tool set
heavily should expect this and, if it matters, steer the model away from the
withheld tools in their own prompt text rather than relying on the protocol
prose to have been trimmed.

The same expansion also carries the GPU-yield note: a profile whose
`tools.builtin` allows `bash` (or omits `tools.builtin`, which allows every
builtin) gets plank's own `# GPU commands` note (`sysprompt::GPU_SUSPEND_NOTE`,
see `docs/ARCHITECTURE.md`) appended after it, the same as the default
prompt, provided the launch has a `ReopenFn` (a local model, not a provider).
A profile that withholds `bash` never sees this note, since there is nothing
for `suspend_model` to apply to.

## The tool allow-list

`tools.builtin`, when present, is the complete list of builtin tools the
model may call under this profile; everything else is withheld. A withheld
tool is not merely refused if called — it is not offered to the model at
all, and if the model calls it anyway (for instance because it remembers
the tool from training, or the name leaks in some other way), the response
is the same "unknown tool" error dispatch gives for a name that was never a
tool to begin with. There is no way for the model to distinguish "this tool
does not exist" from "this tool exists but this profile withheld it", by
design: a distinct error would itself be a hint about what the profile is
hiding.

MCP tools and any other tool source outside the builtin table are unaffected
by this allow-list; it constrains only the tools plank itself implements.

## The settings layer

A profile can carry a `settings` object, applied as its own layer in the
settings stack: `defaults < plugins < profile < ~/.plank < ./.plank < CLI`.
It sits above ordinary plugin settings, so a profile's own choices win over
whatever plugins contribute, but below the user's own `~/.plank` and
project-local `./.plank` configuration and below anything passed on the
command line, so a person running a profile can still override it.

The one thing a profile's `settings` may never touch is `engine.*` — model,
backend, thread count, context size, power mode. Those stay a machine-level
choice made at startup, not something a profile ships. Any `engine.*` keys in
a profile's `settings` block are refused the same way they are refused from
an ordinary plugin's settings.

## What is fatal and what is a warning

Most malformed fields in a profile degrade quietly, as described above, and
plank still launches. Three things are fatal instead, and stop the process
before anything runs:

A `--profile` name that matches no loaded plugin. A name that matches a
plugin, but a plugin whose manifest has no `profile` block at all (asking to
run something that isn't a profile as though it were one). And a
`systemPrompt` file that cannot be read, since a profile with no prompt has
no identity to run under.

Everything else described in this document — a bad accent, a missing or
broken logo, a malformed allow-list, an unusable settings object, a missing
display name, a malformed or unavailable `recommendedModel`, a prompt without the tool-protocol token — warns and falls back rather than refusing to start.

## What actually changes on screen

`displayName` is read in two places: the startup banner (`logo.rs`) and the
terminal window title (`title.rs`). It does not currently reach the Ratatui
status bar footer, despite what the introduction above might suggest — the
footer still shows plank's own segments regardless of the active profile.

### `secondary`: the far end of the shimmer

While a turn runs, the status verb (`Patching…`, `Thinking…`) carries a
highlight that sweeps across the word. `secondary` is the colour the word
**rests** in; the `accent` is the highlight that **travels** across it. The
sweep therefore runs `secondary` at the edges to `accent` at its centre.

The way round to remember: the accent is the thing moving, not the background
it moves over.

`secondary` takes the same two forms as `accent` — a bare ANSI index or a
`#rrggbb` triple — and there are three cases:

* **Neither `accent` nor `secondary`.** The built-in ramp over plank's default
  accent, unchanged, so a plain run and every profile that has not opted in
  look exactly as before.
* **`accent` only.** A resting colour is derived by lightening the accent
  toward white, so the word reads in the profile's own hue while its accent
  sweeps across.
* **`secondary` set.** The word rests in it, whether it is lighter or darker
  than the accent. Nothing requires either to be the brighter one: HAL rests in
  its red `#d0021b` with a white accent washing over it, and a dark resting
  colour such as `"#444444"` gives a dim word the accent lights up as it
  passes.

The shades are quantized to the xterm 6×6×6 color cube, matching the built-in
ramp, so the sweep renders on a 256-color terminal.

`accent` only paints anything on the interactive Ratatui TUI. The
plain-stdout path (used when output is piped, or under `--ui console`)
keeps plank's own colors; it does not read a profile's accent at all. A
profile's visual identity is therefore TUI-only today.

## Sessions are bound to their profile

A saved session records which profile, if any, it was started under, and
refuses to resume under a different one. Resuming plank's own session under
a profile, or a profile's session as plain plank, or one profile's session
under another, are all refused rather than silently mixing an agent's
transcript and tool history with a different agent's identity and allow-list.

This binding is not isolation, though. The `recall` tool searches prior
sessions scoped to the current project, not to the current profile, so a
profile can surface text from a plain-plank session (or another profile's)
that it could never resume directly. Treat the binding as a resume guard,
not a guarantee that one profile's history stays out of another's context.

Session-start context is not profile-aware either: `context.rs` still
prepends the repository's git status and `AGENTS.md` on every run, so a
profile with nothing to do with this codebase's conventions still receives
them.

## Listing profiles

Running `--profile` with no name attached lists the profiles available from
the currently loaded plugins and exits successfully; it is the way to
discover what is installed without guessing a name.

Inside a session, `/plugins` marks each plugin that `--profile` accepts with
`[profile]` after its name, and the one the session is running under with
`[profile, active]`. Profiles installed under `~/.plank/profiles/` are not
scanned, so only the active one appears there.

## Editing the running profile

`/edit-profile` opens the active profile in the built-in editor as one
buffer, the way `/memory` shows every memory file at once. A header comment
comes first: where the profile was loaded from (the plugin origin and root,
with a note when it is the installed copy under `~/.plank/profiles/` rather
than its source), then one row per field with its value and where it comes
from, either `manifest` or the `default` it falls back to. Below it, each
file sits between `<!-- plank-profile: begin ... -->` and `end` markers: the
manifest and the prompt always, the plugin's `settings.json` and `.mcp.json`
only when they exist. The logo is a picture, so it is a header row, not a
section.

Saving checks the whole buffer before writing anything. The manifest must
still parse into a `profile` block with a `systemPrompt`, and the prompt must
not be empty, which is the same gate `/install-profile` applies; a failure
writes nothing and says why. Only the files that changed are written, and
the path in a begin marker is ignored, so editing it cannot redirect a
write.

A profile is set once at startup, so a saved change offers **Restart now**
or **Later**. Restarting saves the session, quits plank the normal way (which
stops its MCP servers and background jobs), and re-executes it from the
launch directory with the same arguments, minus `/resume`, `--worktree`,
`--worktree-pr`, `-p` and `--chdir`, plus `--chdir <session directory>` and
`/resume <session>`. `--profile` is kept as typed: whatever it named is
installed by then, so it resolves without asking. The conversation carries
over; the prompt
changed, so its cache is rebuilt.

`/edit-profile` works only when plank was started with `--profile`, and
editing needs the TUI: the plain REPL prints the same buffer read-only.

## A worked example

`examples/profiles/chatbgt` in this repository is a complete, minimal
profile: a household-budget assistant restricted to reading files and
running `bash`, with no logo, so it also exercises the fallback path. Its
`README.md` shows how to launch it with `--profile examples/profiles/chatbgt`,
which offers to install it on first use.

HAL, a mail profile with a logo, lives in its own repository,
[`aovestdipaperino/plank-profiles`](https://github.com/aovestdipaperino/plank-profiles),
and launches with `plank --profile aovestdipaperino/plank-profiles:HAL`. Its
mail tools come over MCP from Softeria's
[`ms-365-mcp-server`](https://github.com/Softeria/ms-365-mcp-server), started
through `npx` and limited to mail tools, so it can read, flag, draft and file
mail but cannot send or delete it. Its `README.md` covers the builtin
allow-list it ships and how the two limits combine.

## Launching from a path or a repository

`--profile` takes more than an installed name. Its argument is resolved in
this order:

1. **An installed name.** A plugin already loaded for this session that
   declares a profile, or `~/.plank/profiles/<name>/`. It launches directly.
2. **A source installed before.** Every profile install writes the source it
   came from into `.plank-source` in the installed directory. An argument
   matching one launches that installed copy with no fetch and no question.
3. **A local directory**, such as `--profile examples/profiles/chatbgt`.
4. **`owner/repo:folder`**, such as
   `--profile aovestdipaperino/plank-profiles:HAL`: that folder of the
   repository's default branch on GitHub. Plain `owner/repo` works for a
   repository that is a single profile.

For 3 and 4, plank asks `Install the profile <source> into ~/.plank/profiles?`
before fetching anything, installs it exactly as `/install-profile` would, and
launches it. Declining exits. With no terminal to ask on (piped input, or a
headless run), nothing is installed and the error names the `/install-profile`
command that would do it. Because the source is recorded, the same command
line launches the installed copy from then on.

### Versions and updates

A profile's version is its manifest's top-level `version`, as
`MAJOR.MINOR.PATCH`. When `--profile <source>` finds the installed copy by its
recorded source, plank reads the version the source offers now without
fetching the profile: a local directory's manifest, or one request of at most
three seconds for the folder's `plugin.json` through GitHub's contents API
(falling back to `raw.githubusercontent.com` when the API's unauthenticated
hourly limit is spent).

- Not newer (the same or older version): the installed copy launches and
  nothing is downloaded.
- Newer: plank asks `Update profile <name> <installed> -> <available>?`. Yes
  fetches the source and replaces the installed copy, including anything
  changed there with `/edit-profile`; No launches the installed one. With no
  terminal to ask on, it launches the installed copy and says a newer version
  is available.
- The check fails (offline, no `version`, not `MAJOR.MINOR.PATCH`): the
  installed copy launches.

The old copy is moved aside before the new one is installed, and moved back if
the install fails or the source now declares a different profile name, so a
failed update never leaves the profile missing. A profile author releases an
update by raising `version`. The API is asked first because
`raw.githubusercontent.com` is a CDN that can serve an old copy for minutes
after a push, and not the same old copy to every client.

## Installing a profile

`--profile` reads two places: the plugins already loaded for this session, and
a second root, `~/.plank/profiles/`, that exists only for this. That root is
never scanned — `plugins::load_in` does not visit it, so installing a profile
into it contributes nothing to an ordinary session: no skills, no agents, no
hooks, and in particular no MCP server starting behind the user's back. It is
read in exactly two situations: listing the names `--profile` accepts, and
loading the one profile `--profile` named.

`/install-profile <url|owner/repo|path> [name] [--force]` fetches from a git
repository, a marketplace repository, a `.tar.gz`, or a local directory, and
copies the result into `~/.plank/profiles/<name>/`. It is refused when:

- the manifest declares no `profile` block (install it with
  `/install-claude-plugin` instead — it just isn't a profile);
- `systemPrompt` is missing, unreadable, or blank — validated at install time
  rather than left to the launch that treats the same failure as fatal, so a
  broken download is caught before it can strand `--profile`;
- a profile of the same name is already installed — remove it first.

`--force` waives only the unimplemented-hook refusal (hooks that name an event
plank does not fire); the structural refusals above are never waivable.

Both manifest spellings are accepted (`.plank-plugin/plugin.json` and
`.claude-plugin/plugin.json`), which as a side effect makes a plank-spelled
plugin fetchable by `/install-claude-plugin` for the first time.

Removal is manual: `rm -rf ~/.plank/profiles/<name>`. There is no uninstall
command.

### Precedence when a name exists in both places

If a plugin already loaded for this session (from any scan root) has the same
`name` as an installed profile, the scanned plugin wins and the profiles root
is not even consulted — a profile bundled with an installed plugin behaves
like any other plugin unless `--profile` explicitly wants the one in
`~/.plank/profiles/`.

Once `--profile` does resolve to a profile — scanned or installed — it splices
into the plugin set at the highest precedence: its settings layer, its
allow-list and its prompt all take effect as though it were the last (and
therefore winning) plugin loaded.

### On Qwen

A profile runs on a Qwen model too, but its prompt is composed differently,
because the Qwen prompt fences its schemas inside `<tools>` … `</tools>` and
puts the call-format instructions after the fence rather than before a schema
block. The token therefore expands to the whole Tools section of the Qwen
prompt: the fence, holding the allowed builtins (the verbatim Qwen schema
lines, filtered by name), the allowed native extras, and every MCP and WASM
schema, followed by the call format and its reminder. It stops before the
Qwen prompt's `# Rules`, which is plank's prose and is replaced along with the
rest of it; plank's working-style, shell and git sections are left out for
the same reason, as they are under DeepSeek. A prompt without the token gets
the fence alone, appended at the end, so the model can see its tools but is
not told how to call them.
