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
| `tools.builtin` | array of strings | no | every builtin tool is offered |
| `settings` | object | no | no extra settings layer |

`systemPrompt` is the only required field, and deliberately so: a `profile`
block without a prompt is a skin over plank's own identity, and activating a
skin as though it were a full agent would misrepresent what is running. A
manifest with a `profile` object but no `systemPrompt` is not recognized as a
profile at all, and `--profile` treats the plugin as though it had none.

Every other field degrades on its own rather than failing the whole profile.
An `accent` that is neither a bare ANSI index (`"160"`) nor a `#rrggbb` hex
triple is ignored with a warning and plank's default green is used instead.
A `logo` that does not point at a readable, decodable PNG falls back to
plank's own art, also with a warning; the sample profile in this repository
ships without a `logo` key specifically to exercise that fallback. A
`displayName` that is not a non-empty string falls back to the plugin's name.
A `settings` value that is not an object is dropped.

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

## The tool-protocol token

A profile's `systemPrompt` file is composed as-is except for one
substitution: the literal text `{{plank:tool-protocol}}`, wherever it
appears, expands to the trained DSML call-syntax text the model was actually
trained against. A profile prompt should almost always include this token,
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
display name, a prompt without the tool-protocol token — warns and falls back rather than refusing to start.

## What actually changes on screen

`displayName` is read in two places: the startup banner (`logo.rs`) and the
terminal window title (`title.rs`). It does not currently reach the Ratatui
status bar footer, despite what the introduction above might suggest — the
footer still shows plank's own segments regardless of the active profile.

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

## A worked example

`examples/profiles/chatbgt` in this repository is a complete, minimal
profile: a household-budget assistant restricted to reading files and
running `bash`, with no logo, so it also exercises the fallback path. Its
`README.md` shows how to run it directly with `--plugin-dir` or install it
permanently under `~/.plank/plugins/dev/`.

`examples/profiles/hal` is a second worked example, this one installable and
with a logo: a mail-and-calendar profile whose tools arrive over MCP from a
server that does not exist yet. See its own `README.md` for what that means in
practice, the exact allow-list it ships, and the auth methods it deliberately
does not support.

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

### Refused on Qwen

`--profile` is refused outright when the active model is Qwen. Qwen's prompt
is built by an entirely separate path (`sysprompt.rs`'s Qwen branch) that has
no notion of a profile's own `systemPrompt`; running `--profile` there would
silently ignore the very prompt the flag exists to install, so it is refused
up front instead, naming the profile and pointing back at `--profile`.
