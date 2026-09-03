# Writing a plank plugin

This is the guide for someone **writing** a plugin. `WASM-PLUGINS.md` next to it
is the design — read that if you are changing plank itself; read this if you want
plank to run your code.

Everything here describes what is implemented, not what is designed. Where the
two differ, this file wins and says so.

## What a plugin is

A directory:

```
my-plugin/
  .plank-plugin/
    plugin.json        # the manifest
  wasm/
    thing.wasm         # one or more modules
    thing.wasm.minisig # optional signature
```

plank looks for plugins in three places:

1. `--plugin-dir <path>` — one or more, for development. This is the flag to use
   while writing one.
2. `./.plank/plugins/` — project-local, checked in with a repo
3. `~/.plank/plugins/dev/` — user-global, where `/plugins install` puts things

Project-local plugins are **default-deny**: cloning a repo must not hand you
executable code, so the first session in a repo that ships one asks.

> The design document lists `$PLANK_PLUGIN_PATH` as the development location.
> **It is not implemented** — nothing reads that variable. Use `--plugin-dir`.

`.claude-plugin/plugin.json` is accepted as an alternative spelling, for a plugin
that wants to serve both tools.

## The smallest plugin that works

Two files and a build. A Rust guest, using the Extism PDK:

```toml
# Cargo.toml — deliberately not a member of any workspace: it targets wasm32
[workspace]

[package]
name = "hello-plank"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib"]

[dependencies]
extism-pdk = "1"
```

```rust
// src/lib.rs
use extism_pdk::*;

/// Mandatory. Return the ABI major version you were built against.
///
/// Extism cannot type-check exports, so this handshake is what stands in for
/// it: a module that will not say what it speaks is refused at load, with the
/// version named, rather than failing later inside a frame.
#[plugin_fn]
pub fn plank_abi() -> FnResult<String> {
    Ok("1".to_string())
}

#[plugin_fn]
pub fn command_specs() -> FnResult<String> {
    Ok(r#"[{"name": "hello", "args": "", "desc": "say hello"}]"#.to_string())
}

#[plugin_fn]
pub fn command_run(_input: String) -> FnResult<String> {
    Ok(r#"{"print": ["hello from wasm"]}"#.to_string())
}
```

```json
{
  "name": "hello-plank",
  "version": "0.1.0",
  "wasm": [
    {
      "id": "dev.example.hello",
      "module": "hello.wasm",
      "surfaces": ["command"]
    }
  ]
}
```

```sh
cargo build --release --target wasm32-unknown-unknown
mkdir -p my-plugin/.plank-plugin my-plugin/wasm
cp plugin.json my-plugin/.plank-plugin/
cp target/wasm32-unknown-unknown/release/hello_plank.wasm my-plugin/wasm/hello.wasm
plank --plugin-dir my-plugin
```

First run asks you to approve it, listing the surfaces and capabilities. Then
`/hello`.

Note the module is named by the manifest (`hello.wasm`), not by cargo
(`hello_plank.wasm`). A mismatch is a component that loads and then cannot find
its own code.

## Manifest reference

Top level: `name`, `version`, `description`, `author`, and `wasm` — an array of
component objects (a lone object is accepted for the one-component case).

Per component:

| key | meaning |
| --- | --- |
| `id` | reverse-DNS, globally unique. This is the trust identity and what `/plugins info` takes. |
| `module` | file under `wasm/`. |
| `abi` | ABI major version; defaults to the current one. Cross-checked against `plank_abi()`. |
| `surfaces` | any of `command`, `tool`, `segment`, `observer`, `frame`. |
| `capabilities` | any of `log`, `print`, `notify`, `state`, `sound`, `fs`, `net`, `exec`. |
| `events` | which lifecycle events you want (see below). |
| `kind` | `frame` only: `screensaver` or `arcade`. |
| `veiled` | `frame` only: leave the transcript dimly visible underneath. |
| `min_size` | `frame` only: `{"w": 30, "h": 9}`. Opening in a smaller terminal is refused with the numbers. |
| `frames` | `frame` only: named faces this component offers, each addressable as `/plugin:face`. |
| `config` | user-settable options (see below). |

Anything malformed is **dropped and named** in a warning rather than ignored: an
option or event that vanishes silently presents to you as "plank ignores my
manifest", which is the least debuggable failure this system has. Run `/plugins`
to see the warnings.

## Surfaces

A surface is a contract: exports plank will call, and what it does with the
results. Claim only what you implement — claiming a surface without its exports
is a load error naming the missing ones.

### `command` — slash commands

- `command_specs() -> [{name, args, desc}]`, read once at load.
- `command_run(json) -> {print: [..], inject: "..", prompt: ".."}` — write
  scrollback lines, put text in the input box, or submit a prompt as if typed.

Your commands are addressable as `/<plugin>:<name>`, and as the bare `/<name>`
when nothing else claims it.

### `tool` — tools the model can call

- `tool_specs() -> [{name, description, parameters}]` with a JSON Schema.
- `tool_call(json) -> string` — the observation the model sees.

Read **once at load**, and that is a requirement rather than an optimisation:
tool schemas are part of the fingerprinted system prompt, so a list that changed
mid-session would invalidate plank's KV checkpoint every time it changed. This
is also why `/plugins reload` does not exist.

### `segment` — a status-bar cell

- `segment_render(json) -> {text, priority, fg, bg}`.

`priority` decides what survives when the bar overflows: contributed cells are
dropped lowest-priority first. `fg`/`bg` are `[r, g, b]`. Keep `text` short and
expect to be elided — plank's own segments are never dropped on your behalf.

### `observer` — watching, owning nothing

- `on_event(json) -> {}`

### `frame` — the whole terminal

- `frame_open(json) -> {veiled?}` — `OpenParams` carries `{w, h, seed, arg, config}`.
- `frame_step(json) -> bytes` **or** `frame_step_text(json) -> {lines: [...]}`.
- `frame_key(json) -> {stay}|{close: "line"}`
- `frame_mouse(json) -> Outcome` — **optional**. Without it your frame is
  keyboard-only; plank does not strike you for declining an optional export.
- `frame_close() -> {scrollback?}`

Two ways to draw, and you pick one:

**`frame_step_text`** is the one to start with. Return
`{"lines": [{"text": "hi", "fg": [0,255,0], "bold": true}]}` — one entry per row
from the top, each laid out from column zero. Colour and bold are per line, not
per span. Spaces are left undrawn (so a veiled frame shows through your padding),
and rows past `h` or columns past `w` are clipped rather than refused.

**`frame_step`** returns a packed glyph buffer — one `memcpy` out of linear
memory, and what you graduate to when per-glyph colour or a full screen at 30fps
matters. All little-endian:

```text
header: u32 magic 'PGLY' | u16 version | u16 count | u16 w | u16 h
glyph:  u16 x | u16 y | u32 ch (UTF-32) | u8 r | u8 g | u8 b | u8 flags
```

`flags` bit 0 is bold; bit 1 says a background colour follows as three more
bytes, so the common case stays ten bytes per glyph. `guests/support` encodes
this for you, and decoding is total — a malformed buffer costs you the frame,
never the session.

`StepParams` carries `{dt_ms, w, h, now_ms}`, and `dt_ms` is clamped host-side —
a suspended terminal cannot teleport your simulation.

## Capabilities

Declared in the manifest, shown to the user at approval, and **never widened
silently**: an update that asks for more re-prompts even with a valid signature.

Wired today: `log`, `print`, `state` (a per-component KV store — the only
persistence most components need, and it needs no filesystem grant), `sound`,
and `notify`.

`state` has quotas, checked before anything touches the disk: 1 MiB per value,
255-byte keys, 256 keys, 16 MiB in total per component. An over-quota
`plank_state_set` returns an error that starts with your component id
(`'<id>' state value for 'k' is N bytes, more than the ...-byte limit`) and
stores nothing; the store is exactly as it was before the call.

`notify(title, body)` fires a desktop notification. Your component id prefixes
the title whether you like it or not — a notification that does not say which
plugin produced it is one the user cannot act on or switch off — and both
strings are clipped to 200 characters. It respects the user's own notification
setting rather than routing around it: a plugin is not more entitled to
interrupt than plank is.

Declared but reaching nothing yet: `agent`, `session`. plank warns at load if
you ask for one, because approving a capability that does not exist is worse
than refusing it.

Deliberately unimplemented: `fs`, `net`, `exec`. These are the three that undo
the sandbox, and each needs its own decision about what the grant means before
it gets code.

## User-settable options

```json
"config": {
  "difficulty": {"type": "enum", "values": ["easy", "hard"], "default": "hard"},
  "sound":      {"type": "bool", "default": true},
  "speed":      {"type": "int", "min": 1, "max": 10, "default": 5},
  "label":      {"type": "text", "default": "hi"}
}
```

A default is **required**, and a default your own declaration would reject is a
load warning — better you hear it now than a user does later.

Values arrive in `OpenParams.config`, already validated, so you can trust them
without re-checking: bools and ints arrive typed, not as strings. A stored value
that has stopped being acceptable — an enum member you removed in an update —
falls back to your default rather than refusing to open.

You never read a config file. Users set these in the `/config` form's `plugins`
section, or with `/config pluginConfig.<id>.<option> <value>`.

## Events

Implemented: `session_start`, `turn_start`, `user_prompt_submit`,
`pre_tool_use`, `post_tool_use`, `turn_end`, `pre_compact`, `post_compact`,
`session_end`.

`user_prompt_submit` and `post_tool_use` are *transform* (return a replacement
and it is used), `pre_tool_use` is *veto* (block with a reason, which the model
sees as the tool's error), and the rest are notify-only — their replies are
ignored. `turn_start` is deliberately notify: `user_prompt_submit` already owns
refusing a turn, and two events that could both stop one would make "why did
nothing happen" ambiguous.

The design lists around twenty. The rest are not wired, and subscribing to one
warns rather than silently never firing. If you need one, that is the argument
for adding it — an event nothing fires is a promise, and plank has shipped
three of those by accident.

## Packaging, signing, installing

`/plugins install <dir>` copies a plugin into `~/.plank/plugins/`. It copies
rather than links, because a plugin that changed under a running session would
be one whose approved hash no longer describes what is loaded. It refuses to
overwrite: replacing an installed plugin is new bytes under an approved name, so
it is remove-then-install.

`/plugins install <url>` fetches a `.tar.gz` over https (or http to loopback).
Both forms — `<directory>` and `<url>` — scan the source tree for a symlink
before copying it: one that escapes the tree, or that points at a directory,
is refused; one whose target resolves to a plain file inside the tree (the
ordinary "same file under two names" idiom) is allowed.

To sign, with [minisign](https://jedisct1.github.io/minisign/):

```sh
minisign -G -p publisher.pub -s publisher.key     # once
minisign -S -s publisher.key -m my-plugin/wasm/thing.wasm
```

Ship the `.minisig` beside the module. A user runs
`/plugins publisher publisher.pub` once, and thereafter **updates you sign
install without re-prompting** — the hash still changes, but the provenance
does not. A signature never widens capabilities, and a bad signature is refused
outright rather than prompted. Both minisign variants work, prehashed (the
default) and `-l`.

## Diagnostics

- `/plugins` — everything loaded, everything held back, and every load warning.
- `/plugins info <id>` — surfaces, grants with unwired ones marked, strikes,
  module hash and whether it matches the approval, signature status.
- `/plugins disable <id>` — switch one off without forgetting its approval.

**Strikes**: three consecutive failures disable a component for the session.
Every call has a deadline, so a hang is a failure. If your component vanishes
mid-session, `/plugins info` will say why.

**Budgets** are per surface, and these are the targets: `frame_step` 50 ms (a
frame at 30fps has 33 ms for everything, and plank has to draw too),
`frame_key`/`frame_mouse` and `segment_render` 20 ms (input must feel immediate;
the bar repaints while you type), `tool_call` 1000 ms (the user is already
waiting on the model), everything else 200 ms.

A call is only *failed* at four times its target. The measurement is wall-clock,
and a wall-clock measurement on a busy machine measures the machine — enforcing
at the target exactly would strike out a working game on a loaded laptop. Treat
the target as what to design for and the 4× as the line where plank decides you
are broken rather than unlucky.

This is not fuel metering: fuel meters guest instructions and needs wasmtime
configuration Extism does not expose. A guest can still burn its whole budget
inside one host call. What these do catch is the case that matters — a component
ruining the frame rate.

**Memory** is capped at 64 MiB of linear memory (1024 wasm pages, set in the
manifest plank builds for you). Growing past it is a trap, and a trap is a
strike: three and your component is disabled for the session. Keep frame
buffers sized to the `w`/`h` you were handed rather than to a worst case.

## Rules that will bite you

- **Export `plank_abi`.** Nothing else runs until it answers.
- **Payloads only ever gain fields.** Ignore keys you do not know; tolerate
  optional ones being absent. plank holds itself to the same rule. This is how
  a plugin you cannot recompile keeps working.
- **One call at a time.** plank serialises calls per instance; no re-entrancy.
- **Do not expect a filesystem.** Use `state`. A component asking for `fs` today
  is asking for something that does not exist.
- **A frame is a screen, not a stream.** Declaring more glyphs than the area can
  hold is refused rather than allocated on your say-so.

## Worked examples in this repo

- `guests/screensavers` and `guests/arcades` — real `frame` + `command`
  components, sharing `guests/support` for the RNG and glyph packing.
- `spike/text-guest` — the smallest `frame_step_text` component.
- `spike/abi-guest` — one component exercising every surface, used by plank's
  own integration tests.

`guests/build.sh` builds them; `guests/package.sh` assembles installable
directories and records `SHA256SUMS`; `guests/verify.sh` proves a clean rebuild
lands on the same bytes.

The "smallest plugin" section above was written and then **followed literally**
from a directory outside the plank tree — a fresh crate, the manifest as printed,
`--plugin-dir` — and the result discovered with no warnings and printed
`hello from wasm` when its command ran. If the walkthrough stops working, that is
a bug in plank or in this file, not in your setup.
