---
name: verify
description: Use when a change needs to be shown working in the real app rather than only in tests - runs plank itself, in whichever front end the change affects, and reports what actually happened.
argument-hint: [what to verify]
---

# Verify by Running plank

Verify: $ARGUMENTS (if empty, verify the change just made.)

## Before You Run Anything

**Only one plank can hold the model.** Tens of GiB get mapped, and the ds4
engine turns away whoever asks second. So a session the user already has open
rules out starting your own against the real engine, and swapping in the
EchoEngine does not rescue a check about streaming, turn timing or anything
else the model drives. Look before you launch: if something is live, say so
and ask, instead of firing off a run that either fails or walks over theirs.

## Pick the Front End the Change Touches

`main.rs` decides which one you get: TTYs on both ends bring up the Ratatui
TUI, a pipe drops to the plain line REPL, `--non-interactive` runs the
headless stdin protocol. Exercise whichever one your change lands in, keeping
in mind that a slash command is normally written twice - checking it in the
TUI leaves the other half unproven.

**Headless** (the default choice - scriptable, no terminal to drive):

    ./target/debug/plank --non-interactive -p "<prompt that exercises it>"

Session transcripts land in `~/.plank/kvcache/<id>.<family>.kv`; grep them to
prove what actually entered the context. Add `--no-session` when you do not
want the run saved.

**Plain REPL:** pipe the input.

**TUI:** there has to be a real terminal. Drive one with the terminal MCP
tools and take screenshots, then leave with Ctrl+D before the terminal itself
goes away - killing the window strands the session and keeps the model lock
held.

**No model available:** a `cargo build` with `refs/ds4` absent still produces
a plank you can run, backed by the EchoEngine. Command wiring, rendering and
tool dispatch are all testable that way; nothing about generation is.

## Report

Give the command, quote the lines of output that actually demonstrate the
behavior, and answer the question of whether it worked. Name the limits too -
the EchoEngine, or a single front end - because a half-check described as a
full one misleads in a way that no check at all does not.

Clean up what you created: remove a scratch session you saved, and leave the
tree as you found it.
