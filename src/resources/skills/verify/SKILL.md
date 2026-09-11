---
name: verify
description: Use when a change needs to be shown working in the real app rather than only in tests - runs plank itself, in whichever front end the change affects, and reports what actually happened.
argument-hint: [what to verify]
---

# Verify by Running plank

Verify: $ARGUMENTS (if empty, verify the change just made.)

## Before You Run Anything

**The model lock is single-instance.** The ds4 engine maps tens of GiB and
refuses a second process. If the user has a plank session open, you cannot
start another one with the real engine - and the EchoEngine is no substitute
when what you are verifying is turn timing, streaming or anything
model-driven. Check first, and if a session is live, say so and ask rather
than starting a run that will fail or, worse, disturb theirs.

## Pick the Front End the Change Touches

`main.rs` selects: a TTY on both ends gives the Ratatui TUI, piped input gives
the plain line REPL, `--non-interactive` gives the headless stdin protocol.
Verify in the one your change affects - and remember a slash command usually
has two implementations, so a TUI-only check proves half the work.

**Headless** (the default choice - scriptable, no terminal to drive):

    ./target/debug/plank --non-interactive -p "<prompt that exercises it>"

Session transcripts land in `~/.plank/kvcache/<id>.<family>.kv`; grep them to
prove what actually entered the context. Add `--no-session` when you do not
want the run saved.

**Plain REPL:** pipe the input.

**TUI:** needs a real terminal. Drive it through the terminal MCP tools and
screenshot, and quit cleanly with Ctrl+D before closing the terminal - a
killed terminal leaves the session and the model lock behind.

**No model available:** `cargo build` without `refs/ds4` still gives a
runnable plank on the EchoEngine. Enough for command wiring, rendering and
tool dispatch; not enough for anything about generation.

## Report

State the command you ran, quote the output that shows the behavior, and say
plainly whether it worked. If you verified on the EchoEngine or in only one
front end, say which - a partial verification reported as complete is worse
than no verification.

Clean up what you created: remove a scratch session you saved, and leave the
tree as you found it.
