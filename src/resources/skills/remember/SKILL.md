---
name: remember
description: Use when the user wants to review, organize or promote plank's memory - proposes moving durable facts between `~/.plank/MEMORY.md`, `./.plank/MEMORY.md` and `AGENTS.md`, and flags duplicates, stale entries and conflicts across the layers.
---

# Memory Review

Review the memory landscape and produce a report of proposed changes, grouped
by action. Do NOT apply anything - present proposals for approval.

## 1. Gather the Layers

- `~/.plank/MEMORY.md` - user scope, follows the user across projects.
- `./.plank/MEMORY.md` - project scope, tied to this checkout.
- `AGENTS.md` - checked in, and read by every contributor and every agent
  alike. Where the project root holds a single `CLAUDE.md`, `agentsmd.rs` ties
  the two together, so count them as one layer rather than two.

Both MEMORY.md files are already in the session context. Read AGENTS.md.

## 2. Classify Each Entry

| Destination | What belongs there |
|---|---|
| `AGENTS.md` | Project conventions every contributor must follow: build and test commands, architecture, invariants, gotchas |
| `./.plank/MEMORY.md` | This user's working notes about this checkout: goals, in-flight work, local constraints |
| `~/.plank/MEMORY.md` | Who the user is and durable cross-project preferences |
| stays put | Session-specific observations, uncertain patterns |

Distinctions that matter:
- Ask who the sentence is addressed to. AGENTS.md talks to anyone who works
  in this repo; MEMORY.md tracks what this particular user is up to. If the
  next contributor would be worse off not knowing it, it is AGENTS.md.
- If the code, the tests or `git log` already answer it, no layer should. When
  a request like that comes up, find out which part of it was not obvious and
  keep that part.
- Habits around workflow - how branches get named, how a release is cut - sit
  on the fence. Ask whether it is theirs or the project's instead of picking
  for them.

## 3. Find Cleanup

- **Duplicates:** a MEMORY.md entry already stated in AGENTS.md - propose
  dropping the copy.
- **Stale:** an entry contradicted by a newer one, or naming a file, function
  or flag that no longer exists. Verify before proposing: grep for it.
- **Conflicts:** two layers disagreeing - propose a resolution and say which
  is more recent.
- **Relative dates:** "last week", "yesterday" - propose the absolute date.

## 4. Report

Group as: Promotions (with destination and rationale), Cleanup, Ambiguous
(needs the user's call), No action needed. If both MEMORY.md files are empty,
say so and offer to review AGENTS.md for staleness instead.

## Rules

- Present every proposal before changing anything.
- Do not create a file that does not exist yet unless the user approves it.
- Ask about ambiguous entries; do not guess.
