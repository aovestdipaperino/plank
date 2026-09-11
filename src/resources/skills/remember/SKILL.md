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
- `AGENTS.md` - the committed layer every contributor and every agent reads.
  `agentsmd.rs` links it to a lone `CLAUDE.md` in the project root when one
  exists, so treat them as one layer, not two.

Both MEMORY.md files are already in the session context. Read AGENTS.md.

## 2. Classify Each Entry

| Destination | What belongs there |
|---|---|
| `AGENTS.md` | Project conventions every contributor must follow: build and test commands, architecture, invariants, gotchas |
| `./.plank/MEMORY.md` | This user's working notes about this checkout: goals, in-flight work, local constraints |
| `~/.plank/MEMORY.md` | Who the user is and durable cross-project preferences |
| stays put | Session-specific observations, uncertain patterns |

Distinctions that matter:
- AGENTS.md is instructions for whoever works here; MEMORY.md is what *this*
  user is doing. A fact the next contributor needs belongs in AGENTS.md.
- Anything derivable from the code, the tests or git history belongs in none
  of them. If the user asks to remember something like that, ask what was
  non-obvious about it and record that instead.
- Workflow practice (branch naming, release steps) is ambiguous - ask whether
  it is personal or project-wide rather than guessing.

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
