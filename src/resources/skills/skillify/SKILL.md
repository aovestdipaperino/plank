---
name: skillify
description: Use at the end of a repeatable process the user wants to keep - interviews them about what just happened and writes it as a plank skill under `.plank/skills/`.
argument-hint: [description of the process to capture]
---

# Capture This Session's Process as a Skill

Capture: $ARGUMENTS

## 1. Analyze First

Before asking anything, work out from the session: what repeatable process
happened, what its inputs were, the distinct steps in order, what proves each
step done, where the user corrected or steered you, and which tools it needed.
The corrections matter most - they are the difference between the process the
user wanted and the one you would have run.

## 2. Interview

Use the `ask` tool for every question; never ask in plain prose. The user
always gets a free-form option, so do not invent an "I'll provide edits"
choice - offer the substantive ones. `ask.maxOptions` bounds how many.

- **Round 1:** propose a name and a description; confirm the goal and what
  success looks like.
- **Round 2:** present the steps as a numbered list; confirm arguments, and
  ask where it goes - `./.plank/skills/<name>/` for this project, or
  `~/.plank/skills/<name>/` to follow the user everywhere.
- **Round 3:** per step, what it produces that later steps need, what proves
  it succeeded, whether the user must confirm before it proceeds (always for
  irreversible actions), and any hard rules.

Stop once you have enough. Do not over-ask a three-step process.

## 3. Write the SKILL.md

    ---
    name: <slash-command name: ASCII, no whitespace, no '/' or ':'>
    description: <one line - this is what the model matches on when deciding
                 whether to invoke it, so start with "Use when...">
    argument-hint: <shown in the slash menu; omit if it takes no arguments>
    ---

    # <Title>

    <What this does, and the one principle that governs it.>

    ## Steps
    ### 1. <Step>
    What to do, concretely, with commands.
    **Success criteria:** what proves it done.

Rules that come from how `skills.rs` actually loads a skill:

- The directory holds `SKILL.md`; the file name is fixed.
- A skill with an empty body is skipped entirely. A missing `name` falls back
  to the directory name.
- `$ARGUMENTS` is substituted wherever it appears. With no placeholder and
  non-empty arguments, they are appended as a trailing paragraph, so they are
  never silently dropped - but a skill that uses its arguments should say
  where.
- Project skills override user skills of the same name; plugin skills are
  namespaced (`/plugin:name`) and never claim the bare name.
- Write for the model that will read it mid-turn: state the constraint, then
  the reason. A rule with no reason gets rationalized away.

## 4. Confirm and Save

Show the complete SKILL.md in your reply before writing it, ask for approval
with the `ask` tool, then write it. Afterward, tell the user where it landed,
that it is now `/<name>`, and that it is also model-invocable through the
`skill` tool.
