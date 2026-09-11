---
name: skillify
description: Use at the end of a repeatable process the user wants to keep - interviews them about what just happened and writes it as a plank skill under `.plank/skills/`.
argument-hint: [description of the process to capture]
---

# Capture This Session's Process as a Skill

Capture: $ARGUMENTS

## 1. Analyze First

Reconstruct it from the session before you ask a single question: which
repeatable process actually ran, what it started from, the steps in the order
they happened, the evidence that closed each one, the tools involved, and
every point where the user redirected you. Pay most attention to those
redirections - they mark where your default and the process the user wanted
came apart.

## 2. Interview

Questions go through the `ask` tool, not through prose. A free-form answer is
always available to the user, which makes an "I'll provide edits" option pure
padding - spend the slots on real alternatives. How many slots you get is
`ask.maxOptions`.

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

- One file per directory, and its name is not negotiable: `SKILL.md`.
- An empty body means the skill is dropped on load. Leave `name` out and the
  directory name stands in for it.
- Every `$ARGUMENTS` in the text is filled in. Arguments given to a skill with
  no placeholder anywhere are tacked on as a final paragraph rather than
  discarded - still, if a skill expects arguments it should mark where they
  go.
- A project skill shadows a user skill sharing its name. Plugin skills only
  ever answer to `/plugin:name`, never to the bare one.
- Your reader is a model halfway through a turn. Put the constraint first and
  follow it with why it holds; an unexplained rule is one the reader talks
  itself out of.

## 4. Confirm and Save

Show the complete SKILL.md in your reply before writing it, ask for approval
with the `ask` tool, then write it. Afterward, tell the user where it landed,
that it is now `/<name>`, and that it is also model-invocable through the
`skill` tool.
