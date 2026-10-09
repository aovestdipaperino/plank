# plank-lib

plank's user-level files, in one crate that plank and `pt` (`../plank-replay`) both read and
write them with, so the two programs cannot disagree about their format.

## Steering vectors

`vectors::VectorStore` is `~/.plank/models/vectors.json`, a JSON array with one entry per
model, each listing its named directional-steering vectors as base64 of the raw
little-endian `f32` matrix the ds4 engine loads:

```json
[
  { "model": "ds4vision",
    "vectors": [ { "name": "heretic", "value": "Pq3vPLnF..." } ] }
]
```

An entry's `model` is the plank engine name; a GGUF that no engine uses is listed under its
file name, and `vectors::model_keys` gives the keys to try in that order. The store lists
directions, stores and replaces them (`put`), and decodes one into a file named by the
SHA-256 of its bytes (`materialize`), since the engine only loads vectors from files.
Merging another vectors file is two steps: `plan` classifies each incoming vector as new,
unchanged or in conflict without writing anything, and `merge` applies the plan, replacing
conflicts only where the caller says so. Fields the crate does not know are kept on every
write, and every write goes through a temporary sibling renamed into place.

## Profiles

`profiles` reads a profile folder's manifest (`.plank-plugin/plugin.json`, or the Claude Code
spelling): its name, version, `recommendedModel` and `steering` block, which
`profiles::parse_steering` checks the way the command-line flags are checked. Installing is
again a plan, which says whether the profile is new, identical to the installed copy (file by
file, ignoring the source record) or different from it, followed by an install that stages
the new copy beside the old one and swaps them by rename, restoring the old one if the swap
fails. The installed folder gets a `.plank-source` record naming where it came from, which is
how `plank --profile <source>` finds it again. Folders holding symlinks are refused.

## Sources

`source::RepoPath` is a `repo:path` reference: a local folder, a GitHub `owner/repo` or a git
URL, then a path inside it that may not climb out with `..`. Fetching uses a local folder in
place and clones anything else shallowly into a temporary folder that is removed when the
`Checkout` is dropped. Git is told never to prompt for credentials, so a missing or private
repository fails rather than waiting for a username.

## The plank home

`home::plank_dir` is `~/.plank`, or the machine-wide `/Users/.plank` when only that exists,
the same rule plank applies.

## Tests

`cargo test -p plank-lib` runs the crate's tests against scratch directories; none needs the
network or a model.
