# Versioning and release channels

Plank ships through Homebrew only (`aovestdipaperino/homebrew-tap`); it is not
published to crates.io. Releases follow a two-channel scheme driven entirely by
the **patch number**:

- **Stable** — every `vX.Y.0` release. It updates the `plank-agent` formula
  and is published as a full GitHub release.
- **Beta** — every `vX.Y.Z` with `Z >= 1`. It updates the `plank-agent-beta`
  formula and is published as a GitHub prerelease.

A series opens with its stable `.0` and accumulates beta work as patch bumps
on top of it: `v2.5.0` (stable), then `v2.5.1`, `v2.5.2`, … (betas). The next
promotion closes the series by opening `v2.6.0`/`v2.6.1`. The app knows the
rule too: a build whose patch is above 0 shows ` BETA` in its version banner
(`logo::version_label`).

`release.yml` derives the channel automatically: when a GitHub release is
published, it looks at the tag's patch component and routes the bottle to the
matching formula. There is no channel flag anywhere — the tag number *is* the
channel.

Majors are never bumped by automation. A major bump is a deliberate, manual
decision, made by editing `Cargo.toml` and tagging directly; nothing in the
release machinery produces one.

## Promoting a beta to stable

Promotion closes the current series and opens the next minor as a
stable/beta pair built from the same code. For example, promoting while
v2.5.3 is the latest beta creates:

- `v2.6.0` — a full release; `release.yml` builds it and points the stable
  `plank-agent` formula at it.
- `v2.6.1` — a prerelease with identical code (only the version differs);
  `release.yml` seeds the new `plank-agent-beta` formula with it.

Both channels therefore restart aligned; subsequent beta work ships as
`v2.6.2`, `v2.6.3`, and so on.

### Prerequisites

- The `TAP_GITHUB_TOKEN` repository secret must be a PAT with push access to
  **both** `aovestdipaperino/homebrew-tap` and this repo. The releases must be
  created with this PAT rather than the default `GITHUB_TOKEN`, because
  releases created with `GITHUB_TOKEN` do not trigger `release.yml`.

### Running it

Promotion is a manual decision — trigger the **Promote beta to stable**
workflow (`promote.yml`) from the Actions tab, or:

```sh
gh workflow run promote.yml
```

It takes no inputs; it always promotes the highest tagged series. A
`concurrency: promote` group prevents overlapping runs.

### What the workflow does

1. **Finds the highest tagged `MAJOR.MINOR` series** (numeric comparison on
   both components, so `2.10` outranks `2.9`) and computes the next minor.
2. **Bumps `Cargo.toml`/`Cargo.lock` on `main`** to `<major>.<minor+1>.0`,
   commits, and tags it.
3. **Bumps again** to `<major>.<minor+1>.1`, commits, and tags it.
4. **Pushes** `main` and both tags, then **publishes both releases** — the
   `.0` as a full release (it becomes **Latest**), the `.1` as a prerelease.
   `release.yml` fires for each: two fresh builds, one per formula. The tap
   pushes are serialized by a `homebrew-tap` concurrency group.

Unlike the old series-based scheme, promotion never relabels an existing
release or reuses beta bottles — the stable is always built fresh from its
own tag, so the binary's `--version` always matches the formula.

### After promotion

- `brew upgrade plank-agent` picks up the new stable; `plank-agent-beta`
  users move onto the new series with its `.1` release.
- Subsequent beta releases are patch tags on the new series (`.2`, `.3`, …).

### Verifying

- A `v<major>.<minor+1>.0` release shows as **Latest** and a
  `v<major>.<minor+1>.1` prerelease exists.
- `main` has two version-bump commits (`Promote … to stable v….0`,
  `Open v….1 beta channel`).
- `homebrew-tap` has two new commits once both `release.yml` runs finish:
  `Formula/plank-agent.rb` at `.0` and `Formula/plank-agent-beta.rb` at `.1`.

Note the two formulas conflict (both install a `plank` binary), so users
switch channels with `brew uninstall plank-agent && brew install
plank-agent-beta` or the reverse.

## Hotfixing a stable release

Stable is always a `.0`, so a stable fix cannot be a patch bump — a patch
above 0 *is* the beta channel. A stable hotfix is instead a fresh `.0` on the
next minor, cut from the stable tag rather than from `main`:

1. Branch from the stable tag, not `main`:
   `git switch -c hotfix/x.y vX.Y.0`.
2. Commit the fix, keeping it minimal.
3. Bump `Cargo.toml` (and `Cargo.lock`) to `<major>.<minor+1>.0`, commit, and
   tag `v<major>.<minor+1>.0`.
4. Push the branch and the tag, then create a GitHub release for the tag as a
   normal release (**not** a prerelease). Create it with the `gh` CLI or a
   PAT, not the automation `GITHUB_TOKEN`.
5. `release.yml` sees patch 0, builds, and rewrites `Formula/plank-agent.rb`.
   `plank-agent-beta` is left alone.
6. **Forward-port the fix to the beta**: cherry-pick the hotfix commit onto
   `main` and push (do not carry the version bump across), and ship it in the
   next beta patch release.

The hotfix consumes a minor, so the beta series it overtakes is closed by it:
the next promotion computes its pair from the highest series as usual (e.g.
after a `v2.7.0` hotfix while betas were on `v2.6.x`, promotion opens
`v2.8.0`/`v2.8.1`, and new beta work continues as `v2.8.2+`). If a
stable-only change is too large for this, it is not a hotfix; carry it in the
beta and let the next promotion ship it.

## What a bump means for local caches

Version numbers also drive zero-touch maintenance of `~/.plank` on the first
launch after an upgrade (`src/upgrade.rs` reads the `~/.plank/version`
marker, classifies the transition, and cleans up):

| Transition | Maintenance performed automatically |
| --- | --- |
| Patch (`x.y.Z`) | Nothing; the marker advances |
| Minor (`x.Y.0`) | The sysprompt KV checkpoint (`kvcache/sysprompt.kv`) is dropped and rebuilt on the next warm-up |
| Major (`X.0.0`), downgrade, or missing marker | The sysprompt checkpoint **and** the image cache are dropped |

Session transcripts (`kvcache/*.session`) are user data and are never
removed.

Because beta releases are always patch bumps, a sysprompt change shipped in
beta no longer triggers the minor-bump cache drop. That is safe: the
`sysprompt.kv` checkpoint carries a fingerprint of the prompt text
(`Ds4Engine::checkpoint_fingerprint`) and is rebuilt whenever the fingerprint
does not match, so a stale checkpoint is never trusted. The `upgrade.rs` drop
is an optimisation that avoids carrying dead bytes on disk, not the mechanism
that guarantees correctness.

The minor bump still happens at every promotion — both channels cross into
the new series (`.0` for stable users, `.1` for beta users) — so each
beta-to-stable transition clears the checkpoint once. A major bump — the only
transition that also drops the image cache — remains a manual decision,
appropriate when older cached state must not be trusted at all.
