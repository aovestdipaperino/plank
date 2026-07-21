# Versioning and release channels

Plank ships through Homebrew only (`aovestdipaperino/homebrew-tap`); it is not
published to crates.io. Releases follow a two-channel scheme driven entirely by
the `MAJOR.MINOR` series:

- **Beta** — the highest `MAJOR.MINOR` series across all `v*` tags. Releases in
  that series update the `plank-beta` formula and are published as GitHub
  prereleases.
- **Stable** — every series below the highest. Releases there update the
  `plank` formula.

Within the beta series, **every release is a patch bump** — bug fixes and new
features alike. A beta that opens at `v2.4.0` accumulates `v2.4.1`, `v2.4.2`,
and so on for as long as it stays open; the minor never moves while the series
is the beta.

`release.yml` derives the channel automatically: when a GitHub release is
published, it compares the tag's `MAJOR.MINOR` against the highest tagged
series and routes the bottles to the matching formula. There is no channel flag
anywhere — the tag number *is* the channel.

Majors are never bumped by automation. A major bump is a deliberate, manual
decision, made by editing `Cargo.toml` and tagging directly; nothing in the
release machinery produces one.

## Promoting a beta to stable

Promotion flips the current beta series to the stable channel and opens the
next **minor** as the new, initially empty, beta. For example, promoting while
v2.3.7 is the latest beta makes v2.3.7 the stable release and creates v2.4.0
as the new beta.

### Prerequisites

- The beta release you intend to promote must have both bottle assets
  (`plank-<version>.arm64_sequoia.bottle.tar.gz` and
  `plank-<version>.sequoia.bottle.tar.gz`) attached — `release.yml` uploads
  beta bottles under both the `plank-beta` and `plank` names precisely so
  promotion can reuse them without rebuilding. The workflow aborts if either
  is missing.
- The `TAP_GITHUB_TOKEN` repository secret must be a PAT with push access to
  **both** `aovestdipaperino/homebrew-tap` and this repo. The new-beta release
  must be created with this PAT rather than the default `GITHUB_TOKEN`,
  because releases created with `GITHUB_TOKEN` do not trigger `release.yml`.

### Running it

Promotion is a manual decision — trigger the **Promote beta to stable**
workflow (`promote.yml`) from the Actions tab, or:

```sh
gh workflow run promote.yml
```

It takes no inputs; it always promotes the latest release of the highest
series. A `concurrency: promote` group prevents overlapping runs.

### What the workflow does

1. **Finds the current beta**: the highest tagged `MAJOR.MINOR` series, and
   the latest version within it (e.g. `v2.3.7`). The series comparison is
   numeric on both components, so `2.10` correctly outranks `2.9`.
2. **Downloads the stable-named bottles** from that beta release and aborts if
   any expected bottle is missing.
3. **Marks the GitHub release as stable**: clears the prerelease flag and
   marks it `--latest`.
4. **Rewrites `Formula/plank.rb` in the tap** to point at the promoted tag,
   with fresh SHA-256s for the source tarball and both bottles, and pushes the
   commit (`plank <version> (promoted from beta)`).
5. **Opens the next minor beta**: bumps `Cargo.toml` (and `Cargo.lock`) on
   `main` to `<major>.<minor+1>.0`, commits, tags it, and publishes it as a
   prerelease. `release.yml` then fires on that release and seeds the new
   `plank-beta` formula with freshly built bottles.

### After promotion

- `brew upgrade plank` picks up the promoted version; `plank-beta` users move
  onto the new minor with its first beta release.
- Subsequent releases tagged under the new highest series go to beta; patch
  tags under the promoted (now stable) series go straight to the stable
  formula — useful for hotfixing stable without touching the beta.

### Verifying

- The promoted GitHub release shows as **Latest** (not prerelease), and a
  `v<major>.<minor+1>.0` prerelease exists.
- `homebrew-tap` has two new commits: the updated `Formula/plank.rb` and,
  once `release.yml` finishes, the seeded `Formula/plank-beta.rb`.
- `main` has the version-bump commit (`Open v<major>.<minor+1>.0 beta channel`).

Note the two formulas conflict (both install a `plank` binary), so users
switch channels with `brew uninstall plank && brew install plank-beta` or the
reverse.

## Hotfixing a stable release

Once a series has been promoted to stable, the beta lives in a higher minor, so
you can ship a fix to stable without touching the beta. `release.yml` routes by
series automatically: a tag whose `MAJOR.MINOR` is **below** the highest tagged
series updates the `plank` (stable) formula; the highest series updates
`plank-beta`. So a patch tag under the promoted series is a stable-only
release. (The v0.9.10 release was exactly this: a hotfix cut against stable
while the next beta was already open.)

> ⚠️ A stable hotfix must bump **only the patch**. Bumping the minor would
> land the tag in the beta series — or overtake it — and the fix would be
> routed to `plank-beta` instead of `plank`. If a stable-only change is too
> large for a patch, it is not a hotfix; carry it in the beta and let the next
> promotion ship it.

### Running it

Do the work off `main` — `main` tracks the beta series, so bumping it would not
describe a stable patch.

1. Branch from the stable tag, not `main`:
   `git switch -c hotfix/x.y.z vX.Y.Z`.
2. Commit the fix, keeping it minimal.
3. Bump `Cargo.toml` (and `Cargo.lock`) to the next patch under the stable
   series (`X.Y.(Z+1)`), commit, and tag `vX.Y.(Z+1)`.
4. Push the branch and the tag, then create a GitHub release for the tag as a
   normal release (**not** a prerelease). Create it with the `gh` CLI or a PAT,
   not the automation `GITHUB_TOKEN` — releases made with `GITHUB_TOKEN` do not
   trigger `release.yml`.
5. `release.yml` fires, sees the tag's series is below the highest, builds
   bottles, and rewrites `Formula/plank.rb` for the new version. `plank-beta`
   is left alone.
6. **Forward-port the fix to the beta** so it survives the next promotion:
   cherry-pick the hotfix commit onto `main` and push. Do not carry the hotfix
   version bump across — `main` keeps its beta version.

### Verifying

- The hotfix release shows as **Latest** (stable releases outrank the beta
  prerelease), and `homebrew-tap` has one new commit updating
  `Formula/plank.rb`; `Formula/plank-beta.rb` is unchanged.
- `brew upgrade plank` picks up the hotfix; `plank-beta` users are unaffected.
- The fix commit exists on both the hotfix tag and `main`.

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

The minor bump still happens at every promotion, so each beta-to-stable
transition clears the checkpoint once. A major bump — the only transition that
also drops the image cache — remains a manual decision, appropriate when older
cached state must not be trusted at all.
