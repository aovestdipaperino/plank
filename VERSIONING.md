# Versioning and release channels

Plank ships through Homebrew only (`aovestdipaperino/homebrew-tap`); it is not
published to crates.io. Releases follow a two-channel scheme driven entirely by
the major version number:

- **Beta** — the highest major version across all `v*` tags. Releases in that
  major update the `plank-beta` formula and are published as GitHub
  prereleases.
- **Stable** — every major below the highest. Releases there update the
  `plank` formula.

`release.yml` derives the channel automatically: when a GitHub release is
published, it compares the tag's major against the highest tagged major and
routes the bottles to the matching formula. There is no channel flag anywhere —
the tag number *is* the channel.

## Promoting a beta to stable

Promotion flips the current beta major to the stable channel and opens the
next major as the new, initially empty, beta. For example, promoting while
v8.3.1 is the latest beta makes v8.3.1 the stable release and creates v9.0.0
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
major. A `concurrency: promote` group prevents overlapping runs.

### What the workflow does

1. **Finds the current beta**: the highest tagged major, and the latest
   version within it (e.g. `v8.3.1`).
2. **Downloads the stable-named bottles** from that beta release and aborts if
   any expected bottle is missing.
3. **Marks the GitHub release as stable**: clears the prerelease flag and
   marks it `--latest`.
4. **Rewrites `Formula/plank.rb` in the tap** to point at the promoted tag,
   with fresh SHA-256s for the source tarball and both bottles, and pushes the
   commit (`plank <version> (promoted from beta)`).
5. **Opens the next major beta**: bumps `Cargo.toml` (and `Cargo.lock`) on
   `main` to `<major+1>.0.0`, commits, tags `v<major+1>.0.0`, and publishes it
   as a prerelease. `release.yml` then fires on that release and seeds the new
   `plank-beta` formula with freshly built bottles.

### After promotion

- `brew upgrade plank` picks up the promoted version; `plank-beta` users move
  onto the new major with its first beta release.
- Subsequent releases tagged under the new highest major go to beta;
  patch/minor tags under the promoted (now stable) major go straight to the
  stable formula — useful for hotfixing stable without touching the beta.

### Verifying

- The promoted GitHub release shows as **Latest** (not prerelease), and a
  `v<major+1>.0.0` prerelease exists.
- `homebrew-tap` has two new commits: the updated `Formula/plank.rb` and,
  once `release.yml` finishes, the seeded `Formula/plank-beta.rb`.
- `main` has the version-bump commit (`Open v<major+1>.0.0 beta channel`).

Note the two formulas conflict (both install a `plank` binary), so users
switch channels with `brew uninstall plank && brew install plank-beta` or the
reverse.

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
removed. When cutting a release, pick the component accordingly: bump at
least minor when the sysprompt text or engine snapshot format changes, and
major when older cached state must not be trusted at all.
