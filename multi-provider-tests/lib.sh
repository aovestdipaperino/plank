#!/usr/bin/env bash
# Shared setup for the multi-provider tests. Sourced by each run.sh.
#
# One thing to set: REGOLO_API_KEY. Everything else is derived, so these tests
# are reusable as-is by anyone with a regolo.ai key.

# regolo.ai is OpenAI-compatible, so plank speaks to it with `--provider openai`
# and a base URL; plank appends /chat/completions to it.
: "${REGOLO_BASE_URL:=https://api.regolo.ai/v1}"
: "${REGOLO_MODEL:=glm5.2}"

die() { echo "error: $*" >&2; exit 1; }

require_key() {
    [[ -n "${REGOLO_API_KEY:-}" ]] || die "REGOLO_API_KEY is not set

  export REGOLO_API_KEY=...   # get one at https://regolo.ai

Nothing else needs setting: the endpoint (${REGOLO_BASE_URL}) and the model
(${REGOLO_MODEL}) default to regolo.ai's, and both are overridable with
REGOLO_BASE_URL / REGOLO_MODEL if you want a different model."
}

# Locates the plank binary: an explicit PLANK wins, then a sibling checkout's
# release/debug build, then whatever is on PATH.
find_plank() {
    if [[ -n "${PLANK:-}" ]]; then
        [[ -x "$PLANK" ]] || die "PLANK=$PLANK is not executable"
        echo "$PLANK"; return
    fi
    # This directory lives inside the repo, so the checkout root is one level up
    # from lib.sh — no assumption about what the checkout is named.
    local root="${PLANK_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    for candidate in "$root/target/release/plank" "$root/target/debug/plank"; do
        [[ -x "$candidate" ]] && { echo "$candidate"; return; }
    done
    command -v plank >/dev/null && { command -v plank; return; }
    die "no plank binary found

  build one:   (cd $root && cargo build --release)
  or point at one:  PLANK=/path/to/plank $0"
}
