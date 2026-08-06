#!/usr/bin/env bash
# local main agent + REMOTE sub-agent (regolo.ai).
#
# The main engine is the local ds4 model; .plank/agents/remote-reviewer.md sends
# its sidechain to regolo.ai. Set REGOLO_API_KEY and run — nothing else.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=../lib.sh
source ../lib.sh

require_key
PLANK_BIN="$(find_plank)"

# The definition names REGOLO_API_KEY itself (api-key-env), so the key reaches
# the sub-agent through the environment and never through a flag.
export REGOLO_API_KEY

echo "plank: $PLANK_BIN"
echo "main agent: local ds4 model"
echo "sub-agent 'remote-reviewer': $REGOLO_MODEL via $REGOLO_BASE_URL"
echo

exec "$PLANK_BIN" "$@"
