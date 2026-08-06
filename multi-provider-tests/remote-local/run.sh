#!/usr/bin/env bash
# REMOTE main agent (regolo.ai) + local sub-agent.
#
# The main engine is regolo.ai; .plank/agents/cheap-local.md carries
# `provider: local`, so plank loads the local ds4 model alongside it and runs
# that sidechain locally. Expect the local model load at startup — see README.md.
#
# Set REGOLO_API_KEY and run — nothing else.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=../lib.sh
source ../lib.sh

require_key
PLANK_BIN="$(find_plank)"

echo "plank: $PLANK_BIN"
echo "main agent: $REGOLO_MODEL via $REGOLO_BASE_URL"
echo "sub-agent 'cheap-local': the local ds4 model (loaded at startup)"
echo

exec "$PLANK_BIN" \
    --provider openai \
    --model "$REGOLO_MODEL" \
    --base-url "$REGOLO_BASE_URL" \
    --api-key "$REGOLO_API_KEY" \
    "$@"
