#!/usr/bin/env bash
# REMOTE main agent + REMOTE sub-agent, both on regolo.ai, one key.
#
# The main engine is glm5.2; .plank/agents/remote-coder.md sends its sidechain to
# qwen3-coder-next on the same endpoint with the same credential. No local model
# is involved, so this one starts instantly and runs anywhere.
#
# Set REGOLO_API_KEY and run — nothing else.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=../lib.sh
source ../lib.sh

require_key
PLANK_BIN="$(find_plank)"

# The definition names REGOLO_API_KEY itself (api-key-env), so the same key
# reaches the sub-agent through the environment and never through a flag.
export REGOLO_API_KEY

echo "plank: $PLANK_BIN"
echo "main agent: $REGOLO_MODEL via $REGOLO_BASE_URL"
echo "sub-agent 'remote-coder': $REGOLO_SUB_MODEL via $REGOLO_BASE_URL (same key)"
echo

exec "$PLANK_BIN" \
    --provider openai \
    --model "$REGOLO_MODEL" \
    --base-url "$REGOLO_BASE_URL" \
    --api-key "$REGOLO_API_KEY" \
    "$@"
