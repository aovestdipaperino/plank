#!/usr/bin/env bash
# Is the regolo.ai side actually working? One chat completion, straight over
# curl, using the same REGOLO_API_KEY / REGOLO_MODEL / REGOLO_BASE_URL the plank
# tests use — so a failure here tells you the credentials or the endpoint are
# wrong before you go blaming plank's provider path.
#
#   export REGOLO_API_KEY=...
#   ./test-regolo.sh                 # completion against the default model
#   ./test-regolo.sh --models        # list the models the key can reach
#   REGOLO_MODEL=other ./test-regolo.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=lib.sh
source ./lib.sh

require_key
command -v curl >/dev/null || die "curl is required"

# `--models` answers "what can this key reach?", which is the first thing you
# want when a model name is rejected.
if [[ "${1:-}" == "--models" ]]; then
    echo "GET $REGOLO_BASE_URL/models"
    body=$(curl -sS -w '\n%{http_code}' \
        -H "Authorization: Bearer $REGOLO_API_KEY" \
        "$REGOLO_BASE_URL/models")
    code="${body##*$'\n'}"
    payload="${body%$'\n'*}"
    echo "HTTP $code"
    if command -v python3 >/dev/null; then
        # Read the whole body once: a non-200 from regolo.ai is HTML, not JSON,
        # and it has to be shown rather than swallowed by a failed parse.
        printf '%s' "$payload" | python3 -c '
import json, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except ValueError:
    print(raw.strip())
    raise SystemExit(0)
ids = [m.get("id", "?") for m in d.get("data", [])]
print("\n".join(sorted(ids)) if ids else json.dumps(d, indent=2)[:2000])
'
    else
        printf '%s\n' "$payload"
    fi
    [[ "$code" == 200 ]] || { echo "hint: a non-200 here usually means the key is wrong." >&2; exit 1; }
    exit 0
fi

PROMPT="${*:-Reply with exactly: plank-regolo-ok}"

echo "POST $REGOLO_BASE_URL/chat/completions"
echo "model: $REGOLO_MODEL"
echo "prompt: $PROMPT"
echo

# Built with python3 when available so the prompt is escaped properly rather
# than pasted into JSON by hand.
if command -v python3 >/dev/null; then
    REQ=$(REGOLO_MODEL="$REGOLO_MODEL" PROMPT="$PROMPT" python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["REGOLO_MODEL"],
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": 64,
    "stream": False,
}))')
else
    REQ="{\"model\":\"$REGOLO_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":64,\"stream\":false}"
fi

RESP=$(curl -sS -w '\n%{http_code}' \
    -X POST "$REGOLO_BASE_URL/chat/completions" \
    -H "Authorization: Bearer $REGOLO_API_KEY" \
    -H "Content-Type: application/json" \
    -d "$REQ")

CODE="${RESP##*$'\n'}"
BODY="${RESP%$'\n'*}"

if [[ "$CODE" != 200 ]]; then
    echo "HTTP $CODE — the request was refused:" >&2
    printf '%s\n' "$BODY" >&2
    echo >&2
    case "$CODE" in
        401 | 403) echo "hint: REGOLO_API_KEY is wrong, expired, or lacks access." >&2 ;;
        404) echo "hint: model '$REGOLO_MODEL' or the path is wrong — try ./test-regolo.sh --models" >&2 ;;
        429) echo "hint: rate limited or out of quota." >&2 ;;
        5*) echo "hint: regolo.ai returned a server error; retry." >&2 ;;
    esac
    exit 1
fi

if command -v python3 >/dev/null; then
    printf '%s' "$BODY" | python3 -c '
import json, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except ValueError:
    # A 200 that is not JSON is not something to traceback over.
    print("unexpected non-JSON body:")
    print(raw.strip()[:2000])
    raise SystemExit(1)
choice = (d.get("choices") or [{}])[0]
msg = choice.get("message") or {}
text = (msg.get("content") or "").strip()
print("reply:", text if text else "(empty)")
if reason := choice.get("finish_reason"):
    print("finish_reason:", reason)
if usage := d.get("usage"):
    print("tokens: prompt={} completion={} total={}".format(
        usage.get("prompt_tokens", "?"),
        usage.get("completion_tokens", "?"),
        usage.get("total_tokens", "?")))
# An empty reply with a 200 is the shape worth naming: it means the model
# accepted the request and declined to answer, not that the wiring is broken.
if not text:
    print("\nnote: HTTP 200 with no content — the endpoint and key work, but the",
          "model returned nothing. Check finish_reason above.", file=sys.stderr)
    raise SystemExit(2)
'
else
    printf '%s\n' "$BODY"
fi

echo
echo "regolo.ai is reachable with this key. Now try:"
echo "  cd local-remote && ./run.sh     # local main, remote sub-agent"
echo "  cd remote-local && ./run.sh     # remote main, local sub-agent"
