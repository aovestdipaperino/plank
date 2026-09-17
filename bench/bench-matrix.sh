#!/usr/bin/env bash
# Benchmark plank across models, prompts and iterations, keeping the code each
# model produced.
#
# Usage: ./bench-matrix.sh bench.json
#   PLANK=/path/to/plank ./bench-matrix.sh bench.json
#   DRY_RUN=1 ./bench-matrix.sh bench.json     # expand the matrix, run nothing
#
# Unlike scripts/bench-dspark.sh this does not use hyperfine: hyperfine owns
# loop, and every iteration here has to be captured before the next one wipes
# the tree. See docs/superpowers/specs/2026-09-17-bench-matrix-design.md.
set -euo pipefail

# mapfile is bash 4+, and macOS ships 3.2 as /bin/bash. Checked here rather
# than left to fail as `mapfile: command not found` forty minutes into a run.
# 4.4 specifically: this script expands possibly-empty arrays (e.g. a model's
# `args` being `[]`) as "${arr[@]}" under set -u, which is only safe from 4.4
# onward -- 4.0-4.3 throw "unbound variable" on an empty array there.
if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ] || { [ "${BASH_VERSINFO[0]:-0}" -eq 4 ] && [ "${BASH_VERSINFO[1]:-0}" -lt 4 ]; }; then
  echo "bench-matrix: needs bash 4.4 or newer, found ${BASH_VERSION:-unknown}." >&2
  echo "  macOS ships bash 3.2; install a current one with: brew install bash" >&2
  exit 2
fi

SPEC=${1:?usage: bench-matrix.sh <benchmark.json>}
PLANK=${PLANK:-plank}
DRY_RUN=${DRY_RUN:-0}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

command -v jq >/dev/null || { echo "bench-matrix: jq is required" >&2; exit 2; }
command -v bc >/dev/null || { echo "bench-matrix: bc is required" >&2; exit 2; }
[ -r "$SPEC" ] || { echo "bench-matrix: cannot read '$SPEC'" >&2; exit 2; }

# Resolved lazily, only if some prompt actually sets a timeout: stock macOS
# ships neither timeout(1) nor gtimeout(1), and a benchmark definition with no
# timeouts at all must still run on such a machine.
TIMEOUT_BIN=""
resolve_timeout_bin() {
  [ -n "$TIMEOUT_BIN" ] && return 0
  if command -v timeout >/dev/null; then
    TIMEOUT_BIN=$(command -v timeout)
  elif command -v gtimeout >/dev/null; then
    TIMEOUT_BIN=$(command -v gtimeout)
  else
    echo "bench-matrix: this benchmark sets a timeout but neither 'timeout' nor 'gtimeout' is on PATH." >&2
    echo "  install coreutils to get gtimeout: brew install coreutils" >&2
    exit 2
  fi
}

# Ids from the JSON become path components (rundir, OUTDIR entries) that are
# later rm -rf'd; keep them to a safe character set before any directory is
# built from one.
validate_id() {
  case $1 in
    *[!A-Za-z0-9._-]*|'')
      echo "bench-matrix: invalid id '$1' (allowed: letters, digits, '.', '_', '-')" >&2
      exit 2
      ;;
  esac
}

# Resolve the binary up front. A benchmark whose provenance is "some plank" is
# worthless a week later, and this also fails before an hour of inference.
if [ "$DRY_RUN" = 1 ]; then
  PLANK_VERSION="(dry run)"
else
  PLANK_VERSION=$("$PLANK" --version)
fi
echo "benchmarking: $PLANK_VERSION"

NAME=$(jq -r '.name // "bench"' "$SPEC")
ITERATIONS=$(jq -r '.iterations // 3' "$SPEC")
SNAPSHOT_PASS=$(jq -r '.snapshotPass // false' "$SPEC")
STAMP=$(date +%Y%m%d-%H%M%S)

# Runs happen in an otherwise empty /tmp tree, so every file left behind is
# model output. Results are archived under local/, which is gitignored, so
# runs accumulate without ever entering a commit.
RUNROOT=/tmp/plank-bench-$STAMP
OUTDIR=${OUTDIR:-$HERE/../local/bench-$NAME-$STAMP}
mkdir -p "$RUNROOT" "$OUTDIR"

KVDIR=$HOME/.plank/kvcache

# The .kv file that appeared since $1 was taken, or empty. The transcript is
# how tool calls are counted, and it is the only per-run artifact plank leaves
# outside the working directory.
new_transcript() {
  local before=$1
  comm -13 "$before" <(ls "$KVDIR"/*.kv 2>/dev/null | sort) | head -1
}

# One phase: $1 rundir, $2 model id, $3 prompt id, $4 iter, $5 phase name,
# $6 timeout (empty = none), $7 prompt text, rest = model args.
run_phase() {
  local rundir=$1 model=$2 prompt=$3 iter=$4 phase=$5 tmo=$6 text=$7; shift 7
  local margs=("$@")
  local before; before=$(mktemp)
  ls "$KVDIR"/*.kv 2>/dev/null | sort > "$before"

  local -a cmd=("$PLANK" "${PLANK_ARGS[@]}" "${margs[@]}" --chdir "$rundir" -p "$text")
  if [ -n "$tmo" ]; then
    # Real runs need a real timeout(1)/gtimeout(1); a dry run only prints the
    # command it would have run, so it can proceed without one.
    if [ "$DRY_RUN" = 1 ]; then
      cmd=("timeout" "$tmo" "${cmd[@]}")
    else
      resolve_timeout_bin
      cmd=("$TIMEOUT_BIN" "$tmo" "${cmd[@]}")
    fi
  fi

  echo "--- $model/$prompt iter$iter $phase"
  local start end exit_code=0
  start=$(date +%s.%N)
  if [ "$DRY_RUN" = 1 ]; then
    printf '%q ' "${cmd[@]}" > "$rundir/$phase.log"; echo >> "$rundir/$phase.log"
  else
    "${cmd[@]}" > "$rundir/$phase.log" 2>&1 || exit_code=$?
  fi
  end=$(date +%s.%N)

  local status
  case $exit_code in
    0)   status=completed ;;
    3)   status=guard-stopped ;;   # ui::GUARD_STOP_EXIT
    124) status=timeout ;;         # timeout(1) killed it
    *)   status=failed ;;
  esac

  local transcript tools seconds
  transcript=$(new_transcript "$before"); rm -f "$before"
  if [ -n "$transcript" ] && [ -r "$transcript" ]; then
    # The same marker session::Message::is_tool_user keys on, so this survives
    # a front-end change.
    tools=$(grep -c '<tool_result>' "$transcript" || true)
  else
    transcript=""; tools=0
  fi
  # printf, not raw bc output: bc emits a leading-dot number for sub-second
  # phases (".052"), which is not a valid JSON number.
  seconds=$(printf '%.3f' "$(echo "$end - $start" | bc)")

  jq -n \
    --arg model "$model" --arg prompt "$prompt" --arg phase "$phase" \
    --arg status "$status" --arg transcript "$transcript" --arg rundir "$rundir" \
    --argjson iter "$iter" --argjson exit "$exit_code" \
    --argjson seconds "$seconds" \
    --argjson toolCalls "${tools:-0}" \
    --argjson files "$(find "$rundir" -type f -not -name '*.log' -not -name '*.run.json' | wc -l | tr -d ' ')" \
    '{model:$model,prompt:$prompt,iter:$iter,phase:$phase,status:$status,
      exit:$exit,seconds:$seconds,toolCalls:$toolCalls,files:$files,
      transcript:$transcript,rundir:$rundir}' \
    > "$rundir/$phase.run.json"

  echo "    $status  $(printf '%.1f' "$seconds")s  ${tools:-0} tool calls"
}

mapfile -t PLANK_ARGS < <(jq -r '.plankArgs[]? // empty' "$SPEC")

# One iteration: work phase, then the init phase when the prompt asks for it.
# The init phase runs even when the work phase failed -- measuring /init
# against a half-built tree beats leaving a hole in the matrix.
run_iteration() {
  local model=$1 prompt=$2 iter=$3 label=$4; shift 4
  local rundir=$RUNROOT/$model/$prompt/$label
  rm -rf -- "$rundir"; mkdir -p "$rundir"

  local text tmo init_at_end init_tmo
  text=$(jq -r --arg p "$prompt" '.prompts[]|select(.id==$p)|.text' "$SPEC")
  tmo=$(jq -r --arg p "$prompt" '.prompts[]|select(.id==$p)|.timeoutSecs // ""' "$SPEC")
  init_at_end=$(jq -r --arg p "$prompt" '.prompts[]|select(.id==$p)|.runInitAtEnd // false' "$SPEC")
  init_tmo=$(jq -r --arg p "$prompt" '.prompts[]|select(.id==$p)|.initTimeoutSecs // ""' "$SPEC")

  mapfile -t margs < <(jq -r --arg m "$model" \
    '.models[]|select(.id==$m)|.args[]? // empty' "$SPEC")
  # ~ is not expanded inside JSON strings; plank is given a real path.
  local i; for i in "${!margs[@]}"; do margs[$i]=${margs[$i]/#\~/$HOME}; done

  run_phase "$rundir" "$model" "$prompt" "$iter" work "$tmo" "$text" "${margs[@]}"
  if [ "$init_at_end" = true ]; then
    run_phase "$rundir" "$model" "$prompt" "$iter" init "$init_tmo" "/init" "${margs[@]}"
  fi
  # Returned in a variable, not on stdout: run_phase's progress lines go to
  # the terminal as they happen, and capturing this function would swallow
  # them for the length of a run that takes minutes.
  RUNDIR=$rundir
}

# Models outermost, so each model is loaded once per sweep rather than once
# per run.
for model in $(jq -r '.models[].id' "$SPEC"); do
  validate_id "$model"
  for prompt in $(jq -r '.prompts[].id' "$SPEC"); do
    validate_id "$prompt"
    keep=""
    mkdir -p "$OUTDIR/$model/$prompt"
    if [ "$SNAPSHOT_PASS" = true ]; then
      run_iteration "$model" "$prompt" 0 snapshot
      keep=$RUNDIR
    fi
    for ((n = 1; n <= ITERATIONS; n++)); do
      run_iteration "$model" "$prompt" "$n" "iter$n"
      [ -z "$keep" ] && [ "$n" = 1 ] && keep=$RUNDIR
      # A phase's run.json is named for its phase, so iterations would collide
      # in one directory. The iteration prefix is what keeps them apart.
      for f in "$RUNDIR"/*.run.json; do
        [ -e "$f" ] || continue
        cp "$f" "$OUTDIR/$model/$prompt/iter$n.$(basename "$f")"
      done
    done
    # The snapshot: the tree as the model left it, code and AGENTS.md together.
    [ -n "$keep" ] && cp -R "$keep" "$OUTDIR/$model/$prompt/snapshot"
  done
done

ARCHIVE="$OUTDIR/report.md"
{
  echo "# bench-matrix $NAME $STAMP"
  echo
  echo "- plank: \`$PLANK_VERSION\`"
  echo "- definition: \`$SPEC\`"
  echo "- iterations: $ITERATIONS"
  echo "- run root: \`$RUNROOT\`"
  echo
} > "$ARCHIVE"

python3 "$HERE/bench-matrix-summarize.py" "$OUTDIR" | tee -a "$ARCHIVE"

echo
echo "results: $OUTDIR"
