#!/usr/bin/env bash
# Benchmark plank across models, prompts and iterations, keeping the code each
# model produced.
#
# Usage: ./bench-matrix.sh bench.json
#   PLANK=/path/to/plank ./bench-matrix.sh bench.json
#   DRY_RUN=1 ./bench-matrix.sh bench.json     # expand the matrix, run nothing
#
# Needs bash 4.4+, jq and bc. A definition that sets a timeout also needs
# timeout(1), which macOS does not ship: brew install coreutils.
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

# A bad argument in plankArgs is 45 identical 16ms failures and an empty
# report -- which is what `--no-skills` (plank spells it `--skills off`) did to
# a whole session. --dump-config parses the arguments, prints the resolved
# settings and exits without loading a model, so checking each model's full
# vector up front turns hours of nothing into a two-second error. The dummy
# `-p` is required: `--ui chart` and `--ui quiet` reject a run without one,
# and the harness only supplies the real prompt per phase.
#
# It parses arguments; it does not resolve them. --dump-config echoes a `-m`
# path back whether or not a file is there, so that is checked separately --
# the one in the example definition is hand-written and easy to mistype.
preflight_args() {
  [ "$DRY_RUN" = 1 ] && return 0
  local model out i
  mapfile -t MODEL_IDS < <(jq -r '.models[].id' "$SPEC")
  for model in "${MODEL_IDS[@]}"; do
    local -a margs
    mapfile -t margs < <(jq -r --arg m "$model" \
      '.models[]|select(.id==$m)|.args[]? // empty' "$SPEC")
    for i in "${!margs[@]}"; do margs[$i]=${margs[$i]/#\~/$REAL_HOME}; done
    if ! out=$(HOME=$BENCH_HOME "$PLANK" "${PLANK_ARGS[@]}" "${margs[@]}" \
                 -p preflight --dump-config 2>&1 >/dev/null); then
      echo "bench-matrix: plank rejects the arguments for model '$model':" >&2
      echo "  ${PLANK_ARGS[*]} ${margs[*]}" >&2
      echo "$out" | sed 's/^/  /' >&2
      exit 2
    fi
    for i in "${!margs[@]}"; do
      case ${margs[$i]} in
        -m|--model)
          local path=${margs[$((i + 1))]:-}
          if [ -n "$path" ] && [ ! -e "$path" ]; then
            echo "bench-matrix: model '$model' points at a file that is not there:" >&2
            echo "  $path" >&2
            exit 2
          fi
          ;;
      esac
    done
  done
}

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

# plank reads everything it knows about this machine from $HOME/.plank:
# skills and their hooks, plugins, memory, the MCP config, settings.json.
# Left in, they become part of what is measured. The first run showed all
# three models spending passes on the user's brainstorming skill, and DS4
# writing its first draft to a project path lifted from the cached MCP server
# instructions. So plank runs under a scratch HOME whose .plank holds the
# model artifacts and nothing else: the ggufs and models/ are symlinked (the
# ~87 GB stays where it is), the manifests and check stamps are copied so no
# download or network check is triggered. Skills and MCP are simply absent,
# not disabled, so a plank built-in skill is still available -- that is part
# of plank, not of this machine.
#
# One caveat for timing: the system-prompt KV snapshot lives in the home, so
# the first run of each model per session pays a full prefill of it.
REAL_HOME=$HOME
BENCH_HOME=$RUNROOT/home
# Ctrl-C has to end the session, not just the run in front of it. A terminal
# SIGINT reaches the whole foreground process group, so plank takes it as
# "interrupt the generation", saves, and exits -- and the loop then started the
# NEXT of 45 runs, with the model reloaded, as if nothing had happened. Worse,
# killing this script on its own left plank orphaned with the weights still
# mapped.
#
# So the signal is caught here and the child is stopped deliberately: SIGINT
# first, because that is what makes plank save its transcript and write its
# repro, then SIGKILL once it has had its chance. A second Ctrl-C skips the
# waiting. `wait` is what makes this work at all: bash defers a trap until the
# foreground command returns, so the run is started in the background and
# waited on, which a signal can interrupt.
INTERRUPTED=0
CHILD_PID=""
on_interrupt() {
  if [ "$INTERRUPTED" = 1 ]; then
    echo "" >&2; echo "bench-matrix: second interrupt; killing now." >&2
    [ -n "$CHILD_PID" ] && kill_tree "$CHILD_PID" KILL
    exit 130
  fi
  INTERRUPTED=1
  echo "" >&2
  echo "bench-matrix: interrupted; stopping plank and ending the session." >&2
  echo "  (Ctrl-C again to skip waiting for it to save.)" >&2
  if [ -n "$CHILD_PID" ]; then
    kill_tree "$CHILD_PID" INT
    local i
    for ((i = 0; i < 300; i++)); do
      kill -0 "$CHILD_PID" 2>/dev/null || break
      sleep 0.1
    done
    kill_tree "$CHILD_PID" KILL
  fi
  echo "bench-matrix: partial results in $OUTDIR" >&2
  exit 130
}

# Signal $1's whole tree with $2. timeout(1) is usually the direct child and
# relays signals to plank, but when a prompt sets no timeout plank is the child
# itself, and on a KILL nothing relays anything -- so the descendants are
# signalled explicitly rather than trusting the middleman.
kill_tree() {
  local pid=$1 sig=$2 kid
  for kid in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$kid" "$sig"; done
  kill -"$sig" "$pid" 2>/dev/null || true
}

trap on_interrupt INT TERM

KVDIR=$BENCH_HOME/.plank/kvcache
REPRODIR=$BENCH_HOME/.plank/repro
build_bench_home() {
  local src=$REAL_HOME/.plank dst=$BENCH_HOME/.plank f
  mkdir -p "$dst/kvcache" "$dst/repro"
  # -L as well as -e: a dangling link in the real home is reproduced as a
  # dangling link, so a model that would not load there does not load here.
  for f in "$src"/*.gguf "$src"/*.ggd "$src"/models; do
    if [ -e "$f" ] || [ -L "$f" ]; then ln -s "$f" "$dst/$(basename "$f")"; fi
  done
  for f in "$src"/*.manifest "$src"/*.source "$src"/model-speeds.json \
           "$src"/manifest-check "$src"/update-check "$src"/model-check "$src"/version; do
    if [ -e "$f" ]; then cp "$f" "$dst/"; fi
  done
}
build_bench_home

# Sorted listings of the transcripts and repros plank has written so far. In a
# function, not inline: under pipefail an `ls` with nothing to match fails the
# pipeline and set -e ends the whole session. The real kvcache was never empty,
# so that only surfaced once plank ran under a fresh home.
list_transcripts() { { ls "$KVDIR"/*.kv 2>/dev/null || true; } | sort; }
list_repros() { { ls "$REPRODIR"/* 2>/dev/null || true; } | sort; }

# The .kv file that appeared since $1 was taken, or empty. The transcript is
# how tool calls are counted, and it is the only per-run artifact plank leaves
# outside the working directory.
new_transcript() {
  comm -13 "$1" <(list_transcripts) | head -1
}

# Every file under $REPRODIR that appeared since $1 was taken: the repro and
# its sub-agent sidecars. Under --debug plank writes one per run, and it is
# what plank-replay reads, so it is worth more than the transcript afterwards.
new_repros() {
  comm -13 "$1" <(list_repros)
}

# One phase: $1 rundir, $2 metadir, $3 model id, $4 prompt id, $5 iter,
# $6 phase name, $7 timeout (empty = none), $8 prompt text, rest = model args.
#
# rundir is the model's working tree and holds nothing else -- the log and the
# run.json go to metadir, a sibling. They used to land in rundir, where /init
# surveyed them and wrote an AGENTS.md about "a transient benchmark scratch
# directory" instead of about the code, and where they had to be excluded by
# name from the `files` count. An empty tree is the whole point of the run.
run_phase() {
  local rundir=$1 metadir=$2 model=$3 prompt=$4 iter=$5 phase=$6 tmo=$7 text=$8; shift 8
  local margs=("$@")
  local before; before=$(mktemp)
  list_transcripts > "$before"
  local repros_before; repros_before=$(mktemp)
  list_repros > "$repros_before"

  local -a cmd=("$PLANK" "${PLANK_ARGS[@]}" "${margs[@]}" --chdir "$rundir" -p "$text")
  if [ -n "$tmo" ]; then
    # Real runs need a real timeout(1)/gtimeout(1); a dry run only prints the
    # command it would have run, so it can proceed without one.
    # SIGINT, not the default SIGTERM: plank treats SIGINT as "interrupt the
    # generation", ends the turn, and still saves the transcript and writes
    # the repro before exiting. A SIGTERM kill left nothing behind, so every
    # timed-out run in the first session recorded 0 tool calls and could not
    # be analysed at all -- when in fact two of them had finished the file
    # seconds before the kill. -k is the backstop if plank ignores the INT.
    # timeout(1) still exits 124 on expiry, whatever plank returned after it.
    if [ "$DRY_RUN" = 1 ]; then
      cmd=("timeout" "-s" "INT" "-k" "30" "$tmo" "${cmd[@]}")
    else
      resolve_timeout_bin
      cmd=("$TIMEOUT_BIN" "-s" "INT" "-k" "30" "$tmo" "${cmd[@]}")
    fi
  fi

  echo "--- $model/$prompt iter$iter $phase"
  local start end exit_code=0
  start=$(date +%s.%N)
  if [ "$DRY_RUN" = 1 ]; then
    printf '%q ' "${cmd[@]}" > "$metadir/$phase.log"; echo >> "$metadir/$phase.log"
  else
    # Backgrounded and waited on, not run in the foreground: bash holds a
    # trap until a foreground command returns, and a run here lasts an hour.
    HOME=$BENCH_HOME "${cmd[@]}" > "$metadir/$phase.log" 2>&1 &
    CHILD_PID=$!
    wait "$CHILD_PID" || exit_code=$?
    CHILD_PID=""
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
    cp "$transcript" "$metadir/$phase.transcript.kv"
  else
    # No transcript means plank never got to save one, so the count is
    # unknown, not zero: null keeps the summariser from averaging it in.
    transcript=""; tools=null
  fi
  # The repro plank wrote for this phase, and its sidecars, kept beside the
  # log under the phase's name so a run can be replayed from the archive.
  local r; for r in $(new_repros "$repros_before"); do
    cp "$r" "$metadir/$phase.$(basename "$r")"
  done
  rm -f "$repros_before"
  # printf, not raw bc output: bc emits a leading-dot number for sub-second
  # phases (".052"), which is not a valid JSON number.
  seconds=$(printf '%.3f' "$(echo "$end - $start" | bc)")

  jq -n \
    --arg model "$model" --arg prompt "$prompt" --arg phase "$phase" \
    --arg status "$status" --arg transcript "$transcript" --arg rundir "$rundir" \
    --argjson iter "$iter" --argjson exit "$exit_code" \
    --argjson seconds "$seconds" \
    --argjson toolCalls "$tools" \
    --argjson files "$(find "$rundir" -type f | wc -l | tr -d ' ')" \
    '{model:$model,prompt:$prompt,iter:$iter,phase:$phase,status:$status,
      exit:$exit,seconds:$seconds,toolCalls:$toolCalls,files:$files,
      transcript:$transcript,rundir:$rundir}' \
    > "$metadir/$phase.run.json"

  echo "    $status  $(printf '%.1f' "$seconds")s  $tools tool calls"
}

mapfile -t PLANK_ARGS < <(jq -r '.plankArgs[]? // empty' "$SPEC")
preflight_args

# One iteration: work phase, then the init phase when the prompt asks for it.
# The init phase runs even when the work phase failed -- measuring /init
# against a half-built tree beats leaving a hole in the matrix.
run_iteration() {
  local model=$1 prompt=$2 iter=$3 label=$4; shift 4
  local rundir=$RUNROOT/$model/$prompt/$label/tree
  local metadir=$RUNROOT/$model/$prompt/$label/meta
  rm -rf -- "$RUNROOT/$model/$prompt/$label"; mkdir -p "$rundir" "$metadir"

  local text tmo init_at_end init_tmo
  text=$(jq -r --arg p "$prompt" '.prompts[]|select(.id==$p)|.text' "$SPEC")
  tmo=$(jq -r --arg p "$prompt" '.prompts[]|select(.id==$p)|.timeoutSecs // ""' "$SPEC")
  init_at_end=$(jq -r --arg p "$prompt" '.prompts[]|select(.id==$p)|.runInitAtEnd // false' "$SPEC")
  init_tmo=$(jq -r --arg p "$prompt" '.prompts[]|select(.id==$p)|.initTimeoutSecs // ""' "$SPEC")

  local -a margs
  mapfile -t margs < <(jq -r --arg m "$model" \
    '.models[]|select(.id==$m)|.args[]? // empty' "$SPEC")
  # ~ is not expanded inside JSON strings; plank is given a real path. The
  # real home, because plank runs under $BENCH_HOME, where models/ is only a
  # link back here.
  local i; for i in "${!margs[@]}"; do margs[$i]=${margs[$i]/#\~/$REAL_HOME}; done

  run_phase "$rundir" "$metadir" "$model" "$prompt" "$iter" work "$tmo" "$text" "${margs[@]}"
  if [ "$init_at_end" = true ]; then
    run_phase "$rundir" "$metadir" "$model" "$prompt" "$iter" init "$init_tmo" "/init" "${margs[@]}"
  fi
  # Returned in a variable, not on stdout: run_phase's progress lines go to
  # the terminal as they happen, and capturing this function would swallow
  # them for the length of a run that takes minutes.
  RUNDIR=$rundir
  METADIR=$metadir
}

# Copy everything the last run_iteration's phases left in the meta directory
# -- run.json, log, transcript, repro -- into $1, prefixed with $2. A phase's
# files are named for the phase alone, so without the prefix every iteration
# would overwrite the last in one directory. The whole set goes because the
# run tree in /tmp does not survive a reboot and the run.json alone cannot
# answer why a run took as long as it did.
archive_records() {
  local dest=$1 label=$2 f
  for f in "$METADIR"/*; do
    [ -e "$f" ] || continue
    cp "$f" "$dest/$label.$(basename "$f")"
  done
}

# Models outermost, so each model is loaded once per sweep rather than once
# per run.
mapfile -t MODEL_IDS < <(jq -r '.models[].id' "$SPEC")
mapfile -t PROMPT_IDS < <(jq -r '.prompts[].id' "$SPEC")
# mapfile, not $(jq ...): an unquoted command substitution word-splits, so an
# id containing a space became two tokens that each passed validate_id and each
# became a directory -- the matrix quietly ran something other than the spec.
for model in "${MODEL_IDS[@]}"; do
  validate_id "$model"
  for prompt in "${PROMPT_IDS[@]}"; do
    validate_id "$prompt"
    keep=""
    mkdir -p "$OUTDIR/$model/$prompt"
    if [ "$SNAPSHOT_PASS" = true ]; then
      run_iteration "$model" "$prompt" 0 snapshot
      keep=$RUNDIR
      # Archived like any other pass. It is untimed in the sense that its tree
      # is the one kept, not in the sense that it is free: with the example's
      # 1800s init cap it can be tens of minutes per model, and leaving it out
      # of the records made "total wall time" a summary of only part of the
      # session. It records as iter 0, which sorts before iteration 1.
      archive_records "$OUTDIR/$model/$prompt" "snapshot"
    fi
    for ((n = 1; n <= ITERATIONS; n++)); do
      run_iteration "$model" "$prompt" "$n" "iter$n"
      [ -z "$keep" ] && [ "$n" = 1 ] && keep=$RUNDIR
      archive_records "$OUTDIR/$model/$prompt" "iter$n"
    done
    # The snapshot: the tree as the model left it, code and AGENTS.md together.
    # $keep is the model's tree, which now holds nothing the harness wrote, so
    # this copies model output and only model output. When it held the logs and
    # the run.json too, the summariser's recursive glob found every record a
    # second time inside the snapshot and double-counted every run, second and
    # tool call.
    [ -n "$keep" ] && cp -R "$keep" "$OUTDIR/$model/$prompt/snapshot"
    true
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
