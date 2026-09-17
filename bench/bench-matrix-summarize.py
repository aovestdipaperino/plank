#!/usr/bin/env python3
"""Summarise the run.json files bench-matrix.sh writes.

Wall time is reported as a median rather than a mean: an agent run is bimodal
-- a fast path, and a run that spirals into extra tool calls -- so a mean is
dragged around by whichever tail happened to land in this sample. This is the
same reasoning bench-summarize.py applies to its agent pairs.
"""

import json
import pathlib
import statistics
import sys

STATUSES = ("completed", "failed", "guard-stopped", "timeout")

# Keys every record must have to be rendered. `transcript` and `rundir` are
# also written by bench-matrix.sh but nothing here reads them, so they are
# deliberately not required -- a record missing only those still renders.
REQUIRED_KEYS = (
    "model", "prompt", "iter", "phase", "status", "exit",
    "seconds", "toolCalls", "files",
)


# Expected Python type for each required key. `bool` is a subclass of
# `int` in Python and arithmetic/formatting on a bool works fine (True is
# just 1), so it is deliberately accepted wherever an int is -- not worth
# special-casing.
STR_KEYS = ("model", "prompt", "phase", "status")
INT_KEYS = ("iter", "exit", "toolCalls", "files")
NUM_KEYS = ("seconds",)


def _shape_errors(record):
    """List of human-readable problems with record, or [] if it is safe to
    pass to tally/med/detail/rollup/footer."""
    if not isinstance(record, dict):
        return [f"record is {type(record).__name__}, not an object"]
    errors = []
    missing = [k for k in REQUIRED_KEYS if k not in record]
    if missing:
        errors.append(f"missing {', '.join(missing)}")
    for k in STR_KEYS:
        if k in record and not isinstance(record[k], str):
            errors.append(f"{k} is {type(record[k]).__name__}, not a string")
    for k in INT_KEYS:
        if k in record and not isinstance(record[k], int):
            errors.append(f"{k} is {type(record[k]).__name__}, not an integer")
    for k in NUM_KEYS:
        if k in record and not isinstance(record[k], (int, float)):
            errors.append(f"{k} is {type(record[k]).__name__}, not a number")
    return errors


def load(outdir):
    """Every run record under outdir, skipping any that did not parse or
    is not a well-shaped record: not a dict, missing a required key, or a
    required key holding a type that would crash rendering."""
    runs = []
    for path in sorted(outdir.rglob("*.run.json")):
        try:
            with path.open() as fh:
                record = json.load(fh)
        except (json.JSONDecodeError, OSError):
            print(f"<!-- unreadable: {path} -->", file=sys.stderr)
            continue
        errors = _shape_errors(record)
        if errors:
            print(f"<!-- invalid record: {path}: {'; '.join(errors)} -->",
                  file=sys.stderr)
            continue
        runs.append(record)
    return runs


def tally(runs):
    t = {s: sum(1 for r in runs if r["status"] == s) for s in STATUSES}
    other = [r for r in runs if r["status"] not in STATUSES]
    if other:
        for r in other:
            print(f"<!-- unexpected status: {r['status']!r} "
                  f"({r['model']}/{r['prompt']}/{r['phase']}) -->", file=sys.stderr)
        t["other"] = len(other)
    return t


def med(runs, key):
    vals = [r[key] for r in runs]
    return statistics.median(vals) if vals else 0


def detail(runs):
    print("## per model, prompt and phase\n")
    print("| model | prompt | phase | ok | fail | guard | timeout | other "
          "| median s | median tools | seconds |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    seen = []
    for r in runs:
        key = (r["model"], r["prompt"], r["phase"])
        if key not in seen:
            seen.append(key)
    for model, prompt, phase in seen:
        group = [r for r in runs
                 if (r["model"], r["prompt"], r["phase"]) == (model, prompt, phase)]
        t = tally(group)
        each = " ".join(f"{r['seconds']:.0f}" for r in sorted(group, key=lambda r: r["iter"]))
        print(f"| {model} | {prompt} | {phase} | {t['completed']} | {t['failed']} "
              f"| {t['guard-stopped']} | {t['timeout']} | {t.get('other', 0)} "
              f"| {med(group, 'seconds'):.1f} "
              f"| {med(group, 'toolCalls'):.0f} | {each} |")
    print()


def rollup(runs):
    print("## per model\n")
    print("| model | runs | ok | fail | guard | timeout | other | total s | tool calls |")
    print("|---|---|---|---|---|---|---|---|---|")
    for model in dict.fromkeys(r["model"] for r in runs):
        group = [r for r in runs if r["model"] == model]
        t = tally(group)
        print(f"| {model} | {len(group)} | {t['completed']} | {t['failed']} "
              f"| {t['guard-stopped']} | {t['timeout']} | {t.get('other', 0)} "
              f"| {sum(r['seconds'] for r in group):.0f} "
              f"| {sum(r['toolCalls'] for r in group)} |")
    print()


def footer(runs):
    t = tally(runs)
    total = sum(r["seconds"] for r in runs)
    print("## session\n")
    print(f"- runs: {len(runs)}")
    print(f"- completed: {t['completed']}")
    print(f"- failed: {t['failed']}")
    print(f"- guard-stopped: {t['guard-stopped']}")
    print(f"- timed out: {t['timeout']}")
    if "other" in t:
        print(f"- other status: {t['other']}")
    print(f"- total wall time: {total / 3600:.2f} h ({total:.0f} s)")
    print(f"- total tool calls: {sum(r['toolCalls'] for r in runs)}")
    print()


def main():
    outdir = pathlib.Path(sys.argv[1])
    runs = load(outdir)
    if not runs:
        print("no runs found")
        return
    detail(runs)
    rollup(runs)
    footer(runs)


if __name__ == "__main__":
    main()
