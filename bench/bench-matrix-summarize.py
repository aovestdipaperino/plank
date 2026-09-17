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


def load(outdir):
    """Every run record under outdir, skipping any that did not parse."""
    runs = []
    for path in sorted(outdir.rglob("*.run.json")):
        try:
            with path.open() as fh:
                runs.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            print(f"<!-- unreadable: {path} -->")
    return runs


def tally(runs):
    return {s: sum(1 for r in runs if r["status"] == s) for s in STATUSES}


def med(runs, key):
    vals = [r[key] for r in runs]
    return statistics.median(vals) if vals else 0


def detail(runs):
    print("## per model, prompt and phase\n")
    print("| model | prompt | phase | ok | fail | guard | timeout "
          "| median s | median tools | seconds |")
    print("|---|---|---|---|---|---|---|---|---|---|")
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
              f"| {t['guard-stopped']} | {t['timeout']} | {med(group, 'seconds'):.1f} "
              f"| {med(group, 'toolCalls'):.0f} | {each} |")
    print()


def rollup(runs):
    print("## per model\n")
    print("| model | runs | ok | fail | guard | timeout | total s | tool calls |")
    print("|---|---|---|---|---|---|---|---|")
    for model in dict.fromkeys(r["model"] for r in runs):
        group = [r for r in runs if r["model"] == model]
        t = tally(group)
        print(f"| {model} | {len(group)} | {t['completed']} | {t['failed']} "
              f"| {t['guard-stopped']} | {t['timeout']} "
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
