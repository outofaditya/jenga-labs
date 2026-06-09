#!/usr/bin/env python
"""
Append a structured row per (atom, model, method, seq_len) to logs/results/atom_results.jsonl
by parsing the existing memory or time log files, then regenerate logs/results/atom_results.md.

Usage:
    python scripts/aggregate_results.py --atom S4 \
        --row '{"model":"opt-350m","seq_len":8192,"method":"baseline","log":"logs/end2end/memory/opt-350m-8192-a800-baseline.log","metric":"peak_memory_mb"}' \
        --row '{"model":"opt-350m","seq_len":8192,"method":"jenga","log":"logs/end2end/memory/opt-350m-8192-a800.log","metric":"peak_memory_mb"}'

The script is intentionally dumb and additive. Each invocation appends rows; the
markdown table is fully regenerated from the JSONL each time so it never drifts.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


PEAK_MEM_RE = re.compile(r"peak memory: ([\d.]+)")
TOTAL_TIME_RE = re.compile(r"total time:\s*([\d.]+)")


def parse_peak_memory_mb(log_path: str) -> float:
    peak = 0.0
    with open(log_path) as f:
        for line in f:
            m = PEAK_MEM_RE.search(line)
            if m:
                peak = max(peak, float(m.group(1)))
    return peak


def parse_step_time_ms(log_path: str) -> float:
    """Median per step total time in ms from `total time: <ms>` lines, excluding the first (warmup)."""
    vals = []
    with open(log_path) as f:
        for line in f:
            m = TOTAL_TIME_RE.search(line)
            if m:
                vals.append(float(m.group(1)))
    if len(vals) <= 1:
        return float("nan")
    steady = sorted(vals[1:])
    return steady[len(steady) // 2]


METRIC_PARSERS = {
    "peak_memory_mb": parse_peak_memory_mb,
    "step_time_ms": parse_step_time_ms,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atom", required=True)
    parser.add_argument("--row", action="append", required=True,
                        help="JSON dict describing one row; repeatable")
    parser.add_argument("--out", default="logs/results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "atom_results.jsonl"
    md = out_dir / "atom_results.md"

    for raw in args.row:
        row = json.loads(raw)
        metric = row.pop("metric")
        log_path = row.pop("log")
        parser_fn = METRIC_PARSERS[metric]
        value = parser_fn(log_path)
        row["atom"] = args.atom
        row[metric] = value
        row["log"] = log_path
        with jsonl.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print("appended:", json.dumps(row, sort_keys=True))

    rows = []
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    cols = sorted({k for r in rows for k in r.keys()} - {"log"})
    cols = ["atom"] + [c for c in cols if c != "atom"]
    lines = ["# Atom results", "", "| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    md.write_text("\n".join(lines) + "\n")
    print("wrote", md, "with", len(rows), "rows")


if __name__ == "__main__":
    sys.exit(main())
