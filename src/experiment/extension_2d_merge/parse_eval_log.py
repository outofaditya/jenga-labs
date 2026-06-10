"""Parse the measure_2d eval log into a per document CSV.

The I5 eval emits one block per mode:
  === 2d adapter ===
  [baseline_2d] doc N/M loss=... dt=...
  ...
  [merged_2d] doc N/M loss=... dt=...
  ...
"""
import argparse
import csv
import re
from pathlib import Path


DOC_LINE = re.compile(r"^\[(?P<mode>baseline_2d|merged_2d)\] doc (?P<idx>\d+)/(?P<total>\d+) loss=(?P<loss>[0-9.]+) dt=(?P<dt>[0-9.]+)s")


def parse(path: str):
    rows = []
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = DOC_LINE.match(line)
            if m:
                rows.append({
                    "state": m.group("mode"),
                    "doc": int(m.group("idx")),
                    "loss": float(m.group("loss")),
                    "dt": float(m.group("dt")),
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/extensions/token_merging_2d/eval500.log")
    parser.add_argument("--out", default="logs/extensions/token_merging_2d/comparison_500_perdoc.csv")
    args = parser.parse_args()

    rows = parse(args.log)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["state", "doc", "loss", "dt"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")
    by_state = {}
    for r in rows:
        by_state.setdefault(r["state"], []).append(r["loss"])
    for st, vals in by_state.items():
        print(f"  {st:14s} n={len(vals)} mean={sum(vals) / max(len(vals), 1):.4f}")


if __name__ == "__main__":
    main()
