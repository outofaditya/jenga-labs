"""Parse the measure_three_way eval log into a per document CSV.

Reads the stdout capture of measure_three_way.py and emits a row for
every (state, document) pair so a downstream plotter can show the
distribution of forward losses across documents rather than just the
mean. The eval script prints

  === original adapter ===
  [baseline] doc N/M loss=... dt=...
  ...
  [merged] doc N/M loss=... dt=...
  ...
  === retrained adapter ===
  [merged] doc N/M loss=... dt=...
  ...

The section header is what disambiguates the second `[merged]` block
(retrained) from the first one (original).
"""

import argparse
import csv
import re
from pathlib import Path


DOC_LINE = re.compile(
    r"^\[(?P<mode>baseline|merged)\] doc (?P<idx>\d+)/(?P<total>\d+) loss=(?P<loss>[0-9.]+) dt=(?P<dt>[0-9.]+)s"
)
SECTION = re.compile(r"^=== (?P<name>[^=]+?) ===")


def parse(path: str):
    rows = []
    current_section = "original"
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = SECTION.match(line)
            if m:
                current_section = (
                    "retrained"
                    if "retrained" in m.group("name").lower()
                    else "original"
                )
                continue
            m = DOC_LINE.match(line)
            if m:
                state = None
                if current_section == "original" and m.group("mode") == "baseline":
                    state = "orig_hd"
                elif current_section == "original" and m.group("mode") == "merged":
                    state = "orig_merge"
                elif current_section == "retrained" and m.group("mode") == "merged":
                    state = "retrain_merge"
                if state is None:
                    continue
                rows.append(
                    {
                        "state": state,
                        "doc": int(m.group("idx")),
                        "loss": float(m.group("loss")),
                        "dt": float(m.group("dt")),
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/extensions/token_merging/eval500.log")
    parser.add_argument(
        "--out", default="logs/extensions/token_merging/comparison_500_perdoc.csv"
    )
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
