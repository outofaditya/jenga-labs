#!/usr/bin/env python
"""Plot entropy_norm vs retention scatter from the I1 measurement CSV."""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="logs/extensions/adaptive_thresholds/retention.csv")
    parser.add_argument("--out", default="output_figures/extensions/adaptive_thresholds/scatter.pdf")
    args = parser.parse_args()

    by_lam = defaultdict(list)
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            by_lam[float(r["lam"])].append((float(r["entropy_norm"]), float(r["retention"])))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.figure(figsize=(6, 4))
    colors = ["#444", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    for i, (lam, pts) in enumerate(sorted(by_lam.items())):
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        plt.scatter(x, y, label=f"lam={lam}", s=12, alpha=0.6, color=colors[i % len(colors)])
    plt.xlabel("Predictor entropy (normalized)")
    plt.ylabel("Token retention ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
