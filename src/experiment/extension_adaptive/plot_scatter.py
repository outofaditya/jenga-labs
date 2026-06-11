"""Adaptive threshold mechanism check.

Each point is a (layer, batch) pair under one value of lambda. Retention
swings with predictor entropy, demonstrating the runtime hook responds
to input statistics. Drawn in the report's standard style: horizontal
legend top right, bold panel title top left, project palette.
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


PALETTE = ["#255475", "#5D7F84", "#DCBCAC", "#D6838D"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="logs/extensions/adaptive_thresholds/retention.csv")
    parser.add_argument("--out", default="output_figures/improvement/adaptive_thresholds/adaptive-threshold.pdf")
    args = parser.parse_args()

    by_lam = defaultdict(list)
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            by_lam[float(r["lam"])].append((float(r["entropy_norm"]), float(r["retention"])))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.grid(axis="both", linestyle="--", alpha=0.6, zorder=0)

    handles = []
    for i, (lam, pts) in enumerate(sorted(by_lam.items())):
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        color = PALETTE[i % len(PALETTE)]
        ax.scatter(x, y, s=16, alpha=0.7, color=color,
                   edgecolors="black", linewidths=0.3, zorder=3)
        handles.append(mpatches.Patch(facecolor=color, edgecolor="black",
                                      label=f"λ = {lam:.2f}"))

    ax.set_xlabel("Predictor Entropy (Normalized)", fontsize=13)
    ax.set_ylabel("Token Retention Ratio", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)

    header_y = 1.04
    ax.legend(handles=handles, loc="lower right",
              bbox_to_anchor=(1.0, header_y),
              ncol=len(handles), frameon=False,
              fontsize=12, handletextpad=0.4, columnspacing=1.0,
              borderpad=0.0, borderaxespad=0.0)
    ax.text(0.0, header_y, "Adaptive Threshold Response",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=13, fontweight="bold")

    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
