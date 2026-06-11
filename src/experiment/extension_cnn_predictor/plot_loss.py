"""MLP vs CNN attention predictor MSE loss curves.

Drawn in the report's standard style: horizontal legend on the top
right, bold panel title on the top left, project palette, dashed y grid.
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "mlp": "#255475",
    "cnn": "#D6838D",
}
LABELS = {"mlp": "MLP", "cnn": "CNN"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="logs/extensions/cnn_predictor/loss.csv")
    parser.add_argument("--out", default="output_figures/improvement/cnn_predictor/cnn-predictor.pdf")
    args = parser.parse_args()

    rows = defaultdict(list)
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            key = (row["predictor_type"], int(row["seed"]))
            rows[key].append((int(row["epoch"]), float(row["train_loss"])))

    by_type = defaultdict(list)
    for (ptype, _seed), pts in rows.items():
        pts.sort()
        by_type[ptype].append([l for _, l in pts])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)

    handles = []
    for ptype in ["mlp", "cnn"]:
        if ptype not in by_type:
            continue
        runs = np.array(by_type[ptype])
        mean = runs.mean(axis=0)
        std = runs.std(axis=0) if runs.shape[0] > 1 else None
        x = np.arange(1, mean.shape[0] + 1)
        color = PALETTE[ptype]
        ax.plot(x, mean, color=color, linewidth=1.8, zorder=3)
        if std is not None:
            ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=color, zorder=2)
        handles.append(mpatches.Patch(facecolor=color, edgecolor="black", label=LABELS[ptype]))

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("MSE Loss vs Ground Truth Attention", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)

    header_y = 1.04
    ax.legend(handles=handles, loc="lower right",
              bbox_to_anchor=(1.0, header_y), ncol=2, frameon=False,
              fontsize=12, handletextpad=0.4, columnspacing=1.2,
              borderpad=0.0, borderaxespad=0.0)
    ax.text(0.0, header_y, "MLP vs CNN Predictor",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=13, fontweight="bold")

    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
