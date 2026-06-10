#!/usr/bin/env python
"""Plot MLP vs CNN attention predictor MSE loss curves from the CSV."""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="logs/extensions/cnn_predictor/loss.csv")
    parser.add_argument("--out", default="output_figures/extensions/cnn_predictor/cnn-predictor.pdf")
    args = parser.parse_args()

    rows = defaultdict(list)
    with open(args.csv) as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["predictor_type"], int(row["seed"]))
            rows[key].append((int(row["epoch"]), float(row["train_loss"])))

    by_type = defaultdict(list)
    for (ptype, _seed), pts in rows.items():
        pts.sort()
        by_type[ptype].append([l for _, l in pts])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.figure(figsize=(6, 4))
    style = {"mlp": ("--", "#444"), "cnn": ("-", "#1f77b4")}
    for ptype, runs in by_type.items():
        arr = np.array(runs)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0) if arr.shape[0] > 1 else None
        x = np.arange(1, mean.shape[0] + 1)
        ls, color = style.get(ptype, ("-", None))
        plt.plot(x, mean, label=ptype.upper(), linestyle=ls, color=color)
        if std is not None:
            plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss vs ground truth attention")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
