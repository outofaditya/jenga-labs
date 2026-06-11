"""Token merging comparison figures.

  loss-landscape.pdf
    Per held out document forward loss, sorted ascending within each
    state and plotted as a smooth line (no markers, predictor-loss
    idiom). The shape of each curve is the loss distribution for that
    state. Legend sits inside the chart on the upper right.

  comparison.pdf
    Normalized Loss, PPL, and Peak Memory across the trained states as
    a grouped bar chart. Legend aligned to the top right.

Both figures use the project palette, Title Case labels, and
bbox_inches='tight' so labels never clip.
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


PALETTE = ['#255475', '#5D7F84', '#DCBCAC', '#D6838D', '#F3AE75', '#F8F1E4']

STATE_KEYS = ["orig_hd", "orig_merge", "retrain_merge", "trained_2d_merge"]
STATE_LABELS = ["Hard Drop", "Token Merge", "Trained Merge", "Trained 2D Merge"]


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def collect_means(csv_path, csv_2d=None):
    rows = read_rows(csv_path)
    out = {}
    for r in rows:
        lab = r.get("label", "")
        if "hard drop" in lab.lower():
            out["orig_hd"] = r
        elif "Original adapter, token merging" in lab:
            out["orig_merge"] = r
        elif "Retrained" in lab and "merging" in lab.lower():
            out["retrain_merge"] = r
    if csv_2d and os.path.exists(csv_2d):
        for r in read_rows(csv_2d):
            if r.get("mode") == "merged_2d":
                out["trained_2d_merge"] = r
    return out


def collect_perdoc(perdoc_csv, perdoc_2d_csv=None):
    by_state = defaultdict(list)
    for r in read_rows(perdoc_csv):
        by_state[r["state"]].append(float(r["loss"]))
    if perdoc_2d_csv and os.path.exists(perdoc_2d_csv):
        for r in read_rows(perdoc_2d_csv):
            if r["state"] == "merged_2d":
                by_state["trained_2d_merge"].append(float(r["loss"]))
    return by_state


def plot_perdoc(by_state, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    handles = []
    for i, key in enumerate(STATE_KEYS):
        if key not in by_state or not by_state[key]:
            continue
        losses = sorted(by_state[key])
        x = np.linspace(1, len(losses), len(losses))
        ax.plot(x, losses, color=PALETTE[i], linewidth=1.6, zorder=3)
        handles.append(mpatches.Patch(facecolor=PALETTE[i], edgecolor="black",
                                      label=STATE_LABELS[i]))
    ax.set_xlim(0, max((len(by_state[k]) for k in STATE_KEYS if k in by_state and by_state[k]), default=1) + 1)
    ax.set_xlabel("Document Rank (Sorted Ascending Per State)", fontsize=12)
    ax.set_ylabel("Forward Loss", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(0, max((max(by_state[k]) for k in STATE_KEYS if k in by_state and by_state[k]), default=1.0) * 1.1)

    ax.legend(handles=handles, loc="upper right", frameon=False,
              fontsize=12, handletextpad=0.6, labelspacing=0.7,
              borderaxespad=1.0)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def plot_comparison(means, out_path):
    metric_keys = ["mean_loss", "ppl_approx", "peak_memory_mb"]
    metric_labels = ["Loss", "PPL", "Peak Memory"]
    present = [s for s in STATE_KEYS if s in means]
    labels_present = [STATE_LABELS[STATE_KEYS.index(s)] for s in present]
    raw = {m: [float(means[s][m]) for s in present] for m in metric_keys}
    norm = {m: [v / max(raw[m]) for v in raw[m]] for m in metric_keys}

    n_states = len(present)
    bar_width = 0.25
    x = np.arange(n_states)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    for i, m in enumerate(metric_keys):
        offset = (i - 1) * bar_width
        ax.bar(x + offset, norm[m], bar_width,
               color=PALETTE[i], edgecolor="black", zorder=3,
               label=metric_labels[i])

    ax.set_xticks(x)
    ax.set_xticklabels(labels_present, fontsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Normalized To Max", fontsize=12)
    ax.set_ylim(0, 1.18)

    header_y = 1.04
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, header_y),
              ncol=3, frameon=False, fontsize=12, handletextpad=0.4,
              columnspacing=1.2, borderpad=0.0, borderaxespad=0.0)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="logs/extensions/token_merging/comparison_500.csv")
    parser.add_argument("--perdoc_csv", default="logs/extensions/token_merging/comparison_500_perdoc.csv")
    parser.add_argument("--csv_2d", default="logs/extensions/token_merging_2d/comparison_500.csv")
    parser.add_argument("--perdoc_2d_csv", default="logs/extensions/token_merging_2d/comparison_500_perdoc.csv")
    parser.add_argument("--out_dir", default="output_figures/improvement/token_merging")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    means = collect_means(args.csv, args.csv_2d)
    by_state = collect_perdoc(args.perdoc_csv, args.perdoc_2d_csv)

    plot_perdoc(by_state,
                os.path.join(args.out_dir, "loss-landscape.pdf"))
    plot_comparison(means,
                    os.path.join(args.out_dir, "comparison.pdf"))


if __name__ == "__main__":
    main()
