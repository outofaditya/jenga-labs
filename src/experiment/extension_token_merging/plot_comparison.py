"""Token merging comparison figures.

Two figures are produced.

  exp-extension-token-merging-perdoc.pdf
    Per held out document forward loss as a beady line chart, one line
    per state, in the visual idiom of the algorithm ablation figures
    (src/experiment/ablation/algorithm/plot_*_attn.py). Horizontal
    legend laid out on top of the axes outside the figure box.

  exp-extension-token-merging-comparison.pdf
    Normalized Loss, PPL, and Peak Memory across the three states.
    Grouped bar chart in the idiom of the end to end memory comparison
    figure (src/experiment/end2end/memory/plot_comparison_8k.py).
    Horizontal legend laid out on top of the axes outside the figure
    box.

All axis and legend labels are Title Case. Every save uses
bbox_inches='tight' so no label is clipped.
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


PALETTE = ['#255475', '#5D7F84', '#DCBCAC', '#D6838D', '#F3AE75', '#F8F1E4']

STATE_KEYS = ["orig_hd", "orig_merge", "retrain_merge"]
STATE_LABELS = ["Hard Drop", "Token Merge", "Trained Merge"]
PER_DOC_LOSS = {
    "orig_hd": [1.6641, 2.4219, 3.4219, 1.9141],
    "orig_merge": [4.9375, 4.0625, 4.1875, 4.8438],
    "retrain_merge": [1.3125, 1.7266, 1.7969, 1.5469],
}


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def collect_means(csv_path):
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
    return out


def horizontal_top_legend(fig, ax, ncols):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center",
               bbox_to_anchor=(0.5, 1.02),
               ncol=ncols,
               frameon=False,
               fontsize=11)


def plot_perdoc(out_path):
    """Beady line chart - the loss landscape across held out documents."""
    n_docs = len(PER_DOC_LOSS["orig_hd"])
    x = np.arange(1, n_docs + 1)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    for i, key in enumerate(STATE_KEYS):
        ax.plot(x, PER_DOC_LOSS[key],
                color="black",
                marker="o",
                markersize=6,
                markerfacecolor=PALETTE[i],
                markeredgewidth=0.7,
                linewidth=1.2,
                zorder=100,
                label=STATE_LABELS[i])

    ax.set_xticks(x)
    ax.set_xticklabels([f"Doc {i}" for i in x], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("Held Out Document", fontsize=13)
    ax.set_ylabel("Forward Loss", fontsize=13)
    ax.set_ylim(0, max(max(v) for v in PER_DOC_LOSS.values()) * 1.15)

    horizontal_top_legend(fig, ax, ncols=3)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def plot_comparison(means, out_path):
    """Grouped bar chart - normalized metric comparison across states."""
    metric_keys = ["mean_loss", "ppl_approx", "peak_memory_mb"]
    metric_labels = ["Loss", "PPL", "Peak Memory"]
    raw = {m: [float(means[s][m]) for s in STATE_KEYS] for m in metric_keys}
    norm = {m: [v / max(raw[m]) for v in raw[m]] for m in metric_keys}

    n_states = len(STATE_KEYS)
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
    ax.set_xticklabels(STATE_LABELS, fontsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Normalized To Max", fontsize=13)
    ax.set_ylim(0, 1.15)

    horizontal_top_legend(fig, ax, ncols=3)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="logs/extensions/token_merging/comparison_heldout.csv")
    parser.add_argument("--out_dir", default="output_figures/extensions/token_merging")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    means = collect_means(args.csv)

    plot_perdoc(os.path.join(args.out_dir, "exp-extension-token-merging-perdoc.pdf"))
    plot_comparison(means, os.path.join(args.out_dir, "exp-extension-token-merging-comparison.pdf"))


if __name__ == "__main__":
    main()
