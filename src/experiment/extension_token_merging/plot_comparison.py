"""Token merging comparison figures in the codebase's beady line idiom.

Both figures use the visual pattern of src/experiment/ablation/algorithm/
plot_llama2_attn.py: black lines with colored circular markers, dashed y
grid, compact aspect, horizontal legend laid out on top of the axes
outside the figure box. All labels are Title Case and bbox_inches='tight'
is used at save time so no label is ever clipped.

  exp-extension-token-merging-perdoc.pdf
    Forward Loss versus Held Out Document, one line per state.

  exp-extension-token-merging-comparison.pdf
    Normalized Loss, PPL, and Peak Memory across the three states. Each
    metric is normalized so its maximum is 1.0 and drawn as a separate
    beaded line.
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


PALETTE = ['#255475', '#5D7F84', '#DCBCAC', '#D6838D', '#F3AE75', '#F8F1E4']

STATE_KEYS = ["orig_hd", "orig_merge", "retrain_merge"]
STATE_LABELS = ["Original Hard Drop", "Original Token Merging", "Retrained Token Merging"]
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


def beady_line(ax, x, y, color, label):
    ax.plot(x, y,
            color="black",
            marker="o",
            markersize=6,
            markerfacecolor=color,
            markeredgewidth=0.7,
            linewidth=1.2,
            zorder=100,
            label=label)


def horizontal_top_legend(fig, ax, ncols):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center",
               bbox_to_anchor=(0.5, 1.02),
               ncol=ncols,
               frameon=False,
               fontsize=11)


def plot_perdoc(out_path):
    n_docs = len(PER_DOC_LOSS["orig_hd"])
    x = np.arange(1, n_docs + 1)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    for i, key in enumerate(STATE_KEYS):
        beady_line(ax, x, PER_DOC_LOSS[key], PALETTE[i], STATE_LABELS[i])

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
    metric_keys = ["mean_loss", "ppl_approx", "peak_memory_mb"]
    metric_labels = ["Loss", "PPL", "Peak Memory"]
    raw = {m: [float(means[s][m]) for s in STATE_KEYS] for m in metric_keys}
    norm = {m: [v / max(raw[m]) for v in raw[m]] for m in metric_keys}

    x = np.arange(1, len(STATE_KEYS) + 1)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    for i, m in enumerate(metric_keys):
        beady_line(ax, x, norm[m], PALETTE[i], metric_labels[i])

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
