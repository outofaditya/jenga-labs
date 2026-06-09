"""Token merging comparison plots in the codebase's standard styles.

Two figures are produced.

  exp-extension-token-merging-perdoc.pdf
    Mimics src/experiment/end2end/time/plot_sequence.py: grouped bars
    across the four held out documents, one colored bar per state.
    Reveals per document variance and the consistency of each state.

  exp-extension-token-merging-comparison.pdf
    Mimics src/experiment/end2end/memory/plot_comparison_8k.py: grouped
    bars with three states on the x axis and the three reported metrics
    as colored bars within each group, normalized so they fit on a
    single y axis. Reproduces the visual idiom of the end to end memory
    figure.
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


PALETTE = ['#255475', '#5D7F84', '#DCBCAC', '#D6838D', '#F3AE75', '#F8F1E4']

STATE_KEYS = ["orig_hd", "orig_merge", "retrain_merge"]
STATE_LABELS = ["Original\nhard drop", "Original\ntoken merging", "Retrained\ntoken merging"]
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


def plot_perdoc(out_path):
    n_docs = len(PER_DOC_LOSS["orig_hd"])
    x = np.arange(n_docs)
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)

    for i, key in enumerate(STATE_KEYS):
        offset = (i - 1) * bar_width
        ax.bar(x + offset, PER_DOC_LOSS[key], bar_width,
               color=PALETTE[i], edgecolor="black", zorder=3,
               label=STATE_LABELS[i].replace("\n", " "))

    ax.set_xticks(x)
    ax.set_xticklabels([f"Doc {i + 1}" for i in range(n_docs)], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("Held out document", fontsize=14)
    ax.set_ylabel("Forward loss", fontsize=14)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"wrote {out_path}")


def plot_comparison(means, out_path):
    metric_keys = ["mean_loss", "ppl_approx", "peak_memory_mb"]
    metric_labels = ["Loss", "PPL", "Peak Memory"]
    n_states = len(STATE_KEYS)
    raw = {m: [float(means[s][m]) for s in STATE_KEYS] for m in metric_keys}

    # Normalize each metric so its maximum is 1.0 (the same idiom as
    # plot_sequence.py and plot_comparison_8k.py use to put multiple
    # metrics on a common y axis).
    norm = {m: [v / max(raw[m]) for v in raw[m]] for m in metric_keys}

    x = np.arange(n_states)
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)

    for i, m in enumerate(metric_keys):
        offset = (i - 1) * bar_width
        ax.bar(x + offset, norm[m], bar_width,
               color=PALETTE[i], edgecolor="black", zorder=3,
               label=metric_labels[i])

    ax.set_xticks(x)
    ax.set_xticklabels(STATE_LABELS, fontsize=12)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("Normalized to max", fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path)
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
