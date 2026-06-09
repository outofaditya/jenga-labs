"""Token merging four way comparison as grouped bar charts.

The 2x2 experimental design is adapter (Original vs Retrained) crossed
with mode (hard drop vs token merging). Two panels are produced, one
for mean forward loss and one for approximate perplexity, with adapter
on the x axis and mode as the bar color within each group.
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt


PALETTE = ['#5D7F84', '#D6838D']  # hard drop, merging
GROUP_LABELS = ["Original adapter", "Retrained with merging"]
MODE_LABELS = ["Hard drop", "Token merging"]


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def collect(heldout_csv, retrained_csv):
    heldout = read_rows(heldout_csv)
    retrained = read_rows(retrained_csv)
    out = {}
    for r in heldout:
        lab = r.get("label", "")
        if "hard drop" in lab.lower():
            out["orig_hd"] = r
        elif "Original adapter, token merging" in lab:
            out["orig_merge"] = r
    for r in retrained:
        if r["mode"] == "baseline":
            out["retrain_hd"] = r
        elif r["mode"] == "merged":
            out["retrain_merge"] = r
    return out


def grouped_bar(ax, hd, merge, ylabel, value_fmt):
    bar_width = 0.32
    x = [0, 1]
    ax.bar([xi - bar_width / 2 for xi in x], hd, bar_width,
           label=MODE_LABELS[0], color=PALETTE[0], edgecolor="black", zorder=3)
    ax.bar([xi + bar_width / 2 for xi in x], merge, bar_width,
           label=MODE_LABELS[1], color=PALETTE[1], edgecolor="black", zorder=3)
    for xi, hv, mv in zip(x, hd, merge):
        ax.text(xi - bar_width / 2, hv, value_fmt.format(hv),
                ha="center", va="bottom", fontsize=10)
        ax.text(xi + bar_width / 2, mv, value_fmt.format(mv),
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout_csv", default="logs/extensions/token_merging/comparison_heldout.csv")
    parser.add_argument("--retrained_csv", default="logs/extensions/token_merging/retrained_heldout.csv")
    parser.add_argument("--out_loss", default="output_figures/extensions/token_merging/loss.pdf")
    parser.add_argument("--out_ppl", default="output_figures/extensions/token_merging/ppl.pdf")
    args = parser.parse_args()

    by_role = collect(args.heldout_csv, args.retrained_csv)
    hd_loss = [float(by_role["orig_hd"]["mean_loss"]),
               float(by_role["retrain_hd"]["mean_loss"])]
    merge_loss = [float(by_role["orig_merge"]["mean_loss"]),
                  float(by_role["retrain_merge"]["mean_loss"])]
    hd_ppl = [float(by_role["orig_hd"]["ppl_approx"]),
              float(by_role["retrain_hd"]["ppl_approx"])]
    merge_ppl = [float(by_role["orig_merge"]["ppl_approx"]),
                 float(by_role["retrain_merge"]["ppl_approx"])]

    for path, hd, merge, ylabel, fmt, ymax in [
        (args.out_loss, hd_loss, merge_loss, "Mean forward loss",
         "{:.3f}", max(max(hd_loss), max(merge_loss)) * 1.15),
        (args.out_ppl, hd_ppl, merge_ppl, "PPL = exp(loss)",
         "{:.2f}", max(max(hd_ppl), max(merge_ppl)) * 1.15),
    ]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 3.4))
        grouped_bar(ax, hd, merge, ylabel, fmt)
        ax.set_ylim(0, ymax)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
