"""Token merging four way comparison bar charts.

Matches the visual style of src/experiment/end2end/memory/plot_comparison_8k.py:
the project's standard palette, compact 8x2 figures, dashed y grid, 14 pt
ticks, black edged bars.
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt


PALETTE = ['#255475', '#5D7F84', '#DCBCAC', '#D6838D']
BAR_LABELS = [
    "Original\nhard drop",
    "Original\nmerged",
    "Retrained\nhard drop",
    "Retrained\nmerged",
]


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def assemble(heldout_csv, retrained_csv):
    heldout = read_rows(heldout_csv)
    retrained = read_rows(retrained_csv)
    by_role = {}
    for r in heldout:
        lab = r.get("label", "")
        if "hard drop" in lab.lower():
            by_role["orig_hd"] = r
        elif "Original adapter, token merging" in lab:
            by_role["orig_merge"] = r
    for r in retrained:
        if r["mode"] == "baseline":
            by_role["retrain_hd"] = r
        elif r["mode"] == "merged":
            by_role["retrain_merge"] = r
    order = ["orig_hd", "orig_merge", "retrain_hd", "retrain_merge"]
    return [by_role[k] for k in order]


def render_bar(values, ylabel, out, ylim=None):
    plt.figure(figsize=(8, 2))
    x = list(range(len(values)))
    bars = plt.bar(x, values, 0.6, color=PALETTE, edgecolor="black", zorder=3)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.yticks(fontsize=14)
    plt.xticks(x, BAR_LABELS, fontsize=11)
    plt.ylabel(ylabel, fontsize=12)
    if ylim is not None:
        plt.ylim(*ylim)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3g}",
                 ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout_csv", default="logs/extensions/token_merging/comparison_heldout.csv")
    parser.add_argument("--retrained_csv", default="logs/extensions/token_merging/retrained_heldout.csv")
    parser.add_argument("--out_dir", default="output_figures/extensions/token_merging")
    args = parser.parse_args()

    rows = assemble(args.heldout_csv, args.retrained_csv)
    loss = [float(r["mean_loss"]) for r in rows]
    ppl = [float(r["ppl_approx"]) for r in rows]
    mem_gb = [float(r["peak_memory_mb"]) / 1024.0 for r in rows]
    fwd = [float(r["mean_forward_s"]) for r in rows]

    render_bar(loss, "Mean forward loss",
               os.path.join(args.out_dir, "loss.pdf"))
    render_bar(ppl, "PPL = exp(loss)",
               os.path.join(args.out_dir, "ppl.pdf"))
    render_bar(mem_gb, "Peak memory (GB)",
               os.path.join(args.out_dir, "memory.pdf"),
               ylim=(min(mem_gb) - 0.05, max(mem_gb) + 0.05))
    render_bar(fwd, "Mean forward (s)",
               os.path.join(args.out_dir, "time.pdf"))


if __name__ == "__main__":
    main()
