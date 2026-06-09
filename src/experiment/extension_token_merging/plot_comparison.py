"""Three state token merging comparison.

States:
1. Original adapter, hard drop (baseline)
2. Original adapter, token merging (inference time only)
3. Retrained with merging, token merging (joint training)

Renders three simple bar charts, one per metric.
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt


PALETTE = ['#5D7F84', '#D6838D', '#255475']
STATE_LABELS = [
    "Original\nhard drop",
    "Original\ntoken merging",
    "Retrained\ntoken merging",
]


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def collect(csv_path):
    rows = read_rows(csv_path)
    out = {}
    for r in rows:
        lab = r.get("label", "")
        if "hard drop" in lab.lower():
            out["s1"] = r
        elif "Original adapter, token merging" in lab:
            out["s2"] = r
        elif "Retrained" in lab and "merging" in lab.lower():
            out["s3"] = r
    return [out["s1"], out["s2"], out["s3"]]


def render(values, ylabel, out_path, fmt):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = list(range(len(values)))
    ax.bar(x, values, 0.55, color=PALETTE, edgecolor="black", zorder=3)
    for xi, v in zip(x, values):
        ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_LABELS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ymax = max(values) * 1.18 if max(values) > 0 else 1
    ax.set_ylim(0, ymax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="logs/extensions/token_merging/comparison_heldout.csv")
    parser.add_argument("--out_dir", default="output_figures/extensions/token_merging")
    args = parser.parse_args()

    rows = collect(args.csv)
    loss = [float(r["mean_loss"]) for r in rows]
    ppl = [float(r["ppl_approx"]) for r in rows]
    mem_gb = [float(r["peak_memory_mb"]) / 1024.0 for r in rows]

    render(loss, "Mean forward loss",
           os.path.join(args.out_dir, "loss.pdf"), "{:.3f}")
    render(ppl, "PPL = exp(loss)",
           os.path.join(args.out_dir, "ppl.pdf"), "{:.2f}")
    render(mem_gb, "Peak GPU memory (GB)",
           os.path.join(args.out_dir, "memory.pdf"), "{:.2f}")


if __name__ == "__main__":
    main()
