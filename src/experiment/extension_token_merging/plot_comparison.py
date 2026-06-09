"""Render the token merging comparison bar chart from comparison.csv.

Reads up to three rows (baseline hard drop, original adapter merged,
retrained merged) and emits a 2x2 figure: mean_loss, ppl_approx,
peak_memory_mb, mean_forward_s.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(csv_path: str = "logs/extensions/token_merging/comparison.csv",
         out_pdf: str = "output_figures/extensions/token_merging/bar.pdf"):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")

    labels = []
    loss = []
    ppl = []
    mem = []
    fwd = []
    for r in rows:
        labels.append(r["mode"])
        loss.append(float(r["mean_loss"]))
        ppl.append(float(r["ppl_approx"]))
        mem.append(float(r["peak_memory_mb"]) / 1024.0)
        fwd.append(float(r["mean_forward_s"]))

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    metrics = [
        ("Mean forward loss", loss),
        ("Approx PPL", ppl),
        ("Peak memory (GB)", mem),
        ("Mean forward (s)", fwd),
    ]
    for ax, (title, values) in zip(axes.flat, metrics):
        bars = ax.bar(range(len(labels)), values, color=["#888888", "#3366cc", "#cc3333"][:len(labels)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylim(min(values) * 0.97, max(values) * 1.03)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = Path(out_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
