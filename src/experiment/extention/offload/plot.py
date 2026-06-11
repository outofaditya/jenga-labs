"""Paper Figure 19 (b) Sparsity-sensitive Offload panel.

Two bars per sequence length: Naïve (the offload baseline) and Jenga
(offload with token sparsity active). Bars normalized to Naïve = 1.0 per
sequence length. Speedup Naïve / Jenga written inside each Jenga bar.
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np


PALETTE = ["#DCBCAC", "#F3AE75"]
SERIES = ["naive", "jenga"]
SERIES_LABELS = ["Naïve", "Jenga"]
SEQ_KEYS = [4096, 6144, 8192, 10240, 12288, 14336, 16384]
SEQ_LABELS = ["4K", "6K", "8K", "10K", "12K", "14K", "16K"]


def parse_logs(log_dir="logs/extension/offload"):
    times = {s: {k: 0.0 for k in SERIES} for s in SEQ_KEYS}
    for filename in os.listdir(log_dir):
        if not filename.endswith(".log"):
            continue
        path = os.path.join(log_dir, filename)
        try:
            with open(path) as f:
                content = f.read()
        except Exception:
            continue
        if "baseline" in filename:
            method = "naive"
        elif "ours" in filename:
            method = "jenga"
        else:
            continue
        parts = filename.replace(".log", "").split("_")
        seq = None
        for p in parts:
            head = p.split("-")[0]
            if head.isdigit():
                seq = int(head)
                break
        if seq not in times:
            continue
        last = None
        for m in re.finditer(r"total time:\s*([\d.]+)", content):
            last = m
        if last is None:
            continue
        times[seq][method] = float(last.group(1))
    return times


def render(out_path):
    times = parse_logs()
    raw = {s: [times[s]["naive"], times[s]["jenga"]] for s in SEQ_KEYS}
    norm = {s: ([v / raw[s][0] if raw[s][0] else 0.0 for v in raw[s]]) for s in SEQ_KEYS}

    bar_width = 0.32
    x = np.arange(len(SEQ_KEYS))

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)

    for j, _ in enumerate(SERIES):
        heights = [norm[s][j] for s in SEQ_KEYS]
        ax.bar(x + j * bar_width, heights, bar_width,
               color=PALETTE[j], edgecolor="black", zorder=3,
               label=SERIES_LABELS[j])

    for i, s in enumerate(SEQ_KEYS):
        naive, jenga = raw[s]
        if naive > 0 and jenga > 0:
            top = norm[s][1]
            ax.text(x[i] + 1 * bar_width, top + 0.012, f"{naive / jenga:.2f}x",
                    ha="center", va="bottom", fontsize=12, rotation=90, color="#222")

    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(SEQ_LABELS, fontsize=13)
    ax.set_yticks([0.75, 1.0])
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0.7, 1.25)
    ax.set_xlabel("Sequence Length", fontsize=13)
    ax.set_ylabel("Execution Time (Normalized)", fontsize=13)

    header_y = 1.04
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, header_y),
              ncol=2, frameon=False, fontsize=12, handletextpad=0.4,
              columnspacing=1.2, borderpad=0.0, borderaxespad=0.0)
    ax.text(0.0, header_y, "(b) Sparsity-sensitive Offload",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=13, fontweight="bold")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    render("output_figures/extension/offload/offload.pdf")
