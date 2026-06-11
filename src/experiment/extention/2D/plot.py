"""Paper Figure 19 (a) 2D Sparsity panel.

Three bars per sequence length: LoRA (the baseline), Jenga (token sparsity
alone), Jenga-2D (token sparsity composed with LongLoRA shifted attention).
All bars normalized to LoRA = 1.0 per sequence length. Speedups
LoRA / Jenga and LoRA / Jenga-2D are written inside the Jenga and
Jenga-2D bars respectively.
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np


PALETTE = ["#255475", "#5D7F84", "#DCBCAC"]
SERIES = ["lora", "jenga", "jenga_2d"]
SERIES_LABELS = ["LoRA", "Jenga", "Jenga-2D"]
SEQ_KEYS = [16384, 32768, 49152, 65536]
SEQ_LABELS = ["16K", "32K", "48K", "64K"]


def parse_logs(log_dir="logs/extension/2d"):
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
            method = "lora"
        elif "2D" in filename:
            method = "jenga_2d"
        else:
            method = "jenga"
        parts = filename.replace(".log", "").split("-")
        seq = next((int(p) for p in parts if p.isdigit()), None)
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
    raw = {s: [times[s]["lora"], times[s]["jenga"], times[s]["jenga_2d"]] for s in SEQ_KEYS}
    norm = {s: ([v / raw[s][0] if raw[s][0] else 0.0 for v in raw[s]]) for s in SEQ_KEYS}

    bar_width = 0.25
    x = np.arange(len(SEQ_KEYS))

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)

    for j, _ in enumerate(SERIES):
        heights = [norm[s][j] for s in SEQ_KEYS]
        ax.bar(x + j * bar_width, heights, bar_width,
               color=PALETTE[j], edgecolor="black", zorder=3,
               label=SERIES_LABELS[j])

    for i, s in enumerate(SEQ_KEYS):
        lora, jenga, jenga2d = raw[s]
        if lora > 0 and jenga > 0:
            top = norm[s][1]
            ax.text(x[i] + 1 * bar_width, top + 0.015, f"{lora / jenga:.2f}x",
                    ha="center", va="bottom", fontsize=12, rotation=90, color="#222")
        if lora > 0 and jenga2d > 0:
            top = norm[s][2]
            ax.text(x[i] + 2 * bar_width, top + 0.015, f"{lora / jenga2d:.2f}x",
                    ha="center", va="bottom", fontsize=12, rotation=90, color="#222")

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(SEQ_LABELS, fontsize=13)
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0.4, 1.35)
    ax.set_xlabel("Sequence Length", fontsize=13)
    ax.set_ylabel("Execution Time (Normalized)", fontsize=13)

    header_y = 1.04
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, header_y),
              ncol=3, frameon=False, fontsize=12, handletextpad=0.4,
              columnspacing=1.2, borderpad=0.0, borderaxespad=0.0)
    ax.text(0.0, header_y, "(a) 2D Sparsity",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=13, fontweight="bold")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    render("output_figures/extension/2d/2d-sparsity.pdf")
