"""Shared renderer for the two scalability panels (paper Figure 17).

Each panel plots execution time (ms) vs GPU count for three sequence
lengths (1K, 2K, 4K). Legend at top, bold model name inside the chart
upper right.
"""

import os
import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


PALETTE = ["#255475", "#5D7F84", "#DCBCAC"]
SEQ_KEYS = ["1024", "2048", "4096"]
SEQ_LABELS = ["1K", "2K", "4K"]
GPU_COUNTS = [1, 2, 4]


def parse_logs(model_key, log_dir="logs/scalability"):
    times = {seq: {gc: 0.0 for gc in GPU_COUNTS} for seq in SEQ_KEYS}
    for filename in os.listdir(log_dir):
        if not filename.endswith(".log"):
            continue
        if not filename.startswith(model_key + "_"):
            continue
        stem = filename[:-4]
        parts = stem.split("_")
        try:
            seq = parts[-2]
            gc = int(parts[-1])
        except Exception:
            continue
        if seq not in times or gc not in GPU_COUNTS:
            continue
        path = os.path.join(log_dir, filename)
        try:
            with open(path) as f:
                content = f.read()
        except Exception:
            continue
        last = None
        for m in re.finditer(r"total time:\s*([\d.]+)", content):
            last = m
        if last is None:
            continue
        times[seq][gc] = float(last.group(1)) / gc
    return times


def render(model_key, model_display, out_path):
    times = parse_logs(model_key)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.grid(axis="both", linestyle="--", alpha=0.6, zorder=0)

    handles = []
    for i, seq in enumerate(SEQ_KEYS):
        y = [times[seq][gc] for gc in GPU_COUNTS]
        ax.plot(
            GPU_COUNTS,
            y,
            color=PALETTE[i],
            linewidth=1.6,
            zorder=3,
            marker="o",
            markersize=7,
            markerfacecolor=PALETTE[i],
            markeredgecolor=PALETTE[i],
        )
        handles.append(
            mpatches.Patch(facecolor=PALETTE[i], edgecolor="black", label=SEQ_LABELS[i])
        )

    ax.set_xticks(GPU_COUNTS)
    ax.set_xticklabels([str(gc) for gc in GPU_COUNTS], fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel("GPU Number", fontsize=13)
    ax.set_ylabel("Execution Time (ms)", fontsize=13)

    header_y = 1.04
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, header_y),
        ncol=3,
        frameon=False,
        fontsize=12,
        handletextpad=0.5,
        columnspacing=1.4,
        borderpad=0.0,
        borderaxespad=0.0,
    )
    ax.text(
        0.97,
        0.95,
        model_display,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=15,
        fontweight="bold",
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
