"""Shared renderer for the four algorithm-ablation panels (paper Figure 15).

Each panel is one twin-axes line plot: memory retained proportion on the
left y axis (beady line with circular markers), threshold value on the
right y axis. A top legend names the lines and a bold title inside the
top right corner names the layer kind. The OPT 6.7B MLP panel passes
show_threshold=False and gets a 'threshold = 0' annotation instead.
"""
import re

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def parse_log(log_path):
    layers, thresholds, memories = [], [], []
    pattern = re.compile(r"layer:\s*(\d+)\s*,\s*threshold:\s*([0-9.]+)\s*,\s*memory:\s*([0-9.]+)")
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                layers.append(int(m.group(1)) + 1)
                thresholds.append(float(m.group(2)))
                memories.append(float(m.group(3)))
    return layers, thresholds, memories


def render(log_path, out_path, title, memory_color, threshold_color, show_threshold):
    layers, thresholds, memories = parse_log(log_path)
    fig, ax_left = plt.subplots(figsize=(5, 3))
    ax_left.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)

    ax_left.plot(layers, memories,
                 color="black", linewidth=1, zorder=10,
                 marker="o", markersize=4,
                 markerfacecolor=memory_color, markeredgewidth=0.5)
    ax_left.set_ylim(0, 1)
    ax_left.set_xticks([1, 6, 11, 16, 21, 26, 31])
    ax_left.tick_params(axis="both", labelsize=12)
    ax_left.set_ylabel("Memory Retained Proportion", fontsize=12)

    if show_threshold:
        ax_right = ax_left.twinx()
        ax_right.plot(layers, thresholds,
                      color="black", linewidth=1, zorder=10,
                      marker="o", markersize=4,
                      markerfacecolor=threshold_color, markeredgewidth=0.5)
        ax_right.set_ylim(0, 1)
        ax_right.tick_params(axis="y", labelsize=12)
        ax_right.set_ylabel("Threshold Value", fontsize=12)

    handles = [
        mpatches.Patch(facecolor=memory_color, edgecolor="black", label="memory"),
    ]
    if show_threshold:
        handles.append(
            mpatches.Patch(facecolor=threshold_color, edgecolor="black", label="threshold")
        )

    header_y = 1.04
    ax_left.legend(handles=handles, loc="lower left",
                   bbox_to_anchor=(0.0, header_y),
                   ncol=2, frameon=False, fontsize=12,
                   handletextpad=0.5, columnspacing=1.2,
                   borderpad=0.0, borderaxespad=0.0)
    ax_left.text(0.97, 0.95, title, transform=ax_left.transAxes,
                 ha="right", va="top", fontsize=16)

    if not show_threshold:
        ax_left.text(0.97, 0.05, "threshold = 0",
                     transform=ax_left.transAxes, ha="right", va="bottom",
                     fontsize=11, color="#222")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
