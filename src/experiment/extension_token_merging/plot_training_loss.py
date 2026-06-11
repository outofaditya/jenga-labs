"""Plot a LoRA training loss curve in the report's standard line plot style.

Two palettes:
  cool  — used for the Token Merging adapter (I4)
  warm  — used for the 2D sparsity adapter (I5)

The legend mirrors the rest of the report: rectangular swatches with a
black border, Title Case labels, anchored to the top right.
"""
import argparse
import ast
import os
import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


LOSS_PATTERN = re.compile(r"\{'loss': [^}]+\}")
LOGGING_STEPS = 20

PALETTES = {
    "cool": {"raw": "#5D7F84", "smoothed": "#255475"},
    "warm": {"raw": "#F3AE75", "smoothed": "#D6838D"},
}


def parse_log(path):
    steps, losses = [], []
    with open(path, errors="ignore") as f:
        text = f.read()
    for i, m in enumerate(LOSS_PATTERN.finditer(text)):
        try:
            d = ast.literal_eval(m.group(0))
        except Exception:
            continue
        if "loss" not in d:
            continue
        losses.append(float(d["loss"]))
        steps.append(LOGGING_STEPS * (i + 1))
    return steps, losses


def moving_average(values, window):
    if len(values) < window:
        return values[:]
    out = []
    s = sum(values[:window])
    out.append(s / window)
    for i in range(window, len(values)):
        s += values[i] - values[i - window]
        out.append(s / window)
    pad = [out[0]] * (window - 1)
    return pad + out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/extensions/token_merging/train-llama2-8192-a100.log")
    parser.add_argument("--out", default="output_figures/improvement/train-loss-i4.pdf")
    parser.add_argument("--palette", choices=["cool", "warm"], default="cool")
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    steps, losses = parse_log(args.log)
    if not losses:
        raise SystemExit(f"no loss entries parsed from {args.log}")
    smoothed = moving_average(losses, args.window)
    palette = PALETTES[args.palette]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    ax.plot(steps, losses, color=palette["raw"], linewidth=2, zorder=3)
    ax.plot(steps, smoothed, color=palette["smoothed"], linewidth=2, zorder=4)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)

    handles = [
        mpatches.Patch(facecolor=palette["raw"], edgecolor="black", label="Raw"),
        mpatches.Patch(facecolor=palette["smoothed"], edgecolor="black",
                       label=f"{args.window} Step Moving Average"),
    ]
    header_y = 1.04
    ax.legend(handles=handles, loc="lower right",
              bbox_to_anchor=(1.0, header_y), ncol=2, frameon=False,
              fontsize=11, handletextpad=0.4, columnspacing=1.2,
              borderpad=0.0, borderaxespad=0.0)
    plt.savefig(args.out, bbox_inches="tight")
    plt.close()
    print(f"wrote {args.out} ({len(losses)} points)")


if __name__ == "__main__":
    main()
