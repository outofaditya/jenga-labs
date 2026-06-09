"""Plot the I4 LoRA training loss curve.

Matches the visual idiom of src/experiment/ablation/predictor/plot_loss.py:
compact 4 x 3 figure, smooth lines in the project palette, no markers,
dashed y grid, 12 pt ticks. Two traces are drawn, the raw per logging
step loss and a moving average. Horizontal legend on top, Title Case
labels, bbox_inches='tight' on save.
"""
import argparse
import ast
import os
import re

import matplotlib.pyplot as plt


LOSS_PATTERN = re.compile(r"\{'loss': [^}]+\}")
LOGGING_STEPS = 20
PALETTE = ['#255475', '#5D7F84', '#DCBCAC', '#D6838D']


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
    parser.add_argument("--out", default="output_figures/extensions/token_merging/train_loss.pdf")
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    steps, losses = parse_log(args.log)
    if not losses:
        raise SystemExit(f"no loss entries parsed from {args.log}")
    smoothed = moving_average(losses, args.window)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    ax.plot(steps, losses, color=PALETTE[1], linewidth=2, label="Raw")
    ax.plot(steps, smoothed, color=PALETTE[0], linewidth=2, label=f"{args.window} Step Moving Average")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center",
               bbox_to_anchor=(0.5, 1.02),
               ncol=2,
               frameon=False,
               fontsize=11)
    plt.savefig(args.out, bbox_inches="tight")
    plt.close()
    print(f"wrote {args.out} ({len(losses)} points)")


if __name__ == "__main__":
    main()
