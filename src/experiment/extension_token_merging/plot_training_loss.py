"""Parse the I4 training log and plot the LoRA adapter's loss curve.

Reads the trainer's stdout/stderr capture, extracts every
``{'loss': X, ..., 'epoch': Y}`` dict, and emits a two panel figure:
raw per logging step loss on the left, and a smoothed moving average on
the right so the converged plateau is legible.
"""
import argparse
import ast
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS_PATTERN = re.compile(r"\{'loss': [^}]+\}")


def parse_log(path: str):
    losses = []
    steps = []
    text = Path(path).read_text(errors="ignore")
    # logging_steps=5 in the trainer call; each match corresponds to one log
    for i, m in enumerate(LOSS_PATTERN.finditer(text)):
        try:
            d = ast.literal_eval(m.group(0))
        except Exception:
            continue
        if "loss" not in d:
            continue
        losses.append(float(d["loss"]))
        steps.append(5 * (i + 1))  # logging_steps = 5
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
    parser.add_argument("--log", default="logs/extensions/token_merging/train-llama2-8192-a6000.log")
    parser.add_argument("--out", default="output_figures/extensions/token_merging/train_loss.pdf")
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    steps, losses = parse_log(args.log)
    if not losses:
        raise SystemExit(f"no loss entries parsed from {args.log}")
    smoothed = moving_average(losses, args.window)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(steps, losses, color="#3366cc", linewidth=1)
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Per logging step loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, losses, color="#bbbbbb", linewidth=0.8, label="raw")
    axes[1].plot(steps, smoothed, color="#cc3333", linewidth=2.0, label=f"{args.window} window MA")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Smoothed loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Atom I4. LoRA training loss with token merging enabled from step 0")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"parsed {len(losses)} loss entries; wrote {out}")


if __name__ == "__main__":
    main()
