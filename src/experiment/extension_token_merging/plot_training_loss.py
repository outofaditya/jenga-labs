"""Plot the I4 LoRA training loss curve in the repo's standard line plot style.

Matches src/experiment/extension_cnn_predictor/plot_loss.py: 6 x 4 figure,
single dark line for the raw trace, single accent line for the smoothed
trace, alpha shaded band omitted (only one seed). Simple xlabel /
ylabel / legend, tight layout.
"""
import argparse
import ast
import os
import re

import matplotlib.pyplot as plt


LOSS_PATTERN = re.compile(r"\{'loss': [^}]+\}")
LOGGING_STEPS = 20


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
    plt.figure(figsize=(6, 4))
    plt.plot(steps, losses, label="raw", linestyle="--", color="#444", linewidth=1)
    plt.plot(steps, smoothed, label=f"{args.window} step MA", linestyle="-", color="#1f77b4", linewidth=2)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    plt.close()
    print(f"wrote {args.out} ({len(losses)} points)")


if __name__ == "__main__":
    main()
