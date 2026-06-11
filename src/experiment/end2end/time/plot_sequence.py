"""Paper Figure 13 bottom panel. Step time across sequence lengths for
Llama 2 and Llama 3, four bars per sequence: LoRA-Llama2, Jenga-Llama2,
LoRA-Llama3, Jenga-Llama3. Normalized per sequence to the maximum of
the four. OOM bars are rendered as red 'OOM' text.
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np


SEQ_MAP = {"4096": "4k", "8192": "8k", "16384": "16k",
           "32768": "32k", "49152": "48k", "65536": "64k"}
SEQ_ORDER = ["4k", "8k", "16k", "32k", "48k", "64k"]
PALETTE = ["#255475", "#5D7F84", "#DCBCAC", "#D6838D"]
SERIES = ["lora_llama2", "jenga_llama2", "lora_llama3", "jenga_llama3"]
SERIES_LABELS = ["LoRA-Llama2", "Jenga-Llama2", "LoRA-Llama3", "Jenga-Llama3"]


def parse_logs(log_dir):
    data = {model: {method: {s: 0.0 for s in SEQ_ORDER} for method in ["lora", "jenga"]}
            for model in ["llama2", "llama3"]}
    for filename in os.listdir(log_dir):
        if not filename.startswith("checkpoint-") or not filename.endswith(".log"):
            continue
        if "a800" not in filename:
            continue
        if "llora" in filename:
            continue
        parts = filename.split("-")
        model = "llama2" if "llama2" in filename else "llama3" if "llama3" in filename else None
        if model is None:
            continue
        method = "lora" if "baseline" in filename else "jenga"
        seq_digits = next((p for p in parts if p.isdigit() and p in SEQ_MAP), None)
        if not seq_digits:
            continue
        seq_label = SEQ_MAP[seq_digits]
        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath) as f:
                content = f.read()
        except Exception:
            continue
        match = re.search(r"total time:\s*([\d.]+)", content)
        if match:
            data[model][method][seq_label] = float(match.group(1))
    return data


def render(out_path):
    data = parse_logs("logs/end2end/time")
    n = len(SEQ_ORDER)
    raw = {s: [0.0, 0.0, 0.0, 0.0] for s in SEQ_ORDER}
    for s in SEQ_ORDER:
        raw[s][0] = data["llama2"]["lora"][s]
        raw[s][1] = data["llama2"]["jenga"][s]
        raw[s][2] = data["llama3"]["lora"][s]
        raw[s][3] = data["llama3"]["jenga"][s]

    norm = {s: [0.0, 0.0, 0.0, 0.0] for s in SEQ_ORDER}
    for s in SEQ_ORDER:
        ref = max(raw[s] + [1.0])
        norm[s] = [v / ref for v in raw[s]]

    bar_width = 0.2
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    for j in range(4):
        offset = (j - 1.5) * bar_width
        heights = [norm[s][j] for s in SEQ_ORDER]
        ax.bar(x + offset, heights, bar_width,
               color=PALETTE[j], edgecolor="black", zorder=3,
               label=SERIES_LABELS[j])

    for i, s in enumerate(SEQ_ORDER):
        for j in range(4):
            if raw[s][j] > 0:
                continue
            xpos = x[i] + (j - 1.5) * bar_width
            ax.text(xpos, 0.55, "OOM", ha="center", va="bottom",
                    fontsize=11, color="#cc1f1f", rotation=90)
        lora_l2, jenga_l2, lora_l3, jenga_l3 = raw[s]
        if lora_l2 > 0 and jenga_l2 > 0:
            xpos = x[i] + (1 - 1.5) * bar_width
            ax.text(xpos, 0.55, f"{lora_l2 / jenga_l2:.2f}x",
                    ha="center", va="bottom", fontsize=11,
                    rotation=90, color="#222")
        if lora_l3 > 0 and jenga_l3 > 0:
            xpos = x[i] + (3 - 1.5) * bar_width
            ax.text(xpos, 0.55, f"{lora_l3 / jenga_l3:.2f}x",
                    ha="center", va="bottom", fontsize=11,
                    rotation=90, color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(SEQ_ORDER, fontsize=13)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0.5, 1.06)
    ax.set_xlabel("Sequence Length", fontsize=13)
    ax.set_ylabel("Execution Time (Normalized)", fontsize=13)

    header_y = 1.06
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, header_y),
              ncol=4, frameon=False, fontsize=11, handletextpad=0.4,
              columnspacing=1.0, borderpad=0.0, borderaxespad=0.0)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    render("output_figures/end2end/time/time-seq.pdf")
