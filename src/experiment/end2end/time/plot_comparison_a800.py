"""Paper Figure 13 top panel. Step time on the A100 80 GB device.

Six models normalized to LoRA = 1.0. Speedup ratio LoRA / Jenga is
rendered inside the Jenga bar. The script is named `plot_comparison_a800`
for backward compatibility with the artifact layout; the actual device
we ran on is the A100 80 GB and that is the label used in the figure.
"""
import os
import re

import matplotlib.pyplot as plt


def parse_time_logs(log_dir, device_tag, seq_label_k):
    seq_token = str(seq_label_k * 1024)
    data = []
    for filename in os.listdir(log_dir):
        if not filename.endswith(".log"):
            continue
        if filename.startswith("checkpoint"):
            continue
        if seq_token not in filename:
            continue
        if device_tag not in filename:
            continue
        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, "r") as f:
                lines = [line.strip() for line in f if "total time" in line]
        except Exception:
            continue
        base = filename.replace(".log", "")
        if "baseline" in base:
            case_type = "lora"
            base = base.replace("-baseline", "")
        elif "llora" in base:
            case_type = "longlora"
            base = base.replace("-llora", "")
        else:
            case_type = "jenga"
        parts = base.split("-")
        if "opt" not in filename:
            model = parts[0]
        else:
            model = parts[0] + parts[1]
        case_name = f"{model}-{case_type}"
        if not lines:
            total_time = 0
        else:
            match = re.search(r"total time:\s*([\d.]+)", lines[-1])
            total_time = float(match.group(1)) if match else 0
        data.append({"case": case_name, "time": total_time})
    return data


PALETTE = ["#255475", "#5D7F84", "#DCBCAC"]
MODELS = ["llama3", "llama2", "opt6.7b", "opt2.7b", "opt1.3b", "opt350m"]
METHODS = ["lora", "longlora", "jenga"]
METHOD_LABELS = ["LoRA", "LongLoRA", "Jenga"]


def render(seq_label_k, device_tag, device_display, out_path):
    data = parse_time_logs("logs/end2end/time", device_tag, seq_label_k)

    abs_time = {m: [] for m in METHODS}
    for model in MODELS:
        for method in METHODS:
            case = f"{model}-{method}"
            t = next((d["time"] for d in data if d["case"] == case), 0)
            abs_time[method].append(t)

    norm = {m: [] for m in METHODS}
    speedups = []
    for i in range(len(MODELS)):
        ref = max(abs_time["lora"][i], abs_time["longlora"][i], abs_time["jenga"][i], 1.0)
        for m in METHODS:
            norm[m].append(abs_time[m][i] / ref)
        lora = abs_time["lora"][i]
        jenga = abs_time["jenga"][i]
        speedups.append(lora / jenga if jenga > 0 and lora > 0 else None)

    bar_width = 0.25
    x = list(range(len(MODELS)))

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    for i, method in enumerate(METHODS):
        ax.bar([p + i * bar_width for p in x], norm[method], bar_width,
               label=METHOD_LABELS[i], color=PALETTE[i], edgecolor="black", zorder=3)

    for i in range(len(MODELS)):
        for j, method in enumerate(METHODS):
            if abs_time[method][i] == 0:
                xpos = x[i] + j * bar_width
                ax.text(xpos, 0.55, "OOM", ha="center", va="bottom",
                        fontsize=11, color="#cc1f1f", rotation=90)
        s = speedups[i]
        if s is not None:
            xpos = x[i] + 2 * bar_width
            ax.text(xpos, 0.55, f"{s:.2f}x", ha="center", va="bottom",
                    fontsize=12, rotation=90, color="#222")

    ax.set_xticks([p + bar_width for p in x])
    ax.set_xticklabels(MODELS, fontsize=13)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0.5, 1.06)
    ax.set_ylabel("Execution Time (Normalized)", fontsize=13)

    header_y = 1.06
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, header_y),
              ncol=3, frameon=False, fontsize=12, handletextpad=0.4,
              columnspacing=1.2, borderpad=0.0, borderaxespad=0.0)
    ax.text(1.0, header_y, f"GPU: {device_display}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=13, fontweight="bold")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    render(8, "a800", "A100 80GB", "output_figures/end2end/time/time-a100.pdf")
