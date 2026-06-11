"""Paper Figure 14 (a). Memory footprint stacked horizontal bars for
Llama 2 7B at 8 K (LoRA, LongLoRA, Jenga) and at 10 K to 16 K under
Jenga. Categories: model state, activation, others, predictor.
"""
import os
import re

import matplotlib.pyplot as plt


LOG_DIR = "logs/ablations/memory-breakdown"
LOG_FILES = [
    "llama2-8192-a800-baseline.log",
    "llama2-8192-a800-llora.log",
    "llama2-8192-a800.log",
    "llama2-10240-a800.log",
    "llama2-12288-a800.log",
    "llama2-14336-a800.log",
    "llama2-16384-a800.log",
]
PALETTE = ["#255475", "#5D7F84", "#DCBCAC", "#D6838D"]
STAGES = ["model state", "activation", "others", "predictor"]
KEYS = ["model_states", "activations", "others", "predictors"]


def read_logs():
    predictor_path = os.path.join(LOG_DIR, "predictor.log")
    predictor_mem = 0.0
    if os.path.exists(predictor_path):
        with open(predictor_path) as f:
            for line in f:
                m = re.search(r"Total memory:\s*([\d.]+)\s*MB", line)
                if m:
                    predictor_mem = float(m.group(1))

    rows = []
    for filename in LOG_FILES:
        path = os.path.join(LOG_DIR, filename)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            lines = [line.strip() for line in f if "allocaiton" in line]
        if len(lines) < 2:
            continue
        second = re.search(r"allocaiton:\s*([0-9.]+)", lines[1])
        last_alloc = re.search(r"allocaiton:\s*([0-9.]+)", lines[-1])
        reserve = re.search(r"reserve:\s*([0-9.]+)", lines[-1])
        if not (second and last_alloc and reserve):
            continue
        model_states = float(last_alloc.group(1))
        total = float(reserve.group(1))
        activations = float(second.group(1)) - model_states
        others = total - float(second.group(1))
        size_k = int(filename.split("-")[1]) // 1024
        if "baseline" in filename:
            tag = "LoRA"
        elif "llora" in filename:
            tag = "LongLoRA"
        else:
            tag = "Jenga"
        rows.append({
            "case": f"{size_k}K {tag}",
            "model_states": model_states / 1024,
            "activations": activations / 1024,
            "others": others / 1024,
            "predictors": (predictor_mem / 1024) if tag == "Jenga" else 0.0,
        })
    return rows


def render(out_path):
    rows = read_logs()
    rows = rows[::-1]
    cases = [r["case"] for r in rows]
    n = len(rows)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.grid(axis="x", linestyle="--", alpha=0.6, zorder=0)

    bar_height = 0.6
    y_pos = list(range(n))
    for i, r in enumerate(rows):
        left = 0
        for j, key in enumerate(KEYS):
            width = r[key]
            ax.barh(y_pos[i], width, left=left, color=PALETTE[j],
                    edgecolor="black", height=bar_height, zorder=3,
                    label=STAGES[j] if i == 0 else None)
            left += width

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cases, fontsize=13)
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlabel("Memory Footprint (GB)", fontsize=13)
    totals = [r["model_states"] + r["activations"] + r["others"] + r["predictors"] for r in rows]
    ax.set_xlim(0, max(totals) * 1.05)

    header_y = 1.04
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower right",
              bbox_to_anchor=(1.0, header_y), ncol=4, frameon=False,
              fontsize=12, handletextpad=0.4, columnspacing=1.0,
              borderpad=0.0, borderaxespad=0.0)
    ax.text(-0.13, header_y, "(a) Memory Footprint",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=13, fontweight="bold")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    render("output_figures/ablations/memory-breakdown/memory-breakdown.pdf")
