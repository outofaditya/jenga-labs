"""Paper Figure 12 reproduction. Six models at 4 K context, normalized to LongLoRA."""

import os
import re

import matplotlib.pyplot as plt


def parse_memory_logs(log_dir):
    data = []
    for filename in os.listdir(log_dir):
        if not filename.endswith(".log"):
            continue
        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, "r") as f:
                lines = [line.strip() for line in f if "reserve" in line]
        except Exception:
            continue
        base = filename.replace(".log", "")
        if "baseline" in base:
            case_type = "lora"
            base = base.replace("baseline", "")
        elif "llora" in base:
            case_type = "longlora"
            base = base.replace("llora", "")
        else:
            case_type = "jenga"
        parts = base.split("-")
        if "opt" not in filename:
            model = parts[0]
            context = parts[1]
        else:
            model = parts[0] + parts[1]
            context = parts[2]
        context = int(context) // 1024
        case_name = f"{model}-{case_type}-{context}K"
        if len(lines) < 10:
            memory = 0
        else:
            match = re.search(r"reserve:\s*([\d.]+)", lines[-1])
            memory = float(match.group(1)) if match else 0
        data.append({"case": case_name, "memory": memory})
    return data


PALETTE = ["#255475", "#5D7F84", "#DCBCAC"]
MODELS = ["llama3", "llama2", "opt6.7b", "opt2.7b", "opt1.3b", "opt350m"]
METHODS = ["lora", "longlora", "jenga"]
METHOD_LABELS = ["LoRA", "LongLoRA", "Jenga"]


def render(seq_label_k, out_path):
    data = parse_memory_logs("logs/end2end/memory")
    data_seq = [d for d in data if f"-{seq_label_k}K" in d["case"]]

    abs_memory = {m: [] for m in METHODS}
    for model in MODELS:
        for method in METHODS:
            case = f"{model}-{method}-{seq_label_k}K"
            mem = next((d["memory"] for d in data_seq if d["case"] == case), 0)
            abs_memory[method].append(mem)

    norm = {m: [] for m in METHODS}
    savings = []
    for i in range(len(MODELS)):
        ref = max(abs_memory["lora"][i], abs_memory["longlora"][i], 1.0)
        for m in METHODS:
            norm[m].append(abs_memory[m][i] / ref)
        jenga = abs_memory["jenga"][i]
        longlora = abs_memory["longlora"][i]
        savings.append(longlora / jenga if jenga > 0 and longlora > 0 else None)

    bar_width = 0.25
    x = list(range(len(MODELS)))

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    for i, method in enumerate(METHODS):
        ax.bar(
            [p + i * bar_width for p in x],
            norm[method],
            bar_width,
            label=METHOD_LABELS[i],
            color=PALETTE[i],
            edgecolor="black",
            zorder=3,
        )

    for i, s in enumerate(savings):
        if s is None:
            continue
        xpos = x[i] + 2 * bar_width
        ax.text(
            xpos,
            0.05,
            f"{s:.2f}x",
            ha="center",
            va="bottom",
            fontsize=13,
            rotation=90,
            color="#222",
        )

    ax.set_xticks([p + bar_width for p in x])
    ax.set_xticklabels(MODELS, fontsize=13)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Memory Footprint (Normalized)", fontsize=13)

    header_y = 1.06
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, header_y),
        ncol=3,
        frameon=False,
        fontsize=12,
        handletextpad=0.4,
        columnspacing=1.2,
        borderpad=0.0,
        borderaxespad=0.0,
    )
    ax.text(
        1.0,
        header_y,
        f"Sequence Length = {seq_label_k}K",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    render(4, "output_figures/end2end/memory/memory-4k.pdf")
