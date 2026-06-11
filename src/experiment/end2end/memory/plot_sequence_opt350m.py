"""OPT 350M peak memory across sequence lengths. GB axis, model label in corner."""

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


PALETTE = ["#D6838D", "#F3AE75", "#F8F1E4"]
METHODS = ["lora", "longlora", "jenga"]
METHOD_LABELS = ["LoRA", "LongLoRA", "Jenga"]


def render(model_key, model_display, length_order, out_path):
    data = parse_memory_logs("logs/end2end/memory")
    memory_values = {
        length: {method: 0 for method in METHODS} for length in length_order
    }
    for entry in data:
        parts = entry["case"].split("-")
        if parts[0] != model_key:
            continue
        method = parts[1]
        length = parts[-1]
        if length in memory_values and method in METHODS:
            memory_values[length][method] = max(
                memory_values[length][method], entry["memory"]
            )

    bar_width = 0.25
    x = list(range(len(length_order)))

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)

    for i, method in enumerate(METHODS):
        heights = [memory_values[L][method] / 1000.0 for L in length_order]
        ax.bar(
            [p + i * bar_width for p in x],
            heights,
            bar_width,
            label=METHOD_LABELS[i],
            color=PALETTE[i],
            edgecolor="black",
            zorder=3,
        )

    for i, L in enumerate(length_order):
        for j, method in enumerate(METHODS):
            if memory_values[L][method] == 0:
                xpos = x[i] + j * bar_width
                ax.text(
                    xpos,
                    1.5,
                    "OOM",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    color="#cc1f1f",
                    rotation=90,
                )
        jenga = memory_values[L]["jenga"]
        longlora = memory_values[L]["longlora"]
        if jenga > 0 and longlora > 0:
            ratio = longlora / jenga
            xpos = x[i] + 2 * bar_width
            ax.text(
                xpos,
                (jenga / 1000.0) + 0.6,
                f"{ratio:.2f}x",
                ha="center",
                va="bottom",
                fontsize=12,
                rotation=90,
                color="#222",
            )

    ax.set_xticks([p + bar_width for p in x])
    ax.set_xticklabels(length_order, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel("Memory Footprint (GB)", fontsize=13)

    header_y = 1.08
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
        f"Model: {model_display}",
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
    render(
        "opt350m",
        "OPT 350M",
        ["4K", "8K", "16K", "32K", "64K"],
        "output_figures/end2end/memory/memory-opt350m.pdf",
    )
