"""Paper Figure 14 (b). Execution time stacked horizontal bars for
Llama 2 7B at 8 K LoRA, 8 K Jenga, 10 K LoRA, 10 K Jenga, 12 K Jenga,
14 K Jenga, 16 K Jenga. Categories: forward, backward, optimizer step,
prediction.
"""

import os
import re

import matplotlib.pyplot as plt


LOG_DIR = "logs/ablations/time-breakdown"
LOG_FILES = [
    "llama2-8192-a800-baseline.log",
    "llama2-8192-a800.log",
    "llama2-10240-a800-baseline.log",
    "llama2-10240-a800.log",
    "llama2-12288-a800.log",
    "llama2-14336-a800.log",
    "llama2-16384-a800.log",
]
PALETTE = ["#255475", "#5D7F84", "#DCBCAC", "#D6838D"]
STAGES = ["forward", "backward", "optimizer step", "prediction"]


def read_logs():
    predictor_times = {}
    predictor_path = os.path.join(LOG_DIR, "predictor.log")
    if os.path.exists(predictor_path):
        with open(predictor_path) as f:
            for line in f:
                m = re.search(r"seq_len:\s*(\d+)\s*time:\s*([\d.]+)", line)
                if m:
                    predictor_times[int(m.group(1))] = float(m.group(2))

    rows = []
    for filename in LOG_FILES:
        path = os.path.join(LOG_DIR, filename)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            lines = f.readlines()
        time_line = next(
            (line for line in reversed(lines) if "forward time" in line), None
        )
        if not time_line:
            continue
        m = re.search(
            r"forward time: ([\d.]+), backward time: ([\d.]+), optimizer step time: ([\d.]+)",
            time_line,
        )
        if not m:
            continue
        fwd, bwd, opt = map(float, m.groups())
        size_k = int(filename.split("-")[1]) // 1024
        tag = "LoRA" if "baseline" in filename else "Jenga"
        pred = predictor_times.get(size_k * 1024, 0.0) if tag == "Jenga" else 0.0
        rows.append(
            {
                "case": f"{size_k}K {tag}",
                "forward": fwd,
                "backward": bwd,
                "optimizer step": opt,
                "prediction": pred,
            }
        )
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
        for j, stage in enumerate(STAGES):
            width = r[stage]
            ax.barh(
                y_pos[i],
                width,
                left=left,
                color=PALETTE[j],
                edgecolor="black",
                height=bar_height,
                zorder=3,
                label=stage if i == 0 else None,
            )
            left += width

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cases, fontsize=13)
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlabel("Execution Time (ms)", fontsize=13)
    totals = [
        r["forward"] + r["backward"] + r["optimizer step"] + r["prediction"]
        for r in rows
    ]
    ax.set_xlim(0, max(totals) * 1.05)

    header_y = 1.04
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(1.0, header_y),
        ncol=4,
        frameon=False,
        fontsize=12,
        handletextpad=0.4,
        columnspacing=1.0,
        borderpad=0.0,
        borderaxespad=0.0,
    )
    ax.text(
        -0.13,
        header_y,
        "(b) Execution Time",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    render("output_figures/ablations/time-breakdown/time-breakdown.pdf")
