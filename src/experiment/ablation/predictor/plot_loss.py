"""Paper Figure 16. Predictor training loss across Llama 2 7B and OPT 6.7B
on LongAlign (LA) and RedPajama (RP). One panel, four smooth lines, legend
inside the chart on the upper right.
"""
import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def load_loss(path):
    with open(path) as f:
        return [entry["loss"] for entry in json.load(f)["log_history"] if "loss" in entry]


LIMIT = 400
SERIES = [
    ("Llama2-LA",     "logs/ablations/predictor/llama2_16384_predictor_la/trainer_state.json",     "#255475"),
    ("Llama2-RP",     "logs/ablations/predictor/llama2_16384_predictor_red/trainer_state.json",    "#5D7F84"),
    ("OPT6.7B-LA",    "logs/ablations/predictor/opt-6.7b_16384_predictor_la/trainer_state.json",   "#DCBCAC"),
    ("OPT6.7B-RP",    "logs/ablations/predictor/opt-6.7b_16384_predictor_red/trainer_state.json",  "#D6838D"),
]


def render(out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)

    for label, path, color in SERIES:
        losses = load_loss(path)[:LIMIT]
        ax.plot(losses, color=color, linewidth=1.6, zorder=3)

    ax.tick_params(axis="both", labelsize=12)

    handles = [mpatches.Patch(facecolor=c, edgecolor="black", label=lbl)
               for lbl, _, c in SERIES]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              fontsize=13, handletextpad=0.6, labelspacing=0.7,
              borderaxespad=1.0)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    render("output_figures/ablations/predictor/predictor-loss.pdf")
