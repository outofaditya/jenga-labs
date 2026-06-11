"""Paper Figure 15 (bottom right). OPT 6.7B MLP layer ablation.
Threshold is zero across all layers so only the memory line is drawn.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _algo_render import render  # noqa: E402


if __name__ == "__main__":
    render(
        log_path="logs/ablations/algorithm/opt-6.7b-mlp.log",
        out_path="output_figures/ablations/algorithm/algorithm-opt-mlp.pdf",
        title="MLP",
        memory_color="#DCBCAC",
        threshold_color=None,
        show_threshold=False,
    )
