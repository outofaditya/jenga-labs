"""Paper Figure 15 (top right). Llama 2 7B MLP layer ablation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _algo_render import render  # noqa: E402


if __name__ == "__main__":
    render(
        log_path="logs/ablations/algorithm/llama2-mlp.log",
        out_path="output_figures/ablations/algorithm/algorithm-llama2-mlp.pdf",
        title="MLP",
        memory_color="#DCBCAC",
        threshold_color="#D6838D",
        show_threshold=True,
    )
