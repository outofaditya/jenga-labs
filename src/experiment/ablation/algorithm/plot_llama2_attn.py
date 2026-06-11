"""Paper Figure 15 (top left). Llama 2 7B attention layer ablation:
memory retained proportion and threshold value per layer.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _algo_render import render  # noqa: E402


if __name__ == "__main__":
    render(
        log_path="logs/ablations/algorithm/llama2-attn.log",
        out_path="output_figures/ablations/algorithm/algorithm-llama2-attn.pdf",
        title="Attention",
        memory_color="#255475",
        threshold_color="#5D7F84",
        show_threshold=True,
    )
