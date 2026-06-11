"""Paper Figure 15 (bottom left). OPT 6.7B attention layer ablation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _algo_render import render  # noqa: E402


if __name__ == "__main__":
    render(
        log_path="logs/ablations/algorithm/opt-6.7b-attn.log",
        out_path="output_figures/ablations/algorithm/algorithm-opt-attn.pdf",
        title="Attention",
        memory_color="#255475",
        threshold_color="#5D7F84",
        show_threshold=True,
    )
