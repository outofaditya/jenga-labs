"""Paper Figure 17 (left panel). Multi GPU scaling on Llama 2 7B."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _scale_render import render  # noqa: E402


if __name__ == "__main__":
    render(
        model_key="llama2",
        model_display="Llama2-7B",
        out_path="output_figures/scalability/scalability-llama2.pdf",
    )
