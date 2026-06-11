"""Paper Figure 17 (right panel). Multi GPU scaling on OPT 6.7B."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _scale_render import render  # noqa: E402


if __name__ == "__main__":
    render(
        model_key="opt-6.7b",
        model_display="OPT-6.7B",
        out_path="output_figures/scalability/scalability-opt.pdf",
    )
