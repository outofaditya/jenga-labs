"""Paper Figure 12 reproduction. Six models at 8 K context, normalized to LongLoRA."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_comparison_4k import render  # noqa: E402


if __name__ == "__main__":
    render(8, "output_figures/end2end/memory/memory-8k.pdf")
