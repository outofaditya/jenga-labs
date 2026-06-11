"""Paper Figure 13 middle panel. Step time on the A40 48 GB device."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_comparison_a800 import render  # noqa: E402


if __name__ == "__main__":
    render(4, "a40", "A40 48GB", "output_figures/end2end/time/time-a40.pdf")
