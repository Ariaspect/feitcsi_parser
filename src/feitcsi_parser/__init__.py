"""FeitCSI parser and realtime heatmap visualizer.

Wraps CSIKit for parsing FeitCSI .dat files (Intel AX200/AX210 NIC)
and provides Bokeh-based realtime amplitude/phase heatmaps.
"""

from .parser import FeitCSICapture, load_capture, tail_window
from .visualizer import plot_heatmap

__all__ = ["FeitCSICapture", "load_capture", "tail_window", "plot_heatmap", "main"]


def main() -> None:
    """Entry point. Prints usage."""
    print("FeitCSI parser. Run realtime heatmap with:")
    print("  uv run bokeh serve src/feitcsi_parser/bokeh_app.py --show")
