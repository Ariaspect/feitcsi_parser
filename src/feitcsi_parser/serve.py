"""Entry point to launch the Bokeh server app.

Exposes a console script so users can run:

    uv run feitcsi-serve

instead of:

    uv run bokeh serve src/feitcsi_parser/bokeh_app.py --show --port 5006

Any CLI args are forwarded to bokeh serve (e.g. `--port 5007`).
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from bokeh.command.bootstrap import main as bokeh_main

    app_path = str(Path(__file__).resolve().parent / "bokeh_app.py")
    # Default args; user args (if any) override/extend.
    # We pass --show --port 5006 by default; user can pass --port 5007 etc.
    # bokeh serve uses argparse, so later args win for single-value options.
    argv = ["bokeh", "serve", app_path, "--show", "--port", "5006", *sys.argv[1:]]
    bokeh_main(argv)


if __name__ == "__main__":
    main()
