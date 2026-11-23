"""Launch the annotation UI.

    python 04_app/run_app.py

Equivalent to ``panel serve 04_app/annotation_app.py --show``, but it also sets
the MKL threading layer first.  Some Intel MKL builds abort the process with a
stack buffer overrun the first time a LAPACK routine runs when their threading
layer is resolved dynamically, and that has to be set before NumPy loads.

Options::

    python 04_app/run_app.py --port 5010     serve on another port
    python 04_app/run_app.py --no-show       do not open a browser
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must precede any import that pulls in NumPy.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

HERE = Path(__file__).resolve().parent
APP = HERE / "annotation_app.py"


def main():
    parser = argparse.ArgumentParser(description="Launch the annotation UI.")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--no-show", action="store_true",
                        help="do not open a browser window")
    parser.add_argument("--address", default=None,
                        help="bind address, e.g. 0.0.0.0 to allow other machines")
    args = parser.parse_args()

    from bokeh.command.bootstrap import main as bokeh_main

    argv = ["bokeh", "serve", str(APP), "--port", str(args.port)]
    if not args.no_show:
        argv.append("--show")
    if args.address:
        argv += ["--address", args.address, "--allow-websocket-origin",
                 f"{args.address}:{args.port}"]

    print(f"Serving {APP.name} on http://localhost:{args.port}/annotation_app")
    print("Press Ctrl+C to stop.\n")
    sys.exit(bokeh_main(argv))


if __name__ == "__main__":
    main()
