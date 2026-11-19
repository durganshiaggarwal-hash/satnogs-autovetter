"""Start the application::

    python -m satrecovery.gui
    python -m satrecovery.gui --rml24 "C:/Users/me/Downloads/RML24_IQdata"

If a backend is already running at the address given (by default
http://127.0.0.1:8000), the window uses it.  Otherwise it starts one
itself, in the background, and stops it when the window closes.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Satellite Signal Recovery")
    parser.add_argument("--url", default="http://127.0.0.1:8000",
                        help="backend to use if one is running there")
    parser.add_argument("--data-dir", default="satrecovery_data",
                        help="where a backend started by this window keeps its files")
    parser.add_argument("--rml24", default=os.environ.get("SATREC_RML24"),
                        help="the RML24 dataset, so benchmarks can use real recordings")
    args = parser.parse_args()

    from PySide6.QtWidgets import QApplication

    from . import theme
    from .server import connect
    from .window import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Satellite Signal Recovery")
    theme.load_fonts()
    theme.configure_plots()
    app.setStyleSheet(theme.stylesheet())

    client, server = connect(args.url, args.data_dir, args.rml24)
    window = MainWindow(client, embedded=server is not None)
    window.show()
    try:
        return app.exec()
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    sys.exit(main())
