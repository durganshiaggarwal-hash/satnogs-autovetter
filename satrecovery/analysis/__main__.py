"""Start the analysis app::

    python -m satrecovery.analysis --rml24 "C:/Users/you/Downloads/RML24_IQdata"

or, once the dataset folder has been chosen inside the app (it is
remembered), simply::

    python -m satrecovery.analysis

The AI classifier names the signal and the receiver recovers the
message, played as an animation.  The original app is unchanged and
still starts with ``python -m satrecovery.gui``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def default_data_dir() -> str:
    """Where the trained AI and the history live.

    Normally ``satrecovery_data`` in the folder the app is started from.
    The double-click app (``SATREC_DATA_DIR`` set by its launcher) keeps
    it next to itself instead, because a double-clicked program does not
    start in a predictable folder.
    """
    return os.environ.get("SATREC_DATA_DIR", "satrecovery_data")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Satellite Signal Analyser")
    parser.add_argument("--rml24", default=os.environ.get("SATREC_RML24"),
                        help="the RML24 dataset (RML24_IQdata.h5, its folder, or a sample .npz); "
                             "if left out, the folder chosen last time in the app")
    parser.add_argument("--data-dir", default=default_data_dir(),
                        help="where the trained models and the history live (shared with the original app)")
    parser.add_argument("--classifier", default=None,
                        help="the trained classifier (default: <data-dir>/models/classifier.pt)")
    parser.add_argument("--dark", action="store_true",
                        help="open in the dark look (the launcher does this, to match Live satellites)")
    args = parser.parse_args(argv)

    from PySide6.QtWidgets import QApplication

    from ..gui import theme
    from .gui.client import AnalysisClient
    from .gui.server import AnalysisServer
    from .gui.window import AnalysisWindow
    from .preferences import Preferences

    preferences = Preferences()
    rml24 = args.rml24
    if not rml24 and preferences.rml24_path and Path(preferences.rml24_path).exists():
        rml24 = preferences.rml24_path

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Satellite Signal Analyser")
    theme.load_fonts()

    def apply_theme(dark: bool) -> None:
        theme.use_dark(dark)
        theme.configure_plots()
        app.setStyleSheet(theme.stylesheet())

    apply_theme(args.dark or preferences.dark_mode)
    server = AnalysisServer(data_dir=args.data_dir, rml24_path=rml24, classifier_path=args.classifier).start()
    client = AnalysisClient(server.url)
    windows: list[AnalysisWindow] = []

    def open_window(geometry=None, tab: int = 0) -> None:
        window = AnalysisWindow(client, preferences)
        if geometry is not None:
            window.restoreGeometry(geometry)
        window.theme_changed.connect(lambda dark, w=window: rebuild(w, dark))
        window.show()
        if tab:
            window.tabs.setCurrentIndex(tab)
        windows.append(window)

    def rebuild(old: AnalysisWindow, dark: bool) -> None:
        """Dark switch: the same window, built again in the new colours."""
        apply_theme(dark)
        geometry, tab = old.saveGeometry(), old.tabs.currentIndex()
        open_window(geometry, tab)
        windows.remove(old)
        old.close()
        old.deleteLater()

    open_window()
    try:
        return app.exec()
    finally:
        server.stop()


if __name__ == "__main__":
    sys.exit(main())
