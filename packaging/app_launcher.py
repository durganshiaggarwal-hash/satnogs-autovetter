"""The double-click app's starting point (used only by the PyInstaller build).

A double-clicked program does not start in a predictable folder, so
before anything else this decides where the app keeps its files --
the trained AI and the history -- and tells the app through
``SATREC_DATA_DIR``:

* Windows: in ``satrecovery_data`` inside the app's own folder, next to
  ``Satellite Signal Analyser.exe``.
* Mac: in ``satrecovery_data`` next to ``Satellite Signal Analyser.app``.

Put ``classifier.pt`` and ``classifier.json`` in ``satrecovery_data/models``
there.  The dataset is chosen inside the app and remembered.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def app_folder() -> Path:
    executable = Path(sys.executable).resolve()
    for parent in executable.parents:
        if parent.suffix == ".app":          # Mac: .../Satellite Signal Analyser.app/Contents/MacOS/...
            return parent.parent
    return executable.parent                 # Windows: the folder holding the .exe


def main() -> int:
    if getattr(sys, "frozen", False):
        os.environ.setdefault("SATREC_DATA_DIR", str(app_folder() / "satrecovery_data"))
    from satrecovery.analysis.__main__ import main as run_app

    return run_app([])


if __name__ == "__main__":
    sys.exit(main())
