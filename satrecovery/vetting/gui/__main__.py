"""Open the vetting station on the observations downloaded so far.

    python -m satrecovery.vetting.gui
    python -m satrecovery.vetting.gui --data satrecovery_data/satnogs --screenshot shot.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SatNOGS vetting station")
    parser.add_argument("--data", default=str(ROOT / "satrecovery_data" / "satnogs"))
    parser.add_argument("--screenshot", help="save a picture of the window and quit (for the report)")
    parser.add_argument("--no-orbit", action="store_true", help="skip the orbit view (works offline)")
    parser.add_argument("--select", type=int, help="open on this observation id")
    args = parser.parse_args(argv)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from ...gui.theme import load_fonts
    from .. import ArtifactCache, Manifest
    from ..session import VettingSession
    from .window import VettingWindow

    data = Path(args.data)
    manifest = Manifest(data / "manifest.jsonl")
    print(f"Loading {len(manifest)} observations from {data} ...", flush=True)
    app = QApplication.instance() or QApplication(sys.argv)
    load_fonts()
    from ...core.exceptions import ValidationError

    try:
        unvetted = Manifest(data / "unvetted.jsonl")
        print(f"  and {len(unvetted)} that nobody has checked yet", flush=True)
        session = VettingSession.from_cache(
            manifest, ArtifactCache(data), log_path=data / "decisions.jsonl", unvetted=unvetted,
            progress=lambda done, total: print(f"  measuring waterfalls: {done}/{total}", flush=True))
    except ValidationError as error:
        print(f"Cannot open the vetting station: {error.reason}.\n"
              f"Download some first:  python scripts/fetch_satnogs.py --per-label 200")
        return 1
    explainer = None
    if not args.no_orbit:
        from ..orbit import SatnogsCatalog, StationDirectory, TrackExplainer

        cache = ArtifactCache(data)
        explainer = TrackExplainer(cache, SatnogsCatalog(data), StationDirectory(data))
    window = VettingWindow(session, explainer)
    window.show()
    if args.select:
        window.select(args.select)
    if args.screenshot:
        def capture() -> None:
            window.orbit.wait()
            app.processEvents()
            window.grab().save(args.screenshot)
            print(f"Saved {args.screenshot}")
            app.quit()
        QTimer.singleShot(1500, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
