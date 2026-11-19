"""Open the Satellite Signal Analyser: the launcher, Lab analysis and Live satellites.

    python -m satrecovery.vetting.gui                 # launcher; fetches the newest recordings
    python -m satrecovery.vetting.gui --offline       # no network: saved recordings only
    python -m satrecovery.vetting.gui --expert        # straight to the vetting station
    python -m satrecovery.vetting.gui --replay 15489000

Normally it runs from the trained vetter in ``satrecovery_data/models`` (built by
``scripts/build_vetter_bundle.py``) and the latest recordings it fetches itself.
Without that bundle it trains at startup from the downloaded training data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Satellite Signal Analyser")
    parser.add_argument("--data", default=str(ROOT / "satrecovery_data" / "satnogs"))
    parser.add_argument("--models", default=str(ROOT / "satrecovery_data" / "models"))
    parser.add_argument("--latest", type=int, default=75, help="how many of the newest recordings to load (50-100)")
    parser.add_argument("--offline", action="store_true", help="do not contact SatNOGS: saved recordings only")
    parser.add_argument("--screenshot", help="save a picture of the window and quit (for the report)")
    parser.add_argument("--no-orbit", action="store_true", help="skip the orbit view")
    parser.add_argument("--select", type=int, help="expert mode: open on this observation id")
    parser.add_argument("--expert", action="store_true", help="open the vetting station (expert mode) directly")
    parser.add_argument("--replay", type=int, help="open the replay of this observation")
    parser.add_argument("--screen", choices=("launcher", "intro", "home"), default="launcher")
    parser.add_argument("--at", type=float, help="with --replay --screenshot: pause this far into the pass (0..1)")
    args = parser.parse_args(argv)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from ...core.exceptions import ValidationError
    from ...gui.theme import load_fonts
    from .. import ArtifactCache, Manifest
    from ..bundle import VetterBundle
    from ..session import VettingSession
    from .window import VettingWindow

    data = Path(args.data)
    cache = ArtifactCache(data)
    app = QApplication.instance() or QApplication(sys.argv)
    load_fonts()
    feed = None
    showcase: tuple = ()
    try:
        if VetterBundle.exists(args.models):
            bundle = VetterBundle.load(args.models)
            latest = Manifest(data / "latest.jsonl")
            newest = sorted(latest, key=lambda o: o.start.timestamp() if o.start else 0, reverse=True)[:args.latest]
            showcase = bundle.showcase
            print(f"Trained vetter: {bundle.n_train:,} observations.  Saved recent recordings: {len(newest)}.",
                  flush=True)
            session = VettingSession.from_bundle(bundle, list(newest) + list(showcase), cache,
                                                 log_path=data / "decisions.jsonl")
            if not args.offline:
                from .live import LiveFeed

                feed = LiveFeed(cache, latest, count=args.latest)
        else:
            manifest = Manifest(data / "manifest.jsonl")
            print(f"No trained vetter in {args.models}; training on {len(manifest)} observations "
                  f"(run scripts/build_vetter_bundle.py to skip this) ...", flush=True)
            session = VettingSession.from_cache(
                manifest, cache, log_path=data / "decisions.jsonl", unvetted=Manifest(data / "unvetted.jsonl"),
                progress=lambda done, total: print(f"  measuring waterfalls: {done}/{total}", flush=True))
    except ValidationError as error:
        print(f"Cannot open: {error.reason}.\n"
              f"Download some first:  python scripts/fetch_satnogs.py --per-label 200")
        return 1

    explainer = None
    if not args.no_orbit:
        from ..orbit import SatnogsCatalog, StationDirectory, TrackExplainer

        explainer = TrackExplainer(cache, SatnogsCatalog(data), StationDirectory(data))
    if args.expert or args.select:
        window = VettingWindow(session, explainer)
        window.show()
        if args.select:
            window.select(args.select)
    else:
        from .mission import MissionWindow

        window = MissionWindow(session, explainer, feed=feed, showcase=showcase,
                               start="home" if args.replay else args.screen)
        window.show()
        if args.replay:
            window.open_replay(args.replay)
    if feed is not None and not args.screenshot:
        feed.start()
    if args.screenshot:
        def capture() -> None:
            from PySide6.QtCore import QThreadPool

            QThreadPool.globalInstance().waitForDone(30000)
            app.processEvents()
            if hasattr(window, "replay") and args.replay:
                if args.at is not None:
                    window.replay.seek(args.at)
                else:
                    window.replay.finish()
            app.processEvents()
            window.grab().save(args.screenshot)
            print(f"Saved {args.screenshot}")
            app.quit()
        QTimer.singleShot(1500, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
