"""The live feed: fetch the newest SatNOGS recordings in the background, hand each over as it lands.

DESIGN NOTE: OBSERVER (Qt signals across threads)
-------------------------------------------------
The download runs on a background thread; each finished recording is
*emitted* as a signal, which Qt delivers on the window's own thread.
The screens subscribe to :attr:`LiveFeed.arrived` and never touch the
network themselves.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from .. import ArtifactCache, Manifest, SatnogsApiSource
from ..unvetted import fetch_latest


class _Closed(Exception):
    """The window closed while the feed was running."""


class LiveFeed(QObject):
    arrived = Signal(object)      # an Observation, ready in the cache
    status = Signal(str)
    finished = Signal(int)

    def __init__(self, cache: ArtifactCache, manifest: Manifest, count: int = 75, parent=None) -> None:
        super().__init__(parent)
        self._cache, self._manifest, self.count = cache, manifest, count
        self.running = False
        self.stopped = False

    def stop(self) -> None:
        """Ask a running fetch to stop at its next step (the window is closing)."""
        self.stopped = True

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.stopped = False
        self.status.emit("Asking SatNOGS for its newest recordings…")
        feed = self

        def say(signal, value) -> None:
            """Emit, unless the window (and this feed) has already gone: then stop quietly."""
            if feed.stopped:
                raise _Closed()
            try:
                signal.emit(value)
            except RuntimeError:          # "Signal source has been deleted": the app was closed
                feed.stopped = True
                raise _Closed() from None

        class Job:
            def run(self) -> None:
                added = 0
                try:
                    added = fetch_latest(SatnogsApiSource(), feed._cache, feed._manifest, feed.count, workers=6,
                                         progress=lambda text: say(feed.status, text),
                                         added=lambda obs: say(feed.arrived, obs))
                except _Closed:
                    return
                except Exception as error:  # noqa: BLE001 - no network must not take the window down
                    try:
                        say(feed.status, f"Could not reach SatNOGS ({type(error).__name__}); showing saved recordings.")
                    except _Closed:
                        return
                finally:
                    feed.running = False
                try:
                    say(feed.finished, added)
                except _Closed:
                    pass

        # A daemon thread, not Qt's thread pool: closing the app must not wait
        # for a slow SatNOGS request (Qt's pool is joined when the app quits).
        import threading

        self._job = Job()
        threading.Thread(target=self._job.run, name="satnogs-live-feed", daemon=True).start()
