"""The live feed: fetch the newest SatNOGS recordings in the background, hand each over as it lands.

DESIGN NOTE: OBSERVER (Qt signals across threads)
-------------------------------------------------
The download runs on a worker thread; each finished recording is
*emitted* as a signal, which Qt delivers on the window's own thread.
The screens subscribe to :attr:`LiveFeed.arrived` and never touch the
network themselves.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from .. import ArtifactCache, Manifest, SatnogsApiSource
from ..unvetted import fetch_latest


class LiveFeed(QObject):
    arrived = Signal(object)      # an Observation, ready in the cache
    status = Signal(str)
    finished = Signal(int)

    def __init__(self, cache: ArtifactCache, manifest: Manifest, count: int = 75, parent=None) -> None:
        super().__init__(parent)
        self._cache, self._manifest, self.count = cache, manifest, count
        self.running = False

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.status.emit("Asking SatNOGS for its newest recordings…")
        feed = self

        class Job(QRunnable):
            def run(self) -> None:
                added = 0
                try:
                    added = fetch_latest(SatnogsApiSource(), feed._cache, feed._manifest, feed.count, workers=6,
                                         progress=feed.status.emit, added=feed.arrived.emit)
                except Exception as error:  # noqa: BLE001 - no network must not take the window down
                    feed.status.emit(f"Could not reach SatNOGS ({type(error).__name__}); showing saved recordings.")
                finally:
                    feed.running = False
                    feed.finished.emit(added)

        QThreadPool.globalInstance().start(Job())
