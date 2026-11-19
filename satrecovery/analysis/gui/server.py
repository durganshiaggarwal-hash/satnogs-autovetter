"""The analysis backend, running in a thread of the app's own process."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from ...gui.client import BackendError
from ...gui.server import free_port, is_backend


class AnalysisServer:
    """Starts :func:`~satrecovery.analysis.api.create_analysis_app` in the background."""

    def __init__(self, data_dir: str | Path = "satrecovery_data", rml24_path: str | Path | None = None,
                 classifier_path: str | Path | None = None, host: str = "127.0.0.1",
                 port: int | None = None) -> None:
        self._data_dir = Path(data_dir)
        self._rml24 = rml24_path
        self._classifier = classifier_path
        self._host = host
        self._port = int(port) if port else free_port(host)
        self._server = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def start(self, timeout: float = 30.0) -> "AnalysisServer":
        import uvicorn

        from ...backend.settings import Settings
        from ..api import create_analysis_app

        settings = Settings(data_dir=self._data_dir, host=self._host, port=self._port, rml24_path=self._rml24)
        app = create_analysis_app(settings, self._classifier)
        self._server = uvicorn.Server(uvicorn.Config(app, host=self._host, port=self._port, log_level="warning"))
        self._thread = threading.Thread(target=self._server.run, name="satrec-analysis", daemon=True)
        self._thread.start()
        deadline = time.time() + timeout
        while not is_backend(self.url, timeout=0.5):
            if not self._thread.is_alive():
                raise BackendError(0, "the analysis backend failed to start")
            if time.time() > deadline:
                raise BackendError(0, f"the analysis backend did not answer within {timeout:.0f}s")
            time.sleep(0.1)
        return self

    def stop(self, timeout: float = 10.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None

    def __enter__(self) -> "AnalysisServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


__all__ = ["AnalysisServer"]
