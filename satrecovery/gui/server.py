"""Starting the backend for the user, so the GUI is one command.

The GUI is a client of the Phase 7 server.  Asking a user to open a
second terminal and start the server first would be a poor first
impression, so the GUI checks whether one is already running and, if
not, starts one itself inside the same process.

A server the user started separately is used as it is.  That keeps the
option of running the backend on its own -- to use ``/docs`` alongside
the GUI, say -- without the GUI fighting it for the port.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

from .client import BackendError, HttpBackendClient


def free_port(host: str = "127.0.0.1") -> int:
    """A TCP port nothing is listening on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def is_backend(url: str, timeout: float = 1.0) -> bool:
    """Is *our* backend answering at ``url`` -- not just anything?"""
    try:
        health = HttpBackendClient(url, timeout=timeout).health()
    except BackendError:
        return False
    return isinstance(health, dict) and health.get("status") == "ok"


class EmbeddedServer:
    """The backend, running in a background thread of the GUI's process.

    Parameters
    ----------
    data_dir:
        Where results and models are kept -- the same folder the
        stand-alone server uses by default, so both see the same data.
    port:
        ``None`` picks a free port, which avoids clashing with anything
        else on the machine.
    """

    def __init__(self, data_dir: str | Path = "satrecovery_data", port: int | None = None,
                 host: str = "127.0.0.1", rml24_path: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir)
        self._rml24_path = rml24_path
        self._host = host
        self._port = int(port) if port else free_port(host)
        self._server = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, timeout: float = 30.0) -> "EmbeddedServer":
        """Start serving and wait until the first health check answers."""
        import uvicorn

        from ..backend.app import create_app
        from ..backend.settings import Settings

        if self.running:
            return self
        app = create_app(Settings(data_dir=self._data_dir, host=self._host, port=self._port,
                                  rml24_path=self._rml24_path))
        config = uvicorn.Config(app, host=self._host, port=self._port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="satrec-backend", daemon=True)
        self._thread.start()

        deadline = time.time() + timeout
        while not is_backend(self.url, timeout=0.5):
            if not self._thread.is_alive():
                raise BackendError(0, f"the backend failed to start on port {self._port}")
            if time.time() > deadline:
                raise BackendError(0, f"the backend did not answer within {timeout:.0f}s")
            time.sleep(0.1)
        return self

    def stop(self, timeout: float = 10.0) -> None:
        """Ask the server to finish and wait for its thread."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None

    def __enter__(self) -> "EmbeddedServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"EmbeddedServer({self.url}, {'running' if self.running else 'stopped'})"


def connect(url: str = "http://127.0.0.1:8000", data_dir: str | Path = "satrecovery_data",
            rml24_path: str | Path | None = None):
    """Use a running backend at ``url``, or start one. Returns (client, server or None)."""
    if is_backend(url):
        return HttpBackendClient(url), None
    server = EmbeddedServer(data_dir=data_dir, rml24_path=rml24_path).start()
    return HttpBackendClient(server.url), server


__all__ = ["EmbeddedServer", "connect", "free_port", "is_backend"]
