"""Where the backend keeps things, and the limits it enforces.

One object, built once when the server starts and handed to everything
that needs it.  Nothing in the backend reads an environment variable or
hard-codes a path anywhere else, so a test can point the whole server at
a temporary folder by building a different ``Settings``.

The limits exist because the API runs real computation on request.  A
benchmark of every modulation at every noise level with a thousand
frames each would occupy the machine for hours; a request for that is
almost certainly a mistake, and refusing it with a clear message is
kinder than accepting it.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..core.exceptions import ValidationError


class Settings:
    """Server configuration.

    Parameters
    ----------
    data_dir:
        Root folder for everything the server writes.  The results
        database and trained models live inside it.
    host:
        Interface to listen on.  ``127.0.0.1`` by default, which means
        only programs on the same computer can connect -- the right
        default for a desktop application talking to its own backend.
        Opening it to a network is a deliberate choice, not an accident.
    port:
        TCP port.
    max_frames_per_scenario, max_benchmark_cells:
        Upper bounds on what one benchmark request may ask for.
    rml24_path:
        The RML24 dataset (``RML24_IQdata.h5``, its folder, or a sample
        ``.npz``), or ``None``.  When given, benchmarks can use the
        ``recorded`` profile: real recordings instead of the simulator.
    """

    __slots__ = (
        "_data_dir",
        "_host",
        "_port",
        "_max_frames",
        "_max_cells",
        "_rml24_path",
    )

    def __init__(
        self,
        data_dir: str | Path = "satrecovery_data",
        host: str = "127.0.0.1",
        port: int = 8000,
        max_frames_per_scenario: int = 200,
        max_benchmark_cells: int = 2000,
        rml24_path: str | Path | None = None,
    ) -> None:
        port = int(port)
        if not 1 <= port <= 65535:
            raise ValidationError("port", port, "must be between 1 and 65535")
        for name, value in (
            ("max_frames_per_scenario", max_frames_per_scenario),
            ("max_benchmark_cells", max_benchmark_cells),
        ):
            if int(value) < 1:
                raise ValidationError(name, value, "must be at least 1")

        self._data_dir = Path(data_dir).expanduser().resolve()
        self._host = str(host)
        self._port = port
        self._max_frames = int(max_frames_per_scenario)
        self._max_cells = int(max_benchmark_cells)
        self._rml24_path = None if not rml24_path else Path(rml24_path).expanduser().resolve()

    @classmethod
    def from_env(cls) -> "Settings":
        """Read overrides from ``SATREC_*`` environment variables.

        Unset variables keep their defaults, so running the server with
        no configuration at all works.
        """
        return cls(
            data_dir=os.environ.get("SATREC_DATA_DIR", "satrecovery_data"),
            host=os.environ.get("SATREC_HOST", "127.0.0.1"),
            port=int(os.environ.get("SATREC_PORT", "8000")),
            rml24_path=os.environ.get("SATREC_RML24") or None,
        )

    # ------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def database_path(self) -> Path:
        """The SQLite file of benchmark results."""
        return self._data_dir / "results.db"

    @property
    def models_dir(self) -> Path:
        """Folder of trained demappers, one set of files per modulation."""
        return self._data_dir / "models"

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def rml24_path(self) -> Path | None:
        """The RML24 dataset, if one was configured."""
        return self._rml24_path

    @property
    def max_frames_per_scenario(self) -> int:
        return self._max_frames

    @property
    def max_benchmark_cells(self) -> int:
        return self._max_cells

    def ensure_directories(self) -> None:
        """Create the data and model folders if they are missing."""
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"Settings(data_dir={str(self._data_dir)!r}, {self._host}:{self._port})"


__all__ = ["Settings"]
