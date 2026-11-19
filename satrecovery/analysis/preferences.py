"""Small things the app remembers between runs.

:class:`Preferences` keeps them in one small JSON file in the user's
home folder: which RML24 folder to open, and whether the window is dark.
A missing or damaged file just means "nothing remembered yet"; the app
never fails because of it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class Preferences:
    """Remembered settings, saved the moment they change."""

    FILE_NAME = ".satellite_signal_analyser.json"

    def __init__(self, path: str | Path | None = None) -> None:
        default = os.environ.get("SATREC_PREFERENCES") or (Path.home() / self.FILE_NAME)
        self._path = Path(path) if path else Path(default)
        self._values = self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> dict[str, Any]:
        try:
            values = json.loads(self._path.read_text(encoding="utf-8"))
            return values if isinstance(values, dict) else {}
        except (OSError, ValueError):
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def set(self, key: str, value: Any) -> bool:
        """Remember ``value``; returns False if it could not be written to disk."""
        self._values[key] = value
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._values, indent=2), encoding="utf-8")
            return True
        except OSError:
            return False

    # -- the two things remembered ------------------------------------------
    @property
    def rml24_path(self) -> str | None:
        value = self.get("rml24_path")
        return str(value) if value else None

    @rml24_path.setter
    def rml24_path(self, value: str | Path | None) -> None:
        self.set("rml24_path", str(value) if value else None)

    @property
    def dark_mode(self) -> bool:
        return bool(self.get("dark_mode", False))

    @dark_mode.setter
    def dark_mode(self, on: bool) -> None:
        self.set("dark_mode", bool(on))


__all__ = ["Preferences"]
