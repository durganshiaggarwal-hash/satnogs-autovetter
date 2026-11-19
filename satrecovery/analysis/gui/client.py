"""The app's connection to the backend, with the analysis endpoints added.

:class:`AnalysisClient` *inherits* everything the Phase 8 client can
already do and adds one method per new endpoint -- inheritance used for
exactly what it is for, extending without editing.
"""

from __future__ import annotations

import base64
import urllib.parse
from pathlib import Path

from ...gui.client import HttpBackendClient


class AnalysisClient(HttpBackendClient):
    def analysis_status(self) -> dict:
        return self._request("GET", "/api/analysis/status")

    def pick_recordings(self, modulation: str | None = None, snr_db: float | None = None,
                        symbol_rate: float | None = None, count: int = 1, seed: int | None = None) -> list[dict]:
        query = {k: v for k, v in (("modulation", modulation), ("snr_db", snr_db),
                                   ("symbol_rate", symbol_rate), ("count", count), ("seed", seed)) if v is not None}
        return self._request("GET", "/api/analysis/recordings?" + urllib.parse.urlencode(query))

    def analyse_recording(self, row: int) -> dict:
        return self._request("POST", "/api/analysis/recording", {"row": int(row)})

    def evaluate_classifier(self, per_group: int = 6, seed: int = 0) -> dict:
        return self._request("POST", "/api/analysis/evaluate", {"per_group": int(per_group), "seed": int(seed)})

    def analyse_simulated(self, body: dict) -> dict:
        return self._request("POST", "/api/analysis/simulated", body)

    # -- history ---------------------------------------------------------
    def history(self, query: str = "", limit: int = 200) -> list[dict]:
        return self._request("GET", "/api/analysis/history?" + urllib.parse.urlencode({"q": query, "limit": limit}))

    def history_entry(self, entry: int) -> dict:
        return self._request("GET", f"/api/analysis/history/{int(entry)}")

    def clear_history(self) -> dict:
        return self._request("DELETE", "/api/analysis/history")

    # -- dataset and many-at-once ------------------------------------------
    def open_dataset(self, path: str | Path) -> dict:
        return self._request("POST", "/api/analysis/dataset", {"path": str(path)})

    def batch_rows(self, modulation: str | None, snr_db: float | None, symbol_rate: float | None,
                   count: int, seed: int | None = None) -> list[int]:
        query = {k: v for k, v in (("modulation", modulation), ("snr_db", snr_db), ("symbol_rate", symbol_rate),
                                   ("count", count), ("seed", seed)) if v is not None}
        return self._request("GET", "/api/analysis/batch-rows?" + urllib.parse.urlencode(query))

    def quick(self, row: int) -> dict:
        return self._request("POST", "/api/analysis/quick", {"row": int(row)})

    def make_test_file(self, source: dict) -> dict:
        """A test file of an analysed signal: ``{"name", "content_base64", "info"}``."""
        return self._request("POST", "/api/analysis/test-file", {"input": source})

    def analyse_file(self, path: str | Path, symbol_rate: float | None, bits_path: str | Path | None = None) -> dict:
        path = Path(path)
        body = {"name": path.name, "content_base64": base64.b64encode(path.read_bytes()).decode("ascii")}
        if bits_path is not None:
            bits_path = Path(bits_path)
            body["bits_name"] = bits_path.name
            body["bits_base64"] = base64.b64encode(bits_path.read_bytes()).decode("ascii")
        if symbol_rate is not None:
            body["symbol_rate"] = float(symbol_rate)
        return self._request("POST", "/api/analysis/file", body)

    # -- comparing models ----------------------------------------------------
    def models(self) -> dict:
        """The compared models, their results, and which one the app uses."""
        return self._request("GET", "/api/analysis/models")

    def use_model(self, name: str) -> dict:
        return self._request("POST", "/api/analysis/models/use", {"model": str(name)})


__all__ = ["AnalysisClient"]
