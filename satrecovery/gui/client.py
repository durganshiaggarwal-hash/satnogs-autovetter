"""How the GUI talks to the backend.

The GUI never imports a receiver, a simulator or a model.  Everything it
shows comes from the Phase 7 backend, through the small interface
defined here.  That keeps the two halves honest: the GUI cannot quietly
compute something differently from what the API would return, because
it has no way to compute anything at all.

DESIGN NOTE: PROGRAM TO AN INTERFACE
------------------------------------
Every page receives a :class:`BackendClient` and calls its methods.
Only one implementation exists for real use, :class:`HttpBackendClient`,
but the pages do not know or care which one they hold.  The tests take
advantage of exactly that: they hand pages a client pointed at a
throwaway server, and could hand them a fake that never touches the
network at all.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Any


class BackendError(Exception):
    """The backend refused a request, or could not be reached.

    Carries the HTTP status and the backend's own explanation, so the
    GUI can show the user the actual reason -- "SNR must be between -20
    and 30 dB" -- rather than a generic failure.
    """

    def __init__(self, status: int, detail: str, field: str | None = None) -> None:
        self.status = int(status)
        self.detail = str(detail)
        self.field = field
        super().__init__(f"{status}: {detail}")

    @property
    def unreachable(self) -> bool:
        """True when nothing answered at all, as opposed to a refusal."""
        return self.status == 0


class BackendClient(ABC):
    """Everything the GUI can ask of the backend."""

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Where the backend lives, for the status bar."""

    @abstractmethod
    def _request(self, method: str, path: str, body: dict | None = None) -> Any:
        """Send one request; return decoded JSON or raise BackendError."""

    # -- the API, one method per endpoint -------------------------------
    def health(self) -> dict:
        return self._request("GET", "/api/health")

    def modulations(self) -> list[dict]:
        return self._request("GET", "/api/modulations")

    def profiles(self) -> list[dict]:
        return self._request("GET", "/api/profiles")

    def dataset(self) -> dict:
        return self._request("GET", "/api/dataset")

    def receivers(self, modulation: str | None = None) -> list[dict]:
        query = f"?modulation={urllib.parse.quote(modulation)}" if modulation else ""
        return self._request("GET", f"/api/receivers{query}")

    def experiment(self, body: dict) -> dict:
        return self._request("POST", "/api/experiments", body)

    def stages(self, body: dict) -> dict:
        return self._request("POST", "/api/experiments/stages", body)

    def start_benchmark(self, body: dict) -> dict:
        return self._request("POST", "/api/benchmarks", body)

    def runs(self) -> list[dict]:
        return self._request("GET", "/api/runs")

    def cells(self, run_id: int, **filters: str) -> list[dict]:
        query = urllib.parse.urlencode({k: v for k, v in filters.items() if v})
        return self._request("GET", f"/api/runs/{int(run_id)}/cells" + (f"?{query}" if query else ""))

    def delete_run(self, run_id: int) -> None:
        self._request("DELETE", f"/api/runs/{int(run_id)}")

    def models(self) -> list[dict]:
        return self._request("GET", "/api/models")

    def train(self, modulation: str, body: dict | None = None) -> dict:
        return self._request("POST", f"/api/models/{urllib.parse.quote(modulation)}/train", body or {})

    def job(self, job_id: str) -> dict:
        return self._request("GET", f"/api/jobs/{urllib.parse.quote(job_id)}")


class HttpBackendClient(BackendClient):
    """Talks to a running backend over HTTP, using only the standard library.

    No extra dependency: ``urllib`` is enough for a client that sends a
    few JSON requests, and one less package is one less thing to install
    on another laptop.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = float(timeout)

    @property
    def base_url(self) -> str:
        return self._base

    def _request(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self._base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            raise self._explain(error) from None
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            reason = getattr(error, "reason", error)
            raise BackendError(0, f"cannot reach the backend at {self._base} ({reason})") from None

    @staticmethod
    def _explain(error: urllib.error.HTTPError) -> BackendError:
        """Turn an HTTP error body into a sentence a user can act on."""
        try:
            payload = json.loads(error.read() or b"{}")
        except ValueError:
            return BackendError(error.code, error.reason or "request failed")

        detail = payload.get("detail", error.reason)
        field = payload.get("field")
        if isinstance(detail, list) and detail:
            # FastAPI's own type errors: a list of {loc, msg}.
            first = detail[0]
            location = [str(p) for p in first.get("loc", []) if p not in ("body",)]
            field = location[-1] if location else None
            detail = f"{'.'.join(location)}: {first.get('msg', 'invalid')}" if location else first.get("msg")
        return BackendError(error.code, str(detail), field)

    def __repr__(self) -> str:
        return f"HttpBackendClient({self._base!r})"


__all__ = ["BackendClient", "BackendError", "HttpBackendClient"]
