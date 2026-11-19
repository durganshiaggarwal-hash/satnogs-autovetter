"""Where observations come from: the live SatNOGS API, or a saved manifest.

DESIGN NOTE: ABSTRACTION + POLYMORPHISM
---------------------------------------
:class:`ObservationSource` says only "give me observations matching this
query".  :class:`SatnogsApiSource` answers from the internet;
:class:`ManifestSource` answers from a file on disk, so training and the
tests never need a network.  Code that builds a dataset takes any
``ObservationSource`` and cannot tell which it got.

DESIGN NOTE: DEPENDENCY INJECTION
---------------------------------
:class:`SatnogsApiSource` does not open connections itself; it is handed
a :class:`Transport`.  The real one uses ``urllib``; the tests hand it a
fake that replays recorded pages, which is how pagination, retries and
rate limiting are tested without touching SatNOGS.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.exceptions import SatRecoveryError, ValidationError
from .observation import Observation, WaterfallLabel

USER_AGENT = "satrecovery-vetting/0.1 (research; +https://network.satnogs.org)"


class SatnogsError(SatRecoveryError):
    """The SatNOGS API could not be read."""


# ----------------------------------------------------------------------
# Query
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ObservationQuery:
    """What to ask for.  Every field is optional; ``None`` means "any"."""

    waterfall_label: WaterfallLabel | None = None
    status: str | None = None              # good / bad / failed / future
    satellite: int | None = None           # NORAD id
    ground_station: int | None = None
    start: str | None = None               # ISO date: observations starting after
    end: str | None = None                 # ISO date: observations ending before

    def __post_init__(self) -> None:
        if self.waterfall_label is WaterfallLabel.UNKNOWN:
            raise ValidationError("waterfall_label", "unknown", "the API cannot filter for unvetted waterfalls")

    def to_params(self) -> dict[str, str]:
        params: dict[str, str] = {"format": "json"}
        if self.waterfall_label is not None:
            params["waterfall_status"] = self.waterfall_label.api_filter or ""
        if self.status:
            params["status"] = self.status
        if self.satellite is not None:
            params["norad_cat_id"] = str(self.satellite)  # checked live; "satellite=" is ignored
        if self.ground_station is not None:
            params["ground_station"] = str(self.ground_station)
        if self.start:
            params["start"] = self.start
        if self.end:
            params["end"] = self.end
        return params


# ----------------------------------------------------------------------
# Transport
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Response:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


class Transport(ABC):
    """Fetches one URL.  Swapped for a fake in the tests."""

    @abstractmethod
    def get(self, url: str, timeout: float = 60.0) -> Response:
        """Return the response; never raise for an HTTP error status."""


class UrllibTransport(Transport):
    def get(self, url: str, timeout: float = 60.0) -> Response:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as reply:  # noqa: S310 - fixed https hosts
                return Response(reply.status, {k.lower(): v for k, v in reply.headers.items()}, reply.read())
        except urllib.error.HTTPError as error:
            return Response(error.code, {k.lower(): v for k, v in error.headers.items()}, error.read() or b"")
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise SatnogsError(f"could not reach {urllib.parse.urlsplit(url).netloc}: {error}") from error


_NEXT = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


def next_link(headers: dict[str, str]) -> str | None:
    """The ``rel="next"`` URL of a ``Link`` header (the API pages by cursor)."""
    match = _NEXT.search(headers.get("link", ""))
    return match.group(1) if match else None


# ----------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------
class ObservationSource(ABC):
    """Anything that can list observations."""

    @abstractmethod
    def fetch(self, query: ObservationQuery, limit: int | None = None) -> Iterator[Observation]:
        """Observations matching ``query``, newest first, at most ``limit``."""


class SatnogsApiSource(ObservationSource):
    """The public SatNOGS Network API.

    It is a volunteer service, so this source is deliberately slow:
    at most one request every ``min_interval`` seconds, and it backs off
    when the server says it is busy (429/5xx).
    """

    BASE_URL = "https://network.satnogs.org/api/observations/"

    def __init__(self, transport: Transport | None = None, min_interval: float = 1.0,
                 max_retries: int = 4, sleep=time.sleep, clock=time.monotonic) -> None:
        if min_interval < 0:
            raise ValidationError("min_interval", min_interval, "must not be negative")
        self._transport = transport or UrllibTransport()
        self._min_interval = float(min_interval)
        self._max_retries = int(max_retries)
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self.requests_made = 0

    def _get(self, url: str) -> Response:
        for attempt in range(self._max_retries + 1):
            if self._last_request is not None:
                wait = self._min_interval - (self._clock() - self._last_request)
                if wait > 0:
                    self._sleep(wait)
            self._last_request = self._clock()
            self.requests_made += 1
            try:
                response = self._transport.get(url)
            except SatnogsError:
                # A timeout or dropped connection: wait and try again, like a busy server.
                if attempt == self._max_retries:
                    raise
                self._sleep(min(2.0 * 2 ** attempt, 120.0))
                continue
            if response.status == 200:
                return response
            if response.status == 429 or response.status >= 500:
                retry_after = response.headers.get("retry-after", "")
                delay = float(retry_after) if retry_after.isdigit() else 2.0 * 2 ** attempt
                self._sleep(min(delay, 120.0))
                continue
            raise SatnogsError(f"SatNOGS answered {response.status}: {response.body[:200]!r}")
        raise SatnogsError(f"SatNOGS stayed busy after {self._max_retries + 1} tries")

    def fetch(self, query: ObservationQuery, limit: int | None = None) -> Iterator[Observation]:
        url: str | None = f"{self.BASE_URL}?{urllib.parse.urlencode(query.to_params())}"
        given = 0
        while url and (limit is None or given < limit):
            response = self._get(url)
            rows = response.json()
            if not isinstance(rows, list):
                raise SatnogsError(f"unexpected answer from SatNOGS: {str(rows)[:200]}")
            for row in rows:
                yield Observation.from_api(row)
                given += 1
                if limit is not None and given >= limit:
                    return
            url = next_link(response.headers)


class ManifestSource(ObservationSource):
    """Observations saved earlier in a :class:`Manifest` file."""

    def __init__(self, manifest: "Manifest") -> None:
        self._manifest = manifest

    def fetch(self, query: ObservationQuery, limit: int | None = None) -> Iterator[Observation]:
        given = 0
        for obs in self._manifest:
            if limit is not None and given >= limit:
                return
            if query.waterfall_label is not None and obs.waterfall_label is not query.waterfall_label:
                continue
            if query.status and obs.status != query.status:
                continue
            if query.satellite is not None and obs.norad_cat_id != query.satellite:
                continue
            if query.ground_station is not None and obs.ground_station != query.ground_station:
                continue
            yield obs
            given += 1


# ----------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------
class Manifest:
    """A JSON-lines file of observations, one per line, no duplicates.

    Appending is safe to interrupt: a download can be stopped and
    restarted, and observations already saved are skipped.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._items: dict[int, Observation] = {}
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        obs = Observation.from_dict(json.loads(line))
                    except (ValueError, TypeError, ValidationError):
                        continue  # a half-written last line from an interrupted run
                    self._items[obs.id] = obs

    @property
    def path(self) -> Path:
        return self._path

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, obs_id: object) -> bool:
        return obs_id in self._items

    def __iter__(self) -> Iterator[Observation]:
        return iter(list(self._items.values()))

    def add(self, obs: Observation) -> bool:
        """Save ``obs``; ``False`` if it was already there."""
        if obs.id in self._items:
            return False
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(obs.to_dict()) + "\n")
        self._items[obs.id] = obs
        return True

    def counts(self) -> dict[str, int]:
        found: dict[str, int] = {}
        for obs in self._items.values():
            found[obs.waterfall_label.value] = found.get(obs.waterfall_label.value, 0) + 1
        return found
