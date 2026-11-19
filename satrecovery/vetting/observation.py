"""One SatNOGS observation, as an immutable value object.

A SatNOGS observation is one ground station listening to one satellite
pass.  The network keeps, for each one, a *waterfall* (a picture of the
spectrum over the pass), an *audio* file and any frames the station
decoded.  Volunteers then look at the waterfall and mark it
"with signal" or "without signal".  That human mark is the label this
project learns from.

DESIGN NOTE: ENCAPSULATION
--------------------------
:class:`Observation` is frozen: once built from the API's JSON it cannot
change, so a label can never be edited by accident between download and
training.  The awkward parts of the raw JSON (string dates, ``None``
for "not vetted", the TLE split over two fields) are dealt with once, in
:meth:`Observation.from_api`, and never seen again.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..core.exceptions import ValidationError


class WaterfallLabel(Enum):
    """What a volunteer said about an observation's waterfall."""

    SIGNAL = "with-signal"
    NO_SIGNAL = "without-signal"
    UNKNOWN = "unknown"

    @property
    def api_filter(self) -> str | None:
        """The value the API's ``waterfall_status`` filter expects.

        The API *returns* ``"with-signal"``/``"without-signal"`` but only
        *accepts* ``1``/``0`` as a filter (checked against the live API,
        November 2026).  There is no filter for unknown.
        """
        return {WaterfallLabel.SIGNAL: "1", WaterfallLabel.NO_SIGNAL: "0"}.get(self)

    @classmethod
    def parse(cls, value: str | None) -> "WaterfallLabel":
        try:
            return cls(value) if value else cls.UNKNOWN
        except ValueError:
            return cls.UNKNOWN


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(frozen=True)
class Observation:
    """Everything this project needs about one observation."""

    id: int
    norad_cat_id: int | None
    ground_station: int | None
    station_name: str
    start: datetime | None
    end: datetime | None
    status: str
    waterfall_label: WaterfallLabel
    labelled_by_human: bool
    transmitter_mode: str
    transmitter_baud: float | None
    frequency_hz: int | None
    n_decoded_frames: int
    max_altitude_deg: float | None
    waterfall_url: str | None
    audio_url: str | None
    tle: tuple[str, str] | None = field(default=None, compare=False)
    # Where the station is (degrees, metres).  Missing in manifests written
    # before November 2026; StationDirectory fills them in.
    station_lat: float | None = None
    station_lng: float | None = None
    station_alt: float | None = None

    def __post_init__(self) -> None:
        if self.id <= 0:
            raise ValidationError("id", self.id, "observation ids are positive")

    # ------------------------------------------------------------------
    @classmethod
    def from_api(cls, row: dict[str, Any]) -> "Observation":
        """Build from one row of ``/api/observations/``."""
        if "id" not in row:
            raise ValidationError("row", sorted(row)[:5], "not an observation: no 'id'")
        tle = (row["tle1"], row["tle2"]) if row.get("tle1") and row.get("tle2") else None
        baud = row.get("transmitter_baud")
        return cls(
            id=int(row["id"]),
            norad_cat_id=row.get("norad_cat_id"),
            ground_station=row.get("ground_station"),
            station_name=row.get("station_name") or "",
            start=_parse_time(row.get("start")),
            end=_parse_time(row.get("end")),
            status=row.get("status") or "unknown",
            waterfall_label=WaterfallLabel.parse(row.get("waterfall_status")),
            labelled_by_human=row.get("waterfall_status_user") is not None,
            transmitter_mode=row.get("transmitter_mode") or "",
            transmitter_baud=float(baud) if baud is not None else None,
            frequency_hz=row.get("observation_frequency"),
            n_decoded_frames=len(row.get("demoddata") or []),
            max_altitude_deg=row.get("max_altitude"),
            waterfall_url=row.get("waterfall") or None,
            audio_url=row.get("payload") or None,
            tle=tle,
            station_lat=row.get("station_lat"),
            station_lng=row.get("station_lng"),
            station_alt=row.get("station_alt"),
        )

    # ------------------------------------------------------------------
    @property
    def duration_s(self) -> float | None:
        if self.start is None or self.end is None:
            return None
        return (self.end - self.start).total_seconds()

    @property
    def is_trainable(self) -> bool:
        """A human label and a waterfall to learn it from."""
        return (
            self.labelled_by_human
            and self.waterfall_label is not WaterfallLabel.UNKNOWN
            and self.waterfall_url is not None
        )

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Plain JSON, for the manifest file."""
        data = asdict(self)
        data["start"] = self.start.isoformat() if self.start else None
        data["end"] = self.end.isoformat() if self.end else None
        data["waterfall_label"] = self.waterfall_label.value
        data["tle"] = list(self.tle) if self.tle else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        data = dict(data)
        data["start"] = _parse_time(data.get("start"))
        data["end"] = _parse_time(data.get("end"))
        data["waterfall_label"] = WaterfallLabel.parse(data.get("waterfall_label"))
        data["tle"] = tuple(data["tle"]) if data.get("tle") else None
        return cls(**data)

    def __repr__(self) -> str:
        who = "human" if self.labelled_by_human else "unvetted"
        return (f"Observation({self.id}, sat {self.norad_cat_id}, {self.transmitter_mode or '?'}, "
                f"{self.waterfall_label.value} ({who}))")
