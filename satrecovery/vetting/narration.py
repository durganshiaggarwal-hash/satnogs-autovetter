"""Subtitles for a pass replay: what is happening, in plain sentences.

:class:`PassNarrator` reads the same facts the replay draws -- where the
satellite is, how high above the station, how far, whether the recording
shows a signal at that moment, and what the orbit check says made it --
and turns each moment into one or two sentences::

    "The satellite is rising over the horizon, 2,300 km away over Iceland.
     The antenna starts listening."

:class:`Places` names the country under the satellite (Natural Earth
outlines), or the sea it is over.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .orbit import GroundTrack, TrackExplanation

COUNTRIES_FILE = Path(__file__).resolve().parent / "gui" / "assets" / "countries_110m.json"

#: Names that read badly after "over" -> what to say instead.
NICER = {"United States of America": "the United States", "United Kingdom": "the United Kingdom",
         "Russia": "Russia", "Dem. Rep. Congo": "the Congo", "Central African Rep.": "the Central African Republic",
         "Dominican Rep.": "the Dominican Republic", "Czechia": "the Czech Republic", "Netherlands": "the Netherlands",
         "Philippines": "the Philippines", "W. Sahara": "Western Sahara", "Fr. S. Antarctic Lands": "the French "
         "Southern Lands", "Bosnia and Herz.": "Bosnia", "S. Sudan": "South Sudan", "Solomon Is.": "the Solomon Islands",
         "Falkland Is.": "the Falkland Islands", "Antarctica": "Antarctica"}


class Places:
    """Which country (or roughly which sea) a point is over."""

    def __init__(self, path: Path = COUNTRIES_FILE) -> None:
        from matplotlib.path import Path as Shape

        self._shapes: list[tuple[str, object, tuple[float, float, float, float]]] = []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))["countries"]
        except (OSError, ValueError, KeyError):
            data = []
        for country in data:
            for ring in country["rings"]:
                points = np.asarray(ring, dtype=float)
                box = (points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max())
                self._shapes.append((country["name"], Shape(points), box))

    def name(self, lat: float, lon: float) -> str:
        """'Iceland', 'the United States', or 'the North Atlantic' and so on."""
        for name, shape, (x0, x1, y0, y1) in self._shapes:
            if x0 <= lon <= x1 and y0 <= lat <= y1 and shape.contains_point((lon, lat)):
                return NICER.get(name, name)
        return sea(lat, lon)


def sea(lat: float, lon: float) -> str:
    """A rough name for the water at a point: good enough for a subtitle, not for a chart."""
    lon = (lon + 180.0) % 360.0 - 180.0
    if lat > 66:
        return "the Arctic Ocean"
    if lat < -55:
        return "the Southern Ocean"
    if 30 < lat < 46 and -6 < lon < 36:
        return "the Mediterranean"
    if -70 < lon < 20 or (lat > 0 and -100 < lon <= -70):
        return "the North Atlantic" if lat >= 0 else "the South Atlantic"
    if 20 <= lon < 120 and lat < 25:
        return "the Indian Ocean"
    return "the North Pacific" if lat >= 0 else "the South Pacific"


def _km(value: float) -> str:
    return f"{value:,.0f} km"


class PassNarrator:
    """One or two sentences for any moment of the replay."""

    def __init__(self, track: GroundTrack, satellite: str, places: Places | None = None,
                 activity=None, explanation: TrackExplanation | None = None) -> None:
        """``activity(seconds) -> 0..1``: how strong the recording's signal is at that moment."""
        self._track = track
        self._satellite = satellite
        self._activity = activity or (lambda seconds: 0.0)
        self._explanation = explanation
        places = places or Places()
        # Work out the place under the satellite once per sample, not per frame.
        self._places = [places.name(float(a), float(b)) for a, b in zip(track.lat, track.lon)]
        up = np.flatnonzero(track.elevation > 0)
        self._rise = int(up[0]) if len(up) else None
        self._set = int(up[-1]) if len(up) else None
        self._top = int(np.argmax(track.elevation)) if len(up) else None

    def set_explanation(self, explanation: TrackExplanation | None) -> None:
        self._explanation = explanation

    def line(self, seconds: float) -> str:
        track = self._track
        i = int(np.clip(np.searchsorted(track.seconds, seconds), 0, len(track.seconds) - 1))
        place = self._places[i]
        elevation = float(track.elevation[i])
        height = float(track.height_km[i])
        distance = float(track.range_km[i]) if track.range_km is not None else None
        name = self._satellite
        if self._rise is None:
            return f"{name} never rises above this station's horizon during the recording. It is over {place}."
        if i < self._rise:
            return f"{name} is still below the station's horizon, flying over {place}, {_km(height)} up."
        if i > self._set:
            return f"{name} has set below the horizon. The pass is over; it is now above {place}."

        span = max(self._set - self._rise, 1)
        where = (i - self._rise) / span
        away = f", {_km(distance)} away" if distance is not None else ""
        if where < 0.12:
            listening = " The antenna starts listening." if seconds >= 0 else " The recording hasn't started yet."
            motion = f"{name} is rising over the horizon{away}, above {place}.{listening}"
        elif abs(i - self._top) <= max(span * 0.06, 1):
            overhead = ", almost straight overhead" if elevation > 70 else ""
            motion = (f"Closest point of the pass: {elevation:.0f}° up{overhead}, {_km(distance or height)} from the "
                      f"station, over {place}.")
        elif where > 0.88:
            motion = f"It is sinking towards the horizon{away}, over {place}. The pass is nearly over."
        elif i < self._top:
            motion = f"Climbing: {elevation:.0f}° above the horizon, {_km(height)} up, over {place}."
        else:
            motion = f"Going down: {elevation:.0f}° above the horizon, now over {place}."
        return f"{motion} {self._signal(seconds)}".strip()

    def _signal(self, seconds: float) -> str:
        loud = self._activity(seconds) > 0.35
        source = self._explanation.source if self._explanation is not None else None
        best = (self._explanation.matches[0].curve.name.split(" (")[0]
                if self._explanation is not None and self._explanation.matches else "")
        if seconds < 0:
            return ""
        if not loud:
            if source == "ground":
                return "No signal at the centre: the satellite itself is quiet."
            return "Only noise in the recording right now."
        if source == "ground":
            return "The bright line isn't the satellite: it's a fixed transmitter on the ground."
        if source == "other":
            return f"That bright line is {best}, passing at the same time, not {self._satellite}."
        if source == "timing":
            return "Its signal is there, but drifting: the satellite is a little off schedule."
        if source is None:
            return "Something bright is in the recording; the physics check is still working out what made it."
        if source == "unknown":
            return "Something bright is in the recording, but no known satellite or transmitter explains it."
        return f"The station is hearing {self._satellite}: that's the bright line in the middle."
