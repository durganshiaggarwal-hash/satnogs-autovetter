"""Where the stars and the Sun are in a station's sky at a given moment.

Stars are fixed on the celestial sphere (right ascension, declination);
the Earth turning under them makes them rise and set.  For a station at
latitude phi and longitude lambda, at a time whose Greenwich sidereal
time is GMST, a star at (RA, Dec) has hour angle::

    H = GMST + lambda - RA

and is seen at::

    altitude = asin( sin(phi) sin(Dec) + cos(phi) cos(Dec) cos(H) )
    azimuth  = atan2( -sin(H) cos(Dec),  sin(Dec) cos(phi) - cos(Dec) sin(phi) cos(H) )

The Sun's RA and Dec come from the Astronomical Almanac's low-precision
formula (good to about 0.01 degrees, plenty for a sky chart).  Star
positions are the catalogue's (J2000); the 26 years of precession since
then move them by about a third of a degree, too little to see here.

Stars and constellation lines: d3-celestial by Olaf Frohn (BSD-3-Clause),
from the Hipparcos catalogue, stars to magnitude 6 (the naked-eye limit).
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

SKY_FILE = Path(__file__).resolve().parent / "gui" / "assets" / "sky.json"


def gmst_degrees(when: datetime) -> float:
    """Greenwich mean sidereal time, as an angle (skyfield's built-in timescale: no download)."""
    from .orbit import _timescale

    return float(_timescale().from_datetime(when).gmst) * 15.0


def altaz(ra_deg: np.ndarray, dec_deg: np.ndarray, lat: float, lon: float, when: datetime,
          gmst: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Altitude and azimuth (degrees; azimuth from north through east) of sky positions."""
    gmst = gmst_degrees(when) if gmst is None else gmst
    hour = np.radians(gmst + lon - np.asarray(ra_deg, float))
    dec, phi = np.radians(np.asarray(dec_deg, float)), math.radians(lat)
    alt = np.arcsin(np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(hour))
    az = np.arctan2(-np.sin(hour) * np.cos(dec), np.sin(dec) * np.cos(phi) - np.cos(dec) * np.sin(phi) * np.cos(hour))
    return np.degrees(alt), np.degrees(az) % 360.0


def sun_radec(when: datetime) -> tuple[float, float]:
    """The Sun's right ascension and declination (degrees), low-precision almanac formula."""
    j2000 = datetime(2000, 1, 1, 12, tzinfo=when.tzinfo)
    n = (when - j2000).total_seconds() / 86400.0
    mean_longitude = (280.460 + 0.9856474 * n) % 360
    anomaly = math.radians((357.528 + 0.9856003 * n) % 360)
    ecliptic = math.radians(mean_longitude + 1.915 * math.sin(anomaly) + 0.020 * math.sin(2 * anomaly))
    tilt = math.radians(23.439 - 0.0000004 * n)
    ra = math.degrees(math.atan2(math.cos(tilt) * math.sin(ecliptic), math.cos(ecliptic))) % 360
    dec = math.degrees(math.asin(math.sin(tilt) * math.sin(ecliptic)))
    return ra, dec


def sky_state(sun_altitude: float) -> str:
    if sun_altitude > 0:
        return "day"
    if sun_altitude > -6:
        return "twilight"
    if sun_altitude > -12:
        return "dusk"
    return "night"


class StarCatalogue:
    """The naked-eye stars, the constellation lines and names."""

    def __init__(self, path: Path = SKY_FILE) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {"stars": [], "lines": [], "names": []}
        stars = np.asarray(data.get("stars", []), dtype=float).reshape(-1, 3)
        self.ra, self.dec, self.mag = stars[:, 0], stars[:, 1], stars[:, 2]
        self.lines = [np.asarray(line, dtype=float) for line in data.get("lines", [])]
        self.names = [(name, float(ra), float(dec)) for name, ra, dec in data.get("names", [])]

    def __len__(self) -> int:
        return len(self.mag)
