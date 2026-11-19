"""Where each satellite's signal *should* appear on a waterfall.

The physics
-----------
A satellite moving towards the station is heard slightly higher in
frequency, moving away slightly lower: the Doppler shift,
``f_heard = f_sent * (1 - v_r / c)`` with ``v_r`` the range rate.  For a
low satellite at 435 MHz it sweeps through about +-10 kHz in a pass.

The station removes the shift *for the scheduled satellite*: it retunes
all through the pass using that satellite's orbit (its TLE).  So the
scheduled satellite sits on the centre line, and every other satellite
transmitting nearby is left with the *difference* between the two
shifts.  On the waterfall that difference is the S-shaped curve::

    offset(t) = f_other * (1 - v_other(t)/c)  -  f_tuned * (1 - v_scheduled(t)/c)

Given the orbits (SatNOGS DB), the transmitter frequencies (SatNOGS DB)
and the station's position, every term is known, so the curve of every
satellite in view can be drawn before looking at the picture -- and the
bright track in the picture can be matched to one of them.

DESIGN NOTE: SINGLE RESPONSIBILITY
----------------------------------
:class:`SatnogsCatalog` only fetches and caches the catalogue,
:class:`StationDirectory` only knows where stations are,
:func:`doppler` only does the physics, and :class:`PassPrediction` only
combines them for one observation.  Each can be tested alone.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from ..core.exceptions import SatRecoveryError, ValidationError
from .axes import WaterfallAxes
from .observation import Observation
from .source import SatnogsError, Transport, UrllibTransport

C = 299_792_458.0  # m/s


class OrbitError(SatRecoveryError):
    """An orbit or catalogue entry could not be used."""


@dataclass(frozen=True)
class Site:
    lat: float
    lng: float
    alt_m: float = 0.0


@dataclass(frozen=True)
class Transmitter:
    norad: int
    frequency_hz: float
    mode: str
    description: str


# ----------------------------------------------------------------------
class _CachedJson:
    """A JSON download kept on disk and refreshed when older than ``max_age``."""

    def __init__(self, url: str, path: Path, transport: Transport, max_age: float) -> None:
        self._url, self._path, self._transport, self._max_age = url, path, transport, max_age

    def get(self) -> list:
        fresh = self._path.exists() and time.time() - self._path.stat().st_mtime < self._max_age
        if not fresh:
            try:
                response = self._transport.get(self._url, timeout=120.0)
                if response.status == 200:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    self._path.write_bytes(response.body)
            except SatnogsError:
                pass  # fall back to an older copy below
        if not self._path.exists():
            raise OrbitError(f"could not download {self._url} and no saved copy exists")
        return json.loads(self._path.read_text(encoding="utf-8"))


class SatnogsCatalog:
    """Every satellite's current orbit and every transmitter, from SatNOGS DB."""

    TLE_URL = "https://db.satnogs.org/api/tle/?format=json"
    TRANSMITTER_URL = "https://db.satnogs.org/api/transmitters/?format=json"

    def __init__(self, root: str | Path, transport: Transport | None = None, max_age: float = 86400.0) -> None:
        root = Path(root)
        transport = transport or UrllibTransport()
        self._tle_file = _CachedJson(self.TLE_URL, root / "catalog" / "tle.json", transport, max_age)
        self._tx_file = _CachedJson(self.TRANSMITTER_URL, root / "catalog" / "transmitters.json", transport, max_age)
        self._tles: dict[int, tuple[str, str, str]] | None = None
        self._transmitters: list[Transmitter] | None = None

    def tles(self) -> dict[int, tuple[str, str, str]]:
        """NORAD id -> (name, line 1, line 2)."""
        if self._tles is None:
            found = {}
            for row in self._tle_file.get():
                try:
                    found[int(row["norad_cat_id"])] = (str(row.get("tle0", "")).removeprefix("0 ").strip(),
                                                       row["tle1"], row["tle2"])
                except (KeyError, TypeError, ValueError):
                    continue
            self._tles = found
        return self._tles

    def transmitters(self) -> list[Transmitter]:
        """Transmitters that are alive and active and have a downlink frequency."""
        if self._transmitters is None:
            found = []
            for row in self._tx_file.get():
                try:
                    if not row.get("alive") or row.get("status") != "active" or row.get("type") == "Transceiver" and \
                            row.get("downlink_low") is None:
                        continue
                    if row.get("downlink_low") is None or row.get("norad_cat_id") is None:
                        continue
                    found.append(Transmitter(int(row["norad_cat_id"]), float(row["downlink_low"]),
                                             str(row.get("mode") or ""), str(row.get("description") or "")))
                except (TypeError, ValueError):
                    continue
            self._transmitters = found
        return self._transmitters

    def name(self, norad: int) -> str:
        entry = self.tles().get(norad)
        return entry[0] if entry and entry[0] else f"NORAD {norad}"


class StationDirectory:
    """Ground-station positions, asked of the SatNOGS Network API once each and kept."""

    URL = "https://network.satnogs.org/api/stations/?format=json&id={}"

    def __init__(self, root: str | Path, transport: Transport | None = None) -> None:
        self._path = Path(root) / "catalog" / "stations.json"
        self._transport = transport or UrllibTransport()
        try:
            self._known = {int(k): v for k, v in json.loads(self._path.read_text(encoding="utf-8")).items()}
        except (OSError, ValueError):
            self._known = {}

    def site(self, obs: Observation) -> Site:
        if obs.station_lat is not None and obs.station_lng is not None:
            return Site(float(obs.station_lat), float(obs.station_lng), float(obs.station_alt or 0.0))
        if obs.ground_station is None:
            raise OrbitError(f"observation {obs.id} has no ground station")
        if obs.ground_station not in self._known:
            response = self._transport.get(self.URL.format(obs.ground_station))
            rows = response.json() if response.status == 200 else []
            if not rows:
                raise OrbitError(f"station {obs.ground_station} not found")
            row = rows[0]
            self._known[obs.ground_station] = {"lat": row["lat"], "lng": row["lng"], "alt": row.get("altitude") or 0}
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._known), encoding="utf-8")
        entry = self._known[obs.ground_station]
        return Site(float(entry["lat"]), float(entry["lng"]), float(entry["alt"]))


# ----------------------------------------------------------------------
_TIMESCALE = None


def _timescale():
    global _TIMESCALE
    if _TIMESCALE is None:
        from skyfield.api import load

        _TIMESCALE = load.timescale(builtin=True)  # no download needed
    return _TIMESCALE


def look(tle: tuple[str, str], site: Site, times: Sequence[datetime]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(range rate m/s, elevation deg, azimuth deg)`` of a satellite seen from ``site`` at ``times``."""
    from skyfield.api import EarthSatellite, wgs84

    ts = _timescale()
    try:
        satellite = EarthSatellite(tle[0], tle[1], ts=ts)
    except (ValueError, IndexError) as error:
        raise OrbitError(f"bad TLE: {error}") from error
    place = wgs84.latlon(site.lat, site.lng, elevation_m=site.alt_m)
    t = ts.from_datetimes(list(times))
    seen = (satellite - place).at(t)
    elevation, azimuth, _ = seen.altaz()
    *_, range_rate = seen.frame_latlon_and_rates(place)
    rate = np.atleast_1d(range_rate.km_per_s * 1000.0)
    if not np.all(np.isfinite(rate)):
        raise OrbitError("orbit could not be propagated to these times")
    return rate, np.atleast_1d(elevation.degrees), np.atleast_1d(azimuth.degrees)


def doppler_offset_khz(f_other_hz: float, rate_other: np.ndarray, f_tuned_hz: float,
                       rate_scheduled: np.ndarray) -> np.ndarray:
    """Where a transmitter appears on a waterfall Doppler-corrected for the scheduled satellite."""
    return (f_other_hz * (1 - rate_other / C) - f_tuned_hz * (1 - rate_scheduled / C)) / 1000.0


# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Curve:
    """One satellite's predicted track on one waterfall."""

    norad: int
    name: str
    transmitter: str
    frequency_hz: float
    seconds: np.ndarray = field(repr=False)
    khz: np.ndarray = field(repr=False)
    elevation: np.ndarray = field(repr=False)
    azimuth: np.ndarray = field(repr=False)
    scheduled: bool = False

    @property
    def visible(self) -> np.ndarray:
        return self.elevation > 0.0


@dataclass(frozen=True)
class PassPrediction:
    """Every satellite whose signal could show on this waterfall, with its curve."""

    obs: Observation
    axes: WaterfallAxes
    curves: tuple[Curve, ...]

    @classmethod
    def build(cls, obs: Observation, axes: WaterfallAxes, catalog: SatnogsCatalog, site: Site,
              samples: int = 120) -> "PassPrediction":
        if obs.start is None or obs.tle is None or not obs.frequency_hz:
            raise OrbitError(f"observation {obs.id} lacks a start time, orbit or frequency")
        seconds = np.linspace(axes.seconds_bottom, axes.seconds_top, samples)
        times = [obs.start + timedelta(seconds=float(s)) for s in seconds]
        f_tuned = float(obs.frequency_hz)
        scheduled_rate, elevation, azimuth = look(obs.tle, site, times)

        # What the station tuned for (the observation's own TLE) sits at 0 by construction.
        curves = [Curve(int(obs.norad_cat_id or 0), catalog.name(int(obs.norad_cat_id or 0)) + " (as scheduled)",
                        "scheduled", f_tuned, seconds, np.zeros(samples), elevation, azimuth, scheduled=True)]

        # A transmitter that does not move (on the ground, or geostationary):
        # range rate 0, so after the station's correction it traces the mirror
        # image of the scheduled satellite's Doppler.  Local interference.
        curves.append(Curve(0, "Fixed transmitter (ground interference)", "not a satellite", f_tuned, seconds,
                            doppler_offset_khz(f_tuned, np.zeros(samples), f_tuned, scheduled_rate),
                            np.full(samples, 90.0), np.zeros(samples)))

        # Search the band, plus how far Doppler can move a signal into it.
        reach_khz = max(abs(axes.khz_left), abs(axes.khz_right)) + f_tuned * 2 * 8000 / C / 1000
        tles = catalog.tles()
        seen: set[tuple[int, int]] = set()
        for tx in catalog.transmitters():
            if abs(tx.frequency_hz - f_tuned) / 1000 > reach_khz or tx.norad not in tles:
                continue
            key = (tx.norad, round(tx.frequency_hz))
            if key in seen:
                continue
            seen.add(key)
            name, line1, line2 = tles[tx.norad]
            try:
                rate, elev, az = look((line1, line2), site, times)
            except OrbitError:
                continue
            khz = doppler_offset_khz(tx.frequency_hz, rate, f_tuned, scheduled_rate)
            on_screen = (elev > 0) & (khz >= axes.khz_left) & (khz <= axes.khz_right)
            if not on_screen.any():
                continue
            is_scheduled = tx.norad == obs.norad_cat_id
            label = f"{name or 'NORAD %d' % tx.norad}" + (" (today's orbit)" if is_scheduled else "")
            curves.append(Curve(tx.norad, label, tx.description or tx.mode, tx.frequency_hz,
                                seconds, khz, elev, az))
        return cls(obs, axes, tuple(curves))

    def others(self) -> list[Curve]:
        return [c for c in self.curves if not c.scheduled]

    def time_shifted(self, site: Site, shift_s: float) -> Curve:
        """The scheduled satellite's curve if it were really ``shift_s`` seconds ahead of its orbit.

        Newly launched satellites, and ones whose orbit data is a few days
        old, are often ahead or behind where their TLE puts them.  The
        station then corrects Doppler for the wrong moment, and the
        satellite's own signal drifts off the centre line in an S-shape.
        """
        scheduled = self.curves[0]
        times = [self.obs.start + timedelta(seconds=float(s)) for s in scheduled.seconds]
        later = [t + timedelta(seconds=float(shift_s)) for t in times]
        rate_now, _, _ = look(self.obs.tle, site, times)
        rate_real, elev, az = look(self.obs.tle, site, later)
        khz = doppler_offset_khz(scheduled.frequency_hz, rate_real, scheduled.frequency_hz, rate_now)
        return Curve(scheduled.norad, f"{scheduled.name.removesuffix(' (as scheduled)')} "
                     f"({shift_s:+.0f} s off its orbit)", "scheduled", scheduled.frequency_hz,
                     scheduled.seconds, khz, elev, az, scheduled=True)


# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Match:
    curve: Curve
    share: float          # share of the track's points that lie on the curve
    shift_khz: float      # constant offset applied (transmitters are often a few kHz off nominal)
    points: int
    residual_khz: float = 0.0   # mean distance of the agreeing points from the curve (breaks ties)

    @property
    def score(self) -> tuple[float, float]:
        """More points on the curve first; among equals, the closer fit."""
        return round(self.share, 2), -self.residual_khz


def match_track(prediction: PassPrediction, track_seconds: np.ndarray, track_khz: np.ndarray,
                tolerance_khz: float = 1.0, max_shift_khz: float = 5.0) -> list[Match]:
    """How well each predicted curve explains the bright track, best first.

    Each curve may slide sideways by up to ``max_shift_khz`` as a whole
    (a transmitter a little off its listed frequency), but not bend: the
    *shape* has to fit, which is what makes the match meaningful.
    """
    track_seconds, track_khz = np.asarray(track_seconds, float), np.asarray(track_khz, float)
    if len(track_seconds) < 5:
        return []
    results = []
    for curve in prediction.curves:
        visible = curve.visible
        if visible.sum() < 2:
            continue
        predicted = np.interp(track_seconds, curve.seconds, curve.khz)
        above = np.interp(track_seconds, curve.seconds, curve.elevation) > 0
        if above.sum() < 5:
            continue
        residual = track_khz[above] - predicted[above]
        # Try every sideways offset and keep the one that puts the most track
        # points on the curve.  (A median is pulled off by stray points from
        # interference; counting agreement is not.)
        offsets = np.arange(-max_shift_khz, max_shift_khz + 1e-9, tolerance_khz / 4)
        agree = (np.abs(residual[None, :] - offsets[:, None]) <= tolerance_khz).mean(axis=1)
        best = int(np.argmax(agree))
        inliers = np.abs(residual - offsets[best]) <= tolerance_khz
        shift = float(np.median(residual[inliers])) if inliers.any() else float(offsets[best])
        share = float(agree[best] * above.mean())
        residual_khz = float(np.mean(np.abs(residual[inliers] - shift))) if inliers.any() else tolerance_khz
        results.append(Match(curve, share, shift, int(above.sum()), residual_khz))
    return sorted(results, key=lambda m: m.score, reverse=True)


def fit_time_shift(prediction: PassPrediction, site: Site, track_seconds: np.ndarray, track_khz: np.ndarray,
                   max_shift_s: float = 600.0, step_s: float = 5.0, max_offset_khz: float = 5.0) -> Match | None:
    """The time shift of the scheduled satellite's orbit that best explains the track, if any helps."""
    best: Match | None = None
    for shift in np.arange(-max_shift_s, max_shift_s + step_s / 2, step_s):
        if abs(shift) < step_s / 2:
            continue
        curve = prediction.time_shifted(site, float(shift))
        trial = match_track(PassPrediction(prediction.obs, prediction.axes, (curve,)), track_seconds, track_khz,
                            max_shift_khz=max_offset_khz)
        if trial and (best is None or trial[0].score > best.score):
            best = trial[0]
    return best
