"""Tests for the orbit physics: Doppler curves, the catalogue, and matching a track to a source (no network)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("skyfield")

from satrecovery.vetting import ArtifactCache, Observation, Response, Transport  # noqa: E402
from satrecovery.vetting.axes import WaterfallAxes  # noqa: E402
from satrecovery.vetting.orbit import (  # noqa: E402
    C,
    Curve,
    PassPrediction,
    SatnogsCatalog,
    Site,
    StationDirectory,
    doppler_offset_khz,
    fit_time_shift,
    look,
    match_track,
)

# A real TLE from SatNOGS DB (FRONTIERSAT, November 2026).
TLE = ("1 69015U 26100AM  26323.14274952  .00009061  00000-0  37966-3 0  9990",
       "2 69015  97.3828 218.4872 0006677 207.2028 152.8861 15.23733342 30400")
SITE = Site(39.963, -85.93, 250.0)


def find_pass(tle=TLE, site=SITE) -> datetime:
    """A moment the satellite is high above the site, found by search."""
    start = datetime(2026, 11, 19, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=m) for m in range(0, 24 * 60, 1)]
    _, elevation, _ = look(tle, site, times)
    return times[int(np.argmax(elevation))]


class FakeTransport(Transport):
    def __init__(self, pages: dict[str, object]) -> None:
        self.pages = pages
        self.requests: list[str] = []

    def get(self, url, timeout=60.0):
        self.requests.append(url)
        if url not in self.pages:
            return Response(404, {}, b"")
        return Response(200, {}, json.dumps(self.pages[url]).encode())


class TestPhysics:
    def test_satellite_motion_is_plausible(self):
        peak = find_pass()
        times = [peak + timedelta(seconds=s) for s in range(-400, 401, 20)]
        rate, elevation, azimuth = look(TLE, SITE, times)
        assert elevation.max() > 10
        assert np.all(np.abs(rate) < 8000)            # a low orbit: under 8 km/s along the line of sight
        assert rate[0] < 0 < rate[-1]                 # approaching (closing) then receding
        assert np.all((0 <= azimuth) & (azimuth < 360))

    def test_scheduled_satellite_sits_on_the_centre_line(self):
        rate = np.linspace(-7000, 7000, 50)
        assert np.allclose(doppler_offset_khz(436e6, rate, 436e6, rate), 0.0)

    def test_fixed_transmitter_is_the_mirror_of_the_satellite(self):
        rate = np.linspace(-7000, 7000, 50)
        offset = doppler_offset_khz(436e6, np.zeros(50), 436e6, rate)
        assert np.allclose(offset, 436e6 * rate / C / 1000)
        assert offset[0] < 0 < offset[-1] and abs(offset).max() == pytest.approx(436e6 * 7000 / C / 1000)

    def test_another_transmitter_10khz_up_is_10khz_off(self):
        rate = np.zeros(5)
        assert np.allclose(doppler_offset_khz(436.010e6, rate, 436.0e6, rate), 10.0)


class TestCatalogue:
    def _catalog(self, tmp_path, transmitters):
        tles = [{"norad_cat_id": "69015", "tle0": "0 FRONTIERSAT", "tle1": TLE[0], "tle2": TLE[1]}]
        transport = FakeTransport({SatnogsCatalog.TLE_URL: tles, SatnogsCatalog.TRANSMITTER_URL: transmitters})
        return SatnogsCatalog(tmp_path, transport), transport

    def test_reads_and_caches(self, tmp_path):
        catalog, transport = self._catalog(tmp_path, [
            {"norad_cat_id": 69015, "downlink_low": 436150000, "alive": True, "status": "active", "mode": "FSK"},
            {"norad_cat_id": 1, "downlink_low": 145800000, "alive": False, "status": "active"},
            {"norad_cat_id": 2, "downlink_low": None, "alive": True, "status": "active"},
        ])
        assert catalog.name(69015) == "FRONTIERSAT"
        assert [t.norad for t in catalog.transmitters()] == [69015]
        again = SatnogsCatalog(tmp_path, FakeTransport({}))  # offline: the saved copy is used
        assert len(again.tles()) == 1 and len(transport.requests) == 2

    def test_station_from_the_observation_or_the_api(self, tmp_path):
        obs = Observation.from_api({"id": 1, "ground_station": 837, "station_lat": 1.0, "station_lng": 2.0,
                                    "station_alt": 3})
        directory = StationDirectory(tmp_path, FakeTransport({}))
        assert directory.site(obs) == Site(1.0, 2.0, 3.0)
        transport = FakeTransport({StationDirectory.URL.format(837): [{"lat": 39.9, "lng": -85.9, "altitude": 850}]})
        bare = Observation.from_api({"id": 2, "ground_station": 837})
        assert StationDirectory(tmp_path, transport).site(bare) == Site(39.9, -85.9, 850.0)
        assert StationDirectory(tmp_path, FakeTransport({})).site(bare).lat == 39.9  # remembered


def observation_at(peak: datetime, frequency: float = 436150000) -> Observation:
    start = peak - timedelta(seconds=300)
    return Observation.from_api({
        "id": 9, "norad_cat_id": 69015, "ground_station": 837, "start": start.isoformat(),
        "end": (start + timedelta(seconds=600)).isoformat(), "observation_frequency": frequency,
        "tle1": TLE[0], "tle2": TLE[1], "station_lat": SITE.lat, "station_lng": SITE.lng, "station_alt": SITE.alt_m,
    })


class TestPassPrediction:
    def test_finds_a_neighbour_and_ignores_far_ones(self, tmp_path):
        peak = find_pass()
        obs = observation_at(peak)
        tles = [{"norad_cat_id": "69015", "tle0": "0 FRONTIERSAT", "tle1": TLE[0], "tle2": TLE[1]}]
        transmitters = [
            {"norad_cat_id": 69015, "downlink_low": 436150000 + 8000, "alive": True, "status": "active"},
            {"norad_cat_id": 69015, "downlink_low": 437500000, "alive": True, "status": "active"},
        ]
        catalog = SatnogsCatalog(tmp_path, FakeTransport({SatnogsCatalog.TLE_URL: tles,
                                                          SatnogsCatalog.TRANSMITTER_URL: transmitters}))
        prediction = PassPrediction.build(obs, WaterfallAxes(-24, 24, 0, 600), catalog, SITE)
        names = [c.name for c in prediction.curves]
        assert prediction.curves[0].scheduled and np.allclose(prediction.curves[0].khz, 0)
        assert any("ground" in n.lower() for n in names)
        satellites = [c for c in prediction.others() if c.norad == 69015]
        assert len(satellites) == 1 and np.allclose(satellites[0].khz, 8.0, atol=0.05)  # same orbit, 8 kHz up

    def test_time_shift_moves_the_track_off_centre(self, tmp_path):
        peak = find_pass()
        obs = observation_at(peak)
        catalog = SatnogsCatalog(tmp_path, FakeTransport({SatnogsCatalog.TLE_URL: [],
                                                          SatnogsCatalog.TRANSMITTER_URL: []}))
        prediction = PassPrediction.build(obs, WaterfallAxes(-24, 24, 0, 600), catalog, SITE)
        shifted = prediction.time_shifted(SITE, 60.0)
        assert np.abs(shifted.khz).max() > 0.5
        # A track drawn from a 60 s shift -- only while the satellite is up,
        # as a real one would be -- is recognised as a 60 s shift.
        up = shifted.visible
        best = fit_time_shift(prediction, SITE, shifted.seconds[up][::2], shifted.khz[up][::2],
                              max_shift_s=120, step_s=10)
        assert best is not None and "+60 s" in best.curve.name and best.share > 0.9


class TestMatching:
    def _prediction(self):
        seconds = np.linspace(0, 600, 121)
        axes = WaterfallAxes(-24, 24, 0, 600)
        s_curve = 10 * np.tanh((seconds - 300) / 100)
        curves = (
            Curve(1, "centre", "", 1, seconds, np.zeros_like(seconds), np.full_like(seconds, 30), seconds, True),
            Curve(2, "S-curve", "", 1, seconds, s_curve, np.full_like(seconds, 30), seconds),
        )
        return PassPrediction(None, axes, curves), seconds, s_curve  # type: ignore[arg-type]

    def test_picks_the_curve_the_track_follows_even_when_shifted(self):
        prediction, seconds, s_curve = self._prediction()
        rng = np.random.default_rng(0)
        track = s_curve[::3] + 2.0 + rng.normal(0, 0.2, len(s_curve[::3]))
        best = match_track(prediction, seconds[::3], track, max_shift_khz=3.0)[0]
        assert best.curve.name == "S-curve" and best.share > 0.9 and best.shift_khz == pytest.approx(2.0, abs=0.3)

    def test_stray_points_do_not_drag_the_offset(self):
        prediction, seconds, s_curve = self._prediction()
        track = np.r_[s_curve[::3], np.full(10, -20.0)]
        times = np.r_[seconds[::3], np.linspace(0, 600, 10)]
        best = match_track(prediction, times, track, max_shift_khz=3.0)[0]
        assert best.curve.name == "S-curve" and abs(best.shift_khz) < 0.3

    def test_scattered_noise_matches_nothing_well(self):
        prediction, seconds, _ = self._prediction()
        rng = np.random.default_rng(1)
        results = match_track(prediction, rng.uniform(0, 600, 40), rng.uniform(-24, 24, 40))
        assert results[0].share < 0.3

    def test_too_few_points(self):
        prediction, seconds, _ = self._prediction()
        assert match_track(prediction, seconds[:3], np.zeros(3)) == []


class TestAxesCache:
    def test_failure_is_remembered_and_success_is_read(self, tmp_path):
        cache = ArtifactCache(tmp_path, FakeTransport({}))
        obs = Observation.from_api({"id": 5, "waterfall": "https://cdn.example/5.png"})
        cache.axes_path(obs).parent.mkdir(parents=True)
        cache.axes_path(obs).write_text(json.dumps({"ok": False, "reason": "unreadable"}))
        assert cache.axes(obs) is None
        cache.axes_path(obs).write_text(json.dumps({"ok": True, "khz_left": -24, "khz_right": 24,
                                                    "seconds_bottom": 0, "seconds_top": 400}))
        assert cache.axes(obs) == WaterfallAxes(-24, 24, 0, 400)
        assert cache.axes(Observation.from_api({"id": 6}), download=True) is None  # no waterfall to fetch


class TestVerdicts:
    """The words the window shows, for each kind of source."""

    def _match(self, name, share, norad=5, scheduled=False):
        seconds = np.linspace(0, 10, 3)
        curve = Curve(norad, name, "", 1.0, seconds, seconds, seconds, seconds, scheduled)
        from satrecovery.vetting.orbit import Match

        return Match(curve, share, 0.0, 20)

    def _words(self, matches, timing=None):
        from satrecovery.vetting.orbit import TrackExplainer

        return TrackExplainer(None, None, None)._words(matches, timing, 30)  # type: ignore[arg-type]

    def test_each_source_has_its_own_words(self):
        assert self._words([self._match("ORESAT (as scheduled)", 0.8, scheduled=True)])[2] == "scheduled"
        assert self._words([self._match("ORESAT (today's orbit)", 0.8)])[2] == "scheduled"
        verdict, _, source = self._words([self._match("Ground transmitter (fixed frequency)", 0.9, norad=0)])
        assert source == "ground" and "interference" in verdict
        verdict, _, source = self._words([self._match("GEMINI-POLLUX", 0.42)])
        assert source == "other" and "probably GEMINI-POLLUX" in verdict
        assert self._words([self._match("GEMINI-POLLUX", 0.1)])[2] == "unknown"
        assert self._words([])[2] == "unknown"

    def test_timing_wins_only_when_clearly_better(self):
        best = self._match("GEMINI-POLLUX", 0.5)
        assert self._words([best], self._match("ORESAT (+40 s off its orbit)", 0.6, scheduled=True))[2] == "other"
        verdict, _, source = self._words([best], self._match("ORESAT (+40 s off its orbit)", 0.8, scheduled=True))
        assert source == "timing" and "+40 s" in verdict

    def test_plausible_timing_error_grows_with_age(self):
        from satrecovery.vetting.orbit import plausible_time_error_s, tle_epoch

        epoch = tle_epoch(TLE[0])
        assert epoch.year == 2026 and epoch.month == 11
        fresh = observation_at(epoch + timedelta(hours=6))
        stale = observation_at(epoch + timedelta(days=4))
        assert plausible_time_error_s(fresh) < plausible_time_error_s(stale) <= 600
        new_launch = Observation.from_api({"id": 3, "norad_cat_id": 98528, "start": "2026-11-20T00:00:00Z",
                                           "tle1": TLE[0], "tle2": TLE[1]})
        assert plausible_time_error_s(new_launch) == 600


class TestLinkPhysics:
    def test_delay_doppler_speed_and_next_passes(self):
        from satrecovery.vetting.orbit import ground_track, next_passes

        peak = find_pass()
        track = ground_track(TLE, SITE, peak - timedelta(seconds=400), np.linspace(0, 800, 81))
        i_top = int(np.argmax(track.elevation))
        assert track.delay_ms(i_top) == pytest.approx(track.range_km[i_top] / 299792.458 * 1000)
        assert 6.5 < float(np.median(track.speed_kms)) < 8.0                    # low Earth orbit speed
        first, last = 5, 75
        assert track.doppler_hz(first, 436e6) > 0 > track.doppler_hz(last, 436e6)  # higher, then lower
        assert abs(track.doppler_hz(i_top, 436e6)) < abs(track.doppler_hz(first, 436e6))
        passes = next_passes(TLE, SITE, peak + timedelta(minutes=20), hours=24, min_elevation=0)
        assert passes and all(p.rise < p.culminate < p.set and p.max_elevation >= 0 for p in passes)
        assert passes[0].rise > peak
