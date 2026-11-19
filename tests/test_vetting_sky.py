"""Tests for the sky map's astronomy (satrecovery.vetting.sky) and its view."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from satrecovery.vetting.sky import StarCatalogue, altaz, sky_state, sun_radec  # noqa: E402

DORSET = (50.77, -2.02)


def test_the_catalogue_is_the_naked_eye_sky():
    stars = StarCatalogue()
    assert 4000 < len(stars) < 9000 and stars.mag.max() <= 6.5
    assert stars.mag.min() < -1.0                       # Sirius
    assert len(stars.lines) > 80 and len(stars.names) == 89


def test_noon_sun_height_matches_latitude_and_season():
    noon = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)
    ra, dec = sun_radec(noon)
    assert dec == pytest.approx(-19.5, abs=0.5)        # mid-November
    alt, _ = altaz(np.array([ra]), np.array([dec]), *DORSET, noon)
    assert alt[0] == pytest.approx(90 - DORSET[0] + dec, abs=1.0)


def test_polaris_stands_at_the_latitude_and_due_north():
    when = datetime(2026, 11, 19, 21, 0, tzinfo=timezone.utc)
    alt, az = altaz(np.array([37.95]), np.array([89.26]), *DORSET, when)
    assert alt[0] == pytest.approx(DORSET[0], abs=1.0)
    assert min(az[0], 360 - az[0]) < 2.0


def test_stars_rise_in_the_east():
    from datetime import timedelta

    when = datetime(2026, 11, 19, 21, 0, tzinfo=timezone.utc)
    ra, dec = np.array([150.0]), np.array([0.0])        # an equator star
    before, az = altaz(ra, dec, *DORSET, when)
    after, _ = altaz(ra, dec, *DORSET, when + timedelta(minutes=10))
    if 0 < az[0] < 180:                                 # in the east: climbing
        assert after[0] > before[0]
    else:
        assert after[0] < before[0]


def test_sky_states():
    assert [sky_state(a) for a in (10, -3, -9, -30)] == ["day", "twilight", "dusk", "night"]


def test_sky_map_draws_day_and_night():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from satrecovery.vetting.gui.skymap import SkyMap
    from satrecovery.vetting.orbit import Site, ground_track

    QApplication.instance() or QApplication([])
    tle = ("1 69015U 26100AM  26323.14274952  .00009061  00000-0  37966-3 0  9990",
           "2 69015  97.3828 218.4872 0006677 207.2028 152.8861 15.23733342 30400")
    for hour in (12, 22):
        start = datetime(2026, 11, 19, hour, tzinfo=timezone.utc)
        track = ground_track(tle, Site(*DORSET), start, np.linspace(0, 6000, 200))
        sky = SkyMap()
        sky.resize(700, 520)
        sky.set_pass(DORSET, "test", track, start, "FRONTIERSAT")
        sky.set_time(3000.0, 0.5, "a subtitle")
        assert not sky.grab().isNull()
