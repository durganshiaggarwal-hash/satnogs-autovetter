"""Tests for the replay subtitles (satrecovery.vetting.narration)."""

from __future__ import annotations

import numpy as np
import pytest

from satrecovery.vetting.narration import PassNarrator, Places, sea
from satrecovery.vetting.orbit import Curve, GroundTrack, Match, TrackExplanation


@pytest.fixture(scope="module")
def places() -> Places:
    return Places()


def test_countries_and_seas(places):
    assert places.name(51.5, -0.1) == "the United Kingdom"
    assert places.name(64.9, -18.6) == "Iceland"
    assert places.name(48.8, 2.3) == "France"
    assert places.name(35.0, -40.0) == "the North Atlantic"
    assert sea(-60, 10) == "the Southern Ocean" and sea(10, 80) == "the Indian Ocean" and sea(0, -150).endswith("Pacific")


def track(elevation: np.ndarray) -> GroundTrack:
    n = len(elevation)
    return GroundTrack(np.linspace(-100, 500, n), np.linspace(20, 70, n), np.full(n, 0.0), np.full(n, 600.0),
                       elevation, np.full(n, 25.0), np.full(n, 1500.0))


def explanation(source: str) -> TrackExplanation:
    s = np.linspace(0, 1, 3)
    match = Match(Curve(7, "GEMINI-POLLUX (x)", "", 1.0, s, s, s, s), 0.9, 0.0, 30)
    return TrackExplanation(None, None, None, s, s, (match,), None, "", "", source)  # type: ignore[arg-type]


class TestNarrator:
    ELEVATION = np.r_[np.full(10, -5.0), np.linspace(1, 80, 20), np.linspace(80, 1, 20), np.full(10, -5.0)]

    def test_follows_the_pass(self, places):
        narrator = PassNarrator(track(self.ELEVATION), "ORESAT", places)
        seconds = narrator._track.seconds  # noqa: SLF001
        assert "below the station's horizon" in narrator.line(seconds[2])
        assert "rising over the horizon" in narrator.line(seconds[10])
        assert "Climbing" in narrator.line(seconds[18])
        assert "Closest point" in narrator.line(seconds[29]) and "overhead" in narrator.line(seconds[29])
        assert "Going down" in narrator.line(seconds[38])
        assert "sinking" in narrator.line(seconds[48])
        assert "set below the horizon" in narrator.line(seconds[57])

    def test_says_what_the_signal_is(self, places):
        loud = PassNarrator(track(self.ELEVATION), "ORESAT", places, activity=lambda s: 1.0)
        moment = loud._track.seconds[29]  # noqa: SLF001
        assert "still working out" in loud.line(moment)
        loud.set_explanation(explanation("ground"))
        assert "transmitter on the ground" in loud.line(moment)
        loud.set_explanation(explanation("other"))
        assert "GEMINI-POLLUX" in loud.line(moment)
        loud.set_explanation(explanation("scheduled"))
        assert "hearing ORESAT" in loud.line(moment)
        quiet = PassNarrator(track(self.ELEVATION), "ORESAT", places, activity=lambda s: 0.0)
        assert "Only noise" in quiet.line(moment)

    def test_a_pass_that_never_rises(self, places):
        narrator = PassNarrator(track(np.full(20, -10.0)), "ORESAT", places)
        assert "never rises" in narrator.line(100.0)
