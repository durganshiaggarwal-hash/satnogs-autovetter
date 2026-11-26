"""Tests for the plain-English verdicts (satrecovery.vetting.story)."""

from __future__ import annotations

import numpy as np

from satrecovery.vetting import Observation
from satrecovery.vetting.features import FEATURES
from satrecovery.vetting.orbit import Curve, Match, TrackExplanation
from satrecovery.vetting.session import Item, Policy
from satrecovery.vetting.story import tell

POLICY = Policy(0.15, 0.85)


def item(p: float) -> Item:
    obs = Observation.from_api({"id": 1, "norad_cat_id": 5})
    return Item(obs, p, POLICY.verdict(p), np.zeros(len(FEATURES)))


def explanation(source: str, name: str = "GEMINI-POLLUX", share: float = 0.9, norad: int = 7) -> TrackExplanation:
    s = np.linspace(0, 1, 3)
    match = Match(Curve(norad, name, "", 1.0, s, s, s, s), share, 0.0, 30)
    return TrackExplanation(None, None, None, s, s, (match,), None, "", "", source)  # type: ignore[arg-type]


def test_ground_interference_is_not_heard_and_says_why():
    story = tell(item(0.5), explanation("ground", "Ground transmitter (fixed frequency)", norad=0), "ORESAT")
    assert story.mark == "not heard" and story.symbol == "x" and "ground" in story.sentence and story.dots == 5


def test_another_satellite_is_named():
    story = tell(item(0.7), explanation("other", "GEMINI-POLLUX (x)", share=0.44), "ORESAT")
    assert story.headline == "That was GEMINI-POLLUX, not ORESAT" and story.dots == 3


def test_model_alone_when_there_is_no_track():
    assert tell(item(0.95), None, "ORESAT").mark == "heard"
    quiet = tell(item(0.05), explanation("no track"), "ORESAT")
    assert quiet.mark == "not heard" and "noise" in quiet.sentence and quiet.dots == 5


def test_physics_confirming_the_scheduled_satellite():
    assert tell(item(0.6), explanation("scheduled", "ORESAT (as scheduled)"), "ORESAT").mark == "heard"


def test_unsure_says_so_and_why():
    story = tell(item(0.5), explanation("unknown", share=0.1), "ORESAT")
    assert story.mark == "unsure" and story.symbol == "?" and "no known" in story.sentence and story.dots == 1
