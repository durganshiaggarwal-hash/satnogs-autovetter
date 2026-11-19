"""Tests for the saved vetter bundle, the bundle-based session and the latest-recordings feed (no network)."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from satrecovery.core.exceptions import ValidationError  # noqa: E402
from satrecovery.vetting import ArtifactCache, Manifest, Observation, WaterfallImage  # noqa: E402
from satrecovery.vetting.bundle import VetterBundle  # noqa: E402
from satrecovery.vetting.session import Policy, VettingSession  # noqa: E402
from satrecovery.vetting.source import ManifestSource  # noqa: E402
from satrecovery.vetting.unvetted import fetch_latest, is_finished_recording  # noqa: E402


def synthetic(signal: bool, rng, size: int = 64) -> np.ndarray:
    power = rng.normal(0.4, 0.05, (size, size))
    if signal:
        for row in rng.choice(size - 4, size=5, replace=False):
            power[row:row + 3, size // 2 - 3:size // 2 + 3] += 0.45
    return np.clip(power, 0, 1).astype(np.float32)


def observation(i: int, label: str = "unknown", minutes_ago: float = 30.0, **changes) -> Observation:
    start = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    row = {"id": i, "ground_station": i % 12, "norad_cat_id": 1000 + i % 5, "start": start.isoformat(),
           "end": (start + timedelta(minutes=9)).isoformat(), "waterfall": f"https://cdn.example/{i}.png",
           "waterfall_status": label, "waterfall_status_user": None if label == "unknown" else 1,
           "status": "unknown"}
    row.update(changes)
    return Observation.from_api(row)


@pytest.fixture()
def trained(tmp_path):
    rng = np.random.default_rng(3)
    cache = ArtifactCache(tmp_path)
    manifest = Manifest(tmp_path / "manifest.jsonl")
    for i in range(1, 81):
        obs = observation(i, "with-signal" if i % 2 else "without-signal")
        manifest.add(obs)
        WaterfallImage(synthetic(bool(i % 2), rng)).save(cache.waterfall_path(obs))
    return VetterBundle.train(manifest, cache), cache, tmp_path, rng


class TestBundle:
    def test_round_trip_without_the_training_data(self, trained):
        bundle, cache, root, _ = trained
        bundle.save(root / "models")
        for path in (root / "waterfalls").glob("*.npz"):
            path.unlink()  # the app must not need them
        (root / "features.npz").unlink()
        again = VetterBundle.load(root / "models")
        assert again.n_train == 80
        assert np.allclose(again.oof_probability, bundle.oof_probability)
        assert again.measured_error(Policy()) == bundle.measured_error(Policy())
        assert 0.0 <= again.automatable(Policy()) <= 1.0

    def test_missing_bundle_is_a_clear_error(self, tmp_path):
        assert not VetterBundle.exists(tmp_path)
        with pytest.raises(ValidationError, match="build_vetter_bundle"):
            VetterBundle.load(tmp_path)

    def test_stricter_thresholds_decide_less(self, trained):
        bundle = trained[0]
        assert bundle.automatable(Policy(0.01, 0.99)) <= bundle.automatable(Policy(0.3, 0.7))


class TestSessionFromBundle:
    def test_starts_empty_and_adds_as_recordings_arrive(self, trained):
        bundle, cache, root, rng = trained
        session = VettingSession.from_bundle(bundle, [], cache)
        assert len(session) == 0 and session.modes == ("latest",) and session.mode == "latest"
        assert session.measured_error() == bundle.measured_error(session.policy)
        fresh = []
        for i in range(500, 505):
            obs = observation(i)
            WaterfallImage(synthetic(i % 2 == 0, rng)).save(cache.waterfall_path(obs))
            fresh.append(obs)
        added = session.add(fresh)
        assert len(added) == 5 and len(session) == 5
        assert session.add(fresh) == []            # no duplicates
        assert all(0 <= i.probability <= 1 for i in session.ordered())


class TestLatestFeed:
    def test_keeps_the_newest_finished_recordings(self, tmp_path):
        rng = np.random.default_rng(0)
        cache = ArtifactCache(tmp_path)
        listing = Manifest(tmp_path / "listing.jsonl")
        for i, extra in [(1, {}), (2, {"waterfall": None}), (3, {"status": "future"}), (4, {}),
                         (5, {"waterfall_status": "with-signal", "waterfall_status_user": 2})]:
            obs = observation(i, **extra)
            listing.add(obs)
            if obs.waterfall_url:
                WaterfallImage(synthetic(True, rng)).save(cache.waterfall_path(obs))
        assert not is_finished_recording(observation(9, waterfall=None))
        latest = Manifest(tmp_path / "latest.jsonl")
        arrived = []
        n = fetch_latest(ManifestSource(listing), cache, latest, count=10, added=arrived.append)
        assert n == 3 and sorted(o.id for o in latest) == [1, 4, 5] and len(arrived) == 3

    def test_old_recordings_are_left_out(self, tmp_path):
        cache = ArtifactCache(tmp_path)
        listing = Manifest(tmp_path / "listing.jsonl")
        obs = observation(1, minutes_ago=60 * 72)
        listing.add(obs)
        WaterfallImage(synthetic(True, np.random.default_rng(1))).save(cache.waterfall_path(obs))
        assert fetch_latest(ManifestSource(listing), cache, Manifest(tmp_path / "l.jsonl"), 10, max_age_hours=48) == 0


pytest.importorskip("PySide6")


def test_expert_mode_opens_from_a_replay_in_the_live_app(trained):
    """Regression: the live app has one 'latest' queue, and select() tried to switch to 'unvetted'."""
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication

    from satrecovery.vetting.gui.mission import MissionWindow

    app = QApplication.instance() or QApplication([])
    bundle, cache, root, rng = trained
    obs = observation(700)
    WaterfallImage(synthetic(True, rng)).save(cache.waterfall_path(obs))
    session = VettingSession.from_bundle(bundle, [obs], cache)
    win = MissionWindow(session, explainer=None)
    try:
        win.open_replay(obs.id)
        win.open_expert(obs.id)
        assert win._expert is not None and win._expert._current == obs.id  # noqa: SLF001
    finally:
        win.close()
        win.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()


def test_feed_stops_quietly_when_its_window_is_gone(tmp_path):
    from satrecovery.vetting.gui.live import LiveFeed, _Closed

    feed = LiveFeed(ArtifactCache(tmp_path), Manifest(tmp_path / "l.jsonl"))
    feed.stop()
    assert feed.stopped and issubclass(_Closed, Exception)
