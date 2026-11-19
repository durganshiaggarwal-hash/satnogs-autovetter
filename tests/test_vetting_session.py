"""Tests for the vetting session and the vetting station window (no network, no display)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from satrecovery.core.exceptions import ValidationError  # noqa: E402
from satrecovery.vetting import ArtifactCache, Manifest, Observation, WaterfallImage, WaterfallLabel  # noqa: E402
from satrecovery.vetting.features import FEATURES  # noqa: E402
from satrecovery.vetting.session import (  # noqa: E402
    DecisionLog,
    Explainer,
    FeatureStore,
    Policy,
    Verdict,
    VettingSession,
    out_of_fold,
    track_points,
)


def synthetic(signal: bool, rng: np.random.Generator, size: int = 64) -> np.ndarray:
    power = rng.normal(0.4, 0.05, (size, size))
    if signal:
        for row in rng.choice(size - 4, size=5, replace=False):
            power[row:row + 3, size // 2 - 3:size // 2 + 3] += 0.45
    return np.clip(power, 0, 1).astype(np.float32)


@pytest.fixture()
def downloaded(tmp_path):
    """60 labelled observations from 12 stations, waterfalls in the cache."""
    rng = np.random.default_rng(4)
    manifest = Manifest(tmp_path / "manifest.jsonl")
    cache = ArtifactCache(tmp_path)
    for i in range(1, 61):
        signal = i % 2 == 0
        obs = Observation.from_api({
            "id": i, "ground_station": i % 12, "norad_cat_id": 1000 + i % 5,
            "start": f"2026-11-{1 + i % 28:02d}T00:00:00Z", "end": f"2026-11-{1 + i % 28:02d}T00:09:00Z",
            "waterfall": f"https://cdn.example/{i}.png", "waterfall_status_user": 1,
            "waterfall_status": "with-signal" if signal else "without-signal",
        })
        manifest.add(obs)
        WaterfallImage(synthetic(signal, rng)).save(cache.waterfall_path(obs))
    return manifest, cache, tmp_path


class TestPolicy:
    def test_three_zones(self):
        policy = Policy(0.2, 0.8)
        assert policy.verdict(0.9) is Verdict.SIGNAL
        assert policy.verdict(0.1) is Verdict.NO_SIGNAL
        assert policy.verdict(0.5) is Verdict.UNSURE
        assert Verdict.UNSURE.label is None and Verdict.SIGNAL.label is WaterfallLabel.SIGNAL

    def test_checked(self):
        with pytest.raises(ValidationError):
            Policy(0.9, 0.1)


class TestExplainer:
    def test_points_the_right_way(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(1.0, 0.1, (50, len(FEATURES)))
        none = rng.normal(0.0, 0.1, (50, len(FEATURES)))
        explainer = Explainer(np.vstack([signal, none]), np.r_[np.ones(50), np.zeros(50)])
        assert all(toward for _, _, toward in explainer.reasons(np.full(len(FEATURES), 1.2)))
        assert not any(toward for _, _, toward in explainer.reasons(np.full(len(FEATURES), -0.2)))

    def test_needs_both_labels(self):
        with pytest.raises(ValidationError):
            Explainer(np.zeros((5, len(FEATURES))), np.ones(5))


class TestDecisionLog:
    def test_latest_wins_and_reloads(self, tmp_path):
        from satrecovery.vetting.session import Decision

        log = DecisionLog(tmp_path / "d.jsonl")
        log.record(Decision(1, WaterfallLabel.SIGNAL, "auto", 0.9, "t1"))
        log.record(Decision(1, WaterfallLabel.NO_SIGNAL, "you", 0.9, "t2"))
        again = DecisionLog(tmp_path / "d.jsonl")
        assert len(again) == 1 and again.get(1).label is WaterfallLabel.NO_SIGNAL and again.get(1).by == "you"


class TestSession:
    def test_scores_are_out_of_fold_by_station(self):
        rng = np.random.default_rng(1)
        features = rng.normal(size=(40, len(FEATURES))).astype(np.float32)
        targets = np.array([i % 2 for i in range(40)])
        groups = [i % 4 for i in range(40)]
        # A station whose rows are copies: a model that saw the station would score them perfectly.
        assert out_of_fold(features, targets, groups).shape == (40,)

    def test_builds_orders_decides_and_auto_vets(self, downloaded):
        manifest, cache, root = downloaded
        session = VettingSession.from_cache(manifest, cache, Policy(0.3, 0.7), log_path=root / "d.jsonl")
        assert len(session) == 60
        unsure_first = session.ordered("unsure first")
        assert unsure_first[0].uncertainty >= unsure_first[-1].uncertainty
        first = unsure_first[0].obs.id
        session.decide(first, WaterfallLabel.SIGNAL)
        assert session.decision(first).by == "you"
        assert session.ordered("unsure first")[-1].obs.id == first  # decided ones go to the end
        done = session.auto_vet()
        stats = session.stats()
        assert stats["auto"] == done == stats["confident"] - (session.item(first).verdict is not Verdict.UNSURE)
        assert stats["yours"] == 1
        assert 0.0 <= stats["automatable_share"] <= 1.0
        assert len(DecisionLog(root / "d.jsonl")) == 1 + done
        with pytest.raises(ValidationError):
            session.decide(first, WaterfallLabel.UNKNOWN)

    def test_easy_synthetic_signals_are_mostly_confident_and_right(self, downloaded):
        manifest, cache, _ = downloaded
        session = VettingSession.from_cache(manifest, cache)
        stats = session.stats()
        assert stats["confident_error"] < 0.2

    def test_policy_change_moves_verdicts(self, downloaded):
        manifest, cache, _ = downloaded
        session = VettingSession.from_cache(manifest, cache, Policy(0.3, 0.7))
        before = session.stats()["confident"]
        session.set_policy(Policy(0.01, 0.99))
        assert session.stats()["confident"] <= before

    def test_feature_store_reuses_saved_measurements(self, downloaded):
        manifest, cache, root = downloaded
        kept, first = FeatureStore(cache).get(list(manifest))
        assert (root / "features.npz").exists()
        for path in (root / "waterfalls").glob("*.npz"):
            path.unlink()  # a second store must not need the pictures again
        kept_again, second = FeatureStore(cache).get(kept)
        assert np.allclose(first, second)

    def test_too_little_data(self, tmp_path):
        with pytest.raises(ValidationError):
            VettingSession.from_cache(Manifest(tmp_path / "m.jsonl"), ArtifactCache(tmp_path))


def test_track_points_find_the_bursts():
    power = np.zeros((32, 64), dtype=np.float32)
    power[5, 30] = power[20, 33] = 6.0
    rows, cols = track_points(power)
    assert rows.tolist() == [5, 20] and cols.tolist() == [30, 33]


# ----------------------------------------------------------------------
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")


@pytest.fixture()
def window(downloaded, qapp):
    from satrecovery.vetting.gui import VettingWindow

    manifest, cache, root = downloaded
    session = VettingSession.from_cache(manifest, cache, Policy(0.3, 0.7), log_path=root / "d.jsonl")
    win = VettingWindow(session)
    yield win, session
    # Delete the window (and its pyqtgraph scene) now, while the QApplication
    # still exists; left to interpreter shutdown, the order is undefined and
    # Qt aborts with "pure virtual function called".
    win.close()
    win.deleteLater()
    from PySide6.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qapp.processEvents()


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class TestWindow:
    def test_opens_on_the_most_uncertain(self, window):
        win, session = window
        assert win._current == session.ordered("unsure first")[0].obs.id
        assert win.obs_title.text() == f"#{win._current}"

    def test_keyboard_decision_saves_and_moves_on(self, window):
        from PySide6.QtTest import QTest
        from PySide6.QtCore import Qt

        win, session = window
        first = win._current
        win.show()
        from PySide6.QtWidgets import QApplication as App

        App.processEvents()
        assert App.focusWidget() is win.table  # keys go to vetting, not to a threshold box
        QTest.keyClick(App.focusWidget(), Qt.Key_G)
        assert session.decision(first).label is WaterfallLabel.SIGNAL
        assert win._current != first
        assert win.c_you.value.text() == "1"

    def test_answer_is_hidden_until_decided(self, window):
        win, session = window
        assert "hidden" in win.answer.text()
        win._decide(WaterfallLabel.NO_SIGNAL)
        win.table.selectRow(win.model.row_of(next(d.obs_id for d in session._log)))
        assert "volunteer said" in win.answer.text()

    def test_auto_vet_button_and_policy(self, window):
        win, session = window
        win._auto_vet()
        assert int(win.c_auto.value.text()) == session.stats()["auto"]
        win.high.setValue(0.99)
        assert session.policy.high == pytest.approx(0.99)


# ----------------------------------------------------------------------
from satrecovery.vetting.source import ManifestSource  # noqa: E402
from satrecovery.vetting.unvetted import fetch_unvetted, is_unvetted  # noqa: E402


def unvetted_obs(i: int, **changes) -> Observation:
    row = {"id": i, "ground_station": 99, "norad_cat_id": 7, "start": "2099-01-01T00:00:00Z",
           "end": "2099-01-01T00:09:00Z", "waterfall": f"https://cdn.example/{i}.png", "status": "unknown",
           "waterfall_status": "unknown", "waterfall_status_user": None}
    row.update(changes)
    return Observation.from_api(row)


class TestUnvetted:
    def test_what_counts_as_unvetted(self):
        assert is_unvetted(unvetted_obs(1))
        assert not is_unvetted(unvetted_obs(2, waterfall_status="with-signal", waterfall_status_user=3))
        assert not is_unvetted(unvetted_obs(3, waterfall=None))
        assert not is_unvetted(unvetted_obs(4, status="future"))

    def test_fetch_keeps_only_new_unvetted_ones(self, tmp_path):
        rng = np.random.default_rng(0)
        cache = ArtifactCache(tmp_path)
        listing = Manifest(tmp_path / "listing.jsonl")
        for obs in [unvetted_obs(1), unvetted_obs(2, waterfall_status="with-signal", waterfall_status_user=3),
                    unvetted_obs(3), unvetted_obs(4, status="future"), unvetted_obs(5)]:
            listing.add(obs)
            WaterfallImage(synthetic(True, rng)).save(cache.waterfall_path(obs))
        queue = Manifest(tmp_path / "unvetted.jsonl")
        added = fetch_unvetted(ManifestSource(listing), cache, queue, count=10, skip={5})
        assert added == 2 and sorted(o.id for o in queue) == [1, 3]

    def test_session_has_two_modes(self, downloaded):
        manifest, cache, root = downloaded
        rng = np.random.default_rng(9)
        fresh = []
        for i in range(500, 510):
            obs = unvetted_obs(i)
            WaterfallImage(synthetic(i % 2 == 0, rng)).save(cache.waterfall_path(obs))
            fresh.append(obs)
        session = VettingSession.from_cache(manifest, cache, unvetted=fresh)
        assert session.mode == "unvetted" and len(session) == 10 and session.count("test") == 60
        assert all(not i.known for i in session.ordered())
        error_unvetted = session.stats()["confident_error"]
        session.set_mode("test")
        assert len(session) == 60 and all(i.known for i in session.ordered())
        assert session.stats()["confident_error"] == error_unvetted  # always measured on the test set
        session.set_mode("unvetted")
        session.auto_vet()
        assert all(session.decision(i.obs.id) is None for i in session.ordered("unsure first")
                   if i.known)  # auto-vet touched only this mode
        with pytest.raises(ValidationError):
            session.set_mode("everything")


class TestMissionWindow:
    @pytest.fixture()
    def mission(self, downloaded, qapp):
        from PySide6.QtCore import QCoreApplication, QEvent

        from satrecovery.vetting.gui.mission import MissionWindow

        manifest, cache, root = downloaded
        session = VettingSession.from_cache(manifest, cache, Policy(0.3, 0.7))
        win = MissionWindow(session, explainer=None)
        yield win, session
        win.close()
        win.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()

    def test_opens_on_the_launcher_then_how_it_works_then_home(self, mission):
        win, _ = mission
        assert win.stack.currentWidget() is win.launcher
        win.launcher.live.emit()
        assert win.stack.currentWidget() is win.intro
        win.intro.start.emit()
        assert win.stack.currentWidget() is win.home
        win.home.back.emit()
        win.launcher.live.emit()          # the intro is shown once
        assert win.stack.currentWidget() is win.home

    def test_replay_plays_to_a_plain_verdict(self, mission):
        win, session = mission
        obs_id = session.ordered()[0].obs.id
        win.open_replay(obs_id)
        assert win.stack.currentWidget() is win.replay
        assert f"#{obs_id}" in win.replay.subtitle.text()
        assert "Listening" in win.replay.headline.text() or "Working" in win.replay.sentence.text()
        win.replay.finish()
        assert win.replay.symbol.text() and "Listening" not in win.replay.headline.text()
        win.replay.back.emit()
        assert win.stack.currentWidget() is win.home

    def test_world_map_has_land(self):
        from satrecovery.vetting.gui.mission import _land

        rings = _land()
        assert len(rings) > 100 and all(r.shape[1] == 2 for r in rings)
