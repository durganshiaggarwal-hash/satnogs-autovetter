"""Tests for the vetting dataset, splits and baseline model (no network)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from satrecovery.core.exceptions import ModelNotTrainedError, ValidationError
from satrecovery.vetting import (
    NO_SIGNAL,
    SIGNAL,
    ArtifactCache,
    Manifest,
    Observation,
    StationSplit,
    TimeSplit,
    VettingSet,
    WaterfallImage,
    WaterfallVetter,
    normalise,
)
from satrecovery.vetting.model import pick_device


def obs(i: int, station: int, signal: bool = True, start: str = "2026-11-10T00:00:00Z") -> Observation:
    return Observation.from_api({
        "id": i, "ground_station": station, "norad_cat_id": 100 + i % 3, "start": start,
        "end": "2026-11-10T00:08:00Z", "waterfall": f"https://cdn.example/{i}.png",
        "waterfall_status": "with-signal" if signal else "without-signal", "waterfall_status_user": 1,
    })


def synthetic(signal: bool, rng: np.random.Generator, size: int = 64) -> np.ndarray:
    """Noise, plus -- if ``signal`` -- a bright burst train near the centre frequency."""
    power = rng.normal(0.4, 0.05, (size, size))
    if signal:
        centre = size // 2 + rng.integers(-3, 4)
        for row in rng.choice(size - 4, size=4, replace=False):
            power[row:row + 3, centre - 3:centre + 3] += 0.4
    return np.clip(power, 0, 1).astype(np.float32)


class TestNormalise:
    def test_same_pattern_on_different_scales_matches(self):
        rng = np.random.default_rng(0)
        power = synthetic(True, rng)
        assert np.allclose(normalise(power), normalise(power * 0.5 + 0.2), atol=1e-4)

    def test_flat_picture_does_not_divide_by_zero(self):
        assert np.all(np.isfinite(normalise(np.full((10, 10), 0.3))))


class TestVettingSet:
    def test_builds_from_the_cache_and_skips_what_is_missing(self, tmp_path):
        manifest = Manifest(tmp_path / "m.jsonl")
        cache = ArtifactCache(tmp_path)
        rng = np.random.default_rng(1)
        for i in range(1, 7):
            o = obs(i, station=i % 2, signal=i % 2 == 0)
            manifest.add(o)
            if i != 6:  # observation 6 was never downloaded
                WaterfallImage(synthetic(o.waterfall_label.value == "with-signal", rng)).save(cache.waterfall_path(o))
        data = VettingSet.build(manifest, cache)
        assert len(data) == 5 and data.inputs.shape == (5, 1, 64, 64)
        assert list(data.targets) == [NO_SIGNAL, SIGNAL, NO_SIGNAL, SIGNAL, NO_SIGNAL]
        assert data.subset([1, 3]).targets.tolist() == [SIGNAL, SIGNAL]

    def test_empty_is_an_error(self, tmp_path):
        with pytest.raises(ValidationError):
            VettingSet.build(Manifest(tmp_path / "m.jsonl"), ArtifactCache(tmp_path))

    def test_parts_must_line_up(self):
        with pytest.raises(ValidationError):
            VettingSet(np.zeros((2, 1, 8, 8)), np.zeros(3), (obs(1, 1), obs(2, 1)))


class TestSplits:
    def test_no_station_is_on_both_sides(self):
        observations = [obs(i, station=i % 10) for i in range(1, 201)]
        train, test = StationSplit(0.2, seed=3).split(observations)
        assert not StationSplit.stations(observations, train) & StationSplit.stations(observations, test)
        assert 0.1 <= len(test) / 200 <= 0.35

    def test_one_huge_station_does_not_become_the_whole_test_set(self):
        observations = [obs(i, station=0) for i in range(1, 151)] + [obs(i, station=i) for i in range(151, 201)]
        for seed in range(5):
            _, test = StationSplit(0.2, seed=seed).split(observations)
            assert len(test) < 100

    def test_time_split(self):
        early = [obs(i, 1, start="2026-11-01T00:00:00Z") for i in range(1, 4)]
        late = [obs(i, 1, start="2026-11-15T00:00:00Z") for i in range(4, 6)]
        train, test = TimeSplit(datetime(2026, 11, 10, tzinfo=timezone.utc)).split(early + late)
        assert train.tolist() == [0, 1, 2] and test.tolist() == [3, 4]

    def test_share_is_checked(self):
        with pytest.raises(ValidationError):
            StationSplit(1.5)


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(7)
    labels = np.array([i % 2 for i in range(240)])
    inputs = np.stack([normalise(synthetic(bool(y), rng))[None] for y in labels])
    return inputs, labels


class TestWaterfallVetter:
    def test_refuses_to_answer_untrained(self):
        with pytest.raises(ModelNotTrainedError):
            WaterfallVetter(size=32, device=pick_device(False)).predict_proba(np.zeros((1, 1, 64, 64)))

    def test_learns_an_easy_signal_and_survives_save_load(self, toy, tmp_path):
        inputs, labels = toy
        model = WaterfallVetter(channels=(8, 16), size=64, device=pick_device(False), seed=1)
        model.fit(inputs[:200], labels[:200], validation=(inputs[200:], labels[200:]), epochs=8, verbose=False)
        probability = model.predict_proba(inputs[200:])
        assert probability.shape == (40,) and np.all((0 <= probability) & (probability <= 1))
        assert np.mean((probability >= 0.5) == labels[200:]) > 0.9

        again = WaterfallVetter.from_file(model.save(tmp_path / "v.pt"), device=pick_device(False))
        assert np.allclose(again.predict_proba(inputs[200:]), probability, atol=1e-5)

    def test_checks_its_inputs(self, toy):
        inputs, labels = toy
        model = WaterfallVetter(channels=(8,), size=32, device=pick_device(False))
        with pytest.raises(ValidationError):
            model.fit(inputs[:, 0], labels)
        with pytest.raises(ValidationError):
            model.fit(inputs, labels + 1)


# ----------------------------------------------------------------------
from satrecovery.vetting import FEATURES, FeatureVetter, Vetter, feature_matrix, waterfall_features  # noqa: E402


def moving_track(rng: np.random.Generator, size: int = 64) -> np.ndarray:
    """Another satellite's S-curve sweeping across the band -- "without signal"."""
    power = rng.normal(0.4, 0.05, (size, size))
    for row in range(size):
        col = int(size / 2 + 0.35 * size * np.tanh((row - size / 2) / (size / 6)))
        power[row, max(col - 1, 0):col + 1] += 0.5
    return np.clip(power, 0, 1).astype(np.float32)


class TestFeatures:
    def test_one_value_per_named_feature(self):
        rng = np.random.default_rng(0)
        values = waterfall_features(normalise(synthetic(True, rng)))
        assert values.shape == (len(FEATURES),) and np.all(np.isfinite(values))

    def test_bursts_at_the_centre_light_the_centre_band(self):
        rng = np.random.default_rng(1)
        names = list(FEATURES)
        signal = waterfall_features(normalise(synthetic(True, rng)))
        noise = waterfall_features(normalise(synthetic(False, rng)))
        lit = names.index("centre_lit_rows")
        assert signal[lit] > noise[lit]
        assert signal[names.index("centre_excess")] > noise[names.index("centre_excess")]

    def test_a_moving_track_wanders_and_a_centred_one_does_not(self):
        rng = np.random.default_rng(2)
        names = list(FEATURES)
        moving = waterfall_features(normalise(moving_track(rng)))
        steady = waterfall_features(normalise(synthetic(True, rng)))
        assert moving[names.index("track_wander")] > steady[names.index("track_wander")]
        assert moving[names.index("track_steadiness")] < steady[names.index("track_steadiness")]

    def test_flat_and_tiny_inputs(self):
        assert np.all(np.isfinite(waterfall_features(np.zeros((32, 32)))))
        with pytest.raises(ValidationError):
            waterfall_features(np.zeros((8, 8)))
        assert feature_matrix(np.zeros((3, 1, 32, 32))).shape == (3, len(FEATURES))


class TestFeatureVetter:
    def test_is_a_vetter_and_refuses_untrained(self):
        model = FeatureVetter()
        assert isinstance(model, Vetter)
        with pytest.raises(ModelNotTrainedError):
            model.predict_proba(np.zeros((1, 1, 32, 32)))

    def test_learns_and_explains_and_round_trips(self, toy, tmp_path):
        inputs, labels = toy
        model = FeatureVetter(n_trees=50, seed=0)
        model.fit(inputs[:200], labels[:200])
        probability = model.predict_proba(inputs[200:])
        assert np.mean(model.decide(inputs[200:]) == labels[200:]) > 0.9
        assert set(model.importances()) == set(FEATURES)
        assert abs(sum(model.importances().values()) - 1.0) < 1e-6
        again = FeatureVetter.from_file(model.save(tmp_path / "f.pt"))
        assert np.allclose(again.predict_proba(inputs[200:]), probability)

    def test_needs_both_labels(self, toy):
        inputs, labels = toy
        with pytest.raises(ValidationError):
            FeatureVetter().fit(inputs[labels == 1], labels[labels == 1])

    def test_a_cnn_and_a_forest_answer_the_same_way(self, toy):
        inputs, labels = toy
        vetters: list[Vetter] = [FeatureVetter(n_trees=20), WaterfallVetter(channels=(8,), size=32, device=pick_device(False))]
        for vetter in vetters:
            vetter.fit(inputs[:100], labels[:100], **({"epochs": 1, "verbose": False} if isinstance(vetter, WaterfallVetter) else {}))
            assert vetter.predict_proba(inputs[:5]).shape == (5,)
            assert set(vetter.decide(inputs[:5]).tolist()) <= {0, 1}
