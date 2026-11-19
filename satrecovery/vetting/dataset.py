"""Training and test sets of vetted waterfalls, split fairly.

The trap with SatNOGS data is that one ground station's waterfalls all
look alike: the same antenna, the same local interference, the same
noise floor.  Split observations at random and a station's quirks land
in both halves, so the test score rewards recognising stations, not
signals.  :class:`StationSplit` keeps every station wholly on one side,
which is the honest question: *does it work on a station it has never
seen?*

DESIGN NOTE: STRATEGY + TEMPLATE METHOD
---------------------------------------
:class:`VettingSplit` is the strategy interface.  Its public
:meth:`~VettingSplit.split` is a template: it calls the subclass's
:meth:`~VettingSplit._choose`, then checks the answer -- no index on
both sides, none lost -- so no subclass can leak test data into training
by mistake.  :class:`StationSplit` and :class:`TimeSplit` only choose.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np

from ..core.exceptions import ValidationError
from .artifacts import ArtifactCache, ArtifactError
from .observation import Observation, WaterfallLabel
from .source import Manifest

#: target values: 1 = the satellite's signal is there
SIGNAL, NO_SIGNAL = 1, 0


def normalise(power: np.ndarray) -> np.ndarray:
    """Each waterfall on its own scale: median 0, spread 1.

    SatNOGS draws every picture with its own colour-bar range, so the same
    brightness means different power in different pictures.  Only the
    pattern inside one picture can be compared, so each is standardised
    by its own median and spread (robust to the bright signal itself).
    """
    power = np.asarray(power, dtype=np.float32)
    centre = np.median(power)
    spread = np.subtract(*np.percentile(power, [75, 25])) / 1.349
    return ((power - centre) / max(float(spread), 1e-3)).astype(np.float32)


@dataclass(frozen=True)
class VettingSet:
    """Waterfalls ``(n, 1, time, freq)``, labels ``(n,)`` and the observations they came from."""

    inputs: np.ndarray
    targets: np.ndarray
    observations: tuple[Observation, ...]

    def __post_init__(self) -> None:
        if not (len(self.inputs) == len(self.targets) == len(self.observations)):
            raise ValidationError("VettingSet", (len(self.inputs), len(self.targets), len(self.observations)),
                                  "inputs, targets and observations must line up")

    def __len__(self) -> int:
        return len(self.targets)

    def subset(self, index: Sequence[int] | np.ndarray) -> "VettingSet":
        index = np.asarray(index, dtype=np.int64)
        return VettingSet(self.inputs[index], self.targets[index], tuple(self.observations[i] for i in index))

    @property
    def share_with_signal(self) -> float:
        return float(np.mean(self.targets == SIGNAL)) if len(self) else 0.0

    @classmethod
    def build(cls, manifest: Manifest | Sequence[Observation], cache: ArtifactCache) -> "VettingSet":
        """Every human-labelled observation whose waterfall is in the cache."""
        inputs, targets, kept = [], [], []
        for obs in manifest:
            if not obs.is_trainable or not cache.has_waterfall(obs):
                continue
            try:
                power = cache.waterfall(obs).power
            except (ArtifactError, OSError, ValueError):
                continue
            inputs.append(normalise(power)[None])
            targets.append(SIGNAL if obs.waterfall_label is WaterfallLabel.SIGNAL else NO_SIGNAL)
            kept.append(obs)
        if not kept:
            raise ValidationError("manifest", 0, "no labelled observation has a downloaded waterfall")
        shapes = {x.shape for x in inputs}
        if len(shapes) > 1:
            raise ValidationError("waterfalls", sorted(shapes), "all waterfalls must be the same size")
        return cls(np.stack(inputs), np.asarray(targets, dtype=np.int64), tuple(kept))


# ----------------------------------------------------------------------
class VettingSplit(ABC):
    """Splits observations into train and test, by some rule."""

    def split(self, observations: Sequence[Observation]) -> tuple[np.ndarray, np.ndarray]:
        train, test = self._choose(observations)
        train, test = np.sort(np.asarray(train, dtype=np.int64)), np.sort(np.asarray(test, dtype=np.int64))
        if np.intersect1d(train, test).size:
            raise ValidationError("split", "overlap", "an observation is on both sides")
        if len(train) + len(test) != len(observations):
            raise ValidationError("split", (len(train), len(test)), f"must cover all {len(observations)}")
        return train, test

    @abstractmethod
    def _choose(self, observations: Sequence[Observation]) -> tuple[Sequence[int], Sequence[int]]:
        """Indices for training and for testing."""


class StationSplit(VettingSplit):
    """Whole ground stations go to the test side until it holds ``test_share``."""

    def __init__(self, test_share: float = 0.2, seed: int = 0) -> None:
        if not 0.0 < test_share < 1.0:
            raise ValidationError("test_share", test_share, "must be between 0 and 1")
        self._share = float(test_share)
        self._seed = int(seed)

    def _choose(self, observations):
        stations: dict[object, list[int]] = {}
        for i, obs in enumerate(observations):
            stations.setdefault(obs.ground_station, []).append(i)
        names = list(stations)
        np.random.default_rng(self._seed).shuffle(names)
        target = self._share * len(observations)
        test: list[int] = []
        for name in names:
            if len(test) >= target:
                break
            if len(test) + len(stations[name]) > 1.5 * target and test:
                continue  # one huge station would swallow the test set; try a smaller one
            test.extend(stations[name])
        test_set = set(test)
        return [i for i in range(len(observations)) if i not in test_set], test

    @staticmethod
    def stations(observations: Sequence[Observation], index: np.ndarray) -> set:
        return {observations[i].ground_station for i in index}


class TimeSplit(VettingSplit):
    """Train on observations before ``cutoff``, test on the ones after."""

    def __init__(self, cutoff: datetime) -> None:
        self._cutoff = cutoff

    def _choose(self, observations):
        train, test = [], []
        for i, obs in enumerate(observations):
            (test if obs.start is not None and obs.start >= self._cutoff else train).append(i)
        return train, test
