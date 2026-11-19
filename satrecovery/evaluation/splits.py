"""Dividing scenarios into the ones a model learns from and the ones it is judged on.

How the split is made decides what a result means.

A **random split** draws training and test scenarios from the same pool.
The test then asks: *having seen conditions like these, can the model
handle more of them?*  That is the easy question, and the one most
published results answer.

A **held-out split** keeps a whole region out of training -- every
scenario below 4 dB, or every scenario with Doppler -- and tests only
there.  That asks the question a deployed receiver actually faces:
*can it handle conditions nobody showed it?*

Phase 4 already saw one model collapse when moved from clean training
data to impaired test data.  This module makes that kind of test a
one-line choice instead of a special experiment, so every learned method
in the project can be put through it the same way.

DESIGN NOTE: THE STRATEGY PATTERN
---------------------------------
Each way of splitting is a class with one method, and the experiment
takes whichever one it is given::

    for strategy in (RandomSplit(0.3), HoldOutSNR(-10, 4), HoldOutProfile("full")):
        train, test = strategy.split(grid)
        ...

The experiment code does not change between strategies, which is what
makes their results comparable.  The base class also enforces the two
rules every split must obey -- the halves are disjoint, and neither is
empty -- so a subclass cannot produce a split that quietly leaks test
scenarios into training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.exceptions import ValidationError
from .scenarios import ScenarioGrid


class SplitStrategy(ABC):
    """Divides a grid into training and test scenarios."""

    def __init__(self, name: str | None = None) -> None:
        self._name = str(name) if name else type(self).__name__

    @property
    def name(self) -> str:
        return self._name

    def split(self, grid: ScenarioGrid) -> tuple[ScenarioGrid, ScenarioGrid]:
        """Return ``(train, test)``.

        Template method: the subclass chooses the test scenarios, and
        this method derives the training set as everything else, then
        checks the two rules.  Deriving training as the complement is
        what makes leakage impossible rather than merely unlikely.
        """
        if not isinstance(grid, ScenarioGrid):
            raise ValidationError("grid", type(grid).__name__, "expected a ScenarioGrid")

        test = self._choose_test(grid) & grid
        train = grid - test

        if len(test) == 0:
            raise ValidationError(
                "split", self._name, "selected no test scenarios from this grid"
            )
        if len(train) == 0:
            raise ValidationError(
                "split", self._name, "left nothing to train on"
            )
        # Holds by construction; asserted anyway because a leak here
        # would invalidate every result built on the split.
        assert len(train & test) == 0
        return train, test

    @abstractmethod
    def _choose_test(self, grid: ScenarioGrid) -> ScenarioGrid:
        """Pick the scenarios to hold back for testing."""

    def describe(self) -> str:
        """One line for result tables."""
        return self._name

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.describe()!r})"


class RandomSplit(SplitStrategy):
    """Hold back a random fraction of scenarios. The easy test.

    Splits *scenarios*, not frames.  Splitting frames would put frames
    from the same scenario on both sides, and the test would measure
    little more than whether the model can recognise conditions it has
    literally seen.
    """

    def __init__(self, test_fraction: float = 0.3, seed: int = 0) -> None:
        test_fraction = float(test_fraction)
        if not 0.0 < test_fraction < 1.0:
            raise ValidationError("test_fraction", test_fraction, "must lie in (0, 1)")
        super().__init__(f"random {test_fraction:.0%}")
        self._fraction = test_fraction
        self._seed = int(seed)

    def _choose_test(self, grid):
        rng = np.random.default_rng(self._seed)
        n_test = max(1, int(round(self._fraction * len(grid))))
        picks = rng.choice(len(grid), size=min(n_test, len(grid) - 1), replace=False)
        return ScenarioGrid(grid[int(i)] for i in picks)


class HoldOutSNR(SplitStrategy):
    """Test only on noise levels inside a band; train on everything else.

    The natural use is to hold out the *low* end -- train on reasonable
    signals, test on weak ones -- because that is where a deployed
    receiver meets conditions worse than anything in its training data.
    """

    def __init__(self, low_db: float, high_db: float) -> None:
        low_db, high_db = float(low_db), float(high_db)
        if low_db > high_db:
            raise ValidationError("low_db", low_db, f"must not exceed high_db ({high_db})")
        super().__init__(f"held-out SNR {low_db:+g} to {high_db:+g} dB")
        self._low = low_db
        self._high = high_db

    def _choose_test(self, grid):
        return grid.where(snr_min=self._low, snr_max=self._high)


class HoldOutProfile(SplitStrategy):
    """Test only on one impairment profile; train on the others.

    The hardest of the three for a learned model, and the most
    realistic: the satellite's Doppler or the multipath environment at a
    new ground station is exactly the kind of thing that was not in the
    training set.
    """

    def __init__(self, profile: str) -> None:
        profile = str(profile)
        super().__init__(f"held-out profile '{profile}'")
        self._profile = profile

    def _choose_test(self, grid):
        return grid.where(profile=self._profile)


__all__ = ["HoldOutProfile", "HoldOutSNR", "RandomSplit", "SplitStrategy"]
