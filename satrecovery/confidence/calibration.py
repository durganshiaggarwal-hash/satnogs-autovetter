"""Fixing a confidence that is informative but dishonest.

There are two separate things wrong with a typical receiver's
confidence, and they need separating because only one of them is
repairable.

**Mis-ranking** is when the receiver is no more confident about the bits
it got right than the ones it got wrong.  Nothing applied afterwards can
fix that.  The information was never there.

**Mis-scaling** is when the ranking is fine but the numbers are on the
wrong scale -- the receiver says 0.999 for bits it gets right 97% of the
time.  That is repairable, cheaply, and that is what this module does.

Every calibrator here is **monotone**: it maps higher confidence to
higher confidence.  Three consequences follow, and they are the reason
the project separates the two problems in the first place:

1. No bit decision changes.  The bit error rate after calibration is
   exactly the bit error rate before it.
2. The ranking does not change, so the risk-coverage curve and every
   number derived from it are **identical** before and after.
3. Only the calibration numbers move.

So a calibrator can make a receiver honest.  It cannot make it better.
The tests assert all three.

DESIGN NOTE: ABSTRACTION PLUS A LIFECYCLE, SHARED WITH THE MODELS
-------------------------------------------------------------------
A calibrator has to be *fitted* before use, which makes it the same kind
of object as a learned model: it has a state, it is useless before
training, and using it untrained silently produces nonsense rather than
an error.  Phase 4 solved that with
:class:`~satrecovery.learning.base.TrainableModel`, and the same guard
is used here rather than invented again.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..core.exceptions import ModelNotTrainedError, ValidationError

#: Probabilities are held away from the ends before taking logs.
_EPS = 1e-9

#: Temperature is searched over this range of powers of ten.  A
#: temperature of 10 flattens almost anything to near 0.5; 0.1 sharpens
#: almost anything to near certainty.  Nothing useful lies outside.
_LOG_T_RANGE = (-1.0, 1.0)


def _to_logit(confidence: np.ndarray) -> np.ndarray:
    """Turn a probability into log-odds, safely."""
    p = np.clip(np.asarray(confidence, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


class Calibrator(ABC):
    """Rescales confidence numbers so they mean what they say.

    Subclasses implement :meth:`_fit` and :meth:`_transform`; the base
    class owns validation, the fitted-state guard, and the public
    methods, so a subclass is a few lines of arithmetic.
    """

    def __init__(self, name: str | None = None) -> None:
        self._name = str(name) if name else type(self).__name__
        self._fitted = False

    @property
    def name(self) -> str:
        """Short name, for labelling results."""
        return self._name

    @property
    def is_fitted(self) -> bool:
        """Has this calibrator been shown held-out data yet?"""
        return self._fitted

    # ------------------------------------------------------------------
    # Public interface -- fixed here
    # ------------------------------------------------------------------
    def fit(self, confidence: np.ndarray, correct: np.ndarray) -> "Calibrator":
        """Learn the correction from decisions whose outcome is known.

        The data passed here must be **held out**.  Fitting a calibrator
        on the same frames it will be judged on measures how well it can
        memorise, not how well it calibrates, and the demo keeps the two
        sets strictly apart.

        Returns ``self``, so a calibrator can be built and fitted in one
        expression.
        """
        confidence = np.asarray(confidence, dtype=float).ravel()
        correct = np.asarray(correct).ravel()

        if confidence.size != correct.size:
            raise ValidationError(
                "correct",
                correct.size,
                f"must have one outcome per confidence value ({confidence.size})",
            )
        if confidence.size == 0:
            raise ValidationError("confidence", 0, "cannot fit on an empty set")
        if confidence.min() < 0.0 or confidence.max() > 1.0:
            raise ValidationError(
                "confidence", "<array>", "values must be probabilities in [0, 1]"
            )

        self._fit(confidence, correct.astype(bool))
        self._fitted = True
        return self

    def apply(self, confidence: np.ndarray) -> np.ndarray:
        """Correct a set of confidence values."""
        if not self._fitted:
            raise ModelNotTrainedError(self._name)
        confidence = np.asarray(confidence, dtype=float).ravel()
        if confidence.size and (confidence.min() < 0.0 or confidence.max() > 1.0):
            raise ValidationError(
                "confidence", "<array>", "values must be probabilities in [0, 1]"
            )
        return np.clip(self._transform(confidence), 0.0, 1.0)

    def __call__(self, confidence: np.ndarray) -> np.ndarray:
        """``calibrator(c)`` is :meth:`apply`, so one can be passed as a function."""
        return self.apply(confidence)

    # ------------------------------------------------------------------
    # What subclasses fill in
    # ------------------------------------------------------------------
    @abstractmethod
    def _fit(self, confidence: np.ndarray, correct: np.ndarray) -> None:
        """Learn whatever this calibrator needs from held-out outcomes."""

    @abstractmethod
    def _transform(self, confidence: np.ndarray) -> np.ndarray:
        """Apply the learned correction."""

    @staticmethod
    def _nll(confidence: np.ndarray, correct: np.ndarray) -> float:
        """Negative log likelihood, the quantity a calibrator minimises."""
        p = np.clip(confidence, _EPS, 1.0 - _EPS)
        return float(-np.mean(np.log(np.where(correct, p, 1.0 - p))))

    def __repr__(self) -> str:
        state = "fitted" if self._fitted else "unfitted"
        return f"{type(self).__name__}({state})"


class IdentityCalibrator(Calibrator):
    """Changes nothing.

    The control.  Every comparison of calibrators needs a row showing
    what happens when nothing is done, and making that row a real object
    rather than a special case in the reporting code means it travels
    through exactly the same path as the others -- including the
    held-out split, which is where accidental advantages usually hide.
    """

    def __init__(self) -> None:
        super().__init__("None")

    def _fit(self, confidence, correct):
        return None

    def _transform(self, confidence):
        return confidence


class TemperatureScaling(Calibrator):
    """Divides the log-odds by one learned number.

    The whole correction is::

        calibrated = sigmoid(logit(confidence) / T)

    with a single parameter ``T`` fitted by minimising the negative log
    likelihood on held-out decisions.  ``T > 1`` softens an
    overconfident receiver; ``T < 1`` sharpens a timid one.

    One parameter is the appeal.  It cannot overfit in any meaningful
    sense, it needs very little held-out data, and because dividing by a
    positive number preserves order it provably cannot change a single
    bit or reorder a single pair -- so a reader does not have to take on
    trust that the error rate is unaffected.

    The parameter is found by **golden-section search** over
    ``log10(T)``.  The negative log likelihood is convex in the
    temperature for this problem, so a bracketing search finds the
    optimum without gradients, without an optimiser, and without
    PyTorch -- which keeps calibration available even when the project
    is run on a machine with no deep-learning stack installed.
    """

    #: Golden-section search shrinks the bracket by this factor a step.
    _INV_PHI = (np.sqrt(5.0) - 1.0) / 2.0

    def __init__(self, iterations: int = 60, name: str | None = None) -> None:
        iterations = int(iterations)
        if iterations < 5:
            raise ValidationError("iterations", iterations, "too few to converge")
        super().__init__(name or "Temperature")
        self._iterations = iterations
        self._temperature = 1.0

    @property
    def temperature(self) -> float:
        """The fitted temperature. Above 1 means the source was overconfident."""
        return self._temperature

    def _fit(self, confidence, correct):
        logits = _to_logit(confidence)

        def loss(log_t: float) -> float:
            scaled = 1.0 / (1.0 + np.exp(-logits / (10.0**log_t)))
            return self._nll(scaled, correct)

        lo, hi = _LOG_T_RANGE
        a = hi - self._INV_PHI * (hi - lo)
        b = lo + self._INV_PHI * (hi - lo)
        fa, fb = loss(a), loss(b)

        for _ in range(self._iterations):
            if fa < fb:
                hi, b, fb = b, a, fa
                a = hi - self._INV_PHI * (hi - lo)
                fa = loss(a)
            else:
                lo, a, fa = a, b, fb
                b = lo + self._INV_PHI * (hi - lo)
                fb = loss(b)
            if hi - lo < 1e-6:
                break

        self._temperature = float(10.0 ** (0.5 * (lo + hi)))

    def _transform(self, confidence):
        return 1.0 / (1.0 + np.exp(-_to_logit(confidence) / self._temperature))

    def __repr__(self) -> str:
        if not self._fitted:
            return "TemperatureScaling(unfitted)"
        return f"TemperatureScaling(T={self._temperature:.3f})"


class HistogramBinning(Calibrator):
    """Replaces each confidence with the accuracy actually seen near it.

    Non-parametric, and so able to fix a distortion of any shape -- a
    receiver that is overconfident in the middle of its range and
    underconfident at the top is beyond temperature scaling and well
    within this.

    The cost is the usual one.  With many bins and little held-out data
    each bin's accuracy is itself a noisy estimate, and the calibrator
    starts fitting that noise.  It also cannot extrapolate: a confidence
    higher than anything seen during fitting is simply clamped to the
    top bin.

    Bins are equal-mass rather than equal-width, which matters a great
    deal here.  A demapper at useful signal-to-noise puts nearly every
    bit above 0.99, so equal-width bins would leave one bin holding 95%
    of the data and the correction would be a single number for almost
    the whole range -- a worse temperature scaling, with more ways to go
    wrong.
    """

    def __init__(self, n_bins: int = 20, name: str | None = None) -> None:
        n_bins = int(n_bins)
        if n_bins < 2:
            raise ValidationError("n_bins", n_bins, "need at least two bins")
        super().__init__(name or f"Histogram({n_bins})")
        self._n_bins = n_bins
        self._edges = np.array([0.0, 1.0])
        self._values = np.array([0.5])

    @property
    def n_bins(self) -> int:
        """How many bins were requested."""
        return self._n_bins

    @property
    def bin_values(self) -> np.ndarray:
        """Read-only fitted accuracy for each bin."""
        view = self._values.view()
        view.flags.writeable = False
        return view

    def _fit(self, confidence, correct):
        quantiles = np.linspace(0.0, 1.0, self._n_bins + 1)
        edges = np.unique(np.quantile(confidence, quantiles))
        if edges.size < 2:
            # Every value identical: one bin holding everything.
            edges = np.array([edges[0] - _EPS, edges[0] + _EPS])
        self._edges = edges

        index = np.digitize(confidence, edges[1:-1], right=True)
        n_bins = edges.size - 1
        counts = np.bincount(index, minlength=n_bins).astype(float)
        hits = np.bincount(index, weights=correct.astype(float), minlength=n_bins)

        # An empty bin inherits the midpoint of its own range rather
        # than a neighbour's accuracy, so a gap in the fitting data
        # cannot invent a correction that was never observed.
        midpoints = 0.5 * (edges[:-1] + edges[1:])
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.where(counts > 0, hits / np.maximum(counts, 1), midpoints)

        # Enforce monotonicity.  Without it a noisy bin can invert the
        # order of two confidences, which would change the ranking and
        # silently break the guarantee that calibration leaves the
        # risk-coverage curve alone.
        self._values = np.maximum.accumulate(values)

    def _transform(self, confidence):
        index = np.digitize(confidence, self._edges[1:-1], right=True)
        return self._values[np.clip(index, 0, self._values.size - 1)]

    def __repr__(self) -> str:
        state = f"{self._values.size} bins" if self._fitted else "unfitted"
        return f"HistogramBinning({state})"


__all__ = [
    "Calibrator",
    "HistogramBinning",
    "IdentityCalibrator",
    "TemperatureScaling",
]
