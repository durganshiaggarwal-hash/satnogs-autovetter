"""Five different ways of knowing you might be wrong.

Each one is cheap to write because :class:`ConfidenceEstimator` already
owns the housekeeping; each one is a genuinely different *theory* of
where self-knowledge comes from.

:class:`LLRConfidence`
    From the decision margin, expressed as odds.  The receiver's own
    opinion, made explicit so it can be graded alongside the others.

:class:`EVMConfidence`
    From signal quality: how far the symbol landed from any legal
    point.  Knows nothing about the decision it is judging, which means
    it works on a receiver that reports no confidence at all -- and
    which is also its weakness.

:class:`MarginConfidence`
    From how clearly the winner beat the runner-up.  The rule the hard
    demapper uses, pulled out so it can be applied to any receiver.

:class:`MCDropoutConfidence`
    From the network's own instability: run it several times with parts
    of it switched off at random and see whether the answer holds.

:class:`ModelEnsembleConfidence`
    From disagreement between separately trained models.  The most
    expensive and, in the published literature, usually the best.

The last two can do something none of the first three can: report a
confidence **below 0.5**, meaning they positively disagree with the bit
the receiver output.  Whether that extra expressiveness is worth the
compute is a Phase 5 measurement, not an assumption.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from ..core.exceptions import ValidationError
from ..modulation.base import Modulation
from ..receivers.demapper import Demapper
from .base import ConfidenceEstimator, SymbolQualityEstimator

#: Shared with the demappers: beyond this the sigmoid saturates.
_MAX_LLR = 30.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic function, written to avoid overflow on large negatives."""
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


class LLRConfidence(ConfidenceEstimator):
    """The receiver's own odds, turned into a probability.

    A log-likelihood ratio of ``L`` means the favoured bit is ``e**|L|``
    times more likely than the other.  The logistic function turns that
    into the probability of being right, which is the number every other
    estimator here is competing with.

    This is the baseline, and in the one case where the theory is exact
    -- additive Gaussian noise, nothing else wrong, noise power known --
    it is provably the best any estimator can do.  Phase 5's question is
    what happens everywhere else, where the noise power is *estimated*
    and the damage is not only Gaussian.
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name or "LLR")

    def _scores(self, result, modulation):
        llr = self._require_llr(result, self._name)
        return _sigmoid(np.abs(llr))


class EVMConfidence(SymbolQualityEstimator):
    """From how far the symbol landed from the nearest legal point.

    Error vector magnitude is the standard measure of signal quality in
    radio engineering, and every piece of test equipment reports it.
    The appeal here is that it needs nothing from the receiver except
    the symbols: no log-likelihood ratios, no probability model, no
    cooperation at all.  It can therefore judge the end-to-end neural
    receiver, a black box bought from a vendor, or a hardware
    demodulator that reports only bits.

    The mapping from distance to probability is a **deliberately
    unprincipled** squashing function::

        confidence = 0.5 + 0.5 * exp(-(d / (scale * d_min/2))**2)

    where ``d_min/2`` is how far a symbol may stray before the decision
    flips.  There is no claim that the output is a probability, and the
    demo shows that it is not: it is badly calibrated almost everywhere.

    That is the point of including it.  Calibration is fixable after the
    fact -- :class:`~satrecovery.confidence.calibration.TemperatureScaling`
    does it in one parameter -- while a bad *ranking* is not fixable at
    all.  Separating a confidence that is merely mis-scaled from one
    that is genuinely uninformative is most of what Phase 5 is for, and
    this estimator is the clean example of the first kind.

    Parameters
    ----------
    scale:
        Width of the squashing function, in units of half the minimum
        distance.  Larger means more forgiving.  Its value changes the
        calibration and cannot change the ranking, which is exactly the
        property the tests assert.
    """

    def __init__(self, scale: float = 1.0, name: str | None = None) -> None:
        scale = float(scale)
        if scale <= 0.0:
            raise ValidationError("scale", scale, "must be positive")
        super().__init__(name or "EVM")
        self._scale = scale

    @property
    def scale(self) -> float:
        """Width of the squashing function."""
        return self._scale

    def _symbol_scores(self, symbols, modulation):
        distances = modulation.decision_distances(symbols)
        reference = self._scale * 0.5 * modulation.minimum_distance
        return 0.5 + 0.5 * np.exp(-((distances / reference) ** 2))


class MarginConfidence(SymbolQualityEstimator):
    """From how clearly the nearest point beat the runner-up.

    Where :class:`EVMConfidence` asks *how damaged is this symbol?*,
    this asks *how close was the call?* -- which is a different question
    with a different failure mode.  A symbol can land far from every
    point, giving terrible EVM, and still be an easy decision if it
    drifted directly away from the constellation rather than towards a
    neighbour.  On a channel with a gain error, that happens to every
    symbol at once.

    The rule is the one
    :class:`~satrecovery.receivers.demapper.HardDemapper` uses
    internally.  Lifting it out into an estimator means it can be
    applied to the output of any receiver, including the soft and
    learned ones that never compute it.
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name or "Margin")

    def _symbol_scores(self, symbols, modulation):
        constellation = modulation.constellation
        if constellation.size < 2:
            return np.ones(symbols.size, dtype=float)

        distances = np.abs(symbols[:, None] - constellation[None, :])
        nearest_two = np.partition(distances, 1, axis=1)[:, :2]
        margin = nearest_two[:, 1] - nearest_two[:, 0]

        scaled = margin / (0.5 * modulation.minimum_distance)
        return 0.5 + 0.5 * np.clip(scaled, 0.0, 1.0)


class SampledConfidence(ConfidenceEstimator):
    """Base for estimators built from several predictions of the same bit.

    Dropout sampling and model ensembles differ in where the several
    predictions come from, and in nothing else.  Both produce an array
    of shape ``(passes, n_bits)`` holding each pass's probability that
    the bit is a one; both then have to turn that into a single
    confidence in the bit the receiver actually reported.

    So the second half lives here, with two ways of doing it:

    ``"mean"``
        Average the predicted probabilities, then read off how much
        probability the pool gave to the receiver's answer.  Standard
        practice, and what the deep-ensembles literature reports.
    ``"spread"``
        The same, then pulled towards 0.5 in proportion to how much the
        passes disagreed.  Encodes the intuition that unanimous
        confidence and averaged-out confidence are not the same thing,
        which the mean alone cannot distinguish.

    Both are offered because the project is supposed to find out which
    works, not announce it.
    """

    _AGGREGATIONS = ("mean", "spread")

    def __init__(
        self,
        passes: int,
        aggregate: str = "mean",
        name: str | None = None,
    ) -> None:
        passes = int(passes)
        if passes < 2:
            raise ValidationError(
                "passes", passes, "need at least two predictions to compare"
            )
        if aggregate not in self._AGGREGATIONS:
            raise ValidationError(
                "aggregate", aggregate, f"expected one of {list(self._AGGREGATIONS)}"
            )
        super().__init__(name)
        self._passes = passes
        self._aggregate = aggregate

    @property
    def passes(self) -> int:
        """How many predictions are pooled."""
        return self._passes

    @property
    def aggregate(self) -> str:
        """How the pooled predictions are reduced to one number."""
        return self._aggregate

    # ------------------------------------------------------------------
    @abstractmethod
    def _sample(
        self, result: Any, modulation: Modulation
    ) -> np.ndarray:
        """Return ``(passes, n_bits)`` probabilities that each bit is a one."""

    def _scores(self, result, modulation):
        samples = np.asarray(self._sample(result, modulation), dtype=float)
        if samples.ndim != 2 or samples.shape[1] != result.n_bits:
            raise ValidationError(
                f"{self._name}._sample",
                samples.shape,
                f"expected shape (passes, {result.n_bits})",
            )

        mean_p1 = np.mean(samples, axis=0)
        bits = np.asarray(result.bits, dtype=float)

        # Probability the pool assigned to the bit the receiver chose.
        # This is allowed to fall below 0.5, which is the whole reason
        # these estimators can outrank the receiver's own numbers.
        agreement = np.where(bits > 0.5, mean_p1, 1.0 - mean_p1)

        if self._aggregate == "mean":
            return agreement

        # Disagreement between passes, scaled so that total disagreement
        # (half the passes each way) pulls the score all the way to 0.5.
        # The maximum standard deviation of a probability in [0, 1] is
        # 0.5, which makes the normalisation exact rather than a guess.
        spread = np.std(samples, axis=0) / 0.5
        return 0.5 + (agreement - 0.5) * np.clip(1.0 - spread, 0.0, 1.0)


class MCDropoutConfidence(SampledConfidence):
    """From a network's instability when parts of it are switched off.

    Dropout is normally a training trick: random units are disabled each
    step so the network cannot lean on any one of them.  Leaving it
    switched on at *inference* turns one network into a population of
    slightly different networks, and the spread of their answers is a
    cheap stand-in for the spread a properly Bayesian treatment would
    give.

    Cheap is the operative word.  It needs one trained model rather than
    five, and the extra cost is one forward pass per sample.  The
    literature finds it is usually worse than a real ensemble and much
    better than nothing, and Phase 5 checks whether that holds here.

    The model must have been built with ``dropout > 0``; a network with
    no dropout layers produces identical passes and the estimator says
    so rather than silently returning a constant.
    """

    def __init__(
        self,
        model: Any,
        passes: int = 32,
        aggregate: str = "mean",
        seed: int = 0,
        name: str | None = None,
    ) -> None:
        if not hasattr(model, "sample_bit_probabilities"):
            raise ValidationError(
                "model",
                type(model).__name__,
                "needs a sample_bit_probabilities(symbols, passes, seed) method "
                "(MLPDemapper has one)",
            )
        if getattr(model, "dropout", 0.0) <= 0.0:
            raise ValidationError(
                "model.dropout",
                getattr(model, "dropout", None),
                "dropout sampling needs a model built with dropout > 0",
            )
        super().__init__(passes, aggregate, name or f"MC-dropout x{passes}")
        self._model = model
        self._seed = int(seed)

    @property
    def model(self) -> Any:
        """The model being sampled."""
        return self._model

    def _sample(self, result, modulation):
        symbols = self._require_symbols(result, self._name)
        return self._model.sample_bit_probabilities(
            symbols, passes=self._passes, seed=self._seed
        )


class ModelEnsembleConfidence(SampledConfidence):
    """From disagreement between separately trained models.

    Several demappers, trained from different random starts on
    different shuffles of the data, will agree on the easy bits and
    diverge on the hard ones.  That divergence is the signal.

    Unlike dropout sampling the members are genuinely independent
    models, and unlike both of the symbol-quality estimators this one
    can notice a bit that the receiver got wrong for a *systematic*
    reason -- a channel the receiver was not trained for -- because the
    members will have absorbed that failure differently.

    Any :class:`~satrecovery.receivers.demapper.Demapper` can be a
    member, learned or classical.  A soft demapper and three networks is
    a perfectly good ensemble, and arguably a better one than four
    networks, because its members fail in unrelated ways.
    """

    def __init__(
        self,
        members: list[Demapper],
        aggregate: str = "mean",
        name: str | None = None,
    ) -> None:
        members = list(members)
        for member in members:
            if not isinstance(member, Demapper):
                raise ValidationError(
                    "members",
                    type(member).__name__,
                    "every member must be a Demapper",
                )
        super().__init__(
            len(members), aggregate, name or f"Ensemble x{len(members)}"
        )
        self._members = members

    @property
    def members(self) -> list[Demapper]:
        """Copy of the list of demappers in the ensemble."""
        return list(self._members)

    def _sample(self, result, modulation):
        symbols = self._require_symbols(result, self._name)
        rows = []
        for member in self._members:
            bits, confidence, llr = member.demap(symbols, modulation)
            if llr is not None:
                # Project convention: a positive ratio favours a zero.
                rows.append(_sigmoid(-np.asarray(llr, dtype=float)))
            else:
                # A hard demapper gives no odds, only a decision and how
                # sure it is; that is enough to reconstruct the implied
                # probability that the bit is a one.
                bits = np.asarray(bits, dtype=float)
                confidence = np.asarray(confidence, dtype=float)
                rows.append(np.where(bits > 0.5, confidence, 1.0 - confidence))
        return np.stack(rows, axis=0)


__all__ = [
    "EVMConfidence",
    "LLRConfidence",
    "MCDropoutConfidence",
    "MarginConfidence",
    "ModelEnsembleConfidence",
    "SampledConfidence",
]
