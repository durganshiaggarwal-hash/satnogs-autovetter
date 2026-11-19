"""Measuring whether a confidence number means anything.

Phase 1 built a measuring apparatus before it built a receiver, and
checked it against textbook formulas first.  This module does the same
job for the project's second question.  Nothing here produces a
confidence; everything here *grades* one, and every class is checked in
the tests against a case whose answer is known in advance.

Two different questions, two classes, because they are genuinely
different and a receiver can pass one while failing the other.

:class:`CalibrationReport` asks **are the numbers honest?**
    When a receiver says it is 90% sure, is it right 90% of the time?
    A receiver can be useless and perfectly calibrated (always say 50%,
    be right half the time), so this is not a measure of quality.  It is
    a measure of truthfulness.

:class:`RiskCoverageCurve` asks **are the numbers useful?**
    If we throw away the bits the receiver is least sure about, does the
    error rate among the ones we keep actually fall?  A receiver can be
    badly calibrated and still rank its bits perfectly, which is all
    that matters if the number is used to decide what to discard or
    re-request.

Both are standard; neither is invented here.  What this project adds is
that both are applied to the *same* receivers under the *same* channel
conditions, so classical and learned methods can be compared on
self-knowledge rather than only on bit error rate.

DESIGN NOTE: ENCAPSULATION, AND WHY THESE ARE VALUE OBJECTS
------------------------------------------------------------
Both classes are immutable.  They are built once from predictions by a
classmethod, they expose read-only views of their arrays, and they have
no setters.  A report that could be edited after the fact is a report
nobody can trust -- and in a project whose whole argument is "we
measured it", the measurement objects are exactly the ones that must
not be quietly adjustable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.exceptions import ValidationError

#: Probabilities are clipped away from 0 and 1 before taking logs.
#: Without this a single confidently-wrong bit sends the negative log
#: likelihood to infinity and one outlier destroys the whole metric.
_EPS = 1e-12


def _check_pair(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate a (confidence, correctness) pair the same way everywhere."""
    confidence = np.asarray(confidence, dtype=float).ravel()
    correct = np.asarray(correct).ravel()

    if confidence.size != correct.size:
        raise ValidationError(
            "correct",
            correct.size,
            f"must have one outcome per confidence value ({confidence.size})",
        )
    if confidence.size == 0:
        raise ValidationError("confidence", 0, "cannot grade an empty set")
    if confidence.min() < 0.0 or confidence.max() > 1.0:
        raise ValidationError(
            "confidence", "<array>", "values must be probabilities in [0, 1]"
        )

    correct = correct.astype(bool)
    return confidence, correct


class CalibrationReport:
    """Are the stated probabilities honest?

    Built by sorting every decision into a bin by its stated confidence,
    then comparing, within each bin, the average stated confidence
    against the fraction actually correct.  A perfectly calibrated
    receiver sits on the diagonal: the bin where it claimed 0.8 is right
    80% of the time.

    The headline number is the **expected calibration error**, the
    average gap between claim and reality weighted by how many decisions
    fell in each bin.  Zero is perfect; the sign is lost, which is why
    :attr:`overconfidence` is reported alongside it -- claiming too much
    and claiming too little are both errors but they are not the same
    mistake, and in practice receivers almost always do the first.

    Parameters
    ----------
    confidence:
        Stated probability of being correct, one per decision.
    correct:
        Whether that decision was in fact right.
    n_bins:
        How many bins to sort the decisions into.
    binning:
        ``"width"`` splits the confidence range into equal slices --
        the textbook choice, and what published numbers use.
        ``"mass"`` instead puts an equal *number of decisions* in each
        bin.  The difference matters here more than in most fields: a
        demapper at high signal-to-noise puts almost every bit above
        0.999, so equal-width bins leave one bin holding everything and
        the resulting error is an average over a bin that is itself
        wildly varied.  Both are offered because disagreeing numbers
        from the two is itself a finding worth seeing.
    lo:
        Bottom of the binning range.  Defaults to 0.5 because a
        confidence below even odds would mean the receiver should have
        made the opposite decision; values below ``lo`` land in the
        first bin.
    """

    __slots__ = (
        "_confidence",
        "_correct",
        "_edges",
        "_bin_confidence",
        "_bin_accuracy",
        "_bin_counts",
        "_label",
    )

    def __init__(
        self,
        confidence: np.ndarray,
        correct: np.ndarray,
        n_bins: int = 15,
        binning: str = "width",
        lo: float = 0.5,
        label: str = "",
    ) -> None:
        confidence, correct = _check_pair(confidence, correct)
        n_bins = int(n_bins)
        if n_bins < 2:
            raise ValidationError("n_bins", n_bins, "need at least two bins")
        if binning not in ("width", "mass"):
            raise ValidationError("binning", binning, "expected 'width' or 'mass'")
        lo = float(lo)
        if not 0.0 <= lo < 1.0:
            raise ValidationError("lo", lo, "must lie in [0, 1)")

        self._confidence = confidence.copy()
        self._confidence.flags.writeable = False
        self._correct = correct.copy()
        self._correct.flags.writeable = False
        self._label = str(label)

        if binning == "width":
            edges = np.linspace(lo, 1.0, n_bins + 1)
        else:
            # Equal-mass bins: the edges are the quantiles of the data.
            # Duplicates are removed, so a distribution concentrated on a
            # few values honestly produces fewer bins rather than a row
            # of empty ones.
            quantiles = np.linspace(0.0, 1.0, n_bins + 1)
            edges = np.unique(np.quantile(confidence, quantiles))
            if edges.size < 2:
                edges = np.array([confidence[0] - 1e-9, confidence[0] + 1e-9])
        self._edges = edges
        self._edges.flags.writeable = False

        # ``np.digitize`` with right=True puts a value exactly on an edge
        # into the lower bin, which is what keeps the top bin from being
        # the only one that can contain confidence == 1.0.
        index = np.digitize(confidence, edges[1:-1], right=True)

        n_actual = edges.size - 1
        counts = np.bincount(index, minlength=n_actual).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            sum_conf = np.bincount(index, weights=confidence, minlength=n_actual)
            sum_corr = np.bincount(
                index, weights=correct.astype(float), minlength=n_actual
            )
            mean_conf = np.where(counts > 0, sum_conf / np.maximum(counts, 1), np.nan)
            mean_corr = np.where(counts > 0, sum_corr / np.maximum(counts, 1), np.nan)

        self._bin_counts = counts
        self._bin_counts.flags.writeable = False
        self._bin_confidence = mean_conf
        self._bin_confidence.flags.writeable = False
        self._bin_accuracy = mean_corr
        self._bin_accuracy.flags.writeable = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_result(
        cls,
        result: Any,
        true_bits: np.ndarray,
        confidence: np.ndarray | None = None,
        **options: Any,
    ) -> "CalibrationReport":
        """Grade a :class:`~satrecovery.receivers.result.RecoveryResult`.

        Pass ``confidence`` to grade a replacement estimate rather than
        the one the receiver itself reported -- which is exactly what
        comparing estimators on a single recovered frame requires.
        """
        scores = result.confidence if confidence is None else confidence
        options.setdefault("label", result.receiver)
        return cls(scores, ~result.error_mask(true_bits), **options)

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------
    @property
    def label(self) -> str:
        """What produced the numbers being graded."""
        return self._label

    @property
    def n_decisions(self) -> int:
        """How many decisions were graded."""
        return int(self._confidence.size)

    @property
    def bin_edges(self) -> np.ndarray:
        """Read-only bin boundaries, length ``n_bins + 1``."""
        return self._edges

    @property
    def bin_confidence(self) -> np.ndarray:
        """Mean stated confidence within each bin; NaN where empty."""
        return self._bin_confidence

    @property
    def bin_accuracy(self) -> np.ndarray:
        """Fraction actually correct within each bin; NaN where empty."""
        return self._bin_accuracy

    @property
    def bin_counts(self) -> np.ndarray:
        """How many decisions fell in each bin."""
        return self._bin_counts

    # ------------------------------------------------------------------
    # The numbers
    # ------------------------------------------------------------------
    @property
    def accuracy(self) -> float:
        """Fraction of decisions that were right."""
        return float(np.mean(self._correct))

    @property
    def mean_confidence(self) -> float:
        """Average probability the receiver claimed."""
        return float(np.mean(self._confidence))

    @property
    def overconfidence(self) -> float:
        """Mean claimed probability minus the fraction actually right.

        Positive means the receiver flatters itself.  This is the single
        most informative number in the report and the one a reader
        understands without being told what a calibration error is.
        """
        return self.mean_confidence - self.accuracy

    @property
    def ece(self) -> float:
        """Expected calibration error: the headline figure.

        The average distance between claim and reality, weighted by how
        many decisions fell in each bin.  Lower is better; 0 is perfect.
        """
        occupied = self._bin_counts > 0
        if not occupied.any():
            return 0.0
        gaps = np.abs(self._bin_accuracy[occupied] - self._bin_confidence[occupied])
        weights = self._bin_counts[occupied] / self._bin_counts.sum()
        return float(np.sum(weights * gaps))

    @property
    def mce(self) -> float:
        """Maximum calibration error: the worst bin.

        Reported because an average can hide a single badly broken
        region of the confidence range, and that region is usually the
        one a downstream system would rely on most.
        """
        occupied = self._bin_counts > 0
        if not occupied.any():
            return 0.0
        return float(
            np.max(np.abs(self._bin_accuracy[occupied] - self._bin_confidence[occupied]))
        )

    @property
    def brier(self) -> float:
        """Mean squared error between stated probability and outcome.

        Unlike the calibration error this is a *proper scoring rule*: it
        cannot be improved by lying.  A receiver can drive its
        calibration error to zero by reporting the same uninformative
        number every time; it cannot drive the Brier score down that
        way.  The two together say more than either alone.
        """
        return float(np.mean((self._confidence - self._correct.astype(float)) ** 2))

    @property
    def nll(self) -> float:
        """Negative log likelihood of the outcomes under the stated odds.

        Punishes confident mistakes far harder than the Brier score
        does, which is the behaviour wanted when the question is whether
        a receiver can be trusted at its extremes.
        """
        p = np.clip(self._confidence, _EPS, 1.0 - _EPS)
        outcomes = np.where(self._correct, p, 1.0 - p)
        return float(-np.mean(np.log(outcomes)))

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """One row of the calibration table."""
        return {
            "label": self._label,
            "n": self.n_decisions,
            "accuracy": round(self.accuracy, 6),
            "mean_confidence": round(self.mean_confidence, 6),
            "overconfidence": round(self.overconfidence, 6),
            "ece": round(self.ece, 6),
            "mce": round(self.mce, 6),
            "brier": round(self.brier, 6),
            "nll": round(self.nll, 6),
        }

    def __len__(self) -> int:
        return self.n_decisions

    def __repr__(self) -> str:
        name = f"{self._label} " if self._label else ""
        return (
            f"CalibrationReport({name}n={self.n_decisions}, "
            f"ECE={self.ece:.4f}, overconfidence={self.overconfidence:+.4f})"
        )


class RiskCoverageCurve:
    """Are the confidence numbers *useful* for deciding what to keep?

    Sort every decision from most to least confident, then walk down the
    list.  At each point, *coverage* is the fraction kept and *risk* is
    the error rate among those kept.  A receiver whose confidence means
    something produces a curve that starts near zero risk and rises; one
    whose confidence is noise produces a flat line at its overall error
    rate.

    This is the measurement that matters operationally.  A satellite
    ground station that can ask for a retransmission does not care
    whether "0.9" literally means 90%; it cares whether discarding the
    worst 10% of bits actually removes most of the errors.

    Two reference curves are computed alongside, because the raw area
    under the curve is meaningless without them:

    ``oracle``
        What a perfect ranking would achieve -- every correct bit ahead
        of every wrong one.  Its area is not zero, because once coverage
        exceeds the accuracy there is nothing left to keep but errors.
        This is the floor.
    ``random``
        What an uninformative confidence achieves: risk equal to the
        overall error rate at every coverage.  This is the ceiling worth
        beating.

    The number to quote is :attr:`excess_aurc`, the gap to the oracle.
    It is zero for a perfect ranking and equals the overall error rate
    minus the oracle area for a useless one.

    Ties are handled explicitly.  Saturated demappers report exactly
    1.0 for huge numbers of bits, and the order *within* a tie would
    otherwise be decided by array position -- which, for a frame where
    early symbols are cleaner, silently flatters the result.  Tied
    decisions are therefore shuffled with a fixed seed before sorting,
    so the curve reflects the expected behaviour rather than an accident
    of ordering.
    """

    __slots__ = ("_coverage", "_risk", "_order", "_accuracy", "_label", "_n")

    def __init__(
        self,
        confidence: np.ndarray,
        correct: np.ndarray,
        label: str = "",
        seed: int = 0,
    ) -> None:
        confidence, correct = _check_pair(confidence, correct)
        self._n = int(confidence.size)
        self._label = str(label)
        self._accuracy = float(np.mean(correct))

        # Break ties fairly: permute first, then a *stable* sort keeps
        # that permutation inside each group of equal confidences.
        rng = np.random.default_rng(int(seed))
        shuffle = rng.permutation(self._n)
        order = shuffle[np.argsort(-confidence[shuffle], kind="stable")]
        self._order = order
        self._order.flags.writeable = False

        wrong = (~correct)[order].astype(float)
        kept = np.arange(1, self._n + 1, dtype=float)
        self._coverage = kept / self._n
        self._coverage.flags.writeable = False
        self._risk = np.cumsum(wrong) / kept
        self._risk.flags.writeable = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_result(
        cls,
        result: Any,
        true_bits: np.ndarray,
        confidence: np.ndarray | None = None,
        **options: Any,
    ) -> "RiskCoverageCurve":
        """Build the curve for one recovered frame."""
        scores = result.confidence if confidence is None else confidence
        options.setdefault("label", result.receiver)
        return cls(scores, ~result.error_mask(true_bits), **options)

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------
    @property
    def label(self) -> str:
        """What produced the ranking being graded."""
        return self._label

    @property
    def coverage(self) -> np.ndarray:
        """Fraction kept, ascending from ``1/n`` to 1."""
        return self._coverage

    @property
    def risk(self) -> np.ndarray:
        """Error rate among the kept decisions, at each coverage."""
        return self._risk

    @property
    def accuracy(self) -> float:
        """Fraction of all decisions that were right."""
        return self._accuracy

    @property
    def error_rate(self) -> float:
        """Error rate at full coverage: the ordinary bit error rate."""
        return 1.0 - self._accuracy

    # ------------------------------------------------------------------
    # The numbers
    # ------------------------------------------------------------------
    @property
    def aurc(self) -> float:
        """Area under the risk-coverage curve. Lower is better."""
        return float(np.mean(self._risk))

    @property
    def aurc_oracle(self) -> float:
        """Area a perfect ranking would reach, given this error rate.

        The floor against which :attr:`aurc` is worth comparing.  It is
        not zero: beyond a coverage equal to the accuracy, even a
        perfect ranking is forced to keep wrong decisions.
        """
        n_wrong = int(round((1.0 - self._accuracy) * self._n))
        if n_wrong == 0:
            return 0.0
        kept = np.arange(1, self._n + 1, dtype=float)
        n_right = self._n - n_wrong
        wrong_so_far = np.clip(kept - n_right, 0.0, None)
        return float(np.mean(wrong_so_far / kept))

    @property
    def aurc_random(self) -> float:
        """Area an uninformative confidence reaches: the error rate."""
        return self.error_rate

    @property
    def excess_aurc(self) -> float:
        """Gap to a perfect ranking. Zero means the ranking is perfect."""
        return self.aurc - self.aurc_oracle

    @property
    def ranking_quality(self) -> float:
        """Where this ranking sits between useless (0) and perfect (1).

        The single number to put in a results table.  Defined as the
        fraction of the achievable improvement that was actually
        achieved, so it does not depend on the error rate and curves
        measured at different signal-to-noise ratios can be compared
        directly.

        Returns 1.0 when there is nothing to rank -- no errors at all --
        because in that case every ranking is as good as every other and
        reporting a failure would be misleading.
        """
        span = self.aurc_random - self.aurc_oracle
        if span <= 0.0:
            return 1.0
        return float(np.clip((self.aurc_random - self.aurc) / span, -1.0, 1.0))

    def risk_at_coverage(self, coverage: float) -> float:
        """Error rate if this fraction of the most confident bits is kept."""
        coverage = float(coverage)
        if not 0.0 < coverage <= 1.0:
            raise ValidationError("coverage", coverage, "must lie in (0, 1]")
        index = max(0, int(np.ceil(coverage * self._n)) - 1)
        return float(self._risk[index])

    def coverage_at_risk(self, target_risk: float) -> float:
        """Largest fraction keepable while staying under an error rate.

        The operational question stated plainly: *how much of this frame
        can be trusted at a one-in-a-thousand error rate?*  Returns 0.0
        when even the single most confident decision cannot meet the
        target.
        """
        target_risk = float(target_risk)
        if target_risk < 0.0:
            raise ValidationError("target_risk", target_risk, "cannot be negative")
        acceptable = self._risk <= target_risk
        if not acceptable.any():
            return 0.0
        return float(self._coverage[np.flatnonzero(acceptable)[-1]])

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------
    def summary(self, target_risk: float = 1e-3) -> dict[str, Any]:
        """One row of the risk-coverage table."""
        return {
            "label": self._label,
            "n": self._n,
            "error_rate": round(self.error_rate, 8),
            "aurc": round(self.aurc, 8),
            "aurc_oracle": round(self.aurc_oracle, 8),
            "excess_aurc": round(self.excess_aurc, 8),
            "ranking_quality": round(self.ranking_quality, 4),
            f"coverage_at_{target_risk:g}": round(
                self.coverage_at_risk(target_risk), 4
            ),
        }

    def __len__(self) -> int:
        return self._n

    def __repr__(self) -> str:
        name = f"{self._label} " if self._label else ""
        return (
            f"RiskCoverageCurve({name}n={self._n}, BER={self.error_rate:.3e}, "
            f"ranking quality {self.ranking_quality:.3f})"
        )


__all__ = ["CalibrationReport", "RiskCoverageCurve"]
