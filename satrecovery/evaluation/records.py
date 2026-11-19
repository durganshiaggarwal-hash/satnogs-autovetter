"""What a benchmark remembers about each receiver in each scenario.

One :class:`CellResult` per (receiver, scenario) pair -- a cell of the
results table.  It holds everything Phases 1 to 5 taught the project to
measure, computed once and frozen:

* how often the receiver was wrong, with an interval;
* whether its confidence was honest (calibration);
* whether its confidence was useful (ranking);
* how many whole frames it lost, and whether its confidence could tell
  which ones (triage).

Per-frame counts are kept as well as the pooled totals, because Phase 5
showed that the pooled numbers can hide the one thing that matters, and
the cluster bootstrap needs them to put honest intervals on anything.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..confidence.metrics import CalibrationReport, RiskCoverageCurve
from ..core.exceptions import ValidationError
from .scenarios import Scenario
from .statistics import ErrorRate, FrameBootstrap

#: A frame with more than this fraction of bits wrong is counted as
#: *lost*: the receiver failed to lock rather than made scattered
#: mistakes.  Phase 5's frames split cleanly either side of it -- locked
#: frames stayed below about 5%, lost ones sat at 10% or far above.
LOST_FRAME_BER = 0.1


def _frozen(values: Any, dtype: Any) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True).ravel()
    array.flags.writeable = False
    return array


class CellResult:
    """One receiver's measured behaviour in one scenario.

    Build it with :meth:`from_outcomes`, which does every calculation in
    one place; the constructor only stores and validates.
    """

    __slots__ = (
        "_receiver",
        "_scenario",
        "_frame_errors",
        "_frame_bits",
        "_frame_confidence",
        "_calibration",
        "_ranking",
        "_seconds",
        "_source",
    )

    def __init__(
        self,
        receiver: str,
        scenario: Scenario,
        frame_errors: np.ndarray,
        frame_bits: np.ndarray,
        frame_confidence: np.ndarray,
        calibration: dict[str, float],
        ranking: dict[str, float],
        seconds: float,
        source: str = "",
    ) -> None:
        if not isinstance(scenario, Scenario):
            raise ValidationError(
                "scenario", type(scenario).__name__, "expected a Scenario"
            )
        errors = _frozen(frame_errors, np.int64)
        bits = _frozen(frame_bits, np.int64)
        confidence = _frozen(frame_confidence, float)
        if not (errors.size == bits.size == confidence.size) or errors.size == 0:
            raise ValidationError(
                "frame_errors", errors.size, "need one entry per frame, at least one frame"
            )
        self._receiver = str(receiver)
        self._scenario = scenario
        self._frame_errors = errors
        self._frame_bits = bits
        self._frame_confidence = confidence
        self._calibration = dict(calibration)
        self._ranking = dict(ranking)
        self._seconds = float(seconds)
        self._source = str(source)

    # ------------------------------------------------------------------
    @classmethod
    def from_outcomes(
        cls,
        receiver: str,
        scenario: Scenario,
        confidences: list[np.ndarray],
        correct: list[np.ndarray],
        seconds: float,
        source: str = "",
    ) -> "CellResult":
        """Compute every metric from per-frame, per-bit outcomes.

        Parameters
        ----------
        confidences, correct:
            One array per frame: the receiver's per-bit confidence and
            whether each bit was right.
        """
        if len(confidences) != len(correct) or not confidences:
            raise ValidationError(
                "confidences", len(confidences), "need one array per frame, at least one"
            )
        frame_bits = np.array([c.size for c in correct])
        frame_errors = np.array([int(np.count_nonzero(~c)) for c in correct])
        frame_confidence = np.array([float(np.mean(c)) for c in confidences])

        pooled_conf = np.concatenate(confidences)
        pooled_correct = np.concatenate(correct)
        report = CalibrationReport(pooled_conf, pooled_correct)
        curve = RiskCoverageCurve(pooled_conf, pooled_correct)

        calibration = {
            "ece": report.ece,
            "overconfidence": report.overconfidence,
            "brier": report.brier,
        }
        ranking = {
            "ranking_quality": curve.ranking_quality,
            "coverage_at_1e-3": curve.coverage_at_risk(1e-3),
        }
        return cls(
            receiver,
            scenario,
            frame_errors,
            frame_bits,
            frame_confidence,
            calibration,
            ranking,
            seconds,
            source,
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    def receiver(self) -> str:
        return self._receiver

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    @property
    def source(self) -> str:
        """Where the frames came from: simulated, or a recorded dataset."""
        return self._source

    @property
    def n_frames(self) -> int:
        return int(self._frame_errors.size)

    @property
    def seconds(self) -> float:
        """Wall-clock time spent recovering, for the speed column."""
        return self._seconds

    # ------------------------------------------------------------------
    # Per-frame data, read-only
    # ------------------------------------------------------------------
    @property
    def frame_errors(self) -> np.ndarray:
        return self._frame_errors

    @property
    def frame_bits(self) -> np.ndarray:
        return self._frame_bits

    @property
    def frame_confidence(self) -> np.ndarray:
        """Mean confidence over each frame: the triage score."""
        return self._frame_confidence

    # ------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------
    @property
    def error_rate(self) -> ErrorRate:
        """Pooled bit errors, as a value with a Wilson interval."""
        return ErrorRate(int(self._frame_errors.sum()), int(self._frame_bits.sum()))

    @property
    def ber(self) -> float:
        return self.error_rate.rate

    def ber_interval(self, method: str = "conservative") -> tuple[float, float]:
        """95% interval for the bit error rate.

        ``"wilson"``
            Treats every bit as independent.  Far too narrow when frames
            are lost, because their errors arrive in clumps of hundreds.
        ``"bootstrap"``
            Resamples whole frames, so it respects clumping.  But with
            few frames and few errors it is itself rough, and it can come
            out *narrower* than Wilson when errors are spread unusually
            evenly across frames.
        ``"conservative"`` (default)
            The span covering both.  Since each can be wrong in a
            different direction, a project that promises not to
            overclaim reports the wider one.
        """
        wilson = self.error_rate.interval()
        if method == "wilson" or self.n_frames < 2:
            return wilson
        bootstrap = FrameBootstrap(resamples=1000).interval(
            self._frame_errors, self._frame_bits
        )
        if method == "bootstrap":
            return bootstrap
        if method == "conservative":
            return (min(wilson[0], bootstrap[0]), max(wilson[1], bootstrap[1]))
        raise ValidationError(
            "method", method, "expected 'wilson', 'bootstrap' or 'conservative'"
        )

    # ------------------------------------------------------------------
    # Lost frames
    # ------------------------------------------------------------------
    @property
    def frame_lost(self) -> np.ndarray:
        """Which frames failed outright."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self._frame_errors / np.maximum(self._frame_bits, 1)) > LOST_FRAME_BER

    @property
    def lost_frames(self) -> int:
        return int(np.count_nonzero(self.frame_lost))

    @property
    def triage_quality(self) -> float:
        """How well mean confidence picks out the lost frames.

        Ranking quality at frame level, exactly as Phase 5 measured it.
        NaN when there is nothing to triage -- no lost frames, or every
        frame lost -- rather than a flattering 1.0.
        """
        lost = self.frame_lost
        if not lost.any() or lost.all():
            return float("nan")
        return RiskCoverageCurve(self._frame_confidence, ~lost).ranking_quality

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------
    @property
    def ece(self) -> float:
        return float(self._calibration["ece"])

    @property
    def overconfidence(self) -> float:
        return float(self._calibration["overconfidence"])

    @property
    def ranking_quality(self) -> float:
        return float(self._ranking["ranking_quality"])

    @property
    def coverage_at_1e3(self) -> float:
        return float(self._ranking["coverage_at_1e-3"])

    # ------------------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        """One flat row, for tables, storage and the API."""
        low, high = self.ber_interval()
        return {
            "receiver": self._receiver,
            **self._scenario.as_dict(),
            "source": self._source,
            "n_frames": self.n_frames,
            "bits": int(self._frame_bits.sum()),
            "errors": int(self._frame_errors.sum()),
            "ber": self.ber,
            "ber_low": low,
            "ber_high": high,
            "ece": self.ece,
            "overconfidence": self.overconfidence,
            "ranking_quality": self.ranking_quality,
            "coverage_at_1e-3": self.coverage_at_1e3,
            "lost_frames": self.lost_frames,
            "triage_quality": self.triage_quality,
            "seconds": self._seconds,
        }

    def __repr__(self) -> str:
        return (
            f"CellResult({self._receiver}, {self._scenario!r}, "
            f"BER {self.ber:.2e}, lost {self.lost_frames}/{self.n_frames})"
        )


__all__ = ["CellResult", "LOST_FRAME_BER"]
