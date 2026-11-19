"""How sure we are about a measured error rate.

A bit error rate measured on a few thousand bits is an estimate, and
reporting it without an interval invites exactly the kind of claim this
project says it will not make: "receiver A beats receiver B" when the
difference is smaller than the measurement's own wobble.

Two ways of putting an interval on it, because they answer different
questions and the difference between them turned out to matter.

:class:`ErrorRate` -- the **Wilson interval**
    Treats every bit as an independent coin flip.  Standard, cheap, and
    honest about one thing the naive answer is not: zero errors in 2000
    bits does *not* mean an error rate of zero.  It means somewhere
    below about 1 in 500, and the interval says so.

:class:`FrameBootstrap` -- the **cluster bootstrap**
    Phase 5 found that errors are not independent.  A frame where the
    receiver loses lock is wrong almost everywhere, so errors arrive in
    clumps of hundreds.  Treating those as independent flips makes the
    Wilson interval far too narrow.  The bootstrap instead resamples
    *whole frames*, keeping each clump intact, and the width of the
    interval it produces reflects how much the answer really depends on
    which frames happened to be drawn.

The demo shows the two side by side, and on an impaired channel the
honest interval is several times wider than the textbook one.

DESIGN NOTE: OPERATOR OVERLOADING ON A VALUE OBJECT
---------------------------------------------------
``ErrorRate`` supports ``+``: adding two measurements pools their
counts, which is the correct way to combine them -- averaging two rates
would weight a 100-bit measurement the same as a million-bit one.
``sum(rates)`` therefore does the right thing with no special code.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np

from ..core.exceptions import ValidationError


def _z(confidence: float) -> float:
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValidationError("confidence", confidence, "must lie in (0, 1)")
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


class ErrorRate:
    """A count of errors out of a count of trials, with its uncertainty.

    Immutable.  Combine measurements with ``+``; never average rates.
    """

    __slots__ = ("_errors", "_trials")

    def __init__(self, errors: int, trials: int) -> None:
        errors, trials = int(errors), int(trials)
        if trials < 0 or errors < 0:
            raise ValidationError("errors", errors, "counts cannot be negative")
        if errors > trials:
            raise ValidationError(
                "errors", errors, f"cannot exceed the number of trials ({trials})"
            )
        self._errors = errors
        self._trials = trials

    @property
    def errors(self) -> int:
        """How many were wrong."""
        return self._errors

    @property
    def trials(self) -> int:
        """How many were tried."""
        return self._trials

    @property
    def rate(self) -> float:
        """The point estimate. Zero if nothing was measured."""
        return self._errors / self._trials if self._trials else 0.0

    def interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """Wilson score interval.

        Chosen over the simpler "rate plus or minus two standard errors"
        because that one collapses to zero width when no errors are
        seen, which is the case that most needs an honest upper bound.
        """
        if self._trials == 0:
            return (0.0, 1.0)
        z = _z(confidence)
        n, p = self._trials, self.rate
        denominator = 1.0 + z * z / n
        centre = (p + z * z / (2.0 * n)) / denominator
        half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
        # At the edges the bound is exactly 0 or 1; computing it leaves
        # rounding dust like 1e-19, which would print as a nonzero rate.
        low = 0.0 if self._errors == 0 else max(0.0, float(centre - half))
        high = 1.0 if self._errors == self._trials else min(1.0, float(centre + half))
        return (low, high)

    def __add__(self, other: "ErrorRate") -> "ErrorRate":
        if not isinstance(other, ErrorRate):
            return NotImplemented
        return ErrorRate(self._errors + other._errors, self._trials + other._trials)

    def __radd__(self, other: Any) -> "ErrorRate":
        # Lets ``sum(list_of_rates)`` work: sum starts from the integer 0.
        if other == 0:
            return self
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ErrorRate):
            return NotImplemented
        return (self._errors, self._trials) == (other._errors, other._trials)

    def __hash__(self) -> int:
        return hash((self._errors, self._trials))

    def __repr__(self) -> str:
        low, high = self.interval()
        return (
            f"ErrorRate({self._errors}/{self._trials} = {self.rate:.3e}, "
            f"95% [{low:.2e}, {high:.2e}])"
        )


class Comparison:
    """The measured difference between two receivers on the same frames.

    Built by :meth:`FrameBootstrap.compare`.  The verdict is only
    "better" when the whole interval for the difference sits on one side
    of zero; otherwise the honest answer is that the frames tested
    cannot tell the two apart.
    """

    __slots__ = ("_first", "_second", "_difference", "_interval", "_confidence")

    def __init__(
        self,
        first: str,
        second: str,
        difference: float,
        interval: tuple[float, float],
        confidence: float,
    ) -> None:
        self._first = str(first)
        self._second = str(second)
        self._difference = float(difference)
        self._interval = (float(interval[0]), float(interval[1]))
        self._confidence = float(confidence)

    @property
    def difference(self) -> float:
        """First receiver's error rate minus the second's. Negative favours the first."""
        return self._difference

    @property
    def interval(self) -> tuple[float, float]:
        """Bootstrap interval for the difference."""
        return self._interval

    @property
    def is_decisive(self) -> bool:
        """Does the interval exclude zero?"""
        low, high = self._interval
        return high < 0.0 or low > 0.0

    @property
    def verdict(self) -> str:
        """Plain-language conclusion, deliberately conservative."""
        low, high = self._interval
        if high < 0.0:
            return f"{self._first} has fewer errors"
        if low > 0.0:
            return f"{self._second} has fewer errors"
        return "no detectable difference on these frames"

    def as_dict(self) -> dict[str, Any]:
        return {
            "first": self._first,
            "second": self._second,
            "difference": self._difference,
            "low": self._interval[0],
            "high": self._interval[1],
            "confidence": self._confidence,
            "verdict": self.verdict,
        }

    def __repr__(self) -> str:
        low, high = self._interval
        return (
            f"Comparison({self._first} - {self._second} = {self._difference:+.2e}, "
            f"{self._confidence:.0%} [{low:+.2e}, {high:+.2e}]: {self.verdict})"
        )


class FrameBootstrap:
    """Intervals that respect the fact that errors come in clumps.

    Parameters
    ----------
    resamples:
        How many bootstrap draws.  Two thousand gives interval ends that
        are stable to about the second significant figure, which is all
        any result here is quoted to.
    seed:
        Fixed by default, so the same data always gives the same
        interval and a result table can be regenerated exactly.
    """

    def __init__(self, resamples: int = 2000, seed: int = 0) -> None:
        resamples = int(resamples)
        if resamples < 100:
            raise ValidationError("resamples", resamples, "use at least 100")
        self._resamples = resamples
        self._seed = int(seed)

    @property
    def resamples(self) -> int:
        """Bootstrap draws per interval."""
        return self._resamples

    @staticmethod
    def _check(errors: np.ndarray, bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        errors = np.asarray(errors, dtype=float).ravel()
        bits = np.asarray(bits, dtype=float).ravel()
        if errors.size != bits.size:
            raise ValidationError(
                "bits", bits.size, f"need one count per frame ({errors.size})"
            )
        if errors.size == 0:
            raise ValidationError("errors", 0, "no frames to resample")
        if np.any(errors < 0) or np.any(errors > bits):
            raise ValidationError("errors", "<array>", "must lie between 0 and the bit count")
        return errors, bits

    def _draws(self, n_frames: int) -> np.ndarray:
        rng = np.random.default_rng(self._seed)
        return rng.integers(0, n_frames, size=(self._resamples, n_frames))

    def interval(
        self,
        frame_errors: np.ndarray,
        frame_bits: np.ndarray,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Percentile interval for the pooled error rate."""
        errors, bits = self._check(frame_errors, frame_bits)
        _z(confidence)  # validates the level
        draws = self._draws(errors.size)
        rates = errors[draws].sum(axis=1) / bits[draws].sum(axis=1)
        tail = (1.0 - confidence) / 2.0
        return (
            float(np.quantile(rates, tail)),
            float(np.quantile(rates, 1.0 - tail)),
        )

    def compare(
        self,
        first_errors: np.ndarray,
        second_errors: np.ndarray,
        frame_bits: np.ndarray,
        names: tuple[str, str] = ("first", "second"),
        confidence: float = 0.95,
    ) -> Comparison:
        """Is one receiver really better than the other on these frames?

        **Paired**: both receivers must have been run on the *same*
        frames, and each bootstrap draw resamples frames for both at
        once.  That cancels the frame-to-frame luck they share -- a hard
        frame is hard for both -- and leaves only the difference between
        the receivers.  It is why the benchmark runner feeds every
        receiver identical frames rather than generating fresh ones for
        each.
        """
        first, bits = self._check(first_errors, frame_bits)
        second, _ = self._check(second_errors, frame_bits)
        _z(confidence)

        total = bits.sum()
        difference = (first.sum() - second.sum()) / total

        draws = self._draws(first.size)
        drawn_bits = bits[draws].sum(axis=1)
        deltas = (first[draws].sum(axis=1) - second[draws].sum(axis=1)) / drawn_bits
        tail = (1.0 - confidence) / 2.0
        low, high = np.quantile(deltas, tail), np.quantile(deltas, 1.0 - tail)

        return Comparison(names[0], names[1], difference, (low, high), confidence)


__all__ = ["Comparison", "ErrorRate", "FrameBootstrap"]
