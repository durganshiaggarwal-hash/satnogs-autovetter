"""Turning cleaned-up symbols back into bits, with a confidence attached.

The last step of any receiver, and the one where this project's second
question lives.  A demapper does not only have to decide what was sent --
it has to say how sure it is.

Two implementations, deliberately unequal:

:class:`HardDemapper`
    Picks the nearest constellation point and rates its confidence by how
    much nearer that point was than the runner-up.  Simple, fast, and
    only roughly calibrated.

:class:`SoftDemapper`
    Works out, per bit, the log-likelihood ratio: how much more probable
    a zero is than a one, given the noise. Properly grounded in the
    statistics, and its output is what error-correcting codes actually
    want.

Having both is the point.  Phase 5 measures whether the better-grounded
confidence is in fact better calibrated, and by how much.

WHY SOFT OUTPUTS MATTER
-----------------------
A hard decision throws information away.  Telling a decoder "this is a 1"
loses the difference between a symbol sitting squarely on its
constellation point and one that landed halfway to its neighbour.  Modern
error-correcting codes exploit that difference, and the standard figure
for what it buys is around 2 dB of coding gain -- which in a satellite
link budget is roughly 60% more transmit power, or a dish a quarter
wider.  That is why every real demodulator emits log-likelihood ratios.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.exceptions import ValidationError
from ..modulation.base import Modulation

#: Beyond this the sigmoid saturates and the exponential overflows.
#: A log-likelihood ratio of 30 already means odds of about 10**13 to 1.
_MAX_LLR = 30.0


def estimate_noise_power(
    symbols: np.ndarray, modulation: Modulation, floor: float = 1e-4
) -> float:
    """Noise power, read off how far symbols sit from their decisions.

    The mean squared distance from each symbol to its nearest legal
    point.  Needs nothing but the symbols, so it can be computed fresh
    for every frame -- and Phase 6 found that doing so is where most of
    a receiver's ability to recognise a lost frame comes from.

    Phase 5 found its weakness: once symbols routinely land nearer a
    wrong point, it measures the distance to that point and comes out
    low.  The *level* is biased; the frame-to-frame *variation* is
    informative.  Both findings are about this one function.
    """
    symbols = np.asarray(symbols, dtype=np.complex128).ravel()
    if symbols.size == 0:
        return float(floor)
    squared = np.abs(symbols[:, None] - modulation.constellation[None, :]) ** 2
    return max(float(np.mean(np.min(squared, axis=1))), float(floor))


class Demapper(ABC):
    """Base class for turning symbols into bits plus confidence."""

    def __init__(self, name: str | None = None) -> None:
        self._name = str(name) if name else type(self).__name__

    @property
    def name(self) -> str:
        """Short name of this demapper."""
        return self._name

    @abstractmethod
    def demap(
        self, symbols: np.ndarray, modulation: Modulation
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Decide the bits.

        Returns
        -------
        (bits, confidence, llr)
            ``bits`` is one value per transmitted bit.  ``confidence`` is
            the probability, per bit, that the decision is right, and so
            lies in ``[0.5, 1]``.  ``llr`` is the log-likelihood ratio
            where the demapper computes one, otherwise ``None``.
        """

    @staticmethod
    def _validate(symbols: np.ndarray, modulation: Modulation) -> np.ndarray:
        if not isinstance(modulation, Modulation):
            raise ValidationError(
                "modulation",
                type(modulation).__name__,
                "expected a Modulation instance",
            )
        return np.asarray(symbols, dtype=np.complex128).ravel()

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class HardDemapper(Demapper):
    """Nearest point wins; confidence comes from how clear the win was.

    If the closest constellation point is much closer than the next one,
    the decision is safe.  If two points are nearly equidistant, the
    symbol landed on a decision boundary and the answer is close to a
    coin flip.  That margin, scaled against the constellation's own
    spacing, is the confidence.

    It is a reasonable heuristic rather than a probability, and it is
    assigned per symbol, so every bit in a symbol inherits the same
    number.  :class:`SoftDemapper` improves on both counts.
    """

    def __init__(self) -> None:
        super().__init__("Hard")

    def demap(self, symbols, modulation):
        symbols = self._validate(symbols, modulation)
        if symbols.size == 0:
            empty_u8 = np.empty(0, dtype=np.uint8)
            return empty_u8, np.empty(0, dtype=float), None

        constellation = modulation.constellation
        distances = np.abs(symbols[:, None] - constellation[None, :])

        indices = np.argmin(distances, axis=1)
        bits = modulation.indices_to_bits(indices)

        if constellation.size == 1:
            margin = np.ones(symbols.size)
        else:
            nearest_two = np.partition(distances, 1, axis=1)[:, :2]
            margin = nearest_two[:, 1] - nearest_two[:, 0]

        # Half the minimum distance is how far a symbol can stray before
        # the decision flips, so it is the natural yardstick.
        scaled = margin / (0.5 * modulation.minimum_distance)
        per_symbol = 0.5 + 0.5 * np.clip(scaled, 0.0, 1.0)

        confidence = np.repeat(per_symbol, modulation.bits_per_symbol)
        return bits, confidence, None


class SoftDemapper(Demapper):
    """Per-bit log-likelihood ratios, using the max-log approximation.

    For each bit the demapper asks: what is the closest constellation
    point that would have sent a zero here, and the closest that would
    have sent a one?  The difference between those two squared
    distances, divided by the noise power, is the log-likelihood ratio --
    positive means a zero is more likely, and the size says by how much.

    The exact expression sums over every constellation point; the
    max-log form keeps only the nearest in each set.  The simplification
    costs a few tenths of a decibel and is what real hardware uses.

    The noise power is **estimated from the signal itself**, as the mean
    squared distance from each symbol to its nearest constellation
    point.  No part of the receiver is told the true noise level, which
    would be cheating -- and the estimate is one of the things Phase 5
    will check for bias.
    """

    def __init__(self, noise_floor: float = 1e-4) -> None:
        super().__init__("Soft")
        self._noise_floor = float(noise_floor)

    def demap(self, symbols, modulation):
        symbols = self._validate(symbols, modulation)
        if symbols.size == 0:
            return np.empty(0, dtype=np.uint8), np.empty(0, dtype=float), None

        constellation = modulation.constellation
        bits_per_symbol = modulation.bits_per_symbol

        # Squared distance from every symbol to every constellation point.
        squared = np.abs(symbols[:, None] - constellation[None, :]) ** 2

        noise_power = self._estimate_noise_power(squared)

        # Which constellation indices carry a 0, and which a 1, in each
        # bit position. Computed once; it depends only on the scheme.
        labels = modulation.indices_to_bits(
            np.arange(constellation.size)
        ).reshape(constellation.size, bits_per_symbol)

        llr = np.zeros((symbols.size, bits_per_symbol), dtype=float)
        for bit in range(bits_per_symbol):
            zeros = labels[:, bit] == 0
            nearest_zero = np.min(squared[:, zeros], axis=1)
            nearest_one = np.min(squared[:, ~zeros], axis=1)
            llr[:, bit] = (nearest_one - nearest_zero) / noise_power

        llr = np.clip(llr, -_MAX_LLR, _MAX_LLR).ravel()

        # Positive means a zero is the more likely bit.
        bits = (llr <= 0).astype(np.uint8)

        # The sigmoid turns a log-odds into the probability that this
        # decision is the right one.
        confidence = 1.0 / (1.0 + np.exp(-np.abs(llr)))

        return bits, confidence, llr

    def _estimate_noise_power(self, squared_distances: np.ndarray) -> float:
        # Kept as a method, working on the distances already computed,
        # so that a subclass can override it -- which is exactly how the
        # Phase 6 ablation held the noise fixed to test what it does.
        """Noise power, read off the spread around the decisions.

        Each symbol's distance to its nearest constellation point is
        mostly noise, so averaging those gives the noise power without
        anybody having to be told it.
        """
        residual = float(np.mean(np.min(squared_distances, axis=1)))
        return max(residual, self._noise_floor)


class NoiseNormalisedDemapper(Demapper):
    """Gives any demapper the per-frame noise awareness a soft demapper has.

    THE FINDING THIS COMES FROM
    ---------------------------
    Phase 6 measured how well a receiver's average confidence over a
    frame picks out frames where it lost lock.  The classical receiver
    scored 0.970; the hybrid, with a trained network as its demapper,
    scored 0.829 -- with identical bit decisions.

    The ablation that explained it: hold the classical demapper's noise
    estimate fixed instead of re-measuring it per frame, and its score
    falls to 0.835, the hybrid's level.  The soft demapper divides by a
    fresh noise estimate on every frame, so on a frame that has gone
    wrong every one of its odds shrinks together.  A trained network
    has fixed weights and no such step.

    THE FIX
    -------
    Wrap the network and supply the missing step: multiply its
    log-likelihood ratios by ``reference / this frame's noise``.  On a
    typical frame the factor is near one and nothing changes; on a
    noisy or broken frame every ratio shrinks.  Signs never change, so
    **no bit decision changes** -- only the confidence does.

    DESIGN NOTE: THE DECORATOR PATTERN
    ----------------------------------
    The wrapper *is* a demapper, *holds* a demapper, and adds one
    behaviour on the way through.  It works on any demapper -- learned,
    classical, or another wrapper -- and the receiver it is plugged into
    cannot tell the difference.

    The idea is not new: feeding a noise or SNR estimate to a learned
    demapper is a known technique.  What this project contributes is
    the measurement that shows *why* it matters here, and how much.

    Parameters
    ----------
    inner:
        The demapper whose confidence is to be normalised.
    reference_noise:
        The noise power at which the inner demapper's odds are left
        unchanged.  Fit it with :meth:`fit_reference` on training
        frames, so it reflects the conditions the inner model saw.
    """

    def __init__(self, inner: Demapper, reference_noise: float) -> None:
        if not isinstance(inner, Demapper):
            raise ValidationError("inner", type(inner).__name__, "expected a Demapper")
        reference_noise = float(reference_noise)
        if not reference_noise > 0.0:
            raise ValidationError("reference_noise", reference_noise, "must be positive")
        super().__init__(f"{inner.name}+noise")
        self._inner = inner
        self._reference = reference_noise

    @property
    def inner(self) -> Demapper:
        """The demapper being wrapped."""
        return self._inner

    @property
    def reference_noise(self) -> float:
        """Noise power at which the inner odds pass through unchanged."""
        return self._reference

    @staticmethod
    def fit_reference(symbol_frames: list[np.ndarray], modulation: Modulation) -> float:
        """Typical per-frame noise power over a set of training frames.

        The median rather than the mean, so a few lost frames in the
        training set -- whose noise estimates are enormous -- cannot drag
        the reference away from what a normal frame looks like.
        """
        if not symbol_frames:
            raise ValidationError("symbol_frames", 0, "need at least one frame")
        return float(np.median([estimate_noise_power(s, modulation) for s in symbol_frames]))

    def demap(self, symbols, modulation):
        symbols = self._validate(symbols, modulation)
        bits, _, llr = self._inner.demap(symbols, modulation)
        if llr is None:
            raise ValidationError(
                "inner",
                self._inner.name,
                "needs a demapper that produces log-likelihood ratios",
            )
        if symbols.size == 0:
            return bits, np.empty(0, dtype=float), np.asarray(llr, dtype=float)

        scale = self._reference / estimate_noise_power(symbols, modulation)
        llr = np.clip(np.asarray(llr, dtype=float) * scale, -_MAX_LLR, _MAX_LLR)
        confidence = 1.0 / (1.0 + np.exp(-np.abs(llr)))
        return bits, confidence, llr

    def __repr__(self) -> str:
        return f"NoiseNormalisedDemapper({self._inner!r}, reference={self._reference:.4g})"


__all__ = [
    "Demapper",
    "HardDemapper",
    "NoiseNormalisedDemapper",
    "SoftDemapper",
    "estimate_noise_power",
]
