"""Pulse shaping filters.

A modulator does not transmit bare symbols -- it transmits a smooth
waveform built by passing the symbols through a pulse shaping filter.
This matters for two reasons:

1. **Bandwidth.** Abrupt symbol transitions spread energy across a huge
   frequency range.  Satellite spectrum is licensed and crowded, so the
   transmitted signal has to be confined to its allotted band.
2. **Inter-symbol interference.** A well-chosen pulse is zero at every
   *other* symbol instant, so neighbouring symbols do not smear into each
   other at the sampling points.

The root-raised-cosine filter is the standard answer, and the one RML24
uses (with roll-off 0.35).  Splitting the shaping between transmitter and
receiver -- a *root*-raised-cosine at each end -- gives a combined
raised-cosine response that satisfies the zero-interference condition
while also maximising the signal-to-noise ratio at the decision point.

DESIGN NOTE: ABSTRACTION OVER AN ALGORITHM
------------------------------------------
:class:`PulseShape` declares what any pulse shaping filter must be able
to do -- produce its taps -- and then implements ``shape`` and
``matched_filter`` once, on top of that.  Swapping
:class:`RootRaisedCosine` for :class:`Rectangular` in a test changes one
constructor argument and nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np

from ..core.exceptions import ValidationError


class PulseShape(ABC):
    """Base class for pulse shaping filters."""

    def __init__(self, samples_per_symbol: int, name: str | None = None) -> None:
        if int(samples_per_symbol) < 1:
            raise ValidationError(
                "samples_per_symbol", samples_per_symbol, "must be at least 1"
            )
        self._samples_per_symbol = int(samples_per_symbol)
        self._name = str(name) if name else type(self).__name__

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------
    @abstractmethod
    def _build_taps(self) -> np.ndarray:
        """Return the filter's impulse response, before energy normalising."""

    @abstractmethod
    def _rebuilt(self, samples_per_symbol: int) -> "PulseShape":
        """A filter of the same family and parameters at a new rate."""

    # ------------------------------------------------------------------
    # DESIGN NOTE: POLYMORPHISM WITH A SAFETY PURPOSE
    # ------------------------------------------------------------------
    def at_rate(self, samples_per_symbol: int) -> "PulseShape":
        """This filter, rebuilt for a different oversampling factor.

        Returns ``self`` when the rate already matches.  Otherwise every
        subclass rebuilds itself -- a root-raised-cosine keeps its
        roll-off and span, a rectangle stays a rectangle -- and the
        caller never needs to know which kind it is holding.

        This exists because of a real defect.  The receiver used to apply
        whatever pulse it was given, at whatever rate that pulse was
        built for.  Every simulated frame was 8 samples per symbol, so
        nothing ever went wrong.  RML24 frames are 10, 5 or 2, and a
        matched filter at the wrong rate does not fail loudly: it
        produces a plausible-looking, quietly worse constellation.
        """
        samples_per_symbol = int(samples_per_symbol)
        if samples_per_symbol < 1:
            raise ValidationError(
                "samples_per_symbol", samples_per_symbol, "must be at least 1"
            )
        if samples_per_symbol == self._samples_per_symbol:
            return self
        return self._rebuilt(samples_per_symbol)

    # ------------------------------------------------------------------
    # Shared properties
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Short display name of the filter."""
        return self._name

    @property
    def samples_per_symbol(self) -> int:
        """Oversampling factor the filter is built for."""
        return self._samples_per_symbol

    @cached_property
    def taps(self) -> np.ndarray:
        """Unit-energy impulse response.

        ``cached_property`` computes this once on first access and then
        reuses it -- the taps depend only on constructor arguments, which
        never change.  A small example of encapsulation paying for
        itself: because callers go through a property, we could add
        caching later without touching any calling code.
        """
        taps = np.asarray(self._build_taps(), dtype=float)
        if taps.ndim != 1 or taps.size == 0:
            raise ValidationError(
                f"{self._name}.taps", taps.shape, "expected a non-empty 1-D array"
            )
        energy = float(np.sum(taps**2))
        if energy <= 0:
            raise ValidationError(f"{self._name}.taps", "<array>", "zero energy")
        normalised = taps / np.sqrt(energy)
        normalised.flags.writeable = False
        return normalised

    # ------------------------------------------------------------------
    # Shared behaviour
    # ------------------------------------------------------------------
    def shape(self, symbols: np.ndarray) -> np.ndarray:
        """Turn symbols into an oversampled, band-limited waveform.

        Returns ``len(symbols) * samples_per_symbol`` samples, with
        symbol ``k`` centred on sample ``k * samples_per_symbol``.
        """
        symbols = np.asarray(symbols, dtype=np.complex128).ravel()
        if symbols.size == 0:
            return np.empty(0, dtype=np.complex128)

        # Upsample: one symbol, then sps-1 zeros.  Filtering this
        # impulse train is what spreads each symbol into a pulse.
        upsampled = np.zeros(
            symbols.size * self._samples_per_symbol, dtype=np.complex128
        )
        upsampled[:: self._samples_per_symbol] = symbols
        return self._circular_filter(upsampled)

    def matched_filter(self, samples: np.ndarray) -> np.ndarray:
        """Apply the receive-side matched filter.

        For our real, symmetric filters the matched filter has the same
        taps as the transmit filter, so this is deliberately the same
        operation.  It is given its own name because its *purpose* is
        different: at the transmitter the filter limits bandwidth, at the
        receiver it maximises the signal-to-noise ratio at the decision
        instants.  Phase 2's classical receiver calls this first.
        """
        samples = np.asarray(samples, dtype=np.complex128).ravel()
        if samples.size == 0:
            return np.empty(0, dtype=np.complex128)
        return self._circular_filter(samples)

    def _circular_filter(self, samples: np.ndarray) -> np.ndarray:
        """Filter with periodic wrap-around, preserving length and timing.

        Why circular rather than ordinary convolution: a normal
        convolution leaves the first and last few symbols of a frame with
        only part of their pulse, so those symbols suffer extra errors
        that have nothing to do with the channel.  Wrapping around models
        a continuously transmitted stream instead, which means **every**
        symbol in the frame sees the same conditions and every bit label
        stays valid.  With a 2048-sample frame that saves roughly 3% of
        the bits from being quietly corrupted by an artefact of framing.
        """
        taps = self.taps
        n = samples.size
        if taps.size == 1:
            return samples * taps[0]

        padded = np.zeros(n, dtype=float)
        usable = min(taps.size, n)
        padded[:usable] = taps[:usable]

        # Rotate so the filter's centre tap sits at index 0, which keeps
        # the output time-aligned with the input.
        padded = np.roll(padded, -(usable // 2))

        spectrum = np.fft.fft(samples) * np.fft.fft(padded)
        return np.fft.ifft(spectrum)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(sps={self._samples_per_symbol})"


class RootRaisedCosine(PulseShape):
    """The root-raised-cosine filter used by real satellite modems.

    Parameters
    ----------
    samples_per_symbol:
        Oversampling factor.
    rolloff:
        Excess bandwidth factor, between 0 and 1.  Small values are
        spectrally tighter but ring for longer in time, making the link
        more sensitive to timing error.  RML24 uses 0.35, so we default
        to the same.
    span_symbols:
        How many symbol periods of the (infinitely long) ideal response
        to keep.

        Truncation is the one place where this filter stops being
        perfect: cutting the tails off breaks the zero-interference
        condition slightly, leaving a little residual inter-symbol
        interference.  Measured peak residue for 16QAM at eight samples
        per symbol:

        ===========  ====================
        ``span``     Peak residual ISI
        ===========  ====================
        4            1.4e-01
        8            2.9e-02
        12           4.8e-03
        24           2.6e-03
        ===========  ====================

        The default of 12 keeps the residue around 1.5% of the closest
        spacing in our tightest constellation (64QAM), which is far below
        the noise at any signal-to-noise ratio we test.  Going further
        buys little and costs taps.  Real modems make the same trade.

        The length needed depends on the roll-off, because a tighter
        spectrum rings for longer in time.  To reach a residue below
        1e-2, roll-off 0.5 needs a span of about 8, roll-off 0.35 about
        12, and roll-off 0.22 about 20.  Lower the roll-off and the span
        should go up with it.
    """

    def __init__(
        self,
        samples_per_symbol: int,
        rolloff: float = 0.35,
        span_symbols: int = 12,
    ) -> None:
        rolloff = float(rolloff)
        if not 0.0 < rolloff <= 1.0:
            raise ValidationError(
                "rolloff", rolloff, "must be greater than 0 and at most 1"
            )
        if int(span_symbols) < 1:
            raise ValidationError(
                "span_symbols", span_symbols, "must be at least 1"
            )

        self._rolloff = rolloff
        self._span_symbols = int(span_symbols)
        super().__init__(samples_per_symbol, name="RRC")

    @property
    def rolloff(self) -> float:
        """Excess bandwidth factor (``beta``)."""
        return self._rolloff

    @property
    def span_symbols(self) -> int:
        """Length of the filter in symbol periods."""
        return self._span_symbols

    @property
    def occupied_bandwidth_factor(self) -> float:
        """Occupied bandwidth as a multiple of the symbol rate."""
        return 1.0 + self._rolloff

    def _build_taps(self) -> np.ndarray:
        """Evaluate the standard root-raised-cosine impulse response.

        Two points in the expression are removable singularities -- at
        ``t = 0`` and at ``t = T/(4*beta)`` -- where the general formula
        divides by zero.  Both are handled with their analytic limits.
        """
        beta = self._rolloff
        sps = self._samples_per_symbol
        half = (self._span_symbols * sps) // 2

        # Time axis in symbol periods.
        tau = np.arange(-half, half + 1, dtype=float) / sps
        taps = np.zeros_like(tau)

        at_origin = np.isclose(tau, 0.0)
        at_singularity = np.isclose(np.abs(4.0 * beta * tau), 1.0)
        general = ~(at_origin | at_singularity)

        # Limit as t -> 0.
        taps[at_origin] = 1.0 + beta * (4.0 / np.pi - 1.0)

        # Limit as t -> +/- T/(4*beta).
        taps[at_singularity] = (beta / np.sqrt(2.0)) * (
            (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
            + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
        )

        t = tau[general]
        numerator = np.sin(np.pi * t * (1.0 - beta)) + 4.0 * beta * t * np.cos(
            np.pi * t * (1.0 + beta)
        )
        denominator = np.pi * t * (1.0 - (4.0 * beta * t) ** 2)
        taps[general] = numerator / denominator

        return taps

    def _rebuilt(self, samples_per_symbol: int) -> "RootRaisedCosine":
        return RootRaisedCosine(
            samples_per_symbol, rolloff=self._rolloff, span_symbols=self._span_symbols
        )

    def __repr__(self) -> str:
        return (
            f"RootRaisedCosine(sps={self._samples_per_symbol}, "
            f"rolloff={self._rolloff}, span={self._span_symbols})"
        )


class Rectangular(PulseShape):
    """A plain rectangular pulse -- each symbol held for its full period.

    Spectrally wasteful, so no real satellite uses it, but invaluable in
    tests: with ``samples_per_symbol=1`` it reduces to a pass-through,
    which lets us check the modulation and noise classes against textbook
    bit error rate formulas with no filtering in the way.
    """

    def __init__(self, samples_per_symbol: int) -> None:
        super().__init__(samples_per_symbol, name="Rect")

    def _build_taps(self) -> np.ndarray:
        return np.ones(self._samples_per_symbol, dtype=float)

    def _rebuilt(self, samples_per_symbol: int) -> "Rectangular":
        return Rectangular(samples_per_symbol)


__all__ = ["PulseShape", "Rectangular", "RootRaisedCosine"]
