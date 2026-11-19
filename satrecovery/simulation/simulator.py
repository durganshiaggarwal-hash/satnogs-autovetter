"""The signal simulator -- our transmitter plus channel.

This is the class the GUI drives.  Every time the user moves a slider,
the backend asks the simulator for a fresh frame under the new
conditions, and the plots redraw.

WHY A SIMULATOR AT ALL, GIVEN WE HAVE RML24?
--------------------------------------------
RML24 is a fixed recording: its frames were captured under the conditions
the authors chose, and nothing can change them afterwards.  Our project
has to let a user *turn the noise up and watch what happens*, which a
recording cannot do.  So the two sources play different roles:

* the **simulator** drives the interactive experiments and generates
  unlimited labelled training data on a laptop;
* **RML24** is the realistic benchmark the trained models are finally
  measured against, including the hardware imperfections no simulator
  reproduces.

Both produce :class:`~satrecovery.core.frame.Frame` objects, so every
receiver, metric and plot works with either without modification.  That
interchangeability is the whole point of having defined ``Frame`` as a
type rather than passing tuples of arrays around.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ..channel.pipeline import ChannelFactory, ChannelPipeline
from ..core.conditions import ChannelConditions
from ..core.exceptions import ValidationError
from ..core.frame import Frame
from ..core.signal import Signal
from ..modulation.base import Modulation
from ..modulation.registry import ModulationRegistry
from .pulse import PulseShape, RootRaisedCosine


class SignalSimulator:
    """Generates labelled frames: random bits in, received signal out.

    Parameters
    ----------
    sample_rate:
        Sampling frequency in hertz.  Defaults to 1 MHz, matching RML24.
    frame_length:
        Target number of samples per frame.  Defaults to 2048, matching
        RML24.  Rounded down to a whole number of symbols.
    samples_per_symbol:
        Oversampling factor.  Eight samples per symbol leaves plenty of
        room for the timing recovery loop in Phase 2.
    pulse:
        Pulse shaping filter.  Defaults to a root-raised-cosine with
        roll-off 0.35, again matching RML24.
    seed:
        Seed for the random generator.  Set it and every run is
        reproducible -- which for a project whose output is a set of
        measured results is not a nicety but a requirement.
    """

    #: Defaults chosen to match RML24's published parameters.
    DEFAULT_SAMPLE_RATE = 1_000_000.0
    DEFAULT_FRAME_LENGTH = 2048
    DEFAULT_SAMPLES_PER_SYMBOL = 8

    def __init__(
        self,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        frame_length: int = DEFAULT_FRAME_LENGTH,
        samples_per_symbol: int = DEFAULT_SAMPLES_PER_SYMBOL,
        pulse: PulseShape | None = None,
        seed: int | None = None,
    ) -> None:
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValidationError(
                "sample_rate", sample_rate, "must be a positive finite number"
            )
        if int(samples_per_symbol) < 1:
            raise ValidationError(
                "samples_per_symbol", samples_per_symbol, "must be at least 1"
            )

        self._sample_rate = float(sample_rate)
        self._samples_per_symbol = int(samples_per_symbol)

        n_symbols = int(frame_length) // self._samples_per_symbol
        if n_symbols < 1:
            raise ValidationError(
                "frame_length",
                frame_length,
                f"must be at least samples_per_symbol "
                f"({self._samples_per_symbol})",
            )
        self._n_symbols = n_symbols

        # Dependency injection: the caller may supply any PulseShape.
        # The simulator is written against the abstract base class, so it
        # works with a filter type that does not exist yet.
        if pulse is None:
            pulse = RootRaisedCosine(self._samples_per_symbol)
        elif not isinstance(pulse, PulseShape):
            raise ValidationError(
                "pulse", type(pulse).__name__, "expected a PulseShape instance"
            )
        elif pulse.samples_per_symbol != self._samples_per_symbol:
            raise ValidationError(
                "pulse",
                pulse.samples_per_symbol,
                f"filter is built for {pulse.samples_per_symbol} samples per "
                f"symbol but the simulator uses {self._samples_per_symbol}",
            )
        self._pulse = pulse

        self._seed = seed
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------
    @property
    def sample_rate(self) -> float:
        """Sampling frequency in hertz."""
        return self._sample_rate

    @property
    def samples_per_symbol(self) -> int:
        """Oversampling factor."""
        return self._samples_per_symbol

    @property
    def n_symbols(self) -> int:
        """Symbols carried by one frame."""
        return self._n_symbols

    @property
    def frame_length(self) -> int:
        """Samples per frame (a whole number of symbols)."""
        return self._n_symbols * self._samples_per_symbol

    @property
    def symbol_rate(self) -> float:
        """Symbol rate in baud."""
        return self._sample_rate / self._samples_per_symbol

    @property
    def pulse(self) -> PulseShape:
        """The pulse shaping filter in use."""
        return self._pulse

    @property
    def seed(self) -> int | None:
        """The seed this simulator was created with, if any."""
        return self._seed

    # ------------------------------------------------------------------
    # Link budget arithmetic
    # ------------------------------------------------------------------
    def effective_ebn0_db(
        self, snr_db: float, modulation: str | Modulation
    ) -> float:
        """Convert a channel signal-to-noise ratio into energy per bit.

        **This conversion is the single easiest thing to get wrong in the
        whole project, so it lives in one place.**

        ``ChannelConditions.snr_db`` is the ratio of signal power to noise
        power measured on the *oversampled* samples -- the numbers the
        user sees on the GUI slider.  Textbook bit error rate curves are
        plotted against Eb/N0, the energy per *bit* over the noise
        spectral density.  The two differ by two factors:

        * **Processing gain.**  The matched filter integrates each symbol
          over all ``samples_per_symbol`` samples while rejecting the
          out-of-band noise, which buys ``10*log10(sps)`` dB -- about 9 dB
          at eight samples per symbol.
        * **Bits per symbol.**  Energy per symbol is shared among the
          bits the symbol carries, costing ``10*log10(k)`` dB.

        So a frame generated at 0 dB on the slider lands near 6 dB Eb/N0
        for QPSK, where the theoretical bit error rate is about 2.3e-3 --
        which is what we measure.  Forgetting the processing gain would
        make a perfectly correct receiver look nine decibels too good,
        and nothing in the plot would look wrong.

        Returns
        -------
        float
            Eb/N0 in dB, suitable for passing to
            :meth:`~satrecovery.modulation.base.Modulation.theoretical_ber`.
        """
        scheme = self._resolve_modulation(modulation)
        processing_gain_db = 10.0 * np.log10(self._samples_per_symbol)
        bits_penalty_db = 10.0 * np.log10(scheme.bits_per_symbol)
        return float(snr_db) + processing_gain_db - bits_penalty_db

    def snr_db_for_ebn0(
        self, ebn0_db: float, modulation: str | Modulation
    ) -> float:
        """Inverse of :meth:`effective_ebn0_db`.

        Lets the benchmark screen sweep the axis the literature uses
        while still driving the simulator in the units the GUI exposes.
        """
        scheme = self._resolve_modulation(modulation)
        processing_gain_db = 10.0 * np.log10(self._samples_per_symbol)
        bits_penalty_db = 10.0 * np.log10(scheme.bits_per_symbol)
        return float(ebn0_db) - processing_gain_db + bits_penalty_db

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        """Restart the random generator, for reproducible repeat runs."""
        self._seed = seed if seed is not None else self._seed
        self._rng = np.random.default_rng(self._seed)

    def generate_frame(
        self,
        modulation: str | Modulation,
        conditions: ChannelConditions | None = None,
        bits: np.ndarray | None = None,
    ) -> Frame:
        """Generate one labelled frame.

        Parameters
        ----------
        modulation:
            Either a scheme name (``"QPSK"``) or a
            :class:`~satrecovery.modulation.base.Modulation` object.
            Accepting both is deliberate: the API layer has a string, and
            internal code has an object, and neither should have to
            convert for the other's benefit.
        conditions:
            Channel conditions.  Defaults to a moderate 10 dB AWGN link.
        bits:
            Specific bits to transmit.  Random bits by default.  Passing
            them in lets a test transmit a known pattern, and lets the
            GUI keep the payload fixed while the user varies the channel
            -- so what changes on screen is the channel's doing and not
            the payload's.
        """
        scheme = self._resolve_modulation(modulation)
        conditions = conditions or ChannelConditions()

        n_bits = self._n_symbols * scheme.bits_per_symbol
        if bits is None:
            bits = self._rng.integers(0, 2, size=n_bits, dtype=np.uint8)
        else:
            bits = np.asarray(bits, dtype=np.uint8).ravel()
            if bits.size != n_bits:
                raise ValidationError(
                    "bits",
                    bits.size,
                    f"{scheme.name} needs exactly {n_bits} bits for a "
                    f"{self._n_symbols}-symbol frame",
                )

        # --- Transmitter -------------------------------------------
        symbols = scheme.map_bits(bits)
        waveform = self._pulse.shape(symbols)
        transmitted = Signal(
            waveform, self._sample_rate, description=f"tx:{scheme.name}"
        )

        # --- Channel -----------------------------------------------
        channel = ChannelFactory.from_conditions(conditions)
        received = channel.apply(transmitted, self._rng)

        # --- Label it ----------------------------------------------
        return Frame(
            signal=received,
            bits=bits,
            modulation=scheme.name,
            conditions=conditions,
            tx_symbols=symbols,
            samples_per_symbol=self._samples_per_symbol,
            symbol_rate=self.symbol_rate,
            source="simulated",
        )

    def generate_batch(
        self,
        modulations: Iterable[str | Modulation],
        conditions_list: Iterable[ChannelConditions],
        frames_each: int = 1,
    ) -> list[Frame]:
        """Generate frames for every (modulation, conditions) combination.

        Used to build training sets and to run benchmark sweeps.
        """
        if int(frames_each) < 1:
            raise ValidationError(
                "frames_each", frames_each, "must be at least 1"
            )

        schemes = [self._resolve_modulation(m) for m in modulations]
        conditions_list = list(conditions_list)
        if not schemes or not conditions_list:
            return []

        frames: list[Frame] = []
        for scheme in schemes:
            for conditions in conditions_list:
                for _ in range(int(frames_each)):
                    frames.append(self.generate_frame(scheme, conditions))
        return frames

    def snr_sweep(
        self,
        modulation: str | Modulation,
        snr_values_db: Sequence[float],
        base_conditions: ChannelConditions | None = None,
        frames_each: int = 1,
    ) -> list[Frame]:
        """Generate frames across a range of signal-to-noise ratios.

        The workhorse of the benchmark screen: holding everything else
        fixed and sweeping the noise is what produces a bit-error-rate
        curve.  ``ChannelConditions.clone`` makes "everything else fixed"
        literally true, which is what makes the curve interpretable.
        """
        base = base_conditions or ChannelConditions()
        conditions_list = [base.clone(snr_db=float(snr)) for snr in snr_values_db]
        return self.generate_batch([modulation], conditions_list, frames_each)

    def transmit_only(
        self, modulation: str | Modulation, bits: np.ndarray | None = None
    ) -> Frame:
        """Generate a frame through a perfect channel.

        The reference the GUI draws as the clean constellation, and the
        sanity check every receiver must pass with zero bit errors.
        """
        scheme = self._resolve_modulation(modulation)
        ideal: ChannelPipeline = ChannelFactory.ideal()

        n_bits = self._n_symbols * scheme.bits_per_symbol
        if bits is None:
            bits = self._rng.integers(0, 2, size=n_bits, dtype=np.uint8)

        symbols = scheme.map_bits(bits)
        waveform = self._pulse.shape(symbols)
        signal = ideal.apply(
            Signal(waveform, self._sample_rate, description=f"tx:{scheme.name}"),
            self._rng,
        )

        return Frame(
            signal=signal,
            bits=bits,
            modulation=scheme.name,
            conditions=ChannelConditions(snr_db=30.0),
            tx_symbols=symbols,
            samples_per_symbol=self._samples_per_symbol,
            symbol_rate=self.symbol_rate,
            source="simulated-ideal",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_modulation(modulation: str | Modulation) -> Modulation:
        """Accept either a name or an object and always return an object.

        A tiny piece of polymorphism that removes a whole category of
        boilerplate from every caller.
        """
        if isinstance(modulation, Modulation):
            return modulation
        return ModulationRegistry.create(modulation)

    def __repr__(self) -> str:
        return (
            f"SignalSimulator(sample_rate={self._sample_rate:g} Hz, "
            f"frame_length={self.frame_length}, "
            f"sps={self._samples_per_symbol}, "
            f"symbol_rate={self.symbol_rate:g} Bd, "
            f"pulse={self._pulse!r}, seed={self._seed})"
        )


__all__ = ["SignalSimulator"]
