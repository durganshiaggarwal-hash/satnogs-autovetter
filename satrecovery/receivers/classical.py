"""The classical receiver: the baseline everything else is measured against.

This is the traditional way of recovering bits from a damaged signal,
built entirely from signal-processing mathematics with nothing learned
from data.  Six stages in order, then a demapper::

    matched filter -> coarse frequency -> timing -> gain
                   -> carrier loop -> equaliser -> demap

It matters that this is right before anything else is built.  Phase 4's
neural networks and Phase 5's hybrid are both measured against this, so
if the baseline is quietly broken, every later comparison is meaningless
in a way that is very hard to notice -- the plots still look plausible.
That is why Phase 2 ends by checking the measured bit error rate against
published theory rather than against intuition.

DESIGN NOTE: COMPOSITION BUILDS THE BEHAVIOUR
---------------------------------------------
:class:`ClassicalReceiver` contains almost no logic of its own.  It
assembles a :class:`~satrecovery.receivers.base.ReceiverChain` out of
stage objects, runs it, and hands the result to a demapper.  Changing the
receiver means changing which objects go in the list.

That is what makes Phase 5 cheap.  A hybrid receiver is this same chain
with the demapper swapped for a neural network -- no inheritance
gymnastics, no copied code, just a different object in the same slot.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.exceptions import ValidationError
from ..core.frame import Frame
from ..modulation.base import Modulation
from ..modulation.registry import ModulationRegistry
from ..simulation.pulse import PulseShape, RootRaisedCosine
from .base import Receiver, ReceiverChain, RecoveryContext
from .demapper import Demapper, SoftDemapper
from .result import RecoveryResult
from .stages import (
    AutomaticGainControl,
    CoarseFrequencyCorrection,
    DecisionDirectedPLL,
    GardnerTimingRecovery,
    LMSEqualizer,
    MatchedFilterStage,
)


class ClassicalReceiver(Receiver):
    """A full signal-processing receive chain.

    Parameters
    ----------
    pulse:
        The pulse shaping filter to match against.  Defaults to the
        root-raised-cosine the simulator uses.
    pilot_symbols:
        How many symbols at the start of the frame the receiver already
        knows.  See the note below -- this is not optional in any real
        system, and sixteen is a short preamble by practical standards.
    use_equalizer:
        Whether to include the adaptive equaliser.  **Off by default**,
        which is a measured decision rather than an oversight: on the
        RML24 channel the echoes arrive a single sample apart -- an
        eighth of a symbol -- and the matched filter already absorbs
        them, so the equaliser finds no inter-symbol interference to
        remove and merely adapts against noise, costing a little
        accuracy. Switch it on when the echoes are a symbol or more
        apart, which is what ``tap_spacing`` on
        :class:`~satrecovery.channel.effects.MultipathRicianFading`
        produces.
    frequency_search:
        Run the chain both with and without blind frequency correction
        and keep the better result.  See :meth:`_best_attempt`.
    demapper:
        How symbols become bits.  Defaults to
        :class:`~satrecovery.receivers.demapper.SoftDemapper`.
    modulation:
        Force a particular scheme.  By default the receiver reads it from
        the frame -- this project measures *recovery*, so which scheme is
        in use is taken as known, exactly as it is in a real link where
        both ends agreed in advance.

    Notes
    -----
    **Why a known preamble is not cheating.** A constellation has
    rotational symmetry -- turn QPSK by 90 degrees and it is
    indistinguishable from where it started.  So no receiver, however
    clever, can work out which way up it is from the signal alone: the
    information is not there to be found.  Every real link solves this the
    same way, by beginning each frame with a short pattern the receiver
    already knows.  This receiver uses the first ``pilot_symbols`` symbols
    for that, and records ``n_pilot_bits`` in the result so that Phase 6
    can exclude them from the error count.
    """

    #: How many spectral peaks are tried as frequency hypotheses.
    FREQUENCY_HYPOTHESES = 5
    #: A lock quality at or above this ends the search early.
    LOCK_ENOUGH = 0.5

    def __init__(
        self,
        pulse: PulseShape | None = None,
        pilot_symbols: int = 16,
        use_equalizer: bool = False,
        demapper: Demapper | None = None,
        modulation: Modulation | None = None,
        frequency_search: bool = True,
        name: str | None = None,
    ) -> None:
        if int(pilot_symbols) < 0:
            raise ValidationError(
                "pilot_symbols", pilot_symbols, "cannot be negative"
            )
        if pulse is not None and not isinstance(pulse, PulseShape):
            raise ValidationError(
                "pulse", type(pulse).__name__, "expected a PulseShape instance"
            )
        if demapper is not None and not isinstance(demapper, Demapper):
            raise ValidationError(
                "demapper", type(demapper).__name__, "expected a Demapper"
            )

        super().__init__(name or "Classical")
        self._pulse = pulse
        self._pilot_symbols = int(pilot_symbols)
        self._use_equalizer = bool(use_equalizer)
        self._demapper = demapper or SoftDemapper()
        self._modulation = modulation
        self._frequency_search = bool(frequency_search)

    # ------------------------------------------------------------------
    # Read-only configuration
    # ------------------------------------------------------------------
    @property
    def pilot_symbols(self) -> int:
        """Length of the known preamble, in symbols."""
        return self._pilot_symbols

    @property
    def uses_equalizer(self) -> bool:
        """Whether the adaptive equaliser is in the chain."""
        return self._use_equalizer

    @property
    def demapper(self) -> Demapper:
        """The demapper turning symbols into bits."""
        return self._demapper

    # ------------------------------------------------------------------
    # The chain
    # ------------------------------------------------------------------
    def build_chain(
        self,
        samples_per_symbol: int,
        use_coarse_frequency: bool = True,
        frequency_candidate: int = 0,
    ) -> ReceiverChain:
        """Assemble the repair stages, in the order physics requires.

        The ordering is the real content of this method, and none of it
        is arbitrary:

        1. **Matched filter** first, because every later estimate is made
           from these samples and they should be as clean as possible.
        2. **Coarse frequency** next. A large frequency error spins the
           constellation so fast that timing recovery cannot lock, so
           most of it has to go before anything else is attempted.
        3. **Timing recovery**, which also decimates to one sample per
           symbol. Everything after this runs at the symbol rate.
        4. **Gain control**, so the constellation is the right size
           before anything measures distances on it.
        5. **Carrier loop**, cleaning up the rotation the coarse stage
           left behind.
        6. **Equaliser** last, because it adapts against its own
           decisions and those are only trustworthy once the
           constellation is upright and correctly scaled.
        """
        if samples_per_symbol < 1:
            raise ValidationError(
                "samples_per_symbol", samples_per_symbol, "must be at least 1"
            )

        # The frame decides the rate, not the receiver.  A pulse passed in
        # at construction keeps its family and parameters but is rebuilt
        # at the frame's rate, so one receiver can serve frames at 10, 5
        # and 2 samples per symbol -- which one RML24 file contains.
        pulse = (
            self._pulse.at_rate(samples_per_symbol)
            if self._pulse is not None
            else RootRaisedCosine(samples_per_symbol)
        )
        chain = ReceiverChain(name="ClassicalChain")

        # Frequency correction comes FIRST, before the matched filter.
        # The filter is circular, and a frequency offset makes the frame's
        # end and start disagree in phase; filtering first blended that
        # mismatch into symbol 0 of every frame with Doppler.  The stage
        # still estimates on a matched-filtered copy, so it keeps the
        # noise rejection it needs.  See CoarseFrequencyCorrection.
        if use_coarse_frequency:
            chain.add(CoarseFrequencyCorrection(prefilter=pulse, candidate=frequency_candidate))

        chain.add(MatchedFilterStage(pulse))

        if samples_per_symbol >= 2:
            chain.add(GardnerTimingRecovery())

        chain.add(AutomaticGainControl())
        chain.add(DecisionDirectedPLL())

        if self._use_equalizer:
            chain.add(LMSEqualizer())

        return chain

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------
    def _recover(self, frame: Frame) -> RecoveryResult:
        scheme = self._resolve_modulation(frame)
        pilot = self._extract_pilot(frame, scheme)

        cleaned, context, chose = self._best_attempt(frame, scheme, pilot)

        bits, confidence, llr = self._demapper.demap(cleaned.samples, scheme)

        diagnostics: dict[str, Any] = context.diagnostics
        diagnostics["frequency_branch"] = chose
        diagnostics["demapper"] = self._demapper.name
        diagnostics["n_pilot_symbols"] = self._pilot_symbols
        diagnostics["n_pilot_bits"] = self._pilot_symbols * scheme.bits_per_symbol
        diagnostics["evm_percent"] = self._evm(cleaned.samples, scheme)

        return RecoveryResult(
            bits=bits,
            confidence=confidence,
            symbols=cleaned.samples,
            modulation=scheme.name,
            receiver=self._name,
            llr=llr,
            diagnostics=diagnostics,
        )

    def trace(self, frame: Frame) -> tuple[RecoveryResult, list[tuple[str, Any]]]:
        """Recover a frame, keeping what the signal looked like after each stage.

        What the GUI's step-by-step view and the API's ``/stages``
        endpoint show.  The recovery runs normally first, so the trace
        follows the same branch -- with or without coarse frequency
        correction -- that the receiver actually chose, and the last
        snapshot is exactly the symbols the bits were decided from.

        Returns
        -------
        (result, snapshots)
            ``snapshots`` is a list of ``(stage name, Signal)`` pairs,
            starting with ``"received"``.
        """
        result = self.recover(frame)
        scheme = self._resolve_modulation(frame)
        branch = str(result.diagnostics.get("frequency_branch", "corrected"))
        context = RecoveryContext(
            modulation=scheme,
            samples_per_symbol=frame.samples_per_symbol,
            sample_rate=frame.signal.sample_rate,
            pilot_symbols=self._extract_pilot(frame, scheme),
        )
        chain = self.build_chain(
            frame.samples_per_symbol,
            use_coarse_frequency=branch.startswith("corrected"),
            frequency_candidate=int(result.diagnostics.get("frequency_hypothesis") or 0),
        )

        snapshots: list[tuple[str, Any]] = [("received", frame.signal)]
        signal = frame.signal
        for stage in chain:
            signal = stage.apply(signal, context)
            snapshots.append((stage.name, signal))
        return result, snapshots

    def _best_attempt(
        self, frame: Frame, scheme: Modulation, pilot: np.ndarray | None
    ):
        """Run the chain under several frequency hypotheses; keep the one that locks.

        The blind frequency estimator is excellent when it works and
        destructive when it does not, and from inside the receiver there
        is no reliable way to tell the two apart beforehand -- a
        confidence threshold that separates them on one frame length
        fails on another.

        So rather than guess, the receiver tries it both ways and keeps
        whichever produced the constellation that is more firmly *locked*
        (see :meth:`lock_quality`).  That measurement uses no knowledge
        of what was transmitted, so this is a legitimate thing for a real
        receiver to do, and it is close to what acquisition logic does in
        practice.

        It used to keep the branch with the lower error vector magnitude,
        and the real recordings showed why that is wrong for 8PSK: a
        constellation that is still *spinning* smears its points round
        the ring, and every point on a ring is close to one of eight
        targets, so the smear scores a lower EVM than a correctly locked
        but noisy constellation.  On one RML24 frame at 8 dB the receiver
        threw away a correct 577 Hz estimate that way, and lost the
        frame.
        The cost is running the chain twice, which is cheap next to
        getting a whole frame wrong.

        Where the constellation is unsuitable for blind estimation at all
        (16APSK), both branches would be identical, so only one runs.
        """
        # Hypotheses, in order: the strongest spectral peak; no
        # correction at all; then the next few peaks.  The first is right
        # almost always, so the search stops there if the constellation
        # locks.  Only if it does not are the others tried -- and the
        # winner is whichever locks best, never a threshold's say-so.
        # Measured on a real 8PSK recording at 10 dB: the strongest peak
        # was noise (lock 0.07), and the true tone was the fourth (lock
        # 0.75).  This is what acquisition logic in real receivers does.
        attempts: list[tuple[bool, int]] = [(True, 0)]
        if self._frequency_search and scheme.blind_frequency_recoverable:
            attempts += [(False, 0)] + [(True, k) for k in range(1, self.FREQUENCY_HYPOTHESES)]

        best: tuple[float, bool, Any, RecoveryContext] | None = None
        for use_coarse, candidate in attempts:
            if best is not None and -best[0] >= self.LOCK_ENOUGH:
                break
            context = RecoveryContext(
                modulation=scheme,
                samples_per_symbol=frame.samples_per_symbol,
                sample_rate=frame.signal.sample_rate,
                pilot_symbols=pilot,
            )
            chain = self.build_chain(
                frame.samples_per_symbol,
                use_coarse_frequency=use_coarse,
                frequency_candidate=candidate,
            )
            cleaned = chain.apply(frame.signal, context)
            score = -self.lock_quality(cleaned.samples, scheme)
            if best is None or score < best[0]:
                best = (score, use_coarse, cleaned, context)
                context.record("frequency_hypothesis", candidate if use_coarse else None)
                context.record("lock_quality", round(-score, 4))

        assert best is not None
        _, use_coarse, cleaned, context = best
        chose = "corrected" if use_coarse else "uncorrected"
        if len(attempts) == 1:
            chose = "corrected (only option)"
        return cleaned, context, chose

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_modulation(self, frame: Frame) -> Modulation:
        """Which constellation to decode against."""
        if self._modulation is not None:
            return self._modulation
        return ModulationRegistry.create(frame.modulation)

    def _extract_pilot(
        self, frame: Frame, scheme: Modulation
    ) -> np.ndarray | None:
        """The known preamble, as symbols.

        Taken from the frame's own transmitted symbols where they are
        available, and otherwise rebuilt by mapping the first few known
        bits.  Both routes use only what a real receiver would already
        have agreed with the transmitter in advance.
        """
        if self._pilot_symbols == 0:
            return None

        n = self._pilot_symbols

        transmitted = frame.tx_symbols
        if transmitted is not None and transmitted.size >= n:
            return np.array(transmitted[:n], dtype=np.complex128)

        needed = n * scheme.bits_per_symbol
        if frame.bits.size >= needed:
            return scheme.map_bits(frame.bits[:needed])

        return None

    @staticmethod
    def lock_quality(symbols: np.ndarray, scheme: Modulation) -> float:
        """How firmly the constellation stands still: 0 (spinning) to 1.

        Raise every symbol to the constellation's rotational symmetry and
        the data cancels, exactly as in blind frequency estimation; what
        remains points the same way for every symbol if, and only if,
        the constellation is not rotating.  The length of the average,
        relative to the average length, measures that agreement.  Noise
        lowers it, but rotation destroys it, which is the difference that
        matters here.  Magnitudes are capped at the fourth power for the
        same reason as in
        :class:`~satrecovery.receivers.stages.CoarseFrequencyCorrection`.
        """
        symbols = np.asarray(symbols, dtype=np.complex128)
        if symbols.size == 0:
            return 0.0
        m = scheme.rotational_symmetry
        magnitude = np.abs(symbols)
        unit = symbols / np.maximum(magnitude, 1e-30)
        weight = magnitude ** min(m, 4)
        total = float(np.sum(weight))
        if total <= 0:
            return 0.0
        return float(np.abs(np.sum(weight * unit**m)) / total)

    @staticmethod
    def _evm(symbols: np.ndarray, scheme: Modulation) -> float:
        """Error vector magnitude, as a percentage.

        The standard way of saying how far a received constellation sits
        from where it should: the root-mean-square distance between each
        symbol and the point it was decided to be, relative to the
        constellation's own size.  Engineers quote it the way they quote
        signal-to-noise ratio, and unlike bit error rate it can be
        measured on a live link without knowing what was sent.
        """
        if symbols.size == 0:
            return 0.0
        constellation = scheme.constellation
        distances = np.abs(symbols[:, None] - constellation[None, :])
        residual = np.min(distances, axis=1)
        reference = np.sqrt(np.mean(np.abs(constellation) ** 2))
        return round(float(np.sqrt(np.mean(residual**2)) / reference * 100.0), 3)

    def describe(self) -> str:
        parts = [
            f"{self._name}",
            f"pilot={self._pilot_symbols}",
            f"demapper={self._demapper.name}",
        ]
        if self._use_equalizer:
            parts.append("equaliser")
        if self._frequency_search:
            parts.append("freq search")
        return ", ".join(parts)


__all__ = ["ClassicalReceiver"]
