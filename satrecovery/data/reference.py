"""Working out what was sent, when the dataset does not say.

RML24's published file has no transmitted bits (see
:mod:`satrecovery.data.layout`).  Without them no bit error rate can be
measured, and measuring bit error rates is the point of this project.

THE IDEA
--------
Take the *cleanest* recordings -- the ones labelled 20 dB -- and decode
them very carefully, offline, with every advantage a live receiver does
not have.  On a clean enough recording the decisions are right with
overwhelming probability, so they can stand in for the transmitted
bits.  Then add noise ourselves, by a measured amount, and ask every
receiver to recover the bits from the noisy copy.  What is being tested
is still real: the pulse distortion, frequency error, drifting gain and
phase all come from the satellite hardware and the air.  Only the noise
is ours, and noise is the one thing the simulator already models
exactly.

Two things keep this honest:

* **The reference is not one of the contestants.**  :class:`ReferenceDecoder`
  shares no code path with the receivers it labels for.  It looks at
  the whole recording at once, smooths forwards *and* backwards in
  time, and knows the modulation -- privileges no receiver gets.  A
  reference built by the classical receiver would grade the classical
  receiver against itself.
* **It refuses recordings it cannot vouch for.**  Every decoded
  recording gets an estimated symbol error rate from how tightly its
  points cluster.  Above the limit (one error per ten thousand symbols
  by default) it is rejected, and the rejection counts are reported.
  For 64QAM at 20 dB, measured: essentially every recording is
  rejected, because 64 points need more than 20 dB to be told apart.
  That is a finding, not a failure.

The reference also fixes how the frame is cut.  It finds where in the
recording the symbols fall and crops the start so that each symbol sits
within half a sample of a whole multiple of the oversampling factor --
the grid every receiver in the project samples on.  Without that, a
receiver whose timing estimate wrapped round to the other side of a
symbol would be graded one symbol out of step, and every bit would
look wrong.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.exceptions import ValidationError
from ..modulation.base import Modulation
from ..modulation.registry import ModulationRegistry
from ..receivers.stages import interpolate_cubic_many
from ..simulation.pulse import RootRaisedCosine
from . import layout
from .recording import Recording


class Reference:
    """What the reference decoder concluded about one recording. Immutable.

    The symbols and bits cover the *whole* cropped frame, because
    receivers take their preamble from the start.  Only the symbols in
    :attr:`graded` -- the middle, minus any the decoder could not vouch
    for -- are ever used to grade anyone.
    """

    __slots__ = (
        "_row", "_modulation", "_symbols", "_bits", "_crop_start", "_n_samples",
        "_timing_residual", "_frequency_hz", "_evm_percent", "_symbol_error_estimate",
        "_graded", "_accepted", "_reason", "_bits_per_symbol",
    )

    def __init__(
        self,
        row: int,
        modulation: str,
        symbols: np.ndarray,
        bits: np.ndarray,
        crop_start: int,
        n_samples: int,
        timing_residual: float,
        frequency_hz: float,
        evm_percent: float,
        symbol_error_estimate: float,
        graded: np.ndarray,
        accepted: bool,
        reason: str,
        bits_per_symbol: int,
    ) -> None:
        self._row = int(row)
        self._modulation = str(modulation)
        self._symbols = np.array(symbols, dtype=np.complex128, copy=True)
        self._bits = np.array(bits, dtype=np.uint8, copy=True)
        for array in (self._symbols, self._bits):
            array.flags.writeable = False
        self._crop_start = int(crop_start)
        self._n_samples = int(n_samples)
        self._timing_residual = float(timing_residual)
        self._frequency_hz = float(frequency_hz)
        self._evm_percent = float(evm_percent)
        self._symbol_error_estimate = float(symbol_error_estimate)
        self._graded = np.array(graded, dtype=bool, copy=True)
        if self._graded.size != self._symbols.size:
            raise ValidationError("graded", self._graded.size, "needs one entry per symbol")
        self._graded.flags.writeable = False
        self._accepted = bool(accepted)
        self._reason = str(reason)
        self._bits_per_symbol = int(bits_per_symbol)

    # -- read-only ------------------------------------------------------
    @property
    def row(self) -> int:
        return self._row

    @property
    def modulation(self) -> str:
        return self._modulation

    @property
    def symbols(self) -> np.ndarray:
        """Decided constellation points for every symbol of the cropped frame."""
        return self._symbols

    @property
    def bits(self) -> np.ndarray:
        """Bits for every symbol, edges included."""
        return self._bits

    @property
    def graded(self) -> np.ndarray:
        """Which symbols receivers are graded on."""
        return self._graded

    @property
    def graded_bits(self) -> np.ndarray:
        """The bits receivers are graded on."""
        return self._bits[np.repeat(self._graded, self._bits_per_symbol)]

    @property
    def n_symbols(self) -> int:
        return int(self._symbols.size)

    @property
    def n_ungraded(self) -> int:
        """Symbols left out: the edges, plus any the decoder could not read."""
        return int(self._graded.size - self._graded.sum())

    @property
    def crop(self) -> slice:
        """The samples of the recording that make up the frame."""
        return slice(self._crop_start, self._crop_start + self._n_samples)

    @property
    def timing_residual(self) -> float:
        """Where symbols fall after cropping, in samples; within +-0.5."""
        return self._timing_residual

    @property
    def frequency_hz(self) -> float:
        """Carrier frequency error the decoder measured."""
        return self._frequency_hz

    @property
    def evm_percent(self) -> float:
        """Error vector magnitude of the decoded middle, as a percentage."""
        return self._evm_percent

    @property
    def symbol_error_estimate(self) -> float:
        """Estimated chance that any one reference symbol is itself wrong."""
        return self._symbol_error_estimate

    @property
    def accepted(self) -> bool:
        """Clean enough to grade receivers against."""
        return self._accepted

    @property
    def reason(self) -> str:
        """Why it was accepted or rejected, in words."""
        return self._reason

    def metadata(self) -> dict[str, Any]:
        return {
            "row": self._row,
            "modulation": self._modulation,
            "accepted": self._accepted,
            "reason": self._reason,
            "n_symbols": self.n_symbols,
            "ungraded_symbols": self.n_ungraded,
            "crop_start": self._crop_start,
            "timing_residual": round(self._timing_residual, 4),
            "frequency_hz": round(self._frequency_hz, 1),
            "evm_percent": round(self._evm_percent, 2),
            "symbol_error_estimate": self._symbol_error_estimate,
        }

    def __bool__(self) -> bool:
        return self._accepted

    def __repr__(self) -> str:
        verdict = "accepted" if self._accepted else "rejected"
        return (
            f"Reference(row {self._row}, {self._modulation}, {verdict}, "
            f"EVM {self._evm_percent:.1f}%, est. symbol errors {self._symbol_error_estimate:.1e})"
        )


class ReferenceDecoder:
    """Offline, non-causal decoding of clean recordings into reference labels.

    Parameters
    ----------
    guard_symbols:
        Symbols at each end left ungraded.  The matched filter is 12
        symbols long, so the outer 6 at each end are missing part of
        their pulse; 8 leaves a margin.
    max_frequency_hz:
        Search range for the carrier error.  Measured on RML24, real
        errors are within about 2 kHz; searching wider only invites a
        noise peak to win.
    window:
        Symbols averaged when tracking gain and phase.  RML24's gain
        drifts by up to seven times across one recording, slowly enough
        that 32 symbols see it as constant.
    unreadable_distance:
        A symbol further than this many minimum distances from its
        decided point is *unreadable*: the reference cannot vouch for it,
        so nobody is graded on it.  On the clean RML24 recordings these
        are not noise -- noise that large would be seven standard
        deviations out -- but glitches: at 250 kbaud, one symbol in every
        92 is corrupted, like clockwork.
    max_unreadable:
        Highest fraction of unreadable symbols a recording may have.
        Above it, the recording is rejected rather than graded on a
        selection of its easiest symbols.
    max_symbol_error:
        Highest estimated error rate for the symbols that *are* graded.
    """

    #: Powers tried for blind carrier estimation, smallest first.  For each
    #: scheme the smallest one that collapses its constellation is used:
    #: 2 for BPSK, 4 for QPSK and QAM, 8 for 8PSK, 12 for DVB-S2 16APSK.
    TONE_POWERS = (2, 4, 8, 12)

    #: Rounds of decision-aided DC estimation (see decode, step 5).
    DC_PASSES = 2

    def __init__(
        self,
        guard_symbols: int = 8,
        max_frequency_hz: float = 5_000.0,
        window: int = 32,
        refine_window: int = 16,
        refine_passes: int = 3,
        unreadable_distance: float = 0.35,
        max_unreadable: float = 0.03,
        max_symbol_error: float = 1e-4,
    ) -> None:
        if int(guard_symbols) < 0:
            raise ValidationError("guard_symbols", guard_symbols, "cannot be negative")
        if int(window) < 2 or int(refine_window) < 2:
            raise ValidationError("window", window, "needs at least 2 symbols")
        if not 0.0 < float(max_symbol_error) < 1.0:
            raise ValidationError("max_symbol_error", max_symbol_error, "must lie in (0, 1)")
        self._guard = int(guard_symbols)
        self._max_frequency_hz = float(max_frequency_hz)
        self._window = int(window)
        self._refine_window = int(refine_window)
        self._refine_passes = int(refine_passes)
        self._max_symbol_error = float(max_symbol_error)
        if not 0.0 < float(unreadable_distance) <= 0.5:
            raise ValidationError("unreadable_distance", unreadable_distance, "must lie in (0, 0.5]")
        if not 0.0 <= float(max_unreadable) < 1.0:
            raise ValidationError("max_unreadable", max_unreadable, "must lie in [0, 1)")
        self._unreadable_distance = float(unreadable_distance)
        self._max_unreadable = float(max_unreadable)

    @property
    def guard_symbols(self) -> int:
        return self._guard

    @property
    def max_symbol_error(self) -> float:
        return self._max_symbol_error

    # ------------------------------------------------------------------
    def decode(self, recording: Recording) -> Reference:
        """Decode one recording. Never raises on a bad recording: rejects it."""
        if not isinstance(recording, Recording):
            raise ValidationError("recording", type(recording).__name__, "expected a Recording")
        scheme = ModulationRegistry.create(recording.modulation)
        sps = recording.samples_per_symbol
        rate = recording.symbol_rate

        # 1. Remove the radio's DC offset, then matched-filter the whole
        #    recording with an ordinary (not circular) convolution: a
        #    recording is a window on a stream, not one period of a loop.
        x = recording.samples - np.mean(recording.samples)
        taps = RootRaisedCosine(sps, layout.ROLLOFF).taps
        y = np.convolve(x, taps, mode="same")
        n_total = y.size // sps

        # 2. Timing: the instant, to a sixteenth of a sample, where the
        #    filtered signal carries most energy.
        timing = self._timing(y, sps, n_total)
        crop = int(math.floor(timing + 0.5))
        residual = timing - crop
        if crop >= sps:
            crop -= sps
            residual = timing - sps - crop
        n_symbols = (y.size - crop) // sps
        n_samples = n_symbols * sps
        positions = crop + np.arange(n_symbols) * sps + residual
        s = interpolate_cubic_many(y, positions)

        # 3. Carrier frequency, from the tone the right power leaves.
        power = self.tone_power(scheme)
        frequency = self._frequency(s, rate, power)
        spin = np.exp(-2j * np.pi * frequency * np.arange(n_symbols) / rate)
        s = s * spin

        # 4. Gain and phase, tracked blind, then refined from decisions.
        points, gains = self._track(s, scheme, power)

        # 5. The radio's DC offset, estimated properly.  Subtracting the
        #    recording's mean in step 1 removed the offset *plus* the
        #    data's own mean, which over 200 symbols is several percent
        #    of the signal -- measured, it alone pushed 8PSK's EVM from
        #    3.5% to 11.7% on a 30 dB test recording.  With decisions in
        #    hand, what is left after taking out gain x decided point is
        #    noise plus whatever DC remains; its average is the DC.
        for _ in range(self.DC_PASSES):
            decided = scheme.constellation[scheme.nearest_indices(points)]
            middle = slice(self._guard, max(n_symbols - self._guard, self._guard + 1))
            residue = (s - gains * decided) * np.conj(spin)
            dc = np.mean(residue[middle])
            s = s - dc * spin
            points, gains = self._track(s, scheme, power)

        # 6. Decide, and judge how far to trust the decisions.
        indices = scheme.nearest_indices(points)
        decided = scheme.constellation[indices]
        bits = scheme.indices_to_bits(indices)
        graded = np.zeros(n_symbols, dtype=bool)
        graded[self._guard : n_symbols - self._guard] = True
        middle = int(graded.sum())
        if middle < 16:
            return self._rejected(recording, scheme, decided, bits, crop, n_samples, residual,
                                  frequency, "too few symbols left after the guards")

        error = np.abs(points - decided)
        readable = error <= self._unreadable_distance * scheme.minimum_distance
        unreadable = float(np.mean(~readable[graded]))
        graded &= readable
        evm, ser = self._quality(points[graded], decided[graded], scheme)

        if unreadable > self._max_unreadable:
            accepted = False
            reason = (f"{unreadable:.1%} of symbols unreadable, above the {self._max_unreadable:.0%} limit "
                      f"(EVM {evm:.1f}%)")
        elif not np.isfinite(ser) or ser > self._max_symbol_error:
            accepted = False
            reason = (f"estimated {ser:.1e} symbol errors, above the {self._max_symbol_error:.0e} limit "
                      f"(EVM {evm:.1f}%)")
        else:
            accepted = True
            reason = (f"estimated {ser:.1e} symbol errors, {unreadable:.1%} unreadable "
                      f"(EVM {evm:.1f}%)")

        return Reference(
            row=recording.row,
            modulation=scheme.name,
            symbols=decided,
            bits=bits,
            crop_start=crop,
            n_samples=n_samples,
            timing_residual=residual,
            frequency_hz=frequency,
            evm_percent=evm,
            symbol_error_estimate=ser,
            graded=graded,
            accepted=accepted,
            reason=reason,
            bits_per_symbol=scheme.bits_per_symbol,
        )

    # ------------------------------------------------------------------
    @classmethod
    def tone_power(cls, scheme: Modulation) -> int:
        """Smallest power that collapses the constellation onto one angle.

        Judged the same way as
        :attr:`~satrecovery.modulation.base.Modulation.blind_frequency_recoverable`:
        by the magnitude-weighted agreement of the raised points'
        angles.  Derived from the geometry, so it needs no table.
        """
        points = scheme.constellation
        for m in cls.TONE_POWERS:
            raised = points**m
            magnitude = np.abs(raised)
            concentration = abs(np.sum(raised)) / max(float(np.sum(magnitude)), 1e-300)
            if concentration >= 0.4:
                return m
        return cls.TONE_POWERS[-1]

    def _timing(self, y: np.ndarray, sps: int, n_symbols: int) -> float:
        """Offset in [0, sps) that maximises energy at the symbol instants."""
        inner = np.arange(self._guard, max(n_symbols - self._guard - 1, self._guard + 1))
        steps = 16 * sps
        candidates = np.arange(steps) * (sps / steps)
        energy = [np.mean(np.abs(interpolate_cubic_many(y, inner * sps + t)) ** 2) for t in candidates]
        return float(candidates[int(np.argmax(energy))])

    def _frequency(self, s: np.ndarray, rate: float, power: int) -> float:
        """Blind carrier estimate within +-max_frequency_hz."""
        unit = s / max(float(np.mean(np.abs(s))), 1e-12)
        n_fft = 1 << 16
        spectrum = np.abs(np.fft.fft(unit**power, n_fft))
        freqs = np.fft.fftfreq(n_fft, d=1.0 / rate) / power
        allowed = np.abs(freqs) <= self._max_frequency_hz
        if not allowed.any():
            return 0.0
        return float(freqs[allowed][int(np.argmax(spectrum[allowed]))])

    def _track(
        self, s: np.ndarray, scheme: Modulation, power: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Undo slowly drifting gain and phase, looking both ways in time.

        Returns the corrected points and the complex gain divided out of
        each, so that ``points * gains`` is the input again.
        """
        points = scheme.constellation
        energy = float(np.mean(np.abs(points) ** 2))

        # Blind first: gain from the local power, and ONE phase for the
        # whole recording, from the tone left by raising to the right
        # power.  The constellation's own angle at that power is taken
        # out, or QPSK -- whose points sit at 45 degrees -- would be
        # locked 45 degrees off.
        #
        # One phase, not a phase per symbol.  An earlier version tracked
        # the tone symbol by symbol and unwrapped it; where the tone was
        # weak (16QAM) the unwrap occasionally jumped a quarter turn and
        # back, rotating a stretch of symbols onto *other* constellation
        # points.  The points still sat neatly on the grid, so the
        # quality check passed them -- a test against known bits found
        # nine wrong symbols in an accepted reference.  Slow phase drift
        # is left to the decision-directed passes below, which cannot
        # slip that way because they never unwrap anything.
        gain = np.sqrt(_smooth(np.abs(s) ** 2, self._window) / energy)
        own_angle = np.conj(np.sum(points**power))
        levelled = s / np.maximum(gain, 1e-12)
        phase = np.angle(np.sum(levelled**power) * own_angle) / power
        total = np.maximum(gain, 1e-12) * np.exp(1j * phase)
        z = s / total

        # Then decision-directed: estimate a complex gain from what each
        # symbol was decided to be, smooth it, divide it out.  Repeating
        # lets better decisions give a better estimate.
        for k in range(self._refine_passes):
            # Wide first, to follow the drift without chasing noise, then
            # down to the narrow window for the final estimate.
            window = self._window if k == 0 else self._refine_window
            decided = points[scheme.nearest_indices(z)]
            numerator = _smooth(z * np.conj(decided), window)
            denominator = _smooth(np.abs(decided) ** 2, window)
            step = np.where(np.abs(numerator) > 1e-12, numerator / denominator, 1.0)
            z = z / step
            total = total * step
        return z, total

    def _rejected(self, recording, scheme, decided, bits, crop, n_samples, residual, frequency, reason):
        return Reference(
            row=recording.row, modulation=scheme.name, symbols=decided, bits=bits,
            crop_start=crop, n_samples=n_samples, timing_residual=residual,
            frequency_hz=frequency, evm_percent=float("inf"), symbol_error_estimate=float("inf"),
            graded=np.zeros(decided.size, dtype=bool), accepted=False, reason=reason,
            bits_per_symbol=scheme.bits_per_symbol,
        )

    @staticmethod
    def _quality(points: np.ndarray, decided: np.ndarray, scheme: Modulation) -> tuple[float, float]:
        """EVM (%) and an estimate of the symbol error rate.

        The noise level is estimated *robustly*, from the median distance
        to the decided point.  Assuming circular Gaussian noise, that
        distance follows a Rayleigh law whose median is
        ``sigma * sqrt(2 ln 2)``.  Each point is then charged
        ``Q(d / 2 sigma)`` for every neighbour within 1.5 minimum
        distances -- the nearest-neighbour union bound -- and the charges
        are averaged over the constellation.
        """
        if points.size == 0:
            return float("inf"), float("inf")
        error = np.abs(points - decided)
        energy = float(np.mean(np.abs(scheme.constellation) ** 2))
        evm = float(np.sqrt(np.mean(error**2) / energy) * 100.0)

        sigma = float(np.median(error)) / math.sqrt(2.0 * math.log(2.0))
        if sigma <= 0:
            return evm, 0.0
        constellation = scheme.constellation
        dmin = scheme.minimum_distance
        gaps = np.abs(constellation[:, None] - constellation[None, :])
        np.fill_diagonal(gaps, np.inf)
        per_point = [
            sum(0.5 * math.erfc(g / (2.0 * sigma) / math.sqrt(2.0)) for g in row if g <= 1.5 * dmin)
            for row in gaps
        ]
        return evm, float(np.mean(per_point))


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that stays unbiased at the ends."""
    kernel = np.ones(int(window))
    total = np.convolve(values, kernel, mode="same")
    count = np.convolve(np.ones(values.size), kernel, mode="same")
    return total / count


__all__ = ["Reference", "ReferenceDecoder"]
