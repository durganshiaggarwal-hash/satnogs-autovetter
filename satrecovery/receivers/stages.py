"""The repair stages of a classical receiver.

Phase 1 broke the signal in five ways.  These stages undo four of them.
(The fifth, noise, cannot be undone -- only decided around, which is the
demapper's job.)

===============================  ==========================================
Stage                            Undoes
===============================  ==========================================
:class:`MatchedFilterStage`      Noise outside the signal's own band
:class:`CoarseFrequencyCorrection`  Most of the Doppler shift, blind
:class:`GardnerTimingRecovery`   Sampling at the wrong instants
:class:`AutomaticGainControl`    Fading having changed the signal's size
:class:`DecisionDirectedPLL`     Whatever rotation is left
:class:`LMSEqualizer`            Echoes smearing symbols into each other
===============================  ==========================================

Two ideas run through nearly all of them.

**Blind estimation.** The receiver does not get told the frequency
offset or the timing -- it has to work them out from the signal itself.
Each stage finds a quantity that *should* be zero when everything is
aligned, measures how far from zero it actually is, and corrects.

**Feedback loops.** Most corrections are not one-shot calculations but
control loops that converge over the frame: measure the error, nudge the
estimate, measure again.  They all use the same second-order
proportional-integral loop, so :func:`pi_loop_gains` is written once.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.exceptions import ValidationError
from ..simulation.pulse import PulseShape
from .base import ReceiverStage, RecoveryContext


# --------------------------------------------------------------------------
# Shared loop mathematics
# --------------------------------------------------------------------------
def pi_loop_gains(
    normalised_bandwidth: float, damping: float = 0.707
) -> tuple[float, float]:
    """Proportional and integral gains for a second-order tracking loop.

    Every feedback loop in this module faces the same trade-off.  A wide
    loop locks quickly and follows a drifting error, but lets noise into
    the estimate.  A narrow loop is steady but slow, and may never catch
    up with a moving target.  ``normalised_bandwidth`` -- loop bandwidth
    as a fraction of the sample rate -- is that dial.

    The damping factor controls overshoot; 0.707 is the textbook default
    that reaches lock fastest without ringing around it.

    Returns
    -------
    (proportional, integral)
        Gains for the update ``estimate += kp * error + accumulated``.
    """
    if not 0 < normalised_bandwidth < 0.5:
        raise ValidationError(
            "normalised_bandwidth",
            normalised_bandwidth,
            "must be between 0 and 0.5",
        )
    if damping <= 0:
        raise ValidationError("damping", damping, "must be positive")

    theta = normalised_bandwidth / (damping + 1.0 / (4.0 * damping))
    denominator = 1.0 + 2.0 * damping * theta + theta * theta
    kp = 4.0 * damping * theta / denominator
    ki = 4.0 * theta * theta / denominator
    return float(kp), float(ki)


def interpolate_cubic_many(
    samples: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Vectorised form of :func:`interpolate_cubic`.

    The block timing estimator evaluates every symbol position at a
    dozen or more candidate offsets, so a Python loop over positions
    would dominate the whole receiver's runtime.
    """
    n = samples.size
    base = np.floor(positions).astype(np.int64)
    mu = positions - base

    x0 = samples[(base - 1) % n]
    x1 = samples[base % n]
    x2 = samples[(base + 1) % n]
    x3 = samples[(base + 2) % n]

    c0 = -mu * (mu - 1.0) * (mu - 2.0) / 6.0
    c1 = (mu + 1.0) * (mu - 1.0) * (mu - 2.0) / 2.0
    c2 = -(mu + 1.0) * mu * (mu - 2.0) / 2.0
    c3 = (mu + 1.0) * mu * (mu - 1.0) / 6.0

    return c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3


def interpolate_cubic(samples: np.ndarray, position: float) -> complex:
    """Read the waveform at a fractional sample position.

    Timing recovery needs the signal at instants that fall *between*
    the samples we actually have.  Four-point cubic interpolation
    reconstructs those, accurately enough at eight samples per symbol
    that the error disappears under the noise.

    Indices wrap around, matching the periodic convolution the pulse
    shaper uses, so the first and last symbols of a frame are no worse
    off than the ones in the middle.
    """
    n = samples.size
    base = int(np.floor(position))
    mu = position - base

    x0 = samples[(base - 1) % n]
    x1 = samples[base % n]
    x2 = samples[(base + 1) % n]
    x3 = samples[(base + 2) % n]

    # Lagrange basis polynomials for nodes at -1, 0, 1, 2.
    c0 = -mu * (mu - 1.0) * (mu - 2.0) / 6.0
    c1 = (mu + 1.0) * (mu - 1.0) * (mu - 2.0) / 2.0
    c2 = -(mu + 1.0) * mu * (mu - 2.0) / 2.0
    c3 = (mu + 1.0) * mu * (mu - 1.0) / 6.0

    return complex(c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3)


# --------------------------------------------------------------------------
# Stage 1: matched filter
# --------------------------------------------------------------------------
class MatchedFilterStage(ReceiverStage):
    """Filter with a copy of the transmit pulse.

    The first thing any receiver does, and the only stage here that is
    optimal in a provable sense: among all linear filters, this one
    maximises the signal-to-noise ratio at the decision instants.  It
    works by collecting the energy the transmitter spread across each
    symbol while rejecting the noise outside the signal's band -- which
    is where the 9 dB of processing gain in this project comes from.
    """

    def __init__(self, pulse: PulseShape) -> None:
        if not isinstance(pulse, PulseShape):
            raise ValidationError(
                "pulse", type(pulse).__name__, "expected a PulseShape instance"
            )
        super().__init__("MatchedFilter")
        self._pulse = pulse

    @property
    def pulse(self) -> PulseShape:
        """The filter being matched against."""
        return self._pulse

    def _process(self, samples, sample_rate, context):
        return self._pulse.matched_filter(samples), sample_rate

    def parameters(self) -> dict[str, Any]:
        return {"pulse": self._pulse.name}


# --------------------------------------------------------------------------
# Stage 2: coarse frequency correction
# --------------------------------------------------------------------------
class CoarseFrequencyCorrection(ReceiverStage):
    """Estimate the Doppler shift without knowing a single transmitted bit.

    The trick is the constellation's rotational symmetry.  Raise a QPSK
    signal to the fourth power and every one of its four points lands in
    the same place -- the data cancels itself out, and what is left is a
    single tone at four times the frequency error.  Find that tone,
    divide by four, and the offset is known.

    This is why :attr:`Modulation.rotational_symmetry` exists.  The stage
    never asks what kind of modulation it is looking at; it asks the
    modulation object for its symmetry and uses that.  Adding a new
    scheme requires no change here at all.

    THE FAILURE MODE, AND WHY THERE IS AN ACQUISITION RANGE
    -------------------------------------------------------
    Raising to the Nth power raises the *noise* to the Nth power as well.
    Below roughly -5 dB on our links the noise buries the tone, the peak
    of the spectrum is just the largest noise spike, and the estimate
    becomes a random number.

    That failure is worse than useless, because a wrong estimate
    *introduces* a rotation into a constellation that was standing still.
    Measured on QPSK frames with no frequency offset at all, this stage
    returned estimates of several kilohertz -- and the receiver then
    faithfully spun the signal into nonsense.

    The guard is the one real receivers use: a specified acquisition
    range.  A link is engineered knowing roughly how far the carrier can
    drift -- from the orbit and from the oscillator specification -- and
    an estimate outside that range is known to be wrong before it is
    even examined.  Rejecting it and leaving the residue to
    :class:`DecisionDirectedPLL` is strictly better than acting on it.

    Widening the range to cover a larger Doppler shift costs robustness
    at low signal-to-noise ratio, because more noise-driven estimates
    fall inside it.  That trade-off is real and is the reason this is a
    parameter rather than a constant.

    A second, unrelated limit: the estimate is ambiguous beyond
    ``sample_rate / (2 * symmetry)``, since the Nth power folds the
    spectrum.  At eight samples per symbol that is tens of kilohertz.

    DRIFTING STRENGTH -- A BUG THE REAL RECORDINGS FOUND
    ----------------------------------------------------
    The simulator's signals keep the same strength for a whole frame.
    RML24's do not: within one recording the strength drifts by up to
    seven times.  Raised to the eighth power for 8PSK, a seven-fold
    difference becomes almost six million-fold, so the tone was set by
    the loudest dozen symbols and the rest contributed nothing.  The
    estimate was then tens of kilohertz out on recordings where the
    true offset was under one, and 8PSK on real data failed entirely --
    a bit error rate of 0.48 at 12 dB.

    Levelling alone did not fix it, and the reason was more
    interesting.  Raising an oversampled signal to the Nth power leaves
    a tone not only at ``N x offset`` but at ``symbol rate + N x
    offset``, ``2 x symbol rate + N x offset`` and so on -- the
    symbol clock shows through.  At 10 samples per symbol the line next
    to the symbol rate was the *strongest* in the spectrum: estimates of
    "10 kHz" and "13 kHz" were the 100 kHz clock line divided by eight.
    Only the line nearest zero carries the offset alone, and it must lie
    within half a symbol rate of zero, so the peak is now searched for
    there and nowhere else.  The simulator never showed this because at
    its 8 samples per symbol the zero line happened to win.

    The levelling stays, as the second half of the fix: divide by the
    local strength, averaged over ``envelope_symbols`` symbols.  That
    removes slow drift but keeps the *relative* sizes of neighbouring
    symbols, which QAM relies on (its corner points are what make the
    tone).  On the simulator's steady signals the local strength is
    constant, so this changes nothing there.

    WHERE IN THE CHAIN IT GOES -- A BUG PHASE 6 FOUND
    --------------------------------------------------
    Originally this stage ran *after* the matched filter.  The matched
    filter is circular: it treats the frame as if its end joined back
    onto its start.  With no frequency offset that join is seamless.
    With one, the end of a 2048-sample frame has rotated by about ten
    radians relative to the start, so the filter blended badly rotated
    samples from the end into the first symbol -- and the first symbol
    of every frame with Doppler came out wrong.

    No unit test noticed, because the damage is one symbol in 256.  The
    Phase 6 benchmark noticed, as an error floor: 16QAM on a Doppler
    channel stuck at a bit error rate of 1.3e-3 however strong the
    signal, and every one of those errors was symbol 0.

    The fix is the order real receivers use.  Given ``prefilter``, the
    stage *estimates* on a matched-filtered copy -- keeping the noise
    rejection the estimate needs -- but *removes* the offset from the
    raw signal, and the chain puts it before the matched filter.  The
    filter then never sees a rotation to blend.
    """

    #: Highest power the magnitude is weighted by (see _estimate_offset).
    MAX_MAGNITUDE_POWER = 4

    def __init__(
        self,
        zero_padding: int = 4,
        max_offset_fraction: float = 0.05,
        prefilter: PulseShape | None = None,
        envelope_symbols: int | None = 32,
        candidate: int = 0,
    ) -> None:
        if int(candidate) < 0:
            raise ValidationError("candidate", candidate, "cannot be negative")
        self._candidate = int(candidate)
        if envelope_symbols is not None and int(envelope_symbols) < 1:
            raise ValidationError(
                "envelope_symbols", envelope_symbols, "must be at least 1, or None"
            )
        self._envelope_symbols = None if envelope_symbols is None else int(envelope_symbols)
        if int(zero_padding) < 1:
            raise ValidationError(
                "zero_padding", zero_padding, "must be at least 1"
            )
        if not 0 < max_offset_fraction <= 0.5:
            raise ValidationError(
                "max_offset_fraction",
                max_offset_fraction,
                "must be between 0 and 0.5 of the symbol rate",
            )
        if prefilter is not None and not isinstance(prefilter, PulseShape):
            raise ValidationError(
                "prefilter", type(prefilter).__name__, "expected a PulseShape"
            )
        super().__init__("CoarseFreq")
        self._zero_padding = int(zero_padding)
        self._max_offset_fraction = float(max_offset_fraction)
        self._prefilter = prefilter

    @property
    def prefilter(self) -> PulseShape | None:
        """Filter applied to a copy for estimation only, or ``None``."""
        return self._prefilter

    def _process(self, samples, sample_rate, context):
        modulation = context.modulation

        if not modulation.blind_frequency_recoverable:
            # The constellation does not collapse under its own symmetry,
            # so there is no tone to find and any number this stage
            # produced would be a noise peak. 16APSK is the case here.
            context.record("coarse_freq_offset_hz", None)
            context.record("coarse_freq_accepted", False)
            context.record("coarse_freq_skipped", "constellation unsuitable")
            return np.array(samples, dtype=np.complex128), sample_rate

        symmetry = modulation.rotational_symmetry
        # Estimate on a filtered copy when asked to; the correction below
        # is still applied to ``samples`` exactly as they arrived.
        observed = (
            self._prefilter.matched_filter(samples)
            if self._prefilter is not None
            else samples
        )
        if self._envelope_symbols is not None:
            observed = self._levelled(
                observed, self._envelope_symbols * max(context.samples_per_symbol, 1)
            )
        symbol_rate = sample_rate / max(context.samples_per_symbol, 1)
        estimate = self._estimate_offset(
            observed, sample_rate, symmetry,
            search_hz=symbol_rate / (2.0 * symmetry), rank=self._candidate,
        )

        acquisition_range = self._max_offset_fraction * symbol_rate

        if abs(estimate) > acquisition_range:
            # Outside the range this receiver is specified for, so the
            # estimate cannot be right. Leave the signal alone and let
            # the carrier loop deal with whatever is actually there.
            context.record("coarse_freq_offset_hz", round(estimate, 3))
            context.record("coarse_freq_accepted", False)
            return np.array(samples, dtype=np.complex128), sample_rate

        context.record("coarse_freq_offset_hz", round(estimate, 3))
        context.record("coarse_freq_accepted", True)

        if estimate == 0.0:
            return np.array(samples, dtype=np.complex128), sample_rate

        n = np.arange(samples.size)
        corrected = samples * np.exp(-2j * np.pi * estimate * n / sample_rate)
        return corrected, sample_rate

    @staticmethod
    def _levelled(samples: np.ndarray, window: int) -> np.ndarray:
        """Divide by the local RMS strength over ``window`` samples."""
        window = max(1, min(int(window), samples.size))
        kernel = np.ones(window)
        power = np.convolve(np.abs(samples) ** 2, kernel, mode="same")
        count = np.convolve(np.ones(samples.size), kernel, mode="same")
        local = np.sqrt(power / count)
        floor = 1e-12 + 1e-6 * float(np.max(local))
        return samples / np.maximum(local, floor)

    def _estimate_offset(
        self,
        samples: np.ndarray,
        sample_rate: float,
        symmetry: int,
        search_hz: float | None = None,
        rank: int = 0,
    ) -> float:
        """Find the tone left behind after the modulation is cancelled.

        ``search_hz`` limits the search to offsets within that distance
        of zero, which keeps the symbol-clock lines out (see the class
        notes).
        """
        # Normalise first: raising numbers much larger than 1 to the
        # eighth power overflows nothing here, but keeping the scale
        # near unity keeps the spectrum well conditioned.
        power = float(np.sqrt(np.mean(np.abs(samples) ** 2)))
        if power == 0.0:
            return 0.0
        # Viterbi & Viterbi: multiply the *angle* by the symmetry, but
        # weight by the magnitude raised to at most the fourth power.
        # Raising the magnitude to the eighth as well, as plain x**8
        # does for 8PSK, hands the estimate to the pulse's overshoots
        # between symbols, whose phase is neither symbol's.  Measured on
        # 30 simulated 8PSK frames with 800 Hz of Doppler: 12 to 22
        # wrong estimates at 6-14 dB with x**8, none with this.  For
        # BPSK, QPSK and QAM nothing changes, because their symmetry is
        # already 4 or less.
        magnitude = np.abs(samples) / power
        unit = samples / np.maximum(np.abs(samples), 1e-30)
        stripped = magnitude ** min(symmetry, self.MAX_MAGNITUDE_POWER) * unit**symmetry

        n_fft = 1
        target = samples.size * self._zero_padding
        while n_fft < target:
            n_fft *= 2

        spectrum = np.abs(np.fft.fft(stripped, n_fft))
        if search_hz is not None:
            offsets = np.fft.fftfreq(n_fft, d=1.0 / sample_rate) / symmetry
            allowed = np.flatnonzero(np.abs(offsets) <= search_hz)
        else:
            allowed = np.arange(n_fft)
        if allowed.size == 0:
            allowed = np.arange(n_fft)
        # Distinct peaks (local maxima), strongest first; ``rank`` picks
        # one.  Rank 0 is the ordinary estimate; the receiver tries rank 1
        # as a second hypothesis (see ClassicalReceiver._best_attempt).
        is_peak = (spectrum[allowed] >= spectrum[(allowed - 1) % n_fft]) & (
            spectrum[allowed] >= spectrum[(allowed + 1) % n_fft]
        )
        peaks = allowed[is_peak] if is_peak.any() else allowed
        peaks = peaks[np.argsort(spectrum[peaks])[::-1]]
        peak = int(peaks[min(int(rank), peaks.size - 1)])

        # Parabolic interpolation around the peak, which recovers the
        # part of the frequency that falls between two bins. Without it
        # the estimate is quantised to a bin width and the residual is
        # large enough to rotate the constellation noticeably.
        left = spectrum[(peak - 1) % n_fft]
        centre = spectrum[peak]
        right = spectrum[(peak + 1) % n_fft]
        denominator = left - 2.0 * centre + right
        delta = 0.0
        if denominator != 0.0:
            delta = 0.5 * (left - right) / denominator
            delta = float(np.clip(delta, -0.5, 0.5))

        refined = peak + delta
        if refined > n_fft / 2:
            refined -= n_fft  # negative frequencies live in the upper half

        return float(refined / n_fft * sample_rate / symmetry)

    def parameters(self) -> dict[str, Any]:
        return {
            "zero_padding": self._zero_padding,
            "envelope_symbols": self._envelope_symbols,
            "candidate": self._candidate,
        }


# --------------------------------------------------------------------------
# Stage 3: timing recovery
# --------------------------------------------------------------------------
class GardnerTimingRecovery(ReceiverStage):
    """Find the right instant to sample, then sample there.

    The receiver has eight samples for every symbol and needs the one
    where the symbol is cleanest.  Nothing in the signal is labelled, and
    the transmitter's clock drifts against the receiver's, so the
    instants have to be tracked continuously.

    Gardner's method asks a neat question: look at the point *halfway*
    between two decisions.  If the timing is right, that midpoint sits
    symmetrically between them and contributes nothing.  If the timing is
    early or late, the midpoint leans towards one side, and the direction
    of that lean says which way to move.

    Its appeal is that it needs no knowledge of the carrier phase, so it
    can run before the phase is known -- which is why it sits ahead of
    the phase-locked loop in the chain.

    This stage also **decimates**: eight samples per symbol go in, one
    comes out, and the signal's sample rate becomes the symbol rate.

    CHANGING GEAR
    -------------
    A single loop bandwidth cannot do both jobs well.  Measured on these
    frames: a narrow loop (0.003) settles to within a hundredth of a
    sample of the truth, but needs well over a thousand symbols to get
    there -- a 512-symbol frame with 1.5 samples of offset only reached
    0.76.  A wide loop (0.03) arrives inside a hundred symbols but then
    jitters around the answer, and that jitter is sampling error on every
    symbol.

    So the loop changes gear, as real receivers do: wide for the first
    :attr:`acquisition_symbols` to find the timing quickly, then narrow
    to hold it steadily.  Acquisition happens during the preamble, where
    errors cost nothing because those symbols are already known.
    """

    #: Proportional and integral gains while acquiring, then while
    #: tracking. These are *not* produced by :func:`pi_loop_gains`, whose
    #: formula assumes the controller output accumulates into the
    #: quantity being tracked -- true of a carrier loop following a phase
    #: ramp, false of this one, which estimates a standing offset
    #: directly. Using those gains here gives a loop whose time constant
    #: is roughly 500 symbols: measured, it reached 0.19 of a true
    #: 1.5-sample offset over a whole frame. These were chosen for the
    #: structure that is actually here.
    ACQUISITION_GAINS = (0.30, 0.060)
    TRACKING_GAINS = (0.10, 0.008)

    #: Bisection steps used by the block estimator. Twenty halvings of a
    #: one-symbol range reach a resolution of about a millionth of a
    #: sample, far finer than the noise warrants, and cost almost nothing.
    BISECTION_STEPS = 20

    #: Grid points scanned across one symbol period to bracket the rising
    #: zero crossing before bisection refines it.
    GRID_POINTS = 16

    def __init__(
        self,
        acquisition_symbols: int = 96,
        tracking_gains: tuple[float, float] | None = None,
        acquisition_gains: tuple[float, float] | None = None,
        mode: str = "block",
    ) -> None:
        super().__init__("GardnerTiming")
        if mode not in ("block", "tracking"):
            raise ValidationError(
                "mode", mode, "must be 'block' or 'tracking'"
            )
        self._mode = mode
        if int(acquisition_symbols) < 0:
            raise ValidationError(
                "acquisition_symbols", acquisition_symbols, "cannot be negative"
            )
        self._acquisition_symbols = int(acquisition_symbols)
        self._kp, self._ki = tracking_gains or self.TRACKING_GAINS
        self._acq_kp, self._acq_ki = acquisition_gains or self.ACQUISITION_GAINS

    @property
    def acquisition_symbols(self) -> int:
        """How many symbols the loop spends in its fast, noisy gear."""
        return self._acquisition_symbols

    def mean_timing_error(
        self, samples: np.ndarray, sps: int, n_symbols: int, tau: float
    ) -> float:
        """Average Gardner error across the whole frame at one candidate offset.

        This is the S-curve, evaluated at a point.  It passes through
        zero at the correct timing and rises monotonically either side
        of it over roughly half a symbol, which is what makes the zero
        findable by bisection.
        """
        positions = np.arange(n_symbols, dtype=float) * sps + tau
        current = interpolate_cubic_many(samples, positions)
        midpoint = interpolate_cubic_many(samples, positions - sps / 2.0)

        previous = current[:-1]
        latest = current[1:]
        errors = np.real(np.conj(midpoint[1:]) * (latest - previous))
        scale = np.abs(latest) ** 2 + np.abs(previous) ** 2 + 1e-12
        return float(np.mean(errors / scale))

    def _estimate_timing_block(
        self, samples: np.ndarray, sps: int, n_symbols: int
    ) -> float:
        """Find the timing by locating the zero of the averaged S-curve.

        A tracking loop sees one noisy error sample per symbol and has to
        filter them over time, which at low signal-to-noise ratio means
        it wanders.  Measured on BPSK frames, the loop was the *entire*
        source of this receiver's implementation loss -- removing it left
        a loss of about -0.1 dB, and keeping it cost up to 3 dB, varying
        wildly from frame to frame as it settled on different answers.

        Working on a whole frame at once there is no need to filter over
        time: the error can be averaged over every symbol at once, and
        the noise falls away as the square root of the frame length.
        Then the answer is wherever that average crosses zero, which
        bisection finds in about twenty evaluations.

        The honest trade-off: this assumes the timing does not move
        within a frame.  Over 512 symbols with ordinary crystals that
        holds comfortably.  A receiver running continuously against a
        drifting clock needs the loop, which is why
        ``mode="tracking"`` keeps it available.
        """
        half_symbol = sps / 2.0

        # The S-curve repeats every symbol period, so it has two zeros in
        # each one: a rising crossing at the correct timing, and a
        # falling crossing half a symbol away where sampling is at its
        # worst. Both ends of the full range land on that second zero,
        # which is why bracketing the whole range finds nothing -- the
        # endpoints have the same sign by construction.
        #
        # So: scan a grid first to find the *rising* crossing, then
        # bisect inside that bracket.
        grid = np.linspace(-half_symbol, half_symbol, self.GRID_POINTS + 1)[:-1]
        errors = np.array(
            [self.mean_timing_error(samples, sps, n_symbols, t) for t in grid]
        )

        # Wrap around so a crossing between the last and first point is
        # not missed.
        taus = np.append(grid, grid[0] + 2.0 * half_symbol)
        values = np.append(errors, errors[0])

        bracket = None
        for i in range(len(taus) - 1):
            if values[i] <= 0.0 < values[i + 1]:  # rising through zero
                bracket = (taus[i], values[i], taus[i + 1], values[i + 1])
                break

        if bracket is None:
            # Too noisy to show an S-curve at all. Sampling where we
            # already are is the safe answer; it is what a receiver with
            # no timing information can honestly do.
            return 0.0

        low, error_low, high, _ = bracket
        for _ in range(self.BISECTION_STEPS):
            middle = 0.5 * (low + high)
            error_mid = self.mean_timing_error(samples, sps, n_symbols, middle)
            if error_mid > 0.0:
                high = middle
            else:
                low, error_low = middle, error_mid

        centre = 0.5 * (low + high)
        # Fold back into the half-symbol window around zero.
        period = 2.0 * half_symbol
        return float((centre + half_symbol) % period - half_symbol)

    def _process(self, samples, sample_rate, context):
        sps = context.samples_per_symbol
        if sps < 2:
            raise ValidationError(
                "samples_per_symbol",
                sps,
                "Gardner timing recovery needs at least 2 samples per symbol",
            )

        n_symbols = samples.size // sps
        if n_symbols < 2:
            raise ValidationError(
                "samples", samples.size, "too few samples to recover timing"
            )

        if self._mode == "block":
            tau_final = self._estimate_timing_block(samples, sps, n_symbols)
            positions = np.arange(n_symbols, dtype=float) * sps + tau_final
            output = interpolate_cubic_many(samples, positions)

            context.samples_per_symbol = 1
            context.record("timing_offset_samples", round(tau_final, 4))
            context.record("timing_mode", "block")
            return output, sample_rate / sps

        half = sps / 2.0

        tau = 0.0          # running estimate of the timing error, in samples
        accumulated = 0.0  # integral term of the loop
        previous = 0j
        history = np.zeros(n_symbols, dtype=float)

        for k in range(n_symbols):
            position = k * sps + tau
            current = interpolate_cubic(samples, position)
            midpoint = interpolate_cubic(samples, position - half)
            history[k] = tau

            if k > 0:
                # Gardner's timing error detector. The midpoint is
                # weighted by the change between consecutive decisions,
                # so a symmetric midpoint contributes nothing.
                error = float(
                    np.real(np.conj(midpoint) * (current - previous))
                )
                # Normalise by signal power so the loop behaves the same
                # whether the signal arrived strong or weak.
                scale = abs(current) ** 2 + abs(previous) ** 2 + 1e-12
                error /= scale

                # Put the error into units of samples.
                #
                # The raw detector output is linear in the timing error
                # but with a slope set by the pulse shape, not by
                # anything the loop knows: measured on this project's
                # root-raised-cosine at eight samples per symbol, one
                # sample of error produces about 0.08 units of detector
                # output. Feeding that straight into the loop makes the
                # effective gain a twelfth of the nominal one, and the
                # loop creeps towards the answer without ever arriving --
                # measured, it reached 0.02 of a true 1.5-sample offset.
                #
                # Multiplying by the oversampling factor brings the slope
                # to within about a third of unity, which is close enough
                # that the configured loop bandwidth means roughly what
                # it says.
                error *= sps

                # Fast gear while acquiring, slow gear once locked.
                if k < self._acquisition_symbols:
                    kp, ki = self._acq_kp, self._acq_ki
                else:
                    kp, ki = self._kp, self._ki

                # Proportional-integral control, where the loop's output
                # *is* the timing estimate rather than a nudge to it.
                #
                # The difference matters. Accumulating the controller
                # output into tau instead would integrate twice, and a
                # double integrator only settles when its inner integral
                # returns to zero -- so it has to overshoot the answer
                # and come back. With wide gains it does not come back at
                # all: measured, tau ran to the clip limit of a whole
                # symbol and stayed there. Here the integral term simply
                # holds the steady offset while the error sits at zero.
                accumulated += ki * error
                tau = float(np.clip(-(kp * error + accumulated), -sps, sps))

            previous = current

        # Second pass: re-sample the whole frame at the timing the loop
        # settled on.
        #
        # A streaming receiver cannot do this -- it must emit each symbol
        # as it arrives, so the symbols during acquisition are sampled at
        # the wrong instants and are simply lost. Measured here, that
        # transient alone put the bit error rate at 1.6e-2 on a 28 dB
        # channel where it should have been zero.
        #
        # Working on a whole frame at once, there is no reason to accept
        # that. Estimate the timing first, then sample once with the
        # answer, and every symbol in the frame benefits -- including the
        # ones that arrived before the loop had locked.
        settled = history[max(len(history) // 2, 1):]
        tau_final = float(np.median(settled)) if settled.size else tau

        positions = np.arange(n_symbols) * sps + tau_final
        output = np.array(
            [interpolate_cubic(samples, p) for p in positions],
            dtype=np.complex128,
        )

        context.samples_per_symbol = 1
        context.record("timing_offset_samples", round(tau_final, 4))
        context.record("timing_jitter_samples", round(float(np.std(settled)), 4))

        return output, sample_rate / sps

    def parameters(self) -> dict[str, Any]:
        if self._mode == "block":
            return {"mode": "block", "bisection_steps": self.BISECTION_STEPS}
        return {
            "mode": "tracking",
            "acquisition_symbols": self._acquisition_symbols,
            "tracking_gains": (self._kp, self._ki),
        }


# --------------------------------------------------------------------------
# Stage 4: gain control
# --------------------------------------------------------------------------
class AutomaticGainControl(ReceiverStage):
    """Rescale the signal so it matches the constellation's size.

    Fading does not only rotate a constellation, it shrinks or stretches
    it.  Since the demapper decides by measuring distances to fixed
    points, a constellation arriving at half size would be read as
    nonsense -- every symbol would look closest to whatever sits nearest
    the origin.

    Our constellations are all normalised to unit average energy, so the
    fix is to scale the received block to unit average power.  Real
    receivers run this as a continuously adapting loop; a per-frame
    estimate is enough at the frame lengths used here.
    """

    def __init__(self, target_power: float = 1.0) -> None:
        if target_power <= 0:
            raise ValidationError(
                "target_power", target_power, "must be positive"
            )
        super().__init__("AGC")
        self._target_power = float(target_power)

    def _process(self, samples, sample_rate, context):
        power = float(np.mean(np.abs(samples) ** 2))
        if power <= 0.0:
            return np.array(samples, dtype=np.complex128), sample_rate

        gain = np.sqrt(self._target_power / power)
        context.record("agc_gain", round(float(gain), 4))
        return samples * gain, sample_rate

    def parameters(self) -> dict[str, Any]:
        return {"target_power": self._target_power}


# --------------------------------------------------------------------------
# Stage 5: fine carrier recovery
# --------------------------------------------------------------------------
class DecisionDirectedPLL(ReceiverStage):
    """Track whatever rotation the coarse stage left behind.

    By this point the constellation is roughly the right size and roughly
    still, but a residual frequency error of even a few hertz winds up
    over a frame into a visible rotation.  This loop closes it.

    The method is called *decision-directed*: assume the nearest
    constellation point is what was sent, measure the angle between the
    received symbol and that point, and treat the angle as the error.  It
    is circular reasoning that works, because the decisions are right
    most of the time and the loop averages over many symbols.

    **The phase ambiguity.** No blind loop can tell which way up a
    constellation is.  Rotate QPSK by 90 degrees and it looks untouched,
    so the loop is equally happy locked to any of four rotations -- three
    of which decode every symbol wrongly.  There is no clever fix: the
    information simply is not in the signal.  Real links solve it by
    sending a short known preamble, and so does this stage.  Given
    ``context.pilot_symbols`` it starts already aligned; without them it
    runs blind and says so in the diagnostics.
    """

    def __init__(
        self, loop_bandwidth: float = 0.005, damping: float = 0.707
    ) -> None:
        super().__init__("CarrierPLL")
        self._loop_bandwidth = float(loop_bandwidth)
        self._kp, self._ki = pi_loop_gains(loop_bandwidth, damping)

    def _process(self, samples, sample_rate, context):
        if context.samples_per_symbol != 1:
            raise ValidationError(
                "samples_per_symbol",
                context.samples_per_symbol,
                "the carrier loop runs at the symbol rate; "
                "put timing recovery before it",
            )

        constellation = context.modulation.constellation
        pilot = context.pilot_symbols

        phase = 0.0
        if pilot is not None and pilot.size:
            # The preamble says which way up the constellation is. One
            # line, and the ambiguity is gone.
            usable = min(pilot.size, samples.size)
            phase = float(
                np.angle(np.sum(samples[:usable] * np.conj(pilot[:usable])))
            )
            context.record("initial_phase_rad", round(phase, 4))
        else:
            context.record("initial_phase_rad", "blind")

        frequency = 0.0
        accumulated = 0.0
        output = np.zeros(samples.size, dtype=np.complex128)

        for k in range(samples.size):
            rotated = samples[k] * np.exp(-1j * phase)
            output[k] = rotated

            # Which point do we believe was sent?
            nearest = constellation[
                int(np.argmin(np.abs(rotated - constellation)))
            ]
            if nearest == 0:
                continue

            # The angle between what arrived and what we believe was
            # sent is the phase error.
            error = float(np.angle(rotated * np.conj(nearest)))

            accumulated += self._ki * error
            frequency = accumulated
            phase += self._kp * error + frequency

        context.record("residual_freq_rad_per_symbol", round(frequency, 6))
        context.record(
            "residual_freq_hz", round(frequency * sample_rate / (2 * np.pi), 3)
        )
        return output, sample_rate

    def parameters(self) -> dict[str, Any]:
        return {"loop_bandwidth": self._loop_bandwidth}


# --------------------------------------------------------------------------
# Stage 6: equalisation
# --------------------------------------------------------------------------
class LMSEqualizer(ReceiverStage):
    """Undo echoes that smear each symbol into the next.

    When a signal arrives by several paths at once, a delayed copy of
    one symbol lands on top of the next one.  No amount of rotating or
    rescaling fixes that, because the damage mixes symbols together.  An
    equaliser is a short filter that learns to unmix them.

    It learns by least-mean-squares: predict a symbol, compare the
    prediction with what the symbol should have been, and nudge the
    filter taps a little in the direction that would have reduced the
    error.  Repeat for every symbol.

    Where "what it should have been" comes from changes partway through.
    Over the known preamble the answer is certain, so the taps can be
    trained properly.  After that the equaliser switches to its own
    decisions -- the same circular reasoning as the carrier loop, working
    for the same reason.

    The step size is the familiar trade-off, with a hard edge to it.
    Too small and the taps never reach the answer within a frame; too
    large and the loop is unstable and the taps run away to infinity.
    Measured on symbol-spaced echoes at 0 dB: a step of 0.01 took the
    bit error rate from 2.6e-1 down to 8.8e-2, while 0.05 made things
    worse and 0.15 diverged outright.  :attr:`MAX_TAP_ENERGY` catches
    that last case.

    WHEN THIS STAGE IS WORTH HAVING
    -------------------------------
    Only when the echoes are a meaningful fraction of a symbol apart.
    On the RML24 channel they arrive one *sample* apart -- an eighth of
    a symbol -- and the matched filter has already absorbed them, so
    there is no inter-symbol interference left to remove and the
    equaliser just adapts against noise, costing a few percent in bit
    error rate.  That is why
    :class:`~satrecovery.receivers.classical.ClassicalReceiver` leaves
    it switched off by default.
    """

    #: Total tap energy past which the loop is declared unstable. A
    #: healthy equaliser sits near 1; a diverging one passes this within
    #: a few dozen symbols.
    MAX_TAP_ENERGY = 1e4

    def __init__(self, n_taps: int = 7, step_size: float = 0.01) -> None:
        if int(n_taps) < 1 or int(n_taps) % 2 == 0:
            raise ValidationError(
                "n_taps", n_taps, "must be a positive odd number"
            )
        if not 0 < step_size < 1:
            raise ValidationError(
                "step_size", step_size, "must be between 0 and 1"
            )
        super().__init__("Equalizer")
        self._n_taps = int(n_taps)
        self._step_size = float(step_size)

    def _process(self, samples, sample_rate, context):
        if context.samples_per_symbol != 1:
            raise ValidationError(
                "samples_per_symbol",
                context.samples_per_symbol,
                "the equaliser runs at the symbol rate",
            )

        n = samples.size
        taps = np.zeros(self._n_taps, dtype=np.complex128)
        taps[self._n_taps // 2] = 1.0  # start as a pass-through

        constellation = context.modulation.constellation
        pilot = context.pilot_symbols
        n_pilot = 0 if pilot is None else int(pilot.size)

        half = self._n_taps // 2
        offsets = np.arange(-half, half + 1)
        output = np.zeros(n, dtype=np.complex128)

        diverged = False
        for k in range(n):
            window = samples[(k + offsets) % n]
            estimate = complex(np.dot(taps, window))
            output[k] = estimate

            if k < n_pilot:
                wanted = pilot[k]          # training: the answer is known
            else:
                wanted = constellation[    # tracking: trust our own decision
                    int(np.argmin(np.abs(estimate - constellation)))
                ]

            error = wanted - estimate
            taps += self._step_size * error * np.conj(window)

            # Least-mean-squares is only stable for step sizes below a
            # limit that depends on the signal power, and past it the
            # taps grow without bound: each oversized output produces a
            # larger error, which grows the taps further. Left alone it
            # reaches infinity within a frame and every stage downstream
            # inherits the wreckage.
            if k % 64 == 0 and float(np.sum(np.abs(taps) ** 2)) > self.MAX_TAP_ENERGY:
                diverged = True
                break

        if diverged:
            # Hand back what came in. A pass-through is a poor equaliser
            # but an honest one, and far better than the alternative.
            context.record("equalizer_diverged", True)
            context.record("equalizer_residual", None)
            return np.array(samples, dtype=np.complex128), sample_rate

        residual = float(
            np.mean(np.abs(output - self._decisions(output, constellation)) ** 2)
        )
        context.record("equalizer_diverged", False)
        context.record("equalizer_residual", round(residual, 5))
        context.record(
            "equalizer_tap_spread", round(float(np.std(np.abs(taps))), 4)
        )
        return output, sample_rate

    @staticmethod
    def _decisions(symbols: np.ndarray, constellation: np.ndarray) -> np.ndarray:
        """Nearest constellation point for each symbol."""
        distances = np.abs(symbols[:, None] - constellation[None, :])
        return constellation[np.argmin(distances, axis=1)]

    def parameters(self) -> dict[str, Any]:
        return {"n_taps": self._n_taps, "step_size": self._step_size}


__all__ = [
    "AutomaticGainControl",
    "CoarseFrequencyCorrection",
    "DecisionDirectedPLL",
    "GardnerTimingRecovery",
    "LMSEqualizer",
    "MatchedFilterStage",
    "interpolate_cubic",
    "pi_loop_gains",
]
