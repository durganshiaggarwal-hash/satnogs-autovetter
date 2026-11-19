"""Concrete channel effects.

Five impairments, each a subclass of
:class:`~satrecovery.channel.base.ChannelEffect`:

===========================  ===================================================
Class                        Physical cause
===========================  ===================================================
:class:`AdditiveWhiteGaussianNoise`  Thermal noise in the receiver front end
:class:`FlatRicianFading`            One direct path plus diffuse scattering
:class:`MultipathRicianFading`       Several delayed echoes (the RML24 model)
:class:`CarrierFrequencyOffset`      Doppler shift and oscillator mismatch
:class:`CarrierPhaseOffset`          Unknown absolute carrier phase
===========================  ===================================================

Together these are the four knobs the GUI exposes, and they are chosen to
match the impairments RML24 models so that a simulated frame and a real
RML24 frame are directly comparable.

DESIGN NOTE: POLYMORPHISM IN ACTION
-----------------------------------
All five classes implement the same two-line contract -- ``_process`` and
``parameters`` -- and nothing else.  Notice that no class below contains
an ``if`` statement about *which* effect it is.  The decision of what
happens to a signal is made once, when the pipeline is assembled, and is
then encoded in the objects themselves rather than in branching code.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from ..core.exceptions import ValidationError
from .base import ChannelEffect

#: Relative tap magnitudes of the RML24 satellite channel model, in dB.
#: Taken from the dataset's published parameters so our simulated frames
#: and the real RML24 frames see the same multipath profile.
RML24_TAP_GAINS_DB: Final[tuple[float, ...]] = (0.0, -0.97, -5.23)


class AdditiveWhiteGaussianNoise(ChannelEffect):
    """Thermal noise, set to hit a target signal-to-noise ratio.

    The noise power is computed from the *measured* power of the incoming
    signal rather than assumed, so "SNR = 10 dB" means the same thing
    whatever modulation, pulse shaping or fading came before.  Without
    that, a comparison between schemes would be measuring the wrong
    thing.

    Parameters
    ----------
    snr_db:
        Target ratio of signal power to noise power, in dB.
    """

    def __init__(self, snr_db: float) -> None:
        super().__init__("AWGN")
        self._snr_db = float(snr_db)

    @property
    def snr_db(self) -> float:
        """Target signal-to-noise ratio in dB."""
        return self._snr_db

    @property
    def snr_linear(self) -> float:
        """Target signal-to-noise ratio as a power ratio."""
        return 10.0 ** (self._snr_db / 10.0)

    def _process(self, samples, sample_rate, rng):
        signal_power = float(np.mean(np.abs(samples) ** 2))
        if signal_power <= 0.0:
            return np.array(samples, dtype=np.complex128)

        noise_power = signal_power / self.snr_linear

        # A complex Gaussian with total power P splits evenly between the
        # real and imaginary parts, so each gets variance P/2.
        sigma = np.sqrt(noise_power / 2.0)
        noise = rng.normal(0.0, sigma, samples.size) + 1j * rng.normal(
            0.0, sigma, samples.size
        )
        return samples + noise

    def parameters(self) -> dict[str, Any]:
        return {"snr_db": self._snr_db}


class FlatRicianFading(ChannelEffect):
    """Flat Rician fading: one complex gain applied to the whole frame.

    Models a link with a dominant line-of-sight path plus weaker
    scattered paths, which is the normal situation for a satellite.  The
    K-factor is the power ratio between the direct and scattered parts:
    a high K means an almost clean channel, K = 0 dB means the scattered
    energy is as strong as the direct path.

    "Flat" means the whole signal bandwidth fades together, which is a
    good approximation when the delay spread is small compared with the
    symbol period.  The gain is normalised so that ``E[|h|**2] = 1`` and
    fading therefore changes the *shape* of the constellation without
    quietly changing the signal-to-noise ratio.
    """

    def __init__(self, k_factor_db: float = 10.0, name: str = "FlatRician") -> None:
        super().__init__(name)
        self._k_factor_db = float(k_factor_db)

    @property
    def k_factor_db(self) -> float:
        """Rician K-factor in dB."""
        return self._k_factor_db

    @property
    def k_factor_linear(self) -> float:
        """Rician K-factor as a power ratio."""
        return 10.0 ** (self._k_factor_db / 10.0)

    def _draw_gain(self, rng: np.random.Generator) -> complex:
        """Draw one unit-power Rician channel coefficient."""
        k = self.k_factor_linear
        direct = np.sqrt(k / (k + 1.0))
        scatter_sigma = np.sqrt(1.0 / (2.0 * (k + 1.0)))
        scattered = scatter_sigma * (rng.normal() + 1j * rng.normal())
        return complex(direct + scattered)

    def _process(self, samples, sample_rate, rng):
        return samples * self._draw_gain(rng)

    def parameters(self) -> dict[str, Any]:
        return {"k_factor_db": self._k_factor_db}


class MultipathRicianFading(FlatRicianFading):
    """Multipath Rician fading -- several delayed, independently faded echoes.

    DESIGN NOTE: EXTENDING A SIBLING, NOT COPYING IT
    ------------------------------------------------
    This class inherits from :class:`FlatRicianFading` rather than from
    ``ChannelEffect`` directly, because it needs exactly the same
    coefficient-drawing mathematics -- just once per tap instead of once
    per frame.  It reuses :meth:`FlatRicianFading._draw_gain` unchanged
    and overrides only :meth:`_process`.

    That is inheritance used correctly: there is real shared
    implementation, and the subclass genuinely *is* a kind of Rician
    fading.  Multipath fading with a single tap reduces exactly to flat
    fading, so the relationship holds mathematically too.

    The default tap profile is the one published for RML24.
    """

    def __init__(
        self,
        k_factor_db: float = 10.0,
        tap_gains_db: tuple[float, ...] = RML24_TAP_GAINS_DB,
        tap_spacing: int = 1,
    ) -> None:
        super().__init__(k_factor_db, name="MultipathRician")

        if int(tap_spacing) < 1:
            raise ValidationError(
                "tap_spacing", tap_spacing, "must be at least 1 sample"
            )
        self._tap_spacing = int(tap_spacing)

        if len(tap_gains_db) == 0:
            raise ValidationError(
                "tap_gains_db", tap_gains_db, "needs at least one tap"
            )
        self._tap_gains_db = tuple(float(gain) for gain in tap_gains_db)

    @property
    def tap_gains_db(self) -> tuple[float, ...]:
        """Relative power of each echo, in dB."""
        return self._tap_gains_db

    @property
    def n_taps(self) -> int:
        """Number of resolvable paths."""
        return len(self._tap_gains_db)

    @property
    def tap_spacing(self) -> int:
        """Delay between consecutive echoes, in samples.

        The default of 1 matches RML24, where the echoes arrive a single
        sample apart -- an eighth of a symbol at our oversampling, so
        short that the matched filter absorbs most of the damage.

        Set it to the samples-per-symbol figure and the echoes land a
        whole symbol apart instead, which is genuine inter-symbol
        interference: a delayed copy of one symbol sitting on top of the
        next. No amount of rotating or rescaling fixes that, which is
        the situation :class:`~satrecovery.receivers.stages.LMSEqualizer`
        exists for.
        """
        return self._tap_spacing

    def _process(self, samples, sample_rate, rng):
        # dB power -> linear amplitude, then scaled so the total channel
        # power is 1 and the signal-to-noise ratio stays meaningful.
        amplitudes = 10.0 ** (np.asarray(self._tap_gains_db) / 20.0)
        amplitudes = amplitudes / np.sqrt(np.sum(amplitudes**2))

        drawn = [amp * self._draw_gain(rng) for amp in amplitudes]

        # Space the echoes out, leaving silence between them.
        taps = np.zeros(
            (len(drawn) - 1) * self._tap_spacing + 1, dtype=np.complex128
        )
        taps[:: self._tap_spacing] = drawn

        # Each echo is the signal delayed by one sample and scaled, so the
        # received signal is the convolution of the signal with the taps.
        # Trimming back to the original length models the receiver
        # observing a fixed-length window.
        faded = np.convolve(samples, taps, mode="full")
        return faded[: samples.size]

    def parameters(self) -> dict[str, Any]:
        return {
            "k_factor_db": self._k_factor_db,
            "tap_gains_db": list(self._tap_gains_db),
            "tap_spacing": self._tap_spacing,
        }


class CarrierFrequencyOffset(ChannelEffect):
    """A residual carrier frequency error, from Doppler or oscillator drift.

    A satellite in low Earth orbit moves at roughly 7.5 km/s, which at
    S-band produces tens of kilohertz of Doppler shift.  Left
    uncorrected, the offset makes the whole constellation spin, and a
    nearest-neighbour demapper fails completely.  Undoing it is the main
    job of the classical receiver's carrier recovery loop in Phase 2 --
    and the main thing a learned demodulator has to discover for itself.
    """

    def __init__(self, offset_hz: float) -> None:
        super().__init__("FreqOffset")
        self._offset_hz = float(offset_hz)

    @property
    def offset_hz(self) -> float:
        """Frequency error in hertz."""
        return self._offset_hz

    def _process(self, samples, sample_rate, rng):
        if self._offset_hz == 0.0:
            return np.array(samples, dtype=np.complex128)
        n = np.arange(samples.size)
        rotation = np.exp(2j * np.pi * self._offset_hz * n / sample_rate)
        return samples * rotation

    def parameters(self) -> dict[str, Any]:
        return {"offset_hz": self._offset_hz}


class SampleTimingOffset(ChannelEffect):
    """The receiver's sampling clock is not aligned with the symbols.

    A receiver has to decide *when* to look at the incoming waveform.
    Symbols are at their cleanest at one precise instant and smeared into
    their neighbours everywhere else, so sampling even a fraction of a
    symbol early or late costs real signal quality.

    Nothing in the transmitter tells the receiver when those instants
    are, and the two ends run on separate crystals that drift apart.
    Working the timing out from the signal alone is the job of
    :class:`~satrecovery.receivers.stages.GardnerTimingRecovery`.

    Parameters
    ----------
    delay_samples:
        How far the sampling instants are out, in samples.  Fractional
        values are the interesting case, since a whole-sample error just
        shifts which symbol is which.

    Notes
    -----
    The delay is applied as a phase ramp in the frequency domain, which
    is an exact fractional-sample shift for a periodic signal -- matching
    the wrap-around convolution the pulse shaper uses.  It goes in before
    the noise stage, which is harmless: thermal noise is white, so a
    delayed version of it has identical statistics.
    """

    def __init__(self, delay_samples: float) -> None:
        super().__init__("TimingOffset")
        self._delay_samples = float(delay_samples)

    @property
    def delay_samples(self) -> float:
        """Sampling misalignment, in samples."""
        return self._delay_samples

    def _process(self, samples, sample_rate, rng):
        if self._delay_samples == 0.0:
            return np.array(samples, dtype=np.complex128)
        n = samples.size
        freqs = np.fft.fftfreq(n)
        ramp = np.exp(-2j * np.pi * freqs * self._delay_samples)
        return np.fft.ifft(np.fft.fft(samples) * ramp)

    def parameters(self) -> dict[str, Any]:
        return {"delay_samples": self._delay_samples}


class CarrierPhaseOffset(ChannelEffect):
    """A fixed, unknown carrier phase -- the constellation arrives rotated.

    Always present in practice: the receiver has no way to know the
    transmitter's absolute phase reference.  Real links solve it with a
    known preamble or with differential encoding.
    """

    def __init__(self, phase_rad: float) -> None:
        super().__init__("PhaseOffset")
        self._phase_rad = float(phase_rad)

    @property
    def phase_rad(self) -> float:
        """Phase error in radians."""
        return self._phase_rad

    def _process(self, samples, sample_rate, rng):
        if self._phase_rad == 0.0:
            return np.array(samples, dtype=np.complex128)
        return samples * np.exp(1j * self._phase_rad)

    def parameters(self) -> dict[str, Any]:
        return {"phase_rad": self._phase_rad}


__all__ = [
    "RML24_TAP_GAINS_DB",
    "AdditiveWhiteGaussianNoise",
    "CarrierFrequencyOffset",
    "CarrierPhaseOffset",
    "FlatRicianFading",
    "MultipathRicianFading",
    "SampleTimingOffset",
]
