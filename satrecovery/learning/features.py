"""Hand-computed features for the classical machine-learning baseline.

Before deep learning arrived, modulation recognition worked by computing
a few dozen carefully chosen numbers from a signal and feeding them to an
ordinary classifier.  Those numbers are still worth having, for two
reasons: they train in seconds on a laptop, and -- unlike a neural
network -- you can look at one and say why it decided what it decided.

This module provides them as interchangeable
:class:`FeatureExtractor` objects, so the Random Forest in
:mod:`~satrecovery.learning.forest` can be given a different description
of the same signal without any other code changing.

WHAT THE FEATURES ACTUALLY MEASURE
----------------------------------
:class:`CumulantFeatures`
    Higher-order statistics.  The clever part is that cumulants above
    second order are *zero* for Gaussian noise, so they describe the
    constellation's shape while largely ignoring the noise sitting on
    top of it.  Normalised properly they are also blind to the signal's
    amplitude and rotation, which is exactly what a classifier wants:
    the same answer whether the signal arrived strong or weak, upright
    or spinning.

:class:`SpectralFeatures`
    Cheaper, more intuitive descriptors -- how much the amplitude varies,
    how much the phase varies, peak-to-average power. Weaker on their
    own, useful alongside the cumulants.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.exceptions import ValidationError


class FeatureExtractor(ABC):
    """Turns a block of complex samples into a fixed row of numbers.

    Every extractor returns the same number of features in the same
    order every time, because a classifier trained on one layout cannot
    read another.  :attr:`feature_names` makes that layout inspectable,
    which is what lets the forest report *which* feature it relied on.
    """

    def __init__(self, name: str | None = None) -> None:
        self._name = str(name) if name else type(self).__name__

    @property
    def name(self) -> str:
        """Short name of this extractor."""
        return self._name

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Name of each feature, in the order :meth:`extract` returns them."""

    @property
    def n_features(self) -> int:
        """How many numbers this extractor produces."""
        return len(self.feature_names)

    @abstractmethod
    def _compute(self, samples: np.ndarray) -> np.ndarray:
        """Do the arithmetic on one normalised block of samples."""

    def extract(self, samples: np.ndarray) -> np.ndarray:
        """Features for one block of samples.

        A Template Method again: the base class normalises the input to
        unit power first, so that every extractor is automatically blind
        to how strong the signal happened to arrive. Without that step a
        classifier would learn to recognise loudness rather than shape.
        """
        samples = np.asarray(samples, dtype=np.complex128).ravel()
        if samples.size < 8:
            raise ValidationError(
                "samples", samples.size, "need at least 8 samples to compute features"
            )

        power = float(np.mean(np.abs(samples) ** 2))
        if power > 0:
            samples = samples / np.sqrt(power)

        features = np.asarray(self._compute(samples), dtype=float).ravel()
        if features.size != self.n_features:
            raise ValidationError(
                f"{self._name}.features",
                features.size,
                f"expected {self.n_features} values",
            )
        # A feature that came out as NaN would poison every tree that
        # splits on it, so clamp rather than propagate.
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def extract_many(self, blocks: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """Features for several blocks, as a (rows, features) array."""
        rows = [self.extract(block) for block in blocks]
        if not rows:
            return np.empty((0, self.n_features), dtype=float)
        return np.vstack(rows)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.n_features} features)"


class CumulantFeatures(FeatureExtractor):
    """Higher-order cumulants: the standard features for this problem.

    A cumulant is a statistic built from averages of powers of the
    signal.  The second-order ones measure power; the fourth- and
    sixth-order ones measure the *shape* of the constellation in a way
    that survives noise, because Gaussian noise contributes nothing to
    cumulants above second order.

    Each one is normalised by the right power of ``C21`` (the signal's
    own power), which makes it independent of how loud the signal was.
    Taking magnitudes makes it independent of rotation too. What remains
    describes the constellation and very little else -- which is why
    these numbers separate QPSK from 16QAM even at poor signal quality.
    """

    #: Normalised cumulant magnitudes, plus two ratios that sharpen the
    #: distinction between constellations with similar fourth-order
    #: statistics (16QAM and 16APSK, for instance).
    FEATURES = (
        "abs_C20",
        "abs_C21",
        "abs_C40",
        "abs_C41",
        "abs_C42",
        "abs_C60",
        "abs_C61",
        "abs_C63",
        "C42_over_C21sq",
        "C63_over_C42",
    )

    def __init__(self) -> None:
        super().__init__("Cumulants")

    @property
    def feature_names(self) -> list[str]:
        return list(self.FEATURES)

    def _compute(self, x: np.ndarray) -> np.ndarray:
        conj = np.conj(x)
        magnitude_squared = np.abs(x) ** 2

        # --- second order -------------------------------------------
        m20 = np.mean(x**2)
        m21 = np.mean(magnitude_squared)

        # --- fourth order moments ------------------------------------
        m40 = np.mean(x**4)
        m41 = np.mean(x**3 * conj)
        m42 = np.mean(magnitude_squared**2)

        # --- fourth order cumulants ----------------------------------
        c20 = m20
        c21 = m21
        c40 = m40 - 3.0 * m20**2
        c41 = m41 - 3.0 * m20 * m21
        c42 = m42 - np.abs(m20) ** 2 - 2.0 * m21**2

        # --- sixth order (standard expressions) -----------------------
        m60 = np.mean(x**6)
        m61 = np.mean(x**5 * conj)
        m63 = np.mean(x**3 * conj**3)

        c60 = m60 - 15.0 * m20 * m40 + 30.0 * m20**3
        c61 = m61 - 5.0 * m21 * m40 - 10.0 * m20 * m41 + 30.0 * m20**2 * m21
        c63 = (
            m63
            - 9.0 * c21 * c42
            - 6.0 * c21**3
        )

        # Normalise each by the power of C21 that cancels its own scale,
        # so a weak signal and a strong one give the same answer.
        power = abs(c21) if abs(c21) > 1e-12 else 1e-12
        normalised = np.array(
            [
                abs(c20) / power,
                abs(c21) / power,
                abs(c40) / power**2,
                abs(c41) / power**2,
                abs(c42) / power**2,
                abs(c60) / power**3,
                abs(c61) / power**3,
                abs(c63) / power**3,
                abs(c42) / power**2,
                abs(c63) / max(abs(c42), 1e-12) / power,
            ],
            dtype=float,
        )
        return normalised


class SpectralFeatures(FeatureExtractor):
    """Simpler descriptors: how the amplitude, phase and spectrum behave.

    Weaker than cumulants on their own, but cheap and easy to interpret.
    Peak-to-average power alone nearly separates the constant-amplitude
    schemes (PSK) from the ones that vary (QAM, APSK), which is a useful
    sanity check on whatever the forest decides.
    """

    FEATURES = (
        "amplitude_std",
        "amplitude_kurtosis",
        "phase_std",
        "phase_kurtosis",
        "papr_db",
        "spectral_flatness",
        "peak_to_mean_spectrum",
        "zero_crossing_rate",
    )

    def __init__(self) -> None:
        super().__init__("Spectral")

    @property
    def feature_names(self) -> list[str]:
        return list(self.FEATURES)

    @staticmethod
    def _kurtosis(values: np.ndarray) -> float:
        centred = values - np.mean(values)
        variance = float(np.mean(centred**2))
        if variance <= 1e-18:
            return 0.0
        return float(np.mean(centred**4) / variance**2)

    def _compute(self, x: np.ndarray) -> np.ndarray:
        amplitude = np.abs(x)
        # Unwrapping stops a jump from +pi to -pi being read as a huge
        # phase change when the signal merely crossed the negative axis.
        phase = np.unwrap(np.angle(x))

        # An all-zero block has no peak and no mean, and the ratio of
        # the two is not 0 dB but undefined. Flooring both keeps the
        # logarithm finite rather than letting a NaN reach the forest.
        peak = max(float(np.max(amplitude) ** 2), 1e-12)
        mean_power = max(float(np.mean(amplitude**2)), 1e-12)
        papr_db = 10.0 * np.log10(peak / mean_power)

        spectrum = np.abs(np.fft.fft(x)) ** 2
        spectrum = np.maximum(spectrum, 1e-20)
        # Flatness: a tone has a spiky spectrum (low flatness), noise has
        # a flat one (high). The geometric-over-arithmetic mean ratio is
        # the standard way of putting a number on that.
        flatness = float(
            np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum)
        )

        real = np.real(x)
        crossings = float(np.mean(np.diff(np.signbit(real)) != 0))

        return np.array(
            [
                float(np.std(amplitude)),
                self._kurtosis(amplitude),
                float(np.std(np.diff(phase))),
                self._kurtosis(np.diff(phase)),
                papr_db,
                flatness,
                float(np.max(spectrum) / np.mean(spectrum)),
                crossings,
            ],
            dtype=float,
        )


class CombinedFeatures(FeatureExtractor):
    """Several extractors glued together.

    DESIGN NOTE: THE COMPOSITE PATTERN, A THIRD TIME
    -------------------------------------------------
    The same shape as :class:`~satrecovery.channel.pipeline.ChannelPipeline`
    and :class:`~satrecovery.receivers.base.ReceiverChain`: it *is* a
    :class:`FeatureExtractor` and it *has* a list of them.  Anything
    that accepts one extractor accepts a combination of extractors
    without knowing the difference.
    """

    def __init__(self, *extractors: FeatureExtractor) -> None:
        if not extractors:
            raise ValidationError(
                "extractors", 0, "need at least one feature extractor"
            )
        for position, extractor in enumerate(extractors):
            if not isinstance(extractor, FeatureExtractor):
                raise ValidationError(
                    f"extractors[{position}]",
                    type(extractor).__name__,
                    "every element must be a FeatureExtractor",
                )
        super().__init__("+".join(e.name for e in extractors))
        self._extractors = list(extractors)

    @property
    def extractors(self) -> tuple[FeatureExtractor, ...]:
        """The extractors being combined."""
        return tuple(self._extractors)

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for extractor in self._extractors:
            names.extend(f"{extractor.name}.{n}" for n in extractor.feature_names)
        return names

    def _compute(self, samples: np.ndarray) -> np.ndarray:
        # The base class already normalised the samples, and each
        # sub-extractor's own extract() would normalise again -- which is
        # harmless, since normalising twice changes nothing.
        return np.concatenate(
            [extractor.extract(samples) for extractor in self._extractors]
        )


def default_extractor() -> CombinedFeatures:
    """The feature set the forest classifier uses unless told otherwise."""
    return CombinedFeatures(CumulantFeatures(), SpectralFeatures())


__all__ = [
    "CombinedFeatures",
    "CumulantFeatures",
    "FeatureExtractor",
    "SpectralFeatures",
    "default_extractor",
]
