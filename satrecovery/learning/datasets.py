"""Turning labelled frames into the arrays a learning algorithm wants.

Phase 1 produces :class:`~satrecovery.core.frame.Frame` objects: a
received signal plus the bits that were actually sent.  Every model in
Phase 4 needs that same information, but each wants it arranged
differently -- one wants a row of summary statistics per frame, another
wants one row per symbol, a third wants the raw waveform.

Rather than write three ad-hoc conversion scripts that drift apart,
each arrangement is a :class:`DatasetBuilder` subclass.  They share the
batching, shuffling and splitting; they differ only in how one frame
becomes rows.

ONE DESIGN DECISION WORTH UNDERSTANDING
---------------------------------------
:class:`DemappingDataset` runs each frame through the **Phase 2 receiver
chain first**, and trains on the symbols that come out the other side.

That is deliberate.  The neural demapper is not trying to replace the
whole receiver -- it replaces the last step, the one that turns a
cleaned-up symbol into bits.  Training it on symbols that have already
been filtered, de-spun and re-timed means it learns the job it will
actually be asked to do, and the comparison against the classical
demapper is like for like: both see identical input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ..core.conditions import ChannelConditions
from ..core.exceptions import ValidationError
from ..core.frame import Frame
from ..modulation.registry import ModulationRegistry
from ..receivers.classical import ClassicalReceiver
from ..simulation.simulator import SignalSimulator
from .features import FeatureExtractor, default_extractor


class DatasetBuilder(ABC):
    """Converts frames into ``(inputs, targets)`` arrays for one task."""

    def __init__(self, name: str | None = None) -> None:
        self._name = str(name) if name else type(self).__name__

    @property
    def name(self) -> str:
        """Short name of this builder."""
        return self._name

    @abstractmethod
    def _rows(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        """Turn one frame into ``(input_rows, target_rows)``."""

    def build(self, frames: Iterable[Frame]) -> tuple[np.ndarray, np.ndarray]:
        """Stack every frame's rows into two arrays.

        Frames that produce nothing usable are skipped rather than
        crashing the build -- a receiver can legitimately fail on a
        frame that is too noisy, and losing one frame out of a thousand
        matters far less than losing the run.
        """
        input_blocks: list[np.ndarray] = []
        target_blocks: list[np.ndarray] = []

        for frame in frames:
            try:
                inputs, targets = self._rows(frame)
            except Exception:  # noqa: BLE001 - a bad frame must not stop the build
                continue
            if len(inputs) == 0:
                continue
            input_blocks.append(np.asarray(inputs))
            target_blocks.append(np.asarray(targets))

        if not input_blocks:
            raise ValidationError(
                "frames", 0, "no usable rows were produced from these frames"
            )

        return np.concatenate(input_blocks), np.concatenate(target_blocks)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class ClassificationDataset(DatasetBuilder):
    """One row of summary statistics per frame, labelled with its scheme.

    This is the task almost every published paper on RML24 tackles:
    *which modulation is this?*  Worth having as a point of comparison,
    and worth being clear that it is not the question this project is
    really about -- knowing the modulation does not get anybody's data
    back.
    """

    def __init__(
        self,
        extractor: FeatureExtractor | None = None,
        samples_per_symbol: int = 8,
        matched_filter: Any = None,
    ) -> None:
        super().__init__("Classification")
        self._extractor = extractor or default_extractor()
        self._sps = int(samples_per_symbol)
        self._pulse = matched_filter

    @property
    def extractor(self) -> FeatureExtractor:
        """The feature extractor in use."""
        return self._extractor

    def _rows(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        samples = frame.signal.samples
        if self._pulse is not None:
            samples = self._pulse.matched_filter(samples)
        symbols = samples[:: max(frame.samples_per_symbol, 1)]

        features = self._extractor.extract(symbols)
        label = np.array([frame.modulation])
        return features[None, :], label


class DemappingDataset(DatasetBuilder):
    """One row per symbol: a window of received symbols, and its bits.

    The window is the interesting part.  A classical demapper looks at
    one symbol in isolation and picks the nearest constellation point.
    Giving the network its neighbours as well means it *can* learn to
    compensate for symbols smearing into each other -- something the
    nearest-neighbour rule cannot do by construction.

    Whether it actually does is the experiment.
    """

    def __init__(
        self,
        receiver: ClassicalReceiver,
        window: int = 2,
        skip_pilot: bool = True,
    ) -> None:
        if int(window) < 0:
            raise ValidationError("window", window, "cannot be negative")
        super().__init__("Demapping")
        self._receiver = receiver
        self._window = int(window)
        self._skip_pilot = bool(skip_pilot)

    @property
    def window(self) -> int:
        """How many symbols either side of the centre are included."""
        return self._window

    @property
    def width(self) -> int:
        """Total symbols per row, ``2 * window + 1``."""
        return 2 * self._window + 1

    @property
    def n_features(self) -> int:
        """Numbers per row: in-phase and quadrature for each symbol."""
        return 2 * self.width

    def _rows(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        scheme = ModulationRegistry.create(frame.modulation)
        result = self._receiver.recover(frame)

        symbols = result.symbols
        if symbols is None or symbols.size == 0:
            return np.empty((0, self.n_features)), np.empty((0, scheme.bits_per_symbol))

        bits = frame.bits.reshape(-1, scheme.bits_per_symbol)
        usable = min(symbols.size, bits.shape[0])
        symbols = symbols[:usable]
        bits = bits[:usable]

        start = self._receiver.pilot_symbols if self._skip_pilot else 0
        start = max(start, self._window)
        stop = usable - self._window
        if stop <= start:
            return np.empty((0, self.n_features)), np.empty((0, scheme.bits_per_symbol))

        centres = np.arange(start, stop)
        offsets = np.arange(-self._window, self._window + 1)
        windows = symbols[centres[:, None] + offsets[None, :]]

        # Interleave real and imaginary parts into one flat row.
        rows = np.empty((centres.size, self.n_features), dtype=float)
        rows[:, 0::2] = np.real(windows)
        rows[:, 1::2] = np.imag(windows)

        return rows, bits[centres].astype(np.float32)


class EndToEndDataset(DatasetBuilder):
    """The whole received waveform in, every bit out.

    The most ambitious arrangement and the least forgiving.  The network
    gets the raw oversampled signal with the frequency offset, timing
    error and fading still in it, and has to produce the bits -- meaning
    it must discover carrier recovery, timing recovery and demapping for
    itself, from examples alone.

    Phase 2 needed six carefully designed stages to do that. Expecting a
    small network to rediscover all of them is optimistic, and finding
    out how far it gets is the point of including it.
    """

    def __init__(self, max_symbols: int | None = None) -> None:
        super().__init__("EndToEnd")
        self._max_symbols = max_symbols

    def _rows(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        scheme = ModulationRegistry.create(frame.modulation)
        iq = frame.signal.as_iq_array().astype(np.float32)

        # Normalise per frame: the network should learn the shape of the
        # signal, not how loud this particular capture happened to be.
        power = float(np.mean(iq**2))
        if power > 0:
            iq = iq / np.sqrt(power)

        bits = frame.bits.astype(np.float32)
        if self._max_symbols is not None:
            keep_bits = self._max_symbols * scheme.bits_per_symbol
            keep_samples = self._max_symbols * frame.samples_per_symbol
            iq = iq[:, :keep_samples]
            bits = bits[:keep_bits]

        return iq[None, ...], bits[None, :]


# --------------------------------------------------------------------------
# Generating the frames themselves
# --------------------------------------------------------------------------
def training_conditions(
    snr_range: tuple[float, float] = (-6.0, 20.0),
    steps: int = 14,
    with_impairments: bool = True,
    rng: np.random.Generator | None = None,
) -> list[ChannelConditions]:
    """A spread of channel conditions to train across.

    Training on one signal-to-noise ratio produces a model that works at
    that ratio and nowhere else, so the set spans the whole usable range.

    ``with_impairments`` adds random frequency, phase and timing errors
    on top. For a demapper that sits behind the Phase 2 chain those are
    mostly corrected before it ever sees anything, but including them
    means it learns on the *residue* a real receiver leaves behind,
    rather than on an unrealistically clean signal.
    """
    generator = rng or np.random.default_rng(0)
    low, high = snr_range
    conditions: list[ChannelConditions] = []

    for snr in np.linspace(low, high, int(steps)):
        settings: dict[str, Any] = {"snr_db": float(snr)}
        if with_impairments:
            settings.update(
                freq_offset_hz=float(generator.uniform(-1500, 1500)),
                phase_offset_rad=float(generator.uniform(-np.pi, np.pi)),
                timing_offset=float(generator.uniform(-0.5, 0.5)),
            )
            if generator.random() < 0.5:
                settings["fading"] = "multipath_rician"
                settings["rician_k_db"] = float(generator.uniform(6.0, 16.0))
        conditions.append(ChannelConditions(**settings))

    return conditions


def generate_frames(
    simulator: SignalSimulator,
    modulations: Sequence[str],
    conditions: Sequence[ChannelConditions],
    frames_each: int = 2,
) -> list[Frame]:
    """Frames for every modulation and condition combination."""
    return simulator.generate_batch(modulations, conditions, frames_each)


def split_rows(
    inputs: np.ndarray,
    targets: np.ndarray,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split into training and validation halves.

    A plain random split, which is the right thing *within* a set of
    conditions. Phase 6 adds the harder splits -- holding out whole
    signal-to-noise ranges and whole modulations -- which is where
    learned methods usually show their limits.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValidationError(
            "validation_fraction", validation_fraction, "must be between 0 and 1"
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(inputs))
    cut = int(len(inputs) * (1.0 - validation_fraction))
    if cut < 1 or cut >= len(inputs):
        raise ValidationError(
            "inputs", len(inputs), "too few rows to split into train and validation"
        )

    train, validate = order[:cut], order[cut:]
    return inputs[train], targets[train], inputs[validate], targets[validate]


__all__ = [
    "ClassificationDataset",
    "DatasetBuilder",
    "DemappingDataset",
    "EndToEndDataset",
    "generate_frames",
    "split_rows",
    "training_conditions",
]
