"""One recording from the archive, before anything has been done to it.

A :class:`Recording` is deliberately *not* a
:class:`~satrecovery.core.frame.Frame`.  A frame is labelled: it knows
its transmitted bits, which is what makes it gradeable.  A recording
from RML24 knows only what the dataset says about it -- the modulation,
an SNR label and a symbol rate.  Turning one into the other needs bits,
and producing those honestly is a step of its own
(:mod:`satrecovery.data.reference`).  Two classes keep that step from
being skipped by accident.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.exceptions import ValidationError
from . import layout


class Recording:
    """Raw samples and the dataset's labels for them. Immutable."""

    __slots__ = ("_samples", "_modulation", "_snr_label_db", "_symbol_rate", "_row", "_origin")

    def __init__(
        self,
        samples: np.ndarray,
        modulation: str,
        snr_label_db: float,
        symbol_rate: float,
        row: int,
        origin: str = "RML24",
    ) -> None:
        samples = np.array(samples, copy=True)
        if samples.ndim == 2 and samples.shape[1] == 2:
            # The archive stores (I, Q) float pairs.
            samples = samples[:, 0].astype(np.float64) + 1j * samples[:, 1].astype(np.float64)
        samples = np.asarray(samples, dtype=np.complex128).ravel()
        if samples.size < 64:
            raise ValidationError("samples", samples.size, "a recording needs at least 64 samples")
        if not np.all(np.isfinite(samples)):
            raise ValidationError("samples", "<array>", "contains NaN or infinity")

        self._samples = samples
        self._samples.flags.writeable = False
        self._modulation = str(modulation)
        self._snr_label_db = float(snr_label_db)
        self._symbol_rate = float(symbol_rate)
        layout.samples_per_symbol(self._symbol_rate)  # validates the rate
        self._row = int(row)
        self._origin = str(origin)

    @property
    def samples(self) -> np.ndarray:
        """The complex samples, read-only."""
        return self._samples

    @property
    def modulation(self) -> str:
        return self._modulation

    @property
    def snr_label_db(self) -> float:
        """The dataset's own SNR label.

        Called a *label* on purpose: it is the authors' figure, defined
        their way, and is used here only to pick recordings.  It is not
        the SNR of a frame this project grades -- that is set by the
        noise :class:`~satrecovery.data.source.RecordedSource` adds,
        on the same scale as the simulator's.
        """
        return self._snr_label_db

    @property
    def symbol_rate(self) -> float:
        return self._symbol_rate

    @property
    def samples_per_symbol(self) -> int:
        return layout.samples_per_symbol(self._symbol_rate)

    @property
    def sample_rate(self) -> float:
        return layout.SAMPLE_RATE_HZ

    @property
    def row(self) -> int:
        """Position in the archive, so any result can be traced back."""
        return self._row

    @property
    def origin(self) -> str:
        return self._origin

    def metadata(self) -> dict[str, Any]:
        return {
            "origin": self._origin,
            "row": self._row,
            "modulation": self._modulation,
            "snr_label_db": self._snr_label_db,
            "symbol_rate": self._symbol_rate,
            "samples_per_symbol": self.samples_per_symbol,
            "n_samples": int(self._samples.size),
        }

    def __len__(self) -> int:
        return int(self._samples.size)

    def __repr__(self) -> str:
        return (
            f"Recording({self._modulation}, row {self._row}, label {self._snr_label_db:+.0f} dB, "
            f"{self._symbol_rate / 1e3:.0f} kbaud, {self._samples.size} samples)"
        )


__all__ = ["Recording"]
