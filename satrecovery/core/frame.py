"""The :class:`Frame` -- one labelled signal observation.

A frame bundles everything the system knows about a single captured
block of signal:

* the received samples (a :class:`~satrecovery.core.signal.Signal`),
* the **bits that were actually transmitted** (our ground truth),
* the ideal transmitted symbols, for plotting a reference constellation,
* which modulation produced it, and under what channel conditions.

This mirrors the labelling of RML24, which stores IQ data, bit data,
modulation class, SNR and symbol rate for every frame.  Keeping the same
shape means a frame from our simulator and a frame loaded from RML24 are
interchangeable everywhere downstream -- see
:class:`~satrecovery.data.repository.FrameRepository` in Phase 3.

The bit labels are the reason this project can measure *recovery* (bit
error rate) and not just classification accuracy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .conditions import ChannelConditions
from .exceptions import ValidationError
from .signal import Signal


class Frame:
    """An immutable, fully labelled signal observation."""

    __slots__ = (
        "_signal",
        "_bits",
        "_modulation",
        "_conditions",
        "_tx_symbols",
        "_samples_per_symbol",
        "_symbol_rate",
        "_source",
        "_graded",
    )

    def __init__(
        self,
        signal: Signal,
        bits: np.ndarray,
        modulation: str,
        conditions: ChannelConditions,
        tx_symbols: np.ndarray | None = None,
        samples_per_symbol: int = 1,
        symbol_rate: float | None = None,
        source: str = "simulated",
        graded_symbols: np.ndarray | None = None,
    ) -> None:
        if not isinstance(signal, Signal):
            raise ValidationError(
                "signal", type(signal).__name__, "expected a Signal instance"
            )
        if not isinstance(conditions, ChannelConditions):
            raise ValidationError(
                "conditions",
                type(conditions).__name__,
                "expected a ChannelConditions instance",
            )

        # Copy for the same reason as Signal: the ground-truth bits must
        # not be reachable through any array the caller still holds.
        bits = np.array(bits, dtype=np.uint8, copy=True, order="C")
        if bits.ndim != 1:
            raise ValidationError("bits", bits.shape, "expected a 1-D array")
        if bits.size and not np.all((bits == 0) | (bits == 1)):
            raise ValidationError("bits", "<array>", "values must be 0 or 1")

        if int(samples_per_symbol) < 1:
            raise ValidationError(
                "samples_per_symbol", samples_per_symbol, "must be at least 1"
            )

        if graded_symbols is None:
            graded = None
        else:
            graded = np.array(graded_symbols, dtype=bool, copy=True).ravel()
            expected = signal.samples.size // max(int(samples_per_symbol), 1)
            if graded.size != expected:
                raise ValidationError(
                    "graded_symbols",
                    graded.size,
                    f"needs one entry per symbol of the frame ({expected})",
                )
            if not graded.any():
                raise ValidationError("graded_symbols", 0, "grades no symbols at all")
            graded.flags.writeable = False

        self._signal = signal
        self._bits = bits
        self._graded = graded
        self._modulation = str(modulation)
        self._conditions = conditions
        self._samples_per_symbol = int(samples_per_symbol)
        self._source = str(source)

        if tx_symbols is None:
            self._tx_symbols = None
        else:
            self._tx_symbols = np.array(
                tx_symbols, dtype=np.complex128, copy=True, order="C"
            )

        if symbol_rate is None:
            self._symbol_rate = signal.sample_rate / self._samples_per_symbol
        else:
            self._symbol_rate = float(symbol_rate)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------
    @property
    def signal(self) -> Signal:
        """The received signal (already immutable, so safe to hand out)."""
        return self._signal

    @property
    def bits(self) -> np.ndarray:
        """Read-only view of the transmitted bits (the ground truth)."""
        view = self._bits.view()
        view.flags.writeable = False
        return view

    @property
    def tx_symbols(self) -> np.ndarray | None:
        """Read-only view of the ideal transmitted symbols, if known."""
        if self._tx_symbols is None:
            return None
        view = self._tx_symbols.view()
        view.flags.writeable = False
        return view

    @property
    def modulation(self) -> str:
        """Name of the modulation scheme, e.g. ``"QPSK"``."""
        return self._modulation

    @property
    def conditions(self) -> ChannelConditions:
        """The channel conditions this frame was observed under."""
        return self._conditions

    @property
    def samples_per_symbol(self) -> int:
        """Oversampling factor of the received samples."""
        return self._samples_per_symbol

    @property
    def symbol_rate(self) -> float:
        """Symbol rate in symbols per second (baud)."""
        return self._symbol_rate

    @property
    def source(self) -> str:
        """Where the frame came from: ``"simulated"`` or ``"rml24"``."""
        return self._source

    @property
    def graded_symbols(self) -> np.ndarray | None:
        """Which of the frame's symbols are graded; ``None`` means all.

        ``None`` for simulated frames: the simulator knows every bit it
        sent and models a continuous transmission with no edges.

        A recording is different.  It is a window cut out of a stream,
        so its first and last few symbols arrive with half their pulse
        missing; and RML24 has the odd corrupted symbol besides.  For
        those, nobody -- not the receivers, not the reference that
        labels them -- can say what was sent, and grading them would
        measure the recording, not the receiver.

        When a mask is given, :attr:`bits` holds labels for the graded
        symbols only, :attr:`tx_symbols` still covers the whole frame
        (receivers take their preamble from its start), and
        :meth:`~satrecovery.receivers.base.Receiver.recover` keeps the
        same symbols of every receiver's answer.  One rule, applied in
        one place, for every receiver.
        """
        if self._graded is None:
            return None
        view = self._graded.view()
        view.flags.writeable = False
        return view

    @property
    def n_bits(self) -> int:
        """Number of transmitted bits carried by this frame."""
        return int(self._bits.size)

    @property
    def n_samples(self) -> int:
        """Number of received samples."""
        return len(self._signal)

    @property
    def snr_db(self) -> float:
        """Shortcut to ``frame.conditions.snr_db``, used constantly."""
        return self._conditions.snr_db

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------
    def metadata(self) -> dict[str, Any]:
        """Everything except the heavy arrays, for logging and the API."""
        return {
            "modulation": self._modulation,
            "source": self._source,
            "n_bits": self.n_bits,
            "n_samples": self.n_samples,
            "samples_per_symbol": self._samples_per_symbol,
            "symbol_rate": self._symbol_rate,
            "graded_symbols": None if self._graded is None else int(self._graded.sum()),
            "conditions": self._conditions.to_dict(),
        }

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return (
            f"Frame({self._modulation}, {self.n_samples} samples, "
            f"{self.n_bits} bits, {self._conditions.describe()}, "
            f"source={self._source!r})"
        )
