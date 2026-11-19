"""What can be analysed: a real RML24 recording, or a simulated signal.

DESIGN NOTE: POLYMORPHISM, ONE MORE TIME
----------------------------------------
:class:`SignalInput` says what the analyser needs from anything it is
given: the samples, the symbol rate, and -- once the classifier has
named the modulation -- a :class:`~satrecovery.core.frame.Frame` the
receivers can work on, labelled with the true bits if they are known.
:class:`RecordedInput` and :class:`SimulatedInput` answer those
questions completely differently (one asks the Phase 3 reference
decoder, the other already knows every bit it sent), and the analyser
never asks which it has.

ONE RULE FOR BOTH
-----------------
The receivers decode with the modulation the **classifier** chose, not
the true one -- that is the point of having a classifier.  If it chose
wrongly, the bits cannot be checked (they were read in the wrong
"language"), and the answer says so instead of printing a meaningless
error count.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..core.conditions import ChannelConditions
from ..core.exceptions import ValidationError
from ..core.frame import Frame
from ..core.signal import Signal
from ..data.archive import RecordingArchive
from ..data.recording import Recording
from ..data.reference import Reference, ReferenceDecoder
from ..simulation.simulator import SignalSimulator

#: Schemes whose RML24 bits the Phase 3 reference decoder can vouch for.
CHECKABLE: tuple[str, ...] = ("BPSK", "QPSK", "8PSK")

#: Lowest SNR label at which at least 70% of RML24 recordings pass the
#: reference decoder's test, per scheme and symbol rate.  Measured on the
#: full dataset with ``scripts/check_rate.py`` (30 recordings per cell).
#: ``None``: hardly ever, at any label.  Faster rates leave fewer samples
#: per symbol, so the same label is a noisier signal per symbol.
CHECKABLE_FROM_DB: dict[tuple[str, int], int | None] = {
    ("BPSK", 100_000): 6, ("QPSK", 100_000): 6, ("8PSK", 100_000): 12,
    ("BPSK", 250_000): 8, ("QPSK", 250_000): 10, ("8PSK", 250_000): 16,
    ("BPSK", 500_000): 12, ("QPSK", 500_000): 16, ("8PSK", 500_000): None,
}


def checkable_from(modulation: str, symbol_rate: float) -> int | None:
    """Lowest label (dB) from which this scheme and rate can usually be checked."""
    return CHECKABLE_FROM_DB.get((modulation, int(round(float(symbol_rate)))))


def usually_checkable(modulation: str, snr_label_db: float, symbol_rate: float) -> bool:
    """Do most RML24 recordings with these labels pass the reference test?"""
    start = checkable_from(modulation, symbol_rate)
    return start is not None and float(snr_label_db) >= start


class SignalInput(ABC):
    """Something to analyse."""

    @property
    @abstractmethod
    def samples(self) -> np.ndarray:
        """The complex samples, as received."""

    @property
    @abstractmethod
    def samples_per_symbol(self) -> int: ...

    @property
    @abstractmethod
    def symbol_rate(self) -> float: ...

    @property
    @abstractmethod
    def true_modulation(self) -> str:
        """What it really is (known for RML24 and the simulator)."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Where it came from, for display."""

    @abstractmethod
    def _labelled_frame(self) -> tuple[Frame, str] | tuple[None, str]:
        """A frame labelled with its true bits, or ``None`` and why not."""

    # -- shared ---------------------------------------------------------
    def frame(self, modulation: str) -> tuple[Frame, bool, str]:
        """A frame to decode as ``modulation``: ``(frame, checkable, note)``.

        Labelled with the true bits only when the classifier named the
        modulation correctly and the true bits are known.  Otherwise the
        receivers decode blind and nothing is graded.
        """
        if modulation == self.true_modulation:
            labelled, note = self._labelled_frame()
            if labelled is not None:
                return labelled, True, note
        else:
            note = (f"The AI got this one wrong, so the receiver read the signal as {modulation} when it is really "
                    f"{self.true_modulation} -- like translating Hindi as if it were Tamil. These bits are wrong "
                    f"by construction and are not checked.")
        sps = self.samples_per_symbol
        n = (len(self.samples) // sps) * sps
        samples = self.samples[:n] - np.mean(self.samples[:n])
        blind = Frame(
            Signal(samples, self.symbol_rate * sps),
            bits=np.zeros(0, dtype=np.uint8),
            modulation=modulation,
            conditions=ChannelConditions(snr_db=10.0),
            samples_per_symbol=sps,
            symbol_rate=self.symbol_rate,
            source="analysis",
        )
        return blind, False, note


class RecordedInput(SignalInput):
    """One recording from RML24, by row number."""

    def __init__(self, archive: RecordingArchive, row: int, decoder: ReferenceDecoder | None = None,
                 file_name: str | None = None) -> None:
        if not isinstance(archive, RecordingArchive):
            raise ValidationError("archive", type(archive).__name__, "expected a RecordingArchive")
        self._recording: Recording = archive.recording(int(row))
        self._decoder = decoder or ReferenceDecoder()
        self._reference: Reference | None = None
        #: Set when the recording arrived in a file (a test file or an RML24 export).
        self._file_name = file_name

    @property
    def recording(self) -> Recording:
        return self._recording

    @property
    def samples(self) -> np.ndarray:
        return self._recording.samples

    @property
    def samples_per_symbol(self) -> int:
        return self._recording.samples_per_symbol

    @property
    def symbol_rate(self) -> float:
        return self._recording.symbol_rate

    @property
    def true_modulation(self) -> str:
        return self._recording.modulation

    def reference(self) -> Reference:
        if self._reference is None:
            self._reference = self._decoder.decode(self._recording)
        return self._reference

    def describe(self) -> dict[str, Any]:
        described = {
            "kind": "recorded",
            "origin": self._recording.origin,
            "row": self._recording.row,
            "true_modulation": self.true_modulation,
            "snr_label_db": self._recording.snr_label_db,
            "symbol_rate": self.symbol_rate,
            "samples_per_symbol": self.samples_per_symbol,
        }
        if self._file_name:
            described["file"] = self._file_name
        return described

    def _labelled_frame(self):
        reference = self.reference()
        if not reference.accepted:
            if self.true_modulation not in CHECKABLE:
                return None, (f"Why there is no check: RML24 does not include the bits that were sent. Our Phase 3 "
                              f"reference decoder can work them out only for {', '.join(CHECKABLE)}; "
                              f"{self.true_modulation} packs its points too closely for that to be reliable. "
                              f"So the receiver's own confidence is shown instead. "
                              f"Pick BPSK, QPSK or 8PSK to see the bits checked one by one.")
            rate_kbaud = self.symbol_rate / 1000
            label = self._recording.snr_label_db
            start = checkable_from(self.true_modulation, self.symbol_rate)
            intro = ("Why there is no check: RML24 does not include the bits that were sent, so the Phase 3 "
                     "reference decoder works them out first and only trusts the answer if almost no symbol "
                     f"is in doubt. This recording (labelled {label:+.0f} dB) did not pass that strict test, "
                     "so it is left unchecked rather than checked against bits that might be wrong. ")
            if start is None:
                return None, intro + (f"{self.true_modulation} at {rate_kbaud:.0f} kbaud hardly ever passes: "
                                      "at that speed each symbol has only 2 samples. Choose 100 or 250 kbaud.")
            if label >= start:
                return None, intro + (f"Most {self.true_modulation} recordings at {rate_kbaud:.0f} kbaud and "
                                      f"{label:+.0f} dB do pass; click 'Pick another recording'.")
            return None, intro + (f"{self.true_modulation} at {rate_kbaud:.0f} kbaud can usually be checked "
                                  f"from {start:+d} dB upwards (measured on RML24).")
        samples = self._recording.samples - np.mean(self._recording.samples)
        frame = Frame(
            Signal(samples[reference.crop], self._recording.sample_rate, description=f"RML24 row {reference.row}"),
            bits=reference.graded_bits,
            modulation=reference.modulation,
            conditions=ChannelConditions(snr_db=float(np.clip(self._recording.snr_label_db, -20, 30))),
            tx_symbols=reference.symbols,
            samples_per_symbol=self.samples_per_symbol,
            symbol_rate=self.symbol_rate,
            source="rml24",
            graded_symbols=reference.graded,
        )
        return frame, ("Checked one by one against the bits the Phase 3 reference decoder worked out for this "
                       f"recording ({reference.n_ungraded} symbols at the edges or on glitches left out).")


class SimulatedInput(SignalInput):
    """A signal from the project's simulator, at one of RML24's symbol rates.

    The simulator runs at RML24's settings -- 1 MHz, 2048 samples, the
    same pulse -- and at the symbol rate chosen here, so the classifier
    (trained on RML24) sees a signal of the shape it knows.
    """

    def __init__(self, modulation: str, conditions: ChannelConditions, symbol_rate: float = 250_000.0,
                 seed: int = 0) -> None:
        from ..data import layout

        sps = layout.samples_per_symbol(symbol_rate)
        self._simulator = SignalSimulator(samples_per_symbol=sps, seed=int(seed))
        self._frame = self._simulator.generate_frame(modulation, conditions)
        self._seed = int(seed)

    @property
    def samples(self) -> np.ndarray:
        return self._frame.signal.samples

    @property
    def samples_per_symbol(self) -> int:
        return self._frame.samples_per_symbol

    @property
    def symbol_rate(self) -> float:
        return self._frame.symbol_rate

    @property
    def true_modulation(self) -> str:
        return self._frame.modulation

    @property
    def simulated_frame(self) -> Frame:
        """The frame the simulator made, with the bits it sent (for saving as a test file)."""
        return self._frame

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "simulated",
            "seed": self._seed,
            "true_modulation": self.true_modulation,
            "conditions": self._frame.conditions.to_dict(),
            "symbol_rate": self.symbol_rate,
            "samples_per_symbol": self.samples_per_symbol,
        }

    def _labelled_frame(self):
        return self._frame, "Checked against the bits the simulator sent."


__all__ = ["CHECKABLE", "CHECKABLE_FROM_DB", "checkable_from", "usually_checkable", "RecordedInput", "SignalInput", "SimulatedInput"]
