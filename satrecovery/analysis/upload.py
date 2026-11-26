"""Your own signal file: give the app a recording, get the answer.

Like a medical-scan app that takes a scan and says what it shows, this
takes a file of signal samples and says what kind of signal it is, then
reads its bits.

Formats accepted
----------------
``.csv`` / ``.txt``
    Two columns of numbers, I and Q (the two halves of each sample),
    one sample per line.  A header line is skipped.
``.npy``
    A NumPy array: complex numbers, or pairs as shape ``(n, 2)``.
``.npz``
    A recording exported from RML24 by this project
    (:meth:`~satrecovery.data.archive.SampleArchive.write`).  Its true
    type is stored with it, so the app can say whether the AI was right
    and check the bits.  Any other ``.npz`` is read as its first array.

Like RML24, the file must be sampled at 1 MHz; the user says the symbol
rate.  At most the first 2048 samples are analysed -- the length every RML24
recording has, and so the length the AI was trained on.

DESIGN NOTE: INHERITANCE AND POLYMORPHISM
-----------------------------------------
:class:`FileInput` is one more :class:`~.inputs.SignalInput`.  The
:class:`~.analyser.Analyser` never asks which kind of input it has: a
file, an RML24 recording and a simulated signal are all analysed by the
same code.  :class:`SignalFile` keeps the reading of each format in one
place (encapsulation).
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..core.conditions import ChannelConditions
from ..core.exceptions import ValidationError
from ..core.frame import Frame
from ..core.signal import Signal
from ..data import layout
from ..data.archive import SampleArchive
from .inputs import RecordedInput, SignalInput

#: Samples analysed: the length of every RML24 recording.
LENGTH = 2048

#: Fewest samples accepted.  A little under LENGTH, because a signal made
#: of whole symbols is not always exactly 2048 samples long (204 symbols
#: of 10 samples is 2040); the AI's measurements barely change for that.
MIN_LENGTH = 2000

#: Largest file accepted, in bytes.
MAX_BYTES = 20 * 1024 * 1024


class SignalFile:
    """Reads a signal file into complex samples."""

    FORMATS = (".csv", ".txt", ".npy", ".npz")

    @classmethod
    def read(cls, name: str, data: bytes) -> np.ndarray:
        """All the samples in the file, as complex numbers."""
        suffix = Path(name).suffix.lower()
        if suffix not in cls.FORMATS:
            raise ValidationError("file", name, f"use one of {', '.join(cls.FORMATS)}")
        if not data:
            raise ValidationError("file", name, "the file is empty")
        if len(data) > MAX_BYTES:
            raise ValidationError("file", name, f"larger than {MAX_BYTES // (1024 * 1024)} MB")
        try:
            if suffix in (".csv", ".txt"):
                values = cls._text(data, name)
            elif suffix == ".npy":
                values = np.load(io.BytesIO(data), allow_pickle=False)
            else:
                with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                    if not archive.files:
                        raise ValidationError("file", name, "the .npz archive is empty")
                    key = "iq" if "iq" in archive.files else archive.files[0]
                    values = np.asarray(archive[key])
                    if key == "iq" and values.ndim == 3:
                        values = values[0]
        except ValidationError:
            raise
        except Exception as error:  # a damaged file: say so plainly
            if suffix in (".npy", ".npz") and not data.startswith((b"\x93NUMPY", b"PK")):
                raise ValidationError("file", name, f"is not a valid {suffix} file") from None
            raise ValidationError("file", name, f"could not be read ({error})") from None
        return cls._to_complex(np.asarray(values), name)

    @staticmethod
    def _text(data: bytes, name: str = "text") -> np.ndarray:
        text = data.decode("utf-8", errors="replace").replace(",", " ").replace(";", " ").replace("\t", " ")
        rows = []
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                rows.append([float(p) for p in parts[:2]])
            except ValueError:
                if rows:                     # only the first line may be a header
                    raise ValidationError("file", name, f"not a number on line {len(rows) + 2}") from None
        if not rows or any(len(r) != 2 for r in rows):
            raise ValidationError("file", name, "expected two columns, I and Q")
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _to_complex(values: np.ndarray, name: str) -> np.ndarray:
        if np.iscomplexobj(values):
            z = values.ravel()
        elif values.ndim == 2 and values.shape[1] == 2:
            z = values[:, 0] + 1j * values[:, 1]
        elif values.ndim == 2 and values.shape[0] == 2:
            z = values[0] + 1j * values[1]
        else:
            raise ValidationError("file", name, f"shape {values.shape}: expected complex numbers or I,Q pairs")
        z = np.asarray(z, dtype=np.complex128)
        if not np.all(np.isfinite(z)):
            raise ValidationError("file", name, "contains empty or infinite values")
        if z.size and not np.any(z):
            raise ValidationError("file", name, "is silent: every sample is zero")
        return z


class FileInput(SignalInput):
    """A signal from the user's own file.  Its true type is not known."""

    def __init__(self, name: str, samples: np.ndarray, symbol_rate: float) -> None:
        samples = np.asarray(samples, dtype=np.complex128).ravel()
        if samples.size < MIN_LENGTH:
            raise ValidationError("file", name, f"has {samples.size:,} samples; at least {MIN_LENGTH:,} are needed")
        if int(round(float(symbol_rate))) not in (100_000, 250_000, 500_000):
            raise ValidationError("symbol_rate", symbol_rate, "choose 100, 250 or 500 kbaud")
        self._name = str(name)
        self._total = int(samples.size)
        self._samples = samples[:LENGTH].copy()
        self._samples.setflags(write=False)
        self._symbol_rate = float(symbol_rate)

    @property
    def samples(self) -> np.ndarray:
        return self._samples

    @property
    def samples_per_symbol(self) -> int:
        return layout.samples_per_symbol(self._symbol_rate)

    @property
    def symbol_rate(self) -> float:
        return self._symbol_rate

    @property
    def true_modulation(self) -> str | None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"kind": "file", "name": self._name, "samples_in_file": self._total, "samples_used": int(self._samples.size),
                "symbol_rate": self._symbol_rate, "samples_per_symbol": self.samples_per_symbol}

    def _labelled_frame(self):
        return None, ""

    def frame(self, modulation: str) -> tuple[Frame, bool, str]:
        """Always decoded blind: nobody told us which bits were sent."""
        sps = self.samples_per_symbol
        n = (self._samples.size // sps) * sps
        samples = self._samples[:n] - np.mean(self._samples[:n])
        blind = Frame(Signal(samples, self._symbol_rate * sps), bits=np.zeros(0, dtype=np.uint8),
                      modulation=modulation, conditions=ChannelConditions(snr_db=10.0),
                      samples_per_symbol=sps, symbol_rate=self._symbol_rate, source="file")
        note = ("Not checked: a file does not say which bits were sent, so there is nothing to compare "
                "with. The receiver's own confidence in each bit is shown instead.")
        return blind, False, note


def input_from_file(name: str, data: bytes, symbol_rate: float | None) -> SignalInput:
    """The right kind of input for an uploaded file.

    A recording exported from RML24 keeps its labels, so it becomes an
    ordinary :class:`RecordedInput` -- checked like any RML24 recording.
    Anything else is a :class:`FileInput`.
    """
    if Path(name).suffix.lower() == ".npz" and len(data) <= MAX_BYTES:
        from .testfile import TestFile

        try:
            simulated = TestFile.input_from(name, data)
        except (OSError, ValueError):
            simulated = None
        if simulated is not None:
            return simulated
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "upload.npz"
            path.write_bytes(data)
            try:
                archive = SampleArchive(path)
            except (ValidationError, ValueError, OSError):
                archive = None
            if archive is not None and len(archive) > 0:
                row = int(archive.select(archive.modulations()[0])[0]) if archive.modulations() else None
                if row is not None:
                    return RecordedInput(archive, row, file_name=name)
    samples = SignalFile.read(name, data)          # a file that cannot be read says so first
    if symbol_rate is None:
        raise ValidationError("symbol_rate", None, "say the symbol rate: 100, 250 or 500 kbaud")
    return FileInput(name, samples, symbol_rate)


__all__ = ["FileInput", "LENGTH", "MIN_LENGTH", "SignalFile", "input_from_file"]
