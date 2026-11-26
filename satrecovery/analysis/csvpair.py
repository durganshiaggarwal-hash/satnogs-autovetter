"""Test files as two CSV files, readable in Excel.

A CSV test file is a pair with matching names, kept in the same folder:

``<name>_signal.csv``
    The received signal: a header ``I,Q``, then one sample per line.
``<name>_bits.csv``
    The answer key.  First some labelled lines, two columns each::

        type,QPSK            (required)
        speed_baud,250000    (required: 100000, 250000 or 500000)
        snr_db,8             (optional, for the record)
        answer_key,simulator (optional: simulator, phase3, none or by hand)

    then a line ``bit``, then one 0 or 1 per line.

:class:`CsvTestFile` writes a pair from a ``.npz`` test file (export)
and turns a pair back into something to analyse (import).  A pair made
by hand works too, as long as the number of bits fits the signal.

For a real RML24 recording the bits file shows the Phase 3 answer key
so it can be read; loading the pair back works the answer key out again
from the signal itself, exactly as the Real signal tab does, so the
result is the same as with the ``.npz``.
"""

from __future__ import annotations

import csv
import io
import json
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
from ..data.recording import Recording
from ..modulation.registry import ModulationRegistry
from .inputs import RecordedInput, SignalInput
from .testfile import TestFile, _open_npz
from .upload import MIN_LENGTH, SignalFile

SIGNAL_SUFFIX = "_signal.csv"
BITS_SUFFIX = "_bits.csv"
SPEEDS = (100_000, 250_000, 500_000)
#: Longest signal accepted from a CSV pair, in samples.
MAX_SAMPLES = 16_384


class CsvTestFile:
    """Writes and reads the signal + bits CSV pair."""

    # -- names -------------------------------------------------------------------
    @staticmethod
    def base_name(name: str) -> str:
        """``test_QPSK.npz`` or ``test_QPSK_signal.csv`` -> ``test_QPSK``."""
        stem = Path(name).name
        for suffix in (SIGNAL_SUFFIX, BITS_SUFFIX, ".npz", ".csv"):
            if stem.lower().endswith(suffix):
                return stem[: -len(suffix)]
        return Path(stem).stem

    @staticmethod
    def partner(signal_path: str | Path) -> Path | None:
        """The bits file beside a ``..._signal.csv``, if it is there."""
        path = Path(signal_path)
        if not path.name.lower().endswith(SIGNAL_SUFFIX):
            return None
        candidate = path.with_name(path.name[: -len(SIGNAL_SUFFIX)] + BITS_SUFFIX)
        return candidate if candidate.is_file() else None

    # -- export ------------------------------------------------------------------
    @staticmethod
    def from_test_file(data: bytes) -> tuple[str, str]:
        """``(signal.csv text, bits.csv text)`` for a ``.npz`` test file."""
        archive = _open_npz(data)
        info = TestFile.describe(data)
        if archive is None or info is None:
            raise ValidationError("file", "test file", "is not a test file made by this app")
        with archive:
            if info["kind"] == "simulated":
                samples = np.asarray(archive["samples"], dtype=np.complex128)
                bits = np.asarray(archive["bits"], dtype=np.uint8)
                conditions = json.loads(str(archive["conditions"]))
                meta = [("type", info["modulation"]), ("speed_baud", int(info["symbol_rate"])),
                        ("snr_db", conditions.get("snr_db")), ("doppler_hz", conditions.get("freq_offset_hz", 0.0)),
                        ("answer_key", "simulator"), ("source", "simulator")]
            else:
                iq = np.asarray(archive["iq"][0], dtype=np.float64)
                samples = iq[:, 0] + 1j * iq[:, 1]
                recording = Recording(samples, info["modulation"], info["snr_db"], info["symbol_rate"], info["row"])
                reference = RecordedInput(_single(recording), info["row"]).reference()
                bits = np.asarray(reference.bits, dtype=np.uint8) if reference.accepted else np.zeros(0, np.uint8)
                meta = [("type", info["modulation"]), ("speed_baud", int(info["symbol_rate"])),
                        ("snr_db", info["snr_db"]), ("answer_key", "phase3" if reference.accepted else "none"),
                        ("source", "rml24"), ("row", info["row"]),
                        ("note", "worked out again from the signal when loaded, as on the Real signal tab")]
        signal = io.StringIO()
        writer = csv.writer(signal, lineterminator="\n")
        writer.writerow(["I", "Q"])
        writer.writerows((f"{z.real:.8f}", f"{z.imag:.8f}") for z in samples)
        bits_text = io.StringIO()
        writer = csv.writer(bits_text, lineterminator="\n")
        writer.writerows((key, value) for key, value in meta if value is not None)
        writer.writerow(["bit"])
        writer.writerows([int(b)] for b in bits)
        return signal.getvalue(), bits_text.getvalue()

    @staticmethod
    def write_pair(folder: str | Path, base: str, data: bytes) -> tuple[Path, Path]:
        """Write ``<base>_signal.csv`` and ``<base>_bits.csv`` from a ``.npz`` test file."""
        folder = Path(folder)
        signal_text, bits_text = CsvTestFile.from_test_file(data)
        signal_path = folder / (CsvTestFile.base_name(base) + SIGNAL_SUFFIX)
        bits_path = folder / (CsvTestFile.base_name(base) + BITS_SUFFIX)
        signal_path.write_text(signal_text, encoding="utf-8")
        bits_path.write_text(bits_text, encoding="utf-8")
        return signal_path, bits_path

    # -- import ------------------------------------------------------------------
    @staticmethod
    def read_bits(text: str | bytes, name: str = "bits file") -> dict[str, Any]:
        """The labelled lines and the bits of a bits CSV."""
        if isinstance(text, bytes):
            text = text.decode("utf-8-sig", errors="replace")
        meta: dict[str, str] = {}
        bits: list[int] = []
        in_bits = False
        for number, raw in enumerate(text.splitlines(), start=1):
            cells = [c.strip() for c in raw.replace(";", ",").replace("\t", ",").split(",")]
            cells = [c for c in cells if c != ""]
            if not cells:
                continue
            if not in_bits:
                if cells[0].lower() == "bit":
                    in_bits = True
                elif len(cells) >= 2:
                    meta[cells[0].lower()] = cells[1]
                else:
                    raise ValidationError("bits", name, f"line {number}: expected a label and a value, "
                                                        f"like 'type,QPSK', or the line 'bit'")
                continue
            if cells[0] not in ("0", "1"):
                raise ValidationError("bits", name, f"line {number}: expected 0 or 1, found {cells[0]!r}")
            bits.append(int(cells[0]))
        if not in_bits:
            raise ValidationError("bits", name, "has no line 'bit' before the bits")
        kind = meta.get("type", "").upper().replace(" ", "")
        if kind not in layout.SUPPORTED:
            raise ValidationError("bits", name, f"needs a line 'type,…' with one of {', '.join(layout.SUPPORTED)}")
        try:
            speed = int(float(meta.get("speed_baud", "nan")))
        except ValueError:
            speed = 0
        if speed not in SPEEDS:
            raise ValidationError("bits", name, "needs a line 'speed_baud,…' of 100000, 250000 or 500000")
        snr = meta.get("snr_db")
        return {"type": kind, "speed_baud": speed, "snr_db": float(snr) if snr not in (None, "") else None,
                "answer_key": meta.get("answer_key", "by hand"), "source": meta.get("source", "by hand").lower(),
                "row": int(float(meta["row"])) if meta.get("row") else None,
                "bits": np.asarray(bits, dtype=np.uint8)}

    @staticmethod
    def summary(info: dict[str, Any], bits_name: str) -> str:
        n = int(info["bits"].size)
        where = {"simulator": "made by the simulator", "phase3": "RML24 recording, Phase 3 answer key",
                 "none": "RML24 recording, true type only"}.get(info["answer_key"], "made by hand")
        return (f"Answer key found: {bits_name}\n{info['type']} · {info['speed_baud'] // 1000} kbaud · "
                f"{n:,} bits · {where}")

    @staticmethod
    def input_from(signal_name: str, signal_data: bytes, bits_name: str, bits_data: bytes) -> SignalInput:
        """Something to analyse, graded against the bits file."""
        info = CsvTestFile.read_bits(bits_data, bits_name)
        samples = SignalFile.read(signal_name, signal_data)
        if info["source"] == "rml24":
            recording = Recording(samples, info["type"], info["snr_db"] or 0.0, info["speed_baud"], info["row"] or 0)
            return RecordedInput(_single(recording), recording.row, file_name=signal_name)
        return CsvPairInput(signal_name, samples, info)


def _single(recording: Recording) -> SampleArchive:
    """An archive holding one recording, so the Real signal tab's own code can grade it."""
    with tempfile.TemporaryDirectory() as folder:
        return SampleArchive(SampleArchive.write(Path(folder) / "one.npz", [recording]))


class CsvPairInput(SignalInput):
    """A signal CSV graded against its bits CSV."""

    def __init__(self, name: str, samples: np.ndarray, info: dict[str, Any]) -> None:
        samples = np.asarray(samples, dtype=np.complex128).ravel()
        if samples.size < MIN_LENGTH:
            raise ValidationError("file", name, f"has {samples.size:,} samples; at least {MIN_LENGTH:,} are needed")
        if samples.size > MAX_SAMPLES:
            raise ValidationError("file", name, f"has {samples.size:,} samples; at most {MAX_SAMPLES:,} are accepted")
        self._name = str(name)
        self._modulation = info["type"]
        self._symbol_rate = float(info["speed_baud"])
        self._sps = layout.samples_per_symbol(self._symbol_rate)
        self._samples = samples[: (samples.size // self._sps) * self._sps].copy()
        self._samples.setflags(write=False)
        scheme = ModulationRegistry.create(self._modulation)
        expected = (self._samples.size // self._sps) * scheme.bits_per_symbol
        bits = info["bits"]
        if bits.size != expected:
            raise ValidationError(
                "bits", name, f"the bits file has {bits.size:,} bits, but a {self._modulation} signal of "
                f"{samples.size:,} samples at {self._symbol_rate / 1000:.0f} kbaud carries {expected:,}. "
                "Add or remove bits so the counts match")
        self._bits = bits
        self._symbols = scheme.map_bits(bits)
        snr = info.get("snr_db")
        self._conditions = ChannelConditions(snr_db=float(np.clip(snr if snr is not None else 10.0, -20, 30)))
        self._answer_key = info.get("answer_key", "by hand")

    @property
    def samples(self) -> np.ndarray:
        return self._samples

    @property
    def samples_per_symbol(self) -> int:
        return self._sps

    @property
    def symbol_rate(self) -> float:
        return self._symbol_rate

    @property
    def true_modulation(self) -> str:
        return self._modulation

    def describe(self) -> dict[str, Any]:
        return {"kind": "file", "name": self._name, "test_file": "csv", "answer_key": self._answer_key,
                "samples_in_file": int(self._samples.size), "samples_used": int(self._samples.size),
                "true_modulation": self._modulation, "symbol_rate": self._symbol_rate,
                "samples_per_symbol": self._sps}

    def _labelled_frame(self):
        frame = Frame(Signal(self._samples, self._symbol_rate * self._sps, description=f"CSV pair {self._name}"),
                      bits=self._bits, modulation=self._modulation, conditions=self._conditions,
                      tx_symbols=self._symbols, samples_per_symbol=self._sps, symbol_rate=self._symbol_rate,
                      source="csv-test-file")
        return frame, "Checked against the bits in the bits CSV."


__all__ = ["BITS_SUFFIX", "CsvPairInput", "CsvTestFile", "SIGNAL_SUFFIX"]
