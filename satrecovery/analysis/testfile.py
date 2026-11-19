"""Test files: a signal that carries its own answer key.

A plain signal file holds only what was received, so nothing can be
checked.  A **test file** also holds what the signal really is and,
where known, which bits were sent -- so loading it in the "Your own
file" tab grades the AI and the receiver, exactly as the tab it was
saved from did.

Two kinds, one ``.npz`` format (marked ``format = satellite-test-file``):

``simulated``
    Saved from the Simulated signal tab.  Holds the received samples,
    the bits and symbols the simulator sent, the channel conditions and
    the speed.  The answer key is exact.
``rml24``
    Saved from the Real signal tab.  Laid out like a
    :class:`~satrecovery.data.archive.SampleArchive` of one recording, so
    loading it runs the very same Phase 3 reference decoder and the very
    same check as the Real signal tab.  RML24 stores no bits, so this
    answer key is a reconstruction, trusted only where the reference
    decoder accepts it; otherwise only the true type is known.

DESIGN NOTE: INHERITANCE AND POLYMORPHISM, ONCE MORE
----------------------------------------------------
:class:`TestFileInput` is one more :class:`~.inputs.SignalInput`.  It
overrides only :meth:`_labelled_frame` -- the answer key -- and the
:class:`~.analyser.Analyser` grades it with no idea it came from a file.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.conditions import ChannelConditions
from ..core.exceptions import ValidationError
from ..core.frame import Frame
from ..core.signal import Signal
from ..data import layout
from .inputs import RecordedInput, SignalInput, SimulatedInput

FORMAT = "satellite-test-file"
VERSION = 1


def _open_npz(data: bytes):
    """The archive inside ``data``, or None if it is not an .npz archive at all."""
    try:
        loaded = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError, EOFError):
        return None
    if isinstance(loaded, np.lib.npyio.NpzFile):
        return loaded
    return None


def _name_part(value: float, unit: str) -> str:
    return f"{value:+.0f}{unit}"


class TestFile:
    """Writes test files, and says what is inside one."""

    __test__ = False          # a class named Test* that pytest must not collect

    # -- writing ------------------------------------------------------------
    @staticmethod
    def from_simulated(signal: SimulatedInput) -> tuple[str, bytes]:
        """A test file of a simulated signal: ``(suggested name, bytes)``."""
        frame = signal.simulated_frame
        conditions = frame.conditions.to_dict()
        name = (f"test_{frame.modulation}_{_name_part(conditions['snr_db'], 'dB')}"
                f"_{frame.symbol_rate / 1000:.0f}kbaud")
        if conditions.get("freq_offset_hz"):
            name += f"_doppler{conditions['freq_offset_hz']:.0f}"
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            format=np.array(FORMAT), version=np.array(VERSION), kind=np.array("simulated"),
            samples=np.asarray(frame.signal.samples, dtype=np.complex128),
            bits=np.asarray(frame.bits, dtype=np.uint8),
            tx_symbols=np.asarray(frame.tx_symbols if frame.tx_symbols is not None else [], dtype=np.complex128),
            modulation=np.array(frame.modulation),
            symbol_rate=np.array(float(frame.symbol_rate)),
            samples_per_symbol=np.array(int(frame.samples_per_symbol)),
            sample_rate=np.array(float(frame.signal.sample_rate)),
            conditions=np.array(json.dumps(conditions)),
            answer_key=np.array("simulator"),
        )
        return name + ".npz", buffer.getvalue()

    @staticmethod
    def from_recording(signal: RecordedInput) -> tuple[str, bytes]:
        """A test file of one RML24 recording: ``(suggested name, bytes)``.

        Written in the sample-archive layout, so loading it repeats the
        Real signal tab's reference decoding and check exactly.
        """
        recording = signal.recording
        reference = signal.reference()
        answer_key = "phase3" if reference.accepted else "none"
        name = (f"test_RML24_{recording.row}_{recording.modulation}"
                f"_{_name_part(recording.snr_label_db, 'dB')}_{recording.symbol_rate / 1000:.0f}kbaud")
        samples = recording.samples
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            # the sample-archive part (see SampleArchive.KEYS)
            rows=np.array([recording.row], dtype=np.int64),
            iq=np.stack([samples.real, samples.imag], axis=1)[None, ...].astype(np.float32),
            cls=np.array([layout.class_index(recording.modulation)], dtype=np.int32),
            snr=np.array([int(round(recording.snr_label_db))], dtype=np.int32),
            rate=np.array([int(round(recording.symbol_rate))], dtype=np.int32),
            class_names=np.array(layout.CLASS_NAMES),
            # the test-file part
            format=np.array(FORMAT), version=np.array(VERSION), kind=np.array("rml24"),
            answer_key=np.array(answer_key),
        )
        return name + ".npz", buffer.getvalue()

    # -- reading ---------------------------------------------------------------
    @staticmethod
    def describe(data: bytes) -> dict[str, Any] | None:
        """What a test file holds, in a few words; ``None`` for any other file."""
        archive = _open_npz(data)
        if archive is None:
            return None
        try:
            with archive:
                if "format" not in archive.files or str(archive["format"]) != FORMAT:
                    return None
                kind = str(archive["kind"])
                answer_key = str(archive["answer_key"])
                if kind == "simulated":
                    conditions = json.loads(str(archive["conditions"]))
                    return {"kind": kind, "modulation": str(archive["modulation"]),
                            "symbol_rate": float(archive["symbol_rate"]), "snr_db": conditions.get("snr_db"),
                            "freq_offset_hz": conditions.get("freq_offset_hz", 0.0), "answer_key": answer_key,
                            "n_bits": int(archive["bits"].size)}
                return {"kind": kind, "modulation": layout.class_name(int(archive["cls"][0])),
                        "symbol_rate": float(archive["rate"][0]), "snr_db": float(archive["snr"][0]),
                        "row": int(archive["rows"][0]), "answer_key": answer_key}
        except (OSError, ValueError, KeyError, IndexError):
            return None

    @staticmethod
    def describe_path(path: str | Path) -> dict[str, Any] | None:
        path = Path(path)
        if path.suffix.lower() != ".npz":
            return None
        try:
            return TestFile.describe(path.read_bytes())
        except OSError:
            return None

    @staticmethod
    def summary(info: dict[str, Any]) -> str:
        """One or two plain sentences for the screen."""
        if info["kind"] == "simulated":
            text = f"✓ Test file: answer key inside. Made by the simulator · {info['snr_db']:+.0f} dB"
            if info.get("freq_offset_hz"):
                text += f" · {info['freq_offset_hz']:.0f} Hz Doppler"
            return text + "."
        text = f"Test file: real RML24 recording #{info['row']:,}."
        if info["answer_key"] == "phase3":
            return "✓ " + text + " Answer key worked out by the Phase 3 decoder."
        return text + " True type inside; no answer key for the bits (too noisy to work them out)."

    @staticmethod
    def input_from(name: str, data: bytes) -> SignalInput | None:
        """The input a test file describes, or ``None`` if it is not a simulated test file."""
        archive = _open_npz(data)
        if archive is None:
            return None
        with archive:
            if "format" not in archive.files or str(archive["format"]) != FORMAT:
                return None
            if str(archive.get("kind", "")) != "simulated":
                return None
            return TestFileInput(name, {key: archive[key] for key in archive.files})


class TestFileInput(SignalInput):
    """A simulated test file: the signal, its true type, and the exact bits sent."""

    __test__ = False

    def __init__(self, name: str, fields: dict[str, np.ndarray]) -> None:
        try:
            self._name = str(name)
            self._samples = np.asarray(fields["samples"], dtype=np.complex128).ravel()
            self._bits = np.asarray(fields["bits"], dtype=np.uint8).ravel()
            symbols = np.asarray(fields["tx_symbols"], dtype=np.complex128).ravel()
            self._symbols = symbols if symbols.size else None
            self._modulation = str(fields["modulation"])
            self._symbol_rate = float(fields["symbol_rate"])
            self._sps = int(fields["samples_per_symbol"])
            self._sample_rate = float(fields["sample_rate"])
            self._conditions = ChannelConditions.from_dict(json.loads(str(fields["conditions"])))
        except (KeyError, ValueError, TypeError) as error:
            raise ValidationError("file", name, f"is a damaged test file ({error})") from None
        if self._modulation not in layout.SUPPORTED:
            raise ValidationError("file", name, f"holds an unknown signal type {self._modulation!r}")
        self._samples.setflags(write=False)

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
        return {"kind": "file", "name": self._name, "test_file": "simulated",
                "samples_in_file": int(self._samples.size), "samples_used": int(self._samples.size),
                "true_modulation": self._modulation, "conditions": self._conditions.to_dict(),
                "symbol_rate": self._symbol_rate, "samples_per_symbol": self._sps}

    def _labelled_frame(self):
        frame = Frame(Signal(self._samples, self._sample_rate, description=f"test file {self._name}"),
                      bits=self._bits, modulation=self._modulation, conditions=self._conditions,
                      tx_symbols=self._symbols, samples_per_symbol=self._sps, symbol_rate=self._symbol_rate,
                      source="test-file")
        return frame, "Checked against the bits stored in the test file (made by the simulator, so they are exact)."


__all__ = ["FORMAT", "TestFile", "TestFileInput"]
