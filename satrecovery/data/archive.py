"""Reading recordings: the full 20 GB archive, or a small sample of it.

DESIGN NOTE: ONE INTERFACE, TWO STORAGE FORMATS
-----------------------------------------------
:class:`RecordingArchive` says what any collection of recordings can
do: list its labels, find the rows matching a request, and hand back
one :class:`~satrecovery.data.recording.Recording`.  Two subclasses
differ only in *where the numbers live*:

* :class:`RML24File` reads the published HDF5 file a row at a time.  It
  never loads the 20 GB of samples, only the three small label columns.
* :class:`SampleArchive` reads a ``.npz`` of a few hundred recordings --
  what ``scripts/probe_rml24.py`` exports, and what the tests build from
  the simulator.  It lets the whole of Phase 3 be developed and tested
  without the big file, and lets anyone without 45 GB of disk run
  the demo.

Everything downstream -- the reference decoder, the frame source, the
benchmark -- holds a ``RecordingArchive`` and cannot tell which it has.

Both are context managers, so the HDF5 file is closed however the
``with`` block ends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from ..core.exceptions import ValidationError
from . import layout
from .recording import Recording


class RecordingArchive(ABC):
    """Anything that stores labelled recordings by row number."""

    def __init__(self, name: str) -> None:
        self._name = str(name)
        self._labels_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    # -- what subclasses provide ----------------------------------------
    @abstractmethod
    def _load_labels(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(rows, class, snr, symbol_rate)``, one entry per recording."""

    @abstractmethod
    def _read_samples(self, row: int) -> np.ndarray:
        """The raw stored samples of one row."""

    def close(self) -> None:
        """Release any open file. Safe to call twice."""

    # -- what everyone gets ---------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    def _labels(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._labels_cache is None:
            rows, classes, snrs, rates = self._load_labels()
            self._labels_cache = tuple(np.asarray(a) for a in (rows, classes, snrs, rates))
        return self._labels_cache

    def __len__(self) -> int:
        return int(self._labels()[0].size)

    def select(
        self,
        modulation: str,
        snr_label_db: float | None = None,
        symbol_rate: float | None = None,
    ) -> np.ndarray:
        """Row numbers matching a request, in storage order.

        Only the label columns are consulted, so this is fast even on
        the full archive.
        """
        rows, classes, snrs, rates = self._labels()
        mask = classes == layout.class_index(modulation)
        if snr_label_db is not None:
            mask &= snrs == int(round(float(snr_label_db)))
        if symbol_rate is not None:
            mask &= rates == int(round(float(symbol_rate)))
        return rows[mask]

    def symbol_rates(self, modulation: str) -> list[int]:
        """The symbol rates a scheme was recorded at."""
        rows, classes, _, rates = self._labels()
        return sorted(int(r) for r in np.unique(rates[classes == layout.class_index(modulation)]))

    def modulations(self) -> list[str]:
        """Which of the supported schemes this archive actually holds."""
        present = set(int(c) for c in np.unique(self._labels()[1]))
        return [name for name, index in layout.SUPPORTED.items() if index in present]

    def recording(self, row: int) -> Recording:
        """One recording, read from storage now."""
        rows, classes, snrs, rates = self._labels()
        where = np.flatnonzero(rows == int(row))
        if where.size == 0:
            raise ValidationError("row", row, f"not in {self._name}")
        i = int(where[0])
        name = layout.class_name(int(classes[i]))
        return Recording(
            self._read_samples(int(row)),
            modulation=name,
            snr_label_db=float(snrs[i]),
            symbol_rate=float(rates[i]),
            row=int(row),
            origin=self._name,
        )

    def recordings(self, rows: Iterable[int]) -> Iterator[Recording]:
        """Read several rows, one at a time."""
        for row in rows:
            yield self.recording(int(row))

    def summary(self) -> dict[str, dict[str, object]]:
        """Per supported scheme: how many recordings, at which rates and labels."""
        rows, classes, snrs, rates = self._labels()
        out: dict[str, dict[str, object]] = {}
        for name in self.modulations():
            mask = classes == layout.SUPPORTED[name]
            out[name] = {
                "recordings": int(mask.sum()),
                "symbol_rates": sorted(int(r) for r in np.unique(rates[mask])),
                "snr_labels_db": sorted(int(s) for s in np.unique(snrs[mask])),
            }
        return out

    def __enter__(self) -> "RecordingArchive":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._name!r})"


class RML24File(RecordingArchive):
    """The published ``RML24_IQdata.h5``, read a row at a time.

    Parameters
    ----------
    path:
        The ``.h5`` file, or the folder it was unzipped into.
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path).expanduser()
        if path.is_dir():
            found = sorted(path.glob("*.h5")) + sorted(path.glob("*.hdf5"))
            if not found:
                raise ValidationError("path", str(path), "no .h5 file in this folder")
            path = found[0]
        if not path.is_file():
            raise ValidationError("path", str(path), "file not found")
        try:
            import h5py
        except ImportError:  # pragma: no cover - depends on the machine
            raise ValidationError(
                "h5py", "missing", "reading RML24 needs h5py: pip install h5py"
            ) from None

        super().__init__(path.name)
        self._path = path
        try:
            self._file = h5py.File(path, "r")
        except OSError as error:          # not HDF5 at all, or damaged
            raise ValidationError("path", str(path), f"not an HDF5 file ({error.__class__.__name__})") from None
        missing = [k for k in ("IQ_data", "class", "snr", "symbol_rate") if k not in self._file]
        if missing:
            self._file.close()
            raise ValidationError("path", str(path), f"not an RML24 file: no {', '.join(missing)}")

    @property
    def path(self) -> Path:
        return self._path

    def _load_labels(self):
        classes = self._file["class"][:]
        return (
            np.arange(classes.size),
            classes,
            self._file["snr"][:],
            self._file["symbol_rate"][:],
        )

    def _read_samples(self, row: int) -> np.ndarray:
        if self._file is None:
            raise ValidationError("archive", self._name, "has been closed")
        return self._file["IQ_data"][int(row)]

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class SampleArchive(RecordingArchive):
    """A small ``.npz`` of recordings, with their original row numbers.

    Holds everything in memory -- a few hundred recordings are a few
    megabytes.  The format is what ``scripts/probe_rml24.py`` writes;
    :meth:`write` produces the same thing from any recordings, which is
    how the tests build an archive from the simulator.
    """

    KEYS = ("rows", "iq", "cls", "snr", "rate")

    def __init__(self, path: str | Path) -> None:
        path = Path(path).expanduser()
        if not path.is_file():
            raise ValidationError("path", str(path), "file not found")
        try:
            data = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:      # not a zip, damaged, or pickled objects
            raise ValidationError("path", str(path), f"cannot be read as .npz ({error.__class__.__name__})") from None
        with data:
            missing = [k for k in self.KEYS if k not in data.files]
            if missing:
                raise ValidationError("path", str(path), f"not a sample archive: no {', '.join(missing)}")
            self._rows = np.asarray(data["rows"], dtype=np.int64)
            self._iq = np.asarray(data["iq"])
            self._classes = np.asarray(data["cls"])
            self._snrs = np.asarray(data["snr"])
            self._rates = np.asarray(data["rate"])
        super().__init__(path.name)
        self._index = {int(r): i for i, r in enumerate(self._rows)}

    def _load_labels(self):
        return self._rows, self._classes, self._snrs, self._rates

    def _read_samples(self, row: int) -> np.ndarray:
        return self._iq[self._index[int(row)]]

    @staticmethod
    def write(path: str | Path, recordings: Iterable[Recording]) -> Path:
        """Save recordings in this format; returns the path written."""
        recordings = list(recordings)
        if not recordings:
            raise ValidationError("recordings", 0, "nothing to write")
        length = min(len(r) for r in recordings)
        iq = np.stack(
            [np.stack([r.samples[:length].real, r.samples[:length].imag], axis=1) for r in recordings]
        ).astype(np.float32)
        path = Path(path)
        np.savez_compressed(
            path,
            rows=np.array([r.row for r in recordings], dtype=np.int64),
            iq=iq,
            cls=np.array([layout.class_index(r.modulation) for r in recordings], dtype=np.int32),
            snr=np.array([int(round(r.snr_label_db)) for r in recordings], dtype=np.int32),
            rate=np.array([int(round(r.symbol_rate)) for r in recordings], dtype=np.int32),
            class_names=np.array(layout.CLASS_NAMES),
        )
        return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")


def open_archive(path: str | Path) -> RecordingArchive:
    """Open whichever kind of archive ``path`` is.

    A small factory, so scripts and the backend can take one path from
    the user and not care which format it points at.
    """
    path = Path(path).expanduser()
    if path.suffix.lower() == ".npz":
        return SampleArchive(path)
    return RML24File(path)


__all__ = ["RecordingArchive", "RML24File", "SampleArchive", "open_archive"]
