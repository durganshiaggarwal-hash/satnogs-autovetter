"""Training and test sets for the classifier, drawn from RML24.

RML24 was built for exactly this task -- naming the modulation of a
recording -- so every recording comes labelled.  For each of the six
schemes, at every SNR label and every symbol rate, this takes the first
``train_per_cell`` recordings for training and the *last*
``test_per_cell`` for testing.  The two never overlap, so the test
accuracy is measured on recordings the network has never seen, at every
noise level, in the proportions the dataset holds them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.exceptions import ValidationError
from ..data.archive import RecordingArchive
from .classifier import ModulationClassifier


@dataclass(frozen=True)
class LabelledSet:
    """Prepared recordings with their class index and SNR label."""

    inputs: np.ndarray
    targets: np.ndarray
    snr_db: np.ndarray
    rows: np.ndarray

    def __len__(self) -> int:
        return int(self.targets.size)

    def subset(self, count: int, seed: int = 0) -> "LabelledSet":
        """A random subset, e.g. a small validation set for training."""
        count = min(int(count), len(self))
        pick = np.sort(np.random.default_rng(seed).choice(len(self), size=count, replace=False))
        return LabelledSet(self.inputs[pick], self.targets[pick], self.snr_db[pick], self.rows[pick])


def split_rows(
    archive: RecordingArchive,
    classes: tuple[str, ...],
    train_per_cell: int,
    test_per_cell: int,
    snr_labels_db: tuple[float, ...] | None = None,
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """``(row, class index, snr)`` for training and for testing."""
    if train_per_cell < 1 or test_per_cell < 1:
        raise ValidationError("per_cell", (train_per_cell, test_per_cell), "need at least one of each")
    train, test = [], []
    summary = archive.summary()
    for index, name in enumerate(classes):
        if name not in summary:
            raise ValidationError("classes", name, f"not in {archive.name}")
        labels = summary[name]["snr_labels_db"] if snr_labels_db is None else snr_labels_db
        for snr in labels:
            for rate in summary[name]["symbol_rates"]:
                rows = archive.select(name, snr, rate)
                if rows.size < 2:
                    continue
                # First rows for training, last rows for testing; if the
                # cell is small (a sample archive), split it in half.
                n_test = min(test_per_cell, rows.size // 2)
                n_train = min(train_per_cell, rows.size - n_test)
                train += [(int(r), index, float(snr)) for r in rows[:n_train]]
                test += [(int(r), index, float(snr)) for r in rows[rows.size - n_test:]]
    return train, test


def load_set(archive: RecordingArchive, picks: list[tuple[int, int, float]], report=None) -> LabelledSet:
    """Read and prepare the chosen recordings.

    From the full HDF5 file, rows are read in contiguous runs -- one read
    per run of neighbouring rows instead of one per recording, which is
    many times faster on a 20 GB file.  Any other archive is read one
    recording at a time.
    """
    if not picks:
        raise ValidationError("picks", 0, "nothing to load")
    rows = np.array([p[0] for p in picks], dtype=np.int64)
    targets = np.array([p[1] for p in picks], dtype=np.int64)
    snrs = np.array([p[2] for p in picks], dtype=np.float32)

    raw = _read_runs(archive, rows, report)
    if raw is None:
        length = len(archive.recording(int(rows[0])))
        inputs = np.empty((len(rows), 2, length), dtype=np.float32)
        for i, row in enumerate(rows):
            inputs[i] = ModulationClassifier.prepare(archive.recording(int(row)).samples[:length])[0]
            if report is not None and (i + 1) % 2000 == 0:
                report(i + 1, len(rows))
    else:
        inputs = raw
    return LabelledSet(inputs, targets, snrs, rows)


def _read_runs(archive: RecordingArchive, rows: np.ndarray, report=None) -> np.ndarray | None:
    """Prepared recordings read straight from the HDF5 file, or ``None``."""
    from ..data.archive import RML24File

    if not isinstance(archive, RML24File):
        return None
    import h5py

    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    breaks = np.flatnonzero(np.diff(sorted_rows) != 1) + 1
    runs = np.split(sorted_rows, breaks)
    out = None
    done = 0
    with h5py.File(archive.path, "r") as handle:
        data = handle["IQ_data"]
        for run in runs:
            block = data[int(run[0]) : int(run[-1]) + 1]
            prepared = ModulationClassifier.prepare(block)
            if out is None:
                out = np.empty((len(rows), 2, prepared.shape[2]), dtype=np.float32)
            out[order[done : done + len(run)]] = prepared
            done += len(run)
            if report is not None and done // 5000 != (done - len(run)) // 5000:
                report(done, len(rows))
    return out


def accuracy_by_snr(model: ModulationClassifier, data: LabelledSet) -> dict[float, float]:
    """Held-out accuracy at each SNR label -- the standard way to report it."""
    predicted = np.argmax(model.predict_proba(data.inputs), axis=1)
    return {
        float(snr): float(np.mean(predicted[data.snr_db == snr] == data.targets[data.snr_db == snr]))
        for snr in np.unique(data.snr_db)
    }


def confusion(model: ModulationClassifier, data: LabelledSet, min_snr_db: float | None = None) -> np.ndarray:
    """Counts: row = true class, column = predicted class."""
    keep = np.ones(len(data), dtype=bool) if min_snr_db is None else data.snr_db >= min_snr_db
    predicted = np.argmax(model.predict_proba(data.inputs[keep]), axis=1)
    n = len(model.classes)
    matrix = np.zeros((n, n), dtype=np.int64)
    np.add.at(matrix, (data.targets[keep], predicted), 1)
    return matrix


__all__ = ["LabelledSet", "accuracy_by_snr", "confusion", "load_set", "split_rows"]
