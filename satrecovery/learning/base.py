"""The contract every learned component signs.

Phase 2's receiver was built from mathematics: it knew the shape of the
problem in advance and applied a formula.  Phase 4's components know
nothing in advance and have to be shown examples first.  That difference
needs exactly one new idea in the design -- a model has a *lifecycle*.
It starts untrained, becomes trained, and can be saved and restored.

:class:`TrainableModel` captures that lifecycle and nothing else.  What
the model actually does is left to its subclasses, because a modulation
classifier and a neural demapper have almost nothing else in common.

DESIGN NOTE: MULTIPLE INHERITANCE, USED FOR ONE PRECISE REASON
---------------------------------------------------------------
The most important class in this phase is
:class:`~satrecovery.learning.neural.mlp.MLPDemapper`, and it inherits
from **two** bases::

    class MLPDemapper(Demapper, TrainableModel)

That is not cleverness for its own sake.  It says the thing we need:
a learned demapper *is a demapper* -- it satisfies the interface Phase 2
already defined -- and it is *also* something that must be trained.
Because the first half holds, it drops straight into the existing
receiver with no adapter and no special case::

    ClassicalReceiver(demapper=MLPDemapper(...))

That one line is the hybrid receiver of Phase 5, and it exists purely
because Phase 2 defined ``Demapper`` as an interface rather than writing
the nearest-neighbour rule inline.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ..core.exceptions import ModelNotTrainedError, ValidationError


class TrainingHistory:
    """What happened during training, recorded epoch by epoch.

    Kept as a plain object rather than a dictionary so that the GUI and
    the report can rely on the fields existing, and so a typo in a key
    name fails loudly instead of silently plotting nothing.
    """

    __slots__ = ("_epochs", "_records", "_name")

    def __init__(self, name: str = "") -> None:
        self._name = str(name)
        self._epochs: list[int] = []
        self._records: list[dict[str, float]] = []

    def record(self, epoch: int, **metrics: float) -> None:
        """Note one epoch's results."""
        self._epochs.append(int(epoch))
        self._records.append({k: float(v) for k, v in metrics.items()})

    @property
    def name(self) -> str:
        """Name of the model this history belongs to."""
        return self._name

    @property
    def epochs(self) -> list[int]:
        """Epoch numbers, in order."""
        return list(self._epochs)

    @property
    def n_epochs(self) -> int:
        """How many epochs were recorded."""
        return len(self._epochs)

    def series(self, key: str) -> list[float]:
        """One metric across every epoch, for plotting a training curve."""
        return [record[key] for record in self._records if key in record]

    def keys(self) -> list[str]:
        """Every metric name that was recorded at least once."""
        seen: list[str] = []
        for record in self._records:
            for key in record:
                if key not in seen:
                    seen.append(key)
        return seen

    def last(self) -> dict[str, float]:
        """The final epoch's metrics."""
        return dict(self._records[-1]) if self._records else {}

    def best(self, key: str, lowest: bool = True) -> float | None:
        """Best value a metric reached."""
        values = self.series(key)
        if not values:
            return None
        return min(values) if lowest else max(values)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, for saving alongside a model."""
        return {
            "name": self._name,
            "epochs": self._epochs,
            "records": self._records,
        }

    def __len__(self) -> int:
        return len(self._epochs)

    def __repr__(self) -> str:
        if not self._records:
            return f"TrainingHistory({self._name!r}, not yet trained)"
        final = ", ".join(f"{k}={v:.4g}" for k, v in self.last().items())
        return f"TrainingHistory({self._name!r}, {self.n_epochs} epochs, {final})"


class TrainableModel(ABC):
    """Anything that must be shown examples before it is useful.

    Subclasses implement :meth:`fit` and the two halves of persistence.
    The base class owns the state flag and the guard that stops an
    untrained model being used by accident -- a mistake that otherwise
    produces confident nonsense rather than an error.
    """

    def __init__(self, name: str | None = None) -> None:
        self._name = str(name) if name else type(self).__name__
        self._trained = False
        self._history = TrainingHistory(self._name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Short name, used to label results and saved files."""
        return self._name

    @property
    def is_trained(self) -> bool:
        """Has this model been fitted or loaded?"""
        return self._trained

    @property
    def history(self) -> TrainingHistory:
        """What happened during training."""
        return self._history

    def _mark_trained(self) -> None:
        """Called by subclasses once fitting has actually succeeded."""
        self._trained = True

    def _require_trained(self) -> None:
        """Raise unless the model is ready to be used.

        Without this, an untrained network returns its random
        initialisation -- which looks exactly like a trained model
        performing badly. Failing loudly turns a confusing result into
        an obvious one.
        """
        if not self._trained:
            raise ModelNotTrainedError(self._name)

    # ------------------------------------------------------------------
    # What subclasses must provide
    # ------------------------------------------------------------------
    @abstractmethod
    def fit(self, inputs: np.ndarray, targets: np.ndarray, **options: Any) -> TrainingHistory:
        """Learn from examples. Must call :meth:`_mark_trained` on success."""

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """Write the trained model to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> "TrainableModel":
        """Restore a model from disk. Must call :meth:`_mark_trained`."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def describe(self) -> str:
        """One-line summary for logs and the models screen."""
        state = "trained" if self._trained else "untrained"
        return f"{self._name} ({state})"

    @staticmethod
    def _check_pair(inputs: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Validate that inputs and targets line up before training starts."""
        inputs = np.asarray(inputs)
        targets = np.asarray(targets)
        if inputs.ndim < 2:
            raise ValidationError(
                "inputs", inputs.shape, "expected at least two dimensions (samples, features)"
            )
        if len(inputs) != len(targets):
            raise ValidationError(
                "targets",
                len(targets),
                f"must have one row per input ({len(inputs)})",
            )
        if len(inputs) == 0:
            raise ValidationError("inputs", 0, "cannot train on an empty set")
        return inputs, targets

    @staticmethod
    def _write_sidecar(path: Path, metadata: dict[str, Any]) -> None:
        """Save a small JSON file beside the model weights.

        Weights alone are not enough to use a model: you also need to
        know what it was trained on and what shape it expects. Keeping
        that next to the file means a checkpoint found six months later
        is still usable.
        """
        path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

    @staticmethod
    def _read_sidecar(path: Path) -> dict[str, Any]:
        """Read the metadata written by :meth:`_write_sidecar`."""
        sidecar = path.with_suffix(".json")
        if not sidecar.exists():
            return {}
        return json.loads(sidecar.read_text(encoding="utf-8"))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self._name!r}, trained={self._trained})"


__all__ = ["TrainableModel", "TrainingHistory"]
