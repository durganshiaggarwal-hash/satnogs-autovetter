"""Trained demappers on disk: training them, saving them, loading them.

The hybrid receiver needs one trained network per modulation.  Training
takes tens of seconds; loading takes milliseconds.  So the backend
trains a model once -- as a background job, on request -- saves it, and
from then on loads it whenever a hybrid receiver is asked for.

Each modulation's model is three files in the models folder:

``16QAM.pt``
    The network's weights.
``16QAM.json``
    What the network is: its window, layer sizes and training history,
    written by :meth:`MLPDemapper.save` itself.
``16QAM.reference.json``
    The typical per-frame noise power on the training data, which the
    noise-normalised hybrid from Phase 6 needs, and when and how the
    model was trained.

DESIGN NOTE: ENCAPSULATING A RESOURCE THAT OTHER THREADS SHARE
---------------------------------------------------------------
Training runs in a background thread while request handlers load
models in others.  :class:`ModelStore` owns the folder and an in-memory
cache, and every access goes through a lock it holds privately.  No
other code touches the files, so no other code can get the locking
wrong.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..core.exceptions import ModelNotTrainedError, ValidationError
from ..modulation.registry import ModulationRegistry

#: A progress callback: ``report(done, total, message)``.
Progress = Callable[[int, int, str], None]


class TrainingSpec:
    """What to train a hybrid demapper on, and for how long.

    The defaults are the settings Phase 6's fairness check arrived at:
    enough data that the network matches the classical receiver rather
    than losing to it for want of examples.
    """

    __slots__ = (
        "snrs_db", "profiles", "frames_each", "epochs", "patience",
        "window", "hidden", "seed",
    )

    def __init__(
        self,
        snrs_db: list[float] | None = None,
        profiles: list[str] | None = None,
        frames_each: int = 18,
        epochs: int = 200,
        patience: int = 12,
        window: int = 2,
        hidden: tuple[int, ...] = (128, 128),
        seed: int = 2,
    ) -> None:
        self.snrs_db = [float(s) for s in (snrs_db or [0, 4, 8, 12, 16])]
        self.profiles = list(profiles or ["clean", "doppler", "fading", "full"])
        self.frames_each = int(frames_each)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.window = int(window)
        self.hidden = tuple(int(h) for h in hidden)
        self.seed = int(seed)
        for name in ("frames_each", "epochs", "patience"):
            if getattr(self, name) < 1:
                raise ValidationError(name, getattr(self, name), "must be at least 1")
        if not self.snrs_db or not self.profiles:
            raise ValidationError("snrs_db", self.snrs_db, "need at least one SNR and profile")

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


class ModelStore:
    """The folder of trained hybrid demappers, with a cache in front of it."""

    def __init__(self, folder: str | Path) -> None:
        self._folder = Path(folder)
        self._folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[Any, float]] = {}

    @property
    def folder(self) -> Path:
        return self._folder

    @staticmethod
    def _name(modulation: str) -> str:
        return ModulationRegistry.create(modulation).name

    def _paths(self, name: str) -> tuple[Path, Path, Path]:
        base = self._folder / name
        return (
            base.with_suffix(".pt"),
            base.with_suffix(".json"),
            self._folder / f"{name}.reference.json",
        )

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def has(self, modulation: str) -> bool:
        """Is a complete trained model on disk for this modulation?"""
        name = self._name(modulation)
        return all(path.exists() for path in self._paths(name))

    def trained(self) -> list[dict[str, Any]]:
        """Every modulation with a model, and what it was trained on."""
        out = []
        for scheme in ModulationRegistry.available():
            if self.has(scheme):
                _, _, reference = self._paths(self._name(scheme))
                record = json.loads(reference.read_text(encoding="utf-8"))
                record["modulation"] = self._name(scheme)
                out.append(record)
        return out

    def load(self, modulation: str) -> tuple[Any, float]:
        """The trained demapper and its noise reference, from cache or disk."""
        from ..learning.neural import MLPDemapper

        name = self._name(modulation)
        with self._lock:
            if name in self._cache:
                return self._cache[name]
            if not self.has(name):
                raise ModelNotTrainedError(f"hybrid demapper for {name}")

            weights, sidecar, reference_file = self._paths(name)
            shape = json.loads(sidecar.read_text(encoding="utf-8"))
            reference = json.loads(reference_file.read_text(encoding="utf-8"))
            model = MLPDemapper(
                ModulationRegistry.create(name),
                window=int(shape["window"]),
                hidden=tuple(shape["hidden"]),
                dropout=float(shape.get("dropout", 0.0)),
                device="cpu",
            )
            model.load(weights)
            entry = (model, float(reference["noise_reference"]))
            self._cache[name] = entry
            return entry

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def train(
        self,
        modulation: str,
        spec: TrainingSpec | None = None,
        report: Progress | None = None,
    ) -> dict[str, Any]:
        """Train, save and cache a hybrid demapper. Returns its record.

        Runs for tens of seconds with the default spec, so the API calls
        it from a background job and passes ``report`` to show progress.
        """
        from ..evaluation.scenarios import ImpairmentProfile, ScenarioGrid
        from ..evaluation.sources import SimulatedSource
        from ..learning.datasets import DemappingDataset, split_rows
        from ..learning.neural import MLPDemapper
        from ..receivers.classical import ClassicalReceiver
        from ..receivers.demapper import NoiseNormalisedDemapper, SoftDemapper

        spec = spec or TrainingSpec()
        name = self._name(modulation)
        report = report or (lambda done, total, message: None)
        scheme = ModulationRegistry.create(name)

        profiles = {p.name: p for p in ImpairmentProfile.standard()}
        unknown = [p for p in spec.profiles if p not in profiles]
        if unknown:
            raise ValidationError("profiles", unknown, f"known profiles: {sorted(profiles)}")
        grid = ScenarioGrid.product([name], spec.snrs_db, [profiles[p] for p in spec.profiles])

        # Training frames come from their own seed, never shared with the
        # frames a benchmark grades on.
        source = SimulatedSource(seed=spec.seed)
        classical = ClassicalReceiver(pulse=source.simulator.pulse, demapper=SoftDemapper())

        started = time.time()
        frames, symbol_frames = [], []
        for index, scenario in enumerate(grid):
            for frame in source.frames(scenario, spec.frames_each):
                frames.append(frame)
            report(index + 1, len(grid) + 2, f"generated frames for {scenario!r}")

        inputs, targets = DemappingDataset(classical, window=spec.window).build(frames)
        symbol_frames = [classical.recover(f).symbols for f in frames[:: max(1, spec.frames_each // 4)]]
        reference = NoiseNormalisedDemapper.fit_reference(symbol_frames, scheme)
        report(len(grid) + 1, len(grid) + 2, f"training on {len(inputs)} symbols")

        x_train, y_train, x_val, y_val = split_rows(inputs, targets, 0.2, seed=0)
        model = MLPDemapper(scheme, window=spec.window, hidden=spec.hidden, seed=0, device="cpu")
        model.fit(
            x_train, y_train, validation=(x_val, y_val),
            epochs=spec.epochs, patience=spec.patience, verbose=False,
        )

        record = {
            "modulation": name,
            "trained_at": time.time(),
            "seconds": round(time.time() - started, 1),
            "rows": int(len(inputs)),
            "epochs_run": model.history.n_epochs,
            "best_val_loss": model.history.best("val_loss"),
            "noise_reference": reference,
            "spec": spec.as_dict(),
        }
        with self._lock:
            weights, _, reference_file = self._paths(name)
            model.save(weights)
            reference_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
            self._cache[name] = (model, reference)
        report(len(grid) + 2, len(grid) + 2, "saved")
        return record

    def delete(self, modulation: str) -> bool:
        """Remove a model. Returns whether there was one."""
        name = self._name(modulation)
        with self._lock:
            self._cache.pop(name, None)
            removed = False
            for path in self._paths(name):
                if path.exists():
                    path.unlink()
                    removed = True
            return removed

    def __repr__(self) -> str:
        return f"ModelStore({str(self._folder)!r}, {len(self.trained())} trained)"


__all__ = ["ModelStore", "TrainingSpec"]
