"""Comparing different AI models on the same job.

The app's classifier is a ResNet with expert features.  Is that the
right choice?  The only honest way to answer is to train other kinds of
model on exactly the same recordings, test them on exactly the same
held-out recordings, and put the results side by side.  This module
holds what that needs:

* :class:`ForestClassifier` -- a Random Forest (no neural network at all)
  that decides from the 12 expert measurements alone;
* :data:`COMPARISON_MODELS` -- the six models compared, in plain words;
* :func:`load_classifier` -- opens any saved model, whatever kind it is;
* :class:`ModelComparison` -- reads the results the comparison script
  wrote, and remembers which model the app should use.

DESIGN NOTE: POLYMORPHISM
-------------------------
:class:`ForestClassifier` is a :class:`ModulationClassifier` -- it
answers ``classify()``, ``predict_proba()``, ``save()`` and ``load()``
just like the neural networks do -- but *how* it decides is completely
different (decision trees voting, not layers of neurons).  The rest of
the app, the :class:`~satrecovery.analysis.analyser.Analyser` included,
never needs to know which kind it was given.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..core.exceptions import ValidationError
from ..learning.base import TrainableModel, TrainingHistory
from .classifier import ModulationClassifier, _ExpertFeatures, pick_device

#: The models compared, in the order the tab shows them, with a short
#: description of each in simple words.
COMPARISON_MODELS: dict[str, dict[str, str]] = {
    "resnet": {
        "label": "ResNet + expert features",
        "about": "The app's model: a deep network with shortcut connections, plus 12 classical measurements.",
    },
    "resnet_raw": {
        "label": "ResNet (no expert features)",
        "about": "The same network learning from the raw signal only. Shows what the expert features add.",
    },
    "cnn": {
        "label": "Plain CNN",
        "about": "A simple stack of convolutions, like the project's first version (no augmentation).",
    },
    "lstm": {
        "label": "LSTM",
        "about": "A network that reads the signal in order and remembers what came before.",
    },
    "transformer": {
        "label": "Transformer",
        "about": "Attention: every part of the signal is compared with every other part.",
    },
    "forest": {
        "label": "Random Forest",
        "about": "No neural network: 300 decision trees vote, using only the 12 expert measurements.",
    },
}


def expert_features(inputs: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """The 12 expert measurements for prepared recordings ``(n, 2, length)``.

    The same measurements the ResNet receives, computed by the same code.
    """
    import torch

    module = _ExpertFeatures()
    rows = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.as_tensor(np.asarray(inputs[start:start + batch_size], dtype=np.float32))
            rows.append(module(batch).numpy())
    out = np.concatenate(rows) if rows else np.empty((0, _ExpertFeatures.COUNT), dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=50.0, neginf=-50.0)


class ForestClassifier(ModulationClassifier):
    """A Random Forest that names the modulation from the expert measurements.

    It *is a* :class:`ModulationClassifier` (same questions, same answers),
    so the app can use it in place of the network.  It does not build a
    network, so it starts from :class:`TrainableModel` directly.
    """

    ARCHITECTURES = ("forest",)

    def __init__(self, classes: tuple[str, ...] | list[str] | None = None, n_trees: int = 300,
                 seed: int = 0, name: str = "ForestClassifier", **_: Any) -> None:
        TrainableModel.__init__(self, name)
        classes = tuple(str(c) for c in (classes or self.DEFAULT_CLASSES))
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValidationError("classes", classes, "need at least two distinct classes")
        if int(n_trees) < 1:
            raise ValidationError("n_trees", n_trees, "need at least one tree")
        self._classes = classes
        self._channels = ()
        self._n_trees = int(n_trees)
        self._seed = int(seed)
        self._device = "cpu"
        self._architecture = "forest"
        self._forest: Any = None
        self._test_accuracy = None
        self._trained_at = None

    @property
    def n_parameters(self) -> int:
        """Decision points in all the trees -- its equivalent of 'weights'."""
        if self._forest is None:
            return 0
        return int(sum(tree.tree_.node_count for tree in self._forest.estimators_))

    def describe(self) -> str:
        state = "trained" if self.is_trained else "untrained"
        accuracy = f", {self._test_accuracy:.1%} on held-out recordings" if self._test_accuracy else ""
        return f"{self._name} (random forest, {self._n_trees} trees, {state}{accuracy})"

    def fit(self, inputs: np.ndarray, targets: np.ndarray, validation: Any = None, verbose: bool = True,
            report: Any = None, **_: Any) -> TrainingHistory:
        from sklearn.ensemble import RandomForestClassifier

        inputs, targets = self._check_pair(inputs, targets)
        targets = np.asarray(targets, dtype=np.int64)
        if targets.min() < 0 or targets.max() >= len(self._classes):
            raise ValidationError("targets", (targets.min(), targets.max()), "class index out of range")
        self._forest = RandomForestClassifier(n_estimators=self._n_trees, min_samples_leaf=2,
                                              n_jobs=-1, random_state=self._seed)
        self._forest.fit(expert_features(inputs), targets)
        self._trained_at = time.time()
        self._mark_trained()
        history = TrainingHistory(self._name)
        metrics = {"loss": 0.0}
        if validation is not None:
            metrics["val_accuracy"] = self.evaluate(validation[0], validation[1])
        history.record(1, **metrics)
        if report is not None:
            report(1, 1, metrics)
        if verbose and "val_accuracy" in metrics:
            print(f"  forest: validation accuracy {metrics['val_accuracy']:.1%}")
        self._history = history
        return history

    def predict_proba(self, samples: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        self._require_trained()
        x = samples if (isinstance(samples, np.ndarray) and samples.ndim == 3 and samples.shape[1] == 2
                        and samples.dtype == np.float32) else self.prepare(samples)
        if len(x) == 0:
            return np.empty((0, len(self._classes)))
        # Many processor cores help a big batch; for one recording, starting them costs more than it saves.
        self._forest.n_jobs = -1 if len(x) > 500 else 1
        found = self._forest.predict_proba(expert_features(x, batch_size))
        # A class absent from the training data has no column; give it 0.
        out = np.zeros((len(x), len(self._classes)))
        out[:, self._forest.classes_.astype(int)] = found
        return out

    def save(self, path: str | Path) -> Path:
        self._require_trained()
        path = Path(path).with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self._forest, handle)
        self._write_sidecar(path, {
            "model": self._name, "kind": "ForestClassifier", "architecture": "forest",
            "classes": list(self._classes), "n_trees": self._n_trees, "n_parameters": self.n_parameters,
            "test_accuracy": self._test_accuracy, "trained_at": self._trained_at,
            "history": self._history.to_dict() if self._history is not None else {},
        })
        return path

    def load(self, path: str | Path) -> "ForestClassifier":
        path = Path(path).with_suffix(".pt")
        if not path.exists():
            raise ValidationError("path", str(path), "no saved classifier here")
        meta = self._read_sidecar(path)
        if meta.get("classes") and tuple(meta["classes"]) != self._classes:
            raise ValidationError("path", str(path), f"saved for classes {meta['classes']}")
        with open(path, "rb") as handle:
            self._forest = pickle.load(handle)  # noqa: S301 - a file this app wrote itself
        self._test_accuracy = meta.get("test_accuracy")
        self._trained_at = meta.get("trained_at")
        self._mark_trained()
        return self

    @classmethod
    def from_file(cls, path: str | Path, device: Any = None) -> "ForestClassifier":
        path = Path(path).with_suffix(".pt")
        meta = cls._read_sidecar(path) if path.exists() else {}
        return cls(classes=meta.get("classes"), n_trees=meta.get("n_trees", 300)).load(path)


def build_model(name: str, classes: Any = None, seed: int = 0, device: Any = None) -> ModulationClassifier:
    """A fresh, untrained model of one of the compared kinds."""
    if name not in COMPARISON_MODELS:
        raise ValidationError("model", name, f"must be one of {tuple(COMPARISON_MODELS)}")
    if name == "forest":
        return ForestClassifier(classes=classes, seed=seed)
    return ModulationClassifier(classes=classes, seed=seed, device=device or pick_device(), architecture=name,
                                name=COMPARISON_MODELS[name]["label"])


def load_classifier(path: str | Path, device: Any = None) -> ModulationClassifier:
    """Open a saved model, whichever kind it is (network or forest)."""
    path = Path(path).with_suffix(".pt")
    meta = TrainableModel._read_sidecar(path) if path.exists() else {}
    if meta.get("kind") == "ForestClassifier":
        return ForestClassifier.from_file(path)
    return ModulationClassifier.from_file(path, device=device)


def time_per_recording(model: ModulationClassifier, inputs: np.ndarray, repeats: int = 3) -> float:
    """Milliseconds to classify one recording, one at a time (as the app does)."""
    sample = inputs[: min(20, len(inputs))]
    model.predict_proba(sample[:1])           # warm up
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        for one in sample:
            model.predict_proba(one[None])
        best = min(best, (time.perf_counter() - start) / len(sample))
    return best * 1000.0


class ModelComparison:
    """The comparison results on disk, and which model the app is using.

    Everything lives in one folder (``satrecovery_data/models/compare``):
    one saved model per kind, ``results.json`` from the comparison
    script, and ``active.json`` naming the model the app should use.
    """

    RESULTS = "results.json"
    ACTIVE = "active.json"
    #: The name for the app's own model (``classifier.pt``), always offered.
    MAIN = "main"

    def __init__(self, folder: str | Path, main_path: str | Path) -> None:
        self._folder = Path(folder)
        self._main = Path(main_path).with_suffix(".pt")

    @property
    def folder(self) -> Path:
        return self._folder

    def results(self) -> dict[str, Any]:
        path = self._folder / self.RESULTS
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def path_for(self, name: str) -> Path:
        if name == self.MAIN:
            return self._main
        if name not in COMPARISON_MODELS:
            raise ValidationError("model", name, "not one of the compared models")
        return self._folder / f"{name}.pt"

    def active(self) -> str:
        try:
            name = json.loads((self._folder / self.ACTIVE).read_text(encoding="utf-8")).get("model", self.MAIN)
        except (OSError, ValueError, AttributeError):
            return self.MAIN
        if name != self.MAIN and (name not in COMPARISON_MODELS or not self.path_for(name).exists()):
            return self.MAIN
        return name

    def set_active(self, name: str) -> Path:
        path = self.path_for(name)
        if not path.exists():
            raise ValidationError("model", name, "has not been trained yet")
        self._folder.mkdir(parents=True, exist_ok=True)
        (self._folder / self.ACTIVE).write_text(json.dumps({"model": name}), encoding="utf-8")
        return path

    def listing(self) -> dict[str, Any]:
        """Everything the "Compare models" tab shows."""
        results = self.results()
        found = results.get("models", {})
        active = self.active()
        models = [{
            "name": self.MAIN, "label": "App's main model (classifier.pt)",
            "about": "The model trained with train_classifier.py, used unless you pick another.",
            "available": self._main.exists(), "active": active == self.MAIN, "results": None,
        }]
        for name, info in COMPARISON_MODELS.items():
            models.append({"name": name, "label": info["label"], "about": info["about"],
                           "available": self.path_for(name).exists(), "active": active == name,
                           "results": found.get(name)})
        return {"active": active, "models": models, "setup": results.get("setup"),
                "snr_labels_db": results.get("snr_labels_db", []), "folder": str(self._folder),
                "has_results": bool(found)}


__all__ = ["COMPARISON_MODELS", "ForestClassifier", "ModelComparison", "build_model", "expert_features",
           "load_classifier", "time_per_recording"]
