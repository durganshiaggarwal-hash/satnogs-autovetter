"""The trained vetter in one place, so the app no longer needs the training data.

Training needs thousands of volunteer-checked waterfalls (hundreds of MB).
Using the result needs only a few things, all saved together here:

* the forest itself, and the CNN that reads the waterfall picture (their
  probabilities are averaged: the six-model comparison found the pair
  better than either alone, on every seed);
* the explainer's typical values (for the reasons in words; these come
  from the forest's measurements);
* the **out-of-fold** scores of every training observation -- each from a
  forest that never saw that observation's station -- with the
  volunteers' answers, so "error when sure" and "share it can check
  alone" can be worked out for *any* thresholds without the data;
* a few showcase observations (their metadata), so the demo has known
  examples even with no internet.

Build it with ``python scripts/build_vetter_bundle.py``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..core.exceptions import ValidationError
from .artifacts import ArtifactCache
from .model import FeatureVetter
from .observation import Observation, WaterfallLabel
from .session import Explainer, FeatureStore, Policy, Verdict, out_of_fold

def _confidence(probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probability, dtype=float)
    return np.maximum(p, 1 - p), p >= 0.5


def fit_temperature(probability: np.ndarray, target: np.ndarray) -> float:
    """Fit the original project's TemperatureScaling on out-of-fold decisions."""
    from ..confidence.calibration import TemperatureScaling

    confidence, says_signal = _confidence(probability)
    correct = says_signal == (np.asarray(target) == 1)
    return TemperatureScaling().fit(confidence, correct).temperature


def apply_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    from ..confidence.calibration import TemperatureScaling

    confidence, says_signal = _confidence(probability)
    if confidence.size == 0:
        return np.asarray(probability, dtype=float)
    calibrated = TemperatureScaling.with_temperature(temperature).apply(np.clip(confidence, 0.5, 1.0))
    return np.where(says_signal, calibrated, 1 - calibrated)


CNN_FILE = "vetter_cnn.pt"


def waterfall_images(observations, cache: ArtifactCache) -> np.ndarray:
    """Normalised waterfalls at the CNN's 128 x 128 (2 x 2 averaging of the stored 256 x 256)."""
    from .dataset import normalise

    stack = []
    for obs in observations:
        power = normalise(cache.waterfall(obs).power)
        height, width = power.shape
        stack.append(power[:height // 2 * 2, :width // 2 * 2]
                     .reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))[None])
    return np.stack(stack).astype(np.float32) if stack else np.empty((0, 1, 128, 128), np.float32)


def train_cnn(images: np.ndarray, targets: np.ndarray, groups: np.ndarray, epochs: int = 30, seed: int = 0):
    """A WaterfallVetter, early-stopped on a slice of stations it does not train on."""
    from sklearn.model_selection import GroupShuffleSplit

    from .model import WaterfallVetter

    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
                    .split(images, targets, groups))
    model = WaterfallVetter(seed=seed)
    model.fit(images[fit], targets[fit], validation=(images[val], targets[val]), epochs=epochs,
              verbose=False, patience=6)
    return model


def cnn_out_of_fold(images: np.ndarray, targets: np.ndarray, groups: np.ndarray, folds: int = 5,
                    epochs: int = 30, seed: int = 0, progress=None) -> np.ndarray:
    """P(signal) from a CNN that never saw that recording's station, for every recording."""
    from sklearn.model_selection import GroupKFold

    out = np.full(len(targets), 0.5)
    for k, (train, test) in enumerate(GroupKFold(folds).split(images, targets, groups), 1):
        out[test] = train_cnn(images[train], targets[train], groups[train], epochs, seed).predict_proba(images[test])
        if progress:
            progress(k, folds)
    return out


#: Observations worth showing in a demo (each explained by the orbit physics).
SHOWCASE = (15489000, 15490378)


@dataclass
class VetterBundle:
    model: FeatureVetter
    explainer: Explainer
    oof_probability: np.ndarray       # out of fold, one per training observation
    oof_target: np.ndarray            # the volunteers' answers (1 = signal)
    trained_at: float
    showcase: tuple[Observation, ...] = ()
    temperature: float = 1.0          # TemperatureScaling (the original project's), fitted on out-of-fold
    oof_station: np.ndarray | None = None
    cnn: object | None = None         # a WaterfallVetter, or None: forest alone
    oof_forest: np.ndarray | None = None   # the forest's own out-of-fold scores, for comparison

    FILES = ("vetter_bundle.pt", "vetter_bundle.npz", "vetter_bundle.json")

    # ------------------------------------------------------------------
    @classmethod
    def train(cls, observations: Iterable[Observation], cache: ArtifactCache, progress=None,
              cnn: bool = True, epochs: int = 30, cnn_progress=None) -> "VetterBundle":
        labelled = [o for o in observations if o.is_trainable]
        kept, features = FeatureStore(cache).get(labelled, progress)
        if len(kept) < 50:
            raise ValidationError("observations", len(kept), "need at least 50 labelled, downloaded waterfalls")
        targets = np.array([1 if o.waterfall_label is WaterfallLabel.SIGNAL else 0 for o in kept])
        stations = np.array([o.ground_station if o.ground_station is not None else -1 for o in kept])
        forest_oof = out_of_fold(features, targets, stations)
        model = FeatureVetter()
        model.fit_features(features, targets)
        explainer = Explainer(features, targets, model.importances())
        showcase = tuple(o for o in kept if o.id in SHOWCASE)
        network, probability = None, forest_oof
        if cnn:
            images = waterfall_images(kept, cache)
            cnn_oof = cnn_out_of_fold(images, targets, stations, epochs=epochs, progress=cnn_progress)
            probability = (forest_oof + cnn_oof) / 2.0
            network = train_cnn(images, targets, stations, epochs)
        temperature = fit_temperature(probability, targets)
        return cls(model, explainer, probability, targets, time.time(), showcase, temperature, stations,
                   network, forest_oof)

    # ------------------------------------------------------------------
    def predict(self, features: np.ndarray, images: np.ndarray | None = None) -> np.ndarray:
        """Calibrated P(signal): the forest and (if present) the CNN averaged, then temperature-scaled."""
        probability = self.model.predict_proba_features(features)
        if self.cnn is not None and images is not None and len(images):
            probability = (probability + self.cnn.predict_proba(images)) / 2.0
        return self.calibrate(probability)

    @property
    def description(self) -> str:
        what = "Random forest + CNN" if self.cnn is not None else "Random forest"
        return f"{what}, trained on {self.n_train:,} volunteer-checked observations"

    # ------------------------------------------------------------------
    def calibrate(self, probability: np.ndarray) -> np.ndarray:
        """P(signal) made honest: "90%" right about 9 times in 10 (no decision changes)."""
        return apply_temperature(probability, self.temperature)

    @property
    def calibrated_oof(self) -> np.ndarray:
        return self.calibrate(self.oof_probability)

    # ------------------------------------------------------------------
    def measured_error(self, policy: Policy) -> tuple[float, int]:
        """``(share wrong, how many)`` of confident out-of-fold answers at these thresholds."""
        verdicts = [policy.verdict(float(p)) for p in self.calibrated_oof]
        sure = [(v, y) for v, y in zip(verdicts, self.oof_target) if v is not Verdict.UNSURE]
        wrong = sum((v is Verdict.SIGNAL) != bool(y) for v, y in sure)
        return (wrong / len(sure) if sure else float("nan")), len(sure)

    def automatable(self, policy: Policy) -> float:
        """Share of out-of-fold observations the vetter is sure enough to decide alone."""
        return float(np.mean([policy.verdict(float(p)) is not Verdict.UNSURE for p in self.calibrated_oof]))

    @property
    def n_train(self) -> int:
        return int(len(self.oof_target))

    # ------------------------------------------------------------------
    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save(folder / self.FILES[0])
        extra = {"oof_station": self.oof_station} if self.oof_station is not None else {}
        if self.oof_forest is not None:
            extra["oof_forest"] = self.oof_forest
        if self.cnn is not None:
            self.cnn.save(folder / CNN_FILE)
        elif (folder / CNN_FILE).exists():
            (folder / CNN_FILE).unlink()   # a forest-only bundle must not pick up an old CNN
        np.savez(folder / self.FILES[1], oof_probability=self.oof_probability, oof_target=self.oof_target,
                 **extra, **self.explainer.to_arrays())
        (folder / self.FILES[2]).write_text(json.dumps({
            "trained_at": self.trained_at, "n_train": self.n_train, "temperature": self.temperature,
            "showcase": [o.to_dict() for o in self.showcase],
        }, indent=1), encoding="utf-8")
        return folder

    @classmethod
    def exists(cls, folder: str | Path) -> bool:
        return all((Path(folder) / name).exists() for name in cls.FILES)

    @classmethod
    def load(cls, folder: str | Path) -> "VetterBundle":
        folder = Path(folder)
        if not cls.exists(folder):
            raise ValidationError("bundle", str(folder), "no trained vetter here; run scripts/build_vetter_bundle.py")
        model = FeatureVetter.from_file(folder / cls.FILES[0])
        with np.load(folder / cls.FILES[1]) as arrays:
            explainer = Explainer.from_arrays(arrays)
            probability, target = arrays["oof_probability"].copy(), arrays["oof_target"].copy()
            stations = arrays["oof_station"].copy() if "oof_station" in arrays.files else None
            forest_oof = arrays["oof_forest"].copy() if "oof_forest" in arrays.files else None
        meta = json.loads((folder / cls.FILES[2]).read_text(encoding="utf-8"))
        showcase = tuple(Observation.from_dict(d) for d in meta.get("showcase", []))
        network = None
        if (folder / CNN_FILE).exists():
            try:
                from .model import WaterfallVetter

                network = WaterfallVetter.from_file(folder / CNN_FILE)
            except Exception:  # noqa: BLE001 - no PyTorch, or a damaged file: run on the forest alone
                network = None
        return cls(model, explainer, probability, target, float(meta.get("trained_at", 0)), showcase,
                   float(meta.get("temperature", 1.0)), stations, network, forest_oof)
