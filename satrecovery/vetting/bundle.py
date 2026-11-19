"""The trained vetter in one place, so the app no longer needs the training data.

Training needs thousands of volunteer-checked waterfalls (hundreds of MB).
Using the result needs only a few things, all saved together here:

* the forest itself;
* the explainer's typical values (for the reasons in words);
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

    FILES = ("vetter_bundle.pt", "vetter_bundle.npz", "vetter_bundle.json")

    # ------------------------------------------------------------------
    @classmethod
    def train(cls, observations: Iterable[Observation], cache: ArtifactCache, progress=None) -> "VetterBundle":
        labelled = [o for o in observations if o.is_trainable]
        kept, features = FeatureStore(cache).get(labelled, progress)
        if len(kept) < 50:
            raise ValidationError("observations", len(kept), "need at least 50 labelled, downloaded waterfalls")
        targets = np.array([1 if o.waterfall_label is WaterfallLabel.SIGNAL else 0 for o in kept])
        probability = out_of_fold(features, targets, [o.ground_station for o in kept])
        model = FeatureVetter()
        model.fit_features(features, targets)
        explainer = Explainer(features, targets, model.importances())
        showcase = tuple(o for o in kept if o.id in SHOWCASE)
        temperature = fit_temperature(probability, targets)
        stations = np.array([o.ground_station if o.ground_station is not None else -1 for o in kept])
        return cls(model, explainer, probability, targets, time.time(), showcase, temperature, stations)

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
        meta = json.loads((folder / cls.FILES[2]).read_text(encoding="utf-8"))
        showcase = tuple(Observation.from_dict(d) for d in meta.get("showcase", []))
        return cls(model, explainer, probability, target, float(meta.get("trained_at", 0)), showcase,
                   float(meta.get("temperature", 1.0)), stations)
