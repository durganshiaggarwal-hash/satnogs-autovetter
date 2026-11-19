"""The baseline auto-vetters: a CNN on the picture, and a forest on measurements.

These are *baselines* on purpose: they see only the waterfall, as earlier
public work does, so later steps (audio, the orbit check, calibrated
abstention) can be measured against them.

DESIGN NOTE: INHERITANCE + POLYMORPHISM
---------------------------------------
:class:`Vetter` is the interface: "given waterfalls, say how likely each
holds the satellite's signal".  It is itself a
:class:`~satrecovery.learning.TrainableModel`, so every vetter inherits
the "refuse to answer until trained" guard and the JSON sidecar.

* :class:`WaterfallVetter` -- a small 2-D ResNet, trained with the
  Phase 4 :class:`~satrecovery.learning.trainer.Trainer` unchanged.
* :class:`FeatureVetter` -- 300 decision trees voting on the twelve
  measurements in :mod:`~satrecovery.vetting.features`; it can say *why*.

The training script, the evaluation and (later) the app take any
``Vetter`` and never ask which kind.
"""

from __future__ import annotations

import pickle
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ..core.exceptions import ValidationError
from ..learning.base import TrainableModel, TrainingHistory
from ..learning.trainer import EarlyStopping, ProgressPrinter, Trainer
from .features import FEATURES, feature_matrix


class Vetter(TrainableModel):
    """Anything that scores waterfalls ``(n, 1, time, freq)`` for "signal present"."""

    @abstractmethod
    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        """P(signal) for each waterfall, shape ``(n,)``."""

    def decide(self, inputs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """1 (signal) or 0 (no signal) for each waterfall."""
        return (self.predict_proba(inputs) >= threshold).astype(np.int64)

    @staticmethod
    def _check_binary(inputs: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inputs, targets = TrainableModel._check_pair(inputs, targets)
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValidationError("inputs", inputs.shape, "expected (n, 1, time, frequency)")
        if not set(np.unique(targets).tolist()) <= {0, 1}:
            raise ValidationError("targets", np.unique(targets)[:5], "labels must be 0 or 1")
        return inputs, np.asarray(targets, dtype=np.int64)


def pick_device(prefer_gpu: bool = True) -> Any:
    """CUDA, then Apple's GPU (MPS), then the CPU."""
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_network(channels: tuple[int, ...], size: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # GroupNorm, not BatchNorm: BatchNorm's running averages are still
    # settling in the early epochs that early stopping often restores on a
    # small dataset, and then the restored model answers the same thing for
    # every picture in eval mode (measured: 86% in train mode, 36% in eval).
    # GroupNorm has no running state, so train and eval agree.
    def norm(width: int):
        return nn.GroupNorm(min(8, width), width)

    class Augment(nn.Module):
        """Training only: flip time and frequency, shift frequency a little."""

        def forward(self, x):
            if not self.training:
                return x
            n = x.shape[0]
            flip_t = torch.rand(n, device=x.device) < 0.5
            flip_f = torch.rand(n, device=x.device) < 0.5
            x = torch.where(flip_t[:, None, None, None], x.flip(-2), x)
            x = torch.where(flip_f[:, None, None, None], x.flip(-1), x)
            shift = int(torch.randint(-x.shape[-1] // 16, x.shape[-1] // 16 + 1, (1,)))
            return torch.roll(x, shifts=shift, dims=-1)

    class Block(nn.Module):
        def __init__(self, width_in: int, width_out: int, stride: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(width_in, width_out, 3, stride, 1, bias=False)
            self.bn1 = norm(width_out)
            self.conv2 = nn.Conv2d(width_out, width_out, 3, 1, 1, bias=False)
            self.bn2 = norm(width_out)
            self.skip = (nn.Identity() if stride == 1 and width_in == width_out else
                         nn.Sequential(nn.Conv2d(width_in, width_out, 1, stride, bias=False), norm(width_out)))

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            return F.relu(self.bn2(self.conv2(out)) + self.skip(x))

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.size = size
            self.augment = Augment()
            self.stem = nn.Sequential(nn.Conv2d(1, channels[0], 5, 2, 2, bias=False),
                                      norm(channels[0]), nn.ReLU())
            layers, width = [], channels[0]
            for out in channels:
                layers.append(Block(width, out, 2))
                width = out
            self.blocks = nn.Sequential(*layers)
            self.head = nn.Linear(width, 1)

        def forward(self, x):
            if x.shape[-2:] != (self.size, self.size):
                x = F.adaptive_avg_pool2d(x, (self.size, self.size))
            # The band edges are far below the median (about -6 spreads);
            # clipping keeps them from dominating the first layer.
            x = self.blocks(self.stem(self.augment(x.clamp(-4.0, 8.0))))
            return self.head(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))

    return Network()


class WaterfallVetter(Vetter):
    """P(the satellite's signal is in this waterfall)."""

    def __init__(self, channels: tuple[int, ...] = (16, 32, 64, 96), size: int = 128,
                 learning_rate: float = 2e-3, seed: int = 0, device: Any = None,
                 name: str = "WaterfallVetter") -> None:
        import torch

        super().__init__(name)
        if size < 32:
            raise ValidationError("size", size, "at least 32")
        self._channels = tuple(int(c) for c in channels)
        self._size = int(size)
        self._lr = float(learning_rate)
        self._seed = int(seed)
        self._device = device or pick_device()
        torch.manual_seed(self._seed)
        self._net = _build_network(self._channels, self._size).to(self._device)
        self._meta: dict[str, Any] = {}

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self._net.parameters())

    @property
    def device(self) -> Any:
        return self._device

    # ------------------------------------------------------------------
    def fit(self, inputs: np.ndarray, targets: np.ndarray, validation: tuple[np.ndarray, np.ndarray] | None = None,
            epochs: int = 30, batch_size: int = 64, patience: int = 6, verbose: bool = True,
            **_: Any) -> TrainingHistory:
        import torch
        import torch.nn as nn

        inputs, targets = self._check_binary(inputs, targets)

        # Balance the two labels in the loss, whatever the mix of the data.
        share = float(np.mean(targets))
        pos_weight = torch.tensor([(1 - share) / max(share, 1e-3)], device=self._device)
        callbacks: list[Any] = [EarlyStopping(
            monitor="val_loss" if validation is not None else "loss", patience=patience,
            snapshot=lambda: {k: v.detach().clone() for k, v in self._net.state_dict().items()},
            restore=self._net.load_state_dict)]
        if verbose:
            callbacks.insert(0, ProgressPrinter(name=self.name))
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self._lr, weight_decay=1e-4)
        trainer = Trainer(model=self._net, loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight),
                          optimizer=optimizer, device=self._device, callbacks=callbacks, name=self.name)
        column = lambda y: np.asarray(y, dtype=np.float32)[:, None]  # noqa: E731
        started = time.time()
        self._history = trainer.fit(inputs, column(targets),
                                    *(validation[0], column(validation[1])) if validation is not None else (None, None),
                                    epochs=epochs, batch_size=batch_size, seed=self._seed)
        self._meta.update(trained_at=time.time(), train_seconds=time.time() - started,
                          n_train=int(len(inputs)), share_with_signal=share)
        self._mark_trained()
        return self._history

    def predict_proba(self, inputs: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """P(signal) for each waterfall, shape ``(n,)``."""
        import torch

        self._require_trained()
        inputs = np.asarray(inputs, dtype=np.float32)
        if inputs.ndim == 2:
            inputs = inputs[None, None]
        elif inputs.ndim == 3:
            inputs = inputs[:, None]
        self._net.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                batch = torch.as_tensor(inputs[start:start + batch_size]).to(self._device)
                out.append(torch.sigmoid(self._net(batch)).squeeze(1).cpu().numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import torch

        self._require_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path)
        self._write_sidecar(path, {"model": type(self).__name__, "channels": list(self._channels),
                                   "size": self._size, "n_parameters": self.n_parameters,
                                   **self._meta, "history": self._history.to_dict()})
        return path

    def load(self, path: str | Path) -> "WaterfallVetter":
        import torch

        path = Path(path)
        meta = self._read_sidecar(path)
        if meta and (tuple(meta.get("channels", self._channels)) != self._channels
                     or int(meta.get("size", self._size)) != self._size):
            self._channels, self._size = tuple(meta["channels"]), int(meta["size"])
            self._net = _build_network(self._channels, self._size).to(self._device)
        self._net.load_state_dict(torch.load(path, map_location=self._device, weights_only=True))
        self._meta = {k: v for k, v in meta.items() if k not in ("history", "channels", "size")}
        self._mark_trained()
        return self

    @classmethod
    def from_file(cls, path: str | Path, device: Any = None) -> "WaterfallVetter":
        return cls(device=device).load(path)

    def describe(self) -> str:
        state = "trained" if self.is_trained else "untrained"
        return f"{self.name} (2-D ResNet, {self.n_parameters:,} weights, {state})"


class FeatureVetter(Vetter):
    """300 decision trees voting on the twelve waterfall measurements.

    Small data suits it (it scored AUROC 0.83 on held-out stations from
    315 observations, where the CNN could not learn), and it can explain
    itself: :meth:`importances` says which measurements it relies on.
    The measurements are computed inside, so it takes the same input as
    every other :class:`Vetter`.

    Saved with ``pickle``: only load files this project wrote.
    """

    def __init__(self, n_trees: int = 300, min_leaf: int = 3, seed: int = 0, name: str = "FeatureVetter") -> None:
        super().__init__(name)
        if n_trees < 1 or min_leaf < 1:
            raise ValidationError("n_trees/min_leaf", (n_trees, min_leaf), "must be positive")
        self._n_trees, self._min_leaf, self._seed = int(n_trees), int(min_leaf), int(seed)
        self._forest: Any = None
        self._meta: dict[str, Any] = {}

    def fit(self, inputs: np.ndarray, targets: np.ndarray, **_: Any) -> TrainingHistory:
        from sklearn.ensemble import RandomForestClassifier

        inputs, targets = self._check_binary(inputs, targets)
        return self.fit_features(feature_matrix(inputs), targets)

    def fit_features(self, features: np.ndarray, targets: np.ndarray) -> TrainingHistory:
        """Train from measurements already computed (see :func:`feature_matrix`)."""
        from sklearn.ensemble import RandomForestClassifier

        features, targets = self._check_pair(features, targets)
        if features.ndim != 2 or features.shape[1] != len(FEATURES):
            raise ValidationError("features", features.shape, f"expected (n, {len(FEATURES)})")
        if len(np.unique(targets)) < 2:
            raise ValidationError("targets", np.unique(targets), "need examples of both labels")
        started = time.time()
        self._forest = RandomForestClassifier(
            n_estimators=self._n_trees, min_samples_leaf=self._min_leaf, class_weight="balanced",
            random_state=self._seed, n_jobs=-1).fit(features, targets)
        self._history = TrainingHistory(self.name)
        self._history.record(1, train_accuracy=float(self._forest.score(features, targets)))
        self._meta.update(trained_at=time.time(), train_seconds=time.time() - started, n_train=int(len(features)))
        self._mark_trained()
        return self._history

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=np.float32)
        if inputs.ndim == 2:
            inputs = inputs[None, None]
        return self.predict_proba_features(feature_matrix(inputs))

    def predict_proba_features(self, features: np.ndarray) -> np.ndarray:
        """P(signal) from measurements already computed."""
        self._require_trained()
        column = list(self._forest.classes_).index(1)
        return self._forest.predict_proba(np.asarray(features, dtype=np.float32))[:, column].astype(np.float32)

    def importances(self) -> dict[str, float]:
        """Each measurement's share of the forest's decisions, largest first."""
        self._require_trained()
        pairs = zip(FEATURES, self._forest.feature_importances_.tolist())
        return dict(sorted(pairs, key=lambda kv: -kv[1]))

    def save(self, path: str | Path) -> Path:
        self._require_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self._forest))
        self._write_sidecar(path, {"model": type(self).__name__, "features": list(FEATURES),
                                   "n_trees": self._n_trees, "min_leaf": self._min_leaf, **self._meta,
                                   "importances": self.importances()})
        return path

    def load(self, path: str | Path) -> "FeatureVetter":
        path = Path(path)
        meta = self._read_sidecar(path)
        if meta.get("features") and meta["features"] != list(FEATURES):
            raise ValidationError("features", meta["features"], "saved with different measurements")
        self._forest = pickle.loads(path.read_bytes())  # noqa: S301 - own files only
        self._meta = {k: v for k, v in meta.items() if k not in ("features", "importances")}
        self._mark_trained()
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> "FeatureVetter":
        return cls().load(path)

    def describe(self) -> str:
        state = "trained" if self.is_trained else "untrained"
        return f"{self.name} ({self._n_trees} trees on {len(FEATURES)} measurements, {state})"
