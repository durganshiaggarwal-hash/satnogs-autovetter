"""The baseline auto-vetter: a small 2-D ResNet that looks at the waterfall.

This is the *baseline* on purpose: it sees only the picture, as earlier
public work does, so later steps (audio features, the orbit check,
calibrated abstention) can be measured against it.

DESIGN NOTE: INHERITANCE
------------------------
:class:`WaterfallVetter` is a :class:`~satrecovery.learning.TrainableModel`,
like the Phase 4 demappers and the RML24 classifier: it inherits the
"refuse to answer until trained" guard and the JSON sidecar, and trains
with the Phase 4 :class:`~satrecovery.learning.trainer.Trainer`
unchanged -- the same loop now fits pictures instead of symbols.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..core.exceptions import ValidationError
from ..learning.base import TrainableModel, TrainingHistory
from ..learning.trainer import EarlyStopping, ProgressPrinter, Trainer


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

    class Block(nn.Module):
        def __init__(self, width_in: int, width_out: int, stride: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(width_in, width_out, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(width_out)
            self.conv2 = nn.Conv2d(width_out, width_out, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(width_out)
            self.skip = (nn.Identity() if stride == 1 and width_in == width_out else
                         nn.Sequential(nn.Conv2d(width_in, width_out, 1, stride, bias=False), nn.BatchNorm2d(width_out)))

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            return F.relu(self.bn2(self.conv2(out)) + self.skip(x))

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.size = size
            self.stem = nn.Sequential(nn.Conv2d(1, channels[0], 5, 2, 2, bias=False),
                                      nn.BatchNorm2d(channels[0]), nn.ReLU())
            layers, width = [], channels[0]
            for out in channels:
                layers.append(Block(width, out, 2))
                width = out
            self.blocks = nn.Sequential(*layers)
            self.head = nn.Linear(width, 1)

        def forward(self, x):
            if x.shape[-2:] != (self.size, self.size):
                x = F.adaptive_avg_pool2d(x, (self.size, self.size))
            x = self.blocks(self.stem(x))
            return self.head(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))

    return Network()


class WaterfallVetter(TrainableModel):
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

        inputs, targets = self._check_pair(inputs, targets)
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValidationError("inputs", inputs.shape, "expected (n, 1, time, frequency)")
        if not set(np.unique(targets)) <= {0, 1}:
            raise ValidationError("targets", np.unique(targets)[:5], "labels must be 0 or 1")

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
