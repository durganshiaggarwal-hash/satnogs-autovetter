"""Training loops, and the hooks that let other code watch them.

DESIGN NOTE: THE OBSERVER PATTERN
---------------------------------
A training run needs to report progress, and different callers want
different things from it.  A terminal wants a line per epoch.  The GUI
in Phase 8 wants to move a progress bar.  A long run wants to stop early
when the model stops improving, and to keep the best weights rather than
the last ones.

:class:`Trainer` knows none of that.  It holds a list of
:class:`TrainingCallback` objects and tells them what happened; each one
decides for itself what to do about it.  Adding a new kind of
observation means writing one class and passing it in -- the training
loop never changes.

Callbacks can also *influence* the run: returning ``False`` from
``on_epoch_end`` stops training, which is how
:class:`EarlyStopping` works without the loop knowing early stopping
exists.
"""

from __future__ import annotations

import time
from abc import ABC
from typing import Any, Callable

import numpy as np

from ..core.exceptions import ValidationError
from .base import TrainingHistory


class TrainingCallback(ABC):
    """Something that watches a training run.

    Every hook has a default that does nothing, so a callback overrides
    only the moments it cares about.  That is what keeps
    :class:`ProgressPrinter` down to a handful of lines.
    """

    def on_train_begin(self, total_epochs: int) -> None:
        """Called once before the first epoch."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> bool:
        """Called after each epoch. Return ``False`` to stop training."""
        return True

    def on_train_end(self, history: TrainingHistory) -> None:
        """Called once after the last epoch."""


class ProgressPrinter(TrainingCallback):
    """Prints one line per epoch."""

    def __init__(self, every: int = 1, name: str = "") -> None:
        self._every = max(int(every), 1)
        self._name = name
        self._started = 0.0

    def on_train_begin(self, total_epochs: int) -> None:
        self._started = time.time()
        label = f" {self._name}" if self._name else ""
        print(f"training{label}: {total_epochs} epochs")

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> bool:
        if epoch % self._every == 0 or epoch == 1:
            parts = "  ".join(f"{k} {v:.5f}" for k, v in metrics.items())
            print(f"  epoch {epoch:>4}  {parts}")
        return True

    def on_train_end(self, history: TrainingHistory) -> None:
        elapsed = time.time() - self._started
        final = "  ".join(f"{k} {v:.5f}" for k, v in history.last().items())
        print(f"  done in {elapsed:.1f}s  ({final})")


class EarlyStopping(TrainingCallback):
    """Stops when the watched metric stops improving, and keeps the best.

    Two jobs in one, and the second matters more than it looks.  A model
    that trained for sixty epochs is not necessarily the model you want:
    if it was best at epoch forty and drifted afterwards, the final
    weights are worse than ones you already had.  This keeps a snapshot
    of the best and restores it at the end.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        minimise: bool = True,
        restore_best: bool = True,
        snapshot: Callable[[], Any] | None = None,
        restore: Callable[[Any], None] | None = None,
    ) -> None:
        self._monitor = monitor
        self._patience = max(int(patience), 1)
        self._minimise = bool(minimise)
        self._restore_best = bool(restore_best)
        self._snapshot = snapshot
        self._restore = restore

        self._best: float | None = None
        self._best_state: Any = None
        self._best_epoch = 0
        self._waiting = 0

    @property
    def best_epoch(self) -> int:
        """Which epoch produced the best value."""
        return self._best_epoch

    @property
    def best_value(self) -> float | None:
        """The best value seen."""
        return self._best

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> bool:
        if self._monitor not in metrics:
            return True

        value = metrics[self._monitor]
        improved = (
            self._best is None
            or (value < self._best if self._minimise else value > self._best)
        )

        if improved:
            self._best = value
            self._best_epoch = epoch
            self._waiting = 0
            if self._restore_best and self._snapshot is not None:
                self._best_state = self._snapshot()
            return True

        self._waiting += 1
        return self._waiting < self._patience

    def on_train_end(self, history: TrainingHistory) -> None:
        if self._best_state is not None and self._restore is not None:
            self._restore(self._best_state)


class Trainer:
    """Runs the training loop for a PyTorch model.

    Deliberately plain.  It shuffles, batches, steps the optimiser,
    evaluates, and tells the callbacks.  Everything optional -- progress
    reporting, early stopping, keeping the best weights -- lives in the
    callbacks rather than in here, so this loop stays short enough to
    read in one go and to trust.
    """

    def __init__(
        self,
        model: Any,
        loss_fn: Any,
        optimizer: Any,
        device: Any = None,
        callbacks: list[TrainingCallback] | None = None,
        name: str = "model",
    ) -> None:
        import torch

        self._torch = torch
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._device = device or torch.device("cpu")
        self._callbacks = list(callbacks or [])
        self._name = name

    @property
    def device(self) -> Any:
        """Where the tensors live."""
        return self._device

    @property
    def callbacks(self) -> tuple[TrainingCallback, ...]:
        """The observers watching this trainer."""
        return tuple(self._callbacks)

    def add_callback(self, callback: TrainingCallback) -> "Trainer":
        """Register another observer; returns ``self`` for chaining."""
        if not isinstance(callback, TrainingCallback):
            raise ValidationError(
                "callback", type(callback).__name__, "expected a TrainingCallback"
            )
        self._callbacks.append(callback)
        return self

    def fit(
        self,
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
        val_inputs: np.ndarray | None = None,
        val_targets: np.ndarray | None = None,
        epochs: int = 40,
        batch_size: int = 512,
        seed: int = 0,
    ) -> TrainingHistory:
        """Train, and return what happened."""
        torch = self._torch

        if int(epochs) < 1:
            raise ValidationError("epochs", epochs, "need at least one epoch")
        if int(batch_size) < 1:
            raise ValidationError("batch_size", batch_size, "must be positive")

        x_train = torch.as_tensor(train_inputs, dtype=torch.float32)
        y_train = torch.as_tensor(train_targets, dtype=torch.float32)

        has_validation = val_inputs is not None and val_targets is not None
        if has_validation:
            x_val = torch.as_tensor(val_inputs, dtype=torch.float32).to(self._device)
            y_val = torch.as_tensor(val_targets, dtype=torch.float32).to(self._device)

        history = TrainingHistory(self._name)
        generator = torch.Generator().manual_seed(int(seed))
        n_rows = x_train.shape[0]

        for callback in self._callbacks:
            callback.on_train_begin(int(epochs))

        for epoch in range(1, int(epochs) + 1):
            self._model.train()
            order = torch.randperm(n_rows, generator=generator)
            running = 0.0
            seen = 0

            for start in range(0, n_rows, batch_size):
                index = order[start : start + batch_size]
                batch_x = x_train[index].to(self._device)
                batch_y = y_train[index].to(self._device)

                self._optimizer.zero_grad(set_to_none=True)
                predictions = self._model(batch_x)
                loss = self._loss_fn(predictions, batch_y)
                loss.backward()
                self._optimizer.step()

                running += float(loss.item()) * len(index)
                seen += len(index)

            metrics = {"loss": running / max(seen, 1)}

            if has_validation:
                self._model.eval()
                with torch.no_grad():
                    predictions = self._model(x_val)
                    metrics["val_loss"] = float(
                        self._loss_fn(predictions, y_val).item()
                    )
                    # Bit error rate on the validation set: the number
                    # that actually matters, as opposed to the loss the
                    # optimiser happens to minimise.
                    decided = (predictions > 0).float()
                    metrics["val_ber"] = float((decided != y_val).float().mean())

            history.record(epoch, **metrics)

            keep_going = True
            for callback in self._callbacks:
                if not callback.on_epoch_end(epoch, metrics):
                    keep_going = False
            if not keep_going:
                break

        for callback in self._callbacks:
            callback.on_train_end(history)

        return history

    def __repr__(self) -> str:
        return f"Trainer(name={self._name!r}, device={self._device}, callbacks={len(self._callbacks)})"


def pick_device(prefer_gpu: bool = True) -> Any:
    """The best device available, without needing to be told.

    Our models are small enough that a laptop CPU trains them in
    minutes, so a missing GPU is an inconvenience rather than a blocker.
    Choosing automatically means the same script runs on a workstation
    and on a student laptop with no edit.
    """
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


__all__ = [
    "EarlyStopping",
    "ProgressPrinter",
    "Trainer",
    "TrainingCallback",
    "pick_device",
]
