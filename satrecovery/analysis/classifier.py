"""The AI classifier: which kind of signal is this?

A convolutional neural network (CNN) looks at a recording's raw samples
and says which of the six modulations it is -- the step a receiver needs
before it can read a single bit, the same way a translator needs to
know the language before translating.

It does not recover bits.  That is the receivers' job (Phases 2-5).  It
answers one question, with a probability for every possible answer, so
the app can say "QPSK, 97% sure" and show the runners-up.

WHAT THE NETWORK SEES
---------------------
The 2048 complex samples, as two rows (in-phase and quadrature), each
recording scaled to the same average power and its DC offset removed
-- otherwise the network could learn "loud means 64QAM" from the
recording levels, which would be a fact about the dataset, not about
the signals.

The network is a small **ResNet**: blocks of convolutions with
"shortcut" connections that let each block learn a correction to what
the previous one found.  It is the design that scored best (about 96%
at high SNR) in the RML24 authors' own tests.  The first version of
this classifier was a plain stack of convolutions, trained briefly on
less data, and reached only 58% on the cleanest recordings -- far below
what the dataset allows, so it was rebuilt (``architecture="cnn"`` still
loads models saved by that version).

**Augmentation.**  While training, every recording is rotated by a
random angle and shifted by a random number of samples each time the
network sees it.  Neither changes what modulation a signal is -- a
satellite's carrier phase and where a recording happens to start are
accidents -- so the network is taught to ignore them, and every
recording becomes many different examples.

DESIGN NOTE: INHERITANCE FROM THE PROJECT'S OWN HIERARCHY
---------------------------------------------------------
:class:`ModulationClassifier` is a
:class:`~satrecovery.learning.base.TrainableModel`, like the Phase 4
networks.  It inherits the "trained or not" state, the guard that stops
an untrained model being used, and the metadata sidecar written next to
saved weights; and it reuses the Phase 4
:class:`~satrecovery.learning.trainer.Trainer` loop and early stopping
unchanged.  What it adds is only what is new: the network, and turning
outputs into a :class:`Classification`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..core.exceptions import ValidationError
from ..data import layout
from ..learning.base import TrainableModel, TrainingHistory
from ..learning.trainer import EarlyStopping, Trainer, TrainingCallback, pick_device


class Classification:
    """The classifier's answer for one recording. Immutable.

    ``probabilities`` maps every class to its probability, most likely
    first, and they sum to one.
    """

    __slots__ = ("_probabilities",)

    def __init__(self, probabilities: dict[str, float]) -> None:
        if not probabilities:
            raise ValidationError("probabilities", probabilities, "needs at least one class")
        ranked = sorted(((str(k), float(v)) for k, v in probabilities.items()), key=lambda kv: -kv[1])
        self._probabilities = tuple(ranked)

    @property
    def label(self) -> str:
        """The most likely class."""
        return self._probabilities[0][0]

    @property
    def confidence(self) -> float:
        """Its probability, between 0 and 1."""
        return self._probabilities[0][1]

    @property
    def probabilities(self) -> dict[str, float]:
        """Every class, most likely first."""
        return dict(self._probabilities)

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        return list(self._probabilities[: int(k)])

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": [{"name": n, "p": p} for n, p in self._probabilities],
        }

    def __repr__(self) -> str:
        return f"Classification({self.label}, {self.confidence:.1%})"


class ModulationClassifier(TrainableModel):
    """A 1-D CNN that names a recording's modulation.

    Parameters
    ----------
    classes:
        The answers it can give, in output order.  Defaults to the six
        schemes this project's receivers handle.
    channels:
        Width of each convolution block.
    """

    DEFAULT_CLASSES = tuple(layout.SUPPORTED)
    ARCHITECTURES = ("resnet", "cnn", "resnet_raw", "lstm", "transformer")
    #: In plain words, for the "Compare models" tab.
    ARCHITECTURE_NAMES = {
        "resnet": "ResNet + expert features (the app's model)",
        "resnet_raw": "ResNet, no expert features",
        "cnn": "Plain CNN (the first version)",
        "lstm": "LSTM (reads the signal in order)",
        "transformer": "Transformer (attention)",
    }

    def __init__(
        self,
        classes: tuple[str, ...] | list[str] | None = None,
        channels: tuple[int, ...] = (32, 64, 96, 128),
        learning_rate: float = 2e-3,
        seed: int = 0,
        device: Any = None,
        name: str = "ModulationClassifier",
        architecture: str = "resnet",
    ) -> None:
        if architecture not in self.ARCHITECTURES:
            raise ValidationError("architecture", architecture, f"must be one of {self.ARCHITECTURES}")
        super().__init__(name)
        classes = tuple(str(c) for c in (classes or self.DEFAULT_CLASSES))
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValidationError("classes", classes, "need at least two distinct classes")
        if not channels or any(int(c) < 1 for c in channels):
            raise ValidationError("channels", channels, "need positive widths")
        import torch

        torch.manual_seed(int(seed))
        self._classes = classes
        self._channels = tuple(int(c) for c in channels)
        self._learning_rate = float(learning_rate)
        self._seed = int(seed)
        self._device = device or pick_device()
        self._architecture = architecture
        build = _BUILDERS[architecture]
        self._net = build(len(classes), self._channels).to(self._device)
        self._test_accuracy: float | None = None
        self._trained_at: float | None = None

    # -- read-only ------------------------------------------------------
    @property
    def classes(self) -> tuple[str, ...]:
        return self._classes

    @property
    def architecture(self) -> str:
        return self._architecture

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self._net.parameters())

    @property
    def device(self) -> Any:
        return self._device

    @property
    def test_accuracy(self) -> float | None:
        """Accuracy on held-out recordings, if it was measured."""
        return self._test_accuracy

    # -- input ----------------------------------------------------------
    @staticmethod
    def prepare(samples: np.ndarray) -> np.ndarray:
        """Recordings in any common shape -> ``(n, 2, length)`` float32.

        Accepts one complex recording, a batch of complex recordings, or
        RML24's ``(n, length, 2)`` float pairs.  Each recording has its
        mean removed and is scaled to unit average power.
        """
        x = np.asarray(samples)
        if np.iscomplexobj(x):
            x = np.atleast_2d(x)
            z = x.astype(np.complex64)
        else:
            if x.ndim == 2 and x.shape[1] == 2:
                x = x[None, ...]
            if x.ndim != 3 or x.shape[-1] != 2:
                raise ValidationError("samples", x.shape, "expected complex samples or (n, length, 2) pairs")
            z = x[..., 0].astype(np.float32) + 1j * x[..., 1].astype(np.float32)
        z = z - z.mean(axis=1, keepdims=True)
        power = np.sqrt(np.mean(np.abs(z).astype(np.float64) ** 2, axis=1, keepdims=True))  # no float32 overflow
        z = z / np.maximum(power, 1e-12)
        return np.stack([z.real, z.imag], axis=1).astype(np.float32)

    # -- training -------------------------------------------------------
    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
        epochs: int = 40,
        batch_size: int = 256,
        patience: int = 10,
        verbose: bool = True,
        report: Any = None,
        **_: Any,
    ) -> TrainingHistory:
        """Learn from ``(prepared recordings, class indices)``.

        ``report(epoch, total, metrics)`` is called after every epoch with
        the loss and the validation accuracy, if given.

        Uses the Phase 4 :class:`Trainer` unchanged.  That loop casts
        targets to floats, so the class indices go in as one-hot rows,
        which ``CrossEntropyLoss`` accepts as target probabilities.
        """
        import torch.nn as nn
        import torch

        inputs, targets = self._check_pair(inputs, targets)
        if inputs.ndim != 3 or inputs.shape[1] != 2:
            raise ValidationError("inputs", inputs.shape, "expected (n, 2, length); use prepare()")
        targets = np.asarray(targets, dtype=np.int64)
        if targets.min() < 0 or targets.max() >= len(self._classes):
            raise ValidationError("targets", (targets.min(), targets.max()), "class index out of range")

        callbacks: list[Any] = [
            EarlyStopping(
                monitor="val_loss" if validation else "loss",
                patience=patience,
                snapshot=lambda: {k: v.detach().clone() for k, v in self._net.state_dict().items()},
                restore=self._net.load_state_dict,
            )
        ]
        if verbose or report is not None:
            check = validation if validation else (inputs[:1000], targets[:1000])
            callbacks.insert(0, AccuracyReporter(self, check[0], check[1], report))
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self._learning_rate, weight_decay=1e-4)
        # Fast at first, gently slower to the end: a cosine schedule,
        # stepped by a callback so the Phase 4 Trainer stays unchanged.
        callbacks.append(CosineSchedule(optimizer, epochs))
        if verbose or report is not None:
            sample = np.random.default_rng(self._seed).choice(len(inputs), size=min(2000, len(inputs)), replace=False)
            callbacks[0].track_training(inputs[np.sort(sample)], targets[np.sort(sample)])
        trainer = Trainer(
            model=self._net,
            loss_fn=nn.CrossEntropyLoss(label_smoothing=0.05),
            optimizer=optimizer,
            device=self._device,
            callbacks=callbacks,
            name=self._name,
        )
        eye = np.eye(len(self._classes), dtype=np.float32)
        val_x, val_y = (validation[0], eye[np.asarray(validation[1], dtype=np.int64)]) if validation else (None, None)
        self._history = trainer.fit(inputs, eye[targets], val_x, val_y, epochs=epochs,
                                    batch_size=batch_size, seed=self._seed)
        self._trained_at = time.time()
        self._mark_trained()
        return self._history

    # -- using it -------------------------------------------------------
    def predict_proba(self, samples: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Class probabilities, one row per recording."""
        import torch

        self._require_trained()
        x = samples if (isinstance(samples, np.ndarray) and samples.ndim == 3 and samples.shape[1] == 2
                        and samples.dtype == np.float32) else self.prepare(samples)
        self._net.eval()
        rows = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                batch = torch.as_tensor(x[start : start + batch_size]).to(self._device)
                rows.append(torch.softmax(self._net(batch), dim=1).cpu().numpy())
        return np.concatenate(rows) if rows else np.empty((0, len(self._classes)))

    def classify(self, samples: np.ndarray) -> Classification:
        """The answer for one recording."""
        probabilities = self.predict_proba(samples)[0]
        return Classification(dict(zip(self._classes, probabilities.tolist())))

    def evaluate(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Fraction of recordings named correctly."""
        predicted = np.argmax(self.predict_proba(inputs), axis=1)
        return float(np.mean(predicted == np.asarray(targets)))

    def record_test_accuracy(self, accuracy: float) -> None:
        self._test_accuracy = float(accuracy)

    # -- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import torch

        self._require_trained()
        path = Path(path).with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path)
        self._write_sidecar(path, {
            "model": self._name,
            "kind": "ModulationClassifier",
            "architecture": self._architecture,
            "classes": list(self._classes),
            "channels": list(self._channels),
            "n_parameters": self.n_parameters,
            "test_accuracy": self._test_accuracy,
            "trained_at": self._trained_at,
            "history": self._history.to_dict(),
        })
        return path

    def load(self, path: str | Path) -> "ModulationClassifier":
        import torch

        path = Path(path).with_suffix(".pt")
        if not path.exists():
            raise ValidationError("path", str(path), "no saved classifier here")
        meta = self._read_sidecar(path)
        if meta.get("classes") and tuple(meta["classes"]) != self._classes:
            raise ValidationError("path", str(path), f"saved for classes {meta['classes']}")
        self._net.load_state_dict(torch.load(path, map_location=self._device, weights_only=True))
        self._test_accuracy = meta.get("test_accuracy")
        self._trained_at = meta.get("trained_at")
        self._mark_trained()
        return self

    @classmethod
    def from_file(cls, path: str | Path, device: Any = None) -> "ModulationClassifier":
        """Build with the saved shape, then load the saved weights."""
        path = Path(path).with_suffix(".pt")
        meta = cls._read_sidecar(path) if path.exists() else {}
        # Models saved before the ResNet existed carry no "architecture".
        model = cls(classes=meta.get("classes"), channels=tuple(meta.get("channels", (32, 64, 96, 128))),
                    device=device, architecture=meta.get("architecture", "cnn"))
        return model.load(path)

    def describe(self) -> str:
        state = "trained" if self.is_trained else "untrained"
        accuracy = f", {self._test_accuracy:.1%} on held-out recordings" if self._test_accuracy else ""
        return f"{self._name} ({self._architecture}, {state}, {self.n_parameters:,} weights{accuracy})"


class AccuracyReporter(TrainingCallback):
    """Prints accuracy after every epoch -- the number that means something here.

    The Phase 4 trainer reports a bit error rate, which is meaningless
    for a classifier, and is shared by models that need it; so rather
    than change it, this callback is *added* and reports what matters.
    """

    def __init__(self, model: "ModulationClassifier", inputs: np.ndarray, targets: np.ndarray,
                 report: Any = None) -> None:
        self._model = model
        self._inputs = inputs
        self._targets = np.asarray(targets)
        self._report = report
        self._total = 0
        self._started = 0.0
        self._train: tuple[np.ndarray, np.ndarray] | None = None

    def track_training(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """Also report accuracy on these training recordings.

        Training accuracy next to validation accuracy is how to tell the
        two ways a model can fall short apart: both low means it has not
        learned enough (bigger model, longer training); training high but
        validation low means it is memorising (more data, augmentation).
        """
        self._train = (inputs, np.asarray(targets))

    def _accuracy(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        import torch

        net = self._model._net
        net.eval()
        correct = 0
        with torch.no_grad():
            for start in range(0, len(inputs), 256):
                batch = torch.as_tensor(inputs[start : start + 256]).to(self._model.device)
                correct += int((net(batch).argmax(1).cpu().numpy() == targets[start : start + 256]).sum())
        return correct / max(len(targets), 1)

    def on_train_begin(self, total_epochs: int) -> None:
        self._total = int(total_epochs)
        self._started = time.time()
        if self._report is None:
            print(f"training {self._model.name}: up to {total_epochs} epochs, "
                  f"{self._model.n_parameters:,} weights, on {self._model.device}")

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> bool:
        # The model is mid-training, so ask the network directly.
        accuracy = self._accuracy(self._inputs, self._targets)
        train_accuracy = self._accuracy(*self._train) if self._train is not None else None
        summary = {"loss": metrics.get("loss"), "val_loss": metrics.get("val_loss"),
                   "val_accuracy": accuracy, "train_accuracy": train_accuracy}
        if self._report is not None:
            self._report(epoch, self._total, summary)
        else:
            val = f"  val loss {summary['val_loss']:.4f}" if summary["val_loss"] is not None else ""
            seen = f"  training {train_accuracy:6.1%}" if train_accuracy is not None else ""
            print(f"  epoch {epoch:3d}  loss {summary['loss']:.4f}{val}{seen}  validation {accuracy:6.1%}"
                  f"   ({time.time() - self._started:.0f} s)")
        return True

    def on_train_end(self, history: TrainingHistory) -> None:
        pass


class CosineSchedule(TrainingCallback):
    """Lowers the learning rate along a half cosine, once per epoch.

    Big steps early to find a good region, small steps late to settle in
    it.  A callback rather than a change to the shared Trainer.
    """

    def __init__(self, optimizer: Any, total_epochs: int, floor: float = 0.02) -> None:
        self._optimizer = optimizer
        self._total = max(int(total_epochs), 1)
        self._floor = float(floor)
        self._base = [group["lr"] for group in optimizer.param_groups]

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> bool:
        import math

        progress = min(epoch / self._total, 1.0)
        scale = self._floor + (1 - self._floor) * 0.5 * (1 + math.cos(math.pi * progress))
        for group, base in zip(self._optimizer.param_groups, self._base):
            group["lr"] = base * scale
        return True


def _augmenter():
    """Random rotation and shift, active only while training."""
    import torch
    import torch.nn as nn

    class Augment(nn.Module):
        def forward(self, x):
            if not self.training:
                return x
            n = x.shape[0]
            angle = torch.rand(n, 1, device=x.device) * (2 * torch.pi)
            c, s = torch.cos(angle), torch.sin(angle)
            i, q = x[:, 0], x[:, 1]
            x = torch.stack([i * c - q * s, i * s + q * c], dim=1)
            shift = int(torch.randint(0, x.shape[2], (1,)))
            return torch.roll(x, shifts=shift, dims=2)

    return Augment()


def _build_resnet(n_classes: int, channels: tuple[int, ...], expert: bool = True):
    """A small 1-D ResNet: a stem, then two residual blocks per width.

    ``expert=False`` leaves out the expert features -- the same network
    learning from the raw samples alone, to measure what they add.
    """
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, width_in: int, width_out: int, stride: int) -> None:
            super().__init__()
            self.main = nn.Sequential(
                nn.Conv1d(width_in, width_out, 5, stride=stride, padding=2, bias=False),
                nn.BatchNorm1d(width_out),
                nn.ReLU(inplace=True),
                nn.Conv1d(width_out, width_out, 5, padding=2, bias=False),
                nn.BatchNorm1d(width_out),
            )
            self.shortcut = (
                nn.Identity() if stride == 1 and width_in == width_out else
                nn.Sequential(nn.Conv1d(width_in, width_out, 1, stride=stride, bias=False),
                              nn.BatchNorm1d(width_out))
            )
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(self.main(x) + self.shortcut(x))

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.augment = _augmenter()
            if expert:
                self.expert = _ExpertFeatures()
                self.expert_head = nn.Sequential(
                    nn.BatchNorm1d(_ExpertFeatures.COUNT),
                    nn.Linear(_ExpertFeatures.COUNT, 64),
                    nn.ReLU(inplace=True),
                )
            self.stem = nn.Sequential(
                nn.Conv1d(2, channels[0], 7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(channels[0]),
                nn.ReLU(inplace=True),
            )
            blocks: list[nn.Module] = []
            width = channels[0]
            for index, out in enumerate(channels):
                stride = 1 if index == 0 else 2
                blocks += [Block(width, out, stride), Block(out, out, 1)]
                width = out
            self.body = nn.Sequential(*blocks)
            self.head = nn.Sequential(
                nn.Linear(2 * width + (64 if expert else 0), 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            x = self.augment(x)
            features = self.body(self.stem(x))
            pooled = torch.cat([features.mean(dim=2), features.amax(dim=2)], dim=1)
            if not expert:
                return self.head(pooled)
            return self.head(torch.cat([pooled, self.expert_head(self.expert(x))], dim=1))

    return Network()


def _expert_features_class():
    import torch
    import torch.nn as nn

    class ExpertFeatures(nn.Module):
        """Classical signal-processing measurements, as extra inputs.

        The network alone confused QPSK with 8PSK and 16QAM with 64QAM:
        in the raw waveform, slowly spun by the carrier error, they look
        alike.  Classical theory separates them directly.  Raise a signal
        to the power *k* and, if *k* is a multiple of the constellation's
        symmetry, the data cancels and a sharp spectral line appears;
        otherwise there is none.  BPSK has a line at k = 2, QPSK first at
        4, 8PSK first at 8, 16APSK first at 12.  The *shape* of the
        amplitude distribution -- how often the signal is loud or quiet --
        tells the QAMs and APSK apart.

        These are measured here, with no learned weights, and handed to
        the network beside its own features.  Rotating the signal or
        shifting where it starts changes none of them, so they need no
        augmentation.  Classical knowledge plus learning: the same idea
        as the project's hybrid receiver, applied to classification.
        """

        POWERS = (2, 4, 8, 12)
        COUNT = 2 * len(POWERS) + 4

        def forward(self, x):
            z = torch.complex(x[:, 0].float(), x[:, 1].float())
            magnitude = z.abs()
            power = magnitude.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12)
            z = z / power.sqrt()
            magnitude = z.abs()
            unit = z / magnitude.clamp_min(1e-12)
            out = []
            for k in self.POWERS:
                for weighted in (unit.pow(k) * magnitude.pow(min(k, 4)), unit.pow(k)):
                    spectrum = torch.fft.fft(weighted, dim=1).abs()
                    out.append(torch.log(spectrum.amax(dim=1) / spectrum.mean(dim=1).clamp_min(1e-12)))
            m2 = magnitude.pow(2)
            out.append(magnitude.std(dim=1))
            out.append(torch.log(m2.pow(2).mean(dim=1) / m2.mean(dim=1).pow(2)))
            out.append(torch.log(m2.pow(3).mean(dim=1) / m2.mean(dim=1).pow(3)))
            out.append((m2 < 0.25).float().mean(dim=1))
            return torch.stack(out, dim=1)

    return ExpertFeatures


class _LazyExpert:
    """Defer building the module class until torch is imported."""

    _cls = None

    @property
    def COUNT(self) -> int:  # noqa: N802 - mirrors the class attribute
        return 2 * 4 + 4

    def __call__(self):
        if _LazyExpert._cls is None:
            _LazyExpert._cls = _expert_features_class()
        return _LazyExpert._cls()


_ExpertFeatures = _LazyExpert()


def _build_network(n_classes: int, channels: tuple[int, ...]):
    """The CNN. Built here so importing this module does not need torch."""
    import torch
    import torch.nn as nn

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            width = 2
            for out in channels:
                layers += [
                    nn.Conv1d(width, out, kernel_size=7, padding=3),
                    nn.BatchNorm1d(out),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out, out, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),
                ]
                width = out
            self.body = nn.Sequential(*layers)
            self.head = nn.Sequential(
                nn.Linear(2 * width, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            features = self.body(x)
            pooled = torch.cat([features.mean(dim=2), features.amax(dim=2)], dim=1)
            return self.head(pooled)

    return Network()


def _downsampling_stem(width: int):
    """Three strided convolutions: 2048 samples become 128 steps of ``width`` numbers.

    An LSTM or a Transformer reading all 2048 samples one by one would be
    very slow; this short front end summarises every 16 samples first.
    """
    import torch.nn as nn

    layers: list[nn.Module] = []
    previous = 2
    for out, stride in ((32, 4), (width, 2), (width, 2)):
        layers += [nn.Conv1d(previous, out, 7, stride=stride, padding=3, bias=False),
                   nn.BatchNorm1d(out), nn.ReLU(inplace=True)]
        previous = out
    return nn.Sequential(*layers)


def _build_lstm(n_classes: int, channels: tuple[int, ...]):
    """An LSTM: reads the signal step by step, remembering what came before.

    Good at patterns in time (it is used for speech and text).  The
    signal is first shortened by a small convolution front end, then a
    two-layer, two-direction LSTM reads it.
    """
    import torch
    import torch.nn as nn

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.augment = _augmenter()
            self.stem = _downsampling_stem(64)
            self.lstm = nn.LSTM(64, 64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
                                      nn.Linear(128, n_classes))

        def forward(self, x):
            steps = self.stem(self.augment(x)).transpose(1, 2)      # (n, steps, 64)
            out, _ = self.lstm(steps)
            return self.head(torch.cat([out.mean(dim=1), out.amax(dim=1)], dim=1))

    return Network()


def _build_transformer(n_classes: int, channels: tuple[int, ...]):
    """A Transformer: every part of the signal 'looks at' every other part.

    The design behind modern language models.  The signal is shortened
    by a small convolution front end, then two attention layers compare
    all 128 steps with each other.  No position numbers are added: what
    modulation a signal is does not depend on where in the recording a
    pattern happens, and the convolutions already capture local order.
    """
    import torch
    import torch.nn as nn

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.augment = _augmenter()
            self.stem = _downsampling_stem(64)
            layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1,
                                               batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
                                      nn.Linear(128, n_classes))

        def forward(self, x):
            steps = self.stem(self.augment(x)).transpose(1, 2)
            out = self.encoder(steps)
            return self.head(torch.cat([out.mean(dim=1), out.amax(dim=1)], dim=1))

    return Network()


_BUILDERS = {
    "resnet": _build_resnet,
    "cnn": _build_network,
    "resnet_raw": lambda n, c: _build_resnet(n, c, expert=False),
    "lstm": _build_lstm,
    "transformer": _build_transformer,
}


__all__ = ["AccuracyReporter", "Classification", "CosineSchedule", "ModulationClassifier"]
