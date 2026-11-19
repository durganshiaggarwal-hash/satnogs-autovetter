"""The learned receivers: a neural demapper and an end-to-end network.

Two models with very different ambitions.

:class:`MLPDemapper` replaces **one step** of the Phase 2 chain -- the
final decision that turns a cleaned-up symbol into bits.  Everything
before it stays as it was.

:class:`CNNDemodulator` replaces **the whole chain**.  It takes the raw
received waveform, with the frequency offset and timing error still in
it, and outputs bits.  To do that it would have to discover carrier
recovery, timing recovery and demapping for itself, from examples.

Including both is the point.  A project that only built the ambitious
one would learn nothing when it failed, and a project that only built
the modest one would never find out where the limit is.

DESIGN NOTE: WHY THE DEMAPPER INHERITS FROM TWO BASES
------------------------------------------------------
``class MLPDemapper(Demapper, TrainableModel)`` says exactly two things:
this satisfies the demapper interface Phase 2 defined, and it must be
trained before use.  Because the first half holds, it drops into the
existing receiver with no adapter::

    ClassicalReceiver(demapper=MLPDemapper(...))

That line is Phase 5's hybrid receiver, available already, and it
exists only because Phase 2 defined ``Demapper`` as an interface instead
of writing the nearest-neighbour rule inline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.exceptions import ValidationError
from ..modulation.base import Modulation
from ..receivers.demapper import Demapper
from .base import TrainableModel, TrainingHistory
from .trainer import EarlyStopping, Trainer, pick_device

#: Beyond this a log-likelihood ratio saturates the sigmoid; matching
#: the classical demapper's limit keeps the two comparable.
_MAX_LLR = 30.0


def _build_mlp(
    n_inputs: int,
    hidden: tuple[int, ...],
    n_outputs: int,
    dropout: float = 0.0,
) -> Any:
    """A plain fully-connected network with ReLU activations.

    ``dropout`` inserts a dropout layer after each hidden activation.
    During training that is ordinary regularisation.  Phase 5 uses it
    for a second purpose -- left switched on at inference it turns one
    network into a population of slightly different ones, whose
    disagreement is a usable uncertainty estimate.
    """
    import torch.nn as nn

    layers: list[Any] = []
    width = n_inputs
    for size in hidden:
        layers.append(nn.Linear(width, size))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        width = size
    layers.append(nn.Linear(width, n_outputs))
    return nn.Sequential(*layers)


class MLPDemapper(Demapper, TrainableModel):
    """A small network that decides bits from a window of symbols.

    The classical demapper looks at one symbol and picks the nearest
    constellation point.  That rule is provably optimal when the only
    damage is additive Gaussian noise -- so on a clean channel, the best
    a learned demapper can do is tie.

    The interesting case is when the damage is something else.  This
    network sees the symbol **together with its neighbours**, so it can
    in principle notice that a symbol has been pulled off its point by
    the one before it, and compensate.  The nearest-neighbour rule
    cannot do that by construction: it never looks sideways.

    Whether it actually learns to is the experiment, and the answer is
    allowed to be no.

    Parameters
    ----------
    modulation:
        The scheme this demapper is trained for.  One model per scheme,
        which is how it would be deployed -- both ends of a link agree
        the modulation in advance.
    window:
        How many symbols either side of the centre the network sees.
        Zero makes it a pure per-symbol decision, directly comparable
        with the classical rule.
    hidden:
        Width of each hidden layer.
    """

    def __init__(
        self,
        modulation: Modulation,
        window: int = 2,
        hidden: tuple[int, ...] = (64, 64),
        learning_rate: float = 1e-3,
        seed: int = 0,
        device: Any = None,
        name: str | None = None,
        dropout: float = 0.0,
    ) -> None:
        if not isinstance(modulation, Modulation):
            raise ValidationError(
                "modulation", type(modulation).__name__, "expected a Modulation"
            )
        if int(window) < 0:
            raise ValidationError("window", window, "cannot be negative")
        if not hidden:
            raise ValidationError("hidden", hidden, "need at least one hidden layer")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValidationError("dropout", dropout, "must lie in [0, 1)")

        label = name or f"MLP-{modulation.name}"
        Demapper.__init__(self, label)
        TrainableModel.__init__(self, label)

        self._modulation = modulation
        self._window = int(window)
        self._hidden = tuple(int(h) for h in hidden)
        self._learning_rate = float(learning_rate)
        self._seed = int(seed)
        self._dropout = float(dropout)
        self._device = device or pick_device()

        import torch

        torch.manual_seed(self._seed)
        self._net = _build_mlp(
            self.n_features,
            self._hidden,
            modulation.bits_per_symbol,
            dropout=self._dropout,
        ).to(self._device)

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------
    @property
    def modulation(self) -> Modulation:
        """The scheme this demapper was built for."""
        return self._modulation

    @property
    def window(self) -> int:
        """Symbols either side of the centre that the network sees."""
        return self._window

    @property
    def width(self) -> int:
        """Total symbols per decision."""
        return 2 * self._window + 1

    @property
    def n_features(self) -> int:
        """Numbers the network takes in: I and Q for each symbol."""
        return 2 * self.width

    @property
    def n_parameters(self) -> int:
        """How many weights the network has."""
        return sum(p.numel() for p in self._net.parameters())

    @property
    def dropout(self) -> float:
        """Dropout probability. Zero means the network is deterministic."""
        return self._dropout

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
        epochs: int = 40,
        batch_size: int = 1024,
        patience: int = 8,
        verbose: bool = True,
        **_: Any,
    ) -> TrainingHistory:
        """Learn from ``(symbol window, bits)`` pairs.

        Trained with binary cross-entropy on each bit independently,
        which is the right loss here: the bits of one symbol are decided
        together but scored separately, exactly as a bit error rate
        counts them.
        """
        import torch
        import torch.nn as nn

        inputs, targets = self._check_pair(inputs, targets)
        if inputs.shape[1] != self.n_features:
            raise ValidationError(
                "inputs",
                inputs.shape[1],
                f"this demapper expects {self.n_features} values per row "
                f"({self.width} symbols)",
            )
        if targets.shape[1] != self._modulation.bits_per_symbol:
            raise ValidationError(
                "targets",
                targets.shape[1],
                f"{self._modulation.name} carries "
                f"{self._modulation.bits_per_symbol} bits per symbol",
            )

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._learning_rate)
        callbacks: list[Any] = [
            EarlyStopping(
                monitor="val_loss" if validation else "loss",
                patience=patience,
                snapshot=lambda: {
                    k: v.detach().clone() for k, v in self._net.state_dict().items()
                },
                restore=self._net.load_state_dict,
            )
        ]
        if verbose:
            from .trainer import ProgressPrinter

            callbacks.insert(0, ProgressPrinter(every=5, name=self._name))

        trainer = Trainer(
            model=self._net,
            loss_fn=nn.BCEWithLogitsLoss(),
            optimizer=optimizer,
            device=self._device,
            callbacks=callbacks,
            name=self._name,
        )

        val_inputs, val_targets = validation if validation else (None, None)
        history = trainer.fit(
            inputs,
            targets,
            val_inputs,
            val_targets,
            epochs=epochs,
            batch_size=batch_size,
            seed=self._seed,
        )
        self._history = history
        self._mark_trained()
        return history

    # ------------------------------------------------------------------
    # Using it -- the Demapper interface
    # ------------------------------------------------------------------
    def demap(self, symbols, modulation):
        """Decide bits for a run of symbols.

        Satisfies the same contract as
        :class:`~satrecovery.receivers.demapper.SoftDemapper`, returning
        bits, per-bit confidence and log-likelihood ratios.

        The network's raw outputs *are* log-odds, which is what a
        log-likelihood ratio is.  So where the classical demapper has to
        compute distances and divide by an estimated noise power, this
        one reads its answer straight off the final layer -- and the two
        are directly comparable, because they mean the same thing.
        """
        import torch

        self._require_trained()
        symbols = self._validate(symbols, modulation)

        if modulation.name != self._modulation.name:
            raise ValidationError(
                "modulation",
                modulation.name,
                f"this demapper was trained for {self._modulation.name}",
            )

        if symbols.size == 0:
            return (
                np.empty(0, dtype=np.uint8),
                np.empty(0, dtype=float),
                np.empty(0, dtype=float),
            )

        rows = self._windows(symbols)

        self._net.eval()
        with torch.no_grad():
            logits = (
                self._net(torch.as_tensor(rows, dtype=torch.float32).to(self._device))
                .cpu()
                .numpy()
            )

        # The network predicts the probability that a bit is 1, so its
        # logit is log P(1)/P(0). The project's convention elsewhere is
        # the other way round, so negate to match.
        llr = np.clip(-logits.ravel(), -_MAX_LLR, _MAX_LLR)
        bits = (llr <= 0).astype(np.uint8)
        confidence = 1.0 / (1.0 + np.exp(-np.abs(llr)))
        return bits, confidence, llr

    def sample_bit_probabilities(
        self, symbols: np.ndarray, passes: int = 32, seed: int = 0
    ) -> np.ndarray:
        """Run the network repeatedly with dropout left switched on.

        Normally dropout is disabled at inference: every unit is used
        and the answer is deterministic.  Leaving it on makes each pass
        a different thinned network, so the same input gives slightly
        different answers, and the spread of those answers says
        something about how much the decision depended on any one part
        of the model.

        Returns
        -------
        np.ndarray
            Shape ``(passes, n_bits)``: each pass's probability that
            the corresponding bit is a one.

        Raises
        ------
        ValidationError
            If the network was built without dropout, in which case
            every pass would be identical and the resulting "spread" of
            exactly zero would look like perfect confidence.  Failing
            is better than returning a flattering constant.
        """
        import torch

        self._require_trained()
        if self._dropout <= 0.0:
            raise ValidationError(
                "dropout",
                self._dropout,
                "sampling needs a demapper built with dropout > 0",
            )
        passes = int(passes)
        if passes < 2:
            raise ValidationError("passes", passes, "need at least two passes")

        symbols = np.asarray(symbols, dtype=np.complex128).ravel()
        if symbols.size == 0:
            return np.empty((passes, 0), dtype=float)

        rows = torch.as_tensor(self._windows(symbols), dtype=torch.float32).to(
            self._device
        )

        # train() switches the dropout layers back on.  The linear
        # layers behave identically either way, so this changes the
        # stochasticity and nothing else -- and no gradients are taken,
        # so no weights move.
        self._net.train()
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(int(seed))
        try:
            with torch.no_grad():
                draws = [
                    torch.sigmoid(self._net(rows)).cpu().numpy().ravel()
                    for _ in range(passes)
                ]
        finally:
            # Leave the model and the global random state exactly as
            # they were found; a confidence estimate that perturbs the
            # experiment it is measuring is worse than no estimate.
            self._net.eval()
            torch.random.set_rng_state(generator_state)

        return np.stack(draws, axis=0).astype(float)

    def _windows(self, symbols: np.ndarray) -> np.ndarray:
        """Build one input row per symbol, padding the ends by wrapping.

        Wrapping matches the periodic convolution the pulse shaper uses,
        so the first and last symbols of a frame are treated the same as
        the ones in the middle rather than being quietly degraded.
        """
        count = symbols.size
        offsets = np.arange(-self._window, self._window + 1)
        index = (np.arange(count)[:, None] + offsets[None, :]) % count
        windows = symbols[index]

        rows = np.empty((count, self.n_features), dtype=np.float32)
        rows[:, 0::2] = np.real(windows)
        rows[:, 1::2] = np.imag(windows)
        return rows

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import torch

        self._require_trained()
        path = Path(path).with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path)
        self._write_sidecar(
            path,
            {
                "model": self._name,
                "kind": "MLPDemapper",
                "modulation": self._modulation.name,
                "window": self._window,
                "hidden": list(self._hidden),
                "dropout": self._dropout,
                "n_parameters": self.n_parameters,
                "history": self._history.to_dict(),
            },
        )
        return path

    def load(self, path: str | Path) -> "MLPDemapper":
        import torch

        path = Path(path).with_suffix(".pt")
        if not path.exists():
            raise ValidationError("path", str(path), "no saved model at this path")
        self._net.load_state_dict(
            torch.load(path, map_location=self._device, weights_only=True)
        )
        self._net.to(self._device)
        self._mark_trained()
        return self

    def describe(self) -> str:
        state = "trained" if self.is_trained else "untrained"
        return (
            f"{self._name} ({state}, {self.width}-symbol window, "
            f"{self.n_parameters:,} weights, {self._device})"
        )


class CNNDemodulator(TrainableModel):
    """A convolutional network taking raw samples straight to bits.

    The ambitious one.  It is handed the received waveform with the
    frequency offset, timing error and fading still present, and asked
    for the transmitted bits -- which means rediscovering, from examples
    alone, what Phase 2 needed six designed stages to do.

    The architecture gives it a fighting chance rather than an easy
    ride.  Convolutions with stride reduce eight samples per symbol down
    to one, which is the same decimation the timing recovery performs,
    except that where the signal is sampled is learned rather than
    computed.  After that, a per-symbol head produces the bits.

    What it cannot be given is any notion of a constellation. It sees
    numbers.

    Setting expectations honestly: small networks tend to do well here
    at high signal-to-noise and fall apart where the classical chain
    still works, because the loops in that chain integrate information
    across a whole frame while a convolution sees only its own receptive
    field. Measuring that gap is the useful outcome.
    """

    def __init__(
        self,
        modulation: Modulation,
        samples_per_symbol: int = 8,
        channels: tuple[int, ...] = (32, 64, 64),
        learning_rate: float = 1e-3,
        seed: int = 0,
        device: Any = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(modulation, Modulation):
            raise ValidationError(
                "modulation", type(modulation).__name__, "expected a Modulation"
            )
        super().__init__(name or f"CNN-{modulation.name}")

        self._modulation = modulation
        self._sps = int(samples_per_symbol)
        self._channels = tuple(int(c) for c in channels)
        self._learning_rate = float(learning_rate)
        self._seed = int(seed)
        self._device = device or pick_device()

        import torch

        torch.manual_seed(self._seed)
        self._net = _CNNBody(
            samples_per_symbol=self._sps,
            channels=self._channels,
            bits_per_symbol=modulation.bits_per_symbol,
        ).to(self._device)

    @property
    def modulation(self) -> Modulation:
        """The scheme this model was built for."""
        return self._modulation

    @property
    def n_parameters(self) -> int:
        """How many weights the network has."""
        return sum(p.numel() for p in self._net.parameters())

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
        epochs: int = 60,
        batch_size: int = 32,
        patience: int = 12,
        verbose: bool = True,
        **_: Any,
    ) -> TrainingHistory:
        """Learn from ``(IQ waveform, bits)`` pairs."""
        import torch
        import torch.nn as nn

        inputs, targets = self._check_pair(inputs, targets)
        if inputs.ndim != 3 or inputs.shape[1] != 2:
            raise ValidationError(
                "inputs", inputs.shape, "expected shape (frames, 2, samples)"
            )

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._learning_rate)
        callbacks: list[Any] = [
            EarlyStopping(
                monitor="val_loss" if validation else "loss",
                patience=patience,
                snapshot=lambda: {
                    k: v.detach().clone() for k, v in self._net.state_dict().items()
                },
                restore=self._net.load_state_dict,
            )
        ]
        if verbose:
            from .trainer import ProgressPrinter

            callbacks.insert(0, ProgressPrinter(every=5, name=self._name))

        trainer = Trainer(
            model=self._net,
            loss_fn=nn.BCEWithLogitsLoss(),
            optimizer=optimizer,
            device=self._device,
            callbacks=callbacks,
            name=self._name,
        )

        val_inputs, val_targets = validation if validation else (None, None)
        history = trainer.fit(
            inputs,
            targets,
            val_inputs,
            val_targets,
            epochs=epochs,
            batch_size=batch_size,
            seed=self._seed,
        )
        self._history = history
        self._mark_trained()
        return history

    def predict_bits(self, iq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bits, confidence and log-likelihood ratios for one or more frames."""
        import torch

        self._require_trained()
        iq = np.asarray(iq, dtype=np.float32)
        if iq.ndim == 2:
            iq = iq[None, ...]

        self._net.eval()
        with torch.no_grad():
            logits = (
                self._net(torch.as_tensor(iq).to(self._device)).cpu().numpy()
            )

        llr = np.clip(-logits, -_MAX_LLR, _MAX_LLR)
        bits = (llr <= 0).astype(np.uint8)
        confidence = 1.0 / (1.0 + np.exp(-np.abs(llr)))
        return bits, confidence, llr

    def save(self, path: str | Path) -> Path:
        import torch

        self._require_trained()
        path = Path(path).with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path)
        self._write_sidecar(
            path,
            {
                "model": self._name,
                "kind": "CNNDemodulator",
                "modulation": self._modulation.name,
                "samples_per_symbol": self._sps,
                "channels": list(self._channels),
                "n_parameters": self.n_parameters,
                "history": self._history.to_dict(),
            },
        )
        return path

    def load(self, path: str | Path) -> "CNNDemodulator":
        import torch

        path = Path(path).with_suffix(".pt")
        if not path.exists():
            raise ValidationError("path", str(path), "no saved model at this path")
        self._net.load_state_dict(
            torch.load(path, map_location=self._device, weights_only=True)
        )
        self._net.to(self._device)
        self._mark_trained()
        return self

    def describe(self) -> str:
        state = "trained" if self.is_trained else "untrained"
        return (
            f"{self._name} ({state}, {self.n_parameters:,} weights, {self._device})"
        )


def _make_cnn_body():
    """Defined lazily so importing this module does not require torch."""
    import torch.nn as nn

    class CNNBody(nn.Module):
        """Strided convolutions down to the symbol rate, then a bit head."""

        def __init__(self, samples_per_symbol, channels, bits_per_symbol):
            super().__init__()
            self.bits_per_symbol = bits_per_symbol

            # Factor the decimation into strides of 2, so the network
            # reduces to one position per symbol in stages rather than
            # in one brutal step.
            strides: list[int] = []
            remaining = samples_per_symbol
            while remaining > 1 and remaining % 2 == 0:
                strides.append(2)
                remaining //= 2
            if remaining > 1:
                strides.append(remaining)

            layers: list[nn.Module] = []
            in_channels = 2
            for index, out_channels in enumerate(channels):
                stride = strides[index] if index < len(strides) else 1
                layers += [
                    nn.Conv1d(
                        in_channels, out_channels, kernel_size=7,
                        stride=stride, padding=3,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
                in_channels = out_channels

            # Anything left over after the channel list runs out.
            for stride in strides[len(channels):]:
                layers += [
                    nn.Conv1d(
                        in_channels, in_channels, kernel_size=7,
                        stride=stride, padding=3,
                    ),
                    nn.BatchNorm1d(in_channels),
                    nn.ReLU(),
                ]

            self.features = nn.Sequential(*layers)
            # One 1x1 convolution turns each symbol position into its
            # bits, which keeps the model independent of frame length.
            self.head = nn.Conv1d(in_channels, bits_per_symbol, kernel_size=1)

        def forward(self, x):
            x = self.features(x)
            x = self.head(x)                   # (batch, bits, symbols)
            return x.transpose(1, 2).flatten(1)  # (batch, symbols * bits)

    return CNNBody


class _CNNLazy:
    """Builds the network class on first use, keeping torch optional."""

    _cls: Any = None

    def __call__(self, **kwargs: Any) -> Any:
        if _CNNLazy._cls is None:
            _CNNLazy._cls = _make_cnn_body()
        return _CNNLazy._cls(**kwargs)


_CNNBody = _CNNLazy()


__all__ = ["CNNDemodulator", "MLPDemapper"]
