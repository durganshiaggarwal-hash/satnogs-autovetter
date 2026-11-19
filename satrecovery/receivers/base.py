"""The receiver framework: stages, chains and the Receiver contract.

DESIGN NOTE: A SYMMETRY WORTH NOTICING
--------------------------------------
Phase 1 built :class:`~satrecovery.channel.base.ChannelEffect` and
:class:`~satrecovery.channel.pipeline.ChannelPipeline`: a set of
interchangeable impairments, chained together, that *break* a signal.

This module is the mirror image.  :class:`ReceiverStage` and
:class:`ReceiverChain` are a set of interchangeable repairs, chained
together, that *fix* one.  Same abstract base plus composite shape, same
Strategy and Template Method patterns, opposite purpose::

    channel  = ChannelPipeline([Fading, FreqOffset, Noise])     # breaks
    receiver = ReceiverChain([MatchedFilter, Carrier, Timing])  # repairs

The symmetry is not decoration.  Because both sides are built from small
swappable objects, a Phase 5 hybrid receiver can keep the first three
repair stages and replace only the last one with a neural network --
without rewriting anything, and without the surrounding code noticing.

One deliberate difference: a channel effect must preserve the number of
samples, while a receiver stage may change it.  Timing recovery consumes
eight samples per symbol and emits one, which is the whole point of it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from ..core.exceptions import ValidationError
from ..core.frame import Frame
from ..core.signal import Signal
from ..modulation.base import Modulation
from ..modulation.registry import ModulationRegistry
from .result import RecoveryResult


class RecoveryContext:
    """Working state shared down a receiver chain.

    A stage needs to know more than the samples in front of it: which
    constellation is expected, how many samples there currently are per
    symbol, and -- for the stages that need a reference -- the known
    preamble symbols.

    It also collects **diagnostics**.  Each stage records what it worked
    out (the frequency offset it estimated, the timing it settled on),
    and those end up in the :class:`RecoveryResult`.  That turns the
    receiver from a black box into something you can interrogate when a
    result looks wrong, which during development is most of the time.

    Unlike the value objects in ``core``, this one is deliberately
    mutable: it is a scratchpad, and the stages write on it as they go.
    """

    __slots__ = (
        "_modulation",
        "samples_per_symbol",
        "sample_rate",
        "_pilot_symbols",
        "_diagnostics",
    )

    def __init__(
        self,
        modulation: Modulation,
        samples_per_symbol: int,
        sample_rate: float,
        pilot_symbols: np.ndarray | None = None,
    ) -> None:
        if not isinstance(modulation, Modulation):
            raise ValidationError(
                "modulation",
                type(modulation).__name__,
                "expected a Modulation instance",
            )
        self._modulation = modulation
        self.samples_per_symbol = int(samples_per_symbol)
        self.sample_rate = float(sample_rate)
        self._pilot_symbols = (
            None
            if pilot_symbols is None
            else np.asarray(pilot_symbols, dtype=np.complex128).ravel()
        )
        self._diagnostics: dict[str, Any] = {}

    @property
    def modulation(self) -> Modulation:
        """The constellation the receiver is decoding against."""
        return self._modulation

    @property
    def pilot_symbols(self) -> np.ndarray | None:
        """Known symbols at the start of the frame, if the link uses them.

        Every real link sends a preamble the receiver already knows --
        it is how the receiver works out which way up the constellation
        is.  See :class:`~satrecovery.receivers.classical.ClassicalReceiver`
        for why that cannot be done blind.
        """
        return self._pilot_symbols

    @property
    def n_pilot(self) -> int:
        """Length of the known preamble, in symbols."""
        return 0 if self._pilot_symbols is None else int(self._pilot_symbols.size)

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Everything the stages have reported so far."""
        return dict(self._diagnostics)

    def record(self, key: str, value: Any) -> None:
        """Note something a stage worked out."""
        self._diagnostics[str(key)] = value

    def __repr__(self) -> str:
        return (
            f"RecoveryContext({self._modulation.name}, "
            f"sps={self.samples_per_symbol}, pilot={self.n_pilot})"
        )


class ReceiverStage(ABC):
    """One repair applied to a received signal.

    The same Template Method split as
    :class:`~satrecovery.channel.base.ChannelEffect`: ``apply`` is
    concrete and handles the bookkeeping, ``_process`` is abstract and
    holds the mathematics.
    """

    def __init__(self, name: str | None = None) -> None:
        self._name = str(name) if name else type(self).__name__

    def apply(self, signal: Signal, context: RecoveryContext) -> Signal:
        """Return a new :class:`Signal` with this stage applied."""
        if not isinstance(signal, Signal):
            raise ValidationError(
                "signal", type(signal).__name__, "expected a Signal instance"
            )
        if not isinstance(context, RecoveryContext):
            raise ValidationError(
                "context",
                type(context).__name__,
                "expected a RecoveryContext instance",
            )

        samples, sample_rate = self._process(
            signal.samples, signal.sample_rate, context
        )
        samples = np.asarray(samples, dtype=np.complex128).ravel()
        if samples.size == 0:
            raise ValidationError(
                f"{self._name}.output", 0, "a stage must emit at least one sample"
            )

        previous = signal.description
        description = f"{previous}|{self._name}" if previous else self._name
        context.sample_rate = float(sample_rate)
        return Signal(samples, sample_rate, description=description)

    def __call__(self, signal: Signal, context: RecoveryContext) -> Signal:
        return self.apply(signal, context)

    @abstractmethod
    def _process(
        self,
        samples: np.ndarray,
        sample_rate: float,
        context: RecoveryContext,
    ) -> tuple[np.ndarray, float]:
        """Transform the samples.

        Returns the new samples **and** their sample rate, because a
        stage is allowed to change it: timing recovery decimates to one
        sample per symbol, so the rate that comes out is the symbol rate.
        A stage that changes the rate must also update
        ``context.samples_per_symbol``.
        """

    @property
    def name(self) -> str:
        """Short name, used in signal descriptions and diagnostics."""
        return self._name

    def parameters(self) -> dict[str, Any]:
        """Settings of this stage, for logging and the API."""
        return {}

    def describe(self) -> str:
        """One-line human-readable summary."""
        params = self.parameters()
        if not params:
            return self._name
        inner = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self._name}({inner})"

    def __repr__(self) -> str:
        return self.describe()


class ReceiverChain(ReceiverStage):
    """An ordered sequence of repairs, which is itself a repair.

    The Composite pattern again, for the same payoff: a chain can be
    dropped in anywhere a single stage is accepted, and a chain can
    contain another chain.  Phase 5's hybrid receiver reuses this
    directly -- it takes the classical chain, drops the last stage, and
    bolts a neural network on the end.
    """

    def __init__(
        self, stages: Sequence[ReceiverStage] = (), name: str = "ReceiverChain"
    ) -> None:
        super().__init__(name)
        for position, stage in enumerate(stages):
            if not isinstance(stage, ReceiverStage):
                raise ValidationError(
                    f"stages[{position}]",
                    type(stage).__name__,
                    "every element must be a ReceiverStage",
                )
        self._stages: list[ReceiverStage] = list(stages)

    @property
    def stages(self) -> tuple[ReceiverStage, ...]:
        """The chain as an immutable tuple."""
        return tuple(self._stages)

    def add(self, stage: ReceiverStage) -> "ReceiverChain":
        """Append a stage and return ``self``, so calls can be chained."""
        if not isinstance(stage, ReceiverStage):
            raise ValidationError(
                "stage", type(stage).__name__, "expected a ReceiverStage"
            )
        self._stages.append(stage)
        return self

    def _process(self, samples, sample_rate, context):
        signal = Signal(samples, sample_rate)
        for stage in self._stages:
            signal = stage.apply(signal, context)
        return signal.samples, signal.sample_rate

    def __len__(self) -> int:
        return len(self._stages)

    def __iter__(self) -> Iterator[ReceiverStage]:
        return iter(self._stages)

    def __getitem__(self, index: int) -> ReceiverStage:
        return self._stages[index]

    def parameters(self) -> dict[str, Any]:
        return {
            "stages": [
                {"stage": s.name, **s.parameters()} for s in self._stages
            ]
        }

    def describe(self) -> str:
        if not self._stages:
            return "ReceiverChain(pass-through)"
        return " -> ".join(stage.describe() for stage in self._stages)


class Receiver(ABC):
    """Anything that turns a received :class:`Frame` into recovered bits.

    This is the single most important abstraction in the project.  Every
    method we compare -- the classical chain of Phase 2, the neural
    networks of Phase 4, the hybrid of Phase 5 -- is a subclass with the
    same one-method interface::

        result = receiver.recover(frame)

    Phase 6's evaluator holds a list of these and loops over it.  It has
    no idea which is which, and that is precisely what makes the
    comparison fair: there is no code path where one method gets treated
    differently from another.
    """

    def __init__(self, name: str | None = None) -> None:
        self._name = str(name) if name else type(self).__name__

    def recover(self, frame: Frame) -> RecoveryResult:
        """Recover the transmitted bits from ``frame``.

        A Template Method: it validates the input, delegates the real
        work to :meth:`_recover`, and checks the answer is well formed
        before handing it back.  Subclasses implement only ``_recover``,
        and get the contract enforcement for free.
        """
        if not isinstance(frame, Frame):
            raise ValidationError(
                "frame", type(frame).__name__, "expected a Frame instance"
            )

        result = self._recover(frame)

        if not isinstance(result, RecoveryResult):
            raise ValidationError(
                f"{self._name}._recover",
                type(result).__name__,
                "must return a RecoveryResult",
            )
        if frame.graded_symbols is not None and result.n_bits != frame.n_bits:
            # Some of a recorded frame's symbols are not graded (see
            # Frame.graded_symbols). Selecting here, once, means no
            # receiver can forget to, and none can be graded on a
            # different set of symbols from the others.  "Once" matters:
            # a receiver that wraps others (ReceiverBank) gets back an
            # answer its member already trimmed, and must not trim again.
            result = result.select_symbols(
                frame.graded_symbols,
                ModulationRegistry.create(frame.modulation).bits_per_symbol,
            )
            if result.n_bits != frame.n_bits:
                raise ValidationError(
                    f"{self._name}._recover",
                    result.n_bits,
                    f"after selecting the graded symbols, expected {frame.n_bits} bits",
                )
        return result

    @abstractmethod
    def _recover(self, frame: Frame) -> RecoveryResult:
        """Do the actual work. Implemented by every concrete receiver."""

    @property
    def name(self) -> str:
        """Short name, used to label rows in the results table."""
        return self._name

    def describe(self) -> str:
        """One-line summary of how this receiver is configured."""
        return self._name

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self._name!r})"


__all__ = [
    "Receiver",
    "ReceiverChain",
    "ReceiverStage",
    "RecoveryContext",
]
