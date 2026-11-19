"""Abstract base class for channel effects.

DESIGN NOTE: THE STRATEGY PATTERN
---------------------------------
Everything that can happen to a signal between the satellite's amplifier
and the ground station's demodulator -- thermal noise, multipath fading,
Doppler shift, oscillator drift -- is modelled as a :class:`ChannelEffect`.

Each one is a separate class with the same interface, so they are
*interchangeable strategies*.  The simulator holds a list of them and
applies them in order; it never asks what kind each one is.  Adding phase
noise or an interfering carrier later means writing one new class.

DESIGN NOTE: THE TEMPLATE METHOD PATTERN
----------------------------------------
``apply()`` is concrete and ``_process()`` is abstract.  The base class
therefore owns all the bookkeeping that is identical for every effect --
type checking, defaulting the random generator, rebuilding an immutable
:class:`~satrecovery.core.signal.Signal`, appending a description -- and
each subclass writes only the few lines of mathematics that make it
different.

The leading underscore marks ``_process`` as internal: it is part of the
contract between this class and its subclasses, not part of the public
API.  Callers use ``apply()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..core.exceptions import ValidationError
from ..core.signal import Signal


class ChannelEffect(ABC):
    """One physical impairment that can be applied to a signal."""

    def __init__(self, name: str | None = None) -> None:
        self._name = str(name) if name else type(self).__name__

    # ------------------------------------------------------------------
    # Public interface -- the template method
    # ------------------------------------------------------------------
    def apply(
        self, signal: Signal, rng: np.random.Generator | None = None
    ) -> Signal:
        """Return a new :class:`Signal` with this effect applied.

        The input signal is never modified; every stage produces a fresh
        immutable object, so a frame's history stays reconstructable and
        two receivers can never interfere with each other's input.
        """
        if not isinstance(signal, Signal):
            raise ValidationError(
                "signal", type(signal).__name__, "expected a Signal instance"
            )
        if rng is None:
            rng = np.random.default_rng()
        elif not isinstance(rng, np.random.Generator):
            raise ValidationError(
                "rng", type(rng).__name__, "expected a numpy Generator"
            )

        samples = self._process(signal.samples, signal.sample_rate, rng)
        samples = np.asarray(samples, dtype=np.complex128)

        if samples.shape != signal.samples.shape:
            raise ValidationError(
                f"{self._name}.output",
                samples.shape,
                f"effects must preserve length (expected {signal.samples.shape})",
            )

        previous = signal.description
        description = f"{previous}|{self._name}" if previous else self._name
        return signal.with_samples(samples, description=description)

    def __call__(
        self, signal: Signal, rng: np.random.Generator | None = None
    ) -> Signal:
        """Allow effects to be used as plain callables.

        Operator overloading is a form of polymorphism: an effect can now
        be passed anywhere a function is expected, so
        ``list(map(effect, signals))`` works.
        """
        return self.apply(signal, rng)

    # ------------------------------------------------------------------
    # What subclasses must provide
    # ------------------------------------------------------------------
    @abstractmethod
    def _process(
        self,
        samples: np.ndarray,
        sample_rate: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Transform the raw sample array.

        Receives a read-only array, so implementations must build and
        return a new one rather than writing in place.  The returned array
        must have the same length as the input.
        """

    # ------------------------------------------------------------------
    # Introspection, shared by all effects
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Short name of this effect, used in signal descriptions."""
        return self._name

    def parameters(self) -> dict[str, Any]:
        """Settings of this effect, for logging and the JSON API.

        The default returns nothing; effects with parameters override it.
        """
        return {}

    def describe(self) -> str:
        """One-line human-readable summary, e.g. for GUI tooltips."""
        params = self.parameters()
        if not params:
            return self._name
        inner = ", ".join(f"{key}={value}" for key, value in params.items())
        return f"{self._name}({inner})"

    def __repr__(self) -> str:
        return self.describe()
