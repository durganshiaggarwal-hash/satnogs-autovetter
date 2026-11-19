"""The :class:`Signal` value object.

DESIGN NOTE: ENCAPSULATION
--------------------------
A :class:`Signal` wraps a NumPy array of complex samples.  The array is
stored in a *private* attribute ``_samples`` and is never handed out in a
writable form:

* ``signal.samples`` returns a **read-only view**, so callers can read
  and compute but cannot corrupt the object's state behind its back.
* There is no ``samples`` setter.  To get a modified signal you call
  :meth:`Signal.with_samples`, which returns a *new* object.

This makes ``Signal`` effectively immutable, which matters a lot here:
a signal frame travels through a chain of channel effects and then
through several receivers.  If any one of them could mutate the array in
place, the comparison between receivers would silently become unfair.
Immutability makes that class of bug impossible rather than unlikely.
"""

from __future__ import annotations

import numpy as np

from .exceptions import EmptySignalError, ValidationError


class Signal:
    """A block of complex baseband samples with a known sample rate.

    Parameters
    ----------
    samples:
        Complex baseband samples.  Real input is accepted and promoted to
        complex.
    sample_rate:
        Sampling frequency in hertz.  Must be strictly positive.
    description:
        Optional free-text note, e.g. ``"after AWGN"``.  Purely for
        debugging and GUI labels.
    """

    # ``__slots__`` stops callers from inventing new attributes on an
    # instance, which is another small piece of encapsulation: the set of
    # fields a Signal has is fixed by the class, not by its users.
    __slots__ = ("_samples", "_sample_rate", "_description")

    def __init__(
        self,
        samples: np.ndarray,
        sample_rate: float,
        description: str = "",
    ) -> None:
        # Always copy.  ``np.ascontiguousarray`` would skip the copy when
        # the input already has the right dtype and layout, leaving this
        # Signal sharing memory with the caller's array -- and a caller
        # who then edited their own array would silently change our
        # supposedly immutable state.  Immutability has to be owned
        # memory, not just a read-only view over someone else's.
        samples = np.array(samples, dtype=np.complex128, copy=True, order="C")

        if samples.ndim != 1:
            raise ValidationError(
                "samples", samples.shape, "expected a one-dimensional array"
            )
        if samples.size == 0:
            raise EmptySignalError("a Signal must contain at least one sample")
        if not np.all(np.isfinite(samples)):
            raise ValidationError(
                "samples", "<array>", "contains NaN or infinite values"
            )
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValidationError(
                "sample_rate", sample_rate, "must be a positive finite number"
            )

        self._samples = samples
        self._sample_rate = float(sample_rate)
        self._description = str(description)

    # ------------------------------------------------------------------
    # Read-only properties -- the public face of the object
    # ------------------------------------------------------------------
    @property
    def samples(self) -> np.ndarray:
        """Read-only view of the complex samples.

        Attempting ``signal.samples[0] = 5`` raises ``ValueError``, which
        is exactly what we want: the only legitimate way to change a
        signal is to build a new one.
        """
        view = self._samples.view()
        view.flags.writeable = False
        return view

    @property
    def sample_rate(self) -> float:
        """Sampling frequency in hertz."""
        return self._sample_rate

    @property
    def description(self) -> str:
        """Human-readable note about this signal's provenance."""
        return self._description

    @property
    def duration(self) -> float:
        """Length of the signal in seconds."""
        return self._samples.size / self._sample_rate

    @property
    def real(self) -> np.ndarray:
        """In-phase (I) component."""
        return np.real(self._samples)

    @property
    def imag(self) -> np.ndarray:
        """Quadrature (Q) component."""
        return np.imag(self._samples)

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------
    def power(self) -> float:
        """Average power, i.e. the mean of ``|x[n]|**2``."""
        return float(np.mean(np.abs(self._samples) ** 2))

    def peak_amplitude(self) -> float:
        """Largest sample magnitude in the block."""
        return float(np.max(np.abs(self._samples)))

    def normalized(self, target_power: float = 1.0) -> "Signal":
        """Return a copy scaled to ``target_power`` average power."""
        if target_power <= 0:
            raise ValidationError(
                "target_power", target_power, "must be positive"
            )
        current = self.power()
        if current == 0.0:
            return self
        scale = np.sqrt(target_power / current)
        return self.with_samples(
            self._samples * scale, description=f"{self._description}|normalized"
        )

    def with_samples(
        self, samples: np.ndarray, description: str | None = None
    ) -> "Signal":
        """Return a new :class:`Signal` with the same sample rate.

        This is the replacement for a setter.  Channel effects use it so
        that each stage produces a fresh immutable object and the chain
        of transformations stays traceable.
        """
        return Signal(
            samples,
            self._sample_rate,
            self._description if description is None else description,
        )

    def as_iq_array(self) -> np.ndarray:
        """Return the samples as a real ``(2, N)`` array of I and Q rows.

        This is the layout RML24 uses and the layout our neural networks
        expect, so it is the standard hand-off point between the signal
        world and the machine-learning world.
        """
        return np.stack([self.real, self.imag], axis=0)

    @classmethod
    def from_iq_array(
        cls, iq: np.ndarray, sample_rate: float, description: str = ""
    ) -> "Signal":
        """Build a :class:`Signal` from a real ``(2, N)`` I/Q array.

        Design note: this is an *alternative constructor*.  Rather than
        overloading ``__init__`` with flags, we expose a named classmethod
        whose purpose is obvious at the call site.
        """
        iq = np.asarray(iq)
        if iq.ndim != 2 or iq.shape[0] != 2:
            raise ValidationError(
                "iq", iq.shape, "expected an array of shape (2, N)"
            )
        return cls(iq[0] + 1j * iq[1], sample_rate, description)

    # ------------------------------------------------------------------
    # Python protocol methods (operator overloading = polymorphism)
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self._samples.size)

    def __getitem__(self, index: int | slice) -> "complex | Signal":
        """Index a single sample, or slice out a sub-signal."""
        if isinstance(index, slice):
            return self.with_samples(self._samples[index])
        return complex(self._samples[index])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Signal):
            return NotImplemented
        return (
            self._sample_rate == other._sample_rate
            and self._samples.shape == other._samples.shape
            and bool(np.allclose(self._samples, other._samples))
        )

    def __repr__(self) -> str:
        note = f", description={self._description!r}" if self._description else ""
        return (
            f"Signal(n={len(self)}, sample_rate={self._sample_rate:g} Hz, "
            f"power={self.power():.4g}{note})"
        )
