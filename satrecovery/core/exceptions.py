"""Exception hierarchy for the satellite signal recovery system.

DESIGN NOTE: INHERITANCE
------------------------
Every exception in this project descends from a single root,
``SatRecoveryError``.  That gives callers a choice of how precise they
want to be:

    try:
        ...
    except UnknownModulationError:   # very specific
        ...
    except ModulationError:          # any modulation problem
        ...
    except SatRecoveryError:         # anything this project raises
        ...

Because Python matches ``except`` clauses against the *class hierarchy*,
a single ``except SatRecoveryError`` catches all of our errors while
still letting genuine Python bugs (``TypeError``, ``KeyError``, ...)
escape so we notice them.
"""

from __future__ import annotations


class SatRecoveryError(Exception):
    """Root of every exception raised by this project."""


# --------------------------------------------------------------------------
# Configuration / validation
# --------------------------------------------------------------------------
class ConfigurationError(SatRecoveryError):
    """A configuration value is missing, malformed or out of range."""


class ValidationError(ConfigurationError):
    """A value failed a validity check inside a property setter."""

    def __init__(self, field: str, value: object, reason: str) -> None:
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(f"invalid value for {field!r}: {value!r} ({reason})")


# --------------------------------------------------------------------------
# Modulation
# --------------------------------------------------------------------------
class ModulationError(SatRecoveryError):
    """Base class for problems with modulation schemes."""


class UnknownModulationError(ModulationError):
    """The requested modulation scheme is not registered."""

    def __init__(self, name: str, available: list[str] | None = None) -> None:
        self.name = name
        self.available = sorted(available or [])
        message = f"unknown modulation scheme {name!r}"
        if self.available:
            message += f"; available: {', '.join(self.available)}"
        super().__init__(message)


class BitLengthError(ModulationError):
    """The number of bits is not a whole multiple of bits-per-symbol."""

    def __init__(self, n_bits: int, bits_per_symbol: int) -> None:
        self.n_bits = n_bits
        self.bits_per_symbol = bits_per_symbol
        super().__init__(
            f"cannot map {n_bits} bits with {bits_per_symbol} bits/symbol: "
            f"{n_bits} is not a multiple of {bits_per_symbol}"
        )


# --------------------------------------------------------------------------
# Signals and channels
# --------------------------------------------------------------------------
class SignalError(SatRecoveryError):
    """Base class for problems with signal objects."""


class EmptySignalError(SignalError):
    """A signal was created with no samples."""


class ChannelError(SatRecoveryError):
    """Base class for problems inside a channel effect."""


class SampleRateMismatchError(ChannelError):
    """Two signals that must share a sample rate do not."""

    def __init__(self, first: float, second: float) -> None:
        super().__init__(
            f"sample rate mismatch: {first:g} Hz vs {second:g} Hz"
        )


# --------------------------------------------------------------------------
# Receivers (used from Phase 2 onwards)
# --------------------------------------------------------------------------
class ReceiverError(SatRecoveryError):
    """Base class for problems inside a receiver."""


class UnknownReceiverError(ReceiverError):
    """A receiver was asked for by a name nobody registered."""

    def __init__(self, name: str, available: list[str] | None = None) -> None:
        self.name = name
        self.available = sorted(available or [])
        message = f"unknown receiver {name!r}"
        if self.available:
            message += f"; available: {', '.join(self.available)}"
        super().__init__(message)


class ModelNotTrainedError(ReceiverError):
    """A learning-based receiver was used before it was trained."""

    def __init__(self, receiver_name: str) -> None:
        self.name = receiver_name
        super().__init__(
            f"{receiver_name!r} has not been trained yet; "
            f"call fit(...) or load(...) first"
        )
