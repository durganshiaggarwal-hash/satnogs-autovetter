"""Registry and factory for modulation schemes.

DESIGN NOTE: THE FACTORY PATTERN
--------------------------------
The GUI sends the backend a string -- ``"16QAM"`` -- because that is what
a dropdown produces and what JSON can carry.  Something has to turn that
string into an object.

The naive version of that is a chain of ``if`` statements, repeated
wherever it is needed:

    if name == "BPSK":  return BPSK()
    elif name == "QPSK": return QPSK()
    ...

:class:`ModulationRegistry` replaces it with a lookup table built once.
Adding 32APSK later means registering one class; no existing code
changes.  That is the Open/Closed Principle in practice: the system is
open to extension but closed to modification.

The registry also *caches* instances.  Modulation objects are immutable
and building a 64QAM constellation involves real work, so handing out the
same shared instance is both safe and faster.
"""

from __future__ import annotations

from typing import Callable

from ..core.exceptions import UnknownModulationError, ValidationError
from .apsk import APSK16
from .base import Modulation
from .psk import BPSK, PSK8, QPSK, PhaseShiftKeying
from .qam import QAM16, QAM64, QuadratureAmplitudeModulation


class ModulationRegistry:
    """Maps names to modulation classes and caches the instances."""

    # Canonical name -> zero-argument callable that builds the scheme.
    _factories: dict[str, Callable[[], Modulation]] = {}

    # Alternative spellings the GUI or a dataset file might use.
    _aliases: dict[str, str] = {}

    # Shared instances, built on first request.
    _cache: dict[str, Modulation] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    @classmethod
    def register(
        cls,
        name: str,
        factory: Callable[[], Modulation],
        aliases: tuple[str, ...] = (),
    ) -> None:
        """Add a scheme to the registry."""
        if not callable(factory):
            raise ValidationError("factory", factory, "must be callable")

        key = cls._normalise(name)
        cls._factories[key] = factory
        cls._cache.pop(key, None)
        for alias in aliases:
            cls._aliases[cls._normalise(alias)] = key

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    @classmethod
    def create(cls, name: str) -> Modulation:
        """Return the modulation scheme called ``name``.

        Raises :class:`UnknownModulationError` -- which carries the list
        of valid names -- rather than a bare ``KeyError``, so the API can
        turn it straight into a helpful HTTP 400 response.
        """
        if not isinstance(name, str):
            raise ValidationError("name", type(name).__name__, "expected a string")

        key = cls._resolve(name)
        if key is None:
            raise UnknownModulationError(name, cls.available())

        if key not in cls._cache:
            cls._cache[key] = cls._factories[key]()
        return cls._cache[key]

    @classmethod
    def available(cls) -> list[str]:
        """Display names of every registered scheme, in a sensible order.

        Sorted by constellation size so the GUI dropdown runs from the
        most robust scheme to the most fragile, which is the order a user
        naturally explores.
        """
        schemes = [cls.create(key) for key in cls._factories]
        schemes.sort(key=lambda m: (m.bits_per_symbol, m.name))
        return [scheme.name for scheme in schemes]

    @classmethod
    def all_schemes(cls) -> list[Modulation]:
        """Every registered scheme as an object, in the same order."""
        return [cls.create(name) for name in cls.available()]

    @classmethod
    def describe_all(cls) -> list[dict[str, object]]:
        """Summary of every scheme, for the ``GET /modulations`` endpoint."""
        return [
            {
                "name": scheme.name,
                "family": _family_of(scheme),
                "order": scheme.order,
                "bits_per_symbol": scheme.bits_per_symbol,
                "minimum_distance": round(scheme.minimum_distance, 4),
                "has_theory_curve": scheme.theoretical_ber(10.0) is not None,
            }
            for scheme in cls.all_schemes()
        ]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """``True`` if ``name`` (or an alias of it) is known."""
        return isinstance(name, str) and cls._resolve(name) is not None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @classmethod
    def _resolve(cls, name: str) -> str | None:
        key = cls._normalise(name)
        if key in cls._factories:
            return key
        return cls._aliases.get(key)

    @staticmethod
    def _normalise(name: str) -> str:
        """Fold case, spaces, hyphens and underscores so lookup is forgiving."""
        return (
            str(name)
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        )


# --------------------------------------------------------------------------
# Default registrations -- the six schemes this project compares.
#
# Chosen to span the whole robustness/throughput range (1 to 6 bits per
# symbol) and all three constellation families, while staying inside the
# 22 schemes RML24 covers.
# --------------------------------------------------------------------------
ModulationRegistry.register("BPSK", BPSK, aliases=("2psk", "bpsk"))
ModulationRegistry.register("QPSK", QPSK, aliases=("4psk", "qpsk"))
ModulationRegistry.register("8PSK", PSK8, aliases=("psk8", "8psk"))
ModulationRegistry.register("16QAM", QAM16, aliases=("qam16", "16qam"))
ModulationRegistry.register("64QAM", QAM64, aliases=("qam64", "64qam"))
ModulationRegistry.register("16APSK", APSK16, aliases=("apsk16", "16apsk"))


def _family_of(scheme: Modulation) -> str:
    """Name of the most general non-abstract ancestor of ``scheme``.

    Walks the method resolution order to find the family class -- e.g.
    ``BPSK`` reports ``PhaseShiftKeying``.  This is introspection used
    for documentation rather than for control flow: nothing in the system
    behaves differently based on the answer.
    """
    chain = [
        klass
        for klass in type(scheme).__mro__
        if isinstance(klass, type)
        and issubclass(klass, Modulation)
        and klass is not Modulation
    ]
    return chain[-1].__name__ if chain else type(scheme).__name__


def get_modulation(name: str) -> Modulation:
    """Convenience wrapper around :meth:`ModulationRegistry.create`."""
    return ModulationRegistry.create(name)


__all__ = [
    "APSK16",
    "BPSK",
    "Modulation",
    "ModulationRegistry",
    "PSK8",
    "PhaseShiftKeying",
    "QAM16",
    "QAM64",
    "QPSK",
    "QuadratureAmplitudeModulation",
    "get_modulation",
]
