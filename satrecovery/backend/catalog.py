"""The receivers the API offers, by name.

A request says ``"receivers": ["classical", "hybrid"]``.  Something has
to turn those strings into configured receiver objects, know which ones
exist, and know that ``"hybrid"`` for 64QAM is only available once a
64QAM model has been trained.  That is this module's whole job.

DESIGN NOTE: FACTORY AND REGISTRY, AGAIN
----------------------------------------
Phase 1's :class:`~satrecovery.modulation.registry.ModulationRegistry`
maps a name to a class.  The catalog maps a name to a *builder* -- a
function that, given a modulation, returns a ready receiver -- because a
receiver needs configuring, and a hybrid needs a model loaded from disk
before it can be built at all.

Each entry also says whether it needs a trained model, so the catalog
can report honestly which receivers are usable right now rather than
offering one that will fail when asked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..core.exceptions import UnknownReceiverError, ValidationError
from ..modulation.registry import ModulationRegistry
from ..receivers.base import Receiver
from ..receivers.classical import ClassicalReceiver
from ..receivers.demapper import HardDemapper, NoiseNormalisedDemapper, SoftDemapper
from ..simulation.pulse import PulseShape
from .models import ModelStore

#: Given a modulation name, return a ready receiver.
Builder = Callable[[str], Receiver]


@dataclass(frozen=True)
class CatalogEntry:
    """One named receiver the API can build."""

    name: str
    label: str
    description: str
    builder: Builder
    needs_model: bool = False


class ReceiverCatalog:
    """Named receivers, built on request for a given modulation."""

    def __init__(self, pulse: PulseShape, models: ModelStore | None = None) -> None:
        if not isinstance(pulse, PulseShape):
            raise ValidationError("pulse", type(pulse).__name__, "expected a PulseShape")
        self._pulse = pulse
        self._models = models
        self._entries: dict[str, CatalogEntry] = {}

    def register(self, entry: CatalogEntry) -> "ReceiverCatalog":
        """Add a receiver. Names must be unique. Returns self for chaining."""
        if entry.name in self._entries:
            raise ValidationError("name", entry.name, "already registered")
        self._entries[entry.name] = entry
        return self

    # ------------------------------------------------------------------
    def names(self) -> list[str]:
        return list(self._entries)

    def entry(self, name: str) -> CatalogEntry:
        try:
            return self._entries[name]
        except KeyError:
            raise UnknownReceiverError(name, self.names()) from None

    def is_available(self, name: str, modulation: str) -> bool:
        """Can this receiver be built for this modulation right now?"""
        entry = self.entry(name)
        if not entry.needs_model:
            return True
        return self._models is not None and self._models.has(modulation)

    def describe(self, modulation: str | None = None) -> list[dict]:
        """Every receiver, and for which modulations it is ready."""
        schemes = (
            [ModulationRegistry.create(modulation).name]
            if modulation
            else ModulationRegistry.available()
        )
        return [
            {
                "name": e.name,
                "label": e.label,
                "description": e.description,
                "needs_model": e.needs_model,
                "available_for": [m for m in schemes if self.is_available(e.name, m)],
            }
            for e in self._entries.values()
        ]

    def build(self, name: str, modulation: str) -> Receiver:
        """A configured receiver for one modulation.

        Raises :class:`ModelNotTrainedError` (through the model store)
        when a hybrid is asked for before its model exists, which the
        API turns into a clear "train it first" response.
        """
        modulation = ModulationRegistry.create(modulation).name
        return self.entry(name).builder(modulation)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    @classmethod
    def standard(cls, pulse: PulseShape, models: ModelStore | None = None) -> "ReceiverCatalog":
        """The receivers this project compares."""
        catalog = cls(pulse, models)

        catalog.register(CatalogEntry(
            "classical", "Classical",
            "Matched filter, blind carrier and timing recovery, PLL, and a soft "
            "demapper that re-measures the noise on every frame.",
            lambda m: ClassicalReceiver(pulse=pulse, demapper=SoftDemapper(), name="classical"),
        ))
        catalog.register(CatalogEntry(
            "classical-hard", "Classical (hard decisions)",
            "The same chain with the simplest demapper: nearest point wins, "
            "confidence from the winning margin. Same bits, cruder confidence.",
            lambda m: ClassicalReceiver(pulse=pulse, demapper=HardDemapper(), name="classical-hard"),
        ))
        catalog.register(CatalogEntry(
            "classical-equalised", "Classical + equaliser",
            "The classical chain with an adaptive LMS equaliser added, for "
            "channels whose echoes arrive whole symbols apart.",
            lambda m: ClassicalReceiver(pulse=pulse, use_equalizer=True, name="classical-equalised"),
        ))

        if models is not None:
            def hybrid(modulation: str) -> Receiver:
                network, _ = models.load(modulation)
                return ClassicalReceiver(pulse=pulse, demapper=network, name="hybrid")

            def hybrid_noise(modulation: str) -> Receiver:
                network, reference = models.load(modulation)
                return ClassicalReceiver(
                    pulse=pulse,
                    demapper=NoiseNormalisedDemapper(network, reference),
                    name="hybrid-noise",
                )

            catalog.register(CatalogEntry(
                "hybrid", "Hybrid",
                "The classical chain with a trained neural network making the final "
                "decision from a window of five symbols.",
                hybrid, needs_model=True,
            ))
            catalog.register(CatalogEntry(
                "hybrid-noise", "Hybrid + per-frame noise",
                "The hybrid with its confidence rescaled by each frame's measured "
                "noise -- the Phase 6 fix. Same bits; better at spotting lost frames.",
                hybrid_noise, needs_model=True,
            ))
        return catalog


__all__ = ["CatalogEntry", "ReceiverCatalog"]
