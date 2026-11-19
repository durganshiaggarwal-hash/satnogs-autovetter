"""Chaining channel effects together.

DESIGN NOTE: THE COMPOSITE PATTERN
----------------------------------
:class:`ChannelPipeline` is the most interesting class in this layer
because it is two things at once:

* it **is a** :class:`~satrecovery.channel.base.ChannelEffect`
  (inheritance), so anywhere a single effect is accepted, a whole chain
  of effects can be passed instead; and
* it **has a** list of :class:`ChannelEffect` objects (composition),
  which is how it does its work.

That combination is the Composite pattern.  Its payoff is that nothing
else in the program needs to distinguish "one effect" from "a chain of
effects" -- a pipeline can even contain another pipeline, and the
simulator would not notice.

DESIGN NOTE: COMPOSITION OVER INHERITANCE
-----------------------------------------
Note what we did *not* do: build a ``NoisyFadedOffsetChannel`` class by
inheriting from three effect classes at once.  That approach needs a new
class for every combination -- eight classes for three effects, and the
method resolution order becomes a maze.  Composing a list of small
objects at runtime gives every combination for free.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from ..core.conditions import ChannelConditions
from ..core.exceptions import ValidationError
from ..core.signal import Signal
from .base import ChannelEffect
from .effects import (
    AdditiveWhiteGaussianNoise,
    CarrierFrequencyOffset,
    CarrierPhaseOffset,
    FlatRicianFading,
    MultipathRicianFading,
    SampleTimingOffset,
)


class ChannelPipeline(ChannelEffect):
    """An ordered chain of channel effects, which is itself an effect."""

    def __init__(
        self,
        effects: Sequence[ChannelEffect] = (),
        name: str = "ChannelPipeline",
    ) -> None:
        super().__init__(name)

        for position, effect in enumerate(effects):
            if not isinstance(effect, ChannelEffect):
                raise ValidationError(
                    f"effects[{position}]",
                    type(effect).__name__,
                    "every element must be a ChannelEffect",
                )
        self._effects: list[ChannelEffect] = list(effects)

    # ------------------------------------------------------------------
    # Composite behaviour
    # ------------------------------------------------------------------
    @property
    def effects(self) -> tuple[ChannelEffect, ...]:
        """The chain as an immutable tuple, so callers cannot reorder it."""
        return tuple(self._effects)

    def add(self, effect: ChannelEffect) -> "ChannelPipeline":
        """Append an effect and return ``self`` so calls can be chained.

        ``pipeline.add(fading).add(offset).add(noise)`` reads in the same
        order the signal experiences the impairments, which makes the
        physical ordering obvious at the call site.
        """
        if not isinstance(effect, ChannelEffect):
            raise ValidationError(
                "effect", type(effect).__name__, "expected a ChannelEffect"
            )
        self._effects.append(effect)
        return self

    def _process(self, samples, sample_rate, rng):
        """Run every effect in order.

        Each stage is applied through the sub-effect's own public
        ``apply`` method, so each one still gets its validation and its
        description bookkeeping.  The pipeline adds no special cases: it
        does not know or care what the effects are.
        """
        signal = Signal(samples, sample_rate)
        for effect in self._effects:
            signal = effect.apply(signal, rng)
        return signal.samples

    # ------------------------------------------------------------------
    # Container protocol -- more operator overloading
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._effects)

    def __iter__(self) -> Iterator[ChannelEffect]:
        return iter(self._effects)

    def __getitem__(self, index: int) -> ChannelEffect:
        return self._effects[index]

    def parameters(self) -> dict[str, Any]:
        return {
            "stages": [
                {"effect": effect.name, **effect.parameters()}
                for effect in self._effects
            ]
        }

    def describe(self) -> str:
        if not self._effects:
            return "ChannelPipeline(ideal channel)"
        return " -> ".join(effect.describe() for effect in self._effects)


class ChannelFactory:
    """Builds a :class:`ChannelPipeline` from a declarative description.

    DESIGN NOTE: THE FACTORY PATTERN, SEPARATING *WHAT* FROM *HOW*
    --------------------------------------------------------------
    The GUI knows what the user wants: "10 dB signal-to-noise ratio,
    multipath fading, 200 Hz of Doppler".  It expresses that as a
    :class:`~satrecovery.core.conditions.ChannelConditions` object.  It
    should not have to know which classes implement those impairments, nor
    the order they must be applied in.

    This factory holds that knowledge in exactly one place.  The physical
    ordering below is not arbitrary and is the kind of domain rule that
    must not be duplicated across a GUI, an API and a test suite.
    """

    @staticmethod
    def from_conditions(conditions: ChannelConditions) -> ChannelPipeline:
        """Assemble the pipeline described by ``conditions``.

        The stages are applied in the order the physics dictates:

        1. **Timing offset** -- the propagation delay, which leaves the
           receiver sampling at the wrong instants.
        2. **Fading** -- happens along the propagation path.
        3. **Frequency offset** -- Doppler accumulates over that path, and
           the receiver's own oscillator error adds to it.
        4. **Phase offset** -- the unknown absolute carrier phase.
        5. **Noise** -- added *last*, because thermal noise is generated
           in the receiver's own front end. Adding it before the fading
           stage would let the fading attenuate the noise along with the
           signal, which would quietly overstate the link quality.
        """
        if not isinstance(conditions, ChannelConditions):
            raise ValidationError(
                "conditions",
                type(conditions).__name__,
                "expected a ChannelConditions instance",
            )

        pipeline = ChannelPipeline(name="SatelliteChannel")

        if conditions.timing_offset != 0.0:
            pipeline.add(SampleTimingOffset(conditions.timing_offset))

        if conditions.fading == "flat_rician":
            pipeline.add(FlatRicianFading(conditions.rician_k_db))
        elif conditions.fading == "multipath_rician":
            pipeline.add(MultipathRicianFading(conditions.rician_k_db))

        if conditions.freq_offset_hz != 0.0:
            pipeline.add(CarrierFrequencyOffset(conditions.freq_offset_hz))

        if conditions.phase_offset_rad != 0.0:
            pipeline.add(CarrierPhaseOffset(conditions.phase_offset_rad))

        pipeline.add(AdditiveWhiteGaussianNoise(conditions.snr_db))

        return pipeline

    @staticmethod
    def ideal() -> ChannelPipeline:
        """An empty pipeline -- a perfect channel, for sanity checks.

        Running a receiver through this must give a bit error rate of
        exactly zero.  If it does not, the bug is in the receiver, not in
        the channel, and that is worth being able to establish in one
        line.
        """
        return ChannelPipeline(name="IdealChannel")


__all__ = ["ChannelFactory", "ChannelPipeline"]
