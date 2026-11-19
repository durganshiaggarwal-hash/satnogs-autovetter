"""Where test frames come from.

The benchmark should not care whether a frame was simulated a moment ago
or recorded over the air two years ago.  It asks a *source* for frames
matching a scenario and grades whatever comes back.

There are two: the simulator here, and
:class:`~satrecovery.data.source.RecordedSource`, added in Phase 3 once
the RML24 recordings were downloaded.  Every experiment reruns on real
data without a line of the runner changing -- the whole reason this
module exists.

DESIGN NOTE: ABSTRACTION THAT ANTICIPATES A SECOND IMPLEMENTATION
------------------------------------------------------------------
:class:`FrameSource` is written before the second implementation exists,
and it says exactly what a real-data source will need to say:

* :meth:`~FrameSource.frames` -- give me this many frames for this
  scenario;
* :meth:`~FrameSource.supports` -- can you produce that scenario at all?

The second method is there because real data cannot do everything a
simulator can.  RML24 has no "clean" profile -- every recording carries
whatever the real channel did to it -- so a recorded source will decline
scenarios a simulated one accepts, and the runner skips them honestly
rather than silently substituting something else.
"""

from __future__ import annotations

import zlib
from abc import ABC, abstractmethod
from typing import Iterator

from ..core.exceptions import ValidationError
from ..core.frame import Frame
from ..simulation.simulator import SignalSimulator
from .scenarios import Scenario


class FrameSource(ABC):
    """Anything that can produce labelled frames for a scenario."""

    def __init__(self, name: str | None = None) -> None:
        self._name = str(name) if name else type(self).__name__

    @property
    def name(self) -> str:
        """Short label, stored with every result so its origin is known."""
        return self._name

    def frames(self, scenario: Scenario, count: int) -> Iterator[Frame]:
        """Yield ``count`` frames for ``scenario``.

        Template method: checks the request once, then hands over to the
        subclass.  A generator rather than a list, so a long benchmark
        never holds more than one frame in memory at a time.
        """
        if not isinstance(scenario, Scenario):
            raise ValidationError(
                "scenario", type(scenario).__name__, "expected a Scenario"
            )
        count = int(count)
        if count < 1:
            raise ValidationError("count", count, "must be at least 1")
        if not self.supports(scenario):
            raise ValidationError(
                "scenario", repr(scenario), f"{self._name} cannot produce it"
            )
        yield from self._frames(scenario, count)

    @abstractmethod
    def supports(self, scenario: Scenario) -> bool:
        """Can this source produce frames for ``scenario``?"""

    @abstractmethod
    def _frames(self, scenario: Scenario, count: int) -> Iterator[Frame]:
        """Produce the frames. The request has already been checked."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._name!r})"


class SimulatedSource(FrameSource):
    """Frames from the project's own simulator.

    **Reproducible per scenario.**  Every scenario gets its own random
    seed, derived from the source's base seed and the scenario's key.
    Two consequences, both deliberate:

    * Running a grid in a different order, or running half of it, gives
      exactly the same frames for each scenario.  A result can be
      reproduced by rerunning just the scenario in question.
    * Two benchmarks with the same seed see identical frames, so a
      difference between them is the receivers' doing and not luck.

    The seed is derived with CRC-32 rather than Python's ``hash``,
    because ``hash`` of a string changes from one interpreter run to the
    next and would quietly make every run unrepeatable.
    """

    def __init__(
        self,
        seed: int = 0,
        simulator_options: dict | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name or "simulated")
        self._seed = int(seed)
        self._options = dict(simulator_options or {})
        self._options.pop("seed", None)
        # One simulator built up front, so its settings can be inspected
        # and so a bad option fails at construction rather than mid-run.
        self._reference = SignalSimulator(seed=self._seed, **self._options)

    @property
    def seed(self) -> int:
        """Base seed every scenario's own seed is derived from."""
        return self._seed

    @property
    def simulator(self) -> SignalSimulator:
        """A simulator with this source's settings, for training and pulses."""
        return self._reference

    def scenario_seed(self, scenario: Scenario) -> int:
        """The seed this scenario's frames are drawn with."""
        return (self._seed * 1_000_003 + zlib.crc32(scenario.key.encode())) % (2**32)

    def supports(self, scenario: Scenario) -> bool:
        # Everything except real recordings: asked for "recorded", a
        # simulator could only hand back simulated frames under a false
        # name.
        return scenario.profile.name != "recorded"

    def _frames(self, scenario, count):
        simulator = SignalSimulator(seed=self.scenario_seed(scenario), **self._options)
        conditions = scenario.conditions()
        for _ in range(count):
            yield simulator.generate_frame(scenario.modulation, conditions)


class SourceRouter(FrameSource):
    """Several sources behind one, each scenario sent to the first that can.

    DESIGN NOTE: COMPOSITE
    ----------------------
    A router *is* a :class:`FrameSource` and *holds* several, so the
    benchmark runner -- which takes exactly one source -- can run a grid
    mixing simulated profiles and real recordings without knowing there
    is more than one.  Each scenario goes to the first source that
    :meth:`supports` it; the order given is the order of preference.
    """

    def __init__(self, sources: list[FrameSource], name: str | None = None) -> None:
        sources = list(sources)
        if not sources or not all(isinstance(s, FrameSource) for s in sources):
            raise ValidationError("sources", sources, "needs one or more FrameSource objects")
        super().__init__(name or "+".join(s.name for s in sources))
        self._sources = sources

    @property
    def sources(self) -> list[FrameSource]:
        return list(self._sources)

    def source_for(self, scenario: Scenario) -> FrameSource | None:
        """Which source a scenario would go to, or ``None``."""
        return next((s for s in self._sources if s.supports(scenario)), None)

    def supports(self, scenario: Scenario) -> bool:
        return self.source_for(scenario) is not None

    def _frames(self, scenario, count):
        yield from self.source_for(scenario).frames(scenario, count)


__all__ = ["FrameSource", "SimulatedSource", "SourceRouter"]
