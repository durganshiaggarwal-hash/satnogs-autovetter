"""What to test: scenarios, impairment profiles, and grids of both.

A benchmark is only as fair as its description of what was tested.  If
two receivers are compared on "16QAM at 6 dB", every reader has to be
able to reconstruct exactly what that meant -- which impairments, how
strong, how many frames.  This module makes that description a set of
objects instead of a set of loose arguments.

:class:`ImpairmentProfile`
    A named bundle of everything wrong with the channel *except* the
    noise level.  "clean", "doppler", "fading", "full".  Naming them
    is what lets a held-out test say "trained on clean, tested on full"
    and mean something precise.

:class:`Scenario`
    One cell of an experiment: a modulation, a signal-to-noise ratio and
    a profile.  Immutable and hashable, so it can be a dictionary key,
    a database key, and a set member, and two equal scenarios are
    guaranteed to describe the same test.

:class:`ScenarioGrid`
    Every combination of a set of modulations, ratios and profiles.  A
    benchmark runs a grid; a held-out split divides one.

DESIGN NOTE: VALUE OBJECTS AND OPERATOR OVERLOADING
---------------------------------------------------
``Scenario`` defines ``__eq__`` and ``__hash__`` from its contents, so
identity follows meaning: two scenarios built separately with the same
settings are equal and collide in a set.  ``ScenarioGrid`` overloads
``+`` (union), ``-`` (difference), ``len`` and iteration, so a held-out
split reads as plain arithmetic::

    test = grid - train
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator

import numpy as np

from ..core.conditions import ChannelConditions
from ..core.exceptions import ValidationError
from ..modulation.registry import ModulationRegistry


class ImpairmentProfile:
    """Everything wrong with a channel apart from its noise level.

    Parameters
    ----------
    name:
        Short label, used in tables and as part of a scenario's key.
    settings:
        Keyword arguments for
        :class:`~satrecovery.core.conditions.ChannelConditions`, minus
        ``snr_db``.  Validated immediately by building a test
        conditions object, so a misspelt field fails here rather than
        halfway through a long benchmark.
    """

    __slots__ = ("_name", "_settings")

    def __init__(self, name: str, **settings: Any) -> None:
        name = str(name).strip()
        if not name:
            raise ValidationError("name", name, "a profile needs a name")
        if "snr_db" in settings:
            raise ValidationError(
                "snr_db",
                settings["snr_db"],
                "the noise level belongs to the scenario, not the profile",
            )
        # Build once to validate every field and value up front.
        ChannelConditions(snr_db=10.0, **settings)
        self._name = name
        self._settings = dict(settings)

    @property
    def name(self) -> str:
        """Short label."""
        return self._name

    @property
    def settings(self) -> dict[str, Any]:
        """Copy of the channel settings this profile applies."""
        return dict(self._settings)

    def conditions(self, snr_db: float) -> ChannelConditions:
        """The full channel description at one noise level."""
        return ChannelConditions(snr_db=float(snr_db), **self._settings)

    # ------------------------------------------------------------------
    # The standard set used throughout the project.  Built by
    # classmethods rather than module constants so each call returns a
    # fresh object and nothing can be mutated behind a caller's back.
    # ------------------------------------------------------------------
    @classmethod
    def clean(cls) -> "ImpairmentProfile":
        """Noise and nothing else: the case the theory covers exactly."""
        return cls("clean")

    @classmethod
    def doppler(cls) -> "ImpairmentProfile":
        """Carrier frequency and phase error, as from a moving satellite."""
        return cls("doppler", freq_offset_hz=800.0, phase_offset_rad=0.7)

    @classmethod
    def fading(cls) -> "ImpairmentProfile":
        """Several delayed copies of the signal arriving together."""
        return cls("fading", fading="multipath_rician", rician_k_db=10.0)

    @classmethod
    def full(cls) -> "ImpairmentProfile":
        """Everything at once: the conditions Phase 5 found hardest."""
        return cls(
            "full",
            freq_offset_hz=800.0,
            phase_offset_rad=0.7,
            timing_offset=0.4,
            fading="multipath_rician",
            rician_k_db=10.0,
        )

    @classmethod
    def recorded(cls) -> "ImpairmentProfile":
        """Whatever the real channel did: RML24 recordings (Phase 3).

        It carries no settings because nothing about the channel is
        *chosen* -- the frequency error, drifting gain and pulse
        distortion are all in the recording.  Only a
        :class:`~satrecovery.data.source.RecordedSource` produces it;
        the simulator declines it rather than pass off clean frames as
        real ones.
        """
        return cls("recorded")

    @classmethod
    def standard(cls) -> list["ImpairmentProfile"]:
        """All four, in order of increasing difficulty."""
        return [cls.clean(), cls.doppler(), cls.fading(), cls.full()]

    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImpairmentProfile):
            return NotImplemented
        return self._name == other._name and self._settings == other._settings

    def __hash__(self) -> int:
        return hash((self._name, tuple(sorted(self._settings.items()))))

    def __repr__(self) -> str:
        return f"ImpairmentProfile({self._name!r})"


class Scenario:
    """One test condition: a modulation, a noise level and a profile."""

    __slots__ = ("_modulation", "_snr_db", "_profile")

    def __init__(
        self, modulation: str, snr_db: float, profile: ImpairmentProfile
    ) -> None:
        if not isinstance(profile, ImpairmentProfile):
            raise ValidationError(
                "profile", type(profile).__name__, "expected an ImpairmentProfile"
            )
        # Normalise through the registry so "qpsk" and "QPSK" are one
        # scenario rather than two that never meet in a comparison.
        self._modulation = ModulationRegistry.create(str(modulation)).name
        self._snr_db = float(snr_db)
        self._profile = profile
        # Fail now, not mid-run, if the noise level is out of range.
        self.conditions()

    @property
    def modulation(self) -> str:
        """Scheme name, as the registry spells it."""
        return self._modulation

    @property
    def snr_db(self) -> float:
        """Signal-to-noise ratio in decibels."""
        return self._snr_db

    @property
    def profile(self) -> ImpairmentProfile:
        """What else is wrong with the channel."""
        return self._profile

    @property
    def key(self) -> str:
        """Stable text identifier, for databases and seeding.

        Formatted with a fixed number of decimals so that 6, 6.0 and
        6.000 produce the same key and therefore the same frames.
        """
        return f"{self._modulation}|{self._snr_db:+.2f}|{self._profile.name}"

    def conditions(self) -> ChannelConditions:
        """The channel this scenario describes."""
        return self._profile.conditions(self._snr_db)

    def as_dict(self) -> dict[str, Any]:
        """Plain fields, for tables and storage."""
        return {
            "modulation": self._modulation,
            "snr_db": self._snr_db,
            "profile": self._profile.name,
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Scenario):
            return NotImplemented
        return self.key == other.key and self._profile == other._profile

    def __hash__(self) -> int:
        return hash(self.key)

    def __lt__(self, other: "Scenario") -> bool:
        """Sort by modulation, then profile, then noise level."""
        if not isinstance(other, Scenario):
            return NotImplemented
        return (self._modulation, self._profile.name, self._snr_db) < (
            other._modulation,
            other._profile.name,
            other._snr_db,
        )

    def __repr__(self) -> str:
        return (
            f"Scenario({self._modulation}, {self._snr_db:+g} dB, "
            f"{self._profile.name})"
        )


class ScenarioGrid:
    """An ordered set of scenarios, usually every combination of three lists.

    Behaves like a set with a stable order: duplicates are dropped,
    iteration is sorted, and ``+``/``-`` are union and difference.
    """

    __slots__ = ("_scenarios",)

    def __init__(self, scenarios: Iterable[Scenario] = ()) -> None:
        unique: dict[Scenario, None] = {}
        for item in scenarios:
            if not isinstance(item, Scenario):
                raise ValidationError(
                    "scenarios", type(item).__name__, "every member must be a Scenario"
                )
            unique[item] = None
        self._scenarios = tuple(sorted(unique))

    @classmethod
    def product(
        cls,
        modulations: Iterable[str],
        snrs_db: Iterable[float],
        profiles: Iterable[ImpairmentProfile] | None = None,
    ) -> "ScenarioGrid":
        """Every combination of modulation, noise level and profile."""
        modulations = list(modulations)
        snrs_db = [float(s) for s in snrs_db]
        profiles = list(profiles) if profiles is not None else [ImpairmentProfile.clean()]
        for name, values in (
            ("modulations", modulations),
            ("snrs_db", snrs_db),
            ("profiles", profiles),
        ):
            if not values:
                raise ValidationError(name, [], "cannot build a grid from an empty list")
        return cls(
            Scenario(m, s, p) for m in modulations for p in profiles for s in snrs_db
        )

    # ------------------------------------------------------------------
    @property
    def modulations(self) -> list[str]:
        """Distinct modulations, in order."""
        return list(dict.fromkeys(s.modulation for s in self._scenarios))

    @property
    def snrs_db(self) -> list[float]:
        """Distinct noise levels, ascending."""
        return sorted({s.snr_db for s in self._scenarios})

    @property
    def profiles(self) -> list[str]:
        """Distinct profile names, in order of first appearance."""
        return list(dict.fromkeys(s.profile.name for s in self._scenarios))

    def filter(self, predicate: Callable[[Scenario], bool]) -> "ScenarioGrid":
        """The scenarios for which ``predicate`` is true."""
        return ScenarioGrid(s for s in self._scenarios if predicate(s))

    def where(
        self,
        modulation: str | None = None,
        profile: str | None = None,
        snr_min: float = -np.inf,
        snr_max: float = np.inf,
    ) -> "ScenarioGrid":
        """The common filters, without writing a lambda each time."""
        wanted = ModulationRegistry.create(modulation).name if modulation else None
        return self.filter(
            lambda s: (wanted is None or s.modulation == wanted)
            and (profile is None or s.profile.name == profile)
            and snr_min <= s.snr_db <= snr_max
        )

    # ------------------------------------------------------------------
    # Set arithmetic
    # ------------------------------------------------------------------
    def __add__(self, other: "ScenarioGrid") -> "ScenarioGrid":
        if not isinstance(other, ScenarioGrid):
            return NotImplemented
        return ScenarioGrid(self._scenarios + other._scenarios)

    def __sub__(self, other: "ScenarioGrid") -> "ScenarioGrid":
        if not isinstance(other, ScenarioGrid):
            return NotImplemented
        removed = set(other._scenarios)
        return ScenarioGrid(s for s in self._scenarios if s not in removed)

    def __and__(self, other: "ScenarioGrid") -> "ScenarioGrid":
        if not isinstance(other, ScenarioGrid):
            return NotImplemented
        kept = set(other._scenarios)
        return ScenarioGrid(s for s in self._scenarios if s in kept)

    def __contains__(self, item: object) -> bool:
        return item in self._scenarios

    def __iter__(self) -> Iterator[Scenario]:
        return iter(self._scenarios)

    def __len__(self) -> int:
        return len(self._scenarios)

    def __getitem__(self, index: int) -> Scenario:
        return self._scenarios[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScenarioGrid):
            return NotImplemented
        return self._scenarios == other._scenarios

    def __hash__(self) -> int:
        return hash(self._scenarios)

    def __repr__(self) -> str:
        return (
            f"ScenarioGrid({len(self)} scenarios: {len(self.modulations)} "
            f"modulations x {len(self.snrs_db)} SNRs x {len(self.profiles)} profiles)"
        )


__all__ = ["ImpairmentProfile", "Scenario", "ScenarioGrid"]
