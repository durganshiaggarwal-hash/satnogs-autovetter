"""Run every receiver on every scenario, fairly, and keep the results.

The fairness rules are few and enforced here rather than trusted to
whoever writes an experiment:

1. **Every receiver sees the identical frames.**  Each frame is drawn
   once from the source and handed to each receiver in turn.  Any
   difference between two receivers is then the receivers' doing, and
   the paired bootstrap can cancel the frame-to-frame luck they share.
2. **Every receiver is graded by the same code**, in
   :meth:`CellResult.from_outcomes`.  There is no path by which one
   method's numbers are computed differently from another's.
3. **Nothing is skipped silently.**  A receiver that cannot handle a
   modulation, or a source that cannot produce a scenario, is recorded
   as skipped, with the reason, and the skip appears in the summary.

DESIGN NOTE: THE OBSERVER PATTERN
---------------------------------
A long benchmark should report progress, but the runner has no business
knowing *how*: printing to a terminal, moving a progress bar in the
GUI, or pushing an update over the API's websocket are three different
jobs.  So the runner announces events to any number of
:class:`BenchmarkObserver` objects and each decides what to do.  The
same pattern Phase 4 used for training callbacks, for the same reason.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable

import numpy as np

from ..confidence.base import ConfidenceEstimator
from ..core.exceptions import ValidationError
from ..receivers.base import Receiver
from .records import CellResult
from .scenarios import Scenario, ScenarioGrid
from .sources import FrameSource
from .statistics import Comparison, FrameBootstrap


# ======================================================================
# Observers
# ======================================================================
class BenchmarkObserver:
    """Receives progress events. Override whichever hooks are needed.

    Not abstract on purpose: every hook has a do-nothing default, so an
    observer that only cares about the end of the run implements one
    method rather than four.
    """

    def on_start(self, n_scenarios: int, receivers: list[str]) -> None:
        """Called once, before the first scenario."""

    def on_cell(self, cell: CellResult, done: int, total: int) -> None:
        """Called after each receiver finishes each scenario."""

    def on_skip(self, receiver: str, scenario: Scenario, reason: str) -> None:
        """Called when a cell cannot be run."""

    def on_finish(self, result: "BenchmarkResult") -> None:
        """Called once, after the last scenario."""


class ConsoleProgress(BenchmarkObserver):
    """Prints one line per cell. What the demo scripts use."""

    def __init__(self, every: int = 1) -> None:
        self._every = max(1, int(every))
        self._started = 0.0

    def on_start(self, n_scenarios, receivers):
        self._started = time.perf_counter()
        print(f"  {n_scenarios} scenarios x {len(receivers)} receivers")

    def on_cell(self, cell, done, total):
        if done % self._every == 0 or done == total:
            elapsed = time.perf_counter() - self._started
            print(
                f"  [{done:>4}/{total}] {cell.receiver:<12s} {cell.scenario!r:<34s} "
                f"BER {cell.ber:.2e}  ({elapsed:.0f}s)"
            )

    def on_skip(self, receiver, scenario, reason):
        print(f"  skipped {receiver} on {scenario!r}: {reason}")


class CallbackObserver(BenchmarkObserver):
    """Forwards progress to a plain function: the hook the GUI will use.

    ``callback(done, total, cell)`` is called after every cell.
    """

    def __init__(self, callback: Callable[[int, int, CellResult], Any]) -> None:
        if not callable(callback):
            raise ValidationError("callback", type(callback).__name__, "must be callable")
        self._callback = callback

    def on_cell(self, cell, done, total):
        self._callback(done, total, cell)


# ======================================================================
# Results
# ======================================================================
class BenchmarkResult:
    """Every cell of one benchmark run, with ways to ask questions of it."""

    def __init__(
        self,
        cells: Iterable[CellResult],
        skipped: Iterable[tuple[str, Scenario, str]] = (),
        source: str = "",
    ) -> None:
        self._cells: dict[tuple[str, Scenario], CellResult] = {}
        for cell in cells:
            self._cells[(cell.receiver, cell.scenario)] = cell
        self._skipped = list(skipped)
        self._source = str(source)

    @property
    def source(self) -> str:
        return self._source

    @property
    def cells(self) -> list[CellResult]:
        return list(self._cells.values())

    @property
    def skipped(self) -> list[tuple[str, Scenario, str]]:
        """Every (receiver, scenario, reason) that was not run."""
        return list(self._skipped)

    @property
    def receivers(self) -> list[str]:
        return list(dict.fromkeys(r for r, _ in self._cells))

    @property
    def scenarios(self) -> ScenarioGrid:
        return ScenarioGrid(s for _, s in self._cells)

    def cell(self, receiver: str, scenario: Scenario) -> CellResult:
        """One cell, or a clear error naming what was asked for."""
        try:
            return self._cells[(receiver, scenario)]
        except KeyError:
            raise ValidationError(
                "cell", f"{receiver} / {scenario!r}", "not in this result"
            ) from None

    def select(
        self,
        receiver: str | None = None,
        modulation: str | None = None,
        profile: str | None = None,
    ) -> list[CellResult]:
        """Cells matching every filter given, in scenario order."""
        chosen = [
            c
            for c in self._cells.values()
            if (receiver is None or c.receiver == receiver)
            and (modulation is None or c.scenario.modulation == modulation)
            and (profile is None or c.scenario.profile.name == profile)
        ]
        return sorted(chosen, key=lambda c: (c.scenario, c.receiver))

    def curve(
        self, receiver: str, modulation: str, profile: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A bit-error-rate curve: (snr, ber, interval) for plotting."""
        cells = self.select(receiver, modulation, profile)
        snr = np.array([c.scenario.snr_db for c in cells])
        ber = np.array([c.ber for c in cells])
        interval = np.array([c.ber_interval() for c in cells]).reshape(-1, 2)
        return snr, ber, interval

    def compare(
        self,
        first: str,
        second: str,
        scenarios: Iterable[Scenario] | None = None,
    ) -> Comparison:
        """Paired comparison of two receivers, pooled over scenarios.

        Valid because both saw the identical frames, in the same order,
        in every scenario -- which the runner guarantees.
        """
        chosen = list(scenarios) if scenarios is not None else list(self.scenarios)
        a, b, bits = [], [], []
        for scenario in chosen:
            if (first, scenario) not in self._cells or (second, scenario) not in self._cells:
                continue
            ca, cb = self._cells[(first, scenario)], self._cells[(second, scenario)]
            a.append(ca.frame_errors)
            b.append(cb.frame_errors)
            bits.append(ca.frame_bits)
        if not a:
            raise ValidationError(
                "scenarios", len(chosen), f"none were run for both {first} and {second}"
            )
        return FrameBootstrap().compare(
            np.concatenate(a), np.concatenate(b), np.concatenate(bits), (first, second)
        )

    def rows(self) -> list[dict[str, Any]]:
        """Every cell as a flat dictionary: the results table."""
        return [c.as_dict() for c in sorted(self._cells.values(), key=lambda c: (c.scenario, c.receiver))]

    def __add__(self, other: "BenchmarkResult") -> "BenchmarkResult":
        """Merge two runs, e.g. the same grid on two data sources."""
        if not isinstance(other, BenchmarkResult):
            return NotImplemented
        source = self._source if self._source == other._source else "mixed"
        return BenchmarkResult(
            list(self._cells.values()) + list(other._cells.values()),
            self._skipped + other._skipped,
            source,
        )

    def __len__(self) -> int:
        return len(self._cells)

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult({len(self)} cells, {len(self.receivers)} receivers, "
            f"{len(self._skipped)} skipped)"
        )


# ======================================================================
# The runner
# ======================================================================
class BenchmarkRunner:
    """Grades a set of receivers on a grid of scenarios.

    Parameters
    ----------
    source:
        Where frames come from.
    receivers:
        Mapping from display name to receiver.  A dictionary rather than
        a list so that two configurations of the same class -- a
        classical receiver with and without its equaliser -- can sit side
        by side under different names.
    frames_per_scenario:
        How many frames each cell is measured on.
    estimator:
        How per-bit confidence is read from each result.  ``None`` uses
        the confidence each receiver reports itself, which is the fair
        default: it grades the receiver as built.
    """

    def __init__(
        self,
        source: FrameSource,
        receivers: dict[str, Receiver],
        frames_per_scenario: int = 20,
        estimator: ConfidenceEstimator | None = None,
    ) -> None:
        if not isinstance(source, FrameSource):
            raise ValidationError("source", type(source).__name__, "expected a FrameSource")
        if not receivers:
            raise ValidationError("receivers", {}, "need at least one receiver")
        for name, receiver in receivers.items():
            if not isinstance(receiver, Receiver):
                raise ValidationError(
                    "receivers", f"{name}: {type(receiver).__name__}", "expected a Receiver"
                )
        if int(frames_per_scenario) < 1:
            raise ValidationError(
                "frames_per_scenario", frames_per_scenario, "must be at least 1"
            )
        if estimator is not None and not isinstance(estimator, ConfidenceEstimator):
            raise ValidationError(
                "estimator", type(estimator).__name__, "expected a ConfidenceEstimator"
            )

        self._source = source
        self._receivers = dict(receivers)
        self._frames = int(frames_per_scenario)
        self._estimator = estimator
        self._observers: list[BenchmarkObserver] = []

    def add_observer(self, observer: BenchmarkObserver) -> "BenchmarkRunner":
        """Register someone to be told about progress. Returns self for chaining."""
        if not isinstance(observer, BenchmarkObserver):
            raise ValidationError(
                "observer", type(observer).__name__, "expected a BenchmarkObserver"
            )
        self._observers.append(observer)
        return self

    def _notify(self, event: str, *args: Any) -> None:
        for observer in self._observers:
            getattr(observer, event)(*args)

    @staticmethod
    def _can_run(receiver: Receiver, scenario: Scenario) -> str | None:
        """Why a receiver cannot handle a scenario, or ``None`` if it can.

        Uses an optional ``supports`` method -- duck typing -- so that a
        receiver which handles everything need not implement anything.
        """
        supports = getattr(receiver, "supports", None)
        if callable(supports) and not supports(scenario.modulation):
            return f"no member for {scenario.modulation}"
        return None

    def run(self, grid: ScenarioGrid | Iterable[Scenario]) -> BenchmarkResult:
        """Measure every receiver on every scenario in ``grid``."""
        grid = grid if isinstance(grid, ScenarioGrid) else ScenarioGrid(grid)
        if len(grid) == 0:
            raise ValidationError("grid", 0, "nothing to run")

        names = list(self._receivers)
        total = len(grid) * len(names)
        self._notify("on_start", len(grid), names)

        cells: list[CellResult] = []
        skipped: list[tuple[str, Scenario, str]] = []
        done = 0

        for scenario in grid:
            if not self._source.supports(scenario):
                for name in names:
                    reason = f"{self._source.name} cannot produce this scenario"
                    skipped.append((name, scenario, reason))
                    self._notify("on_skip", name, scenario, reason)
                done += len(names)
                continue

            active = {}
            for name in names:
                reason = self._can_run(self._receivers[name], scenario)
                if reason:
                    skipped.append((name, scenario, reason))
                    self._notify("on_skip", name, scenario, reason)
                    done += 1
                else:
                    active[name] = self._receivers[name]

            confidences = {n: [] for n in active}
            correct = {n: [] for n in active}
            seconds = {n: 0.0 for n in active}

            # Rule 1: draw each frame once, hand it to every receiver.
            for frame in self._source.frames(scenario, self._frames):
                truth = np.asarray(frame.bits)
                for name, receiver in active.items():
                    started = time.perf_counter()
                    result = receiver.recover(frame)
                    seconds[name] += time.perf_counter() - started
                    scores = (
                        result.confidence
                        if self._estimator is None
                        else self._estimator.estimate(result)
                    )
                    confidences[name].append(np.asarray(scores, dtype=float))
                    correct[name].append(~result.error_mask(truth))

            # Rule 2: grade everyone with the same code.
            for name in active:
                cell = CellResult.from_outcomes(
                    name,
                    scenario,
                    confidences[name],
                    correct[name],
                    seconds[name],
                    source=self._source.name,
                )
                cells.append(cell)
                done += 1
                self._notify("on_cell", cell, done, total)

        result = BenchmarkResult(cells, skipped, self._source.name)
        self._notify("on_finish", result)
        return result

    def __repr__(self) -> str:
        return (
            f"BenchmarkRunner({self._source.name}, receivers={list(self._receivers)}, "
            f"{self._frames} frames each)"
        )


__all__ = [
    "BenchmarkObserver",
    "BenchmarkResult",
    "BenchmarkRunner",
    "CallbackObserver",
    "ConsoleProgress",
]
