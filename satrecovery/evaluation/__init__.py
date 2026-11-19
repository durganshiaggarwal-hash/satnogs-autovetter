"""Phase 6 -- grading every receiver the same way, and saying how sure we are.

``scenarios``
    What to test: impairment profiles, scenarios, and grids of them.
``sources``
    Where frames come from.  The simulator today; the RML24 recordings
    plug in beside it without anything else changing.
``runner``
    Runs every receiver on identical frames, grades them with identical
    code, and reports progress to observers.
``statistics``
    Intervals on error rates: the textbook Wilson interval, and a frame
    bootstrap that respects how errors cluster.  Paired comparisons
    between receivers.
``splits``
    Random and held-out divisions of a grid, for testing learned models
    on conditions they never saw.
``store``
    SQLite storage, so results outlive the run that made them.
"""

from __future__ import annotations

from .records import LOST_FRAME_BER, CellResult
from .runner import (
    BenchmarkObserver,
    BenchmarkResult,
    BenchmarkRunner,
    CallbackObserver,
    ConsoleProgress,
)
from .scenarios import ImpairmentProfile, Scenario, ScenarioGrid
from .sources import FrameSource, SimulatedSource, SourceRouter
from .splits import HoldOutProfile, HoldOutSNR, RandomSplit, SplitStrategy
from .statistics import Comparison, ErrorRate, FrameBootstrap
from .store import ResultStore

__all__ = [
    "BenchmarkObserver",
    "BenchmarkResult",
    "BenchmarkRunner",
    "CallbackObserver",
    "CellResult",
    "Comparison",
    "ConsoleProgress",
    "ErrorRate",
    "FrameBootstrap",
    "FrameSource",
    "HoldOutProfile",
    "HoldOutSNR",
    "ImpairmentProfile",
    "LOST_FRAME_BER",
    "RandomSplit",
    "ResultStore",
    "Scenario",
    "ScenarioGrid",
    "SimulatedSource",
    "SourceRouter",
    "SplitStrategy",
]
