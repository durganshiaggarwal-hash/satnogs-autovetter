"""What the API actually does, with no HTTP in sight.

Every endpoint is a thin wrapper around a method here.  The split is
deliberate: these classes can be called directly from a test, a script
or the GUI without a web server, and the web layer stays small enough to
read in one sitting.

DESIGN NOTE: LAYERED ARCHITECTURE
---------------------------------
Three layers, each depending only on the one below::

    app.py        HTTP: routes, status codes, JSON           (FastAPI)
      |
    services.py   use cases: run an experiment, start a job  (this file)
      |
    satrecovery   the science: receivers, channels, metrics  (Phases 1-6)

Nothing in Phases 1-6 imports anything from here.  The science does not
know it is being served over HTTP, which is why every test written
before this phase still runs unchanged.
"""

from __future__ import annotations

import math
import secrets
import threading
import time
from typing import Any

import numpy as np

from ..confidence.metrics import CalibrationReport, RiskCoverageCurve
from ..core.exceptions import ModelNotTrainedError, SatRecoveryError, ValidationError
from ..core.frame import Frame
from ..evaluation.runner import BenchmarkRunner, CallbackObserver
from ..evaluation.scenarios import ImpairmentProfile, ScenarioGrid
from ..evaluation.sources import SimulatedSource, SourceRouter
from ..evaluation.statistics import ErrorRate
from ..evaluation.store import ResultStore
from ..modulation.registry import ModulationRegistry
from ..receivers.bank import ReceiverBank
from ..receivers.result import RecoveryResult
from ..simulation.simulator import SignalSimulator
from .catalog import ReceiverCatalog
from .jobs import Job, JobManager
from .models import ModelStore, TrainingSpec
from .schemas import BenchmarkRequest, ExperimentRequest, StagesRequest, TrainRequest
from .settings import Settings


# ======================================================================
# JSON helpers
# ======================================================================
def jsonable(value: Any) -> Any:
    """Turn numpy types, NaN and infinities into plain JSON values.

    JSON has no NaN.  Python's encoder writes one anyway, producing a
    file other languages refuse to parse, so every NaN becomes ``null``.
    """
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, complex):
        return {"i": value.real, "q": value.imag}
    if value is None or isinstance(value, str):
        return value
    return str(value)


def iq(samples: np.ndarray) -> dict[str, list[float]]:
    samples = np.asarray(samples, dtype=np.complex128).ravel()
    return {"i": samples.real.tolist(), "q": samples.imag.tolist()}


# ======================================================================
# Experiments: one frame, several receivers
# ======================================================================
class ExperimentService:
    """The interactive part: build one frame and show what each receiver makes of it."""

    def __init__(self, catalog: ReceiverCatalog) -> None:
        self._catalog = catalog

    @staticmethod
    def _seed(seed: int | None) -> int:
        """Use the seed given, or pick one and report it.

        Every response carries the seed it used, so a result someone
        found interesting in the GUI can be reproduced exactly by
        sending the same request with that seed.
        """
        return int(seed) if seed is not None else secrets.randbelow(2**31)

    def _frame(self, modulation: str, conditions: Any, seed: int) -> Frame:
        simulator = SignalSimulator(seed=seed)
        return simulator.generate_frame(modulation, conditions.to_domain())

    def _frame_out(self, frame: Frame, seed: int) -> dict[str, Any]:
        scheme = ModulationRegistry.create(frame.modulation)
        tx_symbols = frame.tx_symbols
        transmitted = SignalSimulator(seed=0).pulse.at_rate(frame.samples_per_symbol).shape(tx_symbols)
        return {
            "modulation": scheme.name,
            "conditions": jsonable(frame.conditions.to_dict()),
            "seed": seed,
            "n_symbols": int(tx_symbols.size),
            "n_bits": int(frame.n_bits),
            "samples_per_symbol": int(frame.samples_per_symbol),
            "sample_rate": float(frame.signal.sample_rate),
            "received": iq(frame.signal.samples),
            "transmitted": iq(transmitted),
            "tx_symbols": iq(tx_symbols),
            "constellation": iq(scheme.constellation),
        }

    def _result_out(
        self, name: str, result: RecoveryResult, frame: Frame, seconds: float
    ) -> dict[str, Any]:
        truth = np.asarray(frame.bits)
        wrong = result.error_mask(truth)
        correct = ~wrong
        bps = ModulationRegistry.create(frame.modulation).bits_per_symbol
        errors = int(wrong.sum())
        low, high = ErrorRate(errors, result.n_bits).interval()
        counts, edges = np.histogram(result.confidence, bins=20, range=(0.5, 1.0))

        return {
            "receiver": name,
            "label": self._catalog.entry(name).label,
            "n_bits": result.n_bits,
            "bit_errors": errors,
            "ber": result.ber(truth),
            "ber_low": low,
            "ber_high": high,
            "mean_confidence": result.mean_confidence,
            "ece": CalibrationReport(result.confidence, correct).ece,
            "ranking_quality": (
                RiskCoverageCurve(result.confidence, correct).ranking_quality if errors else None
            ),
            "evm_percent": jsonable(result.diagnostics.get("evm_percent")),
            "symbols": iq(result.symbols if result.symbols is not None else []),
            "symbol_wrong": wrong[: (wrong.size // bps) * bps].reshape(-1, bps).any(axis=1).tolist(),
            "confidence_histogram": {"edges": edges.tolist(), "counts": counts.tolist()},
            "seconds": round(seconds, 4),
            "diagnostics": jsonable(result.diagnostics),
        }

    def run(self, request: ExperimentRequest) -> dict[str, Any]:
        """Every requested receiver, on the *same* frame."""
        names = list(dict.fromkeys(request.receivers))   # drop repeats, keep order
        for name in names:
            self._catalog.entry(name)                      # unknown name -> 404 before any work
        seed = self._seed(request.seed)
        frame = self._frame(request.modulation, request.conditions, seed)

        results = []
        for name in names:
            receiver = self._catalog.build(name, frame.modulation)
            started = time.perf_counter()
            result = receiver.recover(frame)
            results.append(self._result_out(name, result, frame, time.perf_counter() - started))
        return {"frame": self._frame_out(frame, seed), "results": results}

    def stages(self, request: StagesRequest) -> dict[str, Any]:
        """One receiver, with the signal shown after every stage."""
        seed = self._seed(request.seed)
        frame = self._frame(request.modulation, request.conditions, seed)
        receiver = self._catalog.build(request.receiver, frame.modulation)
        trace = getattr(receiver, "trace", None)
        if not callable(trace):
            raise ValidationError("receiver", request.receiver, "does not support stage tracing")

        started = time.perf_counter()
        result, snapshots = trace(frame)
        seconds = time.perf_counter() - started
        sample_count = frame.signal.samples.size
        stages = [
            {
                "name": stage,
                "rate": "sample" if signal.samples.size == sample_count else "symbol",
                "samples": iq(signal.samples),
            }
            for stage, signal in snapshots
        ]
        return {
            "frame": self._frame_out(frame, seed),
            "stages": stages,
            "result": self._result_out(request.receiver, result, frame, seconds),
        }


# ======================================================================
# Benchmarks: a grid, as a background job, saved to the database
# ======================================================================
class BenchmarkService:
    """Starts benchmark jobs and reads their stored results."""

    def __init__(
        self,
        settings: Settings,
        catalog: ReceiverCatalog,
        jobs: JobManager,
        recordings: "RecordingService | None" = None,
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._jobs = jobs
        self._recordings = recordings

    def _store(self) -> ResultStore:
        # A fresh connection per call: SQLite connections must not be
        # shared between threads, and the job thread writes while
        # request threads read.
        return ResultStore(self._settings.database_path)

    def _grid(self, request: BenchmarkRequest) -> ScenarioGrid:
        profiles = {p.name: p for p in ImpairmentProfile.standard()}
        if self._recordings is not None and self._recordings.available:
            profiles["recorded"] = ImpairmentProfile.recorded()
        unknown = [p for p in request.profiles if p not in profiles]
        if unknown:
            hint = ""
            if "recorded" in unknown:
                hint = " ('recorded' needs the RML24 dataset: start with --rml24 PATH)"
            raise ValidationError("profiles", unknown, f"known profiles: {sorted(profiles)}{hint}")
        return ScenarioGrid.product(
            request.modulations, request.snrs_db, [profiles[p] for p in request.profiles]
        )

    def start(self, request: BenchmarkRequest) -> Job:
        """Check everything that can be checked now, then queue the work.

        A request that is going to fail should fail in the response to
        the request, not three minutes later in a job log.  So the grid
        is built, the size limits applied and every receiver name and
        model looked up before anything is queued.
        """
        grid = self._grid(request)
        names = list(dict.fromkeys(request.receivers))
        if request.frames_per_scenario > self._settings.max_frames_per_scenario:
            raise ValidationError(
                "frames_per_scenario", request.frames_per_scenario,
                f"the limit is {self._settings.max_frames_per_scenario}",
            )
        cells = len(grid) * len(names)
        if cells > self._settings.max_benchmark_cells:
            raise ValidationError(
                "receivers", cells,
                f"{len(grid)} scenarios x {len(names)} receivers exceeds the limit of "
                f"{self._settings.max_benchmark_cells} cells",
            )
        for name in names:
            self._catalog.entry(name)
            if not any(self._catalog.is_available(name, m) for m in grid.modulations):
                raise ModelNotTrainedError(f"{name} for {', '.join(grid.modulations)}")

        catalog, settings = self._catalog, self._settings
        recordings = self._recordings
        uses_recordings = any(s.profile.name == "recorded" for s in grid)

        def work(report):
            receivers = {}
            for name in names:
                members = {
                    m: catalog.build(name, m)
                    for m in grid.modulations
                    if catalog.is_available(name, m)
                }
                receivers[name] = ReceiverBank(members, name=name)

            source = SimulatedSource(seed=request.seed)
            if uses_recordings:
                # One source for the runner, two behind it: each scenario
                # goes to whichever can produce it.
                source = SourceRouter([source, recordings.source(seed=request.seed)])
            runner = BenchmarkRunner(
                source, receivers,
                frames_per_scenario=request.frames_per_scenario,
            )
            runner.add_observer(CallbackObserver(
                lambda done, total, cell: report(done, total, f"{cell.receiver}: {cell.scenario!r}")
            ))
            result = runner.run(grid)
            with ResultStore(settings.database_path) as store:
                run_id = store.save(result, request.name, notes=request.notes)
            return {
                "run_id": run_id,
                "cells": len(result),
                "skipped": [
                    {"receiver": r, "scenario": repr(s), "reason": why}
                    for r, s, why in result.skipped
                ],
            }

        return self._jobs.submit(
            "benchmark", f"{request.name}: {len(grid)} scenarios x {len(names)} receivers", work
        )

    # -- reading stored results ---------------------------------------
    def runs(self) -> list[dict[str, Any]]:
        with self._store() as store:
            return store.runs()

    def cells(self, run_id: int, **filters: Any) -> list[dict[str, Any]]:
        with self._store() as store:
            if not any(r["id"] == run_id for r in store.runs()):
                raise KeyError(run_id)
            return jsonable(store.cells(run_id, **filters))

    def delete(self, run_id: int) -> None:
        with self._store() as store:
            if not any(r["id"] == run_id for r in store.runs()):
                raise KeyError(run_id)
            store.delete(run_id)


# ======================================================================
# Recordings: the RML24 dataset, if one is configured
# ======================================================================
class RecordingService:
    """Opens the dataset once, on first use, and shares it.

    Opening is deferred because the full file is 20 GB on a disk that
    may be slow, and a user who never asks for real data should not wait
    for it.  A lock guards the open, since requests arrive on several
    threads.
    """

    def __init__(self, path) -> None:
        self._path = path
        self._archive = None
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return self._path is not None

    def archive(self):
        from ..data import open_archive

        with self._lock:
            if self._archive is None and self._error is None and self._path is not None:
                try:
                    self._archive = open_archive(self._path)
                except SatRecoveryError as error:
                    self._error = str(error)
            return self._archive

    @property
    def available(self) -> bool:
        return self.archive() is not None

    def source(self, seed: int = 0):
        """A fresh :class:`RecordedSource` over the shared archive."""
        from ..data import RecordedSource

        archive = self.archive()
        if archive is None:
            raise ValidationError("profiles", "recorded", self.describe()["detail"])
        return RecordedSource(archive, seed=seed)

    def describe(self) -> dict[str, Any]:
        if self._path is None:
            return {"available": False, "detail": "No dataset configured. Start with --rml24 PATH "
                    "(the RML24_IQdata.h5 file, its folder, or a sample .npz)."}
        archive = self.archive()
        if archive is None:
            return {"available": False, "path": str(self._path), "detail": self._error or "could not open"}
        return {
            "available": True,
            "path": str(self._path),
            "name": archive.name,
            "recordings": len(archive),
            "modulations": archive.modulations(),
            "detail": "BPSK, QPSK and 8PSK recordings are clean enough to grade receivers on.",
        }

    def close(self) -> None:
        with self._lock:
            if self._archive is not None:
                self._archive.close()
                self._archive = None


# ======================================================================
# Models: training hybrids as background jobs
# ======================================================================
class ModelService:
    def __init__(self, models: ModelStore, jobs: JobManager) -> None:
        self._models = models
        self._jobs = jobs

    def trained(self) -> list[dict[str, Any]]:
        return jsonable(self._models.trained())

    def train(self, modulation: str, request: TrainRequest) -> Job:
        name = ModulationRegistry.create(modulation).name   # bad name -> 422 now
        spec = TrainingSpec(
            snrs_db=request.snrs_db, profiles=request.profiles,
            frames_each=request.frames_each, epochs=request.epochs, patience=request.patience,
        )
        models = self._models
        return self._jobs.submit(
            "training", f"hybrid demapper for {name}",
            lambda report: jsonable(models.train(name, spec, report)),
        )

    def delete(self, modulation: str) -> bool:
        return self._models.delete(modulation)


__all__ = [
    "BenchmarkService", "ExperimentService", "ModelService", "RecordingService", "iq", "jsonable",
]
