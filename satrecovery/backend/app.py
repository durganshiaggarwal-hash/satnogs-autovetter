"""The HTTP layer: routes, status codes, and turning errors into answers.

Start it with::

    python -m satrecovery.backend

then open http://127.0.0.1:8000/docs for interactive documentation of
every endpoint -- each one can be tried from the browser, which is the
quickest way to see what the backend does.

Every route is a few lines: parse the request, call a service, return
the result.  Anything longer belongs in ``services.py``.

DESIGN NOTE: AN APPLICATION FACTORY AND DEPENDENCY INJECTION
-------------------------------------------------------------
:func:`create_app` builds a fresh application from a
:class:`~satrecovery.backend.settings.Settings`, rather than the module
holding one global app.  The tests build an app pointed at a temporary
folder, and the real server builds one pointed at the user's data
folder, from the same code.  Everything the routes need is created once
in a :class:`Backend` and handed to them, never constructed inside a
route -- so a route cannot accidentally open its own database or train
its own model.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from .. import __version__
from ..core.exceptions import (
    ModelNotTrainedError,
    SatRecoveryError,
    UnknownModulationError,
    UnknownReceiverError,
    ValidationError,
)
from ..evaluation.scenarios import ImpairmentProfile
from ..modulation.registry import ModulationRegistry
from ..simulation.pulse import RootRaisedCosine
from .catalog import ReceiverCatalog
from .jobs import JobManager
from .models import ModelStore
from .schemas import (
    BenchmarkRequest,
    ErrorOut,
    ExperimentRequest,
    ExperimentResponse,
    JobOut,
    ModulationInfo,
    ProfileInfo,
    ReceiverInfo,
    RunOut,
    StagesRequest,
    StagesResponse,
    TrainRequest,
)
from .services import BenchmarkService, ExperimentService, ModelService, RecordingService, iq
from .settings import Settings


class Backend:
    """Everything the routes depend on, built once per application."""

    def __init__(self, settings: Settings) -> None:
        settings.ensure_directories()
        self.settings = settings
        self.models = ModelStore(settings.models_dir)
        self.catalog = ReceiverCatalog.standard(RootRaisedCosine(8), self.models)
        self.jobs = JobManager()
        self.experiments = ExperimentService(self.catalog)
        self.recordings = RecordingService(settings.rml24_path)
        self.benchmarks = BenchmarkService(settings, self.catalog, self.jobs, self.recordings)
        self.model_service = ModelService(self.models, self.jobs)

    def close(self) -> None:
        self.jobs.shutdown()
        self.recordings.close()


def _backend(request: Request) -> Backend:
    return request.app.state.backend


_ERRORS = {
    404: {"model": ErrorOut, "description": "Unknown receiver, run or job"},
    409: {"model": ErrorOut, "description": "A model must be trained first"},
    422: {"model": ErrorOut, "description": "A value the science does not accept"},
}


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an application. Each call gives an independent one."""
    settings = settings or Settings.from_env()
    backend = Backend(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        backend.close()

    app = FastAPI(
        title="Satellite Signal Recovery",
        version=__version__,
        summary="Classical, learned and hybrid receivers for distorted satellite signals.",
        lifespan=lifespan,
    )
    app.state.backend = backend

    # ------------------------------------------------------------------
    # Turning domain errors into HTTP answers, in one place
    # ------------------------------------------------------------------
    @app.exception_handler(ValidationError)
    async def _validation(request: Request, error: ValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": "invalid value", "detail": str(error), "field": error.field},
        )

    @app.exception_handler(UnknownModulationError)
    async def _modulation(request: Request, error: UnknownModulationError):
        return JSONResponse(
            status_code=422,
            content={"error": "unknown modulation", "detail": str(error), "field": "modulation"},
        )

    @app.exception_handler(UnknownReceiverError)
    async def _receiver(request: Request, error: UnknownReceiverError):
        return JSONResponse(
            status_code=404,
            content={"error": "unknown receiver", "detail": str(error), "field": "receivers"},
        )

    @app.exception_handler(ModelNotTrainedError)
    async def _untrained(request: Request, error: ModelNotTrainedError):
        return JSONResponse(
            status_code=409,
            content={
                "error": "model not trained",
                "detail": f"{error.name} has not been trained yet. "
                "Train it with POST /api/models/{modulation}/train.",
            },
        )

    @app.exception_handler(SatRecoveryError)
    async def _other(request: Request, error: SatRecoveryError):
        return JSONResponse(status_code=400, content={"error": type(error).__name__, "detail": str(error)})

    # ------------------------------------------------------------------
    # Reference data: what can be asked for
    # ------------------------------------------------------------------
    @app.get("/api/health", tags=["reference"])
    def health(request: Request) -> dict[str, Any]:
        """Is the server up, and where is it keeping its files?"""
        b = _backend(request)
        return {
            "status": "ok",
            "version": __version__,
            "data_dir": str(b.settings.data_dir),
            "trained_models": [m["modulation"] for m in b.models.trained()],
        }

    @app.get("/api/modulations", response_model=list[ModulationInfo], tags=["reference"])
    def modulations() -> list[dict[str, Any]]:
        """Every scheme, with its constellation for plotting."""
        out = []
        for name in ModulationRegistry.available():
            scheme = ModulationRegistry.create(name)
            out.append({
                "name": scheme.name,
                "bits_per_symbol": scheme.bits_per_symbol,
                "order": scheme.order,
                "constellation": iq(scheme.constellation),
            })
        return out

    @app.get("/api/profiles", response_model=list[ProfileInfo], tags=["reference"])
    def profiles() -> list[dict[str, Any]]:
        """The named impairment profiles benchmarks are built from."""
        return [{"name": p.name, "settings": p.settings} for p in ImpairmentProfile.standard()]

    @app.get("/api/dataset", tags=["reference"])
    def dataset(request: Request) -> dict[str, Any]:
        """Is a real-recordings dataset (RML24) available for benchmarks?

        When it is, benchmarks accept the ``recorded`` profile: frames cut
        from real satellite recordings, labelled by the Phase 3 reference
        decoder, at whatever noise level is asked for.
        """
        return _backend(request).recordings.describe()

    @app.get("/api/receivers", response_model=list[ReceiverInfo], tags=["reference"])
    def receivers(request: Request, modulation: str | None = None) -> list[dict[str, Any]]:
        """Every receiver, and which modulations it is ready for right now."""
        return _backend(request).catalog.describe(modulation)

    # ------------------------------------------------------------------
    # Experiments: the interactive part
    # ------------------------------------------------------------------
    @app.post("/api/experiments", response_model=ExperimentResponse, responses=_ERRORS, tags=["experiments"])
    def run_experiment(request: Request, body: ExperimentRequest) -> dict[str, Any]:
        """Build one frame and recover it with every listed receiver.

        All receivers see the identical frame, so any difference between
        their results is theirs.  The response includes the seed, so the
        same frame can be requested again.
        """
        return _backend(request).experiments.run(body)

    @app.post("/api/experiments/stages", response_model=StagesResponse, responses=_ERRORS, tags=["experiments"])
    def run_stages(request: Request, body: StagesRequest) -> dict[str, Any]:
        """Follow one frame through every stage of one receiver."""
        return _backend(request).experiments.stages(body)

    # ------------------------------------------------------------------
    # Benchmarks and stored results
    # ------------------------------------------------------------------
    @app.post("/api/benchmarks", status_code=202, response_model=JobOut, responses=_ERRORS, tags=["benchmarks"])
    def start_benchmark(request: Request, body: BenchmarkRequest) -> dict[str, Any]:
        """Queue a benchmark. Poll the returned job for progress."""
        return _backend(request).benchmarks.start(body).snapshot()

    @app.get("/api/runs", response_model=list[RunOut], tags=["benchmarks"])
    def list_runs(request: Request) -> list[dict[str, Any]]:
        """Every stored benchmark run, newest first."""
        return _backend(request).benchmarks.runs()

    @app.get("/api/runs/{run_id}/cells", responses=_ERRORS, tags=["benchmarks"])
    def run_cells(
        request: Request,
        run_id: int,
        receiver: str | None = None,
        modulation: str | None = Query(None, examples=["16QAM"]),
        profile: str | None = None,
    ) -> list[dict[str, Any]]:
        """One run's results, optionally filtered."""
        try:
            return _backend(request).benchmarks.cells(
                run_id, receiver=receiver, modulation=modulation, profile=profile
            )
        except KeyError:
            raise HTTPException(404, f"no run {run_id}") from None

    @app.delete("/api/runs/{run_id}", status_code=204, tags=["benchmarks"])
    def delete_run(request: Request, run_id: int) -> None:
        try:
            _backend(request).benchmarks.delete(run_id)
        except KeyError:
            raise HTTPException(404, f"no run {run_id}") from None

    # ------------------------------------------------------------------
    # Trained models
    # ------------------------------------------------------------------
    @app.get("/api/models", tags=["models"])
    def list_models(request: Request) -> list[dict[str, Any]]:
        """Which hybrid demappers are trained, and on what."""
        return _backend(request).model_service.trained()

    @app.post("/api/models/{modulation}/train", status_code=202, response_model=JobOut, tags=["models"])
    def train_model(request: Request, modulation: str, body: TrainRequest | None = None) -> dict[str, Any]:
        """Train (or retrain) the hybrid demapper for one modulation, as a job."""
        return _backend(request).model_service.train(modulation, body or TrainRequest()).snapshot()

    @app.delete("/api/models/{modulation}", status_code=204, tags=["models"])
    def delete_model(request: Request, modulation: str) -> None:
        if not _backend(request).model_service.delete(modulation):
            raise HTTPException(404, f"no trained model for {modulation}")

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------
    @app.get("/api/jobs", response_model=list[JobOut], tags=["jobs"])
    def list_jobs(request: Request) -> list[dict[str, Any]]:
        return _backend(request).jobs.list()

    @app.get("/api/jobs/{job_id}", response_model=JobOut, tags=["jobs"])
    def get_job(request: Request, job_id: str) -> dict[str, Any]:
        """Progress of one job. The GUI polls this to draw a progress bar."""
        try:
            return _backend(request).jobs.get(job_id).snapshot()
        except KeyError:
            raise HTTPException(404, f"no job {job_id}") from None

    return app


__all__ = ["Backend", "create_app"]
