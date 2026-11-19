"""The analysis endpoints, added on top of the existing backend.

DESIGN NOTE: OPEN FOR EXTENSION, CLOSED FOR MODIFICATION
--------------------------------------------------------
:func:`create_analysis_app` builds the Phase 7 application exactly as
before -- with :func:`~satrecovery.backend.app.create_app`, untouched --
and then *adds* a router of new endpoints to it.  Not one line of the
existing backend changes, and every existing endpoint keeps working, so
the original app and this one can share a data folder and a set of
trained models.

New endpoints, all under ``/api/analysis``:

``GET  /status``        is the classifier trained, and is RML24 open?
``GET  /recordings``    pick RML24 recordings by scheme, SNR label and rate
``POST /recording``     classify and recover one RML24 recording
``POST /simulated``     classify and recover one simulated signal
``POST /evaluate``      test the classifier on held-out recordings: where does it fail?
``POST /file``          classify and recover a signal from the user's own file
``GET  /history``       every analysis so far, newest first (``?q=`` filters)
``GET  /history/{id}``  one past analysis, exactly as it was
``DELETE /history``     forget them all
``POST /dataset``       open a different RML24 folder while the app runs
``GET  /batch-rows``    pick many recordings at once, for "Many at once"
``POST /quick``         analyse one recording and return only its summary
``POST /test-file``     a test file (signal + answer key) of an analysed signal, to save
"""

from __future__ import annotations

import base64
import binascii
import threading
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from ..backend.app import create_app
from ..backend.schemas import ConditionsIn
from ..backend.settings import Settings
from ..core.exceptions import SatRecoveryError, ValidationError
from ..data import layout
from .analyser import Analyser
from .classifier import ModulationClassifier
from .compare import ModelComparison, load_classifier
from .inputs import CHECKABLE, RecordedInput, SimulatedInput, usually_checkable
from .history import AnalysisHistory
from .testfile import TestFile
from .trust import TrustCheck
from .upload import MAX_BYTES, input_from_file


class RecordingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row: int = Field(..., ge=0)


class SimulatedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modulation: str = "QPSK"
    conditions: ConditionsIn = Field(default_factory=ConditionsIn)
    symbol_rate: float = Field(250_000.0, description="100000, 250000 or 500000, as in RML24")
    seed: int | None = Field(None, ge=0)


class FileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=1, max_length=255, description="the file's name, for its extension")
    content_base64: str = Field(..., min_length=1, description="the file's bytes, base64-encoded")
    symbol_rate: float | None = Field(None, description="100000, 250000 or 500000; not needed for an RML24 export")
    bits_name: str | None = Field(None, max_length=255, description="the answer key's file name (a ..._bits.csv)")
    bits_base64: str | None = Field(None, description="the answer key's bytes, base64-encoded")


class TestFileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: dict[str, Any] = Field(..., description="the 'input' part of an analysis result")


class DatasetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., min_length=1, max_length=4096, description="the RML24 folder, its .h5 file, or a sample .npz")


class UseModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(..., min_length=1, max_length=40, description="'main', or one of the compared models")


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    per_group: int = Field(6, ge=1, le=40, description="recordings per scheme, SNR label and rate")
    seed: int = Field(0, ge=0)


#: The training script keeps the *last* recordings of every group for
#: testing (``--test-per-cell``, 40 by default).  The report card draws
#: only from those, so it never tests the AI on something it trained on.
HELD_OUT_PER_GROUP = 40


class AnalysisService:
    """Loads the classifier lazily and runs analyses."""

    def __init__(self, settings: Settings, backend: Any, classifier_path: str | Path | None = None) -> None:
        self._settings = settings
        self._backend = backend
        main = Path(classifier_path) if classifier_path else settings.models_dir / "classifier.pt"
        # The "Compare models" tab can make another model the app's choice.
        self._comparison = ModelComparison(Path(settings.models_dir) / "compare", main)
        active = self._comparison.active()
        self._path = main if active == ModelComparison.MAIN else self._comparison.path_for(active)
        self._classifier: ModulationClassifier | None = None
        self._lock = threading.Lock()
        self._history = AnalysisHistory(Path(settings.data_dir) / "analysis_history.db")

    @property
    def history(self) -> AnalysisHistory:
        return self._history

    def _remember(self, result: dict[str, Any]) -> dict[str, Any]:
        """Keep the analysis in the history.  A history problem never spoils the analysis itself."""
        try:
            result["history_id"] = self._history.add(result)
        except Exception:  # noqa: BLE001 - the answer matters more than the record of it
            result["history_id"] = None
        return result

    def history_entry(self, entry: int) -> dict[str, Any]:
        try:
            return self._history.get(entry)
        except ValidationError:
            raise HTTPException(status_code=404, detail=f"Analysis {entry} is not in the history.") from None

    # -- choosing the dataset while the app runs -------------------------
    def open_dataset(self, path: str) -> dict[str, Any]:
        """Open a different RML24 folder; the old one stays open if the new one fails."""
        from ..backend.services import RecordingService
        from ..data import open_archive

        try:
            probe = open_archive(Path(path).expanduser())
        except SatRecoveryError as error:
            raise HTTPException(status_code=422, detail=f"That is not an RML24 dataset: {error}") from None
        probe.close()
        old = self._backend.recordings
        self._backend.recordings = RecordingService(str(Path(path).expanduser()))
        old.close()
        return self._backend.recordings.describe()

    # -- test files ------------------------------------------------------------
    def test_file(self, source: dict[str, Any]) -> dict[str, Any]:
        """A test file of an analysed signal: its samples plus its answer key."""
        kind = source.get("kind")
        if kind == "recorded" and not source.get("file"):
            name, data = TestFile.from_recording(RecordedInput(self.archive(), int(source["row"])))
        elif kind == "simulated":
            from ..core.conditions import ChannelConditions

            try:
                signal = SimulatedInput(str(source["true_modulation"]),
                                        ChannelConditions.from_dict(dict(source["conditions"])),
                                        float(source["symbol_rate"]), int(source["seed"]))
            except (KeyError, TypeError, ValueError) as error:
                raise HTTPException(status_code=422, detail=f"Not enough detail to rebuild this signal ({error}).") from None
            name, data = TestFile.from_simulated(signal)
        else:
            raise HTTPException(status_code=422, detail="Only signals from the Real signal and Simulated signal "
                                                        "tabs can be saved as test files.")
        return {"name": name, "content_base64": base64.b64encode(data).decode("ascii"),
                "info": TestFile.describe(data)}

    # -- many at once ------------------------------------------------------
    def batch_rows(self, modulation: str | None, snr_db: float | None, symbol_rate: float | None,
                   count: int, seed: int | None) -> list[int]:
        archive = self.archive()
        names = [modulation] if modulation else archive.modulations()
        rows = np.concatenate([archive.select(n, snr_db, symbol_rate) for n in names]) if names else np.array([])
        if rows.size == 0:
            return []
        rng = np.random.default_rng(seed)
        return sorted(int(r) for r in rng.choice(rows, size=min(int(count), rows.size), replace=False))

    def quick(self, row: int) -> dict[str, Any]:
        """One recording, analysed in full but answered in one line (not kept in the history)."""
        signal = RecordedInput(self.archive(), row)
        result = self.analyser().analyse(signal)
        c, check = result["classification"], result["recovery"]["check"]
        source = result["input"]
        return {
            "row": int(row), "really": c["true_label"], "ai_said": c["label"], "confidence": c["confidence"],
            "correct": bool(c["correct"]), "snr_label_db": source.get("snr_label_db"),
            "symbol_rate": source.get("symbol_rate"), "checked": bool(check.get("available")),
            "bit_errors": check.get("bit_errors"), "n_bits": check.get("n_bits"), "ber": check.get("ber"),
            "bits_recovered": result["recovery"]["n_bits"],
            "trust": result["trust"]["level"],
            "warning": " ".join(result["trust"]["reasons"]),
        }

    @property
    def classifier_path(self) -> Path:
        return self._path

    def classifier(self) -> ModulationClassifier | None:
        with self._lock:
            if self._classifier is None and self._path.with_suffix(".pt").exists():
                self._classifier = load_classifier(self._path)
            return self._classifier

    # -- comparing models --------------------------------------------------
    def models(self) -> dict[str, Any]:
        """The compared models, their results, and which one the app uses."""
        return self._comparison.listing()

    def use_model(self, name: str) -> dict[str, Any]:
        """Make the app use another model from now on (remembered between runs)."""
        try:
            path = self._comparison.set_active(name)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=f"Cannot use that model: {error}") from None
        with self._lock:
            self._path = path
            self._classifier = None
        return self.models()

    def analyser(self) -> Analyser:
        model = self.classifier()
        if model is None:
            raise HTTPException(status_code=409, detail=(
                "The AI classifier has not been trained yet. Train it once with: "
                "python scripts/train_classifier.py <path to RML24>"))
        return Analyser(model, self._backend.catalog)

    def archive(self):
        archive = self._backend.recordings.archive()
        if archive is None:
            raise HTTPException(status_code=409, detail=self._backend.recordings.describe()["detail"])
        return archive

    def status(self) -> dict[str, Any]:
        model = self.classifier()
        return {
            "classifier": {
                "available": model is not None,
                "path": str(self._path.with_suffix(".pt")),
                "classes": list(model.classes) if model else list(ModulationClassifier.DEFAULT_CLASSES),
                "test_accuracy": model.test_accuracy if model else None,
                "description": model.describe() if model else "not trained yet",
            },
            "dataset": self._backend.recordings.describe(),
            "symbol_rates": [100_000, 250_000, 500_000],
            "snr_labels_db": list(layout.SNR_LABELS_DB),
        }

    #: How many recordings to try when looking for a checkable one.
    CHECKABLE_TRIES = 8

    def recordings(self, modulation: str | None, snr_db: float | None, symbol_rate: float | None,
                   count: int, seed: int | None) -> list[dict[str, Any]]:
        archive = self.archive()
        names = [modulation] if modulation else archive.modulations()
        rows = np.concatenate([archive.select(n, snr_db, symbol_rate) for n in names]) if names else np.array([])
        if rows.size == 0:
            return []
        rng = np.random.default_rng(seed)
        picked = rng.choice(rows, size=min(int(count), rows.size), replace=False)
        # One recording of a checkable type, at a quality where most pass:
        # look a little further for one whose bits can actually be checked,
        # rather than hand out one of the few that cannot.  Said openly in
        # the GUI, and every recording returned says whether it is checkable.
        if int(count) == 1 and modulation in CHECKABLE and snr_db is not None and symbol_rate is not None \
                and usually_checkable(modulation, snr_db, symbol_rate):
            for row in rng.choice(rows, size=min(self.CHECKABLE_TRIES, rows.size), replace=False):
                if RecordedInput(archive, int(row)).reference().accepted:
                    picked = np.array([row])
                    break
        out = []
        for row in sorted(int(r) for r in picked):
            rec = archive.recording(row)
            checkable = rec.modulation in CHECKABLE and RecordedInput(archive, row).reference().accepted
            out.append({"row": row, "modulation": rec.modulation, "snr_label_db": rec.snr_label_db,
                        "symbol_rate": rec.symbol_rate, "samples_per_symbol": rec.samples_per_symbol,
                        "checkable": bool(checkable)})
        return out

    def evaluate(self, request: EvaluateRequest) -> dict[str, Any]:
        """Test the classifier on held-out RML24 recordings and report where it fails."""
        from .dataset import load_set

        model = self.classifier()
        if model is None:
            self.analyser()                                  # raises the "train it first" answer
        archive = self.archive()
        rng = np.random.default_rng(request.seed)
        summary = archive.summary()
        chosen = []
        speed_of: dict[int, float] = {}
        for index, name in enumerate(model.classes):
            if name not in summary:
                continue
            for snr in summary[name]["snr_labels_db"]:
                for rate in summary[name]["symbol_rates"]:
                    rows = archive.select(name, snr, rate)
                    if rows.size < 2:
                        continue
                    held_out = rows[rows.size - min(HELD_OUT_PER_GROUP, rows.size // 2):]
                    take = min(request.per_group, held_out.size)
                    picked = np.sort(rng.choice(held_out, size=take, replace=False))
                    chosen += [(int(r), index, float(snr)) for r in picked]
                    speed_of.update({int(r): float(rate) for r in picked})
        data = load_set(archive, chosen)
        probabilities = model.predict_proba(data.inputs)
        predicted = probabilities.argmax(axis=1)
        right = predicted == data.targets
        classes = list(model.classes)

        by_snr = []
        for snr in sorted(np.unique(data.snr_db)):
            keep = data.snr_db == snr
            by_snr.append({"snr_db": float(snr), "accuracy": float(right[keep].mean()), "n": int(keep.sum())})
        matrix = np.zeros((len(classes), len(classes)), dtype=int)
        clean = data.snr_db >= 0
        np.add.at(matrix, (data.targets[clean], predicted[clean]), 1)
        by_class = []
        for index, name in enumerate(classes):
            keep = (data.targets == index) & clean
            by_class.append({"name": name, "accuracy": float(right[keep].mean()) if keep.any() else None})
        # The warning, judged on the same held-out recordings: how many of the
        # AI's mistakes does it flag, and how many correct answers does it
        # flag needlessly?
        levels = []
        for i in range(len(data)):
            samples = data.inputs[i, 0].astype(np.float64) + 1j * data.inputs[i, 1].astype(np.float64)
            ranked = {classes[k]: float(probabilities[i, k]) for k in range(len(classes))}
            levels.append(TrustCheck.assess(ranked, samples, speed_of.get(int(data.rows[i]), 250_000.0)).level)
        flagged = np.array([level != "confident" for level in levels])
        n_wrong, n_right = int((~right).sum()), int(right.sum())
        warnings = {
            "flagged": int(flagged.sum()),
            "mistakes_flagged": float((flagged & ~right).sum() / n_wrong) if n_wrong else None,
            "correct_flagged": float((flagged & right).sum() / n_right) if n_right else None,
            "accuracy_when_confident": float(right[~flagged].mean()) if (~flagged).any() else None,
            "accuracy_when_flagged": float(right[flagged].mean()) if flagged.any() else None,
        }
        wrong = np.flatnonzero(~right)
        wrong = wrong[np.argsort(-data.snr_db[wrong], kind="stable")][:300]
        mistakes = [{
            "row": int(data.rows[i]),
            "snr_label_db": float(data.snr_db[i]),
            "true": classes[int(data.targets[i])],
            "predicted": classes[int(predicted[i])],
            "confidence": float(probabilities[i, predicted[i]]),
            "warning": levels[i],
        } for i in wrong]
        high = data.snr_db >= 10
        return {
            "n": int(len(data)),
            "accuracy": float(right.mean()),
            "accuracy_10db_up": float(right[high].mean()) if high.any() else None,
            "by_snr": by_snr,
            "confusion": {"classes": classes, "matrix": matrix.tolist(), "min_snr_db": 0},
            "by_class": by_class,
            "mistakes": mistakes,
            "n_mistakes": int((~right).sum()),
            "warnings": warnings,
        }

    def analyse_recording(self, row: int) -> dict[str, Any]:
        # The archive refuses an unknown row itself, with a clear message.
        return self._remember(self.analyser().analyse(RecordedInput(self.archive(), row)))

    def analyse_file(self, request: FileRequest) -> dict[str, Any]:
        if len(request.content_base64) > MAX_BYTES * 4 // 3 + 8:
            raise HTTPException(status_code=413, detail=f"The file is larger than {MAX_BYTES // (1024 * 1024)} MB.")
        try:
            data = base64.b64decode(request.content_base64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=422, detail="The file's contents did not arrive intact.") from None
        if request.bits_base64:
            from .csvpair import CsvTestFile

            try:
                bits = base64.b64decode(request.bits_base64, validate=True)
            except (binascii.Error, ValueError):
                raise HTTPException(status_code=422, detail="The bits file did not arrive intact.") from None
            signal = CsvTestFile.input_from(request.name, data, request.bits_name or "bits file", bits)
            return self._remember(self.analyser().analyse(signal))
        return self._remember(self.analyser().analyse(input_from_file(request.name, data, request.symbol_rate)))

    def analyse_simulated(self, request: SimulatedRequest) -> dict[str, Any]:
        seed = request.seed if request.seed is not None else int(np.random.default_rng().integers(0, 2**31))
        signal = SimulatedInput(request.modulation, request.conditions.to_domain(), request.symbol_rate, seed)
        return self._remember(self.analyser().analyse(signal))


def analysis_router() -> APIRouter:
    router = APIRouter(prefix="/api/analysis", tags=["analysis"])

    def service(request: Request) -> AnalysisService:
        return request.app.state.analysis

    @router.get("/status")
    def status(request: Request) -> dict[str, Any]:
        """Is the classifier trained, and which dataset is open?"""
        return service(request).status()

    @router.get("/recordings")
    def recordings(request: Request, modulation: str | None = None, snr_db: float | None = None,
                   symbol_rate: float | None = None, count: int = 1, seed: int | None = None) -> list[dict]:
        """Pick RML24 recordings at random, optionally filtered."""
        return service(request).recordings(modulation, snr_db, symbol_rate, max(1, min(count, 50)), seed)

    @router.post("/recording")
    def analyse_recording(request: Request, body: RecordingRequest) -> dict[str, Any]:
        """Step 1, the AI names the modulation; step 2, a receiver recovers the bits."""
        return service(request).analyse_recording(body.row)

    @router.post("/evaluate")
    def evaluate(request: Request, body: EvaluateRequest) -> dict[str, Any]:
        """How good is the AI? Accuracy by SNR, confusions and every mistake, on held-out recordings."""
        return service(request).evaluate(body)

    @router.post("/simulated")
    def analyse_simulated(request: Request, body: SimulatedRequest) -> dict[str, Any]:
        """The same two steps, on a signal made by the simulator."""
        return service(request).analyse_simulated(body)

    @router.get("/history")
    def history(request: Request, q: str = "", limit: int = 200) -> list[dict]:
        """Every analysis so far, newest first."""
        return service(request).history.entries(q, limit)

    @router.get("/history/{entry}")
    def history_entry(request: Request, entry: int) -> dict[str, Any]:
        """One past analysis, exactly as it was."""
        return service(request).history_entry(entry)

    @router.delete("/history")
    def clear_history(request: Request) -> dict[str, int]:
        """Forget every past analysis."""
        return {"removed": service(request).history.clear()}

    @router.post("/dataset")
    def open_dataset(request: Request, body: DatasetRequest) -> dict[str, Any]:
        """Use a different RML24 folder from now on."""
        return service(request).open_dataset(body.path)

    @router.get("/batch-rows")
    def batch_rows(request: Request, modulation: str | None = None, snr_db: float | None = None,
                   symbol_rate: float | None = None, count: int = 100, seed: int | None = None) -> list[int]:
        """Pick up to 1000 recordings for "Many at once"."""
        return service(request).batch_rows(modulation, snr_db, symbol_rate, max(1, min(count, 1000)), seed)

    @router.post("/quick")
    def quick(request: Request, body: RecordingRequest) -> dict[str, Any]:
        """Analyse one recording; return only its one-line summary."""
        return service(request).quick(body.row)

    @router.post("/test-file")
    def test_file(request: Request, body: TestFileRequest) -> dict[str, Any]:
        """A test file of an analysed signal: its samples plus its answer key."""
        return service(request).test_file(body.input)

    @router.get("/models")
    def models(request: Request) -> dict[str, Any]:
        """Compare models: every model's results, and which one the app uses."""
        return service(request).models()

    @router.post("/models/use")
    def use_model(request: Request, body: UseModelRequest) -> dict[str, Any]:
        """Make the app use another of the compared models."""
        return service(request).use_model(body.model)

    @router.post("/file")
    def analyse_file(request: Request, body: FileRequest) -> dict[str, Any]:
        """The same two steps, on a signal from the user's own file."""
        return service(request).analyse_file(body)

    return router


def create_analysis_app(settings: Settings | None = None, classifier_path: str | Path | None = None) -> FastAPI:
    """The existing backend, unchanged, plus the analysis endpoints."""
    settings = settings or Settings.from_env()
    app = create_app(settings)
    app.state.analysis = AnalysisService(settings, app.state.backend, classifier_path)
    app.include_router(analysis_router())
    return app


__all__ = ["AnalysisService", "DatasetRequest", "TestFileRequest", "EvaluateRequest", "FileRequest", "RecordingRequest", "SimulatedRequest", "UseModelRequest", "analysis_router", "create_analysis_app"]
