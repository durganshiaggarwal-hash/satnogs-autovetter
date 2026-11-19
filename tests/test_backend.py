"""Tests for Phase 7 -- the backend, through HTTP and without it.

Most tests go through FastAPI's test client, which calls the real app
in-process: real routing, real validation, real JSON, no network.  The
app is pointed at a temporary folder, so nothing touches the user's
data.

The properties tested are the ones a GUI will rely on without checking:
every receiver in one experiment sees the same frame, a seed reproduces
a frame exactly, a bad request fails *before* any work starts and says
which field was wrong, and nothing in a response is invalid JSON.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="the backend needs FastAPI")
from fastapi.testclient import TestClient  # noqa: E402

from satrecovery.backend import Settings, create_app  # noqa: E402
from satrecovery.backend.catalog import CatalogEntry, ReceiverCatalog  # noqa: E402
from satrecovery.backend.jobs import JobManager, JobState  # noqa: E402
from satrecovery.backend.services import jsonable  # noqa: E402
from satrecovery.core.exceptions import UnknownReceiverError, ValidationError  # noqa: E402
from satrecovery.simulation.pulse import RootRaisedCosine  # noqa: E402

TINY_TRAINING = {"snrs_db": [4, 10], "profiles": ["clean"], "frames_each": 3, "epochs": 4}


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("backend")


@pytest.fixture(scope="module")
def client(data_dir):
    with TestClient(create_app(Settings(data_dir=data_dir))) as c:
        yield c


def wait_for(client, job_id, timeout=120.0):
    deadline = time.time() + timeout
    while True:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in ("done", "failed"):
            return job
        assert time.time() < deadline, f"job still {job['state']}"
        time.sleep(0.05)


def experiment(client, **body):
    response = client.post("/api/experiments", json=body)
    assert response.status_code == 200, response.text
    return response.json()


# ======================================================================
# Reference data
# ======================================================================
class TestReference:
    def test_health(self, client, data_dir):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["data_dir"] == str(data_dir.resolve())

    def test_every_modulation_comes_with_its_constellation(self, client):
        schemes = client.get("/api/modulations").json()
        assert [s["name"] for s in schemes] == ["BPSK", "QPSK", "8PSK", "16APSK", "16QAM", "64QAM"]
        for scheme in schemes:
            assert len(scheme["constellation"]["i"]) == scheme["order"]

    def test_the_four_standard_profiles(self, client):
        assert [p["name"] for p in client.get("/api/profiles").json()] == \
            ["clean", "doppler", "fading", "full"]

    def test_receivers_report_what_they_need(self, client):
        receivers = {r["name"]: r for r in client.get("/api/receivers?modulation=QPSK").json()}
        assert receivers["classical"]["available_for"] == ["QPSK"]
        assert receivers["hybrid"]["needs_model"]

    def test_the_interactive_docs_are_served(self, client):
        assert client.get("/docs").status_code == 200
        assert "/api/experiments" in client.get("/openapi.json").json()["paths"]


# ======================================================================
# Experiments
# ======================================================================
class TestExperiments:
    def test_one_result_per_receiver_on_one_frame(self, client):
        body = experiment(client, modulation="16QAM", receivers=["classical", "classical-hard"], seed=3,
                          conditions={"snr_db": 4})
        assert [r["receiver"] for r in body["results"]] == ["classical", "classical-hard"]
        assert body["frame"]["n_bits"] == 1024
        assert len(body["frame"]["received"]["i"]) == 2048

    def test_every_receiver_sees_the_same_frame(self, client):
        """Soft and hard demappers decide identically, so on one shared
        frame their errors must match symbol for symbol."""
        body = experiment(client, modulation="16QAM", receivers=["classical", "classical-hard"],
                          seed=11, conditions={"snr_db": 2})
        soft, hard = body["results"]
        assert soft["bit_errors"] == hard["bit_errors"] > 0
        assert soft["symbol_wrong"] == hard["symbol_wrong"]

    def test_a_seed_reproduces_everything(self, client):
        request = dict(modulation="8PSK", seed=42, conditions={"snr_db": 3, "freq_offset_hz": 400})
        first, second = experiment(client, **request), experiment(client, **request)
        assert first["frame"]["received"] == second["frame"]["received"]
        assert first["results"][0]["bit_errors"] == second["results"][0]["bit_errors"]

    def test_an_unseeded_request_reports_its_seed(self, client):
        first = experiment(client, modulation="QPSK", conditions={"snr_db": 0})
        again = experiment(client, modulation="QPSK", conditions={"snr_db": 0}, seed=first["frame"]["seed"])
        assert first["frame"]["received"] == again["frame"]["received"]

    def test_wrong_symbols_agree_with_wrong_bits(self, client):
        result = experiment(client, modulation="16QAM", seed=5, conditions={"snr_db": 2})["results"][0]
        wrong_symbols = sum(result["symbol_wrong"])
        assert math.ceil(result["bit_errors"] / 4) <= wrong_symbols <= result["bit_errors"]
        assert len(result["symbol_wrong"]) == 256

    def test_ranking_is_none_when_there_is_nothing_to_rank(self, client):
        result = experiment(client, modulation="QPSK", seed=1, conditions={"snr_db": 25})["results"][0]
        assert result["bit_errors"] == 0
        assert result["ranking_quality"] is None

    def test_repeated_receivers_are_run_once(self, client):
        body = experiment(client, receivers=["classical", "classical", "classical-hard"], seed=1)
        assert len(body["results"]) == 2

    def test_responses_are_strict_json(self, client):
        """No NaN or Infinity: other languages' JSON parsers reject them."""
        raw = client.post("/api/experiments", json={"seed": 2, "conditions": {"snr_db": 30}}).text
        assert "NaN" not in raw and "Infinity" not in raw

    def test_the_clean_waveform_is_included_for_comparison(self, client):
        frame = experiment(client, seed=4, conditions={"snr_db": 30})["frame"]
        received = np.array(frame["received"]["i"]) + 1j * np.array(frame["received"]["q"])
        clean = np.array(frame["transmitted"]["i"]) + 1j * np.array(frame["transmitted"]["q"])
        # At 30 dB the received wave is the clean one plus a whisper of noise.
        assert np.mean(np.abs(received - clean) ** 2) < 0.01 * np.mean(np.abs(clean) ** 2)


class TestExperimentErrors:
    def test_a_value_the_domain_rejects_names_the_field(self, client):
        response = client.post("/api/experiments", json={"conditions": {"snr_db": 99}})
        assert response.status_code == 422
        assert response.json()["field"] == "snr_db"

    def test_an_unknown_modulation(self, client):
        response = client.post("/api/experiments", json={"modulation": "QAM-9000"})
        assert response.status_code == 422
        assert response.json()["field"] == "modulation"

    def test_an_unknown_receiver_lists_the_real_ones(self, client):
        response = client.post("/api/experiments", json={"receivers": ["magic"]})
        assert response.status_code == 404
        assert "classical" in response.json()["detail"]

    def test_a_misspelt_field_is_refused_not_ignored(self, client):
        assert client.post("/api/experiments", json={"modulaton": "16QAM"}).status_code == 422

    def test_an_unknown_fading_model(self, client):
        assert client.post("/api/experiments", json={"conditions": {"fading": "rayleigh"}}).status_code == 422

    def test_no_receivers_is_refused(self, client):
        assert client.post("/api/experiments", json={"receivers": []}).status_code == 422

    def test_an_untrained_hybrid_says_how_to_fix_it(self, client):
        response = client.post("/api/experiments", json={"modulation": "8PSK", "receivers": ["hybrid"]})
        assert response.status_code == 409
        assert "/api/models/" in response.json()["detail"]


# ======================================================================
# Stage tracing
# ======================================================================
class TestStages:
    def test_the_last_stage_is_what_the_bits_came_from(self, client):
        body = client.post("/api/experiments/stages", json={
            "modulation": "16QAM", "seed": 7, "conditions": {"snr_db": 12, "freq_offset_hz": 600},
        }).json()
        assert body["stages"][-1]["samples"] == body["result"]["symbols"]

    def test_stages_start_at_the_antenna_and_drop_to_symbol_rate(self, client):
        body = client.post("/api/experiments/stages", json={"seed": 1}).json()
        names = [s["name"] for s in body["stages"]]
        rates = [s["rate"] for s in body["stages"]]
        assert names[0] == "received" and rates[0] == "sample"
        assert rates[-1] == "symbol"
        assert "MatchedFilter" in names

    def test_frequency_correction_precedes_the_matched_filter(self, client):
        """The Phase 6 bug fix, visible through the API."""
        body = client.post("/api/experiments/stages", json={
            "modulation": "QPSK", "seed": 3, "conditions": {"snr_db": 10, "freq_offset_hz": 900},
        }).json()
        names = [s["name"] for s in body["stages"]]
        assert names.index("CoarseFreq") < names.index("MatchedFilter")


# ======================================================================
# Benchmarks and stored runs
# ======================================================================
@pytest.fixture(scope="module")
def finished(client):
    response = client.post("/api/benchmarks", json={
        "name": "test run", "modulations": ["QPSK", "16QAM"], "snrs_db": [0, 8],
        "receivers": ["classical", "classical-hard"], "frames_per_scenario": 2,
    })
    assert response.status_code == 202
    return wait_for(client, response.json()["id"])


class TestBenchmarks:
    def test_a_benchmark_runs_to_completion(self, finished):
        assert finished["state"] == "done", finished["error"]
        assert finished["result"]["cells"] == 2 * 2 * 2
        assert finished["fraction"] == 1.0

    def test_the_run_is_stored_and_listed(self, client, finished):
        runs = client.get("/api/runs").json()
        assert any(r["id"] == finished["result"]["run_id"] and r["name"] == "test run" for r in runs)

    def test_cells_can_be_filtered(self, client, finished):
        run_id = finished["result"]["run_id"]
        cells = client.get(f"/api/runs/{run_id}/cells", params={"receiver": "classical", "modulation": "QPSK"}).json()
        assert len(cells) == 2
        assert {c["receiver"] for c in cells} == {"classical"}
        assert all(len(c["frame_errors"]) == 2 for c in cells)

    def test_a_run_can_be_deleted(self, client):
        job = client.post("/api/benchmarks", json={
            "name": "doomed", "modulations": ["BPSK"], "snrs_db": [6],
            "receivers": ["classical"], "frames_per_scenario": 1,
        }).json()
        run_id = wait_for(client, job["id"])["result"]["run_id"]
        assert client.delete(f"/api/runs/{run_id}").status_code == 204
        assert client.get(f"/api/runs/{run_id}/cells").status_code == 404
        assert client.delete(f"/api/runs/{run_id}").status_code == 404

    def test_an_unknown_profile_fails_before_queuing(self, client):
        before = len(client.get("/api/jobs").json())
        response = client.post("/api/benchmarks", json={
            "name": "x", "modulations": ["QPSK"], "snrs_db": [0], "profiles": ["lunar"],
            "receivers": ["classical"],
        })
        assert response.status_code == 422
        assert len(client.get("/api/jobs").json()) == before

    def test_an_untrained_hybrid_fails_before_queuing(self, client):
        response = client.post("/api/benchmarks", json={
            "name": "x", "modulations": ["64QAM"], "snrs_db": [0], "receivers": ["hybrid"],
        })
        assert response.status_code == 409

    def test_missing_things_are_404(self, client):
        assert client.get("/api/jobs/nope").status_code == 404
        assert client.get("/api/runs/99999/cells").status_code == 404


@pytest.fixture(scope="module")
def tight(tmp_path_factory):
    settings = Settings(
        data_dir=tmp_path_factory.mktemp("tight"),
        max_frames_per_scenario=5, max_benchmark_cells=4,
    )
    with TestClient(create_app(settings)) as c:
        yield c


class TestLimits:
    def test_too_many_frames_is_refused(self, tight):
        response = tight.post("/api/benchmarks", json={
            "name": "x", "modulations": ["QPSK"], "snrs_db": [0], "receivers": ["classical"],
            "frames_per_scenario": 50,
        })
        assert response.status_code == 422
        assert response.json()["field"] == "frames_per_scenario"

    def test_too_large_a_grid_is_refused(self, tight):
        response = tight.post("/api/benchmarks", json={
            "name": "x", "modulations": ["QPSK", "16QAM"], "snrs_db": [0, 4, 8],
            "receivers": ["classical"], "frames_per_scenario": 1,
        })
        assert response.status_code == 422
        assert "exceeds the limit" in response.json()["detail"]


# ======================================================================
# Training hybrids
# ======================================================================
@pytest.fixture(scope="module")
def trained(client):
    response = client.post("/api/models/QPSK/train", json=TINY_TRAINING)
    assert response.status_code == 202
    return wait_for(client, response.json()["id"])


class TestModels:
    def test_training_completes_and_is_listed(self, client, trained):
        assert trained["state"] == "done", trained["error"]
        assert [m["modulation"] for m in client.get("/api/models").json()] == ["QPSK"]

    def test_the_hybrid_becomes_available(self, client, trained):
        receivers = {r["name"]: r for r in client.get("/api/receivers?modulation=QPSK").json()}
        assert receivers["hybrid"]["available_for"] == ["QPSK"]
        assert receivers["hybrid-noise"]["available_for"] == ["QPSK"]

    def test_the_hybrid_runs_on_the_same_frame(self, client, trained):
        body = experiment(client, modulation="QPSK", seed=8, conditions={"snr_db": 6},
                          receivers=["classical", "hybrid", "hybrid-noise"])
        hybrid, noise = body["results"][1], body["results"][2]
        # The noise-normalised wrapper changes confidence, never decisions.
        assert hybrid["symbol_wrong"] == noise["symbol_wrong"]

    def test_a_restarted_server_finds_the_model(self, data_dir, trained):
        with TestClient(create_app(Settings(data_dir=data_dir))) as fresh:
            assert "QPSK" in fresh.get("/api/health").json()["trained_models"]

    def test_a_bad_modulation_is_refused_at_once(self, client):
        assert client.post("/api/models/QAM9/train", json=TINY_TRAINING).status_code == 422

    def test_deleting_a_model(self, client, data_dir, trained):
        assert client.delete("/api/models/QPSK").status_code == 204
        assert client.get("/api/models").json() == []
        assert client.delete("/api/models/QPSK").status_code == 404


# ======================================================================
# The pieces, without HTTP
# ======================================================================
class TestJobManager:
    def test_a_job_moves_through_its_states(self):
        import threading

        manager = JobManager()
        seen = []
        handle = {}
        ready = threading.Event()

        def work(report):
            ready.wait(5)                       # until the test holds the handle
            seen.append(handle["job"].state)
            report(1, 2, "halfway")
            return {"answer": 42}

        job = handle["job"] = manager.submit("test", "a job", work)
        ready.set()
        final = manager.wait(job.id, timeout=10)
        assert seen == [JobState.RUNNING]
        assert final["state"] == "done" and final["result"] == {"answer": 42}
        assert final["done"] == final["total"] == 2
        manager.shutdown()

    def test_a_failing_job_reports_why_and_the_worker_survives(self):
        manager = JobManager()
        bad = manager.submit("test", "fails", lambda report: 1 / 0)
        assert manager.wait(bad.id)["state"] == "failed"
        assert "ZeroDivisionError" in manager.wait(bad.id)["error"]
        good = manager.submit("test", "runs after", lambda report: "fine")
        assert manager.wait(good.id)["result"] == "fine"
        manager.shutdown()

    def test_an_expected_error_is_reported_in_plain_words(self):
        manager = JobManager()

        def work(report):
            raise ValidationError("snr_db", 99, "too high")

        job = manager.submit("test", "x", work)
        assert manager.wait(job.id)["error"] == "invalid value for 'snr_db': 99 (too high)"
        manager.shutdown()

    def test_jobs_run_one_at_a_time_in_order(self):
        manager = JobManager()
        order = []
        jobs = [manager.submit("t", str(i), lambda r, i=i: (time.sleep(0.02), order.append(i))) for i in range(4)]
        for job in jobs:
            manager.wait(job.id)
        assert order == [0, 1, 2, 3]
        manager.shutdown()

    def test_an_unknown_job_is_a_key_error(self):
        with pytest.raises(KeyError):
            JobManager().get("missing")


class TestPieces:
    def test_settings_validate(self):
        with pytest.raises(ValidationError):
            Settings(port=0)
        with pytest.raises(ValidationError):
            Settings(max_frames_per_scenario=0)

    def test_settings_read_the_environment(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SATREC_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("SATREC_PORT", "8123")
        settings = Settings.from_env()
        assert settings.port == 8123
        assert settings.data_dir == tmp_path.resolve()
        assert settings.host == "127.0.0.1"

    def test_the_server_listens_only_locally_by_default(self):
        assert Settings().host == "127.0.0.1"

    def test_jsonable_makes_everything_plain(self):
        out = jsonable({"a": np.float32(1.5), "b": float("nan"), "c": np.arange(3),
                        "d": np.bool_(True), "e": (np.int64(2), None), "f": float("inf")})
        assert out == {"a": 1.5, "b": None, "c": [0, 1, 2], "d": True, "e": [2, None], "f": None}

    def test_the_catalog_refuses_duplicates_and_unknowns(self):
        catalog = ReceiverCatalog(RootRaisedCosine(8))
        entry = CatalogEntry("x", "X", "", lambda m: None)
        catalog.register(entry)
        with pytest.raises(ValidationError):
            catalog.register(entry)
        with pytest.raises(UnknownReceiverError):
            catalog.build("y", "QPSK")


# ======================================================================
# Real recordings (Phase 3), when a dataset is configured
# ======================================================================
class TestRecordings:
    def test_without_a_dataset_it_says_how_to_add_one(self, client):
        body = client.get("/api/dataset").json()
        assert body["available"] is False and "--rml24" in body["detail"]

    def test_the_recorded_profile_is_refused_with_the_reason(self, client):
        response = client.post("/api/benchmarks", json={
            "name": "x", "modulations": ["QPSK"], "snrs_db": [4], "profiles": ["recorded"],
            "receivers": ["classical"], "frames_per_scenario": 1,
        })
        assert response.status_code == 422 and "RML24" in response.json()["detail"]

    def test_a_mixed_benchmark_runs_on_both_sources(self, tmp_path):
        """Simulated and recorded profiles in one grid, routed per scenario."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent))
        from test_data import synthetic

        from satrecovery.data import SampleArchive

        recordings = [synthetic("QPSK", 4, seed=200 + k, row=k)[0] for k in range(3)]
        sample = SampleArchive.write(tmp_path / "sample.npz", recordings)
        settings = Settings(data_dir=tmp_path / "data", rml24_path=sample)
        with TestClient(create_app(settings)) as c:
            info = c.get("/api/dataset").json()
            assert info["available"] and info["modulations"] == ["QPSK"]
            job = c.post("/api/benchmarks", json={
                "name": "mixed", "modulations": ["QPSK", "16QAM"], "snrs_db": [6],
                "profiles": ["clean", "recorded"], "receivers": ["classical"],
                "frames_per_scenario": 2,
            }).json()
            done = wait_for(c, job["id"])
            assert done["state"] == "done", done
            cells = c.get(f"/api/runs/{done['result']['run_id']}/cells").json()
            profiles = sorted((cell["modulation"], cell["profile"]) for cell in cells)
            assert profiles == [("16QAM", "clean"), ("QPSK", "clean"), ("QPSK", "recorded")]
            # 16QAM was not recorded cleanly enough: skipped, and said so.
            assert any("16QAM" in s["scenario"] for s in done["result"]["skipped"])
