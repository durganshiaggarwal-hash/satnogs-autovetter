"""Tests for the analysis app: the AI classifier, the analyser, its API and its screens.

The classifier tests train real (tiny) networks on simulated signals, so
they check learning actually happens, not only that the code runs.  The
analyser tests use an "oracle" classifier -- a subclass that always
gives a chosen answer -- so the recovery path can be tested both when
the AI is right and when it is wrong, without depending on training.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from satrecovery.analysis import (  # noqa: E402
    STAGES,
    Analyser,
    Classification,
    ModulationClassifier,
    RecordedInput,
    SimulatedInput,
)
from satrecovery.analysis.dataset import accuracy_by_snr, confusion, load_set, split_rows  # noqa: E402
from satrecovery.backend.catalog import ReceiverCatalog  # noqa: E402
from satrecovery.core.conditions import ChannelConditions  # noqa: E402
from satrecovery.core.exceptions import ModelNotTrainedError, ValidationError  # noqa: E402
from satrecovery.data import SampleArchive  # noqa: E402
from satrecovery.simulation.pulse import RootRaisedCosine  # noqa: E402
from satrecovery.simulation.simulator import SignalSimulator  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_data import synthetic  # noqa: E402


class Oracle(ModulationClassifier):
    """Always answers ``answer`` -- a stand-in for a perfectly (or wrongly) trained model."""

    def __init__(self, answer: str | None = None) -> None:
        super().__init__(device="cpu")
        self.answer = answer
        self._mark_trained()

    def classify(self, samples):
        return Classification({self.answer: 0.9, **{c: 0.1 / 5 for c in self.classes if c != self.answer}})


@pytest.fixture(scope="module")
def archive_file(tmp_path_factory):
    recordings, row = [], 0
    for modulation in ("BPSK", "QPSK", "8PSK"):
        for k in range(4):
            recordings.append(synthetic(modulation, 4, seed=300 + row, row=row)[0])
            row += 1
    return SampleArchive.write(tmp_path_factory.mktemp("an") / "s.npz", recordings)


@pytest.fixture(scope="module")
def archive(archive_file):
    return SampleArchive(archive_file)


@pytest.fixture(scope="module")
def catalog():
    return ReceiverCatalog.standard(RootRaisedCosine(8))


# ======================================================================
# The classifier
# ======================================================================
class TestClassification:
    def test_answers_are_ranked_and_complete(self):
        c = Classification({"QPSK": 0.2, "BPSK": 0.7, "8PSK": 0.1})
        assert c.label == "BPSK" and c.confidence == pytest.approx(0.7)
        assert list(c.probabilities) == ["BPSK", "QPSK", "8PSK"]
        assert c.top(2) == [("BPSK", 0.7), ("QPSK", 0.2)]

    def test_an_empty_answer_is_refused(self):
        with pytest.raises(ValidationError):
            Classification({})


class TestModulationClassifier:
    def test_it_is_a_trainable_model(self):
        from satrecovery.learning.base import TrainableModel

        assert isinstance(ModulationClassifier(device="cpu"), TrainableModel)

    def test_recordings_are_levelled_before_the_network_sees_them(self):
        loud = 50 * (np.exp(1j * np.arange(2048)) + 0.3)
        x = ModulationClassifier.prepare(loud)
        assert x.shape == (1, 2, 2048)
        z = x[0, 0] + 1j * x[0, 1]
        assert abs(z.mean()) < 1e-3 and np.mean(np.abs(z) ** 2) == pytest.approx(1.0, rel=1e-3)

    def test_rml24_pairs_and_complex_samples_agree(self):
        z = np.random.default_rng(0).standard_normal(2048) + 1j
        pairs = np.stack([z.real, z.imag], axis=1)[None].astype(np.float32)
        assert np.allclose(ModulationClassifier.prepare(z), ModulationClassifier.prepare(pairs), atol=1e-5)

    def test_an_untrained_model_refuses_to_answer(self):
        with pytest.raises(ModelNotTrainedError):
            ModulationClassifier(device="cpu").classify(np.ones(2048, complex))

    def test_it_learns_to_tell_bpsk_from_qpsk(self, tmp_path):
        """A real (small) training run: it must do much better than a coin."""
        rng = np.random.default_rng(1)
        sim = SignalSimulator(samples_per_symbol=4, seed=1)
        classes = ("BPSK", "QPSK")
        x, y = [], []
        for k in range(240):
            frame = sim.generate_frame(classes[k % 2], ChannelConditions(
                snr_db=15, freq_offset_hz=float(rng.uniform(-500, 500)), phase_offset_rad=float(rng.uniform(0, 6))))
            x.append(frame.signal.samples)
            y.append(k % 2)
        model = ModulationClassifier(classes=classes, channels=(16, 32), device="cpu")
        x = model.prepare(np.array(x))
        y = np.array(y)
        model.fit(x[:200], y[:200], validation=(x[200:], y[200:]), epochs=6, batch_size=20, patience=6, verbose=False)
        assert model.evaluate(x[200:], y[200:]) >= 0.85

        model.record_test_accuracy(0.9)
        saved = model.save(tmp_path / "clf")
        again = ModulationClassifier.from_file(saved, device="cpu")
        assert again.classes == classes and again.test_accuracy == 0.9
        assert np.allclose(again.predict_proba(x[200:205]), model.predict_proba(x[200:205]), atol=1e-5)

    def test_loading_a_model_saved_for_other_classes_is_refused(self, tmp_path):
        model = ModulationClassifier(classes=("BPSK", "QPSK"), channels=(8,), device="cpu")
        model._mark_trained()
        path = model.save(tmp_path / "two")
        with pytest.raises(ValidationError):
            ModulationClassifier(classes=("8PSK", "QPSK"), channels=(8,), device="cpu").load(path)


class TestDataset:
    def test_training_and_test_recordings_never_overlap(self, archive):
        train, test = split_rows(archive, ("BPSK", "QPSK", "8PSK"), 100, 40)
        assert train and test
        assert not {r for r, _, _ in train} & {r for r, _, _ in test}

    def test_a_scheme_missing_from_the_archive_is_refused(self, archive):
        with pytest.raises(ValidationError):
            split_rows(archive, ("16QAM",), 10, 5)

    def test_sets_load_with_labels_and_measure(self, archive):
        _, test = split_rows(archive, ("BPSK", "QPSK", "8PSK"), 100, 40)
        data = load_set(archive, test)
        assert data.inputs.shape[1:] == (2, 2048) and len(data) == len(test)
        model = ModulationClassifier(classes=("BPSK", "QPSK", "8PSK"), channels=(8,), device="cpu")
        model._mark_trained()                       # untrained weights: only the bookkeeping is tested
        assert set(accuracy_by_snr(model, data)) == {20.0}
        assert confusion(model, data).sum() == len(data)


# ======================================================================
# The analyser
# ======================================================================
class TestAnalyser:
    def test_right_answer_gives_checked_bits_and_a_full_movie(self, archive, catalog):
        row = int(archive.select("QPSK")[0])
        result = Analyser(Oracle("QPSK"), catalog).analyse(RecordedInput(archive, row))
        assert result["classification"]["correct"]
        recovery = result["recovery"]
        assert [s["name"] for s in recovery["stages"]][0] == "received"
        assert {s["name"] for s in recovery["stages"]} <= set(STAGES)
        sizes = {len(s["points"]["i"]) for s in recovery["stages"]}
        assert len(sizes) == 1                       # every stage: one point per symbol
        assert len(recovery["colours"]) == sizes.pop()
        assert recovery["check"]["available"] and recovery["check"]["bit_errors"] == 0

    def test_a_wrong_answer_is_decoded_as_told_and_not_graded(self, archive, catalog):
        row = int(archive.select("QPSK")[0])
        result = Analyser(Oracle("8PSK"), catalog).analyse(RecordedInput(archive, row))
        assert not result["classification"]["correct"]
        assert result["recovery"]["modulation"] == "8PSK"
        assert not result["recovery"]["check"]["available"]
        assert "wrong" in result["recovery"]["check"]["detail"]

    def test_a_simulated_signal_is_checked_against_what_was_sent(self, catalog):
        signal = SimulatedInput("8PSK", ChannelConditions(snr_db=12, freq_offset_hz=500), 100_000, seed=4)
        assert signal.samples_per_symbol == 10
        result = Analyser(Oracle("8PSK"), catalog).analyse(signal)
        assert result["recovery"]["check"]["available"]
        assert result["recovery"]["check"]["ber"] < 0.01

    def test_points_end_up_in_their_clusters(self, archive, catalog):
        """The last frame of the movie: every point near a constellation point."""
        row = int(archive.select("BPSK")[0])
        recovery = Analyser(Oracle("BPSK"), catalog).analyse(RecordedInput(archive, row))["recovery"]
        last = np.array(recovery["stages"][-1]["points"]["i"]) + 1j * np.array(recovery["stages"][-1]["points"]["q"])
        ideal = np.array(recovery["constellation"]["i"]) + 1j * np.array(recovery["constellation"]["q"])
        distance = np.min(np.abs(last[8:-8, None] - ideal[None]), axis=1)
        assert np.median(distance) < 0.3

    def test_only_a_classifier_is_accepted(self, catalog):
        with pytest.raises(ValidationError):
            Analyser(object(), catalog)


# ======================================================================
# The API: the existing backend plus new endpoints
# ======================================================================
class TestApi:
    @pytest.fixture
    def make_client(self, tmp_path, archive_file):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from satrecovery.analysis.api import create_analysis_app
        from satrecovery.backend import Settings

        def build(with_model: bool):
            path = tmp_path / "models" / "classifier.pt"
            if with_model:
                model = ModulationClassifier(channels=(8,), device="cpu")
                model._mark_trained()
                model.save(path)
            settings = Settings(data_dir=tmp_path / "data", rml24_path=archive_file)
            return TestClient(create_analysis_app(settings, path))

        return build

    def test_without_a_trained_classifier_it_says_how_to_train_one(self, make_client):
        with make_client(False) as c:
            assert c.get("/api/analysis/status").json()["classifier"]["available"] is False
            response = c.post("/api/analysis/simulated", json={"modulation": "QPSK"})
            assert response.status_code == 409 and "train_classifier.py" in response.json()["detail"]

    def test_the_endpoints_answer(self, make_client):
        with make_client(True) as c:
            assert c.get("/api/health").json()["status"] == "ok"            # the old API still there
            picks = c.get("/api/analysis/recordings", params={"modulation": "8PSK", "count": 2, "seed": 1}).json()
            assert len(picks) == 2 and {p["modulation"] for p in picks} == {"8PSK"}
            assert all(isinstance(p["checkable"], bool) for p in picks)
            result = c.post("/api/analysis/recording", json={"row": picks[0]["row"]})
            assert result.status_code == 200 and result.json()["input"]["kind"] == "recorded"
            sim = c.post("/api/analysis/simulated", json={"modulation": "BPSK", "symbol_rate": 250000, "seed": 2})
            assert sim.status_code == 200 and sim.json()["input"]["kind"] == "simulated"

    def test_without_the_dataset_the_ai_is_tested_on_simulated_signals(self, make_client):
        with make_client(True) as c:
            report = c.post("/api/analysis/evaluate-simulated", json={"per_group": 1, "seed": 3}).json()
            # 6 types x 21 RML24 signal qualities x 3 speeds, one signal each
            assert report["n"] == 6 * 21 * 3 and report["source"] == "simulated"
            assert len(report["by_snr"]) == 21 and len(report["confusion"]["matrix"]) == 6
            assert all({"seed", "symbol_rate", "true", "predicted"} <= set(m) for m in report["mistakes"])
            body = {"modulation": "8PSK", "conditions": {"snr_db": 12}, "symbol_rate": 250000, "seed": 5}
            one = c.post("/api/analysis/simulated-quick", json=body).json()
            assert one["really"] == "8PSK" and one["seed"] == 5 and one["snr_label_db"] == 12
            again = c.post("/api/analysis/simulated-quick", json=body).json()
            assert again["ai_said"] == one["ai_said"]                      # the same seed makes the same signal
            assert c.get("/api/analysis/history").json() == []              # quick answers are not kept

    def test_the_model_card_reads_what_was_saved_with_the_model(self, make_client):
        with make_client(True) as c:
            card = c.get("/api/analysis/model-card").json()
            assert card["classes"] and len(card["loss"]) == len(card["val_loss"])

    def test_bad_requests_are_refused(self, make_client):
        with make_client(True) as c:
            assert c.post("/api/analysis/recording", json={"row": -1}).status_code == 422
            assert c.post("/api/analysis/simulated", json={"modulation": "QPSK", "colour": 1}).status_code == 422
            assert c.post("/api/analysis/simulated", json={"symbol_rate": 300000}).status_code == 422


# ======================================================================
# The screens
# ======================================================================
class TestScreens:
    @pytest.fixture(scope="class")
    @classmethod
    def app(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        from satrecovery.gui import theme

        application = QApplication.instance() or QApplication([])
        theme.load_fonts()
        theme.configure_plots()
        return application

    def test_the_shared_page_is_abstract(self, app):
        from satrecovery.analysis.gui.pages import AnalysisPage

        with pytest.raises(TypeError):
            AnalysisPage(None)

    def test_the_movie_jumps_to_the_end_with_animations_off(self, app, archive, catalog):
        from satrecovery.analysis.gui.widgets import RecoveryAnimation
        from satrecovery.gui.motion import Motion

        recovery = Analyser(Oracle("QPSK"), catalog).analyse(
            RecordedInput(archive, int(archive.select("QPSK")[0])))["recovery"]
        before, Motion.enabled = Motion.enabled, False
        try:
            movie = RecoveryAnimation()
            finished = []
            movie.finished.connect(lambda: finished.append(True))
            movie.play(recovery)
            assert finished and movie.current_stage == movie.n_stages - 1
        finally:
            Motion.enabled = before

    def test_the_movie_plays_through_every_stage(self, app, archive, catalog):
        from satrecovery.analysis.gui.widgets import RecoveryAnimation

        recovery = Analyser(Oracle("BPSK"), catalog).analyse(
            RecordedInput(archive, int(archive.select("BPSK")[0])))["recovery"]
        movie = RecoveryAnimation()
        movie.MOVE_MS, movie.HOLD_MS = 40, 30
        seen, finished = [], []
        movie.stage_changed.connect(seen.append)
        movie.finished.connect(lambda: finished.append(True))
        movie.play(recovery)
        deadline = time.time() + 10
        while not finished and time.time() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert finished and seen == list(range(movie.n_stages))


def test_changing_a_filter_picks_a_matching_recording(tmp_path, archive_file):
    """The recording analysed must always match what the dropdowns say."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from satrecovery.analysis.gui.client import AnalysisClient
    from satrecovery.analysis.gui.pages import RealSignalPage
    from satrecovery.analysis.gui.server import AnalysisServer

    app = QApplication.instance() or QApplication([])
    path = tmp_path / "models" / "classifier.pt"
    model = ModulationClassifier(channels=(8,), device="cpu")
    model._mark_trained()
    model.save(path)
    with AnalysisServer(data_dir=tmp_path / "data", rml24_path=archive_file, classifier_path=path) as server:
        page = RealSignalPage(AnalysisClient(server.url))
        page.ensure_loaded()
        deadline = time.time() + 20
        while page.kind.findData("8PSK") < 0 and time.time() < deadline:
            app.processEvents()
            time.sleep(0.01)
        page.kind.setCurrentIndex(page.kind.findData("8PSK"))
        deadline = time.time() + 20
        while (page._row is None or "8PSK" not in page._picked_modulation()) and time.time() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert page._row is not None and page._picked_modulation() == "8PSK"
        page.rate.setCurrentIndex(page.rate.findData(250_000.0))
        marked = [page.quality.itemText(i) for i in range(page.quality.count()) if "bits checked" in page.quality.itemText(i)]
        assert marked and all(int(t.split()[0]) >= 16 for t in marked)


def test_a_phase_slip_at_the_start_of_a_real_recording_is_caught(catalog):
    """RML24 #187,269 (8PSK, +20 dB): the loop slipped 45 degrees at the cut edge.

    Before the fix every symbol came out one bit wrong (487 of 1,461
    bits).  The check pattern after the edge must turn it back.
    """
    from satrecovery.data import SampleArchive

    archive = SampleArchive(Path(__file__).parent / "data" / "rml24_row_187269.npz")
    recovery = Analyser(Oracle("8PSK"), catalog).analyse(RecordedInput(archive, 187269))["recovery"]
    assert recovery["check"]["available"]
    assert recovery["check"]["bit_errors"] == 0
    assert "slipped" in recovery["check"]["detail"]


def test_a_phase_slip_in_the_middle_of_a_real_recording_is_caught(catalog):
    """RML24 #184,700 (8PSK, +18 dB): slipped at symbol 38, after the start.

    A check only at the start missed it (460 of 1,458 bits wrong); the
    pattern repeated every 32 symbols must catch it.
    """
    from satrecovery.data import SampleArchive

    archive = SampleArchive(Path(__file__).parent / "data" / "rml24_row_184700.npz")
    recovery = Analyser(Oracle("8PSK"), catalog).analyse(RecordedInput(archive, 184700))["recovery"]
    assert recovery["check"]["available"]
    # A couple of symbols right at the slip may still be wrong; the rest
    # of the frame must not be.  The pattern's own bits are not counted.
    assert recovery["check"]["bit_errors"] / recovery["check"]["n_bits"] < 0.01
    assert recovery["check"]["pilot_bits_excluded"] > 0


def test_where_bits_can_be_checked_follows_the_measurements():
    from satrecovery.analysis.inputs import checkable_from, usually_checkable

    assert checkable_from("8PSK", 250_000) == 16 and checkable_from("8PSK", 500_000) is None
    assert usually_checkable("8PSK", 18, 250_000) and not usually_checkable("8PSK", 14, 250_000)
    assert usually_checkable("BPSK", 8, 250_000) and not usually_checkable("64QAM", 20, 100_000)


# ======================================================================
# Your own file, and the PDF report
# ======================================================================
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


class TestOwnFile:
    def _samples(self, n=2048):
        rng = np.random.default_rng(5)
        return np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, n)))

    def test_a_csv_with_a_header_is_read(self):
        from satrecovery.analysis.upload import SignalFile

        z = self._samples(10)
        text = "I,Q\n" + "".join(f"{v.real},{v.imag}\n" for v in z)
        assert np.allclose(SignalFile.read("s.csv", text.encode()), z)

    def test_numpy_files_are_read_as_complex_or_pairs(self):
        import io

        from satrecovery.analysis.upload import SignalFile

        z = self._samples(10)
        for array in (z, np.stack([z.real, z.imag], axis=1)):
            buffer = io.BytesIO()
            np.save(buffer, array)
            assert np.allclose(SignalFile.read("s.npy", buffer.getvalue()), z)

    def test_bad_files_are_refused_in_plain_words(self):
        from satrecovery.analysis.upload import FileInput, SignalFile

        with pytest.raises(ValidationError, match="use one of"):
            SignalFile.read("s.doc", b"123")
        with pytest.raises(ValidationError, match="two columns"):
            SignalFile.read("s.csv", b"1\n2\n3\n")
        with pytest.raises(ValidationError, match="at least"):
            FileInput("s.csv", self._samples(100), 250_000)
        with pytest.raises(ValidationError, match="kbaud"):
            FileInput("s.csv", self._samples(), 300_000)

    def test_a_file_is_analysed_blind_and_its_truth_is_unknown(self, catalog):
        from satrecovery.analysis.upload import FileInput

        result = Analyser(Oracle("QPSK"), catalog).analyse(FileInput("s.csv", self._samples(), 250_000))
        assert result["input"]["kind"] == "file"
        assert result["classification"]["true_label"] is None and not result["classification"]["correct"]
        assert result["recovery"]["n_bits"] > 0 and not result["recovery"]["check"]["available"]

    def test_an_rml24_export_keeps_its_labels(self):
        from satrecovery.analysis.upload import input_from_file

        signal = input_from_file("row.npz", (EXAMPLES / "rml24_8psk_recording_184700.npz").read_bytes(), None)
        assert isinstance(signal, RecordedInput) and signal.true_modulation == "8PSK"

    def test_the_example_files_open(self):
        from satrecovery.analysis.upload import input_from_file

        for name, rate in (("qpsk_250kbaud.csv", 250_000), ("16qam_100kbaud.npy", 100_000)):
            signal = input_from_file(name, (EXAMPLES / name).read_bytes(), rate)
            assert signal.samples.size >= 2000 and signal.samples_per_symbol == 1_000_000 // rate

    def test_the_file_endpoint(self, tmp_path, archive_file):
        pytest.importorskip("fastapi")
        import base64

        from fastapi.testclient import TestClient

        from satrecovery.analysis.api import create_analysis_app
        from satrecovery.backend import Settings

        path = tmp_path / "models" / "classifier.pt"
        model = ModulationClassifier(channels=(8,), device="cpu")
        model._mark_trained()
        model.save(path)
        with TestClient(create_analysis_app(Settings(data_dir=tmp_path / "d", rml24_path=archive_file), path)) as c:
            def send(name, rate=None):
                body = {"name": name, "content_base64": base64.b64encode((EXAMPLES / name).read_bytes()).decode()}
                if rate:
                    body["symbol_rate"] = rate
                return c.post("/api/analysis/file", json=body)

            csv = send("qpsk_250kbaud.csv", 250000)
            assert csv.status_code == 200 and csv.json()["input"]["kind"] == "file"
            assert send("rml24_8psk_recording_184700.npz").json()["input"]["kind"] == "recorded"
            assert send("qpsk_250kbaud.csv").status_code == 422             # speed missing
            bad = c.post("/api/analysis/file", json={"name": "x.csv", "content_base64": "%%%"})
            assert bad.status_code == 422


class TestPdfReport:
    @pytest.fixture(scope="class")
    @classmethod
    def app(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        from satrecovery.gui import theme

        application = QApplication.instance() or QApplication([])
        theme.load_fonts()
        theme.configure_plots()
        return application

    def test_a_report_is_written_for_every_kind_of_signal(self, app, tmp_path, archive, catalog):
        from PySide6.QtGui import QPixmap

        from satrecovery.analysis.gui.pdf import AnalysisReportWriter
        from satrecovery.analysis.upload import FileInput

        analyser = Analyser(Oracle("QPSK"), catalog)
        results = [
            analyser.analyse(RecordedInput(archive, int(archive.select("QPSK")[0]))),
            analyser.analyse(SimulatedInput("QPSK", ChannelConditions(snr_db=12.0), 250_000, seed=1)),
            analyser.analyse(FileInput("mine.csv", TestOwnFile()._samples(), 250_000)),
        ]
        picture = QPixmap(200, 100)
        picture.fill()
        for k, result in enumerate(results):
            written = AnalysisReportWriter(result, [("A picture", picture)]).write(tmp_path / f"r{k}")
            assert written.suffix == ".pdf" and written.read_bytes()[:4] == b"%PDF"
        assert "cannot be marked right or wrong" in AnalysisReportWriter(results[2]).html()
        assert "Checked:" in AnalysisReportWriter(results[0]).html()

    def test_the_page_saves_its_latest_analysis(self, app, tmp_path, archive, catalog):
        from satrecovery.analysis.gui.pages import FilePage

        page = FilePage(None)
        assert not page.save_button.isEnabled() and page.write_report(str(tmp_path / "none.pdf")) is None
        page._on_result(Analyser(Oracle("BPSK"), catalog).analyse(
            RecordedInput(archive, int(archive.select("BPSK")[0]))))
        page._reveal_bits()
        assert page.save_button.isEnabled()
        written = page.write_report(str(tmp_path / "report.pdf"))
        assert written and Path(written).read_bytes()[:4] == b"%PDF"
