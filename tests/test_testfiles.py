"""Tests for test files: a signal saved together with its answer key."""

from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from satrecovery.analysis import Analyser, RecordedInput, SimulatedInput  # noqa: E402
from satrecovery.analysis.classifier import ModulationClassifier  # noqa: E402
from satrecovery.analysis.testfile import FORMAT, TestFile, TestFileInput  # noqa: E402
from satrecovery.analysis.upload import FileInput, input_from_file  # noqa: E402
from satrecovery.core.conditions import ChannelConditions  # noqa: E402
from satrecovery.core.exceptions import ValidationError  # noqa: E402
from satrecovery.data import SampleArchive  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_analysis import Oracle, archive, archive_file, catalog  # noqa: E402,F401
from test_data import synthetic  # noqa: E402

DATA = Path(__file__).parent / "data"
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(autouse=True)
def private_preferences(tmp_path, monkeypatch):
    monkeypatch.setenv("SATREC_PREFERENCES", str(tmp_path / "prefs.json"))


def _simulated(seed=21, snr=8.0, doppler=600.0, modulation="QPSK"):
    return SimulatedInput(modulation, ChannelConditions(snr_db=snr, freq_offset_hz=doppler), 250_000, seed=seed)


class TestSimulatedTestFiles:
    def test_the_file_is_graded_exactly_as_the_simulated_tab_was(self, catalog):
        signal = _simulated()
        name, data = TestFile.from_simulated(signal)
        assert name == "test_QPSK_+8dB_250kbaud_doppler600.npz"
        original = Analyser(Oracle("QPSK"), catalog).analyse(signal)
        loaded = input_from_file(name, data, None)
        assert isinstance(loaded, TestFileInput) and loaded.true_modulation == "QPSK"
        again = Analyser(Oracle("QPSK"), catalog).analyse(loaded)
        assert again["recovery"]["check"]["available"]
        for key in ("bit_errors", "n_bits"):
            assert again["recovery"]["check"][key] == original["recovery"]["check"][key]
        assert again["classification"]["correct"] and again["input"]["kind"] == "file"

    def test_a_wrong_ai_answer_is_marked_wrong(self, catalog):
        name, data = TestFile.from_simulated(_simulated())
        result = Analyser(Oracle("8PSK"), catalog).analyse(input_from_file(name, data, None))
        assert result["classification"]["true_label"] == "QPSK" and not result["classification"]["correct"]
        assert not result["recovery"]["check"]["available"]

    def test_the_answer_key_is_the_simulators_own(self):
        signal = _simulated()
        _, data = TestFile.from_simulated(signal)
        with np.load(io.BytesIO(data)) as archive:
            assert str(archive["format"]) == FORMAT
            assert np.array_equal(archive["bits"], signal.simulated_frame.bits)
            assert np.allclose(archive["samples"], signal.samples)

    def test_what_is_inside_is_described_in_plain_words(self):
        _, data = TestFile.from_simulated(_simulated())
        info = TestFile.describe(data)
        assert info["answer_key"] == "simulator" and info["symbol_rate"] == 250_000 and info["n_bits"] > 0
        assert "answer key inside" in TestFile.summary(info) and "600 Hz Doppler" in TestFile.summary(info)

    def test_a_damaged_test_file_is_refused_plainly(self):
        buffer = io.BytesIO()
        np.savez(buffer, format=np.array(FORMAT), kind=np.array("simulated"), samples=np.zeros(10))
        with pytest.raises(ValidationError, match="damaged test file"):
            input_from_file("bad.npz", buffer.getvalue(), None)


class TestRealTestFiles:
    def test_an_rml24_test_file_is_graded_like_the_real_signal_tab(self, catalog):
        recording = RecordedInput(SampleArchive(DATA / "rml24_row_184700.npz"), 184700)
        name, data = TestFile.from_recording(recording)
        assert name == "test_RML24_184700_8PSK_+18dB_250kbaud.npz"
        assert TestFile.describe(data)["answer_key"] == "phase3"
        original = Analyser(Oracle("8PSK"), catalog).analyse(recording)
        loaded = input_from_file(name, data, None)
        again = Analyser(Oracle("8PSK"), catalog).analyse(loaded)
        assert again["recovery"]["check"]["bit_errors"] == original["recovery"]["check"]["bit_errors"]
        assert again["recovery"]["check"]["n_bits"] == original["recovery"]["check"]["n_bits"]
        assert again["input"]["file"] == name and again["input"]["row"] == 184700

    def test_a_recording_too_noisy_to_check_keeps_only_its_true_type(self, tmp_path, catalog):
        noisy = synthetic("8PSK", 4, seed=5, snr_db=4.0, row=9, snr_label=4.0)[0]
        archive = SampleArchive(SampleArchive.write(tmp_path / "n.npz", [noisy]))
        name, data = TestFile.from_recording(RecordedInput(archive, 9))
        info = TestFile.describe(data)
        assert info["answer_key"] == "none" and "no answer key" in TestFile.summary(info)
        result = Analyser(Oracle("8PSK"), catalog).analyse(input_from_file(name, data, None))
        assert result["classification"]["correct"] and not result["recovery"]["check"]["available"]

    def test_plain_files_are_not_test_files(self):
        assert TestFile.describe((EXAMPLES / "16qam_100kbaud.npy").read_bytes()) is None
        assert TestFile.describe(b"I,Q\n1,2\n") is None
        assert isinstance(input_from_file("q.csv", (EXAMPLES / "qpsk_250kbaud.csv").read_bytes(), 250_000), FileInput)

    def test_the_example_test_files_open(self):
        for name in ("test_QPSK_+8dB_250kbaud_doppler600.npz", "test_RML24_184700_8PSK_+18dB_250kbaud.npz"):
            assert TestFile.describe_path(EXAMPLES / name) is not None
            input_from_file(name, (EXAMPLES / name).read_bytes(), None)


class TestApi:
    @pytest.fixture
    def client(self, tmp_path, archive_file):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from satrecovery.analysis.api import create_analysis_app
        from satrecovery.backend import Settings

        path = tmp_path / "models" / "classifier.pt"
        model = ModulationClassifier(channels=(8,), device="cpu")
        model._mark_trained()
        model.save(path)
        with TestClient(create_analysis_app(Settings(data_dir=tmp_path / "data", rml24_path=archive_file), path)) as c:
            yield c

    def test_save_then_load_round_trip(self, client):
        simulated = client.post("/api/analysis/simulated", json={"modulation": "BPSK", "seed": 4}).json()
        made = client.post("/api/analysis/test-file", json={"input": simulated["input"]}).json()
        assert made["name"].startswith("test_BPSK") and made["info"]["answer_key"] == "simulator"
        loaded = client.post("/api/analysis/file", json={"name": made["name"],
                                                         "content_base64": made["content_base64"]}).json()
        assert loaded["classification"]["true_label"] == "BPSK" and loaded["input"]["test_file"] == "simulated"

        row = client.get("/api/analysis/recordings", params={"modulation": "QPSK", "seed": 1}).json()[0]["row"]
        real = client.post("/api/analysis/recording", json={"row": row}).json()
        made = client.post("/api/analysis/test-file", json={"input": real["input"]}).json()
        loaded = client.post("/api/analysis/file", json={"name": made["name"],
                                                         "content_base64": made["content_base64"]}).json()
        assert loaded["input"]["row"] == row and loaded["input"]["file"] == made["name"]

    def test_only_real_and_simulated_signals_can_be_saved(self, client):
        refused = client.post("/api/analysis/test-file", json={"input": {"kind": "file", "name": "a.csv"}})
        assert refused.status_code == 422
        assert client.post("/api/analysis/test-file", json={"input": {"kind": "simulated"}}).status_code == 422


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

    def test_the_file_tab_recognises_a_test_file(self, app):
        from satrecovery.analysis.gui.pages import FilePage

        page = FilePage(None)
        page.rate.setCurrentIndex(0)
        page.set_file(EXAMPLES / "test_QPSK_+8dB_250kbaud_doppler600.npz")
        assert page.test_info.isVisibleTo(page) and "answer key inside" in page.test_info.text()
        assert not page.rate.isEnabled() and page.rate.currentData() == 250_000.0
        page.set_file(EXAMPLES / "qpsk_250kbaud.csv")
        assert not page.test_info.isVisibleTo(page) and page.rate.isEnabled()

    def test_only_real_and_simulated_tabs_offer_to_save(self, app, catalog):
        from satrecovery.analysis.gui.pages import FilePage, SimulatedSignalPage
        from satrecovery.gui.motion import Motion

        result = Analyser(Oracle("QPSK"), catalog).analyse(_simulated())
        before, Motion.enabled = Motion.enabled, False
        try:
            page = SimulatedSignalPage(None)
            page._on_result(result)
            deadline = time.time() + 5
            while not page.save_button.isEnabled() and time.time() < deadline:
                app.processEvents()
                time.sleep(0.01)
            assert page.test_button.isEnabled() and page.test_file_source()["seed"] == 21
            assert not FilePage(None).test_button.isVisibleTo(FilePage(None))
        finally:
            Motion.enabled = before

    def test_saving_writes_the_file(self, app, tmp_path):
        from satrecovery.analysis.gui.pages import SimulatedSignalPage

        name, data = TestFile.from_simulated(_simulated())
        page = SimulatedSignalPage(None)
        written = page.write_test_file(str(tmp_path / "mine"), {"name": name, "content_base64": base64.b64encode(data).decode(),
                                                                "info": TestFile.describe(data)})
        assert written.endswith(".npz") and Path(written).read_bytes() == data

    def test_an_rml24_file_reopens_in_the_file_tab(self, app, tmp_path):
        from satrecovery.analysis.gui.client import AnalysisClient
        from satrecovery.analysis.gui.window import AnalysisWindow
        from satrecovery.analysis.preferences import Preferences

        window = AnalysisWindow(AnalysisClient("http://127.0.0.1:9"), Preferences(tmp_path / "p.json"))
        assert window.page_for({"input": {"kind": "recorded", "file": "test_RML24.npz"}}) is window.file
        assert window.page_for({"input": {"kind": "recorded"}}) is window.real
        window.close()
