"""Tests for the analyser's additions: history, remembered settings, the
dataset button, many-at-once, the help guide, dark mode and the window."""

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("torch")

from satrecovery.analysis import Analyser, RecordedInput, SimulatedInput  # noqa: E402
from satrecovery.analysis.classifier import ModulationClassifier  # noqa: E402
from satrecovery.analysis.history import AnalysisHistory  # noqa: E402
from satrecovery.analysis.preferences import Preferences  # noqa: E402
from satrecovery.core.conditions import ChannelConditions  # noqa: E402
from satrecovery.core.exceptions import ValidationError  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_analysis import Oracle, archive, archive_file, catalog  # noqa: E402,F401

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(autouse=True)
def private_preferences(tmp_path, monkeypatch):
    """Never touch the real home folder's settings."""
    monkeypatch.setenv("SATREC_PREFERENCES", str(tmp_path / "prefs.json"))


@pytest.fixture
def results(archive, catalog):
    analyser = Analyser(Oracle("QPSK"), catalog)
    return [
        analyser.analyse(RecordedInput(archive, int(archive.select("QPSK")[0]))),
        analyser.analyse(SimulatedInput("QPSK", ChannelConditions(snr_db=12.0), 250_000, seed=3)),
        Analyser(Oracle("8PSK"), catalog).analyse(RecordedInput(archive, int(archive.select("BPSK")[0]))),
    ]


# ======================================================================
class TestHistory:
    def test_analyses_come_back_newest_first_and_exactly_as_they_were(self, tmp_path, results):
        history = AnalysisHistory(tmp_path / "h.db")
        ids = [history.add(r) for r in results]
        entries = history.entries()
        assert [e["id"] for e in entries] == ids[::-1] and len(history) == 3
        again = history.get(ids[0])
        assert again["classification"] == results[0]["classification"]
        assert again["recovery"]["n_bits"] == results[0]["recovery"]["n_bits"]
        assert again["history"]["id"] == ids[0]

    def test_the_filter_finds_by_any_word(self, tmp_path, results):
        history = AnalysisHistory(tmp_path / "h.db")
        for r in results:
            history.add(r)
        assert {e["kind"] for e in history.entries("Simulated")} == {"simulated"}
        assert any(e["verdict"].startswith("wrong") for e in history.entries("wrong"))
        assert all(e["ai_label"] == "8PSK" for e in history.entries("8PSK"))
        assert history.entries("nothing-like-this") == []

    def test_the_summary_says_right_wrong_or_unknown(self, results):
        assert AnalysisHistory.summarise(results[0])["verdict"] == "right"
        assert AnalysisHistory.summarise(results[2])["verdict"] == "wrong (really BPSK)"

    def test_it_forgets_on_request_and_keeps_only_the_newest(self, tmp_path, results, monkeypatch):
        history = AnalysisHistory(tmp_path / "h.db")
        monkeypatch.setattr(AnalysisHistory, "LIMIT", 2)
        for r in results:
            history.add(r)
        assert len(history) == 2
        assert history.clear() == 2 and len(history) == 0
        with pytest.raises(ValidationError):
            history.get(1)
        with pytest.raises(ValidationError):
            history.add({"not": "an analysis"})


class TestPreferences:
    def test_settings_are_remembered_between_runs(self, tmp_path):
        first = Preferences(tmp_path / "p.json")
        first.rml24_path = "/data/RML24_IQdata"
        first.dark_mode = True
        second = Preferences(tmp_path / "p.json")
        assert second.rml24_path == "/data/RML24_IQdata" and second.dark_mode is True

    def test_a_damaged_file_means_nothing_remembered(self, tmp_path):
        path = tmp_path / "p.json"
        path.write_text("{ not json", encoding="utf-8")
        prefs = Preferences(path)
        assert prefs.rml24_path is None and prefs.dark_mode is False


class TestBatchSummary:
    def test_the_score_and_the_csv(self, tmp_path):
        from satrecovery.analysis.gui.batch import BatchSummary

        summary = BatchSummary()
        summary.add({"row": 1, "correct": True, "checked": True, "bit_errors": 2, "n_bits": 1000, "ai_said": "8PSK"})
        summary.add({"row": 2, "correct": False, "checked": False, "ai_said": "QPSK"})
        summary.add({"row": 3, "correct": True, "checked": True, "bit_errors": 0, "n_bits": 1000, "ai_said": "8PSK"})
        assert (summary.n, summary.correct, len(summary.checked)) == (3, 2, 2)
        assert summary.ber == pytest.approx(0.001)
        written = summary.write_csv(tmp_path / "out")
        lines = written.read_text(encoding="utf-8").splitlines()
        assert written.suffix == ".csv" and lines[0].startswith("row,") and len(lines) == 4

    def test_an_empty_batch_has_no_error_rate(self):
        from satrecovery.analysis.gui.batch import BatchSummary

        assert BatchSummary().ber is None


class TestDarkTheme:
    def test_switching_changes_the_colours_and_back(self):
        from satrecovery.gui import theme

        light_bg, light_ink = theme.BG, theme.INK
        try:
            theme.use_dark(True)
            assert theme.DARK and theme.BG != light_bg and theme.INK != light_ink
            assert theme.BRAND == (theme.PEACH, theme.BLUSH, theme.SAND)
            assert theme.tint("#000000", 1.0) == theme.SURFACE           # tints stay dark
            assert theme.INK in theme.stylesheet()
        finally:
            theme.use_dark(False)
        assert (theme.BG, theme.INK, theme.DARK) == (light_bg, light_ink, False)
        assert theme.tint("#000000", 1.0) == "#ffffff"                    # the light look is untouched


# ======================================================================
class TestApiAdditions:
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
        with TestClient(create_analysis_app(Settings(data_dir=tmp_path / "data"), path)) as c:
            yield c

    def test_choosing_a_dataset_while_running(self, client, archive_file, tmp_path):
        assert client.get("/api/analysis/status").json()["dataset"]["available"] is False
        bad = client.post("/api/analysis/dataset", json={"path": str(tmp_path / "nowhere")})
        assert bad.status_code == 422 and "not an RML24 dataset" in bad.json()["detail"]
        good = client.post("/api/analysis/dataset", json={"path": str(archive_file)})
        assert good.status_code == 200 and good.json()["recordings"] == 12
        assert client.get("/api/analysis/status").json()["dataset"]["available"] is True

    def test_every_analysis_lands_in_the_history(self, client, archive_file):
        client.post("/api/analysis/dataset", json={"path": str(archive_file)})
        row = client.get("/api/analysis/recordings", params={"modulation": "QPSK", "seed": 1}).json()[0]["row"]
        first = client.post("/api/analysis/recording", json={"row": row}).json()
        client.post("/api/analysis/simulated", json={"modulation": "BPSK", "seed": 2})
        body = {"name": "qpsk_250kbaud.csv", "symbol_rate": 250000,
                "content_base64": base64.b64encode((EXAMPLES / "qpsk_250kbaud.csv").read_bytes()).decode()}
        client.post("/api/analysis/file", json=body)
        entries = client.get("/api/analysis/history").json()
        assert [e["kind"] for e in entries] == ["file", "simulated", "recorded"]
        again = client.get(f"/api/analysis/history/{first['history_id']}").json()
        assert again["classification"] == first["classification"]
        assert client.get("/api/analysis/history", params={"q": "Simulated"}).json()[0]["kind"] == "simulated"
        assert client.get("/api/analysis/history/99999").status_code == 404
        assert client.delete("/api/analysis/history").json()["removed"] == 3
        assert client.get("/api/analysis/history").json() == []

    def test_many_at_once(self, client, archive_file):
        client.post("/api/analysis/dataset", json={"path": str(archive_file)})
        rows = client.get("/api/analysis/batch-rows", params={"modulation": "8PSK", "count": 50, "seed": 0}).json()
        assert len(rows) == 4 and rows == sorted(rows)
        one = client.post("/api/analysis/quick", json={"row": rows[0]}).json()
        assert one["really"] == "8PSK" and one["row"] == rows[0] and "stages" not in one
        assert client.get("/api/analysis/history").json() == []          # quick ones are not kept


# ======================================================================
class TestWindow:
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

    def test_six_tabs_no_dataset_and_results_open_in_the_right_one(self, app, tmp_path):
        from satrecovery.analysis.gui.client import AnalysisClient
        from satrecovery.analysis.gui.window import AnalysisWindow

        window = AnalysisWindow(AnalysisClient("http://127.0.0.1:9"), Preferences(tmp_path / "p.json"))
        # The Real signal tab (it needed the RML24 dataset) is gone; everything runs on the trained AI alone.
        assert [p.title for p in window.pages] == ["Simulated signal", "Your own file", "History",
                                                   "Many at once", "How good is the AI?", "Compare models"]
        assert not hasattr(window, "dataset_button")
        assert window.page_for({"input": {"kind": "simulated"}}) is window.simulated
        assert window.page_for({"input": {"kind": "file"}}) is window.file
        assert window.page_for({"input": {"kind": "recorded"}}) is window.file   # older RML24 analyses
        dialog = window.show_help()
        dialog.show_page(1)
        assert dialog.pages.currentIndex() == 1 and dialog.words_button.isChecked()
        dialog.close()
        window.close()

    def test_dark_switch_waits_while_an_analysis_runs(self, app, tmp_path):
        from satrecovery.analysis.gui.client import AnalysisClient
        from satrecovery.analysis.gui.window import AnalysisWindow

        prefs = Preferences(tmp_path / "p.json")
        window = AnalysisWindow(AnalysisClient("http://127.0.0.1:9"), prefs)
        asked = []
        window.theme_changed.connect(asked.append)
        window._busy.add(window.simulated)
        window.dark_switch.setChecked(True)
        assert asked == [] and not window.dark_switch.isChecked() and prefs.dark_mode is False
        window._busy.clear()
        window.dark_switch.setChecked(True)
        assert asked == [True] and prefs.dark_mode is True
        window.close()

    def test_a_reopened_analysis_says_so(self, app, archive, catalog):
        from satrecovery.analysis.gui.pages import SimulatedSignalPage
        from satrecovery.gui.motion import Motion

        result = Analyser(Oracle("QPSK"), catalog).analyse(
            SimulatedInput("QPSK", ChannelConditions(snr_db=12.0), 250_000, seed=4))
        result["history"] = {"id": 7, "saved_at": time.time()}
        before, Motion.enabled = Motion.enabled, False
        try:
            page = SimulatedSignalPage(None)
            page.show_saved(result)
            deadline = time.time() + 5
            while not page.save_button.isEnabled() and time.time() < deadline:
                app.processEvents()
                time.sleep(0.01)
            assert page.reopened.isVisibleTo(page) and "Opened from History" in page.reopened.text()
            assert page.save_button.isEnabled() and page.last_result is result
        finally:
            Motion.enabled = before

    def test_the_history_tab_lists_what_it_is_given(self, app):
        from satrecovery.analysis.gui.history import HistoryPage

        page = HistoryPage(None)
        page._show([{"id": 3, "saved_at": time.time(), "kind": "recorded", "source": "RML24 #184,700, +18 dB",
                     "ai_label": "8PSK", "confidence": 0.97, "verdict": "right", "bits": "2 of 1,080 wrong"},
                    {"id": 2, "saved_at": time.time(), "kind": "file", "source": "File a.csv",
                     "ai_label": "QPSK", "confidence": 0.9, "verdict": "unknown", "bits": "not checked"}])
        opened = []
        page.open_entry.connect(opened.append)
        page._open(1, 0)
        assert page.table.rowCount() == 2 and opened == [2] and not page.empty.isVisibleTo(page)


def test_the_double_click_launcher_keeps_data_beside_the_app(monkeypatch, tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packaging"))
    import app_launcher

    mac = tmp_path / "dist" / "Satellite Signal Analyser.app" / "Contents" / "MacOS" / "Satellite Signal Analyser"
    monkeypatch.setattr(sys, "executable", str(mac))
    assert app_launcher.app_folder() == tmp_path / "dist"
    windows = tmp_path / "Satellite Signal Analyser" / "Satellite Signal Analyser.exe"
    monkeypatch.setattr(sys, "executable", str(windows))
    assert app_launcher.app_folder() == tmp_path / "Satellite Signal Analyser"
