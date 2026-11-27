"""The 'How good is the AI?' screen: reading saved comparisons, the table, the training command."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")


def _result(accuracy, auroc, alone5, alone10, ece, seconds):
    return {"accuracy": accuracy, "balanced_accuracy": accuracy, "auroc": auroc, "ece": ece,
            "decide_alone_5pc": alone5, "decide_alone_10pc": alone10, "train_seconds": seconds}


@pytest.fixture()
def board(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from satrecovery.vetting.gui import ai_view

    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "run_a_full.json").write_text(json.dumps({
        "n": 100, "stations": 10, "folds": 5, "epochs": 30, "limit": 0, "finished": "2026-11-25 10:00",
        "results": {"forest": _result(0.78, 0.86, 0.30, 0.59, 0.027, 1.0),
                    "cnn": _result(0.77, 0.86, 0.33, 0.54, 0.033, 40.0),
                    "cnn+forest": _result(0.79, 0.88, 0.40, 0.65, 0.034, 41.0)},
        "curves": {"forest": {"coverage": [0.1, 0.5, 1.0], "risk": [0.0, 0.05, 0.2]}}}), encoding="utf-8")
    monkeypatch.setattr(ai_view, "REPORT", tmp_path / "no_report.json")   # keep the real report out of the test
    widget = ai_view.ModelBoard(in_use="cnn+forest", runs_dir=runs)
    return widget, runs


class TestModelBoard:
    def test_reads_the_saved_run_and_fills_the_table(self, board):
        widget, _ = board
        assert widget.picker.count() == 1 and "full run" in widget.picker.itemText(0)
        assert widget.table.rowCount() == 3
        names = [widget.table.item(r, 0).text() for r in range(3)]
        assert names[2].startswith("CNN + forest") and "in use" in names[2]
        assert widget._run.best("auroc", True) == "cnn+forest"
        assert widget._run.best("ece", False) == "forest"           # lower is better for calibration error

    def test_the_combined_model_keeps_its_parts(self, board):
        widget, _ = board
        widget.checks["cnn"].setChecked(False)
        assert not widget.checks["cnn+forest"].isChecked()          # it cannot run without the CNN
        widget.checks["cnn+forest"].setChecked(True)
        assert widget.checks["cnn"].isChecked() and widget.checks["forest"].isChecked()

    def test_the_training_command_writes_to_the_apps_own_folder(self, board):
        widget, runs = board
        for name, box in widget.checks.items():
            box.setChecked(name in ("logistic", "forest"))
        widget._set_mode("Quick trial")
        args = widget.command()
        assert args[args.index("--models") + 1: args.index("--out")] == ["logistic", "forest"]
        assert args[args.index("--out") + 1] == str(runs) and "--limit" in args
        widget._set_mode("Full run")
        assert "--limit" not in widget.command()

    def test_switching_charts_and_picking_a_model(self, board):
        widget, _ = board
        widget.set_chart("Deciding alone")
        assert widget.charts.currentWidget() is widget.alone_plot
        widget.table.selectRow(0)
        assert widget._focus == "forest"


def _per_recording():
    # 6 recordings: two stations, two transmitter types, low and high passes
    return {"id": [1, 2, 3, 4, 5, 6], "label": [1, 1, 0, 0, 1, 0], "station": [7, 7, 7, 8, 8, 8],
            "elevation": [10.0, 70.0, 15.0, 65.0, 30.0, None], "mode": ["FM", "FM", "FM", "CW", "CW", "CW"],
            "decoded": [0, 0, 0, 0, 3, 0], "norad": [1] * 6, "lat": [50.0] * 3 + [None] * 3,
            "lng": [5.0] * 3 + [None] * 3,
            "scores": {"forest": [0.9, 0.4, 0.95, 0.05, 0.2, 0.5]}}


class TestRecordingResults:
    def test_confusion_uses_the_apps_rules(self):
        from satrecovery.vetting.gui.report_card import RecordingResults

        r = RecordingResults(_per_recording(), "forest")
        table = r.confusion()
        # rec 1: signal, sure -> heard; rec 2: signal, 0.4 -> unsure; rec 5: decoded packets -> heard
        assert table[0].tolist() == [2, 0, 1]
        # rec 3: no signal but 0.95 -> heard (wrong); rec 4: 0.05 -> not heard; rec 6: 0.5 -> unsure
        assert table[1].tolist() == [1, 1, 1]
        assert table.sum() == len(r) == 6

    def test_mistakes_are_most_confident_first_and_slices_count_right(self):
        from satrecovery.vetting.gui.report_card import RecordingResults

        r = RecordingResults(_per_recording(), "forest")
        assert [int(r.ids[i]) for i in r.mistakes()][:2] == [3, 5]      # 0.95 wrong, then 0.2 wrong
        modes, accuracy, count = r.heatmap()
        assert modes == ["FM", "CW"] and count.sum() == 5                 # rec 6 has no elevation
        assert np.isnan(accuracy).all()                                   # too few per cell to show
        assert [s[3] for s in r.stations()] == [3]                       # station 8 has no position

    def test_reliability_bins_cover_every_recording(self):
        from satrecovery.vetting.gui.report_card import RecordingResults

        r = RecordingResults(_per_recording(), "forest")
        assert sum(n for *_, n in r.reliability()) == 6

    def test_a_run_without_per_recording_results_shows_a_note(self, board):
        widget, _ = board
        assert widget._run.recording_results("forest") is None
        assert "no per-recording results" in widget.report.note.text()
