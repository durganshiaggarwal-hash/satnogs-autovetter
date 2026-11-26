"""The 'How good is the AI?' screen: reading saved comparisons, the table, the training command."""

from __future__ import annotations

import json
import os

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
