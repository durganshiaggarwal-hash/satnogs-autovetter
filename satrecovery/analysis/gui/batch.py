"""The "Many at once" tab: analyse a whole group of recordings in one go.

Pick a type, a quality, a speed and how many; the app analyses them one
after another, keeps a running score, and ends with one summary and a
table that can be saved for Excel (CSV).

One recording at a time, each as its own background call, so the
progress bar moves, the window never freezes, and Stop works at once.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...gui import theme
from ...gui.motion import PillButton
from ...gui.pages.base import Page
from ...gui.widgets import card


class BatchSummary:
    """The running score of a batch: kept apart from the screen so it can be tested on its own."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, row: dict) -> None:
        self.rows.append(dict(row))

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def correct(self) -> int:
        return sum(1 for r in self.rows if r.get("correct"))

    @property
    def checked(self) -> list[dict]:
        return [r for r in self.rows if r.get("checked")]

    @property
    def warned(self) -> int:
        """Answers the warning flagged (unsure or unfamiliar)."""
        return sum(1 for r in self.rows if r.get("trust") in ("unsure", "unfamiliar"))

    @property
    def bit_errors(self) -> int:
        return sum(int(r.get("bit_errors") or 0) for r in self.checked)

    @property
    def bits_checked(self) -> int:
        return sum(int(r.get("n_bits") or 0) for r in self.checked)

    @property
    def ber(self) -> float | None:
        return self.bit_errors / self.bits_checked if self.bits_checked else None

    FIELDS = ("row", "really", "ai_said", "confidence", "correct", "trust", "warning", "snr_label_db", "symbol_rate",
              "bits_recovered", "checked", "bit_errors", "n_bits", "ber")

    def write_csv(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
        return path


class BatchPage(Page):
    """Analyse many recordings, one after another, and sum them up."""

    def __init__(self, client, parent: QWidget | None = None) -> None:
        super().__init__(client, parent)
        self._queue: list[int] = []
        self._running = False
        self.summary = BatchSummary()

        # -- left: what to analyse -------------------------------------------
        box, form = card("Analyse many at once", "Choose a group of real RML24 recordings, then press Start.")
        self.kind = QComboBox()
        self.quality = QComboBox()
        self.quality.addItem("Any quality", None)
        for snr in range(20, -21, -2):
            self.quality.addItem(f"{snr:+d} dB", float(snr))
        self.quality.setCurrentIndex(self.quality.findData(16.0))
        self.rate = QComboBox()
        self.rate.addItem("Any speed", None)
        for rate in (100_000, 250_000, 500_000):
            self.rate.addItem(f"{rate // 1000} kbaud", float(rate))
        self.rate.setCurrentIndex(self.rate.findData(250_000.0))
        self.how_many = QSpinBox()
        self.how_many.setRange(5, 1000)
        self.how_many.setValue(50)
        self.how_many.setSingleStep(10)
        for label, widget in (("Signal type", self.kind), ("Signal quality", self.quality),
                              ("Speed", self.rate), ("How many recordings", self.how_many)):
            form.addWidget(QLabel(label))
            form.addWidget(widget)
        self.start_button = PillButton("Start", "primary")
        self.start_button.clicked.connect(self.toggle)
        form.addWidget(self.start_button)
        note = QLabel("Each recording takes about a second. Results here are not added to History.")
        note.setObjectName("Muted")
        note.setWordWrap(True)
        form.addWidget(note)
        left = QWidget()
        left.setFixedWidth(300)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(box)
        left_layout.addStretch(1)

        # -- right: progress, summary, table -------------------------------
        results, right = card("Results")
        self.status = QLabel("Not started.")
        self.status.setObjectName("Hint")
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(12)
        self.progress.setStyleSheet(
            f"QProgressBar {{ background: {theme.SUNK}; border: none; border-radius: 6px; }}"
            f"QProgressBar::chunk {{ background: {theme.gradient()}; border-radius: 6px; }}")
        numbers = QHBoxLayout()
        numbers.setSpacing(theme.GAP)
        self.ai_number = self._number(numbers, "AI named correctly")
        self.bits_number = self._number(numbers, "Bits wrong, where checked")
        self.unchecked_number = self._number(numbers, "Recordings not checkable")
        self.warned_number = self._number(numbers, "Answers flagged ⚠")
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Recording", "Quality", "Speed", "Really", "AI said", "How sure", "Bits"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMinimumHeight(360)
        self.save_button = PillButton("Save for Excel (.csv)", "secondary")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_csv)
        right.addWidget(self.status)
        right.addWidget(self.progress)
        right.addLayout(numbers)
        right.addWidget(self.table, 1)
        right.addWidget(self.save_button, 0, Qt.AlignLeft)

        top = QHBoxLayout()
        top.setSpacing(theme.GAP)
        top.addWidget(left, 0, Qt.AlignTop)
        top.addWidget(results, 1)
        page = QVBoxLayout(self)
        page.setContentsMargins(28, 18, 14, 10)
        page.setSpacing(10)
        page.addWidget(self.heading(self.title, "Test a whole group of real recordings and get one summary."))
        page.addLayout(top, 1)

    @staticmethod
    def _number(row: QHBoxLayout, caption: str) -> QLabel:
        column = QVBoxLayout()
        label = QLabel(caption)
        label.setObjectName("Muted")
        value = QLabel("—")
        value.setObjectName("BigNumber")
        column.addWidget(label)
        column.addWidget(value)
        row.addLayout(column, 1)
        return value

    @property
    def title(self) -> str:
        return "Many at once"

    @property
    def running(self) -> bool:
        return self._running

    def refresh(self) -> None:
        self.tasks.run(self.client.analysis_status, self._on_status, self.report_error)

    def _on_status(self, status: dict) -> None:
        dataset = status["dataset"]
        if self.kind.count() == 0:
            self.kind.addItem("Any type", None)
            for name in status["classifier"]["classes"]:
                self.kind.addItem(name, name)
            self.kind.setCurrentIndex(max(self.kind.findData("8PSK"), 0))
        ready = bool(dataset.get("available")) and bool(status["classifier"]["available"])
        self.start_button.setEnabled(ready)
        if not dataset.get("available"):
            self.status.setText("Choose the RML24 dataset first (Dataset, at the top right).")
        elif not status["classifier"]["available"]:
            self.status.setText("Train the AI first: python scripts/train_classifier.py <path to RML24>")

    # -- running -------------------------------------------------------------
    def toggle(self) -> None:
        if self._running:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        kind, quality, rate = self.kind.currentData(), self.quality.currentData(), self.rate.currentData()
        count = int(self.how_many.value())
        self.summary = BatchSummary()
        self.table.setRowCount(0)
        self._show_numbers()
        self._running = True
        self.start_button.setText("Stop")
        self.save_button.setEnabled(False)
        self.status.setText("Picking recordings…")
        self.tasks.run(lambda: self.client.batch_rows(kind, quality, rate, count), self._got_rows, self._failed)

    def stop(self) -> None:
        self._queue = []
        self._finish(stopped=True)

    def _got_rows(self, rows: list[int]) -> None:
        if not self._running:
            return
        if not rows:
            self.status.setText("No recordings match those choices.")
            self._finish()
            return
        self._queue = list(rows)
        self._total = len(rows)
        self.progress.setRange(0, self._total)
        self.progress.setValue(0)
        self._next()

    def _next(self) -> None:
        if not self._running:
            return
        if not self._queue:
            self._finish()
            return
        row = self._queue.pop(0)
        self.status.setText(f"Analysing… {self.summary.n + 1} of {self._total}")
        self.tasks.run(lambda: self.client.quick(row), self._got_one, self._failed_one)

    def _got_one(self, result: dict) -> None:
        if not self._running:
            return
        self.summary.add(result)
        self._add_row(result)
        self.progress.setValue(self.summary.n)
        self._show_numbers()
        self._next()

    def _failed_one(self, error) -> None:
        self.report_error(error)
        self._next()

    def _failed(self, error) -> None:
        self.report_error(error)
        self._finish(stopped=True)

    def _finish(self, stopped: bool = False) -> None:
        self._running = False
        self.start_button.setText("Start")
        self.save_button.setEnabled(self.summary.n > 0)
        if self.summary.n:
            word = "Stopped" if stopped else "Done"
            self.status.setText(f"{word}: {self.summary.n} recordings analysed.")
            if not stopped:
                self.notify.emit(f"Many at once: {self.summary.correct} of {self.summary.n} named correctly.",
                                 "success")

    # -- showing -------------------------------------------------------------
    def _show_numbers(self) -> None:
        s = self.summary
        self.ai_number.setText(f"{s.correct} of {s.n}" if s.n else "—")
        self.bits_number.setText(f"{100 * s.ber:.2f}%" if s.ber is not None else "—")
        self.unchecked_number.setText(str(s.n - len(s.checked)) if s.n else "—")
        self.warned_number.setText(str(s.warned) if s.n else "—")

    def _add_row(self, r: dict) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        bits = (f"{r['bit_errors']:,} of {r['n_bits']:,} wrong" if r.get("checked") else "not checked")
        cells = [f"#{r['row']:,}", f"{r['snr_label_db']:+.0f} dB" if r.get("snr_label_db") is not None else "",
                 f"{(r.get('symbol_rate') or 0) / 1000:.0f} kbaud", r.get("really") or "", r["ai_said"],
                 f"{r['confidence']:.0%}" + ("  ⚠" if r.get("trust") in ("unsure", "unfamiliar") else ""), bits]
        for column, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if column == 4:
                item.setForeground(QColor(theme.GOOD if r.get("correct") else theme.CRITICAL))
            if column == 5 and r.get("warning"):
                item.setToolTip(r["warning"])
            self.table.setItem(row, column, item)

    def save_csv(self) -> None:
        if not self.summary.n:
            return
        suggested = str(Path.home() / f"many_at_once_{datetime.now():%Y%m%d_%H%M%S}.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Save for Excel", suggested, "CSV files (*.csv)")
        if path:
            try:
                written = self.summary.write_csv(path)
            except OSError as error:
                self.report_error(f"Could not save: {error}")
                return
            self.notify.emit(f"Saved: {written.name}", "success")


__all__ = ["BatchPage", "BatchSummary"]
