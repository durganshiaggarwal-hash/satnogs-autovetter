"""The History tab: every analysis so far, ready to open again.

Every analysis is saved by the backend as it happens (see
:class:`~satrecovery.analysis.history.AnalysisHistory`).  This tab lists
them, newest first, with a filter box; double-clicking one opens it
again in the tab it came from, exactly as it was.
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...gui import theme
from ...gui.motion import PillButton
from ...gui.pages.base import Page
from ...gui.widgets import card


class HistoryPage(Page):
    """Past analyses, newest first."""

    #: Emitted when an entry is double-clicked: its number in the history.
    open_entry = Signal(int)

    COLUMNS = ("When", "Signal", "AI said", "How sure", "Right?", "Bits")

    def __init__(self, client, parent: QWidget | None = None) -> None:
        super().__init__(client, parent)
        self._entries: list[dict] = []

        box, layout = card("Every analysis, newest first",
                           "Each analysis on the other tabs is saved here by itself. "
                           "Double-click one to open it again, exactly as it was.")
        controls = QHBoxLayout()
        controls.setSpacing(theme.GAP)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter: 8PSK, wrong, RML24, a file name…")
        self.filter.setClearButtonEnabled(True)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self.refresh)
        self.filter.textChanged.connect(lambda _text: self._debounce.start())
        self.count = QLabel("")
        self.count.setObjectName("Hint")
        self.clear_button = PillButton("Clear all", "secondary")
        self.clear_button.clicked.connect(self.clear_all)
        controls.addWidget(self.filter, 1)
        controls.addWidget(self.count)
        controls.addWidget(self.clear_button)
        layout.addLayout(controls)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setMinimumHeight(420)
        self.table.cellDoubleClicked.connect(self._open)
        layout.addWidget(self.table, 1)
        self.empty = QLabel("Nothing yet. Analyse a signal on any tab and it appears here.")
        self.empty.setObjectName("Hint")
        layout.addWidget(self.empty)

        page = QVBoxLayout(self)
        page.setContentsMargins(28, 18, 14, 10)
        page.setSpacing(10)
        page.addWidget(self.heading(self.title, "Every analysis is kept, so a good example is never lost."))
        page.addWidget(box, 1)

    @property
    def title(self) -> str:
        return "History"

    # -- loading -----------------------------------------------------------
    def refresh(self) -> None:
        query = self.filter.text().strip()
        self.tasks.run(lambda: self.client.history(query), self._show, self.report_error)

    def showEvent(self, event) -> None:  # noqa: N802
        # New analyses arrive while the tab is hidden, so look again each time it is shown.
        super().showEvent(event)
        if self.loaded:
            self.refresh()

    def _show(self, entries: list[dict]) -> None:
        self._entries = list(entries)
        self.table.setRowCount(0)
        for entry in self._entries:
            row = self.table.rowCount()
            self.table.insertRow(row)
            when = datetime.fromtimestamp(entry["saved_at"]).strftime("%d %b, %H:%M")
            verdict = entry["verdict"]
            cells = [when, entry["source"], entry["ai_label"], f"{entry['confidence']:.0%}",
                     "right" if verdict == "right" else (verdict if verdict.startswith("wrong") else "unknown"),
                     entry["bits"]]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, int(entry["id"]))
                if column == 4 and verdict == "right":
                    item.setForeground(QColor(theme.GOOD))
                elif column == 4 and verdict.startswith("wrong"):
                    item.setForeground(QColor(theme.CRITICAL))
                self.table.setItem(row, column, item)
        shown = len(self._entries)
        self.count.setText(f"{shown:,} shown" if shown else "")
        self.empty.setVisible(shown == 0)
        self.clear_button.setEnabled(shown > 0 or bool(self.filter.text()))

    # -- actions -------------------------------------------------------------
    def _open(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if item is not None:
            self.open_entry.emit(int(item.data(Qt.UserRole)))

    def clear_all(self) -> None:
        answer = QMessageBox.question(self, "Clear the history",
                                      "Forget every saved analysis? This cannot be undone.",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            self.tasks.run(self.client.clear_history, self._cleared, self.report_error)

    def _cleared(self, answer: dict) -> None:
        self.notify.emit(f"History cleared ({answer.get('removed', 0):,} removed).", "success")
        self.refresh()


__all__ = ["HistoryPage"]
