"""Results, as score cards and as a full table.

The score cards are what a user reads first: one per receiver, its
colour across the top, how many bits it got wrong counting up to the
new number.  The table holds every metric for anyone who wants them,
tucked into a section that opens on request so it does not crowd the
screen.

The headings are plain words.  Hovering over each one gives the
technical name and what it measures, so the numbers work both for
someone meeting these ideas for the first time and for an examiner who
wants to know exactly which metric is shown.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..motion import CountUpLabel


def percent(value: float | None) -> str:
    """An error rate as a readable percentage."""
    if value is None:
        return "–"
    if value == 0:
        return "0"
    pct = 100.0 * value
    if pct >= 1:
        return f"{pct:.1f}%"
    if pct >= 0.01:
        return f"{pct:.2f}%"
    return f"{pct:.0e}%"


COLUMNS = [
    ("Receiver", "Which receiver produced this row."),
    ("Bits wrong", "How many of the signal's bits the receiver got wrong."),
    ("Error rate", "Bit error rate: the fraction of bits that were wrong."),
    ("Likely range", "95% confidence interval for the error rate. One signal is a small sample, "
                     "so the true rate could be anywhere in here."),
    ("Spots its mistakes", "Ranking quality, 0 to 1. If the receiver threw away the bits it was "
                           "least sure about, how well would that remove its errors? 1 is perfect, "
                           "0 is no better than chance. Blank when there were no errors to find."),
    ("Honesty gap", "Expected calibration error. When the receiver says it is 90% sure, how far "
                    "is that from how often it is actually right? Smaller is more honest."),
    ("Signal quality", "Error vector magnitude: how far the recovered symbols sit from the ideal "
                       "points, as a percentage. Smaller is cleaner."),
]


class ScoreCard(QFrame):
    """One receiver's headline: its colour, its name, its count of wrong bits."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._colour = QColor(theme.INK_3)
        self.name = QLabel("")
        self.name.setObjectName("CardTitle")
        self.number = CountUpLabel(lambda v: f"{v:,.0f}")
        self.number.setObjectName("BigNumber")
        self.caption = QLabel("")
        self.caption.setObjectName("Hint")
        self.detail = QLabel("")
        self.detail.setObjectName("Muted")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(2)
        layout.addWidget(self.name)
        layout.addWidget(self.number)
        layout.addWidget(self.caption)
        layout.addWidget(self.detail)
        self.setMinimumWidth(170)

    def show_result(self, result: dict) -> None:
        self._colour = QColor(theme.receiver_colour(result["receiver"]))
        self.name.setText(result["label"])
        self.number.set_value(result["bit_errors"])
        if result["bit_errors"] == 0:
            self.caption.setText(f"bits wrong of {result['n_bits']:,}: every one right")
        else:
            self.caption.setText(f"bits wrong of {result['n_bits']:,}, {percent(result['ber'])}")
        ranking = result.get("ranking_quality")
        self.detail.setText(
            "No mistakes to spot" if ranking is None else f"Spots its mistakes: {ranking:.2f} of 1"
        )
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - the colour bar across the top
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._colour)
        painter.drawRoundedRect(14, 8, 34, 5, 2.5, 2.5)


class ResultsTable(QTableWidget):
    """One row per receiver, in the receiver's colour, numbers aligned."""

    ROW = 36

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, len(COLUMNS), parent)
        for column, (name, tip) in enumerate(COLUMNS):
            header = QTableWidgetItem(name)
            header.setToolTip(tip)
            header.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter if column == 0 else Qt.AlignRight | Qt.AlignVCenter)
            self.setHorizontalHeaderItem(column, header)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setFocusPolicy(Qt.NoFocus)
        self.setShowGrid(False)
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, len(COLUMNS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.verticalHeader().setDefaultSectionSize(self.ROW)
        self._mono = QFont(theme.MONO, theme.BASE_PT - 1)

    def show_results(self, results: list[dict]) -> None:
        self.setRowCount(len(results))
        for row, result in enumerate(results):
            ranking = result.get("ranking_quality")
            evm = result.get("evm_percent")
            values = [
                result["label"],
                f"{result['bit_errors']} of {result['n_bits']}",
                percent(result["ber"]),
                f"{percent(result['ber_low'])} – {percent(result['ber_high'])}",
                "–" if ranking is None else f"{ranking:.2f}",
                f"{result['ece']:.3f}",
                "–" if evm is None else f"{evm:.1f}%",
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column == 0:
                    item.setForeground(QColor(theme.INK))
                    item.setData(Qt.DecorationRole, QColor(theme.receiver_colour(result["receiver"])))
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                else:
                    item.setFont(self._mono)
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    if column == 1 and result["bit_errors"] > 0:
                        item.setForeground(QColor(theme.CRITICAL))
                self.setItem(row, column, item)
        self.setFixedHeight(self.horizontalHeader().height() + self.ROW * max(len(results), 1) + 6)


__all__ = ["COLUMNS", "ResultsTable", "ScoreCard", "percent"]
