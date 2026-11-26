"""The AI's report card: how good is the classifier, and where does it fail?

A classifier that only ever shows its successes proves nothing.  This
tab tests the RML24-trained classifier on freshly simulated signals --
made the way RML24 was made, at every one of its signal qualities and
speeds, so no dataset is needed -- and shows three things:

* **accuracy at every signal quality**, from hopeless to perfectly clean;
* **which types it mixes up**, as a confusion matrix;
* **every mistake**, worst-first (mistakes on clean signals are the
  interesting ones).  Double-click one to make that very signal again in
  the Simulated signal tab and watch the AI get it wrong.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...gui import theme
from ...gui.motion import CountUpLabel, PillButton, fade_in, font
from ...gui.pages.base import Page
from ...gui.widgets import card


class ConfusionMatrix(QWidget):
    """Rows: what it really was.  Columns: what the AI said.  Shaded by share."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._classes: list[str] = []
        self._shares = np.zeros((0, 0))
        self._counts = np.zeros((0, 0), dtype=int)
        self.setMinimumSize(420, 420)

    def show_matrix(self, classes: list[str], matrix: list[list[int]]) -> None:
        self._classes = list(classes)
        self._counts = np.asarray(matrix, dtype=int)
        totals = np.maximum(self._counts.sum(axis=1, keepdims=True), 1)
        self._shares = self._counts / totals
        self.update()

    def share(self, really: str, said: str) -> float:
        i, j = self._classes.index(really), self._classes.index(said)
        return float(self._shares[i, j])

    def paintEvent(self, event) -> None:  # noqa: N802
        n = len(self._classes)
        if n == 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        left, top = 70.0, 30.0
        size = min((self.width() - left - 8) / n, (self.height() - top - 8) / n)
        painter.setFont(font(theme.SANS_MEDIUM, 8.5))
        painter.setPen(QColor(theme.INK_2))
        painter.drawText(QRectF(left, 0, size * n, 14), Qt.AlignCenter, "what the AI said  →")
        for j, name in enumerate(self._classes):
            painter.drawText(QRectF(left + j * size, 14, size, 16), Qt.AlignCenter, name)
        for i, name in enumerate(self._classes):
            painter.setPen(QColor(theme.INK_2))
            painter.drawText(QRectF(0, top + i * size, left - 8, size), Qt.AlignRight | Qt.AlignVCenter, name)
            for j in range(n):
                share = self._shares[i, j]
                rect = QRectF(left + j * size + 1.5, top + i * size + 1.5, size - 3, size - 3)
                if i == j:
                    base = QColor(theme.GOOD)
                else:
                    base = QColor(theme.CRITICAL)
                fill = QColor(base)
                fill.setAlphaF(0.08 + 0.85 * share if share > 0 else 0.04)
                painter.setPen(Qt.NoPen)
                painter.setBrush(fill)
                painter.drawRoundedRect(rect, 6, 6)
                if self._counts[i].sum() and share >= 0.005:
                    painter.setPen(QColor("white") if share > 0.55 else QColor(theme.INK))
                    painter.drawText(rect, Qt.AlignCenter, f"{share:.0%}")
        painter.end()


class ReportPage(Page):
    """How good is the AI?  The RML24-trained AI, tested on freshly simulated signals."""

    #: Emitted when a mistake is double-clicked: the signal's type, quality, speed and seed.
    open_signal = Signal(dict)

    def __init__(self, client, parent: QWidget | None = None) -> None:
        super().__init__(client, parent)
        self._result: dict | None = None
        self._rml24_accuracy: float | None = None

        # -- left: the test, and the headline numbers ------------------
        box, form = card(
            "Test the AI",
            "The AI was trained on RML24. Here it names brand-new simulated signals of all six types, made "
            "the way RML24 was made, at every one of RML24's 21 signal qualities and 3 speeds.")
        form.addWidget(QLabel("Signals per group"))
        self.per_group = QSpinBox()
        self.per_group.setRange(1, 20)
        self.per_group.setValue(3)
        self.per_group.setToolTip("A group is one type x one signal quality x one speed: 378 groups in all. "
                                  "3 per group is 1,134 signals, about half a minute.")
        form.addWidget(self.per_group)
        self.run_button = PillButton("Run the test", "primary")
        self.run_button.clicked.connect(self.run)
        form.addWidget(self.run_button)
        self.status = QLabel("Not run yet.")
        self.status.setObjectName("Hint")
        self.status.setWordWrap(True)
        form.addWidget(self.status)

        scores, scores_layout = card("The score")
        self.overall = CountUpLabel(lambda v: f"{v:.1f}%")
        self.overall.setObjectName("BigNumber")
        self.clean = CountUpLabel(lambda v: f"{v:.1f}%")
        self.clean.setObjectName("BigNumber")
        self.mistakes_count = QLabel("")
        self.mistakes_count.setObjectName("Hint")
        self.mistakes_count.setWordWrap(True)
        # How well the "unsure / unfamiliar" warning catches the AI's mistakes.
        self.warning_score = QLabel("")
        self.warning_score.setObjectName("Hint")
        self.warning_score.setWordWrap(True)
        for caption, widget in (("Overall, every signal quality", self.overall),
                                ("Clean signals, +10 dB and above", self.clean)):
            label = QLabel(caption)
            label.setObjectName("Muted")
            scores_layout.addWidget(label)
            scores_layout.addWidget(widget)
        scores_layout.addWidget(self.mistakes_count)
        scores_layout.addWidget(self.warning_score)

        left = QWidget()
        left.setFixedWidth(300)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(theme.GAP)
        left_layout.addWidget(box)
        left_layout.addWidget(scores)
        left_layout.addStretch(1)

        # -- right: the curve, the confusions, the mistakes ------------
        curve_card, curve_layout = card("Accuracy at every signal quality",
                                        "Below about -8 dB the noise is stronger than the signal, and no "
                                        "classifier can do better than guessing (the dashed line).")
        self.curve = pg.PlotWidget()
        self.curve.setMinimumHeight(360)
        self.curve.setMouseEnabled(False, False)
        self.curve.setMenuEnabled(False)
        self.curve.setYRange(0, 100)
        self.curve.setLabel("left", "named correctly (%)")
        self.curve.setLabel("bottom", "signal quality, SNR label (dB)")
        self.curve.showGrid(x=True, y=True, alpha=0.2)
        curve_layout.addWidget(self.curve)

        matrix_card, matrix_layout = card("Which types it mixes up",
                                          "Recordings at 0 dB and above. Green: right. Red: the mix-ups.")
        self.matrix = ConfusionMatrix()
        matrix_layout.addWidget(self.matrix)
        self.weakest = QLabel("")
        self.weakest.setObjectName("Hint")
        self.weakest.setWordWrap(True)
        matrix_layout.addWidget(self.weakest)

        table_card, table_layout = card("Only the mistakes, cleanest signals first",
                                        "This list shows ONLY the signals the AI got wrong, so the two "
                                        "columns never match -- that is what makes them mistakes. The ones it "
                                        "got right are not listed. Double-click one to make that exact "
                                        "signal again in the Simulated signal tab.")
        self.table_summary = QLabel("")
        self.table_summary.setWordWrap(True)
        self.table_summary.setStyleSheet(f"font-family: '{theme.SANS_SEMIBOLD}';")
        table_layout.addWidget(self.table_summary)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Signal", "Quality", "Really was", "AI said", "How sure", "Warned?"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMinimumHeight(460)
        self.table.cellDoubleClicked.connect(self._open)
        table_layout.addWidget(self.table)

        # One page that scrolls: the test and the curve on top, the
        # confusion matrix next, every mistake at the bottom -- each big
        # enough to read.
        top = QHBoxLayout()
        top.setSpacing(theme.GAP)
        top.addWidget(left, 0, Qt.AlignTop)
        top.addWidget(curve_card, 1)
        middle = QHBoxLayout()
        middle.setSpacing(theme.GAP)
        middle.addWidget(matrix_card, 1)
        content = QWidget()
        stack = QVBoxLayout(content)
        stack.setContentsMargins(0, 0, 14, 8)
        stack.setSpacing(theme.GAP)
        stack.addLayout(top)
        stack.addLayout(middle)
        stack.addWidget(table_card)
        body = QVBoxLayout()
        body.addWidget(self.scrolling(content), 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 18, 14, 10)
        layout.setSpacing(10)
        layout.addWidget(self.heading(self.title, "Where the AI is right, where it fails, and why — on signals "
                                                  "it has never seen."))
        layout.addLayout(body, 1)

    @property
    def title(self) -> str:
        return "How good is the AI?"

    def refresh(self) -> None:
        """Only the model's own RML24 score, for comparison; the test runs on request."""
        self.tasks.run(self.client.analysis_status, self._got_status, lambda _error: None)

    def _got_status(self, status: dict) -> None:
        self._rml24_accuracy = (status.get("classifier") or {}).get("test_accuracy")

    def run(self) -> None:
        per_group = int(self.per_group.value())
        self.run_button.setEnabled(False)
        self.status.setText(f"Making and testing {per_group * 378:,} simulated signals… "
                            f"(about {max(per_group * 10, 10)} s)")
        self.tasks.run(lambda: self.client.evaluate_simulated(per_group), self._show, self._failed)

    def _failed(self, error) -> None:
        self.run_button.setEnabled(True)
        self.status.setText("The test could not run.")
        self.report_error(error)

    def _show(self, result: dict) -> None:
        self._result = result
        self.run_button.setEnabled(True)
        self.status.setText(f"Tested on {result['n']:,} simulated signals the AI had never seen.")
        self.overall.set_value(100 * result["accuracy"])
        if result.get("accuracy_10db_up") is not None:
            self.clean.set_value(100 * result["accuracy_10db_up"])
        text = (f"{result['n_mistakes']:,} mistakes in all. Most are on very noisy signals, where nobody "
                "could tell.")
        if self._rml24_accuracy is not None:
            text += f" For comparison, on real RML24 test recordings it scored {100 * self._rml24_accuracy:.1f}%."
        self.mistakes_count.setText(text)

        self.curve.clear()
        xs = [row["snr_db"] for row in result["by_snr"]]
        ys = [100 * row["accuracy"] for row in result["by_snr"]]
        guess = 100 / max(len(result["confusion"]["classes"]), 1)
        self.curve.addItem(pg.InfiniteLine(guess, angle=0, pen=pg.mkPen(theme.INK_3, width=1, style=Qt.DashLine)))
        self.curve.plot(xs, ys, pen=pg.mkPen(theme.MAUVE, width=2.5), symbol="o", symbolSize=7,
                        symbolBrush=theme.MAUVE, symbolPen=None)

        classes = result["confusion"]["classes"]
        self.matrix.show_matrix(classes, result["confusion"]["matrix"])
        ranked = sorted((c for c in result["by_class"] if c["accuracy"] is not None), key=lambda c: c["accuracy"])
        if ranked:
            matrix = np.asarray(result["confusion"]["matrix"])
            worst = ranked[0]["name"]
            i = classes.index(worst)
            off = matrix[i].copy()
            off[i] = 0
            confused_with = classes[int(off.argmax())] if off.any() else None
            text = f"Hardest for the AI: {worst}, named correctly {ranked[0]['accuracy']:.0%} of the time"
            text += f", most often mistaken for {confused_with}." if confused_with else "."
            self.weakest.setText(text)
            fade_in(self.weakest)

        warnings = result.get("warnings") or {}
        if warnings.get("mistakes_flagged") is not None:
            text = (f"Warning check: it flagged {100 * warnings['mistakes_flagged']:.0f}% of the AI's mistakes, "
                    f"and {100 * (warnings.get('correct_flagged') or 0):.0f}% of its correct answers.")
            if warnings.get("accuracy_when_confident") is not None:
                text += (f" When it says \"Confident\", the AI is right "
                         f"{100 * warnings['accuracy_when_confident']:.1f}% of the time.")
            self.warning_score.setText(text)
        right_count = result["n"] - result["n_mistakes"]
        self.table_summary.setText(
            f"Tested {result['n']:,} signals: {right_count:,} named correctly (not listed), "
            f"{result['n_mistakes']:,} mistakes (listed below"
            + (f", the first {len(result['mistakes'])}" if len(result["mistakes"]) < result["n_mistakes"] else "")
            + ").")
        self.table.setRowCount(0)
        for mistake in result["mistakes"]:
            row = self.table.rowCount()
            self.table.insertRow(row)
            warned = mistake.get("warning")
            cells = [f"#{mistake['row']:,}", f"{mistake['snr_label_db']:+.0f} dB", mistake["true"],
                     mistake["predicted"], f"{mistake['confidence']:.0%}",
                     ("flagged: " + warned) if warned in ("unsure", "unfamiliar") else ("no" if warned else "")]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, dict(mistake))
                if column == 3:
                    item.setForeground(QColor(theme.CRITICAL))
                self.table.setItem(row, column, item)
        self.notify.emit(f"Tested: {100 * result['accuracy']:.1f}% overall, "
                         f"{100 * (result.get('accuracy_10db_up') or 0):.1f}% on clean signals.", "success")

    def _open(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if item is not None:
            self.open_signal.emit(dict(item.data(Qt.UserRole)))


__all__ = ["ConfusionMatrix", "ReportPage"]
