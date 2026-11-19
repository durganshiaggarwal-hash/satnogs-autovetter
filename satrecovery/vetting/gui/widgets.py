"""The vetting station's parts: the queue, the scope, the meter and the reasons."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QFrame, QLabel, QStyledItemDelegate, QVBoxLayout, QWidget

from ..session import Item, Verdict, VettingSession, track_points
from . import theme

pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")


def caps(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("caps")
    return label


def rule() -> QFrame:
    line = QFrame()
    line.setObjectName("rule")
    return line


# ----------------------------------------------------------------------
class QueueModel(QAbstractTableModel):
    """The list of observations, in the chosen order."""

    HEADERS = ("OBS", "SAT", "MODE", "P(SIGNAL)", "DONE")
    MODE_CHARS = 10

    def __init__(self, session: VettingSession) -> None:
        super().__init__()
        self._session = session
        self._order = "unsure first"
        self._rows: list[Item] = session.ordered(self._order)

    def set_order(self, order: str) -> None:
        self.beginResetModel()
        self._order = order
        self._rows = self._session.ordered(order)
        self.endResetModel()

    def refresh(self) -> None:
        """Re-read decisions without reordering (so the list does not jump while you work)."""
        self._rows = [self._session.item(i.obs.id) for i in self._rows]  # verdicts change with the policy
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))

    def item_at(self, row: int) -> Item | None:
        return self._rows[row] if 0 <= row < len(self._rows) else None

    def row_of(self, obs_id: int) -> int:
        for row, item in enumerate(self._rows):
            if item.obs.id == obs_id:
                return row
        return -1

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        item = self.item_at(index.row())
        if item is None:
            return None
        column = index.column()
        decision = self._session.decision(item.obs.id)
        if role == Qt.DisplayRole:
            if column == 0:
                return str(item.obs.id)
            if column == 1:
                return str(item.obs.norad_cat_id or "?")
            if column == 2:
                return (item.obs.transmitter_mode or "?")[:self.MODE_CHARS]
            if column == 3:
                return f"{item.probability:.2f}"
            if column == 4:
                if decision is None:
                    return ""
                mark = "S" if decision.label.value == "with-signal" else "N"
                return f"{mark}·{'you' if decision.by == 'you' else 'auto'}"
        if role == Qt.ForegroundRole:
            if column == 4 and decision is not None:
                return QColor(theme.SIGNAL if decision.label.value == "with-signal" else theme.NO_SIGNAL)
            if column in (0, 1, 2):
                return QColor(theme.FAINT if decision is not None else theme.TEXT)
        if role == Qt.UserRole:
            return item
        return None


class ProbabilityDelegate(QStyledItemDelegate):
    """Draws P(signal) as a thin bar with the two thresholds marked."""

    def __init__(self, session: VettingSession, parent=None) -> None:
        super().__init__(parent)
        self._session = session

    def paint(self, painter: QPainter, option, index) -> None:
        item: Item = index.data(Qt.UserRole)
        if item is None:
            return super().paint(painter, option, index)
        painter.save()
        rect = option.rect.adjusted(6, 0, -6, 0)
        bar = QRectF(rect.left(), rect.center().y() - 2, rect.width() - 34, 4)
        painter.fillRect(bar, QColor(theme.RULE))
        filled = QRectF(bar.left(), bar.top(), bar.width() * item.probability, bar.height())
        painter.fillRect(filled, QColor(theme.verdict_colour(item.verdict.value)))
        painter.setPen(QPen(QColor(theme.DIM), 1))
        for edge in (self._session.policy.low, self._session.policy.high):
            x = bar.left() + bar.width() * edge
            painter.drawLine(int(x), int(bar.top() - 3), int(x), int(bar.bottom() + 3))
        painter.setPen(QColor(theme.TEXT))
        font = QFont(theme.MONO)
        font.setPixelSize(10)
        painter.setFont(font)
        painter.drawText(QRectF(bar.right() + 4, rect.top(), 30, rect.height()),
                         Qt.AlignVCenter | Qt.AlignLeft, f"{item.probability:.2f}")
        painter.restore()


# ----------------------------------------------------------------------
class Scope(pg.GraphicsLayoutWidget):
    """The waterfall, with the frequency profile below and the centre-band strength beside it."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setBackground(theme.BG)
        self.ci.setSpacing(2)
        axis_pen = pg.mkPen(theme.FAINT)
        text_pen = pg.mkPen(theme.DIM)

        self.plot = self.addPlot(row=0, col=0)
        self.image = pg.ImageItem()
        self.image.setColorMap(pg.colormap.get("inferno"))
        self.plot.addItem(self.image)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.setLabel("left", "seconds into pass")
        self.plot.getAxis("bottom").setStyle(showValues=False)
        self.centre_lines = [pg.InfiniteLine(angle=90, pen=pg.mkPen(theme.ACCENT, width=1, style=Qt.DashLine))
                             for _ in range(2)]
        for line in self.centre_lines:
            self.plot.addItem(line)
        self.track = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(86, 200, 216, 170))
        self.plot.addItem(self.track)

        self.side = self.addPlot(row=0, col=1)
        self.side.setMaximumWidth(110)
        self.side.setYLink(self.plot)
        self.side.hideAxis("left")
        self.side.setLabel("bottom", "centre band")
        self.side.setMenuEnabled(False)
        self.side.hideButtons()
        self.side_curve = self.side.plot(pen=pg.mkPen(theme.ACCENT, width=1.2))

        self.below = self.addPlot(row=1, col=0)
        self.below.setMaximumHeight(110)
        self.below.setXLink(self.plot)
        self.below.setLabel("bottom", "frequency  (centre = where the scheduled satellite should be)")
        self.below.getAxis("bottom").setStyle(showValues=False)
        self.below.setMenuEnabled(False)
        self.below.hideButtons()
        self.below_curve = self.below.plot(pen=pg.mkPen(theme.TEXT, width=1.1))

        for plot in (self.plot, self.side, self.below):
            for name in ("left", "bottom"):
                axis = plot.getAxis(name)
                axis.setPen(axis_pen)
                axis.setTextPen(text_pen)
                axis.enableAutoSIPrefix(False)  # no "(x0.001)" scale labels
        self.show_track = True

    def show(self, power: np.ndarray, duration_s: float | None) -> None:  # noqa: A003 - Qt naming
        n_time, n_freq = power.shape
        height = float(duration_s) if duration_s else float(n_time)
        self.image.setImage(power, levels=(-2.0, 6.0))
        self.image.setRect(QRectF(0, 0, n_freq, height))
        seconds = (np.arange(n_time) + 0.5) * height / n_time

        edge = n_freq // 8
        inner = n_freq - 2 * edge
        centre = edge + inner / 2
        half = max(inner // 20, 2)
        for line, x in zip(self.centre_lines, (centre - half, centre + half)):
            line.setValue(x)
        band = power[:, int(centre - half):int(centre + half)].mean(axis=1)
        self.side_curve.setData(band, seconds)
        self.below_curve.setData(np.arange(n_freq) + 0.5, power.mean(axis=0))

        rows, cols = track_points(power)
        if self.show_track and len(rows):
            self.track.setData(cols + 0.5, seconds[rows])
        else:
            self.track.clear()
        self.plot.setRange(xRange=(0, n_freq), yRange=(0, height), padding=0)


# ----------------------------------------------------------------------
class Meter(QWidget):
    """P(signal) on one line, with the 'decide alone' zones at each end."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(44)
        self._p, self._low, self._high, self._verdict = 0.5, 0.15, 0.85, "unsure"

    def set_value(self, probability: float, low: float, high: float, verdict: str) -> None:
        self._p, self._low, self._high, self._verdict = probability, low, high, verdict
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width() - 2
        y = 16
        painter.fillRect(QRectF(1, y, width, 6), QColor(theme.RULE))
        painter.fillRect(QRectF(1, y, width * self._low, 6), QColor(theme.NO_SIGNAL).darker(260))
        painter.fillRect(QRectF(1 + width * self._high, y, width * (1 - self._high), 6), QColor(theme.SIGNAL).darker(260))
        x = 1 + width * self._p
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(theme.verdict_colour(self._verdict)))
        painter.drawRect(QRectF(x - 2, y - 5, 4, 16))
        painter.setPen(QColor(theme.DIM))
        font = QFont(theme.MONO)
        font.setPixelSize(9)
        painter.setFont(font)
        # zone names below the bar, threshold values above it: they never collide
        painter.drawText(QRectF(1, y + 12, 80, 12), Qt.AlignLeft, "NO SIGNAL")
        painter.drawText(QRectF(width - 80, y + 12, 80, 12), Qt.AlignRight, "SIGNAL")
        painter.drawText(QRectF(width * self._low - 20, 0, 40, 12), Qt.AlignHCenter, f"{self._low:.2f}")
        painter.drawText(QRectF(width * self._high - 20, 0, 40, 12), Qt.AlignHCenter, f"{self._high:.2f}")


class Reasons(QWidget):
    """The model's reasons, one line each, marked by which way they point."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

    def set_reasons(self, reasons: list[tuple[str, str, bool]]) -> None:
        while self._layout.count():
            widget = self._layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        for title, detail, toward_signal in reasons:
            colour = theme.SIGNAL if toward_signal else theme.NO_SIGNAL
            label = QLabel(f"<span style='color:{colour}; font-family:\"{theme.MONO}\"'>{'+' if toward_signal else '−'}</span>"
                           f"&nbsp; {title}<br><span style='color:{theme.DIM}; font-family:\"{theme.MONO}\"; "
                           f"font-size:10px'>&nbsp;&nbsp;&nbsp;{detail}</span>")
            label.setTextFormat(Qt.RichText)
            label.setWordWrap(True)
            self._layout.addWidget(label)


def verdict_text(verdict: Verdict) -> str:
    return {Verdict.SIGNAL: "CONFIDENT · SIGNAL", Verdict.NO_SIGNAL: "CONFIDENT · NO SIGNAL",
            Verdict.UNSURE: "UNSURE · CHECK BY EYE"}[verdict]
