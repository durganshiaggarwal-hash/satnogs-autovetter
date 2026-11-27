"""The report card for the Live AI: where it is right, where it fails, and why.

The Live counterpart of the Lab's "How good is the AI?" tab.  Everything is
drawn from one comparison run's per-recording results -- each recording's
score from a model that never saw its ground station, the volunteer's
answer, and the pass details -- so every chart is an honest held-out view:

* **Confusion**: what volunteers said against what the AI did (heard,
  not heard, or handed to a person), with the app's own rules;
* **Can you trust its %?**: a reliability diagram -- when it says 80%,
  is it right 80% of the time?
* **Heatmap**: accuracy for each transmitter type at each pass elevation;
* **By elevation**: accuracy and the share decided alone, low pass to overhead;
* **Mistakes**: its most confident wrong answers; double-click to replay one;
* **Stations**: every ground station on a world map, coloured by accuracy.

The data side (:class:`RecordingResults`) has no Qt in it and is tested alone.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..session import Policy
from . import theme

#: elevation bands for the heatmap (degrees)
BANDS = ((0, 20), (20, 40), (40, 60), (60, 91))
#: a cell needs this many recordings before its accuracy is shown
MIN_CELL = 15
POLICY = Policy()


# ----------------------------------------------------------------------
class RecordingResults:
    """One model's held-out answers on every recording of one run, ready to slice."""

    def __init__(self, per_recording: dict, model: str, policy: Policy = POLICY) -> None:
        scores = per_recording.get("scores", {}).get(model)
        if scores is None:
            raise KeyError(model)
        self.model = model
        self.ids = np.asarray(per_recording["id"], dtype=np.int64)
        self.label = np.asarray(per_recording["label"], dtype=np.int64)
        self.p = np.asarray(scores, dtype=float)
        self.station = np.asarray(per_recording.get("station", [-1] * len(self.ids)), dtype=np.int64)
        self.elevation = np.array([np.nan if e is None else e for e in per_recording.get("elevation", [])],
                                  dtype=float) if per_recording.get("elevation") else np.full(len(self.ids), np.nan)
        self.mode = list(per_recording.get("mode", ["unknown"] * len(self.ids)))
        self.decoded = np.asarray(per_recording.get("decoded", [0] * len(self.ids)), dtype=np.int64)
        self.norad = np.asarray(per_recording.get("norad", [0] * len(self.ids)), dtype=np.int64)
        self.lat = np.array([np.nan if v is None else v for v in per_recording.get("lat", [None] * len(self.ids))],
                            dtype=float)
        self.lng = np.array([np.nan if v is None else v for v in per_recording.get("lng", [None] * len(self.ids))],
                            dtype=float)
        self.predicted = (self.p >= 0.5).astype(np.int64)
        self.right = self.predicted == self.label
        # The app's own verdicts: decoded packets are proof; otherwise sure only outside the policy's band.
        self.verdict = np.array([POLICY.verdict(float(p), int(d)).value for p, d in zip(self.p, self.decoded)])
        self._policy = policy

    def __len__(self) -> int:
        return len(self.ids)

    # -- confusion -------------------------------------------------------
    def confusion(self) -> np.ndarray:
        """Rows: volunteer said signal / no signal.  Columns: AI heard / not heard / unsure."""
        out = np.zeros((2, 3), dtype=np.int64)
        columns = {"signal": 0, "no signal": 1, "unsure": 2}
        for truth, verdict in zip(self.label, self.verdict):
            out[0 if truth == 1 else 1, columns[verdict]] += 1
        return out

    # -- calibration -----------------------------------------------------
    def reliability(self, bins: int = 10) -> list[tuple[float, float, int]]:
        """(mean predicted P(signal), share that really had a signal, count) for each probability bin."""
        edges = np.linspace(0, 1, bins + 1)
        index = np.clip(np.digitize(self.p, edges) - 1, 0, bins - 1)
        out = []
        for b in range(bins):
            mask = index == b
            if mask.sum():
                out.append((float(self.p[mask].mean()), float(self.label[mask].mean()), int(mask.sum())))
        return out

    # -- slices ------------------------------------------------------------
    def modes(self, top: int = 8) -> list[str]:
        """The most common transmitter types, most common first."""
        return [m for m, _ in Counter(self.mode).most_common(top)]

    def heatmap(self, top: int = 8) -> tuple[list[str], np.ndarray, np.ndarray]:
        """(modes, accuracy[mode, band], count[mode, band]); accuracy is NaN where there are too few."""
        modes = self.modes(top)
        mode = np.asarray(self.mode, dtype=object)
        accuracy = np.full((len(modes), len(BANDS)), np.nan)
        count = np.zeros((len(modes), len(BANDS)), dtype=np.int64)
        for i, m in enumerate(modes):
            for j, (low, high) in enumerate(BANDS):
                mask = (mode == m) & (self.elevation >= low) & (self.elevation < high)
                count[i, j] = int(mask.sum())
                if count[i, j] >= MIN_CELL:
                    accuracy[i, j] = float(self.right[mask].mean())
        return modes, accuracy, count

    def by_elevation(self, step: int = 10) -> list[tuple[float, float, float, int]]:
        """(band centre, accuracy, share decided alone, count) for each elevation band of ``step`` degrees."""
        out = []
        for low in range(0, 90, step):
            mask = (self.elevation >= low) & (self.elevation < low + step + (1 if low + step >= 90 else 0))
            n = int(mask.sum())
            if n >= MIN_CELL:
                out.append((low + step / 2, float(self.right[mask].mean()),
                            float(np.mean(self.verdict[mask] != "unsure")), n))
        return out

    def mistakes(self, limit: int = 200) -> list[int]:
        """Indices of the wrong answers, most confident first."""
        wrong = np.where(~self.right)[0]
        confidence = np.abs(self.p[wrong] - 0.5)
        return [int(i) for i in wrong[np.argsort(-confidence)][:limit]]

    def stations(self) -> list[tuple[float, float, float, int]]:
        """(lat, lng, accuracy, count) for every station with a known position."""
        out = []
        for s in np.unique(self.station):
            mask = self.station == s
            lat, lng = self.lat[mask][0], self.lng[mask][0]
            if np.isfinite(lat) and np.isfinite(lng):
                out.append((float(lat), float(lng), float(self.right[mask].mean()), int(mask.sum())))
        return out


# ----------------------------------------------------------------------
def accuracy_colour(value: float, alpha: int = 255) -> QColor:
    """Red (60% or worse) through yellow to green (90% or better)."""
    t = float(np.clip((value - 0.6) / 0.3, 0, 1))
    low, mid, high = QColor(theme.NO_SIGNAL), QColor(theme.UNSURE), QColor(theme.SIGNAL)
    a, b, u = (low, mid, t / 0.5) if t < 0.5 else (mid, high, (t - 0.5) / 0.5)
    colour = QColor(int(a.red() + (b.red() - a.red()) * u), int(a.green() + (b.green() - a.green()) * u),
                    int(a.blue() + (b.blue() - a.blue()) * u))
    colour.setAlpha(alpha)
    return colour


def _font(size: int, bold: bool = False) -> QFont:
    font = QFont(theme.SANS_BOLD if bold else theme.SANS)
    font.setPixelSize(size)
    return font


class _Painted(QWidget):
    """A chart drawn with QPainter, with a message when there is nothing to show."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(300)
        self.results: RecordingResults | None = None

    def show_results(self, results: RecordingResults | None) -> None:
        self.results = results
        self.update()

    def _empty(self, painter: QPainter, text: str) -> None:
        painter.setPen(QColor(theme.DIM))
        painter.setFont(_font(14))
        painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, text)


class ConfusionChart(_Painted):
    COLUMNS = ("Heard", "Not heard", "Unsure: asks a person")
    ROWS = ("Volunteer: signal", "Volunteer: no signal")

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.results is None:
            self._empty(painter, "No per-recording results in this run.")
            return
        table = self.results.confusion()
        left, top = 190, 44
        width = (self.width() - left - 16) / 3
        height = min((self.height() - top - 16) / 2, 120)
        painter.setFont(_font(13, bold=True))
        painter.setPen(QColor(theme.DIM))
        for c, name in enumerate(self.COLUMNS):
            painter.drawText(QRectF(left + c * width, 8, width, top - 12), Qt.AlignCenter, name.upper())
        for r, name in enumerate(self.ROWS):
            painter.setPen(QColor(theme.DIM))
            painter.setFont(_font(14, bold=True))
            painter.drawText(QRectF(0, top + r * height, left - 14, height), Qt.AlignRight | Qt.AlignVCenter, name)
            total = max(int(table[r].sum()), 1)
            for c in range(3):
                n = int(table[r, c])
                good = (r == 0 and c == 0) or (r == 1 and c == 1)
                colour = QColor(theme.SIGNAL if good else theme.UNSURE if c == 2 else theme.NO_SIGNAL)
                colour.setAlpha(int(40 + 180 * n / total))
                rect = QRectF(left + c * width + 3, top + r * height + 3, width - 6, height - 6)
                painter.fillRect(rect, colour)
                painter.setPen(QColor(theme.TEXT))
                painter.setFont(_font(24, bold=True))
                painter.drawText(rect.adjusted(0, 0, 0, -height * 0.25), Qt.AlignCenter, f"{n:,}")
                painter.setFont(_font(13))
                painter.drawText(rect.adjusted(0, height * 0.35, 0, 0), Qt.AlignCenter, f"{n / total:.0%} of the row")


class TrustChart(pg.PlotWidget):
    """Reliability diagram: predicted P(signal) against the share that really had a signal."""

    def show_results(self, results: RecordingResults | None) -> None:
        self.clear()
        if results is None:
            return
        self.plot([0, 1], [0, 1], pen=pg.mkPen(theme.DIM, width=1, style=Qt.DashLine))
        points = results.reliability()
        xs, ys, ns = zip(*points) if points else ((), (), ())
        self.plot(list(xs), list(ys), pen=pg.mkPen(theme.ACCENT, width=2.5))
        biggest = max(ns) if ns else 1
        self.addItem(pg.ScatterPlotItem(x=list(xs), y=list(ys), size=[8 + 22 * (n / biggest) ** 0.5 for n in ns],
                                        brush=QColor(theme.ACCENT_FILL), pen=pg.mkPen(theme.TEXT, width=1)))
        self.setXRange(0, 1, padding=0.02)
        self.setYRange(0, 1, padding=0.02)
        self.setLabel("bottom", "how sure the AI was that there is a signal")
        self.setLabel("left", "how often there really was one")


class HeatmapChart(_Painted):
    """Accuracy for each transmitter type (rows) at each pass elevation (columns)."""

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.results is None:
            self._empty(painter, "No per-recording results in this run.")
            return
        modes, accuracy, count = self.results.heatmap()
        left, top = 150, 34
        width = (self.width() - left - 12) / len(BANDS)
        height = max(min((self.height() - top - 8) / max(len(modes), 1), 46), 26)
        painter.setFont(_font(13, bold=True))
        painter.setPen(QColor(theme.DIM))
        for j, (low, high) in enumerate(BANDS):
            painter.drawText(QRectF(left + j * width, 4, width, top - 8), Qt.AlignCenter,
                             f"{low}-{min(high, 90)}° HIGH")
        for i, mode in enumerate(modes):
            painter.setFont(_font(13))
            painter.setPen(QColor(theme.TEXT))
            painter.drawText(QRectF(0, top + i * height, left - 12, height), Qt.AlignRight | Qt.AlignVCenter,
                             mode[:20])
            for j in range(len(BANDS)):
                rect = QRectF(left + j * width + 2, top + i * height + 2, width - 4, height - 4)
                if np.isnan(accuracy[i, j]):
                    painter.fillRect(rect, QColor(theme.RAISED))
                    painter.setPen(QColor(theme.FAINT))
                    painter.drawText(rect, Qt.AlignCenter, f"n={count[i, j]}" if count[i, j] else "–")
                    continue
                painter.fillRect(rect, accuracy_colour(accuracy[i, j], 210))
                painter.setPen(QColor("#1f1f1f"))
                painter.setFont(_font(13, bold=True))
                painter.drawText(rect, Qt.AlignCenter, f"{accuracy[i, j]:.0%}   n={count[i, j]}")


class ElevationChart(pg.PlotWidget):
    """Accuracy and the share decided alone, from low passes to overhead ones."""

    def show_results(self, results: RecordingResults | None) -> None:
        self.clear()
        if results is None:
            return
        rows = results.by_elevation()
        if not rows:
            return
        x, accuracy, alone, n = (list(v) for v in zip(*rows))
        self.plot(x, accuracy, pen=pg.mkPen(theme.SIGNAL, width=3), symbol="o", symbolSize=9,
                  symbolBrush=QColor(theme.SIGNAL), name="right")
        self.plot(x, alone, pen=pg.mkPen(theme.ACCENT, width=2, style=Qt.DashLine), symbol="s", symbolSize=7,
                  symbolBrush=QColor(theme.ACCENT), name="decided alone")
        for xi, yi, ni in zip(x, accuracy, n):
            label = pg.TextItem(f"n={ni}", color=theme.DIM, anchor=(0.5, 1.4))
            label.setPos(xi, yi)
            self.addItem(label)
        self.setXRange(0, 90, padding=0.02)
        self.setYRange(0.3, 1.0, padding=0)
        self.setLabel("bottom", "highest point of the pass above the horizon (degrees)")
        self.setLabel("left", "share")


class StationMap(_Painted):
    """Every ground station with a known position, coloured by how often the AI is right on its recordings."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from .mission import _land

        self._land = _land()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        scale = min(w / 360, h / 160)
        ox, oy = (w - 360 * scale) / 2, (h - 160 * scale) / 2

        def at(lat, lng):
            return QPointF(ox + (lng + 180) * scale, oy + (80 - lat) * scale)

        painter.fillRect(QRectF(ox, oy, 360 * scale, 160 * scale), QColor("#2a3038"))
        painter.setPen(QPen(QColor("#757575"), 0.6))
        painter.setBrush(QColor("#4f4f4f"))
        for ring in self._land:
            path = QPainterPath()
            path.moveTo(at(ring[0][1], ring[0][0]))
            for lng, lat in ring[1:]:
                path.lineTo(at(max(lat, -80), lng))
            painter.drawPath(path)
        if self.results is None:
            self._empty(painter, "No per-recording results in this run.")
            return
        stations = self.results.stations()
        if not stations:
            self._empty(painter, "Station positions are not saved for this run.")
            return
        biggest = max(n for *_, n in stations)
        for lat, lng, accuracy, n in sorted(stations, key=lambda s: -s[3]):
            painter.setPen(QPen(QColor(theme.BG), 1))
            painter.setBrush(accuracy_colour(accuracy, 230))
            radius = 3 + 9 * (n / biggest) ** 0.5
            painter.drawEllipse(at(lat, lng), radius, radius)
        painter.setPen(QColor(theme.DIM))
        painter.setFont(_font(12))
        painter.drawText(QRectF(ox + 8, oy + 160 * scale - 24, 360 * scale, 20), Qt.AlignLeft,
                         f"{len(stations)} stations  ·  dot size: recordings  ·  colour: share right "
                         "(red 60% or worse, green 90% or better)")


# ----------------------------------------------------------------------
class ReportCard(QWidget):
    """Six views of one model's held-out answers, switched by tabs."""

    open_recording = Signal(int)
    VIEWS = ("Confusion", "Can you trust its %?", "Heatmap", "By elevation", "Mistakes", "Stations")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.results: RecordingResults | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        tabs = QHBoxLayout()
        tabs.setSpacing(0)
        self.buttons: dict[str, QPushButton] = {}
        for name in self.VIEWS:
            button = QPushButton(name)
            button.setObjectName("tab")
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, n=name: self.set_view(n))
            self.buttons[name] = button
            tabs.addWidget(button)
        tabs.addStretch(1)
        layout.addLayout(tabs)

        self.stack = QStackedWidget()
        self.confusion = ConfusionChart()
        self.trust = TrustChart(background=theme.PANEL)
        self.heatmap = HeatmapChart()
        self.elevation = ElevationChart(background=theme.PANEL)
        for plot in (self.trust, self.elevation):
            plot.setMenuEnabled(False)
            plot.hideButtons()
            plot.setMouseEnabled(False, False)
            plot.setMinimumHeight(300)
            for side in ("left", "bottom"):
                axis = plot.getAxis(side)
                axis.setPen(pg.mkPen(theme.FAINT))
                axis.setTextPen(pg.mkPen(theme.DIM))
                axis.setStyle(tickFont=_font(13))
                axis.enableAutoSIPrefix(False)
        self.mistakes = QTableWidget(0, 7)
        self.mistakes.setHorizontalHeaderLabels(["Recording", "Transmitter", "Elevation", "Volunteer said",
                                                 "AI said", "How sure", "Station"])
        self.mistakes.verticalHeader().hide()
        self.mistakes.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.mistakes.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.mistakes.setSelectionMode(QAbstractItemView.SingleSelection)
        self.mistakes.setShowGrid(False)
        self.mistakes.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mistakes.setStyleSheet("QHeaderView::section { letter-spacing: 0px; padding: 6px 2px; font-size: 13px; }")
        self.mistakes.cellDoubleClicked.connect(self._open)
        self.stations = StationMap()
        for widget in (self.confusion, self.trust, self.heatmap, self.elevation, self.mistakes, self.stations):
            self.stack.addWidget(widget)
        layout.addWidget(self.stack, 1)
        self.note = QLabel("")
        self.note.setObjectName("body")
        self.note.setWordWrap(True)
        self.note.setMinimumHeight(44)
        layout.addWidget(self.note)
        self._view = self.VIEWS[0]
        self.set_view(self._view)

    # ------------------------------------------------------------------
    def show_results(self, results: RecordingResults | None, title: str = "") -> None:
        self.results, self._title = results, title
        for chart in (self.confusion, self.trust, self.heatmap, self.elevation, self.stations):
            chart.show_results(results)
        self._fill_mistakes()
        self._explain()

    def set_view(self, name: str) -> None:
        self._view = name
        self.stack.setCurrentIndex(self.VIEWS.index(name))
        for key, button in self.buttons.items():
            button.setChecked(key == name)
        self._explain()

    def _fill_mistakes(self) -> None:
        r = self.results
        rows = r.mistakes() if r is not None else []
        self.mistakes.setRowCount(len(rows))
        for row, i in enumerate(rows):
            said = "signal" if r.label[i] else "no signal"
            ai = "signal" if r.predicted[i] else "no signal"
            sure = abs(r.p[i] - 0.5) * 2
            elevation = "–" if np.isnan(r.elevation[i]) else f"{r.elevation[i]:.0f}°"
            cells = [f"#{r.ids[i]}", r.mode[i], elevation, said, ai, f"{max(r.p[i], 1 - r.p[i]):.0%}",
                     str(r.station[i])]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                item.setData(Qt.UserRole, int(r.ids[i]))
                if column == 4:
                    item.setForeground(QColor(theme.NO_SIGNAL if sure > 0.5 else theme.UNSURE))
                self.mistakes.setItem(row, column, item)

    def _open(self, row: int, _column: int) -> None:
        item = self.mistakes.item(row, 0)
        if item is not None:
            self.open_recording.emit(int(item.data(Qt.UserRole)))

    def _explain(self) -> None:
        r = self.results
        if r is None:
            self.note.setText("This run has no per-recording results. Run Train and compare (Full run) to make "
                              "them; earlier runs only kept the totals.")
            return
        table = r.confusion()
        decided = int(table[:, :2].sum())
        wrong = int(table[0, 1] + table[1, 0])
        texts = {
            "Confusion": f"With the app's rules ({POLICY.low:.0%} / {POLICY.high:.0%}, decoded packets count as "
                         f"heard) it decides {decided:,} of {len(r):,} recordings alone, {wrong / max(decided, 1):.1%} "
                         "of them wrong; the rest go to a person.",
            "Can you trust its %?": "On the dashed line its confidence is honest: when it says 80%, 8 in 10 really "
                                    "have a signal. Above the line it is too cautious, below it too bold. Dot size: "
                                    "how many recordings.",
            "Heatmap": f"Share right for each transmitter type at each pass height. Grey: fewer than {MIN_CELL} "
                       "recordings, too few to say. Low passes and some transmitter types are clearly harder.",
            "By elevation": "Solid: share right. Dashed: share it decides without a person. Low passes are "
                            "short and faint, so both drop.",
            "Mistakes": f"{int((~r.right).sum()):,} wrong answers at 50%, most confident first. Some are the "
                        "volunteer's mistake, not the AI's. Double-click one to replay it.",
            "Stations": "Each dot is a ground station the AI was tested on without ever training on it.",
        }
        self.note.setText(texts.get(self._view, ""))


__all__ = ["RecordingResults", "ReportCard", "accuracy_colour"]
