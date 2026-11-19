"""The orbit view: which source made the track, the sky above the station, and the timing slider.

Everything here answers to the physics in :mod:`satrecovery.vetting.orbit`:
the curves on the waterfall, the dots on the sky chart and the slider are
all computed from orbits, frequencies and the station's position -- none
of it is read off the picture.

DESIGN NOTE: CONCURRENCY BEHIND AN INTERFACE
--------------------------------------------
Working out a pass takes a second (and the first time, downloading the
catalogue takes several), so :class:`OrbitWorker` runs it on a thread
pool and *signals* the answer.  The panel only ever sees finished
:class:`~satrecovery.vetting.orbit.TrackExplanation` objects, and drops
answers for an observation you have already moved past.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from ..orbit import Curve, OrbitError, TrackExplainer, TrackExplanation
from . import theme
from .widgets import caps

PALETTE = ["#e6a3ff", "#ffd479", "#7fd3ff", "#ff7f9e", "#b8f28a", "#9fb0ff", "#ffa64d"]  # far apart in hue


def colour_for(curve: Curve, index: int) -> str:
    if curve.scheduled:
        return theme.ACCENT
    if curve.norad == 0:
        return "#ffffff"
    return PALETTE[index % len(PALETTE)]


# ----------------------------------------------------------------------
class _Signals(QObject):
    done = Signal(int, object)
    failed = Signal(int, str)


class OrbitWorker(QRunnable):
    def __init__(self, explainer: TrackExplainer, obs, power: np.ndarray) -> None:
        super().__init__()
        self.signals = _Signals()
        self._explainer, self._obs, self._power = explainer, obs, power

    def run(self) -> None:
        try:
            self.signals.done.emit(self._obs.id, self._explainer.explain(self._obs, self._power))
        except (OrbitError, OSError, ValueError) as error:
            self.signals.failed.emit(self._obs.id, str(error))
        except Exception as error:  # noqa: BLE001 - a background failure must not take the window down
            self.signals.failed.emit(self._obs.id, f"{type(error).__name__}: {error}")


# ----------------------------------------------------------------------
class SkyPlot(QWidget):
    """The sky over the station: centre = straight up, edge = horizon, north up."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(190, 190)
        self._tracks: list[tuple[np.ndarray, np.ndarray, str, bool]] = []
        self._seconds: np.ndarray | None = None
        self._now: float | None = None

    def set_tracks(self, seconds: np.ndarray | None, tracks: list[tuple[np.ndarray, np.ndarray, str, bool]]) -> None:
        self._seconds, self._tracks, self._now = seconds, tracks, None
        self.update()

    def set_time(self, seconds: float) -> None:
        self._now = seconds
        self.update()

    def _point(self, az: float, el: float, centre: QPointF, radius: float) -> QPointF:
        r = radius * (1 - max(el, 0) / 90)
        a = math.radians(az)
        return QPointF(centre.x() + r * math.sin(a), centre.y() - r * math.cos(a))

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        side = min(self.width(), self.height()) - 24
        centre = QPointF(self.width() / 2, self.height() / 2)
        radius = side / 2
        painter.setPen(QPen(QColor(theme.RULE), 1))
        for el in (0, 30, 60):
            r = radius * (1 - el / 90)
            painter.drawEllipse(centre, r, r)
        font = QFont(theme.MONO)
        font.setPixelSize(9)
        painter.setFont(font)
        painter.setPen(QColor(theme.DIM))
        for label, az in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
            p = self._point(az, -8, centre, radius + 8)
            painter.drawText(QRectF(p.x() - 6, p.y() - 6, 12, 12), Qt.AlignCenter, label)
        for az, el, colour, strong in self._tracks:
            up = el > 0
            if up.sum() < 2:
                continue
            painter.setPen(QPen(QColor(colour), 2.0 if strong else 1.0))
            points = [self._point(a, e, centre, radius) for a, e, u in zip(az, el, up) if u]
            for a, b in zip(points, points[1:]):
                painter.drawLine(a, b)
            if self._now is not None and self._seconds is not None:
                a_now = float(np.interp(self._now, self._seconds, np.unwrap(np.radians(az))) % (2 * math.pi))
                e_now = float(np.interp(self._now, self._seconds, el))
                if e_now > 0:
                    painter.setBrush(QColor(colour))
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(self._point(math.degrees(a_now), e_now, centre, radius), 4, 4)
                    painter.setBrush(Qt.NoBrush)


# ----------------------------------------------------------------------
class OrbitPanel(QWidget):
    """Under the waterfall: the verdict on the track, the candidates, the sky, the slider."""

    def __init__(self, explainer: TrackExplainer | None, scope, parent=None) -> None:
        super().__init__(parent)
        self._explainer, self._scope = explainer, scope
        self._pool = QThreadPool.globalInstance()
        self._workers: list[OrbitWorker] = []
        self._current: int | None = None
        self._explanation: TrackExplanation | None = None
        self._colours: dict[str, str] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(16)
        self.sky = SkyPlot()
        sky_box = QVBoxLayout()
        sky_box.addWidget(caps("sky over the station"))
        sky_box.addWidget(self.sky, 1)
        layout.addLayout(sky_box)

        text = QVBoxLayout()
        text.setSpacing(4)
        text.addWidget(caps("what made the track  ·  orbit physics"))
        self.verdict = QLabel("–")
        self.verdict.setObjectName("counter")
        self.verdict.setWordWrap(True)
        text.addWidget(self.verdict)
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet(f"color: {theme.DIM}")
        text.addWidget(self.detail)
        self.candidates = QLabel()
        self.candidates.setTextFormat(Qt.RichText)
        self.candidates.setObjectName("mono")
        self.candidates.setWordWrap(True)
        text.addWidget(self.candidates)
        text.addStretch(1)

        slider_row = QHBoxLayout()
        slider_row.addWidget(caps("what if its orbit were off by"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(-60, 60)       # x 5 s
        self.slider.setValue(0)
        self.slider.setFocusPolicy(Qt.NoFocus)
        self.slider.valueChanged.connect(self._slide)
        slider_row.addWidget(self.slider, 1)
        self.slider_value = QLabel("0 s")
        self.slider_value.setObjectName("mono")
        self.slider_value.setMinimumWidth(50)
        slider_row.addWidget(self.slider_value)
        text.addLayout(slider_row)
        layout.addLayout(text, 1)
        self._what_if: Curve | None = None

        scope.on_hover = self.sky.set_time

    # ------------------------------------------------------------------
    def show_observation(self, obs, power: np.ndarray) -> None:
        self._current = obs.id
        self._obs, self._power = obs, power
        self._explanation = None
        self._what_if = None
        self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.slider_value.setText("0 s")
        self.sky.set_tracks(None, [])
        self.candidates.setText("")
        if self._explainer is None:
            self.verdict.setText("Orbit view off")
            self.detail.setText("")
            return
        self.verdict.setText("Working out the pass…")
        self.detail.setText("Orbits from SatNOGS DB, the station's position, the waterfall's kHz scale.")
        worker = OrbitWorker(self._explainer, obs, power)
        worker.signals.done.connect(self._done)
        worker.signals.failed.connect(self._failed)
        self._workers.append(worker)
        self._pool.start(worker)

    def wait(self, ms: int = 30000) -> bool:
        """For tests and screenshots: block until background work is finished."""
        return self._pool.waitForDone(ms)

    def _failed(self, obs_id: int, message: str) -> None:
        if obs_id != self._current:
            return
        self.verdict.setText("No orbit view for this one")
        self.detail.setText(message)

    def _done(self, obs_id: int, explanation: TrackExplanation) -> None:
        if obs_id != self._current:
            return  # you have moved on
        self._explanation = explanation
        self._scope.show(self._power, self._obs.duration_s, explanation.prediction.axes)
        colour = {"scheduled": theme.SIGNAL, "other": theme.NO_SIGNAL, "ground": theme.NO_SIGNAL,
                  "timing": theme.UNSURE, "unknown": theme.UNSURE, "no track": theme.DIM}[explanation.source]
        self.verdict.setText(f"<span style='color:{colour}'>{explanation.verdict}</span>")
        self.detail.setText(explanation.detail)
        self._colours = {c.name: colour_for(c, i) for i, c in enumerate(explanation.prediction.curves)}
        shares = {m.curve.name: m for m in explanation.matches}
        rows = []
        for curve in explanation.prediction.curves:
            if curve.name.endswith("(today's orbit)"):
                continue
            match = shares.get(curve.name)
            share = f"{match.share:>4.0%}" if match else "  – "
            where = "" if curve.norad == 0 else f"{curve.frequency_hz / 1e6:.4f} MHz"
            rows.append(f"<span style='color:{self._colours[curve.name]}'>■</span> {share}  {curve.name[:40]}"
                        f" <span style='color:{theme.DIM}'>{where}</span>")
        self.candidates.setText("<br>".join(rows[:6]) + (f"<br><span style='color:{theme.DIM}'>+{len(rows) - 6} more"
                                                         f"</span>" if len(rows) > 6 else ""))
        self.sky.set_tracks(explanation.prediction.curves[0].seconds,
                            [(c.azimuth, c.elevation, self._colours[c.name], c.scheduled)
                             for c in explanation.prediction.curves if c.norad != 0
                             and not c.name.endswith("(today's orbit)")])
        self._draw()

    def _slide(self, value: int) -> None:
        shift = value * 5
        self.slider_value.setText(f"{shift:+d} s")
        if self._explanation is None:
            return
        self._what_if = None if shift == 0 else self._explanation.prediction.time_shifted(self._explanation.site,
                                                                                             float(shift))
        self._draw()

    def _draw(self) -> None:
        explanation = self._explanation
        if explanation is None:
            return
        best = explanation.matches[0].curve.name if explanation.matches else None
        offsets = {m.curve.name: m.shift_khz for m in explanation.matches}
        curves = []
        for curve in explanation.prediction.curves:
            if curve.scheduled and not curve.name.endswith("(today's orbit)"):
                continue  # the centre line already marks it
            up = curve.visible
            strong = curve.name == best and explanation.source in ("other", "ground", "scheduled")
            shift = offsets.get(curve.name, 0.0) if strong else 0.0
            label = "ground" if curve.norad == 0 else curve.name.split(" (")[0][:18]
            curves.append((curve.khz[up] + shift, curve.seconds[up], self._colours.get(curve.name, "#fff"), label, strong))
        if self._what_if is not None:
            up = self._what_if.visible
            curves.append((self._what_if.khz[up], self._what_if.seconds[up], theme.ACCENT,
                           f"if {self.slider_value.text()} off", True))
        elif explanation.timing is not None and explanation.source == "timing":
            t = explanation.timing
            up = t.curve.visible
            curves.append((t.curve.khz[up] + t.shift_khz, t.curve.seconds[up], theme.UNSURE, "timing fit", True))
        self._scope.draw_curves(curves)
