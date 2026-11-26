"""The radio link, live: distance, delay, Doppler, speed, and the pass timeline.

Every number is computed from the satellite's orbit at the replay's
current moment -- nothing is simulated:

* **signal delay** = distance / speed of light;
* **Doppler shift** = -f x (range rate) / c, i.e. how far the satellite's
  signal is shifted *before* the station corrects for it;
* **changing at** = how fast that shift moves (it is fastest at the top
  of the pass, which is why the station has to keep retuning).
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

from ..orbit import GroundTrack
from . import theme


def _caps(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("caps")
    return label


class LinkPanel(QFrame):
    """Six live readouts of the station-satellite link."""

    FIELDS = (("distance", "DISTANCE"), ("delay", "SIGNAL DELAY"), ("doppler", "DOPPLER SHIFT"),
              ("rate", "SHIFT CHANGING AT"), ("pointing", "ANTENNA POINTS AT"), ("speed", "SATELLITE SPEED"))

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        grid = QGridLayout(self)
        grid.setContentsMargins(14, 10, 14, 10)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(2)
        self._values: dict[str, QLabel] = {}
        for n, (key, title) in enumerate(self.FIELDS):
            row, column = (n // 3) * 2, n % 3
            grid.addWidget(_caps(title), row, column)
            value = QLabel("–")
            value.setObjectName("counter")
            grid.addWidget(value, row + 1, column)
            self._values[key] = value
        self.note = QLabel("")
        self.note.setObjectName("body")
        self.note.setWordWrap(True)
        grid.addWidget(self.note, 4, 0, 1, 3)

    def show_moment(self, track: GroundTrack | None, seconds: float, frequency_hz: float | None) -> None:
        if track is None or track.range_km is None:
            for value in self._values.values():
                value.setText("–")
            self.note.setText("No orbit for this recording.")
            return
        i = int(np.clip(np.searchsorted(track.seconds, seconds), 0, len(track.seconds) - 1))
        up = track.elevation[i] > 0
        self._values["distance"].setText(f"{track.range_km[i]:,.0f} km")
        self._values["delay"].setText(f"{track.delay_ms(i):.2f} ms")
        if frequency_hz:
            doppler = track.doppler_hz(i, frequency_hz)
            j = min(i + 1, len(track.seconds) - 1)
            k = max(j - 1, 0) if j == i else i
            dt = max(track.seconds[j] - track.seconds[k], 1e-6)
            rate = (track.doppler_hz(j, frequency_hz) - track.doppler_hz(k, frequency_hz)) / dt
            self._values["doppler"].setText(f"{doppler / 1000:+.2f} kHz")
            self._values["rate"].setText(f"{rate:+.0f} Hz/s")
        else:
            self._values["doppler"].setText("–")
            self._values["rate"].setText("–")
        if track.azimuth is not None:
            self._values["pointing"].setText(f"{track.elevation[i]:.0f}° up, {_compass(track.azimuth[i])}"
                                             if up else "below horizon")
        if track.speed_kms is not None:
            self._values["speed"].setText(f"{track.speed_kms[i]:.2f} km/s")
        colour = theme.TEXT if up else theme.FAINT
        for value in self._values.values():
            value.setStyleSheet(f"color: {colour}")
        if not up:
            self.note.setText("The satellite is below the horizon: no radio link right now.")
        elif track.range_rate_kms is not None:
            towards = track.range_rate_kms[i] < 0
            self.note.setText(("Coming closer, so its signal is heard higher in pitch."
                               if towards else "Moving away, so its signal is heard lower in pitch.")
                              + "  Delay = distance ÷ speed of light.")


def _compass(azimuth: float) -> str:
    names = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")
    return names[int(((azimuth % 360) + 22.5) // 45) % 8]


# ----------------------------------------------------------------------
class PassTimeline(QWidget):
    """The pass at a glance, and the replay's scrubber.

    The curve is the satellite's height above the station's horizon over
    the recording; rise and set are marked with their times; the green
    strip shows where the recording has a signal; click or drag to move.
    """

    scrubbed = Signal(float)   # 0..1 through the recording

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(70)
        self.setCursor(Qt.PointingHandCursor)
        self._seconds = np.zeros(0)
        self._elevation = np.zeros(0)
        self._activity = np.zeros(0)
        self._t0, self._t1, self._now = 0.0, 1.0, 0.0
        self._start = None

    def set_pass(self, start, t0: float, t1: float, track: GroundTrack | None, activity: np.ndarray | None) -> None:
        self._start, self._t0, self._t1 = start, t0, t1
        if track is not None:
            keep = (track.seconds >= t0 - 1) & (track.seconds <= t1 + 1)
            self._seconds, self._elevation = track.seconds[keep], track.elevation[keep]
        else:
            self._seconds, self._elevation = np.zeros(0), np.zeros(0)
        self._activity = np.zeros(0) if activity is None else np.asarray(activity)
        self.update()

    def set_time(self, seconds: float) -> None:
        self._now = seconds
        self.update()

    def _x(self, seconds: float) -> float:
        return 10 + (seconds - self._t0) / max(self._t1 - self._t0, 1e-6) * (self.width() - 20)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._scrub(event.position().x())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if event.buttons():
            self._scrub(event.position().x())

    def _scrub(self, x: float) -> None:
        self.scrubbed.emit(float(np.clip((x - 10) / max(self.width() - 20, 1), 0, 1)))

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        top, bottom = 8.0, self.height() - 22.0
        horizon = bottom - 4
        font = QFont(theme.MONO)
        font.setPixelSize(11)
        painter.setFont(font)
        # Where the recording has a signal.
        if len(self._activity):
            n = len(self._activity)
            for i, a in enumerate(self._activity):
                if a > 0.35:
                    x0 = self._x(self._t0 + (self._t1 - self._t0) * i / n)
                    x1 = self._x(self._t0 + (self._t1 - self._t0) * (i + 1) / n)
                    painter.fillRect(QRectF(x0, bottom + 2, max(x1 - x0, 1), 4), QColor(theme.SIGNAL))
        painter.setPen(QPen(QColor(theme.RULE), 1))
        painter.drawLine(QPointF(10, horizon), QPointF(self.width() - 10, horizon))
        if len(self._seconds) > 1:
            peak = max(float(self._elevation.max()), 10.0)
            y = lambda e: horizon - max(e, 0) / peak * (horizon - top)  # noqa: E731
            path = QPainterPath(QPointF(self._x(self._seconds[0]), y(self._elevation[0])))
            for s, e in zip(self._seconds[1:], self._elevation[1:]):
                path.lineTo(QPointF(self._x(s), y(e)))
            fill = QPainterPath(path)
            fill.lineTo(QPointF(self._x(self._seconds[-1]), horizon))
            fill.lineTo(QPointF(self._x(self._seconds[0]), horizon))
            gradient = QLinearGradient(0, top, 0, horizon)
            gradient.setColorAt(0, QColor(*theme.ACCENT_RGB, 90))
            gradient.setColorAt(1, QColor(*theme.ACCENT_RGB, 10))
            painter.fillPath(fill, gradient)
            painter.setPen(QPen(QColor(theme.ACCENT), 1.5))
            painter.drawPath(path)
            # Rise and set: where the curve crosses the horizon.
            up = self._elevation > 0
            for i in np.flatnonzero(np.diff(up.astype(int)) != 0):
                s = float(self._seconds[i + 1])
                rising = bool(up[i + 1])
                painter.setPen(QPen(QColor(theme.SIGNAL if rising else theme.NO_SIGNAL), 1, Qt.DashLine))
                painter.drawLine(QPointF(self._x(s), top), QPointF(self._x(s), horizon))
                when = (self._start + timedelta(seconds=s)).strftime("%H:%M:%S") if self._start else f"{s:.0f}s"
                painter.setPen(QColor(theme.DIM))
                painter.drawText(QRectF(self._x(s) + 3, top, 120, 12), Qt.AlignLeft,
                                 ("rises " if rising else "sets ") + when)
            i_top = int(np.argmax(self._elevation))
            painter.setPen(QColor(theme.DIM))
            label_y = max(y(self._elevation[i_top]) - 13, 0.0)       # never above the widget
            painter.drawText(QRectF(self._x(self._seconds[i_top]) + 6, label_y, 90, 12),
                             Qt.AlignLeft, f"highest {self._elevation[i_top]:.0f}°")
        # Now.
        x = self._x(self._now)
        painter.setPen(QPen(QColor(theme.TEXT), 1.5))
        painter.drawLine(QPointF(x, top - 2), QPointF(x, bottom + 6))
        painter.setPen(QColor(theme.FAINT))
        painter.drawText(QRectF(10, bottom + 7, 200, 12), Qt.AlignLeft, "start of recording")
        painter.drawText(QRectF(self.width() - 210, bottom + 7, 200, 12), Qt.AlignRight, "end")
