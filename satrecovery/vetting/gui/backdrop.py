"""Motion for the front screens: a starry backdrop, an orbiting-satellites hero,
numbers that count up, screens that fade in.

All drawn with QPainter, all decoration: none of it reads data except the
hero's satellite count, which is only a number to draw.  Timers run only
while the widget is on screen, so a hidden screen costs nothing.
"""

from __future__ import annotations

import math
import random
import re

from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer, QVariantAnimation, QEasingCurve
from PySide6.QtGui import QColor, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QGraphicsOpacityEffect, QLabel, QWidget

from . import theme

CREAM = QColor(theme.TEXT)
SLATE = QColor(theme.ACCENT)
SLATE_FILL = QColor(theme.ACCENT_FILL)


class _Clock:
    """Seconds since start, ticking only while the owner is visible."""

    def __init__(self, owner: QWidget, interval_ms: int = 33) -> None:
        self.owner = owner
        self.timer = QTimer(owner)
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(owner.update)
        self.elapsed = QElapsedTimer()
        self.elapsed.start()

    def start(self) -> None:
        self.timer.start()

    def stop(self) -> None:
        self.timer.stop()

    @property
    def t(self) -> float:
        return self.elapsed.elapsed() / 1000.0


class StarryScreen(QWidget):
    """A screen whose background is a slow, twinkling starfield with satellites drifting across."""

    N_STARS = 260
    N_SATELLITES = 6

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._clock = _Clock(self, 40)
        rng = random.Random(7)
        # (x, y) as fractions of the screen, size, phase and speed of the twinkle
        self._stars = [(rng.random(), rng.random(), rng.choice((0.8, 0.8, 1.1, 1.5)), rng.random() * 6.3,
                        0.6 + rng.random() * 1.6) for _ in range(self.N_STARS)]
        # satellites: start point, direction (radians), speed (fraction of width per second), phase
        self._sats = [(rng.random(), rng.random(), rng.uniform(-0.5, 0.5), rng.uniform(0.012, 0.03),
                       rng.random() * 50) for _ in range(self.N_SATELLITES)]

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        self._clock.start()

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        self._clock.stop()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h, t = self.width(), self.height(), self._clock.t
        painter.fillRect(self.rect(), QColor(theme.BG))
        # a soft slate wash in one corner, like ink spreading on paper
        wash = QRadialGradient(QPointF(w * 0.85, h * 0.1), max(w, h) * 0.7)
        tint = QColor(SLATE_FILL)
        tint.setAlpha(55)
        wash.setColorAt(0, tint)
        wash.setColorAt(1, QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), wash)
        painter.setPen(Qt.NoPen)
        for x, y, size, phase, speed in self._stars:
            star = QColor(CREAM)
            star.setAlpha(int(60 + 120 * (0.5 + 0.5 * math.sin(t * speed + phase))))
            painter.setBrush(star)
            painter.drawEllipse(QPointF(x * w, y * h), size, size)
        for x0, y0, angle, speed, phase in self._sats:
            travel = ((t + phase) * speed) % 1.6 - 0.3          # off one edge, across, off the other
            dx, dy = math.cos(angle), math.sin(angle)
            x, y = (x0 * 0.2 + travel) * w, (y0 + dy * travel * 0.4) % 1.0 * h
            for k in range(14):                                  # a fading trail
                trail = QColor(SLATE)
                trail.setAlpha(int(110 * (1 - k / 14)))
                painter.setBrush(trail)
                painter.drawEllipse(QPointF(x - k * 4 * dx, y - k * 4 * dy * 0.4), 1.3, 1.3)
            painter.setBrush(CREAM)
            painter.drawEllipse(QPointF(x, y), 2.2, 2.2)


class OrbitHero(QWidget):
    """A small Earth with a swarm of satellites on inclined orbits: drag it to spin, hover to speed it up."""

    def __init__(self, n_satellites: int = 140, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 300)
        self.setCursor(Qt.OpenHandCursor)
        self.setMouseTracking(True)
        self._clock = _Clock(self, 33)
        rng = random.Random(11)
        # each satellite: orbit radius (Earth radii), inclination, node, phase, speed
        self._sats = [(rng.choice((1.12, 1.18, 1.25, 1.35, 1.55)), math.radians(rng.choice((51.6, 53, 97.5, 70, 87.9))
                        + rng.uniform(-3, 3)), rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi),
                       rng.uniform(0.18, 0.35)) for _ in range(n_satellites)]
        self._spin = 0.0        # extra rotation from dragging
        self._drag: float | None = None
        self._hover = 0.0
        self._last_t = 0.0
        self._phase = 0.0

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._clock.start()

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        self._clock.stop()

    def enterEvent(self, event) -> None:  # noqa: N802
        self._hover = 1.0

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = 0.0

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._drag = event.position().x()
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag is not None:
            x = event.position().x()
            self._spin += (x - self._drag) * 0.01
            self._drag = x

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag = None
        self.setCursor(Qt.OpenHandCursor)

    def paintEvent(self, event) -> None:  # noqa: N802
        t = self._clock.t
        dt, self._last_t = t - self._last_t, t
        self._phase += max(min(dt, 0.1), 0.0) * (1.0 + 2.5 * self._hover)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cx, cy = self.width() / 2, self.height() / 2
        r = min(self.width(), self.height()) * 0.28
        rot = self._phase * 0.15 + self._spin
        tilt = math.radians(20)

        def project(radius, inc, node, phase):
            # position on the orbit, then the view rotation and tilt
            x, y = math.cos(phase), math.sin(phase)
            x, y, z = x, y * math.cos(inc), y * math.sin(inc)
            x, y = x * math.cos(node + rot) - y * math.sin(node + rot), x * math.sin(node + rot) + y * math.cos(node + rot)
            y, z = y * math.cos(tilt) - z * math.sin(tilt), y * math.sin(tilt) + z * math.cos(tilt)
            return cx + x * radius * r, cy - z * radius * r, y          # y < 0: in front of the Earth

        points = [project(radius, inc, node, phase + self._phase * speed)
                  for radius, inc, node, phase, speed in self._sats]
        painter.setPen(Qt.NoPen)
        for x, y, depth in points:                                     # behind the Earth first
            if depth > 0:
                dot = QColor(SLATE)
                dot.setAlpha(90)
                painter.setBrush(dot)
                painter.drawEllipse(QPointF(x, y), 1.6, 1.6)
        # the Earth: an ink sphere with a slate rim and faint meridians
        body = QRadialGradient(QPointF(cx - r * 0.35, cy - r * 0.35), r * 1.5)
        body.setColorAt(0, QColor("#7d8fa3"))
        body.setColorAt(0.6, QColor(theme.ACCENT_FILL))
        body.setColorAt(1, QColor("#3a4552"))
        painter.setBrush(body)
        painter.drawEllipse(QPointF(cx, cy), r, r)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 227, 40), 1))
        for k in range(6):
            angle = rot + k * math.pi / 6
            painter.drawEllipse(QPointF(cx, cy), abs(r * math.cos(angle)), r)
        for lat in (-60, -30, 0, 30, 60):
            y = cy - r * math.sin(math.radians(lat)) * math.cos(tilt)
            half = r * math.cos(math.radians(lat))
            painter.drawEllipse(QRectF(cx - half, y - half * 0.18, 2 * half, half * 0.36))
        painter.setPen(Qt.NoPen)
        for x, y, depth in points:                                     # then the ones in front
            if depth <= 0:
                glow = QColor(CREAM)
                glow.setAlpha(230)
                painter.setBrush(glow)
                painter.drawEllipse(QPointF(x, y), 2.2, 2.2)
        painter.setPen(QColor(theme.DIM))
        painter.drawText(QRectF(0, self.height() - 22, self.width(), 18), Qt.AlignCenter,
                         "drag to spin  ·  hover to speed up")


_NUMBER = re.compile(r"^(\d[\d,]*)(.*)$")


def count_up(label: QLabel, text: str, ms: int = 900) -> None:
    """Show ``text`` on the label, counting its leading number up from zero."""
    match = _NUMBER.match(text)
    if not match:
        label.setText(text)
        return
    target = int(match.group(1).replace(",", ""))
    suffix = match.group(2)
    animation = QVariantAnimation(label)
    animation.setStartValue(0)
    animation.setEndValue(target)
    animation.setDuration(ms)
    animation.setEasingCurve(QEasingCurve.OutCubic)
    animation.valueChanged.connect(lambda v: label.setText(f"{int(v):,}{suffix}"))
    animation.finished.connect(lambda: label.setText(text))
    label.setText(f"0{suffix}")
    animation.start(QVariantAnimation.DeleteWhenStopped)


def fade_in(widget: QWidget, ms: int = 280) -> None:
    """Fade a screen in when it is shown; the effect is removed afterwards so painting stays fast."""
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    animation = QVariantAnimation(widget)
    animation.setStartValue(0.0)
    animation.setEndValue(1.0)
    animation.setDuration(ms)
    animation.setEasingCurve(QEasingCurve.OutCubic)
    animation.valueChanged.connect(effect.setOpacity)
    animation.finished.connect(lambda: widget.setGraphicsEffect(None))
    animation.start(QVariantAnimation.DeleteWhenStopped)
