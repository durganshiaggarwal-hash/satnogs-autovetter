"""The sky over the station at the replay's moment: stars, constellations, the Sun, the satellite.

Drawn as you would see it lying on your back with your head to the north:
the zenith in the middle, the horizon round the edge, and so **east on
the left, west on the right** (a sky chart is a mirror of a ground map).
The projection is stereographic from the zenith, which keeps
constellation shapes recognisable right down to the horizon.

The sky turns as the replay plays (stars rise in the east), the
background is dark at night and lighter in twilight or day, and the
satellite's path is drawn with a dot at its current place.
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

from ..orbit import GroundTrack
from ..sky import StarCatalogue, altaz, gmst_degrees, sky_state, sun_radec
from . import theme
from .mission import INK, WorldMap

SKY = {"day": ("#2a4f78", "#16304d"), "twilight": ("#1a2f4f", "#0c1628"),
       "dusk": ("#0f1c33", "#070d18"), "night": ("#070c18", "#03060c")}


class SkyMap(QWidget):
    """Same interface as the map and the globe: ``set_pass`` and ``set_time``."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 300)
        self._stars = StarCatalogue()
        self._station: tuple[float, float] | None = None
        self._station_name = ""
        self._track: GroundTrack | None = None
        self._start = None
        self._satellite = "the satellite"
        self._t = 0.0
        self._pulse = 0.0
        self._caption = ""

    def set_pass(self, station: tuple[float, float], name: str, track: GroundTrack | None,
                 start=None, satellite: str = "") -> None:
        self._station, self._station_name, self._track = station, name, track
        self._start = start if start is not None else self._start
        self._satellite = satellite or self._satellite
        self.update()

    def set_time(self, seconds: float, pulse: float, caption: str = "") -> None:
        self._t, self._pulse, self._caption = seconds, pulse, caption
        self.update()

    # -- projection: zenith in the middle, horizon at radius R, east on the left
    def _geometry(self) -> tuple[float, float, float]:
        return self.width() / 2, (self.height() - 34) / 2 + 4, min(self.width(), self.height() - 50) * 0.47

    def _xy(self, alt, az):
        cx, cy, R = self._geometry()
        r = R * np.tan(np.radians(90.0 - np.asarray(alt, float)) / 2.0)   # stereographic: tan(45°) = 1 at the horizon
        a = np.radians(np.asarray(az, float))
        return cx - r * np.sin(a), cy - r * np.cos(a)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#05080d"))
        cx, cy, R = self._geometry()
        if self._station is None or self._start is None:
            painter.setPen(QColor(theme.DIM))
            painter.drawText(self.rect(), Qt.AlignCenter, "No station position for this recording.")
            return
        lat, lon = self._station
        when = self._start + timedelta(seconds=float(self._t))
        gmst = gmst_degrees(when)
        sun_ra, sun_dec = sun_radec(when)
        sun_alt, sun_az = (float(v[0]) for v in altaz(np.array([sun_ra]), np.array([sun_dec]), lat, lon, when, gmst))
        state = sky_state(sun_alt)

        inner, outer = SKY[state]
        dome = QRadialGradient(QPointF(cx, cy), R)
        dome.setColorAt(0.0, QColor(outer))
        dome.setColorAt(1.0, QColor(inner))
        painter.setPen(Qt.NoPen)
        painter.setBrush(dome)
        painter.drawEllipse(QPointF(cx, cy), R, R)
        painter.save()
        clip = QRectF(cx - R, cy - R, 2 * R, 2 * R)
        from PySide6.QtGui import QPainterPath

        path = QPainterPath()
        path.addEllipse(clip)
        painter.setClipPath(path)
        self._rings(painter)
        self._constellations(painter, lat, lon, when, gmst, state)
        self._starfield(painter, lat, lon, when, gmst, state)
        if sun_alt > -1:
            self._sun(painter, sun_alt, sun_az)
        if self._track is not None:
            self._satellite_path(painter)
        painter.restore()
        self._compass(painter)
        WorldMap._subtitle(self, painter)
        painter.setPen(QColor(theme.FAINT))
        font = QFont(theme.SANS)
        font.setPixelSize(10)
        painter.setFont(font)
        local = f"Sun {sun_alt:+.0f}° · {state}"
        painter.drawText(QRectF(8, self.height() - 18, self.width() - 16, 14), Qt.AlignLeft,
                         f"Looking up from the station, head to the north (so east is on the left).  {local}.  "
                         f"Stars: Hipparcos via d3-celestial.")

    def _rings(self, painter: QPainter) -> None:
        cx, cy, R = self._geometry()
        painter.setPen(QPen(QColor(255, 255, 255, 26), 0.8, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        for alt in (30, 60):
            r = R * math.tan(math.radians(90 - alt) / 2)
            painter.drawEllipse(QPointF(cx, cy), r, r)
        font = QFont(theme.MONO)
        font.setPixelSize(9)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255, 60))
        for alt in (30, 60):
            r = R * math.tan(math.radians(90 - alt) / 2)
            painter.drawText(QRectF(cx + 3, cy - r - 12, 40, 12), Qt.AlignLeft, f"{alt}°")

    def _starfield(self, painter, lat, lon, when, gmst, state) -> None:
        cat = self._stars
        if not len(cat):
            return
        alt, az = altaz(cat.ra, cat.dec, lat, lon, when, gmst)
        # In daylight only the brightest show; in twilight the brighter ones.
        limit = {"day": -0.5, "twilight": 2.5, "dusk": 4.5, "night": 6.0}[state]
        keep = (alt > 0) & (cat.mag <= limit)
        x, y = self._xy(alt[keep], az[keep])
        mag = cat.mag[keep]
        painter.setPen(Qt.NoPen)
        for xi, yi, m in zip(x, y, mag):
            size = max(0.6, 3.4 - 0.55 * m)
            light = int(np.clip(255 - 28 * m, 90, 255))
            painter.setBrush(QColor(light, light, min(255, light + 12)))
            painter.drawEllipse(QPointF(xi, yi), size, size)

    def _constellations(self, painter, lat, lon, when, gmst, state) -> None:
        if state == "day":
            return
        cat = self._stars
        painter.setPen(QPen(QColor(120, 160, 210, 70 if state == "night" else 40), 0.9))
        for line in cat.lines:
            alt, az = altaz(line[:, 0], line[:, 1], lat, lon, when, gmst)
            x, y = self._xy(alt, az)
            for i in range(len(x) - 1):
                if alt[i] > -5 and alt[i + 1] > -5:
                    painter.drawLine(QPointF(x[i], y[i]), QPointF(x[i + 1], y[i + 1]))
        if not cat.names:
            return
        names = np.array([n for n, _, _ in cat.names])
        ra = np.array([r for _, r, _ in cat.names])
        dec = np.array([d for _, _, d in cat.names])
        alt, az = altaz(ra, dec, lat, lon, when, gmst)
        x, y = self._xy(alt, az)
        font = QFont(theme.SANS)
        font.setPixelSize(10)
        painter.setFont(font)
        painter.setPen(QColor(150, 180, 220, 120))
        for name, a, xi, yi in zip(names, alt, x, y):
            if a > 8:
                painter.drawText(QRectF(xi - 60, yi - 7, 120, 14), Qt.AlignCenter, str(name))

    def _sun(self, painter: QPainter, alt: float, az: float) -> None:
        x, y = self._xy(alt, az)
        glow = QRadialGradient(QPointF(float(x), float(y)), 26)
        glow.setColorAt(0, QColor(255, 220, 120, 230))
        glow.setColorAt(1, QColor(255, 200, 90, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(QPointF(float(x), float(y)), 26, 26)
        painter.setPen(QColor(INK))
        painter.drawText(QRectF(float(x) + 14, float(y) - 6, 60, 14), Qt.AlignLeft, "Sun")

    def _satellite_path(self, painter: QPainter) -> None:
        track = self._track
        if track.azimuth is None:
            return
        up = track.elevation > 0
        x, y = self._xy(np.maximum(track.elevation, 0), track.azimuth)
        now = int(np.clip(np.searchsorted(track.seconds, self._t), 0, len(track.seconds) - 1))
        for i in range(len(x) - 1):
            if up[i] and up[i + 1]:
                past = i < now
                painter.setPen(QPen(QColor(theme.ACCENT if past else theme.FAINT), 2.2 if past else 1.2,
                                    Qt.SolidLine if past else Qt.DashLine))
                painter.drawLine(QPointF(x[i], y[i]), QPointF(x[i + 1], y[i + 1]))
        if up[now]:
            sx, sy = float(x[now]), float(y[now])
            halo = QRadialGradient(QPointF(sx, sy), 14 + 8 * self._pulse)
            halo.setColorAt(0, QColor(86, 200, 216, 200))
            halo.setColorAt(1, QColor(86, 200, 216, 0))
            painter.setPen(Qt.NoPen)
            painter.setBrush(halo)
            painter.drawEllipse(QPointF(sx, sy), 14 + 8 * self._pulse, 14 + 8 * self._pulse)
            WorldMap._satellite_icon(painter, sx, sy)
            painter.setPen(QColor(INK))
            font = QFont(theme.SANS)
            font.setPixelSize(11)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(QRectF(sx + 14, sy - 8, 220, 16), Qt.AlignLeft | Qt.AlignVCenter,
                             f"{self._satellite} · {track.elevation[now]:.0f}° up")

    def _compass(self, painter: QPainter) -> None:
        cx, cy, R = self._geometry()
        painter.setPen(QPen(QColor(theme.ACCENT), 1.2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), R, R)
        font = QFont(theme.MONO)
        font.setPixelSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(INK))
        for label, az in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
            a = math.radians(az)
            x, y = cx - (R + 14) * math.sin(a), cy - (R + 14) * math.cos(a)
            painter.drawText(QRectF(x - 8, y - 8, 16, 16), Qt.AlignCenter, label)
