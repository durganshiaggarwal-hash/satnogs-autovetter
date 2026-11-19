"""A 3D globe, drawn with the same orbit numbers as the flat map.

It is an orthographic projection -- the Earth as seen from far away --
painted with QPainter rather than OpenGL, so it runs anywhere Qt runs
(no graphics driver needed, and it can be screenshotted).  The satellite
is drawn at its true height above the surface (600 km is about a tenth
of the Earth's radius, so it visibly floats), its orbit wraps round the
globe, and parts behind the Earth are hidden.  Drag to turn the globe.

The projection, for a point at latitude phi and longitude lambda seen
from above (phi0, lambda0)::

    x = r cos(phi) sin(lambda - lambda0)
    y = r (cos(phi0) sin(phi) - sin(phi0) cos(phi) cos(lambda - lambda0))
    z = r (sin(phi0) sin(phi) + cos(phi0) cos(phi) cos(lambda - lambda0))

and the point faces us when z > 0.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF, QRadialGradient
from PySide6.QtWidgets import QWidget

from ..orbit import EARTH_RADIUS_KM, GroundTrack
from . import theme
from .mission import INK, STATION, WorldMap, _land


class GlobeMap(QWidget):
    """The same interface as :class:`WorldMap`: ``set_pass`` and ``set_time``."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 300)
        self._land = _land()
        self._station: tuple[float, float] | None = None
        self._station_name = ""
        self._track: GroundTrack | None = None
        self._t = 0.0
        self._pulse = 0.0
        self._caption = ""
        self._lat0, self._lon0 = 30.0, 0.0
        self._drag: QPointF | None = None
        self.setCursor(Qt.OpenHandCursor)

    # -- the shared interface ------------------------------------------
    def set_pass(self, station: tuple[float, float], name: str, track: GroundTrack | None) -> None:
        self._station, self._station_name, self._track = station, name, track
        self._lat0 = float(np.clip(station[0] - 10, -70, 70))
        self._lon0 = station[1]
        self.update()

    def set_time(self, seconds: float, pulse: float, caption: str = "") -> None:
        self._t, self._pulse, self._caption = seconds, pulse, caption
        self.update()

    # -- turning it ----------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._drag = event.position()
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag is None:
            return
        delta = event.position() - self._drag
        self._drag = event.position()
        degrees_per_pixel = 180.0 / max(self._radius() * math.pi, 1)
        self._lon0 = (self._lon0 - delta.x() * degrees_per_pixel + 180) % 360 - 180
        self._lat0 = float(np.clip(self._lat0 + delta.y() * degrees_per_pixel, -80, 80))
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag = None
        self.setCursor(Qt.OpenHandCursor)

    # -- projection ----------------------------------------------------
    def _radius(self) -> float:
        return min(self.width(), self.height() - 40) * 0.46

    def _project(self, lat, lon, height_km=0.0):
        """Screen x, y and whether the point is in front of (or beside) the Earth."""
        lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
        phi0, lam0 = math.radians(self._lat0), math.radians(self._lon0)
        r = self._radius() * (1.0 + np.asarray(height_km, float) / EARTH_RADIUS_KM)
        dl = lon - lam0
        x = r * np.cos(lat) * np.sin(dl)
        y = r * (math.cos(phi0) * np.sin(lat) - math.sin(phi0) * np.cos(lat) * np.cos(dl))
        z = r * (math.sin(phi0) * np.sin(lat) + math.cos(phi0) * np.cos(lat) * np.cos(dl))
        visible = (z > 0) | (x * x + y * y > self._radius() ** 2)
        cx, cy = self.width() / 2, (self.height() - 30) / 2
        return cx + x, cy - y, visible, z

    def _surface_polygon(self, lat, lon) -> QPolygonF | None:
        """A shape on the surface; points round the back are pulled to the edge of the disc."""
        x, y, _, z = self._project(lat, lon)
        if (z <= 0).all():
            return None
        cx, cy, R = self.width() / 2, (self.height() - 30) / 2, self._radius()
        dx, dy = x - cx, y - cy
        length = np.maximum(np.hypot(dx, dy), 1e-9)
        behind = z <= 0
        x = np.where(behind, cx + dx / length * R, x)
        y = np.where(behind, cy + dy / length * R, y)
        return QPolygonF([QPointF(a, b) for a, b in zip(x, y)])

    # -- drawing -------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#06090e"))
        cx, cy, R = self.width() / 2, (self.height() - 30) / 2, self._radius()
        centre = QPointF(cx, cy)

        glow = QRadialGradient(centre, R * 1.18)
        glow.setColorAt(0.80, QColor(86, 200, 216, 70))
        glow.setColorAt(1.0, QColor(86, 200, 216, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(centre, R * 1.18, R * 1.18)
        ocean = QRadialGradient(QPointF(cx - R * 0.35, cy - R * 0.35), R * 1.5)
        ocean.setColorAt(0.0, QColor("#16304a"))
        ocean.setColorAt(1.0, QColor("#060c16"))
        painter.setBrush(ocean)
        painter.drawEllipse(centre, R, R)

        painter.save()
        clip = QPainterPath()
        clip.addEllipse(centre, R, R)
        painter.setClipPath(clip)
        self._graticule(painter)
        painter.setPen(QPen(QColor("#2f4058"), 0.8))
        painter.setBrush(QColor("#1d2b3c"))
        for ring in self._land:
            polygon = self._surface_polygon(ring[:, 1], ring[:, 0])
            if polygon is not None:
                painter.drawPolygon(polygon)
        if self._track is not None:
            self._footprint(painter)
        painter.restore()

        if self._track is not None:
            self._orbit(painter)
        if self._station is not None:
            self._pin(painter)
        WorldMap._subtitle(self, painter)  # the same subtitles as the flat map
        painter.setPen(QColor(theme.FAINT))
        font = QFont(theme.SANS)
        font.setPixelSize(10)
        painter.setFont(font)
        painter.drawText(QRectF(8, self.height() - 18, 420, 14), Qt.AlignLeft,
                         "Drag to turn the globe.  Satellite drawn at its true height.")

    def _graticule(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor(255, 255, 255, 18), 0.8))
        steps = np.linspace(-180, 180, 181)
        for lat in range(-60, 90, 30):
            self._polyline(painter, np.full_like(steps, lat), steps)
        lats = np.linspace(-85, 85, 120)
        for lon in range(-180, 180, 30):
            self._polyline(painter, lats, np.full_like(lats, lon))

    def _polyline(self, painter: QPainter, lat, lon, height_km=0.0) -> None:
        x, y, visible, _ = self._project(lat, lon, height_km)
        for i in range(len(x) - 1):
            if visible[i] and visible[i + 1]:
                painter.drawLine(QPointF(x[i], y[i]), QPointF(x[i + 1], y[i + 1]))

    def _now(self) -> int:
        return int(np.clip(np.searchsorted(self._track.seconds, self._t), 0, len(self._track.seconds) - 1))

    def _footprint(self, painter: QPainter) -> None:
        track, i = self._track, self._now()
        lat, lon, d = map(math.radians, (track.lat[i], track.lon[i], track.footprint_deg[i]))
        lats, lons = [], []
        for bearing in np.linspace(0, 2 * math.pi, 90):
            lat2 = math.asin(math.sin(lat) * math.cos(d) + math.cos(lat) * math.sin(d) * math.cos(bearing))
            lon2 = lon + math.atan2(math.sin(bearing) * math.sin(d) * math.cos(lat),
                                    math.cos(d) - math.sin(lat) * math.sin(lat2))
            lats.append(math.degrees(lat2))
            lons.append(math.degrees(lon2))
        polygon = self._surface_polygon(np.array(lats), np.array(lons))
        if polygon is None:
            return
        visible = track.elevation[i] > 0
        fill = QColor(theme.ACCENT)
        fill.setAlpha(40 if visible else 18)
        painter.setPen(QPen(QColor(theme.ACCENT if visible else theme.FAINT), 1))
        painter.setBrush(fill)
        painter.drawPolygon(polygon)

    def _orbit(self, painter: QPainter) -> None:
        track, i = self._track, self._now()
        x, y, visible, _ = self._project(track.lat, track.lon, track.height_km)
        for j in range(len(x) - 1):
            if visible[j] and visible[j + 1]:
                past = j < i
                painter.setPen(QPen(QColor(theme.ACCENT if past else theme.FAINT), 2.0 if past else 1.0,
                                    Qt.SolidLine if past else Qt.DashLine))
                painter.drawLine(QPointF(x[j], y[j]), QPointF(x[j + 1], y[j + 1]))
        if not visible[i]:
            return
        sx, sy = float(x[i]), float(y[i])
        if track.elevation[i] > 0 and self._station is not None:
            px, py, _, _ = self._project(*self._station)
            link = QColor(theme.SIGNAL)
            link.setAlpha(int(90 + 165 * self._pulse))
            painter.setPen(QPen(link, 1.5 + 2.0 * self._pulse))
            painter.drawLine(QPointF(sx, sy), QPointF(float(px), float(py)))
            if self._pulse > 0.35:
                WorldMap._pulses(painter, sx, sy, float(px), float(py), self._t)
        halo = QRadialGradient(QPointF(sx, sy), 16)
        halo.setColorAt(0, QColor(86, 200, 216, 170))
        halo.setColorAt(1, QColor(86, 200, 216, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(halo)
        painter.drawEllipse(QPointF(sx, sy), 16, 16)
        WorldMap._satellite_icon(painter, sx, sy)
        painter.setPen(QColor(INK))
        font = QFont(theme.MONO)
        font.setPixelSize(10)
        painter.setFont(font)
        painter.drawText(QRectF(sx - 80, sy + 12, 160, 14), Qt.AlignHCenter, f"{track.height_km[i]:,.0f} km up")

    def _pin(self, painter: QPainter) -> None:
        x, y, visible, _ = self._project(*self._station)
        if not visible:
            return
        x, y = float(x), float(y)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(STATION))
        painter.drawEllipse(QPointF(x, y), 4, 4)
        painter.setPen(QColor(INK))
        font = QFont(theme.SANS)
        font.setPixelSize(11)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRectF(x + 8, y - 8, 220, 16), Qt.AlignLeft | Qt.AlignVCenter,
                         f"Station · {self._station_name[:24]}")
