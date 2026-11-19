"""Mission replay: the front door of the app, for people who don't know signals.

Three screens in one window:

* **How it works** -- four pictures: a satellite sends, a station records,
  the AI checks, the physics confirms.
* **Home** -- three numbers and a few recordings to replay.
* **Replay** -- a satellite pass played back like a short film: the
  satellite crossing a world map along its real path, its coverage
  circle, the radio link to the station when it is above the horizon,
  and beside it the recording filling in second by second.  It ends on
  one sentence with a tick or a cross.

Nothing on these screens is decoration.  The path, the circle and the
moment the satellite rises come from its orbit; the picture is the real
recording; the verdict is the vetting model plus the orbit check.
"Expert mode" opens the vetting station with all the evidence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF, QRadialGradient
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..orbit import GroundTrack, OrbitError, TrackExplanation, ground_track
from ..session import Item, Verdict, VettingSession
from ..story import Story, tell
from . import theme
from .orbit_view import OrbitWorker

LAND_FILE = Path(__file__).resolve().parent / "assets" / "land_110m.json"

INK = "#e9eef4"
SOFT = "#9aa5b4"
OCEAN_TOP, OCEAN_BOTTOM = "#0b1320", "#0a0f17"
LAND, COAST, GRID = "#1a2330", "#2b3747", "#141c27"
STATION = "#f0c35a"

STYLE = theme.STYLE + f"""
QLabel#hero {{ font-family: "{theme.SANS}"; font-size: 34px; font-weight: 700; color: {INK}; }}
QLabel#lead {{ font-family: "{theme.SANS}"; font-size: 15px; color: {SOFT}; }}
QLabel#h2 {{ font-family: "{theme.SANS}"; font-size: 20px; font-weight: 700; color: {INK}; }}
QLabel#body {{ font-family: "{theme.SANS}"; font-size: 13px; color: {SOFT}; }}
QLabel#stat {{ font-family: "{theme.SANS}"; font-size: 40px; font-weight: 800; color: {INK}; }}
QFrame#card {{ background: {theme.PANEL}; border: 1px solid {theme.RULE}; border-radius: 10px; }}
QFrame#card:hover {{ border-color: {theme.DIM}; }}
QPushButton#big {{ font-family: "{theme.SANS}"; font-size: 14px; font-weight: 600; padding: 10px 18px;
                   border-radius: 8px; border: 1px solid {theme.ACCENT}; color: {theme.ACCENT}; background: transparent; }}
QPushButton#quiet {{ font-family: "{theme.SANS}"; font-size: 13px; padding: 8px 14px; border-radius: 8px;
                     border: 1px solid {theme.RULE}; color: {SOFT}; background: transparent; }}
"""


def label(text: str, name: str, wrap: bool = True) -> QLabel:
    widget = QLabel(text)
    widget.setObjectName(name)
    widget.setWordWrap(wrap)
    return widget


def _land() -> list[np.ndarray]:
    try:
        rings = json.loads(LAND_FILE.read_text(encoding="utf-8"))["rings"]
    except (OSError, ValueError, KeyError):
        return []
    return [np.asarray(r, dtype=float) for r in rings]


# ----------------------------------------------------------------------
class WorldMap(QWidget):
    """A map centred on the station, with the satellite's path, its coverage circle and the radio link."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(520, 360)
        self._land = _land()
        self._station: tuple[float, float] | None = None
        self._station_name = ""
        self._track: GroundTrack | None = None
        self._t = 0.0
        self._pulse = 0.0
        self._lon0, self._lat0 = 0.0, 20.0
        self._span = 75.0  # degrees of longitude either side of the centre

    def set_pass(self, station: tuple[float, float], name: str, track: GroundTrack | None) -> None:
        self._station, self._station_name, self._track = station, name, track
        self._lon0 = station[1]
        self._lat0 = float(np.clip(station[0], -50, 50))
        self.update()

    def set_time(self, seconds: float, pulse: float) -> None:
        self._t, self._pulse = seconds, pulse
        self.update()

    # -- projection: equirectangular around the station, longitudes wrapped
    def _scale(self) -> float:
        return self.width() / (2 * self._span * math.cos(math.radians(self._lat0)) ** 0.5)

    def _xy(self, lat: float | np.ndarray, lon: float | np.ndarray):
        k = self._scale()
        dlon = (np.asarray(lon) - self._lon0 + 180.0) % 360.0 - 180.0
        x = self.width() / 2 + dlon * k * math.cos(math.radians(self._lat0)) ** 0.5
        y = self.height() / 2 - (np.asarray(lat) - self._lat0) * k
        return x, y

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        sky = QLinearGradient(0, 0, 0, self.height())
        sky.setColorAt(0, QColor(OCEAN_TOP))
        sky.setColorAt(1, QColor(OCEAN_BOTTOM))
        painter.fillRect(self.rect(), sky)
        self._grid(painter)
        self._coasts(painter)
        if self._track is not None:
            self._pass(painter)
        if self._station is not None:
            self._pin(painter)
        painter.setPen(QColor(theme.FAINT))
        font = QFont(theme.SANS)
        font.setPixelSize(10)
        painter.setFont(font)
        painter.drawText(QRectF(8, self.height() - 18, 400, 14), Qt.AlignLeft,
                         "Map: Natural Earth (public domain).  Path and coverage from the satellite's orbit.")

    def _grid(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor(GRID), 1))
        for lat in range(-75, 90, 15):
            _, y = self._xy(lat, self._lon0)
            painter.drawLine(QPointF(0, float(y)), QPointF(self.width(), float(y)))
        for lon in range(-180, 180, 15):
            x, _ = self._xy(self._lat0, lon)
            painter.drawLine(QPointF(float(x), 0), QPointF(float(x), self.height()))

    def _coasts(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor(COAST), 1))
        painter.setBrush(QColor(LAND))
        for ring in self._land:
            lon = np.unwrap(ring[:, 0], period=360.0)
            for shift in (-360.0, 0.0, 360.0):
                x, y = self._xy(ring[:, 1], lon + shift)
                # _xy wraps longitudes; undo that for a continuous ring
                dlon = lon + shift - self._lon0
                x = self.width() / 2 + dlon * self._scale() * math.cos(math.radians(self._lat0)) ** 0.5
                if x.max() < -50 or x.min() > self.width() + 50:
                    continue
                painter.drawPolygon(QPolygonF([QPointF(a, b) for a, b in zip(x, y)]))

    def _pass(self, painter: QPainter) -> None:
        track = self._track
        x, y = self._xy(track.lat, track.lon)
        now = int(np.clip(np.searchsorted(track.seconds, self._t), 0, len(track.seconds) - 1))
        jumps = np.abs(np.diff(x)) > self.width() / 2

        def polyline(start: int, stop: int, pen: QPen) -> None:
            painter.setPen(pen)
            for i in range(max(start, 0), min(stop, len(x) - 1)):
                if not jumps[i]:
                    painter.drawLine(QPointF(x[i], y[i]), QPointF(x[i + 1], y[i + 1]))

        polyline(now, len(x), QPen(QColor(theme.FAINT), 1.2, Qt.DashLine))
        polyline(0, now, QPen(QColor(theme.ACCENT), 2.0))

        lat, lon = float(track.lat[now]), float(track.lon[now])
        visible = track.elevation[now] > 0
        # The coverage circle: everywhere that can see the satellite now.
        ring = self._circle(lat, lon, float(track.footprint_deg[now]))
        fill = QColor(theme.ACCENT)
        fill.setAlpha(36 if visible else 18)
        painter.setPen(QPen(QColor(theme.ACCENT if visible else theme.FAINT), 1))
        painter.setBrush(fill)
        painter.drawPolygon(ring)
        painter.setBrush(Qt.NoBrush)

        sx, sy = float(x[now]), float(y[now])
        if visible and self._station is not None:
            px, py = self._xy(*self._station)
            link = QColor(theme.SIGNAL)
            link.setAlpha(int(90 + 165 * self._pulse))
            painter.setPen(QPen(link, 1.5 + 2.5 * self._pulse))
            painter.drawLine(QPointF(sx, sy), QPointF(float(px), float(py)))
        glow = QRadialGradient(QPointF(sx, sy), 18)
        glow.setColorAt(0, QColor(86, 200, 216, 180))
        glow.setColorAt(1, QColor(86, 200, 216, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(QPointF(sx, sy), 18, 18)
        self._satellite_icon(painter, sx, sy)

    def _circle(self, lat: float, lon: float, radius_deg: float) -> QPolygonF:
        """Points ``radius_deg`` away from (lat, lon) on the sphere."""
        phi, lam, d = math.radians(lat), math.radians(lon), math.radians(radius_deg)
        points = []
        for bearing in np.linspace(0, 2 * math.pi, 90):
            lat2 = math.asin(math.sin(phi) * math.cos(d) + math.cos(phi) * math.sin(d) * math.cos(bearing))
            lon2 = lam + math.atan2(math.sin(bearing) * math.sin(d) * math.cos(phi),
                                    math.cos(d) - math.sin(phi) * math.sin(lat2))
            points.append((math.degrees(lat2), math.degrees(lon2)))
        lats = np.array([p[0] for p in points])
        lons = np.unwrap(np.array([p[1] for p in points]), period=360.0)
        dlon = (lons - self._lon0 + 180.0) % 360.0 - 180.0
        dlon = np.unwrap(dlon, period=360.0)
        x = self.width() / 2 + dlon * self._scale() * math.cos(math.radians(self._lat0)) ** 0.5
        _, y = self._xy(lats, self._lon0)
        return QPolygonF([QPointF(a, b) for a, b in zip(x, y)])

    @staticmethod
    def _satellite_icon(painter: QPainter, x: float, y: float) -> None:
        painter.save()
        painter.translate(x, y)
        painter.rotate(-30)
        painter.setPen(QPen(QColor(INK), 1))
        painter.setBrush(QColor(theme.ACCENT))
        painter.drawRect(QRectF(-12, -3, 8, 6))
        painter.drawRect(QRectF(4, -3, 8, 6))
        painter.setBrush(QColor(INK))
        painter.drawRect(QRectF(-3.5, -4, 7, 8))
        painter.restore()

    def _pin(self, painter: QPainter) -> None:
        x, y = (float(v) for v in self._xy(*self._station))
        glow = QRadialGradient(QPointF(x, y), 14)
        glow.setColorAt(0, QColor(240, 195, 90, 160))
        glow.setColorAt(1, QColor(240, 195, 90, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(QPointF(x, y), 14, 14)
        painter.setBrush(QColor(STATION))
        painter.drawEllipse(QPointF(x, y), 4, 4)
        painter.setPen(QColor(INK))
        font = QFont(theme.SANS)
        font.setPixelSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRectF(x + 10, y - 9, 260, 18), Qt.AlignLeft | Qt.AlignVCenter,
                         f"Ground station · {self._station_name[:28]}")


# ----------------------------------------------------------------------
class Dots(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(110, 18)
        self._n, self._colour = 0, theme.ACCENT

    def set(self, n: int, colour: str) -> None:
        self._n, self._colour = n, colour
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        for i in range(5):
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(self._colour) if i < self._n else QColor(theme.RULE))
            painter.drawEllipse(QPointF(9 + i * 20, 9), 6, 6)


class ReplayScreen(QWidget):
    back = Signal()
    expert = Signal(int)

    SECONDS_PER_PASS = 14.0   # a whole pass plays in this long at 1x

    def __init__(self, session: VettingSession, explainer, parent=None) -> None:
        super().__init__(parent)
        self._session, self._explainer = session, explainer
        self._item: Item | None = None
        self._explanation: TrackExplanation | None = None
        self._track: GroundTrack | None = None
        self._power: np.ndarray | None = None
        self._band: np.ndarray | None = None
        self._speed = 1.0
        self._t = 0.0
        self._t0, self._t1 = 0.0, 1.0
        self._workers: list = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 20, 28, 20)
        outer.setSpacing(14)
        top = QHBoxLayout()
        back = QPushButton("←  Home")
        back.setObjectName("quiet")
        back.clicked.connect(self.back.emit)
        top.addWidget(back)
        top.addSpacing(12)
        titles = QVBoxLayout()
        self.title = label("", "h2")
        self.subtitle = label("", "body")
        titles.addWidget(self.title)
        titles.addWidget(self.subtitle)
        top.addLayout(titles, 1)
        expert = QPushButton("Expert mode  →")
        expert.setObjectName("quiet")
        expert.clicked.connect(lambda: self._item and self.expert.emit(self._item.obs.id))
        top.addWidget(expert)
        outer.addLayout(top)

        middle = QHBoxLayout()
        middle.setSpacing(18)
        self.map = WorldMap()
        middle.addWidget(self.map, 3)
        right = QVBoxLayout()
        right.addWidget(label("WHAT THE ANTENNA HEARD", "caps"))
        self.scope = pg.PlotWidget(background=theme.BG)
        self.scope.setMenuEnabled(False)
        self.scope.hideButtons()
        self.scope.setMouseEnabled(False, False)
        for name in ("left", "bottom"):
            axis = self.scope.getAxis(name)
            axis.setPen(pg.mkPen(theme.FAINT))
            axis.setTextPen(pg.mkPen(theme.DIM))
            axis.enableAutoSIPrefix(False)
        self.scope.setLabel("left", "seconds")
        self.scope.setLabel("bottom", "frequency offset (kHz)")
        self.image = pg.ImageItem()
        self.image.setColorMap(pg.colormap.get("inferno"))
        self.scope.addItem(self.image)
        self.now_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(theme.ACCENT, width=1.5))
        self.scope.addItem(self.now_line)
        self.predicted = pg.PlotDataItem(pen=pg.mkPen("#ffffff", width=1.5, style=Qt.DashLine))
        self.scope.addItem(self.predicted)
        right.addWidget(self.scope, 1)
        self.caption = label("", "body")
        right.addWidget(self.caption)
        middle.addLayout(right, 2)
        outer.addLayout(middle, 1)

        card = QFrame()
        card.setObjectName("card")
        row = QHBoxLayout(card)
        row.setContentsMargins(20, 16, 20, 16)
        self.symbol = QLabel()
        self.symbol.setFixedWidth(64)
        self.symbol.setAlignment(Qt.AlignCenter)
        row.addWidget(self.symbol)
        words = QVBoxLayout()
        self.headline = label("", "h2")
        self.sentence = label("", "body")
        words.addWidget(self.headline)
        words.addWidget(self.sentence)
        sure = QHBoxLayout()
        sure.addWidget(label("HOW SURE", "caps", wrap=False))
        self.dots = Dots()
        sure.addWidget(self.dots)
        self.volunteer = label("", "body", wrap=False)
        sure.addSpacing(16)
        sure.addWidget(self.volunteer)
        sure.addStretch(1)
        words.addLayout(sure)
        row.addLayout(words, 1)
        controls = QVBoxLayout()
        buttons = QHBoxLayout()
        self.play = QPushButton("❚❚  Pause")
        self.play.setObjectName("big")
        self.play.clicked.connect(self.toggle)
        buttons.addWidget(self.play)
        self.speed = QPushButton("1×")
        self.speed.setObjectName("quiet")
        self.speed.clicked.connect(self._cycle_speed)
        buttons.addWidget(self.speed)
        controls.addLayout(buttons)
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, 1000)
        self.timeline.sliderMoved.connect(self._scrub)
        controls.addWidget(self.timeline)
        row.addLayout(controls)
        outer.addWidget(card)

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._clock = QElapsedTimer()
        self._show_story(None)

    # ------------------------------------------------------------------
    def load(self, obs_id: int) -> None:
        self._timer.stop()
        item = self._session.item(obs_id)
        obs = item.obs
        self._item, self._explanation = item, None
        satellite = self._name(obs)
        self.title.setText(f"{satellite} over {obs.station_name or 'a ground station'}")
        minutes = (obs.duration_s or 0) / 60
        when = obs.start.strftime("%d %b %Y, %H:%M UTC") if obs.start else ""
        self.subtitle.setText(f"Recording #{obs.id} · {when} · {minutes:.0f}-minute pass · "
                              + ("nobody has checked this one yet" if not item.known else "checked by a volunteer"))
        self._power = self._session.waterfall(obs.id)
        axes = self._session.cache.axes(obs, download=False)
        self._draw_recording(axes)
        self._track = None
        self.map.set_pass(self._station_or_default(obs), obs.station_name, None)
        try:
            site = self._explainer.stations.site(obs) if self._explainer else None
            if site is not None and obs.tle is not None and obs.start is not None:
                margin = 480.0
                seconds = np.linspace(-margin, (obs.duration_s or 600) + margin, 400)
                self._track = ground_track(obs.tle, site, obs.start, seconds)
                self.map.set_pass((site.lat, site.lng), obs.station_name, self._track)
        except (OrbitError, OSError, ValueError):
            pass
        self._show_story(None)
        if self._explainer is not None:
            worker = OrbitWorker(self._explainer, obs, self._power)
            worker.signals.done.connect(self._explained)
            worker.signals.failed.connect(lambda oid, _msg: oid == obs.id and self._show_story(None, final=True))
            self._workers.append(worker)
            from PySide6.QtCore import QThreadPool

            QThreadPool.globalInstance().start(worker)
        else:
            self._show_story(None, final=True)
        self.restart()

    def _station_or_default(self, obs) -> tuple[float, float]:
        if obs.station_lat is not None and obs.station_lng is not None:
            return float(obs.station_lat), float(obs.station_lng)
        return 20.0, 0.0

    def _name(self, obs) -> str:
        if self._explainer is not None:
            try:
                return self._explainer.catalog.name(int(obs.norad_cat_id or 0))
            except (OrbitError, OSError, ValueError):
                pass
        return f"Satellite {obs.norad_cat_id}"

    def _draw_recording(self, axes) -> None:
        power = self._power
        n_time, n_freq = power.shape
        obs = self._item.obs
        if axes is not None:
            left, right, bottom, top = axes.khz_left, axes.khz_right, axes.seconds_bottom, axes.seconds_top
        else:
            left, right, bottom, top = -n_freq / 2, n_freq / 2, 0.0, float(obs.duration_s or n_time)
        self._extent = (left, right, bottom, top)
        self._t0, self._t1 = bottom, top
        self.image.setRect(QRectF(left, bottom, right - left, top - bottom))
        self.scope.setRange(xRange=(left, right), yRange=(bottom, top), padding=0)
        edge = n_freq // 8
        centre = n_freq // 2
        half = max((n_freq - 2 * edge) // 20, 2)
        band = power[:, centre - half:centre + half].max(axis=1)
        self._band = np.clip((band - 2.0) / 4.0, 0.0, 1.0)

    def _explained(self, obs_id: int, explanation: TrackExplanation) -> None:
        if self._item is None or obs_id != self._item.obs.id:
            return
        self._explanation = explanation
        axes = explanation.prediction.axes
        self._draw_recording(axes)
        self._show_story(explanation, final=True)

    def _show_story(self, explanation: TrackExplanation | None, final: bool = False) -> None:
        if self._item is None:
            return
        self._final_story: Story | None = (tell(self._item, explanation, self._name(self._item.obs),
                                                self._session.policy) if final else None)
        best = explanation.matches[0] if explanation is not None and explanation.matches else None
        if best is not None and explanation.source in ("other", "ground", "scheduled"):
            up = best.curve.visible
            self._curve = (best.curve.khz[up] + best.shift_khz, best.curve.seconds[up])
        else:
            self._curve = None
        if self._item.known:
            said = "with signal" if self._item.obs.waterfall_label.value == "with-signal" else "without signal"
            self.volunteer.setText(f"A volunteer said: {said}")
        else:
            self.volunteer.setText("No volunteer has checked this yet")
        self._render_story()

    def _render_story(self) -> None:
        done = self._t >= self._t1 - 1e-6
        story = self._final_story
        if story is None or not done:
            self.symbol.setText(f"<span style='font-size:40px; color:{theme.DIM}'>…</span>")
            self.headline.setText("Listening…" if story is None or not done else story.headline)
            self.sentence.setText("Playing back the pass. The verdict appears at the end."
                                  if story is not None else "Working out the satellite's path and every "
                                                            "other source that could have made a signal.")
            self.dots.set(0, theme.DIM)
            return
        colour = {"heard": theme.SIGNAL, "not heard": theme.NO_SIGNAL}.get(story.mark, theme.UNSURE)
        self.symbol.setText(f"<span style='font-size:44px; color:{colour}'>{story.symbol}</span>")
        self.headline.setText(f"<span style='color:{colour}'>{story.headline}</span>")
        self.sentence.setText(story.sentence)
        self.dots.set(story.dots, colour)

    # -- playback --------------------------------------------------------
    def restart(self) -> None:
        self._t = self._t0
        self._clock.restart()
        self._timer.start()
        self.play.setText("❚❚  Pause")
        self._update_frame()

    def toggle(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.play.setText("▶  Play")
        else:
            if self._t >= self._t1:
                self.restart()
                return
            self._clock.restart()
            self._timer.start()
            self.play.setText("❚❚  Pause")

    def finish(self) -> None:
        """Jump to the end (tests, screenshots)."""
        self._t = self._t1
        self._timer.stop()
        self.play.setText("↻  Replay")
        self._update_frame()

    def _cycle_speed(self) -> None:
        self._speed = {1.0: 2.0, 2.0: 4.0, 4.0: 1.0}[self._speed]
        self.speed.setText(f"{self._speed:g}×")

    def _scrub(self, value: int) -> None:
        self._t = self._t0 + (self._t1 - self._t0) * value / 1000
        self._update_frame()

    def _tick(self) -> None:
        dt = self._clock.restart() / 1000.0
        self._t += dt * self._speed * (self._t1 - self._t0) / self.SECONDS_PER_PASS
        if self._t >= self._t1:
            self._t = self._t1
            self._timer.stop()
            self.play.setText("↻  Replay")
        self._update_frame()

    def _update_frame(self) -> None:
        if self._power is None:
            return
        n_time = self._power.shape[0]
        fraction = (self._t - self._t0) / max(self._t1 - self._t0, 1e-6)
        rows = int(np.clip(fraction, 0, 1) * n_time)
        shown = np.full_like(self._power, -3.0)
        shown[:rows] = self._power[:rows]
        self.image.setImage(shown, levels=(-2.0, 6.0))
        self.now_line.setValue(self._t)
        if self._curve is not None:
            khz, seconds = self._curve
            keep = seconds <= self._t
            self.predicted.setData(khz[keep], seconds[keep])
        else:
            self.predicted.setData([], [])
        pulse = float(self._band[min(rows, n_time - 1)]) if self._band is not None and rows else 0.0
        self.map.set_time(self._t, pulse)
        self.timeline.blockSignals(True)
        self.timeline.setValue(int(np.clip(fraction, 0, 1) * 1000))
        self.timeline.blockSignals(False)
        caption = f"{max(self._t, 0) // 60:.0f}:{max(self._t, 0) % 60:02.0f} into the recording"
        if self._track is not None:
            i = int(np.clip(np.searchsorted(self._track.seconds, self._t), 0, len(self._track.seconds) - 1))
            elevation, height = self._track.elevation[i], self._track.height_km[i]
            caption += (f"  ·  satellite {elevation:.0f}° above the horizon, {height:.0f} km up" if elevation > 0
                        else "  ·  satellite below the horizon")
        if self._curve is not None:
            caption += "  ·  dashed line: where the physics says the signal comes from"
        self.caption.setText(caption)
        self._render_story()


# ----------------------------------------------------------------------
class Step(QWidget):
    """One picture of 'How it works', drawn, not loaded."""

    def __init__(self, kind: str, parent=None) -> None:
        super().__init__(parent)
        self._kind = kind
        self.setFixedHeight(170)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Draw on a 200 x 130 canvas, scaled up to fit.
        scale = min(self.width() / 200, self.height() / 130) * 1.15
        painter.translate(self.width() / 2, self.height() / 2)
        painter.scale(scale, scale)
        cx, cy = 0.0, 0.0
        accent, ink = QColor(theme.ACCENT), QColor(INK)
        painter.setPen(QPen(accent, 2))
        if self._kind == "send":
            WorldMap._satellite_icon(painter, cx, cy - 20)
            for r in (18, 30, 42):
                painter.drawArc(QRectF(cx - r, cy - 20 - r, 2 * r, 2 * r), -60 * 16, -60 * 16)
        elif self._kind == "record":
            painter.drawLine(QPointF(cx, cy + 40), QPointF(cx, cy - 5))
            painter.drawArc(QRectF(cx - 34, cy - 40, 68, 50), 200 * 16, 140 * 16)
            painter.drawLine(QPointF(cx - 30, cy + 40), QPointF(cx + 30, cy + 40))
        elif self._kind == "check":
            rect = QRectF(cx - 40, cy - 40, 80, 80)
            gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            gradient.setColorAt(0, QColor("#3b0f70"))
            gradient.setColorAt(1, QColor("#000004"))
            painter.fillRect(rect, gradient)
            painter.setPen(QPen(QColor("#fcffa4"), 2))
            painter.drawLine(QPointF(cx, cy - 36), QPointF(cx, cy + 36))
            painter.setPen(QPen(ink, 2))
            painter.drawRect(rect)
        else:  # confirm
            painter.setPen(QPen(QColor(theme.RULE), 1))
            painter.drawEllipse(QPointF(cx, cy), 44, 44)
            painter.drawEllipse(QPointF(cx, cy), 22, 22)
            path = QPainterPath()
            path.moveTo(cx - 40, cy + 18)
            path.cubicTo(cx - 10, cy - 40, cx + 10, cy - 40, cx + 40, cy + 18)
            painter.setPen(QPen(accent, 2))
            painter.drawPath(path)
            painter.setPen(QPen(QColor(theme.SIGNAL), 4))
            painter.drawLine(QPointF(cx + 30, cy - 28), QPointF(cx + 36, cy - 20))
            painter.drawLine(QPointF(cx + 36, cy - 20), QPointF(cx + 50, cy - 40))


class IntroScreen(QWidget):
    start = Signal()

    STEPS = (
        ("send", "1  A satellite sends", "Small satellites circle the Earth and radio data down as they pass."),
        ("record", "2  A station records", "Volunteer stations around the world point an antenna and record the pass."),
        ("check", "3  The AI checks", "It looks at the recording: is the satellite's signal there, or only noise?"),
        ("confirm", "4  Physics confirms", "From the orbit it predicts where every signal should be, and names the "
                                           "source of the one it sees."),
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(60, 60, 60, 50)
        outer.addWidget(label("Listening to satellites", "hero"))
        outer.addWidget(label("Thousands of satellite recordings are made every day by volunteers, and nobody has "
                              "time to check them all. This app checks them, and says when it isn't sure.", "lead"))
        outer.addSpacing(40)
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        for column, (kind, title, text) in enumerate(self.STEPS):
            card = QFrame()
            card.setObjectName("card")
            inside = QVBoxLayout(card)
            inside.setContentsMargins(18, 18, 18, 18)
            inside.addWidget(Step(kind))
            inside.addWidget(label(title, "h2"))
            inside.addWidget(label(text, "body"))
            inside.addStretch(1)
            grid.addWidget(card, 0, column)
        outer.addLayout(grid)
        outer.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        go = QPushButton("Start  →")
        go.setObjectName("big")
        go.clicked.connect(self.start.emit)
        row.addWidget(go)
        outer.addLayout(row)


class HomeScreen(QWidget):
    replay = Signal(int)
    expert = Signal()
    intro = Signal()

    def __init__(self, session: VettingSession, parent=None) -> None:
        super().__init__(parent)
        self._session = session
        outer = QVBoxLayout(self)
        outer.setContentsMargins(60, 44, 60, 40)
        outer.setSpacing(10)
        top = QHBoxLayout()
        top.addWidget(label("Satellite pass checker", "hero"), 1)
        how = QPushButton("How it works")
        how.setObjectName("quiet")
        how.clicked.connect(self.intro.emit)
        top.addWidget(how)
        expert = QPushButton("Expert mode  →")
        expert.setObjectName("quiet")
        expert.clicked.connect(self.expert.emit)
        top.addWidget(expert)
        outer.addLayout(top)
        outer.addWidget(label("Real recordings from the SatNOGS network of volunteer ground stations.", "lead"))
        outer.addSpacing(26)

        stats = QHBoxLayout()
        stats.setSpacing(20)
        for value, text in self._numbers():
            card = QFrame()
            card.setObjectName("card")
            inside = QVBoxLayout(card)
            inside.setContentsMargins(22, 18, 22, 18)
            inside.addWidget(label(value, "stat"))
            inside.addWidget(label(text, "body"))
            stats.addWidget(card)
        outer.addLayout(stats)
        outer.addSpacing(26)
        outer.addWidget(label("Watch a pass", "h2"))
        outer.addWidget(label("Each one plays back the satellite's flight and the recording, then says what was "
                              "heard.", "body"))
        cards = QHBoxLayout()
        cards.setSpacing(20)
        for obs_id, title, text in self._featured():
            card = QPushButton()
            card.setObjectName("quiet")
            card.setMinimumHeight(150)
            card.setText(f"{title}\n\n{text}")
            card.setStyleSheet("text-align: left; padding: 16px; font-size: 13px;")
            card.clicked.connect(lambda _=False, i=obs_id: self.replay.emit(i))
            cards.addWidget(card)
        outer.addLayout(cards)
        outer.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        random_one = QPushButton("Replay a recording nobody has checked  →")
        random_one.setObjectName("big")
        random_one.clicked.connect(self._random)
        row.addWidget(random_one)
        outer.addLayout(row)

    def _numbers(self) -> list[tuple[str, str]]:
        session = self._session
        unvetted = session.count("unvetted")
        mode = session.mode
        session.set_mode("unvetted" if unvetted else "test")
        stats = session.stats()
        session.set_mode(mode)
        error, measured = session.measured_error()
        return [
            (f"{unvetted:,}", "recordings here that no volunteer has checked yet"),
            (f"{stats['automatable_share']:.0%}", "of them the AI is sure enough to check on its own"),
            (f"{1 - error:.0%}" if error == error else "–",
             f"right when it is sure, tested on {measured:,} recordings volunteers had checked"),
        ]

    def _featured(self) -> list[tuple[int, str, str]]:
        """A clear 'heard', a clear 'not heard', and one where physics names the source, if present."""
        session = self._session
        everything = session.all_items
        unvetted = [i for i in everything if not i.known]
        pool = unvetted or everything
        picks: list[tuple[int, str, str]] = []
        heard = max(pool, key=lambda i: i.probability, default=None)
        if heard is not None:
            picks.append((heard.obs.id, "A clear signal", f"#{heard.obs.id} · {heard.obs.transmitter_mode or '?'}"))
        silent = min(pool, key=lambda i: i.probability, default=None)
        if silent is not None and silent is not heard:
            picks.append((silent.obs.id, "Only noise", f"#{silent.obs.id} · {silent.obs.transmitter_mode or '?'}"))
        for obs_id, title in ((15489000, "Not a satellite at all"), (15490378, "The wrong satellite")):
            if any(i.obs.id == obs_id for i in everything):
                picks.append((obs_id, title, f"#{obs_id} · the physics finds the real source"))
        return picks[:4]

    def _random(self) -> None:
        pool = [i for i in self._session.all_items if not i.known] or self._session.all_items
        if pool:
            self.replay.emit(pool[np.random.default_rng().integers(len(pool))].obs.id)


class MissionWindow(QMainWindow):
    """The app's front door: how it works, home, replay; expert mode on request."""

    def __init__(self, session: VettingSession, explainer=None, show_intro: bool = True) -> None:
        super().__init__()
        self._session, self._explainer = session, explainer
        self._expert = None
        self.setWindowTitle("Satellite pass checker")
        self.resize(1440, 880)
        self.setStyleSheet(STYLE)
        self.stack = QStackedWidget()
        self.intro = IntroScreen()
        self.home = HomeScreen(session)
        self.replay = ReplayScreen(session, explainer)
        for screen in (self.intro, self.home, self.replay):
            self.stack.addWidget(screen)
        self.setCentralWidget(self.stack)
        self.intro.start.connect(lambda: self.stack.setCurrentWidget(self.home))
        self.home.intro.connect(lambda: self.stack.setCurrentWidget(self.intro))
        self.home.replay.connect(self.open_replay)
        self.home.expert.connect(lambda: self.open_expert(None))
        self.replay.back.connect(lambda: self.stack.setCurrentWidget(self.home))
        self.replay.expert.connect(self.open_expert)
        self.stack.setCurrentWidget(self.intro if show_intro else self.home)

    def open_replay(self, obs_id: int) -> None:
        self.stack.setCurrentWidget(self.replay)
        self.replay.load(obs_id)

    def open_expert(self, obs_id: int | None) -> None:
        from .window import VettingWindow

        if self._expert is None:
            self._expert = VettingWindow(self._session, self._explainer)
        self._expert.show()
        self._expert.raise_()
        if obs_id is not None:
            self._expert.select(obs_id)
