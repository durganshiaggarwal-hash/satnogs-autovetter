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
    QScrollArea,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QSizePolicy,
    QWidget,
)

from ..orbit import GroundTrack, OrbitError, TrackExplanation, ground_track
from ..session import Item, Verdict, VettingSession
from ..narration import PassNarrator, Places
from ..story import Story, tell
from . import theme
from .backdrop import OrbitHero, StarryScreen, count_up, fade_in
from .orbit_view import OrbitWorker

LAND_FILE = Path(__file__).resolve().parent / "assets" / "land_110m.json"

INK = theme.TEXT
SOFT = "#c9c9c9"
OCEAN_TOP, OCEAN_BOTTOM = "#3a4552", "#2a3038"
LAND, COAST, GRID = "#4f4f4f", "#757575", "#36404b"
STATION = theme.UNSURE

STYLE = theme.STYLE + f"""
QLabel#hero {{ font-family: "{theme.SANS_BOLD}"; font-size: 40px; letter-spacing: 7px; color: {INK}; }}
QLabel#lead {{ font-family: "{theme.SANS_SEMIBOLD}"; font-size: 19px; letter-spacing: 1px; color: {theme.ACCENT}; }}
QLabel#h2 {{ font-family: "{theme.SANS_SEMIBOLD}"; font-size: 21px; letter-spacing: 2px; color: {theme.ACCENT}; }}
QLabel#body {{ font-family: "{theme.SANS}"; font-size: 15px; color: {SOFT}; }}
QLabel#stat {{ font-family: "{theme.SANS_BOLD}"; font-size: 46px; letter-spacing: 2px; color: {INK}; }}
QFrame#card {{ background: rgba(74, 74, 74, 225); border: 1px solid {theme.RULE}; border-radius: 0; }}
QFrame#card:hover {{ border-color: {theme.ACCENT}; }}
QFrame#divider {{ background: {theme.RULE}; max-height: 1px; min-height: 1px; }}
QPushButton#big {{ font-family: "{theme.SANS_SEMIBOLD}"; font-size: 17px; padding: 12px 8px; border: none;
                   border-bottom: 1px solid {INK}; color: {INK}; background: transparent; }}
QPushButton#big:hover {{ color: {theme.ACCENT}; border-bottom: 2px solid {theme.ACCENT}; }}
QPushButton#quiet {{ font-family: "{theme.SANS_SEMIBOLD}"; font-size: 15px; padding: 9px 8px; border: none;
                     border-bottom: 1px solid {theme.LINE}; color: {theme.ACCENT}; background: transparent; }}
QPushButton#quiet:hover {{ border-bottom: 2px solid {theme.ACCENT}; }}
"""


def heading(text: str) -> str:
    """Headings are set in capitals, as in the rest of the app."""
    return text.upper()


def divider() -> QFrame:
    line = QFrame()
    line.setObjectName("divider")
    return line


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
        self.setMinimumSize(520, 300)
        self._land = _land()
        self._station: tuple[float, float] | None = None
        self._station_name = ""
        self._track: GroundTrack | None = None
        self._t = 0.0
        self._pulse = 0.0
        self._lon0, self._lat0 = 0.0, 20.0
        self._span = 75.0  # degrees of longitude either side of the centre
        self._caption = ""

    def set_pass(self, station: tuple[float, float], name: str, track: GroundTrack | None) -> None:
        self._station, self._station_name, self._track = station, name, track
        self._lon0 = station[1]
        self._lat0 = float(np.clip(station[0], -50, 50))
        self.update()

    def set_time(self, seconds: float, pulse: float, caption: str = "") -> None:
        self._t, self._pulse, self._caption = seconds, pulse, caption
        self.update()

    def _subtitle(self, painter: QPainter) -> None:
        """The narration, like film subtitles: centred near the bottom, on a dark band."""
        if not self._caption:
            return
        font = QFont(theme.SANS)
        font.setPixelSize(18)
        font.setWeight(QFont.Medium)
        painter.setFont(font)
        width = min(self.width() - 60, 760)
        box = QRectF((self.width() - width) / 2, 0, width, 1000)
        needed = painter.boundingRect(box.adjusted(16, 10, -16, -10), Qt.TextWordWrap | Qt.AlignHCenter,
                                      self._caption)
        height = needed.height() + 22
        box = QRectF((self.width() - width) / 2, self.height() - height - 30, width, height)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 215))
        painter.drawRect(box)
        painter.fillRect(QRectF(box.left(), box.top(), 3, box.height()), QColor(theme.ACCENT))
        painter.setPen(QColor(INK))
        painter.drawText(box.adjusted(16, 10, -16, -10), Qt.TextWordWrap | Qt.AlignHCenter | Qt.AlignVCenter,
                         self._caption)

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
        self._subtitle(painter)
        painter.setPen(QColor(theme.FAINT))
        font = QFont(theme.SANS)
        font.setPixelSize(12)
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
            if self._pulse > 0.35:
                self._pulses(painter, sx, sy, float(px), float(py), self._t)
        glow = QRadialGradient(QPointF(sx, sy), 18)
        glow.setColorAt(0, QColor(*theme.ACCENT_RGB, 180))
        glow.setColorAt(1, QColor(*theme.ACCENT_RGB, 0))
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
    def _pulses(painter: QPainter, sx: float, sy: float, px: float, py: float, seconds: float) -> None:
        """Bright dots travelling down the link, satellite to station: drawn only while a signal is heard."""
        painter.setPen(Qt.NoPen)
        for k in range(4):
            phase = (seconds * 0.9 + k / 4) % 1.0
            x, y = sx + (px - sx) * phase, sy + (py - sy) * phase
            colour = QColor(theme.SIGNAL)
            colour.setAlpha(int(255 * (1 - abs(phase - 0.5))))
            painter.setBrush(colour)
            painter.drawEllipse(QPointF(x, y), 3.2, 3.2)

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
        font.setPixelSize(14)
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
        self._narrator: PassNarrator | None = None
        self._places: Places | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 20, 28, 20)
        outer.setSpacing(14)
        top = QHBoxLayout()
        back = QPushButton("Home")
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
        outer.addLayout(top)

        from .globe import GlobeMap
        from .link import LinkPanel, PassTimeline
        from .skymap import SkyMap
        from .orbit_view import SkyPlot

        middle = QHBoxLayout()
        middle.setSpacing(18)
        left = QVBoxLayout()
        left.setSpacing(10)
        switch = QHBoxLayout()
        switch.setSpacing(0)
        switch.addWidget(label("THE PASS", "caps", wrap=False))
        switch.addSpacing(18)
        self.view_buttons = {}
        for name in ("Map", "Globe", "Sky"):
            button = QPushButton(name)
            button.setObjectName("tab")
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, n=name: self.set_view(n))
            self.view_buttons[name] = button
            switch.addWidget(button)
        switch.addStretch(1)
        left.addLayout(switch)
        self.views = QStackedWidget()
        self.map = WorldMap()
        self.globe = GlobeMap()
        self.skymap = SkyMap()
        self.views.addWidget(self.map)
        self.views.addWidget(self.globe)
        self.views.addWidget(self.skymap)
        left.addWidget(self.views, 1)
        # The station's sky, radar-style, as an inset in the map's corner.
        self.sky = SkyPlot(self.views)
        self.sky.setFixedSize(170, 170)
        self.sky.filled = True
        self.sky.raise_()
        self.link = LinkPanel()
        middle.addLayout(left, 3)
        # Right column: the recording, or the data packets decoded from it.
        from ..frames import FrameStore
        from .messages import MessagesPanel

        column = QVBoxLayout()
        tabs = QHBoxLayout()
        tabs.setSpacing(0)
        tabs.addWidget(label("THE RECORDING", "caps", wrap=False))
        tabs.addSpacing(18)
        self.page_buttons = {}
        for name in ("Recording", "Messages"):
            button = QPushButton(name)
            button.setObjectName("tab")
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, n=name: self.set_page(n))
            self.page_buttons[name] = button
            tabs.addWidget(button)
        tabs.addStretch(1)
        column.addLayout(tabs)
        self.pages = QStackedWidget()
        recording_page = QWidget()
        right = QVBoxLayout(recording_page)
        right.setContentsMargins(0, 0, 0, 0)
        root_data = Path(__file__).resolve().parents[3] / "satrecovery_data" / "satnogs"
        self.messages = MessagesPanel(FrameStore(root_data) if explainer is not None else None)
        self.messages.counted.connect(lambda n: self.page_buttons["Messages"].setText(f"Messages ({n})"))
        self.pages.addWidget(recording_page)
        self.pages.addWidget(self.messages)
        column.addWidget(self.pages, 1)
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
        right.addWidget(label("THE RADIO LINK, NOW", "caps", wrap=False))
        right.addWidget(self.link)
        middle.addLayout(column, 2)
        outer.addLayout(middle, 1)

        card = QFrame()
        card.setObjectName("card")
        row = QHBoxLayout(card)
        row.setContentsMargins(20, 16, 20, 16)
        # The verdict as a word in a coloured tile (no symbols).
        self.symbol = QLabel()
        self.symbol.setFixedWidth(150)
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
        self.next_passes = label("", "body")
        words.addWidget(self.next_passes)
        row.addLayout(words, 1)
        controls = QVBoxLayout()
        buttons = QHBoxLayout()
        self.play = QPushButton("Pause")
        self.play.setObjectName("big")
        self.play.clicked.connect(self.toggle)
        buttons.addWidget(self.play)
        self.speed = QPushButton("1×")
        self.speed.setObjectName("quiet")
        self.speed.clicked.connect(self._cycle_speed)
        buttons.addWidget(self.speed)
        controls.addLayout(buttons)
        row.addLayout(controls)
        self.pass_line = PassTimeline()
        self.pass_line.scrubbed.connect(lambda f: self._scrub(int(f * 1000)))
        outer.addWidget(self.pass_line)
        outer.addWidget(card)

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._clock = QElapsedTimer()
        self._show_story(None)
        self.set_view("Map")

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._place_sky()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._place_sky()

    def _place_sky(self) -> None:
        self.sky.move(self.views.width() - self.sky.width() - 12, 12)
        self.sky.raise_()

    def set_page(self, name: str) -> None:
        self.pages.setCurrentWidget(self.messages if name == "Messages" else self.pages.widget(0))
        for key, button in self.page_buttons.items():
            button.setChecked(key == name)

    def set_view(self, name: str) -> None:
        self.views.setCurrentWidget({"Globe": self.globe, "Sky": self.skymap}.get(name, self.map))
        # The small sky inset would repeat the Sky view, so hide it there.
        self.sky.setVisible(name != "Sky")
        for key, button in self.view_buttons.items():
            button.setChecked(key == name)
        self._place_sky()   # changing page raises it above the inset

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
        self._narrator = None
        self.map.set_pass(self._station_or_default(obs), obs.station_name, None)
        self.globe.set_pass(self._station_or_default(obs), obs.station_name, None)
        self.skymap.set_pass(self._station_or_default(obs), obs.station_name, None, obs.start, satellite)
        self.sky.set_tracks(None, [])
        self.next_passes.setText("")
        self.messages.show_observation(obs)
        self.set_page("Recording")
        try:
            site = self._explainer.stations.site(obs) if self._explainer else None
            if site is not None and obs.tle is not None and obs.start is not None:
                margin = 480.0
                seconds = np.linspace(-margin, (obs.duration_s or 600) + margin, 400)
                self._track = ground_track(obs.tle, site, obs.start, seconds)
                self.map.set_pass((site.lat, site.lng), obs.station_name, self._track)
                self.globe.set_pass((site.lat, site.lng), obs.station_name, self._track)
                self.skymap.set_pass((site.lat, site.lng), obs.station_name, self._track, obs.start, satellite)
                self.sky.set_tracks(self._track.seconds, [(self._track.azimuth, self._track.elevation,
                                                           theme.ACCENT, True)])
                self._show_next_passes(obs, site)
                self.pass_line.set_pass(obs.start, self._t0, self._t1, self._track,
                                        np.maximum(self._band, self._anywhere))
                self._places = self._places or Places()
                self._narrator = PassNarrator(self._track, satellite, self._places, activity=self._activity)
        except (OrbitError, OSError, ValueError):
            pass
        self._show_story(None)
        if self._explainer is not None:
            worker = OrbitWorker(self._explainer, obs, self._power)
            worker.signals.done.connect(self._explained)
            worker.signals.failed.connect(lambda oid, _msg: oid == obs.id and self._show_story(None, final=True))
            self._workers = OrbitWorker.prune(self._workers) + [worker]
            from PySide6.QtCore import QThreadPool

            QThreadPool.globalInstance().start(worker)
        else:
            self._show_story(None, final=True)
        self.restart()

    def _show_next_passes(self, obs, site) -> None:
        from ..orbit import next_passes

        try:
            passes = next_passes(obs.tle, site, obs.end or obs.start, hours=24)
        except (OrbitError, ValueError):
            return
        if passes:
            when = "  ·  ".join(f"{p.rise:%d %b %H:%M} UTC, {p.max_elevation:.0f}° high" for p in passes[:3])
            self.next_passes.setText(f"Next passes over this station after this recording:  {when}")

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
        # Strong anywhere in the band (not only the centre): a stray satellite or
        # interference crosses the centre only briefly.
        anywhere = power[:, edge:n_freq - edge].max(axis=1)
        self._anywhere = np.clip((anywhere - 3.0) / 3.0, 0.0, 1.0)
        if hasattr(self, "pass_line"):
            self.pass_line.set_pass(obs.start, self._t0, self._t1, getattr(self, "_track", None),
                                    np.maximum(self._band, self._anywhere))

    def _activity(self, seconds: float) -> float:
        """How strong the signal at the centre is at this moment of the recording (0..1)."""
        if self._band is None:
            return 0.0
        fraction = (seconds - self._t0) / max(self._t1 - self._t0, 1e-6)
        row = int(np.clip(fraction, 0, 0.999) * len(self._band))
        return float(max(self._band[row], self._anywhere[row]))

    def _explained(self, obs_id: int, explanation: TrackExplanation) -> None:
        if self._item is None or obs_id != self._item.obs.id:
            return
        self._explanation = explanation
        axes = explanation.prediction.axes
        self._draw_recording(axes)
        if self._narrator is not None:
            self._narrator.set_explanation(explanation)
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
            self._tag("LISTENING", theme.DIM)
            self.headline.setText("Listening…" if story is None or not done else story.headline)
            self.sentence.setText("Playing back the pass. The verdict appears at the end."
                                  if story is not None else "Working out the satellite's path and every "
                                                            "other source that could have made a signal.")
            self.dots.set(0, theme.DIM)
            return
        colour = {"heard": theme.SIGNAL, "not heard": theme.NO_SIGNAL}.get(story.mark, theme.UNSURE)
        self._tag({"heard": "HEARD", "not heard": "NOT HEARD"}.get(story.mark, "UNSURE"), colour)
        self.headline.setText(f"<span style='color:{colour}'>{story.headline}</span>")
        packets = self._item.obs.n_decoded_frames if self._item is not None else 0
        extra = ("  Open Messages to see what it sent." if packets and "decoded" in story.sentence else
                 f"  The station also decoded {packets} data packet{'s' if packets != 1 else ''} from this pass: "
                 f"open Messages to see what it sent." if packets and story.mark != "not heard" else "")
        self.sentence.setText(story.sentence + extra)
        self.dots.set(story.dots, colour)

    def _tag(self, word: str, colour: str) -> None:
        self.symbol.setText(word)
        self.symbol.setStyleSheet(f"color: {colour}; background: {theme.RAISED}; border-left: 3px solid {colour}; "
                                  f"font-family: \"{theme.SANS_BOLD}\"; font-size: 16px; letter-spacing: 3px; "
                                  f"padding: 22px 8px;")

    # -- playback --------------------------------------------------------
    def restart(self) -> None:
        self._t = self._t0
        self._clock.restart()
        self._timer.start()
        self.play.setText("Pause")
        self._update_frame()

    def toggle(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.play.setText("Play")
        else:
            if self._t >= self._t1:
                self.restart()
                return
            self._clock.restart()
            self._timer.start()
            self.play.setText("Pause")

    def seek(self, fraction: float) -> None:
        """Pause at a point of the pass (0 = start of the recording, 1 = end)."""
        self._t = self._t0 + (self._t1 - self._t0) * float(np.clip(fraction, 0, 1))
        self._timer.stop()
        self.play.setText("Play")
        self._update_frame()

    def finish(self) -> None:
        """Jump to the end (tests, screenshots)."""
        self._t = self._t1
        self._timer.stop()
        self.play.setText("Replay")
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
            self.play.setText("Replay")
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
        subtitle = self._narrator.line(self._t) if self._narrator is not None else ""
        self.map.set_time(self._t, pulse, subtitle)
        self.globe.set_time(self._t, pulse, subtitle)
        self.skymap.set_time(self._t, pulse, subtitle)
        self.sky.set_time(self._t)
        self.pass_line.set_time(self._t)
        self.link.show_moment(self._track, self._t, self._item.obs.frequency_hz if self._item else None)
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
            gradient.setColorAt(0, QColor(theme.ACCENT_FILL))
            gradient.setColorAt(1, QColor(theme.BG))
            painter.fillRect(rect, gradient)
            painter.setPen(QPen(QColor(theme.UNSURE), 2))
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
            # The satellite on its predicted path: a solid dot with a ring.
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(theme.SIGNAL))
            painter.drawEllipse(QPointF(cx + 20, cy - 23), 5, 5)
            painter.setPen(QPen(QColor(theme.SIGNAL), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cx + 20, cy - 23), 10, 10)


class IntroScreen(StarryScreen):
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
        outer.addWidget(label(heading("Listening to satellites"), "hero"))
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


def waterfall_thumbnail(power: np.ndarray, width: int = 64, height: int = 72):
    """A small picture of a recording (inferno colours), for cards."""
    from PySide6.QtGui import QImage, QPixmap

    lut = (pg.colormap.get("inferno").getLookupTable(nPts=256, alpha=False)).astype(np.uint8)
    small = power[::-1]                                       # start of the pass at the bottom, as on screen
    rows = np.linspace(0, small.shape[0] - 1, height).astype(int)
    cols = np.linspace(0, small.shape[1] - 1, width).astype(int)
    level = np.clip((small[np.ix_(rows, cols)] + 2.0) / 8.0, 0, 1)
    rgb = np.ascontiguousarray(lut[(level * 255).astype(np.uint8)])
    image = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class RecordingCard(QPushButton):
    """One recording on the home screen: a thumbnail, the satellite, the verdict, the station, when."""

    def __init__(self, item: Item, title: str | None = None, satellite: str = "", picture=None,
                 parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("recording")
        self.setFixedHeight(112)
        self.setMinimumWidth(250)
        self.setCursor(Qt.PointingHandCursor)
        obs = item.obs
        self.obs_id = obs.id
        mark, colour = {Verdict.SIGNAL: ("HEARD", theme.SIGNAL), Verdict.NO_SIGNAL: ("NOTHING HEARD", theme.NO_SIGNAL)
                        }.get(item.verdict, ("UNSURE", theme.UNSURE))
        ago = ""
        if obs.end is not None:
            from datetime import datetime, timezone

            minutes = (datetime.now(timezone.utc) - obs.end).total_seconds() / 60
            ago = f"{minutes:.0f} min ago" if minutes < 120 else f"{minutes / 60:.0f} h ago"
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 10, 12, 10)
        row.setSpacing(12)
        if picture is not None:
            thumb = QLabel()
            thumb.setPixmap(picture)
            thumb.setFixedSize(picture.size())
            thumb.setAttribute(Qt.WA_TransparentForMouseEvents)
            row.addWidget(thumb)
        words = QVBoxLayout()
        words.setSpacing(2)
        top = QHBoxLayout()
        verdict = QLabel((title or mark).upper())
        verdict.setStyleSheet(f"color: {colour}; font-family: \"{theme.SANS_SEMIBOLD}\"; font-size: 12px; "
                              f"letter-spacing: 2px;")
        top.addWidget(verdict, 1)
        when = QLabel("" if title else ago)      # a titled card has a long title: its age goes below
        when.setStyleSheet(f"color: {theme.DIM}; font-size: 13px;")
        top.addWidget(when)
        words.addLayout(top)
        name = QLabel(satellite or f"NORAD {obs.norad_cat_id}")
        name.setStyleSheet(f"color: {INK}; font-family: \"{theme.SANS_BOLD}\"; font-size: 17px; letter-spacing: 1px;")
        words.addWidget(name)
        detail = QLabel(f"{obs.transmitter_mode or '?'}  ·  {(obs.station_name or 'unknown station')[:28]}"
                        + (f"  ·  {ago}" if title and ago else ""))
        detail.setStyleSheet(f"color: {SOFT}; font-size: 13px;")
        words.addWidget(detail)
        words.addStretch(1)
        for label_ in (verdict, when, name, detail):
            label_.setAttribute(Qt.WA_TransparentForMouseEvents)
        for label_ in (verdict, name, detail):
            # long titles are cut at the card's edge instead of widening it, so every card stays on the grid
            label_.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        row.addLayout(words, 1)
        self.setStyleSheet(f"QPushButton#recording {{ background: {theme.PANEL}; border: 1px solid {theme.RULE}; "
                           f"border-left: 3px solid {colour}; border-radius: 0; padding: 0; text-align: left; }}"
                           f"QPushButton#recording:hover {{ background: {theme.RAISED}; border-color: {theme.SELECTED}; "
                           f"border-left-color: {colour}; }}")


class HomeScreen(StarryScreen):
    replay = Signal(int)
    ai = Signal()
    intro = Signal()
    refresh_requested = Signal()
    back = Signal()

    SHOWN = 8
    COLUMNS = 4

    def __init__(self, session: VettingSession, showcase: tuple = (), parent=None, names=None) -> None:
        super().__init__(parent)
        self._session = session
        self._showcase = tuple(showcase)
        self._names = names or (lambda obs: f"NORAD {obs.norad_cat_id}")
        self._thumbs: dict[int, object] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(60, 36, 60, 32)
        outer.setSpacing(10)
        top = QHBoxLayout()
        back = QPushButton("Back")
        back.setObjectName("quiet")
        back.clicked.connect(self.back.emit)
        top.addWidget(back)
        top.addWidget(label(heading("Live satellites"), "hero"), 1)
        for text, signal in (("How it works", self.intro), ("How good is the AI  →", self.ai)):
            button = QPushButton(text)
            button.setObjectName("quiet")
            button.clicked.connect(signal.emit)
            top.addWidget(button)
        outer.addLayout(top)
        outer.addWidget(label("The newest recordings from the SatNOGS network of volunteer ground stations, "
                              "checked as they arrive.", "lead"))
        outer.addSpacing(18)

        self._stats = self._grid()
        self._stat_labels = []
        for column in range(self.COLUMNS):
            card = QFrame()
            card.setObjectName("card")
            inside = QVBoxLayout(card)
            inside.setContentsMargins(22, 16, 22, 16)
            value, text = label("–", "stat"), label("", "body")
            inside.addWidget(value)
            inside.addWidget(text)
            card.setFixedHeight(132)
            self._stat_labels.append((value, text))
            self._stats.addWidget(card, 0, column)
        outer.addLayout(self._stats)
        outer.addSpacing(18)

        outer.addWidget(divider())
        outer.addSpacing(10)
        row = QHBoxLayout()
        row.addWidget(label(heading("Latest recordings"), "h2"), 1)
        self.feed_status = label("", "body", wrap=False)
        row.addWidget(self.feed_status)
        row.addSpacing(16)
        refresh = QPushButton("Fetch newest  →")
        refresh.setObjectName("quiet")
        refresh.clicked.connect(self.refresh_requested.emit)
        row.addWidget(refresh)
        outer.addLayout(row)
        self._latest = self._grid()
        outer.addLayout(self._latest)
        outer.addSpacing(18)
        outer.addWidget(divider())
        outer.addSpacing(10)
        outer.addWidget(label(heading("Explained by the physics"), "h2"))
        self._examples = self._grid()
        outer.addLayout(self._examples)
        outer.addStretch(1)
        self.refresh()

    def _grid(self) -> QGridLayout:
        """Every row of cards on one grid: equal columns, so boxes line up down the page."""
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(18)
        for column in range(self.COLUMNS):
            grid.setColumnStretch(column, 1)
        return grid

    def set_status(self, text: str) -> None:
        self.feed_status.setText(text[:90])

    def refresh(self) -> None:
        session = self._session
        showcase_ids = {o.id for o in self._showcase}
        latest = sorted((i for i in session.all_items if i.obs.id not in showcase_ids),
                        key=lambda i: i.obs.start.timestamp() if i.obs.start else 0, reverse=True)
        sure = sum(i.verdict is not Verdict.UNSURE for i in latest)
        unchecked = sum(not i.known for i in latest)
        error, measured = session.measured_error()
        numbers = [
            (f"{len(latest)}", f"latest recordings loaded · {unchecked} not checked by any volunteer"),
            (f"{sure / len(latest):.0%}" if latest else "–", "of them the AI is sure enough to check on its own"),
            (f"{1 - error:.0%}" if error == error else "–",
             f"right when it is sure, tested on {measured:,} answers volunteers had checked"),
            (f"{len({i.obs.norad_cat_id for i in latest})}", "different satellites among them"),
        ]
        for (value, text), (v, t) in zip(self._stat_labels, numbers):
            if value.text() != v:
                count_up(value, v)      # numbers roll up when they change
            text.setText(t)
        self._fill(self._latest, [self._card(i) for i in latest[:self.SHOWN]], columns=self.COLUMNS,
                   empty="Recordings appear here as they download.")
        titles = {15489000: "Not a satellite at all", 15490378: "The wrong satellite"}
        examples = [self._card(session.item(o.id), titles.get(o.id)) for o in self._showcase
                    if o.id in {i.obs.id for i in session.all_items}]
        self._fill(self._examples, examples, columns=self.COLUMNS, empty="")

    def _card(self, item: Item, title: str | None = None) -> RecordingCard:
        if item.obs.id not in self._thumbs:
            try:
                self._thumbs[item.obs.id] = waterfall_thumbnail(self._session.waterfall(item.obs.id))
            except Exception:  # noqa: BLE001 - a missing picture must not break the home screen
                self._thumbs[item.obs.id] = None
        try:
            name = self._names(item.obs)
        except Exception:  # noqa: BLE001
            name = f"NORAD {item.obs.norad_cat_id}"
        return RecordingCard(item, title, name, self._thumbs[item.obs.id])

    def _fill(self, layout, cards, columns: int = 0, empty: str = "") -> None:
        while layout.count():
            widget = layout.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        if not cards and empty:
            layout.addWidget(label(empty, "body"), 0, 0) if columns else layout.addWidget(label(empty, "body"))
            return
        for n, card in enumerate(cards):
            card.clicked.connect(lambda _=False, i=card.obs_id: self.replay.emit(i))
            if columns:
                layout.addWidget(card, n // columns, n % columns)
            else:
                layout.addWidget(card)
        if columns:
            # empty placeholders keep a short last row on the same columns as the rows above
            for n in range(len(cards), -(-len(cards) // columns) * columns):
                spacer = QWidget()
                spacer.setStyleSheet("background: transparent;")
                layout.addWidget(spacer, n // columns, n % columns)
        if not columns:
            layout.addStretch(1)


class LauncherScreen(StarryScreen):
    """The front door: the two halves of the project."""

    lab = Signal()
    live = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(60, 30, 60, 50)
        top = QHBoxLayout()
        words = QVBoxLayout()
        words.addStretch(1)
        words.addWidget(label(heading("Satellite Signal Analyser"), "hero"))
        words.addSpacing(8)
        words.addWidget(label("Identifying and recovering satellite signals with AI: first on a lab dataset, "
                              "then on a live network of real ground stations.", "lead"))
        words.addStretch(1)
        top.addLayout(words, 3)
        self.hero = OrbitHero()
        self.hero.setFixedHeight(270)
        top.addWidget(self.hero, 2)
        outer.addLayout(top)
        outer.addSpacing(24)
        doors = QHBoxLayout()
        doors.setSpacing(28)
        for title, text, details, signal, kind in (
            ("Lab analysis", "What kind of signal is it, and what message does it carry?",
             "RML24: 1.3 million recordings from a satellite test bench.  The AI names the signal type, the "
             "receiver cleans the signal and recovers the bits, step by step.", self.lab, "check"),
            ("Live satellites", "Did the station really hear the satellite, and if not, what did it hear?",
             "SatNOGS: real passes recorded minutes ago by volunteer stations worldwide.  The AI checks each one; "
             "orbit physics names the source of every signal.", self.live, "confirm"),
        ):
            card = QFrame()
            card.setObjectName("card")
            inside = QVBoxLayout(card)
            inside.setContentsMargins(26, 24, 26, 24)
            inside.addWidget(Step(kind))
            inside.addWidget(label(heading(title), "hero"))
            inside.addWidget(label(text, "h2"))
            inside.addWidget(label(details, "body"))
            inside.addStretch(1)
            go = QPushButton(f"Open {title.lower()}  →")
            go.setObjectName("big")
            go.clicked.connect(signal.emit)
            inside.addWidget(go)
            doors.addWidget(card)
        outer.addLayout(doors)
        outer.addStretch(1)


class ScrollingStack(QStackedWidget):
    """A stack whose screens scroll when the window is smaller than they are.

    Each screen sits in its own frameless scroll area, so the window fits a
    1366 x 768 laptop; callers still add, show and ask for the screens
    themselves, never the scroll areas.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._areas: dict[QWidget, QScrollArea] = {}

    def addWidget(self, screen: QWidget) -> int:  # noqa: N802 - Qt naming
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setStyleSheet("QScrollArea { background: transparent; border: 0; }")
        area.setWidget(screen)
        self._areas[screen] = area
        return super().addWidget(area)

    def setCurrentWidget(self, screen: QWidget) -> None:  # noqa: N802
        super().setCurrentWidget(self._areas.get(screen, screen))

    def _screen(self, widget):
        return widget.widget() if isinstance(widget, QScrollArea) else widget

    def currentWidget(self):  # noqa: N802
        return self._screen(super().currentWidget())

    def widget(self, index: int):
        return self._screen(super().widget(index))


class MissionWindow(QMainWindow):
    """The app: the launcher and the live half (how it works, home, replay, how good is the AI)."""

    def __init__(self, session: VettingSession, explainer=None, show_intro: bool = True, feed=None,
                 showcase: tuple = (), start: str = "launcher") -> None:
        super().__init__()
        self._session, self._explainer, self._feed = session, explainer, feed
        self._expert = None
        self._lab_process = None
        self.setWindowTitle("Satellite Signal Analyser")
        self.resize(1440, 880)
        self.setStyleSheet(STYLE)
        self.stack = ScrollingStack()
        self.setMinimumSize(960, 600)        # smaller screens scroll instead of pushing the window off screen
        self.launcher = LauncherScreen()
        self.intro = IntroScreen()
        names = (lambda obs: explainer.catalog.name(int(obs.norad_cat_id or 0))) if explainer is not None else None
        self.home = HomeScreen(session, showcase, names=names)
        self.replay = ReplayScreen(session, explainer)
        from .ai_view import ModelBoard

        bundle = getattr(session, "_bundle", None)
        self.ai = ModelBoard(in_use="cnn+forest" if bundle is not None and bundle.cnn is not None else "forest")
        for screen in (self.launcher, self.intro, self.home, self.replay, self.ai):
            self.stack.addWidget(screen)
        self.setCentralWidget(self.stack)
        self._seen_intro = not show_intro
        self.launcher.live.connect(self.open_live)
        self.launcher.lab.connect(self.open_lab)
        self.intro.start.connect(lambda: self.stack.setCurrentWidget(self.home))
        self.home.intro.connect(lambda: self.stack.setCurrentWidget(self.intro))
        self.home.back.connect(lambda: self.stack.setCurrentWidget(self.launcher))
        self.home.replay.connect(self.open_replay)
        self.home.ai.connect(lambda: self.stack.setCurrentWidget(self.ai))
        self.ai.back.connect(lambda: self.stack.setCurrentWidget(self.home))
        self.ai.open_recording.connect(self.open_any)
        self.home.refresh_requested.connect(self.fetch_newest)
        self._replay_from = self.home
        self.replay.back.connect(lambda: self.stack.setCurrentWidget(self._replay_from))

        # New recordings from the feed: score them, then redraw at most a few times a second.
        self._redraw = QTimer(self)
        self._redraw.setSingleShot(True)
        self._redraw.setInterval(400)
        self._redraw.timeout.connect(self._redraw_lists)
        if feed is not None:
            feed.arrived.connect(self._arrived)
            feed.status.connect(self.home.set_status)
            feed.finished.connect(lambda n: self.home.set_status(f"Up to date · {n} recordings checked just now"))

        self.stack.currentChanged.connect(self._fade)
        screens = {"launcher": self.launcher, "intro": self.intro, "home": self.home, "ai": self.ai}
        self.stack.setCurrentWidget(screens.get(start, self.launcher))

    # ------------------------------------------------------------------
    def _fade(self, index: int) -> None:
        widget = self.stack.widget(index)
        if widget in (self.launcher, self.intro, self.home, self.ai):
            fade_in(widget)

    def open_live(self) -> None:
        self.stack.setCurrentWidget(self.home if self._seen_intro else self.intro)
        self._seen_intro = True

    def open_lab(self) -> None:
        """The original Lab analyser (RML24), as its own window."""
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[3]
        if self._lab_process is None or self._lab_process.poll() is not None:
            self._lab_process = subprocess.Popen([sys.executable, "-m", "satrecovery.analysis", "--dark"],
                                                 cwd=str(root))

    def fetch_newest(self) -> None:
        if self._feed is not None:
            self._feed.start()

    def _arrived(self, obs) -> None:
        try:
            self._session.add([obs])
        except (OSError, ValueError) as error:
            self.home.set_status(f"Could not read {obs.id}: {error}")
            return
        self._redraw.start()

    def _redraw_lists(self) -> None:
        self.home.refresh()
        if self._expert is not None:
            self._expert.refresh_queue()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._feed is not None:
            self._feed.stop()
        if self._expert is not None:
            self._expert.close()
        super().closeEvent(event)

    def open_replay(self, obs_id: int) -> None:
        current = self.stack.currentWidget()
        self._replay_from = current if current in (self.home, self.ai) else self.home
        self.stack.setCurrentWidget(self.replay)
        self.replay.load(obs_id)

    def open_any(self, obs_id: int) -> bool:
        """Replay any saved recording, e.g. a mistake from the report card, even if it is not on the home screen."""
        known = {i.obs.id for i in self._session.all_items}
        if obs_id not in known:
            from .. import Manifest

            root = self._session.cache_root
            found = []
            for name in ("manifest.jsonl", "latest.jsonl", "unvetted.jsonl"):
                if (root / name).exists():
                    found = [o for o in Manifest(root / name) if o.id == obs_id]
                    if found:
                        break
            if not found:
                found = self._fetch_observation(obs_id)       # not saved here (e.g. a copy without the data)
            try:
                for obs in found:
                    self._session._cache.waterfall(obs)        # downloads it if it is not on this computer
                added = self._session.add(found) if found else []
            except Exception:  # noqa: BLE001 - no network, or SatNOGS no longer has it
                added = []
            if not added:
                self.ai.status.setText(f"Recording #{obs_id} cannot be replayed here: its waterfall is not on this "
                                       "computer and could not be downloaded.")
                return False
        self.open_replay(obs_id)
        return True

    @staticmethod
    def _fetch_observation(obs_id: int) -> list:
        """One observation's details from the SatNOGS API, or nothing."""
        from ..frames import OBSERVATION_URL
        from ..observation import Observation
        from ..source import UrllibTransport

        try:
            response = UrllibTransport().get(OBSERVATION_URL.format(obs_id), timeout=15)
            return [Observation.from_api(response.json())] if response.status == 200 else []
        except Exception:  # noqa: BLE001
            return []

    def open_expert(self, obs_id: int | None) -> None:
        from .window import VettingWindow

        if self._expert is None:
            self._expert = VettingWindow(self._session, self._explainer)
        self._expert.show()
        self._expert.raise_()
        if obs_id is not None:
            self._expert.select(obs_id)
