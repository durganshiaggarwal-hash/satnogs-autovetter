"""The main window: a header, four sliding tabs, and toasts.

The window knows almost nothing about what the pages do.  It lays them
out, loads each the first time it is shown, routes the few messages that
cross between pages, and shows status -- quietly in the status bar for
routine news, as a toast for anything worth interrupting for.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from . import theme
from .client import BackendClient
from .motion import AnimatedTabs, LoadingBar, Motion, ToastHost, ToggleSwitch, brand_gradient
from .pages import BenchmarkPage, ExperimentPage, ModelsPage, Page, StagesPage


class Logo(QWidget):
    """A dish sending out signal arcs, on the palette's soft gradient."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(40, 40)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        painter.setPen(Qt.NoPen)
        painter.setBrush(brand_gradient(rect))
        painter.drawRoundedRect(rect, 0 if theme.DARK else 12, 0 if theme.DARK else 12)
        ink = QColor(theme.BRAND_DEEP)
        painter.setPen(QPen(ink, 2.2, Qt.SolidLine, Qt.RoundCap))
        painter.setBrush(Qt.NoBrush)
        centre = QPointF(13, 27)
        for radius in (7, 12, 17):
            painter.drawArc(QRectF(centre.x() - radius, centre.y() - radius, 2 * radius, 2 * radius),
                            0 * 16, 90 * 16)
        painter.setPen(Qt.NoPen)
        painter.setBrush(ink)
        painter.drawEllipse(centre, 3, 3)


class StatusDot(QWidget):
    """A small dot: green when connected, red when the backend went away."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(10, 10)
        self._colour = QColor(theme.GOOD)

    def set_ok(self, ok: bool) -> None:
        self._colour = QColor(theme.GOOD if ok else theme.CRITICAL)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._colour)
        painter.drawEllipse(1, 1, 8, 8)


class MainWindow(QMainWindow):
    """Satellite Signal Recovery."""

    def __init__(self, client: BackendClient, embedded: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Satellite Signal Recovery")
        self.resize(1440, 920)
        self.setMinimumSize(1180, 760)
        self._client = client
        self._busy: set[Page] = set()

        self.experiment = ExperimentPage(client)
        self.stages = StagesPage(client)
        self.benchmark = BenchmarkPage(client)
        self.models = ModelsPage(client)
        self.pages: list[Page] = [self.experiment, self.stages, self.benchmark, self.models]

        self.tabs = AnimatedTabs()
        for page in self.pages:                       # polymorphism: no page is special
            self.tabs.addTab(page, page.title)
            page.message.connect(self._show_message)
            page.notify.connect(self._toast)
            page.tasks.busy_changed.connect(lambda busy, p=page: self._set_busy(p, busy))

        # -- header -----------------------------------------------------
        header = QFrame()
        header.setObjectName("Header")
        header.setStyleSheet(
            f"QFrame#Header {{ background: {theme.SURFACE}; border-bottom: 1px solid {theme.LINE}; }}"
        )
        name = QLabel("Satellite Signal Recovery")
        name.setStyleSheet(f"font-family: '{theme.SANS_EXTRABOLD}'; font-size: 12.5pt;")
        tagline = QLabel("Classical, AI and hybrid receivers, compared honestly")
        tagline.setObjectName("Muted")
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        brand_text.addWidget(name)
        brand_text.addWidget(tagline)

        self.motion_switch = ToggleSwitch("Animations")
        self.motion_switch.setChecked(Motion.enabled)
        self.motion_switch.setToolTip("Turn every animation off for a still interface.")
        self.motion_switch.toggled.connect(self._set_motion)

        self._dot = StatusDot()
        origin = "started by this window" if embedded else "already running"
        self._dot.setToolTip(f"Backend at {client.base_url} ({origin})")
        connected = QLabel("Connected")
        connected.setObjectName("Muted")
        connected.setToolTip(self._dot.toolTip())

        row = QHBoxLayout(header)
        row.setContentsMargins(24, 12, 24, 12)
        row.setSpacing(14)
        row.addWidget(Logo())
        row.addLayout(brand_text)
        row.addSpacing(28)
        row.addWidget(self.tabs.bar)
        row.addStretch(1)
        row.addWidget(self.motion_switch)
        row.addSpacing(10)
        row.addWidget(self._dot)
        row.addWidget(connected)

        self.loading = LoadingBar()

        shell = QWidget()
        shell.setObjectName("Shell")
        column = QVBoxLayout(shell)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(header)
        column.addWidget(self.loading)
        column.addWidget(self.tabs, 1)
        self.setCentralWidget(shell)

        self.toasts = ToastHost(self)

        # The few messages that cross between pages.
        self.experiment.follow_requested.connect(self._follow)
        self.models.models_changed.connect(self._models_changed)
        self.tabs.currentChanged.connect(lambda index: self.pages[index].ensure_loaded())

        self._connection = QLabel(f"Backend at {client.base_url} ({origin})")
        self.statusBar().addPermanentWidget(self._connection)
        self.statusBar().setSizeGripEnabled(False)

        QTimer.singleShot(0, self.experiment.ensure_loaded)

    # ------------------------------------------------------------------
    def _set_motion(self, on: bool) -> None:
        Motion.enabled = bool(on)
        self._toast("Animations on." if on else "Animations off.", "info")

    def _set_busy(self, page: Page, busy: bool) -> None:
        (self._busy.add if busy else self._busy.discard)(page)
        self.loading.set_busy(bool(self._busy))

    def _toast(self, text: str, kind: str = "info") -> None:
        self.toasts.show(text, kind)

    def _show_message(self, text: str, is_error: bool) -> None:
        if is_error:
            self.statusBar().setStyleSheet(f"color: {theme.CRITICAL};")
            self.statusBar().showMessage(text, 12_000)
            self._toast(text, "error")
            if "cannot reach" in text:
                self._dot.set_ok(False)
        else:
            self.statusBar().setStyleSheet("")
            self.statusBar().showMessage(text, 8_000)
            self._dot.set_ok(True)

    def _follow(self, request: dict) -> None:
        self.stages.ensure_loaded()
        self.tabs.setCurrentWidget(self.stages)
        self.stages.follow(request)

    def _models_changed(self) -> None:
        """A new model changes which receivers exist: tell every page that has loaded."""
        for page in self.pages:
            if page is not self.models and page.loaded:
                page.refresh()
        if self.experiment.loaded:
            self.experiment.refresh_receivers()


__all__ = ["MainWindow"]
