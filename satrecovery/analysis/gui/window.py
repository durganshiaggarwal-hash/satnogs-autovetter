"""The analysis app's window: a header, a row of tabs, toasts.

Built from the Phase 8 parts -- the logo, the animated tabs, the loading
bar, the toasts, the theme -- imported, not copied.

The header holds the app's name and the things that apply everywhere:
the Dark and Animations switches and the "?" guide.  The tabs sit in
their own row beneath it.

No dataset is needed: the AI was trained on RML24 once, and every tab
works on signals the user gives it (simulated, or from a file).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from ...gui import theme
from ...gui.motion import AnimatedTabs, LoadingBar, Motion, PillButton, ToastHost, ToggleSwitch
from ...gui.pages.base import Page
from ...gui.window import Logo, StatusDot
from ..preferences import Preferences
from .batch import BatchPage
from .client import AnalysisClient
from .help import HelpDialog
from .history import HistoryPage
from .pages import AnalysisPage, FilePage, SimulatedSignalPage
from .compare import ComparePage
from .report import ReportPage


class AnalysisWindow(QMainWindow):
    """Satellite Signal Analyser."""

    #: The Dark switch was flipped: the app rebuilds the window in the new colours.
    theme_changed = Signal(bool)

    def __init__(self, client: AnalysisClient, preferences: Preferences | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Satellite Signal Analyser")
        self.resize(1440, 920)
        self.setMinimumSize(1180, 760)
        self._busy: set[Page] = set()
        self._client = client
        self.preferences = preferences or Preferences()

        self.simulated = SimulatedSignalPage(client)
        self.file = FilePage(client)
        self.history = HistoryPage(client)
        self.batch = BatchPage(client)
        self.report = ReportPage(client)
        self.compare = ComparePage(client)
        self.pages: list[Page] = [self.simulated, self.file, self.history, self.batch, self.report, self.compare]

        self.tabs = AnimatedTabs()
        for page in self.pages:
            self.tabs.addTab(page, page.title)
            page.message.connect(self._show_message)
            page.notify.connect(lambda text, kind: self.toasts.show(text, kind))
            page.tasks.busy_changed.connect(lambda busy, p=page: self._set_busy(p, busy))
        self.tabs.currentChanged.connect(lambda index: self.pages[index].ensure_loaded())

        # -- header: name on the left, app-wide controls on the right ------
        header = QFrame()
        header.setObjectName("Header")
        header.setStyleSheet(f"QFrame#Header {{ background: {theme.SURFACE}; border-bottom: 1px solid {theme.LINE}; }}")
        name = QLabel("Satellite Signal Analyser")
        name.setStyleSheet(f"font-family: '{theme.SANS_EXTRABOLD}'; font-size: 12.5pt;")
        tagline = QLabel("The AI names the signal; the receiver reads the message")
        tagline.setObjectName("Muted")
        text = QVBoxLayout()
        text.setSpacing(0)
        text.addWidget(name)
        text.addWidget(tagline)
        self.dark_switch = ToggleSwitch("Dark")
        self.dark_switch.setChecked(theme.DARK)
        self.dark_switch.toggled.connect(self._dark_toggled)
        self.motion_switch = ToggleSwitch("Animations")
        self.motion_switch.setChecked(Motion.enabled)
        self.motion_switch.toggled.connect(lambda on: setattr(Motion, "enabled", bool(on)))
        self.help_button = PillButton("?", "secondary")
        self.help_button.setFixedWidth(42)
        self.help_button.setToolTip("How to use the app, and what the words mean")
        self.help_button.clicked.connect(self.show_help)
        self._dot = StatusDot()
        self._dot.setToolTip(f"Backend at {client.base_url}")
        connected = QLabel("Connected")
        connected.setObjectName("Muted")
        row = QHBoxLayout()
        row.setContentsMargins(24, 12, 24, 6)
        row.setSpacing(14)
        row.addWidget(Logo())
        row.addLayout(text)
        row.addStretch(1)
        row.addWidget(self.dark_switch)
        row.addWidget(self.motion_switch)
        row.addWidget(self.help_button)
        row.addSpacing(6)
        row.addWidget(self._dot)
        row.addWidget(connected)
        tabs_row = QHBoxLayout()
        tabs_row.setContentsMargins(20, 0, 24, 8)
        tabs_row.addWidget(self.tabs.bar)
        tabs_row.addStretch(1)
        header_stack = QVBoxLayout(header)
        header_stack.setContentsMargins(0, 0, 0, 0)
        header_stack.setSpacing(0)
        header_stack.addLayout(row)
        header_stack.addLayout(tabs_row)

        self.loading = LoadingBar()
        shell = QWidget()
        column = QVBoxLayout(shell)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(header)
        column.addWidget(self.loading)
        column.addWidget(self.tabs, 1)
        self.setCentralWidget(shell)
        self.toasts = ToastHost(self)
        self.statusBar().setSizeGripEnabled(False)

        self.report.open_signal.connect(self._open_mistake)
        self.history.open_entry.connect(self.open_history_entry)
        self.simulated.ensure_loaded()

    # -- history ------------------------------------------------------------------
    def open_history_entry(self, entry: int) -> None:
        self.history.tasks.run(lambda: self._client.history_entry(entry), self._show_saved,
                               self.history.report_error)

    def page_for(self, result: dict) -> AnalysisPage:
        """The tab an analysis belongs to, by where its signal came from."""
        source = result.get("input", {})
        if source.get("file"):
            return self.file                        # an RML24 recording that arrived in a file
        kind = source.get("kind")
        # Older analyses of RML24 recordings reopen in the file tab: the dataset is no longer needed.
        return {"simulated": self.simulated, "file": self.file}.get(kind, self.file)

    def _show_saved(self, result: dict) -> None:
        page = self.page_for(result)
        page.ensure_loaded()
        self.tabs.setCurrentWidget(page)
        page.show_saved(result)

    def _open_mistake(self, signal: dict) -> None:
        """A mistake from the report card: the very same simulated signal, opened in the Simulated tab."""
        self.simulated.ensure_loaded()
        self.tabs.setCurrentWidget(self.simulated)
        self.simulated.open_simulated(signal["true"], signal["snr_label_db"], signal.get("symbol_rate", 250_000.0),
                                      signal.get("seed"))

    # -- help and dark mode ---------------------------------------------------------
    def show_help(self) -> HelpDialog:
        dialog = HelpDialog(self)
        dialog.open()
        return dialog

    def _dark_toggled(self, on: bool) -> None:
        if self._busy:
            # Rebuilding the window mid-analysis would lose the answer.
            self.dark_switch.blockSignals(True)
            self.dark_switch.setChecked(not on)
            self.dark_switch.blockSignals(False)
            self.toasts.show("Wait for the current analysis to finish, then switch.", "info")
            return
        self.preferences.dark_mode = on
        self.theme_changed.emit(bool(on))

    # -- shared -----------------------------------------------------------------------
    def _set_busy(self, page: Page, busy: bool) -> None:
        (self._busy.add if busy else self._busy.discard)(page)
        self.loading.set_busy(bool(self._busy))

    def _show_message(self, text: str, is_error: bool) -> None:
        self.statusBar().setStyleSheet(f"color: {theme.CRITICAL};" if is_error else "")
        self.statusBar().showMessage(text, 12_000 if is_error else 8_000)
        if is_error:
            self.toasts.show(text, "error")
            self._dot.set_ok("cannot reach" not in text)


__all__ = ["AnalysisWindow"]
