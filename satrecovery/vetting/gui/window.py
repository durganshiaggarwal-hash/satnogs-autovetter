"""The vetting station window.

    ┌ top bar: automatable share · error when confident · to check · auto · you ┐
    │ queue          │ waterfall (+ centre band, + track)  │ observation        │
    │ (unsure first) │                       │ band trace  │ P(signal) meter    │
    │                │ frequency profile     │             │ reasons · keys     │
    └ G signal · B no signal · A accept · J/K next/previous · T track · V answer ┘

Keyboard first: a person should be able to vet a hundred observations
without touching the mouse.  After each decision the next undecided row
is selected.

DESIGN NOTE: OBSERVER (Qt signals)
----------------------------------
The window does not poll.  The queue's selection model *signals* a new
row; the window answers by showing it.  The session never knows a window
exists.
"""

from __future__ import annotations

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ..observation import WaterfallLabel
from ..session import Policy, Verdict, VettingSession
from . import theme
from .orbit_view import OrbitPanel
from .widgets import Meter, ProbabilityDelegate, QueueModel, Reasons, Scope, caps, rule, verdict_text

SATNOGS_OBSERVATION = "https://network.satnogs.org/observations/{}/"


class Counter(QWidget):
    def __init__(self, title: str, colour: str = theme.TEXT) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(0)
        self.value = QLabel("–")
        self.value.setObjectName("counter")
        self.value.setStyleSheet(f"color: {colour}")
        layout.addWidget(caps(title))
        layout.addWidget(self.value)


class VettingWindow(QMainWindow):
    def __init__(self, session: VettingSession, explainer=None) -> None:
        super().__init__()
        self._session = session
        self._explainer = explainer
        self._current: int | None = None
        self.setWindowTitle("SatNOGS vetting station")
        self.resize(1440, 860)
        self.setStyleSheet(theme.STYLE)

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._top_bar())
        outer.addWidget(rule())
        self.banner = QLabel()
        self.banner.setObjectName("mono")
        self.banner.setContentsMargins(16, 6, 16, 6)
        self.banner.setTextFormat(Qt.RichText)
        outer.addWidget(self.banner)
        outer.addWidget(rule())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.addWidget(self._queue_panel())
        self.scope = Scope()
        middle = QSplitter(Qt.Vertical)
        middle.setHandleWidth(1)
        middle.addWidget(self.scope)
        self.orbit = OrbitPanel(explainer, self.scope)
        middle.addWidget(self.orbit)
        middle.setSizes([600, 240])
        splitter.addWidget(middle)
        splitter.addWidget(self._detail_panel())
        splitter.setSizes([440, 680, 320])
        outer.addWidget(splitter, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage(
            "G signal   B no signal   A accept suggestion   J/K next/previous   N next unsure   "
            "T track   O orbit curves   V volunteer's answer   hover the waterfall to move the sky chart")

        self._keys()
        self._update_mode()
        self._refresh_counts()
        self.table.selectRow(0)
        # Start in the queue: otherwise the first focusable widget (a threshold
        # box) takes focus and swallows the first G or B typed.
        for widget in (self.low, self.high, self.order):
            widget.setFocusPolicy(Qt.ClickFocus)
        for button in self.findChildren(QPushButton):
            button.setFocusPolicy(Qt.NoFocus)
        self.reveal.setFocusPolicy(Qt.NoFocus)
        self.table.setFocus()

    # ------------------------------------------------------------------
    def _top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("panel")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 10, 16, 10)
        titles = QVBoxLayout()
        title = QLabel("SATNOGS · VETTING STATION")
        title.setObjectName("title")
        sub = QLabel(self._session.model_name)
        sub.setObjectName("sub")
        titles.addWidget(title)
        titles.addWidget(sub)
        layout.addLayout(titles)
        layout.addSpacing(24)

        # Which queue: the real work, or the test set that measures the model.
        self.mode_buttons = {}
        for mode in self._session.modes:
            button = QPushButton()
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, m=mode: self._set_mode(m))
            self.mode_buttons[mode] = button
            layout.addWidget(button)
        layout.addStretch(1)

        self.c_auto_share = Counter("automatable", theme.ACCENT)
        self.c_error = Counter("error when sure (test set)")
        self.c_check = Counter("to check", theme.UNSURE)
        self.c_auto = Counter("auto-vetted")
        self.c_you = Counter("by you")
        for counter in (self.c_auto_share, self.c_error, self.c_check, self.c_auto, self.c_you):
            layout.addWidget(counter)

        layout.addSpacing(12)
        policy = QGridLayout()
        policy.setVerticalSpacing(2)
        self.low = QDoubleSpinBox()
        self.high = QDoubleSpinBox()
        for box, value in ((self.low, self._session.policy.low), (self.high, self._session.policy.high)):
            box.setRange(0.0, 1.0)
            box.setSingleStep(0.05)
            box.setDecimals(2)
            box.setValue(value)
            box.valueChanged.connect(self._policy_changed)
        policy.addWidget(caps("no signal below"), 0, 0)
        policy.addWidget(caps("signal above"), 0, 1)
        policy.addWidget(self.low, 1, 0)
        policy.addWidget(self.high, 1, 1)
        layout.addLayout(policy)
        layout.addSpacing(12)
        self.auto_button = QPushButton("AUTO-VET CONFIDENT")
        self.auto_button.setObjectName("primary")
        self.auto_button.clicked.connect(self._auto_vet)
        layout.addWidget(self.auto_button)
        return bar

    def _queue_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        row = QHBoxLayout()
        row.addWidget(caps("queue"))
        row.addStretch(1)
        self.order = QComboBox()
        self.order.addItems(VettingSession.ORDERS)
        self.order.currentTextChanged.connect(self._order_changed)
        row.addWidget(self.order)
        layout.addLayout(row)

        self.model = QueueModel(self._session)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setItemDelegateForColumn(3, ProbabilityDelegate(self._session, self.table))
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setShowGrid(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        for column, width in enumerate((72, 60, 84, 0, 52)):
            if width:
                header.resizeSection(column, width)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setMinimumSectionSize(40)
        self.table.selectionModel().currentRowChanged.connect(lambda current, _: self._show_row(current.row()))
        layout.addWidget(self.table, 1)
        return panel

    def _detail_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        layout.addWidget(caps("observation"))
        self.obs_title = QLabel("–")
        self.obs_title.setObjectName("big")
        layout.addWidget(self.obs_title)
        self.meta = QLabel()
        self.meta.setObjectName("mono")
        self.meta.setTextFormat(Qt.RichText)
        layout.addWidget(self.meta)
        layout.addWidget(rule())

        layout.addWidget(caps("model"))
        self.verdict = QLabel()
        self.verdict.setObjectName("counter")
        layout.addWidget(self.verdict)
        self.meter = Meter()
        layout.addWidget(self.meter)
        layout.addWidget(caps("why"))
        self.reasons = Reasons()
        layout.addWidget(self.reasons)
        layout.addWidget(rule())

        layout.addWidget(caps("your call"))
        buttons = QHBoxLayout()
        self.b_signal = QPushButton("[G] SIGNAL")
        self.b_signal.setObjectName("signal")
        self.b_none = QPushButton("[B] NO SIGNAL")
        self.b_none.setObjectName("nosignal")
        self.b_signal.clicked.connect(lambda: self._decide(WaterfallLabel.SIGNAL))
        self.b_none.clicked.connect(lambda: self._decide(WaterfallLabel.NO_SIGNAL))
        buttons.addWidget(self.b_signal)
        buttons.addWidget(self.b_none)
        layout.addLayout(buttons)
        self.decision = QLabel()
        self.decision.setObjectName("mono")
        layout.addWidget(self.decision)

        self.reveal = QCheckBox("Show the volunteer's answer before deciding (V)")
        self.reveal.toggled.connect(lambda _: self._show_answer())
        layout.addWidget(self.reveal)
        self.answer = QLabel()
        self.answer.setObjectName("mono")
        layout.addWidget(self.answer)
        layout.addStretch(1)
        link = QPushButton("OPEN ON SATNOGS ↗")
        link.clicked.connect(self._open_satnogs)
        layout.addWidget(link)
        return panel

    def _keys(self) -> None:
        bindings = {
            "G": lambda: self._decide(WaterfallLabel.SIGNAL),
            "B": lambda: self._decide(WaterfallLabel.NO_SIGNAL),
            "A": self._accept,
            "J": lambda: self._step(1),
            "K": lambda: self._step(-1),
            "N": self._next_unsure,
            "T": self._toggle_track,
            "V": lambda: self.reveal.toggle(),
            "O": self._toggle_curves,
        }
        self._shortcuts = []
        for key, action in bindings.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(action)
            self._shortcuts.append(shortcut)

    # ------------------------------------------------------------------
    def _show_row(self, row: int) -> None:
        item = self.model.item_at(row)
        if item is None:
            return
        self._current = item.obs.id
        obs = item.obs
        self.obs_title.setText(f"#{obs.id}")
        duration = f"{obs.duration_s / 60:.1f} min" if obs.duration_s else "?"
        rows = [
            ("satellite", f"NORAD {obs.norad_cat_id}"),
            ("mode", f"{obs.transmitter_mode or '?'}" + (f" · {obs.transmitter_baud:,.0f} Bd" if obs.transmitter_baud else "")),
            ("station", f"{obs.ground_station} · {obs.station_name[:22]}"),
            ("pass", f"{duration} · max elev {obs.max_altitude_deg:.0f}°" if obs.max_altitude_deg else duration),
            ("frames decoded", str(obs.n_decoded_frames)),
            ("SatNOGS rating", self._rating(obs.status)),
        ]
        self.meta.setText("<table cellspacing='0' cellpadding='1'>" + "".join(
            f"<tr><td style='color:{theme.DIM}; padding-right:10px'>{k}</td><td>{v}</td></tr>" for k, v in rows)
            + "</table>")
        colour = theme.verdict_colour(item.verdict.value)
        self.verdict.setText(f"<span style='color:{colour}'>{verdict_text(item.verdict)}</span>"
                             f"&nbsp;&nbsp;<span style='color:{theme.DIM}'>p = {item.probability:.2f}</span>")
        self.meter.set_value(item.probability, self._session.policy.low, self._session.policy.high, item.verdict.value)
        self.reasons.set_reasons(self._session.reasons(obs.id))
        try:
            power = self._session.waterfall(obs.id)
            # The kHz scale, if already known (no network here: the orbit worker fetches it).
            axes = self._session.cache.axes(obs, download=False)
            self.scope.show(power, obs.duration_s, axes)
            self.orbit.show_observation(obs, power)
        except Exception as error:  # a damaged cache file should not close the window
            self.statusBar().showMessage(f"Could not draw {obs.id}: {error}", 6000)
        self._show_decision()
        self._show_answer()

    @staticmethod
    def _rating(status: str) -> str:
        """SatNOGS's own automatic rating, which is not a look at the waterfall."""
        note = {
            "good": "automatic: data frames were decoded",
            "bad": "automatic",
            "failed": "automatic: the station sent no results",
            "unknown": "not rated",
        }.get(status, "")
        return f"{status}" + (f" <span style='color:{theme.DIM}'>({note})</span>" if note else "")

    def _set_mode(self, mode: str) -> None:
        self._session.set_mode(mode)
        self._update_mode()
        self.model.set_order(self.order.currentText())
        self._refresh_counts()
        if self.model.rowCount():
            self.table.selectRow(0)
        self.table.setFocus()

    def _update_mode(self) -> None:
        for mode, button in self.mode_buttons.items():
            title = {"unvetted": "UNVETTED", "test": "TEST SET", "latest": "LATEST"}[mode]
            button.setText(f"{title} · {self._session.count(mode)}")
            button.setChecked(mode == self._session.mode)
            button.setObjectName("primary" if mode == self._session.mode else "")
            button.style().unpolish(button)
            button.style().polish(button)
        if self._session.mode == "latest":
            self.banner.setText(
                f"<span style='color:{theme.ACCENT}'>LIVE</span>&nbsp;&nbsp;The newest recordings from SatNOGS, "
                f"most of them not checked by any volunteer yet.  Your decisions are saved on this computer only; "
                f"nothing is sent to SatNOGS.")
        elif self._session.mode == "unvetted":
            self.banner.setText(
                f"<span style='color:{theme.ACCENT}'>REAL WORK</span>&nbsp;&nbsp;No volunteer has checked these on "
                f"SatNOGS, so the verdicts here are new information.  Your decisions are saved on this computer only; "
                f"nothing is sent to SatNOGS.")
        else:
            self.banner.setText(
                f"<span style='color:{theme.UNSURE}'>TEST SET</span>&nbsp;&nbsp;These already have a volunteer's answer "
                f"on SatNOGS.  Each is scored by a model that never saw its station, to measure how often it is right.")
        self.reveal.setVisible(self._session.mode == "test")

    def _show_decision(self) -> None:
        decision = self._session.decision(self._current) if self._current else None
        if decision is None:
            self.decision.setText(f"<span style='color:{theme.FAINT}'>not decided yet</span>")
            return
        colour = theme.SIGNAL if decision.label is WaterfallLabel.SIGNAL else theme.NO_SIGNAL
        text = "signal" if decision.label is WaterfallLabel.SIGNAL else "no signal"
        self.decision.setText(f"decided: <span style='color:{colour}'>{text}</span> "
                              f"<span style='color:{theme.DIM}'>({decision.by})</span>")

    def _show_answer(self) -> None:
        if self._current is None:
            return
        obs = self._session.item(self._current).obs
        decided = self._session.decision(obs.id) is not None
        if obs.waterfall_label is WaterfallLabel.UNKNOWN:
            self.answer.setText(f"<span style='color:{theme.FAINT}'>no volunteer has checked this on SatNOGS yet: "
                                f"your call is the first</span>")
        elif self.reveal.isChecked() or decided:
            colour = theme.SIGNAL if obs.waterfall_label is WaterfallLabel.SIGNAL else theme.NO_SIGNAL
            self.answer.setText(f"volunteer said: <span style='color:{colour}'>{obs.waterfall_label.value}</span>")
        else:
            self.answer.setText(f"<span style='color:{theme.FAINT}'>volunteer's answer hidden until you decide</span>")

    def _refresh_counts(self) -> None:
        stats = self._session.stats()
        self.c_auto_share.value.setText(f"{stats['automatable_share']:.0%}")
        error = stats["confident_error"]
        self.c_error.value.setText("–" if error != error else f"{error:.1%}")
        self.c_check.value.setText(str(stats["to_check"]))
        self.c_auto.value.setText(str(stats["auto"]))
        self.c_you.value.setText(str(stats["yours"]))
        remaining = stats["confident"] - stats["auto"]
        self.auto_button.setText(f"AUTO-VET CONFIDENT ({max(remaining, 0)})")

    # ------------------------------------------------------------------
    def _decide(self, label: WaterfallLabel) -> None:
        if self._current is None:
            return
        self._session.decide(self._current, label)
        self.model.refresh()
        self._refresh_counts()
        self._show_decision()
        self._show_answer()
        self._next_undecided()

    def _accept(self) -> None:
        if self._current is None:
            return
        label = self._session.item(self._current).verdict.label
        if label is None:
            self.statusBar().showMessage("The model is unsure about this one: press G or B.", 4000)
            return
        self._decide(label)

    def _step(self, delta: int) -> None:
        row = max(0, min(self.table.currentIndex().row() + delta, self.model.rowCount() - 1))
        self.table.selectRow(row)

    def _next_where(self, test) -> None:
        start = self.table.currentIndex().row()
        for offset in range(1, self.model.rowCount() + 1):
            row = (start + offset) % self.model.rowCount()
            item = self.model.item_at(row)
            if item is not None and test(item):
                self.table.selectRow(row)
                return
        self.statusBar().showMessage("Nothing left in this list.", 4000)

    def _next_undecided(self) -> None:
        self._next_where(lambda item: self._session.decision(item.obs.id) is None)

    def _next_unsure(self) -> None:
        self._next_where(lambda item: item.verdict is Verdict.UNSURE and self._session.decision(item.obs.id) is None)

    def _toggle_curves(self) -> None:
        self.scope.show_curves = not self.scope.show_curves
        self._show_row(self.table.currentIndex().row())

    def _toggle_track(self) -> None:
        self.scope.show_track = not self.scope.show_track
        self._show_row(self.table.currentIndex().row())

    def _auto_vet(self) -> None:
        done = self._session.auto_vet()
        self.model.refresh()
        self._refresh_counts()
        self._show_decision()
        self.statusBar().showMessage(f"Auto-vetted {done} confident observations. The unsure ones are yours.", 6000)

    def _policy_changed(self) -> None:
        low, high = self.low.value(), self.high.value()
        if low >= high:
            return
        current = self._current
        self._session.set_policy(Policy(low, high))
        self.model.refresh()
        self._refresh_counts()
        if current is not None:
            self._show_row(self.model.row_of(current))
        self.table.setFocus()  # back to vetting keys

    def _order_changed(self, order: str) -> None:
        self.model.set_order(order)
        self.table.selectRow(0)
        self.table.setFocus()

    def _open_satnogs(self) -> None:
        if self._current is not None:
            QDesktopServices.openUrl(QUrl(SATNOGS_OBSERVATION.format(self._current)))

    def refresh_queue(self) -> None:
        """New recordings arrived: rebuild the list, keeping the current selection."""
        current = self._current
        self.model.set_order(self.order.currentText())
        self._update_mode()
        self._refresh_counts()
        row = self.model.row_of(current) if current is not None else 0
        if row >= 0:
            self.table.blockSignals(True)
            self.table.selectRow(row)
            self.table.blockSignals(False)

    def select(self, obs_id: int) -> None:
        """Open a given observation, switching mode if needed."""
        item = self._session.item(obs_id)
        mode = "test" if item.known else "unvetted"
        if mode != self._session.mode:
            self._set_mode(mode)
        row = self.model.row_of(obs_id)
        if row >= 0:
            self.table.selectRow(row)
