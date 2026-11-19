"""The main screen: set up a channel, and watch each receiver cope.

Every receiver is run on the *same* frame -- same bits, same noise --
so whatever differs between their plots is the receivers' doing.  With
"Keep this signal" on, moving a slider changes only that one condition;
turn it off and every run draws a fresh frame.

The layout reads top to bottom as the story does: the wave that
arrived, what each receiver recovered from it, then the score.  The
full table of numbers is folded away until someone asks for it.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..client import BackendError
from ..motion import Collapsible, PillButton, ToggleSwitch
from ..widgets import (
    ConditionsPanel,
    ConstellationPlot,
    ReceiverPicker,
    ResultsTable,
    ScoreCard,
    WaveformPlot,
    card,
)
from ..widgets.tables import percent
from .base import Page


class ExperimentPage(Page):
    """Channel controls on the left; what each receiver made of it on the right."""

    #: Ask the main window to open the step-by-step view on this frame.
    follow_requested = Signal(dict)

    MAX_RECEIVERS = 5

    def __init__(self, client, parent: QWidget | None = None) -> None:
        super().__init__(client, parent)
        self._seed: int | None = None
        self._want_new = False           # "New signal" pressed, not yet sent
        self._in_flight = False
        self._dirty = False
        self._last: dict | None = None

        # -- left: the controls -----------------------------------------
        self.conditions = ConditionsPanel()
        self.receivers = ReceiverPicker()

        actions, actions_layout = card()
        self.run_button = PillButton("Recover this signal", "primary")
        self.new_button = PillButton("New signal", "secondary")
        self.new_button.setToolTip("Draw new random bits and new noise.")
        self.keep = ToggleSwitch("Keep this signal")
        self.keep.setChecked(True)
        self.keep.setToolTip(
            "Same bits and same noise each time, so a change in the results comes from "
            "the setting you moved and nothing else."
        )
        self.live = ToggleSwitch("Update as I drag")
        self.live.setChecked(True)
        self.seed_label = QLabel("")
        self.seed_label.setObjectName("Muted")
        actions_layout.addWidget(self.run_button)
        actions_layout.addWidget(self.new_button)
        actions_layout.addSpacing(2)
        actions_layout.addWidget(self.keep)
        actions_layout.addWidget(self.live)
        actions_layout.addWidget(self.seed_label)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 12, 0)
        controls_layout.setSpacing(theme.GAP)
        controls_layout.addWidget(actions)
        controls_layout.addWidget(self.conditions)
        controls_layout.addWidget(self.receivers)
        controls_layout.addStretch(1)

        # -- right: what happened ---------------------------------------
        self.wave = WaveformPlot(
            "The wave", "Dashed is what the satellite sent; solid is what reached the dish. "
            "The first 24 symbols.",
        )
        self.wave.setMinimumHeight(220)
        self.arrived = ConstellationPlot("What arrived", "Every sample, before any processing.")
        self._recovered: list[ConstellationPlot] = []
        self._plots = QGridLayout()
        self._plots.setSpacing(theme.GAP)
        self._plots.addWidget(self.arrived, 0, 0)

        self._scores: list[ScoreCard] = []
        self._score_row = QHBoxLayout()
        self._score_row.setSpacing(theme.GAP)
        self.summary = QLabel("Set up a channel on the left. Results appear here as you change it.")
        self.summary.setObjectName("Hint")
        self.summary.setWordWrap(True)
        self.follow_button = PillButton("Follow this signal step by step", "secondary")
        self.follow_button.setEnabled(False)
        self.table = ResultsTable()
        self.details = Collapsible("All the numbers", self.table)

        results, results_layout = card()
        top = QHBoxLayout()
        top.setSpacing(theme.GAP)
        top.addWidget(self.summary, 1)
        top.addWidget(self.follow_button, 0, Qt.AlignTop)
        results_layout.addLayout(top)
        results_layout.addLayout(self._score_row)
        results_layout.addWidget(self.details)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 12, 0)
        right_layout.setSpacing(theme.GAP)
        right_layout.addWidget(self.wave)
        plots_box = QWidget()
        plots_box.setLayout(self._plots)
        plots_box.setMinimumHeight(300)
        plots_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(plots_box, 1)
        right_layout.addWidget(results)

        body = QHBoxLayout()
        body.setSpacing(theme.GAP + 4)
        body.addWidget(self.scrolling(controls, 352))
        body.addWidget(self.scrolling(right), 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 16, 12)
        layout.setSpacing(14)
        layout.addWidget(self.heading(
            "Experiment",
            "Damage a satellite signal on the way down, then see how each receiver recovers it.",
        ))
        layout.addLayout(body, 1)

        # -- behaviour ---------------------------------------------------
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(160)
        self._debounce.timeout.connect(self.run)

        self.run_button.clicked.connect(self.run)
        self.new_button.clicked.connect(self.new_signal)
        self.conditions.changed.connect(self._on_controls_changed)
        self.receivers.changed.connect(self._on_controls_changed)
        self.follow_button.clicked.connect(self._follow)
        self.keep.toggled.connect(lambda on: None if on else self._forget_seed())

    # ------------------------------------------------------------------
    @property
    def title(self) -> str:
        return "Experiment"

    def refresh(self) -> None:
        """Load the modulations, profiles and receivers the backend offers."""
        def load():
            return (self.client.modulations(), self.client.profiles())

        def loaded(data):
            schemes, profiles = data
            self.conditions.blockSignals(True)
            self.conditions.set_modulations([s["name"] for s in schemes])
            self.conditions.set_profiles(profiles)
            self.conditions.blockSignals(False)
            self.refresh_receivers()

        self.tasks.run(load, loaded, self.report_error)

    def refresh_receivers(self) -> None:
        """Re-read which receivers are ready for the chosen modulation."""
        modulation = self.conditions.modulation()
        self.tasks.run(
            lambda: self.client.receivers(modulation),
            lambda infos: (self.receivers.set_receivers(infos, modulation), self.run()),
            self.report_error,
        )

    # ------------------------------------------------------------------
    def _on_controls_changed(self) -> None:
        sender = self.sender()
        if sender is self.conditions and self._last is not None \
                and self._last["frame"]["modulation"] != self.conditions.modulation():
            self.refresh_receivers()      # a new modulation may change what is trained
            return
        if self.live.isChecked():
            self._debounce.start()

    def _forget_seed(self) -> None:
        self._seed = None

    def new_signal(self) -> None:
        # A run already in flight was sent with the old seed; when it
        # returns it must not put that seed back.  ``_want_new`` holds
        # until a request without a seed has actually been sent.
        self._seed = None
        self._want_new = True
        self.run()

    def request_body(self) -> dict:
        return {
            "modulation": self.conditions.modulation(),
            "conditions": self.conditions.conditions(),
            "receivers": self.receivers.selected()[: self.MAX_RECEIVERS],
            "seed": self._seed if self.keep.isChecked() else None,
        }

    def run(self) -> None:
        """Send the experiment. While one is running, at most one more waits."""
        if not self.conditions.modulation() or not self.receivers.selected():
            return
        if self._in_flight:
            self._dirty = True             # coalesce a burst of slider moves into one run
            return
        self._in_flight = True
        body = self.request_body()
        self._want_new = False
        self.run_button.setEnabled(False)
        self.tasks.run(lambda: self.client.experiment(body), self._show, self._failed)

    def _failed(self, error: BackendError) -> None:
        self._in_flight = False
        self.run_button.setEnabled(True)
        self.report_error(error)
        if self._dirty:
            self._dirty = False
            self.run()

    def _show(self, response: dict) -> None:
        self._in_flight = False
        self.run_button.setEnabled(True)
        self._last = response
        frame = response["frame"]
        if self.keep.isChecked() and not self._want_new:
            self._seed = frame["seed"]
        self.seed_label.setText(f"Signal number {frame['seed']}")

        self.wave.show_wave(frame["received"], frame["samples_per_symbol"], reference=frame["transmitted"])
        self.arrived.show_points(frame["received"], theme.RAW, cloud=True)
        self._layout_recovered(response["results"], frame)
        self._show_scores(response["results"])
        self.table.show_results(response["results"])
        self.summary.setText(self._summarise(response["results"]))
        self.follow_button.setEnabled(True)
        self.report(f"{frame['modulation']} at {frame['conditions']['snr_db']:g} dB, "
                    f"signal {frame['seed']}: recovered by {len(response['results'])} receivers")

        if self._dirty:
            self._dirty = False
            self.run()

    def _layout_recovered(self, results: list[dict], frame: dict) -> None:
        """One constellation per receiver, side by side, created as needed.

        A receiver keeps its own plot from one run to the next, so its
        points glide to their new places rather than being redrawn.
        """
        by_name = {getattr(p, "receiver_name", None): p for p in self._recovered}
        ordered = []
        for result in results:
            plot = by_name.pop(result["receiver"], None)
            if plot is None:
                plot = ConstellationPlot()
                plot.receiver_name = result["receiver"]
            ordered.append(plot)
        for spare in by_name.values():
            self._plots.removeWidget(spare)
            spare.setVisible(False)
        self._recovered = ordered + list(by_name.values())

        for plot in ordered:
            self._plots.removeWidget(plot)
        columns = 1 + len(results)
        per_row = 3 if columns > 4 else columns
        for index, (plot, result) in enumerate(zip(ordered, results), start=1):
            self._plots.addWidget(plot, index // per_row, index % per_row)
            plot.setVisible(True)
            wrong = sum(result["symbol_wrong"])
            plot.set_title(result["label"], f"{wrong} of {len(result['symbol_wrong'])} symbols wrong"
                           if wrong else "Every symbol right")
            plot.show_points(result["symbols"], theme.receiver_colour(result["receiver"]),
                             reference=frame["constellation"], wrong=result["symbol_wrong"])

    def _show_scores(self, results: list[dict]) -> None:
        while len(self._scores) < len(results):
            score = ScoreCard()
            self._scores.append(score)
            self._score_row.addWidget(score)
        for index, score in enumerate(self._scores):
            score.setVisible(index < len(results))
        for score, result in zip(self._scores, results):
            score.show_result(result)

    @staticmethod
    def _summarise(results: list[dict]) -> str:
        """One plain sentence about what just happened."""
        if not results:
            return ""
        errors = [r["bit_errors"] for r in results]
        if max(errors) == 0:
            sentence = "Every receiver recovered every bit."
        elif len(set(errors)) == 1:
            sentence = f"Every receiver got the same {errors[0]} bits wrong ({percent(results[0]['ber'])})."
        else:
            best = min(results, key=lambda r: r["bit_errors"])
            sentence = f"{best['label']} did best, with {best['bit_errors']} bits wrong."
        if len(results) > 1:
            sentence += " They all saw the same signal, so the differences are theirs."
        return sentence

    def _follow(self) -> None:
        if self._last is not None:
            frame = self._last["frame"]
            self.follow_requested.emit({
                "modulation": frame["modulation"],
                "conditions": self.conditions.conditions(),
                "seed": frame["seed"],
            })


__all__ = ["ExperimentPage"]
