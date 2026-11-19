"""Many signals, many conditions: how the receivers compare overall.

One experiment is one signal, and one signal can be lucky.  A benchmark
runs many signals at every combination of modulation, channel and
strength, and plots the error rate with its uncertainty -- the
measurement every claim in the project rests on.

It runs as a background job on the backend: the page shows its progress
and stays usable, and the results are saved, so they are still here the
next time the application opens.
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..client import BackendError
from ..motion import Collapsible, PillButton, SmoothProgress, fade_in
from ..tasks import JobWatcher
from ..widgets import BerCurvePlot, CheckList, card, section
from ..widgets.tables import percent
from .base import Page

PROFILE_LABELS = {"clean": "Clean", "doppler": "Moving", "fading": "Echoes", "full": "Everything",
                  "recorded": "Real recordings"}
STANDARD_PROFILES = {"clean", "doppler", "fading", "full"}


class BenchmarkPage(Page):
    """Set up a grid, run it in the background, read the curves."""

    def __init__(self, client, parent: QWidget | None = None) -> None:
        super().__init__(client, parent)
        self._watcher: JobWatcher | None = None
        self._receiver_info: list[dict] = []
        self._cells: list[dict] = []
        accent = theme.page_accent(self.title)

        # -- left: the form ---------------------------------------------
        form_card, form = card("New benchmark", "Pick what to test; it runs in the background.")

        self.name = QLineEdit()
        self.name.setPlaceholderText("A name you will recognise later")
        form.addWidget(section("Name"))
        form.addWidget(self.name)

        form.addWidget(section("Modulations"))
        self.modulations = CheckList([], [])
        self._mod_holder = QVBoxLayout()
        self._mod_holder.addWidget(self.modulations)
        form.addLayout(self._mod_holder)

        form.addWidget(section("Channels"))
        self.profiles = CheckList([(k, v) for k, v in PROFILE_LABELS.items()], ["clean", "full"],
                                  columns=2, accent=accent)
        form.addWidget(self.profiles)

        form.addWidget(section("Signal strengths"))
        strengths = QFormLayout()
        strengths.setHorizontalSpacing(12)
        strengths.setVerticalSpacing(8)
        self.snr_from = self._spin(-10, 30, 0)
        self.snr_to = self._spin(-10, 30, 16)
        self.snr_step = self._spin(1, 10, 4)
        strengths.addRow("From", self.snr_from)
        strengths.addRow("To", self.snr_to)
        strengths.addRow("In steps of", self.snr_step)
        form.addLayout(strengths)

        form.addWidget(section("Receivers"))
        self._rx_holder = QVBoxLayout()
        self.receivers = CheckList([], [])
        self._rx_holder.addWidget(self.receivers)
        form.addLayout(self._rx_holder)

        frames_row = QFormLayout()
        self.frames = QSpinBox()
        self.frames.lineEdit().setTextMargins(6, 0, 0, 0)
        self.frames.setRange(1, 200)
        self.frames.setValue(20)
        self.frames.setToolTip("More signals per condition means narrower uncertainty and a longer run.")
        frames_row.addRow("Signals per condition", self.frames)
        form.addLayout(frames_row)

        self.size_label = QLabel("")
        self.size_label.setObjectName("Muted")
        self.size_label.setWordWrap(True)
        form.addWidget(self.size_label)

        self.run_button = PillButton("Run benchmark", "primary")
        self.progress = SmoothProgress()
        self.progress.setVisible(False)
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("Muted")
        self.progress_label.setWordWrap(True)
        form.addWidget(self.run_button)
        form.addWidget(self.progress)
        form.addWidget(self.progress_label)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 12, 0)
        controls_layout.addWidget(form_card)
        controls_layout.addStretch(1)

        # -- right: results ---------------------------------------------
        self.runs = QComboBox()
        self.runs.setMinimumWidth(260)
        self.runs.setAccessibleName("Saved benchmark")
        self.delete_button = PillButton("Delete", "quiet")
        self.view_modulation = QComboBox()
        self.view_profile = QComboBox()
        self.view_profile.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.view_profile.setMinimumContentsLength(14)
        for combo in (self.runs, self.view_modulation, self.view_profile):
            combo.setCursor(Qt.PointingHandCursor)
        picker_card, picker_layout = card()
        picker = QHBoxLayout()
        picker.setSpacing(10)
        picker.addWidget(QLabel("Saved results"))
        picker.addWidget(self.runs, 1)
        picker.addWidget(self.delete_button)
        picker.addSpacing(18)
        picker.addWidget(QLabel("Show"))
        picker.addWidget(self.view_modulation)
        picker.addWidget(self.view_profile)
        picker_layout.setContentsMargins(theme.PAD, 12, theme.PAD, 12)
        picker_layout.addLayout(picker)

        self.curve = BerCurvePlot(
            "Error rate as the signal gets stronger",
            "Shaded bands are 95% ranges. A hollow triangle means no errors were seen: the "
            "true rate is somewhere below the bottom of the chart, not zero.",
        )
        self.curve.setMinimumHeight(360)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Receiver", "Strength", "Error rate", "Likely range", "Lost signals", "Spots its mistakes"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setShowGrid(False)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMinimumHeight(220)
        table_card, table_layout = card()
        table_layout.addWidget(Collapsible("The numbers behind the chart", self.table, expanded=False))

        self.empty = QLabel("No saved results yet. Set up a benchmark on the left and run it.")
        self.empty.setObjectName("Hint")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setMinimumHeight(200)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 12, 0)
        right_layout.setSpacing(theme.GAP)
        right_layout.addWidget(picker_card)
        right_layout.addWidget(self.empty)
        right_layout.addWidget(self.curve, 1)
        right_layout.addWidget(table_card)
        self._result_widgets = (self.curve, table_card)

        body = QHBoxLayout()
        body.setSpacing(theme.GAP + 4)
        body.addWidget(self.scrolling(controls, 372))
        body.addWidget(self.scrolling(right), 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 16, 12)
        layout.setSpacing(14)
        layout.addWidget(self.heading(
            "Benchmark",
            "Run every receiver on many signals across many conditions, and compare error rates "
            "with their uncertainty.",
        ))
        layout.addLayout(body, 1)

        for widget in (self.snr_from, self.snr_to, self.snr_step, self.frames):
            widget.valueChanged.connect(self._update_size)
        self.profiles.changed.connect(self._update_size)
        self.run_button.clicked.connect(self.start)
        self.runs.currentIndexChanged.connect(self._load_selected_run)
        self.view_modulation.currentIndexChanged.connect(self._draw)
        self.view_profile.currentIndexChanged.connect(self._draw)
        self.delete_button.clicked.connect(self._delete_run)

    @staticmethod
    def _spin(low: float, high: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setValue(value)
        spin.setSuffix(" dB")
        spin.lineEdit().setTextMargins(6, 0, 0, 0)   # Qt ignores padding-left on spin boxes
        return spin

    # ------------------------------------------------------------------
    @property
    def title(self) -> str:
        return "Benchmark"

    def refresh(self) -> None:
        def load():
            return (self.client.modulations(), self.client.receivers(), self.client.runs(),
                    self.client.dataset())

        self.tasks.run(load, self._on_loaded, self.report_error)

    def _on_loaded(self, data) -> None:
        schemes, receivers, runs, dataset = data
        self._show_dataset(dataset)
        self._receiver_info = receivers
        names = [s["name"] for s in schemes]

        chosen = self.modulations.selected() or ["16QAM"]
        replacement = CheckList([(n, n) for n in names], chosen, columns=2,
                                accent=theme.page_accent(self.title))
        self._mod_holder.replaceWidget(self.modulations, replacement)
        self.modulations.deleteLater()
        self.modulations = replacement
        self.modulations.changed.connect(self._on_modulations)

        picked = self.receivers.selected() or ["classical", "hybrid", "hybrid-noise"]
        rx = CheckList([(r["name"], r["label"]) for r in receivers], picked, columns=1,
                       colour_of=theme.receiver_colour)
        self._rx_holder.replaceWidget(self.receivers, rx)
        self.receivers.deleteLater()
        self.receivers = rx
        self.receivers.changed.connect(self._update_size)
        self._on_modulations()
        self._fill_runs(runs)

    def _show_dataset(self, dataset: dict) -> None:
        """Offer "Real recordings" only when the backend has RML24 open."""
        available = bool(dataset.get("available"))
        enabled = set(STANDARD_PROFILES) | ({"recorded"} if available else set())
        self.profiles.set_enabled_options(enabled, dataset.get("detail", ""))
        if available:
            self.profiles.set_tooltip(
                "recorded",
                f"Real satellite recordings from {dataset.get('name', 'RML24')}, with noise added to "
                "reach each strength. BPSK, QPSK and 8PSK only: the others are not clean enough to "
                "label reliably, and are skipped."
            )

    def _on_modulations(self) -> None:
        """A receiver is offered if it is ready for at least one chosen modulation."""
        chosen = set(self.modulations.selected())
        ready = {r["name"] for r in self._receiver_info if chosen & set(r["available_for"])}
        self.receivers.set_enabled_options(ready, "Not trained for these modulations yet: see the Models tab.")
        self._update_size()

    def _snr_values(self) -> list[float]:
        low, high, step = self.snr_from.value(), self.snr_to.value(), self.snr_step.value()
        if high < low:
            return []
        return [round(float(v), 2) for v in np.arange(low, high + 1e-9, step)]

    def _update_size(self) -> None:
        conditions = len(self.modulations.selected()) * len(self.profiles.selected()) * len(self._snr_values())
        receivers = len(self.receivers.selected())
        signals = conditions * self.frames.value()
        self.size_label.setText(f"{conditions} conditions × {receivers} receivers, {signals:,} signals in all.")
        self.run_button.setEnabled(conditions > 0 and receivers > 0 and self._watcher is None)

    # -- running --------------------------------------------------------
    def start(self) -> None:
        body = {
            "name": self.name.text().strip() or time.strftime("Benchmark %d %b %H:%M"),
            "modulations": self.modulations.selected(),
            "snrs_db": self._snr_values(),
            "profiles": self.profiles.selected(),
            "receivers": self.receivers.selected(),
            "frames_per_scenario": self.frames.value(),
        }
        self.run_button.setEnabled(False)
        self.tasks.run(lambda: self.client.start_benchmark(body), self._started, self._start_failed)

    def _start_failed(self, error: BackendError) -> None:
        self.progress_label.setText(error.detail)
        self.report_error(error)
        self._update_size()

    def _started(self, job: dict) -> None:
        self.progress.reset()
        self.progress.setVisible(True)
        fade_in(self.progress)
        self.progress_label.setText("Starting…")
        self._watcher = JobWatcher(self.client, job["id"], parent=self)
        self._watcher.progress.connect(self._on_progress)
        self._watcher.finished.connect(self._on_finished)
        self._watcher.failed.connect(self._on_failed)
        self._watcher.start()
        self.notify.emit("Benchmark started. You can keep using the app while it runs.", "info")

    def _on_progress(self, fraction: float, message: str) -> None:
        self.progress.set_fraction(fraction)
        self.progress_label.setText(f"{fraction:.0%}  {message}")

    def _on_finished(self, job: dict) -> None:
        self._watcher = None
        self.progress.set_fraction(1.0)
        result = job["result"]
        skipped = len(result.get("skipped", []))
        note = f", {skipped} skipped (no trained model)" if skipped else ""
        self.progress_label.setText(
            f"Finished in {job['elapsed_seconds']:.0f} s: {result['cells']} measurements{note}."
        )
        self.notify.emit(f"Benchmark finished: {result['cells']} measurements saved.", "success")
        self._update_size()
        new_id = result["run_id"]
        self.tasks.run(self.client.runs, lambda runs: self._fill_runs(runs, select=new_id), self.report_error)

    def _on_failed(self, message: str) -> None:
        self._watcher = None
        self.progress.setVisible(False)
        self.progress_label.setText(f"The benchmark failed: {message}")
        self.notify.emit(f"Benchmark failed: {message}", "error")
        self._update_size()

    # -- reading saved results -------------------------------------------
    def _fill_runs(self, runs: list[dict], select: int | None = None) -> None:
        self.runs.blockSignals(True)
        self.runs.clear()
        for run in runs:
            when = time.strftime("%d %b %H:%M", time.localtime(run["created"]))
            self.runs.addItem(f"{run['name']}  ({when})", run["id"])
        self.runs.blockSignals(False)
        has_runs = bool(runs)
        self.empty.setVisible(not has_runs)
        for widget in self._result_widgets:
            widget.setVisible(has_runs)
        self.delete_button.setEnabled(has_runs)
        if has_runs:
            index = self.runs.findData(select) if select is not None else 0
            self.runs.setCurrentIndex(max(index, 0))
            self._load_selected_run()

    def _load_selected_run(self) -> None:
        run_id = self.runs.currentData()
        if run_id is None:
            return
        self.tasks.run(lambda: self.client.cells(run_id), self._cells_loaded, self.report_error)

    def _cells_loaded(self, cells: list[dict]) -> None:
        self._cells = cells
        for combo, key, labels in (
            (self.view_modulation, "modulation", {}),
            (self.view_profile, "profile", PROFILE_LABELS),
        ):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for value in dict.fromkeys(c[key] for c in cells):
                combo.addItem(labels.get(value, value), value)
            index = combo.findData(current)
            combo.setCurrentIndex(max(index, 0))
            combo.blockSignals(False)
        self._draw()

    def _draw(self) -> None:
        modulation = self.view_modulation.currentData()
        profile = self.view_profile.currentData()
        chosen = [c for c in self._cells if c["modulation"] == modulation and c["profile"] == profile]
        labels = {r["name"]: r["label"] for r in self._receiver_info}

        series = []
        for receiver in dict.fromkeys(c["receiver"] for c in chosen):
            rows = sorted((c for c in chosen if c["receiver"] == receiver), key=lambda c: c["snr_db"])
            series.append({
                "name": receiver,
                "label": labels.get(receiver, receiver),
                "snr": [c["snr_db"] for c in rows],
                "ber": [c["ber"] for c in rows],
                "low": [c["ber_low"] for c in rows],
                "high": [c["ber_high"] for c in rows],
            })
        caption = ("Shaded bands are 95% ranges. A hollow triangle means no errors were seen: the true "
                   "rate is somewhere below the bottom of the chart, not zero.")
        twins = [
            (a["label"], b["label"])
            for i, a in enumerate(series) for b in series[i + 1:]
            if a["snr"] == b["snr"] and a["ber"] == b["ber"]
        ]
        if twins:
            first, second = twins[0]
            caption += (f" {first} and {second} made identical decisions, so one line hides the "
                        "other: they differ only in confidence.")
        self.curve.set_title(
            (f"{modulation}, real satellite recordings: error rate as the signal gets stronger"
             if profile == "recorded" else
             f"{modulation}, {PROFILE_LABELS.get(profile, profile)} channel: error rate as the signal gets stronger"),
            caption,
        )
        self.curve.show_curves(series)

        rows = sorted(chosen, key=lambda c: (c["snr_db"], c["receiver"]))
        self.table.setRowCount(len(rows))
        for r, cell in enumerate(rows):
            ranking = cell.get("ranking_quality")
            values = [
                labels.get(cell["receiver"], cell["receiver"]),
                f"{cell['snr_db']:g} dB",
                percent(cell["ber"]),
                f"{percent(cell['ber_low'])} – {percent(cell['ber_high'])}",
                f"{cell['lost_frames']} of {cell['n_frames']}",
                "–" if ranking is None or cell["errors"] == 0 else f"{ranking:.2f}",
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column == 0:
                    item.setData(Qt.DecorationRole, QColor(theme.receiver_colour(cell["receiver"])))
                else:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, column, item)

    def _delete_run(self) -> None:
        run_id = self.runs.currentData()
        if run_id is None:
            return
        self.tasks.run(lambda: (self.client.delete_run(run_id), self.client.runs())[1],
                       self._fill_runs, self.report_error)
        self.notify.emit("Saved result deleted.", "info")


__all__ = ["BenchmarkPage"]
