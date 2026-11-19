"""Training the hybrid receiver's neural networks.

The hybrid needs one trained network per modulation.  This page shows
one card for each: whether a network exists, what it was trained on,
and a button to train it, in the background, with a progress bar.  When
a network finishes, the other pages are told, so the hybrid appears in
their receiver lists without a restart.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .. import theme
from ..motion import PillButton, SmoothProgress, ToggleSwitch, fade_in
from ..tasks import JobWatcher
from ..widgets import card
from .base import Page

#: The "quick" option: enough to try the hybrid, not enough to trust it.
QUICK = {"frames_each": 6, "epochs": 60}


class ModelCard(QWidget):
    """One modulation's network: its state, and the button that trains it."""

    train_clicked = Signal(str)

    def __init__(self, modulation: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.modulation = modulation
        frame, layout = card()
        self.name = QLabel(modulation)
        self.name.setObjectName("PageTitle")
        self.badge = QLabel("")
        self.badge.setObjectName("Badge")
        self.detail = QLabel("")
        self.detail.setObjectName("Hint")
        self.detail.setWordWrap(True)
        self.detail.setMinimumHeight(38)
        self.button = PillButton("Train", "secondary")
        self.button.clicked.connect(lambda: self.train_clicked.emit(self.modulation))
        top = QHBoxLayout()
        top.addWidget(self.name)
        top.addStretch(1)
        top.addWidget(self.badge)
        layout.addLayout(top)
        layout.addWidget(self.detail)
        layout.addWidget(self.button)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)

    def show_state(self, model: dict | None, training: bool, busy: bool) -> None:
        if training:
            text, fill, ink = "Training…", theme.page_fill("Models"), theme.page_accent("Models")
            self.detail.setText("Learning from examples in the background.")
        elif model:
            text, fill, ink = "Trained", theme.tint(theme.SAGE, 0.3), "#3f5a37"
            when = time.strftime("%d %b %H:%M", time.localtime(model["trained_at"]))
            self.detail.setText(f"Learned from {model['rows']:,} symbols in {model['seconds']:.0f} s, on {when}.")
        else:
            text, fill, ink = "Not trained", theme.SUNK, theme.INK_3
            self.detail.setText("The hybrid cannot be used with this modulation until it is trained.")
        self.badge.setText(text)
        self.badge.setStyleSheet(f"background: {fill}; color: {ink};")
        self.button.setText("Retrain" if model else "Train")
        self.button.setEnabled(not busy)


class ModelsPage(Page):
    """One card per modulation: trained or not, and a button to train."""

    #: Emitted when a model finishes training, so other pages can refresh.
    models_changed = Signal()

    def __init__(self, client, parent: QWidget | None = None) -> None:
        super().__init__(client, parent)
        self._watcher: JobWatcher | None = None
        self._training: str | None = None
        self.cards: dict[str, ModelCard] = {}

        intro, intro_layout = card(
            "Hybrid receiver models",
            "The hybrid keeps the classical receiver's front end and lets a small neural network make "
            "the final decision. Each modulation needs its own network. Full training takes from tens "
            "of seconds to a few minutes and runs in the background.",
        )
        self.quick = ToggleSwitch("Quick training: about five times faster, less accurate")
        self.quick.setToolTip(
            "Fewer examples and fewer passes over them. Fine for trying the hybrid out; use full "
            "training before quoting any result, because an under-trained hybrid loses for reasons "
            "that have nothing to do with the method (the Phase 6 fairness check)."
        )
        self.progress = SmoothProgress()
        self.progress.setVisible(False)
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("Muted")
        self.progress_label.setVisible(False)
        intro_layout.addWidget(self.quick)
        intro_layout.addWidget(self.progress)
        intro_layout.addWidget(self.progress_label)

        self._grid = QGridLayout()
        self._grid.setSpacing(theme.GAP)

        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(0, 0, 12, 0)
        column.setSpacing(theme.GAP)
        column.addWidget(intro)
        column.addLayout(self._grid)
        column.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 16, 12)
        layout.setSpacing(14)
        layout.addWidget(self.heading("Models", "Train the neural networks the hybrid receiver needs."))
        layout.addWidget(self.scrolling(content), 1)

    @property
    def title(self) -> str:
        return "Models"

    def refresh(self) -> None:
        self.tasks.run(lambda: (self.client.modulations(), self.client.models()),
                       self._show, self.report_error)

    def _show(self, data) -> None:
        schemes, models = data
        trained = {m["modulation"]: m for m in models}
        for index, scheme in enumerate(schemes):
            name = scheme["name"]
            if name not in self.cards:
                model_card = ModelCard(name)
                model_card.train_clicked.connect(self.train)
                self.cards[name] = model_card
                self._grid.addWidget(model_card, index // 3, index % 3)
                fade_in(model_card)
            self.cards[name].show_state(trained.get(name), name == self._training, self._watcher is not None)

    def train(self, modulation: str) -> None:
        body = dict(QUICK) if self.quick.isChecked() else {}
        self._training = modulation
        self.tasks.run(lambda: self.client.train(modulation, body), self._started, self._failed_to_start)

    def _failed_to_start(self, error) -> None:
        self._training = None
        self.report_error(error)

    def _started(self, job: dict) -> None:
        self.progress.reset()
        self.progress.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_label.setText(f"Training the {self._training} network…")
        self._watcher = JobWatcher(self.client, job["id"], parent=self)
        self._watcher.progress.connect(lambda f, m: (self.progress.set_fraction(f),
                                                     self.progress_label.setText(f"{f:.0%}  {m}")))
        self._watcher.finished.connect(self._finished)
        self._watcher.failed.connect(self._failed)
        self._watcher.start()
        self.notify.emit(f"Training the {self._training} network in the background.", "info")
        self.refresh()

    def _finished(self, job: dict) -> None:
        result = job["result"]
        self.progress.set_fraction(1.0)
        self.progress_label.setText(
            f"{result['modulation']} trained on {result['rows']:,} symbols in {result['seconds']:.0f} s."
        )
        self.notify.emit(f"The {result['modulation']} hybrid is ready to use.", "success")
        self._watcher = None
        self._training = None
        self.refresh()
        self.models_changed.emit()

    def _failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.progress_label.setText(f"Training failed: {message}")
        self.notify.emit(f"Training failed: {message}", "error")
        self._watcher = None
        self._training = None
        self.refresh()


__all__ = ["ModelCard", "ModelsPage", "QUICK"]
