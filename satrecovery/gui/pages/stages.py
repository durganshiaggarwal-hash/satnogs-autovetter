"""Follow one signal through every stage of a receiver.

The receiver is a sequence of six steps, so this page numbers them --
here the numbers mean something.  Each step shows what the signal looks
like after it, and says in one or two sentences what it did, quoting
the receiver's own estimates against the settings the user chose.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..motion import PillButton, fade_in
from ..widgets import ConditionsPanel, ConstellationPlot, WaveformPlot, card
from ..widgets.tables import percent
from .base import Page

#: Plain names for the stages, in the receiver's own order.
STAGE_NAMES = {
    "received": "What arrived",
    "CoarseFreq": "Remove Doppler",
    "MatchedFilter": "Filter noise",
    "GardnerTiming": "Find the timing",
    "AGC": "Fix the size",
    "CarrierPLL": "Lock the phase",
}


def explain(stage: str, diagnostics: dict, conditions: dict, frame: dict) -> str:
    """What this stage did, in a sentence or two, with real numbers.

    Every number quoted is either something the user set or something
    the receiver worked out without being told -- and the sentence says
    which is which, because the point is that the receiver had to find
    them blind.
    """
    sps = frame["samples_per_symbol"]
    if stage == "received":
        return (
            f"What the dish picked up: {sps} samples for every symbol, with noise and whatever "
            "the channel did. Plotted as every sample, it is a smear. The symbols are in there, "
            "blurred together."
        )
    if stage == "CoarseFreq":
        found = diagnostics.get("coarse_freq_offset_hz")
        found_text = "an estimate it rejected" if not diagnostics.get("coarse_freq_accepted") else f"{found:+.0f} Hz"
        return (
            f"The satellite's motion shifted the frequency by {conditions['freq_offset_hz']:+.0f} Hz. "
            f"Without being told, the receiver measured {found_text} and removed it. This comes "
            "before the filter on purpose: doing it after was a bug Phase 6 found."
        )
    if stage == "MatchedFilter":
        return (
            "A filter shaped exactly like the transmitted pulse. It keeps what belongs to the "
            "signal and discards noise outside its band, which is where most of the receiver's "
            "sensitivity comes from. Still every sample, so still a smear, but a tighter one."
        )
    if stage == "GardnerTiming":
        found = diagnostics.get("timing_offset_samples")
        found_text = ""
        if found is not None:
            found_text = f" It measured {found:+.2f} samples; you set {conditions['timing_offset']:+.1f}."
            if conditions.get("fading", "none") != "none":
                found_text += (" With echoes the two need not match: the delayed copies move the "
                               "clearest instant too, and the receiver finds that, not your setting.")
        return (
            f"Picks one sample per symbol, at the instant each one is clearest: {sps} samples "
            f"become 1, and the smear becomes points.{found_text}"
        )
    if stage == "AGC":
        gain = diagnostics.get("agc_gain")
        gain_text = "" if gain is None else f" It scaled the signal by {gain:.2f}."
        return (
            "Automatic gain control: resizes the constellation so the points can line up with "
            f"the rings. Fading changes the signal's strength, and this undoes that.{gain_text}"
        )
    if stage == "CarrierPLL":
        phase = diagnostics.get("initial_phase_rad")
        residual = diagnostics.get("residual_freq_hz")
        parts = ["Locks onto the remaining rotation and holds it steady. The bits are decided from these points."]
        if phase is not None:
            parts.append(
                f"It found a phase of {math.degrees(phase):+.0f}°; you set "
                f"{math.degrees(conditions['phase_offset_rad']):+.0f}°."
            )
        if residual is not None and abs(residual) >= 0.5:
            parts.append(f"It also caught {residual:+.1f} Hz of Doppler the earlier stage missed.")
        return " ".join(parts)
    return ""


class StagesPage(Page):
    """One receiver, one signal, one stage at a time."""

    def __init__(self, client, parent: QWidget | None = None) -> None:
        super().__init__(client, parent)
        self._response: dict | None = None
        self._seed: int | None = None
        self._index = 0

        self.conditions = ConditionsPanel()
        self.receiver = QComboBox()
        self.receiver.setAccessibleName("Receiver")
        self.receiver.setCursor(Qt.PointingHandCursor)
        self.run_button = PillButton("Follow a signal through", "primary")
        chooser, chooser_layout = card("Receiver", "Which receiver to follow.")
        chooser_layout.addWidget(self.receiver)
        chooser_layout.addWidget(self.run_button)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 12, 0)
        controls_layout.setSpacing(theme.GAP)
        controls_layout.addWidget(chooser)
        controls_layout.addWidget(self.conditions)
        controls_layout.addStretch(1)
        scroll = self.scrolling(controls, 352)

        # -- the step strip ---------------------------------------------
        self._steps = QHBoxLayout()
        self._steps.setSpacing(8)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.idClicked.connect(self.show_stage)
        self.prev_button = PillButton("Previous", "secondary")
        self.next_button = PillButton("Next", "secondary")
        self.prev_button.clicked.connect(lambda: self.show_stage(self._index - 1))
        self.next_button.clicked.connect(lambda: self.show_stage(self._index + 1))
        strip = QHBoxLayout()
        strip.setSpacing(0)
        strip.addLayout(self._steps, 1)

        self.stage_title = QLabel("Choose a channel and press Follow a signal through.")
        self.stage_title.setObjectName("CardTitle")
        self.explanation = QLabel("")
        self.explanation.setObjectName("Hint")
        self.explanation.setWordWrap(True)
        self.explanation.setMinimumHeight(44)
        self.explanation.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        story = QFrame()
        story.setObjectName("Story")
        fill = theme.page_fill(self.title)
        story.setStyleSheet(
            f"QFrame#Story {{ background: {theme.tint(fill, 0.55)}; border: 1px solid {theme.tint(fill, 0.2)};"
            f" border-radius: {theme.RADIUS_CARD}px; }}"
            f"QFrame#Story QLabel {{ background: transparent; }}"
        )
        story_layout = QVBoxLayout(story)
        story_layout.setContentsMargins(theme.PAD, 14, theme.PAD, 14)
        story_layout.setSpacing(6)
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title_row.addWidget(self.stage_title, 1)
        title_row.addWidget(self.prev_button)
        title_row.addWidget(self.next_button)
        story_layout.addLayout(title_row)
        story_layout.addWidget(self.explanation)

        self.points = ConstellationPlot("The signal at this step", "")
        self.wave = WaveformPlot("Over time", "In-phase part, first 24 symbols.")
        plots = QGridLayout()
        plots.setSpacing(theme.GAP)
        plots.addWidget(self.points, 0, 0)
        plots.addWidget(self.wave, 0, 1)
        plots.setColumnStretch(0, 1)
        plots.setColumnStretch(1, 1)

        right = QVBoxLayout()
        right.setSpacing(theme.GAP)
        right.addLayout(strip)
        right.addWidget(story)
        right.addLayout(plots, 1)

        body = QHBoxLayout()
        body.setSpacing(theme.GAP + 4)
        body.addWidget(scroll)
        body.addLayout(right, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 18)
        layout.setSpacing(14)
        layout.addWidget(self.heading(
            "Step by step",
            "Follow one signal through the receiver, one stage at a time, and see what each stage fixes.",
        ))
        layout.addLayout(body, 1)

        self.run_button.clicked.connect(self.run)
        self._update_nav()

    # ------------------------------------------------------------------
    @property
    def title(self) -> str:
        return "Step by step"

    def refresh(self) -> None:
        def load():
            return self.client.modulations(), self.client.profiles(), self.client.receivers()

        def loaded(data):
            schemes, profiles, receivers = data
            current = self.conditions.modulation()
            self.conditions.set_modulations([s["name"] for s in schemes], current or "16QAM")
            self.conditions.set_profiles(profiles)
            chosen = self.receiver.currentData()
            self.receiver.clear()
            for info in receivers:
                if info["available_for"]:
                    self.receiver.addItem(info["label"], info["name"])
            index = self.receiver.findData(chosen or "classical")
            self.receiver.setCurrentIndex(max(index, 0))

        self.tasks.run(load, loaded, self.report_error)

    def follow(self, request: dict) -> None:
        """Open a specific signal from the Experiment page."""
        self._seed = request.get("seed")
        self.conditions.set_state(request["modulation"], request.get("conditions", {}))
        self.run(keep_seed=True)

    def run(self, keep_seed: bool = False) -> None:
        if not self.conditions.modulation():
            return
        body = {
            "modulation": self.conditions.modulation(),
            "conditions": self.conditions.conditions(),
            "receiver": self.receiver.currentData() or "classical",
            "seed": self._seed if keep_seed else None,
        }
        self.run_button.setEnabled(False)
        self.tasks.run(lambda: self.client.stages(body), self._show, self._failed)

    def _failed(self, error) -> None:
        self.run_button.setEnabled(True)
        self.report_error(error)

    def _show(self, response: dict) -> None:
        self.run_button.setEnabled(True)
        self._response = response
        self._seed = response["frame"]["seed"]

        for button in self._group.buttons():
            self._group.removeButton(button)
            self._steps.removeWidget(button)
            button.deleteLater()
        for number, stage in enumerate(response["stages"]):
            button = PillButton(f"{number + 1}  {STAGE_NAMES.get(stage['name'], stage['name'])}", "step",
                                accent=theme.page_fill(self.title), accent_ink=theme.page_accent(self.title))
            button.setToolTip(stage["name"])
            button.setMinimumWidth(button.sizeHint().width())
            self._group.addButton(button, number)
            self._steps.addWidget(button, 1)
        result = response["result"]
        self.report(
            f"Followed signal {self._seed} through {len(response['stages'])} stages: "
            f"{result['bit_errors']} of {result['n_bits']} bits wrong ({percent(result['ber'])})"
        )
        self.show_stage(0)

    def show_stage(self, index: int) -> None:
        if self._response is None:
            return
        stages = self._response["stages"]
        index = max(0, min(int(index), len(stages) - 1))
        self._index = index
        stage = stages[index]
        frame = self._response["frame"]
        result = self._response["result"]
        last = index == len(stages) - 1

        button = self._group.button(index)
        if button is not None:
            button.setChecked(True)

        name = STAGE_NAMES.get(stage["name"], stage["name"])
        self.stage_title.setText(f"Step {index + 1} of {len(stages)}: {name}")
        self.explanation.setText(explain(stage["name"], result["diagnostics"],
                                         frame["conditions"], frame))
        fade_in(self.explanation)

        colour = theme.receiver_colour(result["receiver"])
        if stage["rate"] == "sample":
            self.points.set_title("The signal at this step", "Every sample, so symbols blur together.")
            self.points.show_points(stage["samples"], theme.RAW, cloud=True)
            self.wave.show_wave(stage["samples"], frame["samples_per_symbol"],
                                reference=frame["transmitted"], colour=theme.INK,
                                trace_label="after this step")
        else:
            wrong = result["symbol_wrong"] if last else None
            caption = (f"{sum(wrong)} symbols decided wrongly, shown as red crosses."
                       if last and wrong and any(wrong)
                       else "One point per symbol; rings are where they should be.")
            self.points.set_title("The symbols at this step", caption)
            self.points.show_points(stage["samples"], colour, reference=frame["constellation"], wrong=wrong)
            self.wave.show_wave(stage["samples"], 1, reference=frame["tx_symbols"], colour=colour,
                                dots=True, reference_label="what was sent", trace_label="after this step")
        self._update_nav()

    def _update_nav(self) -> None:
        count = len(self._response["stages"]) if self._response else 0
        self.prev_button.setEnabled(count > 0 and self._index > 0)
        self.next_button.setEnabled(count > 0 and self._index < count - 1)


__all__ = ["StagesPage", "explain", "STAGE_NAMES"]
