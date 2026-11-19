"""The controls a user sets: channel sliders and receiver choices.

DESIGN NOTE: ENCAPSULATION OF A UNIT CONVERSION
-----------------------------------------------
A ``QSlider`` only knows integers.  The channel needs floats in physical
units -- decibels, hertz, degrees -- and the API needs radians where the
user thinks in degrees.  :class:`LabeledSlider` hides the integer steps
completely: callers read and write ``value`` in real units and never see
a tick count.  The conversion to the API's units happens in exactly one
place, :meth:`ConditionsPanel.conditions`, so a degrees-versus-radians
mistake cannot hide in some other corner of the GUI.
"""

from __future__ import annotations

import math
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..motion import Chip, PillButton


def card(title: str = "", hint: str = "") -> tuple[QFrame, QVBoxLayout]:
    """A white rounded card with an optional title and one-line hint."""
    frame = QFrame()
    frame.setObjectName("Card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(theme.PAD, theme.PAD - 4, theme.PAD, theme.PAD)
    layout.setSpacing(12)
    if title:
        heading = QLabel(title)
        heading.setObjectName("CardTitle")
        layout.addWidget(heading)
    if hint:
        line = QLabel(hint)
        line.setObjectName("Hint")
        line.setWordWrap(True)
        layout.addWidget(line)
    return frame, layout


def section(text: str) -> QLabel:
    """A small muted label that groups the controls beneath it."""
    label = QLabel(text)
    label.setObjectName("Muted")
    return label


class LabeledSlider(QWidget):
    """A slider in real units, with a name and a live readout chip.

    Parameters
    ----------
    label:
        What the user is changing, in plain words.
    minimum, maximum, step:
        Range and resolution in real units.
    unit:
        Shown after the readout.
    decimals:
        Digits after the point in the readout.
    hint:
        Tooltip: the technical name and what it means physically.
    """

    valueChanged = Signal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        step: float,
        value: float,
        unit: str = "",
        decimals: int = 1,
        hint: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not maximum > minimum or not step > 0:
            raise ValueError("need maximum > minimum and a positive step")
        self._minimum = float(minimum)
        self._step = float(step)
        self._unit = unit
        self._decimals = int(decimals)
        ticks = int(round((maximum - minimum) / step))
        self._maximum = self._minimum + ticks * self._step

        self._name = QLabel(label)
        self._name.setObjectName("FieldLabel")
        self._readout = QLabel()
        self._readout.setObjectName("ReadoutChip")
        self._readout.setAlignment(Qt.AlignCenter)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, ticks)
        self._slider.setFocusPolicy(Qt.StrongFocus)
        self._slider.setAccessibleName(label)
        self._slider.setCursor(Qt.PointingHandCursor)
        if hint:
            for widget in (self._name, self._slider):
                widget.setToolTip(hint)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)
        layout.addWidget(self._name, 0, 0)
        layout.addWidget(self._readout, 0, 1, Qt.AlignRight)
        layout.addWidget(self._slider, 1, 0, 1, 2)

        self._slider.valueChanged.connect(self._on_ticks)
        self.value = value
        self._refresh_readout()

    # -- the public face: real units only -------------------------------
    @property
    def value(self) -> float:
        return round(self._minimum + self._slider.value() * self._step, 10)

    @value.setter
    def value(self, value: float) -> None:
        clamped = min(max(float(value), self._minimum), self._maximum)
        self._slider.setValue(int(round((clamped - self._minimum) / self._step)))

    @property
    def minimum(self) -> float:
        return self._minimum

    @property
    def maximum(self) -> float:
        return self._maximum

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 - Qt naming
        super().setEnabled(enabled)
        self._name.setObjectName("FieldLabel" if enabled else "Muted")
        self._name.style().polish(self._name)

    # -- internals ------------------------------------------------------
    def _on_ticks(self, _ticks: int) -> None:
        self._refresh_readout()
        self.valueChanged.emit(self.value)

    def _refresh_readout(self) -> None:
        text = f"{self.value:+.{self._decimals}f}" if self._minimum < 0 else f"{self.value:.{self._decimals}f}"
        self._readout.setText(f"{text} {self._unit}".strip())


class Swatch(QWidget):
    """A small filled circle in a receiver's colour, beside its name."""

    def __init__(self, colour: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._colour = QColor(colour)
        self.setFixedSize(12, 12)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._colour)
        painter.drawEllipse(1, 1, 10, 10)


class ConditionsPanel(QFrame):
    """Everything about the channel, as sliders a person can drag.

    Emits ``changed`` whenever anything moves.  Read the full state with
    :meth:`conditions` and :meth:`modulation`, which return exactly what
    the API expects.
    """

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")

        title = QLabel("The channel")
        title.setObjectName("CardTitle")
        hint = QLabel("What happens to the signal between the satellite and your dish.")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)

        self._modulation = QComboBox()
        self._modulation.setAccessibleName("Modulation")
        self._modulation.setCursor(Qt.PointingHandCursor)
        self._modulation.setToolTip(
            "How bits are packed into each symbol. More bits per symbol means more "
            "data, and a signal that breaks more easily."
        )

        self.snr = LabeledSlider(
            "Signal strength", -10, 30, 0.5, 10, "dB", 1,
            "Signal-to-noise ratio. Below about 0 dB the noise is stronger than the signal.",
        )
        self.doppler = LabeledSlider(
            "Doppler shift", -3000, 3000, 50, 0, "Hz", 0,
            "Carrier frequency offset, caused by the satellite moving towards or away from you.",
        )
        self.phase = LabeledSlider(
            "Phase error", -180, 180, 5, 0, "°", 0,
            "Carrier phase offset: the whole constellation rotated by a fixed angle.",
        )
        self.timing = LabeledSlider(
            "Clock error", -4, 4, 0.1, 0, "samples", 1,
            "Sampling timing offset. Eight samples make one symbol, so 4 is half a symbol out.",
        )
        self._echoes = QComboBox()
        self._echoes.addItem("None", "none")
        self._echoes.addItem("One path, fading", "flat_rician")
        self._echoes.addItem("Several paths", "multipath_rician")
        self._echoes.setAccessibleName("Echoes")
        self._echoes.setCursor(Qt.PointingHandCursor)
        self._echoes.setToolTip(
            "Rician fading: the signal arriving by more than one path, reflected off "
            "the ground or buildings near the dish."
        )
        self.k_factor = LabeledSlider(
            "Direct path strength", 0, 20, 0.5, 10, "dB", 1,
            "Rician K-factor: how much stronger the direct path is than the echoes.",
        )

        self._presets = QGridLayout()
        self._presets.setHorizontalSpacing(8)
        self._presets.setVerticalSpacing(8)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.PAD, theme.PAD - 4, theme.PAD, theme.PAD)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(section("Start from a standard channel"))
        layout.addLayout(self._presets)
        layout.addSpacing(4)
        layout.addWidget(self._labelled("Modulation", self._modulation))
        layout.addWidget(self.snr)
        layout.addSpacing(2)
        layout.addWidget(section("Damage on the way down"))
        for slider in (self.doppler, self.phase, self.timing):
            layout.addWidget(slider)
        layout.addWidget(self._labelled("Echoes", self._echoes))
        layout.addWidget(self.k_factor)

        for slider in (self.snr, self.doppler, self.phase, self.timing, self.k_factor):
            slider.valueChanged.connect(lambda _v: self.changed.emit())
        self._modulation.currentIndexChanged.connect(lambda _i: self.changed.emit())
        self._echoes.currentIndexChanged.connect(self._on_echoes)
        self._on_echoes()

    @staticmethod
    def _labelled(text: str, widget: QWidget) -> QWidget:
        box = QWidget()
        row = QVBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        label = QLabel(text)
        label.setObjectName("FieldLabel")
        row.addWidget(label)
        row.addWidget(widget)
        return box

    def _on_echoes(self, *_args) -> None:
        self.k_factor.setEnabled(self._echoes.currentData() != "none")
        self.changed.emit()

    # -- filling in what the backend offers -----------------------------
    def set_modulations(self, names: list[str], default: str = "16QAM") -> None:
        self._modulation.blockSignals(True)
        self._modulation.clear()
        self._modulation.addItems(names)
        if default in names:
            self._modulation.setCurrentText(default)
        self._modulation.blockSignals(False)

    def set_profiles(self, profiles: list[dict]) -> None:
        """One button per standard profile; clicking it sets the sliders.

        The same four channels the benchmark is built on, so a user can
        reproduce any benchmark condition here and look at it frame by
        frame.  The signal strength is left alone: it is the thing a user
        sweeps, and a preset resetting it would be a surprise.
        """
        names = {"clean": "Clean", "doppler": "Moving", "fading": "Echoes", "full": "Everything"}
        while self._presets.count():
            item = self._presets.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for index, profile in enumerate(profiles):
            button = PillButton(names.get(profile["name"], profile["name"].title()), "secondary")
            button.setToolTip(", ".join(f"{k} = {v}" for k, v in profile["settings"].items()) or "noise only")
            button.clicked.connect(lambda _c=False, p=profile["settings"]: self.apply_profile(p))
            self._presets.addWidget(button, index // 2, index % 2)

    def apply_profile(self, settings: dict) -> None:
        """Set every slider from a profile's settings, emitting one change."""
        self.blockSignals(True)
        self.doppler.value = settings.get("freq_offset_hz", 0.0)
        self.phase.value = math.degrees(settings.get("phase_offset_rad", 0.0))
        self.timing.value = settings.get("timing_offset", 0.0)
        index = self._echoes.findData(settings.get("fading", "none"))
        self._echoes.setCurrentIndex(max(index, 0))
        self.k_factor.value = settings.get("rician_k_db", 10.0)
        self.blockSignals(False)
        self._on_echoes()

    def set_state(self, modulation: str, conditions: dict) -> None:
        """Set the modulation and every slider at once, emitting one change.

        The inverse of :meth:`modulation` plus :meth:`conditions`: what
        one panel reports, another can be set to, which is how the
        Experiment page hands a signal to the Step-by-step page.
        """
        self.blockSignals(True)
        if self._modulation.findText(modulation) < 0:
            self._modulation.addItem(modulation)
        self._modulation.setCurrentText(modulation)
        self.snr.value = conditions.get("snr_db", self.snr.value)
        self.apply_profile(conditions)
        self.blockSignals(False)
        self.changed.emit()

    # -- reading the state, in the API's units -------------------------
    def modulation(self) -> str:
        return self._modulation.currentText()

    def conditions(self) -> dict:
        """Exactly the ``conditions`` object the API expects."""
        return {
            "snr_db": self.snr.value,
            "fading": self._echoes.currentData(),
            "rician_k_db": self.k_factor.value,
            "freq_offset_hz": self.doppler.value,
            "phase_offset_rad": math.radians(self.phase.value),
            "timing_offset": self.timing.value,
        }


class ReceiverPicker(QFrame):
    """Which receivers to compare: one colour chip each.

    A receiver that needs a trained model it does not have yet is shown
    but disabled, with a tooltip saying how to fix that -- hiding it
    would leave the user wondering where the hybrid went.
    """

    changed = Signal()

    def __init__(self, title: str = "Receivers to compare",
                 default: tuple[str, ...] = ("classical", "hybrid", "hybrid-noise"),
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._default = set(default)
        self._boxes: dict[str, Chip] = {}
        heading = QLabel(title)
        heading.setObjectName("CardTitle")
        hint = QLabel("Each keeps its colour on every plot.")
        hint.setObjectName("Hint")
        self._rows = QVBoxLayout()
        self._rows.setSpacing(8)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.PAD, theme.PAD - 4, theme.PAD, theme.PAD)
        layout.setSpacing(10)
        layout.addWidget(heading)
        layout.addWidget(hint)
        layout.addLayout(self._rows)

    def set_receivers(self, receivers: list[dict], modulation: str) -> None:
        """Rebuild the list from ``GET /api/receivers``, keeping choices."""
        previous = {name for name, box in self._boxes.items() if box.isChecked()} or self._default
        while self._rows.count():
            item = self._rows.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._boxes.clear()

        for info in receivers:
            available = modulation in info["available_for"]
            chip = Chip(info["label"], theme.receiver_colour(info["name"]))
            chip.setChecked(available and info["name"] in previous)
            chip.setEnabled(available)
            tip = info["description"]
            if not available:
                tip += f"\n\nNot trained for {modulation} yet: train it on the Models tab."
            chip.setToolTip(tip)
            chip.toggled.connect(lambda _t: self.changed.emit())
            self._rows.addWidget(chip)
            self._boxes[info["name"]] = chip
        if not self.selected():
            first = next((b for b in self._boxes.values() if b.isEnabled()), None)
            if first is not None:
                first.setChecked(True)
        self.changed.emit()

    def selected(self) -> list[str]:
        """Chosen receivers, in catalogue order (so colours never shuffle)."""
        return [name for name, box in self._boxes.items() if box.isChecked() and box.isEnabled()]

    def set_selected(self, names: list[str]) -> None:
        for name, box in self._boxes.items():
            box.setChecked(box.isEnabled() and name in names)


class CheckList(QWidget):
    """Several options as chips, any number of which can be chosen."""

    changed = Signal()

    def __init__(self, options: list[tuple[str, str]], checked: list[str] | None = None,
                 columns: int = 3, colour_of: Callable[[str], str] | None = None,
                 accent: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        self._boxes: dict[str, Chip] = {}
        for index, (value, label) in enumerate(options):
            colour = colour_of(value) if colour_of is not None else (accent or theme.BRAND_DEEP)
            chip = Chip(label, colour)
            chip.setChecked(value in (checked or []))
            chip.toggled.connect(lambda _t: self.changed.emit())
            grid.addWidget(chip, index // columns, index % columns)
            self._boxes[value] = chip
        for column in range(columns):
            grid.setColumnStretch(column, 1)

    def selected(self) -> list[str]:
        return [value for value, box in self._boxes.items() if box.isChecked() and box.isEnabled()]

    def set_tooltip(self, value: str, text: str) -> None:
        """Explain one option on hover."""
        if value in self._boxes:
            self._boxes[value].setToolTip(text)

    def set_enabled_options(self, enabled: set[str], reason: str = "") -> None:
        for value, box in self._boxes.items():
            ok = value in enabled
            box.setEnabled(ok)
            if not ok:
                box.setChecked(False)
                box.setToolTip(reason)
            else:
                box.setToolTip("")


__all__ = ["CheckList", "ConditionsPanel", "LabeledSlider", "ReceiverPicker", "Swatch", "card", "section"]
