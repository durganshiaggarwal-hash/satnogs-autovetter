"""The pieces of the analysis screen: the AI's answer, the recovery animation, the bits.

THE RECOVERY ANIMATION
----------------------
:class:`RecoveryAnimation` plays the receiver's work as one continuous
movie instead of a set of steps to click through.  The analyser reduces
every stage to one point per symbol, so point number ``k`` is the same
symbol in every stage; the animation slides each point smoothly from
where it was after one stage to where it is after the next.  Each point
is coloured by the symbol it is *finally* decided to be, so a jumble of
colours visibly sorts itself into clean, single-coloured clusters --
which is what "recovering the signal" looks like.

It runs on one timer and a clock, not a queue of animations, so pausing,
replaying and changing speed are all just changes to the clock.  With
animations switched off (``Motion.enabled``) it jumps to the end.
"""

from __future__ import annotations

import colorsys

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ...gui import theme
from ...gui.motion import CountUpLabel, Motion, PillButton, animate, brand_gradient, fade_in, font


def _complex(points: dict | None) -> np.ndarray:
    if not points:
        return np.zeros(0, dtype=complex)
    return np.asarray(points["i"], dtype=float) + 1j * np.asarray(points["q"], dtype=float)


def palette(count: int) -> list[QColor]:
    """``count`` clearly different colours, soft enough for the theme."""
    count = max(int(count), 1)
    colours = []
    for k in range(count):
        hue = (k / count + 0.03) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.62, 0.86)
        colours.append(QColor(int(r * 255), int(g * 255), int(b * 255)))
    return colours


def _ease(t: float) -> float:
    """Smooth start and stop (cubic in-out)."""
    t = min(max(t, 0.0), 1.0)
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


# ======================================================================
# Step 1: the AI's answer
# ======================================================================
class ProbabilityBars(QWidget):
    """One bar per class, growing to its probability; the winner in the brand gradient."""

    ROW = 26

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[tuple[str, float]] = []
        self._grow = 1.0
        self.setMinimumHeight(6 * self.ROW)

    def show_probabilities(self, rows: list[tuple[str, float]]) -> None:
        self._rows = list(rows)
        self.setMinimumHeight(max(len(self._rows), 1) * self.ROW)
        animate(self, 0.0, 1.0, Motion.SLOW * 3, lambda v: (setattr(self, "_grow", v), self.update()))

    def clear(self) -> None:
        self._rows = []
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        name_w, pct_w = 64, 48
        bar_w = max(self.width() - name_w - pct_w - 8, 10)
        for i, (name, p) in enumerate(self._rows):
            y = i * self.ROW
            painter.setFont(font(theme.SANS_SEMIBOLD if i == 0 else theme.SANS_MEDIUM, 9))
            painter.setPen(QColor(theme.INK if i == 0 else theme.INK_2))
            painter.drawText(QRectF(0, y, name_w, self.ROW), Qt.AlignVCenter | Qt.AlignLeft, name)
            track = QRectF(name_w, y + 8, bar_w, self.ROW - 16)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(theme.SUNK))
            painter.drawRoundedRect(track, 5, 5)
            filled = QRectF(track.left(), track.top(), max(track.width() * p * self._grow, 2), track.height())
            painter.setBrush(brand_gradient(filled) if i == 0 else QColor(theme.LINE))
            painter.drawRoundedRect(filled, 5, 5)
            painter.setFont(font(theme.MONO, 8))
            painter.setPen(QColor(theme.INK_2))
            painter.drawText(QRectF(name_w + bar_w + 8, y, pct_w, self.ROW), Qt.AlignVCenter | Qt.AlignRight,
                             f"{100 * p * self._grow:.0f}%")


class ClassifierCard(QFrame):
    """Step 1: what the AI thinks the signal is, and whether it was right."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.PAD, theme.PAD - 4, theme.PAD, theme.PAD)
        layout.setSpacing(8)
        step = QLabel("STEP 1  ·  THE AI CLASSIFIER")
        step.setObjectName("Muted")
        title = QLabel("What kind of signal is this?")
        title.setObjectName("CardTitle")
        self.answer = QLabel("—")
        self.answer.setObjectName("BigNumber")
        self.sure = QLabel("Press Analyse to ask the AI.")
        self.sure.setObjectName("Hint")
        self.sure.setWordWrap(True)
        self.bars = ProbabilityBars()
        self.verdict = QLabel("")
        self.verdict.setWordWrap(True)
        self.verdict.setStyleSheet("font-family: 'Manrope SemiBold';")
        # How far to trust the answer: confident, unsure or unfamiliar.
        self.trust = QLabel("")
        self.trust.setWordWrap(True)
        self.trust.setVisible(False)
        for widget in (step, title, self.answer, self.sure, self.bars, self.trust, self.verdict):
            layout.addWidget(widget)
        layout.addStretch(1)

    #: Badge colours by level: (fill, text).
    def _trust_colours(self, level: str) -> tuple[str, str]:
        if level == "confident":
            return theme.tint(theme.GOOD, 0.82), theme.INK
        if level == "unsure":
            return theme.tint("#e0a030", 0.72), theme.INK
        return theme.tint(theme.CRITICAL, 0.75), theme.INK

    def show_trust(self, trust: dict | None) -> None:
        """The warning badge: how far to trust this answer (hidden for older results without one)."""
        if not trust:
            self.trust.setVisible(False)
            return
        mark = {"confident": "✓", "unsure": "⚠", "unfamiliar": "⚠"}.get(trust["level"], "")
        text = f"{mark}  {trust['title']}"
        if trust.get("reasons"):
            text += "\n" + "\n".join(f"•  {reason}" for reason in trust["reasons"])
        fill, ink = self._trust_colours(trust["level"])
        self.trust.setStyleSheet(f"background: {fill}; color: {ink}; border-radius: 8px; padding: 8px 10px;")
        self.trust.setText(text)
        self.trust.setVisible(True)
        fade_in(self.trust)

    def thinking(self) -> None:
        self.trust.setVisible(False)
        self.answer.setText("…")
        self.sure.setText("The AI is looking at the signal.")
        self.bars.clear()
        self.verdict.setText("")

    def show_result(self, classification: dict) -> None:
        self.answer.setText(classification["label"])
        fade_in(self.answer)
        self.sure.setText(f"{classification['confidence']:.0%} sure. The other possibilities, as the AI rates them:")
        self.bars.show_probabilities([(row["name"], row["p"]) for row in classification["probabilities"]])
        if classification.get("true_label") is None:
            self.verdict.setText("Your file does not say what it really is, so this answer cannot be marked "
                                 "right or wrong.")
            self.verdict.setStyleSheet(f"color: {theme.INK_2}; font-family: 'Manrope SemiBold';")
        elif classification.get("correct"):
            self.verdict.setText(f"✓  Right: it really is {classification['true_label']}.")
            self.verdict.setStyleSheet(f"color: {theme.GOOD}; font-family: 'Manrope SemiBold';")
        else:
            self.verdict.setText(f"✕  Wrong: it is really {classification['true_label']}.")
            self.verdict.setStyleSheet(f"color: {theme.CRITICAL}; font-family: 'Manrope SemiBold';")
        fade_in(self.verdict)


# ======================================================================
# Step 2: the recovery, as an animation
# ======================================================================
class StageStrip(QWidget):
    """Dots joined by a line, one per stage; the line fills as the animation advances."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._names: list[str] = []
        self._position = 0.0           # 0 .. len-1, fractional during a move
        self.setFixedHeight(46)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_stages(self, names: list[str]) -> None:
        self._names = list(names)
        self._position = 0.0
        self.update()

    def set_position(self, position: float) -> None:
        self._position = float(position)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        if not self._names:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        n = len(self._names)
        margin = 40
        span = max(self.width() - 2 * margin, 1)
        xs = [margin + span * (i / max(n - 1, 1)) for i in range(n)]
        y = 12
        painter.setPen(QPen(QColor(theme.LINE), 3, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(xs[0], y), QPointF(xs[-1], y))
        reach = xs[0] + (xs[-1] - xs[0]) * (self._position / max(n - 1, 1))
        painter.setPen(QPen(QColor(theme.MAUVE), 3, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(xs[0], y), QPointF(reach, y))
        current = int(round(self._position))
        for i, x in enumerate(xs):
            done = i <= self._position + 1e-6
            painter.setPen(QPen(QColor(theme.MAUVE if done else theme.LINE), 2))
            painter.setBrush(QColor(theme.MAUVE) if i == current else QColor(theme.BLUSH if done else theme.SURFACE))
            painter.drawEllipse(QPointF(x, y), 7 if i == current else 5.5, 7 if i == current else 5.5)
            painter.setPen(QColor(theme.INK if i == current else theme.INK_3))
            painter.setFont(font(theme.SANS_SEMIBOLD if i == current else theme.SANS_MEDIUM, 8))
            painter.drawText(QRectF(x - 60, y + 11, 120, 20), Qt.AlignHCenter | Qt.AlignTop, self._names[i])


class RecoveryAnimation(QFrame):
    """Step 2: the receiver's stages, played automatically, in the original app's plots.

    The same two views as the main app's Step-by-step tab: the
    **constellation** (one point per symbol once timing is found, with
    rings where symbols belong and red crosses on symbols read wrongly)
    and the **wave over time** (what arrived, against what was sent where
    that is known).  Instead of clicking through the stages, it plays
    them one after another, with one sentence of explanation each; the
    stage buttons still work, and pause the playback where clicked.

    The constellation glides from one stage to the next on its own (the
    original plot already animates), so the movie is smooth without a
    second animation system.  With animations off it shows the end at once.
    """

    stage_changed = Signal(int)
    finished = Signal()

    #: Milliseconds spent on each stage: the glide, then a rest to read.
    MOVE_MS = 700
    HOLD_MS = 1900

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._recovery: dict | None = None
        self._stages: list[dict] = []
        self._index = -1
        self._playing = False
        self._speed = 1.0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance)

        step = QLabel("STEP 2  ·  RECOVERING THE MESSAGE")
        step.setObjectName("Muted")

        # Stage buttons, numbered: the order means something here.
        self._buttons_row = QHBoxLayout()
        self._buttons_row.setSpacing(8)
        self._buttons: list[PillButton] = []

        # The story box, as in the Step-by-step tab.
        self.stage_title = QLabel("Waiting for a signal")
        self.stage_title.setObjectName("CardTitle")
        self.stage_text = QLabel("Once the AI has named the signal, the receiver repairs it here, stage by stage.")
        self.stage_text.setObjectName("Hint")
        self.stage_text.setWordWrap(True)
        self.stage_text.setMinimumHeight(38)
        self.stage_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        story = QFrame()
        story.setObjectName("Story")
        fill = theme.page_fill("Step by step")
        story.setStyleSheet(
            f"QFrame#Story {{ background: {theme.tint(fill, 0.55)}; border: 1px solid {theme.tint(fill, 0.2)};"
            f" border-radius: {theme.RADIUS_CARD}px; }} QFrame#Story QLabel {{ background: transparent; }}")
        story_layout = QVBoxLayout(story)
        story_layout.setContentsMargins(theme.PAD, 12, theme.PAD, 12)
        story_layout.setSpacing(4)
        story_layout.addWidget(self.stage_title)
        story_layout.addWidget(self.stage_text)

        from ...gui.widgets import ConstellationPlot, WaveformPlot

        self.points = ConstellationPlot("The signal at this stage", "")
        self.wave = WaveformPlot("Over time", "In-phase part, first 24 symbols.")
        for plot in (self.points, self.wave):
            plot.setObjectName("Plain")
            plot.setStyleSheet("QFrame#Plain { background: transparent; border: none; }")
        for plot in (self.points, self.wave):
            plot.setMinimumHeight(480)
        plots = QHBoxLayout()
        plots.setSpacing(theme.GAP)
        plots.addWidget(self.points, 1)
        plots.addWidget(self.wave, 1)

        self.replay = PillButton("Replay", "secondary")
        self.pause = PillButton("Pause", "quiet")
        self.speed = PillButton("Speed 1×", "quiet")
        self.replay.clicked.connect(self.restart)
        self.pause.clicked.connect(self.toggle)
        self.speed.clicked.connect(self._cycle_speed)
        for button in (self.replay, self.pause, self.speed):
            button.setEnabled(False)
        controls = QHBoxLayout()
        controls.addWidget(self.replay)
        controls.addWidget(self.pause)
        controls.addStretch(1)
        controls.addWidget(self.speed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.PAD, theme.PAD - 4, theme.PAD, theme.PAD - 4)
        layout.setSpacing(10)
        layout.addWidget(step)
        layout.addLayout(self._buttons_row)
        layout.addWidget(story)
        layout.addLayout(plots, 1)
        layout.addLayout(controls)

    # -- public ---------------------------------------------------------
    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def current_stage(self) -> int:
        return self._index

    @property
    def n_stages(self) -> int:
        return len(self._stages)

    def clear(self) -> None:
        self._timer.stop()
        self._playing = False
        self._recovery = None
        self._stages = []
        self._index = -1
        self._set_buttons([])
        self.stage_title.setText("Waiting for the AI")
        self.stage_text.setText("")
        for button in (self.replay, self.pause, self.speed):
            button.setEnabled(False)

    def play(self, recovery: dict) -> None:
        """Start the playback for one analysis result."""
        self._recovery = recovery
        self._stages = list(recovery["stages"])
        self._set_buttons([s["title"] for s in self._stages])
        for button in (self.replay, self.pause, self.speed):
            button.setEnabled(True)
        self.restart()

    def restart(self) -> None:
        if not self._stages:
            return
        self._timer.stop()
        if not Motion.enabled:
            # Still emit every stage, in order, so listeners stay in step.
            for index in range(len(self._stages)):
                self.show_stage(index)
            self._finish()
            return
        self._playing = True
        self.pause.setText("Pause")
        self.show_stage(0)
        self._schedule()

    def toggle(self) -> None:
        if not self._stages:
            return
        if self._playing:
            self._playing = False
            self._timer.stop()
            self.pause.setText("Resume")
        else:
            if self._index >= len(self._stages) - 1:
                self.restart()
                return
            self._playing = True
            self.pause.setText("Pause")
            self._schedule()

    def jump_to_end(self) -> None:
        self._timer.stop()
        if self._stages:
            self.show_stage(len(self._stages) - 1)
            self._finish()

    def show_stage(self, index: int) -> None:
        """Draw one stage in both plots and tell its story."""
        if not self._stages or self._recovery is None:
            return
        index = max(0, min(int(index), len(self._stages) - 1))
        self._index = index
        stage = self._stages[index]
        recovery = self._recovery
        n = len(self._stages)
        last = index == n - 1
        for k, button in enumerate(self._buttons):
            button.setChecked(k == index)
        self.stage_title.setText(f"Stage {index + 1} of {n}  ·  {stage['title']}")
        self.stage_text.setText(stage["text"])
        fade_in(self.stage_text)

        samples = stage.get("samples") or stage["points"]
        sps = int(recovery.get("samples_per_symbol", 1))
        colour = theme.receiver_colour(recovery.get("receiver", "classical"))
        reference = recovery.get("reference_symbols")
        constellation = recovery.get("constellation_raw") or recovery.get("constellation")
        if stage.get("rate", "symbol" if not stage.get("per_sample") else "sample") == "sample":
            self.points.set_title("The signal at this stage", "Every sample, so the symbols blur together.")
            self.points.show_points(samples, theme.RAW, cloud=True)
            self.wave.show_wave(samples, sps, colour=theme.INK, trace_label="at this stage")
        else:
            wrong = recovery.get("wrong_symbols") if last else None
            if last and wrong is not None and any(wrong):
                caption = f"{sum(wrong)} symbols read wrongly, shown as red crosses."
            elif last and wrong is not None:
                caption = "Every checked symbol read correctly."
            else:
                caption = "One point per symbol; rings are where they should land."
            self.points.set_title("The symbols at this stage", caption)
            self.points.show_points(samples, colour, reference=constellation, wrong=wrong)
            if reference is not None:
                self.wave.show_wave(samples, 1, reference=reference, colour=colour, dots=True,
                                    reference_label="what was sent", trace_label="at this stage")
            else:
                self.wave.show_wave(samples, 1, colour=colour, dots=True, trace_label="at this stage")
        self.stage_changed.emit(index)

    # -- playback -------------------------------------------------------
    def _set_buttons(self, titles: list[str]) -> None:
        for button in self._buttons:
            self._buttons_row.removeWidget(button)
            button.deleteLater()
        self._buttons = []
        short = {"What arrived": "Arrived", "Remove Doppler": "Doppler", "Filter noise": "Filter",
                 "Find timing": "Timing", "Fix size": "Size", "Lock phase": "Phase", "Undo echoes": "Echoes"}
        for number, title in enumerate(titles):
            button = PillButton(f"{number + 1}  {short.get(title, title)}", "step", accent=theme.page_fill("Step by step"),
                                accent_ink=theme.page_accent("Step by step"))
            button.setMinimumWidth(button.sizeHint().width())
            button.setToolTip(title)
            button.clicked.connect(lambda _c=False, k=number: self._clicked(k))
            self._buttons_row.addWidget(button, 1)
            self._buttons.append(button)

    def _clicked(self, index: int) -> None:
        # Clicking a stage takes over from the autoplay.
        self._timer.stop()
        self._playing = False
        self.pause.setText("Resume")
        self.show_stage(index)
        if index == len(self._stages) - 1:
            self._finish()

    def _schedule(self) -> None:
        self._timer.start(int((self.MOVE_MS + self.HOLD_MS) / self._speed))

    def _advance(self) -> None:
        if not self._playing:
            return
        if self._index >= len(self._stages) - 1:
            self._finish()
            return
        self.show_stage(self._index + 1)
        if self._index >= len(self._stages) - 1:
            self._finish()
        else:
            self._schedule()

    def _finish(self) -> None:
        self._timer.stop()
        self._playing = False
        self.pause.setText("Play again")
        self.finished.emit()

    def _cycle_speed(self) -> None:
        self._speed = {1.0: 2.0, 2.0: 0.5, 0.5: 1.0}[self._speed]
        self.speed.setText("Speed " + {1.0: "1×", 2.0: "2×", 0.5: "½×"}[self._speed])


# ======================================================================
# The recovered message
# ======================================================================
class BitsCard(QFrame):
    """The answer at the end: how many bits, how trustworthy, and were they right."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._text = ""
        self._typed = 0
        self._typer = QTimer(self)
        self._typer.setInterval(18)
        self._typer.timeout.connect(self._type)

        step = QLabel("THE RECOVERED MESSAGE")
        step.setObjectName("Muted")
        self.by = QLabel("")
        self.by.setObjectName("Hint")
        self.count = CountUpLabel(lambda v: f"{v:,.0f} bits")
        self.count.setObjectName("BigNumber")
        self.trust = QLabel("")
        self.trust.setObjectName("Hint")
        self.check = QLabel("")
        self.check.setWordWrap(True)
        self.detail = QLabel("")
        self.detail.setObjectName("Hint")
        self.detail.setWordWrap(True)
        self.bits = QLabel("")
        self.bits.setWordWrap(True)
        self.bits.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.bits.setStyleSheet(f"font-family: '{theme.MONO}'; font-size: 10pt; color: {theme.INK_2};"
                                f" background: {theme.SUNK}; border-radius: 8px; padding: 8px 10px;")

        left = QVBoxLayout()
        left.setSpacing(4)
        left.addWidget(step)
        left.addWidget(self.count)
        left.addWidget(self.by)
        left.addWidget(self.trust)
        right = QVBoxLayout()
        right.setSpacing(6)
        right.addWidget(self.check)
        right.addWidget(self.detail)
        right.addWidget(self.bits)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.PAD, theme.PAD - 4, theme.PAD, theme.PAD - 4)
        layout.setSpacing(10)
        layout.addLayout(left)
        layout.addLayout(right)
        layout.addStretch(1)
        self.reset()

    def reset(self) -> None:
        self._typer.stop()
        self.count.set_value(0)
        self.by.setText("Appears when the recovery finishes.")
        self.trust.setText("")
        self.check.setText("")
        self.detail.setText("")
        self.bits.setText("")

    def show_result(self, recovery: dict) -> None:
        receiver = {"hybrid": "the hybrid receiver (maths + AI)", "classical": "the classical receiver"}
        self.by.setText(f"Read as {recovery['modulation']} by {receiver.get(recovery['receiver'], recovery['receiver'])}.")
        self.count.set_value(recovery["n_bits"])
        self.trust.setText(f"The receiver is sure of {recovery['trusted_fraction']:.0%} of them.")
        check = recovery["check"]
        if check.get("available"):
            errors, total = check["bit_errors"], check["n_bits"]
            good = errors == 0
            mark = "✓" if good else "✕"
            self.check.setText(f"{mark}  Checked: {errors:,} of {total:,} bits wrong ({100 * errors / max(total, 1):.2f}%)")
            self.check.setStyleSheet(f"color: {theme.GOOD if good else theme.CRITICAL};"
                                     f" font-family: '{theme.SANS_SEMIBOLD}'; font-size: 12pt;")
        else:
            self.check.setText("Not checked bit by bit")
            self.check.setStyleSheet(f"color: {theme.INK_2}; font-family: '{theme.SANS_SEMIBOLD}'; font-size: 12pt;")
        self.detail.setText(check.get("detail", ""))
        fade_in(self.check)
        preview = recovery["bits_preview"]
        self._text = " ".join(preview[i : i + 8] for i in range(0, len(preview), 8)) + "  …"
        self._typed = 0
        if Motion.enabled:
            self._typer.start()
        else:
            self.bits.setText(self._text)

    def _type(self) -> None:
        self._typed = min(self._typed + 3, len(self._text))
        self.bits.setText(self._text[: self._typed])
        if self._typed >= len(self._text):
            self._typer.stop()


__all__ = ["BitsCard", "ClassifierCard", "ProbabilityBars", "RecoveryAnimation", "StageStrip", "palette"]
