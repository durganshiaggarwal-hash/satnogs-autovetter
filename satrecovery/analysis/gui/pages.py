"""The two tabs of the analysis app: a real RML24 signal, or a simulated one.

DESIGN NOTE: TEMPLATE METHOD, AGAIN
-----------------------------------
:class:`AnalysisPage` does everything the two tabs have in common: the
layout, the three result cards, and the choreography -- ask the backend,
show the AI's answer, play the recovery, then reveal the bits.  The
subclasses fill in only what differs: which controls sit on the left
(:meth:`_build_controls`) and what to send (:meth:`_request`).  Both are
abstract, so a third kind of signal is one more small subclass.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...gui import theme
from ...gui.motion import PillButton, ToggleSwitch, fade_in
from ...gui.pages.base import Page
from ...gui.widgets import LabeledSlider, card
from ..inputs import CHECKABLE, usually_checkable
from .pdf import AnalysisReportWriter
from .widgets import BitsCard, ClassifierCard, RecoveryAnimation

#: Pause between the AI's answer and the start of the movie, so the eye
#: has a moment to read the answer first.
REVEAL_DELAY_MS = 900


class _Column(QWidget):
    """A fixed-width column that asks for the full height its wrapped text needs.

    Without this, Qt sizes a column of word-wrapped labels as if the text
    sat on one line, and on a short window the labels get squeezed on top
    of the buttons.  Measuring the height at the column's real width fixes it.
    """

    def _needed(self) -> int:
        layout = self.layout()
        if layout is None:
            return 0
        width = self.width() if self.width() > 0 else self.maximumWidth()
        return layout.totalHeightForWidth(width) if layout.hasHeightForWidth() else layout.totalMinimumSize().height()

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(super().minimumSizeHint().width(), self._needed())

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(super().sizeHint().width(), max(super().sizeHint().height(), self._needed()))


class AnalysisPage(Page):
    """Controls on the left; the AI's answer, the recovery and the bits on the right."""

    def __init__(self, client, parent: QWidget | None = None) -> None:
        super().__init__(client, parent)
        self._status: dict | None = None
        self._pending: dict | None = None

        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        self.banner.setStyleSheet(
            f"background: {theme.tint(theme.PEACH, 0.5)}; color: {theme.INK}; border-radius: 10px;"
            f" padding: 10px 14px;")
        # Shown when an analysis is opened again from the History tab.
        self.reopened = QLabel("")
        self.reopened.setVisible(False)
        self.reopened.setStyleSheet(
            f"background: {theme.tint(theme.SAGE, 0.4)}; color: {theme.INK}; border-radius: 10px;"
            f" padding: 8px 14px; font-family: '{theme.SANS_SEMIBOLD}';")

        controls = _Column()
        controls.setFixedWidth(300)
        column = QVBoxLayout(controls)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(theme.GAP)
        self.analyse_button = PillButton(self.ACTION, "primary")
        self.analyse_button.clicked.connect(self.analyse)
        self.save_button = PillButton("Save report (PDF)", "secondary")
        self.save_button.setToolTip("Save this analysis as a PDF: the AI's answer, the recovered bits and the pictures.")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_report)
        self.test_button = PillButton("Save as test file", "secondary")
        self.test_button.setToolTip("Save this signal together with its answer key (its true type and the bits "
                                    "that were sent), to load later in the Your own file tab.")
        self.test_button.setEnabled(False)
        self.test_button.setVisible(self.CAN_SAVE_TEST_FILE)
        self.test_button.clicked.connect(self.save_test_file)
        self._build_controls(column)
        column.addWidget(self.analyse_button)
        column.addWidget(self.save_button)
        column.addWidget(self.test_button)
        column.addStretch(1)

        self.classifier = ClassifierCard()
        self.classifier.setFixedWidth(340)
        self.animation = RecoveryAnimation()
        self.animation.finished.connect(self._reveal_bits)
        self.result = BitsCard()

        # One page that scrolls, so nothing is squashed: the choice and the
        # two answers side by side on top, the recovery full width below,
        # with room for its plots to be read properly.
        top = QHBoxLayout()
        top.setSpacing(theme.GAP)
        top.addWidget(controls, 0, Qt.AlignTop)
        top.addWidget(self.classifier, 0)
        top.addWidget(self.result, 1)

        content = QWidget()
        stack = QVBoxLayout(content)
        stack.setContentsMargins(0, 0, 14, 8)
        stack.setSpacing(theme.GAP)
        # The banner, with room beside it for a button that fixes what it says.
        self.banner_layout = QHBoxLayout()
        self.banner_layout.setSpacing(theme.GAP)
        self.banner_layout.addWidget(self.banner, 1)
        stack.addLayout(self.banner_layout)
        stack.addWidget(self.reopened)
        stack.addLayout(top)
        stack.addWidget(self.animation)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 18, 14, 10)
        layout.setSpacing(10)
        layout.addWidget(self.heading(self.title, self.HINT))
        layout.addWidget(self.scrolling(content), 1)

    # -- what subclasses supply -----------------------------------------
    ACTION = "Analyse"
    HINT = ""
    #: Can this tab's signals be saved as test files (signal + answer key)?
    CAN_SAVE_TEST_FILE = False

    @abstractmethod
    def _build_controls(self, column: QVBoxLayout) -> None:
        """Add this tab's controls to the left-hand column."""

    @abstractmethod
    def _request(self) -> Callable[[], dict] | None:
        """A call that asks the backend for one analysis, or None if not ready."""

    # -- shared ---------------------------------------------------------
    def refresh(self) -> None:
        self.tasks.run(self.client.analysis_status, self._on_status, self.report_error)

    def _on_status(self, status: dict) -> None:
        self._status = status
        if not status["classifier"]["available"]:
            self.banner.setText(
                "The AI classifier has not been trained yet, so there is nothing to ask. Train it once "
                "(a few minutes on the GPU):   python scripts/train_classifier.py \"path to RML24\"   "
                "and then restart this app.")
            self.banner.setVisible(True)
        else:
            accuracy = status["classifier"].get("test_accuracy")
            if accuracy is not None:
                self.analyse_button.setToolTip(f"The classifier names {accuracy:.0%} of held-out RML24 recordings correctly.")
        self._status_ready(status)

    def _status_ready(self, status: dict) -> None:
        """Hook for subclasses once the backend's status is known."""

    def analyse(self) -> None:
        call = self._request()
        if call is None:
            return
        self.analyse_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.test_button.setEnabled(False)
        self.reopened.setVisible(False)
        self.classifier.thinking()
        self.animation.clear()
        self.result.reset()
        self.tasks.run(call, self._on_result, self._on_failed)

    def _on_failed(self, error) -> None:
        self.analyse_button.setEnabled(True)
        self.classifier.answer.setText("—")
        self.classifier.sure.setText("Could not analyse this signal.")
        self.report_error(error)

    def _on_result(self, result: dict) -> None:
        self._pending = result
        self.analyse_button.setEnabled(True)
        self.classifier.show_result(result["classification"])
        self.classifier.show_trust(result.get("trust"))
        delay = REVEAL_DELAY_MS if self.animation.isVisible() and _motion_on() else 0
        QTimer.singleShot(delay, lambda: self.animation.play(result["recovery"]))
        if result["classification"].get("true_label") is None:
            verdict = "the true type is not known"
        else:
            verdict = "right" if result["classification"]["correct"] else "wrong"
        self.report(f"The AI says {result['classification']['label']} "
                    f"({result['classification']['confidence']:.0%} sure) -- {verdict}.")

    def show_saved(self, result: dict) -> None:
        """Show an analysis again, as it was -- opened from the History tab."""
        from datetime import datetime

        saved = result.get("history", {}).get("saved_at")
        when = datetime.fromtimestamp(saved).strftime("%d %B, %H:%M") if saved else "earlier"
        self.save_button.setEnabled(False)
        self.test_button.setEnabled(False)
        self.animation.clear()
        self.result.reset()
        self.reopened.setText(f"Opened from History: analysed on {when}.")
        self.reopened.setVisible(True)
        fade_in(self.reopened)
        self._on_result(result)

    def _reveal_bits(self) -> None:
        if self._pending is not None:
            self.result.show_result(self._pending["recovery"])
            self.save_button.setEnabled(True)
            self.test_button.setEnabled(self.CAN_SAVE_TEST_FILE and self.test_file_source() is not None)

    # -- test files (signal + answer key) ---------------------------------
    def test_file_source(self) -> dict | None:
        """What the backend needs to rebuild the latest signal, or None if it cannot be saved."""
        if self._pending is None:
            return None
        source = self._pending.get("input", {})
        if source.get("kind") == "recorded" and not source.get("file"):
            return source
        if source.get("kind") == "simulated" and "seed" in source:
            return source
        return None

    def save_test_file(self) -> None:
        source = self.test_file_source()
        if source is None:
            return
        self.tasks.run(lambda: self.client.make_test_file(source), self._ask_where, self.report_error)

    #: The two choices in the save window.
    NPZ_FILTER = "Test file (*.npz)"
    CSV_FILTER = "Two CSV files for Excel (*.csv)"

    def _ask_where(self, answer: dict) -> None:
        suggested = str(Path.home() / answer["name"])
        path, chosen = QFileDialog.getSaveFileName(self, "Save as test file", suggested,
                                                   f"{self.NPZ_FILTER};;{self.CSV_FILTER}")
        if not path:
            return
        if chosen == self.CSV_FILTER or path.lower().endswith(".csv"):
            self.write_test_csv(path, answer)
        else:
            self.write_test_file(path, answer)

    def write_test_csv(self, path: str, answer: dict) -> list[str] | None:
        """Write a test file as ``..._signal.csv`` + ``..._bits.csv``; returns where they went."""
        import base64

        from ..csvpair import CsvTestFile

        target = Path(path)
        try:
            written = CsvTestFile.write_pair(target.parent, target.name, base64.b64decode(answer["content_base64"]))
        except (OSError, ValueError, KeyError) as error:
            self.report_error(f"Could not save the CSV files: {error}")
            return None
        self.notify.emit(f"Saved for Excel: {written[0].name} and {written[1].name}", "success")
        self.report(f"Saved {written[0]} and {written[1]}")
        return [str(p) for p in written]

    def write_test_file(self, path: str, answer: dict) -> str | None:
        """Write a test file the backend made; returns where it went."""
        import base64

        target = Path(path)
        if target.suffix.lower() != ".npz":
            target = target.with_suffix(".npz")
        try:
            target.write_bytes(base64.b64decode(answer["content_base64"]))
        except (OSError, ValueError, KeyError) as error:
            self.report_error(f"Could not save the test file: {error}")
            return None
        info = answer.get("info") or {}
        has_key = info.get("answer_key") in ("simulator", "phase3")
        self.notify.emit(f"Test file saved: {target.name}" + ("" if has_key else " (true type only, no bit answer key)"),
                         "success")
        self.report(f"Test file saved to {target}")
        return str(target)

    # -- the PDF report --------------------------------------------------
    def save_report(self) -> None:
        """Ask where to save, then write the report there."""
        if self._pending is None:
            return
        from datetime import datetime

        suggested = str(Path.home() / f"signal_report_{datetime.now():%Y%m%d_%H%M%S}.pdf")
        path, _ = QFileDialog.getSaveFileName(self, "Save report", suggested, "PDF files (*.pdf)")
        if path:
            self.write_report(path)

    def write_report(self, path: str) -> str | None:
        """Write the latest analysis to ``path`` as a PDF; returns where it went."""
        if self._pending is None:
            return None
        if self.animation.n_stages and self.animation.current_stage != self.animation.n_stages - 1:
            self.animation.show_stage(self.animation.n_stages - 1)   # the picture shows the final result
        pictures = [("The recovery, after the last cleaning step", self.animation.grab())]
        try:
            written = AnalysisReportWriter(self._pending, pictures).write(path)
        except (OSError, ValueError) as error:
            self.report_error(f"Could not save the report: {error}")
            return None
        self.notify.emit(f"Report saved: {written.name}", "success")
        self.report(f"Report saved to {written}")
        return str(written)

    @property
    def last_result(self) -> dict | None:
        return self._pending


def _motion_on() -> bool:
    from ...gui.motion import Motion

    return Motion.enabled


# ======================================================================
class RealSignalPage(AnalysisPage):
    """Analyse a real recording from RML24."""

    CAN_SAVE_TEST_FILE = True

    #: Asks the window to let the user choose the RML24 folder.
    choose_dataset = Signal()

    def __init__(self, client, parent: QWidget | None = None) -> None:
        super().__init__(client, parent)
        # Beside the "no dataset" banner, where the eye already is.
        self.banner_layout.addWidget(self.dataset_button, 0, Qt.AlignVCenter)

    ACTION = "Analyse this signal"
    HINT = "Pick a real satellite recording from RML24. The AI says what it is; the receiver reads it."

    @property
    def title(self) -> str:
        return "Real signal"

    def _build_controls(self, column: QVBoxLayout) -> None:
        box, layout = card("Pick a real recording",
                           "1.3 million real recordings. Narrow them down, or leave everything on Any.")
        self.kind = QComboBox()
        self.quality = QComboBox()
        self.rate = QComboBox()
        self.kind.addItem("Any type", None)
        self.quality.addItem("Any quality", None)
        for snr in range(20, -21, -2):
            words = "very clean" if snr >= 16 else "clean" if snr >= 8 else "noisy" if snr >= 0 else "very noisy"
            self.quality.addItem(f"{snr:+d} dB  ({words})", float(snr))
        self.rate.addItem("Any speed", None)
        for rate in (100_000, 250_000, 500_000):
            self.rate.addItem(f"{rate // 1000} kbaud", float(rate))
        for label, combo in (("Signal type", self.kind), ("Signal quality", self.quality), ("Speed", self.rate)):
            combo.setCursor(Qt.PointingHandCursor)
            layout.addWidget(QLabel(label))
            layout.addWidget(combo)
        self.pick = PillButton("Pick another recording", "secondary")
        self.pick.clicked.connect(self.pick_recording)
        # Changing a filter picks a matching recording at once, so what is
        # analysed always matches what the dropdowns say.  (Before, the
        # old recording stayed chosen until the button was pressed, and a
        # user who chose "8PSK" could analyse a 64QAM recording.)
        for combo in (self.kind, self.quality, self.rate):
            combo.currentIndexChanged.connect(lambda _i: self._filters_changed())
        for combo in (self.kind, self.rate):
            combo.currentIndexChanged.connect(lambda _i: self._mark_checkable_qualities())
        layout.addWidget(self.pick)
        note = QLabel("Bits can be checked one by one only for BPSK, QPSK and 8PSK, on clean enough "
                      "recordings: RML24 does not include the bits that were sent, and Phase 3 can only work "
                      "them out reliably then. Measured on RML24, checks usually work from:\n"
                      "  100 kbaud:  BPSK +6 dB,  QPSK +6 dB,  8PSK +12 dB\n"
                      "  250 kbaud:  BPSK +8 dB,  QPSK +10 dB,  8PSK +16 dB\n"
                      "  500 kbaud:  BPSK +12 dB,  QPSK +16 dB,  8PSK hardly ever\n"
                      "Qualities marked ✓ in the list can usually be checked; there, a recording that can "
                      "be checked is picked when one is found.")
        note.setObjectName("Muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.dataset_button = PillButton("Choose dataset folder…", "primary")
        self.dataset_button.setToolTip("Pick the RML24_IQdata folder. The app remembers it for next time.")
        self.dataset_button.clicked.connect(self.choose_dataset.emit)
        self.dataset_button.setVisible(False)
        self.chosen = QLabel("No recording chosen yet.")
        self.chosen.setObjectName("Hint")
        self.chosen.setWordWrap(True)
        layout.addWidget(self.chosen)
        column.addWidget(box)
        self._row: int | None = None

    def _status_ready(self, status: dict) -> None:
        dataset = status["dataset"]
        if not dataset.get("available"):
            self.banner.setText("No RML24 dataset is open yet. Choose the RML24_IQdata folder once and the app "
                                "remembers it. The Simulated signal and Your own file tabs work without it.")
            self.banner.setToolTip(dataset.get("detail", ""))
            self.banner.setVisible(True)
            self.dataset_button.setVisible(True)
            self.pick.setEnabled(False)
            self.analyse_button.setEnabled(False)
            return
        # A dataset may have just been chosen: undo the "no dataset" state.
        self.dataset_button.setVisible(False)
        if status["classifier"]["available"]:
            self.banner.setVisible(False)
        self.pick.setEnabled(True)
        self.analyse_button.setEnabled(True)
        names = dataset.get("modulations", [])
        classes = status["classifier"]["classes"]
        for name in names:
            if name in classes and self.kind.findData(name) < 0:
                checkable = name in CHECKABLE
                self.kind.addItem(f"{name}   {'(bits checked)' if checkable else '(bits not checkable)'}", name)
        if self._row is None:
            self.pick_recording()

    def _mark_checkable_qualities(self) -> None:
        """Put a ✓ beside the qualities whose bits can usually be checked."""
        kind, rate = self.kind.currentData(), self.rate.currentData()
        for i in range(1, self.quality.count()):
            snr = self.quality.itemData(i)
            words = "very clean" if snr >= 16 else "clean" if snr >= 8 else "noisy" if snr >= 0 else "very noisy"
            text = f"{snr:+.0f} dB  ({words})"
            if kind in CHECKABLE and rate is not None and usually_checkable(kind, snr, rate):
                text += "   ✓ bits checked"
            self.quality.setItemText(i, text)

    def _picked_modulation(self) -> str:
        """The true type of the recording now chosen (for tests; never shown before the AI answers)."""
        return getattr(self, "_picked", "")

    def _filters_changed(self) -> None:
        if self._status is not None and self._status["dataset"].get("available"):
            self._row = None
            self.chosen.setText("Picking a recording that matches…")
            self.pick_recording()

    def pick_recording(self) -> None:
        kind, quality, rate = self.kind.currentData(), self.quality.currentData(), self.rate.currentData()
        seed = int(np.random.default_rng().integers(0, 2**31))
        # Only the newest pick counts: changing two dropdowns quickly sends
        # two requests, and the older answer must not arrive last and win.
        self._pick_token = getattr(self, "_pick_token", 0) + 1
        token = self._pick_token
        self.tasks.run(lambda: self.client.pick_recordings(kind, quality, rate, 1, seed),
                       lambda rows: self._on_picked(rows) if token == self._pick_token else None,
                       self.report_error)

    def _on_picked(self, rows: list[dict]) -> None:
        if not rows:
            self.chosen.setText("No recording matches those choices.")
            self._row = None
            return
        rec = rows[0]
        self._row = rec["row"]
        self._picked = rec["modulation"]
        self.chosen.setText(
            f"Recording #{rec['row']:,}  ·  {rec['symbol_rate'] / 1000:.0f} kbaud  ·  labelled {rec['snr_label_db']:+.0f} dB."
            + ("\n✓ Its bits can be checked one by one." if rec.get("checkable") else
               "\nIts bits cannot be checked one by one." if self.kind.currentData() in CHECKABLE else "")
            + "\nIts type stays hidden until the AI has answered.")
        fade_in(self.chosen)

    def _request(self):
        if self._row is None:
            self.report("Pick a recording first: none matches the current choices yet.")
            return None
        row = self._row
        return lambda: self.client.analyse_recording(row)

    def open_row(self, row: int, snr_label_db: float | None = None) -> None:
        """Analyse one particular recording -- e.g. a mistake from the report card."""
        self._row = int(row)
        label = f"  ·  labelled {snr_label_db:+.0f} dB" if snr_label_db is not None else ""
        self.chosen.setText(f"Recording #{int(row):,}{label}, opened from the AI's report card.")
        self.analyse()


# ======================================================================
class SimulatedSignalPage(AnalysisPage):
    """Make a signal with the simulator, damage it, and analyse it."""

    CAN_SAVE_TEST_FILE = True

    ACTION = "Make a signal and analyse it"
    HINT = "Make your own signal and damage it with the sliders. The AI says what it is; the receiver reads it."

    @property
    def title(self) -> str:
        return "Simulated signal"

    def _build_controls(self, column: QVBoxLayout) -> None:
        box, layout = card("Make a signal", "A fresh one every time you press the button.")
        self.kind = QComboBox()
        self.kind.setCursor(Qt.PointingHandCursor)
        self.rate = QComboBox()
        self.rate.setCursor(Qt.PointingHandCursor)
        for rate in (100_000, 250_000, 500_000):
            self.rate.addItem(f"{rate // 1000} kbaud", float(rate))
        self.rate.setCurrentIndex(1)
        self.strength = LabeledSlider("Signal strength", -10, 25, 1, 8, "dB", 0,
                                      "Signal-to-noise ratio: higher is cleaner.")
        self.doppler = LabeledSlider("Doppler shift", -2000, 2000, 50, 600, "Hz", 0,
                                     "The satellite's motion shifts the frequency.")
        self.echoes = ToggleSwitch("Echoes on the way down")
        layout.addWidget(QLabel("Signal type"))
        layout.addWidget(self.kind)
        layout.addWidget(QLabel("Speed"))
        layout.addWidget(self.rate)
        layout.addWidget(self.strength)
        layout.addWidget(self.doppler)
        layout.addWidget(self.echoes)
        column.addWidget(box)

    def _status_ready(self, status: dict) -> None:
        if self.kind.count() == 0:
            for name in status["classifier"]["classes"]:
                self.kind.addItem(name, name)
            self.kind.setCurrentIndex(max(self.kind.findData("QPSK"), 0))

    def request_body(self) -> dict[str, Any]:
        return {
            "modulation": self.kind.currentData() or "QPSK",
            "symbol_rate": self.rate.currentData(),
            "conditions": {
                "snr_db": self.strength.value,
                "freq_offset_hz": self.doppler.value,
                "fading": "multipath_rician" if self.echoes.isChecked() else "none",
            },
        }

    def _request(self):
        body = self.request_body()
        return lambda: self.client.analyse_simulated(body)


# ======================================================================
class FilePage(AnalysisPage):
    """Analyse a signal from the user's own file: give it data, get the answer."""

    ACTION = "Analyse this file"
    HINT = "Give the app your own signal file. The AI says what it is; the receiver reads it."

    #: Where the example files ship, next to the project.
    EXAMPLES = Path(__file__).resolve().parents[3] / "examples"

    @property
    def title(self) -> str:
        return "Your own file"

    def _build_controls(self, column: QVBoxLayout) -> None:
        box, layout = card("Your signal file", "Choose a file of signal samples, then press Analyse.")
        self.choose = PillButton("Choose file…", "secondary")
        self.choose.clicked.connect(self.choose_file)
        self.chosen = QLabel("No file chosen yet.")
        self.chosen.setObjectName("Hint")
        self.chosen.setWordWrap(True)
        # Shown when the file is a test file: what its answer key is.
        self.test_info = QLabel("")
        self.test_info.setWordWrap(True)
        self.test_info.setVisible(False)
        self.test_info.setStyleSheet(
            f"background: {theme.tint(theme.SAGE, 0.4)}; color: {theme.INK}; border-radius: 8px; padding: 8px 10px;")
        self.choose_bits = PillButton("Choose bits file…", "secondary")
        self.choose_bits.setToolTip("The answer key for a CSV signal: a ..._bits.csv file (see the walkthrough).")
        self.choose_bits.clicked.connect(self.choose_bits_file)
        self.choose_bits.setVisible(False)
        self.export_csv = PillButton("Export as CSV (for Excel)", "secondary")
        self.export_csv.setToolTip("Write this test file as ..._signal.csv and ..._bits.csv, to open in Excel.")
        self.export_csv.clicked.connect(self.export_as_csv)
        self.export_csv.setVisible(False)
        self.rate = QComboBox()
        self.rate.setCursor(Qt.PointingHandCursor)
        for rate in (100_000, 250_000, 500_000):
            self.rate.addItem(f"{rate // 1000} kbaud", float(rate))
        self.rate.setCurrentIndex(1)
        note = QLabel("Accepted: a CSV or TXT file with two columns (I and Q), a NumPy .npy file, or a "
                      "recording exported from RML24 (.npz). Sampled at 1 MHz like RML24, with at least "
                      "2,000 samples; up to the first 2,048 are analysed.\n"
                      "Speed: the symbol rate the signal was sent at (an RML24 export already knows it).\n"
                      "A test file (saved with \"Save as test file\" on the Real or Simulated tab) also "
                      "holds its answer key, so the AI and the bits are graded. So does a pair of CSV files, "
                      "..._signal.csv with ..._bits.csv beside it.\n"
                      "To try it, use the files in the project's examples folder.")
        note.setObjectName("Muted")
        note.setWordWrap(True)
        layout.addWidget(self.choose)
        layout.addWidget(self.chosen)
        layout.addWidget(self.test_info)
        layout.addWidget(self.choose_bits)
        layout.addWidget(self.export_csv)
        layout.addWidget(QLabel("Speed"))
        layout.addWidget(self.rate)
        layout.addWidget(note)
        column.addWidget(box)
        self._path: Path | None = None
        self._bits_path: Path | None = None

    def choose_file(self) -> None:
        start = str(self.EXAMPLES if self.EXAMPLES.is_dir() else Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Choose a signal file", start,
                                              "Signal files (*.csv *.txt *.npy *.npz);;All files (*)")
        if path:
            self.set_file(path)

    def set_file(self, path: str | Path) -> None:
        from ..csvpair import CsvTestFile
        from ..testfile import TestFile

        self._path = Path(path)
        self._bits_path = None
        size = self._path.stat().st_size if self._path.exists() else 0
        # Zero-width spaces after underscores let a long file name wrap instead of being cut off.
        shown = self._path.name.replace("_", "_\u200b")
        self.chosen.setText(f"{shown}  ·  {size / 1024:,.0f} KB")
        fade_in(self.chosen)
        info = TestFile.describe_path(self._path)
        is_text = self._path.suffix.lower() in (".csv", ".txt")
        self.choose_bits.setVisible(is_text)
        self.export_csv.setVisible(info is not None)
        if info is not None:
            self._show_key(TestFile.summary(info), True, info["symbol_rate"])
            return
        partner = CsvTestFile.partner(self._path) if is_text else None
        if partner is not None:
            self.set_bits_file(partner)
        else:
            self.test_info.setVisible(False)
            self.rate.setEnabled(True)

    def choose_bits_file(self) -> None:
        start = str(self._path.parent if self._path else Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Choose the bits file (the answer key)", start,
                                              "Bits files (*_bits.csv *.csv *.txt);;All files (*)")
        if path:
            self.set_bits_file(path)

    def set_bits_file(self, path: str | Path) -> None:
        """Use ``path`` as the answer key for the chosen CSV signal."""
        from ..csvpair import CsvTestFile
        from ...core.exceptions import ValidationError

        path = Path(path)
        try:
            info = CsvTestFile.read_bits(path.read_bytes(), path.name)
        except (OSError, ValidationError) as error:
            self._bits_path = None
            self._show_key(f"✕ {path.name}: {error}", False)
            return
        self._bits_path = path
        self._show_key(CsvTestFile.summary(info, path.name), True, info["speed_baud"])

    def _show_key(self, text: str, good: bool, speed: float | None = None) -> None:
        fill = theme.SAGE if good else theme.CRITICAL
        self.test_info.setStyleSheet(f"background: {theme.tint(fill, 0.4 if good else 0.75)}; color: {theme.INK};"
                                     f" border-radius: 8px; padding: 8px 10px;")
        self.test_info.setText(text.replace("_", "_\u200b"))
        self.test_info.setVisible(True)
        fade_in(self.test_info)
        self.rate.setEnabled(speed is None)
        if speed is not None:
            index = self.rate.findData(float(speed))
            if index >= 0:
                self.rate.setCurrentIndex(index)

    def export_as_csv(self) -> None:
        if self._path is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Where should the two CSV files go?", str(self._path.parent))
        if folder:
            self.export_to(folder)

    def export_to(self, folder: str | Path) -> list[str] | None:
        """Write the chosen .npz test file as ``..._signal.csv`` + ``..._bits.csv`` in ``folder``."""
        from ..csvpair import CsvTestFile
        from ...core.exceptions import ValidationError

        if self._path is None:
            return None
        try:
            written = CsvTestFile.write_pair(folder, self._path.name, self._path.read_bytes())
        except (OSError, ValidationError) as error:
            self.report_error(f"Could not export: {error}")
            return None
        self.notify.emit(f"Saved for Excel: {written[0].name} and {written[1].name}", "success")
        return [str(p) for p in written]

    def _request(self):
        if self._path is None:
            self.report("Choose a file first.")
            return None
        path, rate, bits = self._path, self.rate.currentData(), self._bits_path
        return lambda: self.client.analyse_file(path, rate, bits)


__all__ = ["AnalysisPage", "FilePage", "RealSignalPage", "SimulatedSignalPage"]
