"""The "?" button's guide: what each tab does, and what the words mean.

Written for someone who has never studied signals -- an examiner trying
the app, say.  Plain Qt widgets; the words live in two lists at the top
so they are easy to read and change.
"""

from __future__ import annotations

import html

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QButtonGroup, QDialog, QHBoxLayout, QLabel, QScrollArea, QStackedWidget, QVBoxLayout, QWidget

from ...gui import theme
from ...gui.motion import PillButton

#: (tab or button, what it is for)
HOW_TO_USE: list[tuple[str, str]] = [
    ("Simulated signal", "Make your own signal and damage it with the sliders: weaker signal, Doppler shift, "
                         "echoes. Then watch the AI and the receiver deal with it."),
    ("Your own file", "Give the app a file of signal samples (CSV, .npy, or a recording exported from RML24). "
                      "It tells you what the signal is and reads its bits. Try the files in the examples folder."),
    ("History", "Every analysis is saved here by itself. Double-click one to open it again."),
    ("Many at once", "Make a whole group of new signals, for example 100 of 8PSK at +16 dB, let the AI name "
                     "each one, and get one summary table. Save it for Excel."),
    ("How good is the AI?", "Tests the AI (trained on RML24) on new simulated signals of every type and quality, "
                            "and shows where it goes wrong. Double-click a mistake to make it again."),
    ("Compare models", "The model the app uses: what it is, its score on RML24, and how its training went. Other "
                       "kinds of AI appear here once they have been trained on the RML24 dataset."),
    ("Save report (PDF)", "After an analysis, saves a short report of it as a PDF."),
    ("Save as test file", "On the Simulated tab: saves the signal together with its answer key (its "
                          "true type and the bits sent). Load it in Your own file and everything is graded. Choose 'Two CSV files for Excel' in "
                          "the save window to get a signal CSV and a bits CSV you can open in Excel."),
    ("Dark and Animations (top bar)", "Dark switches the colours; Animations turns the moving pictures on or off."),
]

#: (word, what it means)
WORDS: list[tuple[str, str]] = [
    ("Satellite signal", "A radio wave that carries a message (bits) from a satellite down to a dish."),
    ("Modulation", "The \"language\" the signal uses to carry bits. The app knows six: BPSK, QPSK, 8PSK, "
                   "16QAM, 64QAM and 16APSK."),
    ("BPSK, QPSK, 8PSK", "Points placed on a circle: 2, 4 or 8 of them. More points carry more bits at once, "
                         "but are easier to confuse."),
    ("16QAM, 64QAM", "Points placed on a square grid: 16 or 64 of them."),
    ("16APSK", "16 points on two circles, used by satellite TV (DVB-S2)."),
    ("Symbol", "One point sent at a time. Each symbol carries 1 to 6 bits, depending on the modulation."),
    ("Bit", "A single 0 or 1 of the message."),
    ("Constellation", "The dot picture of a signal: one dot per symbol. A clean signal forms neat clusters."),
    ("SNR (dB)", "Signal-to-noise ratio: how clean the signal is. Higher is cleaner. At 0 dB the noise is as "
                 "strong as the signal; below 0 it is stronger."),
    ("kbaud", "Thousands of symbols per second: the speed of the signal."),
    ("Doppler shift", "The satellite's movement shifts the signal's frequency, which makes the dots spin."),
    ("Receiver", "The part that cleans up the signal step by step and reads the bits."),
    ("AI classifier", "A neural network trained on RML24 that names the modulation, with how sure it is."),
    ("CNN, ResNet, LSTM, Transformer", "Kinds of neural network. A CNN looks for short patterns; a ResNet is a "
                                       "deeper CNN with shortcuts; an LSTM reads in order and remembers; a "
                                       "Transformer compares every part with every other part."),
    ("Random Forest", "Not a neural network: hundreds of simple yes/no decision trees that vote."),
    ("Expert features", "12 measurements from classical signal theory that help tell similar types apart."),
    ("RML24", "A public collection of 1.3 million real satellite signal recordings, used to train the AI once. The app keeps only the trained AI, so the dataset is not needed to use it."),
    ("Test file", "A signal file that also holds its answer key, so the app can mark its own answers."),
    ("Warning badge", "Under the AI's answer: Confident, Unsure (weak or close answer, very noisy signal) or "
                      "Unfamiliar (mostly noise, or a signal width that does not fit the speed). It cannot catch "
                      "every mistake; the How good is the AI? tab shows how many it catches."),
    ("Check pattern", "A few known symbols every 32, so the receiver can confirm it is reading the right way up. "
                      "They are not counted in the score."),
]


def _rows(pairs: list[tuple[str, str]]) -> QWidget:
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 8, 0)
    layout.setSpacing(12)
    for name, text in pairs:
        label = QLabel(f"<b>{html.escape(name)}</b><br>{html.escape(text)}")
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText)
        layout.addWidget(label)
    layout.addStretch(1)
    return box


class HelpDialog(QDialog):
    """A small guide with two pages: how to use the app, and the words explained."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("How to use the Satellite Signal Analyser")
        self.resize(620, 640)
        self.setStyleSheet(f"QDialog {{ background: {theme.BG}; }}")
        self.pages = QStackedWidget()
        for pairs in (HOW_TO_USE, WORDS):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(_rows(pairs))
            self.pages.addWidget(scroll)
        self.how_button = PillButton("How to use", "step")
        self.words_button = PillButton("Words explained", "step")
        group = QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self.how_button)
        group.addButton(self.words_button)
        self.how_button.setChecked(True)
        self.how_button.clicked.connect(lambda: self.show_page(0))
        self.words_button.clicked.connect(lambda: self.show_page(1))
        switch = QHBoxLayout()
        switch.setSpacing(10)
        switch.addWidget(self.how_button)
        switch.addWidget(self.words_button)
        switch.addStretch(1)
        close = PillButton("Close", "secondary")
        close.clicked.connect(self.accept)
        title = QLabel("How to use the app")
        title.setObjectName("PageTitle")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addLayout(switch)
        layout.addWidget(self.pages, 1)
        layout.addWidget(close, 0, Qt.AlignRight)
        self._title = title

    def show_page(self, index: int) -> None:
        self.pages.setCurrentIndex(int(index))
        self._title.setText("How to use the app" if index == 0 else "Words explained")
        (self.how_button if index == 0 else self.words_button).setChecked(True)


__all__ = ["HOW_TO_USE", "HelpDialog", "WORDS"]
