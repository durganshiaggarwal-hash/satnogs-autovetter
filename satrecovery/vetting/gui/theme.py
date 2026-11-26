"""The app's look: the Ink wash palette, flat boxes, one slate accent.

* Surfaces are ink greys; boxes are flat with square corners and thin
  grey dividers, never shadows or gradients.
* Headings are cream, upper case and widely letter-spaced; sub-headings
  and section labels are slate with wide letter-spacing.
* Buttons are text with a thin underline, not boxes; tabs are rows of
  dark tiles where the chosen one is a lighter grey.
* Colour carries meaning: lime = signal, red = no signal, amber = unsure.

Fonts are the project's own (Manrope, IBM Plex Mono), loaded by
:func:`satrecovery.gui.theme.load_fonts`.
"""

from __future__ import annotations

# The "Ink wash" palette: #4A4A4A ink, #CBCBCB grey, #FFFFE3 cream, #6D8196 slate.
BG = "#2e2e2e"          # window background: a deeper wash of the ink
PANEL = "#4A4A4A"       # boxes: the ink
RAISED = "#565656"      # tiles, inputs
SELECTED = "#6D8196"    # the chosen tile: slate
RULE = "#5c5c5c"        # dividers
LINE = "#CBCBCB"        # button underlines: the grey
TEXT = "#FFFFE3"        # cream
DIM = "#CBCBCB"
FAINT = "#8f8f8f"
ACCENT = "#a9bccf"      # slate, lightened so text and thin lines stay readable on the ink
ACCENT_RGB = (169, 188, 207)
ACCENT_FILL = "#6D8196" # slate itself, for fills
SIGNAL = "#bfe08f"      # soft lime
NO_SIGNAL = "#ee8a76"   # soft red
UNSURE = "#f2d47c"      # warm amber

MONO = "IBM Plex Mono"
SANS = "Manrope"
SANS_SEMIBOLD = "Manrope SemiBold"
SANS_BOLD = "Manrope ExtraBold"


def verdict_colour(verdict_value: str) -> str:
    return {"signal": SIGNAL, "no signal": NO_SIGNAL}.get(verdict_value, UNSURE)


STYLE = f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-family: "{SANS}"; font-size: 14px; }}
QFrame#panel {{ background: {PANEL}; border: none; }}
QLabel, QCheckBox {{ background: transparent; }}
QFrame#rule {{ background: {RULE}; max-height: 1px; min-height: 1px; }}
QLabel#title {{ font-family: "{SANS_BOLD}"; font-size: 17px; letter-spacing: 4px; color: {TEXT}; }}
QLabel#sub {{ color: {DIM}; font-size: 13px; }}
QLabel#caps {{ font-family: "{SANS_SEMIBOLD}"; font-size: 12px; letter-spacing: 2.5px; color: {ACCENT}; }}
QLabel#mono {{ font-family: "{MONO}"; font-size: 13px; }}
QLabel#big {{ font-family: "{SANS_BOLD}"; font-size: 24px; letter-spacing: 2px; }}
QLabel#counter {{ font-family: "{MONO}"; font-size: 19px; }}
QTableView {{ background: {PANEL}; border: none; gridline-color: {PANEL}; font-family: "{MONO}"; font-size: 13px;
             selection-background-color: {ACCENT_FILL}; selection-color: {TEXT}; outline: 0; }}
QTableView::item {{ padding: 4px 8px; border-bottom: 1px solid {RULE}; }}
QTableView::item:hover {{ background: {RAISED}; }}
QHeaderView::section {{ background: {PANEL}; color: {ACCENT}; border: none; border-bottom: 1px solid {RULE};
                       font-family: "{SANS_SEMIBOLD}"; font-size: 12px; letter-spacing: 1.5px; padding: 8px 8px; }}
QPushButton {{ background: transparent; border: none; border-bottom: 1px solid {LINE}; border-radius: 0;
              padding: 8px 6px; font-family: "{SANS_SEMIBOLD}"; font-size: 14px; color: {TEXT}; }}
QPushButton:hover {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QPushButton:pressed {{ color: {DIM}; }}
QPushButton:checked {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QPushButton#signal {{ color: {SIGNAL}; border-bottom-color: {SIGNAL}; }}
QPushButton#nosignal {{ color: {NO_SIGNAL}; border-bottom-color: {NO_SIGNAL}; }}
QPushButton#primary {{ color: {ACCENT}; border-bottom-color: {ACCENT}; }}
QPushButton#tab {{ background: {PANEL}; border: none; border-right: 1px solid {BG}; padding: 9px 20px; min-width: 70px;
                  font-family: "{SANS_SEMIBOLD}"; font-size: 14px; color: {DIM}; }}
QPushButton#tab:hover {{ background: {RAISED}; color: {TEXT}; border: none; border-right: 1px solid {BG}; }}
QPushButton#tab:checked {{ background: {ACCENT_FILL}; color: {TEXT}; border: none; border-right: 1px solid {BG}; }}
QComboBox, QDoubleSpinBox, QLineEdit {{ background: {RAISED}; border: none; border-bottom: 1px solid {LINE};
                             border-radius: 0; padding: 6px 10px; font-family: "{MONO}"; font-size: 13px; }}
QComboBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-bottom-color: {ACCENT}; }}
QToolButton#thresholds {{ background: transparent; border: none; border-bottom: 1px solid {LINE}; border-radius: 0;
                          padding: 8px 6px; font-family: "{SANS_SEMIBOLD}"; font-size: 14px; color: {TEXT}; }}
QToolButton#thresholds:hover {{ color: {ACCENT}; border-bottom-color: {ACCENT}; }}
QToolButton#thresholds::menu-indicator {{ image: none; }}
QMenu {{ background: {PANEL}; border: 1px solid {RULE}; }}
QComboBox QAbstractItemView {{ background: {PANEL}; selection-background-color: {ACCENT_FILL}; }}
QCheckBox {{ color: {DIM}; font-size: 13px; }}
QStatusBar {{ background: {BG}; color: {DIM}; font-family: "{MONO}"; font-size: 12px; border-top: 1px solid {RULE}; }}
QSplitter::handle {{ background: {RULE}; }}
QSlider::groove:horizontal {{ height: 3px; background: {RULE}; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QScrollBar:vertical {{ background: {BG}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {RAISED}; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT_FILL}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {RULE}; padding: 6px 8px; font-size: 13px; }}
"""

#: Expert mode packs a lot onto one screen: slightly tighter type there.
EXPERT = f"""
QLabel#counter {{ font-size: 17px; }}
QLabel#mono {{ font-size: 12px; }}
QLabel#big {{ font-size: 22px; }}
QTableView {{ font-size: 13px; }}
"""
