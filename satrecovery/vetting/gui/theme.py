"""The vetting station's look: a dark instrument panel, not a dashboard.

Dense, monospace numbers, thin rules instead of cards, colour only where
it carries meaning (green = signal, coral = no signal, amber = unsure).
Fonts are the project's own (Manrope, IBM Plex Mono), loaded by
:func:`satrecovery.gui.theme.load_fonts`.
"""

from __future__ import annotations

BG = "#0d1015"
PANEL = "#12161c"
RAISED = "#181d25"
RULE = "#232a34"
TEXT = "#d6dde6"
DIM = "#7b8594"
FAINT = "#4a5260"
ACCENT = "#56c8d8"
SIGNAL = "#5fd68e"
NO_SIGNAL = "#f07a5f"
UNSURE = "#f0c35a"

MONO = "IBM Plex Mono"
SANS = "Manrope"


def verdict_colour(verdict_value: str) -> str:
    return {"signal": SIGNAL, "no signal": NO_SIGNAL}.get(verdict_value, UNSURE)


STYLE = f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-family: "{SANS}"; font-size: 12px; }}
QFrame#panel {{ background: {PANEL}; border: none; }}
QLabel, QCheckBox {{ background: transparent; }}
QFrame#rule {{ background: {RULE}; max-height: 1px; min-height: 1px; }}
QLabel#title {{ font-family: "{MONO}"; font-size: 13px; letter-spacing: 2px; color: {TEXT}; }}
QLabel#sub {{ color: {DIM}; font-size: 11px; }}
QLabel#caps {{ font-family: "{MONO}"; font-size: 10px; letter-spacing: 1.5px; color: {DIM}; }}
QLabel#mono {{ font-family: "{MONO}"; }}
QLabel#big {{ font-family: "{MONO}"; font-size: 22px; }}
QLabel#counter {{ font-family: "{MONO}"; font-size: 16px; }}
QTableView {{ background: {PANEL}; border: none; gridline-color: {PANEL}; font-family: "{MONO}"; font-size: 11px;
             selection-background-color: {RAISED}; selection-color: {TEXT}; outline: 0; }}
QTableView::item {{ padding: 2px 6px; border-bottom: 1px solid {BG}; }}
QHeaderView::section {{ background: {PANEL}; color: {DIM}; border: none; border-bottom: 1px solid {RULE};
                       font-family: "{MONO}"; font-size: 10px; padding: 4px 6px; }}
QPushButton {{ background: {RAISED}; border: 1px solid {RULE}; border-radius: 3px; padding: 6px 10px;
              font-family: "{MONO}"; font-size: 11px; color: {TEXT}; }}
QPushButton:hover {{ border-color: {DIM}; }}
QPushButton:pressed {{ background: {RULE}; }}
QPushButton#signal {{ border-color: {SIGNAL}; color: {SIGNAL}; }}
QPushButton#nosignal {{ border-color: {NO_SIGNAL}; color: {NO_SIGNAL}; }}
QPushButton#primary {{ border-color: {ACCENT}; color: {ACCENT}; }}
QComboBox, QDoubleSpinBox {{ background: {RAISED}; border: 1px solid {RULE}; border-radius: 3px; padding: 3px 6px;
                             font-family: "{MONO}"; font-size: 11px; }}
QComboBox QAbstractItemView {{ background: {RAISED}; selection-background-color: {RULE}; }}
QCheckBox {{ color: {DIM}; font-size: 11px; }}
QStatusBar {{ background: {PANEL}; color: {DIM}; font-family: "{MONO}"; font-size: 10px; border-top: 1px solid {RULE}; }}
QSplitter::handle {{ background: {RULE}; }}
QScrollBar:vertical {{ background: {PANEL}; width: 8px; }}
QScrollBar::handle:vertical {{ background: {RULE}; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""
