"""How the application looks: colours, type, spacing, and one stylesheet.

Every colour, font and radius in the GUI comes from this module, so the
whole look can be changed here -- including swapping in a palette of
your own: replace the hex values under *Surface and ink* and *Brand*,
and every screen follows.

THE DIRECTION: DAYLIGHT LAB, WITH ONE SPLASH OF AURORA
-------------------------------------------------------
Light and roomy, so nothing feels cramped: a pale lavender background,
white cards with soft rounded corners, generous gaps.

Colourful in one confident place rather than everywhere: a signature
indigo-to-violet-to-pink gradient -- an aurora over a ground station --
on the header, the main action buttons and the sliders' filled tracks.
Each tab also carries its own colour, so the user always knows where
they are.

Friendly but serious type: Manrope, a rounded geometric sans, for
everything; IBM Plex Mono for numbers that should line up.

THE ONE RULE THAT DOES NOT BEND
-------------------------------
Inside the plots, colour means *which receiver*.  Those colours come
from a palette validated for colour-vision deficiency, each receiver
keeps one colour on every screen, and red is reserved for wrong symbols
and always comes with a different marker shape.  The interface around
the plots can be as colourful as it likes; the data cannot.
"""

from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------
# Surface and ink -- from the user's palette:
#   Peach Fuzz #FFC9B5, Powder Blush #F7B1AB, Desert Sand #D8AA96,
#   Dusty Mauve #807182, Ash Grey #C7D3BF
# The palette is soft, so it is used for fills; text and outlines use
# deeper shades of the same hues, which is what keeps it readable.
# ----------------------------------------------------------------------
PEACH = "#FFC9B5"
BLUSH = "#F7B1AB"
SAND = "#D8AA96"
MAUVE = "#807182"
SAGE = "#C7D3BF"

BG = "#fdf7f4"            # window background: the palest peach
SURFACE = "#ffffff"       # cards
SUNK = "#faeee9"          # inputs, wells, hover fills
LINE = "#efdfd8"          # card borders
LINE_SOFT = "#f6ebe6"     # grid lines
INK = "#3a2f3d"           # primary text: Dusty Mauve, deepened to plum
INK_2 = "#6b5f6d"         # secondary text
INK_3 = "#a3969f"         # muted text, axes
FOCUS = MAUVE             # keyboard focus ring

# ----------------------------------------------------------------------
# Brand: a peach-to-blush-to-sand gradient, used in a few places on purpose
# ----------------------------------------------------------------------
BRAND = (PEACH, BLUSH, SAND)                    # the soft gradient for fills
BRAND_DEEP = MAUVE                              # the one strong colour: outlines, handles, ticks
BRAND_SOFT = ("#fff1eb", "#fdeceb", "#f7eee9")  # palest tints for hovers and chips
ON_BRAND = INK                                  # text on a gradient fill: dark, not white

#: Each tab's own colour, as (fill, ink): a pastel from the palette for the
#: tab's pill, and a deeper shade of the same hue for its text.
PAGE_ACCENTS = {
    "Experiment": (BLUSH, "#a14f48"),
    "Step by step": (SAND, "#86523d"),
    "Benchmark": (SAGE, "#4e6b44"),
    "Models": ("#d6cdd7", "#5e4f61"),
}


def page_accent(title: str) -> str:
    """The tab's strong colour, for text and small marks."""
    if DARK:
        return BRAND_DEEP          # one accent for every tab in the dark look
    return PAGE_ACCENTS.get(title, (BLUSH, MAUVE))[1]


def page_fill(title: str) -> str:
    """The tab's soft colour, for its pill and highlights."""
    if DARK:
        return "#6D8196"           # the chosen tab is a slate tile
    return PAGE_ACCENTS.get(title, (BLUSH, MAUVE))[0]


def corner(height: float) -> float:
    """Corner radius of pills and tabs: round in the light look, square in the dark one."""
    return 0.0 if DARK else height / 2


# ----------------------------------------------------------------------
# Data colours -- by receiver identity, fixed, validated (see module doc)
# ----------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]

RECEIVER_COLOURS = {
    "classical": SERIES[0],
    "hybrid": SERIES[1],
    "hybrid-noise": SERIES[2],
    "classical-hard": SERIES[3],
    "classical-equalised": SERIES[4],
}

OTHER = INK_3
CRITICAL = "#d03b3b"     # wrong symbols and failures, never a series colour
GOOD = "#3f8f4a"
RAW = "#9a8c99"          # the received signal before any receiver touched it


def receiver_colour(name: str) -> str:
    """The one colour a receiver wears everywhere."""
    return RECEIVER_COLOURS.get(name, OTHER)


def tint(hex_colour: str, amount: float) -> str:
    """Mix a colour towards the card colour: 0 leaves it, 1 makes it the card colour.

    In the light look that is white, as it always was; in the dark look
    it is the dark card colour, so a tinted fill stays dark behind light text.
    """
    hex_colour = hex_colour.lstrip("#")
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (0, 2, 4))
    target = SURFACE.lstrip("#") if DARK else "ffffff"
    tr, tg, tb = (int(target[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c, t: round(c + (t - c) * amount)  # noqa: E731
    return f"#{mix(r, tr):02x}{mix(g, tg):02x}{mix(b, tb):02x}"


# ----------------------------------------------------------------------
# Dark mode (added for the Satellite Signal Analyser's "Dark" switch)
# ----------------------------------------------------------------------
#: Is the dark look in use?  Changed only through use_dark().
DARK = False

_LIGHT_VALUES: dict[str, object] = {}

#: The dark look: the same hues, deepened for fills and lightened for text,
#: so every pairing used in the light look (light text on soft fills
#: becomes light text on deep fills) stays readable.
_DARK_VALUES: dict[str, object] = {
    # Retuned (November 2026) to the "Ink wash" palette shared with the Live
    # satellites half: #4A4A4A ink, #CBCBCB grey, #FFFFE3 cream, #6D8196 slate.
    # Flat and square.  Hue roles are unchanged: the "brand" fills are slate.
    "PEACH": "#6d8196", "BLUSH": "#6d8196", "SAND": "#5b6d80", "MAUVE": "#a9bccf", "SAGE": "#5f7a55",
    "BG": "#2e2e2e", "SURFACE": "#4a4a4a", "SUNK": "#3c3c3c", "LINE": "#5c5c5c", "LINE_SOFT": "#444444",
    "INK": "#ffffe3", "INK_2": "#cbcbcb", "INK_3": "#9a9a9a", "FOCUS": "#a9bccf",
    "BRAND_DEEP": "#a9bccf", "BRAND_SOFT": ("#3d4650", "#414b56", "#39424c"),
    "CRITICAL": "#ee8a76", "GOOD": "#bfe08f", "RAW": "#b0b0b0",
    "RADIUS_CARD": 0, "RADIUS_CONTROL": 0, "BASE_PT": 12,
}


def use_dark(on: bool) -> None:
    """Switch every colour in this module to the dark look, or back.

    Widgets read these colours when they are built and painted, so the
    window is rebuilt after a switch (the app does that); the stylesheet
    and the plot defaults are set again by the caller.
    """
    global DARK, BRAND, ON_BRAND, OTHER
    module = globals()
    if not _LIGHT_VALUES:
        _LIGHT_VALUES.update({name: module[name] for name in _DARK_VALUES})
    for name, value in (_DARK_VALUES if on else _LIGHT_VALUES).items():
        module[name] = value
    DARK = bool(on)
    BRAND = (module["PEACH"], module["BLUSH"], module["SAND"])
    ON_BRAND = module["INK"]   # cream text on the slate fill
    OTHER = module["INK_3"]


# ----------------------------------------------------------------------
# Type and spacing
# ----------------------------------------------------------------------
ASSETS = Path(__file__).resolve().parent / "assets"
FONT_DIR = ASSETS / "fonts"

#: Manrope registers each weight as its own family name, so weights are
#: chosen by family -- which works identically on Windows and Linux,
#: where relying on Qt to match "weight 600" does not.
SANS = "Manrope"
SANS_MEDIUM = "Manrope Medium"
SANS_SEMIBOLD = "Manrope SemiBold"
SANS_EXTRABOLD = "Manrope ExtraBold"
MONO = "IBM Plex Mono"
BASE_PT = 10

RADIUS_CARD = 14
RADIUS_CONTROL = 10
GAP = 16          # space between cards
PAD = 20          # space inside cards


def load_fonts() -> list[str]:
    """Register the bundled typefaces with Qt. Returns the families loaded.

    Bundled rather than assumed, so the application looks the same on a
    laptop that has never had these fonts installed.  If loading fails,
    Qt falls back to the system face and nothing breaks.
    """
    from PySide6.QtGui import QFontDatabase

    families: set[str] = set()
    for path in sorted(FONT_DIR.glob("*.ttf")):
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id >= 0:
            families.update(QFontDatabase.applicationFontFamilies(font_id))
    return sorted(families)


def gradient(direction: str = "horizontal", colours: tuple[str, ...] | None = None) -> str:
    """A Qt stylesheet gradient across the brand colours."""
    colours = colours or BRAND
    x2, y2 = (1, 0) if direction == "horizontal" else (0, 1)
    stops = ", ".join(
        f"stop:{i / (len(colours) - 1):.2f} {c}" for i, c in enumerate(colours)
    )
    return f"qlineargradient(x1:0, y1:0, x2:{x2}, y2:{y2}, {stops})"


def asset(name: str) -> str:
    return (ASSETS / name).as_posix()


def stylesheet() -> str:
    """One Qt stylesheet for every standard widget.

    The bespoke animated widgets in :mod:`satrecovery.gui.motion` paint
    themselves; this covers everything Qt draws.
    """
    return f"""
    QWidget {{
        font-family: "{SANS}"; font-size: {BASE_PT}pt; color: {INK};
    }}
    QMainWindow, QWidget#Page, QWidget#Shell {{ background: {BG}; }}

    QFrame#Card {{
        background: {SURFACE}; border: 1px solid {LINE}; border-radius: {RADIUS_CARD}px;
    }}
    QFrame#CardTinted {{
        background: {SURFACE}; border: 1px solid {LINE}; border-radius: {RADIUS_CARD}px;
    }}
    QLabel#PageTitle {{ font-family: "{SANS_EXTRABOLD}"; font-size: 19pt; }}
    QLabel#CardTitle {{ font-family: "{SANS_SEMIBOLD}"; font-size: 11pt; }}
    QLabel#Hint {{ color: {INK_2}; }}
    QLabel#Muted {{ color: {INK_3}; font-size: 9pt; }}
    QLabel#Readout {{ font-family: "{MONO}"; color: {INK_2}; font-size: 9pt; }}
    QLabel#FieldLabel {{ font-family: "{SANS_MEDIUM}"; color: {INK}; }}
    QLabel#Error {{ color: {CRITICAL}; }}
    QLabel#ReadoutChip {{
        font-family: "{MONO}"; font-size: 9pt; color: {BRAND_DEEP};
        background: {BRAND_SOFT[0]}; border-radius: 8px; padding: 2px 9px;
    }}
    QLabel#BigNumber {{ font-family: "{SANS_EXTRABOLD}"; font-size: 24pt; color: {INK}; }}
    QLabel#Badge {{ border-radius: 9px; padding: 2px 10px; font-family: "{SANS_SEMIBOLD}"; font-size: 9pt; }}

    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
        background: {SURFACE}; border: 1px solid {LINE}; border-radius: {RADIUS_CONTROL}px;
        padding: 6px 12px; min-height: 22px; selection-background-color: {BRAND_SOFT[1]};
    }}
    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: {SAND}; }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
        border: 1px solid {FOCUS};
    }}
    QComboBox::drop-down {{ border: none; width: 28px; }}
    QComboBox::down-arrow {{ image: url("{asset('chevron-down.svg')}"); }}
    QComboBox QAbstractItemView {{
        background: {SURFACE}; border: 1px solid {LINE}; border-radius: 8px; padding: 4px;
        selection-background-color: {BRAND_SOFT[0]}; selection-color: {INK}; outline: none;
    }}
    QAbstractSpinBox {{ padding: 6px 30px 6px 12px; }}
    QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
        subcontrol-origin: border; width: 22px; border: none; background: transparent;
    }}
    QAbstractSpinBox::up-button {{ subcontrol-position: top right; margin-top: 3px; margin-right: 3px; }}
    QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; margin-bottom: 3px; margin-right: 3px; }}
    QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
        background: {SUNK}; border-radius: 5px;
    }}
    QAbstractSpinBox::up-arrow {{ image: url("{asset('chevron-up.svg')}"); width: 10px; height: 6px; }}
    QAbstractSpinBox::down-arrow {{ image: url("{asset('chevron-down.svg')}"); width: 10px; height: 6px; }}

    QSlider {{ min-height: 24px; }}
    QSlider::groove:horizontal {{ height: 6px; background: {SUNK}; border-radius: 3px; }}
    QSlider::sub-page:horizontal {{ background: {gradient()}; border-radius: 3px; }}
    QSlider::handle:horizontal {{
        background: {SURFACE}; width: 18px; height: 18px; margin: -7px 0;
        border-radius: 10px; border: 2px solid {BRAND_DEEP};
    }}
    QSlider::handle:horizontal:hover {{ border: 3px solid {BRAND_DEEP}; }}
    QSlider::handle:horizontal:pressed {{ background: {BLUSH}; border: 3px solid {BRAND_DEEP}; }}
    QSlider::handle:horizontal:focus {{ border: 3px solid {FOCUS}; }}
    QSlider:disabled {{ }}
    QSlider::sub-page:horizontal:disabled {{ background: {LINE}; }}
    QSlider::handle:horizontal:disabled {{ border-color: {LINE}; }}

    QCheckBox {{ spacing: 8px; }}
    QCheckBox::indicator {{
        width: 16px; height: 16px; border: 1.5px solid {INK_3}; border-radius: 5px;
        background: {SURFACE};
    }}
    QCheckBox::indicator:hover {{ border-color: {BRAND_DEEP}; }}
    QCheckBox::indicator:checked {{
        background: {BRAND_DEEP}; border-color: {BRAND_DEEP}; image: url("{asset('check.svg')}");
    }}
    QCheckBox:disabled {{ color: {INK_3}; }}

    QTableWidget {{
        background: {SURFACE}; border: none; gridline-color: {LINE_SOFT};
        selection-background-color: {SUNK}; selection-color: {INK};
    }}
    QTableWidget::item {{ padding: 0 12px; border-bottom: 1px solid {LINE_SOFT}; }}
    QHeaderView::section {{
        background: {SURFACE}; color: {INK_3}; border: none;
        border-bottom: 1px solid {LINE}; padding: 8px 12px; font-family: "{SANS_SEMIBOLD}";
        font-size: 9pt;
    }}

    QScrollArea {{ border: none; background: transparent; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 4px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: {INK_3}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {LINE}; border-radius: 4px; min-width: 30px; }}

    QStatusBar {{ background: {SURFACE}; border-top: 1px solid {LINE}; color: {INK_2}; }}
    QStatusBar QLabel {{ color: {INK_3}; font-size: 9pt; }}
    QToolTip {{
        background: {INK}; color: {SURFACE}; border: none; padding: 6px 10px; border-radius: 6px;
    }}
    """ + (_DARK_EXTRA.format(ink=INK, accent=BRAND_DEEP, line=LINE, sunk=SUNK, mono=MONO,
                              bold=SANS_EXTRABOLD, semibold=SANS_SEMIBOLD) if DARK else "")


#: Headings and dividers in the dark look: cream bold headings with wide
#: letter-spacing, slate section titles, thin grey rules.
_DARK_EXTRA = """
    QLabel#PageTitle {{ font-family: "{bold}"; font-size: 24pt; letter-spacing: 4px; color: {ink}; }}
    QLabel#CardTitle {{ font-family: "{semibold}"; font-size: 14pt; letter-spacing: 1.5px; color: {accent}; }}
    QLabel#BigNumber {{ font-size: 28pt; letter-spacing: 1px; }}
    QLabel#Muted {{ font-size: 11pt; }}
    QLabel#Readout, QLabel#ReadoutChip {{ font-size: 11pt; }}
    QLabel#Badge {{ font-size: 11pt; }}
    QStatusBar QLabel {{ font-size: 11pt; }}
    QHeaderView::section {{ color: {accent}; letter-spacing: 1px; font-size: 11pt; }}
    QFrame#Card, QFrame#CardTinted {{ border: 1px solid {line}; }}
    QToolTip {{ background: {sunk}; color: {ink}; border: 1px solid {line}; border-radius: 0; }}
    QLabel#ReadoutChip {{ border-radius: 0; }}
    QLabel#Badge {{ border-radius: 0; }}
"""


def configure_plots() -> None:
    """Global pyqtgraph defaults: white plots, quiet axes, antialiasing.

    White, because the colour validator measured four of the five
    receiver colours under 3:1 contrast on a grey background and only
    three on white -- and those three are always named on the plot that
    shows them, so identity never rests on colour alone.
    """
    import pyqtgraph as pg

    pg.setConfigOptions(antialias=True, background=SURFACE, foreground=INK_3)


__all__ = [
    "ASSETS", "BG", "BLUSH", "BRAND", "BRAND_DEEP", "BRAND_SOFT", "MAUVE", "ON_BRAND", "PEACH", "SAGE", "SAND", "CRITICAL", "FONT_DIR", "GAP", "GOOD", "INK",
    "INK_2", "INK_3", "LINE", "LINE_SOFT", "MONO", "OTHER", "PAD", "PAGE_ACCENTS",
    "RADIUS_CARD", "RADIUS_CONTROL", "RAW", "RECEIVER_COLOURS", "SANS", "SANS_EXTRABOLD",
    "SANS_MEDIUM", "SANS_SEMIBOLD", "SERIES", "SUNK", "SURFACE", "asset", "configure_plots",
    "gradient", "load_fonts", "page_accent", "page_fill", "receiver_colour", "stylesheet", "tint", "use_dark",
]
