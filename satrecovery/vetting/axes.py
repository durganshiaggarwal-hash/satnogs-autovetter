"""Reading the numbers on a SatNOGS waterfall's axes.

To draw where a satellite's signal *should* be, the picture needs a
scale: how many kHz one column is, and how many seconds one row is.
SatNOGS does not publish it, and it varies -- about +-24 kHz for most
modes, +-29 kHz for 9600-baud FSK, +-38 kHz for 9600-baud BPSK (seen on
the live network, November 2026).  It is printed on the picture, though:
the tick labels under the plot (kHz) and to its right (seconds).

Every SatNOGS waterfall is drawn by matplotlib in its default font, so
this is a very small OCR problem.  :class:`GlyphReader` renders the
characters ``0-9``, minus and point in that font, and matches each blob of
ink on the axes against them.  :func:`read_axes` then fits a straight
line through (pixel, value) pairs, and refuses the answer unless the
labels lie on that line -- a misread digit shows up as a bad fit, not as
a wrong scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.exceptions import SatRecoveryError
from .waterfall import WaterfallDecodeError, locate_plot

CHARS = "0123456789−."
BOX = (20, 14)          # every glyph is compared on a 20 x 14 grid


class AxisReadError(SatRecoveryError):
    """The axis labels could not be read with confidence."""


def _normalise(mask: np.ndarray) -> np.ndarray:
    """A glyph's ink, cropped and scaled to fill :data:`BOX`."""
    rows, cols = np.nonzero(mask)
    crop = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1].astype(np.float32)
    out = np.zeros(BOX, dtype=np.float32)
    h, w = crop.shape
    scale = min(BOX[0] / h, BOX[1] / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    ys = (np.arange(nh) / scale).astype(int).clip(0, h - 1)
    xs = (np.arange(nw) / scale).astype(int).clip(0, w - 1)
    top, left = (BOX[0] - nh) // 2, (BOX[1] - nw) // 2
    out[top:top + nh, left:left + nw] = crop[np.ix_(ys, xs)]
    return out


class GlyphReader:
    """Recognises the digits, minus sign and point that matplotlib draws."""

    _templates: dict[str, np.ndarray] | None = None

    @classmethod
    def templates(cls) -> dict[str, np.ndarray]:
        if cls._templates is None:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            found = {}
            for char in CHARS:
                if char in "−.":
                    continue  # recognised by shape, below
                fig = plt.figure(figsize=(1, 1), dpi=100)
                fig.text(0.5, 0.5, char, fontsize=40, ha="center", va="center")
                fig.canvas.draw()
                rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3] / 255.0
                plt.close(fig)
                found[char] = _normalise(rgb.mean(axis=2) < 0.5)
            cls._templates = found
        return cls._templates

    def read(self, mask: np.ndarray, digit_height: float) -> str:
        """One glyph.  ``digit_height`` (pixels) tells a minus or point from a digit."""
        rows, cols = np.nonzero(mask)
        height = rows.max() - rows.min() + 1
        width = cols.max() - cols.min() + 1
        if height <= 0.35 * digit_height:
            return "−" if width >= 1.8 * height else "."
        from scipy.ndimage import gaussian_filter

        # Blur both: a 10-pixel glyph and a 40-pixel template differ in stroke
        # detail, not in shape, and blurring compares shape.
        glyph = gaussian_filter(_normalise(mask), 1.2).ravel()
        best, best_score = "?", -np.inf
        for char, template in self.templates().items():
            t = gaussian_filter(template, 1.2).ravel()
            score = float(np.dot(glyph - glyph.mean(), t - t.mean())
                          / (np.linalg.norm(glyph - glyph.mean()) * np.linalg.norm(t - t.mean()) + 1e-9))
            if score > best_score:
                best, best_score = char, score
        return best if best_score > 0.5 else "?"


@dataclass(frozen=True)
class Label:
    text: str
    x: float          # centre, pixels
    y: float

    @property
    def value(self) -> float | None:
        try:
            return float(self.text.replace("−", "-"))
        except ValueError:
            return None


def read_labels(ink: np.ndarray, reader: GlyphReader | None = None, vertical_gap: float = 0.6) -> list[Label]:
    """Every label in a strip of ink: blobs grouped into words, words read."""
    from scipy import ndimage

    reader = reader or GlyphReader()
    labelled, count = ndimage.label(ink, structure=np.ones((3, 3)))  # 8-way: strokes touch diagonally
    if count == 0:
        return []
    boxes = ndimage.find_objects(labelled)
    blobs = []
    for index, box in enumerate(boxes, start=1):
        ys, xs = box
        if (ys.stop - ys.start) * (xs.stop - xs.start) < 2:
            continue
        blobs.append((index, ys, xs))
    if not blobs:
        return []
    digit_height = float(np.percentile([b[1].stop - b[1].start for b in blobs], 75))

    # Join blobs into words: same line (centres close; a minus sign is short
    # but sits mid-height) and a small horizontal gap.  Union-find, so the
    # order blobs were found in does not matter.
    parent = list(range(len(blobs)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (_, ys_a, xs_a) in enumerate(blobs):
        for j in range(i + 1, len(blobs)):
            _, ys_b, xs_b = blobs[j]
            centres = abs((ys_a.start + ys_a.stop) - (ys_b.start + ys_b.stop)) / 2
            gap = max(xs_b.start - xs_a.stop, xs_a.start - xs_b.stop)
            if centres < vertical_gap * digit_height and gap < 0.6 * digit_height:
                parent[root(i)] = root(j)
    groups: dict[int, list[tuple]] = {}
    for i, blob in enumerate(blobs):
        groups.setdefault(root(i), []).append(blob)
    words = [sorted(group, key=lambda b: b[2].start) for group in groups.values()]

    labels = []
    for word in words:
        text = "".join(reader.read(labelled[ys, xs] == index, digit_height) for index, ys, xs in word)
        x0, x1 = min(b[2].start for b in word), max(b[2].stop for b in word)
        y0, y1 = min(b[1].start for b in word), max(b[1].stop for b in word)
        labels.append(Label(text, (x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2))
    return labels


@dataclass(frozen=True)
class WaterfallAxes:
    """Where the decoded waterfall's edges are, in kHz and seconds.

    ``khz_left``/``khz_right`` are the frequency offsets at the left and right
    edges of the plot; ``seconds_bottom``/``seconds_top`` the time since the
    observation started at the bottom and top edges.
    """

    khz_left: float
    khz_right: float
    seconds_bottom: float
    seconds_top: float

    def khz(self, column: np.ndarray | float, n_columns: int) -> np.ndarray:
        """kHz at the centre of a column of an ``n_columns``-wide grid."""
        return self.khz_left + (np.asarray(column) + 0.5) / n_columns * (self.khz_right - self.khz_left)

    def column(self, khz: np.ndarray | float, n_columns: int) -> np.ndarray:
        return (np.asarray(khz) - self.khz_left) / (self.khz_right - self.khz_left) * n_columns - 0.5

    def seconds(self, row: np.ndarray | float, n_rows: int) -> np.ndarray:
        """Seconds since start at the centre of a row (row 0 = start of the pass)."""
        return self.seconds_bottom + (np.asarray(row) + 0.5) / n_rows * (self.seconds_top - self.seconds_bottom)

    def to_dict(self) -> dict[str, float]:
        return {"khz_left": self.khz_left, "khz_right": self.khz_right,
                "seconds_bottom": self.seconds_bottom, "seconds_top": self.seconds_top}


def _fit(positions: list[float], values: list[float], what: str) -> tuple[float, float]:
    """``value = a + b * position``; refuses unless at least three labels agree within a pixel."""
    if len(positions) < 3:
        raise AxisReadError(f"{what}: only {len(positions)} readable labels")
    p, v = np.asarray(positions, dtype=float), np.asarray(values, dtype=float)
    # A misread digit ("200" read as "700") is far off the line the other
    # labels make.  Try every pair of labels as the line, keep the one most
    # labels agree with, and allow at most a third of the labels to disagree.
    best: np.ndarray | None = None
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] == p[j] or v[i] == v[j]:
                continue
            slope = (v[j] - v[i]) / (p[j] - p[i])
            agree = np.abs((v - (v[i] + slope * (p - p[i]))) / slope) <= 1.5
            if best is None or agree.sum() > best.sum():
                best = agree
    if best is None or best.sum() < max(3, int(np.ceil(len(p) * 2 / 3))):
        raise AxisReadError(f"{what}: labels do not lie on a line")
    b, a = np.polyfit(p[best], v[best], 1)
    worst = np.max(np.abs((v[best] - (a + b * p[best])) / b))
    if not np.isfinite(worst) or worst > 1.5:
        raise AxisReadError(f"{what}: labels do not lie on a line (worst {worst:.1f} px)")
    return float(a), float(b)


def _ink(rgb: np.ndarray) -> np.ndarray:
    return rgb.mean(axis=2) < 0.45


def _first_band(profile: np.ndarray, start: int, minimum: int = 3, bridge: int = 3) -> tuple[int, int] | None:
    """First run of ``True`` at least ``minimum`` long, at or after ``start``.

    Gaps of up to ``bridge`` are part of the run: the space between two
    digits of one label must not end the band of labels.
    """
    run = last = None
    for i in range(start, len(profile)):
        if profile[i]:
            if run is None:
                run = i
            last = i
        elif run is not None and i - last > bridge:
            if last + 1 - run >= minimum:
                return run, last + 1
            run = last = None
    return (run, last + 1) if run is not None and last + 1 - run >= minimum else None


def read_axes(rgb: np.ndarray) -> WaterfallAxes:
    """The kHz and seconds scales of a SatNOGS waterfall picture."""
    row0, row1, col0, col1 = locate_plot(rgb)
    ink = _ink(rgb)

    # Frequency: the first band of text below the plot (the tick labels;
    # the axis title "Frequency (kHz)" is a second band further down).
    below = ink[row1 + 1:, max(col0 - 40, 0):col1 + 40]
    band = _first_band(below.any(axis=1), start=4)
    if band is None:
        raise AxisReadError("no frequency labels under the plot")
    strip = below[band[0]:band[1]]
    labels = [l for l in read_labels(strip) if l.value is not None]
    offset = max(col0 - 40, 0)
    a, b = _fit([l.x + offset for l in labels], [l.value for l in labels], "frequency")
    khz_left, khz_right = a + b * (col0 - 0.5), a + b * (col1 - 0.5)
    if not (khz_left < 0 < khz_right):
        raise AxisReadError(f"frequency axis {khz_left:.1f}..{khz_right:.1f} kHz does not span 0")

    # Time: tick labels in seconds.  Newer pictures put them right of the
    # plot (with clock times on the left); older ones put them on the left.
    a = b = None
    problems = []
    # A label on the very edge (often "0" at the bottom) sticks out of the
    # plot's rows by half its height, so look a little above and below.
    top, bottom = max(row0 - 8, 0), row1 + 8
    for side in ("right", "left"):
        if side == "right":
            region = ink[top:bottom, col1 + 1:]
            band = _first_band(region.any(axis=0), start=5)
        else:
            region = ink[top:bottom, :col0][:, ::-1]      # scan outwards from the plot
            band = _first_band(region.any(axis=0), start=5)
        if band is None:
            problems.append(f"{side}: no labels")
            continue
        strip = region[:, band[0]:band[1]]
        if side == "left":
            strip = strip[:, ::-1]
        labels = [l for l in read_labels(strip) if l.value is not None]
        try:
            a, b = _fit([l.y + top for l in labels], [l.value for l in labels], f"time ({side})")
            break
        except AxisReadError as error:
            problems.append(str(error))
    if a is None:
        raise AxisReadError("; ".join(problems))
    seconds_top, seconds_bottom = a + b * (row0 - 0.5), a + b * (row1 - 0.5)
    if not seconds_bottom < seconds_top:
        raise AxisReadError("time does not increase upwards")
    return WaterfallAxes(float(khz_left), float(khz_right), float(seconds_bottom), float(seconds_top))


def read_axes_png(path: str | Path) -> WaterfallAxes:
    from matplotlib import image

    try:
        rgb = np.asarray(image.imread(str(path)), dtype=np.float32)[:, :, :3]
    except (OSError, ValueError, SyntaxError) as error:
        raise WaterfallDecodeError(f"cannot read {Path(path).name}: {error}") from error
    return read_axes(rgb)
