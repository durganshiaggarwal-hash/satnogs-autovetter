"""Turn a SatNOGS waterfall picture back into numbers.

SatNOGS publishes each waterfall as a matplotlib chart: the spectrum
drawn in the *viridis* colour map, with axes, tick labels and a colour
bar around it.  A network should learn from the spectrum, not from the
tick labels, so :class:`WaterfallImage` does two things:

1. **Finds the plot.**  Every pixel is matched to its nearest viridis
   colour.  The plot is the widest block of columns (the colour bar is a
   narrow one) and tallest block of rows that are almost all viridis.
2. **Inverts the colour map.**  Each pixel becomes its position on the
   colour map, 0 (dark purple, weakest) to 1 (yellow, strongest).

The result is a ``(time, frequency)`` array with time running forwards
(the picture puts the start of the pass at the bottom).  The colour bar's
dB range changes from picture to picture and is not read (that would mean
reading text), so values are *relative* power within one observation.

DESIGN NOTE: ENCAPSULATION
--------------------------
The array is read-only and only reachable through properties, and the
object is only made through :meth:`from_png` / :meth:`load` or from a
checked array, so a ``WaterfallImage`` is always a valid spectrum.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.exceptions import SatRecoveryError, ValidationError


class WaterfallDecodeError(SatRecoveryError):
    """The picture does not look like a SatNOGS waterfall."""


_LUT: np.ndarray | None = None
_TREE = None


def _colour_map():
    global _LUT, _TREE
    if _LUT is None:
        from matplotlib import colormaps
        from scipy.spatial import cKDTree

        _LUT = colormaps["viridis"](np.linspace(0.0, 1.0, 256))[:, :3].astype(np.float32)
        _TREE = cKDTree(_LUT)
    return _LUT, _TREE


def _longest_run(flags: np.ndarray) -> tuple[int, int]:
    """``(start, stop)`` of the longest run of ``True``; ``(0, 0)`` if none."""
    best = (0, 0)
    start = None
    for i, flag in enumerate(np.append(flags, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


TOLERANCE = 0.03
COVERAGE = 0.9


_TABLE: tuple[np.ndarray, np.ndarray] | None = None
LEVELS = 128   # colours are matched on a 128-level-per-channel grid


def _colour_table() -> tuple[np.ndarray, np.ndarray]:
    """For every colour on the grid: its nearest viridis index and how far from viridis it is.

    Worked out once (about a second); after that a picture is decoded by
    looking its pixels up, instead of searching a tree for each of 1.3
    million pixels -- which also held Python's lock, so parallel downloads
    queued behind one another.
    """
    global _TABLE
    if _TABLE is None:
        _, tree = _colour_map()
        steps = (np.arange(LEVELS, dtype=np.float32) + 0.5) / LEVELS
        grid = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1).reshape(-1, 3)
        distance, index = tree.query(grid, k=1)
        _TABLE = (index.astype(np.uint8).reshape(LEVELS, LEVELS, LEVELS),
                  distance.astype(np.float32).reshape(LEVELS, LEVELS, LEVELS))
    return _TABLE


def _locate(rgb: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
    """``(row0, row1, col0, col1, colour-map index of every pixel)`` of the plot."""
    table_index, table_distance = _colour_table()
    height, width, _ = rgb.shape
    q = np.clip((np.asarray(rgb, dtype=np.float32) * LEVELS).astype(np.int32), 0, LEVELS - 1)
    index = table_index[q[..., 0], q[..., 1], q[..., 2]]
    on_map = table_distance[q[..., 0], q[..., 1], q[..., 2]] < TOLERANCE

    # Columns first, judged against the best column: margins above and
    # below the plot differ between pictures, so no fixed share works.
    column_cover = on_map.mean(axis=0)
    if column_cover.max() < 0.25:
        raise WaterfallDecodeError("no plot area found (hardly any viridis colours)")
    col0, col1 = _longest_run(column_cover > COVERAGE * column_cover.max())
    if col1 - col0 < 16:
        raise WaterfallDecodeError("no plot area found (no wide block of viridis colours)")
    row0, row1 = _longest_run(on_map[:, col0:col1].mean(axis=1) > COVERAGE)
    if row1 - row0 < 16:
        raise WaterfallDecodeError("no plot area found (no tall block of viridis colours)")
    return row0, row1, col0, col1, index


def locate_plot(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """``(row0, row1, col0, col1)``: the plot inside a waterfall picture (end-exclusive)."""
    return _locate(rgb)[:4]


class WaterfallImage:
    """A waterfall as relative power, ``shape == (time_bins, freq_bins)``."""

    #: kept for compatibility; the module-level values are used
    TOLERANCE = 0.03
    #: a row/column is part of the plot if this share of it is viridis
    COVERAGE = 0.9

    def __init__(self, power: np.ndarray) -> None:
        power = np.asarray(power, dtype=np.float32)
        if power.ndim != 2 or min(power.shape) < 8:
            raise ValidationError("power", power.shape, "expected a 2-D (time, frequency) array, at least 8x8")
        if not np.all(np.isfinite(power)):
            raise ValidationError("power", "non-finite", "contains NaN or infinity")
        power = power.copy()
        power.setflags(write=False)
        self._power = power

    # ------------------------------------------------------------------
    @classmethod
    def from_png(cls, path: str | Path) -> "WaterfallImage":
        from matplotlib import image

        try:
            rgb = np.asarray(image.imread(str(path)), dtype=np.float32)
        except (OSError, ValueError, SyntaxError) as error:
            raise WaterfallDecodeError(f"cannot read {Path(path).name}: {error}") from error
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise WaterfallDecodeError(f"{Path(path).name} is not a colour picture")
        return cls.from_rgb(rgb[:, :, :3])

    @classmethod
    def from_rgb(cls, rgb: np.ndarray) -> "WaterfallImage":
        row0, row1, col0, col1, index = _locate(rgb)
        power = (index[row0:row1, col0:col1] / 255.0).astype(np.float32)
        return cls(power[::-1])  # start of the pass first

    # ------------------------------------------------------------------
    @property
    def power(self) -> np.ndarray:
        return self._power

    @property
    def shape(self) -> tuple[int, int]:
        return self._power.shape  # type: ignore[return-value]

    def resized(self, time_bins: int = 256, freq_bins: int = 256) -> "WaterfallImage":
        """The same spectrum on a fixed grid (block means, then linear interpolation)."""
        if time_bins < 8 or freq_bins < 8:
            raise ValidationError("bins", (time_bins, freq_bins), "at least 8 each")
        out = self._power
        for axis, size in ((0, time_bins), (1, freq_bins)):
            old = out.shape[axis]
            positions = (np.arange(size) + 0.5) * old / size - 0.5
            low = np.clip(np.floor(positions).astype(int), 0, old - 1)
            high = np.clip(low + 1, 0, old - 1)
            weight = np.clip(positions - low, 0.0, 1.0)
            shape = [1, 1]
            shape[axis] = size
            weight = weight.reshape(shape)
            out = np.take(out, low, axis=axis) * (1 - weight) + np.take(out, high, axis=axis) * weight
        return WaterfallImage(out)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, power=(self._power * 65535).astype(np.uint16))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "WaterfallImage":
        with np.load(path) as data:
            return cls(data["power"].astype(np.float32) / 65535.0)

    def __repr__(self) -> str:
        return f"WaterfallImage({self.shape[0]} time x {self.shape[1]} frequency)"
