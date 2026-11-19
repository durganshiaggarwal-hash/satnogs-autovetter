"""The plots: constellations, waveforms and error-rate curves -- in motion.

When new data arrives, a plot does not wipe and redraw.  Points glide
from where they were to where they now are; a wave reshapes itself;
curves fade up.  That motion carries meaning: drag the Doppler slider
and you *watch* the constellation rotate; step from timing recovery to
the phase lock and you see the cloud swing round and settle onto the
rings.  The animation shows what the stage did.

DESIGN NOTE: INHERITANCE FOR SHARED CHROME
-------------------------------------------
Every plot sits in the same kind of card: a title, a one-line caption,
a plot area with quiet axes, no stray menu button.  :class:`PlotPanel`
builds that once.  Each subclass adds only what makes it different --
how to draw and animate a constellation, a wave, a curve.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEasingCurve, Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from .. import theme
from ..motion import Motion, animate, stop


def _complex(iq: dict) -> np.ndarray:
    return np.asarray(iq["i"], dtype=float) + 1j * np.asarray(iq["q"], dtype=float)


def _colour(hex_colour: str, alpha: int = 255) -> QColor:
    colour = QColor(hex_colour)
    colour.setAlpha(alpha)
    return colour


def _ease(t: float) -> float:
    return 1 - (1 - t) ** 3


class PlotPanel(QFrame):
    """A titled card holding one plot, styled like every other."""

    def __init__(self, title: str = "", caption: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._title = QLabel(title)
        self._title.setObjectName("CardTitle")
        self._caption = QLabel(caption)
        self._caption.setObjectName("Muted")
        self._caption.setWordWrap(True)
        self._caption.setVisible(bool(caption))

        self.plot = pg.PlotWidget()
        self.plot.setStyleSheet("background: transparent; border: none;")
        item = self.plot.getPlotItem()
        item.hideButtons()
        item.setMenuEnabled(False)
        item.showGrid(x=True, y=True, alpha=0.25)
        tick_font = QFont(theme.SANS, 8)
        for side in ("left", "bottom"):
            axis = item.getAxis(side)
            axis.setPen(pg.mkPen(theme.LINE))
            axis.setTextPen(pg.mkPen(theme.INK_3))
            axis.setStyle(tickFont=tick_font, tickLength=-4)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(3)
        layout.addWidget(self._title)
        layout.addWidget(self._caption)
        layout.addSpacing(4)
        layout.addWidget(self.plot, 1)

    def set_title(self, title: str, caption: str | None = None) -> None:
        self._title.setText(title)
        if caption is not None:
            self._caption.setText(caption)
            self._caption.setVisible(bool(caption))

    def set_axis_labels(self, bottom: str = "", left: str = "") -> None:
        style = {"color": theme.INK_3, "font-size": "8pt"}
        item = self.plot.getPlotItem()
        item.setLabel("bottom", bottom, **style)
        item.setLabel("left", left, **style)

    def clear(self) -> None:
        self.plot.clear()

    def _legend(self):
        legend = self.plot.getPlotItem().legend
        if legend is None:
            legend = self.plot.addLegend(offset=(-10, 10), labelTextColor=theme.INK_2,
                                         labelTextSize="8pt", brush=_colour(theme.SURFACE, 230),
                                         pen=pg.mkPen(theme.LINE))
        legend.clear()
        return legend


class ConstellationPlot(PlotPanel):
    """Where symbols landed, against where they should have.

    The ideal points are hollow rings.  Symbols decided correctly are
    dots in the receiver's colour; symbols decided wrongly are red
    crosses -- a different shape, so the distinction does not depend on
    seeing colour.  When the same number of points arrives again, each
    one glides from its old position to its new one.
    """

    def __init__(self, title: str = "", caption: str = "", parent: QWidget | None = None) -> None:
        super().__init__(title, caption, parent)
        self.plot.setAspectLocked(True)
        self.plot.setMinimumSize(190, 190)
        self._rings = pg.ScatterPlotItem(size=12, symbol="o", pen=pg.mkPen(theme.INK_3, width=1), brush=None)
        self._points = pg.ScatterPlotItem(pxMode=True)
        self.plot.addItem(self._rings)
        self.plot.addItem(self._points)
        self._shown: np.ndarray | None = None
        self._cloud: bool | None = None
        self._style: dict = {}
        self._animation = None
        self._extent = 1.6

    @property
    def positions(self) -> np.ndarray | None:
        """Where the points are now (after any animation), for tests."""
        return None if self._shown is None else self._shown.copy()

    def show_points(
        self,
        points: dict,
        colour: str,
        reference: dict | None = None,
        wrong: list[bool] | None = None,
        cloud: bool = False,
    ) -> None:
        """Draw one set of symbols, gliding from the previous set if it matches."""
        target = _complex(points)
        spread = float(np.percentile(np.abs(target), 99)) if target.size else 1.0
        extent = 1.12 * spread if spread > 0 else 1.6

        if reference is not None:
            ideal = _complex(reference)
            self._rings.setData(ideal.real, ideal.imag)
            self._rings.show()
            if ideal.size:
                extent = max(1.45 * float(np.max(np.abs(ideal))), 1.05 * spread)
        else:
            self._rings.hide()

        if cloud:
            style = {"size": 3, "pen": None, "brush": _colour(colour, 70), "symbol": "o"}
        else:
            bad = np.zeros(target.size, dtype=bool) if wrong is None else np.asarray(wrong, dtype=bool)[: target.size]
            if bad.size < target.size:
                bad = np.pad(bad, (0, target.size - bad.size))
            good_brush, bad_brush = _colour(colour, 160), _colour(theme.CRITICAL)
            good_pen = pg.mkPen(_colour(colour, 235), width=1)
            style = {
                "size": np.where(bad, 11, 7),
                "symbol": ["x" if b else "o" for b in bad],
                "brush": [bad_brush if b else good_brush for b in bad],
                "pen": [None if b else good_pen for b in bad],
            }

        previous = self._shown
        can_glide = (
            Motion.enabled and previous is not None and previous.size == target.size
            and self._cloud == cloud and target.size > 0
        )
        self._cloud = cloud
        self._style = style

        stop(self._animation)
        self._animation = None

        if can_glide:
            start = previous.copy()
            delta = target - start

            def step(t: float) -> None:
                now = start + delta * t
                self._points.setData(now.real, now.imag, **style)
                self._shown = now

            self._animation = animate(self, 0.0, 1.0, Motion.SLOW, step, QEasingCurve.OutCubic)
        else:
            self._points.setData(target.real, target.imag, **style)
            self._shown = target
            if Motion.enabled:
                self._points.setOpacity(0.0)
                animate(self, 0.0, 1.0, Motion.MEDIUM, self._points.setOpacity)

        self._zoom_to(extent)

    def _zoom_to(self, extent: float) -> None:
        start = self._extent

        def step(t: float) -> None:
            e = start + (extent - start) * t
            self.plot.setRange(xRange=(-e, e), yRange=(-e, e), padding=0)

        self._extent = extent
        animate(self, 0.0, 1.0, Motion.SLOW, step, QEasingCurve.OutCubic)


class WaveformPlot(PlotPanel):
    """The wave itself: the in-phase part over time, for the first symbols.

    Only the first couple of dozen symbols, because 256 symbols of wave
    in one panel is a solid band with no visible shape -- and the shape
    is what this plot is for.  A new wave of the same length morphs out
    of the old one.
    """

    def __init__(self, title: str = "", caption: str = "", symbols_shown: int = 24,
                 parent: QWidget | None = None) -> None:
        super().__init__(title, caption, parent)
        self._shown_symbols = int(symbols_shown)
        self.set_axis_labels("time (symbols)", "")
        self.plot.setMinimumHeight(130)
        self._previous: dict[str, np.ndarray] = {}
        self._animation = None

    def show_wave(
        self,
        trace: dict,
        samples_per_symbol: int,
        reference: dict | None = None,
        colour: str = theme.INK,
        dots: bool = False,
        reference_label: str = "what was sent",
        trace_label: str = "what arrived",
    ) -> None:
        """Draw a trace, and optionally the clean original beneath it."""
        step = 1.0 / samples_per_symbol
        count = self._shown_symbols * samples_per_symbol
        wave = _complex(trace)[:count].real
        x = np.arange(wave.size) * step
        ref = _complex(reference)[:count].real if reference is not None else None
        mode = ("dots" if dots else "line", samples_per_symbol, wave.size)

        previous = self._previous.get(mode) if self._previous.get("mode") == mode else None
        self._previous = {"mode": mode, mode: wave}
        stop(self._animation)
        self._animation = None

        self.clear()
        legend = self._legend()
        if ref is not None:
            if dots:
                item = self.plot.plot(np.arange(ref.size) * step, ref, pen=None, symbol="o", symbolSize=9,
                                      symbolPen=pg.mkPen(theme.INK_3, width=1.3), symbolBrush=None,
                                      name=reference_label)
            else:
                item = self.plot.plot(np.arange(ref.size) * step, ref,
                                      pen=pg.mkPen(theme.INK_3, width=1.4, style=Qt.DashLine),
                                      name=reference_label)
        if dots:
            main = self.plot.plot(x, wave, pen=None, symbol="o", symbolSize=7, symbolPen=None,
                                  symbolBrush=_colour(colour, 230), name=trace_label)
        else:
            main = self.plot.plot(x, wave, pen=pg.mkPen(colour, width=2.2), name=trace_label)
        self.plot.setXRange(0, self._shown_symbols, padding=0.01)

        if previous is not None and Motion.enabled:
            start = previous.copy()

            def morph(t: float) -> None:
                main.setData(x, start + (wave - start) * t)

            self._animation = animate(self, 0.0, 1.0, Motion.SLOW, morph, QEasingCurve.OutCubic)
        elif Motion.enabled:
            main.setOpacity(0.0)
            animate(self, 0.0, 1.0, Motion.MEDIUM, main.setOpacity)


class BerCurvePlot(PlotPanel):
    """Error rate against signal strength, one curve per receiver.

    Logarithmic, because error rates span five orders of magnitude and a
    linear axis would flatten everything interesting into the floor.
    Labelled in percentages rather than powers of ten, because "0.1%"
    means something to a first-time reader and "1e-3" does not.

    A measurement of zero errors cannot sit on a log axis.  It is drawn
    at the floor with a downward triangle, meaning "below what these
    frames could measure" -- not "zero", which no finite test can show.
    """

    FLOOR = 1e-5

    def __init__(self, title: str = "", caption: str = "", parent: QWidget | None = None) -> None:
        super().__init__(title, caption, parent)
        self.set_axis_labels("signal strength (dB)", "bits wrong")
        ticks = [(math.log10(v), label) for v, label in (
            (1e-5, "0.001%"), (1e-4, "0.01%"), (1e-3, "0.1%"), (1e-2, "1%"), (1e-1, "10%"), (1.0, "100%"),
        )]
        self.plot.getPlotItem().getAxis("left").setTicks([ticks, []])
        self.plot.setYRange(math.log10(self.FLOOR) - 0.2, 0.05, padding=0)

    def show_curves(self, series: list[dict]) -> None:
        """Each item: ``{name, label, snr, ber, low, high}`` (lists)."""
        self.clear()
        self._legend()

        def lg(values):
            return np.log10(np.maximum(np.asarray(values, dtype=float), self.FLOOR))

        drawn = []
        for entry in series:
            colour = theme.receiver_colour(entry["name"])
            snr = np.asarray(entry["snr"], dtype=float)
            ber = np.asarray(entry["ber"], dtype=float)
            order = np.argsort(snr)
            snr, ber = snr[order], ber[order]
            low = np.asarray(entry["low"], dtype=float)[order]
            high = np.asarray(entry["high"], dtype=float)[order]

            upper = pg.PlotDataItem(snr, lg(high), pen=None)
            lower = pg.PlotDataItem(snr, lg(low), pen=None)
            band = pg.FillBetweenItem(lower, upper, brush=_colour(colour, 40))
            for item in (upper, lower, band):
                self.plot.addItem(item)

            measured = ber > 0
            line = self.plot.plot(snr, lg(ber), pen=pg.mkPen(colour, width=2.4), name=entry["label"])
            dots = pg.ScatterPlotItem(
                snr[measured], lg(ber[measured]), size=9, symbol="o",
                pen=pg.mkPen(theme.SURFACE, width=1.5), brush=_colour(colour),
            )
            self.plot.addItem(dots)
            drawn += [band, line, dots]
            if (~measured).any():
                floor = pg.ScatterPlotItem(
                    snr[~measured], lg(ber[~measured]), size=10, symbol="t",
                    pen=pg.mkPen(colour, width=1.6), brush=_colour(theme.SURFACE),
                )
                self.plot.addItem(floor)
                drawn.append(floor)
        if series:
            all_snr = np.concatenate([np.asarray(s["snr"], dtype=float) for s in series])
            self.plot.setXRange(all_snr.min() - 0.5, all_snr.max() + 0.5, padding=0)
        self.plot.setYRange(math.log10(self.FLOOR) - 0.2, 0.05, padding=0)

        if Motion.enabled and drawn:
            for item in drawn:
                item.setOpacity(0.0)
            animate(self, 0.0, 1.0, Motion.SLOW, lambda t: [i.setOpacity(t) for i in drawn])


__all__ = ["BerCurvePlot", "ConstellationPlot", "PlotPanel", "WaveformPlot"]
