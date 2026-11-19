"""Motion: every animated widget in the application.

Each animation answers something the user did -- a press, a toggle, a
tab change, a result arriving -- and lasts about a quarter of a second,
so the application feels alive without ever making anyone wait.  A
single switch, :attr:`Motion.enabled`, turns all of it off for anyone
who prefers a still interface; every widget here checks it.

DESIGN NOTE: INHERITANCE THAT KEEPS THE CONTRACT
------------------------------------------------
Every widget here *is* the Qt widget it replaces.  :class:`ToggleSwitch`
is a ``QCheckBox`` that paints itself as a switch; :class:`Chip` is a
``QCheckBox`` that paints itself as a coloured pill; :class:`PillButton`
is a ``QPushButton``.  Each overrides how it is *drawn* and nothing
about how it *behaves* -- ``isChecked()``, ``toggled``, ``clicked`` all
work exactly as before.  So the pages swap them in without changing a
line of their logic, and every test written against the old widgets
still passes.  That is the Liskov substitution principle, earned
rather than recited.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import (
    QEasingCurve,
    QObject,
    QPoint,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import theme


# ======================================================================
# Settings and the one animation helper everything uses
# ======================================================================
class Motion:
    """Global animation settings.

    Durations follow one scale so every movement in the application
    feels related: quick for feedback on a press, medium for a state
    change, slow only for something arriving on screen.
    """

    enabled = True
    FAST = 140
    MEDIUM = 240
    SLOW = 380

    @classmethod
    def ms(cls, duration: int) -> int:
        return int(duration) if cls.enabled else 0


def animate(
    owner: QObject,
    start: float,
    end: float,
    duration: int,
    on_value: Callable[[float], None],
    easing: QEasingCurve.Type = QEasingCurve.OutCubic,
    on_done: Callable[[], None] | None = None,
) -> QVariantAnimation | None:
    """Run ``on_value`` from ``start`` to ``end`` over ``duration`` ms.

    With motion off (or a zero duration) it jumps straight to the end,
    so callers never need a second code path for the still interface.
    The animation is parented to ``owner`` so it lives exactly as long
    as the widget it moves.
    """
    duration = Motion.ms(duration)
    if duration <= 0:
        on_value(end)
        if on_done is not None:
            on_done()
        return None
    animation = QVariantAnimation(owner)
    animation.setStartValue(float(start))
    animation.setEndValue(float(end))
    animation.setDuration(duration)
    animation.setEasingCurve(easing)
    animation.valueChanged.connect(lambda value: on_value(float(value)))
    if on_done is not None:
        animation.finished.connect(on_done)
    animation.finished.connect(animation.deleteLater)
    animation.start()
    return animation


def stop(animation: QVariantAnimation | None) -> None:
    """Stop an animation that may already have finished and deleted itself.

    Finished animations delete themselves so they do not pile up; a
    Python reference kept to one can therefore outlive the Qt object it
    points at.  Stopping it then raises, and there is nothing left to
    stop, so that is ignored.
    """
    if animation is None:
        return
    try:
        animation.stop()
    except RuntimeError:
        pass


def mix(a: str | QColor, b: str | QColor, t: float) -> QColor:
    """Blend two colours: t=0 gives a, t=1 gives b."""
    a, b = QColor(a), QColor(b)
    t = max(0.0, min(1.0, t))
    return QColor(
        round(a.red() + (b.red() - a.red()) * t),
        round(a.green() + (b.green() - a.green()) * t),
        round(a.blue() + (b.blue() - a.blue()) * t),
        round(a.alpha() + (b.alpha() - a.alpha()) * t),
    )


def brand_gradient(rect: QRectF, colours=None) -> QLinearGradient:
    colours = colours or theme.BRAND
    gradient = QLinearGradient(rect.topLeft(), rect.topRight())
    for index, colour in enumerate(colours):
        gradient.setColorAt(index / (len(colours) - 1), QColor(colour))
    return gradient


def font(family: str = theme.SANS, points: float = theme.BASE_PT) -> QFont:
    face = QFont(family)
    face.setPointSizeF(points)
    return face


# ======================================================================
# Buttons
# ======================================================================
class PillButton(QPushButton):
    """A rounded button that answers every hover, press and click.

    Hover warms the fill; pressing dips it; a click sends a ripple out
    from exactly where the pointer was.  Four variants:

    ``primary``   the aurora gradient -- the one main action on a page
    ``secondary`` white with a border -- ordinary actions
    ``quiet``     no fill until hovered -- minor actions
    ``step``      checkable; filled with ``accent`` when chosen
    """

    VARIANTS = ("primary", "secondary", "quiet", "step")

    def __init__(self, text: str = "", variant: str = "secondary", accent: str | None = None,
                 accent_ink: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")
        self._variant = variant
        self._accent = QColor(accent or theme.BLUSH)
        self._accent_ink = QColor(accent_ink or theme.BRAND_DEEP)
        self._hover = 0.0
        self._press = 0.0
        self._checked_level = 0.0
        self._ripples: list[list[float]] = []      # [x, y, radius, alpha]
        self.setCursor(Qt.PointingHandCursor)
        self.setCheckable(variant == "step")
        self.setMinimumHeight(38)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.toggled.connect(self._on_toggled)

    @property
    def variant(self) -> str:
        return self._variant

    def set_accent(self, colour: str, ink: str | None = None) -> None:
        self._accent = QColor(colour)
        if ink is not None:
            self._accent_ink = QColor(ink)
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        metrics = QFontMetrics(font(theme.SANS_SEMIBOLD))
        return QSize(metrics.horizontalAdvance(self.text()) + 40, 38)

    # -- animated state ---------------------------------------------------
    def _set(self, name: str, value: float) -> None:
        setattr(self, name, value)
        self.update()

    def enterEvent(self, event) -> None:  # noqa: N802
        animate(self, self._hover, 1.0, Motion.MEDIUM, lambda v: self._set("_hover", v))
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        animate(self, self._hover, 0.0, Motion.MEDIUM, lambda v: self._set("_hover", v))
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.isEnabled():
            animate(self, self._press, 1.0, Motion.FAST, lambda v: self._set("_press", v))
            self._ripple(event.position())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        animate(self, self._press, 0.0, Motion.MEDIUM, lambda v: self._set("_press", v))
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter) and self.isEnabled():
            self._ripple(QPointF(self.width() / 2, self.height() / 2))
        super().keyPressEvent(event)

    def _ripple(self, at: QPointF) -> None:
        if not Motion.enabled:
            return
        ripple = [at.x(), at.y(), 0.0, 0.32]
        self._ripples.append(ripple)
        reach = max(self.width(), self.height()) * 1.2

        def grow(t: float) -> None:
            ripple[2] = reach * t
            ripple[3] = 0.32 * (1.0 - t)
            self.update()

        animate(self, 0.0, 1.0, Motion.SLOW, grow, QEasingCurve.OutQuad,
                on_done=lambda: (self._ripples.remove(ripple) if ripple in self._ripples else None,
                                 self.update()))

    def _on_toggled(self, checked: bool) -> None:
        animate(self, self._checked_level, 1.0 if checked else 0.0, Motion.MEDIUM,
                lambda v: self._set("_checked_level", v))

    # -- drawing ------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        radius = min(theme.RADIUS_CONTROL + 2, rect.height() / 2)
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        enabled = self.isEnabled()
        dip = self._press * 0.10

        if self._variant == "primary":
            if enabled:
                painter.fillPath(path, QBrush(brand_gradient(rect)))
                warm = QColor(theme.BRAND_DEEP)
                warm.setAlpha(int(40 * self._hover + 255 * dip * 0.8))
                painter.fillPath(path, warm)
                border = mix(QColor(theme.SAND), QColor(theme.BRAND_DEEP), self._hover)
            else:
                painter.fillPath(path, QColor(theme.SUNK))
                border = QColor(theme.LINE)
            text_colour = QColor(theme.ON_BRAND) if enabled else QColor(theme.INK_3)
        elif self._variant == "step":
            base = mix(theme.SURFACE, theme.tint(self._accent.name(), 0.7), self._hover)
            fill = mix(base, self._accent, self._checked_level)
            painter.fillPath(path, mix(fill, QColor(0, 0, 0), dip * 0.4))
            text_colour = mix(theme.INK_2, self._accent_ink, self._checked_level)
            border = mix(theme.LINE, self._accent_ink, max(self._hover * 0.4, self._checked_level * 0.8))
        elif self._variant == "quiet":
            fill = mix(QColor(255, 255, 255, 0), theme.SUNK, self._hover)
            painter.fillPath(path, mix(fill, theme.LINE, dip * 2))
            text_colour = QColor(theme.INK_2 if enabled else theme.INK_3)
            border = None
        else:  # secondary
            fill = mix(theme.SURFACE, theme.BRAND_SOFT[0], self._hover)
            painter.fillPath(path, mix(fill, theme.PEACH, dip * 3))
            text_colour = QColor(theme.INK if enabled else theme.INK_3)
            border = mix(theme.LINE, theme.SAND, self._hover)

        if border is not None:
            painter.setPen(QPen(border, 1))
            painter.drawPath(path)

        if self._ripples:
            painter.save()
            painter.setClipPath(path)
            for x, y, radius_now, alpha in self._ripples:
                ink = QColor(255, 255, 255) if self._variant == "primary" \
                    else QColor(self._accent_ink)
                ink.setAlphaF(max(0.0, alpha))
                painter.setPen(Qt.NoPen)
                painter.setBrush(ink)
                painter.drawEllipse(QPointF(x, y), radius_now, radius_now)
            painter.restore()

        if self.hasFocus() and self.focusPolicy() != Qt.NoFocus:
            ring = QPainterPath()
            ring.addRoundedRect(rect.adjusted(-0.5, -0.5, 0.5, 0.5), radius, radius)
            painter.setPen(QPen(QColor(theme.FOCUS), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(ring)

        painter.setPen(text_colour)
        painter.setFont(font(theme.SANS_SEMIBOLD))
        painter.drawText(rect.translated(0, self._press * 1.0), Qt.AlignCenter, self.text())


# ======================================================================
# Toggles
# ======================================================================
class ToggleSwitch(QCheckBox):
    """A checkbox drawn as a switch whose knob slides when flipped."""

    TRACK_W, TRACK_H = 38, 22

    def __init__(self, text: str = "", accent: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._accent = accent
        self._position = 0.0
        self._hover = 0.0
        self.setCursor(Qt.PointingHandCursor)
        self.toggled.connect(self._slide)

    def setChecked(self, checked: bool) -> None:  # noqa: N802
        super().setChecked(checked)
        if not self.signalsBlocked():
            return
        self._position = 1.0 if checked else 0.0     # programmatic, silent: no animation
        self.update()

    def _slide(self, checked: bool) -> None:
        animate(self, self._position, 1.0 if checked else 0.0, Motion.MEDIUM,
                lambda v: (setattr(self, "_position", v), self.update()), QEasingCurve.OutBack)

    def showEvent(self, event) -> None:  # noqa: N802
        self._position = 1.0 if self.isChecked() else 0.0
        super().showEvent(event)

    def enterEvent(self, event) -> None:  # noqa: N802
        animate(self, self._hover, 1.0, Motion.FAST, lambda v: (setattr(self, "_hover", v), self.update()))
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        animate(self, self._hover, 0.0, Motion.FAST, lambda v: (setattr(self, "_hover", v), self.update()))
        super().leaveEvent(event)

    def sizeHint(self) -> QSize:  # noqa: N802
        metrics = QFontMetrics(font())
        return QSize(self.TRACK_W + 10 + metrics.horizontalAdvance(self.text()) + 4, 28)

    def hitButton(self, pos: QPoint) -> bool:  # noqa: N802 - the whole row is clickable
        return self.rect().contains(pos)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        y = (self.height() - self.TRACK_H) / 2
        track = QRectF(0, y, self.TRACK_W, self.TRACK_H)
        path = QPainterPath()
        path.addRoundedRect(track, self.TRACK_H / 2, self.TRACK_H / 2)

        p = max(0.0, min(1.0, self._position))
        off = mix(theme.LINE, theme.SAND, self._hover * 0.5)
        painter.fillPath(path, off)
        if p > 0:
            on = QColor(self._accent) if self._accent else None
            painter.save()
            painter.setOpacity(p if self.isEnabled() else p * 0.4)
            painter.fillPath(path, on if on is not None else QBrush(
                brand_gradient(track, (theme.BLUSH, theme.BRAND_DEEP))))
            painter.restore()

        knob = self.TRACK_H - 6
        x = 3 + (self.TRACK_W - knob - 6) * self._position
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 28))
        painter.drawEllipse(QRectF(x, y + 4, knob, knob))
        painter.setBrush(QColor(theme.SURFACE))
        painter.drawEllipse(QRectF(x, y + 3, knob, knob))

        if self.hasFocus():
            painter.setPen(QPen(QColor(theme.FOCUS), 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(track.adjusted(-2, -2, 2, 2), self.TRACK_H / 2 + 2, self.TRACK_H / 2 + 2)

        painter.setPen(QColor(theme.INK if self.isEnabled() else theme.INK_3))
        painter.setFont(font())
        painter.drawText(QRectF(self.TRACK_W + 10, 0, self.width() - self.TRACK_W - 10, self.height()),
                         Qt.AlignVCenter | Qt.AlignLeft, self.text())


class Chip(QCheckBox):
    """A receiver as a pill: its colour dot, its name, filled when chosen.

    Chosen chips fill with a pale tint of the receiver's own colour and
    gain a coloured edge, so the selection *is* the legend: the colours
    the user picks here are the colours they will see on every plot.
    """

    def __init__(self, text: str, colour: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._colour = QColor(colour)
        self._level = 0.0
        self._hover = 0.0
        self.setCursor(Qt.PointingHandCursor)
        self.toggled.connect(lambda on: animate(
            self, self._level, 1.0 if on else 0.0, Motion.MEDIUM,
            lambda v: (setattr(self, "_level", v), self.update())))

    def showEvent(self, event) -> None:  # noqa: N802
        self._level = 1.0 if self.isChecked() else 0.0
        super().showEvent(event)

    def enterEvent(self, event) -> None:  # noqa: N802
        animate(self, self._hover, 1.0, Motion.FAST, lambda v: (setattr(self, "_hover", v), self.update()))
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        animate(self, self._hover, 0.0, Motion.FAST, lambda v: (setattr(self, "_hover", v), self.update()))
        super().leaveEvent(event)

    def sizeHint(self) -> QSize:  # noqa: N802
        metrics = QFontMetrics(font(theme.SANS_MEDIUM))
        return QSize(metrics.horizontalAdvance(self.text()) + 54, 36)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self.sizeHint()      # never squeeze the name into the tick

    def hitButton(self, pos: QPoint) -> bool:  # noqa: N802
        return self.rect().contains(pos)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        enabled = self.isEnabled()
        colour = self._colour if enabled else QColor(theme.LINE)

        soft = QColor(theme.tint(colour.name(), 0.86))
        fill = mix(mix(theme.SURFACE, theme.SUNK, self._hover * 0.7), soft, self._level)
        painter.fillPath(path, fill)
        edge = mix(theme.LINE, colour, max(self._level, self._hover * 0.5))
        pen = QPen(edge, 1.5 if self._level > 0.5 else 1)
        if not enabled:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawPath(path)

        cy = rect.center().y()
        painter.setPen(Qt.NoPen)
        painter.setBrush(colour)
        painter.drawEllipse(QPointF(rect.left() + 16, cy), 5, 5)

        painter.setPen(QColor(theme.INK if enabled else theme.INK_3))
        painter.setFont(font(theme.SANS_MEDIUM))
        text_rect = QRectF(rect.left() + 28, rect.top(), rect.width() - 50, rect.height())
        text = QFontMetrics(painter.font()).elidedText(self.text(), Qt.ElideRight, int(text_rect.width()))
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, text)

        if self._level > 0.01:
            painter.save()
            painter.setOpacity(self._level)
            tick = QPainterPath()
            x, y = rect.right() - 20, cy
            tick.moveTo(x, y)
            tick.lineTo(x + 4, y + 4)
            tick.lineTo(x + 11, y - 4)
            painter.setPen(QPen(colour.darker(115), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(tick)
            painter.restore()

        if self.hasFocus():
            painter.setPen(QPen(QColor(theme.FOCUS), 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect.adjusted(-1, -1, 1, 1), rect.height() / 2 + 1, rect.height() / 2 + 1)


# ======================================================================
# Tabs that slide
# ======================================================================
class TabBar(QWidget):
    """The tab strip: each tab has its own colour, and a pill slides between them."""

    clicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._titles: list[str] = []
        self._current = 0
        self._pill = QRectF()
        self._pill_colour = QColor(theme.BLUSH)
        self._hovered = -1
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(44)
        self.setFocusPolicy(Qt.StrongFocus)

    def add(self, title: str) -> None:
        self._titles.append(title)
        self.updateGeometry()
        self._pill = self._tab_rect(self._current)
        self.update()

    def _tab_rect(self, index: int) -> QRectF:
        metrics = QFontMetrics(font(theme.SANS_SEMIBOLD))
        x = 0.0
        for i, title in enumerate(self._titles):
            width = metrics.horizontalAdvance(title) + 46
            if i == index:
                return QRectF(x, 4, width, self.height() - 8)
            x += width + 6
        return QRectF()

    def sizeHint(self) -> QSize:  # noqa: N802
        if not self._titles:
            return QSize(0, 44)
        last = self._tab_rect(len(self._titles) - 1)
        return QSize(int(last.right()) + 2, 44)

    def set_current(self, index: int) -> None:
        start, end = QRectF(self._pill), self._tab_rect(index)
        start_colour = QColor(self._pill_colour)
        end_colour = QColor(theme.page_fill(self._titles[index]))
        self._current = index

        def step(t: float) -> None:
            self._pill = QRectF(
                start.x() + (end.x() - start.x()) * t, end.y(),
                start.width() + (end.width() - start.width()) * t, end.height(),
            )
            self._pill_colour = mix(start_colour, end_colour, t)
            self.update()

        if start.isNull():
            step(1.0)
        else:
            animate(self, 0.0, 1.0, Motion.MEDIUM, step, QEasingCurve.OutCubic)

    def _index_at(self, pos) -> int:
        for i in range(len(self._titles)):
            if self._tab_rect(i).contains(QPointF(pos)):
                return i
        return -1

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        index = self._index_at(event.position())
        if index != self._hovered:
            self._hovered = index
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hovered = -1
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        index = self._index_at(event.position())
        if index >= 0:
            self.clicked.emit(index)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Right, Qt.Key_Left) and self._titles:
            step = 1 if event.key() == Qt.Key_Right else -1
            self.clicked.emit((self._current + step) % len(self._titles))
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if not self._pill.isNull():
            pill = QPainterPath()
            pill.addRoundedRect(self._pill, self._pill.height() / 2, self._pill.height() / 2)
            painter.fillPath(pill, self._pill_colour)
        for i, title in enumerate(self._titles):
            rect = self._tab_rect(i)
            accent = QColor(theme.page_accent(title))
            if i == self._hovered and i != self._current:
                hover = QPainterPath()
                hover.addRoundedRect(rect, rect.height() / 2, rect.height() / 2)
                painter.fillPath(hover, QColor(theme.SUNK))
            painter.setPen(Qt.NoPen)
            painter.setBrush(accent)
            painter.drawEllipse(QPointF(rect.left() + 17, rect.center().y()), 4, 4)
            painter.setPen(accent.darker(125) if i == self._current else QColor(theme.INK_2))
            painter.setFont(font(theme.SANS_SEMIBOLD))
            painter.drawText(rect.adjusted(28, 0, -12, 0), Qt.AlignVCenter | Qt.AlignLeft, title)
        if self.hasFocus() and not self._pill.isNull():
            painter.setPen(QPen(QColor(theme.FOCUS), 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self._pill, self._pill.height() / 2, self._pill.height() / 2)


class AnimatedTabs(QWidget):
    """Pages that slide and fade in, behind a :class:`TabBar`.

    Offers the parts of ``QTabWidget``'s interface the window uses --
    ``addTab``, ``count``, ``tabText``, ``currentWidget``,
    ``setCurrentWidget``, ``currentChanged`` -- so it replaced the old
    tab widget without the window or its tests noticing.  The bar is a
    separate widget, :attr:`bar`, so it can sit in the header rather
    than above the pages.
    """

    currentChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bar = TabBar()
        self._stack = QStackedWidget()
        self._titles: list[str] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack)
        self.bar.clicked.connect(self.setCurrentIndex)

    def addTab(self, widget: QWidget, title: str) -> int:  # noqa: N802
        self._titles.append(title)
        self.bar.add(title)
        index = self._stack.addWidget(widget)
        if index == 0:
            self.bar.set_current(0)
        return index

    def count(self) -> int:
        return self._stack.count()

    def tabText(self, index: int) -> str:  # noqa: N802
        return self._titles[index]

    def currentIndex(self) -> int:  # noqa: N802
        return self._stack.currentIndex()

    def currentWidget(self) -> QWidget:  # noqa: N802
        return self._stack.currentWidget()

    def widget(self, index: int) -> QWidget:
        return self._stack.widget(index)

    def setCurrentWidget(self, widget: QWidget) -> None:  # noqa: N802
        self.setCurrentIndex(self._stack.indexOf(widget))

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802
        if index < 0 or index == self._stack.currentIndex():
            return
        forward = index > self._stack.currentIndex()
        self._stack.setCurrentIndex(index)
        self.bar.set_current(index)
        self._enter(self._stack.widget(index), forward)
        self.currentChanged.emit(index)

    def _enter(self, page: QWidget, forward: bool) -> None:
        """Slide the new page a few pixels in from its side while it fades up."""
        if not Motion.enabled:
            return
        effect = QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(effect)
        offset = 28 if forward else -28
        home = QPoint(0, 0)

        def step(t: float) -> None:
            effect.setOpacity(t)
            page.move(home + QPoint(round(offset * (1 - t)), 0))

        def done() -> None:
            page.move(home)
            # Remove the effect afterwards: while present, Qt renders the
            # whole page to an image every frame, which plots dislike.
            page.setGraphicsEffect(None)

        animate(page, 0.0, 1.0, Motion.SLOW, step, QEasingCurve.OutCubic, done)


# ======================================================================
# Things that arrive
# ======================================================================
class Toast(QFrame):
    """A short message that slides up in the corner, then leaves."""

    KINDS = {"success": (theme.GOOD, "✓"), "info": (theme.BRAND_DEEP, "i"), "error": (theme.CRITICAL, "!")}

    def __init__(self, text: str, kind: str, parent: QWidget) -> None:
        super().__init__(parent)
        colour, glyph = self.KINDS.get(kind, self.KINDS["info"])
        self.setObjectName("Toast")
        self.setStyleSheet(
            f"QFrame#Toast {{ background: {theme.INK}; border-radius: 12px; }}"
            f"QLabel {{ color: {theme.SURFACE}; background: transparent; }}"
        )
        badge = QLabel(glyph)
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(24, 24)
        badge.setStyleSheet(
            f"background: {colour}; color: white; border-radius: 12px; "
            f"font-family: '{theme.SANS_EXTRABOLD}'; font-size: 10pt;"
        )
        label = QLabel(text)
        label.setWordWrap(True)
        label.setMaximumWidth(340)
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 12, 18, 12)
        row.setSpacing(12)
        row.addWidget(badge, 0, Qt.AlignTop)
        row.addWidget(label, 1)
        self.adjustSize()
        self._effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._effect)
        self._effect.setOpacity(0.0)


class ToastHost(QObject):
    """Shows toasts stacked in a window's bottom-right corner."""

    MARGIN = 20

    def __init__(self, window: QWidget) -> None:
        super().__init__(window)
        self._window = window
        self._toasts: list[Toast] = []

    def show(self, text: str, kind: str = "info", seconds: float = 3.5) -> Toast:
        toast = Toast(text, kind, self._window)
        self._toasts.append(toast)
        toast.show()
        toast.raise_()
        self._arrange(new=toast)
        QTimer.singleShot(int(seconds * 1000), lambda: self._dismiss(toast))
        return toast

    def _target(self, index: int) -> QPoint:
        y = self._window.height() - self.MARGIN - 34
        for toast in reversed(self._toasts[index:]):
            y -= toast.height() + 10
        toast = self._toasts[index]
        return QPoint(self._window.width() - toast.width() - self.MARGIN, y + 10)

    def _arrange(self, new: Toast | None = None) -> None:
        for index, toast in enumerate(self._toasts):
            end = self._target(index)
            if toast is new:
                toast.move(end + QPoint(0, 24))
                start_y = toast.y()
                animate(toast, 0.0, 1.0, Motion.SLOW,
                        lambda t, w=toast, s=start_y, e=end: (
                            w.move(e.x(), round(s + (e.y() - s) * t)),
                            w._effect.setOpacity(t)),
                        QEasingCurve.OutBack)
            else:
                start = toast.pos()
                animate(toast, 0.0, 1.0, Motion.MEDIUM,
                        lambda t, w=toast, s=start, e=end: w.move(
                            round(s.x() + (e.x() - s.x()) * t), round(s.y() + (e.y() - s.y()) * t)))

    def _dismiss(self, toast: Toast) -> None:
        if toast not in self._toasts:
            return

        def gone() -> None:
            if toast in self._toasts:
                self._toasts.remove(toast)
            toast.deleteLater()
            self._arrange()

        animate(toast, 1.0, 0.0, Motion.MEDIUM, lambda t: toast._effect.setOpacity(t), on_done=gone)

    @property
    def visible(self) -> list[Toast]:
        return list(self._toasts)


class CountUpLabel(QLabel):
    """A number that counts to its new value instead of jumping."""

    def __init__(self, formatter: Callable[[float], str] = lambda v: f"{v:.0f}",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._formatter = formatter
        self._value = 0.0
        self._target = 0.0
        self.setText(formatter(0.0))

    @property
    def target(self) -> float:
        return self._target

    def set_value(self, value: float) -> None:
        self._target = float(value)
        animate(self, self._value, self._target, Motion.SLOW,
                lambda v: (setattr(self, "_value", v), self.setText(self._formatter(v))))


class LoadingBar(QWidget):
    """A thin aurora streak that sweeps under the header while work is in flight."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(3)
        self._phase = 0.0
        self._opacity = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    @property
    def active(self) -> bool:
        return self._timer.isActive() and self._opacity > 0

    def set_busy(self, busy: bool) -> None:
        if busy:
            if not self._timer.isActive():
                self._timer.start()
            animate(self, self._opacity, 1.0, Motion.FAST, lambda v: setattr(self, "_opacity", v))
        else:
            animate(self, self._opacity, 0.0, Motion.MEDIUM,
                    lambda v: (setattr(self, "_opacity", v), self.update()),
                    on_done=self._timer.stop)

    def _tick(self) -> None:
        self._phase = (self._phase + 0.012) % 1.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        if self._opacity <= 0:
            return
        painter = QPainter(self)
        painter.setOpacity(self._opacity)
        width = self.width()
        streak = width * 0.35
        x = -streak + (width + streak) * self._phase
        rect = QRectF(x, 0, streak, self.height())
        gradient = QLinearGradient(rect.topLeft(), rect.topRight())
        gradient.setColorAt(0.0, QColor(theme.PEACH))
        gradient.setColorAt(0.5, QColor(theme.BLUSH))
        gradient.setColorAt(1.0, QColor(theme.BRAND_DEEP))
        painter.fillRect(rect, gradient)


class SmoothProgress(QWidget):
    """A rounded progress bar in the brand gradient that eases to each new value."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(8)
        self._shown = 0.0
        self._target = 0.0

    def value(self) -> float:
        return self._target

    def set_fraction(self, fraction: float) -> None:
        self._target = max(0.0, min(1.0, float(fraction)))
        animate(self, self._shown, self._target, Motion.SLOW,
                lambda v: (setattr(self, "_shown", v), self.update()))

    def reset(self) -> None:
        self._shown = self._target = 0.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect())
        track = QPainterPath()
        track.addRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        painter.fillPath(track, QColor(theme.SUNK))
        if self._shown > 0:
            done = QRectF(rect.x(), rect.y(), max(rect.height(), rect.width() * self._shown), rect.height())
            bar = QPainterPath()
            bar.addRoundedRect(done, rect.height() / 2, rect.height() / 2)
            painter.fillPath(bar, QBrush(brand_gradient(rect, (theme.BLUSH, theme.BRAND_DEEP))))


class Collapsible(QWidget):
    """A section that opens and closes smoothly when its header is clicked."""

    toggled = Signal(bool)

    def __init__(self, title: str, content: QWidget, expanded: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._content = content
        self._open = expanded
        self.header = PillButton(self._label(), "quiet")
        self.header.clicked.connect(self.toggle)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.header, 0, Qt.AlignLeft)
        layout.addWidget(content)
        content.setMaximumHeight(16777215 if expanded else 0)
        content.setVisible(expanded)

    def _label(self) -> str:
        return f"{'▾' if self._open else '▸'}  {self._title}"

    @property
    def expanded(self) -> bool:
        return self._open

    def toggle(self) -> None:
        self.set_expanded(not self._open)

    def set_expanded(self, expanded: bool) -> None:
        if expanded == self._open:
            return
        self._open = expanded
        self.header.setText(self._label())
        content = self._content
        full = max(content.sizeHint().height(), content.minimumSizeHint().height())
        if expanded:
            content.setMaximumHeight(0)
            content.setVisible(True)
            animate(self, 0, full, Motion.MEDIUM, lambda v: content.setMaximumHeight(int(v)),
                    on_done=lambda: content.setMaximumHeight(16777215))
        else:
            animate(self, content.height(), 0, Motion.MEDIUM, lambda v: content.setMaximumHeight(int(v)),
                    on_done=lambda: content.setVisible(False))
        self.toggled.emit(expanded)


def fade_in(widget: QWidget, duration: int = Motion.MEDIUM) -> None:
    """Fade a widget up from transparent. Removes the effect when done."""
    if not Motion.enabled:
        return
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    effect.setOpacity(0.0)
    animate(widget, 0.0, 1.0, duration, effect.setOpacity,
            on_done=lambda: widget.setGraphicsEffect(None))


__all__ = [
    "AnimatedTabs", "Chip", "Collapsible", "CountUpLabel", "LoadingBar", "Motion",
    "PillButton", "SmoothProgress", "TabBar", "Toast", "ToastHost", "ToggleSwitch",
    "animate", "brand_gradient", "fade_in", "mix", "stop",
]
