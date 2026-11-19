"""What every tab of the application has in common.

DESIGN NOTE: ABSTRACTION -- AND A CASE WHERE PYTHON'S OWN CHECK FAILS
----------------------------------------------------------------------
:class:`Page` is abstract: every tab must say what it is called and how
it loads its data, and the main window relies on both.  Normally
``abc.ABC`` enforces that by refusing to create an object from a class
with unimplemented abstract methods.

Qt widgets break that.  A ``QWidget`` subclass is created by Qt's own
C++ machinery, which never runs the line of Python that does the
refusing.  Measured while building this phase: a ``QWidget`` subclass
with an unimplemented abstract method was created without complaint.

So this class does two things.  It combines ``ABCMeta`` with Qt's
metaclass -- in that order, for the reason given on ``_PageMeta`` -- so
Python still *computes* which methods are missing, and it checks that
list itself, in ``__init__``, so the guarantee holds again.  The test
suite checks it directly: an incomplete page must fail to construct.

DESIGN NOTE: POLYMORPHISM
-------------------------
The main window holds a list of pages and treats them identically: adds
each as a tab under its ``title``, calls ``refresh()`` on each when the
backend connects or a model is trained.  It has no idea which page is
which.  Adding a fifth tab means writing one subclass and adding it to
the list -- nothing else changes.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from .. import theme

from ..client import BackendClient, BackendError
from ..tasks import TaskRunner


class _PageMeta(ABCMeta, type(QWidget)):
    """ABCMeta and Qt's metaclass, combined so a widget can be abstract.

    The order matters, and getting it wrong fails silently.  Qt's
    metaclass does not hand control on to the next metaclass in line, so
    with Qt's listed first, ``ABCMeta`` never runs and Python never works
    out which methods are missing.  Listed first, ``ABCMeta`` does its
    work and then passes control to Qt's.  The first version of this
    file had them the other way round; the test suite caught it.
    """


class Page(QWidget, metaclass=_PageMeta):
    """Base class for every tab.

    Subclasses implement :attr:`title` and :meth:`refresh`, and use
    :attr:`tasks` for every backend call so the window never freezes.
    """

    #: A message for the status bar: (text, is_error).
    message = Signal(str, bool)

    #: Something worth a pop-up: (text, kind), kind being success, info or error.
    notify = Signal(str, str)

    def __init__(self, client: BackendClient, parent: QWidget | None = None) -> None:
        missing = sorted(getattr(type(self), "__abstractmethods__", ()))
        if missing:
            # The check abc would normally make, restored by hand.
            raise TypeError(
                f"Can't instantiate abstract class {type(self).__name__} "
                f"without an implementation for: {', '.join(missing)}"
            )
        super().__init__(parent)
        self.setObjectName("Page")
        self._client = client
        self.tasks = TaskRunner(self)
        self._page_loaded = False

    @property
    def client(self) -> BackendClient:
        return self._client

    @property
    def loaded(self) -> bool:
        """Has this page fetched its data at least once?"""
        return self._page_loaded

    # -- what every page must provide -----------------------------------
    @property
    @abstractmethod
    def title(self) -> str:
        """The tab's label."""

    @abstractmethod
    def refresh(self) -> None:
        """(Re)load whatever this page shows from the backend."""

    # -- shared behaviour -----------------------------------------------
    def ensure_loaded(self) -> None:
        """Load on first view only, so start-up does not wait for every tab."""
        if not self._page_loaded:
            self._page_loaded = True
            self.refresh()

    def report(self, text: str) -> None:
        self.message.emit(text, False)

    def report_error(self, error: BackendError | Exception | str) -> None:
        if isinstance(error, BackendError):
            text = error.detail
        else:
            text = str(error)
        self.message.emit(text, True)

    def heading(self, title: str, hint: str = "") -> QWidget:
        """The page's title beside a badge in the tab's own colour, and one line of purpose."""
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(2, 0, 0, 6)
        row.setSpacing(14)
        row.addWidget(_Badge(theme.page_fill(self.title), theme.page_accent(self.title)), 0, Qt.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(2)
        label = QLabel(title)
        label.setObjectName("PageTitle")
        text.addWidget(label)
        if hint:
            line = QLabel(hint)
            line.setObjectName("Hint")
            line.setWordWrap(True)
            text.addWidget(line)
        row.addLayout(text, 1)
        return box

    @staticmethod
    def scrolling(content: QWidget, width: int | None = None) -> QScrollArea:
        """Wrap a column so it scrolls on small screens instead of squashing."""
        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)
        if width is not None:
            scroll.setFixedWidth(width)
        return scroll

    @staticmethod
    def row(*widgets: QWidget, stretch_last: bool = False) -> QWidget:
        box = QWidget()
        layout = QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for widget in widgets:
            layout.addWidget(widget)
        if not stretch_last:
            layout.addStretch(1)
        return box


__all__ = ["Page"]


class _Badge(QWidget):
    """A rounded square in the tab's soft colour, with a dot in its strong one."""

    def __init__(self, fill: str, ink: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fill, self._ink = QColor(fill), QColor(ink)
        self.setFixedSize(44, 44)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._fill)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 13, 13)
        painter.setBrush(self._ink)
        painter.drawEllipse(QPointF(self.width() / 2, self.height() / 2), 6, 6)
