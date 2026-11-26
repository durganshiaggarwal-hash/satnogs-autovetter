"""Keeping the window responsive while the backend works.

Qt draws the window, and reacts to every click and drag, on one thread.
Anything slow done on that thread -- a network request, a minute-long
benchmark -- freezes the whole application until it finishes.  So every
request goes to a background thread, and the answer comes back as a Qt
*signal*, which Qt delivers safely on the window's own thread.

Two tools:

:class:`TaskRunner`
    Run one call in the background; get its result or its error back as
    a callback.  Used for every request.
:class:`JobWatcher`
    Poll a backend job every few hundred milliseconds and report its
    progress until it finishes.  Used for benchmarks and training.

DESIGN NOTE: THE OBSERVER PATTERN, QT'S WAY
-------------------------------------------
Qt's signals and slots are the Observer pattern built into the toolkit.
A :class:`JobWatcher` *emits* ``progress`` and ``finished``; any number
of widgets *connect* to them.  The watcher does not know who is
listening, which is why the same watcher drives a progress bar, a
status line and a results reload without any of them knowing about the
others.
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot

from .client import BackendClient, BackendError


class _Signals(QObject):
    """Signals for one background call. QRunnable cannot own signals itself."""

    succeeded = Signal(object)
    failed = Signal(object)


class _Call(QRunnable):
    def __init__(self, function: Callable[[], Any]) -> None:
        super().__init__()
        self.signals = _Signals()
        self._function = function

    @Slot()
    def run(self) -> None:
        try:
            result = self._function()
        except BackendError as error:
            self._send(self.signals.failed, error)
        except Exception as error:  # noqa: BLE001 - surfaced to the user, not swallowed
            self._send(self.signals.failed, BackendError(-1, f"{type(error).__name__}: {error}"))
        else:
            self._send(self.signals.succeeded, result)

    @staticmethod
    def _send(signal, value) -> None:
        """Deliver the result, unless the window closed while the call ran."""
        try:
            signal.emit(value)
        except RuntimeError:          # "Signal source has been deleted"
            pass


class TaskRunner(QObject):
    """Runs calls on a thread pool and delivers results on the GUI thread.

    Also tracks how many calls are in flight, so a page can show that it
    is busy and the main window can show it in the status bar.
    """

    busy_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()
        self._pending: set[_Call] = set()

    @property
    def busy(self) -> bool:
        return bool(self._pending)

    def run(
        self,
        function: Callable[[], Any],
        on_success: Callable[[Any], None],
        on_error: Callable[[BackendError], None] | None = None,
    ) -> None:
        call = _Call(function)
        call.setAutoDelete(False)          # we keep it until its signal arrives
        was_busy = self.busy
        self._pending.add(call)

        def finish(handler, value):
            self._pending.discard(call)
            if handler is not None:
                handler(value)
            if not self.busy:
                self.busy_changed.emit(False)

        call.signals.succeeded.connect(lambda value: finish(on_success, value))
        call.signals.failed.connect(lambda error: finish(on_error, error))
        if not was_busy:
            self.busy_changed.emit(True)
        self._pool.start(call)

    def wait(self, timeout_ms: int = 30_000) -> bool:
        """Block until every call has finished. For tests only."""
        return self._pool.waitForDone(timeout_ms)


class JobWatcher(QObject):
    """Follows one backend job to completion.

    Emits ``progress(fraction, message)`` while it runs and exactly one
    of ``finished(job)`` or ``failed(message)`` at the end.
    """

    progress = Signal(float, str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, client: BackendClient, job_id: str, interval_ms: int = 400,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._job_id = job_id
        self._runner = TaskRunner(self)
        self._timer = QTimer(self)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self._poll)
        self._polling = False

    @property
    def job_id(self) -> str:
        return self._job_id

    def start(self) -> "JobWatcher":
        self._timer.start()
        self._poll()
        return self

    def stop(self) -> None:
        self._timer.stop()

    def _poll(self) -> None:
        if self._polling:              # never stack requests on a slow backend
            return
        self._polling = True
        self._runner.run(lambda: self._client.job(self._job_id), self._update, self._lost)

    def _update(self, job: dict) -> None:
        self._polling = False
        self.progress.emit(float(job.get("fraction", 0.0)), str(job.get("message", "")))
        if job["state"] == "done":
            self.stop()
            self.finished.emit(job)
        elif job["state"] == "failed":
            self.stop()
            self.failed.emit(str(job.get("error") or "the job failed"))

    def _lost(self, error: BackendError) -> None:
        self._polling = False
        self.stop()
        self.failed.emit(error.detail)


__all__ = ["JobWatcher", "TaskRunner"]
