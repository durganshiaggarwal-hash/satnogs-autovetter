"""Work that takes too long to do inside one HTTP request.

A benchmark takes minutes; training a model takes tens of seconds.  An
HTTP request should answer in well under a second, so long work becomes
a *job*: the request starts it and immediately gets back an id, and the
GUI asks ``GET /api/jobs/{id}`` every so often to draw a progress bar.

Jobs run one at a time, in order.  Two benchmarks running at once would
each take twice as long and finish no sooner, and a training job racing
a benchmark that uses its model would be asking for trouble.

DESIGN NOTE: A SMALL STATE MACHINE, ENCAPSULATED
-------------------------------------------------
A :class:`Job` moves ``queued -> running -> done`` or ``-> failed``, and
never any other way.  The transitions are private methods called only
by :class:`JobManager`; everything outside reads a snapshot.  So no
request handler can mark a job done that is still running, and the
state a client sees is always one the job really passed through.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Callable

from ..core.exceptions import SatRecoveryError, ValidationError

#: The work a job does.  It is handed a ``report(done, total, message)``
#: function to call as it goes, and returns a JSON-safe result.
Work = Callable[[Callable[[int, int, str], None]], Any]


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Job:
    """One piece of background work and everything known about it."""

    def __init__(self, kind: str, description: str) -> None:
        self._id = uuid.uuid4().hex[:12]
        self._kind = str(kind)
        self._description = str(description)
        self._state = JobState.QUEUED
        self._done = 0
        self._total = 0
        self._message = "waiting to start"
        self._result: Any = None
        self._error: str | None = None
        self._created = time.time()
        self._started: float | None = None
        self._finished: float | None = None
        self._lock = threading.Lock()

    @property
    def id(self) -> str:
        return self._id

    @property
    def state(self) -> JobState:
        with self._lock:
            return self._state

    # -- transitions, used only by JobManager -------------------------
    def _start(self) -> None:
        with self._lock:
            assert self._state is JobState.QUEUED
            self._state = JobState.RUNNING
            self._started = time.time()
            self._message = "started"

    def _progress(self, done: int, total: int, message: str) -> None:
        with self._lock:
            self._done, self._total, self._message = int(done), int(total), str(message)

    def _finish(self, result: Any) -> None:
        with self._lock:
            self._state = JobState.DONE
            self._result = result
            self._finished = time.time()
            self._message = "finished"
            if self._total:
                self._done = self._total

    def _fail(self, error: str) -> None:
        with self._lock:
            self._state = JobState.FAILED
            self._error = error
            self._finished = time.time()
            self._message = "failed"

    # -- what the outside world sees ----------------------------------
    def snapshot(self) -> dict[str, Any]:
        """A consistent copy of the job's state, safe to send as JSON."""
        with self._lock:
            elapsed = None
            if self._started is not None:
                elapsed = round((self._finished or time.time()) - self._started, 2)
            return {
                "id": self._id,
                "kind": self._kind,
                "description": self._description,
                "state": self._state.value,
                "done": self._done,
                "total": self._total,
                "fraction": (self._done / self._total) if self._total else 0.0,
                "message": self._message,
                "result": self._result,
                "error": self._error,
                "created": self._created,
                "elapsed_seconds": elapsed,
            }

    def __repr__(self) -> str:
        return f"Job({self._id}, {self._kind}, {self.state.value})"


class JobManager:
    """Runs jobs one at a time in a background thread and remembers them."""

    def __init__(self, keep: int = 100) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="satrec-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._keep = max(1, int(keep))

    def submit(self, kind: str, description: str, work: Work) -> Job:
        """Queue some work. Returns at once with the job's handle."""
        if not callable(work):
            raise ValidationError("work", type(work).__name__, "must be callable")
        job = Job(kind, description)
        with self._lock:
            self._jobs[job.id] = job
            self._trim()
        self._executor.submit(self._run, job, work)
        return job

    def _run(self, job: Job, work: Work) -> None:
        job._start()
        try:
            job._finish(work(job._progress))
        except SatRecoveryError as error:
            # An expected failure -- a bad request discovered late, an
            # untrained model -- reported in its own words.
            job._fail(str(error))
        except Exception:  # noqa: BLE001 - a job must never kill the worker
            job._fail(traceback.format_exc(limit=3))

    def _trim(self) -> None:
        """Forget the oldest finished jobs beyond ``keep``."""
        finished = [j for j in self._jobs.values() if j.state in (JobState.DONE, JobState.FAILED)]
        excess = len(self._jobs) - self._keep
        for job in sorted(finished, key=lambda j: j.snapshot()["created"])[: max(0, excess)]:
            del self._jobs[job.id]

    def get(self, job_id: str) -> Job:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError:
                raise KeyError(job_id) from None

    def list(self) -> list[dict[str, Any]]:
        """Snapshots of every remembered job, newest first."""
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted((j.snapshot() for j in jobs), key=lambda s: s["created"], reverse=True)

    def wait(self, job_id: str, timeout: float = 60.0) -> dict[str, Any]:
        """Block until a job finishes. For tests and scripts, not the API."""
        deadline = time.time() + timeout
        job = self.get(job_id)
        while job.state in (JobState.QUEUED, JobState.RUNNING):
            if time.time() > deadline:
                raise TimeoutError(f"job {job_id} still {job.state.value} after {timeout}s")
            time.sleep(0.05)
        return job.snapshot()

    def shutdown(self) -> None:
        """Stop accepting work and let the current job finish."""
        self._executor.shutdown(wait=True, cancel_futures=True)


__all__ = ["Job", "JobManager", "JobState"]
