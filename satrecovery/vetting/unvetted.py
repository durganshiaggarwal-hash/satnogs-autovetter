"""The real work: recent observations that no volunteer has checked yet.

The labelled observations (``manifest.jsonl``) are for learning and for
testing -- each already has a volunteer's answer on the SatNOGS website.
The ones here (``unvetted.jsonl``) have none: on a sample of 25 past
observations, 22 had no human verdict and 18 had no rating of any kind
(live network, November 2026).  For these, the vetter's answer is new
information, not a re-run of the website.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from .artifacts import ArtifactCache, ArtifactError
from .observation import Observation, WaterfallLabel
from .source import Manifest, ObservationQuery, ObservationSource, SatnogsError


def is_unvetted(obs: Observation) -> bool:
    """Finished, has a waterfall, and nobody has said whether it shows a signal."""
    return (obs.waterfall_label is WaterfallLabel.UNKNOWN and not obs.labelled_by_human
            and obs.waterfall_url is not None and obs.status != "future")


def fetch_unvetted(source: ObservationSource, cache: ArtifactCache, manifest: Manifest, count: int,
                   skip: set[int] | None = None, max_age_days: float = 14.0, look_at: int | None = None,
                   progress: Callable[[str], None] | None = None) -> int:
    """Download up to ``count`` unvetted observations, newest first; returns how many were added.

    ``skip`` holds ids to leave out (the labelled set).  Observations older
    than ``max_age_days`` are not looked at: recent ones are where an
    unchecked backlog actually helps someone.
    """
    skip = skip or set()
    now = datetime.now(timezone.utc)
    query = ObservationQuery(end=now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    oldest = now - timedelta(days=max_age_days)
    look_at = look_at or count * 20
    added = seen = 0
    try:
        for obs in source.fetch(query):
            seen += 1
            if added >= count or seen > look_at or (obs.start is not None and obs.start < oldest):
                break
            if obs.id in manifest or obs.id in skip or not is_unvetted(obs):
                continue
            try:
                cache.waterfall(obs)
            except ArtifactError as error:
                if progress:
                    progress(f"skip {obs.id}: {error}")
                continue
            manifest.add(obs)
            added += 1
            if progress and added % 10 == 0:
                progress(f"{added}/{count} unvetted downloaded ({seen} looked at)")
    except SatnogsError as error:
        if progress:
            progress(f"SatNOGS problem: {error}")
    return added


def is_finished_recording(obs: Observation) -> bool:
    """A pass that is over and whose waterfall has been uploaded (checked by a volunteer or not)."""
    return obs.waterfall_url is not None and obs.status != "future"


def fetch_latest(source: ObservationSource, cache: ArtifactCache, manifest: Manifest, count: int,
                 max_age_hours: float = 48.0, workers: int = 4, look_at: int | None = None,
                 progress: Callable[[str], None] | None = None,
                 added: Callable[[Observation], None] | None = None) -> int:
    """The newest ``count`` finished recordings, each downloaded as soon as it is listed.

    Listing goes to the API (one page a second); each recording's waterfall
    is downloaded from SatNOGS's file servers on a small pool of workers
    *while* the listing continues, so the first ones are ready in seconds.
    ``added`` is called for each recording once its waterfall is in the cache.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    now = datetime.now(timezone.utc)
    oldest = now - timedelta(hours=max_age_hours)
    query = ObservationQuery(end=now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    look_at = look_at or count * 6
    lock = threading.Lock()
    state = {"done": 0, "wanted": 0, "seen": 0}
    stop = threading.Event()

    def download(obs: Observation) -> None:
        if stop.is_set():
            return
        try:
            cache.waterfall(obs)
        except (ArtifactError, SatnogsError, OSError) as error:
            if progress:
                progress(f"skip {obs.id}: {error}")
            return
        with lock:
            manifest.add(obs)
            state["done"] += 1
            done = state["done"]
        if added:
            added(obs)
        if progress and done % 10 == 0:
            progress(f"{done} downloaded, {state['seen']} listed")

    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        for obs in source.fetch(query):
            state["seen"] += 1
            if state["wanted"] >= count or state["seen"] > look_at or (obs.start is not None and obs.start < oldest):
                break
            if not is_finished_recording(obs):
                continue
            state["wanted"] += 1
            if obs.id in manifest and cache.has_waterfall(obs):
                with lock:
                    state["done"] += 1
                if added:
                    added(obs)
                continue
            pool.submit(download, obs)
            if progress and state["seen"] % 25 == 0:
                progress(f"listing: {state['seen']} looked at, {state['wanted']} to fetch, {state['done']} ready")
    except SatnogsError as error:
        if progress:
            progress(f"SatNOGS problem while listing: {error}")
    except BaseException:
        stop.set()          # e.g. the window closed: let queued downloads end at once
        raise
    finally:
        pool.shutdown(wait=True)
    return state["done"]
