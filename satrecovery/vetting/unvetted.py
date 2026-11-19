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
    """The newest ``count`` finished recordings: listed one API page a second, then downloaded in parallel.

    The listing goes to the API (rate-limited); the waterfall files come from
    SatNOGS's file servers, so a few at once is fine.  ``added`` is called
    for each recording as soon as its waterfall is in the cache.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    now = datetime.now(timezone.utc)
    oldest = now - timedelta(hours=max_age_hours)
    query = ObservationQuery(end=now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    look_at = look_at or count * 6
    wanted: list[Observation] = []
    seen = 0
    try:
        for obs in source.fetch(query):
            seen += 1
            if len(wanted) >= count or seen > look_at or (obs.start is not None and obs.start < oldest):
                break
            if is_finished_recording(obs):
                wanted.append(obs)
    except SatnogsError as error:
        if progress:
            progress(f"SatNOGS problem while listing: {error}")
    if progress:
        progress(f"found {len(wanted)} recent recordings ({seen} looked at); downloading")

    def download(obs: Observation) -> Observation:
        cache.waterfall(obs)
        return obs

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        jobs = {pool.submit(download, obs): obs for obs in wanted}
        for job in as_completed(jobs):
            obs = jobs[job]
            try:
                job.result()
            except (ArtifactError, SatnogsError, OSError) as error:
                if progress:
                    progress(f"skip {obs.id}: {error}")
                continue
            manifest.add(obs)
            done += 1
            if added:
                added(obs)
            if progress and done % 10 == 0:
                progress(f"{done}/{len(wanted)} downloaded")
    return done
