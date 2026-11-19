#!/usr/bin/env python3
"""Run the orbit check on every volunteer-checked recording whose axes are readable; save what it found.

    python scripts/physics_features.py

Writes ``satrecovery_data/satnogs/physics.jsonl``: per observation, how much of
the bright track each kind of source explains.  Restartable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satrecovery.vetting import ArtifactCache, Manifest  # noqa: E402
from satrecovery.vetting.dataset import normalise  # noqa: E402
from satrecovery.vetting.features import CLIP  # noqa: E402
from satrecovery.vetting.orbit import OrbitError, SatnogsCatalog, StationDirectory, TrackExplainer  # noqa: E402
from satrecovery.vetting.physics import physics_row  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    data = ROOT / "satrecovery_data" / "satnogs"
    cache = ArtifactCache(data)
    explainer = TrackExplainer(cache, SatnogsCatalog(data), StationDirectory(data))
    out = data / "physics.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["obs_id"] for line in out.read_text().splitlines() if line.strip()}
    todo = []
    for obs in Manifest(data / "manifest.jsonl"):
        path = cache.axes_path(obs)
        if obs.is_trainable and obs.id not in done and path.exists() and json.loads(path.read_text()).get("ok"):
            todo.append(obs)
    print(f"{len(todo)} to check ({len(done)} already done)", flush=True)
    started = time.time()
    with out.open("a", encoding="utf-8") as handle:
        for n, obs in enumerate(todo, 1):
            try:
                power = np.clip(normalise(cache.waterfall(obs).power), *CLIP)
                row = physics_row(explainer.explain(obs, power))
            except (OrbitError, OSError, ValueError) as error:
                row = {"error": str(error)[:120]}
            row["obs_id"] = obs.id
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}  ({time.time() - started:.0f} s)", flush=True)
    print(f"done in {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
