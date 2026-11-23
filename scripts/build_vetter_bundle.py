#!/usr/bin/env python3
"""Train the vetter on every downloaded, volunteer-checked observation and save it for the app.

    python scripts/build_vetter_bundle.py            # forest + CNN (a few minutes)
    python scripts/build_vetter_bundle.py --no-cnn   # forest alone (seconds)

Saves ``satrecovery_data/models/vetter_bundle.*``.  After that the app runs
from the bundle plus the latest recordings it fetches itself; the training
waterfalls in ``satrecovery_data/satnogs/waterfalls`` are no longer needed to
run it (keep them if you want to retrain).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satrecovery.vetting import ArtifactCache, Manifest  # noqa: E402
from satrecovery.vetting.bundle import VetterBundle  # noqa: E402
from satrecovery.vetting.session import Policy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    data = ROOT / "satrecovery_data" / "satnogs"
    started = time.time()
    manifest = Manifest(data / "manifest.jsonl")
    print(f"Training on up to {len(manifest)} volunteer-checked observations ...")
    bundle = VetterBundle.train(
        manifest, ArtifactCache(data), cnn="--no-cnn" not in sys.argv,
        progress=lambda done, total: print(f"  measured {done}/{total}", flush=True),
        cnn_progress=lambda k, n: print(f"  CNN, stations held out: fold {k}/{n}", flush=True))
    folder = bundle.save(ROOT / "satrecovery_data" / "models")
    policy = Policy()
    error, sure = bundle.measured_error(policy)
    print(f"Trained on {bundle.n_train}.  At thresholds {policy.low}/{policy.high}: decides {bundle.automatable(policy):.0%} "
          f"alone, {error:.1%} of those wrong (each scored with its station held out, {sure} confident answers).")
    print(f"Model: {bundle.description}")
    print(f"Showcase examples kept: {[o.id for o in bundle.showcase]}")
    print(f"Saved to {folder} in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
