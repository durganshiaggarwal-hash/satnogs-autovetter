#!/usr/bin/env python3
"""Download human-vetted SatNOGS observations for training the auto-vetter.

    python scripts/fetch_satnogs.py --per-label 200
    python scripts/fetch_satnogs.py --per-label 5000 --max-per-satellite 150 --audio

What it does:

1. Asks the SatNOGS API for observations a volunteer marked "with signal",
   then "without signal", newest first.
2. Keeps only ones with a human label and a waterfall, and at most
   ``--max-per-satellite`` per satellite, so a few busy satellites
   do not fill the whole dataset.
3. Downloads each waterfall, turns it back into numbers (about 120 KB)
   and saves it in ``satrecovery_data/satnogs/waterfalls/``.
4. Writes every kept observation to ``satrecovery_data/satnogs/manifest.jsonl``.

Stop it any time (Ctrl+C) and run it again: what is already saved is skipped.
SatNOGS is run by volunteers, so this asks for at most one API page per second.
The data is CC BY-SA 4.0: credit "SatNOGS Network, Libre Space Foundation".
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satrecovery.vetting import (  # noqa: E402
    ArtifactCache,
    ArtifactError,
    Manifest,
    ObservationQuery,
    SatnogsApiSource,
    SatnogsError,
    WaterfallLabel,
)

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-label", type=int, default=200, help="observations to keep for each label")
    parser.add_argument("--max-per-satellite", type=int, default=100,
                        help="at most this many per satellite and label (spreads the dataset out)")
    parser.add_argument("--before", default=None, help="only observations ending before this ISO date")
    parser.add_argument("--audio", action="store_true", help="also download the audio (a few MB each)")
    parser.add_argument("--keep-png", action="store_true", help="keep the original waterfall pictures too")
    parser.add_argument("--out", default=str(ROOT / "satrecovery_data" / "satnogs"))
    args = parser.parse_args(argv)

    out = Path(args.out)
    manifest = Manifest(out / "manifest.jsonl")
    cache = ArtifactCache(out, keep_png=args.keep_png)
    source = SatnogsApiSource()
    print(f"Saving to {out}  (already have {len(manifest)}: {manifest.counts() or 'nothing'})")

    started = time.time()
    for label in (WaterfallLabel.SIGNAL, WaterfallLabel.NO_SIGNAL):
        have = Counter(o.norad_cat_id for o in manifest if o.waterfall_label is label)
        kept = sum(have.values())
        seen = skipped = failed = 0
        print(f"\n{label.value}: have {kept}, want {args.per_label}")
        try:
            for obs in source.fetch(ObservationQuery(waterfall_label=label, end=args.before)):
                if kept >= args.per_label:
                    break
                seen += 1
                if obs.id in manifest:
                    continue
                if not obs.is_trainable or have[obs.norad_cat_id] >= args.max_per_satellite:
                    skipped += 1
                    continue
                try:
                    cache.waterfall(obs)
                    if args.audio and obs.audio_url:
                        cache.audio(obs)
                except (ArtifactError, SatnogsError) as error:
                    failed += 1
                    print(f"  skip {obs.id}: {error}")
                    continue
                manifest.add(obs)
                have[obs.norad_cat_id] += 1
                kept += 1
                if kept % 25 == 0:
                    print(f"  {kept}/{args.per_label} kept  ({seen} looked at, {skipped} skipped, "
                          f"{failed} failed, {len(have)} satellites, {time.time() - started:.0f} s)")
        except KeyboardInterrupt:
            print("\nStopped. Run again to carry on where it left off.")
            break
        except SatnogsError as error:
            print(f"  SatNOGS problem: {error}. Run again later to carry on.")
            break
        print(f"  done: {kept} kept from {len(have)} satellites")

    print(f"\nManifest: {manifest.path}  ({len(manifest)} observations: {manifest.counts()})")
    print(f"API pages requested: {source.requests_made}.  Took {time.time() - started:.0f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
