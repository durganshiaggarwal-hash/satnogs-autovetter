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
    parser.add_argument("--unvetted", type=int, default=0, metavar="N",
                        help="instead: download N recent observations nobody has checked yet")
    args = parser.parse_args(argv)

    out = Path(args.out)
    manifest = Manifest(out / "manifest.jsonl")
    cache = ArtifactCache(out, keep_png=args.keep_png)
    source = SatnogsApiSource()

    if args.unvetted:
        from satrecovery.vetting.unvetted import fetch_unvetted

        queue = Manifest(out / "unvetted.jsonl")
        print(f"Unvetted queue: {queue.path} (already {len(queue)}); fetching up to {args.unvetted} more")
        added = fetch_unvetted(source, cache, queue, args.unvetted, skip={o.id for o in manifest},
                               progress=lambda line: print("  " + line, flush=True))
        print(f"Added {added}; the queue now holds {len(queue)}.  API pages requested: {source.requests_made}.")
        return 0
    print(f"Saving to {out}  (already have {len(manifest)}: {manifest.counts() or 'nothing'})")

    # Both labels are fetched in turn, one observation each, so the data is
    # balanced at every moment and a model can be trained before the end.
    started = time.time()
    labels = (WaterfallLabel.SIGNAL, WaterfallLabel.NO_SIGNAL)
    streams = {label: source.fetch(ObservationQuery(waterfall_label=label, end=args.before)) for label in labels}
    have = {label: Counter(o.norad_cat_id for o in manifest if o.waterfall_label is label) for label in labels}
    kept = {label: sum(have[label].values()) for label in labels}
    seen = skipped = failed = 0
    last_report = sum(kept.values())

    def next_usable(label: WaterfallLabel):
        nonlocal seen, skipped
        for obs in streams[label]:
            seen += 1
            if obs.id in manifest:
                continue
            if not obs.is_trainable or have[label][obs.norad_cat_id] >= args.max_per_satellite:
                skipped += 1
                continue
            return obs
        return None

    try:
        active = [label for label in labels if kept[label] < args.per_label]
        while active:
            for label in list(active):
                obs = next_usable(label)
                if obs is None:
                    print(f"  {label.value}: SatNOGS has no more; stopping at {kept[label]}")
                    active.remove(label)
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
                have[label][obs.norad_cat_id] += 1
                kept[label] += 1
                if kept[label] >= args.per_label:
                    active.remove(label)
            total = sum(kept.values())
            if total - last_report >= 50:
                last_report = total
                print(f"  {kept[labels[0]]} with / {kept[labels[1]]} without signal  ({seen} looked at, "
                      f"{skipped} skipped, {failed} failed, {time.time() - started:.0f} s)", flush=True)
    except KeyboardInterrupt:
        print("\nStopped. Run again to carry on where it left off.")
    except SatnogsError as error:
        print(f"  SatNOGS problem: {error}. Run again later to carry on.")

    print(f"\nManifest: {manifest.path}  ({len(manifest)} observations: {manifest.counts()})")
    print(f"API pages requested: {source.requests_made}.  Took {time.time() - started:.0f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
