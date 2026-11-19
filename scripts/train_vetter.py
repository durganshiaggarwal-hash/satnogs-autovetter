#!/usr/bin/env python3
"""Train the baseline auto-vetter on the SatNOGS waterfalls downloaded so far.

    python scripts/train_vetter.py
    python scripts/train_vetter.py --epochs 40 --random-split    # also show the leaky score

What it does:

1. Loads every human-labelled waterfall in ``satrecovery_data/satnogs``.
2. Splits by **ground station**: test stations are never seen in training
   (and validation stations are separate from both).
3. Trains the 2-D ResNet, keeping the weights that did best on validation.
4. Reports accuracy, balanced accuracy, precision and recall for each
   label and AUROC on the held-out stations, and saves the model to
   ``satrecovery_data/models/vetter.pt`` with the numbers in
   ``outputs/vetting/baseline.json``.

``--random-split`` trains a second model on a plain random split, to
measure how much a careless split would flatter the score.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satrecovery.vetting import (  # noqa: E402
    SIGNAL,
    ArtifactCache,
    Manifest,
    StationSplit,
    VettingSet,
    WaterfallVetter,
)

ROOT = Path(__file__).resolve().parent.parent


def scores(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    decided = (probability >= 0.5).astype(int)
    out: dict[str, float] = {"n": int(len(target)), "accuracy": float(np.mean(decided == target))}
    recalls = []
    for label, name in ((1, "signal"), (0, "no_signal")):
        chosen = decided == label
        actual = target == label
        out[f"{name}_precision"] = float(np.mean(target[chosen] == label)) if chosen.any() else float("nan")
        out[f"{name}_recall"] = float(np.mean(decided[actual] == label)) if actual.any() else float("nan")
        recalls.append(out[f"{name}_recall"])
    out["balanced_accuracy"] = float(np.nanmean(recalls))
    out["auroc"] = float(roc_auc_score(target, probability)) if len(set(target.tolist())) == 2 else float("nan")
    return out


def run(data: VettingSet, train_idx, val_idx, test_idx, epochs: int, seed: int, label: str):
    print(f"\n== {label}: train {len(train_idx)}, validation {len(val_idx)}, test {len(test_idx)}")
    model = WaterfallVetter(seed=seed)
    print(f"   {model.describe()} on {model.device}")
    model.fit(data.inputs[train_idx], data.targets[train_idx],
              validation=(data.inputs[val_idx], data.targets[val_idx]), epochs=epochs)
    result = scores(data.targets[test_idx], model.predict_proba(data.inputs[test_idx]))
    print("   " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in result.items()))
    return model, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "satrecovery_data" / "satnogs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-split", action="store_true", help="also train on a leaky random split")
    args = parser.parse_args(argv)

    started = time.time()
    data_dir = Path(args.data)
    data = VettingSet.build(Manifest(data_dir / "manifest.jsonl"), ArtifactCache(data_dir))
    obs = data.observations
    print(f"{len(data)} labelled waterfalls, {data.share_with_signal:.0%} with signal, "
          f"{len({o.ground_station for o in obs})} stations, {len({o.norad_cat_id for o in obs})} satellites")

    rest, test = StationSplit(0.2, seed=args.seed).split(obs)
    inner_train, inner_val = StationSplit(0.15, seed=args.seed + 1).split([obs[i] for i in rest])
    train, val = rest[inner_train], rest[inner_val]
    assert not (StationSplit.stations(obs, test) & StationSplit.stations(obs, train))

    model, held_out = run(data, train, val, test, args.epochs, args.seed, "held-out stations")
    report = {"setup": {"n": len(data), "epochs": args.epochs, "seed": args.seed,
                        "test_stations": len(StationSplit.stations(obs, test))},
              "held_out_stations": held_out}

    if args.random_split:
        order = np.random.default_rng(args.seed).permutation(len(data))
        n_test, n_val = len(test), len(val)
        _, leaky = run(data, order[n_test + n_val:], order[n_test:n_test + n_val], order[:n_test],
                       args.epochs, args.seed, "random split (leaky, for comparison)")
        report["random_split"] = leaky

    model_path = model.save(ROOT / "satrecovery_data" / "models" / "vetter.pt")
    out = ROOT / "outputs" / "vetting" / "baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nModel: {model_path}\nNumbers: {out}\nTook {time.time() - started:.0f} s")
    print(f"(Signal share on test stations: {np.mean(data.targets[test] == SIGNAL):.0%}. "
          "Prior public work: 86.6% accuracy on a 187-image random split.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
