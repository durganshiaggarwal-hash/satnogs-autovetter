#!/usr/bin/env python3
"""Train the baseline auto-vetters on the SatNOGS waterfalls downloaded so far.

    python scripts/train_vetter.py
    python scripts/train_vetter.py --epochs 40 --random-split    # also show the leaky score
    python scripts/train_vetter.py --only features               # skip the CNN (fast)

What it does:

1. Loads every human-labelled waterfall in ``satrecovery_data/satnogs``.
2. Splits by **ground station**: test stations are never seen in training
   (and the CNN's validation stations are separate from both).
3. Trains both baselines on the same split:
   * ``features`` -- FeatureVetter: a forest on twelve waterfall measurements;
   * ``cnn`` -- WaterfallVetter: a small 2-D ResNet on the picture.
   The forest needs no validation set, so it also learns from the
   validation stations; the test stations stay unseen by both.
4. Reports accuracy, balanced accuracy, precision and recall for each
   label and AUROC on the held-out stations; saves the models to
   ``satrecovery_data/models/vetter_<name>.pt`` and the numbers to
   ``outputs/vetting/baseline.json``.

``--random-split`` repeats it on a plain random split of the same size, to
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
    FeatureVetter,
    Manifest,
    StationSplit,
    Vetter,
    VettingSet,
    WaterfallVetter,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"features": "FeatureVetter (forest on 12 measurements)", "cnn": "WaterfallVetter (2-D ResNet)"}


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


def train(name: str, data: VettingSet, train_idx, val_idx, epochs: int, seed: int) -> Vetter:
    if name == "features":
        model: Vetter = FeatureVetter(seed=seed)
        fit_idx = np.concatenate([train_idx, val_idx])
        model.fit(data.inputs[fit_idx], data.targets[fit_idx])
    else:
        model = WaterfallVetter(seed=seed)
        model.fit(data.inputs[train_idx], data.targets[train_idx],
                  validation=(data.inputs[val_idx], data.targets[val_idx]), epochs=epochs, verbose=False)
    return model


def run_split(names, data, train_idx, val_idx, test_idx, epochs, seed, title) -> tuple[dict, dict]:
    print(f"\n== {title}: train {len(train_idx)}, validation {len(val_idx)}, test {len(test_idx)}")
    print(f"   {'model':<44}{'accuracy':>9}{'balanced':>10}{'AUROC':>8}{'sig recall':>12}{'no-sig recall':>15}")
    results, models = {}, {}
    for name in names:
        started = time.time()
        model = train(name, data, train_idx, val_idx, epochs, seed)
        result = scores(data.targets[test_idx], model.predict_proba(data.inputs[test_idx]))
        result["train_seconds"] = round(time.time() - started, 1)
        results[name], models[name] = result, model
        print(f"   {MODELS[name]:<44}{result['accuracy']:>9.3f}{result['balanced_accuracy']:>10.3f}"
              f"{result['auroc']:>8.3f}{result['signal_recall']:>12.3f}{result['no_signal_recall']:>15.3f}")
    return results, models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "satrecovery_data" / "satnogs"))
    parser.add_argument("--only", nargs="+", choices=list(MODELS), default=list(MODELS))
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
    train_idx, val_idx = rest[inner_train], rest[inner_val]
    if StationSplit.stations(obs, test) & StationSplit.stations(obs, np.concatenate([train_idx, val_idx])):
        raise SystemExit("a test station leaked into training -- this is a bug")

    held_out, models = run_split(args.only, data, train_idx, val_idx, test, args.epochs, args.seed,
                                 "held-out stations")
    report: dict = {"setup": {"n": len(data), "epochs": args.epochs, "seed": args.seed,
                              "test_stations": len(StationSplit.stations(obs, test)),
                              "test_signal_share": float(np.mean(data.targets[test] == SIGNAL))},
                    "held_out_stations": held_out}

    if args.random_split:
        order = np.random.default_rng(args.seed).permutation(len(data))
        n_test, n_val = len(test), len(val_idx)
        report["random_split"], _ = run_split(args.only, data, order[n_test + n_val:], order[n_test:n_test + n_val],
                                              order[:n_test], args.epochs, args.seed, "random split (leaky)")

    if "features" in models:
        top = list(models["features"].importances().items())[:5]
        report["feature_importances"] = models["features"].importances()
        print("\nWhat the forest relies on most: " + ", ".join(f"{k} ({v:.0%})" for k, v in top))

    out_models = ROOT / "satrecovery_data" / "models"
    for name, model in models.items():
        model.save(out_models / f"vetter_{name}.pt")
    out = ROOT / "outputs" / "vetting" / "baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nModels: {out_models}/vetter_*.pt\nNumbers: {out}\nTook {time.time() - started:.0f} s")
    print(f"(Signal share on test stations: {report['setup']['test_signal_share']:.0%}. "
          "Prior public work: 86.6% accuracy on a 187-image random split.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
