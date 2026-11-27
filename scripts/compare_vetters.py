#!/usr/bin/env python3
"""Which kind of model checks SatNOGS recordings best?  A fair comparison.

    python scripts/compare_vetters.py                      # every model
    python scripts/compare_vetters.py --models forest boosting --folds 5
    python scripts/compare_vetters.py --limit 400 --epochs 4   # a quick trial

Every model sees exactly the same volunteer-checked recordings and is
tested on exactly the same held-out **ground stations** (5 folds, whole
stations held out), so the only thing that changes is the model:

* ``logistic``  -- logistic regression on the 12 waterfall measurements (a simple baseline)
* ``forest``    -- random forest on the 12 measurements (the app's model)
* ``boosting``  -- gradient-boosted trees on the 12 measurements
* ``svm``       -- support-vector machine (RBF kernel) on the 12 measurements
* ``cnn``       -- deep learning: a 2-D ResNet on the waterfall picture itself
* ``cnn+forest``-- the CNN's and the forest's probabilities averaged

Reported for each: accuracy, balanced accuracy, AUROC, calibration error
(ECE), the share it can decide alone at 5% and 10% error (the original
project's RiskCoverageCurve), and training time.  Writes
``outputs/vetting/model_comparison.json`` and ``.png``.

The JSON also keeps every recording's held-out score from every model,
with the volunteer's answer and the pass details (elevation, transmitter
type, decoded packets, station position where known), so the app can
draw its report card -- confusion matrix, calibration, heatmaps, the
mistakes -- for any model in any run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import warnings

import matplotlib

matplotlib.use("Agg")
# scikit-learn 1.9 deprecates SVC(probability=True); it still works and is what we want here.
warnings.filterwarnings("ignore", message=".*probability.*parameter was deprecated", category=FutureWarning)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satrecovery.confidence.metrics import CalibrationReport, RiskCoverageCurve  # noqa: E402
from satrecovery.vetting import ArtifactCache, Manifest, WaterfallLabel  # noqa: E402
from satrecovery.vetting.dataset import normalise  # noqa: E402
from satrecovery.vetting.session import FeatureStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FEATURE_MODELS = ("logistic", "forest", "boosting", "svm")
ALL = FEATURE_MODELS + ("cnn", "cnn+forest")
LABELS = {"logistic": "Logistic regression", "forest": "Random forest (app)", "boosting": "Gradient boosting",
          "svm": "SVM (RBF)", "cnn": "CNN on the waterfall (deep learning)", "cnn+forest": "CNN + forest, averaged"}


def feature_model(name: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    if name == "forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                      random_state=seed, n_jobs=-1)
    if name == "boosting":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                              l2_regularization=1.0, class_weight="balanced", random_state=seed)
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale", probability=True,
                                                   class_weight="balanced", random_state=seed))
    raise ValueError(name)


def cnn_scores(images: np.ndarray, y: np.ndarray, groups: np.ndarray, train: np.ndarray, test: np.ndarray,
               epochs: int, seed: int) -> np.ndarray:
    """Train the 2-D ResNet on the training stations (a slice of them kept for early stopping)."""
    from sklearn.model_selection import GroupShuffleSplit

    from satrecovery.vetting.model import WaterfallVetter

    inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    fit, val = next(inner.split(train, y[train], groups[train]))
    fit, val = train[fit], train[val]
    model = WaterfallVetter(seed=seed)
    model.fit(images[fit], y[fit], validation=(images[val], y[val]), epochs=epochs, verbose=False, patience=6)
    return model.predict_proba(images[test])


def summary(p: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    p = np.asarray(p, dtype=float)
    confidence, correct = np.maximum(p, 1 - p), (p >= 0.5) == (y == 1)
    curve = RiskCoverageCurve(confidence, correct)
    return {"accuracy": float(correct.mean()), "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
            "auroc": float(roc_auc_score(y, p)), "ece": float(CalibrationReport(confidence, correct, n_bins=10).ece),
            "decide_alone_5pc": float(curve.coverage_at_risk(0.05)),
            "decide_alone_10pc": float(curve.coverage_at_risk(0.10)), "_curve": curve}


def per_recording(kept, y: np.ndarray, scores: dict, data: Path) -> dict:
    """Every recording's held-out score from every model, with what the report card needs to slice it."""
    try:
        sites = json.loads((data / "catalog" / "stations.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        sites = {}

    def site(obs, key):
        if key == "lat" and obs.station_lat is not None:
            return round(float(obs.station_lat), 3)
        if key == "lng" and obs.station_lng is not None:
            return round(float(obs.station_lng), 3)
        entry = sites.get(str(obs.ground_station))
        return round(float(entry[key]), 3) if entry else None

    return {
        "id": [int(o.id) for o in kept],
        "label": [int(v) for v in y],
        "station": [int(o.ground_station) if o.ground_station is not None else -1 for o in kept],
        "elevation": [round(float(o.max_altitude_deg), 1) if o.max_altitude_deg is not None else None for o in kept],
        "mode": [o.transmitter_mode or "unknown" for o in kept],
        "decoded": [int(o.n_decoded_frames or 0) for o in kept],
        "norad": [int(o.norad_cat_id or 0) for o in kept],
        "lat": [site(o, "lat") for o in kept],
        "lng": [site(o, "lng") for o in kept],
        "scores": {name: [round(float(v), 4) for v in s] for name, s in scores.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=ALL, default=list(ALL))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30, help="most passes for the CNN (it stops early)")
    parser.add_argument("--limit", type=int, default=0, help="use only this many recordings (a quick trial)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "vetting"),
                        help="folder for the results (the app keeps its own runs apart from the report's)")
    parser.add_argument("--name", default="model_comparison", help="file name for the results, without extension")
    args = parser.parse_args(argv)
    from sklearn.model_selection import GroupKFold

    started = time.time()
    data = ROOT / "satrecovery_data" / "satnogs"
    cache = ArtifactCache(data)
    labelled = [o for o in Manifest(data / "manifest.jsonl") if o.is_trainable and cache.has_waterfall(o)]
    if args.limit:
        labelled = list(np.random.default_rng(args.seed).permutation(np.array(labelled, dtype=object)))[:args.limit]
    kept, features = FeatureStore(cache).get(labelled)
    y = np.array([1 if o.waterfall_label is WaterfallLabel.SIGNAL else 0 for o in kept])
    groups = np.array([o.ground_station if o.ground_station is not None else -1 for o in kept])
    print(f"{len(kept):,} recordings, {len(set(groups.tolist()))} stations, {y.mean():.0%} with signal; "
          f"{args.folds} folds with whole stations held out\n", flush=True)

    need_images = any(m.startswith("cnn") for m in args.models)
    images = None
    if need_images:
        print("Loading the waterfalls for the CNN ...", flush=True)
        stack = []
        for obs in kept:
            power = normalise(cache.waterfall(obs).power)
            # 256 x 256 -> 128 x 128 by 2 x 2 averaging: what the network uses anyway
            stack.append(power.reshape(128, 2, 128, 2).mean(axis=(1, 3))[None])
        images = np.stack(stack).astype(np.float32)

    folds = list(GroupKFold(args.folds).split(features, y, groups))
    scores = {name: np.full(len(y), np.nan) for name in args.models}
    seconds = {name: 0.0 for name in args.models}
    for k, (train, test) in enumerate(folds, 1):
        for name in args.models:
            if name == "cnn+forest":
                continue
            t0 = time.time()
            if name in FEATURE_MODELS:
                model = feature_model(name, args.seed).fit(features[train], y[train])
                scores[name][test] = model.predict_proba(features[test])[:, 1]
            else:
                scores[name][test] = cnn_scores(images, y, groups, train, test, args.epochs, args.seed)
            seconds[name] += time.time() - t0
        print(f"  fold {k}/{args.folds} done ({time.time() - started:.0f} s)", flush=True)
    if "cnn+forest" in args.models:
        cnn = scores["cnn"] if "cnn" in scores else None
        forest = scores["forest"] if "forest" in scores else None
        if cnn is None or forest is None:
            raise SystemExit("cnn+forest needs both cnn and forest in --models")
        scores["cnn+forest"] = (cnn + forest) / 2
        seconds["cnn+forest"] = seconds["cnn"] + seconds["forest"]

    results = {name: summary(scores[name], y) for name in args.models}
    print(f"\n{'model':38}{'accuracy':>9}{'balanced':>10}{'AUROC':>8}{'ECE':>7}{'alone@5%':>10}{'alone@10%':>11}"
          f"{'train s':>9}")
    for name in args.models:
        r = results[name]
        print(f"{LABELS[name]:38}{r['accuracy']:>9.3f}{r['balanced_accuracy']:>10.3f}{r['auroc']:>8.3f}"
              f"{r['ece']:>7.3f}{r['decide_alone_5pc']:>10.1%}{r['decide_alone_10pc']:>11.1%}{seconds[name]:>9.0f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clean = {name: {k: v for k, v in r.items() if not k.startswith("_")} | {"train_seconds": seconds[name]}
             for name, r in results.items()}
    # Each model's risk-coverage curve, thinned to 60 points, so the app can draw it.
    curves = {}
    for name, r in results.items():
        curve = r["_curve"]
        keep = np.unique(np.linspace(0, len(curve.coverage) - 1, 60).astype(int))
        curves[name] = {"coverage": [round(float(v), 4) for v in np.asarray(curve.coverage)[keep]],
                        "risk": [round(float(v), 4) for v in np.asarray(curve.risk)[keep]]}
    (out / f"{args.name}.json").write_text(json.dumps({
        "n": int(len(y)), "stations": int(len(set(groups.tolist()))), "folds": args.folds,
        "epochs": args.epochs, "limit": args.limit, "seed": args.seed,
        "finished": time.strftime("%Y-%m-%d %H:%M"), "results": clean, "curves": curves,
        "per_recording": per_recording(kept, y, scores, data)}), encoding="utf-8")

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    names = list(args.models)
    colours = ["#7b8594", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"][:len(names)]
    x = np.arange(len(names))
    left.bar(x - 0.2, [results[n]["auroc"] for n in names], 0.4, color=colours, label="AUROC")
    left.bar(x + 0.2, [results[n]["accuracy"] for n in names], 0.4, color=colours, alpha=0.45, label="accuracy")
    left.set_xticks(x, [LABELS[n].split(" (")[0].replace("CNN on the waterfall", "CNN") for n in names],
                    rotation=20, ha="right", fontsize=8)
    left.set(ylim=(0.5, 1.0), title="Overall: AUROC (solid) and accuracy (faint)")
    for n, c in zip(names, colours):
        curve = results[n]["_curve"]
        right.plot(curve.coverage, curve.risk, color=c, label=LABELS[n])
    right.set(xlabel="share of recordings decided automatically", ylabel="share of those that are wrong",
              title="How much each can decide alone", xlim=(0, 1), ylim=(0, 0.3))
    right.legend(fontsize=7)
    fig.suptitle(f"{len(y):,} volunteer-checked SatNOGS recordings, {args.folds} folds, whole stations held out",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"{args.name}.png", dpi=130)
    plt.close(fig)
    print(f"\nSaved {out / (args.name + '.json')} and {out / (args.name + '.png')}  ({time.time() - started:.0f} s)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
