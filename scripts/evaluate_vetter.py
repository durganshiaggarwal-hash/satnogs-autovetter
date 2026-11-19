#!/usr/bin/env python3
"""The numbers and figures for the write-up: how good the vetter is, measured honestly.

    python scripts/evaluate_vetter.py

Uses the saved bundle (``scripts/build_vetter_bundle.py``): every score in it
comes from a model that never saw that observation's ground station.

1. Accuracy, AUROC, and calibration (ECE) before and after temperature
   scaling -- the calibration itself fitted on other stations' scores.
2. The headline: how much can be decided automatically at 1%, 2%, 5%, 10%
   error (the original project's RiskCoverageCurve).
3. The physics experiment (if physics.jsonl exists): does adding the orbit
   check to the picture model improve those numbers?  (It does not, and
   why is reported.)

Writes ``outputs/vetting/evaluation.json`` and ``outputs/vetting/evaluation.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satrecovery.confidence.metrics import CalibrationReport, RiskCoverageCurve  # noqa: E402
from satrecovery.vetting.bundle import VetterBundle, apply_temperature, fit_temperature  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RISKS = (0.01, 0.02, 0.05, 0.10)


def decisions(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.maximum(p, 1 - p), (p >= 0.5) == (y == 1)


def grouped_calibration(p: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5) -> np.ndarray:
    """Temperature-scale each fold with a temperature fitted on the other folds' stations."""
    from sklearn.model_selection import GroupKFold

    out = np.empty_like(p)
    for train, test in GroupKFold(folds).split(p, y, groups):
        out[test] = apply_temperature(p[test], fit_temperature(p[train], y[train]))
    return out


def station_interval(p: np.ndarray, y: np.ndarray, groups: np.ndarray, risk: float, draws: int = 300,
                     seed: int = 0) -> tuple[float, float]:
    """95% range of 'share decided alone at this error', resampling whole stations (they differ a lot)."""
    rng = np.random.default_rng(seed)
    stations = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in stations}
    values = []
    for _ in range(draws):
        pick = np.concatenate([members[g] for g in rng.choice(stations, size=len(stations), replace=True)])
        confidence, correct = decisions(p[pick], y[pick])
        values.append(RiskCoverageCurve(confidence, correct).coverage_at_risk(risk))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def summary(p: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score

    confidence, correct = decisions(p, y)
    report = CalibrationReport(confidence, correct, n_bins=10)
    curve = RiskCoverageCurve(confidence, correct)
    return {"n": int(len(y)), "accuracy": float(correct.mean()), "auroc": float(roc_auc_score(y, p)),
            "ece": float(report.ece), "overconfidence": float(report.overconfidence),
            "decide_alone": {f"{r:.0%}": float(curve.coverage_at_risk(r)) for r in RISKS},
            "_report": report, "_curve": curve}


def physics_experiment(bundle: VetterBundle) -> dict | None:
    """Picture model alone vs picture + orbit check, on the recordings the check could run on."""
    from sklearn.model_selection import GroupKFold

    from satrecovery.vetting import ArtifactCache, Manifest, WaterfallLabel
    from satrecovery.vetting.physics import CombinedVetter, physics_vector
    from satrecovery.vetting.session import FeatureStore, out_of_fold

    data = ROOT / "satrecovery_data" / "satnogs"
    path = data / "physics.jsonl"
    if not path.exists():
        return None
    rows = {json.loads(line)["obs_id"]: json.loads(line) for line in path.read_text().splitlines() if line.strip()}
    labelled = [o for o in Manifest(data / "manifest.jsonl") if o.is_trainable]
    kept, features = FeatureStore(ArtifactCache(data)).get(labelled)
    y = np.array([1 if o.waterfall_label is WaterfallLabel.SIGNAL else 0 for o in kept])
    groups = np.array([o.ground_station for o in kept])
    picture = out_of_fold(features, y, groups)
    index = np.array([i for i, o in enumerate(kept) if o.id in rows and "error" not in rows[o.id]])
    if len(index) < 100:
        return None
    physics = np.array([physics_vector(rows[kept[i].id]) for i in index])
    yy, gg, pp = y[index], groups[index], picture[index]
    combined = np.zeros(len(index))
    for train, test in GroupKFold(5).split(physics, yy, gg):
        combined[test] = CombinedVetter().fit(pp[train], physics[train], yy[train]).predict_proba(pp[test], physics[test])
    clear = {}
    for kind, threshold in (("scheduled", 0.7), ("ground", 0.85), ("other", 0.85)):
        chosen = [i for i in index if rows[kept[i].id].get(kind, 0) >= threshold]
        if chosen:
            clear[kind] = {"n": len(chosen), "volunteers_said_signal": float(np.mean(y[chosen]))}
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731
    return {"n": int(len(index)), "picture_only": strip(summary(pp, yy)),
            "picture_plus_physics": strip(summary(combined, yy)),
            "weights": {k: float(v) for k, v in CombinedVetter().fit(pp, physics, yy).weights().items()},
            "when_physics_is_clear": clear}


def main() -> int:
    bundle = VetterBundle.load(ROOT / "satrecovery_data" / "models")
    p, y = bundle.oof_probability, bundle.oof_target
    groups = bundle.oof_station if bundle.oof_station is not None else np.arange(len(y)) % 5
    raw = summary(p, y)
    calibrated = summary(grouped_calibration(p, y, groups), y)

    print(f"{bundle.n_train:,} volunteer-checked recordings, {len(set(groups.tolist()))} stations, "
          f"{y.mean():.0%} with signal.  Every score: station held out.\n")
    print(f"{'':22}{'accuracy':>9}{'AUROC':>8}{'ECE':>8}{'overconf.':>11}   decide alone at <= 1% / 2% / 5% / 10% error")
    for name, s in (("as trained", raw), ("calibrated", calibrated)):
        alone = " / ".join(f"{v:5.1%}" for v in s["decide_alone"].values())
        print(f"{name:22}{s['accuracy']:>9.3f}{s['auroc']:>8.3f}{s['ece']:>8.3f}{s['overconfidence']:>+11.3f}   {alone}")
    print(f"\nTemperature (fitted on all): {bundle.temperature:.3f}  (above 1 = it was overconfident)")
    intervals = {}
    for r in RISKS:
        intervals[f"{r:.0%}"] = station_interval(p, y, groups, r)
    print("95% ranges (stations resampled), as trained: " + ";  ".join(
        f"<= {k}: {lo:.0%}-{hi:.0%}" for k, (lo, hi) in intervals.items()))

    experiment = physics_experiment(bundle)
    if experiment:
        print(f"\nPhysics experiment on {experiment['n']} recordings with readable axes (stations held out):")
        for name in ("picture_only", "picture_plus_physics"):
            s = experiment[name]
            alone = " / ".join(f"{v:5.1%}" for v in s["decide_alone"].values())
            print(f"  {name:22}acc {s['accuracy']:.3f}  AUROC {s['auroc']:.3f}  decide alone {alone}")
        for kind, d in experiment["when_physics_is_clear"].items():
            print(f"  physics clearly says '{kind}': {d['n']} recordings, volunteers said 'with signal' "
                  f"in {d['volunteers_said_signal']:.0%}")

    out = ROOT / "outputs" / "vetting"
    out.mkdir(parents=True, exist_ok=True)
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731
    (out / "evaluation.json").write_text(json.dumps({
        "n": bundle.n_train, "temperature": bundle.temperature, "as_trained": strip(raw),
        "decide_alone_95pc_range": intervals,
        "calibrated": strip(calibrated), "physics_experiment": experiment}, indent=2), encoding="utf-8")

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, s, colour in (("as trained", raw, "#d95f02"), ("calibrated", calibrated, "#1b9e77")):
        report = s["_report"]
        keep = report.bin_counts > 0
        left.plot(report.bin_confidence[keep], report.bin_accuracy[keep], "o-", color=colour,
                  label=f"{name} (ECE {s['ece']:.3f})")
    left.plot([0.5, 1], [0.5, 1], "--", color="grey", lw=1, label="perfectly honest")
    left.set(xlabel="how sure the vetter says it is", ylabel="how often it is right",
             title="Does '90% sure' mean right 9 times in 10?", xlim=(0.5, 1), ylim=(0.4, 1))
    left.legend(fontsize=8)
    curve = calibrated["_curve"]
    right.plot(curve.coverage, curve.risk, color="#1b9e77")
    for r in RISKS:
        c = curve.coverage_at_risk(r)
        if c < 0.01:
            continue  # nothing can be decided at this error: no marker to draw
        right.plot([c], [r], "o", color="#1b9e77")
        right.annotate(f"{c:.0%} at {r:.0%}", (c, r), textcoords="offset points", xytext=(6, -10), fontsize=8)
    right.set(xlabel="share of recordings decided automatically", ylabel="share of those that are wrong",
              title="How much can it check alone?", xlim=(0, 1), ylim=(0, 0.3))
    fig.suptitle(f"SatNOGS vetter on {bundle.n_train:,} volunteer-checked recordings, each station held out",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "evaluation.png", dpi=130)
    print(f"\nSaved {out / 'evaluation.json'} and {out / 'evaluation.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
