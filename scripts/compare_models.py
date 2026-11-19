#!/usr/bin/env python3
"""Train several kinds of AI model on the same RML24 recordings and compare them.

    python scripts/compare_models.py "C:/Users/you/Downloads/RML24_IQdata"

The six models (see the "Compare models" tab for plain-word descriptions):

    resnet       ResNet + expert features   (the app's design)
    resnet_raw   ResNet, no expert features (shows what the features add)
    cnn          plain CNN                  (the first version)
    lstm         LSTM
    transformer  Transformer
    forest       Random Forest on the 12 expert measurements

A FAIR TEST
-----------
Every model gets exactly the same training recordings, the same
validation recordings (for deciding when to stop), the same number of
passes, and is tested on exactly the same held-out recordings -- the
last 40 of every group, which no model ever trains on.  The only thing
that changes is the model.

WHAT IT SAVES (in satrecovery_data/models/compare/)
---------------------------------------------------
* one trained model per kind (``resnet.pt``, ``lstm.pt``, ...), which the
  app can switch to from the "Compare models" tab;
* ``results.json``: accuracy at every SNR, overall, at +10 dB and above,
  the mix-ups, size, training time and speed of every model;
* figures in ``outputs/compare/``.

HOW LONG
--------
On an RTX-class GPU with the defaults, roughly 10-25 minutes per network
and a minute for the forest -- about 1-2 hours in all.  For a quick try:

    python scripts/compare_models.py <RML24> --per-cell 60 --epochs 8

or only some models:  ``--models cnn lstm forest``.  Results from earlier
runs are kept, so models can be trained one at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satrecovery.analysis.classifier import ModulationClassifier  # noqa: E402
from satrecovery.analysis.compare import (  # noqa: E402
    COMPARISON_MODELS,
    ModelComparison,
    build_model,
    time_per_recording,
)
from satrecovery.analysis.dataset import LabelledSet, accuracy_by_snr, confusion, load_set, split_rows  # noqa: E402
from satrecovery.data import open_archive  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COLOURS = ["#807182", "#eb6834", "#3b82a0", "#5a9e4b", "#c4a000", "#a04b8c"]


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", help="RML24_IQdata.h5, its folder, or a sample .npz")
    parser.add_argument("--models", nargs="+", default=list(COMPARISON_MODELS), choices=list(COMPARISON_MODELS),
                        help="which models to train (default: all six)")
    parser.add_argument("--per-cell", type=int, default=250, help="training recordings per scheme/SNR/rate")
    parser.add_argument("--test-per-cell", type=int, default=40, help="test recordings per scheme/SNR/rate")
    parser.add_argument("--epochs", type=int, default=40, help="most passes for every network")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--out", default=str(ROOT / "satrecovery_data" / "models" / "compare"),
                        help="folder for the models and results.json")
    parser.add_argument("--figures", default=str(ROOT / "outputs" / "compare"))
    parser.add_argument("--cpu", action="store_true", help="train on the CPU even if a GPU is present")
    parser.add_argument("--quiet", action="store_true", help="do not print every epoch")
    args = parser.parse_args(argv)

    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = None
    if args.cpu:
        import torch

        device = torch.device("cpu")
    classes = ModulationClassifier.DEFAULT_CLASSES

    # -- the same data for everyone ------------------------------------
    with open_archive(args.archive) as archive:
        print(f"Dataset: {archive.name}, {len(archive):,} recordings")
        train_rows, test_rows = split_rows(archive, classes, args.per_cell, args.test_per_cell)
        print(f"Reading {len(train_rows):,} training and {len(test_rows):,} test recordings...")
        progress = lambda done, total: print(f"  {done:,} / {total:,}")  # noqa: E731
        everything = load_set(archive, train_rows, progress)
        test = load_set(archive, test_rows, progress)
        dataset_name = archive.name

    rng = np.random.default_rng(0)                    # the same split as train_classifier.py
    pick = rng.permutation(len(everything))
    n_val = min(1500, max(len(everything) // 10, 1))
    val_index, train_index = np.sort(pick[:n_val]), np.sort(pick[n_val:])
    train = LabelledSet(everything.inputs[train_index], everything.targets[train_index],
                        everything.snr_db[train_index], everything.rows[train_index])
    val = LabelledSet(everything.inputs[val_index], everything.targets[val_index],
                      everything.snr_db[val_index], everything.rows[val_index])
    del everything

    # -- earlier results are kept; this run adds to or replaces them ----
    store = ModelComparison(out, ROOT / "satrecovery_data" / "models" / "classifier.pt")
    results = store.results() or {}
    found = dict(results.get("models", {}))
    setup = {"dataset": dataset_name, "train": len(train), "validation": len(val), "test": len(test),
             "per_cell": args.per_cell, "test_per_cell": args.test_per_cell, "epochs": args.epochs,
             "classes": list(classes)}
    if results.get("setup") and {k: results["setup"].get(k) for k in ("per_cell", "test_per_cell", "epochs")} != \
            {k: setup[k] for k in ("per_cell", "test_per_cell", "epochs")}:
        print("\nNote: the settings differ from the earlier run, so earlier results are cleared "
              "to keep the comparison fair.")
        found = {}

    for name in args.models:
        label = COMPARISON_MODELS[name]["label"]
        print(f"\n=== {label} ({name}) ===")
        model = build_model(name, classes=classes, device=device)
        began = time.time()
        model.fit(train.inputs, train.targets, validation=(val.inputs, val.targets), epochs=args.epochs,
                  batch_size=args.batch_size, patience=args.patience, verbose=not args.quiet)
        train_seconds = time.time() - began
        overall = model.evaluate(test.inputs, test.targets)
        by_snr = accuracy_by_snr(model, test)
        model.record_test_accuracy(overall)
        model.save(out / f"{name}.pt")
        high = [a for s, a in by_snr.items() if s >= 10]
        found[name] = {
            "label": label,
            "accuracy": overall,
            "accuracy_10db_plus": float(np.mean(high)) if high else None,
            "accuracy_by_snr": {f"{s:g}": a for s, a in sorted(by_snr.items())},
            "confusion_0db_plus": confusion(model, test, min_snr_db=0).tolist(),
            "n_parameters": int(model.n_parameters),
            "train_seconds": round(train_seconds, 1),
            "epochs_run": int(model.history.n_epochs) if model.history is not None else None,
            "ms_per_recording": round(time_per_recording(model, test.inputs), 2),
            "device": str(getattr(model, "device", "cpu")),
        }
        print(f"  {label}: {overall:.1%} overall"
              + (f", {found[name]['accuracy_10db_plus']:.1%} at +10 dB and above" if high else "")
              + f"; trained in {train_seconds:.0f} s")
        # Saved after every model, so a long run that stops keeps what it finished.
        _write(out, setup, found, by_snr)

    print_table(found)
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    draw_curves(found, figures / "accuracy_vs_snr.png")
    print(f"\nModels and results in {out}")
    print(f"Figure: {figures / 'accuracy_vs_snr.png'}")
    print(f"Done in {time.time() - started:.0f} s.  Open the app's \"Compare models\" tab to see them.")
    return found


def _write(out: Path, setup: dict, found: dict, by_snr: dict) -> None:
    snrs = sorted({float(s) for info in found.values() for s in info["accuracy_by_snr"]})
    data = {"setup": setup, "snr_labels_db": snrs, "models": found, "written_at": time.time()}
    (out / ModelComparison.RESULTS).write_text(json.dumps(data, indent=1), encoding="utf-8")


def print_table(found: dict) -> None:
    print("\n" + "-" * 86)
    print(f"{'model':32} {'overall':>8} {'+10 dB+':>8} {'size':>10} {'train':>8} {'speed':>12}")
    for name in sorted(found, key=lambda n: -(found[n]["accuracy_10db_plus"] or 0)):
        info = found[name]
        high = f"{info['accuracy_10db_plus']:.1%}" if info["accuracy_10db_plus"] is not None else "-"
        print(f"{info['label']:32} {info['accuracy']:>8.1%} {high:>8} {info['n_parameters']:>10,} "
              f"{info['train_seconds']:>7.0f}s {info['ms_per_recording']:>9.2f} ms")
    print("-" * 86)


def draw_curves(found: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for colour, name in zip(COLOURS, found):
        points = sorted((float(s), a) for s, a in found[name]["accuracy_by_snr"].items())
        ax.plot([p[0] for p in points], [100 * p[1] for p in points], "o-", ms=3, lw=1.8, color=colour,
                label=found[name]["label"])
    ax.axhline(100 / 6, color="#a3969f", ls="--", lw=1)
    ax.set_xlabel("SNR label of the recording (dB)")
    ax.set_ylabel("named correctly (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Different AI models on the same held-out RML24 recordings", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
