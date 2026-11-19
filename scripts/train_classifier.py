#!/usr/bin/env python3
"""Train the AI classifier (a small ResNet) on RML24.  Run once, on the GPU if there is one.

    python scripts/train_classifier.py "C:/Users/you/Downloads/RML24_IQdata"

What it does:

1. Takes recordings of the six schemes the receivers handle, at every
   SNR label and every symbol rate: the first 250 of each group for
   training (by default), the last 40 for testing.  The test recordings
   are never seen during training.
2. Trains the network, printing training and validation accuracy after
   every pass (training far above validation would mean memorising).
   Up to 40 passes; stops by itself if it stops improving.
3. Measures accuracy on the held-out recordings, at every SNR, and
   saves the model where the analysis app looks for it:
   ``satrecovery_data/models/classifier.pt``.
4. Draws the figures for the report into ``outputs/classifier/``:
   accuracy against SNR, the confusion matrix, and the training curve.

On an RTX-class GPU: about a minute to read the recordings and roughly
10-25 minutes to train.  It needs about 2 GB of memory for the
recordings; if it runs out, use ``--per-cell 150``.  On a CPU it works
too, only much slower; use ``--per-cell 40 --epochs 10`` for a quick try.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satrecovery.analysis.classifier import ModulationClassifier  # noqa: E402
from satrecovery.analysis.dataset import (  # noqa: E402
    LabelledSet,
    accuracy_by_snr,
    confusion,
    load_set,
    split_rows,
)
from satrecovery.data import open_archive  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", help="RML24_IQdata.h5, its folder, or a sample .npz")
    parser.add_argument("--per-cell", type=int, default=250,
                        help="training recordings per scheme/SNR/rate (lower it if memory runs out)")
    parser.add_argument("--test-per-cell", type=int, default=40, help="test recordings per scheme/SNR/rate")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default=str(ROOT / "satrecovery_data" / "models" / "classifier.pt"),
                        help="where to save the trained model")
    parser.add_argument("--cpu", action="store_true", help="train on the CPU even if a GPU is present")
    args = parser.parse_args()

    started = time.time()
    figures = ROOT / "outputs" / "classifier"
    figures.mkdir(parents=True, exist_ok=True)

    device = None
    if args.cpu:
        import torch

        device = torch.device("cpu")
    model = ModulationClassifier(device=device)

    with open_archive(args.archive) as archive:
        print(f"Dataset: {archive.name}, {len(archive):,} recordings")
        train_rows, test_rows = split_rows(archive, model.classes, args.per_cell, args.test_per_cell)
        print(f"Reading {len(train_rows):,} training and {len(test_rows):,} test recordings...")
        progress = lambda done, total: print(f"  {done:,} / {total:,}")  # noqa: E731
        everything = load_set(archive, train_rows, progress)
        test = load_set(archive, test_rows, progress)

    # A small validation set carved from the *training* recordings, for
    # deciding when to stop; the test set stays untouched until the end.
    rng = np.random.default_rng(0)
    pick = rng.permutation(len(everything))
    n_val = min(1500, max(len(everything) // 10, 1))
    val_index, train_index = np.sort(pick[:n_val]), np.sort(pick[n_val:])
    train = LabelledSet(everything.inputs[train_index], everything.targets[train_index],
                        everything.snr_db[train_index], everything.rows[train_index])
    val = LabelledSet(everything.inputs[val_index], everything.targets[val_index],
                      everything.snr_db[val_index], everything.rows[val_index])
    del everything

    print(f"\nTraining on {len(train):,} recordings ({model.device}); validating on {len(val):,}.")
    model.fit(train.inputs, train.targets, validation=(val.inputs, val.targets),
              epochs=args.epochs, batch_size=args.batch_size, patience=10)

    print("\nTesting on recordings the network never saw...")
    overall = model.evaluate(test.inputs, test.targets)
    by_snr = accuracy_by_snr(model, test)
    model.record_test_accuracy(overall)
    saved = model.save(args.out)

    high = [a for s, a in by_snr.items() if s >= 10]
    print(f"\n  Overall accuracy:          {overall:.1%}")
    if high:
        print(f"  At 10 dB and above:        {np.mean(high):.1%}")
    for snr in sorted(by_snr):
        bar = "#" * int(round(by_snr[snr] * 40))
        print(f"    {snr:+5.0f} dB  {by_snr[snr]:6.1%}  {bar}")

    draw_accuracy(by_snr, figures / "accuracy_vs_snr.png")
    draw_confusion(confusion(model, test, min_snr_db=0), model.classes, figures / "confusion.png")
    draw_training(model, figures / "training.png")
    print(f"\nModel saved to {saved}")
    print(f"Figures in {figures}")
    print(f"Done in {time.time() - started:.0f} s.  Now run:  python -m satrecovery.analysis --rml24 \"{args.archive}\"")


def draw_accuracy(by_snr: dict[float, float], path: Path) -> None:
    snrs = sorted(by_snr)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(snrs, [100 * by_snr[s] for s in snrs], "o-", color="#807182", lw=2)
    ax.axhline(100 / 6, color="#a3969f", ls="--", lw=1)
    ax.text(snrs[0], 100 / 6 + 2, "guessing (1 in 6)", fontsize=8, color="#6b5f6d")
    ax.set_xlabel("SNR label of the recording (dB)")
    ax.set_ylabel("named correctly (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.set_title("AI classifier on held-out RML24 recordings", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_confusion(matrix: np.ndarray, classes, path: Path) -> None:
    rows = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(rows, cmap="PuRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=40)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("the AI said")
    ax.set_ylabel("it really was")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if rows[i, j] >= 0.005:
                ax.text(j, i, f"{rows[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if rows[i, j] > 0.6 else "#3a2f3d")
    ax.set_title("Which schemes get mixed up (recordings at 0 dB and above)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_training(model: ModulationClassifier, path: Path) -> None:
    history = model.history
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(history.epochs, history.series("loss"), label="training", color="#807182")
    if "val_loss" in history.keys():
        ax.plot(history.epochs, history.series("val_loss"), label="validation", color="#eb6834")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
