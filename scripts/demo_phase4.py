#!/usr/bin/env python3
"""Phase 4 demonstration: what the learned receivers can and cannot do.

Run from the project root::

    python3 scripts/demo_phase4.py

Produces two figures in ``outputs/``:

``learned_vs_classical.png``
    The headline comparison, on a channel with echoes a symbol apart.
    Three methods, including the control that shows *why* the learned
    demapper wins when it wins.

``generalisation.png``
    The two places learning falls over: a classifier trained on clean
    signals collapsing on impaired ones, and an end-to-end network that
    cannot learn synchronisation from the data it is given.

Both figures take a few minutes, because they train the models from
scratch rather than loading anything.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satrecovery import (  # noqa: E402
    AdditiveWhiteGaussianNoise,
    ChannelConditions,
    ClassicalReceiver,
    Frame,
    HybridReceiver,
    ModulationRegistry,
    MultipathRicianFading,
    Signal,
    SignalSimulator,
    SoftDemapper,
)
from satrecovery.channel.pipeline import ChannelPipeline  # noqa: E402
from satrecovery.learning import (  # noqa: E402
    ClassificationDataset,
    CNNDemodulator,
    DemappingDataset,
    EndToEndDataset,
    ForestModulationClassifier,
    MLPDemapper,
    generate_frames,
    split_rows,
    training_conditions,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

C_SURFACE = "#fcfcfb"
C_TEXT = "#0b0b0b"
C_MUTED = "#52514e"
C_GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def style(axis) -> None:
    axis.set_facecolor(C_SURFACE)
    axis.tick_params(colors=C_MUTED, labelsize=9)
    for edge in axis.spines.values():
        edge.set_color(C_GRID)


def isi_frame(simulator, scheme, snr_db: float, seed: int, k_db: float = 12.0) -> Frame:
    """A frame whose echoes land a whole symbol apart.

    Genuine inter-symbol interference: a delayed copy of one symbol
    sitting on top of the next. This is the damage a nearest-neighbour
    demapper structurally cannot undo, because it only ever looks at one
    symbol at a time.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(
        0, 2, size=simulator.n_symbols * scheme.bits_per_symbol, dtype=np.uint8
    )
    symbols = scheme.map_bits(bits)

    channel = ChannelPipeline(
        [
            MultipathRicianFading(k_db, tap_spacing=8),
            AdditiveWhiteGaussianNoise(snr_db),
        ]
    )
    received = channel.apply(
        Signal(simulator.pulse.shape(symbols), simulator.sample_rate), rng
    )
    return Frame(
        received,
        bits,
        scheme.name,
        ChannelConditions(snr_db=snr_db),
        tx_symbols=symbols,
        samples_per_symbol=8,
    )


def figure_learned_vs_classical() -> Path:
    """Train two demappers and race them against the classical rule."""
    simulator = SignalSimulator(frame_length=8192, samples_per_symbol=8, seed=11)
    scheme = ModulationRegistry.create("QPSK")
    front_end = ClassicalReceiver(pulse=simulator.pulse, pilot_symbols=64)

    print("  building a training set of interfered frames...")
    frames = [
        isi_frame(simulator, scheme, float(snr), 1000 + i * 37 + j)
        for i, snr in enumerate(np.linspace(-2, 16, 10))
        for j in range(10)
    ]
    inputs, targets = DemappingDataset(front_end, window=3).build(frames)
    train_x, train_y, val_x, val_y = split_rows(inputs, targets, 0.2, seed=1)
    print(f"  {len(frames)} frames -> {inputs.shape[0]:,} training rows")

    demappers = {}
    for window in (0, 3):
        print(f"  training the {2 * window + 1}-symbol demapper...")
        model = MLPDemapper(scheme, window=window, hidden=(96, 96), seed=0)
        if window == 0:
            centre = (inputs.shape[1] // 2 - 1) // 2
            columns = slice(centre * 2, centre * 2 + 2)
            model.fit(
                train_x[:, columns], train_y,
                validation=(val_x[:, columns], val_y),
                epochs=60, batch_size=2048, verbose=False,
            )
        else:
            model.fit(
                train_x, train_y, validation=(val_x, val_y),
                epochs=60, batch_size=2048, verbose=False,
            )
        demappers[window] = model

    methods = {
        "Classical": ClassicalReceiver(
            pulse=simulator.pulse, demapper=SoftDemapper(), pilot_symbols=64
        ),
        "Learned, 1 symbol": HybridReceiver(
            demappers[0], pulse=simulator.pulse, pilot_symbols=64
        ),
        "Learned, 7 symbols": HybridReceiver(
            demappers[3], pulse=simulator.pulse, pilot_symbols=64
        ),
    }

    snr_values = [0.0, 4.0, 8.0, 12.0, 16.0]
    results: dict[str, list[float]] = {name: [] for name in methods}

    print("  measuring on frames none of them have seen...")
    print(f"    {'SNR':>5} " + " ".join(f"{n:>19}" for n in methods))
    for snr_db in snr_values:
        line = f"    {snr_db:5.1f} "
        for name, receiver in methods.items():
            errors = total = 0
            skip = 64 * scheme.bits_per_symbol
            for seed in range(8):
                frame = isi_frame(simulator, scheme, snr_db, 50000 + seed)
                mask = receiver.recover(frame).error_mask(frame.bits)[skip:]
                errors += int(mask.sum())
                total += mask.size
            rate = errors / total
            results[name].append(max(rate, 1e-6))
            line += f"{rate:19.3e} "
        print(line)

    figure, axis = plt.subplots(figsize=(10.2, 6.6), facecolor=C_SURFACE)
    style(axis)

    for colour, (name, rates) in zip(SERIES, results.items()):
        dashed = name == "Learned, 1 symbol"
        axis.semilogy(
            snr_values, rates,
            marker="o", markersize=6, linewidth=2.2, color=colour,
            linestyle="--" if dashed else "-", label=name, zorder=3,
        )
        # Classical and the one-symbol control land on top of each
        # other -- which is the result -- so their labels need pushing
        # apart or they overprint.
        nudge = {"Classical": 9, "Learned, 1 symbol": -9}.get(name, 0)
        axis.annotate(
            name, xy=(snr_values[-1], rates[-1]),
            xytext=(8, nudge), textcoords="offset points",
            fontsize=9.5, color=colour, va="center",
        )

    gain = results["Classical"][-1] / results["Learned, 7 symbols"][-1]
    axis.set_xlabel("Signal to noise ratio (dB)", fontsize=10, color=C_MUTED)
    axis.set_ylabel("Bit error rate", fontsize=10, color=C_MUTED)
    axis.set_xlim(-1.5, 20.5)
    axis.grid(True, which="both", color=C_GRID, linewidth=0.6)
    axis.legend(
        loc="lower left", fontsize=9, frameon=True,
        facecolor=C_SURFACE, edgecolor=C_GRID, labelcolor=C_MUTED,
    )

    figure.suptitle(
        "Where a learned demapper earns its place",
        fontsize=13.5, color=C_TEXT, x=0.012, ha="left", y=0.982,
    )
    figure.text(
        0.012, 0.945,
        s=(
            "QPSK on a channel whose echoes arrive a whole symbol apart.\n"
            "The dashed line is the control: the same network shown only ONE "
            "symbol, the same information the classical rule gets.\nIt ties. "
            f"Shown seven symbols instead, it wins by up to {gain:.1f}x — so the "
            "gain comes from seeing the neighbours,\nnot from the network being "
            "clever."
        ),
        va="top", fontsize=9, color=C_MUTED,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.80))
    path = OUTPUT_DIR / "learned_vs_classical.png"
    figure.savefig(path, dpi=150, facecolor=C_SURFACE)
    plt.close(figure)
    return path


def figure_generalisation() -> Path:
    """The two ways the learned methods fall over."""
    simulator = SignalSimulator(frame_length=4096, samples_per_symbol=8, seed=21)
    tester = SignalSimulator(frame_length=4096, samples_per_symbol=8, seed=777)
    schemes = ModulationRegistry.available()
    builder = ClassificationDataset(matched_filter=simulator.pulse)

    print("  training classifiers on clean and on impaired signals...")
    accuracy: dict[str, list[float]] = {}
    leaned_on: dict[str, str] = {}

    for label, impaired in (("clean only", False), ("impaired", True)):
        frames = generate_frames(
            simulator, schemes,
            training_conditions((-6, 20), 14, with_impairments=impaired),
            frames_each=10,
        )
        features, labels = builder.build(frames)
        forest = ForestModulationClassifier(n_trees=200, seed=0)
        forest.fit(features, labels)
        leaned_on[label] = next(iter(forest.feature_importances())).split(".")[-1]

        scores = []
        for test_impaired in (False, True):
            test_frames = generate_frames(
                tester, schemes,
                training_conditions(
                    (-6, 20), 8, with_impairments=test_impaired,
                    rng=np.random.default_rng(99),
                ),
                frames_each=5,
            )
            correct = sum(
                forest.classify_frame(f, matched_filter=tester.pulse)[0] == f.modulation
                for f in test_frames
            )
            scores.append(correct / len(test_frames))
        accuracy[label] = scores
        print(f"    trained on {label:11s}: clean {scores[0]:.3f}  impaired {scores[1]:.3f}"
              f"   (top feature: {leaned_on[label]})")

    print("  training end-to-end networks...")
    scheme = ModulationRegistry.create("QPSK")
    small = SignalSimulator(frame_length=1024, samples_per_symbol=8, seed=5)
    cnn_scores = []
    for label, impaired in (("aligned signal", False), ("+ Doppler, phase, timing", True)):
        if impaired:
            rng = np.random.default_rng(4)
            conditions = [
                ChannelConditions(
                    snr_db=float(s),
                    freq_offset_hz=float(rng.uniform(-400, 400)),
                    phase_offset_rad=float(rng.uniform(-np.pi, np.pi)),
                    timing_offset=float(rng.uniform(-0.5, 0.5)),
                )
                for s in np.linspace(0, 20, 8)
            ]
        else:
            conditions = [
                ChannelConditions(snr_db=float(s)) for s in np.linspace(0, 20, 8)
            ]

        frames = generate_frames(small, ["QPSK"], conditions, frames_each=60)
        features, labels = EndToEndDataset().build(frames)
        train_x, train_y, val_x, val_y = split_rows(features, labels, 0.2, seed=3)

        model = CNNDemodulator(scheme, samples_per_symbol=8, seed=0)
        history = model.fit(
            train_x, train_y, validation=(val_x, val_y),
            epochs=50, batch_size=32, verbose=False,
        )
        rate = history.last()["val_ber"]
        cnn_scores.append(max(rate, 1e-5))
        print(f"    {label:26s}: bit error rate {rate:.4f}")

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(12.4, 5.8), facecolor=C_SURFACE,
        gridspec_kw={"width_ratios": [1.35, 1]},
    )

    # ---- left: the classifier's train/test mismatch --------------------
    style(left)
    positions = np.arange(2)
    width = 0.34
    for index, (label, scores) in enumerate(accuracy.items()):
        left.bar(
            positions + (index - 0.5) * (width + 0.02), scores, width,
            color=SERIES[index], label=f"trained on {label}", zorder=3,
        )
        for x, value in zip(positions + (index - 0.5) * (width + 0.02), scores):
            left.text(
                x, value + 0.02, f"{value:.0%}",
                ha="center", fontsize=9, color=C_MUTED,
            )

    left.axhline(1 / len(schemes), color=C_MUTED, linewidth=1.1, linestyle=":", zorder=2)
    left.text(
        1.48, 1 / len(schemes) + 0.015, "guessing",
        fontsize=8.5, color=C_MUTED, ha="right",
    )
    left.set_xticks(positions)
    left.set_xticklabels(["tested on clean", "tested on impaired"], fontsize=9.5)
    left.set_ylabel("Modulation recognised correctly", fontsize=10, color=C_MUTED)
    left.set_ylim(0, 1.0)
    left.grid(True, axis="y", color=C_GRID, linewidth=0.6, zorder=0)
    left.legend(
        loc="upper right", fontsize=8.5, frameon=True,
        facecolor=C_SURFACE, edgecolor=C_GRID, labelcolor=C_MUTED,
    )
    left.set_title(
        "A classifier that learned the wrong thing",
        fontsize=11.5, color=C_TEXT, loc="left", pad=10,
    )

    # ---- right: the end-to-end network ---------------------------------
    style(right)
    bars = right.bar(
        [0, 1], cnn_scores, 0.45,
        color=[SERIES[2], SERIES[1]], zorder=3,
    )
    for bar, value in zip(bars, cnn_scores):
        right.text(
            bar.get_x() + bar.get_width() / 2, value * 1.35,
            f"{value:.4f}", ha="center", fontsize=9.5, color=C_MUTED,
        )
    right.set_yscale("log")
    right.set_xticks([0, 1])
    right.set_xticklabels(["already aligned", "must find the\nfrequency and timing"], fontsize=9.5)
    right.set_ylabel("Bit error rate", fontsize=10, color=C_MUTED)
    right.set_ylim(1e-4, 1.0)
    right.grid(True, axis="y", which="both", color=C_GRID, linewidth=0.6, zorder=0)
    right.set_title(
        "What end-to-end learning could not do",
        fontsize=11.5, color=C_TEXT, loc="left", pad=10,
    )

    figure.suptitle(
        "Two things learning did not solve",
        fontsize=13.5, color=C_TEXT, x=0.012, ha="left", y=0.982,
    )
    figure.text(
        0.012, 0.935,
        s=(
            f"Left: trained on clean signals the forest leans on "
            f"{leaned_on['clean only']}, a rotation-invariant statistic; trained on "
            f"impaired ones it switches to {leaned_on['impaired']}, which a Doppler "
            "shift corrupts.\nIt learned a shortcut that does not survive a moving "
            "satellite — visible only because a forest reports what it used.    "
            f"Right: the same network, {cnn_scores[1] / cnn_scores[0]:.0f}x worse when "
            "it must also find the frequency and timing."
        ),
        va="top", fontsize=8.8, color=C_MUTED,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.845))
    path = OUTPUT_DIR / "generalisation.png"
    figure.savefig(path, dpi=150, facecolor=C_SURFACE)
    plt.close(figure)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Learned demapper versus the classical rule")
    print("-" * 74)
    started = time.time()
    path = figure_learned_vs_classical()
    print(f"  wrote {path.relative_to(OUTPUT_DIR.parent)}  ({time.time() - started:.0f}s)\n")

    print("Where learning falls over")
    print("-" * 74)
    started = time.time()
    path = figure_generalisation()
    print(f"  wrote {path.relative_to(OUTPUT_DIR.parent)}  ({time.time() - started:.0f}s)\n")


if __name__ == "__main__":
    main()
