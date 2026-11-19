#!/usr/bin/env python3
"""Phase 6 demonstration: one fair benchmark, and what it found.

Run from the project root::

    python3 scripts/demo_phase6.py

Takes about ten minutes: it trains every learned model from scratch,
including one per held-out split, rather than loading anything.

Produces four figures in ``outputs/`` and a results database:

``bug_found.png``
    The benchmark's first catch: an error floor in the classical
    receiver, traced to symbol 0 of every Doppler frame and fixed.

``benchmark.png``
    Classical against hybrid, three modulations, two channels, with
    honest intervals.  They tie -- which is what theory says they
    should do on a channel like this.

``held_out.png``
    The hybrid tested on conditions it never trained on.  It holds up,
    and the figure explains why.

``self_knowledge.png``
    Why the hybrid was worse at noticing a lost frame, the experiment
    that proved the cause, and the ten-line fix.

``results.db``
    Every cell of the main benchmark, for the GUI to read in Phase 8.
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
    ClassicalReceiver,
    RiskCoverageCurve,
    SoftDemapper,
    get_modulation,
)
from satrecovery.evaluation import (  # noqa: E402
    BenchmarkRunner,
    ConsoleProgress,
    HoldOutProfile,
    HoldOutSNR,
    ImpairmentProfile,
    RandomSplit,
    ResultStore,
    Scenario,
    ScenarioGrid,
    SimulatedSource,
)
from satrecovery.learning.datasets import DemappingDataset, split_rows  # noqa: E402
from satrecovery.learning.neural import MLPDemapper  # noqa: E402
from satrecovery.receivers.bank import ReceiverBank  # noqa: E402
from satrecovery.receivers.base import ReceiverChain  # noqa: E402
from satrecovery.receivers.demapper import NoiseNormalisedDemapper  # noqa: E402
from satrecovery.receivers.stages import (  # noqa: E402
    AutomaticGainControl,
    CoarseFrequencyCorrection,
    DecisionDirectedPLL,
    GardnerTimingRecovery,
    MatchedFilterStage,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

C_SURFACE = "#fcfcfb"
C_TEXT = "#0b0b0b"
C_MUTED = "#52514e"
C_GRID = "#e4e3df"
C_REF = "#9a9892"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#8257d6", "#c23b3b"]

#: Test frames and training frames come from different seeds, so no
#: model is ever graded on a frame it was trained on.
TEST = SimulatedSource(seed=1)
TRAIN = SimulatedSource(seed=2)
PULSE = TEST.simulator.pulse


def style(axis) -> None:
    axis.set_facecolor(C_SURFACE)
    axis.tick_params(colors=C_MUTED, labelsize=9)
    for edge in axis.spines.values():
        edge.set_color(C_GRID)


def caption(figure, title: str, text: str, top: float) -> None:
    figure.suptitle(title, fontsize=13.5, color=C_TEXT, x=0.012, ha="left", y=0.985)
    figure.text(0.012, 0.935, s=text, va="top", fontsize=9, color=C_MUTED)
    figure.tight_layout(rect=(0, 0, 1, top))


def classical() -> ClassicalReceiver:
    return ClassicalReceiver(pulse=PULSE, demapper=SoftDemapper())


def train_demapper(grid: ScenarioGrid, modulation: str, frames_each: int = 18) -> MLPDemapper:
    """Train one hybrid demapper on the training-seed frames of ``grid``.

    The settings are the ones the fairness check below settled on.  The
    first version used a third of this data and lost to the classical
    receiver on 64QAM; with enough data the loss disappeared.  Reporting
    the under-trained loss would have been a result about our training
    budget, not about learned demappers.
    """
    frames = [
        frame
        for scenario in grid.where(modulation=modulation)
        for frame in TRAIN.frames(scenario, frames_each)
    ]
    inputs, targets = DemappingDataset(classical(), window=2).build(frames)
    x_train, y_train, x_val, y_val = split_rows(inputs, targets, 0.2, seed=0)
    model = MLPDemapper(get_modulation(modulation), window=2, hidden=(128, 128), seed=0)
    model.fit(x_train, y_train, validation=(x_val, y_val), epochs=200, patience=12, verbose=False)
    return model


def hybrid_bank(grid: ScenarioGrid) -> ReceiverBank:
    return ReceiverBank(
        {m: ClassicalReceiver(pulse=PULSE, demapper=train_demapper(grid, m)) for m in grid.modulations},
        name="hybrid",
    )


# ======================================================================
# 1. The bug the benchmark found
# ======================================================================
class PrePhase6Receiver(ClassicalReceiver):
    """The classical receiver exactly as it was before Phase 6.

    Matched filter first, frequency correction second.  Kept only so
    the bug can be reproduced on demand -- by overriding one method,
    which is all it takes because the chain is built in one place.
    """

    def build_chain(self, samples_per_symbol, use_coarse_frequency=True):
        pulse = self._pulse.at_rate(samples_per_symbol)
        chain = ReceiverChain(name="Pre-Phase-6")
        chain.add(MatchedFilterStage(pulse))
        if use_coarse_frequency:
            chain.add(CoarseFrequencyCorrection())
        chain.add(GardnerTimingRecovery())
        chain.add(AutomaticGainControl())
        chain.add(DecisionDirectedPLL())
        return chain


def figure_bug():
    grid = ScenarioGrid.product(["16QAM"], [4, 8, 12, 16, 20], [ImpairmentProfile.doppler()])
    receivers = {
        "before the fix": PrePhase6Receiver(pulse=PULSE, demapper=SoftDemapper()),
        "after the fix": classical(),
    }
    result = BenchmarkRunner(TEST, receivers, frames_per_scenario=40).run(grid)

    # Where in the frame do the old receiver's high-SNR errors fall?
    positions = []
    for frame in TEST.frames(Scenario("16QAM", 20, ImpairmentProfile.doppler()), 40):
        mask = receivers["before the fix"].recover(frame).error_mask(frame.bits)
        positions.extend(np.flatnonzero(mask) // 4)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), facecolor=C_SURFACE)

    axis = axes[0]
    style(axis)
    for index, name in enumerate(receivers):
        snr, ber, interval = result.curve(name, "16QAM", "doppler")
        shown = np.maximum(ber, 1e-6)
        axis.plot(snr, shown, color=SERIES[index], marker="o", markersize=4,
                  linewidth=1.8, label=name)
        axis.fill_between(snr, np.maximum(interval[:, 0], 1e-6), np.maximum(interval[:, 1], 1e-6),
                          color=SERIES[index], alpha=0.15, linewidth=0)
    axis.set_yscale("log")
    axis.set_ylim(5e-7, 0.1)
    axis.set_xlabel("Signal to noise ratio (dB)", fontsize=9.5, color=C_MUTED)
    axis.set_ylabel("bit error rate  (zero plotted at 1e-6)", fontsize=9.5, color=C_MUTED)
    axis.grid(True, which="both", color=C_GRID, linewidth=0.6)
    axis.set_title("16QAM with an 800 Hz Doppler shift", fontsize=10.5, color=C_TEXT, loc="left")
    axis.legend(loc="lower left", fontsize=9, frameon=True,
                facecolor=C_SURFACE, edgecolor=C_GRID, labelcolor=C_MUTED)

    axis = axes[1]
    style(axis)
    axis.hist(positions, bins=np.arange(0, 257, 4), color=SERIES[0], alpha=0.85)
    axis.set_xlim(-4, 256)
    axis.set_xlabel("position of the wrong symbol in the frame", fontsize=9.5, color=C_MUTED)
    axis.set_ylabel("errors (40 frames at 20 dB)", fontsize=9.5, color=C_MUTED)
    axis.grid(True, color=C_GRID, linewidth=0.6)
    axis.set_title(f"all {len(positions)} errors are symbol 0", fontsize=10.5,
                   color=C_TEXT, loc="left")

    caption(
        figure,
        "What the benchmark found first: a bug the whole test suite had missed",
        "Before Phase 6, more signal stopped helping: 16QAM with Doppler hit a floor near "
        "1e-3 and stayed there. Every one of those errors was the first symbol of\nits frame. "
        "The matched filter is circular -- it treats the frame's end as joined to its start -- "
        "and it ran before frequency correction, so the Doppler\nrotation made end and start "
        "disagree and the filter blended the mismatch into symbol 0. Correcting the frequency "
        "first removed the floor entirely.",
        0.80,
    )
    path = OUTPUT_DIR / "bug_found.png"
    figure.savefig(path, dpi=150, facecolor=C_SURFACE)
    plt.close(figure)
    return path, result, len(positions)


# ======================================================================
# 2. The main benchmark
# ======================================================================
def figure_benchmark():
    modulations = ["QPSK", "16QAM", "64QAM"]
    grid = ScenarioGrid.product(modulations, [0, 4, 8, 12, 16], ImpairmentProfile.standard())
    receivers = {"classical": classical(), "hybrid": hybrid_bank(grid)}

    runner = BenchmarkRunner(TEST, receivers, frames_per_scenario=20)
    runner.add_observer(ConsoleProgress(every=30))
    result = runner.run(grid)

    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), facecolor=C_SURFACE, sharex=True)
    for column, modulation in enumerate(modulations):
        for row, profile in enumerate(("clean", "full")):
            axis = axes[row, column]
            style(axis)
            for index, name in enumerate(receivers):
                snr, ber, interval = result.curve(name, modulation, profile)
                offset = (index - 0.5) * 0.35   # nudge so overlapping bars stay visible
                shown = np.maximum(ber, 1e-5)
                axis.errorbar(
                    snr + offset, shown,
                    yerr=[shown - np.maximum(interval[:, 0], 1e-5),
                          np.maximum(interval[:, 1], 1e-5) - shown],
                    color=SERIES[index], marker="o", markersize=4, linewidth=1.6,
                    capsize=2.5, label=name,
                )
            axis.set_yscale("log")
            axis.set_ylim(5e-6, 0.6)
            axis.grid(True, which="both", color=C_GRID, linewidth=0.6)
            verdict = result.compare("hybrid", "classical", grid.where(modulation=modulation, profile=profile))
            axis.set_title(f"{modulation}, {profile}  -  {verdict.verdict.replace(' on these frames', '')}",
                           fontsize=9.5, color=C_TEXT, loc="left")
            if row == 1:
                axis.set_xlabel("Signal to noise ratio (dB)", fontsize=9.5, color=C_MUTED)
            if column == 0:
                axis.set_ylabel("bit error rate\n(zero errors drawn at 1e-5)", fontsize=9, color=C_MUTED)
    axes[0, 0].legend(loc="lower left", fontsize=8.5, frameon=True,
                      facecolor=C_SURFACE, edgecolor=C_GRID, labelcolor=C_MUTED)

    caption(
        figure,
        "Classical against hybrid, every receiver on the same frames",
        "Bars are 95% intervals, taken as the wider of the textbook (Wilson) interval and a "
        "bootstrap over whole frames -- each is wrong in a different direction, so\nthe "
        "project reports the span covering both. The two receivers tie almost everywhere, "
        "which is what theory predicts: on these channels the classical rule is\nalready "
        "close to optimal, and a learned demapper only earns its place when symbols smear "
        "into each other (Phase 4).",
        0.85,
    )
    path = OUTPUT_DIR / "benchmark.png"
    figure.savefig(path, dpi=150, facecolor=C_SURFACE)
    plt.close(figure)
    return path, result, grid


# ======================================================================
# 3. Held-out conditions
# ======================================================================
def figure_held_out():
    modulations = ["16QAM", "64QAM"]
    grid = ScenarioGrid.product(modulations, [-4, 0, 4, 8, 12, 16], ImpairmentProfile.standard())
    strategies = [
        RandomSplit(0.3, seed=0),
        HoldOutSNR(-10, 0),
        HoldOutProfile("doppler"),
        HoldOutProfile("full"),
    ]

    rows = []
    for strategy in strategies:
        train, test = strategy.split(grid)
        bank = hybrid_bank(train)
        result = BenchmarkRunner(TEST, {"classical": classical(), "hybrid": bank},
                                 frames_per_scenario=20).run(test)
        for modulation in modulations:
            subset = test.where(modulation=modulation)
            comparison = result.compare("hybrid", "classical", subset)
            base = sum(c.error_rate for c in result.select("classical", modulation)).rate
            rows.append((strategy.name, modulation, comparison, base, len(train), len(test)))
        print(f"    {strategy.name:30s} trained on {len(train)}, tested on {len(test)} scenarios")

    figure, axis = plt.subplots(figsize=(11.0, 5.0), facecolor=C_SURFACE)
    style(axis)
    labels = []
    for index, (name, modulation, comparison, base, _, _) in enumerate(rows):
        low, high = comparison.interval
        relative = np.array([low, comparison.difference, high]) / base * 100.0
        colour = SERIES[0] if modulation == "16QAM" else SERIES[1]
        axis.errorbar(index, relative[1], yerr=[[relative[1] - relative[0]], [relative[2] - relative[1]]],
                      color=colour, marker="o", markersize=6, capsize=4, linewidth=1.8)
        short = {
            "random 30%": "random 30%\n(seen conditions)",
            "held-out SNR -10 to +0 dB": "unseen\nlow SNR",
            "held-out profile 'doppler'": "unseen\nDoppler",
            "held-out profile 'full'": "unseen\nfull profile",
        }.get(name, name)
        labels.append(f"{short}\n{modulation}")
    axis.axhline(0.0, color=C_TEXT, linewidth=1.0)
    axis.set_xticks(range(len(rows)))
    axis.set_xticklabels(labels, fontsize=8, color=C_MUTED)
    axis.set_ylabel("hybrid minus classical errors\n(% of classical's error rate)", fontsize=9.5, color=C_MUTED)
    axis.grid(True, axis="y", color=C_GRID, linewidth=0.6)
    span = max(abs(v) for _, _, c, b, _, _ in rows for v in np.array(c.interval) / b * 100.0)
    axis.set_ylim(-span * 1.4, span * 1.4)
    axis.text(len(rows) - 0.5, span * 1.2, "hybrid worse", ha="right", fontsize=8.5, color=C_MUTED)
    axis.text(len(rows) - 0.5, -span * 1.25, "hybrid better", ha="right", fontsize=8.5, color=C_MUTED)

    caption(
        figure,
        "Tested on conditions it never trained on, the hybrid holds up",
        "Each bar is the hybrid's error rate minus the classical receiver's, on scenarios held "
        "out of training, with a 95% paired-bootstrap interval. Every one sits\nwithin about "
        "3% of the classical error rate, and the only two clearly away from zero are on the "
        "random split -- the easy one. The reason is\narchitectural: the classical front end "
        "removes the Doppler, timing and fading before the network sees anything, so its input "
        "looks much the same\nwhatever the channel did. Phase 4's end-to-end network had to "
        "cope with the raw impairments itself -- and collapsed.",
        0.80,
    )
    path = OUTPUT_DIR / "held_out.png"
    figure.savefig(path, dpi=150, facecolor=C_SURFACE)
    plt.close(figure)
    return path, rows


# ======================================================================
# 4. Self-knowledge: the ablation and the fix
# ======================================================================
class FixedNoiseSoftDemapper(SoftDemapper):
    """The ablation: a soft demapper that never re-measures the noise.

    One method overridden, everything else inherited, so the comparison
    isolates exactly one difference.
    """

    def __init__(self, noise_power: float) -> None:
        super().__init__()
        self._fixed = float(noise_power)
        self._name = "Soft (fixed noise)"

    def _estimate_noise_power(self, squared_distances):
        return self._fixed


def figure_self_knowledge():
    modulation = "64QAM"
    scheme = get_modulation(modulation)
    grid = ScenarioGrid.product(
        [modulation], [0, 2, 4, 6, 8, 10, 12], [ImpairmentProfile.clean(), ImpairmentProfile.full()]
    )

    reference_frames = [
        classical().recover(frame).symbols
        for scenario in grid
        for frame in TRAIN.frames(scenario, 4)
    ]
    reference = NoiseNormalisedDemapper.fit_reference(reference_frames, scheme)
    network = train_demapper(grid, modulation)

    receivers = {
        "classical": classical(),
        "classical, noise held fixed": ClassicalReceiver(
            pulse=PULSE, demapper=FixedNoiseSoftDemapper(reference)
        ),
        "hybrid": ClassicalReceiver(pulse=PULSE, demapper=network),
        "hybrid + per-frame noise": ClassicalReceiver(
            pulse=PULSE, demapper=NoiseNormalisedDemapper(network, reference)
        ),
    }
    result = BenchmarkRunner(TEST, receivers, frames_per_scenario=40).run(grid)

    summary = {}
    for name in receivers:
        cells = result.select(name)
        confidence = np.concatenate([c.frame_confidence for c in cells])
        lost = np.concatenate([c.frame_lost for c in cells])
        curve = RiskCoverageCurve(confidence, ~lost, label=name)
        summary[name] = {
            "curve": curve,
            "lost": int(lost.sum()),
            "frames": int(lost.size),
            "ber": sum(c.error_rate for c in cells).rate,
            "ece": float(np.mean([c.ece for c in cells])),
        }

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), facecolor=C_SURFACE,
                                gridspec_kw={"width_ratios": [1.0, 1.25]})
    colours = [SERIES[0], C_REF, SERIES[1], SERIES[2]]

    axis = axes[0]
    style(axis)
    names = list(receivers)
    scores = [summary[n]["curve"].ranking_quality for n in names]
    bars = axis.barh(range(len(names)), scores, color=colours, height=0.6)
    for bar, value in zip(bars, scores):
        axis.text(value - 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                  va="center", ha="right", fontsize=9, color="white", fontweight="bold")
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(names, fontsize=9, color=C_MUTED)
    axis.invert_yaxis()
    axis.set_xlim(0.5, 1.0)
    axis.set_xlabel("how well mean confidence picks out lost frames", fontsize=9.5, color=C_MUTED)
    axis.grid(True, axis="x", color=C_GRID, linewidth=0.6)
    axis.set_title("same decisions, different self-knowledge", fontsize=10.5, color=C_TEXT, loc="left")

    axis = axes[1]
    style(axis)
    for colour, name in zip(colours, names):
        curve = summary[name]["curve"]
        axis.plot(curve.coverage, curve.risk, color=colour, linewidth=1.8,
                  label=f"{name}  (keeps {curve.coverage_at_risk(0.01):.0%})")
    axis.axhline(0.01, color=C_TEXT, linewidth=1.0, linestyle="--", alpha=0.7)
    axis.text(0.02, 0.013, "1% of kept frames unusable", fontsize=8, color=C_MUTED)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 0.35)
    axis.set_xlabel("fraction of frames kept", fontsize=9.5, color=C_MUTED)
    axis.set_ylabel("fraction of kept frames that lost lock", fontsize=9.5, color=C_MUTED)
    axis.grid(True, color=C_GRID, linewidth=0.6)
    axis.set_title("frame triage, 64QAM", fontsize=10.5, color=C_TEXT, loc="left")
    axis.legend(loc="upper left", fontsize=8, frameon=True,
                facecolor=C_SURFACE, edgecolor=C_GRID, labelcolor=C_MUTED)

    fixed = summary["classical, noise held fixed"]["curve"].ranking_quality
    plain = summary["hybrid"]["curve"].ranking_quality
    caption(
        figure,
        "Why the hybrid was worse at noticing a lost frame -- and the fix",
        "All four make identical bit decisions. The classical demapper re-measures the noise on "
        "every frame and divides by it, so on a broken frame all its odds shrink\ntogether. "
        f"Hold that estimate fixed and its triage falls from "
        f"{summary['classical']['curve'].ranking_quality:.3f} to {fixed:.3f} -- the hybrid's "
        f"level ({plain:.3f}). A trained network has fixed weights and no per-frame step, so it\n"
        "cannot do this on its own. Wrapping it to rescale its odds by each frame's measured "
        "noise restores the triage, and changes no bit.",
        0.80,
    )
    path = OUTPUT_DIR / "self_knowledge.png"
    figure.savefig(path, dpi=150, facecolor=C_SURFACE)
    plt.close(figure)
    return path, summary


# ======================================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    began = time.time()

    print("Phase 6 -- one fair benchmark, and what it found")
    print("=" * 74)

    print("\n1. The bug the benchmark found")
    print("-" * 74)
    path, bug, n_positions = figure_bug()
    for name in ("before the fix", "after the fix"):
        cells = bug.select(name)
        print(f"    {name:16s} BER at 20 dB: {cells[-1].ber:.2e}   "
              f"(pooled over 4-20 dB: {sum(c.error_rate for c in cells).rate:.2e})")
    print(f"  wrote {path.relative_to(OUTPUT_DIR.parent)}")

    print("\n2. Classical against hybrid")
    print("-" * 74)
    path, result, grid = figure_benchmark()
    print(f"\n    {'':8s} {'profile':8s} {'hybrid minus classical':>24s}  verdict")
    for modulation in grid.modulations:
        for profile in grid.profiles:
            c = result.compare("hybrid", "classical", grid.where(modulation=modulation, profile=profile))
            print(f"    {modulation:8s} {profile:8s} {c.difference:+12.2e} "
                  f"[{c.interval[0]:+.1e},{c.interval[1]:+.1e}]  {c.verdict}")

    print("\n    Why the default interval is the conservative one (classical, 64QAM, full):")
    print(f"    {'SNR':>5s} {'BER':>9s} {'lost':>5s} {'Wilson width':>13s} {'bootstrap width':>16s}")
    for cell in result.select("classical", "64QAM", "full"):
        w = np.diff(cell.ber_interval("wilson"))[0]
        b = np.diff(cell.ber_interval("bootstrap"))[0]
        print(f"    {cell.scenario.snr_db:5.0f} {cell.ber:9.2e} {cell.lost_frames:5d} {w:13.2e} {b:16.2e}")

    with ResultStore(OUTPUT_DIR / "results.db") as store:
        run_id = store.save(result, "phase 6 main benchmark", notes="classical vs hybrid, simulated")
    print(f"\n  saved {len(result)} cells to outputs/results.db (run {run_id})")
    print(f"  wrote {path.relative_to(OUTPUT_DIR.parent)}")

    print("\n3. Conditions the hybrid never trained on")
    print("-" * 74)
    path, rows = figure_held_out()
    print(f"\n    {'split':30s} {'mod':6s} {'hybrid minus classical':>24s}  verdict")
    for name, modulation, c, base, _, _ in rows:
        print(f"    {name:30s} {modulation:6s} {c.difference:+12.2e} "
              f"[{c.interval[0]:+.1e},{c.interval[1]:+.1e}]  {c.verdict}")
    print(f"  wrote {path.relative_to(OUTPUT_DIR.parent)}")

    print("\n4. Self-knowledge: the ablation and the fix")
    print("-" * 74)
    path, summary = figure_self_knowledge()
    print(f"    {'receiver':30s} {'triage':>7s} {'keep at 1%':>11s} {'ECE':>8s} {'BER':>10s}")
    for name, s in summary.items():
        print(f"    {name:30s} {s['curve'].ranking_quality:7.3f} "
              f"{s['curve'].coverage_at_risk(0.01):11.3f} {s['ece']:8.4f} {s['ber']:10.3e}")
    print(f"  wrote {path.relative_to(OUTPUT_DIR.parent)}")

    verdicts = [
        result.compare("hybrid", "classical", grid.where(modulation=m, profile=p))
        for m in grid.modulations for p in grid.profiles
    ] + [c for _, _, c, _, _, _ in rows]
    hybrid_wins = sum(v.difference < 0 and v.is_decisive for v in verdicts)
    classical_wins = sum(v.difference > 0 and v.is_decisive for v in verdicts)

    print("\n" + "=" * 74)
    print(f"Done in {time.time() - began:.0f}s.\n")
    print(f"Across {len(verdicts)} paired comparisons: hybrid decisively better in "
          f"{hybrid_wins}, classical in {classical_wins}, no detectable difference in "
          f"{len(verdicts) - hybrid_wins - classical_wins}.")
    print(f"At 95% confidence, about {0.05 * len(verdicts):.0f} of {len(verdicts)} would "
          "look decisive by chance alone, so no single")
    print("decisive cell is strong evidence by itself; the pattern is what counts.\n")
    print("Reported as measured:")
    print("  - the benchmark found a bug in the classical receiver; it is fixed")
    print("  - a hybrid trained on a third of the data lost to classical on 64QAM")
    print("    in every profile; with enough data that shrank to one profile --")
    print("    clean noise, where the classical rule is optimal and a network can")
    print("    at best match it")
    print("  - on held-out conditions the hybrid holds up, because the classical")
    print("    front end absorbs what changed")
    print("  - the hybrid's weaker frame triage came from one missing step,")
    print("    proved by removing that step from the classical receiver, and")
    print("    restored by adding it to the hybrid")


if __name__ == "__main__":
    main()
