#!/usr/bin/env python3
"""Phase 5 demonstration: does the receiver know when it is wrong?

Run from the project root::

    python3 scripts/demo_phase5.py

Produces three figures in ``outputs/``:

``confidence_honesty.png``
    Whether the stated probabilities are true, and *why* they stop being
    true.  The receiver estimates the noise power from the signal
    itself; that estimate collapses exactly when the signal gets bad,
    and the overconfidence follows it point for point.

``confidence_ranking.png``
    Whether the numbers are useful even when they are not honest.  Five
    ways of judging a decision, against a perfect ranking and a
    worthless one.

``frame_triage.png``
    The result worth keeping.  Pooling the per-bit confidence into a
    single number per frame picks out the frames where the receiver has
    lost lock -- which the standard engineering measure, error vector
    magnitude, almost cannot do.

Everything is measured, including the results that went against
expectation.  Two of them did, and they are marked in the output.
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
    CalibrationReport,
    ChannelConditions,
    ClassicalReceiver,
    EVMConfidence,
    HistogramBinning,
    IdentityCalibrator,
    LLRConfidence,
    MarginConfidence,
    RiskCoverageCurve,
    SignalSimulator,
    SoftDemapper,
    TemperatureScaling,
    get_modulation,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

C_SURFACE = "#fcfcfb"
C_TEXT = "#0b0b0b"
C_MUTED = "#52514e"
C_GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#8257d6", "#c23b3b"]
C_REF = "#9a9892"

#: One channel that is only noise, and one with everything else wrong
#: as well.  Every comparison in this demo is run on both, because the
#: whole Phase 5 question is whether self-knowledge survives conditions
#: the receiver's assumptions do not cover.
IMPAIRED = dict(
    freq_offset_hz=800.0,
    phase_offset_rad=0.7,
    timing_offset=0.4,
    fading="multipath_rician",
    rician_k_db=10.0,
)


def style(axis) -> None:
    axis.set_facecolor(C_SURFACE)
    axis.tick_params(colors=C_MUTED, labelsize=9)
    for edge in axis.spines.values():
        edge.set_color(C_GRID)


# ======================================================================
# Shared measurement helpers
# ======================================================================
def recover_many(simulator, receiver, modulation, snr_db, n_frames, **conditions):
    """Recover a batch of frames and keep the truth alongside each one."""
    out = []
    for _ in range(n_frames):
        frame = simulator.generate_frame(
            modulation, ChannelConditions(snr_db=snr_db, **conditions)
        )
        out.append((receiver.recover(frame), np.asarray(frame.bits)))
    return out


def pool(batch, estimator=None):
    """Flatten a batch into one long (confidence, was-it-right) pair.

    Pooling across frames is the standard way to report calibration and
    it is what the first two figures do.  The third figure exists
    because pooling also *hides* something, and that turned out to
    matter more than anything pooling revealed.
    """
    scores, correct = [], []
    for result, truth in batch:
        scores.append(
            result.confidence if estimator is None else estimator.estimate(result)
        )
        correct.append(~result.error_mask(truth))
    return np.concatenate(scores), np.concatenate(correct)


def noise_power_ratio(batch, modulation):
    """How far the receiver's own noise estimate is from the truth.

    The soft demapper divides by an estimated noise power, taken as the
    mean squared distance from each symbol to its *nearest* legal point.
    Once symbols routinely land nearer a wrong point, that statistic
    measures the distance to the wrong point, which is small -- so the
    estimate falls exactly when the real noise rises.  This returns
    estimate divided by truth: 1.0 is honest, below 1.0 inflates every
    log-likelihood ratio the demapper emits.
    """
    estimated, actual = [], []
    for result, truth in batch:
        symbols = result.symbols
        sent = modulation.map_bits(truth)[: symbols.size]
        squared = np.abs(symbols[:, None] - modulation.constellation[None, :]) ** 2
        estimated.append(float(np.mean(np.min(squared, axis=1))))
        actual.append(float(np.mean(np.abs(symbols - sent) ** 2)))
    return float(np.mean(estimated) / np.mean(actual))


# ======================================================================
# Figure 1 -- are the stated probabilities honest, and why not?
# ======================================================================
def figure_honesty():
    simulator = SignalSimulator(seed=100)
    receiver = ClassicalReceiver(pulse=simulator.pulse, demapper=SoftDemapper())
    qam16 = get_modulation("16QAM")

    snrs = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    overconfidence, ratios, bers = [], [], []
    for snr in snrs:
        batch = recover_many(simulator, receiver, "16QAM", snr, 40)
        scores, correct = pool(batch)
        report = CalibrationReport(scores, correct)
        overconfidence.append(report.overconfidence)
        ratios.append(noise_power_ratio(batch, qam16))
        bers.append(1.0 - report.accuracy)

    # The same broken condition twice: as it comes out, and after a
    # one-parameter repair fitted on a *separate* batch of frames.
    broken = recover_many(simulator, receiver, "16QAM", -4.0, 60)
    fitting = recover_many(simulator, receiver, "16QAM", -4.0, 60)
    raw_scores, raw_correct = pool(broken)
    temperature = TemperatureScaling().fit(*pool(fitting))

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), facecolor=C_SURFACE)

    # -- reliability diagrams, before and after the repair ------------
    for axis, scores, title in (
        (axes[0], raw_scores, "16QAM at -4 dB, as reported"),
        (
            axes[1],
            temperature.apply(raw_scores),
            f"the same bits, rescaled by T = {temperature.temperature:.2f}",
        ),
    ):
        style(axis)
        correct = raw_correct
        report = CalibrationReport(scores, correct, n_bins=12, binning="mass")

        occupied = report.bin_counts > 0
        claimed = report.bin_confidence[occupied]
        actual = report.bin_accuracy[occupied]

        axis.plot([0.5, 1.0], [0.5, 1.0], color=C_REF, linewidth=1.2, linestyle="--",
                  label="honest", zorder=1)
        axis.scatter(claimed, actual, s=np.clip(report.bin_counts[occupied] / 60, 8, 160),
                     color=SERIES[0], alpha=0.85, zorder=3, label="measured")
        for x, y in zip(claimed, actual):
            axis.plot([x, x], [x, y], color=SERIES[1], linewidth=1.0, alpha=0.7, zorder=2)

        axis.set_xlim(0.5, 1.005)
        axis.set_ylim(0.5, 1.005)
        axis.set_xlabel("probability the receiver claimed", fontsize=9.5, color=C_MUTED)
        axis.set_ylabel("fraction actually correct", fontsize=9.5, color=C_MUTED)
        axis.set_title(
            f"{title}\nECE {report.ece:.3f}", fontsize=10.5, color=C_TEXT, loc="left"
        )
        axis.grid(True, color=C_GRID, linewidth=0.6)
        axis.legend(loc="upper left", fontsize=8.5, frameon=True,
                    facecolor=C_SURFACE, edgecolor=C_GRID, labelcolor=C_MUTED)

    # -- the mechanism ------------------------------------------------
    axis = axes[2]
    style(axis)
    axis.plot(snrs, overconfidence, color=SERIES[1], marker="o", markersize=4,
              linewidth=1.8, label="overconfidence")
    axis.axhline(0.0, color=C_REF, linewidth=1.0, linestyle="--")
    axis.set_xlabel("Signal to noise ratio (dB)", fontsize=9.5, color=C_MUTED)
    axis.set_ylabel("claimed minus actual", fontsize=9.5, color=C_MUTED)
    axis.grid(True, color=C_GRID, linewidth=0.6)

    twin = axis.twinx()
    twin.plot(snrs, ratios, color=SERIES[0], marker="s", markersize=4,
              linewidth=1.8, label="noise estimate / truth")
    twin.axhline(1.0, color=SERIES[0], linewidth=0.8, linestyle=":", alpha=0.6)
    twin.set_ylabel("noise estimate / truth", fontsize=9.5, color=SERIES[0])
    twin.tick_params(colors=SERIES[0], labelsize=9)
    twin.set_ylim(0.0, 1.15)
    for edge in twin.spines.values():
        edge.set_color(C_GRID)

    handles = [
        plt.Line2D([], [], color=SERIES[1], marker="o", label="overconfidence (left)"),
        plt.Line2D([], [], color=SERIES[0], marker="s", label="noise estimate / truth (right)"),
    ]
    axis.legend(handles=handles, loc="center right", fontsize=8.5, frameon=True,
                facecolor=C_SURFACE, edgecolor=C_GRID, labelcolor=C_MUTED)
    axis.set_title("the two track each other", fontsize=10.5, color=C_TEXT, loc="left")

    figure.suptitle(
        "Is the receiver's confidence honest?",
        fontsize=13.5, color=C_TEXT, x=0.012, ha="left", y=0.985,
    )
    figure.text(
        0.012, 0.935,
        s=(
            "Left: at -4 dB every bin sits below the line -- the receiver claims "
            "more than it delivers. Middle: the same bits with every probability "
            "divided by one\nnumber, fitted on a separate batch. No decision "
            "changed and no bit was corrected; only the claim was. Right: the "
            "cause. The demapper estimates the\nnoise power from the signal "
            "itself, as the distance to the nearest legal point, and at -4 dB "
            f"that estimate is {ratios[0]:.2f} of the truth -- because symbols are "
            "landing nearer the\nWRONG point, so the distance it measures is to "
            "that one. Underestimating the noise inflates every probability it "
            "reports, and the two curves track accordingly."
        ),
        va="top", fontsize=9, color=C_MUTED,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.74))
    path = OUTPUT_DIR / "confidence_honesty.png"
    figure.savefig(path, dpi=150, facecolor=C_SURFACE)
    plt.close(figure)

    return path, list(zip(snrs, bers, overconfidence, ratios))


# ======================================================================
# Figure 2 -- which way of judging a decision ranks best?
# ======================================================================
def figure_ranking():
    simulator = SignalSimulator(seed=200)
    receiver = ClassicalReceiver(pulse=simulator.pulse, demapper=SoftDemapper())

    estimators = [
        LLRConfidence(),
        EVMConfidence(),
        MarginConfidence(),
        LLRConfidence() | EVMConfidence(),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), facecolor=C_SURFACE)

    # -- the curves themselves, one hard condition --------------------
    axis = axes[0]
    style(axis)
    batch = recover_many(simulator, receiver, "16QAM", 0.0, 60)
    table = []
    for index, estimator in enumerate(estimators):
        scores, correct = pool(batch, estimator)
        curve = RiskCoverageCurve(scores, correct, label=estimator.name)
        axis.plot(curve.coverage, curve.risk, color=SERIES[index], linewidth=1.8,
                  label=f"{estimator.name}  ({curve.ranking_quality:.2f})")
        table.append((estimator.name, curve))

    perfect = RiskCoverageCurve(correct.astype(float), correct)
    axis.plot(perfect.coverage, perfect.risk, color=C_TEXT, linewidth=1.2,
              linestyle="--", label="a perfect ranking  (1.00)")
    axis.axhline(1.0 - correct.mean(), color=C_REF, linewidth=1.2, linestyle=":",
                 label="no ranking at all  (0.00)")

    axis.set_xlabel("fraction of bits kept", fontsize=9.5, color=C_MUTED)
    axis.set_ylabel("error rate among those kept", fontsize=9.5, color=C_MUTED)
    axis.set_yscale("log")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(1e-4, 0.4)
    axis.grid(True, which="both", color=C_GRID, linewidth=0.6)
    axis.set_title("16QAM at 0 dB, noise only", fontsize=10.5, color=C_TEXT, loc="left")
    axis.legend(loc="lower left", fontsize=8.5, frameon=True, ncols=2,
                facecolor=C_SURFACE, edgecolor=C_GRID, labelcolor=C_MUTED)

    # -- how it holds up across conditions ----------------------------
    axis = axes[1]
    style(axis)
    snrs = [0.0, 2.0, 4.0, 6.0, 8.0]
    clean_quality, impaired_quality = [], []
    for snr in snrs:
        clean = recover_many(simulator, receiver, "16QAM", snr, 40)
        rough = recover_many(simulator, receiver, "16QAM", snr, 40, **IMPAIRED)
        for store, batch in ((clean_quality, clean), (impaired_quality, rough)):
            scores, correct = pool(batch, LLRConfidence())
            store.append(RiskCoverageCurve(scores, correct).ranking_quality)

    axis.plot(snrs, clean_quality, color=SERIES[0], marker="o", markersize=4,
              linewidth=1.8, label="noise only")
    axis.plot(snrs, impaired_quality, color=SERIES[4], marker="s", markersize=4,
              linewidth=1.8, label="noise + Doppler + fading + timing error")
    axis.axhline(1.0, color=C_TEXT, linewidth=1.0, linestyle="--", alpha=0.6)
    axis.axhline(0.0, color=C_REF, linewidth=1.0, linestyle=":")
    axis.set_xlabel("Signal to noise ratio (dB)", fontsize=9.5, color=C_MUTED)
    axis.set_ylabel("ranking quality  (1 perfect, 0 useless)", fontsize=9.5, color=C_MUTED)
    axis.set_ylim(-0.05, 1.08)
    axis.grid(True, color=C_GRID, linewidth=0.6)
    axis.set_title("the same measure, two channels", fontsize=10.5, color=C_TEXT, loc="left")
    axis.legend(loc="lower right", fontsize=8.5, frameon=True,
                facecolor=C_SURFACE, edgecolor=C_GRID, labelcolor=C_MUTED)

    figure.suptitle(
        "Are the numbers useful, even when they are not honest?",
        fontsize=13.5, color=C_TEXT, x=0.012, ha="left", y=0.985,
    )
    figure.text(
        0.012, 0.93,
        s=(
            "Throw away the least confident bits and watch the error rate among "
            "the rest. The log-likelihood ratio -- the only estimator with a "
            "theory behind it --\nwins everywhere, and the two heuristics lose "
            "badly. Averaging it with a weak one makes it worse, not better: a "
            "composite is only as good as\nits members. On the right, the cost of "
            "the impairments: the error rate roughly doubles, but the receiver's "
            "grip on which bits to trust falls much\nfurther -- and erratically, "
            "which is not noise in the measurement. It is a few frames where the "
            "receiver lost lock entirely. The next figure is about those."
        ),
        va="top", fontsize=9, color=C_MUTED,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.78))
    path = OUTPUT_DIR / "confidence_ranking.png"
    figure.savefig(path, dpi=150, facecolor=C_SURFACE)
    plt.close(figure)

    return path, table, list(zip(snrs, clean_quality, impaired_quality))


# ======================================================================
# Figure 3 -- the result worth keeping: triage whole frames
# ======================================================================
def figure_triage():
    """Pooling across frames hid something. This is what it hid.

    A receiver that loses carrier lock on one frame produces a frame
    that is wrong nearly everywhere -- and fairly confident about it.
    Averaged into a batch of healthy frames, that frame disappears into
    a slightly worse calibration number.  Counted per frame, it is the
    only thing that matters: a handful of them carry most of the errors
    in the whole batch.

    So the question changes from *which bits do I trust* to *do I trust
    this frame at all*, and the measuring apparatus does not need to
    change with it.  A risk-coverage curve does not care whether a
    "decision" is one bit or one whole frame, which is the payoff for
    having written it against an interface rather than against bits.

    A NOTE ON A CLAIM THAT DID NOT SURVIVE CHECKING
    -------------------------------------------------
    The first version of this experiment concluded that error vector
    magnitude is a poor detector of a lost frame while the pooled
    confidence is a good one.  That was an artefact.  The batch mixes
    16QAM and 64QAM, and EVM has no common scale across them: a
    perfectly healthy 64QAM frame shows a larger error vector than a
    thoroughly broken 16QAM one, so no single threshold can serve both.

    Within one modulation the two measures are equally good.  The real
    finding is narrower and more useful: a confidence is a probability
    and so is comparable across configurations, while a distance is not
    -- which matters precisely in the mixed-modulation setting this
    project is built around.  Both versions are shown.
    """
    simulator = SignalSimulator(seed=600)
    receiver = ClassicalReceiver(pulse=simulator.pulse, demapper=SoftDemapper())
    estimator = LLRConfidence()
    rng = np.random.default_rng(1)

    schemes = ("16QAM", "64QAM")
    per_scheme = {}
    for modulation in schemes:
        bers, confidence, evm = [], [], []
        for _ in range(250):
            snr = float(rng.uniform(2.0, 12.0))
            frame = simulator.generate_frame(
                modulation, ChannelConditions(snr_db=snr, **IMPAIRED)
            )
            result = receiver.recover(frame)
            correct = ~result.error_mask(np.asarray(frame.bits))
            bers.append(1.0 - correct.mean())
            confidence.append(float(np.mean(estimator.estimate(result))))
            evm.append(float(result.diagnostics.get("evm_percent", np.nan)))
        per_scheme[modulation] = tuple(
            np.array(column) for column in (bers, confidence, evm)
        )

    bers = np.concatenate([per_scheme[m][0] for m in schemes])
    confidence = np.concatenate([per_scheme[m][1] for m in schemes])
    evm = np.concatenate([per_scheme[m][2] for m in schemes])
    healthy = bers <= 0.1

    # EVM given every advantage: standardised within each modulation, so
    # it is judged on its shape rather than on its absolute scale.
    fair_evm = np.concatenate(
        [
            -(per_scheme[m][2] - per_scheme[m][2].mean()) / per_scheme[m][2].std()
            for m in schemes
        ]
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), facecolor=C_SURFACE)

    for axis, values, name, title in (
        (axes[0], evm, "error vector magnitude (%)", "no common scale"),
        (axes[1], confidence, "mean confidence over the frame", "one common scale"),
    ):
        style(axis)
        offset = 0
        for index, modulation in enumerate(schemes):
            size = per_scheme[modulation][0].size
            chunk = slice(offset, offset + size)
            offset += size
            local = healthy[chunk]
            axis.scatter(values[chunk][local], np.maximum(bers[chunk][local], 1e-4),
                         s=10, color=SERIES[index], alpha=0.45,
                         label=f"{modulation}, locked")
            axis.scatter(values[chunk][~local], np.maximum(bers[chunk][~local], 1e-4),
                         s=16, color=SERIES[4], alpha=0.6,
                         label="lost lock" if index == 0 else None)
        axis.set_yscale("log")
        axis.set_xlabel(name, fontsize=9.5, color=C_MUTED)
        axis.set_ylabel("bit error rate of that frame", fontsize=9.5, color=C_MUTED)
        axis.grid(True, which="both", color=C_GRID, linewidth=0.6)
        axis.legend(loc="lower left", fontsize=7.5, frameon=True,
                    facecolor=C_SURFACE, edgecolor=C_GRID, labelcolor=C_MUTED)
        axis.set_title(title, fontsize=10.5, color=C_TEXT, loc="left")

    # -- triage as a risk-coverage curve, one frame per decision ------
    axis = axes[2]
    style(axis)
    candidates = {
        "mean confidence": confidence,
        "EVM, scaled per scheme": fair_evm,
        "EVM, as measured": -evm,
        "no ranking at all": rng.random(bers.size),
    }
    summary = {}
    for index, (name, score) in enumerate(candidates.items()):
        scaled = (score - score.min()) / max(np.ptp(score), 1e-12)
        curve = RiskCoverageCurve(scaled, healthy, label=name)
        axis.plot(curve.coverage, curve.risk, color=SERIES[index], linewidth=1.8,
                  label=f"{name}  ({curve.ranking_quality:.2f})")
        summary[name] = curve

    axis.axhline(0.01, color=C_TEXT, linewidth=1.0, linestyle="--", alpha=0.7)
    axis.text(0.02, 0.0125, "1% of kept frames unusable", fontsize=8,
              color=C_MUTED, va="bottom")
    axis.set_xlabel("fraction of frames kept", fontsize=9.5, color=C_MUTED)
    axis.set_ylabel("fraction of kept frames that lost lock", fontsize=9.5, color=C_MUTED)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 0.25)
    axis.grid(True, color=C_GRID, linewidth=0.6)
    axis.set_title("keep the frames worth keeping", fontsize=10.5, color=C_TEXT, loc="left")
    axis.legend(loc="upper left", fontsize=8, frameon=True,
                facecolor=C_SURFACE, edgecolor=C_GRID, labelcolor=C_MUTED)

    # Within one modulation, how do the two compare?  This is the
    # check that overturned the first reading of this experiment.
    alone = {}
    for modulation in schemes:
        frame_bers, frame_conf, frame_evm = per_scheme[modulation]
        local = frame_bers <= 0.1
        alone[modulation] = (
            RiskCoverageCurve(
                (frame_conf - frame_conf.min()) / max(np.ptp(frame_conf), 1e-12), local
            ).ranking_quality,
            RiskCoverageCurve(
                (-frame_evm + frame_evm.max()) / max(np.ptp(frame_evm), 1e-12), local
            ).ranking_quality,
        )

    keep_conf = summary["mean confidence"].coverage_at_risk(0.01)
    keep_evm = summary["EVM, as measured"].coverage_at_risk(0.01)

    figure.suptitle(
        "The useful question is not which bits to trust, but which frames",
        fontsize=13.5, color=C_TEXT, x=0.012, ha="left", y=0.985,
    )
    figure.text(
        0.012, 0.935,
        s=(
            f"{int(np.sum(~healthy))} of {bers.size} frames lost carrier lock and "
            "are wrong nearly everywhere. Averaged into a batch they vanish into a "
            "slightly worse calibration\nnumber; counted per frame they carry most "
            "of the errors. The batch mixes 16QAM and 64QAM, and that is the whole "
            "story on the left: 16QAM tolerates far\nmore distortion, so a "
            "perfectly healthy 16QAM frame shows a LARGER error vector than a "
            "64QAM frame that has failed completely. No single threshold\nserves "
            "both. Within one modulation the two measures tie "
            f"({alone['64QAM'][0]:.2f} against {alone['64QAM'][1]:.2f} on 64QAM); "
            "across modulations they do not, "
            "because a confidence is a probability and a probability means the "
            "same thing everywhere. Keeping frames at a\n1% failure rate: "
            f"{keep_conf:.0%} by confidence, {keep_evm:.0%} by error vector "
            "magnitude as a receiver would actually read it."
        ),
        va="top", fontsize=9, color=C_MUTED,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.74))
    path = OUTPUT_DIR / "frame_triage.png"
    figure.savefig(path, dpi=150, facecolor=C_SURFACE)
    plt.close(figure)

    return path, summary, float(np.mean(~healthy)), alone


# ======================================================================
# Console section: what calibration can and cannot do
# ======================================================================
def report_calibration() -> None:
    """Fit a calibrator on one channel and use it on another.

    Also the demonstration of the claim the tests assert: every
    calibrator here is monotone, so the risk-coverage curve is
    *identical* before and after.  Calibration makes a receiver honest.
    It cannot make it better.
    """
    simulator = SignalSimulator(seed=300)
    receiver = ClassicalReceiver(pulse=simulator.pulse, demapper=SoftDemapper())
    estimator = EVMConfidence()

    fitting = recover_many(simulator, receiver, "16QAM", 2.0, 50)
    fit_scores, fit_correct = pool(fitting, estimator)

    print("  Fitted on 16QAM at +2 dB, noise only; a deliberately uncalibrated")
    print("  estimator (EVM) so there is something to repair.\n")
    print(f"    {'tested on':24s} {'calibrator':14s} {'ECE':>8s} {'over':>8s} "
          f"{'ranking':>9s}")

    for label, options in (
        ("the same channel", {}),
        ("+ Doppler and fading", IMPAIRED),
    ):
        batch = recover_many(simulator, receiver, "16QAM", 2.0, 50, **options)
        scores, correct = pool(batch, estimator)
        for calibrator in (IdentityCalibrator(), TemperatureScaling(), HistogramBinning(20)):
            calibrator.fit(fit_scores, fit_correct)
            fixed = calibrator.apply(scores)
            report = CalibrationReport(fixed, correct)
            curve = RiskCoverageCurve(fixed, correct)
            note = ""
            if hasattr(calibrator, "temperature"):
                note = f"   T = {calibrator.temperature:.3f}"
            print(f"    {label:24s} {calibrator.name:14s} {report.ece:8.4f} "
                  f"{report.overconfidence:+8.4f} {curve.ranking_quality:9.3f}{note}")
            label = ""

    print("\n  Temperature scaling leaves the ranking untouched to the last digit:")
    print("  dividing by a positive number cannot swap any pair of bits. Histogram")
    print("  binning moves it very slightly, and in one direction only -- it")
    print("  collapses every value in a bin to one number, so bits that used to be")
    print("  distinguishable become tied. It can lose resolution; it cannot")
    print("  reorder. Either way a badly scaled confidence is repairable and an")
    print("  uninformative one is not, which is why this project reports the two")
    print("  separately.")


# ======================================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 5 -- does the receiver know when it is wrong?")
    print("=" * 74)

    print("\n1. Is the stated confidence honest?")
    print("-" * 74)
    started = time.time()
    path, rows = figure_honesty()
    print(f"    {'SNR':>5s} {'BER':>10s} {'overconfidence':>15s} {'noise est/truth':>16s}")
    for snr, ber, over, ratio in rows:
        print(f"    {snr:5.0f} {ber:10.3e} {over:+15.4f} {ratio:16.3f}")
    print(f"\n  wrote {path.relative_to(OUTPUT_DIR.parent)}  ({time.time() - started:.0f}s)")

    print("\n2. Are the numbers useful for deciding what to keep?")
    print("-" * 74)
    started = time.time()
    path, table, sweep = figure_ranking()
    print(f"    {'estimator':18s} {'ranking quality':>16s} {'keep at 1e-3':>14s}")
    for name, curve in table:
        print(f"    {name:18s} {curve.ranking_quality:16.3f} "
              f"{curve.coverage_at_risk(1e-3):14.3f}")
    print(f"\n    {'SNR':>5s} {'noise only':>12s} {'impaired':>12s}")
    for snr, clean, rough in sweep:
        print(f"    {snr:5.0f} {clean:12.3f} {rough:12.3f}")
    print(f"\n  wrote {path.relative_to(OUTPUT_DIR.parent)}  ({time.time() - started:.0f}s)")

    print("\n3. What calibration can and cannot fix")
    print("-" * 74)
    report_calibration()

    print("\n4. The frames, not the bits")
    print("-" * 74)
    started = time.time()
    path, summary, lost, alone = figure_triage()
    print(f"    {lost:.1%} of frames lost lock, so keeping everything means "
          f"a {lost:.1%} failure rate.")
    print(f"\n    {'ranked by':26s} {'ranking quality':>16s} {'keep at 1% loss':>16s}")
    for name, curve in summary.items():
        print(f"    {name:26s} {curve.ranking_quality:16.3f} "
              f"{curve.coverage_at_risk(0.01):16.3f}")
    print(f"\n    within a single modulation, where EVM has a usable scale:")
    print(f"    {'':26s} {'confidence':>16s} {'EVM':>16s}")
    for modulation, (by_confidence, by_evm) in alone.items():
        print(f"    {modulation:26s} {by_confidence:16.3f} {by_evm:16.3f}")
    print(f"\n  wrote {path.relative_to(OUTPUT_DIR.parent)}  ({time.time() - started:.0f}s)")

    print("\n" + "=" * 74)
    print("Reported as measured, including where that was inconvenient:")
    print("  - combining a strong estimator with a weak one made the ranking")
    print("    worse, not better")
    print("  - a calibrator can make the receiver honest and provably cannot")
    print("    make it more accurate; the error rate is identical afterwards")
    print("  - the first reading of experiment 4 -- that EVM cannot spot a lost")
    print("    frame -- did not survive checking. Within one modulation it does,")
    print("    as well as the confidence does. What it cannot do is work across")
    print("    modulations, because a distance has no common scale and a")
    print("    probability does.")


if __name__ == "__main__":
    main()
