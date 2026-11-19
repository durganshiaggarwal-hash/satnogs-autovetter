#!/usr/bin/env python3
"""Phase 2 demonstration: the classical receiver repairing a signal.

Run from the project root::

    python3 scripts/demo_phase2.py

Produces two figures in ``outputs/``:

``receiver_stages.png``
    One badly damaged frame, shown at the output of every repair stage.
    This is the picture of what a receiver actually does.

``ber_curves.png``
    Measured bit error rate against the published theory, for four
    schemes. The gap between the two is implementation loss -- the price
    of the receiver having to work the frequency and timing out for
    itself instead of being told them.
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
    ChannelConditions,
    ClassicalReceiver,
    ModulationRegistry,
    Signal,
    SignalSimulator,
)
from satrecovery.receivers.base import RecoveryContext  # noqa: E402
from satrecovery.receivers.demapper import SoftDemapper  # noqa: E402
from satrecovery.receivers.stages import (  # noqa: E402
    AutomaticGainControl,
    CoarseFrequencyCorrection,
    DecisionDirectedPLL,
    GardnerTimingRecovery,
    MatchedFilterStage,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

#: Keep measuring a point until this many bit errors have accumulated,
#: so the estimate is not dominated by counting noise.
TARGET_ERRORS = 150
MAX_FRAMES = 160

# Validated for colour-vision deficiency and for contrast in both modes.
C_OK = "#2a78d6"
C_ERR = "#e34948"
C_REF = "#eb6834"
C_SURFACE = "#fcfcfb"
C_TEXT = "#0b0b0b"
C_MUTED = "#52514e"
C_GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]


def implementation_loss_db(scheme, ebn0_db: float, measured: float) -> float:
    """How much extra energy theory would need to be this bad.

    The honest way to quote implementation loss: find the Eb/N0 at which
    the textbook curve predicts the error rate we actually measured, and
    report the difference. Comparing error rates directly is misleading,
    because the curve is steep and a small shift sideways looks enormous.
    """
    if measured <= 0:
        return float("nan")
    lo, hi = -20.0, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        value = scheme.theoretical_ber(mid)
        if value is None:
            return float("nan")
        if float(value) > measured:
            lo = mid
        else:
            hi = mid
    return ebn0_db - 0.5 * (lo + hi)


def style_axes(axis, limit):
    axis.set_facecolor(C_SURFACE)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal")
    axis.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
    axis.tick_params(colors=C_MUTED, labelsize=7, length=3)
    for edge in axis.spines.values():
        edge.set_color(C_GRID)


def figure_stages() -> Path:
    """Follow one damaged frame through every stage of the receiver."""
    simulator = SignalSimulator(frame_length=8192, samples_per_symbol=8, seed=31)
    scheme = ModulationRegistry.create("16QAM")

    # Everything wrong at once: a weak signal, echoes, a Doppler shift,
    # an unknown phase and a misaligned sampling clock.
    conditions = ChannelConditions(
        snr_db=14.0,
        fading="multipath_rician",
        rician_k_db=10.0,
        freq_offset_hz=900.0,
        phase_offset_rad=0.9,
        timing_offset=0.55,
    )
    frame = simulator.generate_frame(scheme, conditions)

    context = RecoveryContext(
        scheme, 8, frame.signal.sample_rate, frame.tx_symbols[:16]
    )

    # Run the chain by hand so every intermediate signal is available.
    signal = frame.signal
    panels = [("As received", signal.samples[::8], 8)]

    signal = MatchedFilterStage(simulator.pulse).apply(signal, context)
    panels.append(("After matched filter", signal.samples[::8], 8))

    signal = CoarseFrequencyCorrection().apply(signal, context)
    panels.append(("After coarse frequency", signal.samples[::8], 8))
    estimated_hz = context.diagnostics["coarse_freq_offset_hz"]

    signal = GardnerTimingRecovery().apply(signal, context)
    panels.append(("After timing recovery", signal.samples, 1))
    estimated_tau = context.diagnostics["timing_offset_samples"]

    signal = AutomaticGainControl().apply(signal, context)
    panels.append(("After gain control", signal.samples, 1))

    signal = DecisionDirectedPLL().apply(signal, context)
    panels.append(("After carrier loop", signal.samples, 1))

    bits, _, _ = SoftDemapper().demap(signal.samples, scheme)
    errors = np.count_nonzero(bits[64:] != frame.bits[64:])
    final_ber = errors / (frame.n_bits - 64)

    figure, axes = plt.subplots(2, 3, figsize=(12.4, 8.6), facecolor=C_SURFACE)

    for axis, (title, samples, _) in zip(axes.ravel(), panels):
        limit = max(1.9, float(np.percentile(np.abs(samples), 99.5)) * 1.25)
        style_axes(axis, limit)

        wrong = np.zeros(samples.size, dtype=bool)
        if title == "After carrier loop":
            decided = scheme.constellation[
                np.argmin(
                    np.abs(samples[:, None] - scheme.constellation[None, :]),
                    axis=1,
                )
            ]
            sent = frame.tx_symbols[: samples.size]
            wrong = ~np.isclose(decided, sent, atol=1e-6)

        axis.scatter(
            samples[~wrong].real, samples[~wrong].imag,
            s=4, alpha=0.3, color=C_OK, linewidths=0, zorder=2,
        )
        if wrong.any():
            axis.scatter(
                samples[wrong].real, samples[wrong].imag,
                s=7, alpha=0.75, color=C_ERR, linewidths=0, zorder=3,
            )
        axis.scatter(
            scheme.constellation.real, scheme.constellation.imag,
            s=40, facecolors="none", edgecolors=C_REF,
            linewidths=1.5, zorder=4,
        )
        axis.set_title(title, fontsize=11, color=C_TEXT, pad=18, loc="left")

    notes = [
        f"{conditions.describe()}",
        "band-limited, noise outside the signal rejected",
        f"estimated {estimated_hz:+.0f} Hz (true {conditions.freq_offset_hz:+.0f} Hz)",
        f"best sampling point {estimated_tau:+.3f} samples — the clock is "
        f"{conditions.timing_offset:+.2f} out, the echoes add the rest",
        "scaled back to the constellation's own size",
        f"upright at last — {errors} bits wrong, BER {final_ber:.1e}",
    ]
    for axis, note in zip(axes.ravel(), notes):
        axis.text(
            0.0, 1.025, note, transform=axis.transAxes,
            fontsize=8, color=C_MUTED, va="bottom",
        )

    figure.suptitle(
        "One damaged 16QAM frame, repaired stage by stage",
        fontsize=13.5, color=C_TEXT, x=0.012, ha="left", y=0.978,
    )
    figure.text(
        0.012, 0.938,
        "Nothing here is told the frequency, the timing or the phase — every "
        "stage works them out from the signal itself. Red marks a symbol the "
        "receiver got wrong.",
        fontsize=9, color=C_MUTED,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.915))
    path = OUTPUT_DIR / "receiver_stages.png"
    figure.savefig(path, dpi=150, facecolor=C_SURFACE)
    plt.close(figure)
    return path


def figure_ber_curves() -> Path:
    """Measured bit error rate against published theory."""
    simulator = SignalSimulator(frame_length=8192, samples_per_symbol=8, seed=5)
    receiver = ClassicalReceiver(pulse=simulator.pulse)

    plan = {
        "BPSK": np.arange(-14.0, 0.1, 2.0),
        "QPSK": np.arange(-11.0, 3.1, 2.0),
        "16QAM": np.arange(-4.0, 10.1, 2.0),
        "64QAM": np.arange(2.0, 16.1, 2.0),
    }

    figure, axis = plt.subplots(figsize=(10.6, 7.2), facecolor=C_SURFACE)
    axis.set_facecolor(C_SURFACE)

    print(f"{'scheme':>7} {'Eb/N0':>7} {'measured':>11} {'theory':>11} {'loss dB':>8}")

    for colour, (name, snr_values) in zip(SERIES, plan.items()):
        scheme = ModulationRegistry.create(name)
        skip = receiver.pilot_symbols * scheme.bits_per_symbol

        ebn0_points, measured_points, theory_points = [], [], []
        for snr_db in snr_values:
            conditions = ChannelConditions(snr_db=float(snr_db))
            errors = total = 0
            # Keep measuring until enough errors have accumulated for the
            # estimate to mean something. A bit error rate computed from
            # three errors is mostly noise.
            for attempt in range(MAX_FRAMES):
                frame = simulator.generate_frame(name, conditions)
                mask = receiver.recover(frame).error_mask(frame.bits)[skip:]
                errors += int(mask.sum())
                total += mask.size
                if attempt >= 5 and errors >= TARGET_ERRORS:
                    break

            ebn0 = simulator.effective_ebn0_db(float(snr_db), name)
            measured = errors / total
            theory = float(scheme.theoretical_ber(ebn0))

            ebn0_points.append(ebn0)
            measured_points.append(max(measured, 1e-7))
            theory_points.append(theory)
            print(
                f"{name:>7} {ebn0:7.2f} {measured:11.3e} {theory:11.3e} "
                f"{implementation_loss_db(scheme, ebn0, measured):8.2f}"
            )

        axis.semilogy(
            ebn0_points, theory_points,
            linestyle="--", linewidth=1.4, color=colour, alpha=0.55, zorder=2,
        )
        axis.semilogy(
            ebn0_points, measured_points,
            marker="o", markersize=5.5, linewidth=2.0, color=colour,
            label=name, zorder=3,
        )
        axis.annotate(
            name,
            xy=(ebn0_points[-1], measured_points[-1]),
            xytext=(6, -2), textcoords="offset points",
            fontsize=9.5, color=colour, va="center", )

    axis.set_xlabel("Energy per bit over noise, Eb/N0 (dB)", fontsize=10, color=C_MUTED)
    axis.set_ylabel("Bit error rate", fontsize=10, color=C_MUTED)
    axis.set_ylim(2e-6, 1.0)
    axis.grid(True, which="both", color=C_GRID, linewidth=0.6)
    axis.tick_params(colors=C_MUTED, labelsize=9)
    for edge in axis.spines.values():
        edge.set_color(C_GRID)

    solid = plt.Line2D([], [], color=C_MUTED, linewidth=2.0, label="measured")
    dashed = plt.Line2D(
        [], [], color=C_MUTED, linewidth=1.4, linestyle="--", label="theory"
    )
    axis.legend(
        handles=[solid, dashed], loc="lower left", fontsize=9,
        frameon=True, facecolor=C_SURFACE, edgecolor=C_GRID, labelcolor=C_MUTED,
    )

    figure.suptitle(
        "The classical receiver against the textbook curves",
        fontsize=13.5, color=C_TEXT, x=0.012, ha="left", y=0.975,
    )
    figure.text(
        0.012, 0.865,
        "Solid lines are measured through the full receive chain; dashed lines are "
        "the published formulas.\nThe small gap between them is implementation "
        "loss — the price of working the frequency and timing out blind.\n"
        "BPSK and QPSK sit on top of each other because they have the same "
        "theoretical error rate: QPSK sends twice the data for the same energy "
        "per bit.",
        fontsize=9, color=C_MUTED,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    path = OUTPUT_DIR / "ber_curves.png"
    figure.savefig(path, dpi=150, facecolor=C_SURFACE)
    plt.close(figure)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Stage-by-stage repair")
    print("-" * 70)
    stages_path = figure_stages()
    print(f"  wrote {stages_path.relative_to(OUTPUT_DIR.parent)}\n")

    print("Bit error rate against theory (this takes a minute)")
    print("-" * 70)
    started = time.time()
    curves_path = figure_ber_curves()
    print(f"\n  wrote {curves_path.relative_to(OUTPUT_DIR.parent)}")
    print(f"  {time.time() - started:.0f} s\n")


if __name__ == "__main__":
    main()
