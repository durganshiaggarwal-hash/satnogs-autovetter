#!/usr/bin/env python3
"""Phase 1 demonstration: constellations and channel impairments.

Run from the project root::

    python3 scripts/demo_phase1.py

Produces two figures in ``outputs/``:

``constellations.png``
    All six modulation schemes at a clean signal-to-noise ratio, showing
    how the layouts differ and why the high-order ones are harder to
    recover.

``impairments.png``
    16QAM under each impairment in turn, which is a preview of what the
    GUI's Signal Lab screen will show interactively.

The script doubles as a worked example of the API, so it is worth
reading alongside the figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to a file; no display needed

import matplotlib.pyplot as plt
import numpy as np

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satrecovery import (  # noqa: E402
    ChannelConditions,
    ModulationRegistry,
    SignalSimulator,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

# Two categorical colours, validated for colour-vision deficiency and for
# contrast against the chart surface.  Received samples are the data;
# ideal points are the reference overlay.
COLOR_RECEIVED = "#2a78d6"
COLOR_IDEAL = "#eb6834"
COLOR_SURFACE = "#fcfcfb"
COLOR_TEXT = "#0b0b0b"
COLOR_MUTED = "#52514e"
COLOR_GRID = "#e4e3df"


def style_axes(axis: plt.Axes, limit: float) -> None:
    """Apply a consistent, recessive style to a constellation panel."""
    axis.set_facecolor(COLOR_SURFACE)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal")
    axis.grid(True, color=COLOR_GRID, linewidth=0.6, zorder=0)
    axis.axhline(0, color=COLOR_GRID, linewidth=0.9, zorder=1)
    axis.axvline(0, color=COLOR_GRID, linewidth=0.9, zorder=1)
    axis.tick_params(colors=COLOR_MUTED, labelsize=7.5, length=3)
    for edge in axis.spines.values():
        edge.set_color(COLOR_GRID)


def plot_constellation(
    axis: plt.Axes,
    received: np.ndarray,
    ideal: np.ndarray,
    title: str,
    subtitle: str,
    limit: float = 1.9,
) -> None:
    """Draw received samples with the ideal constellation on top."""
    style_axes(axis, limit)

    axis.scatter(
        received.real,
        received.imag,
        s=5,
        alpha=0.3,
        color=COLOR_RECEIVED,
        linewidths=0,
        label="Received symbols",
        zorder=2,
    )
    axis.scatter(
        ideal.real,
        ideal.imag,
        s=46,
        facecolors="none",
        edgecolors=COLOR_IDEAL,
        linewidths=1.6,
        label="Ideal points",
        zorder=3,
    )

    # The title sits well clear of the axes so the smaller grey subtitle
    # can occupy the gap between them without colliding.
    axis.set_title(title, fontsize=10.5, color=COLOR_TEXT, pad=20, loc="left")
    axis.text(
        0.0,
        1.022,
        subtitle,
        transform=axis.transAxes,
        fontsize=7.8,
        color=COLOR_MUTED,
        va="bottom",
    )


def figure_constellations() -> Path:
    """All six schemes side by side at a clean signal-to-noise ratio."""
    simulator = SignalSimulator(frame_length=8192, samples_per_symbol=8, seed=1)
    conditions = ChannelConditions(snr_db=12.0)

    names = ModulationRegistry.available()
    figure, axes = plt.subplots(2, 3, figsize=(12.4, 8.4), facecolor=COLOR_SURFACE)

    for axis, name in zip(axes.ravel(), names):
        scheme = ModulationRegistry.create(name)
        frame = simulator.generate_frame(name, conditions)

        # Matched filter, then sample once per symbol -- the same two
        # steps the classical receiver will begin with in Phase 2.
        filtered = simulator.pulse.matched_filter(frame.signal.samples)
        symbols = filtered[:: frame.samples_per_symbol]

        recovered = scheme.demap_symbols(symbols)
        ber = float(np.mean(recovered != frame.bits))
        ebn0 = simulator.effective_ebn0_db(conditions.snr_db, scheme)

        plot_constellation(
            axis,
            symbols,
            scheme.constellation,
            f"{scheme.name}  ·  {scheme.bits_per_symbol} "
            f"bit{'s' if scheme.bits_per_symbol > 1 else ''}/symbol",
            f"Eb/N0 {ebn0:.1f} dB   ·   min. distance {scheme.minimum_distance:.3f}"
            f"   ·   BER {ber:.1e}",
        )

    axes[0, 0].legend(
        loc="upper right",
        fontsize=7.5,
        frameon=True,
        facecolor=COLOR_SURFACE,
        edgecolor=COLOR_GRID,
        labelcolor=COLOR_MUTED,
        markerscale=1.3,
    )

    figure.suptitle(
        "Constellations after matched filtering, channel SNR 12 dB",
        fontsize=13.5,
        color=COLOR_TEXT,
        x=0.012,
        ha="left",
        y=0.978,
    )
    figure.text(
        0.012,
        0.938,
        "More bits per symbol packs the points closer together, so the same "
        "noise causes more errors. That trade-off is what a cognitive radio "
        "has to manage.",
        fontsize=9,
        color=COLOR_MUTED,
    )

    figure.tight_layout(rect=(0, 0, 1, 0.915))
    path = OUTPUT_DIR / "constellations.png"
    figure.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(figure)
    return path


def figure_impairments() -> Path:
    """One scheme under each impairment -- a preview of the Signal Lab screen."""
    simulator = SignalSimulator(frame_length=8192, samples_per_symbol=8, seed=2)
    scheme = ModulationRegistry.create("16QAM")

    cases = [
        ("Ideal channel", ChannelConditions(snr_db=30.0), "nothing added"),
        (
            "Noise only",
            ChannelConditions(snr_db=6.0),
            "points blur outwards, evenly",
        ),
        (
            "Flat fading",
            ChannelConditions(snr_db=18.0, fading="flat_rician", rician_k_db=4.0),
            "whole grid scaled and rotated",
        ),
        (
            "Multipath fading",
            ChannelConditions(
                snr_db=18.0, fading="multipath_rician", rician_k_db=6.0
            ),
            "echoes smear the points",
        ),
        (
            "Frequency offset",
            ChannelConditions(snr_db=18.0, freq_offset_hz=600.0),
            "constellation spins into rings",
        ),
        (
            "Everything at once",
            ChannelConditions(
                snr_db=8.0,
                fading="multipath_rician",
                rician_k_db=5.0,
                freq_offset_hz=300.0,
                phase_offset_rad=0.6,
            ),
            "all of the above",
        ),
    ]

    figure, axes = plt.subplots(2, 3, figsize=(12.4, 8.4), facecolor=COLOR_SURFACE)

    for axis, (title, conditions, note) in zip(axes.ravel(), cases):
        frame = simulator.generate_frame(scheme, conditions)
        filtered = simulator.pulse.matched_filter(frame.signal.samples)
        symbols = filtered[:: frame.samples_per_symbol]

        recovered = scheme.demap_symbols(symbols)
        ber = float(np.mean(recovered != frame.bits))

        plot_constellation(
            axis,
            symbols,
            scheme.constellation,
            f"{title}  ·  BER {ber:.2e}",
            f"{note}   ·   {conditions.describe()}",
            limit=2.3,
        )

    axes[0, 0].legend(
        loc="upper right",
        fontsize=7.5,
        frameon=True,
        facecolor=COLOR_SURFACE,
        edgecolor=COLOR_GRID,
        labelcolor=COLOR_MUTED,
        markerscale=1.3,
    )

    figure.suptitle(
        "16QAM under each channel impairment",
        fontsize=13.5,
        color=COLOR_TEXT,
        x=0.012,
        ha="left",
        y=0.978,
    )
    figure.text(
        0.012,
        0.938,
        "Each impairment distorts the constellation in its own recognisable "
        "way. A frequency offset is the most destructive and the most "
        "correctable -- which is why carrier recovery comes first in Phase 2.",
        fontsize=9,
        color=COLOR_MUTED,
    )

    figure.tight_layout(rect=(0, 0, 1, 0.915))
    path = OUTPUT_DIR / "impairments.png"
    figure.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(figure)
    return path


def print_summary() -> None:
    """Print the registry table and a short bit-error-rate sweep."""
    print("\nRegistered modulation schemes")
    print("-" * 78)
    header = f"{'name':>8}  {'family':>30}  {'bits':>5}  {'min dist':>9}  {'theory':>7}"
    print(header)
    for entry in ModulationRegistry.describe_all():
        print(
            f"{entry['name']:>8}  {entry['family']:>30}  "
            f"{entry['bits_per_symbol']:>5}  {entry['minimum_distance']:>9.4f}  "
            f"{'yes' if entry['has_theory_curve'] else 'no':>7}"
        )

    print("\nMeasured vs theoretical bit error rate (full transmit chain)")
    print("-" * 78)
    simulator = SignalSimulator(frame_length=16384, samples_per_symbol=8, seed=5)
    print(
        f"{'scheme':>8}  {'SNR dB':>7}  {'Eb/N0 dB':>9}  "
        f"{'measured':>10}  {'theory':>10}"
    )
    for name, snr_db in (("BPSK", -11.0), ("QPSK", -8.0), ("16QAM", -1.0)):
        scheme = ModulationRegistry.create(name)
        conditions = ChannelConditions(snr_db=snr_db)

        errors = total = 0
        for _ in range(8):
            frame = simulator.generate_frame(name, conditions)
            filtered = simulator.pulse.matched_filter(frame.signal.samples)
            recovered = scheme.demap_symbols(
                filtered[:: frame.samples_per_symbol]
            )
            errors += int(np.count_nonzero(recovered != frame.bits))
            total += frame.n_bits

        ebn0 = simulator.effective_ebn0_db(snr_db, scheme)
        theory = float(scheme.theoretical_ber(ebn0))
        print(
            f"{name:>8}  {snr_db:>7.1f}  {ebn0:>9.2f}  "
            f"{errors / total:>10.3e}  {theory:>10.3e}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print_summary()
    print("\nWriting figures")
    print("-" * 78)
    for path in (figure_constellations(), figure_impairments()):
        print(f"  {path.relative_to(OUTPUT_DIR.parent)}")
    print()


if __name__ == "__main__":
    main()
