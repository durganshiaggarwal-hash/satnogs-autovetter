#!/usr/bin/env python3
"""Phase 3 demonstration: the receivers on real satellite recordings.

Run from the project root with either the full dataset or the sample::

    python scripts/demo_phase3.py "C:/path/to/RML24_IQdata"
    python scripts/demo_phase3.py rml24_sample.npz --quick

What it does, in order:

1. Lists what the archive holds for the six schemes the project handles.
2. Decodes clean recordings into reference labels, and reports how many
   the reference decoder was willing to vouch for -- and why not.
3. Trains the hybrid's networks on *simulated* data, if they are not
   trained already.  They never see a recording during training.
4. Benchmarks classical, hybrid and hybrid-noise on the recordings, and
   on the simulator at the same SNRs, through the same Phase 6 runner.
5. Draws the figures and saves every result to a database the GUI reads.

Outputs go to ``outputs/phase3/``:

``reference_quality.png``
    What a clean recording of each scheme looks like once decoded, and
    which schemes were clean enough to grade receivers on.
``glitches.png``
    The corrupted symbols at 250 kbaud, one in every 92, like clockwork.
``real_vs_simulated.png``
    Bit error rate against SNR, real recordings beside the simulator.
``results.db``
    Every benchmark cell, for the Benchmark tab.
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

from satrecovery.backend.catalog import ReceiverCatalog  # noqa: E402
from satrecovery.backend.models import ModelStore, TrainingSpec  # noqa: E402
from satrecovery.data import RecordedSource, ReferenceDecoder, open_archive  # noqa: E402
from satrecovery.data import reference as reference_module  # noqa: E402
from satrecovery.evaluation import (  # noqa: E402
    BenchmarkRunner,
    ImpairmentProfile,
    ResultStore,
    ScenarioGrid,
    SimulatedSource,
)
from satrecovery.modulation.registry import ModulationRegistry  # noqa: E402

OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "phase3"
GRADED = ["BPSK", "QPSK", "8PSK"]
ALL = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "16APSK"]
RECEIVERS = ["classical", "hybrid", "hybrid-noise"]
COLOURS = {"classical": "#2a78d6", "hybrid": "#eb6834", "hybrid-noise": "#1baf7a"}
#: 500 kbaud recordings are noisier (about 10 dB on the simulator's scale,
#: against 13-17 for the others), so they would drop out of the curve
#: above 6 dB and change what each point is an average of.  Two rates,
#: present at every SNR, keep every point comparable.
RATES = (100_000, 250_000)


def heading(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", help="RML24_IQdata.h5, the folder holding it, or a sample .npz")
    parser.add_argument("--quick", action="store_true", help="fewer frames and quick training")
    parser.add_argument("--frames", type=int, default=None, help="frames per benchmark cell")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = args.frames or (8 if args.quick else 30)
    started = time.time()

    with open_archive(args.archive) as archive:
        heading(f"1. What {archive.name} holds ({len(archive):,} recordings)")
        for name, info in archive.summary().items():
            rates = ", ".join(f"{r // 1000} kbaud" for r in info["symbol_rates"])
            print(f"  {name:7s} {info['recordings']:>9,} recordings at {rates}")

        source = RecordedSource(archive, symbol_rates=RATES)
        decode_limit = 40 if args.quick else 120

        heading("2. Reference labels: which recordings can be vouched for")
        print(f"  Decoding up to {decode_limit} clean recordings (labelled 20/18/16 dB) per scheme.")
        acceptance = {}
        for name in ALL:
            if name not in archive.modulations():
                continue
            report = source.acceptance(name, limit=decode_limit)
            acceptance[name] = report
            evm = f"{report['median_evm_percent']:.1f}%" if report["median_evm_percent"] else "-"
            line = f"  {name:7s} {report['accepted']:>4}/{report['decoded']:<4} accepted   median EVM {evm}"
            if report["accepted"] == 0 and report["example_rejection"]:
                line += f"\n          e.g. {report['example_rejection']}"
            print(line)
        graded = [m for m in GRADED if acceptance.get(m, {}).get("accepted", 0) > 0]

        figure_reference(archive, source)
        glitch_spacing = figure_glitches(archive, source)

        heading("3. The hybrid's networks, trained on simulated data only")
        store = ModelStore(OUTPUT / "models")
        spec = TrainingSpec(frames_each=6, epochs=60) if args.quick else TrainingSpec()
        for name in graded:
            if store.has(name):
                print(f"  {name}: already trained")
                continue
            t0 = time.time()
            record = store.train(name, spec)
            print(f"  {name}: trained on {record['rows']:,} simulated symbols in {time.time() - t0:.0f} s")

        simulated = SimulatedSource(seed=0)
        catalog = ReceiverCatalog.standard(simulated.simulator.pulse, store)
        snrs = [-4, -2, 0, 2, 4, 6, 8, 10]

        heading(f"4. Benchmark: {frames} frames per cell, same runner for both sources")
        results = {}
        for label, src, profile in (
            ("real", source, ImpairmentProfile.recorded()),
            ("simulated", simulated, ImpairmentProfile.clean()),
        ):
            results[label] = run_per_modulation(src, catalog, graded, snrs, profile, frames)
            print(f"  {label}: done")

        store_path = OUTPUT / "results.db"
        with ResultStore(store_path) as db:
            for label, by_modulation in results.items():
                for modulation, result in by_modulation.items():
                    db.save(result, f"Phase 3 {label} {modulation}",
                            notes="RML24 recordings with added noise" if label == "real" else "simulator, clean")
        figure_curves(results, graded, snrs)

        heading("5. Findings")
        for modulation in graded:
            for receiver in RECEIVERS:
                real = cells_by_snr(results["real"][modulation], receiver)
                sim = cells_by_snr(results["simulated"][modulation], receiver)
                at = 4.0
                if at in real and at in sim:
                    print(f"  {modulation:5s} {receiver:13s} at {at:+.0f} dB:"
                          f"  real {real[at].ber:.2e}   simulated {sim[at].ber:.2e}")
        if glitch_spacing:
            print(f"\n  Corrupted symbols at 250 kbaud repeat every {glitch_spacing} symbols.")
        print(f"\n  Figures and results in {OUTPUT}   ({time.time() - started:.0f} s in all)")


def run_per_modulation(source, catalog, modulations, snrs, profile, frames):
    """One benchmark per modulation, because the hybrid is one network per scheme."""
    out = {}
    for modulation in modulations:
        receivers = {name: catalog.build(name, modulation) for name in RECEIVERS}
        runner = BenchmarkRunner(source, receivers, frames_per_scenario=frames)
        out[modulation] = runner.run(ScenarioGrid.product([modulation], snrs, [profile]))
    return out


def cells_by_snr(result, receiver):
    return {c.scenario.snr_db: c for c in result.cells if c.receiver == receiver}


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def decoded_points(source, row):
    """The decoder's corrected points for one recording (for plotting)."""
    captured = {}
    original = reference_module.ReferenceDecoder._track

    def capture(self, s, scheme, power):
        points, gains = original(self, s, scheme, power)
        captured["points"] = points
        return points, gains

    reference_module.ReferenceDecoder._track = capture
    try:
        reference = ReferenceDecoder().decode(source.archive.recording(row))
    finally:
        reference_module.ReferenceDecoder._track = original
    return reference, captured["points"]


def figure_reference(archive, source):
    names = [m for m in ALL if m in archive.modulations()]
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.6))
    for ax, name in zip(np.atleast_1d(axes), names):
        rows = archive.select(name, 20, 100_000)
        if rows.size == 0:
            rows = archive.select(name, 20)
        reference, points = decoded_points(source, int(rows[0]))
        scheme = ModulationRegistry.create(name)
        guard = source.decoder.guard_symbols
        edge = np.zeros(reference.n_symbols, dtype=bool)
        edge[:guard] = edge[reference.n_symbols - guard:] = True
        good = reference.graded
        unreadable = ~good & ~edge
        ax.scatter(points[edge].real, points[edge].imag, s=6, color="#a3969f", alpha=0.5, lw=0)
        ax.scatter(points[good].real, points[good].imag, s=5, color="#2a78d6", alpha=0.6, lw=0)
        ax.scatter(points[unreadable].real, points[unreadable].imag, s=14, color="#d03b3b", marker="x", lw=1)
        ax.scatter(scheme.constellation.real, scheme.constellation.imag, s=18, facecolor="none",
                   edgecolor="#3a2f3d", lw=0.8)
        verdict = "graded" if reference.accepted else "rejected"
        ax.set_title(f"{name}\nEVM {reference.evm_percent:.1f}%  -  {verdict}", fontsize=9,
                     color="#3a2f3d" if reference.accepted else "#d03b3b")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        lim = 1.6 * np.max(np.abs(scheme.constellation))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    fig.suptitle("One clean (20 dB) recording of each scheme, decoded by the reference decoder.  "
                 "Blue: graded.  Red crosses: unreadable, never graded.  Grey: frame edges, never graded.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUTPUT / "reference_quality.png", dpi=150)
    plt.close(fig)


def figure_glitches(archive, source):
    """Where the unreadable symbols fall, for 250 kbaud recordings."""
    positions = []
    for name in GRADED:
        for row in archive.select(name, 20, 250_000)[:6]:
            reference = source.reference(int(row))
            inner = np.zeros(reference.n_symbols, dtype=bool)
            guard = source.decoder.guard_symbols
            inner[guard : reference.n_symbols - guard] = True
            positions.append(np.flatnonzero(inner & ~reference.graded))
    gaps = np.concatenate([np.diff(p) for p in positions if p.size > 1]) if positions else np.array([])
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for i, p in enumerate(positions):
        ax.scatter(p, np.full(p.size, i), color="#d03b3b", marker="x", s=24)
    ax.set_xlabel("symbol number within the recording")
    ax.set_ylabel("recording")
    spacing = None
    if gaps.size:
        values, counts = np.unique(gaps, return_counts=True)
        spacing = int(values[np.argmax(counts)])
        ax.set_title(f"Unreadable symbols at 250 kbaud: the commonest gap is {spacing} symbols", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT / "glitches.png", dpi=150)
    plt.close(fig)
    return spacing


def figure_curves(results, modulations, snrs):
    fig, axes = plt.subplots(1, len(modulations), figsize=(4.4 * len(modulations), 3.9), sharey=True)
    for ax, modulation in zip(np.atleast_1d(axes), modulations):
        for receiver in RECEIVERS:
            for label, style in (("real", "-"), ("simulated", "--")):
                cells = cells_by_snr(results[label][modulation], receiver)
                xs = [s for s in snrs if s in cells]
                ys = [max(cells[s].ber, 1e-5) for s in xs]
                lows = [max(cells[s].ber_interval()[0], 1e-5) for s in xs]
                highs = [max(cells[s].ber_interval()[1], 1e-5) for s in xs]
                ax.semilogy(xs, ys, style, color=COLOURS[receiver], lw=1.6 if label == "real" else 1.0,
                            marker="o" if label == "real" else None, ms=3,
                            label=f"{receiver} ({label})")
                if label == "real":
                    ax.fill_between(xs, lows, highs, color=COLOURS[receiver], alpha=0.12, lw=0)
        ax.set_title(modulation)
        ax.set_xlabel("SNR (dB, simulator's scale)")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_ylim(1e-5, 0.6)
    np.atleast_1d(axes)[0].set_ylabel("bit error rate")
    np.atleast_1d(axes)[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Solid: real RML24 recordings with added noise.  Dashed: simulator.  "
                 "Bands: 95% intervals.  Points at 1e-5 had no errors.\n"
                 "Hybrid and hybrid-noise make identical decisions, so one line hides the other "
                 "(they differ only in confidence).", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUTPUT / "real_vs_simulated.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
