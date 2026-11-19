#!/usr/bin/env python3
"""Second look at RML24_IQdata.h5: are the true bits there, and a small sample.

The first inspection showed every ``Bit_len`` is 0.  Bit error rates need
the transmitted bits, so this script checks whether ``Bit_data`` holds
any bits at all, learns how the frames are ordered, and saves a small
sample (a few MB) of the six modulations this project's receivers handle,
so the Phase 3 loader can be written and tested against real frames.

Run from the project folder::

    python scripts/probe_rml24.py "C:/path/to/RML24_IQdata.h5"

It prints a report and writes ``rml24_sample.npz`` next to this project.
Send both back.  Reading is done a row at a time: the 20 GB file is
never loaded into memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

#: Class order from the dataset authors' own code (RML24_Baseline/Cm-Acc.py).
#: Their list has 22 names; the published file has 21 classes because the
#: last one, SOQPSK_PM, was withdrawn.  This script checks the order
#: rather than trusting it.
CLASS_NAMES = ["BPSK", "QPSK", "8PSK", "16QAM", "32QAM", "64QAM", "GMSK", "OQPSK", "FQPSK",
               "ARTM", "SOQPSK", "FM", "PM", "16APSK", "32APSK", "BPSK_FM", "BPSK_PM",
               "QPSK_FM", "QPSK_PM", "FSK_PM", "FQPSK_PM"]

#: The modulations this project's receivers implement.
SUPPORTED = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "16APSK"]

SAMPLE_SNRS = (20, 10, 0)
FRAMES_PER_CELL = 3


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    import h5py

    path = Path(sys.argv[1]).expanduser()
    if path.is_dir():
        found = sorted(path.glob("*.h5"))
        if not found:
            print(f"No .h5 file in {path}")
            sys.exit(1)
        path = found[0]

    with h5py.File(path, "r") as handle:
        iq, bits = handle["IQ_data"], handle["Bit_data"]
        classes = handle["class"][:]
        snrs = handle["snr"][:]
        rates = handle["symbol_rate"][:]
        bit_len = handle["Bit_len"][:]
        print(f"file: {path.name}   frames: {len(classes):,}")
        print(f"IQ chunks: {iq.chunks}  compression: {iq.compression}")
        print(f"Bit_data chunks: {bits.chunks}  compression: {bits.compression}")
        print(f"Bit_len: min {bit_len.min()}  max {bit_len.max()}")

        # -- ordering ---------------------------------------------------
        print("\n== ordering ==")
        for name, column in (("class", classes), ("snr", snrs), ("symbol_rate", rates)):
            changes = int(np.count_nonzero(np.diff(column)))
            print(f"{name:12s} changes value {changes:,} times; first 12: {column[:12]}")
        keys = np.stack([classes, snrs, rates], axis=1)
        _, counts = np.unique(keys, axis=0, return_counts=True)
        print(f"distinct (class, snr, rate) cells: {len(counts):,};"
              f" frames per cell: min {counts.min()} max {counts.max()}")

        # -- are there bits anywhere? -----------------------------------
        print("\n== Bit_data: any bits stored? ==")
        rng = np.random.default_rng(0)
        probe_rows = np.sort(rng.choice(len(classes), size=200, replace=False))
        nonzero_rows = 0
        for row in probe_rows:
            values = bits[int(row)]
            if np.any(values):
                nonzero_rows += 1
                if nonzero_rows <= 3:
                    last = int(np.flatnonzero(values)[-1])
                    print(f"row {row}: class {classes[row]} rate {rates[row]}: nonzero up to index {last};"
                          f" distinct {np.unique(values)[:8]} first {values[:16]}")
        print(f"{nonzero_rows} of {len(probe_rows)} random rows contain any nonzero value")

        # -- per-class signal statistics at 20 dB -----------------------
        print("\n== one 20 dB frame per class (checks the class order) ==")
        print(f"{'idx':>3} {'name':10s} {'rate':>7} {'power':>7} {'|amp| cv':>8} {'bits?':>5}")
        for index in range(int(classes.max()) + 1):
            rows = np.flatnonzero((classes == index) & (snrs == 20) & (rates == 100_000))
            if rows.size == 0:
                continue
            row = int(rows[0])
            frame = iq[row]
            x = frame[:, 0].astype(np.float64) + 1j * frame[:, 1]
            amp = np.abs(x)
            name = CLASS_NAMES[index] if index < len(CLASS_NAMES) else "?"
            print(f"{index:>3} {name:10s} {rates[row]:>7} {np.mean(amp**2):7.3f}"
                  f" {amp.std() / amp.mean():8.3f} {'yes' if np.any(bits[row]) else 'no':>5}")

        # -- the sample -------------------------------------------------
        print("\n== saving a sample ==")
        picked = []
        for name in SUPPORTED:
            index = CLASS_NAMES.index(name)
            for snr in SAMPLE_SNRS:
                for rate in np.unique(rates):
                    rows = np.flatnonzero((classes == index) & (snrs == snr) & (rates == rate))
                    picked.extend(int(r) for r in rows[:FRAMES_PER_CELL])
        picked.sort()
        sample_iq = np.stack([iq[r] for r in picked])
        sample_bits = np.stack([bits[r] for r in picked])
        out = Path("rml24_sample.npz").resolve()
        np.savez_compressed(
            out, rows=np.array(picked), iq=sample_iq, bits=sample_bits,
            bit_len=bit_len[picked], cls=classes[picked], snr=snrs[picked], rate=rates[picked],
            class_names=np.array(CLASS_NAMES),
        )
        print(f"{len(picked)} frames -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
