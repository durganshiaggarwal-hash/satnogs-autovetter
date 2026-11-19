#!/usr/bin/env python3
"""How often can a real RML24 recording be checked bit by bit?

RML24 stores no bits, so the Phase 3 reference decoder works them out and
only trusts the answer when almost no symbol is in doubt.  This script
measures, on your copy of the dataset, what fraction of recordings pass
that test for each checkable type and signal quality.  Nothing is changed.

Run from the project folder::

    python scripts/check_rate.py "C:/path/to/RML24_IQdata"

Takes a minute or two.  Prints a table; send a screenshot of it back.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satrecovery.data.archive import RML24File
from satrecovery.data.reference import ReferenceDecoder

TYPES = ("BPSK", "QPSK", "8PSK")
SNRS = (6, 8, 10, 12, 14, 16, 18, 20)
PER_CELL = 30


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    decoder = ReferenceDecoder()
    with RML24File(sys.argv[1]) as archive:
        for rate in archive.symbol_rates("8PSK"):
            print(f"\n=== {rate // 1000} kbaud: % of recordings that can be checked "
                  f"(out of {PER_CELL} each), and typical EVM ===")
            print(f"{'type':6s}" + "".join(f"{s:>+11d}dB" for s in SNRS))
            for name in TYPES:
                cells = []
                for snr in SNRS:
                    rows = archive.select(name, snr, rate)
                    if rows.size == 0:
                        cells.append(f"{'-':>13s}")
                        continue
                    picked = rows[np.linspace(0, rows.size - 1, PER_CELL).astype(int)]
                    refs = [decoder.decode(r) for r in archive.recordings(picked)]
                    passed = sum(r.accepted for r in refs)
                    evm = [r.evm_percent for r in refs if np.isfinite(r.evm_percent)]
                    evm_text = f"{np.median(evm):.0f}%" if evm else "-"
                    cells.append(f"{100 * passed / len(refs):>6.0f}% ({evm_text:>4s})")
                print(f"{name:6s}" + "".join(cells), flush=True)


if __name__ == "__main__":
    main()
