#!/usr/bin/env python3
"""Report what is actually inside a downloaded copy of RML24.

The dataset's published pages do not document its file layout, so the
Phase 3 loader is written against what this script finds rather than
against a guess.

Run it on the folder you unzipped the dataset into::

    python scripts/inspect_rml24.py "D:/datasets/RML24"

Paste the output back.  It only reads structure and a few sample values;
it never loads a whole 20 GB file into memory if it can avoid it.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

#: Pickle has no way to read part of a file, so files above this size
#: are listed but not opened.  The first small one is enough to learn
#: the layout.
PICKLE_LIMIT_BYTES = 2_000_000_000


def human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def describe(value, indent: str = "    ", depth: int = 0) -> None:
    """Print the shape of an object without printing all of it."""
    pad = indent * (depth + 1)
    if depth > 4:
        print(f"{pad}...")
        return
    if isinstance(value, np.ndarray):
        sample = value.ravel()[:6]
        print(f"{pad}ndarray shape={value.shape} dtype={value.dtype} first={sample}")
    elif isinstance(value, dict):
        print(f"{pad}dict with {len(value)} keys")
        for i, (key, item) in enumerate(value.items()):
            if i >= 8:
                print(f"{pad}  ... {len(value) - 8} more keys")
                break
            print(f"{pad}  key {key!r}:")
            describe(item, indent, depth + 1)
    elif isinstance(value, (list, tuple)):
        print(f"{pad}{type(value).__name__} of length {len(value)}")
        if value:
            print(f"{pad}  first element:")
            describe(value[0], indent, depth + 1)
    else:
        text = repr(value)
        print(f"{pad}{type(value).__name__}: {text[:120]}")


def inspect_h5(path: Path) -> None:
    try:
        import h5py
    except ImportError:
        print("    (skipped: run  pip install h5py  and try again)")
        return

    with h5py.File(path, "r") as handle:
        def visit(name, item):
            if isinstance(item, h5py.Dataset):
                count = min(item.shape[0], 3) if item.shape else 0
                sample = item[:count] if count else item[()]
                sample = np.asarray(sample).ravel()[:6]
                print(f"    dataset {name!r}: shape={item.shape} dtype={item.dtype}")
                print(f"        first values: {sample}")
                if item.ndim == 1 and item.shape[0] < 50_000_000:
                    values = np.unique(item[:])
                    if values.size <= 64:
                        print(f"        distinct values ({values.size}): {values}")
            for key, value in item.attrs.items():
                print(f"    attribute {name!r}.{key} = {value!r}"[:160])

        for key, value in handle.attrs.items():
            print(f"    file attribute {key} = {value!r}"[:160])
        handle.visititems(visit)


def inspect_pickle(path: Path) -> None:
    if path.stat().st_size > PICKLE_LIMIT_BYTES:
        print("    (too large to open safely; listing only)")
        return
    with path.open("rb") as handle:
        try:
            value = pickle.load(handle, encoding="latin1")
        except TypeError:
            value = pickle.load(handle)
    describe(value)


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser()
    if not root.exists():
        print(f"Not found: {root}")
        sys.exit(1)

    files = sorted(p for p in root.rglob("*") if p.is_file())
    print(f"{len(files)} files under {root}\n")
    for path in files[:60]:
        print(f"  {human(path.stat().st_size):>10}  {path.relative_to(root)}")
    if len(files) > 60:
        print(f"  ... and {len(files) - 60} more")

    opened_pickle = False
    for path in files:
        suffix = path.suffix.lower()
        if suffix in (".h5", ".hdf5", ".mat"):
            print(f"\n=== {path.name} ===")
            if suffix == ".mat":
                print("    (MATLAB file; send the name and size and I will add a reader)")
            else:
                inspect_h5(path)
        elif suffix in (".pkl", ".pickle") and not opened_pickle:
            print(f"\n=== {path.name} (first pickle; others assumed the same) ===")
            inspect_pickle(path)
            opened_pickle = path.stat().st_size <= PICKLE_LIMIT_BYTES


if __name__ == "__main__":
    main()
