"""What is known about RML24's layout, in one place.

Nothing here was guessed.  Each fact was measured on the downloaded
``RML24_IQdata.h5`` with ``scripts/inspect_rml24.py`` and
``scripts/probe_rml24.py``, or read from the dataset authors' own code,
and the comment beside it says which.

* 1,323,000 recordings of 2048 complex samples, stored as float32 pairs.
* Labels per recording: modulation class (0-20), SNR (-20 to 20 dB in
  steps of 2) and symbol rate.
* **No transmitted bits.**  ``Bit_data`` and ``Bit_len`` exist, but
  every ``Bit_len`` is 0 and 200 random rows of ``Bit_data`` were all
  zeros.  This single fact shapes all of Phase 3: see
  :mod:`satrecovery.data.reference` for how bits are recovered anyway.
* Sample rate 1 MHz (dataset page).  The six modulations this project
  handles were recorded at 100, 250 and 500 kbaud: 10, 4 and 2 samples
  per symbol.
* Pulse shape: root-raised-cosine with roll-off 0.35.  Measured: the
  averaged spectrum of the 20 dB recordings falls by 0.5, 1.4, 3.0, 5.5
  and 9.3 dB at 0.4, 0.45, 0.5, 0.55 and 0.6 of the symbol rate, which
  is the raised-cosine curve for 0.35 to within a few tenths of a
  decibel -- and the same filter our simulator uses.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ..core.exceptions import ValidationError

#: Class order, from the authors' ``RML24_Baseline/Cm-Acc.py``.  Their
#: list has 22 names; the published file has 21 classes because the last,
#: SOQPSK_PM, was withdrawn ("SOQPSK-PM may have some issues").  Checked
#: against the data: the constant-envelope schemes (GMSK, SOQPSK, ARTM,
#: FM) are exactly the classes whose amplitude barely varies.
CLASS_NAMES: Final[tuple[str, ...]] = (
    "BPSK", "QPSK", "8PSK", "16QAM", "32QAM", "64QAM", "GMSK", "OQPSK",
    "FQPSK", "ARTM", "SOQPSK", "FM", "PM", "16APSK", "32APSK", "BPSK_FM",
    "BPSK_PM", "QPSK_FM", "QPSK_PM", "FSK_PM", "FQPSK_PM",
)

#: The schemes this project's receivers implement, by RML24 class number.
#: The other fifteen (frequency and phase modulation, continuous-phase
#: schemes, composites) need different receivers altogether.
SUPPORTED: Final[dict[str, int]] = {
    "BPSK": 0,
    "QPSK": 1,
    "8PSK": 2,
    "16QAM": 3,
    "64QAM": 5,
    "16APSK": 13,
}

SAMPLE_RATE_HZ: Final[float] = 1_000_000.0
ROLLOFF: Final[float] = 0.35
SAMPLES_PER_RECORDING: Final[int] = 2048
SNR_LABELS_DB: Final[tuple[int, ...]] = tuple(range(-20, 21, 2))


def class_index(modulation: str) -> int:
    """RML24's class number for one of the supported schemes."""
    try:
        return SUPPORTED[str(modulation).upper()]
    except KeyError:
        raise ValidationError(
            "modulation",
            modulation,
            f"not one of the schemes this project handles: {', '.join(SUPPORTED)}",
        ) from None


def class_name(index: int) -> str:
    """The dataset's name for a class number."""
    index = int(index)
    if not 0 <= index < len(CLASS_NAMES):
        raise ValidationError("class", index, f"must be 0-{len(CLASS_NAMES) - 1}")
    return CLASS_NAMES[index]


def samples_per_symbol(symbol_rate: float) -> int:
    """Samples per symbol at RML24's sample rate.

    Refuses rates that do not divide the sample rate exactly: a
    fractional oversampling factor would need a resampler first, and
    rounding it silently would put every symbol in the wrong place.
    """
    symbol_rate = float(symbol_rate)
    if not np.isfinite(symbol_rate) or symbol_rate <= 0:
        raise ValidationError("symbol_rate", symbol_rate, "must be a positive number of symbols per second")
    ratio = SAMPLE_RATE_HZ / symbol_rate
    if ratio < 2 or abs(ratio - round(ratio)) > 1e-9:
        raise ValidationError(
            "symbol_rate",
            symbol_rate,
            "must divide 1 MHz into a whole number (at least 2) of samples per symbol",
        )
    return int(round(ratio))


__all__ = [
    "CLASS_NAMES",
    "SUPPORTED",
    "SAMPLE_RATE_HZ",
    "ROLLOFF",
    "SAMPLES_PER_RECORDING",
    "SNR_LABELS_DB",
    "class_index",
    "class_name",
    "samples_per_symbol",
]
