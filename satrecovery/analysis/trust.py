"""Can this answer be trusted?  A warning when the AI is guessing.

The AI always names a type, even when it is guessing, or when the signal
is unlike anything it learned from.  :class:`TrustCheck` looks at the
answer and at the signal itself, and gives one of three verdicts:

``confident``   a clear winner, and a signal of the kind the AI knows
``unsure``      the top answer is weak, or close to the second, or the
                signal is very noisy -- treat the answer with caution
``unfamiliar``  the signal is mostly noise, or its width does not fit
                the chosen speed -- the AI was not built for this

The checks are deliberately simple, so each warning can be explained in
one sentence:

* **How sure, and how far ahead.**  From the AI's own probabilities.
* **How noisy.**  The noise level is read from the part of the spectrum
  the signal does not use, and the signal's strength from the part it
  does.  The answer is given on the same scale as RML24's labels
  (signal-to-noise over the whole 1 MHz band): on the simulator it lands
  within half a dB of the true value, and on two real RML24 recordings
  labelled +18 and +20 dB it reads about +17.
* **How wide** (only on a clearly strong signal).  A signal sent at a given speed fills a known width of
  the spectrum (about 0.47 x the speed holds 90% of its power, measured
  on the simulator and on real RML24 recordings alike).  Much wider or
  narrower means the speed setting is wrong, or the file is not shaped
  like RML24.
* **How sure the receiver is** of the bits it read, when that is known.

It cannot catch every mistake: the AI is sometimes wrong while sure of
itself.  The "How good is the AI?" tab measures how many of its mistakes
the warning catches, on recordings it never trained on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data import layout

#: Below this top probability the AI is "unsure".
MIN_CONFIDENCE = 0.60
#: ... and when the runner-up is closer than this.
MIN_LEAD = 0.25
#: Below this estimated SNR (dB, RML24's scale) the signal is "mostly noise".
NOISE_FLOOR_DB = -5.0
#: Below this it is "very noisy" (the AI is right about half the time there).
NOISY_DB = 0.0
#: 90% of a signal's power lies within this many times its speed of its centre.
WIDTH_PER_BAUD = 0.47
#: A measured width outside these multiples of the expected one does not fit the speed.
WIDTH_RANGE = (0.65, 1.5)
#: The width is read only when at least this share of the power is signal.
CLEAR_SIGNAL = 0.85
#: Below this share of bits the receiver is sure of, it is "unsure" too.
MIN_TRUSTED = 0.5

_SEGMENT = 256


class TrustVerdict:
    """The outcome of a check: a level, a headline, and the reasons."""

    LEVELS = ("confident", "unsure", "unfamiliar")

    def __init__(self, level: str, title: str, reasons: list[str], snr_db: float | None) -> None:
        if level not in self.LEVELS:
            raise ValueError(f"level must be one of {self.LEVELS}")
        self._level, self._title, self._reasons, self._snr_db = level, title, list(reasons), snr_db

    @property
    def level(self) -> str:
        return self._level

    @property
    def warning(self) -> bool:
        return self._level != "confident"

    @property
    def reasons(self) -> list[str]:
        return list(self._reasons)

    @property
    def snr_db(self) -> float | None:
        return self._snr_db

    def as_dict(self) -> dict[str, Any]:
        return {"level": self._level, "title": self._title, "reasons": list(self._reasons),
                "snr_db": None if self._snr_db is None else round(float(self._snr_db), 1)}

    def __repr__(self) -> str:
        return f"TrustVerdict({self._level}: {self._title})"


class TrustCheck:
    """Looks at an answer and its signal, and says how far to trust it."""

    # -- measurements ------------------------------------------------------------
    @staticmethod
    def _spectrum(samples: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(samples, dtype=np.complex128).ravel()
        x = x - x.mean()
        n = (x.size // _SEGMENT) * _SEGMENT
        if n < 2 * _SEGMENT:
            return np.zeros(0), np.zeros(0)
        blocks = x[:n].reshape(-1, _SEGMENT) * np.hanning(_SEGMENT)
        psd = np.mean(np.abs(np.fft.fftshift(np.fft.fft(blocks, axis=1), axes=1)) ** 2, axis=0)
        freqs = np.fft.fftshift(np.fft.fftfreq(_SEGMENT, 1.0 / sample_rate))
        return psd, freqs

    @classmethod
    def estimate_snr_db(cls, samples: np.ndarray, symbol_rate: float,
                        sample_rate: float = layout.SAMPLE_RATE_HZ) -> float | None:
        """Signal-to-noise over the whole band, on RML24's scale; None if it cannot be measured."""
        psd, freqs = cls._spectrum(samples, sample_rate)
        if psd.size == 0:
            return None
        edge = (1 + layout.ROLLOFF) * float(symbol_rate) / 2
        inside = np.abs(freqs) <= edge * 1.05
        outside = np.abs(freqs) >= edge * 1.2
        if outside.sum() < 4:
            return None
        noise_per_bin = float(np.median(psd[outside]))
        if noise_per_bin <= 0:
            return None
        signal = float(psd[inside].sum()) - noise_per_bin * int(inside.sum())
        return float(10 * np.log10(max(signal, 1e-12) / (noise_per_bin * psd.size)))

    @classmethod
    def width_ratio(cls, samples: np.ndarray, symbol_rate: float,
                    sample_rate: float = layout.SAMPLE_RATE_HZ) -> float | None:
        """Measured signal width / the width this speed should have; None if the signal is too weak to tell."""
        psd, freqs = cls._spectrum(samples, sample_rate)
        if psd.size == 0:
            return None
        noise = float(np.percentile(psd, 15))
        excess = np.maximum(psd - noise, 0.0)
        # Read only on a clearly strong signal: measured on the simulator, below
        # about 85% signal the noise itself makes the signal look wider.
        if excess.sum() < CLEAR_SIGNAL * psd.sum():
            return None
        centre = float(np.sum(freqs * excess) / excess.sum())
        distance = np.abs(freqs - centre)
        order = np.argsort(distance)
        share = np.cumsum(excess[order]) / excess.sum()
        width = float(distance[order][min(np.searchsorted(share, 0.9), share.size - 1)])
        return width / (WIDTH_PER_BAUD * float(symbol_rate))

    # -- the verdict ---------------------------------------------------------------
    @classmethod
    def assess(cls, probabilities: list[dict[str, Any]] | dict[str, float], samples: np.ndarray,
               symbol_rate: float, trusted_fraction: float | None = None) -> TrustVerdict:
        """How far to trust the AI's answer for this signal."""
        if isinstance(probabilities, dict):
            ranked = sorted(probabilities.items(), key=lambda kv: -kv[1])
        else:
            ranked = [(row["name"], float(row["p"])) for row in probabilities]
        (top, p1), (second, p2) = ranked[0], ranked[1] if len(ranked) > 1 else ("", 0.0)
        unsure: list[str] = []
        unfamiliar: list[str] = []

        snr = cls.estimate_snr_db(samples, symbol_rate)
        if snr is not None and snr < NOISE_FLOOR_DB:
            unfamiliar.append(f"The signal is mostly noise (about {snr:+.1f} dB). Below about {NOISE_FLOOR_DB:+.0f} dB "
                              "no classifier can tell the type reliably.")
        elif snr is not None and snr < NOISY_DB:
            unsure.append(f"The signal is very noisy (about {snr:+.1f} dB); here the AI is right only about half the time.")

        ratio = cls.width_ratio(samples, symbol_rate)
        if ratio is not None and not WIDTH_RANGE[0] <= ratio <= WIDTH_RANGE[1]:
            how = f"about {ratio:.1f}x wider" if ratio > 1 else f"about {1 / ratio:.1f}x narrower"
            unfamiliar.append(f"The signal is {how} than a {symbol_rate / 1000:.0f} kbaud signal should be. "
                              "Check the speed, or the signal is not shaped like RML24.")

        if p1 < MIN_CONFIDENCE:
            unsure.append(f"The AI is only {p1:.0%} sure.")
        elif p1 - p2 < MIN_LEAD:
            unsure.append(f"Two answers are close: {top} {p1:.0%} vs {second} {p2:.0%}.")
        if trusted_fraction is not None and trusted_fraction < MIN_TRUSTED:
            unsure.append(f"The receiver is sure of only {trusted_fraction:.0%} of the bits.")

        quality = f" Estimated signal quality about {snr:+.0f} dB." if snr is not None else ""
        if unfamiliar:
            return TrustVerdict("unfamiliar", "Unfamiliar signal: do not rely on this answer.",
                                unfamiliar + unsure, snr)
        if unsure:
            return TrustVerdict("unsure", "Unsure: treat this answer with caution.", unsure, snr)
        return TrustVerdict("confident", f"Confident: {p1:.0%} sure, well ahead of the next guess.{quality}", [], snr)


__all__ = ["TrustCheck", "TrustVerdict"]
