"""What a receiver hands back.

:class:`RecoveryResult` is the common currency of the whole project.
Every receiver -- classical, learned or hybrid -- returns one, and every
metric, plot and comparison consumes one.  Fixing the shape of the answer
up front is what lets Phase 6 grade three completely different methods
with a single piece of code.

The class carries more than the recovered bits, because the project's
second question is whether a receiver *knows when it is wrong*.  So
alongside each bit sits a confidence: the receiver's own estimate of the
probability that it got that bit right.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.exceptions import ValidationError


class RecoveryResult:
    """Bits recovered from one frame, with the receiver's own confidence.

    Parameters
    ----------
    bits:
        The recovered bits.
    confidence:
        Probability, per bit, that the bit above is correct.  Lies in
        ``[0.5, 1.0]``: a receiver that reports less than even odds on its
        own decision should have made the other decision.
    symbols:
        The cleaned-up symbols the decisions were made from.  This is what
        the GUI plots, and it is the visible proof that the receiver
        repaired the signal.
    modulation, receiver:
        Names, for labelling results.
    llr:
        Optional log-likelihood ratios, the natural soft output of a
        demapper.  Error-correcting codes want exactly this.
    diagnostics:
        Whatever the receiver learned along the way -- its estimate of the
        frequency offset, the residual error, how many taps the equaliser
        moved.  Not used for grading; used for understanding *why* a
        result came out the way it did.
    """

    __slots__ = (
        "_bits",
        "_confidence",
        "_symbols",
        "_modulation",
        "_receiver",
        "_llr",
        "_diagnostics",
    )

    def __init__(
        self,
        bits: np.ndarray,
        confidence: np.ndarray | None = None,
        symbols: np.ndarray | None = None,
        modulation: str = "",
        receiver: str = "",
        llr: np.ndarray | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        bits = np.array(bits, dtype=np.uint8, copy=True).ravel()
        if bits.size and not np.all((bits == 0) | (bits == 1)):
            raise ValidationError("bits", "<array>", "values must be 0 or 1")
        self._bits = bits

        if confidence is None:
            # A receiver that offers no opinion is recorded as certain.
            # Phase 6 will show this is the worst possible behaviour, and
            # the default makes that visible rather than hiding it.
            confidence = np.ones(bits.size, dtype=float)
        confidence = np.array(confidence, dtype=float, copy=True).ravel()
        if confidence.size != bits.size:
            raise ValidationError(
                "confidence",
                confidence.size,
                f"must have one value per bit ({bits.size})",
            )
        if confidence.size and (
            confidence.min() < 0.0 or confidence.max() > 1.0
        ):
            raise ValidationError(
                "confidence", "<array>", "values must lie in [0, 1]"
            )
        self._confidence = confidence

        self._symbols = (
            None
            if symbols is None
            else np.array(symbols, dtype=np.complex128, copy=True).ravel()
        )
        self._llr = (
            None if llr is None else np.array(llr, dtype=float, copy=True).ravel()
        )
        self._modulation = str(modulation)
        self._receiver = str(receiver)
        self._diagnostics = dict(diagnostics or {})

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------
    @property
    def bits(self) -> np.ndarray:
        """Read-only view of the recovered bits."""
        view = self._bits.view()
        view.flags.writeable = False
        return view

    @property
    def confidence(self) -> np.ndarray:
        """Read-only per-bit probability of being correct."""
        view = self._confidence.view()
        view.flags.writeable = False
        return view

    @property
    def symbols(self) -> np.ndarray | None:
        """Read-only view of the cleaned-up symbols, if the receiver kept them."""
        if self._symbols is None:
            return None
        view = self._symbols.view()
        view.flags.writeable = False
        return view

    @property
    def llr(self) -> np.ndarray | None:
        """Read-only log-likelihood ratios, if the demapper produced them."""
        if self._llr is None:
            return None
        view = self._llr.view()
        view.flags.writeable = False
        return view

    @property
    def modulation(self) -> str:
        """Name of the modulation scheme this was decoded as."""
        return self._modulation

    @property
    def receiver(self) -> str:
        """Name of the receiver that produced this."""
        return self._receiver

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Copy of what the receiver learned on the way."""
        return dict(self._diagnostics)

    @property
    def n_bits(self) -> int:
        """How many bits were recovered."""
        return int(self._bits.size)

    @property
    def mean_confidence(self) -> float:
        """Average self-reported probability of being right."""
        return float(np.mean(self._confidence)) if self._confidence.size else 0.0

    # ------------------------------------------------------------------
    # Grading against the truth
    # ------------------------------------------------------------------
    def error_mask(self, true_bits: np.ndarray) -> np.ndarray:
        """Boolean array marking which recovered bits are wrong.

        Raises if the lengths differ rather than quietly comparing the
        overlap: a length mismatch means a stage dropped or duplicated
        symbols, which is a bug worth finding, not worth averaging over.
        """
        true_bits = np.asarray(true_bits, dtype=np.uint8).ravel()
        if true_bits.size != self._bits.size:
            raise ValidationError(
                "true_bits",
                true_bits.size,
                f"recovered {self._bits.size} bits; cannot compare",
            )
        return self._bits != true_bits

    def bit_errors(self, true_bits: np.ndarray) -> int:
        """Count of wrong bits."""
        return int(np.count_nonzero(self.error_mask(true_bits)))

    def ber(self, true_bits: np.ndarray) -> float:
        """Bit error rate: the project's headline measurement."""
        if self._bits.size == 0:
            return 0.0
        return self.bit_errors(true_bits) / self._bits.size

    def confidence_is_informative(self, true_bits: np.ndarray) -> bool:
        """Is the receiver less confident about the bits it got wrong?

        The weakest possible test of self-knowledge, and a surprising
        number of models fail it.  Phase 5 replaces this with proper
        calibration and risk-coverage measurements; this is here so the
        property can be checked from the first day it exists.
        """
        wrong = self.error_mask(true_bits)
        if not wrong.any() or wrong.all():
            return True  # nothing to separate
        return bool(
            np.mean(self._confidence[wrong]) < np.mean(self._confidence[~wrong])
        )

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------
    def select_symbols(self, keep: np.ndarray, bits_per_symbol: int) -> "RecoveryResult":
        """A copy holding only the symbols where ``keep`` is true.

        Bits, confidences and log-likelihood ratios keep the
        ``bits_per_symbol`` entries of each kept symbol.  A new object
        rather than an edit, because results are immutable.
        """
        keep = np.asarray(keep, dtype=bool).ravel()
        bps = int(bits_per_symbol)
        if bps < 1 or keep.size * bps != self._bits.size:
            raise ValidationError(
                "keep",
                keep.size,
                f"needs one entry per symbol ({self._bits.size} bits / {bps} per symbol)",
            )
        per_bit = np.repeat(keep, bps)
        symbols = None
        if self._symbols is not None and self._symbols.size == keep.size:
            symbols = self._symbols[keep]
        diagnostics = dict(self._diagnostics)
        diagnostics["ungraded_symbols"] = int(keep.size - keep.sum())
        return RecoveryResult(
            bits=self._bits[per_bit],
            confidence=self._confidence[per_bit],
            symbols=symbols,
            modulation=self._modulation,
            receiver=self._receiver,
            llr=None if self._llr is None or self._llr.size != per_bit.size else self._llr[per_bit],
            diagnostics=diagnostics,
        )

    def summary(self, true_bits: np.ndarray | None = None) -> dict[str, Any]:
        """One row of the results table."""
        row: dict[str, Any] = {
            "receiver": self._receiver,
            "modulation": self._modulation,
            "n_bits": self.n_bits,
            "mean_confidence": round(self.mean_confidence, 4),
        }
        if true_bits is not None:
            row["bit_errors"] = self.bit_errors(true_bits)
            row["ber"] = self.ber(true_bits)
        row.update(self._diagnostics)
        return row

    def __len__(self) -> int:
        return self.n_bits

    def __repr__(self) -> str:
        return (
            f"RecoveryResult({self._receiver or '?'}, {self._modulation or '?'}, "
            f"{self.n_bits} bits, mean confidence {self.mean_confidence:.3f})"
        )


__all__ = ["RecoveryResult"]
