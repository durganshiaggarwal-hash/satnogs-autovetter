"""The contract a confidence estimator signs.

Phase 2 and Phase 4 both produced receivers that emit a confidence, but
they emit it as a side effect of how they happen to work: the soft
demapper's confidence falls out of its log-likelihood ratios, the
network's falls out of its final layer.  Neither can be swapped for the
other, and neither can be applied to a receiver that does not produce
one.

This module separates the two ideas.  A *receiver* decides bits.  A
*confidence estimator* judges those decisions.  Once they are separate,
any estimator can be applied to any receiver's output and the question
"which way of knowing you are wrong works best?" becomes a thing that
can actually be measured, rather than a property welded to each
receiver.

DESIGN NOTE: ABSTRACTION VIA THE TEMPLATE METHOD PATTERN
----------------------------------------------------------
:class:`ConfidenceEstimator` defines the public method
:meth:`~ConfidenceEstimator.estimate` **once** and marks a smaller
private method abstract.  The public method does the work every
estimator needs and none of them should be trusted to repeat: resolve
the modulation, reject a result that is missing what the estimator
needs, call the subclass, then check that the subclass returned one
probability per bit and that each one is really a probability.

A subclass therefore cannot get the housekeeping wrong, because it never
writes any.  It fills in one hole.  That is the Template Method pattern,
and it is the reason an estimator in this project is about fifteen lines
long.

DESIGN NOTE: A WIDER CONTRACT THAN THE DEMAPPER'S
--------------------------------------------------
A :class:`~satrecovery.receivers.demapper.Demapper` must report
confidence in ``[0.5, 1]``: it chose the more likely bit by definition,
so it cannot coherently claim worse than even odds on its own decision.

An estimator is not the thing that made the decision, so it is allowed
the full range ``[0, 1]``.  A value below 0.5 means something no
demapper can say: *I think that bit is wrong.*  Ensembles and dropout
sampling produce exactly that when they disagree with the receiver, and
being able to express it is why they can rank a frame's bits better than
the receiver's own numbers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..core.exceptions import ValidationError
from ..modulation.base import Modulation
from ..receivers.result import RecoveryResult


class ConfidenceEstimator(ABC):
    """Judges how likely each recovered bit is to be correct.

    Subclasses implement :meth:`_scores`.  Everything else -- argument
    checking, resolving the modulation, validating the returned array --
    happens here, once.
    """

    def __init__(self, name: str | None = None) -> None:
        self._name = str(name) if name else type(self).__name__

    @property
    def name(self) -> str:
        """Short name, used to label rows in the results table."""
        return self._name

    # ------------------------------------------------------------------
    # The template method: fixed here, the same for every estimator
    # ------------------------------------------------------------------
    def estimate(
        self,
        result: RecoveryResult,
        modulation: Modulation | None = None,
    ) -> np.ndarray:
        """Probability, per bit, that the receiver got that bit right.

        Parameters
        ----------
        result:
            What a receiver produced.  Different estimators read
            different parts of it -- the log-likelihood ratios, the
            cleaned-up symbols -- and each says so clearly if the part
            it needs is missing.
        modulation:
            The scheme the frame was decoded as.  Resolved from the
            result's own label when not given, so ordinary use is
            ``estimator.estimate(result)``.

        Returns
        -------
        np.ndarray
            One probability per bit, in ``[0, 1]``.
        """
        if not isinstance(result, RecoveryResult):
            raise ValidationError(
                "result", type(result).__name__, "expected a RecoveryResult"
            )

        modulation = self._resolve_modulation(result, modulation)
        scores = np.asarray(self._scores(result, modulation), dtype=float).ravel()

        if scores.size != result.n_bits:
            raise ValidationError(
                f"{self._name}._scores",
                scores.size,
                f"must return one value per bit ({result.n_bits})",
            )
        if not np.all(np.isfinite(scores)):
            raise ValidationError(
                f"{self._name}._scores", "<array>", "produced a non-finite value"
            )
        return np.clip(scores, 0.0, 1.0)

    # ------------------------------------------------------------------
    # What subclasses fill in
    # ------------------------------------------------------------------
    @abstractmethod
    def _scores(
        self, result: RecoveryResult, modulation: Modulation
    ) -> np.ndarray:
        """Produce one score per bit. Need not be clipped or checked."""

    # ------------------------------------------------------------------
    # Shared helpers, available to every subclass
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_modulation(
        result: RecoveryResult, modulation: Modulation | None
    ) -> Modulation:
        """Use the modulation given, or look up the one the result names."""
        if isinstance(modulation, Modulation):
            return modulation
        if modulation is not None:
            raise ValidationError(
                "modulation", type(modulation).__name__, "expected a Modulation"
            )
        if not result.modulation:
            raise ValidationError(
                "modulation",
                None,
                "the result does not name a modulation; pass one explicitly",
            )
        from ..modulation.registry import get_modulation

        return get_modulation(result.modulation)

    @staticmethod
    def _require_llr(result: RecoveryResult, who: str) -> np.ndarray:
        """Fetch log-likelihood ratios or say plainly that there are none."""
        if result.llr is None:
            raise ValidationError(
                "result.llr",
                None,
                f"{who} needs log-likelihood ratios; this result came from a "
                f"receiver that does not produce them (try a soft demapper)",
            )
        return np.asarray(result.llr, dtype=float)

    @staticmethod
    def _require_symbols(result: RecoveryResult, who: str) -> np.ndarray:
        """Fetch the recovered symbols or say plainly that there are none."""
        if result.symbols is None:
            raise ValidationError(
                "result.symbols",
                None,
                f"{who} needs the recovered symbols; this result did not keep them",
            )
        return np.asarray(result.symbols, dtype=np.complex128)

    # ------------------------------------------------------------------
    # DESIGN NOTE: OPERATOR OVERLOADING, MIRRORING THE CHANNEL PIPELINE
    # ------------------------------------------------------------------
    def __or__(self, other: "ConfidenceEstimator") -> "CombinedConfidence":
        """``a | b`` builds an estimator that consults both.

        The channel pipeline composes effects the same way.  Using the
        same spelling for the same idea in two unrelated parts of the
        project is deliberate: once a reader has met one composite, the
        other needs no explanation.
        """
        if not isinstance(other, ConfidenceEstimator):
            return NotImplemented
        return CombinedConfidence([self, other])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self._name!r})"


class CombinedConfidence(ConfidenceEstimator):
    """Several estimators consulted together.

    DESIGN NOTE: COMPOSITE
    ----------------------
    A combination *is* an estimator.  It holds a list of estimators,
    satisfies the same interface, and can therefore be combined again
    with no special case anywhere::

        evm | llr | ensemble

    Nothing that consumes an estimator needs to know whether it is
    holding one or a tree of them.

    Parameters
    ----------
    estimators:
        The estimators to consult. Each is asked for its own scores.
    reduce:
        How to merge them.  ``"mean"`` averages, which smooths out an
        estimator having a bad frame.  ``"min"`` takes the most
        pessimistic view, which is the right choice when the cost of a
        confident error is high.  ``"max"`` is the opposite and is
        offered mainly so that the demo can show it is a bad idea.
    """

    _REDUCTIONS = {
        "mean": lambda stack: np.mean(stack, axis=0),
        "min": lambda stack: np.min(stack, axis=0),
        "max": lambda stack: np.max(stack, axis=0),
    }

    def __init__(
        self,
        estimators: list[ConfidenceEstimator],
        reduce: str = "mean",
        name: str | None = None,
    ) -> None:
        estimators = list(estimators)
        if not estimators:
            raise ValidationError("estimators", [], "need at least one estimator")
        for item in estimators:
            if not isinstance(item, ConfidenceEstimator):
                raise ValidationError(
                    "estimators",
                    type(item).__name__,
                    "every member must be a ConfidenceEstimator",
                )
        if reduce not in self._REDUCTIONS:
            raise ValidationError(
                "reduce", reduce, f"expected one of {sorted(self._REDUCTIONS)}"
            )

        # Flatten a combination of combinations that uses the same
        # reduction, so ``a | b | c`` is one three-way merge rather than
        # a nested pair whose first member would be weighted double.
        flattened: list[ConfidenceEstimator] = []
        for item in estimators:
            if isinstance(item, CombinedConfidence) and item._reduce == reduce:
                flattened.extend(item._estimators)
            else:
                flattened.append(item)

        label = name or f"{reduce}({', '.join(e.name for e in flattened)})"
        super().__init__(label)
        self._estimators = flattened
        self._reduce = reduce

    @property
    def estimators(self) -> list[ConfidenceEstimator]:
        """Copy of the list of estimators being consulted."""
        return list(self._estimators)

    @property
    def reduction(self) -> str:
        """How the members' scores are merged."""
        return self._reduce

    def _scores(self, result, modulation):
        stack = np.stack(
            [e.estimate(result, modulation) for e in self._estimators], axis=0
        )
        return self._REDUCTIONS[self._reduce](stack)

    def __len__(self) -> int:
        return len(self._estimators)

    def __iter__(self):
        return iter(self._estimators)

    def __repr__(self) -> str:
        members = ", ".join(e.name for e in self._estimators)
        return f"CombinedConfidence([{members}], reduce={self._reduce!r})"


class SymbolQualityEstimator(ConfidenceEstimator):
    """Base for estimators that judge the *symbol*, not the bit.

    DESIGN NOTE: INHERITANCE THAT EARNS ITS PLACE
    -----------------------------------------------
    Several estimators work the same way: look at where a symbol landed
    relative to the constellation, produce one number for that symbol,
    then give every bit carried by the symbol that same number.  Only
    the middle step differs between them.

    So the first and last steps live here and are written once.  A
    subclass implements :meth:`_symbol_scores` and inherits the
    broadcasting, the length check against the frame, and the error
    message when the receiver did not keep its symbols.

    This is the third level of the hierarchy --
    ``ConfidenceEstimator -> SymbolQualityEstimator -> EVMConfidence`` --
    and each level adds exactly one idea.
    """

    @abstractmethod
    def _symbol_scores(
        self, symbols: np.ndarray, modulation: Modulation
    ) -> np.ndarray:
        """One score per symbol, in ``[0, 1]``."""

    def _scores(self, result, modulation):
        symbols = self._require_symbols(result, self._name)
        per_symbol = np.asarray(
            self._symbol_scores(symbols, modulation), dtype=float
        ).ravel()

        if per_symbol.size != symbols.size:
            raise ValidationError(
                f"{self._name}._symbol_scores",
                per_symbol.size,
                f"must return one value per symbol ({symbols.size})",
            )

        expected = symbols.size * modulation.bits_per_symbol
        if expected != result.n_bits:
            raise ValidationError(
                "result",
                result.n_bits,
                f"{symbols.size} symbols at {modulation.bits_per_symbol} bits "
                f"each is {expected} bits; the result holds {result.n_bits}",
            )

        # Every bit of a symbol inherits that symbol's score.  This is a
        # real limitation, not an implementation detail: within one
        # 16QAM symbol some bits are decided by a much narrower margin
        # than others, and an estimator that cannot see the difference
        # will rank them as equally safe.  Phase 5 measures what that
        # costs rather than assuming it is negligible.
        return np.repeat(per_symbol, modulation.bits_per_symbol)


__all__ = [
    "CombinedConfidence",
    "ConfidenceEstimator",
    "SymbolQualityEstimator",
]
