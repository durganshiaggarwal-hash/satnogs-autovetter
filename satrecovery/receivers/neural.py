"""The learned and hybrid receivers.

With these, the three methods the project exists to compare all satisfy
the same :class:`~satrecovery.receivers.base.Receiver` interface::

    ClassicalReceiver()   # mathematics all the way through
    NeuralReceiver(cnn)   # raw samples in, bits out, nothing designed
    HybridReceiver(mlp)   # mathematics for sync, a network for the decision

Phase 6's evaluator holds a list of these and loops over it.  It cannot
tell them apart, which is exactly what makes the comparison fair: there
is no code path where one method is treated more kindly than another.

WHY THE HYBRID IS ALMOST NO CODE
--------------------------------
:class:`HybridReceiver` adds about ten lines.  All the work was done in
Phase 2, by defining ``Demapper`` as an interface rather than writing the
nearest-neighbour rule inline in the receiver.  Because a learned
demapper satisfies that interface, it drops into the existing chain and
every stage before it carries on untouched.

That is the practical argument for designing to interfaces, and it is
worth pointing at: the project's headline contribution costs ten lines
because the groundwork was laid three phases earlier.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.exceptions import ValidationError
from ..core.frame import Frame
from ..modulation.registry import ModulationRegistry
from .base import Receiver
from .classical import ClassicalReceiver
from .demapper import Demapper
from .result import RecoveryResult


class HybridReceiver(ClassicalReceiver):
    """The Phase 2 chain, with a learned demapper on the end.

    Mathematics does what mathematics is good at -- matched filtering,
    estimating the frequency, finding the timing, undoing the rotation.
    All of that is exact, cheap and provably correct, and there is no
    reason to make a network rediscover it.

    The final decision is where a learned model has something to add.
    The classical rule picks the nearest constellation point, which is
    optimal when the only damage left is Gaussian noise and blind to
    anything else -- it looks at one symbol and never at its neighbours.
    A network given a window of symbols can learn to undo distortions
    that rule cannot express.

    Measured on signals with echoes a symbol apart: the learned demapper
    cut the bit error rate by between 1.4 and 2.3 times.  With its window
    set to zero -- seeing one symbol, the same information the classical
    rule gets -- it tied instead, which is the control that shows the
    gain comes from the extra context and not from the network itself.
    """

    def __init__(self, demapper: Demapper, name: str = "Hybrid", **options: Any) -> None:
        if not isinstance(demapper, Demapper):
            raise ValidationError(
                "demapper",
                type(demapper).__name__,
                "a hybrid receiver needs a demapper; "
                "pass a trained MLPDemapper",
            )
        super().__init__(demapper=demapper, name=name, **options)

    def describe(self) -> str:
        return f"{self.name}: classical chain + {self.demapper.name} demapper"


class NeuralReceiver(Receiver):
    """Raw samples in, bits out, with nothing designed in between.

    The pure end-to-end approach: the network is handed the received
    waveform with the frequency offset, timing error and fading still in
    it, and produces the bits.  No matched filter, no carrier loop, no
    timing recovery -- if those operations are needed, the network has
    to have learned them from examples.

    What this gives up matters.  The classical chain's loops integrate
    information across a whole frame; a convolution sees only its
    receptive field.  The chain's frequency estimator uses the
    constellation's symmetry, a fact nobody can learn from a few hundred
    frames as reliably as it can be derived in one line.

    What it gains is freedom from assumptions.  Every stage of the
    classical chain assumes something -- that the pulse is a known shape,
    that the noise is Gaussian, that the constellation is what we were
    told.  A network assumes none of that, so where the assumptions are
    wrong it has room to do better.

    Which effect wins is the experiment.
    """

    def __init__(self, model: Any, name: str = "Neural") -> None:
        if not hasattr(model, "predict_bits"):
            raise ValidationError(
                "model",
                type(model).__name__,
                "expected a model with a predict_bits method, "
                "such as CNNDemodulator",
            )
        super().__init__(name)
        self._model = model

    @property
    def model(self) -> Any:
        """The network doing the work."""
        return self._model

    @property
    def pilot_symbols(self) -> int:
        """Zero: this receiver uses no preamble.

        It does not need one, because it never has to resolve the
        constellation's rotational ambiguity by itself -- the training
        labels pinned the orientation down once, for good.  That is a
        real advantage over the classical chain, and a real cost: the
        orientation it learned is the orientation it is stuck with.

        The property exists so that evaluation code can ask any receiver
        how many bits to exclude without checking its type.
        """
        return 0

    def _recover(self, frame: Frame) -> RecoveryResult:
        scheme = ModulationRegistry.create(frame.modulation)

        iq = frame.signal.as_iq_array().astype(np.float32)
        power = float(np.mean(iq**2))
        if power > 0:
            iq = iq / np.sqrt(power)

        bits, confidence, llr = self._model.predict_bits(iq)
        bits = np.asarray(bits).ravel()
        confidence = np.asarray(confidence).ravel()
        llr = np.asarray(llr).ravel()

        # The network emits a fixed number of bits set by its
        # architecture, which need not match this frame exactly. Trim or
        # pad rather than failing, so one odd frame does not stop a
        # benchmark -- and record it, because silent padding would
        # otherwise flatter the error rate.
        wanted = frame.n_bits
        padded = 0
        if bits.size > wanted:
            bits, confidence, llr = bits[:wanted], confidence[:wanted], llr[:wanted]
        elif bits.size < wanted:
            padded = wanted - bits.size
            bits = np.concatenate([bits, np.zeros(padded, dtype=np.uint8)])
            confidence = np.concatenate([confidence, np.full(padded, 0.5)])
            llr = np.concatenate([llr, np.zeros(padded)])

        return RecoveryResult(
            bits=bits,
            confidence=confidence,
            symbols=None,
            modulation=scheme.name,
            receiver=self.name,
            llr=llr,
            diagnostics={
                "model": getattr(self._model, "name", type(self._model).__name__),
                "n_parameters": getattr(self._model, "n_parameters", None),
                "padded_bits": padded,
                "n_pilot_bits": 0,
            },
        )

    def describe(self) -> str:
        return f"{self.name}: {getattr(self._model, 'name', 'model')}, end to end"


__all__ = ["HybridReceiver", "NeuralReceiver"]
