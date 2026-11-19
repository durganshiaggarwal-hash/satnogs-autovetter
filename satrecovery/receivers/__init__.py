"""Receivers: everything that turns a damaged signal back into bits.

Phase 2 provides the classical chain. Phases 4 and 5 add learned and
hybrid receivers alongside it, all behind the same :class:`Receiver`
interface so they can be compared on equal terms.
"""

from .base import Receiver, ReceiverChain, ReceiverStage, RecoveryContext
from .classical import ClassicalReceiver
from .demapper import Demapper, HardDemapper, SoftDemapper
from .neural import HybridReceiver, NeuralReceiver
from .result import RecoveryResult
from .stages import (
    AutomaticGainControl,
    CoarseFrequencyCorrection,
    DecisionDirectedPLL,
    GardnerTimingRecovery,
    LMSEqualizer,
    MatchedFilterStage,
    interpolate_cubic,
    pi_loop_gains,
)

__all__ = [
    "AutomaticGainControl",
    "ClassicalReceiver",
    "CoarseFrequencyCorrection",
    "DecisionDirectedPLL",
    "Demapper",
    "GardnerTimingRecovery",
    "HardDemapper",
    "HybridReceiver",
    "LMSEqualizer",
    "MatchedFilterStage",
    "NeuralReceiver",
    "Receiver",
    "ReceiverChain",
    "ReceiverStage",
    "RecoveryContext",
    "RecoveryResult",
    "SoftDemapper",
    "interpolate_cubic",
    "pi_loop_gains",
]
