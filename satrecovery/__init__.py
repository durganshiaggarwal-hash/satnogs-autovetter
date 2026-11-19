"""Satellite signal recovery -- comparing classical, AI and hybrid receivers.

A software system that measures how reliably the transmitted *bits* can be
recovered from a distorted satellite signal, and whether a receiver can
tell when its own output is probably wrong.

Package layout
--------------
``core``
    Domain objects: :class:`~satrecovery.core.signal.Signal`,
    :class:`~satrecovery.core.frame.Frame`,
    :class:`~satrecovery.core.conditions.ChannelConditions`, and the
    exception hierarchy.
``modulation``
    Modulation schemes, as an abstract base class plus six concrete
    implementations and a registry.
``channel``
    Channel impairments as interchangeable effect objects, plus the
    pipeline that chains them.
``simulation``
    Pulse shaping filters and the signal simulator.
``receivers``
    The classical chain, the learned receivers, and the demappers they
    share.
``learning``
    Trainable models: feature extractors, a random-forest classifier,
    a neural demapper and an end-to-end demodulator.
``confidence``
    Whether a receiver knows when it is wrong: confidence estimators,
    post-hoc calibration, and the metrics that grade both.

``evaluation``
    The benchmark runner, statistics, held-out splits and results store.
``backend``
    The FastAPI server the GUI talks to.

Phase 8 adds ``gui``.

The top level re-exports the handful of names most code needs, so
ordinary use is a single import::

    from satrecovery import SignalSimulator, ChannelConditions

Deeper imports stay available for everything else.
"""

from __future__ import annotations

from .channel.base import ChannelEffect
from .channel.effects import (
    AdditiveWhiteGaussianNoise,
    CarrierFrequencyOffset,
    CarrierPhaseOffset,
    FlatRicianFading,
    MultipathRicianFading,
    SampleTimingOffset,
)
from .channel.pipeline import ChannelFactory, ChannelPipeline
from .confidence.base import ConfidenceEstimator
from .confidence.calibration import (
    Calibrator,
    HistogramBinning,
    IdentityCalibrator,
    TemperatureScaling,
)
from .confidence.estimators import (
    EVMConfidence,
    LLRConfidence,
    MarginConfidence,
    ModelEnsembleConfidence,
)
from .confidence.metrics import CalibrationReport, RiskCoverageCurve
from .core.conditions import ChannelConditions
from .core.exceptions import (
    ChannelError,
    ConfigurationError,
    ModulationError,
    SatRecoveryError,
    SignalError,
    UnknownModulationError,
    ValidationError,
)
from .core.frame import Frame
from .core.signal import Signal
from .modulation.apsk import APSK16
from .modulation.base import Modulation
from .modulation.psk import BPSK, PSK8, QPSK, PhaseShiftKeying
from .modulation.qam import QAM16, QAM64, QuadratureAmplitudeModulation
from .modulation.registry import ModulationRegistry, get_modulation
from .receivers.base import Receiver, ReceiverChain, ReceiverStage, RecoveryContext
from .receivers.classical import ClassicalReceiver
from .receivers.demapper import Demapper, HardDemapper, SoftDemapper
from .receivers.neural import HybridReceiver, NeuralReceiver
from .receivers.result import RecoveryResult
from .simulation.pulse import PulseShape, Rectangular, RootRaisedCosine
from .simulation.simulator import SignalSimulator

__version__ = "0.1.0"

__all__ = [
    "APSK16",
    "BPSK",
    "PSK8",
    "QAM16",
    "QAM64",
    "QPSK",
    "AdditiveWhiteGaussianNoise",
    "CalibrationReport",
    "Calibrator",
    "CarrierFrequencyOffset",
    "CarrierPhaseOffset",
    "ChannelConditions",
    "ClassicalReceiver",
    "ChannelEffect",
    "ChannelError",
    "ChannelFactory",
    "ChannelPipeline",
    "ConfidenceEstimator",
    "ConfigurationError",
    "Demapper",
    "EVMConfidence",
    "FlatRicianFading",
    "HistogramBinning",
    "IdentityCalibrator",
    "LLRConfidence",
    "MarginConfidence",
    "ModelEnsembleConfidence",
    "RiskCoverageCurve",
    "TemperatureScaling",
    "Frame",
    "Modulation",
    "NeuralReceiver",
    "ModulationError",
    "ModulationRegistry",
    "MultipathRicianFading",
    "PhaseShiftKeying",
    "PulseShape",
    "HardDemapper",
    "HybridReceiver",
    "QuadratureAmplitudeModulation",
    "Receiver",
    "ReceiverChain",
    "ReceiverStage",
    "RecoveryContext",
    "RecoveryResult",
    "Rectangular",
    "RootRaisedCosine",
    "SampleTimingOffset",
    "SatRecoveryError",
    "Signal",
    "SoftDemapper",
    "SignalError",
    "SignalSimulator",
    "UnknownModulationError",
    "ValidationError",
    "__version__",
    "get_modulation",
]
