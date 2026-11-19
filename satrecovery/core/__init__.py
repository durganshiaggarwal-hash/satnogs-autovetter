"""Domain objects shared by every other layer."""

from .conditions import FADING_MODES, ChannelConditions
from .exceptions import (
    BitLengthError,
    ChannelError,
    ConfigurationError,
    EmptySignalError,
    ModelNotTrainedError,
    ModulationError,
    ReceiverError,
    SatRecoveryError,
    SignalError,
    UnknownModulationError,
    ValidationError,
)
from .frame import Frame
from .signal import Signal

__all__ = [
    "FADING_MODES",
    "BitLengthError",
    "ChannelConditions",
    "ChannelError",
    "ConfigurationError",
    "EmptySignalError",
    "Frame",
    "ModelNotTrainedError",
    "ModulationError",
    "ReceiverError",
    "SatRecoveryError",
    "Signal",
    "SignalError",
    "UnknownModulationError",
    "ValidationError",
]
