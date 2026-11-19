"""Modulation schemes: an abstract base class, six implementations, a registry."""

from .apsk import APSK16
from .base import Modulation, gray_code, q_function
from .psk import BPSK, PSK8, QPSK, PhaseShiftKeying
from .qam import QAM16, QAM64, QuadratureAmplitudeModulation
from .registry import ModulationRegistry, get_modulation

__all__ = [
    "APSK16",
    "BPSK",
    "PSK8",
    "QAM16",
    "QAM64",
    "QPSK",
    "Modulation",
    "ModulationRegistry",
    "PhaseShiftKeying",
    "QuadratureAmplitudeModulation",
    "get_modulation",
    "gray_code",
    "q_function",
]
