"""Channel impairments, modelled as interchangeable effect objects."""

from .base import ChannelEffect
from .effects import (
    RML24_TAP_GAINS_DB,
    AdditiveWhiteGaussianNoise,
    CarrierFrequencyOffset,
    CarrierPhaseOffset,
    FlatRicianFading,
    MultipathRicianFading,
    SampleTimingOffset,
)
from .pipeline import ChannelFactory, ChannelPipeline

__all__ = [
    "RML24_TAP_GAINS_DB",
    "AdditiveWhiteGaussianNoise",
    "CarrierFrequencyOffset",
    "CarrierPhaseOffset",
    "ChannelEffect",
    "ChannelFactory",
    "ChannelPipeline",
    "FlatRicianFading",
    "MultipathRicianFading",
    "SampleTimingOffset",
]
