"""Automatic vetting of SatNOGS observations.

SatNOGS is a worldwide network of volunteer ground stations.  Each
observation's waterfall is still checked by hand ("with signal" /
"without signal").  This package learns that check from the human
labels, and -- the point of the project -- only decides when it is
sure, passing the rest to a person.

Step 1 (this module set): getting the data.

* :class:`Observation`, :class:`WaterfallLabel` -- one observation and its label
* :class:`ObservationSource` -> :class:`SatnogsApiSource`, :class:`ManifestSource`
* :class:`Manifest` -- the observations downloaded so far, on disk
* :class:`ArtifactCache` -- their waterfalls (and audio), downloaded once
* :class:`WaterfallImage` -- a waterfall picture turned back into numbers
"""

from .artifacts import ArtifactCache, ArtifactError
from .observation import Observation, WaterfallLabel
from .source import (
    Manifest,
    ManifestSource,
    ObservationQuery,
    ObservationSource,
    Response,
    SatnogsApiSource,
    SatnogsError,
    Transport,
    UrllibTransport,
)
from .waterfall import WaterfallDecodeError, WaterfallImage
from .dataset import NO_SIGNAL, SIGNAL, StationSplit, TimeSplit, VettingSet, VettingSplit, normalise
from .model import WaterfallVetter

__all__ = [
    "WaterfallVetter",
    "NO_SIGNAL", "SIGNAL", "StationSplit", "TimeSplit", "VettingSet", "VettingSplit", "normalise",
    "ArtifactCache", "ArtifactError", "Manifest", "ManifestSource", "Observation", "ObservationQuery",
    "ObservationSource", "Response", "SatnogsApiSource", "SatnogsError", "Transport", "UrllibTransport",
    "WaterfallDecodeError", "WaterfallImage", "WaterfallLabel",
]
