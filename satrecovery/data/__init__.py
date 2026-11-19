"""Phase 3: real recordings from RML24.

* :mod:`~satrecovery.data.layout` -- what the published file contains.
* :class:`Recording` -- one stored recording and its labels.
* :class:`RecordingArchive` -- the full HDF5 file (:class:`RML24File`)
  or a small sample (:class:`SampleArchive`), behind one interface.
* :class:`ReferenceDecoder` -- recovers the transmitted bits the
  dataset does not store, and refuses recordings it cannot vouch for.
* :class:`RecordedSource` -- turns accepted recordings into graded
  frames at any noise level, for the Phase 6 benchmark.
"""

from . import layout
from .archive import RecordingArchive, RML24File, SampleArchive, open_archive
from .recording import Recording
from .reference import Reference, ReferenceDecoder
from .source import RecordedSource

__all__ = [
    "layout",
    "Recording",
    "RecordingArchive",
    "RML24File",
    "SampleArchive",
    "open_archive",
    "Reference",
    "ReferenceDecoder",
    "RecordedSource",
]
