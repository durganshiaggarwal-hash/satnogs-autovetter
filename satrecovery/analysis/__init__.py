"""Analyse a signal: the AI names it, a receiver reads it.

An addition to the project, not a change to it: nothing outside this
package was edited.  It has its own entry point::

    python -m satrecovery.analysis --rml24 "C:/path/to/RML24_IQdata"

* :class:`ModulationClassifier` -- a CNN trained on RML24 that names a
  recording's modulation ("QPSK, 97% sure").
* :class:`SignalInput` -- a real RML24 recording or a simulated signal,
  behind one interface.
* :class:`Analyser` -- step 1, classify; step 2, recover the bits with
  the receiver for that modulation, keeping every stage for animation.
* :func:`create_analysis_app` -- the existing backend plus new endpoints.
"""

from .analyser import STAGES, Analyser
from .classifier import Classification, ModulationClassifier
from .inputs import RecordedInput, SignalInput, SimulatedInput

__all__ = [
    "Analyser",
    "Classification",
    "ModulationClassifier",
    "RecordedInput",
    "STAGES",
    "SignalInput",
    "SimulatedInput",
]
