"""Phase 5 -- does the receiver know when it is wrong?

Every earlier phase asked how *often* a receiver is right.  This one
asks whether it can tell, without being shown the answer, which of its
own bits to distrust.

That matters operationally.  A ground station that knows which 5% of a
frame is unreliable can ask for those bits again, or hand them to an
error-correcting code with the right weights, or refuse to act on them.
A ground station holding a frame it believes entirely is stuck with it.

The package is three things, deliberately kept apart:

``metrics``
    The measuring apparatus.  :class:`CalibrationReport` asks whether
    the numbers are honest, :class:`RiskCoverageCurve` asks whether they
    are useful.  Built first and validated against cases with known
    answers, in the same order Phase 1 was built.

``estimators``
    Five different theories of where self-knowledge comes from, all
    behind one interface, so they can be swapped and compared rather
    than argued about.

``calibration``
    Post-hoc repairs for a confidence that ranks well but lies about
    the scale.  Every one is monotone, which means it can make a
    receiver honest and provably cannot make it better.

Typical use::

    from satrecovery.confidence import LLRConfidence, RiskCoverageCurve

    result = receiver.recover(frame)
    scores = LLRConfidence().estimate(result)
    curve = RiskCoverageCurve.from_result(result, frame.bits, scores)
    print(curve.coverage_at_risk(1e-3))
"""

from __future__ import annotations

from .base import CombinedConfidence, ConfidenceEstimator, SymbolQualityEstimator
from .calibration import (
    Calibrator,
    HistogramBinning,
    IdentityCalibrator,
    TemperatureScaling,
)
from .estimators import (
    EVMConfidence,
    LLRConfidence,
    MarginConfidence,
    MCDropoutConfidence,
    ModelEnsembleConfidence,
    SampledConfidence,
)
from .metrics import CalibrationReport, RiskCoverageCurve

__all__ = [
    "CalibrationReport",
    "Calibrator",
    "CombinedConfidence",
    "ConfidenceEstimator",
    "EVMConfidence",
    "HistogramBinning",
    "IdentityCalibrator",
    "LLRConfidence",
    "MCDropoutConfidence",
    "MarginConfidence",
    "ModelEnsembleConfidence",
    "RiskCoverageCurve",
    "SampledConfidence",
    "SymbolQualityEstimator",
    "TemperatureScaling",
]
