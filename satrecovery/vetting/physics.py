"""The orbit check as numbers a model can use, and the model that combines them with the picture.

:func:`physics_row` turns a :class:`~satrecovery.vetting.orbit.TrackExplanation`
into a few numbers: how much of the bright track the scheduled satellite,
another satellite and a ground transmitter each explain, whether a timing
error explains it, and whether there was a track at all.

:class:`CombinedVetter` is a small logistic model on top: the picture
model's opinion (as log-odds) plus those numbers.  It is fitted on
out-of-fold picture scores, so it learns how far to trust each kind of
evidence without the picture model's training set leaking in.

DESIGN NOTE: COMPOSITION OVER INHERITANCE
-----------------------------------------
The combined vetter does not subclass the picture vetter; it *uses* its
output as one input among several.  Either half can be replaced alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.exceptions import ValidationError
from .orbit import TrackExplanation

PHYSICS = ("no_track", "scheduled", "other", "ground", "timing", "track_points")


def physics_row(explanation: TrackExplanation) -> dict:
    """The numbers the combined model uses, from one explanation."""
    best = {"scheduled": 0.0, "other": 0.0, "ground": 0.0}
    for match in explanation.matches:
        if match.curve.norad == 0:
            kind = "ground"
        elif match.curve.scheduled or match.curve.name.endswith("(today's orbit)"):
            kind = "scheduled"
        else:
            kind = "other"
        best[kind] = max(best[kind], float(match.share))
    timing = float(explanation.timing.share) if explanation.timing is not None else 0.0
    return {"no_track": float(explanation.source == "no track"), **best, "timing": timing,
            "track_points": float(len(explanation.track_seconds)), "source": explanation.source}


def physics_vector(row: dict) -> np.ndarray:
    values = [float(row.get(name, 0.0)) for name in PHYSICS]
    values[-1] = np.log1p(values[-1]) / 5.0   # track size on a gentle scale
    return np.asarray(values, dtype=np.float64)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


@dataclass
class CombinedVetter:
    """P(signal) from the picture model's opinion and the orbit check, by logistic regression."""

    coefficients: np.ndarray | None = None
    intercept: float = 0.0

    def _design(self, picture_p: np.ndarray, physics: np.ndarray) -> np.ndarray:
        physics = np.atleast_2d(np.asarray(physics, dtype=float))
        if physics.shape[1] != len(PHYSICS):
            raise ValidationError("physics", physics.shape, f"expected (n, {len(PHYSICS)})")
        return np.column_stack([_logit(picture_p), physics])

    def fit(self, picture_p: np.ndarray, physics: np.ndarray, target: np.ndarray, strength: float = 1.0
            ) -> "CombinedVetter":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=strength, max_iter=1000).fit(self._design(picture_p, physics), target)
        self.coefficients, self.intercept = model.coef_[0].copy(), float(model.intercept_[0])
        return self

    def predict_proba(self, picture_p: np.ndarray, physics: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise ValidationError("CombinedVetter", "unfitted", "fit it first")
        z = self._design(picture_p, physics) @ self.coefficients + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def weights(self) -> dict[str, float]:
        names = ("picture model (log-odds)",) + PHYSICS
        return dict(zip(names, (self.coefficients if self.coefficients is not None else [])))

    def to_dict(self) -> dict:
        return {"coefficients": [float(c) for c in self.coefficients], "intercept": self.intercept}

    @classmethod
    def from_dict(cls, data: dict) -> "CombinedVetter":
        return cls(np.asarray(data["coefficients"], dtype=float), float(data["intercept"]))
