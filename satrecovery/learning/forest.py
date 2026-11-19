"""A Random Forest that recognises the modulation scheme.

The classical machine-learning baseline, and the task most published
work on RML24 tackles: given a captured signal, which of the schemes is
it?  A forest of decision trees votes on the answer, each tree having
learned which of the hand-computed features in
:mod:`~satrecovery.learning.features` separate what.

Two reasons it earns a place in a project about deep learning.

**It is a fair point of comparison.**  Papers report deep networks
reaching about 73% across RML24's 22 schemes.  Knowing what a forest on
good features achieves on the same task says whether the network's
accuracy comes from the architecture or merely from the problem being
easier than it looks.

**It can explain itself.**  A forest reports which features it relied
on, and those features mean something physical -- the fourth-order
cumulant, peak-to-average power.  When it gets a frame wrong you can see
why.  That is worth a great deal when the project's second question is
whether a model knows when it is wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.exceptions import ValidationError
from ..core.frame import Frame
from .base import TrainableModel, TrainingHistory
from .features import FeatureExtractor, default_extractor


class ForestModulationClassifier(TrainableModel):
    """Random Forest over hand-computed signal features.

    Parameters
    ----------
    extractor:
        How a signal becomes a row of numbers.  Defaults to the
        cumulants and spectral descriptors combined.
    n_trees:
        How many decision trees vote.  More is steadier and slower;
        accuracy flattens off well before a hundred here.
    max_depth:
        Depth limit per tree.  ``None`` lets them grow until the leaves
        are pure, which for this many features and this much data does
        not overfit noticeably.
    seed:
        Fixes the randomness, so a reported accuracy can be reproduced.
    """

    def __init__(
        self,
        extractor: FeatureExtractor | None = None,
        n_trees: int = 200,
        max_depth: int | None = None,
        seed: int = 0,
        name: str = "Forest",
    ) -> None:
        if int(n_trees) < 1:
            raise ValidationError("n_trees", n_trees, "need at least one tree")

        super().__init__(name)
        self._extractor = extractor or default_extractor()
        self._n_trees = int(n_trees)
        self._max_depth = max_depth
        self._seed = int(seed)
        self._forest: Any = None
        self._classes: list[str] = []

    # ------------------------------------------------------------------
    # Read-only configuration
    # ------------------------------------------------------------------
    @property
    def extractor(self) -> FeatureExtractor:
        """The feature extractor this classifier was built around."""
        return self._extractor

    @property
    def classes(self) -> list[str]:
        """Modulation names this classifier knows, in the order it uses."""
        return list(self._classes)

    @property
    def n_trees(self) -> int:
        """How many trees vote."""
        return self._n_trees

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
        **_: Any,
    ) -> TrainingHistory:
        """Grow the forest on pre-extracted features.

        ``inputs`` is a ``(rows, features)`` array -- already extracted,
        not raw signal -- and ``targets`` is one modulation name per row.
        Keeping extraction outside the model means the same features can
        be reused by a different classifier without recomputing them.
        """
        from sklearn.ensemble import RandomForestClassifier

        inputs, targets = self._check_pair(inputs, targets)
        targets = np.asarray(targets).astype(str).ravel()

        if inputs.shape[1] != self._extractor.n_features:
            raise ValidationError(
                "inputs",
                inputs.shape[1],
                f"{self._extractor.name} produces "
                f"{self._extractor.n_features} features",
            )

        self._forest = RandomForestClassifier(
            n_estimators=self._n_trees,
            max_depth=self._max_depth,
            random_state=self._seed,
            n_jobs=-1,
        )
        self._forest.fit(inputs, targets)
        self._classes = [str(c) for c in self._forest.classes_]
        self._mark_trained()

        history = TrainingHistory(self._name)
        metrics = {
            "train_accuracy": float(self._forest.score(inputs, targets)),
            "n_rows": float(len(inputs)),
        }
        if validation is not None:
            val_inputs, val_targets = validation
            metrics["val_accuracy"] = float(
                self._forest.score(
                    np.asarray(val_inputs), np.asarray(val_targets).astype(str).ravel()
                )
            )
        history.record(1, **metrics)
        self._history = history
        return history

    # ------------------------------------------------------------------
    # Using it
    # ------------------------------------------------------------------
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Most-voted modulation name for each row of features."""
        self._require_trained()
        return np.asarray(self._forest.predict(np.asarray(inputs))).astype(str)

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        """Fraction of trees voting for each class, as ``(rows, classes)``."""
        self._require_trained()
        return np.asarray(self._forest.predict_proba(np.asarray(inputs)))

    def classify_frame(
        self, frame: Frame, matched_filter: Any = None
    ) -> tuple[str, float]:
        """Classify one frame, returning the name and how sure it is.

        The confidence is the share of trees that agreed.  Unanimous
        means the features landed squarely in one class's territory; a
        narrow margin means the frame sat near a boundary.  It is a
        genuinely useful confidence, and it comes free with the method.
        """
        self._require_trained()
        if not isinstance(frame, Frame):
            raise ValidationError(
                "frame", type(frame).__name__, "expected a Frame instance"
            )

        samples = frame.signal.samples
        if matched_filter is not None:
            samples = matched_filter.matched_filter(samples)
        symbols = samples[:: max(frame.samples_per_symbol, 1)]

        features = self._extractor.extract(symbols)[None, :]
        votes = self.predict_proba(features)[0]
        winner = int(np.argmax(votes))
        return self._classes[winner], float(votes[winner])

    def feature_importances(self) -> dict[str, float]:
        """How much each feature contributed, largest first.

        The reason for preferring a forest over a black box as the
        baseline: this tells you *what the model is looking at*, in
        units an engineer recognises.
        """
        self._require_trained()
        pairs = zip(self._extractor.feature_names, self._forest.feature_importances_)
        return dict(sorted(pairs, key=lambda item: item[1], reverse=True))

    def confusion(
        self, inputs: np.ndarray, targets: np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        """Confusion matrix and its labels.

        Which schemes get mistaken for which is more informative than a
        single accuracy number -- 16QAM and 16APSK carry the same number
        of bits and have similar statistics, so confusing those two is a
        different kind of failure from confusing BPSK with 64QAM.
        """
        self._require_trained()
        from sklearn.metrics import confusion_matrix

        predicted = self.predict(inputs)
        targets = np.asarray(targets).astype(str).ravel()
        matrix = confusion_matrix(targets, predicted, labels=self._classes)
        return matrix, list(self._classes)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write the forest and its metadata to disk."""
        self._require_trained()
        import joblib

        path = Path(path).with_suffix(".joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"forest": self._forest, "classes": self._classes}, path
        )
        self._write_sidecar(
            path,
            {
                "model": self._name,
                "kind": "ForestModulationClassifier",
                "extractor": self._extractor.name,
                "feature_names": self._extractor.feature_names,
                "classes": self._classes,
                "n_trees": self._n_trees,
                "history": self._history.to_dict(),
            },
        )
        return path

    def load(self, path: str | Path) -> "ForestModulationClassifier":
        """Restore a forest saved by :meth:`save`."""
        import joblib

        path = Path(path).with_suffix(".joblib")
        if not path.exists():
            raise ValidationError("path", str(path), "no saved model at this path")

        payload = joblib.load(path)
        self._forest = payload["forest"]
        self._classes = [str(c) for c in payload["classes"]]
        self._mark_trained()
        return self

    def describe(self) -> str:
        state = "trained" if self.is_trained else "untrained"
        classes = f", {len(self._classes)} schemes" if self._classes else ""
        return (
            f"{self._name} ({state}, {self._n_trees} trees, "
            f"{self._extractor.n_features} features{classes})"
        )


__all__ = ["ForestModulationClassifier"]
