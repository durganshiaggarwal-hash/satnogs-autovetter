"""A vetting session: the observations, the model's suggestions, and your decisions.

This is everything the vetting window shows, with no Qt in it, so it can
be tested on its own.

* :class:`Policy` turns a probability into a verdict: confident "signal",
  confident "no signal", or **unsure** -- the ones a person must check.
* :class:`Explainer` turns the twelve measurements into reasons in words.
* :class:`DecisionLog` keeps every decision in a file.
* :class:`VettingSession` puts them together.  Its scores are
  **out of fold**: each observation is scored by a forest trained with
  its whole ground station left out, so what you see on screen is what
  the model would do on a station it has never met, not a remembered
  answer.

DESIGN NOTE: COMPOSITION
------------------------
A session *has* a policy, an explainer, a log and a vetter; it is none of
them.  Each part can be swapped (a stricter policy, a different vetter)
without the others changing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..core.exceptions import ValidationError
from .artifacts import ArtifactCache, ArtifactError
from .dataset import normalise
from .features import CLIP, FEATURES, STRONG, feature_matrix, waterfall_features
from .model import FeatureVetter
from .observation import Observation, WaterfallLabel

#: short labels for the twelve measurements, for the reasons list
SHORT: dict[str, str] = {
    "peak_prominence": "Brightest frequency stands out",
    "peak_offset": "Brightest frequency off centre",
    "peak_width": "Width of the bright band",
    "centre_excess": "Centre band brighter than the rest",
    "centre_p99": "Peak strength at the centre",
    "band_p995": "Peak strength anywhere",
    "centre_lit_rows": "Pass with a strong centre signal",
    "any_lit_rows": "Pass with a strong signal anywhere",
    "centre_burstiness": "Centre strength comes in bursts",
    "track_wander": "Bright track moves in frequency",
    "track_steadiness": "Strong moments at the centre",
    "row_noise_variation": "Noise level changes over the pass",
}
#: measurements that are shares, shown as percentages
SHARES = {"peak_offset", "peak_width", "centre_lit_rows", "any_lit_rows", "track_wander", "track_steadiness"}


class Verdict(Enum):
    SIGNAL = "signal"
    NO_SIGNAL = "no signal"
    UNSURE = "unsure"

    @property
    def label(self) -> WaterfallLabel | None:
        return {Verdict.SIGNAL: WaterfallLabel.SIGNAL, Verdict.NO_SIGNAL: WaterfallLabel.NO_SIGNAL}.get(self)


@dataclass(frozen=True)
class Policy:
    """Decide alone only below ``low`` or above ``high``; in between, ask a person."""

    low: float = 0.15
    high: float = 0.85

    def __post_init__(self) -> None:
        if not 0.0 <= self.low < self.high <= 1.0:
            raise ValidationError("policy", (self.low, self.high), "need 0 <= low < high <= 1")

    def verdict(self, probability: float) -> Verdict:
        if probability >= self.high:
            return Verdict.SIGNAL
        if probability <= self.low:
            return Verdict.NO_SIGNAL
        return Verdict.UNSURE


# ----------------------------------------------------------------------
class Explainer:
    """Reasons in words: which measurements look like "signal" and which like "no signal"."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, importances: dict[str, float] | None = None) -> None:
        features, targets = np.asarray(features, dtype=np.float32), np.asarray(targets)
        if features.ndim != 2 or features.shape[1] != len(FEATURES) or len(features) != len(targets):
            raise ValidationError("features", features.shape, f"expected (n, {len(FEATURES)}) lined up with targets")
        if len(set(targets.tolist())) < 2:
            raise ValidationError("targets", sorted(set(targets.tolist())), "need examples of both labels")
        self._typical_signal = np.median(features[targets == 1], axis=0)
        self._typical_none = np.median(features[targets == 0], axis=0)
        low, high = np.percentile(features, [10, 90], axis=0)
        self._spread = np.maximum(high - low, 1e-6)
        weights = importances or {name: 1.0 for name in FEATURES}
        self._weights = np.array([weights.get(name, 0.0) for name in FEATURES], dtype=np.float32)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {"typical_signal": self._typical_signal, "typical_none": self._typical_none,
                "spread": self._spread, "weights": self._weights}

    @classmethod
    def from_arrays(cls, arrays) -> "Explainer":
        """Rebuild from :meth:`to_arrays` without the training data."""
        explainer = cls.__new__(cls)
        explainer._typical_signal = np.asarray(arrays["typical_signal"], dtype=np.float32)
        explainer._typical_none = np.asarray(arrays["typical_none"], dtype=np.float32)
        explainer._spread = np.asarray(arrays["spread"], dtype=np.float32)
        explainer._weights = np.asarray(arrays["weights"], dtype=np.float32)
        if any(len(a) != len(FEATURES) for a in (explainer._typical_signal, explainer._typical_none,
                                                  explainer._spread, explainer._weights)):
            raise ValidationError("explainer", "arrays", f"each needs {len(FEATURES)} values")
        return explainer

    @staticmethod
    def _show(name: str, value: float) -> str:
        return f"{value:.0%}" if name in SHARES else f"{value:.2f}"

    def reasons(self, values: np.ndarray, count: int = 4) -> list[tuple[str, str, bool]]:
        """Up to ``count`` ``(title, detail, points_to_signal)``, most telling first."""
        values = np.asarray(values, dtype=np.float32)
        # How far the value sits from the midpoint between the two typical
        # values, in units of the measurement's usual range; it counts only
        # as much as the two typical values differ, and as the model uses it.
        gap = self._typical_signal - self._typical_none
        offset = values - (self._typical_signal + self._typical_none) / 2
        lean = offset * np.sign(gap)                                   # > 0: like signal
        strength = (np.clip(np.abs(offset) / self._spread, 0, 1)
                    * np.clip(np.abs(gap) / self._spread, 0, 1) * self._weights)
        out = []
        for i in np.argsort(-strength)[:count]:
            name = list(FEATURES)[i]
            detail = (f"{self._show(name, values[i])}  (usually {self._show(name, self._typical_signal[i])} with "
                      f"signal, {self._show(name, self._typical_none[i])} without)")
            out.append((SHORT[name], detail, bool(lean[i] > 0)))
        return out


# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Decision:
    obs_id: int
    label: WaterfallLabel
    by: str                    # "you" or "auto"
    probability: float
    at: str

    def to_dict(self) -> dict:
        return {"obs_id": self.obs_id, "label": self.label.value, "by": self.by,
                "probability": self.probability, "at": self.at}

    @classmethod
    def from_dict(cls, data: dict) -> "Decision":
        return cls(int(data["obs_id"]), WaterfallLabel(data["label"]), str(data["by"]),
                   float(data["probability"]), str(data["at"]))


class DecisionLog:
    """Every decision, appended to a JSON-lines file; the latest one per observation wins."""

    def __init__(self, path: str | Path | None) -> None:
        self._path = Path(path) if path else None
        self._latest: dict[int, Decision] = {}
        if self._path and self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                try:
                    decision = Decision.from_dict(json.loads(line))
                except (ValueError, KeyError, TypeError):
                    continue
                self._latest[decision.obs_id] = decision

    def record(self, decision: Decision) -> None:
        self._latest[decision.obs_id] = decision
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(decision.to_dict()) + "\n")

    def get(self, obs_id: int) -> Decision | None:
        return self._latest.get(obs_id)

    def __len__(self) -> int:
        return len(self._latest)

    def __iter__(self):
        return iter(self._latest.values())


# ----------------------------------------------------------------------
class FeatureStore:
    """The twelve measurements of every cached waterfall, worked out once and kept in a file."""

    def __init__(self, cache: ArtifactCache, path: str | Path | None = None) -> None:
        self._cache = cache
        self._path = Path(path) if path else cache.root / "features.npz"
        self._rows: dict[int, np.ndarray] = {}
        if self._path.exists():
            try:
                with np.load(self._path) as data:
                    if list(data["names"]) == list(FEATURES):
                        self._rows = dict(zip(data["ids"].tolist(), data["values"]))
            except (OSError, ValueError, KeyError):
                self._rows = {}

    def get(self, observations: Sequence[Observation], progress=None) -> tuple[list[Observation], np.ndarray]:
        """``(observations that have a waterfall, their measurements)``; new ones are computed and saved."""
        kept, rows, added = [], [], 0
        for n, obs in enumerate(observations):
            if obs.id not in self._rows:
                if not self._cache.has_waterfall(obs):
                    continue
                try:
                    self._rows[obs.id] = waterfall_features(normalise(self._cache.waterfall(obs).power))
                except (ArtifactError, OSError, ValueError, ValidationError):
                    continue
                added += 1
                if progress and added % 100 == 0:
                    progress(n + 1, len(observations))
            kept.append(obs)
            rows.append(self._rows[obs.id])
        if added:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            ids = np.array(list(self._rows), dtype=np.int64)
            np.savez(self._path, ids=ids, values=np.stack([self._rows[i] for i in ids.tolist()]),
                     names=np.array(list(FEATURES)))
        return kept, (np.stack(rows).astype(np.float32) if rows else np.empty((0, len(FEATURES)), np.float32))


def out_of_fold(features: np.ndarray, targets: np.ndarray, groups: Sequence, folds: int = 5,
                seed: int = 0) -> np.ndarray:
    """P(signal) for each row from a forest that never saw that row's group (station)."""
    from sklearn.model_selection import GroupKFold

    groups = np.asarray([str(g) for g in groups])
    folds = max(2, min(folds, len(set(groups.tolist()))))
    probability = np.full(len(targets), 0.5, dtype=np.float32)
    for train, test in GroupKFold(n_splits=folds).split(features, targets, groups):
        if len(set(targets[train].tolist())) < 2:
            continue
        model = FeatureVetter(seed=seed)
        model.fit_features(features[train], targets[train])
        probability[test] = model.predict_proba_features(features[test])
    return probability


# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Item:
    """One row of the queue."""

    obs: Observation
    probability: float
    verdict: Verdict
    features: np.ndarray

    @property
    def uncertainty(self) -> float:
        return 1.0 - abs(self.probability - 0.5) * 2.0

    @property
    def known(self) -> bool:
        """Has a volunteer already checked it (so it is test data, not new work)?"""
        return self.obs.waterfall_label is not WaterfallLabel.UNKNOWN


class VettingSession:
    """The queue for one sitting, in one of two modes.

    * ``"unvetted"`` -- the real work: observations no volunteer has
      checked.  Scored by a forest trained on every checked observation.
      SatNOGS has no human answer for these, so the verdict is new.
    * ``"test"`` -- observations volunteers *have* checked, each scored by
      a forest that never saw its station.  Here the volunteer's answer
      exists, so this mode measures how often the vetter is right; it is
      where the "error when sure" figure comes from, in both modes.
    """

    ORDERS = ("unsure first", "newest first", "disagreements", "not decided")
    MODES = ("unvetted", "test")

    def __init__(self, items: Sequence[Item], cache: ArtifactCache, explainer: Explainer,
                 policy: Policy = Policy(), log: DecisionLog | None = None, model_name: str = "",
                 model: FeatureVetter | None = None, bundle=None) -> None:
        if not items and bundle is None:
            raise ValidationError("items", 0, "nothing to vet")
        self._model = model
        self._bundle = bundle
        # From a saved bundle there is no test set on disk: one queue, "latest".
        self.modes = ("latest",) if bundle is not None else self.MODES
        self._all = list(items)
        self._by_id = {item.obs.id: item for item in self._all}
        self._cache = cache
        self._explainer = explainer
        # "is not None", not "or": an empty log has len() 0, so it is falsy,
        # and "log or ..." silently swapped the file-backed log for a throwaway one.
        self._log = log if log is not None else DecisionLog(None)
        self.model_name = model_name
        self.policy = policy
        if bundle is not None:
            self.mode = "latest"
        else:
            self.mode = "unvetted" if self.count("unvetted") else "test"

    # ------------------------------------------------------------------
    @classmethod
    def from_cache(cls, observations: Iterable[Observation], cache: ArtifactCache,
                   policy: Policy = Policy(), log_path: str | Path | None = None, progress=None,
                   unvetted: Iterable[Observation] = ()) -> "VettingSession":
        labelled = [o for o in observations if o.is_trainable]
        store = FeatureStore(cache)
        kept, features = store.get(labelled, progress)
        if len(kept) < 20:
            raise ValidationError("observations", len(kept), "need at least 20 downloaded, labelled waterfalls")
        targets = np.array([1 if o.waterfall_label is WaterfallLabel.SIGNAL else 0 for o in kept])
        probability = out_of_fold(features, targets, [o.ground_station for o in kept])
        model = FeatureVetter()
        model.fit_features(features, targets)
        explainer = Explainer(features, targets, model.importances())
        items = [Item(o, float(p), policy.verdict(float(p)), f) for o, p, f in zip(kept, probability, features)]

        known = {o.id for o in kept}
        fresh = [o for o in unvetted if o.id not in known and o.waterfall_label is WaterfallLabel.UNKNOWN]
        fresh_kept, fresh_features = store.get(fresh, progress)
        if fresh_kept:
            fresh_p = model.predict_proba_features(fresh_features)
            items += [Item(o, float(p), policy.verdict(float(p)), f)
                      for o, p, f in zip(fresh_kept, fresh_p, fresh_features)]
        return cls(items, cache, explainer, policy, DecisionLog(log_path),
                   model_name=f"Forest on {len(FEATURES)} measurements, trained on {len(kept)} volunteer-checked "
                              f"observations", model=model)

    @classmethod
    def from_bundle(cls, bundle, observations: Iterable[Observation], cache: ArtifactCache,
                    policy: Policy = Policy(), log_path: str | Path | None = None, progress=None) -> "VettingSession":
        """The app's normal start: a saved, trained vetter and whatever recordings are at hand."""
        session = cls([], cache, bundle.explainer, policy, DecisionLog(log_path),
                      model_name=f"Forest on {len(FEATURES)} measurements, trained on {bundle.n_train:,} "
                                 f"volunteer-checked observations", model=bundle.model, bundle=bundle)
        session.add(list(observations), progress)
        return session

    def add(self, observations: Sequence[Observation], progress=None) -> list[Item]:
        """Score new recordings and put them in the queue; returns the ones added."""
        if self._model is None:
            raise ValidationError("model", None, "this session has no model to score new recordings with")
        fresh = [o for o in observations if o.id not in self._by_id]
        kept, features = FeatureStore(self._cache).get(fresh, progress)
        if not kept:
            return []
        probability = self._model.predict_proba_features(features)
        if self._bundle is not None:
            probability = self._bundle.calibrate(probability)
        items = [Item(o, float(p), self.policy.verdict(float(p)), f) for o, p, f in zip(kept, probability, features)]
        self._all.extend(items)
        self._by_id.update({i.obs.id: i for i in items})
        return items

    # ------------------------------------------------------------------
    @property
    def cache(self) -> ArtifactCache:
        return self._cache

    @property
    def all_items(self) -> list[Item]:
        """Both modes together (read-only use: the home screen's picks)."""
        return list(self._all)

    def count(self, mode: str) -> int:
        if mode == "latest":
            return len(self._all)
        return sum(1 for i in self._all if (i.known if mode == "test" else not i.known))

    def set_mode(self, mode: str) -> None:
        if mode not in self.modes:
            raise ValidationError("mode", mode, f"one of {self.modes}")
        self.mode = mode

    @property
    def _items(self) -> list[Item]:
        if self.mode == "latest":
            return list(self._all)
        test = self.mode == "test"
        return [i for i in self._all if i.known == test]

    def __len__(self) -> int:
        return len(self._items)

    def item(self, obs_id: int) -> Item:
        return self._by_id[obs_id]

    def set_policy(self, policy: Policy) -> None:
        self.policy = policy
        self._all = [Item(i.obs, i.probability, policy.verdict(i.probability), i.features) for i in self._all]
        self._by_id = {item.obs.id: item for item in self._all}

    def ordered(self, order: str = "unsure first") -> list[Item]:
        if order not in self.ORDERS:
            raise ValidationError("order", order, f"one of {self.ORDERS}")
        items = self._items
        if order == "unsure first":
            return sorted(items, key=lambda i: (self.decision(i.obs.id) is not None, -i.uncertainty))
        if order == "newest first":
            return sorted(items, key=lambda i: i.obs.start or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        if order == "disagreements":
            # Test mode: the model against the volunteer.  Unvetted: the model against you.
            def other(i: Item):
                if i.known:
                    return i.obs.waterfall_label
                decision = self.decision(i.obs.id)
                return decision.label if decision is not None and decision.by == "you" else None
            return sorted((i for i in items if i.verdict.label is not None and other(i) not in (None, i.verdict.label)),
                          key=lambda i: -abs(i.probability - 0.5))
        return [i for i in items if self.decision(i.obs.id) is None]

    def reasons(self, obs_id: int, count: int = 4) -> list[tuple[str, str, bool]]:
        return self._explainer.reasons(self.item(obs_id).features, count)

    def waterfall(self, obs_id: int) -> np.ndarray:
        """The normalised, clipped waterfall, ready to draw."""
        return np.clip(normalise(self._cache.waterfall(self.item(obs_id).obs).power), *CLIP)

    # ------------------------------------------------------------------
    def decision(self, obs_id: int) -> Decision | None:
        return self._log.get(obs_id)

    def decide(self, obs_id: int, label: WaterfallLabel, by: str = "you") -> Decision:
        if label is WaterfallLabel.UNKNOWN:
            raise ValidationError("label", label, "decide signal or no signal")
        item = self.item(obs_id)
        decision = Decision(obs_id, label, by, item.probability, datetime.now(timezone.utc).isoformat())
        self._log.record(decision)
        return decision

    def auto_vet(self) -> int:
        """Record the model's answer for every confident observation in this mode not yet decided."""
        done = 0
        for item in self._items:
            if item.verdict.label is not None and self.decision(item.obs.id) is None:
                self.decide(item.obs.id, item.verdict.label, by="auto")
                done += 1
        return done

    def measured_error(self) -> tuple[float, int]:
        """``(share wrong, how many)`` of confident answers on the volunteer-checked set, at this policy."""
        if self._bundle is not None:
            return self._bundle.measured_error(self.policy)
        known = [i for i in self._all if i.known and i.verdict is not Verdict.UNSURE]
        wrong = sum(i.verdict.label is not i.obs.waterfall_label for i in known)
        return (wrong / len(known) if known else float("nan")), len(known)

    def stats(self) -> dict[str, float]:
        """Counts for the top bar (this mode), and the error measured on the checked set."""
        items = self._items
        decided = [self._log.get(i.obs.id) for i in items]
        auto = [(d, i) for d, i in zip(decided, items) if d is not None and d.by == "auto"]
        yours = [(d, i) for d, i in zip(decided, items) if d is not None and d.by == "you"]
        confident = [i for i in items if i.verdict is not Verdict.UNSURE]
        error, measured_on = self.measured_error()
        known_yours = [(d, i) for d, i in yours if i.known]
        return {
            "total": len(items),
            "confident": len(confident),
            "unsure": len(items) - len(confident),
            "auto": len(auto),
            "yours": len(yours),
            "to_check": sum(1 for d, i in zip(decided, items) if d is None and i.verdict is Verdict.UNSURE),
            "automatable_share": len(confident) / len(items) if items else float("nan"),
            "confident_error": error,
            "error_measured_on": measured_on,
            "you_agree_with_volunteers": (float(np.mean([d.label is i.obs.waterfall_label for d, i in known_yours]))
                                          if known_yours else float("nan")),
        }


def track_points(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, columns)`` of the brightest point in each strong moment, for drawing the track."""
    power = np.asarray(power, dtype=np.float32)
    edge = power.shape[1] // 8
    inner = power[:, edge:power.shape[1] - edge]
    strong = np.flatnonzero(inner.max(axis=1) > STRONG)
    return strong, np.argmax(inner[strong], axis=1) + edge


__all__ = ["Decision", "DecisionLog", "Explainer", "FeatureStore", "Item", "Policy", "SHORT", "Verdict",
           "VettingSession", "out_of_fold", "track_points", "feature_matrix"]
