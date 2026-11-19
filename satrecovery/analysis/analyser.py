"""The two-step answer: what is this signal, and what does it say?

Step 1, the **AI classifier** names the modulation.  Step 2, a
**receiver** recovers the bits, decoding with the modulation step 1
chose -- the hybrid if its network has been trained for that scheme,
the classical receiver otherwise.

Besides the bits, the answer carries everything the app needs to
*animate* the recovery: the signal after each stage of the receiver,
each reduced to one point per symbol so that every point can glide
smoothly from where it was at one stage to where it is at the next.
Each point is coloured by the symbol it is finally decided to be, so
the animation shows a jumble of colours sorting itself into clean,
single-coloured clusters.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..backend.catalog import ReceiverCatalog
from ..core.exceptions import ValidationError
from ..modulation.registry import ModulationRegistry
from .classifier import Classification, ModulationClassifier
from .inputs import CHECKABLE, SignalInput
from .trust import TrustCheck

#: What each stage is called and does, in words a newcomer can follow.
STAGES: dict[str, tuple[str, str]] = {
    "received": ("What arrived",
                 "The signal as the dish picked it up: noise, drift and distortion all mixed in."),
    "CoarseFreq": ("Remove Doppler",
                   "The satellite's motion shifted the frequency, which makes the points spin. "
                   "The receiver measures the shift and takes it out."),
    "MatchedFilter": ("Filter noise",
                      "A filter shaped like the transmitted pulse keeps the signal and throws away "
                      "most of the noise."),
    "GardnerTiming": ("Find timing",
                      "Pick the one instant per symbol where each symbol is clearest. The smear turns into points."),
    "AGC": ("Fix size",
            "Scale the points so they line up with where they should be."),
    "CarrierPLL": ("Lock phase",
                   "Hold the constellation steady. Each point now sits in its cluster, and is read as bits."),
    "LMSEqualizer": ("Undo echoes",
                     "A short adaptive filter separates symbols that echoes smeared together."),
}


class Analyser:
    """Classify, then recover, and package the result for the app."""

    def __init__(self, classifier: ModulationClassifier, catalog: ReceiverCatalog) -> None:
        if not isinstance(classifier, ModulationClassifier):
            raise ValidationError("classifier", type(classifier).__name__, "expected a ModulationClassifier")
        self._classifier = classifier
        self._catalog = catalog

    @property
    def classifier(self) -> ModulationClassifier:
        return self._classifier

    def classify(self, signal: SignalInput) -> Classification:
        return self._classifier.classify(signal.samples)

    def receiver_for(self, modulation: str):
        """The hybrid when its network is trained for this scheme, else classical."""
        hybrid_ready = "hybrid" in self._catalog and self._catalog.is_available("hybrid", modulation)
        name = "hybrid" if hybrid_ready else "classical"
        return name, self._catalog.build(name, modulation)

    #: A known check pattern of this many symbols ...
    CHECK_SYMBOLS = 8
    #: ... at the start of every block of this many graded symbols.
    CHECK_EVERY = 32

    def _confirm_orientation(self, receiver, frame, scheme, result, snapshots):
        """Make sure the constellation is the right way up, all the way through.

        The receivers decide which way up the constellation is from a
        short known preamble at the very start of the frame, then trust
        their phase loop for the rest.  On simulated frames that is fine.
        On real recordings the loop can *slip* -- lock one position round
        and stay there -- and every symbol after the slip comes out wrong:

        * #187,269 (8PSK, +20 dB): slipped at the very start, where the
          recording was cut; 487 of 1,461 bits wrong.
        * #184,700 (8PSK, +18 dB): slipped at symbol 38, chasing a
          25 Hz frequency error left by the coarse estimate; 460 of
          1,458 bits wrong.  A check only at the start missed it; with
          patterns every 32 symbols, about 10 bits remain wrong -- the
          symbols right at the slip, before the next pattern.

        Real satellite standards (DVB-S2's pilot blocks) insert a short
        known pattern *at regular intervals*, so a receiver can confirm
        its orientation again and again.  This does the same: every
        ``CHECK_EVERY`` graded symbols, the first ``CHECK_SYMBOLS`` are
        compared with what was sent, in every orientation the
        constellation allows, and each symbol is turned to the
        orientation of its nearest pattern.  A slip then costs at most
        half the gap between patterns, not the rest of the frame.

        Only recordings need it (frames with ungraded edges); simulated
        frames are returned unchanged.  Returns the result, the
        snapshots, and how many stretches had to be turned.
        """
        graded = frame.graded_symbols
        sent = frame.tx_symbols
        if graded is None or sent is None or not hasattr(receiver, "demapper"):
            return result, snapshots, 0
        final_name, final = snapshots[-1]
        points = np.asarray(final.samples)
        positions = np.flatnonzero(np.asarray(graded, dtype=bool))
        if points.size != sent.size or positions.size == 0:
            return result, snapshots, 0
        m = scheme.rotational_symmetry
        sent = np.asarray(sent)
        turned = points.copy()
        corrected = 0
        block_starts = positions[:: self.CHECK_EVERY]
        # The orientation each pattern says is right.
        best_turns, centres = [], []
        for b in range(len(block_starts)):
            pattern = positions[b * self.CHECK_EVERY : b * self.CHECK_EVERY + self.CHECK_SYMBOLS]
            expected = scheme.nearest_indices(sent[pattern])
            agreement = [
                int(np.sum(scheme.nearest_indices(points[pattern] * np.exp(-2j * np.pi * k / m)) == expected))
                for k in range(m)
            ]
            best = int(np.argmax(agreement))
            best_turns.append(best if agreement[best] > agreement[0] else 0)
            centres.append(float(np.mean(pattern)))
        # Every symbol takes the orientation of the *nearest* pattern, so a
        # slip between two patterns costs at most half the gap.
        nearest = np.argmin(np.abs(np.arange(points.size)[:, None] - np.asarray(centres)[None, :]), axis=1)
        per_symbol = np.asarray(best_turns)[nearest]
        turned = points * np.exp(-2j * np.pi * per_symbol / m)
        corrected = int(np.count_nonzero(best_turns))

        # Between the patterns the phase can still drift (#184,700: a
        # 25 Hz error left by the coarse estimate).  Measure the remaining
        # phase at every pattern and take out a smooth line through those
        # measurements -- pilot-aided tracking, as DVB-S2 receivers do.
        phases = []
        for b in range(len(block_starts)):
            pattern = positions[b * self.CHECK_EVERY : b * self.CHECK_EVERY + self.CHECK_SYMBOLS]
            phases.append(float(np.angle(np.sum(turned[pattern] * np.conj(sent[pattern])))))
        drift = 0.0
        if len(phases) >= 2:
            line = np.interp(np.arange(points.size), centres, np.unwrap(phases))
            if np.max(np.abs(line)) > np.radians(3):
                turned = turned * np.exp(-1j * line)
                drift = float(np.degrees(np.max(np.abs(line))))
        if corrected == 0 and drift == 0.0:
            return result, snapshots, 0
        bits, confidence, llr = receiver.demapper.demap(turned, scheme)
        from ..core.signal import Signal
        from ..receivers.result import RecoveryResult

        diagnostics = dict(result.diagnostics)
        diagnostics["orientation_corrected_blocks"] = corrected
        diagnostics["pilot_phase_correction_deg"] = round(drift, 1)
        fixed = RecoveryResult(bits=bits, confidence=confidence, symbols=turned, modulation=scheme.name,
                               receiver=result.receiver, llr=llr, diagnostics=diagnostics)
        fixed = fixed.select_symbols(graded, scheme.bits_per_symbol)
        snapshots = list(snapshots[:-1]) + [(final_name, Signal(turned, final.sample_rate))]
        return fixed, snapshots, corrected

    def analyse(self, signal: SignalInput) -> dict[str, Any]:
        if not isinstance(signal, SignalInput):
            raise ValidationError("signal", type(signal).__name__, "expected a SignalInput")
        started = time.perf_counter()

        # -- step 1: the AI classifier --------------------------------
        classification = self.classify(signal)
        modulation = classification.label
        scheme = ModulationRegistry.create(modulation)

        # -- step 2: recovery, with the classifier's answer -------------
        frame, checkable, note = signal.frame(modulation)
        receiver_name, receiver = self.receiver_for(modulation)
        result, snapshots = receiver.trace(frame)
        result, snapshots, turned = self._confirm_orientation(receiver, frame, scheme, result, snapshots)

        # The animation follows the receiver's own symbol instants.
        sps = frame.samples_per_symbol
        n_symbols = frame.signal.samples.size // sps
        tau = float(result.diagnostics.get("timing_offset_samples") or 0.0)
        instants = np.clip(np.round(np.arange(n_symbols) * sps + tau).astype(int), 0, frame.signal.samples.size - 1)

        final = np.asarray(snapshots[-1][1].samples)
        colour_index = scheme.nearest_indices(final[:n_symbols]) if final.size >= n_symbols else np.zeros(n_symbols, int)
        stages = []
        for name, stage_signal in snapshots:
            values = np.asarray(stage_signal.samples)
            per_sample = values.size == frame.signal.samples.size
            points = values[instants] if per_sample else values[:n_symbols]
            cloud = values[:: max(1, values.size // 1536)] if per_sample else None
            scale = float(np.sqrt(np.mean(np.abs(points) ** 2))) or 1.0
            title, text = STAGES.get(name, (name, ""))
            stages.append({
                "name": name,
                "title": title,
                "text": text,
                "per_sample": per_sample,
                # The stage as the receiver produced it, for the wave plot
                # and the constellation plot of the original app.
                "rate": "sample" if per_sample else "symbol",
                "samples": _iq(values),
                "points": _iq(points / scale),
                "cloud": None if cloud is None else _iq(cloud / scale),
            })

        bits = np.asarray(result.bits, dtype=np.uint8)
        confidence = np.asarray(result.confidence, dtype=float)
        check: dict[str, Any] = {"available": False, "detail": note}
        wrong_symbols: list[bool] | None = None
        if checkable and frame.n_bits == result.n_bits and frame.n_bits > 0:
            # The check-pattern symbols are *known* to the receiver -- they
            # are how it confirms its orientation -- so counting them as
            # correctly recovered would flatter it.  They are left out of
            # the score, exactly as a real link does not count its pilots
            # as data.
            bit_wrong = result.error_mask(frame.bits).reshape(-1, scheme.bits_per_symbol)
            scored = np.ones(bit_wrong.shape[0], dtype=bool)
            if frame.graded_symbols is not None:
                scored = (np.arange(scored.size) % self.CHECK_EVERY) >= self.CHECK_SYMBOLS
            errors = int(bit_wrong[scored].sum())
            n_scored = int(scored.sum() * scheme.bits_per_symbol)
            check = {"available": True, "bit_errors": errors, "n_bits": n_scored,
                     "ber": errors / max(n_scored, 1), "detail": note,
                     "pilot_bits_excluded": int(frame.n_bits - n_scored)}
            # Which symbols were read wrongly, on the whole symbol grid, so
            # the plot can mark them with red crosses.
            per_symbol = bit_wrong.any(axis=1)
            graded = (np.ones(n_symbols, dtype=bool) if frame.graded_symbols is None
                      else np.asarray(frame.graded_symbols, dtype=bool))
            mask = np.zeros(n_symbols, dtype=bool)
            if int(graded.sum()) == per_symbol.size:
                mask[np.flatnonzero(graded)] = per_symbol
                wrong_symbols = mask.tolist()
        reference = frame.tx_symbols

        if check.get("pilot_bits_excluded"):
            check["detail"] = (check.get("detail", "") + f" The {check['pilot_bits_excluded']:,} bits of the known check "
                               f"pattern ({self.CHECK_SYMBOLS} symbols in every {self.CHECK_EVERY}) are not counted: "
                               "the receiver was told those.").strip()
        if turned:
            check["detail"] = (check.get("detail", "") + f" The receiver's phase loop slipped one or more positions "
                               f"round; the check pattern caught it in {turned} "
                               f"stretch{'es' if turned > 1 else ''} and turned the constellation back.").strip()
        return {
            "input": signal.describe(),
            "classification": {**classification.as_dict(),
                               "correct": modulation == signal.true_modulation,
                               "true_label": signal.true_modulation},
            "recovery": {
                "receiver": receiver_name,
                "modulation": modulation,
                "stages": stages,
                "colours": colour_index.astype(int).tolist(),
                "constellation": _iq(scheme.constellation / np.sqrt(np.mean(np.abs(scheme.constellation) ** 2))),
                "n_bits": int(bits.size),
                "bits_preview": "".join(map(str, bits[:96].tolist())),
                "mean_confidence": float(confidence.mean()) if confidence.size else 0.0,
                "trusted_fraction": float(np.mean(confidence >= 0.95)) if confidence.size else 0.0,
                "check": check,
                "checkable_types": list(CHECKABLE),
                "wrong_symbols": wrong_symbols,
                "samples_per_symbol": int(sps),
                "reference_symbols": None if reference is None or not checkable else _iq(reference),
                "constellation_raw": _iq(scheme.constellation),
            },
            # How far to trust the AI's answer: confident, unsure or unfamiliar.
            "trust": TrustCheck.assess(classification.as_dict()["probabilities"], signal.samples, signal.symbol_rate,
                                       float(np.mean(confidence >= 0.95)) if confidence.size else None).as_dict(),
            "seconds": round(time.perf_counter() - started, 3),
        }


def _iq(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.complex128).ravel()
    return {"i": np.round(values.real, 5).tolist(), "q": np.round(values.imag, 5).tolist()}


__all__ = ["Analyser", "STAGES"]
