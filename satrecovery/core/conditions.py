"""Channel conditions -- the knobs the user turns in the GUI.

DESIGN NOTE: ENCAPSULATION WITH VALIDATION
------------------------------------------
:class:`Signal` is immutable because nothing should ever edit a captured
waveform.  :class:`ChannelConditions` is the deliberate opposite: it is
*mutable*, because it is bound to GUI sliders that the user drags around.

That makes it the natural place to show the other half of encapsulation.
Every field is a private attribute reached through a ``@property`` pair,
and **every setter validates its input**:

    conditions.snr_db = 500     ->  ValidationError

The object can therefore never hold a nonsensical state, no matter what
the GUI, the API or a test does to it.  Compare that with a plain dict of
settings, where a typo like ``{"snr_dB": 10}`` silently does nothing.
"""

from __future__ import annotations

from typing import Any, Final

from .exceptions import ValidationError

#: Fading models the channel layer knows how to build.
FADING_MODES: Final[frozenset[str]] = frozenset(
    {"none", "flat_rician", "multipath_rician"}
)

#: Sensible limits, chosen to cover the range RML24 spans.
SNR_MIN_DB: Final[float] = -20.0
SNR_MAX_DB: Final[float] = 30.0


class ChannelConditions:
    """The state of the radio channel for one experiment.

    Parameters
    ----------
    snr_db:
        Signal-to-noise ratio in decibels.
    fading:
        One of ``"none"``, ``"flat_rician"`` or ``"multipath_rician"``.
    rician_k_db:
        Rician K-factor in dB -- the power ratio of the direct
        line-of-sight path to the scattered paths.  Large K means a
        clean, nearly line-of-sight link; K = 0 dB means the direct and
        scattered components are equally strong.
    freq_offset_hz:
        Residual carrier frequency error in hertz.  Caused by Doppler
        shift and by oscillator drift at both ends of the link.
    phase_offset_rad:
        Fixed carrier phase error in radians.
    """

    __slots__ = (
        "_snr_db",
        "_fading",
        "_rician_k_db",
        "_freq_offset_hz",
        "_phase_offset_rad",
        "_timing_offset",
    )

    def __init__(
        self,
        snr_db: float = 10.0,
        fading: str = "none",
        rician_k_db: float = 10.0,
        freq_offset_hz: float = 0.0,
        phase_offset_rad: float = 0.0,
        timing_offset: float = 0.0,
    ) -> None:
        # Assigning through the properties means the validation below
        # runs for the constructor too, not just for later edits.
        self.snr_db = snr_db
        self.fading = fading
        self.rician_k_db = rician_k_db
        self.freq_offset_hz = freq_offset_hz
        self.phase_offset_rad = phase_offset_rad
        self.timing_offset = timing_offset

    # ------------------------------------------------------------------
    # snr_db
    # ------------------------------------------------------------------
    @property
    def snr_db(self) -> float:
        """Signal-to-noise ratio in dB."""
        return self._snr_db

    @snr_db.setter
    def snr_db(self, value: float) -> None:
        value = _as_float("snr_db", value)
        if not SNR_MIN_DB <= value <= SNR_MAX_DB:
            raise ValidationError(
                "snr_db", value, f"must be between {SNR_MIN_DB} and {SNR_MAX_DB} dB"
            )
        self._snr_db = value

    @property
    def snr_linear(self) -> float:
        """Signal-to-noise ratio as a plain power ratio.

        A *derived* property: it has no backing field, it is computed on
        demand.  Callers cannot set it, so the dB value stays the single
        source of truth and the two can never disagree.
        """
        return 10.0 ** (self._snr_db / 10.0)

    # ------------------------------------------------------------------
    # fading
    # ------------------------------------------------------------------
    @property
    def fading(self) -> str:
        """Name of the fading model to apply."""
        return self._fading

    @fading.setter
    def fading(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValidationError("fading", value, "must be a string")
        name = value.strip().lower()
        if name not in FADING_MODES:
            raise ValidationError(
                "fading", value, f"must be one of {sorted(FADING_MODES)}"
            )
        self._fading = name

    @property
    def has_fading(self) -> bool:
        """``True`` when a fading model other than ``"none"`` is selected."""
        return self._fading != "none"

    # ------------------------------------------------------------------
    # rician_k_db
    # ------------------------------------------------------------------
    @property
    def rician_k_db(self) -> float:
        """Rician K-factor in dB."""
        return self._rician_k_db

    @rician_k_db.setter
    def rician_k_db(self, value: float) -> None:
        value = _as_float("rician_k_db", value)
        if not -20.0 <= value <= 40.0:
            raise ValidationError(
                "rician_k_db", value, "must be between -20 and 40 dB"
            )
        self._rician_k_db = value

    @property
    def rician_k_linear(self) -> float:
        """Rician K-factor as a plain power ratio."""
        return 10.0 ** (self._rician_k_db / 10.0)

    # ------------------------------------------------------------------
    # freq_offset_hz
    # ------------------------------------------------------------------
    @property
    def freq_offset_hz(self) -> float:
        """Carrier frequency offset in hertz."""
        return self._freq_offset_hz

    @freq_offset_hz.setter
    def freq_offset_hz(self, value: float) -> None:
        value = _as_float("freq_offset_hz", value)
        if abs(value) > 500_000.0:
            raise ValidationError(
                "freq_offset_hz", value, "magnitude must not exceed 500 kHz"
            )
        self._freq_offset_hz = value

    # ------------------------------------------------------------------
    # phase_offset_rad
    # ------------------------------------------------------------------
    @property
    def phase_offset_rad(self) -> float:
        """Carrier phase offset in radians."""
        return self._phase_offset_rad

    @phase_offset_rad.setter
    def phase_offset_rad(self, value: float) -> None:
        self._phase_offset_rad = _as_float("phase_offset_rad", value)

    # ------------------------------------------------------------------
    # timing_offset
    # ------------------------------------------------------------------
    @property
    def timing_offset(self) -> float:
        """Sampling clock misalignment, in samples.

        Fractional values are the realistic case: the receiver's clock is
        never exactly aligned with the transmitter's, and the error
        drifts. Half a sample out is enough to cost real signal quality.
        """
        return self._timing_offset

    @timing_offset.setter
    def timing_offset(self, value: float) -> None:
        value = _as_float("timing_offset", value)
        if abs(value) > 8.0:
            raise ValidationError(
                "timing_offset", value, "magnitude must not exceed 8 samples"
            )
        self._timing_offset = value

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------
    def clone(self, **overrides: Any) -> "ChannelConditions":
        """Return a copy, optionally changing some fields.

        Used heavily by the benchmark runner, which sweeps one field at a
        time::

            [base.clone(snr_db=s) for s in range(-10, 21, 2)]
        """
        settings = self.to_dict()
        unknown = set(overrides) - set(settings)
        if unknown:
            raise ValidationError(
                "overrides", sorted(unknown), "not valid condition fields"
            )
        settings.update(overrides)
        return ChannelConditions(**settings)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dictionary form, for the JSON API and the results store."""
        return {
            "snr_db": self._snr_db,
            "fading": self._fading,
            "rician_k_db": self._rician_k_db,
            "freq_offset_hz": self._freq_offset_hz,
            "phase_offset_rad": self._phase_offset_rad,
            "timing_offset": self._timing_offset,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChannelConditions":
        """Rebuild from :meth:`to_dict` output, validating as it goes."""
        if not isinstance(data, dict):
            raise ValidationError("data", type(data).__name__, "expected a dict")
        allowed = {
            "snr_db",
            "fading",
            "rician_k_db",
            "freq_offset_hz",
            "phase_offset_rad",
            "timing_offset",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValidationError("data", sorted(unknown), "unexpected keys")
        return cls(**data)

    def describe(self) -> str:
        """One-line summary for GUI status bars and plot titles."""
        parts = [f"SNR {self._snr_db:g} dB"]
        if self.has_fading:
            label = self._fading.replace("_", " ")
            parts.append(f"{label} (K={self._rician_k_db:g} dB)")
        else:
            parts.append("no fading")
        if self._freq_offset_hz:
            parts.append(f"df {self._freq_offset_hz:g} Hz")
        if self._phase_offset_rad:
            parts.append(f"dphi {self._phase_offset_rad:.2f} rad")
        if self._timing_offset:
            parts.append(f"dt {self._timing_offset:+.2f} sa")
        return ", ".join(parts)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChannelConditions):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"ChannelConditions({fields})"


def _as_float(field: str, value: Any) -> float:
    """Coerce to float, turning type errors into our own exception type."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(field, value, "must be a number") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise ValidationError(field, value, "must be finite")
    return result
