"""The shapes of requests and responses -- the API's public contract.

These classes describe what crosses the wire, and nothing else.  They
are deliberately *not* the domain objects: ``ConditionsIn`` is not a
:class:`~satrecovery.core.conditions.ChannelConditions`, it converts to
one.  Keeping the two apart means the internal classes can change shape
without breaking every client, and the API can change without touching
the science.

DESIGN NOTE: DATA TRANSFER OBJECTS, AND ONE SOURCE OF TRUTH FOR RULES
----------------------------------------------------------------------
Pydantic checks *types* here -- a number is a number, a list has at
least one item -- because that is what produces the readable interactive
documentation at ``/docs``.  The *domain* rules -- an SNR between -20
and 30 dB, a known fading model -- are not repeated.  The conversion to
a domain object enforces them, and the API translates the domain's own
:class:`ValidationError` into an HTTP 422.  So each rule is written
exactly once, in the class it belongs to, and the API cannot drift out
of step with it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..core.conditions import ChannelConditions


class _Strict(BaseModel):
    """Base for request bodies: unknown fields are an error, not ignored.

    A misspelt field silently ignored would run the experiment with the
    default value and return a confident answer to a question nobody
    asked.  Rejecting it is the only safe behaviour.
    """

    model_config = ConfigDict(extra="forbid")


# ======================================================================
# Requests
# ======================================================================
class ConditionsIn(_Strict):
    """Everything about the channel, as the user sets it in the GUI."""

    snr_db: float = Field(10.0, description="Signal-to-noise ratio in dB (-20 to 30).")
    fading: Literal["none", "flat_rician", "multipath_rician"] = "none"
    rician_k_db: float = Field(10.0, description="Direct-to-scattered power ratio in dB.")
    freq_offset_hz: float = Field(0.0, description="Doppler / carrier offset in Hz.")
    phase_offset_rad: float = Field(0.0, description="Carrier phase offset in radians.")
    timing_offset: float = Field(0.0, description="Sampling clock offset in samples.")

    def to_domain(self) -> ChannelConditions:
        """Build the domain object, which applies the real validation."""
        return ChannelConditions(**self.model_dump())


class ExperimentRequest(_Strict):
    """One frame, one channel, several receivers -- the GUI's main action."""

    modulation: str = Field("QPSK", examples=["QPSK", "16QAM"])
    conditions: ConditionsIn = Field(default_factory=ConditionsIn)
    receivers: list[str] = Field(
        default_factory=lambda: ["classical"], min_length=1, max_length=8,
        examples=[["classical", "hybrid"]],
    )
    seed: int | None = Field(
        None, ge=0, description="Fix to reproduce a frame exactly; omitted means random.",
    )


class StagesRequest(_Strict):
    """One frame, followed through every stage of one receiver."""

    modulation: str = "QPSK"
    conditions: ConditionsIn = Field(default_factory=ConditionsIn)
    receiver: str = "classical"
    seed: int | None = Field(None, ge=0)


class BenchmarkRequest(_Strict):
    """A grid of scenarios to run every listed receiver on."""

    name: str = Field(..., min_length=1, max_length=100)
    modulations: list[str] = Field(..., min_length=1, max_length=6)
    snrs_db: list[float] = Field(..., min_length=1, max_length=40)
    profiles: list[str] = Field(default_factory=lambda: ["clean"], min_length=1, max_length=5)
    receivers: list[str] = Field(..., min_length=1, max_length=8)
    frames_per_scenario: int = Field(20, ge=1)
    seed: int = Field(1, ge=0)
    notes: str = Field("", max_length=1000)


class TrainRequest(_Strict):
    """How to train a hybrid demapper. Every field has a sensible default."""

    snrs_db: list[float] = Field(default_factory=lambda: [0, 4, 8, 12, 16], min_length=1, max_length=20)
    profiles: list[str] = Field(
        default_factory=lambda: ["clean", "doppler", "fading", "full"], min_length=1, max_length=4,
    )
    frames_each: int = Field(18, ge=1, le=100)
    epochs: int = Field(200, ge=1, le=1000)
    patience: int = Field(12, ge=1, le=100)


# ======================================================================
# Responses
# ======================================================================
class IQ(BaseModel):
    """Complex samples, split into in-phase and quadrature lists."""

    i: list[float]
    q: list[float]


class Histogram(BaseModel):
    edges: list[float]
    counts: list[int]


class FrameOut(BaseModel):
    modulation: str
    conditions: dict[str, Any]
    seed: int
    n_symbols: int
    n_bits: int
    samples_per_symbol: int
    sample_rate: float
    received: IQ
    transmitted: IQ = Field(description="The clean waveform before the channel touched it.")
    tx_symbols: IQ
    constellation: IQ


class ReceiverResultOut(BaseModel):
    receiver: str
    label: str
    n_bits: int
    bit_errors: int
    ber: float
    ber_low: float
    ber_high: float
    mean_confidence: float
    ece: float
    ranking_quality: float | None = Field(
        description="None when the frame has no errors: there is nothing to rank.",
    )
    evm_percent: float | None
    symbols: IQ
    symbol_wrong: list[bool]
    confidence_histogram: Histogram
    seconds: float
    diagnostics: dict[str, Any]


class ExperimentResponse(BaseModel):
    frame: FrameOut
    results: list[ReceiverResultOut]


class StageOut(BaseModel):
    name: str
    rate: Literal["sample", "symbol"]
    samples: IQ


class StagesResponse(BaseModel):
    frame: FrameOut
    stages: list[StageOut]
    result: ReceiverResultOut


class ReceiverInfo(BaseModel):
    name: str
    label: str
    description: str
    needs_model: bool
    available_for: list[str]


class ModulationInfo(BaseModel):
    name: str
    bits_per_symbol: int
    order: int
    constellation: IQ


class ProfileInfo(BaseModel):
    name: str
    settings: dict[str, Any]


class JobOut(BaseModel):
    id: str
    kind: str
    description: str
    state: Literal["queued", "running", "done", "failed"]
    done: int
    total: int
    fraction: float
    message: str
    result: Any = None
    error: str | None = None
    created: float
    elapsed_seconds: float | None = None


class RunOut(BaseModel):
    id: int
    name: str
    created: float
    source: str
    notes: str
    cells: int


class ErrorOut(BaseModel):
    error: str
    detail: str
    field: str | None = None


__all__ = [
    "BenchmarkRequest", "ConditionsIn", "ErrorOut", "ExperimentRequest",
    "ExperimentResponse", "FrameOut", "Histogram", "IQ", "JobOut",
    "ModulationInfo", "ProfileInfo", "ReceiverInfo", "ReceiverResultOut",
    "RunOut", "StageOut", "StagesRequest", "StagesResponse", "TrainRequest",
]
