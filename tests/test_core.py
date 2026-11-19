"""Tests for the core domain objects.

These tests are deliberately written as statements about the *design*,
not just about the arithmetic.  Each one pins down a specific
encapsulation guarantee, so that a later refactor cannot quietly remove
it: ``Signal`` stays immutable, ``ChannelConditions`` stays impossible to
put into an invalid state, and ``Frame`` keeps its ground-truth bits
safe from the code that is supposed to be guessing them.
"""

from __future__ import annotations

import numpy as np
import pytest

from satrecovery import ChannelConditions, Frame, Signal
from satrecovery.core.exceptions import (
    EmptySignalError,
    SatRecoveryError,
    ValidationError,
)


# ==========================================================================
# Signal: encapsulation and immutability
# ==========================================================================
def test_samples_view_is_not_writable():
    """The headline encapsulation guarantee of the whole project.

    A frame passes through several receivers in turn.  If any of them
    could write into the shared sample array, the receiver that ran next
    would be measured on different data -- and the comparison the project
    exists to make would be invalid.  Making that impossible beats
    remembering not to do it.
    """
    signal = Signal(np.array([1 + 1j, 2 - 1j]), sample_rate=1000.0)
    with pytest.raises(ValueError):
        signal.samples[0] = 0.0


def test_constructor_copies_its_input():
    """Mutating the caller's array must not change the Signal."""
    source = np.array([1 + 0j, 2 + 0j, 3 + 0j])
    signal = Signal(source, sample_rate=1.0)
    source[0] = 999.0
    assert signal.samples[0] == 1 + 0j


def test_with_samples_returns_a_new_object():
    """The replacement for a setter: transform by producing a new value."""
    original = Signal(np.ones(4, dtype=complex), sample_rate=10.0)
    modified = original.with_samples(np.zeros(4, dtype=complex))

    assert modified is not original
    assert original.power() == pytest.approx(1.0)
    assert modified.power() == pytest.approx(0.0)
    assert modified.sample_rate == original.sample_rate


def test_real_input_is_promoted_to_complex():
    signal = Signal(np.array([1.0, -1.0, 1.0]), sample_rate=1.0)
    assert signal.samples.dtype == np.complex128
    assert signal.power() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("samples", "sample_rate", "exception"),
    [
        (np.array([]), 1.0, EmptySignalError),
        (np.ones((2, 2)), 1.0, ValidationError),
        (np.array([1.0, np.nan]), 1.0, ValidationError),
        (np.array([1.0, np.inf]), 1.0, ValidationError),
        (np.ones(4), 0.0, ValidationError),
        (np.ones(4), -5.0, ValidationError),
    ],
)
def test_invalid_signals_are_rejected(samples, sample_rate, exception):
    with pytest.raises(exception):
        Signal(samples, sample_rate)


def test_every_project_error_shares_one_root():
    """Inheritance in the exception hierarchy, as callers rely on it."""
    with pytest.raises(SatRecoveryError):
        Signal(np.array([]), 1.0)


def test_derived_properties_are_consistent():
    signal = Signal(np.ones(100, dtype=complex), sample_rate=50.0)
    assert len(signal) == 100
    assert signal.duration == pytest.approx(2.0)
    assert signal.peak_amplitude() == pytest.approx(1.0)


def test_normalized_reaches_the_target_power():
    signal = Signal(np.full(64, 3.0 + 4.0j), sample_rate=1.0)
    assert signal.power() == pytest.approx(25.0)
    assert signal.normalized().power() == pytest.approx(1.0)
    assert signal.normalized(4.0).power() == pytest.approx(4.0)


def test_iq_array_round_trip():
    """The hand-off format between the signal world and the models."""
    original = Signal(np.array([1 + 2j, -3 + 0.5j]), sample_rate=100.0)
    iq = original.as_iq_array()

    assert iq.shape == (2, 2)
    assert np.allclose(iq[0], [1.0, -3.0])
    assert np.allclose(iq[1], [2.0, 0.5])
    assert Signal.from_iq_array(iq, 100.0) == original


def test_slicing_yields_a_sub_signal():
    signal = Signal(np.arange(10) + 0j, sample_rate=1.0)
    piece = signal[2:5]
    assert isinstance(piece, Signal)
    assert len(piece) == 3
    assert isinstance(signal[0], complex)


def test_slots_block_stray_attributes():
    """Encapsulation: the field list belongs to the class, not its users."""
    signal = Signal(np.ones(2, dtype=complex), sample_rate=1.0)
    with pytest.raises(AttributeError):
        signal.snr = 10.0  # type: ignore[attr-defined]


# ==========================================================================
# ChannelConditions: validating setters
# ==========================================================================
def test_setters_validate_on_assignment():
    conditions = ChannelConditions()
    with pytest.raises(ValidationError):
        conditions.snr_db = 500.0
    with pytest.raises(ValidationError):
        conditions.fading = "rayleigh"
    with pytest.raises(ValidationError):
        conditions.snr_db = "quiet"  # type: ignore[assignment]


def test_constructor_validates_through_the_same_setters():
    """One validation path, used by both the constructor and later edits."""
    with pytest.raises(ValidationError):
        ChannelConditions(snr_db=-999.0)


def test_object_state_survives_a_rejected_assignment():
    """A failed edit must leave the old value intact, not a half-written one."""
    conditions = ChannelConditions(snr_db=12.0)
    with pytest.raises(ValidationError):
        conditions.snr_db = 1e9
    assert conditions.snr_db == 12.0


def test_fading_name_is_normalised():
    conditions = ChannelConditions(fading="  FLAT_Rician ")
    assert conditions.fading == "flat_rician"
    assert conditions.has_fading


def test_derived_values_cannot_drift_from_their_source():
    """``snr_linear`` is computed, so it is always consistent with ``snr_db``."""
    conditions = ChannelConditions(snr_db=20.0)
    assert conditions.snr_linear == pytest.approx(100.0)

    conditions.snr_db = 10.0
    assert conditions.snr_linear == pytest.approx(10.0)

    with pytest.raises(AttributeError):
        conditions.snr_linear = 3.0  # type: ignore[misc]


def test_clone_changes_only_what_it_is_asked_to():
    """What makes a benchmark sweep interpretable: one variable at a time."""
    base = ChannelConditions(
        snr_db=5.0, fading="multipath_rician", freq_offset_hz=200.0
    )
    swept = base.clone(snr_db=15.0)

    assert swept.snr_db == 15.0
    assert swept.fading == "multipath_rician"
    assert swept.freq_offset_hz == 200.0
    assert base.snr_db == 5.0


def test_clone_rejects_unknown_fields():
    """Catches the classic silent-typo bug that a settings dict would allow."""
    with pytest.raises(ValidationError):
        ChannelConditions().clone(snr_dB=10.0)


def test_dict_round_trip():
    original = ChannelConditions(
        snr_db=3.5,
        fading="flat_rician",
        rician_k_db=6.0,
        freq_offset_hz=-120.0,
        phase_offset_rad=0.4,
    )
    assert ChannelConditions.from_dict(original.to_dict()) == original


def test_from_dict_rejects_unexpected_keys():
    with pytest.raises(ValidationError):
        ChannelConditions.from_dict({"snr_db": 10.0, "weather": "cloudy"})


def test_describe_mentions_active_impairments_only():
    quiet = ChannelConditions(snr_db=10.0).describe()
    assert "no fading" in quiet
    assert "df" not in quiet

    busy = ChannelConditions(
        snr_db=2.0, fading="multipath_rician", freq_offset_hz=50.0
    ).describe()
    assert "multipath rician" in busy
    assert "df 50" in busy


# ==========================================================================
# Frame
# ==========================================================================
def _frame(n_bits: int = 8, n_samples: int = 16) -> Frame:
    return Frame(
        signal=Signal(np.ones(n_samples, dtype=complex), sample_rate=1000.0),
        bits=np.zeros(n_bits, dtype=np.uint8),
        modulation="QPSK",
        conditions=ChannelConditions(snr_db=6.0),
        samples_per_symbol=4,
    )


def test_ground_truth_bits_are_read_only():
    """The bits are the answer key; nothing downstream may edit them."""
    frame = _frame()
    with pytest.raises(ValueError):
        frame.bits[0] = 1


def test_frame_rejects_non_binary_bits():
    with pytest.raises(ValidationError):
        Frame(
            signal=Signal(np.ones(4, dtype=complex), sample_rate=1.0),
            bits=np.array([0, 1, 5, 0], dtype=np.uint8),
            modulation="QPSK",
            conditions=ChannelConditions(),
        )


def test_frame_rejects_wrong_types():
    with pytest.raises(ValidationError):
        Frame(
            signal="not a signal",  # type: ignore[arg-type]
            bits=np.zeros(2, dtype=np.uint8),
            modulation="BPSK",
            conditions=ChannelConditions(),
        )
    with pytest.raises(ValidationError):
        Frame(
            signal=Signal(np.ones(2, dtype=complex), sample_rate=1.0),
            bits=np.zeros(2, dtype=np.uint8),
            modulation="BPSK",
            conditions={"snr_db": 5},  # type: ignore[arg-type]
        )


def test_symbol_rate_is_derived_when_not_given():
    frame = _frame()
    assert frame.symbol_rate == pytest.approx(1000.0 / 4)


def test_metadata_excludes_heavy_arrays():
    """What gets logged and returned over the API: labels, not megabytes."""
    meta = _frame().metadata()
    assert meta["modulation"] == "QPSK"
    assert meta["conditions"]["snr_db"] == 6.0
    assert "bits" not in meta
    assert "signal" not in meta
