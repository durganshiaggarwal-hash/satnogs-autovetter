"""Tests for pulse shaping and the simulator.

The centrepiece here is
:func:`test_matched_filter_recovers_symbols_without_interference`.  It
checks the property the whole physical layer is built on: a
root-raised-cosine filter at each end produces a combined response that
is zero at every *other* symbol instant, so neighbouring symbols do not
smear into each other at the decision points.

If that holds, the transmit chain is sound and any bit errors we measure
later are the channel's fault rather than our own.
"""

from __future__ import annotations

import numpy as np
import pytest

from satrecovery import (
    ChannelConditions,
    ModulationRegistry,
    PulseShape,
    Rectangular,
    RootRaisedCosine,
    SignalSimulator,
)
from satrecovery.core.exceptions import ValidationError


# ==========================================================================
# Pulse shaping: abstraction
# ==========================================================================
def test_pulse_shape_is_abstract():
    with pytest.raises(TypeError):
        PulseShape(4)  # type: ignore[abstract]


@pytest.mark.parametrize("sps", [1, 2, 4, 8])
def test_rrc_taps_have_unit_energy(sps):
    """Normalisation keeps the signal-to-noise ratio meaningful end to end."""
    taps = RootRaisedCosine(sps).taps
    assert float(np.sum(taps**2)) == pytest.approx(1.0)


def test_rrc_taps_are_symmetric():
    """A symmetric impulse response means linear phase -- no group delay."""
    taps = RootRaisedCosine(8, rolloff=0.35).taps
    assert np.allclose(taps, taps[::-1])


def test_rrc_peaks_at_the_centre():
    taps = RootRaisedCosine(8).taps
    assert int(np.argmax(np.abs(taps))) == taps.size // 2


def test_rrc_singular_points_are_finite():
    """Roll-off 0.25 puts a sample exactly on the removable singularity.

    With eight samples per symbol, ``t = T/(4*beta)`` lands precisely on
    a tap, so the general formula would divide by zero.  Handling it with
    the analytic limit is what keeps the filter finite.
    """
    taps = RootRaisedCosine(8, rolloff=0.25).taps
    assert np.all(np.isfinite(taps))


@pytest.mark.parametrize("rolloff", [-0.1, 0.0, 1.5])
def test_invalid_rolloff_is_rejected(rolloff):
    with pytest.raises(ValidationError):
        RootRaisedCosine(8, rolloff=rolloff)


def test_taps_are_read_only():
    with pytest.raises(ValueError):
        RootRaisedCosine(4).taps[0] = 1.0


def test_taps_are_computed_once_and_cached():
    pulse = RootRaisedCosine(8)
    assert pulse.taps is pulse.taps


def test_rrc_bandwidth_grows_with_rolloff():
    assert RootRaisedCosine(8, rolloff=0.2).occupied_bandwidth_factor == pytest.approx(1.2)
    assert RootRaisedCosine(8, rolloff=0.5).occupied_bandwidth_factor == pytest.approx(1.5)


# ==========================================================================
# Pulse shaping: the zero-interference property
# ==========================================================================
@pytest.mark.parametrize("sps", [4, 8])
@pytest.mark.parametrize(
    ("rolloff", "span"),
    [
        # Tighter roll-off rings for longer in time, so a tighter
        # spectrum needs a longer filter to reach the same residue.
        # This is the real design trade, not a quirk of our code.
        (0.22, 20),
        (0.35, 12),
        (0.50, 8),
    ],
)
def test_matched_filter_recovers_symbols_without_interference(
    sps, rolloff, span
):
    """Transmit, matched filter, sample at the symbol rate, compare.

    The recovered symbols must match the transmitted ones almost
    exactly.  Any residual error is inter-symbol interference, and the
    root-raised-cosine pair exists precisely to drive it to zero.
    """
    pulse = RootRaisedCosine(sps, rolloff=rolloff, span_symbols=span)
    scheme = ModulationRegistry.create("16QAM")

    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=256 * scheme.bits_per_symbol, dtype=np.uint8)
    symbols = scheme.map_bits(bits)

    waveform = pulse.shape(symbols)
    filtered = pulse.matched_filter(waveform)
    recovered = filtered[::sps]

    error = np.max(np.abs(recovered - symbols))

    # The residue is filter truncation, not a bug: the ideal response is
    # infinitely long.  The bound is set well below 16QAM's closest point
    # spacing (0.632) so it cannot cause a bit error on its own.
    assert error < 0.02, f"inter-symbol interference too high: {error:.2e}"


def test_residual_interference_shrinks_with_filter_length():
    """Confirms the residue above really is truncation and nothing else."""
    scheme = ModulationRegistry.create("16QAM")
    rng = np.random.default_rng(0)
    symbols = scheme.map_bits(
        rng.integers(0, 2, size=256 * 4, dtype=np.uint8)
    )

    errors = []
    for span in (4, 8, 24):
        pulse = RootRaisedCosine(8, rolloff=0.35, span_symbols=span)
        recovered = pulse.matched_filter(pulse.shape(symbols))[::8]
        errors.append(float(np.max(np.abs(recovered - symbols))))

    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] < 5e-3


def test_shaping_preserves_timing_alignment():
    """Symbol k must land on sample k*sps, or every later stage misreads it."""
    pulse = RootRaisedCosine(8)
    symbols = np.zeros(32, dtype=complex)
    symbols[10] = 1.0

    peak = int(np.argmax(np.abs(pulse.shape(symbols))))
    assert peak == 10 * 8


def test_shaping_output_length_is_exact():
    pulse = RootRaisedCosine(8)
    assert pulse.shape(np.ones(100, dtype=complex)).size == 800


def test_rectangular_with_one_sample_per_symbol_is_a_pass_through():
    """The configuration the theory tests rely on: no filtering at all."""
    pulse = Rectangular(1)
    symbols = np.array([1 + 1j, -1 - 1j, 0.5 + 0j])
    assert np.allclose(pulse.shape(symbols), symbols)


def test_empty_input_is_handled():
    assert RootRaisedCosine(8).shape(np.array([], dtype=complex)).size == 0


# ==========================================================================
# Simulator: configuration
# ==========================================================================
def test_defaults_match_the_rml24_parameters():
    """Our frames line up with the dataset's, so the two are comparable."""
    simulator = SignalSimulator()
    assert simulator.sample_rate == 1_000_000.0
    assert simulator.frame_length == 2048
    assert simulator.samples_per_symbol == 8
    assert simulator.pulse.rolloff == pytest.approx(0.35)  # type: ignore[attr-defined]


def test_frame_length_rounds_down_to_whole_symbols():
    simulator = SignalSimulator(frame_length=1000, samples_per_symbol=8)
    assert simulator.n_symbols == 125
    assert simulator.frame_length == 1000

    odd = SignalSimulator(frame_length=1005, samples_per_symbol=8)
    assert odd.n_symbols == 125
    assert odd.frame_length == 1000


def test_symbol_rate_is_derived_from_the_sample_rate():
    simulator = SignalSimulator(sample_rate=1_000_000.0, samples_per_symbol=8)
    assert simulator.symbol_rate == pytest.approx(125_000.0)


def test_a_pulse_built_for_a_different_oversampling_is_rejected():
    """Catches a mismatch that would otherwise show up as mysterious errors."""
    with pytest.raises(ValidationError):
        SignalSimulator(samples_per_symbol=8, pulse=RootRaisedCosine(4))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0.0},
        {"sample_rate": -1.0},
        {"samples_per_symbol": 0},
        {"frame_length": 4, "samples_per_symbol": 8},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValidationError):
        SignalSimulator(**kwargs)


def test_dependency_injection_accepts_any_pulse_shape():
    """The simulator is written against the abstract base, not a concrete class."""
    simulator = SignalSimulator(
        frame_length=64, samples_per_symbol=1, pulse=Rectangular(1)
    )
    assert isinstance(simulator.pulse, Rectangular)
    assert simulator.generate_frame("QPSK").n_samples == 64


# ==========================================================================
# Simulator: generated frames
# ==========================================================================
@pytest.mark.parametrize("name", ModulationRegistry.available())
def test_frame_is_labelled_consistently(name):
    """Every frame must be self-describing, whichever scheme produced it."""
    simulator = SignalSimulator(frame_length=512, seed=1)
    scheme = ModulationRegistry.create(name)
    conditions = ChannelConditions(snr_db=12.0)

    frame = simulator.generate_frame(name, conditions)

    assert frame.modulation == name
    assert frame.n_samples == 512
    assert frame.n_bits == simulator.n_symbols * scheme.bits_per_symbol
    assert frame.tx_symbols is not None
    assert frame.tx_symbols.size == simulator.n_symbols
    assert frame.conditions == conditions
    assert frame.source == "simulated"
    assert frame.symbol_rate == pytest.approx(simulator.symbol_rate)


def test_name_and_object_give_the_same_result():
    """Polymorphic input: the API has a string, internal code has an object."""
    scheme = ModulationRegistry.create("QPSK")
    by_name = SignalSimulator(frame_length=256, seed=5).generate_frame("QPSK")
    by_object = SignalSimulator(frame_length=256, seed=5).generate_frame(scheme)
    assert np.allclose(by_name.signal.samples, by_object.signal.samples)


def test_the_same_seed_reproduces_the_same_frame():
    """Non-negotiable for a project that reports measured numbers."""
    conditions = ChannelConditions(snr_db=3.0, fading="multipath_rician")
    first = SignalSimulator(frame_length=256, seed=11).generate_frame(
        "16QAM", conditions
    )
    second = SignalSimulator(frame_length=256, seed=11).generate_frame(
        "16QAM", conditions
    )

    assert np.array_equal(first.bits, second.bits)
    assert np.allclose(first.signal.samples, second.signal.samples)


def test_different_seeds_give_different_frames():
    conditions = ChannelConditions(snr_db=3.0)
    a = SignalSimulator(frame_length=256, seed=1).generate_frame("QPSK", conditions)
    b = SignalSimulator(frame_length=256, seed=2).generate_frame("QPSK", conditions)
    assert not np.array_equal(a.bits, b.bits)


def test_reset_rewinds_the_generator():
    simulator = SignalSimulator(frame_length=128, seed=9)
    first = simulator.generate_frame("QPSK").bits
    simulator.reset()
    assert np.array_equal(simulator.generate_frame("QPSK").bits, first)


def test_supplied_bits_are_transmitted_verbatim():
    """Lets the GUI hold the payload fixed while the user varies the channel."""
    simulator = SignalSimulator(frame_length=64, samples_per_symbol=8, seed=0)
    pattern = np.tile([1, 0], 8).astype(np.uint8)

    frame = simulator.generate_frame("QPSK", bits=pattern)
    assert np.array_equal(frame.bits, pattern)


def test_wrong_number_of_supplied_bits_is_rejected():
    simulator = SignalSimulator(frame_length=64, samples_per_symbol=8)
    with pytest.raises(ValidationError):
        simulator.generate_frame("QPSK", bits=np.zeros(5, dtype=np.uint8))


# ==========================================================================
# Simulator: end-to-end sanity checks
# ==========================================================================
def test_ideal_channel_recovers_every_bit():
    """The cleanest possible end-to-end check of the whole transmit chain.

    Shape, matched filter, sample at the symbol rate, demap -- with no
    channel in the way, the bit error rate must be exactly zero.  Running
    it for all six schemes catches a Gray-mapping or alignment mistake in
    any one of them.
    """
    simulator = SignalSimulator(frame_length=1024, samples_per_symbol=8, seed=4)

    for name in ModulationRegistry.available():
        scheme = ModulationRegistry.create(name)
        frame = simulator.transmit_only(name)

        filtered = simulator.pulse.matched_filter(frame.signal.samples)
        recovered = scheme.demap_symbols(filtered[:: frame.samples_per_symbol])

        assert np.array_equal(recovered, frame.bits), f"{name} lost bits"


def test_noise_degrades_recovery_as_expected():
    """A smoke test of the full chain: more noise, more errors.

    Note the signal-to-noise ratios.  Because the matched filter adds
    about 9 dB of processing gain at eight samples per symbol, QPSK is
    already nearly error-free by 0 dB on the channel.  To see the
    interesting part of the curve the sweep has to go well negative --
    which is exactly the range RML24 covers.
    """
    simulator = SignalSimulator(frame_length=2048, samples_per_symbol=8, seed=6)
    scheme = ModulationRegistry.create("QPSK")
    payload = np.random.default_rng(0).integers(
        0, 2, size=simulator.n_symbols * 2, dtype=np.uint8
    )

    rates = []
    for snr_db in (-12.0, -8.0, -4.0, 0.0):
        frame = simulator.generate_frame(
            "QPSK", ChannelConditions(snr_db=snr_db), bits=payload
        )
        filtered = simulator.pulse.matched_filter(frame.signal.samples)
        recovered = scheme.demap_symbols(filtered[:: frame.samples_per_symbol])
        rates.append(float(np.mean(recovered != frame.bits)))

    assert rates == sorted(rates, reverse=True), rates
    assert rates[0] > 0.05, "-12 dB should produce plenty of errors"
    assert rates[-1] < 0.01, "0 dB should be nearly clean after the filter"


# ==========================================================================
# The link budget conversion
# ==========================================================================
def test_ebn0_conversion_round_trips():
    simulator = SignalSimulator(samples_per_symbol=8)
    for name in ("BPSK", "QPSK", "64QAM"):
        ebn0 = simulator.effective_ebn0_db(-5.0, name)
        assert simulator.snr_db_for_ebn0(ebn0, name) == pytest.approx(-5.0)


def test_processing_gain_is_accounted_for():
    """Eight samples per symbol must buy about 9 dB."""
    simulator = SignalSimulator(samples_per_symbol=8)
    # BPSK carries one bit per symbol, so there is no bits penalty.
    assert simulator.effective_ebn0_db(0.0, "BPSK") == pytest.approx(9.03, abs=0.01)
    # QPSK carries two, costing 3 dB of the gain back.
    assert simulator.effective_ebn0_db(0.0, "QPSK") == pytest.approx(6.02, abs=0.01)


def test_no_oversampling_means_no_processing_gain():
    simulator = SignalSimulator(
        frame_length=64, samples_per_symbol=1, pulse=Rectangular(1)
    )
    assert simulator.effective_ebn0_db(10.0, "BPSK") == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("name", "snr_db", "tolerance"),
    [
        ("BPSK", -11.0, 0.20),
        ("QPSK", -8.0, 0.20),
        ("16QAM", -1.0, 0.35),
    ],
)
def test_full_chain_ber_matches_theory(name, snr_db, tolerance):
    """The end-to-end correctness proof for the whole transmit chain.

    Random bits -> Gray mapping -> root-raised-cosine shaping -> AWGN ->
    matched filter -> symbol-rate sampling -> nearest-neighbour demapping,
    measured against the published formula.  Passing this means the
    modulation, the pulse shaping, the noise calibration, the timing
    alignment and the link budget arithmetic are all simultaneously
    right -- which is the baseline every later claim is measured against.
    """
    simulator = SignalSimulator(
        frame_length=16384, samples_per_symbol=8, seed=21
    )
    scheme = ModulationRegistry.create(name)
    conditions = ChannelConditions(snr_db=snr_db)

    errors = 0
    total = 0
    for _ in range(12):
        frame = simulator.generate_frame(name, conditions)
        filtered = simulator.pulse.matched_filter(frame.signal.samples)
        recovered = scheme.demap_symbols(filtered[:: frame.samples_per_symbol])
        errors += int(np.count_nonzero(recovered != frame.bits))
        total += frame.n_bits

    measured = errors / total
    ebn0_db = simulator.effective_ebn0_db(snr_db, name)
    expected = float(scheme.theoretical_ber(ebn0_db))

    assert expected > 1e-5, "pick a lower SNR so the test is not noise-limited"
    relative_error = abs(measured - expected) / expected
    assert relative_error < tolerance, (
        f"{name} at {snr_db} dB channel SNR (Eb/N0 {ebn0_db:.2f} dB): "
        f"measured {measured:.3e}, theory {expected:.3e}, "
        f"off by {relative_error:.1%}"
    )


def test_frequency_offset_wrecks_an_uncorrected_receiver():
    """Motivates Phase 2: carrier recovery is not optional.

    A spinning constellation defeats a nearest-neighbour demapper
    completely, which is why the classical receiver needs a carrier
    recovery loop -- and why a learned demodulator has to discover the
    same correction for itself.
    """
    simulator = SignalSimulator(frame_length=2048, samples_per_symbol=8, seed=8)
    scheme = ModulationRegistry.create("QPSK")

    clean = ChannelConditions(snr_db=20.0)
    spinning = ChannelConditions(snr_db=20.0, freq_offset_hz=2000.0)

    rates = []
    for conditions in (clean, spinning):
        frame = simulator.generate_frame("QPSK", conditions)
        filtered = simulator.pulse.matched_filter(frame.signal.samples)
        recovered = scheme.demap_symbols(filtered[:: frame.samples_per_symbol])
        rates.append(float(np.mean(recovered != frame.bits)))

    assert rates[0] < 1e-3
    assert rates[1] > 0.1


# ==========================================================================
# Batch generation
# ==========================================================================
def test_batch_covers_every_combination():
    simulator = SignalSimulator(frame_length=128, seed=2)
    conditions = [ChannelConditions(snr_db=s) for s in (0.0, 10.0)]

    frames = simulator.generate_batch(["BPSK", "QPSK"], conditions, frames_each=3)

    assert len(frames) == 2 * 2 * 3
    assert {frame.modulation for frame in frames} == {"BPSK", "QPSK"}
    assert {frame.snr_db for frame in frames} == {0.0, 10.0}


def test_snr_sweep_varies_only_the_noise():
    """What makes a bit-error-rate curve mean anything."""
    simulator = SignalSimulator(frame_length=128, seed=2)
    base = ChannelConditions(snr_db=0.0, fading="flat_rician", rician_k_db=6.0)

    frames = simulator.snr_sweep("QPSK", [-5.0, 0.0, 5.0], base)

    assert [frame.snr_db for frame in frames] == [-5.0, 0.0, 5.0]
    assert all(frame.conditions.fading == "flat_rician" for frame in frames)
    assert all(frame.conditions.rician_k_db == 6.0 for frame in frames)


def test_empty_batch_inputs_give_an_empty_result():
    simulator = SignalSimulator(frame_length=128, seed=2)
    assert simulator.generate_batch([], [ChannelConditions()]) == []
    assert simulator.generate_batch(["QPSK"], []) == []


def test_frames_each_must_be_positive():
    simulator = SignalSimulator(frame_length=128)
    with pytest.raises(ValidationError):
        simulator.generate_batch(["QPSK"], [ChannelConditions()], frames_each=0)
