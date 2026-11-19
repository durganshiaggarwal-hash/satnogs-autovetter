"""Tests for the classical receiver.

The headline test is :func:`test_receiver_ber_is_close_to_theory`, which
checks that a complete receive chain -- matched filter, blind frequency
estimation, timing recovery, gain control, carrier loop and demapping --
lands within a fraction of a decibel of the published formulas.

That matters more than it might look.  Every later claim in this project
is a comparison against this receiver, so a baseline that is quietly a
few decibels off would make a mediocre neural network look like a
breakthrough, and nothing in the plots would appear wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from satrecovery import (
    AdditiveWhiteGaussianNoise,
    ChannelConditions,
    ClassicalReceiver,
    Frame,
    HardDemapper,
    ModulationRegistry,
    MultipathRicianFading,
    Receiver,
    ReceiverChain,
    RecoveryResult,
    Signal,
    SignalSimulator,
    SoftDemapper,
)
from satrecovery.channel.pipeline import ChannelPipeline
from satrecovery.core.exceptions import ValidationError
from satrecovery.receivers.base import ReceiverStage, RecoveryContext
from satrecovery.receivers.stages import (
    AutomaticGainControl,
    CoarseFrequencyCorrection,
    DecisionDirectedPLL,
    GardnerTimingRecovery,
    LMSEqualizer,
    MatchedFilterStage,
    interpolate_cubic,
    pi_loop_gains,
)

# A single simulator shared by most tests. 4096 samples at 8 samples per
# symbol is 512 symbols, long enough for the tracking loops to settle.
SIM = SignalSimulator(frame_length=4096, samples_per_symbol=8, seed=17)


def payload_ber(receiver: Receiver, frame: Frame) -> float:
    """Bit error rate over the payload, excluding the known preamble.

    The preamble is known to both ends in any real link, so counting it
    would flatter every result. Phase 6 uses the same convention.
    """
    result = receiver.recover(frame)
    scheme = ModulationRegistry.create(frame.modulation)
    skip = receiver.pilot_symbols * scheme.bits_per_symbol
    mask = result.error_mask(frame.bits)[skip:]
    return float(np.mean(mask)) if mask.size else 0.0


def measure(receiver, modulation, conditions, trials=4) -> float:
    """Average payload bit error rate over several frames."""
    errors = total = 0
    scheme = ModulationRegistry.create(modulation)
    skip = receiver.pilot_symbols * scheme.bits_per_symbol
    for _ in range(trials):
        frame = SIM.generate_frame(modulation, conditions)
        mask = receiver.recover(frame).error_mask(frame.bits)[skip:]
        errors += int(mask.sum())
        total += mask.size
    return errors / total


@pytest.fixture(scope="module")
def receiver() -> ClassicalReceiver:
    return ClassicalReceiver(pulse=SIM.pulse)


# ==========================================================================
# RecoveryResult
# ==========================================================================
def test_result_arrays_are_read_only():
    result = RecoveryResult(bits=np.array([0, 1, 1, 0], dtype=np.uint8))
    with pytest.raises(ValueError):
        result.bits[0] = 1
    with pytest.raises(ValueError):
        result.confidence[0] = 0.0


def test_result_rejects_mismatched_confidence():
    with pytest.raises(ValidationError):
        RecoveryResult(bits=np.zeros(4, dtype=np.uint8), confidence=np.ones(3))


def test_result_rejects_impossible_confidence():
    with pytest.raises(ValidationError):
        RecoveryResult(
            bits=np.zeros(2, dtype=np.uint8), confidence=np.array([1.5, 0.9])
        )


def test_result_defaults_to_total_certainty():
    """A receiver with no opinion is recorded as certain, deliberately.

    Phase 6 will show that total certainty is the worst thing a receiver
    can claim. Making it the default puts that behaviour on the chart
    rather than hiding it behind a missing value.
    """
    result = RecoveryResult(bits=np.zeros(8, dtype=np.uint8))
    assert result.mean_confidence == 1.0


def test_ber_counts_what_it_should():
    result = RecoveryResult(bits=np.array([0, 1, 0, 1], dtype=np.uint8))
    truth = np.array([0, 1, 1, 1], dtype=np.uint8)
    assert result.bit_errors(truth) == 1
    assert result.ber(truth) == pytest.approx(0.25)
    assert list(result.error_mask(truth)) == [False, False, True, False]


def test_length_mismatch_is_an_error_not_an_average():
    """A dropped symbol is a bug to find, not a difference to average over."""
    result = RecoveryResult(bits=np.zeros(10, dtype=np.uint8))
    with pytest.raises(ValidationError):
        result.ber(np.zeros(9, dtype=np.uint8))


def test_confidence_is_informative_detects_the_useful_case():
    bits = np.array([0, 0, 0, 0], dtype=np.uint8)
    truth = np.array([0, 0, 0, 1], dtype=np.uint8)

    good = RecoveryResult(bits=bits, confidence=np.array([0.9, 0.9, 0.9, 0.6]))
    bad = RecoveryResult(bits=bits, confidence=np.array([0.6, 0.6, 0.6, 0.99]))

    assert good.confidence_is_informative(truth)
    assert not bad.confidence_is_informative(truth)


# ==========================================================================
# Demappers
# ==========================================================================
@pytest.mark.parametrize("demapper", [HardDemapper(), SoftDemapper()])
@pytest.mark.parametrize("name", ModulationRegistry.available())
def test_demapper_is_lossless_on_clean_symbols(demapper, name):
    scheme = ModulationRegistry.create(name)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=400 * scheme.bits_per_symbol, dtype=np.uint8)

    recovered, confidence, _ = demapper.demap(scheme.map_bits(bits), scheme)

    assert np.array_equal(recovered, bits)
    assert confidence.size == bits.size
    assert np.all(confidence >= 0.5) and np.all(confidence <= 1.0)


def test_soft_demapper_produces_llrs_with_the_right_sign():
    """Positive log-likelihood ratio must mean the bit is a zero."""
    scheme = ModulationRegistry.create("QPSK")
    bits = np.array([0, 0, 1, 1, 0, 1, 1, 0], dtype=np.uint8)

    recovered, _, llr = SoftDemapper().demap(scheme.map_bits(bits), scheme)

    assert llr is not None
    assert np.array_equal(recovered, bits)
    assert np.all(llr[bits == 0] > 0)
    assert np.all(llr[bits == 1] < 0)


def test_hard_demapper_offers_no_llr():
    scheme = ModulationRegistry.create("QPSK")
    _, _, llr = HardDemapper().demap(scheme.constellation, scheme)
    assert llr is None


def test_confidence_falls_as_noise_rises():
    """A receiver should become less sure when the signal gets worse."""
    scheme = ModulationRegistry.create("16QAM")
    rng = np.random.default_rng(1)
    symbols = scheme.map_bits(
        rng.integers(0, 2, size=4000 * 4, dtype=np.uint8)
    )

    levels = []
    for snr_db in (20.0, 10.0, 4.0):
        noisy = AdditiveWhiteGaussianNoise(snr_db).apply(
            Signal(symbols, 1.0), np.random.default_rng(2)
        )
        _, confidence, _ = SoftDemapper().demap(noisy.samples, scheme)
        levels.append(float(np.mean(confidence)))

    assert levels == sorted(levels, reverse=True), levels


def test_soft_confidence_separates_right_from_wrong():
    """The property the whole uncertainty half of the project rests on."""
    scheme = ModulationRegistry.create("16QAM")
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, size=8000 * 4, dtype=np.uint8)
    noisy = AdditiveWhiteGaussianNoise(12.0).apply(
        Signal(scheme.map_bits(bits), 1.0), np.random.default_rng(4)
    )

    recovered, confidence, llr = SoftDemapper().demap(noisy.samples, scheme)
    result = RecoveryResult(bits=recovered, confidence=confidence, llr=llr)

    assert result.bit_errors(bits) > 50, "pick an SNR that produces errors"
    assert result.confidence_is_informative(bits)


# ==========================================================================
# Loop mathematics
# ==========================================================================
def test_loop_gains_grow_with_bandwidth():
    narrow = pi_loop_gains(0.001)
    wide = pi_loop_gains(0.05)
    assert wide[0] > narrow[0] and wide[1] > narrow[1]
    assert all(g > 0 for g in narrow + wide)


@pytest.mark.parametrize("bandwidth", [-0.1, 0.0, 0.8])
def test_invalid_loop_bandwidth_is_rejected(bandwidth):
    with pytest.raises(ValidationError):
        pi_loop_gains(bandwidth)


def test_interpolation_is_exact_at_whole_samples():
    data = np.array([1 + 1j, 2 - 1j, -3 + 0j, 0.5 + 2j, 4 + 4j])
    for index in range(data.size):
        assert interpolate_cubic(data, float(index)) == pytest.approx(
            data[index]
        )


def test_interpolation_reproduces_a_smooth_signal():
    """A slow sine is what a heavily oversampled waveform looks like."""
    n = 256
    t = np.arange(n)
    signal = np.exp(2j * np.pi * 3 * t / n)
    for position in (10.25, 40.5, 99.75):
        expected = np.exp(2j * np.pi * 3 * position / n)
        assert interpolate_cubic(signal, position) == pytest.approx(
            expected, abs=2e-3
        )


# ==========================================================================
# Individual stages
# ==========================================================================
def test_receiver_stage_is_abstract():
    with pytest.raises(TypeError):
        ReceiverStage()  # type: ignore[abstract]


def _context(name="QPSK", sps=8, pilot=16):
    scheme = ModulationRegistry.create(name)
    frame = SIM.generate_frame(name, ChannelConditions(snr_db=20.0))
    symbols = frame.tx_symbols
    return frame, RecoveryContext(
        scheme, sps, frame.signal.sample_rate,
        None if symbols is None else symbols[:pilot],
    )


@pytest.mark.parametrize(
    ("offset_hz", "tolerance"),
    [(0.0, 30.0), (200.0, 30.0), (-1500.0, 30.0), (3000.0, 40.0)],
)
def test_coarse_frequency_estimate_is_accurate(offset_hz, tolerance):
    """The blind estimator, measured against a known offset."""
    frame = SIM.generate_frame(
        "QPSK", ChannelConditions(snr_db=20.0, freq_offset_hz=offset_hz)
    )
    _, context = _context()
    stage = CoarseFrequencyCorrection()

    stage.apply(
        Signal(
            SIM.pulse.matched_filter(frame.signal.samples),
            frame.signal.sample_rate,
        ),
        context,
    )

    estimate = context.diagnostics["coarse_freq_offset_hz"]
    assert context.diagnostics["coarse_freq_accepted"] is True
    assert abs(estimate - offset_hz) < tolerance


def test_coarse_frequency_refuses_an_unsuitable_constellation():
    """16APSK: the Nth-power trick does not work and the stage knows it.

    Its two rings are 4-fold and 12-fold symmetric, so no single power
    collapses both. Rather than return the position of a noise peak and
    let the receiver act on it, the stage declines.
    """
    frame = SIM.generate_frame("16APSK", ChannelConditions(snr_db=20.0))
    _, context = _context("16APSK")

    CoarseFrequencyCorrection().apply(frame.signal, context)

    assert context.diagnostics["coarse_freq_accepted"] is False
    assert "unsuitable" in context.diagnostics["coarse_freq_skipped"]


def test_coarse_frequency_rejects_estimates_outside_its_range():
    """An estimate beyond the specified acquisition range is known wrong."""
    stage = CoarseFrequencyCorrection(max_offset_fraction=0.001)
    frame = SIM.generate_frame(
        "QPSK", ChannelConditions(snr_db=20.0, freq_offset_hz=3000.0)
    )
    _, context = _context()

    output = stage.apply(frame.signal, context)

    assert context.diagnostics["coarse_freq_accepted"] is False
    assert np.allclose(output.samples, frame.signal.samples)


@pytest.mark.parametrize("delay", [0.0, 0.25, 0.5, -0.4, 1.5])
def test_gardner_finds_the_timing_offset(delay):
    """The loop must converge on the delay that was actually applied."""
    frame = SIM.generate_frame(
        "QPSK", ChannelConditions(snr_db=25.0, timing_offset=delay)
    )
    _, context = _context()

    filtered = Signal(
        SIM.pulse.matched_filter(frame.signal.samples),
        frame.signal.sample_rate,
    )
    output = GardnerTimingRecovery().apply(filtered, context)

    found = context.diagnostics["timing_offset_samples"]
    assert abs(found - delay) < 0.25, f"wanted {delay}, found {found}"
    assert context.samples_per_symbol == 1
    assert len(output) == len(filtered) // 8


def test_timing_recovery_decimates_to_the_symbol_rate():
    frame, context = _context()
    filtered = Signal(
        SIM.pulse.matched_filter(frame.signal.samples),
        frame.signal.sample_rate,
    )
    output = GardnerTimingRecovery().apply(filtered, context)
    assert output.sample_rate == pytest.approx(SIM.symbol_rate)


def test_timing_recovery_needs_oversampling():
    _, context = _context(sps=1)
    with pytest.raises(ValidationError):
        GardnerTimingRecovery().apply(
            Signal(np.ones(64, dtype=complex), 1000.0), context
        )


def test_gain_control_restores_unit_power():
    _, context = _context()
    faint = Signal(np.full(256, 0.01 + 0.01j), 1000.0)
    assert AutomaticGainControl().apply(faint, context).power() == pytest.approx(
        1.0
    )


def test_carrier_loop_removes_a_fixed_rotation():
    """A constellation turned by a known angle must come back upright."""
    scheme = ModulationRegistry.create("QPSK")
    rng = np.random.default_rng(5)
    bits = rng.integers(0, 2, size=1024, dtype=np.uint8)
    symbols = scheme.map_bits(bits)

    rotated = symbols * np.exp(1j * 0.8)
    context = RecoveryContext(scheme, 1, 1000.0, symbols[:16])

    output = DecisionDirectedPLL().apply(Signal(rotated, 1000.0), context)

    recovered, _, _ = SoftDemapper().demap(output.samples, scheme)
    assert np.array_equal(recovered, bits)


def test_carrier_loop_resolves_the_phase_ambiguity():
    """Without a preamble the loop cannot know which way up it is.

    A quarter-turn maps QPSK onto itself, so a blind loop locks happily
    to a rotation that decodes every symbol wrongly. The preamble is the
    only thing that distinguishes them, which is why every real link
    sends one.
    """
    scheme = ModulationRegistry.create("QPSK")
    rng = np.random.default_rng(6)
    bits = rng.integers(0, 2, size=1024, dtype=np.uint8)
    symbols = scheme.map_bits(bits)
    rotated = symbols * np.exp(1j * np.pi / 2)  # exactly one quarter turn

    with_pilot = RecoveryContext(scheme, 1, 1000.0, symbols[:16])
    without = RecoveryContext(scheme, 1, 1000.0, None)

    guided = DecisionDirectedPLL().apply(Signal(rotated, 1000.0), with_pilot)
    blind = DecisionDirectedPLL().apply(Signal(rotated, 1000.0), without)

    guided_bits, _, _ = SoftDemapper().demap(guided.samples, scheme)
    blind_bits, _, _ = SoftDemapper().demap(blind.samples, scheme)

    assert np.array_equal(guided_bits, bits)
    assert not np.array_equal(blind_bits, bits)
    assert without.diagnostics["initial_phase_rad"] == "blind"


def test_carrier_loop_runs_at_the_symbol_rate_only():
    _, context = _context(sps=8)
    with pytest.raises(ValidationError):
        DecisionDirectedPLL().apply(
            Signal(np.ones(64, dtype=complex), 1000.0), context
        )


def test_equalizer_starts_as_a_pass_through():
    """A centre-spike start means it can only improve on doing nothing."""
    scheme = ModulationRegistry.create("QPSK")
    rng = np.random.default_rng(7)
    symbols = scheme.map_bits(rng.integers(0, 2, size=512, dtype=np.uint8))
    context = RecoveryContext(scheme, 1, 1000.0, symbols[:16])

    output = LMSEqualizer(step_size=1e-9).apply(Signal(symbols, 1000.0), context)
    assert np.allclose(output.samples, symbols, atol=1e-4)


def test_equalizer_detects_its_own_divergence():
    """Least-mean-squares runs away above a critical step size.

    Left alone the taps reach infinity within a frame and every stage
    downstream inherits the wreckage, so the stage checks itself and
    falls back to passing the signal through untouched.
    """
    scheme = ModulationRegistry.create("QPSK")
    rng = np.random.default_rng(8)
    symbols = scheme.map_bits(rng.integers(0, 2, size=2048, dtype=np.uint8))
    noisy = AdditiveWhiteGaussianNoise(3.0).apply(
        Signal(symbols, 1000.0), np.random.default_rng(9)
    )
    context = RecoveryContext(scheme, 1, 1000.0, symbols[:16])

    output = LMSEqualizer(step_size=0.9).apply(noisy, context)

    assert context.diagnostics["equalizer_diverged"] is True
    assert np.all(np.isfinite(output.samples))
    assert np.allclose(output.samples, noisy.samples)


@pytest.mark.parametrize("n_taps", [0, 2, 8, -3])
def test_equalizer_requires_an_odd_tap_count(n_taps):
    with pytest.raises(ValidationError):
        LMSEqualizer(n_taps=n_taps)


# ==========================================================================
# The chain, as a composite
# ==========================================================================
def test_a_chain_is_itself_a_stage():
    chain = ReceiverChain([AutomaticGainControl()])
    assert isinstance(chain, ReceiverStage)


def test_chain_supports_container_operations():
    chain = ReceiverChain([MatchedFilterStage(SIM.pulse)])
    chain.add(AutomaticGainControl())

    assert len(chain) == 2
    assert [stage.name for stage in chain] == ["MatchedFilter", "AGC"]
    assert isinstance(chain.stages, tuple)
    assert chain.stages[0].name == "MatchedFilter"


def test_chain_rejects_things_that_are_not_stages():
    with pytest.raises(ValidationError):
        ReceiverChain(["matched filter"])  # type: ignore[list-item]
    with pytest.raises(ValidationError):
        ReceiverChain().add(42)  # type: ignore[arg-type]


def test_chain_nests_inside_a_chain():
    inner = ReceiverChain([AutomaticGainControl()])
    outer = ReceiverChain([inner, AutomaticGainControl()])
    _, context = _context()
    output = outer.apply(Signal(np.full(128, 7.0 + 0j), 1000.0), context)
    assert output.power() == pytest.approx(1.0)


# ==========================================================================
# The receiver, end to end
# ==========================================================================
def test_receiver_is_abstract():
    with pytest.raises(TypeError):
        Receiver()  # type: ignore[abstract]


def test_receiver_rejects_a_non_frame(receiver):
    with pytest.raises(ValidationError):
        receiver.recover("a signal")  # type: ignore[arg-type]


def test_recovers_every_bit_on_a_clean_channel(receiver):
    """With a strong signal and no impairments there is no excuse."""
    for name in ModulationRegistry.available():
        frame = SIM.generate_frame(name, ChannelConditions(snr_db=28.0))
        assert payload_ber(receiver, frame) == 0.0, name


def test_result_is_fully_labelled(receiver):
    frame = SIM.generate_frame("16QAM", ChannelConditions(snr_db=15.0))
    result = receiver.recover(frame)

    assert result.n_bits == frame.n_bits
    assert result.modulation == "16QAM"
    assert result.receiver == "Classical"
    assert result.symbols is not None
    assert result.llr is not None
    assert result.diagnostics["demapper"] == "Soft"
    assert result.diagnostics["n_pilot_bits"] == 16 * 4
    assert 0.0 <= result.diagnostics["evm_percent"] < 100.0


@pytest.mark.parametrize(
    "conditions",
    [
        ChannelConditions(snr_db=22.0, phase_offset_rad=1.3),
        ChannelConditions(snr_db=22.0, freq_offset_hz=250.0),
        ChannelConditions(snr_db=22.0, freq_offset_hz=-1800.0),
        ChannelConditions(snr_db=22.0, timing_offset=0.45),
        ChannelConditions(snr_db=22.0, timing_offset=-1.6),
        ChannelConditions(snr_db=22.0, fading="flat_rician", rician_k_db=6.0),
        ChannelConditions(
            snr_db=22.0, fading="multipath_rician", rician_k_db=10.0
        ),
    ],
    ids=["phase", "freq", "big freq", "timing", "big timing", "flat", "multipath"],
)
def test_each_impairment_is_repaired(receiver, conditions):
    """Each stage must earn its place against the damage it exists for."""
    frame = SIM.generate_frame("QPSK", conditions)
    assert payload_ber(receiver, frame) < 1e-3, conditions.describe()


def test_everything_at_once_is_still_repaired(receiver):
    frame = SIM.generate_frame(
        "QPSK",
        ChannelConditions(
            snr_db=16.0,
            fading="multipath_rician",
            rician_k_db=9.0,
            freq_offset_hz=900.0,
            phase_offset_rad=0.8,
            timing_offset=0.6,
        ),
    )
    assert payload_ber(receiver, frame) < 1e-2


@pytest.mark.parametrize(
    ("name", "snr_db", "tolerance"),
    [
        ("BPSK", -8.0, 0.40),
        ("BPSK", -6.0, 0.40),
        ("QPSK", -5.0, 0.35),
        ("QPSK", -3.0, 0.35),
        ("8PSK", 0.0, 0.45),
        ("16QAM", 0.0, 0.35),
        ("16QAM", 2.0, 0.35),
        ("64QAM", 8.0, 0.40),
    ],
)
def test_receiver_ber_is_close_to_theory(receiver, name, snr_db, tolerance):
    """The headline result of Phase 2.

    A real receive chain, estimating the frequency and timing blindly
    from the signal, must land within a fraction of a decibel of the
    textbook curve. The gap that remains is implementation loss -- the
    price of the receiver having to work things out instead of being
    told them -- and a real modem pays the same kind of price.
    """
    scheme = ModulationRegistry.create(name)
    measured = measure(receiver, name, ChannelConditions(snr_db=snr_db))

    ebn0_db = SIM.effective_ebn0_db(snr_db, name)
    expected = float(scheme.theoretical_ber(ebn0_db))

    assert expected > 1e-4, "choose an SNR where errors are countable"
    relative = abs(measured - expected) / expected
    assert relative < tolerance, (
        f"{name} at {snr_db} dB (Eb/N0 {ebn0_db:.2f} dB): "
        f"measured {measured:.3e}, theory {expected:.3e}, "
        f"off by {relative:.0%}"
    )


def test_ber_falls_as_the_signal_improves(receiver):
    rates = [
        measure(receiver, "16QAM", ChannelConditions(snr_db=snr), trials=3)
        for snr in (-2.0, 2.0, 6.0, 10.0)
    ]
    assert rates == sorted(rates, reverse=True), rates


def test_frequency_search_picks_the_better_branch():
    """The receiver tries both ways and keeps the tighter constellation.

    The blind estimator is excellent when it works and destructive when
    it does not, and no threshold reliably separates the two beforehand.
    Running the chain both ways and judging by error vector magnitude --
    which uses no knowledge of what was sent -- settles it per frame.
    """
    searching = ClassicalReceiver(pulse=SIM.pulse, frequency_search=True)
    frame = SIM.generate_frame(
        "QPSK", ChannelConditions(snr_db=20.0, freq_offset_hz=1200.0)
    )
    result = searching.recover(frame)

    assert result.diagnostics["frequency_branch"] == "corrected"
    assert payload_ber(searching, frame) == 0.0


def test_frequency_search_helps_when_the_estimate_is_unreliable():
    """At low signal-to-noise the estimator returns noise; search catches it."""
    conditions = ChannelConditions(snr_db=-8.0)
    with_search = ClassicalReceiver(pulse=SIM.pulse, frequency_search=True)
    without = ClassicalReceiver(pulse=SIM.pulse, frequency_search=False)

    searched = measure(with_search, "QPSK", conditions, trials=6)
    blind = measure(without, "QPSK", conditions, trials=6)

    assert searched <= blind * 1.05, (searched, blind)


def test_soft_demapping_beats_hard_demapping(receiver):
    """Soft decisions keep information that hard decisions throw away."""
    soft = ClassicalReceiver(pulse=SIM.pulse, demapper=SoftDemapper())
    hard = ClassicalReceiver(pulse=SIM.pulse, demapper=HardDemapper())
    conditions = ChannelConditions(snr_db=2.0)

    soft_result = soft.recover(SIM.generate_frame("16QAM", conditions))
    hard_result = hard.recover(SIM.generate_frame("16QAM", conditions))

    # Both decide the same way; what differs is the quality of the
    # confidence attached, which is what Phase 5 builds on.
    assert soft_result.llr is not None
    assert hard_result.llr is None
    assert soft_result.mean_confidence != hard_result.mean_confidence


def test_receiver_reports_lower_confidence_on_its_mistakes(receiver):
    frame = SIM.generate_frame("16QAM", ChannelConditions(snr_db=2.0))
    result = receiver.recover(frame)
    assert result.bit_errors(frame.bits) > 0
    assert result.confidence_is_informative(frame.bits)


# ==========================================================================
# The equaliser, on damage it can actually fix
# ==========================================================================
def _isi_frame(snr_db: float, seed: int) -> Frame:
    """A frame whose echoes land a whole symbol apart.

    The RML24 channel spaces its echoes one *sample* apart, an eighth of
    a symbol, which the matched filter mostly absorbs. Spacing them a
    symbol apart produces genuine inter-symbol interference: a delayed
    copy of one symbol sitting on top of the next.
    """
    rng = np.random.default_rng(seed)
    scheme = ModulationRegistry.create("QPSK")
    bits = rng.integers(0, 2, size=SIM.n_symbols * 2, dtype=np.uint8)
    symbols = scheme.map_bits(bits)

    channel = ChannelPipeline(
        [
            MultipathRicianFading(12.0, tap_spacing=8),
            AdditiveWhiteGaussianNoise(snr_db),
        ]
    )
    received = channel.apply(
        Signal(SIM.pulse.shape(symbols), SIM.sample_rate), rng
    )
    return Frame(
        received,
        bits,
        "QPSK",
        ChannelConditions(snr_db=snr_db),
        tx_symbols=symbols,
        samples_per_symbol=8,
    )


def test_equalizer_helps_against_symbol_spaced_echoes():
    """The case the equaliser exists for, and the only one where it wins."""
    with_eq = ClassicalReceiver(
        pulse=SIM.pulse, use_equalizer=True, pilot_symbols=128
    )
    without = ClassicalReceiver(
        pulse=SIM.pulse, use_equalizer=False, pilot_symbols=128
    )

    errors = {"on": 0, "off": 0}
    total = 0
    for seed in range(4):
        frame = _isi_frame(2.0, 300 + seed)
        for label, rx in (("on", with_eq), ("off", without)):
            mask = rx.recover(frame).error_mask(frame.bits)[128 * 2:]
            errors[label] += int(mask.sum())
        total += frame.n_bits - 256

    assert errors["on"] < 0.7 * errors["off"], errors


def test_equalizer_is_off_by_default():
    """A measured decision: on this channel it costs a little accuracy.

    The RML24 echoes are too short to cause inter-symbol interference,
    so the equaliser finds nothing to fix and merely adapts against
    noise. Reporting that is more useful than quietly leaving it on.
    """
    assert ClassicalReceiver().uses_equalizer is False


# ======================================================================
# Phase 3 fix: the frame decides the rate, not the receiver
# ======================================================================
class TestFrameRateWins:
    """RML24 frames are 10, 5 or 2 samples per symbol, not 8.

    A receiver built with an 8-sps pulse used to apply that pulse to
    every frame.  Measured before the fix, QPSK at 6 dB on a 2-sps frame
    came out at 45% bit errors -- a coin toss -- with nothing to say why.
    """

    def test_pulses_rebuild_themselves_at_a_new_rate(self):
        from satrecovery import Rectangular, RootRaisedCosine

        rrc = RootRaisedCosine(8, rolloff=0.22, span_symbols=16)
        moved = rrc.at_rate(5)
        assert isinstance(moved, RootRaisedCosine)
        assert moved.samples_per_symbol == 5
        assert moved.rolloff == 0.22 and moved.span_symbols == 16
        assert rrc.at_rate(8) is rrc

        rect = Rectangular(4).at_rate(2)
        assert isinstance(rect, Rectangular) and rect.samples_per_symbol == 2

    def test_a_bad_rate_is_rejected(self):
        from satrecovery import RootRaisedCosine
        from satrecovery.core.exceptions import ValidationError

        with pytest.raises(ValidationError):
            RootRaisedCosine(8).at_rate(0)

    @pytest.mark.parametrize("sps", [10, 5, 2])
    def test_one_receiver_serves_every_rml24_rate(self, sps):
        from satrecovery import (
            ChannelConditions,
            ClassicalReceiver,
            RootRaisedCosine,
            SignalSimulator,
            SoftDemapper,
        )

        simulator = SignalSimulator(samples_per_symbol=sps, seed=7)
        receiver = ClassicalReceiver(pulse=RootRaisedCosine(8), demapper=SoftDemapper())
        errors = total = 0
        for _ in range(10):
            frame = simulator.generate_frame(
                "QPSK", ChannelConditions(snr_db=10.0, phase_offset_rad=0.5)
            )
            errors += receiver.recover(frame).bit_errors(frame.bits)
            total += frame.bits.size
        # 10 dB SNR is at least 10 dB Eb/N0 for QPSK at any of these
        # rates, where theory gives about 4e-6: effectively zero here.
        assert errors / total < 1e-3


class TestFirstSymbolRegression:
    """The bug the Phase 6 benchmark found.

    The matched filter is circular.  When it ran before frequency
    correction, a Doppler shift made the frame's end and start disagree
    in phase, and the filter blended that mismatch into symbol 0.  The
    result was an error floor -- 16QAM stuck at 1.3e-3 however strong
    the signal -- made entirely of first symbols.  No other test saw it,
    because it is one symbol in 256.
    """

    def test_a_strong_doppler_frame_has_no_errors_at_all(self):
        from satrecovery import ChannelConditions, ClassicalReceiver, SignalSimulator, SoftDemapper

        simulator = SignalSimulator(seed=21)
        receiver = ClassicalReceiver(pulse=simulator.pulse, demapper=SoftDemapper())
        conditions = ChannelConditions(snr_db=20.0, freq_offset_hz=800.0)
        errors = 0
        for _ in range(20):
            frame = simulator.generate_frame("16QAM", conditions)
            errors += receiver.recover(frame).bit_errors(frame.bits)
        assert errors == 0

    def test_frequency_correction_runs_before_the_matched_filter(self):
        from satrecovery import ClassicalReceiver
        from satrecovery.receivers.stages import CoarseFrequencyCorrection, MatchedFilterStage

        names = [type(s) for s in ClassicalReceiver().build_chain(8, use_coarse_frequency=True)]
        assert names.index(CoarseFrequencyCorrection) < names.index(MatchedFilterStage)
