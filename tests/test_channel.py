"""Tests for the channel layer.

Two themes run through this file:

* **Physical correctness.**  An effect that claims to deliver 10 dB of
  signal-to-noise ratio must actually deliver it, and fading must not
  quietly change the average power -- otherwise every later measurement
  is calibrated against the wrong thing.
* **Design guarantees.**  Effects never mutate their input, every effect
  preserves frame length, and a pipeline is interchangeable with a single
  effect (the Composite pattern).
"""

from __future__ import annotations

import numpy as np
import pytest

from satrecovery import (
    AdditiveWhiteGaussianNoise,
    CarrierFrequencyOffset,
    CarrierPhaseOffset,
    ChannelConditions,
    ChannelEffect,
    ChannelFactory,
    ChannelPipeline,
    FlatRicianFading,
    MultipathRicianFading,
    Signal,
)
from satrecovery.core.exceptions import ValidationError


def _test_signal(n: int = 20_000) -> Signal:
    """A unit-power reference signal: random QPSK-like symbols."""
    rng = np.random.default_rng(0)
    phases = rng.integers(0, 4, size=n) * (np.pi / 2) + np.pi / 4
    return Signal(np.exp(1j * phases), sample_rate=1_000_000.0)


def _rng() -> np.random.Generator:
    return np.random.default_rng(42)


ALL_EFFECTS: list[ChannelEffect] = [
    AdditiveWhiteGaussianNoise(10.0),
    FlatRicianFading(8.0),
    MultipathRicianFading(8.0),
    CarrierFrequencyOffset(1500.0),
    CarrierPhaseOffset(0.7),
]


# ==========================================================================
# Abstraction
# ==========================================================================
def test_channel_effect_is_abstract():
    with pytest.raises(TypeError):
        ChannelEffect()  # type: ignore[abstract]


def test_subclass_must_implement_process():
    class Useless(ChannelEffect):
        pass

    with pytest.raises(TypeError):
        Useless()  # type: ignore[abstract]


# ==========================================================================
# Guarantees every effect must honour
# ==========================================================================
@pytest.mark.parametrize("effect", ALL_EFFECTS, ids=lambda e: e.name)
def test_effect_preserves_frame_length(effect):
    signal = _test_signal(4096)
    assert len(effect.apply(signal, _rng())) == len(signal)


@pytest.mark.parametrize("effect", ALL_EFFECTS, ids=lambda e: e.name)
def test_effect_does_not_mutate_its_input(effect):
    """Each stage returns a new object; the previous one stays intact."""
    signal = _test_signal(1024)
    before = np.array(signal.samples)

    output = effect.apply(signal, _rng())

    assert output is not signal
    assert np.array_equal(signal.samples, before)


@pytest.mark.parametrize("effect", ALL_EFFECTS, ids=lambda e: e.name)
def test_effect_records_itself_in_the_description(effect):
    """A frame's history stays readable, which makes debugging tractable."""
    output = effect.apply(_test_signal(256), _rng())
    assert effect.name in output.description


@pytest.mark.parametrize("effect", ALL_EFFECTS, ids=lambda e: e.name)
def test_effect_is_callable_like_a_function(effect):
    """Operator overloading: an effect can go wherever a callable can."""
    signal = _test_signal(256)
    assert np.allclose(
        effect(signal, _rng()).samples,
        effect.apply(signal, _rng()).samples,
    )


@pytest.mark.parametrize("effect", ALL_EFFECTS, ids=lambda e: e.name)
def test_effect_rejects_a_non_signal(effect):
    with pytest.raises(ValidationError):
        effect.apply(np.ones(10, dtype=complex))  # type: ignore[arg-type]


@pytest.mark.parametrize("effect", ALL_EFFECTS, ids=lambda e: e.name)
def test_effect_is_reproducible_from_a_seed(effect):
    """Required for a project whose deliverable is a set of measurements."""
    signal = _test_signal(512)
    first = effect.apply(signal, np.random.default_rng(7)).samples
    second = effect.apply(signal, np.random.default_rng(7)).samples
    assert np.allclose(first, second)


# ==========================================================================
# AWGN: does it deliver the signal-to-noise ratio it promises?
# ==========================================================================
@pytest.mark.parametrize("snr_db", [-5.0, 0.0, 7.0, 15.0, 25.0])
def test_awgn_hits_its_target_snr(snr_db):
    """Measure the noise we actually added and check the ratio.

    This is the calibration every later result rests on.  If 10 dB does
    not mean 10 dB, a bit-error-rate curve is just a shape with no
    meaning.
    """
    signal = _test_signal(200_000)
    noisy = AdditiveWhiteGaussianNoise(snr_db).apply(signal, _rng())

    noise = noisy.samples - signal.samples
    measured_db = 10.0 * np.log10(
        signal.power() / float(np.mean(np.abs(noise) ** 2))
    )
    assert measured_db == pytest.approx(snr_db, abs=0.1)


def test_awgn_noise_is_circularly_symmetric():
    """Real and imaginary noise must be equal-power and uncorrelated."""
    signal = _test_signal(200_000)
    noise = (
        AdditiveWhiteGaussianNoise(10.0).apply(signal, _rng()).samples
        - signal.samples
    )

    assert np.var(noise.real) == pytest.approx(np.var(noise.imag), rel=0.05)
    correlation = np.corrcoef(noise.real, noise.imag)[0, 1]
    assert abs(correlation) < 0.02


def test_higher_snr_means_less_noise():
    signal = _test_signal(50_000)
    powers = [
        float(
            np.mean(
                np.abs(
                    AdditiveWhiteGaussianNoise(snr).apply(signal, _rng()).samples
                    - signal.samples
                )
                ** 2
            )
        )
        for snr in (0.0, 10.0, 20.0)
    ]
    assert powers == sorted(powers, reverse=True)


# ==========================================================================
# Fading
# ==========================================================================
@pytest.mark.parametrize("k_db", [0.0, 5.0, 10.0, 20.0])
def test_flat_rician_fading_has_unit_average_power(k_db):
    """Fading changes the constellation's shape, not the link's budget.

    If ``E[|h|^2]`` drifted from 1, switching fading on would also change
    the effective signal-to-noise ratio, and we would be measuring two
    things at once.
    """
    effect = FlatRicianFading(k_db)
    rng = np.random.default_rng(1)
    gains = np.array([effect._draw_gain(rng) for _ in range(200_000)])
    assert float(np.mean(np.abs(gains) ** 2)) == pytest.approx(1.0, abs=0.01)


def test_high_k_factor_approaches_a_clean_channel():
    """With K very large the direct path dominates and fading nearly vanishes."""
    rng = np.random.default_rng(2)
    strong = FlatRicianFading(40.0)
    gains = np.array([strong._draw_gain(rng) for _ in range(20_000)])
    assert float(np.std(np.abs(gains))) < 0.02


def test_low_k_factor_fades_hard():
    rng = np.random.default_rng(3)
    weak = FlatRicianFading(0.0)
    gains = np.array([weak._draw_gain(rng) for _ in range(20_000)])
    assert float(np.std(np.abs(gains))) > 0.2


def test_multipath_uses_the_rml24_tap_profile():
    """Our simulated channel matches the published RML24 model."""
    effect = MultipathRicianFading()
    assert effect.n_taps == 3
    assert effect.tap_gains_db == (0.0, -0.97, -5.23)


def test_multipath_preserves_average_power():
    signal = _test_signal(100_000)
    faded = MultipathRicianFading(10.0).apply(signal, _rng())
    ratio_db = 10.0 * np.log10(faded.power() / signal.power())
    assert abs(ratio_db) < 1.5


def test_multipath_inherits_from_flat_fading():
    """Inheritance used for genuine shared implementation, not labelling.

    Multipath fading needs exactly the same coefficient mathematics as
    flat fading, applied once per echo instead of once per frame -- and a
    single-tap multipath channel *is* a flat channel, so the
    relationship holds mathematically as well as in code.
    """
    assert issubclass(MultipathRicianFading, FlatRicianFading)
    assert isinstance(MultipathRicianFading(), FlatRicianFading)


def test_single_tap_multipath_equals_flat_fading():
    """The claim above, checked rather than asserted in a comment."""
    signal = _test_signal(1000)
    single = MultipathRicianFading(10.0, tap_gains_db=(0.0,))
    flat = FlatRicianFading(10.0)

    a = single.apply(signal, np.random.default_rng(99)).samples
    b = flat.apply(signal, np.random.default_rng(99)).samples
    assert np.allclose(a, b)


def test_multipath_needs_at_least_one_tap():
    with pytest.raises(ValidationError):
        MultipathRicianFading(10.0, tap_gains_db=())


# ==========================================================================
# Frequency and phase offsets
# ==========================================================================
def test_frequency_offset_preserves_power():
    signal = _test_signal(10_000)
    shifted = CarrierFrequencyOffset(5000.0).apply(signal, _rng())
    assert shifted.power() == pytest.approx(signal.power())


def test_frequency_offset_moves_the_spectrum_by_the_right_amount():
    """Check the physics, not just that something happened."""
    rate = 1_000_000.0
    n = 8192
    tone = Signal(np.ones(n, dtype=complex), sample_rate=rate)
    offset_hz = 25_000.0

    shifted = CarrierFrequencyOffset(offset_hz).apply(tone, _rng())
    spectrum = np.abs(np.fft.fft(shifted.samples))
    peak_hz = np.fft.fftfreq(n, d=1.0 / rate)[int(np.argmax(spectrum))]

    assert peak_hz == pytest.approx(offset_hz, abs=rate / n)


def test_zero_offset_is_a_no_op():
    signal = _test_signal(100)
    assert np.allclose(
        CarrierFrequencyOffset(0.0).apply(signal, _rng()).samples, signal.samples
    )
    assert np.allclose(
        CarrierPhaseOffset(0.0).apply(signal, _rng()).samples, signal.samples
    )


def test_phase_offset_rotates_by_the_requested_angle():
    signal = Signal(np.ones(16, dtype=complex), sample_rate=1.0)
    rotated = CarrierPhaseOffset(np.pi / 3).apply(signal, _rng())
    assert np.allclose(np.angle(rotated.samples), np.pi / 3)


# ==========================================================================
# The Composite pattern
# ==========================================================================
def test_a_pipeline_is_itself_an_effect():
    """The defining property of the Composite pattern."""
    pipeline = ChannelPipeline([CarrierPhaseOffset(0.5)])
    assert isinstance(pipeline, ChannelEffect)


def test_an_empty_pipeline_is_the_identity():
    signal = _test_signal(256)
    assert np.allclose(
        ChannelFactory.ideal().apply(signal, _rng()).samples, signal.samples
    )


def test_pipeline_applies_stages_in_order():
    """Two rotations compose; their total must be their sum."""
    signal = Signal(np.ones(8, dtype=complex), sample_rate=1.0)
    pipeline = ChannelPipeline(
        [CarrierPhaseOffset(0.4), CarrierPhaseOffset(0.9)]
    )
    assert np.allclose(
        np.angle(pipeline.apply(signal, _rng()).samples), 1.3
    )


def test_pipeline_nests_inside_a_pipeline():
    """A composite can contain another composite, transparently."""
    signal = Signal(np.ones(8, dtype=complex), sample_rate=1.0)
    inner = ChannelPipeline([CarrierPhaseOffset(0.25), CarrierPhaseOffset(0.25)])
    outer = ChannelPipeline([inner, CarrierPhaseOffset(0.5)])

    assert np.allclose(np.angle(outer.apply(signal, _rng()).samples), 1.0)
    assert len(outer) == 2


def test_pipeline_supports_container_operations():
    pipeline = ChannelPipeline([CarrierPhaseOffset(0.1)])
    pipeline.add(AdditiveWhiteGaussianNoise(10.0))

    assert len(pipeline) == 2
    assert isinstance(pipeline[0], CarrierPhaseOffset)
    assert [effect.name for effect in pipeline] == ["PhaseOffset", "AWGN"]


def test_add_returns_self_for_chaining():
    pipeline = ChannelPipeline()
    assert pipeline.add(CarrierPhaseOffset(0.1)) is pipeline


def test_pipeline_rejects_non_effects():
    with pytest.raises(ValidationError):
        ChannelPipeline(["noise"])  # type: ignore[list-item]
    with pytest.raises(ValidationError):
        ChannelPipeline().add(42)  # type: ignore[arg-type]


def test_pipeline_effects_tuple_cannot_be_reordered():
    """Encapsulation: hand out a tuple so the chain cannot be edited behind us."""
    pipeline = ChannelPipeline([CarrierPhaseOffset(0.1)])
    assert isinstance(pipeline.effects, tuple)


# ==========================================================================
# The factory
# ==========================================================================
def test_factory_builds_only_the_requested_stages():
    quiet = ChannelFactory.from_conditions(ChannelConditions(snr_db=10.0))
    assert [effect.name for effect in quiet] == ["AWGN"]

    busy = ChannelFactory.from_conditions(
        ChannelConditions(
            snr_db=10.0,
            fading="multipath_rician",
            freq_offset_hz=100.0,
            phase_offset_rad=0.3,
        )
    )
    assert [effect.name for effect in busy] == [
        "MultipathRician",
        "FreqOffset",
        "PhaseOffset",
        "AWGN",
    ]


def test_factory_always_adds_noise_last():
    """Physically essential: thermal noise arises in the receiver.

    Putting the noise before the fading stage would let the fading
    attenuate the noise along with the signal, which would overstate the
    link quality -- a mistake that is easy to make and hard to spot in a
    plot.
    """
    pipeline = ChannelFactory.from_conditions(
        ChannelConditions(snr_db=5.0, fading="flat_rician", freq_offset_hz=10.0)
    )
    assert pipeline[-1].name == "AWGN"


def test_factory_rejects_a_plain_dict():
    with pytest.raises(ValidationError):
        ChannelFactory.from_conditions({"snr_db": 10.0})  # type: ignore[arg-type]


def test_describe_reads_as_a_signal_chain():
    pipeline = ChannelFactory.from_conditions(
        ChannelConditions(snr_db=5.0, fading="flat_rician")
    )
    text = pipeline.describe()
    assert "->" in text
    assert "FlatRician" in text and "AWGN" in text
