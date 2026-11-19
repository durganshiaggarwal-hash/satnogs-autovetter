"""Tests for the modulation layer.

The most important test in this file is
:func:`test_ber_matches_textbook_theory`.  Everything the project claims
later -- that a neural demapper beats a classical one, or that a hybrid
beats both -- is meaningless unless the classical baseline is correct in
the first place.  Checking our measured bit error rate against the
closed-form expressions from the communications literature is how we
establish that.  If this test passes, the transmitter, the Gray mapping,
the noise model and the demapper are all right.
"""

from __future__ import annotations

import numpy as np
import pytest

from satrecovery import (
    AdditiveWhiteGaussianNoise,
    ChannelConditions,
    Modulation,
    ModulationRegistry,
    Rectangular,
    Signal,
    SignalSimulator,
)
from satrecovery.core.exceptions import (
    BitLengthError,
    UnknownModulationError,
    ValidationError,
)
from satrecovery.modulation.base import gray_code

ALL_SCHEMES = ModulationRegistry.available()


# ==========================================================================
# Abstraction
# ==========================================================================
def test_abstract_base_class_cannot_be_instantiated():
    """``Modulation`` is an idea, not a thing -- Python must refuse it."""
    with pytest.raises(TypeError):
        Modulation("fake", 2)  # type: ignore[abstract]


def test_incomplete_subclass_cannot_be_instantiated():
    """A subclass that skips the abstract method is rejected too."""

    class Broken(Modulation):
        pass  # never implements _build_constellation

    with pytest.raises(TypeError):
        Broken("broken", 1)  # type: ignore[abstract]


# ==========================================================================
# Gray coding
# ==========================================================================
def test_gray_code_neighbours_differ_by_one_bit():
    for index in range(63):
        difference = gray_code(index) ^ gray_code(index + 1)
        assert bin(difference).count("1") == 1


def test_gray_code_is_a_bijection():
    for width in (1, 2, 3, 4, 6):
        size = 2**width
        assert sorted(gray_code(i) for i in range(size)) == list(range(size))


# ==========================================================================
# Constellation properties, checked for every scheme
# ==========================================================================
@pytest.mark.parametrize("name", ALL_SCHEMES)
def test_constellation_has_unit_average_energy(name):
    """Normalisation is what makes a fair comparison between schemes."""
    scheme = ModulationRegistry.create(name)
    energy = np.mean(np.abs(scheme.constellation) ** 2)
    assert energy == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("name", ALL_SCHEMES)
def test_constellation_size_matches_bits_per_symbol(name):
    scheme = ModulationRegistry.create(name)
    assert scheme.constellation.shape == (2**scheme.bits_per_symbol,)
    assert scheme.order == len(scheme)


@pytest.mark.parametrize("name", ALL_SCHEMES)
def test_constellation_points_are_distinct(name):
    scheme = ModulationRegistry.create(name)
    assert scheme.minimum_distance > 1e-6


@pytest.mark.parametrize("name", ALL_SCHEMES)
def test_constellation_is_read_only(name):
    """Encapsulation: a caller must not be able to corrupt the geometry."""
    scheme = ModulationRegistry.create(name)
    with pytest.raises(ValueError):
        scheme.constellation[0] = 99.0


# ==========================================================================
# Round-trip behaviour
# ==========================================================================
@pytest.mark.parametrize("name", ALL_SCHEMES)
def test_noiseless_round_trip_is_lossless(name):
    """With no channel at all, every bit must survive."""
    scheme = ModulationRegistry.create(name)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=600 * scheme.bits_per_symbol, dtype=np.uint8)

    recovered = scheme.demap_symbols(scheme.map_bits(bits))

    assert np.array_equal(recovered, bits)


@pytest.mark.parametrize("name", ALL_SCHEMES)
def test_bit_index_conversion_round_trips(name):
    scheme = ModulationRegistry.create(name)
    indices = np.arange(scheme.order)
    bits = scheme.indices_to_bits(indices)
    assert np.array_equal(scheme.bits_to_indices(bits), indices)


@pytest.mark.parametrize("name", ALL_SCHEMES)
def test_wrong_bit_count_is_rejected(name):
    """Fail loudly rather than silently truncating the payload."""
    scheme = ModulationRegistry.create(name)
    if scheme.bits_per_symbol == 1:
        pytest.skip("BPSK accepts any length, so there is nothing to reject")
    bad = np.zeros(scheme.bits_per_symbol + 1, dtype=np.uint8)
    with pytest.raises(BitLengthError):
        scheme.map_bits(bad)


def test_non_binary_bits_are_rejected():
    scheme = ModulationRegistry.create("QPSK")
    with pytest.raises(ValidationError):
        scheme.map_bits(np.array([0, 1, 2, 1], dtype=np.uint8))


# ==========================================================================
# Gray mapping quality
# ==========================================================================
@pytest.mark.parametrize("name", ["BPSK", "QPSK", "8PSK"])
def test_psk_nearest_neighbours_differ_by_exactly_one_bit(name):
    """The property Gray coding exists to provide.

    For PSK the nearest neighbours of any point are its two angular
    neighbours, and a Gray-ordered circle guarantees those labels differ
    in a single bit.  So the most likely error costs one bit, not
    several -- which is why the measured bit error rate below lands on
    the textbook curve.
    """
    scheme = ModulationRegistry.create(name)
    points = scheme.constellation
    distances = np.abs(points[:, None] - points[None, :])
    np.fill_diagonal(distances, np.inf)

    for index in range(scheme.order):
        closest = np.flatnonzero(
            np.isclose(distances[index], scheme.minimum_distance)
        )
        for neighbour in closest:
            differing = bin(int(index) ^ int(neighbour)).count("1")
            assert differing == 1, f"{name}: {index} vs {neighbour}"


# ==========================================================================
# The headline test: measured bit error rate against published theory
# ==========================================================================
@pytest.mark.parametrize(
    ("name", "snr_db", "tolerance"),
    [
        # BPSK and QPSK have exact closed forms, so demand a close match.
        ("BPSK", 4.0, 0.12),
        ("BPSK", 8.0, 0.15),
        ("QPSK", 7.0, 0.12),
        ("QPSK", 11.0, 0.15),
        # The rest are nearest-neighbour approximations, which are only
        # asymptotically tight, so allow more room.
        ("8PSK", 13.0, 0.30),
        ("16QAM", 16.0, 0.30),
        ("64QAM", 22.0, 0.35),
    ],
)
def test_ber_matches_textbook_theory(name, snr_db, tolerance):
    """Measured bit error rate must match the published formula.

    Run at the symbol level -- one sample per symbol, rectangular pulse,
    no filtering -- so the test isolates the modulation and the noise
    model.  The full receiver chain is validated separately in Phase 2.

    Converting signal-to-noise ratio to energy per bit: with unit-energy
    symbols, SNR equals Es/N0, and Eb/N0 = Es/N0 divided by the number of
    bits each symbol carries.
    """
    scheme = ModulationRegistry.create(name)
    rng = np.random.default_rng(12345)

    n_symbols = 400_000
    bits = rng.integers(
        0, 2, size=n_symbols * scheme.bits_per_symbol, dtype=np.uint8
    )
    symbols = scheme.map_bits(bits)

    noisy = AdditiveWhiteGaussianNoise(snr_db).apply(
        Signal(symbols, sample_rate=1.0), rng
    )
    recovered = scheme.demap_symbols(noisy.samples)

    measured = float(np.mean(recovered != bits))

    ebn0_db = snr_db - 10.0 * np.log10(scheme.bits_per_symbol)
    expected = float(scheme.theoretical_ber(ebn0_db))

    assert expected > 1e-6, "pick a lower SNR so the test is not noise-limited"
    relative_error = abs(measured - expected) / expected
    assert relative_error < tolerance, (
        f"{name} at {snr_db} dB (Eb/N0 {ebn0_db:.2f} dB): "
        f"measured {measured:.3e}, theory {expected:.3e}, "
        f"off by {relative_error:.1%}"
    )


def test_apsk_reports_no_theory_curve():
    """Polymorphism: a subclass may answer 'not applicable'."""
    assert ModulationRegistry.create("16APSK").theoretical_ber(10.0) is None
    assert ModulationRegistry.create("QPSK").theoretical_ber(10.0) is not None


def test_ber_falls_as_snr_rises():
    """A monotonicity check that holds for every scheme."""
    scheme = ModulationRegistry.create("QPSK")
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, size=200_000, dtype=np.uint8)
    symbols = scheme.map_bits(bits)

    rates = []
    for index, snr_db in enumerate((0.0, 4.0, 8.0, 12.0)):
        noisy = AdditiveWhiteGaussianNoise(snr_db).apply(
            Signal(symbols, sample_rate=1.0), np.random.default_rng(100 + index)
        )
        recovered = scheme.demap_symbols(noisy.samples)
        rates.append(float(np.mean(recovered != bits)))

    assert rates == sorted(rates, reverse=True), rates


# ==========================================================================
# Registry
# ==========================================================================
def test_registry_is_case_and_separator_insensitive():
    expected = ModulationRegistry.create("16QAM")
    for spelling in ("16qam", "qam16", " 16-QAM ", "16_qam"):
        assert ModulationRegistry.create(spelling) is expected


def test_registry_caches_instances():
    """Immutable objects can safely be shared, and building 64QAM is work."""
    assert ModulationRegistry.create("64QAM") is ModulationRegistry.create("64QAM")


def test_unknown_scheme_raises_a_helpful_error():
    with pytest.raises(UnknownModulationError) as info:
        ModulationRegistry.create("256QAM")
    assert "QPSK" in str(info.value)


def test_registry_orders_from_robust_to_fragile():
    schemes = ModulationRegistry.all_schemes()
    bits = [scheme.bits_per_symbol for scheme in schemes]
    assert bits == sorted(bits)


# ==========================================================================
# Polymorphism, demonstrated end to end
# ==========================================================================
def test_one_loop_handles_every_scheme():
    """The point of the whole hierarchy: no branching on scheme type.

    This loop is exactly what the benchmark runner and the GUI do.  It
    never asks what kind of modulation it is holding.
    """
    simulator = SignalSimulator(
        frame_length=512, samples_per_symbol=1, pulse=Rectangular(1), seed=3
    )
    conditions = ChannelConditions(snr_db=25.0)

    for scheme in ModulationRegistry.all_schemes():
        frame = simulator.generate_frame(scheme, conditions)
        recovered = scheme.demap_symbols(frame.signal.samples)

        assert frame.n_bits == 512 * scheme.bits_per_symbol
        assert recovered.size == frame.n_bits
        ber = float(np.mean(recovered != frame.bits))
        assert ber < 0.05, f"{scheme.name} unexpectedly poor at 25 dB: {ber}"
