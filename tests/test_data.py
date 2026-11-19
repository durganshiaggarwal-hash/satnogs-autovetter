"""Tests for Phase 3 -- real recordings.

The most important test here is the one the real data cannot provide:
**does the reference decoder recover the right bits?**  RML24 has no
transmitted bits, which is the whole reason the decoder exists, so it
cannot be checked against RML24.  Instead the tests build recordings
from the simulator -- where every bit is known -- and give them RML24's
problems: cut out of a longer stream at an arbitrary sample, a carrier
error, strength drifting several-fold across the frame, a DC offset and
a little noise.  The decoder must then reproduce the transmitted symbols
exactly, up to the rotation no blind decoder can know.

Every other Phase 3 claim builds on that one.
"""

from __future__ import annotations

import numpy as np
import pytest

from satrecovery.core.conditions import ChannelConditions
from satrecovery.core.exceptions import ValidationError
from satrecovery.core.frame import Frame
from satrecovery.core.signal import Signal
from satrecovery.data import (
    RecordedSource,
    Recording,
    RecordingArchive,
    ReferenceDecoder,
    SampleArchive,
    layout,
    open_archive,
)
from satrecovery.evaluation import BenchmarkRunner, ImpairmentProfile, Scenario, SimulatedSource
from satrecovery.modulation.registry import ModulationRegistry
from satrecovery.receivers.classical import ClassicalReceiver
from satrecovery.receivers.result import RecoveryResult
from satrecovery.simulation.simulator import SignalSimulator


# ======================================================================
# Synthetic recordings: RML24's problems, with the truth known
# ======================================================================
def synthetic(
    modulation: str,
    sps: int,
    seed: int,
    *,
    snr_db: float = 30.0,
    freq_hz: float = 600.0,
    ramp: tuple[float, float] = (0.4, 1.8),
    dc: complex = 0.05 - 0.03j,
    start: int | None = None,
    row: int = 0,
    snr_label: float = 20.0,
):
    """A recording cut from a stream, plus the symbols really sent.

    The simulator's frames are circular -- one period of a continuous
    transmission -- so a window starting anywhere in a twice-as-long
    frame is a genuine piece of stream, with symbols falling wherever
    the cut happens to put them.
    """
    rng = np.random.default_rng(seed)
    sim = SignalSimulator(samples_per_symbol=sps, frame_length=4096, seed=seed)
    # 30 dB is the simulator's ceiling; noisier recordings get more below.
    frame = sim.generate_frame(modulation, ChannelConditions(snr_db=30.0))
    stream = frame.signal.samples
    start = int(rng.integers(0, 4096 - 2048)) if start is None else int(start)
    x = stream[start : start + 2048].copy()
    n = np.arange(x.size)
    x *= np.linspace(ramp[0], ramp[1], x.size)
    x *= np.exp(2j * np.pi * freq_hz * n / 1e6 + 1j * rng.uniform(0, 2 * np.pi))
    power = np.mean(np.abs(x) ** 2)
    extra = max(power / 10 ** (snr_db / 10) - power / 1e3, 0.0)
    sigma = np.sqrt(extra / 2)
    x += sigma * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)) + dc
    recording = Recording(
        np.stack([x.real, x.imag], axis=1).astype(np.float32),
        modulation, snr_label, 1e6 / sps, row=row, origin="synthetic",
    )
    return recording, frame.tx_symbols, start


def matches_truth(reference, truth: np.ndarray, start: int, sps: int, scheme) -> bool:
    """Do the graded reference symbols equal what was sent, up to rotation?"""
    j = np.flatnonzero(reference.graded)
    positions = reference.crop.start + j * sps + reference.timing_residual + start
    sent = truth[np.round(positions / sps).astype(int) % truth.size]
    got = reference.symbols[j]
    for k in range(scheme.rotational_symmetry):
        turn = np.exp(2j * np.pi * k / scheme.rotational_symmetry)
        if np.allclose(got, sent * turn, atol=1e-6):
            return True
    return False


@pytest.fixture(scope="module")
def archive_path(tmp_path_factory):
    """A small archive of synthetic recordings, in the probe's format."""
    recordings = []
    row = 0
    for modulation in ("BPSK", "QPSK", "8PSK"):
        for sps in (10, 4):
            for k in range(3):
                rec, _, _ = synthetic(modulation, sps, seed=100 + row, row=row)
                recordings.append(rec)
                row += 1
    # One dirty recording, labelled 0 dB, that must never be used.
    dirty, _, _ = synthetic("QPSK", 4, seed=7, snr_db=0.0, row=row, snr_label=0.0)
    recordings.append(dirty)
    path = tmp_path_factory.mktemp("rml") / "sample.npz"
    return SampleArchive.write(path, recordings)


@pytest.fixture(scope="module")
def archive(archive_path):
    return SampleArchive(archive_path)


# ======================================================================
# Layout and recordings
# ======================================================================
class TestLayout:
    def test_supported_classes_point_at_the_right_names(self):
        for name, index in layout.SUPPORTED.items():
            assert layout.CLASS_NAMES[index] == name

    def test_twenty_one_classes(self):
        assert len(layout.CLASS_NAMES) == 21

    def test_symbol_rates_become_whole_oversampling_factors(self):
        assert [layout.samples_per_symbol(r) for r in (100e3, 250e3, 500e3)] == [10, 4, 2]

    def test_a_rate_that_does_not_divide_is_refused(self):
        with pytest.raises(ValidationError):
            layout.samples_per_symbol(300e3)

    def test_an_unsupported_scheme_is_refused_by_name(self):
        with pytest.raises(ValidationError, match="FM"):
            layout.class_index("FM")


class TestRecording:
    def test_iq_pairs_become_complex(self):
        rec = Recording(np.array([[1, 2], [3, 4]] * 40, dtype=np.float32), "QPSK", 20, 250e3, row=5)
        assert rec.samples[0] == 1 + 2j and rec.samples_per_symbol == 4

    def test_samples_are_read_only(self):
        rec, _, _ = synthetic("BPSK", 10, seed=1)
        with pytest.raises(ValueError):
            rec.samples[0] = 0

    def test_nan_is_refused(self):
        samples = np.ones(100, dtype=complex)
        samples[3] = np.nan
        with pytest.raises(ValidationError):
            Recording(samples, "BPSK", 20, 100e3, row=0)


# ======================================================================
# Archives
# ======================================================================
class TestArchive:
    def test_round_trip_through_the_sample_format(self, archive):
        rec = archive.recording(3)
        assert rec.modulation == "BPSK" and rec.samples_per_symbol == 4 and len(rec) == 2048

    def test_select_filters_on_the_labels(self, archive):
        assert len(archive.select("QPSK")) == 7
        assert len(archive.select("QPSK", snr_label_db=20)) == 6
        assert len(archive.select("QPSK", 20, symbol_rate=250e3)) == 3

    def test_summary_lists_what_is_there(self, archive):
        summary = archive.summary()
        assert set(summary) == {"BPSK", "QPSK", "8PSK"}
        assert summary["QPSK"]["symbol_rates"] == [100000, 250000]

    def test_an_unknown_row_is_refused(self, archive):
        with pytest.raises(ValidationError):
            archive.recording(999)

    def test_the_factory_picks_the_format(self, archive_path):
        with open_archive(archive_path) as opened:
            assert isinstance(opened, SampleArchive)

    def test_the_abstract_archive_cannot_be_built(self):
        with pytest.raises(TypeError):
            RecordingArchive("x")

    def test_the_hdf5_reader_matches_the_published_layout(self, tmp_path, archive):
        """A miniature RML24_IQdata.h5 with the real file's field names."""
        h5py = pytest.importorskip("h5py")
        from satrecovery.data import RML24File

        rows = archive.select("QPSK")
        with h5py.File(tmp_path / "RML24_IQdata.h5", "w") as f:
            iq = np.stack([np.stack([r.samples.real, r.samples.imag], 1) for r in archive.recordings(rows)])
            f["IQ_data"] = iq.astype(np.float32)
            f["Bit_data"] = np.zeros((len(rows), 8), np.int32)
            f["Bit_len"] = np.zeros(len(rows), np.int32)
            f["class"] = np.full(len(rows), 1, np.int32)
            f["snr"] = np.array([int(archive.recording(r).snr_label_db) for r in rows], np.int32)
            f["symbol_rate"] = np.array([int(archive.recording(r).symbol_rate) for r in rows], np.int32)
        with RML24File(tmp_path) as real:            # a folder works as well as the file
            assert len(real) == len(rows)
            assert real.modulations() == ["QPSK"]
            first = real.recording(0)
            assert np.allclose(first.samples, archive.recording(int(rows[0])).samples, atol=1e-6)

    def test_a_file_that_is_not_rml24_is_refused(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        from satrecovery.data import RML24File

        with h5py.File(tmp_path / "other.h5", "w") as f:
            f["something"] = np.zeros(3)
        with pytest.raises(ValidationError, match="not an RML24 file"):
            RML24File(tmp_path / "other.h5")


# ======================================================================
# The reference decoder -- checked against known truth
# ======================================================================
class TestReferenceDecoder:
    @pytest.mark.parametrize("modulation", ["BPSK", "QPSK", "8PSK", "16QAM"])
    @pytest.mark.parametrize("sps", [10, 4, 2])
    def test_it_recovers_exactly_what_was_sent(self, modulation, sps):
        """Cut at a random sample, drifting 4.5-fold, 600 Hz off, with DC."""
        scheme = ModulationRegistry.create(modulation)
        for seed in range(3):
            rec, truth, start = synthetic(modulation, sps, seed=seed + 10 * sps)
            reference = ReferenceDecoder().decode(rec)
            assert reference.accepted, reference.reason
            assert matches_truth(reference, truth, start, sps, scheme)

    def test_it_finds_the_carrier_error(self):
        rec, _, _ = synthetic("QPSK", 10, seed=3, freq_hz=-1234.0)
        assert ReferenceDecoder().decode(rec).frequency_hz == pytest.approx(-1234.0, abs=5.0)

    def test_symbols_land_within_half_a_sample_of_the_grid(self):
        for start in range(10):
            rec, _, _ = synthetic("BPSK", 10, seed=4, start=start)
            reference = ReferenceDecoder().decode(rec)
            assert abs(reference.timing_residual) <= 0.5
            assert reference.n_symbols * 10 <= 2048 - reference.crop.start

    def test_the_edges_are_never_graded(self):
        reference = ReferenceDecoder(guard_symbols=8).decode(synthetic("QPSK", 4, seed=5)[0])
        assert not reference.graded[:8].any() and not reference.graded[-8:].any()
        assert reference.graded[8:-8].all()

    def test_a_noisy_recording_is_rejected_not_trusted(self):
        rec, _, _ = synthetic("QPSK", 4, seed=6, snr_db=0.0)
        reference = ReferenceDecoder().decode(rec)
        assert not reference.accepted and "above" in reference.reason

    def test_a_glitch_is_left_ungraded_rather_than_guessed(self):
        """RML24 corrupts one symbol in 92 at 250 kbaud; so do we, here."""
        rec, truth, start = synthetic("QPSK", 4, seed=8)
        samples = rec.samples.copy()
        rms = np.sqrt(np.mean(np.abs(samples) ** 2))
        samples[1002] += 2.5 * rms * (1 - 1j)       # a spike, like RML24's glitches
        broken = Recording(samples, "QPSK", 20, 250e3, row=0)
        reference = ReferenceDecoder().decode(broken)
        assert reference.accepted
        assert reference.n_ungraded > 16            # the guards, plus the glitch
        assert matches_truth(reference, truth, start, 4, ModulationRegistry.create("QPSK"))

    def test_too_many_glitches_reject_the_recording(self):
        rec, _, _ = synthetic("QPSK", 4, seed=9)
        samples = rec.samples.copy()
        for k in range(0, 2048, 40):
            samples[k : k + 4] = 0
        reference = ReferenceDecoder().decode(Recording(samples, "QPSK", 20, 250e3, row=0))
        assert not reference.accepted and "unreadable" in reference.reason

    def test_the_tone_power_comes_from_the_geometry(self):
        power = {m: ReferenceDecoder.tone_power(ModulationRegistry.create(m))
                 for m in ("BPSK", "QPSK", "8PSK", "16QAM", "16APSK")}
        assert power == {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16QAM": 4, "16APSK": 12}

    def test_bad_settings_are_refused(self):
        with pytest.raises(ValidationError):
            ReferenceDecoder(max_symbol_error=2.0)
        with pytest.raises(ValidationError):
            ReferenceDecoder(unreadable_distance=0.9)


# ======================================================================
# Frames with ungraded symbols
# ======================================================================
class TestGradedSymbols:
    def test_the_mask_must_cover_every_symbol(self):
        signal = Signal(np.ones(80, complex), 1e6)
        with pytest.raises(ValidationError, match="one entry per symbol"):
            Frame(signal, np.zeros(4, np.uint8), "BPSK", ChannelConditions(),
                  samples_per_symbol=8, graded_symbols=np.ones(5, bool))

    def test_a_mask_grading_nothing_is_refused(self):
        signal = Signal(np.ones(80, complex), 1e6)
        with pytest.raises(ValidationError):
            Frame(signal, np.zeros(0, np.uint8), "BPSK", ChannelConditions(),
                  samples_per_symbol=8, graded_symbols=np.zeros(10, bool))

    def test_selecting_symbols_keeps_their_bits_together(self):
        result = RecoveryResult(bits=np.array([0, 1, 1, 0, 1, 1], np.uint8),
                                symbols=np.array([1, 2, 3], complex))
        kept = result.select_symbols(np.array([True, False, True]), 2)
        assert list(kept.bits) == [0, 1, 1, 1] and list(kept.symbols) == [1, 3]
        assert kept.diagnostics["ungraded_symbols"] == 1

    def test_a_mask_of_the_wrong_length_is_refused(self):
        result = RecoveryResult(bits=np.zeros(6, np.uint8))
        with pytest.raises(ValidationError):
            result.select_symbols(np.ones(4, bool), 2)


# ======================================================================
# The recorded frame source
# ======================================================================
class TestRecordedSource:
    def test_it_only_produces_the_recorded_profile(self, archive):
        source = RecordedSource(archive)
        assert source.supports(Scenario("QPSK", 6, ImpairmentProfile.recorded()))
        assert not source.supports(Scenario("QPSK", 6, ImpairmentProfile.clean()))
        assert not source.supports(Scenario("16QAM", 6, ImpairmentProfile.recorded()))

    def test_the_simulator_declines_to_fake_a_recording(self):
        assert not SimulatedSource().supports(Scenario("QPSK", 6, ImpairmentProfile.recorded()))

    def test_the_dirty_recording_is_never_drawn(self, archive):
        source = RecordedSource(archive)
        assert 18 not in source.candidates("QPSK")      # labelled 0 dB, so not a candidate
        assert source.acceptance("QPSK")["rate"] == 1.0

    def test_frames_are_labelled_for_their_graded_symbols(self, archive):
        source = RecordedSource(archive)
        for frame in source.frames(Scenario("8PSK", 10, ImpairmentProfile.recorded()), 4):
            assert frame.source == "rml24"
            assert frame.n_bits == int(frame.graded_symbols.sum()) * 3
            assert frame.tx_symbols.size == frame.graded_symbols.size

    def test_the_noise_lands_on_the_simulators_scale(self, archive):
        """Energy per symbol must match the simulator at the same "SNR"."""
        source = RecordedSource(archive)
        row = int(source.candidates("QPSK")[0])
        rec = archive.recording(row)
        clean = source._prepared(rec)[source.reference(row).crop]
        frame = source.frame(row, 4.0, np.random.default_rng(0))
        added = frame.signal.samples - clean
        sample_snr = 10 * np.log10(np.mean(np.abs(clean) ** 2) / np.mean(np.abs(added) ** 2))
        expected = 4.0 + 10 * np.log10(8 / rec.samples_per_symbol)
        assert sample_snr == pytest.approx(expected, abs=0.5)

    def test_it_will_not_pretend_a_recording_is_cleaner_than_it_is(self, archive):
        source = RecordedSource(archive)
        row = int(source.candidates("QPSK")[0])
        too_clean = source.native_snr_db(row) + 1
        with pytest.raises(ValidationError, match="noisier"):
            source.frame(row, too_clean)
        assert not source.supports(Scenario("QPSK", 29, ImpairmentProfile.recorded()))

    def test_the_same_seed_gives_the_same_frames(self, archive):
        scenario = Scenario("BPSK", 2, ImpairmentProfile.recorded())
        a = [f.signal.samples for f in RecordedSource(archive, seed=3).frames(scenario, 3)]
        b = [f.signal.samples for f in RecordedSource(archive, seed=3).frames(scenario, 3)]
        assert all(np.array_equal(x, y) for x, y in zip(a, b))

    def test_more_frames_than_recordings_reuses_them_with_fresh_noise(self, archive):
        frames = list(RecordedSource(archive).frames(Scenario("BPSK", 4, ImpairmentProfile.recorded()), 9))
        assert len(frames) == 9
        assert not np.array_equal(frames[0].signal.samples, frames[6].signal.samples)

    @pytest.mark.parametrize("modulation", ["BPSK", "QPSK", "8PSK"])
    def test_the_classical_receiver_recovers_every_graded_bit(self, archive, modulation):
        """End to end: recording -> reference -> noisy frame -> receiver."""
        source = RecordedSource(archive)
        receiver = ClassicalReceiver()
        for frame in source.frames(Scenario(modulation, 14, ImpairmentProfile.recorded()), 6):
            result = receiver.recover(frame)
            assert result.n_bits == frame.n_bits
            assert result.bit_errors(frame.bits) == 0

    def test_the_benchmark_runs_on_recordings_unchanged(self, archive):
        """Phase 6's promise: the runner needs no change for real data."""
        runner = BenchmarkRunner(RecordedSource(archive), {"classical": ClassicalReceiver()}, frames_per_scenario=3)
        result = runner.run([Scenario("QPSK", s, ImpairmentProfile.recorded()) for s in (0, 8)])
        cells = {c.scenario.snr_db: c for c in result.cells}
        assert cells[0.0].ber >= cells[8.0].ber
        assert result.source == "RML24"


# ======================================================================
# Fixes the real data prompted
# ======================================================================
class TestFixesFromRealData:
    def test_16apsk_uses_the_dvb_s2_layout(self):
        """Outer ring at 15 + 30k degrees, so every inner point lines up with one."""
        points = ModulationRegistry.create("16APSK").constellation
        radius = np.abs(points)
        outer = np.degrees(np.angle(points[radius > radius.mean()])) % 360
        inner = np.degrees(np.angle(points[radius < radius.mean()])) % 360
        assert sorted(np.round(outer).astype(int)) == list(range(15, 360, 30))
        assert all(np.any(np.isclose(a, outer)) for a in inner)

    @pytest.mark.parametrize("sps", [8, 10])
    def test_8psk_survives_doppler(self, sps):
        """Was a bit error rate near 0.5 at 6-10 dB with 800 Hz of Doppler."""
        sim = SignalSimulator(samples_per_symbol=sps, seed=1)
        receiver = ClassicalReceiver()
        errors = bits = 0
        for _ in range(8):
            frame = sim.generate_frame("8PSK", ChannelConditions(snr_db=10, freq_offset_hz=-849, phase_offset_rad=0.7))
            result = receiver.recover(frame)
            errors += result.bit_errors(frame.bits)
            bits += frame.n_bits
        assert errors / bits < 1e-3

    def test_frequency_survives_strength_drifting_sevenfold(self):
        sim = SignalSimulator(seed=2)
        receiver = ClassicalReceiver()
        for _ in range(4):
            frame = sim.generate_frame("8PSK", ChannelConditions(snr_db=14, freq_offset_hz=700))
            drift = np.linspace(0.3, 2.1, frame.n_samples)
            drifted = Frame(Signal(frame.signal.samples * drift, frame.signal.sample_rate), frame.bits,
                            "8PSK", frame.conditions, tx_symbols=frame.tx_symbols,
                            samples_per_symbol=frame.samples_per_symbol)
            result = receiver.recover(drifted)
            assert result.diagnostics["coarse_freq_offset_hz"] == pytest.approx(700, abs=60)


class TestAcquisition:
    def test_lock_quality_tells_still_from_spinning(self):
        scheme = ModulationRegistry.create("8PSK")
        rng = np.random.default_rng(0)
        still = scheme.constellation[rng.integers(0, 8, 400)]
        spinning = still * np.exp(1j * 0.05 * np.arange(400))
        assert ClassicalReceiver.lock_quality(still, scheme) > 0.99
        assert ClassicalReceiver.lock_quality(spinning, scheme) < 0.1

    def test_a_wrong_strongest_peak_is_overruled_by_one_that_locks(self, archive):
        """The receiver must try other hypotheses when the first does not lock."""
        source = RecordedSource(archive)
        receiver = ClassicalReceiver()
        for frame in source.frames(Scenario("8PSK", 14, ImpairmentProfile.recorded()), 6):
            result = receiver.recover(frame)
            assert result.diagnostics["lock_quality"] >= 0.5
            assert result.ber(frame.bits) < 0.01        # locked: at most the odd noise error
