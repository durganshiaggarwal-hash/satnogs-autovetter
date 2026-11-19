"""Tests for Phase 6 -- the benchmark runner and everything it relies on.

The statistics are checked against values computed independently, the
splits are checked for leakage, and the runner is checked for the one
property every comparison in the project depends on: that every receiver
is shown exactly the same frames.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from satrecovery import ClassicalReceiver, HardDemapper, SoftDemapper
from satrecovery.core.exceptions import ReceiverError, ValidationError
from satrecovery.evaluation import (
    BenchmarkObserver,
    BenchmarkResult,
    BenchmarkRunner,
    CallbackObserver,
    CellResult,
    ErrorRate,
    FrameBootstrap,
    FrameSource,
    HoldOutProfile,
    HoldOutSNR,
    ImpairmentProfile,
    RandomSplit,
    ResultStore,
    Scenario,
    ScenarioGrid,
    SimulatedSource,
)
from satrecovery.receivers.bank import ReceiverBank


# ======================================================================
# Fixtures
# ======================================================================
@pytest.fixture(scope="module")
def source():
    return SimulatedSource(seed=3)


@pytest.fixture(scope="module")
def receivers(source):
    pulse = source.simulator.pulse
    return {
        "soft": ClassicalReceiver(pulse=pulse, demapper=SoftDemapper()),
        "hard": ClassicalReceiver(pulse=pulse, demapper=HardDemapper()),
    }


@pytest.fixture(scope="module")
def small_grid():
    return ScenarioGrid.product(
        ["QPSK", "16QAM"], [0.0, 8.0], [ImpairmentProfile.clean(), ImpairmentProfile.full()]
    )


@pytest.fixture(scope="module")
def result(source, receivers, small_grid):
    return BenchmarkRunner(source, receivers, frames_per_scenario=4).run(small_grid)


# ======================================================================
# Scenarios and grids
# ======================================================================
class TestScenarios:
    def test_equal_settings_make_equal_scenarios(self):
        a = Scenario("qpsk", 6, ImpairmentProfile.clean())
        b = Scenario("QPSK", 6.0, ImpairmentProfile.clean())
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_the_key_is_stable_across_spellings_of_a_number(self):
        assert Scenario("QPSK", 6, ImpairmentProfile.clean()).key == \
            Scenario("QPSK", 6.000, ImpairmentProfile.clean()).key

    def test_a_profile_rejects_the_noise_level(self):
        with pytest.raises(ValidationError):
            ImpairmentProfile("bad", snr_db=4.0)

    def test_a_profile_rejects_a_misspelt_field_immediately(self):
        with pytest.raises(Exception):
            ImpairmentProfile("bad", freqoffset=100.0)

    def test_an_unknown_modulation_fails_at_construction(self):
        with pytest.raises(Exception):
            Scenario("QAM-9000", 6, ImpairmentProfile.clean())

    def test_the_standard_profiles_build_valid_channels(self):
        for profile in ImpairmentProfile.standard():
            assert profile.conditions(5.0).snr_db == 5.0

    def test_a_product_grid_has_every_combination(self, small_grid):
        assert len(small_grid) == 2 * 2 * 2
        assert small_grid.modulations == ["16QAM", "QPSK"]
        assert small_grid.snrs_db == [0.0, 8.0]

    def test_duplicates_are_dropped(self):
        s = Scenario("QPSK", 0, ImpairmentProfile.clean())
        assert len(ScenarioGrid([s, s, s])) == 1

    def test_grid_arithmetic_behaves_like_sets(self, small_grid):
        clean = small_grid.where(profile="clean")
        full = small_grid.where(profile="full")
        assert len(clean) + len(full) == len(small_grid)
        assert clean + full == small_grid
        assert small_grid - clean == full
        assert len(clean & full) == 0

    def test_where_filters_on_every_field(self, small_grid):
        picked = small_grid.where(modulation="qpsk", profile="full", snr_min=4.0)
        assert len(picked) == 1
        assert picked[0].modulation == "QPSK" and picked[0].snr_db == 8.0

    def test_empty_inputs_are_refused(self):
        with pytest.raises(ValidationError):
            ScenarioGrid.product([], [0.0])


# ======================================================================
# Sources
# ======================================================================
class TestSources:
    def test_a_scenario_always_yields_the_same_frames(self, small_grid):
        scenario = small_grid[0]
        first = [f.bits.copy() for f in SimulatedSource(seed=9).frames(scenario, 3)]
        again = [f.bits.copy() for f in SimulatedSource(seed=9).frames(scenario, 3)]
        assert all(np.array_equal(a, b) for a, b in zip(first, again))

    def test_frames_do_not_depend_on_the_order_scenarios_are_run(self, small_grid):
        """Running half a grid must reproduce that half exactly."""
        src = SimulatedSource(seed=9)
        target = small_grid[3]
        list(src.frames(small_grid[0], 2))       # run something else first
        after = next(iter(src.frames(target, 1))).bits.copy()
        alone = next(iter(SimulatedSource(seed=9).frames(target, 1))).bits.copy()
        assert np.array_equal(after, alone)

    def test_different_scenarios_get_different_frames(self, small_grid):
        src = SimulatedSource(seed=9)
        a = next(iter(src.frames(small_grid[0], 1))).bits
        b = next(iter(src.frames(small_grid[1], 1))).bits
        assert not np.array_equal(a, b)

    def test_frames_carry_the_scenarios_conditions(self, small_grid):
        scenario = small_grid.where(profile="full")[0]
        frame = next(iter(SimulatedSource().frames(scenario, 1)))
        assert frame.modulation == scenario.modulation
        assert frame.conditions == scenario.conditions()

    def test_a_bad_count_is_refused(self, small_grid):
        with pytest.raises(ValidationError):
            list(SimulatedSource().frames(small_grid[0], 0))


# ======================================================================
# Statistics
# ======================================================================
class TestErrorRate:
    def test_wilson_matches_an_independently_computed_value(self):
        # 10 errors in 100 trials, 95%: the standard published answer
        # is [0.0552, 0.1744] to four places.
        low, high = ErrorRate(10, 100).interval()
        assert low == pytest.approx(0.0552, abs=5e-4)
        assert high == pytest.approx(0.1744, abs=5e-4)

    def test_zero_errors_still_has_an_upper_bound(self):
        """No errors seen is not an error rate of zero."""
        low, high = ErrorRate(0, 2000).interval()
        assert low == 0.0
        # Close to the "rule of three": 3/n for 95%.
        assert 1e-3 < high < 3e-3

    def test_adding_pools_the_counts(self):
        assert ErrorRate(1, 100) + ErrorRate(9, 900) == ErrorRate(10, 1000)
        assert sum([ErrorRate(1, 10), ErrorRate(2, 20)]) == ErrorRate(3, 30)

    def test_pooling_is_not_averaging(self):
        """A tiny measurement must not count as much as a large one."""
        pooled = ErrorRate(5, 10) + ErrorRate(0, 10_000)
        assert pooled.rate < 0.001
        assert pooled.rate != pytest.approx((0.5 + 0.0) / 2)

    def test_impossible_counts_are_refused(self):
        with pytest.raises(ValidationError):
            ErrorRate(5, 3)
        with pytest.raises(ValidationError):
            ErrorRate(-1, 3)


class TestFrameBootstrap:
    def test_clustered_errors_widen_the_interval(self):
        """The reason the bootstrap exists.

        Same total errors, same total bits.  Spread evenly, the Wilson
        interval is about right.  Packed into a few lost frames, the true
        uncertainty is far larger, and the bootstrap must say so.
        """
        bits = np.full(50, 1000)
        spread = np.full(50, 20)                     # 2% everywhere
        clumped = np.zeros(50, dtype=int)
        clumped[:2] = 500                            # two dead frames
        boot = FrameBootstrap(resamples=2000, seed=1)

        spread_width = np.diff(boot.interval(spread, bits))[0]
        clumped_width = np.diff(boot.interval(clumped, bits))[0]
        wilson_width = np.diff(ErrorRate(1000, 50_000).interval())[0]

        assert clumped_width > 5 * spread_width
        assert clumped_width > 5 * wilson_width

    def test_the_interval_contains_the_point_estimate(self):
        rng = np.random.default_rng(0)
        bits = np.full(40, 1024)
        errors = rng.binomial(1024, 0.03, size=40)
        low, high = FrameBootstrap().interval(errors, bits)
        assert low <= errors.sum() / bits.sum() <= high

    def test_identical_receivers_are_indistinguishable(self):
        bits = np.full(30, 1000)
        errors = np.random.default_rng(1).integers(0, 50, 30)
        comparison = FrameBootstrap().compare(errors, errors, bits, ("a", "b"))
        assert comparison.difference == 0.0
        assert not comparison.is_decisive
        assert "no detectable" in comparison.verdict

    def test_a_clearly_better_receiver_is_detected(self):
        bits = np.full(30, 1000)
        rng = np.random.default_rng(2)
        worse = rng.binomial(1000, 0.05, 30)
        better = rng.binomial(1000, 0.01, 30)
        comparison = FrameBootstrap().compare(better, worse, bits, ("new", "old"))
        assert comparison.is_decisive
        assert comparison.verdict == "new has fewer errors"

    def test_pairing_detects_what_unpaired_frames_would_hide(self):
        """Huge frame-to-frame variation, tiny consistent difference.

        Every frame differs wildly in difficulty, but receiver A always
        makes exactly 2 fewer errors.  Pairing cancels the shared
        difficulty and finds the difference; it would be invisible if
        the two had been tested on different frames.
        """
        bits = np.full(40, 1000)
        difficulty = np.random.default_rng(3).integers(10, 400, 40)
        comparison = FrameBootstrap().compare(difficulty - 2, difficulty, bits)
        assert comparison.is_decisive

    def test_mismatched_inputs_are_refused(self):
        with pytest.raises(ValidationError):
            FrameBootstrap().interval(np.ones(3), np.ones(4))
        with pytest.raises(ValidationError):
            FrameBootstrap().interval(np.array([5]), np.array([3]))
        with pytest.raises(ValidationError):
            FrameBootstrap(resamples=10)


# ======================================================================
# Cells
# ======================================================================
class TestCellResult:
    def _cell(self, error_counts, bits_each=100, confidence=0.9):
        correct = []
        for errors in error_counts:
            mask = np.ones(bits_each, dtype=bool)
            mask[:errors] = False
            correct.append(mask)
        confidences = [np.full(bits_each, confidence) for _ in error_counts]
        return CellResult.from_outcomes(
            "rx", Scenario("QPSK", 0, ImpairmentProfile.clean()),
            confidences, correct, seconds=1.0,
        )

    def test_counts_are_pooled_correctly(self):
        cell = self._cell([0, 5, 50])
        assert cell.error_rate == ErrorRate(55, 300)
        assert cell.n_frames == 3

    def test_lost_frames_are_counted_by_the_threshold(self):
        cell = self._cell([0, 5, 50, 90])
        assert cell.lost_frames == 2

    def test_triage_is_undefined_when_nothing_was_lost(self):
        assert math.isnan(self._cell([0, 1, 2]).triage_quality)

    def test_per_frame_arrays_are_read_only(self):
        cell = self._cell([0, 1])
        with pytest.raises(ValueError):
            cell.frame_errors[0] = 99

    def test_clustered_interval_is_wider_on_clustered_errors(self):
        cell = self._cell([0] * 18 + [90, 90])
        wilson = np.diff(cell.ber_interval("wilson"))[0]
        bootstrap = np.diff(cell.ber_interval("bootstrap"))[0]
        assert bootstrap > wilson

    def test_the_conservative_interval_covers_both(self):
        cell = self._cell([0] * 18 + [90, 90])
        low, high = cell.ber_interval()
        for method in ("wilson", "bootstrap"):
            a, b = cell.ber_interval(method)
            assert low <= a and b <= high
        with pytest.raises(ValidationError):
            cell.ber_interval("vibes")


# ======================================================================
# The runner
# ======================================================================
class TestRunner:
    def test_every_cell_is_filled(self, result, small_grid, receivers):
        assert len(result) == len(small_grid) * len(receivers)
        assert result.skipped == []

    def test_every_receiver_sees_identical_frames(self, source, small_grid):
        """The rule every paired comparison depends on.

        Two copies of the *same* receiver must produce identical error
        counts frame by frame.  If the runner drew fresh frames per
        receiver, they would not.
        """
        pulse = source.simulator.pulse
        twins = {
            "a": ClassicalReceiver(pulse=pulse, demapper=SoftDemapper()),
            "b": ClassicalReceiver(pulse=pulse, demapper=SoftDemapper()),
        }
        run = BenchmarkRunner(source, twins, frames_per_scenario=3).run(small_grid)
        for scenario in small_grid:
            assert np.array_equal(
                run.cell("a", scenario).frame_errors,
                run.cell("b", scenario).frame_errors,
            )

    def test_reruns_reproduce_exactly(self, source, receivers, small_grid, result):
        again = BenchmarkRunner(source, receivers, frames_per_scenario=4).run(small_grid)
        for cell in result.cells:
            twin = again.cell(cell.receiver, cell.scenario)
            assert np.array_equal(cell.frame_errors, twin.frame_errors)

    def test_soft_and_hard_decide_identically(self, result):
        """Different confidence, same decisions: an end-to-end sanity check."""
        comparison = result.compare("soft", "hard")
        assert comparison.difference == 0.0

    def test_soft_confidence_ranks_better_than_hard(self, result):
        noisy = [s for s in result.scenarios if s.snr_db == 0.0 and s.profile.name == "clean"]
        for scenario in noisy:
            assert result.cell("soft", scenario).ranking_quality >= \
                result.cell("hard", scenario).ranking_quality

    def test_curves_come_back_in_snr_order(self, result):
        snr, ber, interval = result.curve("soft", "16QAM", "clean")
        assert list(snr) == [0.0, 8.0]
        assert ber[0] > ber[1]
        assert interval.shape == (2, 2)
        assert np.all(interval[:, 0] <= ber + 1e-12)

    def test_observers_hear_every_event(self, source, receivers, small_grid):
        class Recorder(BenchmarkObserver):
            def __init__(self):
                self.events = []

            def on_start(self, n, names):
                self.events.append(("start", n))

            def on_cell(self, cell, done, total):
                self.events.append(("cell", done, total))

            def on_finish(self, result):
                self.events.append(("finish", len(result)))

        recorder = Recorder()
        ticks = []
        grid = small_grid.where(modulation="QPSK", profile="clean")
        BenchmarkRunner(source, receivers, frames_per_scenario=1) \
            .add_observer(recorder) \
            .add_observer(CallbackObserver(lambda d, t, c: ticks.append(d))) \
            .run(grid)

        assert recorder.events[0] == ("start", len(grid))
        assert recorder.events[-1] == ("finish", len(grid) * 2)
        assert ticks == list(range(1, len(grid) * 2 + 1))

    def test_an_unsupported_modulation_is_skipped_and_reported(self, source, small_grid):
        pulse = source.simulator.pulse
        bank = ReceiverBank({"QPSK": ClassicalReceiver(pulse=pulse)}, name="qpsk-only")
        run = BenchmarkRunner(source, {"bank": bank}, frames_per_scenario=1).run(small_grid)
        assert len(run) == len(small_grid.where(modulation="QPSK"))
        assert len(run.skipped) == len(small_grid.where(modulation="16QAM"))
        assert all("16QAM" in reason for _, _, reason in run.skipped)

    def test_a_source_that_declines_is_skipped_not_faked(self, receivers, small_grid):
        class CleanOnly(SimulatedSource):
            def supports(self, scenario):
                return scenario.profile.name == "clean"

        run = BenchmarkRunner(CleanOnly(), receivers, frames_per_scenario=1).run(small_grid)
        assert {c.scenario.profile.name for c in run.cells} == {"clean"}
        assert len(run.skipped) == len(small_grid.where(profile="full")) * 2

    def test_results_from_two_runs_can_be_merged(self, result):
        merged = result + result
        assert len(merged) == len(result)   # same cells, deduplicated by key

    def test_bad_construction_is_refused(self, source, receivers):
        with pytest.raises(ValidationError):
            BenchmarkRunner("not a source", receivers)
        with pytest.raises(ValidationError):
            BenchmarkRunner(source, {})
        with pytest.raises(ValidationError):
            BenchmarkRunner(source, {"x": "not a receiver"})
        with pytest.raises(ValidationError):
            BenchmarkRunner(source, receivers, frames_per_scenario=0)

    def test_asking_for_a_missing_cell_names_it(self, result):
        missing = Scenario("64QAM", 0, ImpairmentProfile.clean())
        with pytest.raises(ValidationError, match="64QAM"):
            result.cell("soft", missing)


# ======================================================================
# Receiver bank
# ======================================================================
class TestReceiverBank:
    def test_frames_go_to_the_right_member(self, source):
        pulse = source.simulator.pulse
        soft = ClassicalReceiver(pulse=pulse, demapper=SoftDemapper(), name="soft")
        hard = ClassicalReceiver(pulse=pulse, demapper=HardDemapper(), name="hard")
        bank = ReceiverBank({"QPSK": soft, "16QAM": hard})
        assert bank.member("qpsk") is soft
        assert bank.member("16QAM") is hard

    def test_a_missing_modulation_is_refused_not_guessed(self, source):
        bank = ReceiverBank({"QPSK": ClassicalReceiver(pulse=source.simulator.pulse)})
        assert not bank.supports("64QAM")
        with pytest.raises(ReceiverError):
            bank.member("64QAM")

    def test_a_fallback_catches_the_rest(self, source):
        fallback = ClassicalReceiver(pulse=source.simulator.pulse)
        bank = ReceiverBank({"QPSK": ClassicalReceiver(pulse=source.simulator.pulse)},
                            fallback=fallback)
        assert bank.supports("64QAM")
        assert bank.member("64QAM") is fallback

    def test_a_bank_is_a_receiver(self, source, small_grid):
        bank = ReceiverBank({"QPSK": ClassicalReceiver(pulse=source.simulator.pulse)})
        frame = next(iter(source.frames(small_grid.where(modulation="QPSK")[0], 1)))
        assert bank.recover(frame).n_bits == frame.n_bits

    def test_an_empty_bank_is_refused(self):
        with pytest.raises(ValidationError):
            ReceiverBank({})


# ======================================================================
# Splits
# ======================================================================
@pytest.fixture(scope="module")
def grid():
    return ScenarioGrid.product(
        ["QPSK", "16QAM"], [-4, 0, 4, 8, 12], ImpairmentProfile.standard()
    )


class TestSplits:
    @pytest.mark.parametrize(
        "strategy",
        [RandomSplit(0.3, seed=1), HoldOutSNR(-10, 0), HoldOutProfile("full")],
        ids=lambda s: s.name,
    )
    def test_halves_are_disjoint_and_cover_the_grid(self, grid, strategy):
        train, test = strategy.split(grid)
        assert len(train & test) == 0
        assert train + test == grid
        assert len(train) > 0 and len(test) > 0

    def test_held_out_snr_tests_only_inside_the_band(self, grid):
        train, test = HoldOutSNR(-10, 0).split(grid)
        assert all(s.snr_db <= 0 for s in test)
        assert all(s.snr_db > 0 for s in train)

    def test_held_out_profile_never_trains_on_it(self, grid):
        train, test = HoldOutProfile("full").split(grid)
        assert test.profiles == ["full"]
        assert "full" not in train.profiles

    def test_random_split_is_reproducible(self, grid):
        assert RandomSplit(0.3, seed=4).split(grid) == RandomSplit(0.3, seed=4).split(grid)

    def test_a_split_that_selects_nothing_is_refused(self, grid):
        with pytest.raises(ValidationError):
            HoldOutSNR(50, 60).split(grid)
        with pytest.raises(ValidationError):
            HoldOutProfile("lunar").split(grid)

    def test_a_split_that_leaves_nothing_to_train_on_is_refused(self, grid):
        with pytest.raises(ValidationError):
            HoldOutSNR(-100, 100).split(grid)

    def test_bad_parameters_are_refused(self):
        with pytest.raises(ValidationError):
            RandomSplit(1.5)
        with pytest.raises(ValidationError):
            HoldOutSNR(10, 0)


# ======================================================================
# Store
# ======================================================================
class TestResultStore:
    def test_a_run_round_trips(self, result):
        with ResultStore(":memory:") as store:
            run_id = store.save(result, "test run", notes="from the tests")
            rows = store.cells(run_id)
            assert len(rows) == len(result)
            original = {(r["receiver"], r["modulation"], r["snr_db"], r["profile"]): r
                        for r in result.rows()}
            for row in rows:
                twin = original[(row["receiver"], row["modulation"], row["snr_db"], row["profile"])]
                assert row["errors"] == twin["errors"]
                assert row["ber"] == pytest.approx(twin["ber"])
                assert sum(row["frame_errors"]) == twin["errors"]

    def test_filters_narrow_the_rows(self, result):
        with ResultStore(":memory:") as store:
            run_id = store.save(result, "r")
            assert len(store.cells(run_id, receiver="soft")) == len(result) // 2
            assert all(r["profile"] == "full" for r in store.cells(run_id, profile="full"))

    def test_runs_are_listed_newest_first(self, result):
        with ResultStore(":memory:") as store:
            first = store.save(result, "first")
            second = store.save(result, "second")
            assert [r["id"] for r in store.runs()] == [second, first]
            assert store.latest() == second
            assert store.latest("first") == first

    def test_deleting_a_run_removes_its_cells(self, result):
        with ResultStore(":memory:") as store:
            run_id = store.save(result, "gone")
            store.delete(run_id)
            assert store.runs() == []
            assert store.cells(run_id) == []

    def test_nan_survives_storage(self, result):
        """No lost frames gives NaN triage; it must not come back as 0."""
        with ResultStore(":memory:") as store:
            run_id = store.save(result, "nan")
            assert any(math.isnan(r["triage_quality"]) for r in store.cells(run_id))

    def test_a_file_database_persists(self, result, tmp_path):
        path = tmp_path / "results.db"
        with ResultStore(path) as store:
            run_id = store.save(result, "on disk")
        with ResultStore(path) as reopened:
            assert len(reopened.cells(run_id)) == len(result)

    def test_a_closed_store_refuses_work(self, result):
        store = ResultStore(":memory:")
        store.close()
        store.close()  # idempotent
        with pytest.raises(ValidationError):
            store.runs()

    def test_bad_saves_are_refused(self, result):
        with ResultStore(":memory:") as store:
            with pytest.raises(ValidationError):
                store.save("not a result", "x")
            with pytest.raises(ValidationError):
                store.save(result, "   ")


# ======================================================================
# The fix that came out of the Phase 6 ablation
# ======================================================================
@pytest.fixture(scope="module")
def symbols_and_scheme():
    from satrecovery import get_modulation

    scheme = get_modulation("16QAM")
    rng = np.random.default_rng(0)
    clean = scheme.map_bits(rng.integers(0, 2, 4 * 400).astype(np.uint8))
    noise = (rng.normal(size=clean.size) + 1j * rng.normal(size=clean.size)) * 0.08
    return clean + noise, scheme


class TestNoiseNormalisedDemapper:
    def test_no_decision_ever_changes(self, symbols_and_scheme):
        from satrecovery.receivers.demapper import NoiseNormalisedDemapper

        symbols, scheme = symbols_and_scheme
        inner = SoftDemapper()
        for reference in (0.001, 0.01, 1.0):
            wrapped = NoiseNormalisedDemapper(inner, reference_noise=reference)
            assert np.array_equal(
                inner.demap(symbols, scheme)[0], wrapped.demap(symbols, scheme)[0]
            )

    def test_a_noisier_frame_gets_lower_confidence(self, symbols_and_scheme):
        from satrecovery.receivers.demapper import NoiseNormalisedDemapper

        symbols, scheme = symbols_and_scheme
        rng = np.random.default_rng(1)
        noisier = symbols + (rng.normal(size=symbols.size) + 1j * rng.normal(size=symbols.size)) * 0.08

        class FixedOdds(SoftDemapper):
            """A demapper with no noise awareness of its own."""
            def _estimate_noise_power(self, squared):
                return 0.01

        wrapped = NoiseNormalisedDemapper(FixedOdds(), reference_noise=0.01)
        quiet = np.mean(wrapped.demap(symbols, scheme)[1])
        loud = np.mean(wrapped.demap(noisier, scheme)[1])
        assert loud < quiet

    def test_the_reference_ignores_a_few_broken_frames(self, symbols_and_scheme):
        from satrecovery.receivers.demapper import NoiseNormalisedDemapper, estimate_noise_power

        symbols, scheme = symbols_and_scheme
        broken = np.random.default_rng(2).normal(size=symbols.size) * 3 + 0j
        typical = estimate_noise_power(symbols, scheme)
        reference = NoiseNormalisedDemapper.fit_reference([symbols] * 5 + [broken], scheme)
        assert reference == pytest.approx(typical)

    def test_a_demapper_without_odds_is_refused(self, symbols_and_scheme):
        from satrecovery.receivers.demapper import NoiseNormalisedDemapper

        symbols, scheme = symbols_and_scheme
        with pytest.raises(ValidationError, match="log-likelihood"):
            NoiseNormalisedDemapper(HardDemapper(), 0.01).demap(symbols, scheme)

    def test_bad_construction_is_refused(self):
        from satrecovery.receivers.demapper import NoiseNormalisedDemapper

        with pytest.raises(ValidationError):
            NoiseNormalisedDemapper("not a demapper", 0.01)
        with pytest.raises(ValidationError):
            NoiseNormalisedDemapper(SoftDemapper(), 0.0)


def test_a_router_sends_each_scenario_to_the_first_source_that_can_take_it():
    from satrecovery.evaluation import ImpairmentProfile, Scenario, SimulatedSource, SourceRouter
    from satrecovery.evaluation.sources import FrameSource

    class Nothing(FrameSource):
        def supports(self, scenario):
            return False

        def _frames(self, scenario, count):
            yield from ()

    simulated = SimulatedSource(seed=1)
    router = SourceRouter([Nothing("nothing"), simulated])
    clean = Scenario("QPSK", 6, ImpairmentProfile.clean())
    assert router.source_for(clean) is simulated and router.name == "nothing+simulated"
    assert len(list(router.frames(clean, 2))) == 2
    assert not router.supports(Scenario("QPSK", 6, ImpairmentProfile.recorded()))
