"""Tests for Phase 5 -- the confidence estimators, calibrators and metrics.

The metrics are tested first and hardest, because everything else in
the phase is judged by them.  Each one is checked against a case whose
answer is known in advance: a source that is honest by construction, a
source that is overconfident by a known amount, a ranking that is
perfect and one that is worthless.

The most important tests in this file are the three that assert what
calibration *cannot* do.  The project's argument is that a badly scaled
confidence and an uninformative one are different problems, and that
only the first is repairable.  If calibration could quietly improve a
ranking, that argument would be wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from satrecovery import (
    ChannelConditions,
    ClassicalReceiver,
    SignalSimulator,
    SoftDemapper,
    get_modulation,
)
from satrecovery.confidence import (
    CalibrationReport,
    CombinedConfidence,
    EVMConfidence,
    HistogramBinning,
    IdentityCalibrator,
    LLRConfidence,
    MarginConfidence,
    MCDropoutConfidence,
    ModelEnsembleConfidence,
    RiskCoverageCurve,
    TemperatureScaling,
)
from satrecovery.core.exceptions import ModelNotTrainedError, ValidationError
from satrecovery.receivers.demapper import HardDemapper
from satrecovery.receivers.result import RecoveryResult

torch = pytest.importorskip("torch", reason="the learned estimators need PyTorch")


# ======================================================================
# Fixtures: synthetic sources whose behaviour is known exactly
# ======================================================================
@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20250515)


@pytest.fixture(scope="module")
def honest(rng):
    """A perfectly calibrated source: right exactly as often as it claims."""
    confidence = rng.uniform(0.5, 1.0, 100_000)
    correct = rng.random(confidence.size) < confidence
    return confidence, correct


@pytest.fixture(scope="module")
def boastful(rng):
    """Claims 0.99 every time; right 80% of the time."""
    confidence = np.full(100_000, 0.99)
    correct = rng.random(confidence.size) < 0.80
    return confidence, correct


@pytest.fixture(scope="module")
def recovered():
    """A real recovered frame with a useful number of errors in it."""
    simulator = SignalSimulator(seed=11)
    receiver = ClassicalReceiver(pulse=simulator.pulse, demapper=SoftDemapper())
    conditions = ChannelConditions(snr_db=4.0, phase_offset_rad=0.3)
    frame = simulator.generate_frame("16QAM", conditions)
    result = receiver.recover(frame)
    return result, np.asarray(frame.bits)


# ======================================================================
# CalibrationReport
# ======================================================================
class TestCalibrationReport:
    def test_honest_source_is_measured_as_honest(self, honest):
        report = CalibrationReport(*honest)
        # The floor is finite-sample noise, not bias: 15 bins over
        # 100k samples cannot resolve better than about a percent.
        assert report.ece < 0.01
        assert abs(report.overconfidence) < 0.01

    def test_overconfidence_is_measured_at_its_true_size(self, boastful):
        report = CalibrationReport(*boastful)
        assert report.overconfidence == pytest.approx(0.19, abs=0.01)
        assert report.ece == pytest.approx(0.19, abs=0.01)

    def test_underconfidence_carries_the_opposite_sign(self, rng):
        confidence = np.full(50_000, 0.60)
        correct = rng.random(confidence.size) < 0.95
        report = CalibrationReport(confidence, correct)
        assert report.overconfidence < -0.3
        # The calibration error itself is unsigned.
        assert report.ece > 0.3

    def test_calibration_error_cannot_tell_useless_from_useful(self, rng):
        """A source that always says 0.5 is perfectly calibrated and useless.

        This is why the project never reports a calibration error on its
        own.  If it did, the best-looking receiver would be the one that
        refuses to express an opinion.
        """
        correct = rng.random(50_000) < 0.5
        useless = CalibrationReport(np.full(correct.size, 0.5), correct)
        assert useless.ece < 0.01
        # ...but a proper scoring rule sees through it.
        informative = CalibrationReport(
            np.where(correct, 0.9, 0.1).astype(float), correct
        )
        assert informative.brier < useless.brier
        assert informative.nll < useless.nll

    def test_confident_mistakes_are_punished_harder_by_nll(self, rng):
        correct = rng.random(20_000) < 0.9
        mild = CalibrationReport(np.full(correct.size, 0.95), correct)
        extreme = CalibrationReport(np.full(correct.size, 0.999_9), correct)
        assert extreme.nll > mild.nll
        # The Brier score moves far less for the same mistake.
        assert (extreme.nll / mild.nll) > (extreme.brier / mild.brier)

    def test_both_binnings_recover_a_known_gap(self, rng):
        """Equal-width and equal-mass bins must agree when the truth is uniform."""
        confidence = np.clip(1.0 - 10.0 ** rng.uniform(-6, -1, 100_000), 0.5, 1.0)
        correct = rng.random(confidence.size) < (confidence - 0.05)
        for binning in ("width", "mass"):
            report = CalibrationReport(confidence, correct, binning=binning)
            assert report.ece == pytest.approx(0.05, abs=0.01)

    def test_bins_sum_to_the_whole_set(self, honest):
        report = CalibrationReport(*honest)
        assert report.bin_counts.sum() == report.n_decisions
        assert report.bin_edges.size == report.bin_counts.size + 1

    def test_reported_arrays_cannot_be_edited(self, honest):
        report = CalibrationReport(*honest)
        with pytest.raises(ValueError):
            report.bin_accuracy[0] = 0.0
        with pytest.raises(ValueError):
            report.bin_edges[0] = 0.0

    def test_from_result_grades_the_receivers_own_numbers(self, recovered):
        result, true_bits = recovered
        report = CalibrationReport.from_result(result, true_bits)
        assert report.n_decisions == result.n_bits
        assert report.label == result.receiver
        assert report.accuracy == pytest.approx(1.0 - result.ber(true_bits))

    def test_from_result_can_grade_a_replacement_estimate(self, recovered):
        result, true_bits = recovered
        scores = EVMConfidence().estimate(result)
        report = CalibrationReport.from_result(result, true_bits, scores)
        assert report.mean_confidence == pytest.approx(float(np.mean(scores)))

    @pytest.mark.parametrize(
        "confidence, correct, field",
        [
            (np.array([]), np.array([]), "confidence"),
            (np.array([0.9, 0.8]), np.array([True]), "correct"),
            (np.array([1.4]), np.array([True]), "confidence"),
            (np.array([-0.1]), np.array([True]), "confidence"),
        ],
    )
    def test_bad_input_is_rejected(self, confidence, correct, field):
        with pytest.raises(ValidationError) as excinfo:
            CalibrationReport(confidence, correct)
        assert excinfo.value.field == field

    def test_bad_options_are_rejected(self, honest):
        with pytest.raises(ValidationError):
            CalibrationReport(*honest, n_bins=1)
        with pytest.raises(ValidationError):
            CalibrationReport(*honest, binning="sideways")
        with pytest.raises(ValidationError):
            CalibrationReport(*honest, lo=1.0)


# ======================================================================
# RiskCoverageCurve
# ======================================================================
class TestRiskCoverageCurve:
    def test_a_perfect_ranking_scores_one(self, rng):
        correct = rng.random(20_000) < 0.9
        curve = RiskCoverageCurve(correct.astype(float), correct)
        assert curve.ranking_quality == pytest.approx(1.0)
        assert curve.excess_aurc == pytest.approx(0.0, abs=1e-12)

    def test_a_worthless_ranking_scores_about_zero(self, rng):
        correct = rng.random(20_000) < 0.9
        curve = RiskCoverageCurve(rng.random(correct.size), correct)
        assert abs(curve.ranking_quality) < 0.05

    def test_ties_are_not_allowed_to_flatter_the_result(self):
        """Every confidence identical must score as worthless, not perfect.

        Without the tie shuffle the errors' positions in the array decide
        the curve.  Put every error at the end -- which is exactly what a
        receiver with an acquisition transient does -- and an unshuffled
        sort would report a perfect ranking from a constant.
        """
        correct = np.r_[np.ones(5_000, bool), np.zeros(5_000, bool)]
        curve = RiskCoverageCurve(np.ones(correct.size), correct)
        assert abs(curve.ranking_quality) < 0.05

    def test_a_partly_useful_ranking_lands_in_between(self, rng):
        correct = rng.random(20_000) < 0.9
        noisy = np.where(
            correct, rng.uniform(0.4, 1.0, correct.size), rng.uniform(0.0, 0.6, correct.size)
        )
        curve = RiskCoverageCurve(noisy, correct)
        assert 0.3 < curve.ranking_quality < 0.99

    @pytest.mark.parametrize("accuracy", [0.5, 0.9, 0.99, 1.0])
    def test_oracle_area_matches_a_brute_force_count(self, accuracy):
        n = 4_000
        correct = np.zeros(n, dtype=bool)
        correct[: int(accuracy * n)] = True
        curve = RiskCoverageCurve(correct.astype(float), correct)

        kept = np.arange(1, n + 1, dtype=float)
        expected = np.mean(np.clip(kept - int(accuracy * n), 0.0, None) / kept)
        assert curve.aurc_oracle == pytest.approx(expected)
        # A perfect ranking achieves exactly the oracle area.
        assert curve.aurc == pytest.approx(expected)

    def test_full_coverage_risk_is_the_ordinary_error_rate(self, rng):
        correct = rng.random(5_000) < 0.8
        curve = RiskCoverageCurve(rng.random(correct.size), correct)
        assert curve.risk_at_coverage(1.0) == pytest.approx(curve.error_rate)
        assert curve.risk[-1] == pytest.approx(curve.error_rate)

    def test_coverage_at_risk_answers_the_operational_question(self):
        correct = np.ones(10_000, dtype=bool)
        correct[:1_000] = False          # 10% wrong
        curve = RiskCoverageCurve(correct.astype(float), correct)
        # A perfect ranking keeps every correct bit at zero risk.
        assert curve.coverage_at_risk(0.0) == pytest.approx(0.9)
        # And relaxing the target can only let more through.
        assert curve.coverage_at_risk(0.01) >= curve.coverage_at_risk(0.0)
        assert curve.coverage_at_risk(1.0) == pytest.approx(1.0)

    def test_an_impossible_target_keeps_nothing(self):
        """When the most confident decision is itself wrong, keep none of it.

        An inverted ranking -- the receiver is surest about exactly the
        bits it got wrong -- is the worst case, and the honest answer is
        a coverage of zero rather than a small comforting number.
        """
        correct = np.r_[np.zeros(100, bool), np.ones(900, bool)]
        inverted = np.r_[np.ones(100), np.zeros(900)]
        curve = RiskCoverageCurve(inverted, correct)
        assert curve.coverage_at_risk(0.0) == 0.0
        assert curve.ranking_quality < -0.5   # worse than useless

    def test_a_flawless_frame_ranks_perfectly_by_definition(self):
        correct = np.ones(500, dtype=bool)
        curve = RiskCoverageCurve(np.full(500, 0.9), correct)
        assert curve.ranking_quality == 1.0
        assert curve.aurc == 0.0

    def test_from_result_uses_the_frames_own_truth(self, recovered):
        result, true_bits = recovered
        curve = RiskCoverageCurve.from_result(result, true_bits)
        assert curve.error_rate == pytest.approx(result.ber(true_bits))

    def test_arrays_are_read_only(self, rng):
        correct = rng.random(100) < 0.9
        curve = RiskCoverageCurve(rng.random(100), correct)
        with pytest.raises(ValueError):
            curve.risk[0] = 0.0
        with pytest.raises(ValueError):
            curve.coverage[0] = 0.0

    def test_bad_coverage_is_rejected(self, rng):
        correct = rng.random(100) < 0.9
        curve = RiskCoverageCurve(rng.random(100), correct)
        for bad in (0.0, -0.5, 1.5):
            with pytest.raises(ValidationError):
                curve.risk_at_coverage(bad)
        with pytest.raises(ValidationError):
            curve.coverage_at_risk(-0.1)


# ======================================================================
# Estimators
# ======================================================================
class TestEstimators:
    def test_every_estimator_returns_one_probability_per_bit(self, recovered):
        result, _ = recovered
        for estimator in (LLRConfidence(), EVMConfidence(), MarginConfidence()):
            scores = estimator.estimate(result)
            assert scores.shape == (result.n_bits,)
            assert scores.min() >= 0.0 and scores.max() <= 1.0

    def test_the_modulation_is_found_without_being_passed(self, recovered):
        result, _ = recovered
        explicit = EVMConfidence().estimate(result, get_modulation("16QAM"))
        implied = EVMConfidence().estimate(result)
        assert np.array_equal(explicit, implied)

    def test_an_estimator_says_plainly_what_it_is_missing(self):
        bare = RecoveryResult(np.zeros(8, dtype=np.uint8), modulation="QPSK")
        with pytest.raises(ValidationError, match="log-likelihood"):
            LLRConfidence().estimate(bare)
        with pytest.raises(ValidationError, match="symbols"):
            EVMConfidence().estimate(bare)

    def test_a_result_that_names_no_modulation_is_rejected(self):
        bare = RecoveryResult(np.zeros(8, dtype=np.uint8))
        with pytest.raises(ValidationError, match="modulation"):
            LLRConfidence().estimate(bare)

    def test_symbol_scores_must_match_the_frame(self, recovered):
        """A result whose symbols and bits disagree is a bug, not a round-off."""
        result, _ = recovered
        mangled = RecoveryResult(
            result.bits[:-4],
            symbols=result.symbols,
            modulation="16QAM",
        )
        with pytest.raises(ValidationError):
            EVMConfidence().estimate(mangled)

    def test_the_evm_scale_changes_calibration_but_never_ranking(self, recovered):
        """The headline property of the whole phase, asserted on real data.

        Two estimators that differ only by a monotone squashing produce
        different calibration numbers and *identical* risk-coverage
        curves.  Scale is repairable; ranking is not.
        """
        result, true_bits = recovered
        correct = ~result.error_mask(true_bits)

        tight = EVMConfidence(scale=1.0).estimate(result)
        loose = EVMConfidence(scale=1.4).estimate(result)

        assert CalibrationReport(tight, correct).ece != pytest.approx(
            CalibrationReport(loose, correct).ece, abs=1e-6
        )
        assert RiskCoverageCurve(tight, correct).aurc == pytest.approx(
            RiskCoverageCurve(loose, correct).aurc, abs=1e-12
        )

    def test_a_bad_scale_is_rejected(self):
        for bad in (0.0, -1.0):
            with pytest.raises(ValidationError):
                EVMConfidence(scale=bad)


class TestCombining:
    def test_the_or_operator_builds_a_composite(self):
        combined = LLRConfidence() | EVMConfidence()
        assert isinstance(combined, CombinedConfidence)
        assert len(combined) == 2

    def test_chaining_flattens_rather_than_nesting(self):
        """``a | b | c`` must weight all three equally, not a and b double."""
        combined = LLRConfidence() | EVMConfidence() | MarginConfidence()
        assert len(combined) == 3
        assert [e.name for e in combined] == ["LLR", "EVM", "Margin"]

    def test_a_composite_is_itself_an_estimator(self, recovered):
        result, _ = recovered
        nested = (LLRConfidence() | EVMConfidence()) | MarginConfidence()
        scores = nested.estimate(result)
        assert scores.shape == (result.n_bits,)

    def test_the_reduction_does_what_it_says(self, recovered):
        result, _ = recovered
        members = [LLRConfidence(), EVMConfidence()]
        lowest = CombinedConfidence(members, reduce="min").estimate(result)
        middle = CombinedConfidence(members, reduce="mean").estimate(result)
        highest = CombinedConfidence(members, reduce="max").estimate(result)
        assert np.all(lowest <= middle + 1e-12)
        assert np.all(middle <= highest + 1e-12)

    def test_a_composite_needs_real_estimators(self):
        with pytest.raises(ValidationError):
            CombinedConfidence([])
        with pytest.raises(ValidationError):
            CombinedConfidence([LLRConfidence(), "not an estimator"])
        with pytest.raises(ValidationError):
            CombinedConfidence([LLRConfidence()], reduce="median")

    def test_or_with_something_else_is_refused(self):
        with pytest.raises(TypeError):
            LLRConfidence() | 7


# ======================================================================
# Calibration
# ======================================================================
class TestCalibration:
    def test_an_unfitted_calibrator_refuses_to_work(self):
        with pytest.raises(ModelNotTrainedError):
            TemperatureScaling().apply(np.array([0.9]))

    def test_temperature_above_one_means_the_source_was_overconfident(self, boastful):
        calibrator = TemperatureScaling().fit(*boastful)
        assert calibrator.is_fitted
        assert calibrator.temperature > 1.0

    def test_temperature_below_one_means_the_source_was_timid(self, rng):
        confidence = np.full(50_000, 0.6)
        correct = rng.random(confidence.size) < 0.95
        calibrator = TemperatureScaling().fit(confidence, correct)
        assert calibrator.temperature < 1.0

    def test_calibration_is_fitted_and_judged_on_different_data(self, rng):
        """Fit on one half, measure on the other. The gap must still close."""
        confidence = np.full(100_000, 0.99)
        correct = rng.random(confidence.size) < 0.80
        fit_half, test_half = slice(None, 50_000), slice(50_000, None)

        calibrator = TemperatureScaling().fit(
            confidence[fit_half], correct[fit_half]
        )
        before = CalibrationReport(confidence[test_half], correct[test_half])
        after = CalibrationReport(
            calibrator.apply(confidence[test_half]), correct[test_half]
        )
        assert after.ece < 0.25 * before.ece
        assert after.nll < before.nll

    def test_temperature_scaling_cannot_change_a_single_bit(self, recovered):
        """A monotone rescaling leaves every decision exactly as it was."""
        result, true_bits = recovered
        correct = ~result.error_mask(true_bits)
        scores = LLRConfidence().estimate(result)

        calibrated = TemperatureScaling().fit(scores, correct).apply(scores)
        # Nothing crossed the even-odds line, so no decision flipped.
        assert np.array_equal(scores >= 0.5, calibrated >= 0.5)

    def test_temperature_scaling_cannot_change_the_ranking(self, recovered):
        """The other half of the same claim, measured rather than argued."""
        result, true_bits = recovered
        correct = ~result.error_mask(true_bits)
        scores = EVMConfidence().estimate(result)

        calibrated = TemperatureScaling().fit(scores, correct).apply(scores)
        assert RiskCoverageCurve(scores, correct).aurc == pytest.approx(
            RiskCoverageCurve(calibrated, correct).aurc, abs=1e-12
        )

    def test_histogram_binning_handles_a_shape_temperature_cannot(self, rng):
        """Overconfident at the top, underconfident in the middle.

        One parameter cannot bend in two directions at once; a
        non-parametric calibrator can, and this is the case that shows
        the difference rather than asserting it.
        """
        confidence = rng.uniform(0.5, 1.0, 120_000)
        truth = np.where(confidence > 0.8, confidence - 0.15, confidence + 0.12)
        correct = rng.random(confidence.size) < np.clip(truth, 0.0, 1.0)

        fit, test = slice(None, 60_000), slice(60_000, None)
        before = CalibrationReport(confidence[test], correct[test]).ece
        temperature = CalibrationReport(
            TemperatureScaling().fit(confidence[fit], correct[fit]).apply(
                confidence[test]
            ),
            correct[test],
        ).ece
        histogram = CalibrationReport(
            HistogramBinning(n_bins=20).fit(confidence[fit], correct[fit]).apply(
                confidence[test]
            ),
            correct[test],
        ).ece
        assert histogram < temperature
        assert histogram < before

    def test_histogram_binning_can_lose_resolution_but_not_reorder(self, recovered):
        """The precise version of the claim, which is weaker than it looks.

        Temperature scaling leaves the ranking identical.  Histogram
        binning collapses every value in a bin to one number, so bits
        that used to be distinguishable become tied -- the ranking can
        lose resolution.  What it still cannot do is put a worse bit
        ahead of a better one, so the quality may fall slightly and
        cannot rise.
        """
        result, true_bits = recovered
        correct = ~result.error_mask(true_bits)
        scores = EVMConfidence().estimate(result)

        before = RiskCoverageCurve(scores, correct).ranking_quality
        binned = HistogramBinning(n_bins=20).fit(scores, correct).apply(scores)
        after = RiskCoverageCurve(binned, correct).ranking_quality
        assert after <= before + 1e-9
        assert after > before - 0.05

    def test_histogram_binning_stays_monotone(self, rng):
        """A noisy bin must not be allowed to reorder two confidences."""
        confidence = rng.uniform(0.5, 1.0, 2_000)
        correct = rng.random(confidence.size) < 0.7
        calibrator = HistogramBinning(n_bins=30).fit(confidence, correct)
        assert np.all(np.diff(calibrator.bin_values) >= -1e-12)

    def test_the_identity_calibrator_is_a_real_control(self, honest):
        confidence, correct = honest
        calibrator = IdentityCalibrator().fit(confidence, correct)
        assert np.array_equal(calibrator.apply(confidence), confidence)

    def test_a_calibrator_can_be_used_as_a_function(self, boastful):
        calibrator = TemperatureScaling().fit(*boastful)
        assert np.array_equal(
            calibrator(boastful[0][:10]), calibrator.apply(boastful[0][:10])
        )

    @pytest.mark.parametrize(
        "confidence, correct",
        [
            (np.array([]), np.array([])),
            (np.array([0.9, 0.8]), np.array([True])),
            (np.array([1.5]), np.array([True])),
        ],
    )
    def test_bad_fitting_data_is_rejected(self, confidence, correct):
        with pytest.raises(ValidationError):
            TemperatureScaling().fit(confidence, correct)

    def test_bad_construction_is_rejected(self):
        with pytest.raises(ValidationError):
            TemperatureScaling(iterations=1)
        with pytest.raises(ValidationError):
            HistogramBinning(n_bins=1)


# ======================================================================
# The sampled estimators, which need a trained network
# ======================================================================
@pytest.fixture(scope="module")
def trained_pair():
    """Two small demappers: one with dropout, one without.

    Deliberately tiny and briefly trained.  These tests check the
    *mechanism* -- that sampling varies, that it leaves the model
    untouched, that a dropout-free network is refused -- not whether the
    network is any good.  That is the demo's job, with real data.
    """
    from satrecovery.learning.datasets import (
        DemappingDataset,
        generate_frames,
        training_conditions,
    )
    from satrecovery.learning.neural import MLPDemapper

    simulator = SignalSimulator(seed=5)
    modulation = get_modulation("QPSK")
    receiver = ClassicalReceiver(pulse=simulator.pulse, demapper=SoftDemapper())

    frames = generate_frames(
        simulator,
        ["QPSK"],
        training_conditions(snr_range=(4.0, 16.0), steps=6, with_impairments=False),
        frames_each=2,
    )
    inputs, targets = DemappingDataset(receiver, window=1).build(frames)

    models = []
    for seed, dropout in ((0, 0.2), (1, 0.0)):
        model = MLPDemapper(
            modulation, window=1, hidden=(32, 32), seed=seed, dropout=dropout
        )
        model.fit(inputs, targets, epochs=6, batch_size=512, verbose=False)
        models.append(model)
    return models


class TestSampledEstimators:
    def test_dropout_sampling_needs_dropout(self, trained_pair):
        _, deterministic = trained_pair
        with pytest.raises(ValidationError, match="dropout"):
            MCDropoutConfidence(deterministic)

    def test_passes_actually_differ(self, trained_pair, recovered):
        stochastic, _ = trained_pair
        symbols = np.exp(1j * np.linspace(0, 6.0, 64)) * 0.7
        draws = stochastic.sample_bit_probabilities(symbols, passes=8, seed=0)
        assert draws.shape == (8, symbols.size * 2)
        assert draws.std(axis=0).max() > 0.0

    def test_sampling_leaves_the_model_as_it_found_it(self, trained_pair):
        stochastic, _ = trained_pair
        symbols = np.full(16, 0.7 + 0.7j)
        before = stochastic.demap(symbols, stochastic.modulation)[2]
        stochastic.sample_bit_probabilities(symbols, passes=4, seed=3)
        after = stochastic.demap(symbols, stochastic.modulation)[2]
        # Still in eval mode, so ordinary demapping is deterministic.
        assert np.array_equal(before, after)

    def test_an_ensemble_may_disagree_with_the_receiver(self):
        """The expressiveness no demapper has: a score below even odds."""
        modulation = get_modulation("QPSK")
        simulator = SignalSimulator(seed=7)
        receiver = ClassicalReceiver(pulse=simulator.pulse, demapper=SoftDemapper())
        frame = simulator.generate_frame("QPSK", ChannelConditions(snr_db=-2.0))
        result = receiver.recover(frame)

        estimator = ModelEnsembleConfidence([SoftDemapper(), HardDemapper()])
        scores = estimator.estimate(result, modulation)
        assert scores.shape == (result.n_bits,)
        assert scores.min() >= 0.0

    def test_an_ensemble_needs_demappers(self):
        with pytest.raises(ValidationError):
            ModelEnsembleConfidence([SoftDemapper(), "not a demapper"])

    def test_too_few_passes_is_rejected(self, trained_pair):
        stochastic, _ = trained_pair
        with pytest.raises(ValidationError):
            MCDropoutConfidence(stochastic, passes=1)
        with pytest.raises(ValidationError):
            stochastic.sample_bit_probabilities(np.ones(4), passes=1)

    def test_an_unknown_aggregation_is_rejected(self, trained_pair):
        stochastic, _ = trained_pair
        with pytest.raises(ValidationError):
            MCDropoutConfidence(stochastic, aggregate="vote")

    def test_the_spread_rule_is_never_more_confident_than_the_mean(self):
        """Disagreement can only reduce confidence, never manufacture it."""
        modulation = get_modulation("QPSK")
        simulator = SignalSimulator(seed=9)
        receiver = ClassicalReceiver(pulse=simulator.pulse, demapper=SoftDemapper())
        frame = simulator.generate_frame("QPSK", ChannelConditions(snr_db=2.0))
        result = receiver.recover(frame)

        members = [SoftDemapper(), HardDemapper()]
        plain = ModelEnsembleConfidence(members, aggregate="mean").estimate(
            result, modulation
        )
        penalised = ModelEnsembleConfidence(members, aggregate="spread").estimate(
            result, modulation
        )
        assert np.all(np.abs(penalised - 0.5) <= np.abs(plain - 0.5) + 1e-12)
