"""Tests for the learned components.

Two kinds of test here.

The ordinary kind checks the machinery: an untrained model refuses to be
used, a saved model reloads, a callback fires.

The more interesting kind checks the *claims*.  Cumulant features are
supposed to equal particular values for particular constellations, and
supposed to be unchanged when a signal is scaled or rotated -- so those
are checked against the published table rather than against whatever the
code happened to produce.  If the features were wrong, every result
built on them would be wrong in a way no accuracy number would reveal.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Phase 4 needs PyTorch")

from satrecovery import (  # noqa: E402
    ChannelConditions,
    ClassicalReceiver,
    Demapper,
    ModulationRegistry,
    Receiver,
    SignalSimulator,
    SoftDemapper,
)
from satrecovery.core.exceptions import (  # noqa: E402
    ModelNotTrainedError,
    ValidationError,
)
from satrecovery.learning import (  # noqa: E402
    ClassificationDataset,
    CNNDemodulator,
    CombinedFeatures,
    CumulantFeatures,
    DemappingDataset,
    EarlyStopping,
    EndToEndDataset,
    FeatureExtractor,
    ForestModulationClassifier,
    MLPDemapper,
    SpectralFeatures,
    TrainableModel,
    TrainingCallback,
    TrainingHistory,
    default_extractor,
    generate_frames,
    split_rows,
    training_conditions,
)
from satrecovery.receivers.neural import HybridReceiver, NeuralReceiver  # noqa: E402

SIM = SignalSimulator(frame_length=2048, samples_per_symbol=8, seed=41)


def clean_symbols(name: str, n: int = 4000, seed: int = 0) -> np.ndarray:
    """Noiseless symbols drawn uniformly from a constellation."""
    scheme = ModulationRegistry.create(name)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n * scheme.bits_per_symbol, dtype=np.uint8)
    return scheme.map_bits(bits)


# ==========================================================================
# Features: checked against the published cumulant table
# ==========================================================================
def test_feature_extractor_is_abstract():
    with pytest.raises(TypeError):
        FeatureExtractor()  # type: ignore[abstract]


@pytest.mark.parametrize(
    ("name", "expected_c40", "expected_c42"),
    [
        # The standard normalised cumulant magnitudes. These are
        # published constants, not numbers this project chose.
        ("BPSK", 2.0, 2.0),
        ("QPSK", 1.0, 1.0),
        ("8PSK", 0.0, 1.0),
        ("16QAM", 0.68, 0.68),
        ("64QAM", 0.62, 0.62),
    ],
)
def test_cumulants_match_the_published_values(name, expected_c40, expected_c42):
    """If these are wrong, every classifier result built on them is too."""
    extractor = CumulantFeatures()
    features = extractor.extract(clean_symbols(name, n=20000))
    names = extractor.feature_names

    measured_c40 = features[names.index("abs_C40")]
    measured_c42 = features[names.index("abs_C42")]

    assert measured_c40 == pytest.approx(expected_c40, abs=0.06), name
    assert measured_c42 == pytest.approx(expected_c42, abs=0.06), name


def test_8psk_fourth_order_cumulant_vanishes():
    """A result worth isolating: 8PSK has |C40| = 0, unlike every other scheme.

    It is what separates 8PSK from QPSK most cleanly, and it falls
    straight out of the constellation's eight-fold symmetry.
    """
    extractor = CumulantFeatures()
    names = extractor.feature_names
    value = extractor.extract(clean_symbols("8PSK", n=20000))[names.index("abs_C40")]
    assert value < 0.08


@pytest.mark.parametrize("name", ["QPSK", "16QAM", "64QAM"])
def test_cumulants_ignore_amplitude_and_rotation(name):
    """The property that makes them usable on a real link.

    A signal arrives at an unknown strength and an unknown phase. A
    feature that changed with either would be describing the channel
    rather than the modulation.
    """
    extractor = CumulantFeatures()
    symbols = clean_symbols(name, n=20000)

    original = extractor.extract(symbols)
    scaled = extractor.extract(symbols * 7.3)
    rotated = extractor.extract(symbols * np.exp(1j * 0.9))

    assert np.allclose(original, scaled, atol=1e-6)
    assert np.allclose(original, rotated, atol=0.02)


def test_papr_separates_constant_envelope_schemes():
    """Peak-to-average power alone tells PSK from QAM."""
    extractor = SpectralFeatures()
    index = extractor.feature_names.index("papr_db")

    psk = extractor.extract(clean_symbols("QPSK", n=8000))[index]
    qam = extractor.extract(clean_symbols("64QAM", n=8000))[index]

    assert psk < 1.5, "PSK points all have the same amplitude"
    assert qam > 3.0, "QAM amplitudes vary a lot"


def test_combined_features_is_a_composite():
    combined = CombinedFeatures(CumulantFeatures(), SpectralFeatures())
    assert isinstance(combined, FeatureExtractor)
    assert combined.n_features == (
        CumulantFeatures().n_features + SpectralFeatures().n_features
    )
    assert len(combined.feature_names) == combined.n_features
    assert combined.extract(clean_symbols("QPSK", 2000)).size == combined.n_features


def test_combined_features_needs_at_least_one_extractor():
    with pytest.raises(ValidationError):
        CombinedFeatures()


def test_features_never_contain_nan():
    """A single NaN would poison every tree that split on that feature."""
    extractor = default_extractor()
    for samples in (
        np.zeros(512, dtype=complex),
        np.ones(512, dtype=complex),
        clean_symbols("16APSK", 512),
    ):
        assert np.all(np.isfinite(extractor.extract(samples)))


# ==========================================================================
# The model lifecycle
# ==========================================================================
def test_trainable_model_is_abstract():
    with pytest.raises(TypeError):
        TrainableModel()  # type: ignore[abstract]


def test_untrained_model_refuses_to_be_used():
    """Silently returning a random initialisation would be far worse.

    An untrained network produces confident nonsense that looks exactly
    like a trained model performing badly.
    """
    scheme = ModulationRegistry.create("QPSK")
    demapper = MLPDemapper(scheme, window=1)

    assert not demapper.is_trained
    with pytest.raises(ModelNotTrainedError):
        demapper.demap(scheme.constellation, scheme)


def test_untrained_forest_refuses_to_predict():
    forest = ForestModulationClassifier()
    with pytest.raises(ModelNotTrainedError):
        forest.predict(np.zeros((2, forest.extractor.n_features)))


def test_history_records_and_reports():
    history = TrainingHistory("demo")
    history.record(1, loss=1.0, val_loss=1.2)
    history.record(2, loss=0.5, val_loss=0.9)

    assert history.n_epochs == 2
    assert history.series("loss") == [1.0, 0.5]
    assert history.last()["val_loss"] == 0.9
    assert history.best("loss") == 0.5
    assert set(history.keys()) == {"loss", "val_loss"}


# ==========================================================================
# Callbacks: the observer pattern
# ==========================================================================
def test_callbacks_are_told_what_happened():
    class Spy(TrainingCallback):
        def __init__(self):
            self.began = 0
            self.epochs = []
            self.ended = 0

        def on_train_begin(self, total_epochs):
            self.began = total_epochs

        def on_epoch_end(self, epoch, metrics):
            self.epochs.append(epoch)
            return True

        def on_train_end(self, history):
            self.ended = history.n_epochs

    spy = Spy()
    scheme = ModulationRegistry.create("QPSK")
    demapper = MLPDemapper(scheme, window=0, hidden=(8,))

    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(256, 2)).astype(np.float32)
    targets = rng.integers(0, 2, size=(256, 2)).astype(np.float32)

    from satrecovery.learning.trainer import Trainer
    import torch.nn as nn

    trainer = Trainer(
        model=demapper._net,
        loss_fn=nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(demapper._net.parameters()),
        callbacks=[spy],
    )
    trainer.fit(inputs, targets, epochs=3, batch_size=64)

    assert spy.began == 3
    assert spy.epochs == [1, 2, 3]
    assert spy.ended == 3


def test_early_stopping_halts_a_run_that_stopped_improving():
    stopper = EarlyStopping(monitor="val_loss", patience=2)

    assert stopper.on_epoch_end(1, {"val_loss": 1.0}) is True
    assert stopper.on_epoch_end(2, {"val_loss": 0.5}) is True   # improved
    assert stopper.on_epoch_end(3, {"val_loss": 0.6}) is True   # waiting 1
    assert stopper.on_epoch_end(4, {"val_loss": 0.7}) is False  # waiting 2, stop
    assert stopper.best_epoch == 2
    assert stopper.best_value == 0.5


def test_early_stopping_ignores_a_metric_it_cannot_see():
    stopper = EarlyStopping(monitor="val_loss", patience=1)
    assert stopper.on_epoch_end(1, {"loss": 1.0}) is True


# ==========================================================================
# Datasets
# ==========================================================================
def test_classification_dataset_gives_one_row_per_frame():
    frames = generate_frames(
        SIM, ["QPSK", "16QAM"], [ChannelConditions(snr_db=12.0)], frames_each=3
    )
    builder = ClassificationDataset(matched_filter=SIM.pulse)
    inputs, targets = builder.build(frames)

    assert inputs.shape == (6, builder.extractor.n_features)
    assert set(targets) == {"QPSK", "16QAM"}


def test_demapping_dataset_rows_match_the_window():
    receiver = ClassicalReceiver(pulse=SIM.pulse)
    builder = DemappingDataset(receiver, window=2)
    frames = generate_frames(
        SIM, ["16QAM"], [ChannelConditions(snr_db=14.0)], frames_each=2
    )
    inputs, targets = builder.build(frames)

    assert inputs.shape[1] == builder.n_features == 10
    assert targets.shape[1] == 4
    assert len(inputs) == len(targets)


def test_demapping_dataset_excludes_the_preamble():
    """Training on the preamble would teach the model to recite it."""
    receiver = ClassicalReceiver(pulse=SIM.pulse, pilot_symbols=32)
    frames = generate_frames(
        SIM, ["QPSK"], [ChannelConditions(snr_db=16.0)], frames_each=1
    )

    with_pilot = DemappingDataset(receiver, window=1, skip_pilot=False).build(frames)
    without = DemappingDataset(receiver, window=1, skip_pilot=True).build(frames)

    assert len(without[0]) < len(with_pilot[0])


def test_end_to_end_dataset_keeps_the_iq_layout():
    frames = generate_frames(
        SIM, ["QPSK"], [ChannelConditions(snr_db=10.0)], frames_each=2
    )
    inputs, targets = EndToEndDataset().build(frames)

    assert inputs.shape == (2, 2, SIM.frame_length)
    assert targets.shape == (2, frames[0].n_bits)
    # Normalised per frame, so loudness carries no information.
    assert np.mean(inputs[0] ** 2) == pytest.approx(1.0, abs=0.02)


def test_training_conditions_span_the_range():
    conditions = training_conditions((-6.0, 20.0), steps=10, with_impairments=False)
    assert len(conditions) == 10
    assert conditions[0].snr_db == pytest.approx(-6.0)
    assert conditions[-1].snr_db == pytest.approx(20.0)
    assert all(c.freq_offset_hz == 0.0 for c in conditions)


def test_training_conditions_can_add_impairments():
    conditions = training_conditions((0.0, 10.0), steps=8, with_impairments=True)
    assert any(c.freq_offset_hz != 0.0 for c in conditions)
    assert any(c.has_fading for c in conditions)


def test_split_is_reproducible_and_complete():
    inputs = np.arange(100).reshape(100, 1).astype(float)
    targets = np.arange(100).reshape(100, 1).astype(float)

    a = split_rows(inputs, targets, 0.2, seed=7)
    b = split_rows(inputs, targets, 0.2, seed=7)

    assert np.array_equal(a[0], b[0])
    assert len(a[0]) == 80 and len(a[2]) == 20
    # Nothing lost, nothing duplicated.
    assert sorted(np.concatenate([a[0], a[2]]).ravel()) == list(range(100))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, 1.5])
def test_invalid_split_fraction_is_rejected(fraction):
    with pytest.raises(ValidationError):
        split_rows(np.zeros((10, 1)), np.zeros((10, 1)), fraction)


# ==========================================================================
# The neural demapper
# ==========================================================================
def test_mlp_demapper_satisfies_both_contracts():
    """The multiple inheritance that makes the hybrid receiver free."""
    scheme = ModulationRegistry.create("QPSK")
    demapper = MLPDemapper(scheme, window=1)

    assert isinstance(demapper, Demapper)
    assert isinstance(demapper, TrainableModel)


def test_mlp_demapper_shapes_are_consistent():
    scheme = ModulationRegistry.create("16QAM")
    demapper = MLPDemapper(scheme, window=3)

    assert demapper.width == 7
    assert demapper.n_features == 14
    assert demapper.n_parameters > 0


def test_mlp_demapper_rejects_the_wrong_modulation():
    """One model per scheme, and it says so rather than guessing."""
    qpsk = ModulationRegistry.create("QPSK")
    qam = ModulationRegistry.create("16QAM")
    demapper = MLPDemapper(qpsk, window=0, hidden=(8,))
    demapper._mark_trained()

    with pytest.raises(ValidationError):
        demapper.demap(qam.constellation, qam)


def test_mlp_demapper_rejects_wrongly_shaped_training_data():
    scheme = ModulationRegistry.create("QPSK")
    demapper = MLPDemapper(scheme, window=2, hidden=(8,))

    with pytest.raises(ValidationError):
        demapper.fit(np.zeros((16, 4)), np.zeros((16, 2)), verbose=False)
    with pytest.raises(ValidationError):
        demapper.fit(np.zeros((16, 10)), np.zeros((16, 5)), verbose=False)


@pytest.mark.parametrize("name", ["BPSK", "QPSK"])
def test_mlp_demapper_learns_a_clean_constellation(name):
    """On noiseless symbols it should reach the right answer every time.

    A small, fast check that training works end to end: the data is
    separable, so anything short of near-zero error means something in
    the pipeline is broken rather than merely hard.
    """
    scheme = ModulationRegistry.create(name)
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, size=4000 * scheme.bits_per_symbol, dtype=np.uint8)
    symbols = scheme.map_bits(bits)

    inputs = np.empty((symbols.size, 2), dtype=np.float32)
    inputs[:, 0] = np.real(symbols)
    inputs[:, 1] = np.imag(symbols)
    targets = bits.reshape(-1, scheme.bits_per_symbol).astype(np.float32)

    demapper = MLPDemapper(scheme, window=0, hidden=(32, 32), seed=0)
    demapper.fit(inputs, targets, epochs=60, batch_size=512, verbose=False)

    assert demapper.is_trained
    recovered, confidence, llr = demapper.demap(symbols[:500], scheme)

    errors = np.mean(recovered != bits[: 500 * scheme.bits_per_symbol])
    assert errors < 0.02, f"{name}: {errors:.3f}"
    assert llr is not None
    assert np.all(confidence >= 0.5) and np.all(confidence <= 1.0)


def test_mlp_demapper_round_trips_through_disk(tmp_path):
    scheme = ModulationRegistry.create("QPSK")
    demapper = MLPDemapper(scheme, window=0, hidden=(16,), seed=0)

    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(512, 2)).astype(np.float32)
    targets = rng.integers(0, 2, size=(512, 2)).astype(np.float32)
    demapper.fit(inputs, targets, epochs=3, batch_size=128, verbose=False)

    path = demapper.save(tmp_path / "mlp")
    assert path.exists()
    assert path.with_suffix(".json").exists()

    before = demapper.demap(clean_symbols("QPSK", 200), scheme)[2]

    restored = MLPDemapper(scheme, window=0, hidden=(16,), seed=99)
    restored.load(path)
    after = restored.demap(clean_symbols("QPSK", 200), scheme)[2]

    assert restored.is_trained
    assert np.allclose(before, after)


def test_loading_a_missing_model_fails_clearly(tmp_path):
    scheme = ModulationRegistry.create("QPSK")
    with pytest.raises(ValidationError):
        MLPDemapper(scheme, window=0, hidden=(8,)).load(tmp_path / "nothing")


# ==========================================================================
# The receivers built on learned parts
# ==========================================================================
def test_hybrid_receiver_is_a_classical_receiver():
    """The point of Phase 2 defining Demapper as an interface."""
    scheme = ModulationRegistry.create("QPSK")
    demapper = MLPDemapper(scheme, window=0, hidden=(8,))
    hybrid = HybridReceiver(demapper, pulse=SIM.pulse)

    assert isinstance(hybrid, ClassicalReceiver)
    assert isinstance(hybrid, Receiver)
    assert hybrid.demapper is demapper


def test_hybrid_receiver_demands_a_demapper():
    with pytest.raises(ValidationError):
        HybridReceiver("not a demapper")  # type: ignore[arg-type]


def test_hybrid_receiver_runs_the_full_chain():
    """A learned demapper must slot in with nothing else changing."""
    scheme = ModulationRegistry.create("QPSK")
    rng = np.random.default_rng(5)
    bits = rng.integers(0, 2, size=8000, dtype=np.uint8)
    symbols = scheme.map_bits(bits)

    inputs = np.empty((symbols.size, 2), dtype=np.float32)
    inputs[:, 0] = np.real(symbols)
    inputs[:, 1] = np.imag(symbols)
    targets = bits.reshape(-1, 2).astype(np.float32)

    demapper = MLPDemapper(scheme, window=0, hidden=(32, 32), seed=0)
    demapper.fit(inputs, targets, epochs=50, batch_size=512, verbose=False)

    frame = SIM.generate_frame("QPSK", ChannelConditions(snr_db=20.0))
    result = HybridReceiver(demapper, pulse=SIM.pulse).recover(frame)

    assert result.n_bits == frame.n_bits
    assert result.receiver == "Hybrid"
    assert result.ber(frame.bits) < 0.05


def test_neural_receiver_is_a_receiver():
    scheme = ModulationRegistry.create("QPSK")
    model = CNNDemodulator(scheme, samples_per_symbol=8)
    receiver = NeuralReceiver(model)

    assert isinstance(receiver, Receiver)
    assert receiver.pilot_symbols == 0


def test_neural_receiver_rejects_something_that_is_not_a_model():
    with pytest.raises(ValidationError):
        NeuralReceiver(object())


def test_every_receiver_shares_one_interface():
    """What makes the Phase 6 comparison fair.

    The evaluator holds a list of receivers and cannot tell them apart.
    There is no branch where one method is treated more kindly.
    """
    scheme = ModulationRegistry.create("QPSK")
    demapper = MLPDemapper(scheme, window=0, hidden=(8,))
    demapper._mark_trained()

    receivers = [
        ClassicalReceiver(pulse=SIM.pulse),
        HybridReceiver(demapper, pulse=SIM.pulse),
        NeuralReceiver(CNNDemodulator(scheme)),
    ]
    for receiver in receivers:
        assert isinstance(receiver, Receiver)
        assert hasattr(receiver, "recover")
        assert isinstance(receiver.pilot_symbols, int)


# ==========================================================================
# The forest
# ==========================================================================
def test_forest_learns_to_tell_schemes_apart():
    frames = generate_frames(
        SIM,
        ["BPSK", "QPSK", "16QAM"],
        [ChannelConditions(snr_db=s) for s in (6.0, 12.0, 18.0)],
        frames_each=6,
    )
    inputs, targets = ClassificationDataset(matched_filter=SIM.pulse).build(frames)

    forest = ForestModulationClassifier(n_trees=60, seed=0)
    history = forest.fit(inputs, targets)

    assert forest.is_trained
    assert history.last()["train_accuracy"] > 0.9
    assert set(forest.classes) == {"BPSK", "QPSK", "16QAM"}

    frame = SIM.generate_frame("16QAM", ChannelConditions(snr_db=15.0))
    name, confidence = forest.classify_frame(frame, matched_filter=SIM.pulse)
    assert name in forest.classes
    assert 0.0 <= confidence <= 1.0


def test_forest_reports_what_it_relied_on():
    """The reason for keeping an explainable baseline.

    It is how we diagnosed the classifier leaning on phase statistics --
    a feature a Doppler shift corrupts -- rather than on the
    rotation-invariant cumulants.
    """
    frames = generate_frames(
        SIM, ["QPSK", "16QAM"], [ChannelConditions(snr_db=12.0)], frames_each=8
    )
    inputs, targets = ClassificationDataset(matched_filter=SIM.pulse).build(frames)

    forest = ForestModulationClassifier(n_trees=40, seed=0)
    forest.fit(inputs, targets)

    importances = forest.feature_importances()
    assert len(importances) == forest.extractor.n_features
    assert abs(sum(importances.values()) - 1.0) < 1e-6
    # Ordered most important first.
    values = list(importances.values())
    assert values == sorted(values, reverse=True)


def test_forest_rejects_features_of_the_wrong_width():
    forest = ForestModulationClassifier()
    with pytest.raises(ValidationError):
        forest.fit(np.zeros((10, 3)), np.array(["QPSK"] * 10))


def test_forest_round_trips_through_disk(tmp_path):
    frames = generate_frames(
        SIM, ["QPSK", "16QAM"], [ChannelConditions(snr_db=14.0)], frames_each=5
    )
    inputs, targets = ClassificationDataset(matched_filter=SIM.pulse).build(frames)

    forest = ForestModulationClassifier(n_trees=30, seed=0)
    forest.fit(inputs, targets)
    before = forest.predict(inputs)

    path = forest.save(tmp_path / "forest")
    restored = ForestModulationClassifier(n_trees=30, seed=0).load(path)

    assert restored.is_trained
    assert np.array_equal(restored.predict(inputs), before)


# ==========================================================================
# The end-to-end network
# ==========================================================================
def test_cnn_produces_the_right_number_of_bits():
    scheme = ModulationRegistry.create("QPSK")
    small = SignalSimulator(frame_length=512, samples_per_symbol=8, seed=2)

    frames = generate_frames(
        small, ["QPSK"], [ChannelConditions(snr_db=15.0)], frames_each=8
    )
    inputs, targets = EndToEndDataset().build(frames)

    model = CNNDemodulator(scheme, samples_per_symbol=8, seed=0)
    model.fit(inputs, targets, epochs=2, batch_size=4, verbose=False)

    bits, confidence, llr = model.predict_bits(inputs[0])
    assert bits.size == targets.shape[1]
    assert confidence.size == bits.size
    assert llr.size == bits.size


def test_cnn_round_trips_through_disk(tmp_path):
    scheme = ModulationRegistry.create("QPSK")
    small = SignalSimulator(frame_length=512, samples_per_symbol=8, seed=2)
    frames = generate_frames(
        small, ["QPSK"], [ChannelConditions(snr_db=15.0)], frames_each=4
    )
    inputs, targets = EndToEndDataset().build(frames)

    model = CNNDemodulator(scheme, samples_per_symbol=8, seed=0)
    model.fit(inputs, targets, epochs=2, batch_size=4, verbose=False)
    before = model.predict_bits(inputs[0])[2]

    path = model.save(tmp_path / "cnn")
    restored = CNNDemodulator(scheme, samples_per_symbol=8, seed=7).load(path)

    assert np.allclose(restored.predict_bits(inputs[0])[2], before)
