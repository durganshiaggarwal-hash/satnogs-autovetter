"""Tests for comparing models: the new networks, the forest, the script, the API and the tab."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from satrecovery.analysis import Analyser, ModulationClassifier  # noqa: E402
from satrecovery.analysis.compare import (  # noqa: E402
    COMPARISON_MODELS,
    ForestClassifier,
    ModelComparison,
    build_model,
    expert_features,
    load_classifier,
)
from satrecovery.backend.catalog import ReceiverCatalog  # noqa: E402
from satrecovery.core.exceptions import ModelNotTrainedError, ValidationError  # noqa: E402
from satrecovery.data import SampleArchive  # noqa: E402
from satrecovery.simulation.pulse import RootRaisedCosine  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_data import synthetic  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ModulationClassifier.DEFAULT_CLASSES


@pytest.fixture(scope="module")
def six_archive(tmp_path_factory):
    """Every class, two SNR labels, enough recordings for a split."""
    recordings, row = [], 0
    for modulation in CLASSES:
        for snr in (0.0, 20.0):
            for _ in range(8):
                recordings.append(synthetic(modulation, 4, seed=500 + row, row=row, snr_db=snr + 2,
                                            snr_label=snr)[0])
                row += 1
    return SampleArchive.write(tmp_path_factory.mktemp("cmp") / "six.npz", recordings)


@pytest.fixture(scope="module")
def prepared(six_archive):
    archive = SampleArchive(six_archive)
    x = np.stack([ModulationClassifier.prepare(archive.recording(r).samples)[0] for r in range(len(archive))])
    y = np.array([CLASSES.index(archive.recording(r).modulation) for r in range(len(archive))])
    return x, y


class TestNewNetworks:
    @pytest.mark.parametrize("architecture", ["resnet_raw", "lstm", "transformer"])
    def test_builds_and_answers_with_probabilities(self, architecture):
        model = ModulationClassifier(device="cpu", architecture=architecture)
        model._mark_trained()
        p = model.predict_proba(np.random.default_rng(0).standard_normal((3, 2, 2048)).astype(np.float32))
        assert p.shape == (3, 6)
        assert np.allclose(p.sum(axis=1), 1, atol=1e-5)

    def test_resnet_without_expert_features_is_the_resnet_minus_them(self):
        full = ModulationClassifier(device="cpu", architecture="resnet")
        raw = ModulationClassifier(device="cpu", architecture="resnet_raw")
        assert raw.n_parameters < full.n_parameters
        assert not hasattr(raw._net, "expert")

    def test_the_apps_resnet_is_unchanged(self):
        # The saved classifier.pt must still load: same layers, same names.
        assert ModulationClassifier(device="cpu").n_parameters == 635_166

    @pytest.mark.parametrize("architecture", ["lstm", "transformer"])
    def test_saves_and_loads_with_its_architecture(self, architecture, tmp_path, prepared):
        x, y = prepared
        model = ModulationClassifier(device="cpu", architecture=architecture)
        model.fit(x, y, epochs=1, batch_size=32, verbose=False)
        path = model.save(tmp_path / "m.pt")
        again = load_classifier(path)
        assert again.architecture == architecture
        assert np.allclose(again.predict_proba(x[:4]), model.predict_proba(x[:4]), atol=1e-5)

    def test_shorter_recordings_still_work(self):
        model = ModulationClassifier(device="cpu", architecture="transformer")
        model._mark_trained()
        assert model.predict_proba(np.ones((1, 2, 512), dtype=np.float32)).shape == (1, 6)

    def test_unknown_architecture_is_refused(self):
        with pytest.raises(ValidationError):
            ModulationClassifier(device="cpu", architecture="gpt")


class TestForest:
    def test_learns_from_the_expert_features(self, prepared):
        x, y = prepared
        forest = ForestClassifier(n_trees=60)
        forest.fit(x[::2], y[::2], verbose=False)
        # Clean simulated signals: the expert measurements alone separate them well.
        assert forest.evaluate(x[1::2], y[1::2]) > 0.6

    def test_is_a_modulation_classifier_so_the_app_can_use_it(self, prepared):
        x, y = prepared
        forest = ForestClassifier(n_trees=20)
        forest.fit(x, y, verbose=False)
        assert isinstance(forest, ModulationClassifier)
        analyser = Analyser(forest, ReceiverCatalog.standard(RootRaisedCosine(8)))
        answer = forest.classify(x[0, 0] + 1j * x[0, 1])
        assert answer.label in CLASSES and 0 <= answer.confidence <= 1
        assert analyser.classifier is forest

    def test_untrained_forest_refuses(self, prepared):
        with pytest.raises(ModelNotTrainedError):
            ForestClassifier().predict_proba(prepared[0][:2])

    def test_save_load_round_trip(self, prepared, tmp_path):
        x, y = prepared
        forest = ForestClassifier(n_trees=10)
        forest.fit(x, y, verbose=False)
        forest.record_test_accuracy(0.5)
        again = load_classifier(forest.save(tmp_path / "forest.pt"))
        assert isinstance(again, ForestClassifier)
        assert again.test_accuracy == 0.5
        assert np.allclose(again.predict_proba(x[:5]), forest.predict_proba(x[:5]))
        assert "random forest" in again.describe()

    def test_expert_features_are_finite_and_twelve(self, prepared):
        f = expert_features(prepared[0][:10])
        assert f.shape == (10, 12) and np.isfinite(f).all()


class TestComparisonStore:
    def test_every_compared_model_can_be_built(self):
        for name in COMPARISON_MODELS:
            assert isinstance(build_model(name, device="cpu"), ModulationClassifier)
        with pytest.raises(ValidationError):
            build_model("nope")

    def test_main_is_used_until_another_is_chosen(self, tmp_path):
        store = ModelComparison(tmp_path / "compare", tmp_path / "classifier.pt")
        assert store.active() == "main"
        with pytest.raises(ValidationError):
            store.set_active("lstm")                    # not trained yet
        (tmp_path / "compare").mkdir()
        (tmp_path / "compare" / "lstm.pt").write_bytes(b"x")
        store.set_active("lstm")
        assert store.active() == "lstm"
        (tmp_path / "compare" / "lstm.pt").unlink()      # model deleted: fall back
        assert store.active() == "main"

    def test_broken_files_do_not_break_it(self, tmp_path):
        folder = tmp_path / "compare"
        folder.mkdir()
        (folder / "results.json").write_text("{not json")
        (folder / "active.json").write_text("[]")
        store = ModelComparison(folder, tmp_path / "classifier.pt")
        assert store.results() == {} and store.active() == "main"
        listing = store.listing()
        assert listing["has_results"] is False
        assert [m["name"] for m in listing["models"]][0] == "main"
        assert len(listing["models"]) == 1 + len(COMPARISON_MODELS)


@pytest.fixture(scope="module")
def compared(six_archive, tmp_path_factory):
    """The comparison script, run for real on the small archive."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import compare_models

    out = tmp_path_factory.mktemp("models") / "compare"
    found = compare_models.main([str(six_archive), "--models", "cnn", "lstm", "forest", "--per-cell", "5",
                                 "--test-per-cell", "3", "--epochs", "1", "--batch-size", "16", "--cpu",
                                 "--quiet", "--out", str(out), "--figures", str(out.parent / "fig")])
    return out, found


class TestScript:
    def test_trains_saves_and_writes_results(self, compared):
        out, found = compared
        assert set(found) == {"cnn", "lstm", "forest"}
        for name in found:
            assert (out / f"{name}.pt").exists()
        data = json.loads((out / "results.json").read_text())
        assert data["setup"]["test"] == 6 * 2 * 3 and data["setup"]["epochs"] == 1
        for info in data["models"].values():
            assert 0 <= info["accuracy"] <= 1
            assert set(info["accuracy_by_snr"]) == {"0", "20"}
            assert np.asarray(info["confusion_0db_plus"]).shape == (6, 6)
            assert info["n_parameters"] > 0 and info["ms_per_recording"] > 0
        assert (out.parent / "fig" / "accuracy_vs_snr.png").exists()

    def test_same_test_recordings_for_every_model(self, compared):
        out, found = compared
        totals = {name: int(np.sum(info["confusion_0db_plus"])) for name, info in found.items()}
        assert len(set(totals.values())) == 1

    def test_a_later_run_adds_to_the_results(self, compared, six_archive):
        import compare_models

        out, _ = compared
        compare_models.main([str(six_archive), "--models", "transformer", "--per-cell", "5", "--test-per-cell", "3",
                             "--epochs", "1", "--batch-size", "16", "--cpu", "--quiet", "--out", str(out),
                             "--figures", str(out.parent / "fig")])
        data = json.loads((out / "results.json").read_text())
        assert set(data["models"]) == {"cnn", "lstm", "forest", "transformer"}


class TestApi:
    @pytest.fixture
    def client(self, compared, tmp_path):
        from fastapi.testclient import TestClient

        from satrecovery.analysis.api import create_analysis_app
        from satrecovery.backend.settings import Settings

        out, _ = compared
        settings = Settings(data_dir=tmp_path)
        models = Path(settings.models_dir)
        models.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copytree(out, models / "compare")
        (models / "compare" / "active.json").unlink(missing_ok=True)
        return TestClient(create_analysis_app(settings))

    def test_lists_models_with_results(self, client):
        listing = client.get("/api/analysis/models").json()
        assert listing["has_results"] and listing["active"] == "main"
        by_name = {m["name"]: m for m in listing["models"]}
        assert by_name["forest"]["available"] and by_name["forest"]["results"]["accuracy"] >= 0
        assert by_name["resnet"]["results"] is None and not by_name["resnet"]["available"]

    def test_switching_model_changes_what_the_app_uses(self, client):
        assert client.post("/api/analysis/models/use", json={"model": "forest"}).json()["active"] == "forest"
        status = client.get("/api/analysis/status").json()
        assert status["classifier"]["available"] and "random forest" in status["classifier"]["description"]
        assert status["classifier"]["path"].endswith("forest.pt")
        client.post("/api/analysis/models/use", json={"model": "lstm"})
        assert "lstm" in client.get("/api/analysis/status").json()["classifier"]["description"]

    def test_choice_is_remembered_by_a_new_app(self, client, tmp_path):
        from fastapi.testclient import TestClient

        from satrecovery.analysis.api import create_analysis_app
        from satrecovery.backend.settings import Settings

        client.post("/api/analysis/models/use", json={"model": "cnn"})
        again = TestClient(create_analysis_app(Settings(data_dir=tmp_path)))
        assert again.get("/api/analysis/models").json()["active"] == "cnn"

    def test_untrained_or_unknown_models_are_refused(self, client):
        assert client.post("/api/analysis/models/use", json={"model": "resnet"}).status_code == 422
        assert client.post("/api/analysis/models/use", json={"model": "bogus"}).status_code == 422


class TestTab:
    @pytest.fixture(scope="class")
    def app(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        from satrecovery.gui import theme

        application = QApplication.instance() or QApplication([])
        theme.load_fonts()
        theme.configure_plots()
        return application

    def _page(self):
        from satrecovery.analysis.gui.client import AnalysisClient
        from satrecovery.analysis.gui.compare import ComparePage

        return ComparePage(AnalysisClient("http://127.0.0.1:9"))

    def test_without_results_it_explains_how_to_make_them(self, app, tmp_path):
        page = self._page()
        page.show_listing(ModelComparison(tmp_path / "c", tmp_path / "classifier.pt").listing())
        assert not page.empty_card.isHidden() and page.results_box.isHidden()
        assert "compare_models.py" in page.empty_text.text()

    def test_with_results_it_shows_lines_table_and_selection(self, app, compared, tmp_path):
        out, found = compared
        page = self._page()
        page.show_listing(ModelComparison(out, tmp_path / "classifier.pt").listing())
        assert page.empty_card.isHidden() and not page.results_box.isHidden()
        assert page.table.rowCount() == len(json.loads((out / "results.json").read_text())["models"])
        names = [page.table.item(r, 0).data(0x0100) for r in range(page.table.rowCount())]
        page.table.selectRow(names.index("forest"))
        assert page.chosen_name.text() == "Random Forest"
        assert page.use_button.isEnabled()
        assert "Best on clean signals" in page.winner.text()
