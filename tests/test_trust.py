"""Tests for the warning: when should the AI's answer not be trusted?"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from satrecovery.analysis import Analyser, SimulatedInput  # noqa: E402
from satrecovery.analysis.classifier import ModulationClassifier  # noqa: E402
from satrecovery.analysis.history import AnalysisHistory  # noqa: E402
from satrecovery.analysis.trust import TrustCheck, TrustVerdict  # noqa: E402
from satrecovery.core.conditions import ChannelConditions  # noqa: E402
from satrecovery.data import SampleArchive  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_analysis import Oracle, archive_file, catalog  # noqa: E402,F401

SURE = [{"name": "QPSK", "p": 0.95}, {"name": "8PSK", "p": 0.03}]


def _signal(snr, rate=250_000, seed=1, doppler=500.0):
    return SimulatedInput("QPSK", ChannelConditions(snr_db=snr, freq_offset_hz=doppler), rate, seed=seed)


class TestMeasurements:
    @pytest.mark.parametrize("rate", [100_000, 250_000, 500_000])
    @pytest.mark.parametrize("snr", [-5.0, 5.0, 15.0])
    def test_the_noise_level_is_read_close_to_the_truth(self, rate, snr):
        estimate = TrustCheck.estimate_snr_db(_signal(snr, rate).samples, rate)
        assert estimate == pytest.approx(snr, abs=1.5)

    def test_a_real_recording_reads_close_to_its_label(self):
        recording = SampleArchive(Path(__file__).parent / "data" / "rml24_row_184700.npz").recording(184700)
        assert 14.0 <= TrustCheck.estimate_snr_db(recording.samples, recording.symbol_rate) <= 20.0

    @pytest.mark.parametrize("rate", [100_000, 250_000, 500_000])
    def test_the_width_fits_the_right_speed(self, rate):
        assert TrustCheck.width_ratio(_signal(15.0, rate).samples, rate) == pytest.approx(1.0, abs=0.25)

    def test_a_weak_signal_is_too_noisy_to_measure_its_width(self):
        assert TrustCheck.width_ratio(_signal(-2.0).samples, 250_000) is None


class TestVerdicts:
    def test_a_clean_clear_answer_is_confident(self):
        verdict = TrustCheck.assess(SURE, _signal(15.0).samples, 250_000)
        assert verdict.level == "confident" and not verdict.warning and "+15 dB" in verdict.as_dict()["title"]

    def test_mostly_noise_is_unfamiliar(self):
        verdict = TrustCheck.assess(SURE, _signal(-10.0).samples, 250_000)
        assert verdict.level == "unfamiliar" and "mostly noise" in verdict.reasons[0]

    def test_very_noisy_is_unsure(self):
        verdict = TrustCheck.assess(SURE, _signal(-3.0).samples, 250_000)
        assert verdict.level == "unsure" and "very noisy" in verdict.reasons[0]

    @pytest.mark.parametrize("told, word", [(100_000, "wider"), (500_000, "narrower")])
    def test_the_wrong_speed_is_spotted(self, told, word):
        verdict = TrustCheck.assess(SURE, _signal(15.0).samples, told)
        assert verdict.level == "unfamiliar" and word in verdict.reasons[0]

    def test_a_weak_or_close_answer_is_unsure(self):
        samples = _signal(15.0).samples
        weak = TrustCheck.assess([{"name": "QPSK", "p": 0.48}, {"name": "8PSK", "p": 0.40}], samples, 250_000)
        close = TrustCheck.assess({"QPSK": 0.62, "8PSK": 0.38}, samples, 250_000)
        assert weak.level == close.level == "unsure"
        assert "only 48% sure" in weak.reasons[0] and "close" in close.reasons[0]

    def test_an_unsure_receiver_is_reported(self):
        verdict = TrustCheck.assess(SURE, _signal(15.0).samples, 250_000, trusted_fraction=0.3)
        assert verdict.level == "unsure" and "30% of the bits" in verdict.reasons[0]

    def test_an_unknown_level_is_refused(self):
        with pytest.raises(ValueError):
            TrustVerdict("maybe", "", [], None)


class TestWhereItShows:
    def test_every_analysis_carries_a_verdict_and_history_marks_warnings(self, catalog, tmp_path):
        clean = Analyser(Oracle("QPSK"), catalog).analyse(_signal(15.0))
        noisy = Analyser(Oracle("QPSK"), catalog).analyse(_signal(-10.0))
        assert clean["trust"]["level"] == "confident" and noisy["trust"]["level"] == "unfamiliar"
        assert AnalysisHistory.summarise(clean)["ai_label"] == "QPSK"
        assert AnalysisHistory.summarise(noisy)["ai_label"] == "QPSK (unfamiliar)"

    def test_the_report_card_measures_the_warning(self, tmp_path, archive_file):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from satrecovery.analysis.api import create_analysis_app
        from satrecovery.backend import Settings

        path = tmp_path / "models" / "classifier.pt"
        model = ModulationClassifier(channels=(8,), device="cpu")
        model._mark_trained()
        model.save(path)
        with TestClient(create_analysis_app(Settings(data_dir=tmp_path / "d", rml24_path=archive_file), path)) as c:
            report = c.post("/api/analysis/evaluate", json={"per_group": 1}).json()
            warnings = report["warnings"]
            assert set(warnings) >= {"flagged", "mistakes_flagged", "correct_flagged", "accuracy_when_confident"}
            assert all(m["warning"] in ("confident", "unsure", "unfamiliar") for m in report["mistakes"])
            row = c.get("/api/analysis/batch-rows", params={"modulation": "QPSK", "count": 1}).json()[0]
            assert c.post("/api/analysis/quick", json={"row": row}).json()["trust"] in TrustVerdict.LEVELS

    def test_many_at_once_counts_the_warnings(self):
        from satrecovery.analysis.gui.batch import BatchSummary

        summary = BatchSummary()
        for level in ("confident", "unsure", "unfamiliar", "confident"):
            summary.add({"trust": level})
        assert summary.warned == 2

    def test_the_badge_shows_and_hides(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        from satrecovery.analysis.gui.widgets import ClassifierCard

        QApplication.instance() or QApplication([])
        card = ClassifierCard()
        card.show_trust({"level": "unsure", "title": "Unsure: treat this answer with caution.",
                         "reasons": ["The AI is only 48% sure."], "snr_db": 3.0})
        assert card.trust.isVisibleTo(card) and "48% sure" in card.trust.text()
        card.show_trust(None)                    # an older result, saved before the warning existed
        assert not card.trust.isVisibleTo(card)
