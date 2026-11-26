"""Tests for test files as two CSV files: export, import, and hand-made pairs."""

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from satrecovery.analysis import Analyser, RecordedInput  # noqa: E402
from satrecovery.analysis.classifier import ModulationClassifier  # noqa: E402
from satrecovery.analysis.csvpair import CsvPairInput, CsvTestFile  # noqa: E402
from satrecovery.analysis.upload import input_from_file  # noqa: E402
from satrecovery.core.exceptions import ValidationError  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_analysis import Oracle, archive_file, catalog  # noqa: E402,F401

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SIM = EXAMPLES / "test_QPSK_+8dB_250kbaud_doppler600.npz"
REAL = EXAMPLES / "test_RML24_184700_8PSK_+18dB_250kbaud.npz"


@pytest.fixture(autouse=True)
def private_preferences(tmp_path, monkeypatch):
    monkeypatch.setenv("SATREC_PREFERENCES", str(tmp_path / "prefs.json"))


def _grade(signal, answer, catalog):
    return Analyser(Oracle(answer), catalog).analyse(signal)


class TestExportAndImport:
    @pytest.mark.parametrize("path, answer", [(SIM, "QPSK"), (REAL, "8PSK")])
    def test_a_pair_grades_exactly_like_its_test_file(self, path, answer, catalog, tmp_path):
        signal_path, bits_path = CsvTestFile.write_pair(tmp_path, path.name, path.read_bytes())
        assert signal_path.name.endswith("_signal.csv") and bits_path.name.endswith("_bits.csv")
        assert CsvTestFile.partner(signal_path) == bits_path
        original = _grade(input_from_file(path.name, path.read_bytes(), None), answer, catalog)
        again = _grade(CsvTestFile.input_from(signal_path.name, signal_path.read_bytes(),
                                              bits_path.name, bits_path.read_bytes()), answer, catalog)
        for key in ("bit_errors", "n_bits"):
            assert again["recovery"]["check"][key] == original["recovery"]["check"][key]
        assert again["classification"]["correct"]

    def test_the_bits_file_reads_well_in_excel(self, tmp_path):
        _, bits_path = CsvTestFile.write_pair(tmp_path, "t", SIM.read_bytes())
        lines = bits_path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "type,QPSK" and lines[1] == "speed_baud,250000"
        start = lines.index("bit")
        assert set(lines[start + 1:]) <= {"0", "1"} and len(lines) - start - 1 == 1024

    def test_a_real_recording_reloads_as_the_recording(self, tmp_path):
        signal_path, bits_path = CsvTestFile.write_pair(tmp_path, REAL.name, REAL.read_bytes())
        loaded = CsvTestFile.input_from(signal_path.name, signal_path.read_bytes(), bits_path.name, bits_path.read_bytes())
        assert isinstance(loaded, RecordedInput) and loaded.describe()["row"] == 184700

    def test_the_example_pair_opens(self):
        signal = EXAMPLES / "test_QPSK_+8dB_250kbaud_doppler600_signal.csv"
        bits = CsvTestFile.partner(signal)
        assert bits is not None
        assert isinstance(CsvTestFile.input_from(signal.name, signal.read_bytes(), bits.name, bits.read_bytes()),
                          CsvPairInput)


class TestHandMadePairs:
    def _signal(self, n=2048):
        rng = np.random.default_rng(1)
        z = np.repeat(np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, n // 4))), 4)
        return ("I,Q\n" + "".join(f"{v.real},{v.imag}\n" for v in z)).encode()

    def test_a_hand_made_pair_is_accepted(self):
        bits = "type,qpsk\nspeed_baud,250000\nbit\n" + "0\n1\n" * 512
        signal = CsvTestFile.input_from("mine.csv", self._signal(), "mine_bits.csv", bits.encode())
        assert signal.true_modulation == "QPSK" and signal.describe()["answer_key"] == "by hand"

    def test_the_wrong_number_of_bits_says_what_was_expected(self):
        bits = "type,QPSK\nspeed_baud,250000\nbit\n" + "1\n" * 1000
        with pytest.raises(ValidationError, match="has 1,000 bits.*carries 1,024"):
            CsvTestFile.input_from("mine.csv", self._signal(), "mine_bits.csv", bits.encode())

    @pytest.mark.parametrize("text, problem", [
        ("speed_baud,250000\nbit\n0\n", "type"),
        ("type,QPSK\nbit\n0\n", "speed_baud"),
        ("type,QPSK\nspeed_baud,250000\n0\n1\n", "label and a value"),
        ("type,QPSK\nspeed_baud,250000\n", "no line 'bit'"),
        ("type,QPSK\nspeed_baud,250000\nbit\n0\n2\n", "expected 0 or 1"),
        ("type,QPSK\nspeed_baud,300000\nbit\n0\n", "speed_baud"),
    ])
    def test_a_damaged_bits_file_is_explained(self, text, problem):
        with pytest.raises(ValidationError, match=problem):
            CsvTestFile.read_bits(text)

    def test_excel_style_files_are_read(self):
        info = CsvTestFile.read_bits("﻿type;QPSK\r\nspeed_baud;250000\r\nbit;\r\n0;\r\n1;\r\n".encode("utf-8"))
        assert info["type"] == "QPSK" and info["bits"].tolist() == [0, 1]


class TestApi:
    def test_the_file_endpoint_takes_a_bits_file(self, tmp_path, archive_file):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from satrecovery.analysis.api import create_analysis_app
        from satrecovery.backend import Settings

        path = tmp_path / "models" / "classifier.pt"
        model = ModulationClassifier(channels=(8,), device="cpu")
        model._mark_trained()
        model.save(path)
        signal_path, bits_path = CsvTestFile.write_pair(tmp_path, "t", SIM.read_bytes())
        body = {"name": signal_path.name, "content_base64": base64.b64encode(signal_path.read_bytes()).decode(),
                "bits_name": bits_path.name, "bits_base64": base64.b64encode(bits_path.read_bytes()).decode()}
        with TestClient(create_analysis_app(Settings(data_dir=tmp_path / "data", rml24_path=archive_file), path)) as c:
            result = c.post("/api/analysis/file", json=body)
            assert result.status_code == 200 and result.json()["classification"]["true_label"] == "QPSK"
            assert result.json()["input"]["test_file"] == "csv"
            body["bits_base64"] = base64.b64encode(b"type,QPSK\nspeed_baud,250000\nbit\n1\n").decode()
            refused = c.post("/api/analysis/file", json=body)
            assert refused.status_code == 422 and "carries 1,024" in refused.json()["detail"]


class TestScreens:
    @pytest.fixture(scope="class")
    @classmethod
    def app(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        from satrecovery.gui import theme

        application = QApplication.instance() or QApplication([])
        theme.load_fonts()
        theme.configure_plots()
        return application

    def test_choosing_the_signal_finds_its_bits(self, app):
        from satrecovery.analysis.gui.pages import FilePage

        page = FilePage(None)
        page.set_file(EXAMPLES / "test_QPSK_+8dB_250kbaud_doppler600_signal.csv")
        assert page._bits_path is not None and page._bits_path.name.endswith("_bits.csv")
        assert "Answer key found" in page.test_info.text() and not page.rate.isEnabled()
        assert page.choose_bits.isVisibleTo(page) and not page.export_csv.isVisibleTo(page)

    def test_a_plain_csv_has_no_key_until_one_is_chosen(self, app, tmp_path):
        from satrecovery.analysis.gui.pages import FilePage

        page = FilePage(None)
        page.set_file(EXAMPLES / "qpsk_250kbaud.csv")
        assert page._bits_path is None and not page.test_info.isVisibleTo(page) and page.rate.isEnabled()
        bad = tmp_path / "bad_bits.csv"
        bad.write_text("bit\n0\n", encoding="utf-8")
        page.set_bits_file(bad)
        assert page._bits_path is None and page.test_info.text().startswith("Could not use")

    def test_a_test_file_can_be_exported(self, app, tmp_path):
        from satrecovery.analysis.gui.pages import FilePage

        page = FilePage(None)
        page.set_file(SIM)
        assert page.export_csv.isVisibleTo(page)
        written = page.export_to(tmp_path)
        assert written and all(Path(p).exists() for p in written)

    def test_saving_as_csv_from_the_simulated_tab(self, app, tmp_path):
        from satrecovery.analysis.gui.pages import SimulatedSignalPage

        data = SIM.read_bytes()
        page = SimulatedSignalPage(None)
        written = page.write_test_csv(str(tmp_path / "mine.csv"),
                                      {"name": SIM.name, "content_base64": base64.b64encode(data).decode()})
        assert [Path(p).name for p in written] == ["mine_signal.csv", "mine_bits.csv"]
