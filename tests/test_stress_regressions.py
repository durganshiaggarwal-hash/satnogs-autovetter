"""Bugs found by the stress test, each kept fixed by a test.

Every one of these used to crash the server (500) or give a wrong or
garbled message.  They are bad inputs a user can really send.
"""

from __future__ import annotations

import base64
import io
import warnings

import numpy as np
import pytest

from satrecovery.analysis.classifier import ModulationClassifier
from satrecovery.analysis.trust import TrustCheck
from satrecovery.analysis.upload import SignalFile, input_from_file
from satrecovery.core.exceptions import ValidationError
from satrecovery.data import layout


def _npy(array) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


class TestSymbolRate:
    @pytest.mark.parametrize("rate", [0, -250_000, float("nan"), float("inf")])
    def test_zero_negative_or_not_a_number_is_refused_clearly(self, rate):
        with pytest.raises(ValidationError, match="symbol_rate"):
            layout.samples_per_symbol(rate)

    def test_the_rml24_speeds_still_work(self):
        assert [layout.samples_per_symbol(r) for r in (100_000, 250_000, 500_000)] == [10, 4, 2]


class TestDatasetPath:
    def test_a_file_that_is_not_hdf5_is_refused_not_a_crash(self, tmp_path):
        from satrecovery.data import open_archive

        fake = tmp_path / "RML24_IQdata.h5"
        fake.write_text("not hdf5")
        with pytest.raises(ValidationError, match="not an HDF5 file"):
            open_archive(fake)

    def test_a_damaged_sample_archive_is_refused(self, tmp_path):
        from satrecovery.data import open_archive

        fake = tmp_path / "sample.npz"
        fake.write_bytes(b"junk")
        with pytest.raises(ValidationError, match="cannot be read as .npz"):
            open_archive(fake)


class TestUploadedFiles:
    def test_a_silent_signal_is_refused(self):
        with pytest.raises(ValidationError, match="silent"):
            SignalFile.read("quiet.npy", _npy(np.zeros(2048, complex)))

    def test_a_broken_file_says_so_before_asking_for_the_speed(self):
        with pytest.raises(ValidationError, match="not a valid .npz file"):
            input_from_file("x.npz", b"not a zip", symbol_rate=None)
        with pytest.raises(ValidationError, match="use one of"):
            input_from_file("x.exe", b"MZ", symbol_rate=None)

    def test_an_empty_npz_is_refused_plainly(self):
        buffer = io.BytesIO()
        np.savez(buffer)
        with pytest.raises(ValidationError, match="empty"):
            SignalFile.read("x.npz", buffer.getvalue())

    def test_csv_errors_name_the_file(self):
        with pytest.raises(ValidationError, match="mine.csv"):
            SignalFile.read("mine.csv", b"hello\n")


class TestNumbers:
    def test_width_check_on_silence_gives_no_ratio_and_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert TrustCheck.width_ratio(np.zeros(2048, complex), 250_000.0) is None

    def test_huge_amplitudes_do_not_overflow(self):
        rng = np.random.default_rng(0)
        signal = ((rng.normal(size=2048) + 1j * rng.normal(size=2048)) * 1e30).astype(np.complex64)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            prepared = ModulationClassifier.prepare(signal)
        assert np.all(np.isfinite(prepared)) and abs(np.mean(prepared[0, 0] ** 2 + prepared[0, 1] ** 2) - 1) < 1e-3


class TestNetwork:
    def test_offline_is_detected_quickly_and_said_plainly(self, monkeypatch):
        import socket

        from satrecovery.vetting.gui import live

        for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.delenv(name, raising=False)

        def refuse(*_args, **_kwargs):
            raise OSError("no route")

        monkeypatch.setattr(socket, "create_connection", refuse)
        assert live.online() is False
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")       # behind a proxy: let the requests decide
        assert live.online() is True
        assert "saved on this computer" in live.plain(
            "SatNOGS problem while listing: could not reach network.satnogs.org: <urlopen error [Errno 61]>")
        assert live.plain("listing: 25 looked at") == "listing: 25 looked at"


class TestGuiLimits:
    @pytest.fixture(scope="class")
    def qapp(self):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        return QApplication.instance() or QApplication([])

    def test_screens_scroll_so_the_window_fits_a_small_laptop(self, qapp):
        from PySide6.QtWidgets import QLabel, QWidget

        from satrecovery.vetting.gui.mission import ScrollingStack

        stack = ScrollingStack()
        big, small = QWidget(), QLabel("x")
        big.setMinimumSize(1400, 900)
        stack.addWidget(big)
        stack.addWidget(small)
        stack.setCurrentWidget(small)
        assert stack.currentWidget() is small and stack.widget(0) is big
        assert stack.minimumSizeHint().width() < 1366 and stack.minimumSizeHint().height() < 768

    def test_finished_orbit_jobs_are_let_go(self, qapp):
        from satrecovery.vetting.gui.orbit_view import OrbitWorker

        class Explainer:
            def explain(self, obs, power):
                return "done"

        class Obs:
            id = 1

        worker = OrbitWorker(Explainer(), Obs(), np.zeros((4, 4)))
        running = OrbitWorker(Explainer(), Obs(), np.zeros((4, 4)))
        worker.run()
        assert worker.finished and worker._power is None
        assert OrbitWorker.prune([worker, running]) == [running]
