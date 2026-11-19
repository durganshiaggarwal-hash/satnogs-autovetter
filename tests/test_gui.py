"""Tests for Phase 8 -- the desktop application.

Qt runs offscreen here, so these tests need no display: the real widgets
are created, driven and inspected exactly as on a desktop, they just
never appear on a screen.  The integration tests start a real backend in
a temporary folder and let the pages talk to it over HTTP.

What is tested is what a user would notice if it broke: units converted
correctly, the right number of results shown, errors reported rather
than swallowed, and the abstract base class actually being abstract --
which, for Qt widgets, Python does not guarantee on its own.
"""

from __future__ import annotations

import math
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6", reason="the GUI needs PySide6")
pytest.importorskip("pyqtgraph", reason="the GUI needs pyqtgraph")
pytest.importorskip("fastapi", reason="the GUI talks to the backend")

from PySide6.QtWidgets import QApplication  # noqa: E402

from satrecovery.backend.schemas import ConditionsIn  # noqa: E402
from satrecovery.gui import theme  # noqa: E402
from satrecovery.gui.client import BackendError, HttpBackendClient  # noqa: E402
from satrecovery.gui.pages import (  # noqa: E402
    BenchmarkPage,
    ExperimentPage,
    ModelsPage,
    Page,
    StagesPage,
)
from satrecovery.gui.pages.stages import explain  # noqa: E402
from satrecovery.gui.server import EmbeddedServer, free_port, is_backend  # noqa: E402
from satrecovery.gui.tasks import JobWatcher  # noqa: E402
from satrecovery.gui.widgets import ConditionsPanel, LabeledSlider  # noqa: E402
from satrecovery.gui.widgets.tables import percent  # noqa: E402
from satrecovery.gui.window import MainWindow  # noqa: E402
from satrecovery.gui import motion  # noqa: E402
from satrecovery.gui.motion import (  # noqa: E402
    AnimatedTabs,
    Chip,
    CountUpLabel,
    Motion,
    PillButton,
    SmoothProgress,
    ToggleSwitch,
)


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    theme.load_fonts()
    theme.configure_plots()
    application.setStyleSheet(theme.stylesheet())
    return application


def pump(app, condition, timeout=30.0):
    """Process Qt events until ``condition()`` is true."""
    deadline = time.time() + timeout
    while not condition():
        app.processEvents()
        if time.time() > deadline:
            raise TimeoutError("condition never became true")
        time.sleep(0.01)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    with EmbeddedServer(data_dir=tmp_path_factory.mktemp("gui")) as running:
        yield running


@pytest.fixture(scope="module")
def client(server):
    return HttpBackendClient(server.url)


# ======================================================================
# Theme
# ======================================================================
class TestTheme:
    def test_every_receiver_has_its_own_colour(self):
        colours = list(theme.RECEIVER_COLOURS.values())
        assert len(colours) == len(set(colours))

    def test_error_red_is_never_a_receiver_colour(self):
        assert theme.CRITICAL not in theme.RECEIVER_COLOURS.values()

    def test_an_unknown_receiver_gets_the_neutral(self):
        assert theme.receiver_colour("someone-new") == theme.OTHER

    def test_the_bundled_typefaces_load(self, app):
        families = theme.load_fonts()
        assert theme.SANS in families and theme.MONO in families

    def test_the_stylesheet_points_at_files_that_exist(self):
        for name in ("check.svg", "chevron-up.svg", "chevron-down.svg"):
            assert (theme.ASSETS / name).exists()
            assert name in theme.stylesheet()


# ======================================================================
# Controls
# ======================================================================
class TestLabeledSlider:
    def test_values_are_in_real_units(self, app):
        slider = LabeledSlider("x", -10, 30, 0.5, 6.0, "dB")
        assert slider.value == 6.0
        slider.value = 12.25          # between steps: snaps to the nearest one
        assert slider.value in (12.0, 12.5)

    def test_values_are_clamped_to_the_range(self, app):
        slider = LabeledSlider("x", -10, 30, 0.5, 0)
        slider.value = 99
        assert slider.value == 30
        slider.value = -99
        assert slider.value == -10

    def test_moving_it_announces_the_real_value(self, app):
        slider = LabeledSlider("x", 0, 100, 5, 0)
        heard = []
        slider.valueChanged.connect(heard.append)
        slider.value = 35
        assert heard == [35.0]

    def test_a_backwards_range_is_refused(self, app):
        with pytest.raises(ValueError):
            LabeledSlider("x", 5, 1, 1, 2)


class TestConditionsPanel:
    def test_it_produces_exactly_what_the_api_accepts(self, app):
        panel = ConditionsPanel()
        conditions = panel.conditions()
        assert set(conditions) == set(ConditionsIn.model_fields)
        ConditionsIn(**conditions).to_domain()          # and the domain accepts it

    def test_degrees_on_screen_become_radians_for_the_api(self, app):
        panel = ConditionsPanel()
        panel.phase.value = 90
        assert panel.conditions()["phase_offset_rad"] == pytest.approx(math.pi / 2)

    def test_a_profile_sets_the_sliders_but_not_the_strength(self, app):
        panel = ConditionsPanel()
        panel.snr.value = 7
        panel.apply_profile({"freq_offset_hz": 800.0, "phase_offset_rad": 0.7,
                             "fading": "multipath_rician", "rician_k_db": 10.0})
        out = panel.conditions()
        assert out["snr_db"] == 7
        assert out["freq_offset_hz"] == 800
        assert out["fading"] == "multipath_rician"
        assert out["phase_offset_rad"] == pytest.approx(math.radians(40), abs=0.05)

    def test_state_round_trips_between_panels(self, app):
        """How the Experiment page hands a signal to Step by step."""
        first, second = ConditionsPanel(), ConditionsPanel()
        first.set_modulations(["QPSK", "16QAM"], "QPSK")
        first.doppler.value = 450
        first.timing.value = 1.2
        second.set_modulations(["QPSK", "16QAM"], "16QAM")
        second.set_state(first.modulation(), first.conditions())
        assert second.modulation() == "QPSK"
        copied, original = second.conditions(), first.conditions()
        assert copied["fading"] == original["fading"]
        numbers = [k for k in original if k != "fading"]
        assert [copied[k] for k in numbers] == pytest.approx([original[k] for k in numbers])

    def test_the_direct_path_slider_follows_the_echoes_choice(self, app):
        panel = ConditionsPanel()
        assert not panel.k_factor.isEnabled()
        panel.apply_profile({"fading": "flat_rician"})
        assert panel.k_factor.isEnabled()


def test_percentages_read_naturally():
    assert percent(0) == "0"
    assert percent(None) == "–"
    assert percent(0.0823) == "8.2%"
    assert percent(0.0017) == "0.17%"
    assert percent(2e-6).endswith("%")


# ======================================================================
# The abstract base class, which Qt would otherwise let slip
# ======================================================================
class TestPageIsAbstract:
    def test_an_incomplete_page_cannot_be_built(self, app, client):
        class Incomplete(Page):
            @property
            def title(self):
                return "half a page"

        with pytest.raises(TypeError, match="refresh"):
            Incomplete(client)

    def test_the_base_itself_cannot_be_built(self, app, client):
        with pytest.raises(TypeError):
            Page(client)

    def test_a_complete_page_can(self, app, client):
        class Complete(Page):
            @property
            def title(self):
                return "whole"

            def refresh(self):
                pass

        assert Complete(client).title == "whole"


# ======================================================================
# Talking to the backend
# ======================================================================
class TestClient:
    def test_an_unreachable_backend_says_so(self):
        dead = HttpBackendClient(f"http://127.0.0.1:{free_port()}", timeout=1)
        with pytest.raises(BackendError) as caught:
            dead.health()
        assert caught.value.unreachable

    def test_a_refusal_carries_the_backends_reason(self, client):
        with pytest.raises(BackendError) as caught:
            client.experiment({"conditions": {"snr_db": 99}})
        assert caught.value.status == 422
        assert caught.value.field == "snr_db"
        assert "between" in caught.value.detail

    def test_a_type_error_is_explained_too(self, client):
        with pytest.raises(BackendError) as caught:
            client.experiment({"receivers": "classical"})     # a string, not a list
        assert caught.value.status == 422
        assert "receivers" in caught.value.detail

    def test_the_embedded_server_is_recognised(self, server):
        assert is_backend(server.url)
        assert not is_backend(f"http://127.0.0.1:{free_port()}", timeout=0.5)


# ======================================================================
# The pages, against a real backend
# ======================================================================
class TestPages:
    def test_the_experiment_page_runs_and_shows_every_receiver(self, app, client):
        page = ExperimentPage(client)
        page.refresh()
        pump(app, lambda: page.table.rowCount() > 0)
        assert page.table.rowCount() == len(page.receivers.selected())
        assert page.seed_label.text().startswith("Signal number")

    def test_keeping_the_signal_reuses_its_seed(self, app, client):
        page = ExperimentPage(client)
        page.refresh()
        pump(app, lambda: page.table.rowCount() > 0)
        first = page.request_body()["seed"]
        page.conditions.snr.value = 3
        page.run()
        pump(app, lambda: not page.tasks.busy)
        assert first is not None and page.request_body()["seed"] == first

    def test_a_new_signal_draws_a_new_seed(self, app, client):
        page = ExperimentPage(client)
        page.refresh()
        pump(app, lambda: page.table.rowCount() > 0)
        before = page.request_body()["seed"]
        page.new_signal()
        pump(app, lambda: not page.tasks.busy and page.request_body()["seed"] != before)

    def test_step_by_step_shows_every_stage_with_words(self, app, client):
        page = StagesPage(client)
        page.refresh()
        pump(app, lambda: page.receiver.count() > 0)
        page.follow({"modulation": "QPSK", "seed": 4,
                     "conditions": {"snr_db": 12, "freq_offset_hz": 600, "fading": "none"}})
        pump(app, lambda: len(page._group.buttons()) > 0)
        assert len(page._group.buttons()) == 6
        for index in range(6):
            page.show_stage(index)
            assert page.explanation.text()
        assert not page.next_button.isEnabled()

    def test_a_failed_request_reaches_the_status_bar(self, app, client):
        page = ExperimentPage(client)
        messages = []
        page.message.connect(lambda text, error: messages.append((text, error)))
        page.conditions.set_modulations(["QPSK"], "QPSK")
        page.tasks.run(lambda: client.experiment({"conditions": {"snr_db": 99}}),
                       lambda r: None, page.report_error)
        pump(app, lambda: messages)
        assert messages[0][1] is True and "snr_db" in messages[0][0]

    def test_a_benchmark_runs_from_the_page_and_appears(self, app, client):
        page = BenchmarkPage(client)
        page.refresh()
        pump(app, lambda: page.modulations.selected())
        page.snr_from.setValue(4)
        page.snr_to.setValue(8)
        page.frames.setValue(2)
        page.start()
        pump(app, lambda: page._watcher is not None, timeout=15)
        pump(app, lambda: page._watcher is None, timeout=120)
        pump(app, lambda: page.runs.count() > 0 and page.table.rowCount() > 0)
        assert "Finished" in page.progress_label.text()

    def test_the_models_page_lists_every_modulation(self, app, client):
        page = ModelsPage(client)
        page.refresh()
        pump(app, lambda: len(page.cards) > 0)
        assert len(page.cards) == 6
        assert all(card.button.text() == "Train" for card in page.cards.values())

    def test_a_job_watcher_reports_to_the_end(self, app, client):
        job = client.start_benchmark({
            "name": "watched", "modulations": ["BPSK"], "snrs_db": [6],
            "receivers": ["classical"], "frames_per_scenario": 1,
        })
        watcher = JobWatcher(client, job["id"], interval_ms=50)
        seen, done = [], []
        watcher.progress.connect(lambda f, m: seen.append(f))
        watcher.finished.connect(done.append)
        watcher.start()
        pump(app, lambda: done)
        assert done[0]["state"] == "done"


class TestWindow:
    def test_every_page_is_a_tab_in_order(self, app, client):
        window = MainWindow(client)
        assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == \
            ["Experiment", "Step by step", "Benchmark", "Models"]

    def test_following_a_signal_switches_tab(self, app, client):
        window = MainWindow(client)
        window.stages.refresh()
        pump(app, lambda: window.stages.receiver.count() > 0)
        window.experiment.follow_requested.emit({
            "modulation": "QPSK", "seed": 2, "conditions": {"snr_db": 10},
        })
        assert window.tabs.currentWidget() is window.stages
        pump(app, lambda: len(window.stages._group.buttons()) > 0)


def test_the_timing_explanation_warns_about_echoes():
    frame = {"samples_per_symbol": 8}
    with_echoes = explain("GardnerTiming", {"timing_offset_samples": 2.1},
                          {"timing_offset": 1.5, "fading": "multipath_rician"}, frame)
    without = explain("GardnerTiming", {"timing_offset_samples": 1.48},
                      {"timing_offset": 1.5, "fading": "none"}, frame)
    assert "echoes" in with_echoes.lower()
    assert "echoes" not in without.lower()


# ======================================================================
# Motion: animated widgets must still honour the Qt widgets they replace
# ======================================================================
@pytest.fixture
def still():
    """Animations off, so every change lands immediately."""
    before = Motion.enabled
    Motion.enabled = False
    yield
    Motion.enabled = before


class TestMotion:
    def test_with_motion_off_an_animation_jumps_to_the_end(self, app, still):
        seen = []
        result = motion.animate(None, 0.0, 1.0, 500, seen.append)
        assert result is None and seen == [1.0]

    def test_motion_off_means_zero_duration(self, still):
        assert Motion.ms(Motion.SLOW) == 0

    def test_the_toggle_switch_is_still_a_checkbox(self, app, still):
        switch = ToggleSwitch("Keep")
        from PySide6.QtWidgets import QCheckBox
        assert isinstance(switch, QCheckBox)          # Liskov: usable wherever a QCheckBox is
        seen = []
        switch.toggled.connect(seen.append)
        switch.click()
        assert switch.isChecked() and seen == [True]

    def test_a_chip_is_still_a_checkbox_and_never_squeezed(self, app, still):
        chip = Chip("16QAM", "#2a78d6")
        chip.click()
        assert chip.isChecked()
        assert chip.minimumSizeHint() == chip.sizeHint()

    def test_a_pill_button_still_clicks(self, app, still):
        button = PillButton("Go", "primary")
        clicks = []
        button.clicked.connect(lambda: clicks.append(1))
        button.click()
        assert clicks == [1]

    def test_a_disabled_pill_button_does_not_click(self, app, still):
        button = PillButton("Go", "secondary")
        button.setEnabled(False)
        clicks = []
        button.clicked.connect(lambda: clicks.append(1))
        button.click()
        assert clicks == []

    def test_an_unknown_variant_is_refused(self, app):
        with pytest.raises(ValueError):
            PillButton("Go", "sparkly")

    def test_count_up_lands_on_the_value(self, app, still):
        label = CountUpLabel(lambda v: f"{v:.0f}")
        label.set_value(42)
        assert label.text() == "42"

    def test_progress_lands_on_the_fraction(self, app, still):
        bar = SmoothProgress()
        bar.set_fraction(0.75)
        assert bar.value() == pytest.approx(0.75)
        bar.reset()
        assert bar.value() == 0

    def test_animated_tabs_behave_like_a_tab_widget(self, app, still):
        from PySide6.QtWidgets import QWidget
        tabs = AnimatedTabs()
        first, second = QWidget(), QWidget()
        tabs.addTab(first, "One")
        tabs.addTab(second, "Two")
        changed = []
        tabs.currentChanged.connect(changed.append)
        tabs.setCurrentWidget(second)
        assert tabs.count() == 2 and tabs.tabText(1) == "Two"
        assert tabs.currentIndex() == 1 and tabs.currentWidget() is second
        assert changed == [1]
