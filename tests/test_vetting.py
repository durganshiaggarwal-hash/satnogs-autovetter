"""Tests for the SatNOGS data layer (satrecovery.vetting).  No network:
the API is replaced by a fake transport that replays pages."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from satrecovery.core.exceptions import ValidationError
from satrecovery.vetting import (
    ArtifactCache,
    ArtifactError,
    Manifest,
    ManifestSource,
    Observation,
    ObservationQuery,
    Response,
    SatnogsApiSource,
    SatnogsError,
    Transport,
    WaterfallDecodeError,
    WaterfallImage,
    WaterfallLabel,
)
from satrecovery.vetting.source import next_link

BASE = SatnogsApiSource.BASE_URL


def api_row(obs_id: int = 15487232, **changes) -> dict:
    """A row shaped like the live API's (fields checked November 2026)."""
    row = {
        "id": obs_id, "start": "2026-11-19T12:32:09Z", "end": "2026-11-19T12:39:58Z",
        "ground_station": 223, "norad_cat_id": 62394, "station_name": "Test station",
        "payload": f"https://cdn.example/satnogs_{obs_id}.ogg",
        "waterfall": f"https://cdn.example/waterfall_{obs_id}.png",
        "demoddata": [{"payload_demod": "x"}, {"payload_demod": "y"}],
        "status": "good", "waterfall_status": "with-signal", "waterfall_status_user": 42,
        "transmitter_mode": "GMSK", "transmitter_baud": 9600.0, "observation_frequency": 436775000,
        "max_altitude": 33.5, "tle1": "1 62394U ...", "tle2": "2 62394 ...",
    }
    row.update(changes)
    return row


class FakeTransport(Transport):
    """Serves fixed responses by URL; records every request."""

    def __init__(self, pages: dict[str, list[Response] | Response]) -> None:
        self._pages = {url: (list(r) if isinstance(r, list) else [r]) for url, r in pages.items()}
        self.requests: list[str] = []

    def get(self, url: str, timeout: float = 60.0) -> Response:
        self.requests.append(url)
        replies = self._pages.get(url)
        if not replies:
            return Response(404, {}, b"not found")
        return replies.pop(0) if len(replies) > 1 else replies[0]


def page(rows: list[dict], next_url: str | None = None) -> Response:
    headers = {"link": f'<{next_url}>; rel="next"'} if next_url else {}
    return Response(200, headers, json.dumps(rows).encode())


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# ----------------------------------------------------------------------
class TestObservation:
    def test_parses_an_api_row(self):
        obs = Observation.from_api(api_row())
        assert obs.id == 15487232
        assert obs.waterfall_label is WaterfallLabel.SIGNAL
        assert obs.labelled_by_human
        assert obs.n_decoded_frames == 2
        assert obs.duration_s == pytest.approx(469.0)
        assert obs.tle == ("1 62394U ...", "2 62394 ...")
        assert obs.is_trainable

    def test_unvetted_and_missing_fields(self):
        obs = Observation.from_api(api_row(waterfall_status="unknown", waterfall_status_user=None,
                                           waterfall=None, demoddata=None, tle1=None, start=None))
        assert obs.waterfall_label is WaterfallLabel.UNKNOWN
        assert not obs.labelled_by_human and not obs.is_trainable
        assert obs.n_decoded_frames == 0 and obs.tle is None and obs.duration_s is None

    def test_unknown_label_text_is_unknown_not_an_error(self):
        assert WaterfallLabel.parse("something-new") is WaterfallLabel.UNKNOWN
        assert WaterfallLabel.parse(None) is WaterfallLabel.UNKNOWN

    def test_round_trips_through_a_dict(self):
        obs = Observation.from_api(api_row())
        again = Observation.from_dict(json.loads(json.dumps(obs.to_dict())))
        assert again == obs and again.tle == obs.tle

    def test_is_immutable_and_checked(self):
        obs = Observation.from_api(api_row())
        with pytest.raises(AttributeError):
            obs.id = 5  # type: ignore[misc]
        with pytest.raises(ValidationError):
            Observation.from_api({"status": "good"})
        with pytest.raises(ValidationError):
            Observation.from_api(api_row(obs_id=0))


class TestQuery:
    def test_label_uses_the_numbers_the_api_accepts(self):
        assert ObservationQuery(waterfall_label=WaterfallLabel.SIGNAL).to_params()["waterfall_status"] == "1"
        assert ObservationQuery(waterfall_label=WaterfallLabel.NO_SIGNAL).to_params()["waterfall_status"] == "0"

    def test_satellite_filter_name(self):
        assert ObservationQuery(satellite=62394).to_params()["norad_cat_id"] == "62394"

    def test_cannot_ask_for_unvetted(self):
        with pytest.raises(ValidationError):
            ObservationQuery(waterfall_label=WaterfallLabel.UNKNOWN)

    def test_next_link(self):
        assert next_link({"link": f'<{BASE}?cursor=abc>; rel="next"'}) == f"{BASE}?cursor=abc"
        assert next_link({"link": f'<{BASE}?cursor=abc>; rel="prev"'}) is None
        assert next_link({}) is None


class TestApiSource:
    def _source(self, pages, **kwargs):
        clock = FakeClock()
        transport = FakeTransport(pages)
        return SatnogsApiSource(transport, sleep=clock.sleep, clock=clock, **kwargs), transport, clock

    def test_follows_pages_until_the_end(self):
        first = f"{BASE}?format=json"
        second = f"{BASE}?cursor=2"
        source, transport, _ = self._source({
            first: page([api_row(1), api_row(2)], second),
            second: page([api_row(3)]),
        })
        assert [o.id for o in source.fetch(ObservationQuery())] == [1, 2, 3]
        assert transport.requests == [first, second]

    def test_limit_stops_early_without_fetching_more_pages(self):
        first = f"{BASE}?format=json"
        source, transport, _ = self._source({first: page([api_row(1), api_row(2)], f"{BASE}?cursor=2")})
        assert [o.id for o in source.fetch(ObservationQuery(), limit=2)] == [1, 2]
        assert len(transport.requests) == 1

    def test_waits_between_requests(self):
        first, second = f"{BASE}?format=json", f"{BASE}?cursor=2"
        source, _, clock = self._source({first: page([api_row(1)], second), second: page([api_row(2)])},
                                        min_interval=1.0)
        list(source.fetch(ObservationQuery()))
        assert clock.sleeps == [pytest.approx(1.0)]

    def test_backs_off_when_busy_then_succeeds(self):
        first = f"{BASE}?format=json"
        source, transport, clock = self._source(
            {first: [Response(429, {"retry-after": "7"}, b""), page([api_row(1)])]}, min_interval=0.0)
        assert [o.id for o in source.fetch(ObservationQuery())] == [1]
        assert 7.0 in clock.sleeps and len(transport.requests) == 2

    def test_gives_up_after_retries(self):
        first = f"{BASE}?format=json"
        source, transport, _ = self._source({first: Response(503, {}, b"")}, min_interval=0.0, max_retries=2)
        with pytest.raises(SatnogsError, match="busy"):
            list(source.fetch(ObservationQuery()))
        assert len(transport.requests) == 3

    def test_a_bad_request_is_an_error_not_a_retry(self):
        first = f"{BASE}?format=json"
        source, transport, _ = self._source({first: Response(400, {}, b'{"bad": "filter"}')})
        with pytest.raises(SatnogsError, match="400"):
            list(source.fetch(ObservationQuery()))
        assert len(transport.requests) == 1


class TestManifest:
    def test_add_skips_duplicates_and_reloads(self, tmp_path):
        manifest = Manifest(tmp_path / "m.jsonl")
        assert manifest.add(Observation.from_api(api_row(1)))
        assert not manifest.add(Observation.from_api(api_row(1)))
        manifest.add(Observation.from_api(api_row(2, waterfall_status="without-signal")))
        again = Manifest(tmp_path / "m.jsonl")
        assert len(again) == 2 and 1 in again
        assert again.counts() == {"with-signal": 1, "without-signal": 1}

    def test_survives_a_half_written_last_line(self, tmp_path):
        path = tmp_path / "m.jsonl"
        Manifest(path).add(Observation.from_api(api_row(1)))
        with path.open("a") as handle:
            handle.write('{"id": 2, "norad')
        assert [o.id for o in Manifest(path)] == [1]

    def test_manifest_source_filters_like_the_api(self, tmp_path):
        manifest = Manifest(tmp_path / "m.jsonl")
        manifest.add(Observation.from_api(api_row(1)))
        manifest.add(Observation.from_api(api_row(2, waterfall_status="without-signal", norad_cat_id=7)))
        source = ManifestSource(manifest)
        assert [o.id for o in source.fetch(ObservationQuery(waterfall_label=WaterfallLabel.NO_SIGNAL))] == [2]
        assert [o.id for o in source.fetch(ObservationQuery(satellite=62394))] == [1]
        assert len(list(source.fetch(ObservationQuery(), limit=1))) == 1


# ----------------------------------------------------------------------
def render_waterfall(power: np.ndarray) -> bytes:
    """Draw ``power`` the way SatNOGS does: viridis, axes, ticks and a colour bar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 7), dpi=100)
    shown = ax.imshow(power, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest",
                      extent=(-24, 24, 0, 400))
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Time (seconds)")
    fig.colorbar(shown, ax=ax, label="Power (dB)")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def known_spectrum() -> np.ndarray:
    rng = np.random.default_rng(3)
    power = rng.random((60, 40)) * 0.3
    power[:10, 18:22] = 1.0          # a strong burst at the START of the pass
    power[:, 5] = 0.6                # a steady carrier
    return power


class TestWaterfallImage:
    def test_recovers_the_plotted_numbers(self, tmp_path, known_spectrum):
        png = tmp_path / "w.png"
        png.write_bytes(render_waterfall(known_spectrum))
        image = WaterfallImage.from_png(png).resized(60, 40)
        assert np.corrcoef(image.power.ravel(), known_spectrum.ravel())[0, 1] > 0.95

    def test_time_runs_forwards(self, tmp_path, known_spectrum):
        png = tmp_path / "w.png"
        png.write_bytes(render_waterfall(known_spectrum))
        image = WaterfallImage.from_png(png).resized(60, 40)
        assert image.power[:10, 18:22].mean() > image.power[-10:, 18:22].mean() + 0.3

    def test_rejects_a_picture_that_is_not_a_waterfall(self, tmp_path):
        from matplotlib import image

        path = tmp_path / "grey.png"
        image.imsave(path, np.full((100, 100), 0.5), cmap="gray")
        with pytest.raises(WaterfallDecodeError):
            WaterfallImage.from_png(path)

    def test_resize_save_load(self, tmp_path, known_spectrum):
        image = WaterfallImage(known_spectrum).resized(32, 16)
        assert image.shape == (32, 16)
        again = WaterfallImage.load(image.save(tmp_path / "w.npz"))
        assert np.allclose(again.power, image.power, atol=1e-4)

    def test_is_read_only_and_checked(self, known_spectrum):
        image = WaterfallImage(known_spectrum)
        with pytest.raises(ValueError):
            image.power[0, 0] = 1.0
        with pytest.raises(ValidationError):
            WaterfallImage(np.zeros(10))
        with pytest.raises(ValidationError):
            WaterfallImage(np.full((10, 10), np.nan))


class TestArtifactCache:
    def test_downloads_once_and_drops_the_png(self, tmp_path, known_spectrum):
        obs = Observation.from_api(api_row(5))
        transport = FakeTransport({obs.waterfall_url: Response(200, {}, render_waterfall(known_spectrum))})
        cache = ArtifactCache(tmp_path, transport, time_bins=32, freq_bins=32)
        assert cache.waterfall(obs).shape == (32, 32)
        assert cache.has_waterfall(obs)
        assert not list((tmp_path / "png").glob("*.png"))
        cache.waterfall(obs)
        assert len(transport.requests) == 1

    def test_errors_are_artifact_errors(self, tmp_path):
        obs = Observation.from_api(api_row(6))
        cache = ArtifactCache(tmp_path, FakeTransport({}))
        with pytest.raises(ArtifactError, match="404"):
            cache.waterfall(obs)
        with pytest.raises(ArtifactError, match="no waterfall"):
            cache.waterfall(Observation.from_api(api_row(7, waterfall=None)))
        assert not list(tmp_path.rglob("*.part"))

    def test_audio_is_saved_as_is(self, tmp_path):
        obs = Observation.from_api(api_row(8))
        cache = ArtifactCache(tmp_path, FakeTransport({obs.audio_url: Response(200, {}, b"OggS...")}))
        assert cache.audio(obs).read_bytes() == b"OggS..."
