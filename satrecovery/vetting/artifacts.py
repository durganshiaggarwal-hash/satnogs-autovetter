"""Downloading an observation's files, once, into a local cache.

:class:`ArtifactCache` keeps one decoded waterfall per observation
(``waterfalls/<id>.npz``, about 120 KB) and, if asked, the audio
(``audio/<id>.ogg``, a few MB).  The original waterfall PNG (about 2 MB)
is thrown away after decoding unless ``keep_png`` is set, so 10,000
observations fit in about 1.2 GB.

Files are written to ``<name>.part`` first and renamed when complete, so
an interrupted download never leaves a half file that looks finished.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.exceptions import SatRecoveryError
from .observation import Observation
from .source import Transport, UrllibTransport
from .waterfall import WaterfallDecodeError, WaterfallImage


class ArtifactError(SatRecoveryError):
    """An observation's file could not be downloaded or read."""


class ArtifactCache:
    #: refuse anything bigger than this (a waterfall PNG is ~2 MB, audio ~5 MB)
    MAX_BYTES = 64 * 1024 * 1024

    def __init__(self, root: str | Path, transport: Transport | None = None,
                 time_bins: int = 256, freq_bins: int = 256, keep_png: bool = False) -> None:
        self._root = Path(root)
        self._transport = transport or UrllibTransport()
        self._bins = (int(time_bins), int(freq_bins))
        self._keep_png = keep_png

    @property
    def root(self) -> Path:
        return self._root

    def waterfall_path(self, obs: Observation | int) -> Path:
        return self._root / "waterfalls" / f"{int(getattr(obs, 'id', obs))}.npz"

    def audio_path(self, obs: Observation | int) -> Path:
        return self._root / "audio" / f"{int(getattr(obs, 'id', obs))}.ogg"

    def has_waterfall(self, obs: Observation | int) -> bool:
        return self.waterfall_path(obs).exists()

    # ------------------------------------------------------------------
    def _download(self, url: str, target: Path, tries: int = 3) -> Path:
        from .source import SatnogsError

        for attempt in range(tries):
            try:
                response = self._transport.get(url, timeout=120.0)
                break
            except SatnogsError:
                if attempt == tries - 1:
                    raise
                import time

                time.sleep(2.0 * (attempt + 1))  # a reset connection usually works on the next try
        if response.status != 200:
            raise ArtifactError(f"{target.name}: server answered {response.status}")
        if len(response.body) > self.MAX_BYTES:
            raise ArtifactError(f"{target.name}: {len(response.body)} bytes is too big")
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        part.write_bytes(response.body)
        part.replace(target)
        return target

    def axes_path(self, obs: Observation | int) -> Path:
        return self._root / "axes" / f"{int(getattr(obs, 'id', obs))}.json"

    def _save_axes(self, obs: Observation, png: Path) -> None:
        """Read the picture's kHz/seconds scales; a failure is saved too, so it is not retried."""
        from .axes import AxisReadError, read_axes_png

        path = self.axes_path(obs)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            record = {"ok": True, **read_axes_png(png).to_dict()}
        except (AxisReadError, WaterfallDecodeError) as error:
            record = {"ok": False, "reason": str(error)}
        path.write_text(json.dumps(record), encoding="utf-8")

    def axes(self, obs: Observation, download: bool = True):
        """The waterfall's :class:`~satrecovery.vetting.axes.WaterfallAxes`, or ``None`` if unreadable.

        Waterfalls downloaded before the axes were read have only the decoded
        numbers left, so with ``download`` the picture is fetched once more.
        """
        from .axes import WaterfallAxes

        path = self.axes_path(obs)
        if not path.exists():
            if not download or not obs.waterfall_url:
                return None
            png = self._download(obs.waterfall_url, self._root / "png" / f"{obs.id}.axes.png")
            try:
                self._save_axes(obs, png)
            finally:
                if not self._keep_png:
                    png.unlink(missing_ok=True)
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record.get("ok"):
            return None
        return WaterfallAxes(record["khz_left"], record["khz_right"], record["seconds_bottom"], record["seconds_top"])

    def waterfall(self, obs: Observation) -> WaterfallImage:
        """The decoded waterfall, downloading it the first time."""
        path = self.waterfall_path(obs)
        if path.exists():
            return WaterfallImage.load(path)
        if not obs.waterfall_url:
            raise ArtifactError(f"observation {obs.id} has no waterfall")
        png = self._download(obs.waterfall_url, self._root / "png" / f"{obs.id}.png")
        try:
            image = WaterfallImage.from_png(png).resized(*self._bins)
            self._save_axes(obs, png)
        except WaterfallDecodeError as error:
            raise ArtifactError(f"observation {obs.id}: {error}") from error
        finally:
            if not self._keep_png:
                png.unlink(missing_ok=True)
        image.save(path)
        return image

    def audio(self, obs: Observation) -> Path:
        """The audio file's path, downloading it the first time."""
        path = self.audio_path(obs)
        if path.exists():
            return path
        if not obs.audio_url:
            raise ArtifactError(f"observation {obs.id} has no audio")
        return self._download(obs.audio_url, path)
