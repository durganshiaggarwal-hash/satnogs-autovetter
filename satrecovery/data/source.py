"""Real recordings as graded frames: the second :class:`FrameSource`.

Phase 6 wrote :class:`~satrecovery.evaluation.sources.FrameSource` before
any real data existed, with a promise: when RML24 arrived, every
experiment would rerun on it "without a line of the runner changing".
This class keeps that promise.  The benchmark runner, the statistics
and the result store all work on recorded frames exactly as they do on
simulated ones.

HOW A RECORDED FRAME IS MADE
----------------------------
1. Pick a clean recording (labelled 20, 18 or 16 dB by default) of the
   scenario's modulation.
2. Decode it with :class:`~satrecovery.data.reference.ReferenceDecoder`.
   If the decoder will not vouch for it, skip it.
3. Crop it to the reference's symbol grid and remove the radio's DC
   offset.
4. Add white Gaussian noise until the frame sits at the scenario's
   signal-to-noise ratio -- measured on the **same scale as the
   simulator's**, so "6 dB" here and "6 dB" there mean the same energy
   per symbol (see :meth:`RecordedSource.sample_snr_db`).  The noise
   already in the recording counts towards the total.
5. Label it with the reference bits, grading only the symbols the
   reference vouched for.

DESIGN NOTE: LISKOV SUBSTITUTION, FOR REAL THIS TIME
----------------------------------------------------
``RecordedSource`` can stand in wherever ``SimulatedSource`` does.  The
one difference is honest and declared through the interface: it
:meth:`supports` only the ``recorded`` profile, because a recording
already carries whatever the real channel did, and cannot be asked to
show "clean" or "doppler" instead.
"""

from __future__ import annotations

import zlib
from typing import Iterator

import numpy as np

from ..core.conditions import ChannelConditions
from ..core.exceptions import ValidationError
from ..core.frame import Frame
from ..core.signal import Signal
from ..evaluation.scenarios import ImpairmentProfile, Scenario
from ..evaluation.sources import FrameSource
from . import layout
from .archive import RecordingArchive
from .recording import Recording
from .reference import Reference, ReferenceDecoder


class RecordedSource(FrameSource):
    """Frames cut from real RML24 recordings, at any noise level.

    Parameters
    ----------
    archive:
        Where the recordings come from: the full file or a sample.
    decoder:
        Builds the reference labels.  The default is strict.
    snr_labels_db:
        Which of the dataset's SNR labels to draw from.  Only clean
        recordings can be decoded reliably, and the cleaner they are,
        the more of the noise in a graded frame is ours.
    symbol_rates:
        Restrict to these rates; ``None`` for all.
    reference_sps:
        The oversampling factor whose SNR scale is matched -- the
        simulator's, 8, by default.
    headroom_db:
        A recording is only used at a target SNR at least this far below
        its own, so that most of the frame's noise is the controlled,
        white, Gaussian noise added here.
    max_scan:
        How many recordings :meth:`supports` may decode looking for a
        usable one before concluding there is none.
    """

    PROFILE = "recorded"

    def __init__(
        self,
        archive: RecordingArchive,
        decoder: ReferenceDecoder | None = None,
        snr_labels_db: tuple[float, ...] = (20, 18, 16),
        symbol_rates: tuple[float, ...] | None = None,
        reference_sps: int = 8,
        headroom_db: float = 3.0,
        seed: int = 0,
        max_scan: int = 200,
        name: str = "RML24",
    ) -> None:
        if not isinstance(archive, RecordingArchive):
            raise ValidationError("archive", type(archive).__name__, "expected a RecordingArchive")
        if decoder is not None and not isinstance(decoder, ReferenceDecoder):
            raise ValidationError("decoder", type(decoder).__name__, "expected a ReferenceDecoder")
        if int(reference_sps) < 1:
            raise ValidationError("reference_sps", reference_sps, "must be at least 1")
        if not snr_labels_db:
            raise ValidationError("snr_labels_db", snr_labels_db, "needs at least one label")
        super().__init__(name)
        self._archive = archive
        self._decoder = decoder or ReferenceDecoder()
        self._labels = tuple(float(v) for v in snr_labels_db)
        self._rates = None if symbol_rates is None else tuple(float(r) for r in symbol_rates)
        self._reference_sps = int(reference_sps)
        self._headroom_db = float(headroom_db)
        self._seed = int(seed)
        self._max_scan = int(max_scan)
        # row -> (reference, signal power, native noise power).  Decoding
        # is the slow part, and the same clean recording is reused at
        # every noise level.
        self._cache: dict[int, tuple[Reference, float, float, int]] = {}

    # ------------------------------------------------------------------
    @property
    def archive(self) -> RecordingArchive:
        return self._archive

    @property
    def decoder(self) -> ReferenceDecoder:
        return self._decoder

    @classmethod
    def profile(cls) -> ImpairmentProfile:
        """The one profile this source produces."""
        return ImpairmentProfile.recorded()

    def candidates(self, modulation: str) -> np.ndarray:
        """Every row this source may draw for ``modulation``."""
        rows = []
        for label in self._labels:
            if self._rates is None:
                rows.append(self._archive.select(modulation, label))
            else:
                for rate in self._rates:
                    rows.append(self._archive.select(modulation, label, rate))
        return np.unique(np.concatenate(rows)) if rows else np.empty(0, dtype=np.int64)

    def reference(self, row: int) -> Reference:
        """The reference for one row, decoded once and remembered."""
        return self._entry(int(row))[0]

    def native_snr_db(self, row: int) -> float:
        """A recording's own SNR, on this source's (simulator) scale."""
        _, signal_power, noise_power, sps = self._entry(int(row))
        sample_snr = 10.0 * np.log10(signal_power / max(noise_power, 1e-30))
        return float(sample_snr - 10.0 * np.log10(self._reference_sps / sps))

    def sample_snr_db(self, snr_db: float, samples_per_symbol: int) -> float:
        """The per-sample SNR that gives the same energy per symbol.

        The simulator measures SNR per *sample* at its own oversampling
        factor; the matched filter then adds ``10 log10(sps)`` dB.  A
        recording at 2 samples per symbol and one at 10 would otherwise
        get very different energy per symbol at the same "SNR", and a
        curve mixing them would mean nothing.  Matching energy per
        symbol makes "6 dB" mean the same thing for every recording and
        for the simulator.
        """
        return float(snr_db) + 10.0 * np.log10(self._reference_sps / int(samples_per_symbol))

    def acceptance(self, modulation: str, limit: int | None = None) -> dict[str, object]:
        """Decode candidates and report how many the reference accepted.

        Decodes up to ``limit`` recordings (all candidates if ``None``).
        """
        rows = self.candidates(modulation)
        if limit is not None:
            rows = rows[: int(limit)]
        refs = [self.reference(r) for r in rows]
        accepted = [r for r in refs if r.accepted]
        return {
            "modulation": modulation,
            "decoded": len(refs),
            "accepted": len(accepted),
            "rate": (len(accepted) / len(refs)) if refs else 0.0,
            "median_evm_percent": float(np.median([r.evm_percent for r in accepted])) if accepted else None,
            "ungraded_fraction": (
                float(np.mean([r.n_ungraded / r.n_symbols for r in accepted])) if accepted else None
            ),
            "example_rejection": next((r.reason for r in refs if not r.accepted), None),
        }

    # ------------------------------------------------------------------
    def supports(self, scenario: Scenario) -> bool:
        if scenario.profile.name != self.PROFILE:
            return False
        if scenario.modulation not in self._archive.modulations():
            return False
        rows = self.candidates(scenario.modulation)[: self._max_scan]
        return any(self._usable(int(row), scenario.snr_db) for row in rows)

    def scenario_seed(self, scenario: Scenario) -> int:
        """Per-scenario seed, derived the same way as the simulator's."""
        return (self._seed * 1_000_003 + zlib.crc32(scenario.key.encode())) % (2**32)

    def _frames(self, scenario: Scenario, count: int) -> Iterator[Frame]:
        rng = np.random.default_rng(self.scenario_seed(scenario))
        rows = self.candidates(scenario.modulation)
        # Shuffled once per scenario, deterministically, so a benchmark
        # draws from every symbol rate and label rather than the first few
        # rows of the file.
        order = rng.permutation(rows)
        produced = 0
        while produced < count:
            progress = False
            for row in order:
                if produced >= count:
                    break
                if not self._usable(int(row), scenario.snr_db):
                    continue
                yield self.frame(int(row), scenario.snr_db, rng)
                produced += 1
                progress = True
            if not progress:
                raise ValidationError(
                    "scenario", repr(scenario), f"{self._name} has no usable recording for it"
                )
            # Fewer usable recordings than frames requested: go round
            # again.  Each pass adds fresh noise, so frames differ even
            # when the recording repeats -- the same way a simulator
            # reuses its channel model with fresh noise.

    def frame(self, row: int, snr_db: float, rng: np.random.Generator | None = None) -> Frame:
        """One graded frame from one recording at one noise level."""
        rng = rng if rng is not None else np.random.default_rng(self._seed)
        recording = self._archive.recording(int(row))
        reference, signal_power, noise_power, _ = self._entry(int(row))
        if not reference.accepted:
            raise ValidationError("row", row, f"reference rejected: {reference.reason}")

        samples = self._prepared(recording)[reference.crop]
        sps = recording.samples_per_symbol
        target = 10.0 ** (self.sample_snr_db(snr_db, sps) / 10.0)
        extra = signal_power / target - noise_power
        if extra < 0:
            raise ValidationError(
                "snr_db", snr_db, f"recording {row} is itself noisier than that"
            )
        sigma = np.sqrt(extra / 2.0)
        noisy = samples + sigma * (rng.standard_normal(samples.size) + 1j * rng.standard_normal(samples.size))

        return Frame(
            Signal(noisy, recording.sample_rate, description=f"RML24 row {row}"),
            bits=reference.graded_bits,
            modulation=reference.modulation,
            conditions=ChannelConditions(snr_db=float(snr_db)),
            tx_symbols=reference.symbols,
            samples_per_symbol=sps,
            symbol_rate=recording.symbol_rate,
            source="rml24",
            graded_symbols=reference.graded,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _prepared(recording: Recording) -> np.ndarray:
        """The samples with the radio's DC offset removed.

        Every real receiver has a DC blocker straight after the radio;
        the project's receivers were written for the simulator, which
        has no DC offset, so it is removed here once, for all of them.
        """
        return recording.samples - np.mean(recording.samples)

    def _entry(self, row: int) -> tuple[Reference, float, float, int]:
        if row not in self._cache:
            recording = self._archive.recording(row)
            reference = self._decoder.decode(recording)
            samples = self._prepared(recording)[reference.crop]
            signal_power, noise_power = self._powers(samples, recording)
            self._cache[row] = (reference, signal_power, noise_power, recording.samples_per_symbol)
        return self._cache[row]

    @staticmethod
    def _powers(samples: np.ndarray, recording: Recording) -> tuple[float, float]:
        """Signal and noise power, from the spectrum.

        A root-raised-cosine signal occupies only ``(1 + 0.35) / 2`` of
        its symbol rate either side of the carrier.  Everything beyond
        that is noise, and if the noise is white -- as the recordings'
        flat floor suggests -- its level out there is its level
        everywhere.  The quietest region is used, staying clear of the
        band edge and of the radio's own anti-aliasing filter.
        """
        n = samples.size
        spectrum = np.abs(np.fft.fft(samples)) ** 2 / n
        freqs = np.abs(np.fft.fftfreq(n, d=1.0 / recording.sample_rate))
        edge = 0.5 * (1.0 + layout.ROLLOFF) * recording.symbol_rate
        outside = (freqs > 1.1 * edge) & (freqs < 0.45 * recording.sample_rate)
        total = float(np.mean(np.abs(samples) ** 2))
        if outside.sum() < 16:
            # Too little spectrum outside the signal to measure: treat
            # the recording as noiseless and let the decoder's EVM speak.
            return total, 0.0
        noise = float(np.median(spectrum[outside])) / np.log(2.0)  # median of an exponential
        noise = min(noise, 0.5 * total)
        return total - noise, noise

    def _usable(self, row: int, snr_db: float) -> bool:
        reference = self.reference(row)
        if not reference.accepted:
            return False
        return self.native_snr_db(row) >= float(snr_db) + self._headroom_db

    def __repr__(self) -> str:
        return f"RecordedSource({self._archive!r}, labels {self._labels})"


__all__ = ["RecordedSource"]
