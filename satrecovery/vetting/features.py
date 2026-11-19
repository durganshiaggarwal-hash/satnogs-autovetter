"""A dozen measurements of a waterfall that a human vetter looks for.

A volunteer checking a waterfall asks: is there a line or bursts near
the centre frequency (where Doppler correction puts the satellite we
scheduled)?  How strong, how wide, how much of the pass?  And is a bright
track *moving* -- the S-curve of a different satellite, which does not
count?  Each measurement below answers one of those questions, so a
model built on them can say *why* it decided, in words.

All measurements take a normalised waterfall (see
:func:`~satrecovery.vetting.dataset.normalise`): values are "spreads
above this picture's median".
"""

from __future__ import annotations

import numpy as np

from ..core.exceptions import ValidationError

#: name -> what it measures, in plain words (shown as the model's reasons)
FEATURES: dict[str, str] = {
    "peak_prominence": "how far the brightest frequency stands above the rest, averaged over the pass",
    "peak_offset": "how far that brightest frequency is from the centre (0 = centre, 0.5 = edge)",
    "peak_width": "how wide the bright frequency band is (share of the band)",
    "centre_excess": "how much brighter the centre band is than the whole band",
    "centre_p99": "how bright the brightest moments in the centre band are",
    "band_p995": "how bright the brightest moments anywhere are",
    "centre_lit_rows": "share of the pass with something strong in the centre band",
    "any_lit_rows": "share of the pass with something strong anywhere",
    "centre_burstiness": "how much the centre band's strength jumps around (packets come in bursts)",
    "track_wander": "how much the brightest point moves in frequency over the pass (a moving track is another satellite)",
    "track_steadiness": "share of strong moments whose brightest point is near the centre",
    "row_noise_variation": "how much the noise level changes over the pass (interference, gain changes)",
}

#: clip normalised values to this range first, as the CNN does
CLIP = (-4.0, 8.0)
#: "strong" means this many spreads above the median
STRONG = 3.0


def waterfall_features(power: np.ndarray) -> np.ndarray:
    """The measurements in :data:`FEATURES` order, for one ``(time, freq)`` waterfall."""
    power = np.asarray(power, dtype=np.float32)
    if power.ndim == 3 and power.shape[0] == 1:
        power = power[0]
    if power.ndim != 2 or min(power.shape) < 16:
        raise ValidationError("power", power.shape, "expected a (time, frequency) waterfall, at least 16x16")
    w = np.clip(power, *CLIP)
    n_time, n_freq = w.shape
    edge = n_freq // 8                      # the receiver filter darkens the outer eighths
    inner = w[:, edge:n_freq - edge]
    n_inner = inner.shape[1]
    centre = n_inner // 2
    half = max(n_inner // 20, 2)            # the centre band: +-5% of the band
    band = inner[:, centre - half:centre + half]

    profile = inner.mean(axis=0)
    peak = int(np.argmax(profile))
    prominence = float(profile[peak] - np.median(profile))
    width = float(np.mean(profile > np.median(profile) + prominence / 2)) if prominence > 0 else 0.0

    band_max = band.max(axis=1)
    inner_max = inner.max(axis=1)
    strong = inner_max > STRONG
    where = np.argmax(inner, axis=1)
    if strong.sum() >= 3:
        wander = float(np.std(where[strong]) / n_inner)
        steady = float(np.mean(np.abs(where[strong] - centre) <= half))
    else:
        wander, steady = 0.0, 0.0
    band_mean = band.mean(axis=1)

    return np.array([
        prominence,
        abs(peak - centre) / n_inner,
        width,
        float(band.mean() - inner.mean()),
        float(np.percentile(band, 99)),
        float(np.percentile(inner, 99.5)),
        float(np.mean(band_max > STRONG)),
        float(np.mean(strong)),
        float(np.std(band_mean)),
        wander,
        steady,
        float(np.std(inner.std(axis=1))),
    ], dtype=np.float32)


def feature_matrix(inputs: np.ndarray) -> np.ndarray:
    """:func:`waterfall_features` for every waterfall in ``(n, 1, time, freq)``."""
    inputs = np.asarray(inputs)
    if inputs.ndim not in (3, 4):
        raise ValidationError("inputs", inputs.shape, "expected (n, time, freq) or (n, 1, time, freq)")
    return np.stack([waterfall_features(x) for x in inputs]) if len(inputs) else np.empty((0, len(FEATURES)))
