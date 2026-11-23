#!/usr/bin/env python3
"""Experiment: can the Lab half's receiver decode a real SatNOGS recording?

    python scripts/decode_bpsk_audio.py satnogs_15491675.ogg --baud 9600

SatNOGS does not upload the raw signal (IQ); it keeps a mono, 48 kHz,
lossy (Ogg Vorbis) audio file.  For a BPSK satellite this script tries
to get the symbols back from that audio with the *original project's*
classical receiver (Phase 2), unchanged:

1. find the packet bursts (energy well above the noise floor);
2. turn the real audio into a complex signal (Hilbert transform);
3. find the carrier: squaring BPSK removes the +-1 modulation and leaves
   a pure tone at twice the carrier frequency;
4. shift to 0 Hz, low-pass, resample to 4 samples per symbol;
5. run the Lab receiver chain: frequency correction, matched filter,
   Gardner timing recovery, gain control, decision-directed PLL.

The same is done to a stretch of plain noise, so two clean clusters can
be told from wishful thinking.  Success is judged by the cluster
separation (and, as a bonus, by packet flag patterns in the bits).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satrecovery.core.signal import Signal  # noqa: E402
from satrecovery.modulation import get_modulation  # noqa: E402
from satrecovery.receivers.base import RecoveryContext  # noqa: E402
from satrecovery.receivers.classical import ClassicalReceiver  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPS = 4


def bursts(audio: np.ndarray, fs: int, block: float = 0.02) -> list[tuple[int, int]]:
    """(start, stop) sample ranges where the band is much louder than usual."""
    n = int(block * fs)
    blocks = len(audio) // n
    energy = np.square(audio[: blocks * n]).reshape(blocks, n).mean(axis=1)
    loud = energy > 4.0 * np.median(energy)
    out, start = [], None
    for i, flag in enumerate(np.append(loud, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= 3:                       # at least 60 ms
                out.append((start * n, i * n))
            start = None
    return sorted(out, key=lambda r: r[0] - r[1])   # longest first


def to_baseband(audio: np.ndarray, fs: int, baud: float) -> tuple[np.ndarray, float, int]:
    """Complex baseband at SPS samples per symbol, and the carrier frequency found."""
    analytic = hilbert(audio)
    squared = np.fft.rfft(np.real(analytic * analytic) * np.hanning(len(analytic)))
    freqs = np.fft.rfftfreq(len(analytic), 1 / fs)
    band = (freqs > 2 * 2000) & (freqs < 2 * 18000)
    carrier = float(freqs[band][np.argmax(np.abs(squared[band]))] / 2)
    t = np.arange(len(analytic)) / fs
    shifted = analytic * np.exp(-2j * np.pi * carrier * t)
    sos = butter(6, 0.7 * baud, fs=fs, output="sos")
    shifted = sosfiltfilt(sos, shifted.real) + 1j * sosfiltfilt(sos, shifted.imag)
    target = int(round(baud * SPS))
    from math import gcd

    g = gcd(target, fs)
    return resample_poly(shifted, target // g, fs // g), carrier, target


def lab_receiver(samples: np.ndarray, rate: int) -> np.ndarray:
    """The original project's classical receiver chain, BPSK, unchanged."""
    bpsk = get_modulation("BPSK")
    chain = ClassicalReceiver(modulation=bpsk, pilot_symbols=0).build_chain(SPS)
    return chain.apply(Signal(samples, rate), RecoveryContext(bpsk, SPS, sample_rate=rate)).samples


def separation(symbols: np.ndarray) -> float:
    """How far apart the two clusters are, in units of their spread (dB). Noise gives about 0-3 dB."""
    s = symbols[len(symbols) // 10:]                  # skip the loops settling
    s = s * np.exp(-1j * np.angle(np.mean(s ** 2)) / 2)   # put the clusters on the real axis
    i = np.abs(s.real)
    return float(10 * np.log10(np.mean(i) ** 2 / (np.var(i) + np.var(s.imag) + 1e-12)))


def flags(bits: np.ndarray) -> int:
    """HDLC flags (01111110) after NRZI decoding -- packet structure, if the bits are right."""
    nrzi = (bits[1:] == bits[:-1]).astype(int)        # NRZI: no change = 1
    pattern = np.array([0, 1, 1, 1, 1, 1, 1, 0])
    windows = np.lib.stride_tricks.sliding_window_view(nrzi, 8)
    return int(np.sum(np.all(windows == pattern, axis=1)))


def main(argv: list[str] | None = None) -> int:
    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio")
    parser.add_argument("--baud", type=float, default=9600)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "vetting"))
    args = parser.parse_args(argv)

    audio, fs = sf.read(args.audio)
    found = bursts(audio, fs)
    print(f"{len(audio) / fs:.0f} s of {fs} Hz audio; {len(found)} bursts found")
    if not found:
        return 1
    start, stop = found[0]
    margin = int(0.05 * fs)
    burst = audio[max(start - margin, 0):stop + margin]
    quiet_start = int(np.argmin([np.square(audio[i:i + len(burst)]).mean()
                                 for i in range(0, len(audio) - len(burst), len(burst))])) * len(burst)
    quiet = audio[quiet_start:quiet_start + len(burst)]

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, (name, segment) in zip(axes, (("packet burst", burst), ("plain noise", quiet))):
        baseband, carrier, rate = to_baseband(segment, fs, args.baud)
        symbols = lab_receiver(baseband, rate)
        sep = separation(symbols)
        bits = (symbols.real > 0).astype(int)
        results[name] = {"seconds": len(segment) / fs, "carrier_hz": carrier, "symbols": int(len(symbols)),
                         "separation_db": sep, "hdlc_flags": flags(bits),
                         "flags_by_chance": float(max(len(bits) - 8, 0) / 256)}
        print(f"{name:13} {len(segment) / fs:5.2f} s  carrier {carrier:7.0f} Hz  {len(symbols):6d} symbols  "
              f"cluster separation {sep:5.1f} dB  HDLC flags {results[name]['hdlc_flags']} "
              f"(chance ~{results[name]['flags_by_chance']:.0f})")
        s = symbols[len(symbols) // 10:]
        s = s * np.exp(-1j * np.angle(np.mean(s ** 2)) / 2)
        ax.plot(s.real, s.imag, ".", ms=1.5, alpha=0.4, color="#1b9e77" if name == "packet burst" else "#7b8594")
        ax.set(title=f"{name}: separation {sep:.1f} dB", xlabel="in phase", ylabel="quadrature", aspect="equal")
        lim = float(np.percentile(np.abs(s), 99)) * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    fig.suptitle(f"Lab receiver on a real SatNOGS BPSK recording ({Path(args.audio).name})", fontsize=10)
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "bpsk_audio_decode.png", dpi=120)
    (out / "bpsk_audio_decode.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {out / 'bpsk_audio_decode.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
