#!/usr/bin/env python3
"""Break a signal, then watch the receiver work out how to fix it.

Run it::

    python try_it.py

Then start changing things. Some suggestions are at the bottom.
"""

from satrecovery import SignalSimulator, ChannelConditions, ClassicalReceiver

sim = SignalSimulator(seed=1)
rx = ClassicalReceiver(pulse=sim.pulse)

# Break the signal in four different ways at once.
conditions = ChannelConditions(
    snr_db=14,                    # how strong the signal is against the noise
    freq_offset_hz=900,           # the satellite is moving, so the frequency shifts
    phase_offset_rad=0.9,         # nobody knows the transmitter's phase
    timing_offset=0.55,           # the sampling clock is misaligned
    fading="multipath_rician",    # the signal arrives by several paths at once
)

frame = sim.generate_frame("16QAM", conditions)
print("Sent  :", frame)

result = rx.recover(frame)
print("Got   :", result)

d = result.diagnostics
print()
print("What the receiver worked out on its own:")
print(f"  frequency offset : {d['coarse_freq_offset_hz']:+.0f} Hz"
      f"   (truth {conditions.freq_offset_hz:+.0f} Hz)")
print(f"  best sample point: {d['timing_offset_samples']:+.3f} samples")
print(f"  constellation fit: {d['evm_percent']:.1f}% error vector magnitude")
print()
print(f"Bits wrong : {result.bit_errors(frame.bits)} of {result.n_bits}")
print(f"Error rate : {result.ber(frame.bits):.2e}")
print(f"Receiver was less sure about the bits it got wrong: "
      f"{result.confidence_is_informative(frame.bits)}")

print()
print("Things to try:")
print("  snr_db=2             weak signal, errors appear")
print('  "64QAM" instead      six bits per symbol, far more fragile')
print("  freq_offset_hz=5000  a bigger Doppler shift")
print("  timing_offset=3.5    nearly half a symbol out")
print("  comment out a line   see which impairment mattered")
