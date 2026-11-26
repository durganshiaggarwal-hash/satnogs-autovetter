# Development log: the signal-recovery core (Phases 1-8)

This is the phase-by-phase engineering log of the Lab half (RML24 signal
recovery), kept as written during development. Test counts and status lines
below are historical; the current overview is in the top-level README.

An object-oriented Python system for comparing **classical**, **AI-based**
and **hybrid** methods of recovering the transmitted bits from a distorted
satellite signal — and for testing whether a receiver can tell when its own
output is probably wrong.

Built around the RML24 satellite TT&C dataset
([Scientific Data, 2026](https://www.nature.com/articles/s41597-026-07182-7)).


---

## What the project does

Most published work on RML24 asks *which modulation is this?* — a
classification question. This project asks two harder ones:

1. **How reliably can the original bits be recovered?** Measured as bit
   error rate against the dataset's own bit labels, not as classification
   accuracy.
2. **Does the receiver know when it is wrong?** Measured with calibration
   error, risk–coverage curves and error-detection AUROC.

The final deliverable is an interactive application where a user changes
the signal conditions, compares recovery methods side by side, and sees
how each performs.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows  (macOS/Linux: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU; skip on CPU-only
pip install -r requirements.txt
python -m pytest -q                    # about 650 tests
python -m satrecovery.gui              # the desktop app (starts its own backend)
```

Other entry points: `python -m satrecovery.backend` (API at http://127.0.0.1:8000/docs)
and `python scripts/demo_phase1.py` … `demo_phase7.py`.

```python
from satrecovery import SignalSimulator, ChannelConditions, get_modulation

simulator = SignalSimulator(seed=42)
scheme = get_modulation("16QAM")

frame = simulator.generate_frame(
    "16QAM",
    ChannelConditions(snr_db=8.0, fading="multipath_rician", freq_offset_hz=200.0),
)

print(frame)
# Frame(16QAM, 2048 samples, 1024 bits, SNR 8 dB, multipath rician (K=10 dB),
#       df 200 Hz, source='simulated')
```

---

## Where the design concepts live

Each of the four pillars is demonstrated deliberately, in a place where it
earns its keep rather than as decoration.

### Encapsulation

| Where | What it guarantees |
|---|---|
| `core/signal.py` | `Signal` is **immutable**. `samples` returns a read-only view and there is no setter; `with_samples()` returns a new object instead. The constructor copies its input, so the object cannot be changed through an array the caller still holds. |
| `core/conditions.py` | `ChannelConditions` is **mutable but always valid**. Every field is a property whose setter validates, so `conditions.snr_db = 500` raises rather than corrupting state. `snr_linear` is derived, so it can never disagree with `snr_db`. |
| `core/frame.py` | The ground-truth `bits` are read-only — the code that is supposed to be *guessing* them cannot edit them. |
| `__slots__` | Used throughout `core` so the set of fields belongs to the class, not to whoever touches an instance. |

Why immutability matters here specifically: one frame is fed to several
receivers in turn. If any receiver could write into the shared sample
array, the next one would be measured on different data and the comparison
the project exists to make would be silently invalid.

### Abstraction

| Abstract base class | Contract | Implementations |
|---|---|---|
| `Modulation` | `_build_constellation()` | `BPSK`, `QPSK`, `PSK8`, `QAM16`, `QAM64`, `APSK16` |
| `ChannelEffect` | `_process(samples, rate, rng)` | 5 impairments + `ChannelPipeline` |
| `PulseShape` | `_build_taps()` | `RootRaisedCosine`, `Rectangular` |

Each is a **Template Method**: the base class owns the algorithm and the
subclass fills in one step. `Modulation` implements `map_bits` and
`demap_symbols` once, for any constellation; a subclass supplies only its
geometry. Python refuses to instantiate a subclass that forgets, so the
contract is enforced by the language rather than by a comment.

### Inheritance

```
Modulation (abstract)
├── PhaseShiftKeying ──── BPSK, QPSK, PSK8
├── QuadratureAmplitudeModulation ──── QAM16, QAM64
└── APSK16

ChannelEffect (abstract)
├── AdditiveWhiteGaussianNoise
├── FlatRicianFading ──── MultipathRicianFading
├── CarrierFrequencyOffset
├── CarrierPhaseOffset
└── ChannelPipeline          (a chain of effects, which is itself an effect)

SatRecoveryError
├── ConfigurationError ──── ValidationError
├── ModulationError ──── UnknownModulationError, BitLengthError
├── SignalError, ChannelError, ReceiverError ...
```

Three-level chains where there is genuine shared implementation, and
deliberately *not* where there is not: QAM does not inherit from PSK,
because a grid and a circle share no code. `MultipathRicianFading` **does**
inherit from `FlatRicianFading`, because it reuses the coefficient
mathematics exactly — and a one-tap multipath channel really is a flat
channel, which `test_single_tap_multipath_equals_flat_fading` checks rather
than merely asserting in a comment.

### Polymorphism

```python
# No branching on type anywhere. This is what the benchmark runner does.
for scheme in ModulationRegistry.all_schemes():
    frame = simulator.generate_frame(scheme, conditions)
    recovered = scheme.demap_symbols(symbols)
```

- **Method overriding**: every scheme builds a different constellation;
  `APSK16.theoretical_ber()` returns `None` where others return a curve,
  and the GUI adapts without knowing why.
- **Operator overloading**: `len()`, `[]`, `==`, `iter()` on `Signal`,
  `Modulation` and `ChannelPipeline`; `ChannelEffect.__call__` lets an
  effect be used wherever a function is expected.
- **Polymorphic input**: `generate_frame()` accepts a name *or* an object,
  because the API layer has a string and internal code has an object.

### Design patterns

| Pattern | Class | Why |
|---|---|---|
| Template Method | `Modulation`, `ChannelEffect`, `PulseShape` | Base owns the algorithm, subclass fills one step |
| Strategy | `ChannelEffect`, `PulseShape` | Interchangeable impairments and filters |
| **Composite** | `ChannelPipeline` | *Is* an effect **and** *has* a list of effects, so a chain is usable anywhere one effect is |
| Factory | `ModulationRegistry`, `ChannelFactory` | Turns a GUI string into objects; holds the correct stage ordering in one place |
| Registry | `ModulationRegistry` | Adding a scheme registers one class; no existing code changes |
| Dependency injection | `SignalSimulator(pulse=...)` | Written against `PulseShape`, so it works with a filter that does not exist yet |

---

## Package layout

```
satrecovery/
├── core/          Signal, Frame, ChannelConditions, exception hierarchy
├── modulation/    Modulation ABC + 6 schemes + registry
├── channel/       ChannelEffect ABC + 6 impairments + pipeline + factory
├── simulation/    PulseShape ABC + RRC/Rectangular + SignalSimulator
├── data/          RML24: layout, Recording, RecordingArchive ABC (HDF5 file
│                  or sample), ReferenceDecoder, RecordedSource
├── receivers/     Receiver ABC, ReceiverStage ABC + chain, 6 DSP stages,
│                  demappers, ClassicalReceiver, ReceiverBank, RecoveryResult
├── learning/      TrainableModel ABC, feature extractors, dataset builders,
│                  Trainer + callbacks, MLPDemapper, CNNDemodulator, forest
├── confidence/    ConfidenceEstimator ABC, estimators, calibrators, metrics
├── evaluation/    Scenario, ImpairmentProfile, FrameSource ABC (simulated,
│                  router), BenchmarkRunner, statistics, ResultStore
├── backend/       FastAPI app, services, job manager, model store
└── gui/           PySide6 window, four pages, animated widgets

tests/             about 650 tests
scripts/           demo_phase1.py ... demo_phase8, inspect_rml24.py, probe_rml24.py
outputs/           generated figures
```

---

## Why a simulator as well as RML24

RML24 is a fixed recording: its frames were captured under the conditions
its authors chose, and nothing can change them afterwards. The application
has to let a user **turn the noise up and watch what happens**, which a
recording cannot do. So the two sources play different roles:

- the **simulator** drives the interactive experiments and generates
  unlimited labelled training data on a laptop;
- **RML24** is the realistic benchmark the trained models are finally
  measured against, including hardware imperfections no simulator
  reproduces.

Both produce `Frame` objects, so every receiver, metric and plot works with
either without modification. The simulator's defaults deliberately match
RML24's published parameters — 1 MHz sample rate, 2048-sample frames,
root-raised-cosine with roll-off 0.35, and the same Rician multipath tap
profile `[0, −0.97, −5.23] dB`.

---

## Correctness: how we know the baseline is right

Everything the project will later claim — that a learned demapper beats a
classical one, that a hybrid beats both — is meaningless unless the
classical baseline is correct in the first place. So the test suite checks
the physics against published theory, not just that the code runs.

**`test_ber_matches_textbook_theory`** measures bit error rate at the
symbol level and compares it with the closed-form expressions from the
communications literature.

**`test_full_chain_ber_matches_theory`** does the same through the entire
chain: random bits → Gray mapping → root-raised-cosine shaping → AWGN →
matched filter → symbol-rate sampling → nearest-neighbour demapping.

Measured against theory, full chain:

| Scheme | Channel SNR | Eb/N0 | Measured BER | Theory |
|---|---|---|---|---|
| BPSK | −11.0 dB | −1.97 dB | 1.268e−01 | 1.298e−01 |
| QPSK | −8.0 dB | −1.98 dB | 1.293e−01 | 1.301e−01 |
| 16QAM | −1.0 dB | 2.01 dB | 9.825e−02 | 9.735e−02 |

Passing these means the modulation, the Gray mapping, the pulse shaping,
the noise calibration, the timing alignment and the link-budget arithmetic
are all simultaneously right.

### One conversion worth understanding

`ChannelConditions.snr_db` — the number on the GUI slider — is **not**
Eb/N0. They differ by two factors:

- **Processing gain.** The matched filter integrates each symbol over all
  `samples_per_symbol` samples while rejecting out-of-band noise, worth
  `10·log₁₀(sps)` dB — about 9 dB at 8 samples per symbol.
- **Bits per symbol.** Symbol energy is shared among the bits it carries,
  costing `10·log₁₀(k)` dB.

Forgetting the processing gain would make a perfectly correct receiver look
nine decibels too good, and nothing in the plot would look wrong. The
conversion therefore lives in exactly one place:
`SignalSimulator.effective_ebn0_db()`.

---

## A bug the tests caught

`test_constructor_copies_its_input` failed on first run. `Signal` used
`np.ascontiguousarray`, which **skips the copy** when the input already has
the right dtype — so the "immutable" `Signal` was sharing memory with the
caller's array, and a caller editing their own array would silently change
it. Fixed by forcing a copy in `Signal` and `Frame`.

Worth recording because it is the point of writing tests as statements
about the design: immutability has to be *owned memory*, not just a
read-only view over somebody else's.

---

## Demo output

`python3 scripts/demo_phase1.py` writes two figures.

**`outputs/constellations.png`** — all six schemes at 12 dB. More bits per
symbol packs the points closer together, so the same noise causes more
errors: at this signal-to-noise ratio everything up to 16QAM is error-free
while 64QAM already shows a bit error rate of 3.7e−03.

**`outputs/impairments.png`** — 16QAM under each impairment. Each one
distorts the constellation in its own recognisable way. The frequency
offset panel is the striking one: the constellation spins into rings and
the bit error rate hits 0.43, barely better than guessing. That is why
carrier recovery is the first thing the classical receiver does in Phase 2,
and the main thing a learned demodulator has to discover for itself.

---

## Phase 2: the classical receiver

Six repair stages and a demapper, built entirely from signal-processing
mathematics with nothing learned from data. Every estimate is made blind —
the receiver is never told the frequency, the timing or the phase.

| Stage | Undoes | How |
|---|---|---|
| `MatchedFilterStage` | Noise outside the signal's band | Filter with a copy of the transmit pulse |
| `CoarseFrequencyCorrection` | Doppler shift | Raise to the constellation's symmetry power; the data cancels and a tone survives |
| `GardnerTimingRecovery` | Sampling at the wrong instant | Find where the averaged Gardner error crosses zero |
| `AutomaticGainControl` | Fading having changed the signal's size | Rescale to unit power |
| `DecisionDirectedPLL` | Residual rotation | Measure the angle to the nearest constellation point and close a loop |
| `LMSEqualizer` | Echoes smearing symbols together | Short adaptive filter, trained on the preamble |

`ReceiverStage` and `ReceiverChain` mirror Phase 1's `ChannelEffect` and
`ChannelPipeline` exactly — same abstract base plus Composite shape,
opposite purpose. One breaks a signal, the other repairs it. That symmetry
is what makes Phase 5 cheap: a hybrid receiver is this chain with the
demapper swapped for a neural network.

### Measured against theory

The headline result. Full chain, blind estimation, bit error rate versus the
published formulas:

| Scheme | Eb/N0 | Measured | Theory | Implementation loss |
|---|---|---|---|---|
| BPSK | 3.03 dB | 2.50e−02 | 2.25e−02 | 0.19 dB |
| BPSK | 9.03 dB | 3.10e−05 | 3.17e−05 | −0.01 dB |
| QPSK | 1.02 dB | 5.51e−02 | 5.59e−02 | −0.04 dB |
| QPSK | 7.02 dB | 6.69e−04 | 7.53e−04 | −0.09 dB |
| 16QAM | 5.01 dB | 4.56e−02 | 4.17e−02 | 0.24 dB |
| 16QAM | 11.01 dB | 5.66e−04 | 5.57e−04 | 0.01 dB |
| 64QAM | 11.25 dB | 1.53e−02 | 1.49e−02 | 0.06 dB |
| 64QAM | 17.25 dB | 3.82e−05 | 2.87e−05 | 0.16 dB |

Under half a decibel across the normal operating range, which is in the
region a real modem achieves. Negative values are measurement noise, not
a receiver beating physics.

### Four things that had to be fixed, and what they taught

**The blind frequency estimator is destructive when it fails.** Raising a
signal to its symmetry power cancels the data and leaves a tone — but it
raises the *noise* to that power too. Below about −5 dB the peak of the
spectrum is just the largest noise spike, and acting on it spins a
constellation that was standing still. Measured on QPSK frames with no
offset at all, it returned estimates of several kilohertz.

No threshold separates good estimates from bad ones reliably: the obvious
detection statistics are not even frame-length independent. So the receiver
runs the chain **both ways** and keeps whichever produced the tighter
constellation, judged by error vector magnitude. That measurement uses no
knowledge of what was transmitted, so it is a legitimate thing for a real
receiver to do.

**16APSK breaks the method outright.** Its two rings are 4-fold and
12-fold symmetric, so no single power collapses both — raising to the
fourth leaves four separate angles rather than one. `Modulation` now
*computes* whether the trick applies to it, by measuring how tightly the
angles cluster after the power. The separation is clean: PSK 1.00,
QAM 0.45–0.52, 16APSK 0.005. DVB-S2 receivers hit the same wall and use
pilot-aided estimation for APSK instead.

**A tracking loop was the wrong tool for the timing.** A loop sees one
noisy error per symbol and must filter over time, so at low signal-to-noise
it wanders — measured, the timing stage was the *entire* source of BPSK's
implementation loss, costing up to 3 dB and varying wildly between frames.
Working on a whole frame there is no need to filter over time: average the
error across every symbol at once and find where it crosses zero. Worst-case
timing error fell from 6.6 samples to 0.31.

Along the way the loop also turned out to be a double integrator, which
only settles by overshooting and with wide gains never settles at all.
Both versions are kept — `mode="tracking"` is what a streaming receiver
must do, and the comparison is the point.

**The equaliser does not help on this channel, and that is the result.**
RML24's echoes arrive one *sample* apart — an eighth of a symbol — and the
matched filter already absorbs them, so the equaliser finds no inter-symbol
interference and merely adapts against noise. Switched off by default.
Spacing the echoes a whole symbol apart, it earns its keep properly:
bit error rate 2.6e−01 → 8.8e−02. It can also diverge above a critical step
size, so it now checks its own tap energy and falls back to passing the
signal through.

### Why the preamble is not cheating

A constellation has rotational symmetry: turn QPSK by 90 degrees and it is
indistinguishable from where it started. So no receiver, however clever,
can work out which way up it is from the signal alone — the information is
not there to be found. Every real link solves this by beginning each frame
with a short known pattern. This receiver uses 16 symbols and records
`n_pilot_bits` in the result so the evaluation can exclude them.
`test_carrier_loop_resolves_the_phase_ambiguity` demonstrates the failure
without one.

### Demo output

`python3 scripts/demo_phase2.py` writes two figures.

**`outputs/receiver_stages.png`** — one 16QAM frame with everything wrong
at once (14 dB, echoes, 900 Hz of Doppler, an unknown phase, a misaligned
clock), shown at the output of every stage. It arrives as a shapeless blob
and leaves as a clean grid with zero bit errors. The coarse stage estimated
+899 Hz against a true +900.

**`outputs/ber_curves.png`** — the waterfall curves, measured against
theory over five orders of magnitude. BPSK and QPSK land on top of each
other, which is the classic result: QPSK carries twice the data for the
same energy per bit.

---

## Phase 3: real satellite recordings (RML24)

```
python scripts/demo_phase3.py "C:/path/to/RML24_IQdata"      # the full 20 GB file, or its folder
python scripts/demo_phase3.py rml24_sample.npz --quick       # the 2.5 MB sample, no big download
python -m satrecovery.gui --rml24 "C:/path/to/RML24_IQdata"  # the app, with "Real recordings" enabled
```

### The problem: the dataset has no bits

The published `RML24_IQdata.h5` holds 1,323,000 recordings with their
modulation, SNR label and symbol rate — but **not the bits that were
sent**. `Bit_len` is 0 for every row, and 200 random rows of `Bit_data`
were all zeros (`scripts/probe_rml24.py`). Without bits, no bit error
rate can be measured, and measuring bit error rates is the point of this
project.

### The solution: decode clean recordings very carefully, then add noise

1. Take the **cleanest** recordings (labelled 20, 18, 16 dB).
2. Decode each with `ReferenceDecoder` — offline, looking at the whole
   recording at once, smoothing forwards *and* backwards in time, and
   knowing the modulation. These are privileges no receiver gets, and
   the decoder shares no code path with the receivers it grades, so a
   receiver is never graded against itself.
3. **Refuse what cannot be vouched for.** Every recording gets an
   estimated symbol error rate; above 1 in 10,000 it is rejected.
   Individual symbols too far from any point are marked *unreadable* and
   never graded; a recording with more than 3 % of them is rejected.
4. Add white Gaussian noise ourselves, by a measured amount, to reach
   each test SNR — on **the simulator's scale**, so "6 dB" means the same
   energy per symbol for a real frame and a simulated one.

Everything else about the frame is real: the pulse distortion, carrier
error, drifting gain and phase, and the glitches all come from the
hardware and the air. Only the noise is ours, and noise is the one thing
the simulator already models exactly.

**How we know the decoder is right**, given that RML24 cannot tell us:
the tests build recordings from the simulator, where every bit is known,
and give them RML24's problems — cut from a stream at a random sample,
600 Hz off, strength drifting 4.5-fold, a DC offset. The decoder must
reproduce the transmitted symbols exactly. It does, for BPSK, QPSK, 8PSK
and 16QAM at 10, 4 and 2 samples per symbol (`tests/test_data.py`). Two
of those tests failed first and found real bugs in the decoder (below).

### What the real data showed

| Finding | Detail |
|---|---|
| Which schemes can be graded | BPSK, QPSK and 8PSK: every clean recording accepted, EVM 6–9 %. 16QAM, 64QAM and 16APSK: rejected — 20 dB is not clean enough to vouch for every symbol of a dense constellation (64QAM: half its symbols unreadable). A finding, not a failure. |
| A periodic glitch | At 250 kbaud, one symbol in every **92** is corrupted — in most recordings at the *same* positions (87, 179, 271, 363, 455). Something that repeats at fixed places within each 2048-sample window is most likely an artefact of how the dataset was produced rather than of the radio channel. Left ungraded. (`glitches.png`) |
| Drifting strength | Within one 2 ms recording the signal strength drifts by up to **7×**. The simulator's channels never do this. |
| QPSK: real ≈ simulated | Within about 1 dB of the simulator at every SNR. |
| BPSK: real is worse | Mostly one recording whose strength ramps from 0.27 to 2.0: its first symbols sit 13 dB below its average. That is real fading, the same for every receiver. |
| 8PSK: real is 2–4 dB worse | The blind frequency estimate is harder on real 8PSK; see the fixes below. |
| Classical vs hybrid | **They tie on real data**, within the 95 % intervals, as on simulated data. The hybrid was trained only on simulated signals and transfers to real recordings with no measurable extra loss — the sim-to-real gap is the same for both. |

### Five bugs the real data found

Real data's most valuable output was not a curve; it was bugs the
simulator could never reveal, because the simulator and the receivers
shared the same assumptions.

1. **Our 16APSK was not DVB-S2.** The outer ring sat at 0°+30°k; the
   standard puts it at 15°+30°k. Decoded against our layout, the real
   recordings' outer ring sat exactly between our points. Fixed in
   `APSK16`; a test pins the layout.
2. **8PSK frequency estimation raised the magnitude to the 8th power.**
   The pulse's overshoots between symbols then decided the estimate.
   Fixed with Viterbi & Viterbi weighting (magnitude to at most the 4th
   power): wrong estimates on simulated 8PSK with Doppler fell from 12–22
   in 30 frames to none, and 8PSK on a Doppler channel went from a bit
   error rate of 0.49 to 0.0005 at 6 dB — **on simulated data too**.
3. **The symbol clock showed through.** At 10 samples per symbol the
   strongest spectral line was the symbol-rate line, not the carrier
   ("10 kHz" was the 100 kHz clock divided by eight). The search is now
   limited to where the carrier line can be.
4. **The receiver chose its frequency branch by EVM**, and a *spinning*
   8PSK constellation scores a lower EVM than a locked noisy one (every
   point on a ring is near some target). It now chooses by **lock
   quality** — whether the constellation stands still — and, when the
   first hypothesis does not lock, tries the next few spectral peaks
   (multi-hypothesis acquisition). Real 8PSK at 10 dB: 1.6e-2 → 1.8e-4.
5. **The decoder's own bugs**, found by the known-bits tests: removing
   the DC offset by subtracting the mean also removed the *data's* mean
   (8PSK EVM 11.7 % instead of 3.5 % on a clean test recording); and a
   per-symbol phase tracker occasionally slipped a quarter turn and back,
   putting nine wrong symbols in an *accepted* reference. Both fixed:
   DC is now estimated from what is left after the decisions, and the
   phase is one estimate per recording plus decision-directed tracking,
   which cannot slip.

### Design in this phase

| Concept | Where |
|---|---|
| Abstraction, two implementations | `RecordingArchive` (ABC): `RML24File` reads the 20 GB HDF5 a row at a time; `SampleArchive` reads a few-MB `.npz`. Everything downstream holds an archive and cannot tell which. `open_archive()` is a factory choosing between them. |
| Liskov substitution | `RecordedSource` stands in for `SimulatedSource` anywhere. The Phase 6 runner, statistics and database run on real data **without a line changed** — the promise Phase 6's `FrameSource` was written to keep. Its one difference is declared through the interface: it `supports()` only the `recorded` profile. |
| Composite | `SourceRouter` *is* a `FrameSource` and *holds* several, so one benchmark can mix simulated profiles and real recordings. |
| Encapsulation | `Recording` and `Reference` are immutable value objects with read-only arrays. Frames gained `graded_symbols`, and the `Receiver.recover` template method applies it — once, for every receiver — so no receiver can forget to, and none is graded on different symbols. |
| Value object vs entity | A `Recording` is deliberately *not* a `Frame`: a frame is labelled with bits, a recording is not, and making one from the other needs the reference decoder. Two classes keep that step from being skipped by accident. |
| A bug worth mentioning | `ReceiverBank` wraps other receivers, so the graded-symbol selection ran twice — once in the member, once in the bank. Found by the backend test that mixes both sources. |

### The app

Start it with `--rml24 PATH`, and the Benchmark tab's channel list gains
**Real recordings**. Without it, the chip is greyed out and its tooltip
says how to enable it. Scenarios real data cannot produce (16QAM, 64QAM,
16APSK, or an SNR above a recording's own) are skipped and listed, never
silently replaced by simulated frames.

---

## Phase 4: the learned receivers

Three models, and a result for each.

| Model | What it replaces | Size |
|---|---|---|
| `ForestModulationClassifier` | Nothing — it answers "which scheme is this?" | 200 trees on 18 features |
| `MLPDemapper` | The final decision only; the Phase 2 chain stays | ~11k weights |
| `CNNDemodulator` | The entire chain — raw samples straight to bits | ~44k weights |

### The hybrid receiver cost ten lines

`MLPDemapper` inherits from **two** bases:

```python
class MLPDemapper(Demapper, TrainableModel)
```

That says exactly two things: it satisfies the demapper interface Phase 2
defined, and it must be trained first. Because the first half holds, it
drops into the existing receiver with no adapter:

```python
HybridReceiver(trained_mlp)   # = ClassicalReceiver with a learned last step
```

Every stage before the demapper — matched filter, frequency estimation,
timing, carrier loop — carries on untouched. The project's headline
contribution is nearly free because Phase 2 defined `Demapper` as an
interface instead of writing the nearest-neighbour rule inline.

### Finding 1: on plain noise, learning cannot win — and does not

| Signal/noise | Classical | Learned |
|---|---|---|
| −2 dB | 1.315e−01 | 1.315e−01 |
| 2 dB | 4.284e−02 | 4.620e−02 |
| 6 dB | 6.384e−03 | 7.140e−03 |

This is the correct result, not a disappointing one. Picking the nearest
constellation point is *provably optimal* when the only damage is
Gaussian noise, so the best any method can do is tie. A learned demapper
that appeared to beat it would mean a bug somewhere.

### Finding 2: on interference, it wins — and we can prove why

Echoes arriving a whole symbol apart, which smears each symbol into the
next. The nearest-neighbour rule cannot undo that by construction: it
looks at one symbol and never at its neighbours.

| Signal/noise | Classical | Learned, 1 symbol | Learned, 7 symbols |
|---|---|---|---|
| 0 dB | 2.993e−01 | 2.978e−01 | 2.132e−01 |
| 4 dB | 2.701e−01 | 2.707e−01 | 1.404e−01 |
| 8 dB | 2.757e−01 | 2.764e−01 | 1.335e−01 |
| 12 dB | 2.514e−01 | 2.527e−01 | 9.193e−02 |
| 16 dB | 2.480e−01 | 2.463e−01 | **6.445e−02** |

The middle column is the control, and it is what makes this credible.
The *same network*, shown only one symbol — the same information the
classical rule gets — ties with classical to three significant figures.
Shown seven symbols, it wins by up to 3.8×. So the gain comes from the
extra context, not from the network being clever.

Note also that the classical curve is nearly **flat**: more signal power
does not help, because the damage is not noise. That flatness is what an
error floor looks like, and it is exactly the situation where a learned
method has room to work.

### Finding 3: two places learning fell over

**A classifier trained on clean signals collapsed on impaired ones.**

| Trained on | Tested clean | Tested impaired |
|---|---|---|
| clean only | 85.0% | **35.8%** |
| impaired | 71.7% | 70.4% |

With six schemes, guessing scores 16.7%. A clean-trained classifier at
35.8% is barely above that.

The diagnosis came from the forest reporting what it used. Trained on
clean signals its top feature was `abs_C40` — a cumulant, which is
rotation-invariant by construction. Trained on impaired signals it
switched to `phase_std`, a statistic a Doppler shift corrupts. It
learned a shortcut that does not survive contact with a moving
satellite. **That diagnosis was only possible because the baseline can
explain itself**, which is the argument for keeping one.

**The end-to-end network could not learn synchronisation.**

| Trained on | Bit error rate |
|---|---|
| already-aligned signal | 0.0006 |
| plus Doppler, phase and timing error | 0.1608 |

The same architecture, 282× worse when it also has to discover carrier
and timing recovery from examples. Phase 2 needed six designed stages
for that job, and a small network given a few hundred frames does not
rediscover them. Worth stating plainly: this is a limit we measured, not
a claim that it is impossible with more data and a bigger model.

### What this says about the project's question

The pattern across all three findings is consistent, and it is the case
for the hybrid. Mathematics wins wherever the problem has known
structure — synchronisation, and decisions under Gaussian noise. Learning
wins wherever the structure is not known in closed form, which here means
interference between symbols. Neither replaces the other, and the
combination is the one that uses each where it is strongest.

---

## Phase 5: does the receiver know when it is wrong?

Every earlier phase asked how *often* a receiver is right. This one asks
whether it can tell, without being shown the answer, which of its own
bits to distrust. Run it with:

```
python scripts/demo_phase5.py      # about 40 seconds, three figures
```

### Two questions, kept apart

| | asks | measured by |
|---|---|---|
| **Calibration** | When it says 90%, is it right 90% of the time? | `CalibrationReport` — ECE, overconfidence, Brier, NLL |
| **Ranking** | If we drop the least confident bits, do errors fall? | `RiskCoverageCurve` — ranking quality, coverage at a target error rate |

A receiver can pass one and fail the other, so the project never reports
one without the other. Both classes were validated against cases with
known answers before anything was built on them: an honest source scores
ECE 0.002, a perfect ranking scores exactly 1.0, a constant scores 0.008
rather than the 1.0 a naive sort would give it.

### Design in this phase

| Concept | Where |
|---|---|
| Abstraction + Template Method | `ConfidenceEstimator.estimate()` does all validation once; subclasses fill in `_scores()` |
| Three-level inheritance | `ConfidenceEstimator → SymbolQualityEstimator → EVMConfidence` — each level adds one idea |
| Composite + operator overloading | `LLRConfidence() \| EVMConfidence()` builds a `CombinedConfidence`, same spelling as the channel pipeline |
| Polymorphism | the demo loops over five estimators and three calibrators with no type checks |
| Encapsulation | reports are immutable value objects with read-only arrays; calibrators refuse to run unfitted |

### Findings

1. **The soft demapper is honest where it works and overconfident where
   it breaks — for a specific reason.** It estimates noise power from the
   signal itself. At 16QAM −4 dB that estimate is 0.25 of the truth,
   because symbols land nearer the *wrong* point and it measures the
   distance to that one. Overconfidence tracks the estimate point for
   point (+0.12 at −4 dB, 0.000 by +8 dB).
2. **The theory-backed estimator wins.** Ranking quality at 16QAM 0 dB:
   LLR 0.81, Margin 0.38, EVM 0.32. Averaging LLR with EVM gives 0.50 —
   a composite is only as good as its members.
3. **Calibration fixes honesty, never accuracy.** Temperature scaling
   took ECE from 0.124 to 0.031 at −4 dB without changing one bit, and
   left the ranking identical to the last digit. Histogram binning can
   lose a little resolution (ties) but cannot reorder.
4. **The failure that matters is per frame, not per bit.** 19% of frames
   on the impaired mixed-modulation channel lost carrier lock. Pooled,
   they hide inside a calibration number. Ranked per frame by mean
   confidence (the same `RiskCoverageCurve` class, one frame per
   decision), 74% of frames can be kept at a 1% failure rate.
5. **A claim that did not survive checking.** The first reading was that
   EVM cannot spot a lost frame (0.53 vs 0.98). Within one modulation it
   ties exactly (0.94 vs 0.94 on 64QAM). Its failure is only across
   modulations: a healthy 16QAM frame has a larger error vector than a
   dead 64QAM one. A probability has a common scale; a distance does not.

### Learned confidence (measured, not in the figures)

Training three MLP demappers plus one with dropout: on noise alone the
learned confidence ranks within 0.01 of the classical LLR. Under
impairments it is worse (0.21 vs 0.62 at 16QAM +6 dB). MC dropout made
calibration *worse* (ECE 0.007 → 0.033), and a three-model ensemble
improved ranking by only 0.003. The cause of the impaired-channel gap is
not settled: the learned confidences are visibly compressed (6% of bits
above 0.9999 against 80% for classical), but the error count in each
method's most confident half is nearly the same, so compression alone
does not explain it. **Answered in Phase 6 (finding 5):** the gap is
between frames, not within them — the classical demapper re-measures the
noise on every frame and the network does not.

---

## Phase 6: one fair benchmark, and what it found

```
python scripts/demo_phase6.py      # about 8 minutes; trains every model from scratch
```

### The framework

| Piece | Job | Design concept |
|---|---|---|
| `ImpairmentProfile`, `Scenario`, `ScenarioGrid` | describe exactly what was tested | value objects; `grid - train` is set arithmetic via operator overloading |
| `FrameSource` → `SimulatedSource` | where frames come from; RML24 plugs in as a second source | abstraction written for an implementation that doesn't exist yet |
| `BenchmarkRunner` | every receiver on **identical frames**, graded by **identical code**, nothing skipped silently | observer pattern for progress (console now, GUI later) |
| `ErrorRate`, `FrameBootstrap`, `Comparison` | intervals and paired comparisons | `ErrorRate + ErrorRate` pools counts, so `sum()` is correct |
| `SplitStrategy` → `RandomSplit`, `HoldOutSNR`, `HoldOutProfile` | train on part of a grid, test on the rest | strategy pattern; the base class makes train/test leakage impossible |
| `ResultStore` | SQLite, so the GUI and API never rerun anything | encapsulated resource, context manager |
| `ReceiverBank` | one hybrid per modulation, presented as one receiver | composite; dispatch on data, not on class |
| `NoiseNormalisedDemapper` | the fix from finding 4 | decorator pattern |

### Findings

1. **The benchmark found a bug the whole test suite had missed.** 16QAM with Doppler
   had a bit error rate floor near 1e-3 however strong the signal. All 53 errors at
   20 dB were symbol 0. The matched filter is circular and ran before frequency
   correction, so the Doppler rotation made the frame's end and start disagree, and
   the filter blended that into the first symbol. Correcting frequency first removed
   the floor (1.29e-3 → 0). A regression test now pins the stage order.
2. **Classical and hybrid tie on these channels.** Across 20 paired comparisons:
   hybrid decisively better in 1, classical in 3, no detectable difference in 16.
   About 1 in 20 would look decisive by chance alone, so the pattern is what counts.
   The one place classical is reliably ahead is clean 64QAM, where its rule is
   optimal and a network can at best match it.
3. **A fairness check changed a conclusion.** The first hybrid lost on 64QAM in every
   profile. It had run out of epochs on a third of the data; with enough data the
   loss shrank to the clean channel only. Reporting the first version would have
   been a result about our training budget, not about learned demappers.
4. **The hybrid generalises.** On held-out low SNRs, unseen Doppler and the unseen
   "full" profile, it stays within about 3% of classical. The classical front end
   removes the impairments before the network sees anything, which is exactly what
   Phase 4's end-to-end network lacked.
5. **Why the hybrid was worse at noticing lost frames, proved by ablation.**
   Triage on 64QAM: classical 0.970, hybrid 0.829. Hold the classical demapper's
   noise estimate fixed (one overridden method) and it falls to 0.835, the hybrid's
   level, with identical decisions. Wrapping the hybrid to rescale its odds by each
   frame's measured noise restores triage to 0.973, raises frames kept at a 1%
   failure rate from 37% to 45%, cuts calibration error from 0.064 to 0.038, and
   changes no bit. (Feeding noise estimates to learned demappers is a known idea;
   the contribution here is the measurement of why it matters.)
6. **Textbook intervals are badly overconfident here.** On 64QAM with impairments,
   a single lost frame out of 20 makes the frame-bootstrap interval about 10× wider
   than Wilson's. The default reported interval covers both.

---

## Phase 7: the backend

Everything the project can do, over HTTP. The GUI in Phase 8 is a client
of this; so is a browser, a script, or a Jupyter notebook.

```
pip install -r requirements.txt
python -m satrecovery.backend          # terminal 1: the server
python scripts/demo_phase7.py          # terminal 2: a client that drives it
```

Then open **http://127.0.0.1:8000/docs**: every endpoint, documented and
runnable from the browser. (That page loads its interface from a public
CDN, so it needs internet; the API itself does not.)

### Endpoints

| | Endpoint | Does |
|---|---|---|
| reference | `GET /api/health`, `/modulations`, `/profiles`, `/receivers` | what can be asked for, and which receivers are ready |
| interactive | `POST /api/experiments` | one frame, several receivers, **all on the same frame**; returns waveforms, constellations, per-symbol errors, confidence |
| | `POST /api/experiments/stages` | one frame followed through every receiver stage |
| long work | `POST /api/benchmarks` → `GET /api/jobs/{id}` | a grid as a background job with progress; saved to SQLite |
| | `POST /api/models/{mod}/train` | train a hybrid demapper as a job; it then appears as `hybrid` and `hybrid-noise` |
| results | `GET /api/runs`, `/api/runs/{id}/cells` | stored benchmark results, filterable |

### Design

```
app.py        HTTP: routes, status codes                 (thin)
services.py   use cases, no HTTP                         (testable directly)
satrecovery   the science, Phases 1-6                    (unchanged, unaware of HTTP)
```

| Concept | Where |
|---|---|
| Layered architecture | science never imports the backend; all 500 earlier tests run unchanged |
| Application factory + dependency injection | `create_app(settings)`: tests point a whole server at a temp folder |
| DTOs vs domain objects | `ConditionsIn` converts to `ChannelConditions`; validation rules live once, in the domain, and surface as HTTP 422 naming the field |
| Factory + registry | `ReceiverCatalog` maps names to builders; knows a hybrid needs a model |
| State machine, encapsulated | `Job`: queued → running → done/failed, transitions private to `JobManager` |
| Thread-safe resource | `ModelStore` owns the model folder and its cache behind a lock |

### Behaviour worth knowing

- **Safe by default.** Listens on `127.0.0.1` only: other computers cannot connect unless you ask.
- **Reproducible.** Every experiment returns the seed it used; send it back to get the identical frame.
- **Fails early.** A bad SNR, unknown receiver, oversized grid or untrained hybrid is refused *before* any work is queued, with the field named. Misspelt fields are refused, not silently ignored.
- **Strict JSON.** NaN and infinity become `null`, so any language can parse responses.
- The first training job after a reboot can take noticeably longer while PyTorch loads from disk.

## Phase 8: the desktop application

```
pip install -r requirements.txt
python -m satrecovery.gui
```

One command. If no backend is running it starts one itself, in the
background, and stops it when the window closes. (If you already started
`python -m satrecovery.backend`, the window uses that instead, so `/docs`
and the GUI can be open together.)

### The four tabs

| Tab | What you do there |
|---|---|
| **Experiment** | Drag sliders for signal strength, Doppler, phase, clock error and echoes; see the wave, the raw samples, and every receiver's recovered constellation side by side, all on the **same signal**. Wrong symbols are red crosses. "Keep this signal" holds the bits and noise fixed so a change comes only from the slider you moved. |
| **Step by step** | Follow one signal through the six receiver stages. Each stage explains itself with the receiver's own estimates against your settings: "it measured +707 Hz; you set +750". |
| **Benchmark** | Choose modulations, channels, strengths and receivers; it runs in the background with a progress bar, is saved, and plots error rate with 95% bands. |
| **Models** | Train the hybrid's neural network for each modulation, in the background. New models appear in the other tabs straight away. |

### Design choices

- **The GUI computes nothing.** Every number comes from the Phase 7 backend through one
  `BackendClient` interface, so the GUI cannot drift from the API.
- **Two palettes with two jobs.** The interface uses the team's Coolors palette — Peach
  Fuzz `#FFC9B5`, Powder Blush `#F7B1AB`, Desert Sand `#D8AA96`, Dusty Mauve `#807182`,
  Ash Grey `#C7D3BF`. Pastels are only ever *fills*; text uses a deep plum ink and a darker
  shade of each pastel, so every label stays readable. Each tab owns one palette colour
  (Experiment blush, Step by step sand, Benchmark sage, Models mauve), so you always know
  where you are. **Data** keeps its own colour-blind-checked palette: each receiver has one
  colour on every screen; red is reserved for errors and always comes with a text label. Where
  contrast is below 3:1, every plot names its receiver so identity never rests on colour alone.
- **Motion that explains, never decorates.** Buttons ripple on click, switches slide, the tab
  pill glides between tabs, pages slide in, constellation points glide to their new places
  when a slider moves (so you *see* what the slider did), numbers count up, and toasts confirm
  background work. An **Animations** switch in the header turns every animation off at once
  (`Motion.enabled`): `animate()` then jumps straight to the end, so there is no second code path.
- **Plain words first, technical terms on hover.** "Spots its mistakes" (ranking quality),
  "Honesty gap" (calibration error).
- **Never freezes.** Every request runs on a background thread; dragging a slider
  coalesces into one request rather than queuing dozens.
- Fonts (Manrope for text, IBM Plex Mono for numbers, both open licence) are bundled, so it
  looks the same on any laptop.

### Design in this phase

| Concept | Where |
|---|---|
| Abstraction that Qt breaks, restored | `Page` is abstract, but Qt widgets bypass Python's abstract check. Two fixes were needed, and the tests caught the second: `ABCMeta` must come **before** Qt's metaclass, and `Page.__init__` enforces the result itself. |
| Polymorphism | `MainWindow` adds and refreshes pages in one loop with no idea which is which |
| Inheritance for shared chrome | `PlotPanel` → `ConstellationPlot`, `WaveformPlot`, `BerCurvePlot` |
| Liskov substitution | `ToggleSwitch` and `Chip` *are* `QCheckBox`es, `PillButton` *is* a `QPushButton`: they only repaint themselves, so any code (and test) written for the plain widget still works |
| Encapsulation of animation | `animate()` and `Motion` are the only place durations and the on/off switch live; no widget touches a timer directly |
| Encapsulation of unit conversion | `LabeledSlider` hides integer ticks; degrees→radians happens in exactly one method |
| Observer | Qt signals: `JobWatcher` emits progress to whoever listens |
| Programming to an interface | Pages hold a `BackendClient`; tests hand them one pointed at a throwaway server |
| A race worth mentioning | Pressing *New signal* while a slider-drag run was still in flight got undone when that older run returned its seed. A test caught it; `_want_new` now holds until a seedless request has actually been sent. |
| A bug worth mentioning | A page method named `_loaded` was silently replaced by the base class's attribute of the same name — the *fragile base class* problem. Renamed, and the base's private names made distinct. |

---

## Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Core classes, modulation, channel, simulator | **done** |
| 2 | Classical receiver (matched filter, carrier recovery, timing, equaliser) | **done** |
| 3 | RML24: loader, reference labels, real-data benchmark | **done** |
| 4 | ML / ANN / DL models and training pipeline | **done** |
| 5 | Hybrid receiver and confidence estimators | **done** |
| 6 | Evaluation: metrics, held-out splits, results database | **done** (simulated and RML24) |
| 7 | FastAPI backend | **done** |
| 8 | PySide6 GUI | **done** |
| 9 | Documentation, UML diagrams, results write-up | |

---

## Team split

| Member | Owns |
|---|---|
| 1 | Signal side: core, modulation, channel, simulator, classical receiver, evaluation, backend API |
| 2 | Intelligence and interface: ML/ANN/DL models, hybrid receiver, uncertainty, GUI |

---

## Reference

Zhang, Y., Zang, B., Ji, H., Li, L., Chen, L., Li, S. *Cognitive Radio for
Satellite TT&C System: A General Dataset Using Software-defined Radio.*
Scientific Data (2026).
[Paper](https://www.nature.com/articles/s41597-026-07182-7) ·
[Dataset](https://zenodo.org/records/17800058) ·
[Code](https://github.com/yiwawa/RML24-Cognitive-Radio-for-Satellite)
