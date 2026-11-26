# Satellite Signal Analyser

A second app that sits beside the original one. You give it a satellite
signal; it tells you **what kind of signal it is** and **what message it
carries**, and shows the recovery as an animation.

Nothing in the original project was changed to add it: every file of the
original is byte-for-byte the same, and `python -m satrecovery.gui` still
opens the original app. Everything new is in `satrecovery/analysis/`,
`scripts/train_classifier.py`, `tests/test_analysis.py` and this file.

---

## What it does, in simple words

A satellite sends a message as a radio wave. On the way down the wave
gets damaged. The app answers two questions about a damaged signal:

1. **"What kind of signal is this?"** — answered by an **AI classifier**,
   a CNN (convolutional neural network) trained on RML24. It works like a
   brain-tumour classifier: in goes the data, out comes an answer with a
   confidence, e.g. *"QPSK, 97% sure"*. It does **not** read the message.
2. **"What message does it carry?"** — answered by a **receiver**, which
   repairs the signal and reads the bits. It uses the AI's answer to know
   *how* to read it, the way a translator needs to know the language
   first. The repair plays as an animation: a cloud of mixed-up coloured
   points slides, stage by stage, into clean clusters — each point
   coloured by the symbol it ends up being read as.

At the end it shows the recovered bits and, where the true bits are
known, **how many were wrong**.

```
Damaged signal --> AI classifier: "QPSK, 97% sure"
                        |
                        v
                  Receiver repairs it   (animated: 6 stages)
                        |
                        v
            Recovered message: 986 bits, 0 wrong
```

## The tabs

| Tab | You do | It shows |
|---|---|---|
| **Real signal** | Pick a real recording from RML24 (filter by type, quality, speed — or leave on *Any*). Its type stays hidden until the AI answers. | The AI's answer and whether it was right; the recovery animation; the bits, checked against the Phase 3 reference. |
| **Simulated signal** | Choose a type and damage it with sliders (strength, Doppler, echoes). | The same, checked against the bits the simulator sent. |
| **Your own file** | Choose a signal file (CSV/TXT with I and Q columns, a NumPy `.npy`, or a recording exported from RML24 as `.npz`) and its speed. Try the files in `examples/`. | The AI's answer and the recovered bits. A plain file does not say which bits were sent, so it is not checked; an RML24 export keeps its labels and is checked like any recording. |
| **History** | Nothing: every analysis on the tabs above is saved by itself. Type in the box to filter. | Every analysis, newest first. Double-click one to open it again, exactly as it was. |
| **Many at once** | Choose a type, quality, speed and how many recordings, then *Start*. | A progress bar, the running score, one summary, and a table you can *Save for Excel (.csv)*. |
| **How good is the AI?** | Press *Run the test*. | Accuracy at every signal quality, which types it mixes up, and every mistake (double-click one to open it). |
| **Compare models** | Train the models once with `scripts/compare_models.py` (or `Compare models.bat`), then open the tab. | One accuracy line per model on one chart; a table of accuracy, size, training time and speed; the selected model's mix-ups; *Use this model* switches the whole app to it. |

The top bar has four things that apply everywhere:

* **Dataset** — choose the RML24 folder in a normal folder window. The app remembers it, so next time
  `python -m satrecovery.analysis` is enough (no `--rml24`).
* **Dark** — switches the whole window to dark colours; remembered too.
* **Animations** — turns the moving pictures on or off.
* **?** — a short guide to every tab, and the technical words in plain language.

**Test files.** On the Real signal and Simulated signal tabs, **Save as test file** saves the
signal together with its answer key: its true type and the bits that were sent. Load it later in
*Your own file*, on any computer, and the AI is marked right or wrong and every bit is checked.
Simulated test files hold the exact bits the simulator sent. RML24 test files hold the recording
itself; RML24 stores no bits, so their answer key is the Phase 3 reference decoder's, trusted only
where it accepts the recording (otherwise only the true type is known). Two examples are in
`examples/`.

**CSV test files for Excel.** In the save window, choose *Two CSV files for Excel* to get
`..._signal.csv` (I, Q) and `..._bits.csv` (labelled lines `type`, `speed_baud`, …, then `bit`, then
one 0 or 1 per line). A `.npz` test file can be converted with *Export as CSV* on the Your own file
tab. To load a pair, choose the `_signal.csv`: the app finds the `_bits.csv` beside it (or use
*Choose bits file…*) and grades everything. A pair made by hand works too, as long as the number of
bits fits the signal; if not, the app says how many it expected.

**The warning badge.** Under the AI's answer, a badge says how far to trust it: **Confident**,
**Unsure** (the top answer is under 60%, or less than 25 points ahead of the next; or the signal is
very noisy; or the receiver is sure of under half the bits) or **Unfamiliar** (the signal is mostly
noise, below about −5 dB, or its width does not fit the chosen speed). The noise level is estimated
from the spectrum on the same scale as RML24's labels; on the simulator it lands within half a dB of
the truth, and on two real RML24 recordings labelled +18 and +20 dB it reads about +17. The warning
cannot catch every mistake -- the AI is sometimes wrong while sure -- so *How good is the AI?*
measures it on held-out recordings: what share of the AI's mistakes it flagged, what share of
correct answers it flagged needlessly, and how often the AI is right when the badge says Confident.
It also shows in History, Many at once (an "Answers flagged" count and column) and the PDF report.

On the first three tabs, **Save report (PDF)** saves the latest analysis as
a one- or two-page PDF: where the signal came from, what the AI said and
how sure it was, what the receiver read, whether the bits were checked,
and a picture of the recovery.

If the AI names the wrong type, the receiver still reads the signal *as
the AI said* — and the app says plainly that those bits cannot be
checked, because they were read in the wrong "language". You can see it
happen: the points do not land on the rings.

---

## How to run it

**Once: train the AI classifier** (a few minutes on the RTX GPU):

```
python scripts\train_classifier.py "C:\Users\chour\Downloads\RML24_IQdata"
```

It prints the accuracy after every pass, then tests on recordings it
never saw, and saves:

- the model to `satrecovery_data\models\classifier.pt`
- figures for the report to `outputs\classifier\`:
  `accuracy_vs_snr.png` (accuracy at every signal quality),
  `confusion.png` (which types get mixed up), `training.png`

**Every time: open the app**

```
python -m satrecovery.analysis --rml24 "C:\Users\chour\Downloads\RML24_IQdata"
```

Without `--rml24`, the Simulated tab still works.

---

## How RML24 is used

- **Training** — the CNN learns from RML24 recordings of the six schemes
  the receivers handle, at all 21 SNR labels and all three symbol rates.
- **Testing** — accuracy is measured on *other* RML24 recordings (the last
  ones of each group, never seen in training).
- **Analysing** — the Real tab picks from all 1.3 million recordings.
- **Checking** — recovered bits are compared with the bits the Phase 3
  reference decoder worked out for that recording.

The classifier is trained only on real recordings. On the Simulated tab
it meets signals from our simulator, which is a good test of whether
what it learned from real data carries over.

---

## Design of this addition

| Concept | Where |
|---|---|
| **Open/closed principle** | `create_analysis_app()` builds the original backend with the unchanged `create_app()`, then *adds* a router of new endpoints. The original code is closed for modification, open for extension. |
| **Inheritance** | `ModulationClassifier` is a `TrainableModel` (Phase 4): it inherits the trained/untrained guard and model saving, and reuses the Phase 4 `Trainer` loop unchanged. `AnalysisClient` inherits every method of the Phase 8 client and adds four. `AccuracyReporter` is a new `TrainingCallback`. |
| **Abstraction + polymorphism** | `SignalInput` (abstract) has three implementations, `RecordedInput`, `SimulatedInput` and `FileInput` (your own file). The `Analyser` treats all three the same. |
| **Template method** | `AnalysisPage` runs the whole screen — ask, show the AI's answer, play the animation, reveal the bits — and its three subclasses (`RealSignalPage`, `SimulatedSignalPage`, `FilePage`) supply only their controls and their request. |
| **Encapsulation** | `Classification` is an immutable answer object; the animation's whole state is one clock, so pause, replay and speed are just changes to it. |

## The double-click app

`packaging/` turns the analyser into a normal program: `Satellite Signal Analyser.exe` on
Windows, `Satellite Signal Analyser.app` on a Mac. Nobody needs Python or the terminal to use it.
How to build it is in `BUILD_APP.txt`.

## Files

```
satrecovery/analysis/
├── classifier.py   the CNN (ModulationClassifier) and its answer (Classification)
├── dataset.py      training/test sets from RML24, accuracy by SNR, confusion matrix
├── inputs.py       SignalInput: a real RML24 recording, or a simulated signal
├── upload.py       your own file: SignalFile reads it, FileInput analyses it
├── testfile.py     test files: a signal saved with its answer key (TestFile, TestFileInput)
├── csvpair.py      the same as two CSV files for Excel (CsvTestFile, CsvPairInput)
├── trust.py        the warning badge: TrustCheck gives a TrustVerdict (confident / unsure / unfamiliar)
├── history.py      AnalysisHistory: every analysis kept in a small database
├── preferences.py  Preferences: the dataset folder and Dark, remembered
├── analyser.py     step 1 classify, step 2 recover; packages the animation
├── api.py          the original backend + /api/analysis endpoints
├── __main__.py     python -m satrecovery.analysis
└── gui/            the window, the tabs, the animation widget, the PDF report (pdf.py),
                    History (history.py), Many at once (batch.py), the guide (help.py)
examples/           signal files to try in the "Your own file" tab
packaging/          the double-click app: see docs/BUILD_APP.txt
scripts/train_classifier.py
tests/test_analysis.py
```
