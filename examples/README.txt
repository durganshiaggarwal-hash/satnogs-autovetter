EXAMPLE SIGNAL FILES - for the "Your own file" tab
==================================================

Open the app, click the "Your own file" tab, press "Choose file...",
pick one of these, set the Speed, and press "Analyse this file".

qpsk_250kbaud.csv
    A QPSK signal made by the project's simulator: +15 dB, 300 Hz Doppler.
    Two columns, I and Q, one sample per line.   Speed: 250 kbaud.

16qam_100kbaud.npy
    A 16QAM signal made by the simulator: +18 dB, -200 Hz Doppler.
    A NumPy file of complex numbers.             Speed: 100 kbaud.

rml24_8psk_recording_184700.npz
    A REAL recording, row 184,700 of RML24 (8PSK, +18 dB), exported by the
    project.  Its true type is stored inside, so the app also says whether
    the AI was right and checks the bits one by one.  (Speed is read from
    the file.)

test_QPSK_+8dB_250kbaud_doppler600.npz
    A TEST FILE made with "Save as test file" on the Simulated signal tab:
    the signal plus its answer key (true type QPSK and every bit that was sent).
    The app marks the AI right or wrong and checks every bit. Speed is read
    from the file.

test_RML24_184700_8PSK_+18dB_250kbaud.npz
    A TEST FILE made with "Save as test file" on the Real signal tab: RML24
    recording #184,700 with its true type. Its bit answer key is worked out by
    the Phase 3 reference decoder, exactly as on the Real signal tab.

test_QPSK_+8dB_250kbaud_doppler600_signal.csv  +  ..._bits.csv
    The same QPSK test file as TWO CSV FILES for Excel: the signal (I, Q) and the
    answer key (type, speed, then one bit per line). In "Your own file", choose
    the _signal.csv; the app finds the _bits.csv beside it and grades everything.
    Make your own pair the same way, or with "Save as test file" -> type
    "Two CSV files for Excel".

For qpsk_250kbaud.csv and 16qam_100kbaud.npy the app does not know the true answer (just like
a real unknown signal), so it shows what the AI says and how sure the receiver
is of each bit, but cannot mark the answer right or wrong.

Your own files: CSV/TXT with two columns (I, Q), or .npy; sampled at 1 MHz;
at least 2,000 samples.
