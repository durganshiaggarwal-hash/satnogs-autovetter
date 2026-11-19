"""Phase 8 -- the desktop application.

Start it with::

    python -m satrecovery.gui

Four tabs:

**Experiment**    damage a signal with sliders; watch each receiver recover it
**Step by step**  follow one signal through every stage of a receiver
**Benchmark**     many signals across many conditions, with uncertainty
**Models**        train the networks the hybrid receiver needs

The GUI computes nothing itself.  Every number on screen comes from the
Phase 7 backend, which it starts automatically if one is not running.

``theme``     colours, type, stylesheet -- the one place the look is set
``client``    the interface to the backend, and its HTTP implementation
``server``    starting a backend inside the application when needed
``tasks``     background calls and job polling, so the window never freezes
``widgets``   sliders, plots and tables the pages are built from
``pages``     the four tabs
``window``    the main window
"""
