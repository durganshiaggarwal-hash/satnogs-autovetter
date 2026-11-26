"""How good is the AI?  Six kinds of model, compared fairly, and trained again on request.

The numbers come from ``scripts/compare_vetters.py``: every model sees the
same volunteer-checked recordings and is tested on ground stations it never
saw during training (whole stations held out, five folds).  This screen

* shows the model the app uses, in four plain numbers;
* lays all the models side by side in a table (best in each column marked)
  and in two charts (overall scores; how much each can decide alone);
* trains and compares them again, as a quick trial or a full run, in a
  separate process so the window stays responsive.  Runs started here are
  kept in ``satrecovery_data/satnogs/comparisons`` and never overwrite the
  report's own results.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .backdrop import StarryScreen, count_up

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "compare_vetters.py"
REPORT = ROOT / "outputs" / "vetting" / "model_comparison.json"

ORDER = ("logistic", "svm", "boosting", "forest", "cnn", "cnn+forest")
NAMES = {"logistic": "Logistic regression", "forest": "Random forest", "boosting": "Gradient boosting",
         "svm": "Support-vector machine", "cnn": "CNN (deep learning)", "cnn+forest": "CNN + forest"}
KINDS = {"logistic": "a straight-line rule on 12 measurements", "forest": "300 decision trees voting",
         "boosting": "trees built one after another, each fixing the last",
         "svm": "a curved boundary between the two kinds",
         "cnn": "a neural network that looks at the picture itself",
         "cnn+forest": "the CNN and the forest, their answers averaged"}
COLOURS = {"logistic": "#9a9a9a", "svm": "#c9a0dc", "boosting": "#f2d47c", "forest": "#a9bccf",
           "cnn": "#ee8a76", "cnn+forest": "#bfe08f"}
#: (key, column title, higher is better, how to show it)
COLUMNS = (("accuracy", "Right", True, "{:.1%}"),
           ("auroc", "Ranking", True, "{:.3f}"),
           ("decide_alone_5pc", "Alone 5%", True, "{:.0%}"),
           ("decide_alone_10pc", "Alone 10%", True, "{:.0%}"),
           ("ece", "Conf. error", False, "{:.1%}"),
           ("train_seconds", "Time", False, "{}"))
TIPS = {"Model": "Hover a model's name to see what kind of model it is.",
        "Right": "Share of recordings it gets right.",
        "Ranking": "Ranking score (AUROC): 1.0 would always rate a recording with a signal above one without; "
                   "0.5 is guessing.",
        "Alone 5%": "Share of recordings it can check without a person while keeping mistakes under 5%.",
        "Alone 10%": "Share of recordings it can check without a person while keeping mistakes under 10%.",
        "Conf. error": "Calibration error: how far its 'how sure' is from how often it is really right. Lower is better.",
        "Time": "Time to train it on all five groups of stations."}

QUICK = ("--limit", "600", "--epochs", "6", "--folds", "3")


def _label(text: str, name: str, wrap: bool = True) -> QLabel:
    widget = QLabel(text)
    widget.setObjectName(name)
    widget.setWordWrap(wrap)
    return widget


def _card() -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("card")
    inside = QVBoxLayout(card)
    inside.setContentsMargins(22, 18, 22, 18)
    inside.setSpacing(8)
    return card, inside


def _duration(seconds: float) -> str:
    if seconds < 1:
        return "<1 s"
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min"


class Run:
    """One saved comparison (a JSON file)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        data = json.loads(path.read_text(encoding="utf-8"))
        self.n, self.folds, self.limit = data.get("n", 0), data.get("folds", 0), data.get("limit", 0)
        self.results: dict = data.get("results", {})
        self.curves: dict = data.get("curves", {})
        self.finished = data.get("finished") or time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))

    @property
    def title(self) -> str:
        kind = "report run" if self.path == REPORT else ("quick trial" if self.limit else "full run")
        return f"{self.finished}  ·  {kind}  ·  {self.n:,} recordings, {self.folds} folds"

    def models(self) -> list[str]:
        return [m for m in ORDER if m in self.results]

    def best(self, key: str, higher: bool) -> str | None:
        models = self.models()
        if not models:
            return None
        values = [self.results[m][key] for m in models]
        return models[int(np.argmax(values) if higher else np.argmin(values))]


class ModelBoard(StarryScreen):
    """The 'How good is the AI?' screen."""

    back = Signal()

    def __init__(self, in_use: str = "cnn+forest", runs_dir: Path | None = None, parent=None) -> None:
        super().__init__(parent)
        self._in_use = in_use
        self._runs_dir = Path(runs_dir or ROOT / "satrecovery_data" / "satnogs" / "comparisons")
        self._runs: list[Run] = []
        self._run: Run | None = None
        self._focus: str | None = None
        self._process: QProcess | None = None
        self._log = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(60, 32, 60, 28)
        outer.setSpacing(10)
        top = QHBoxLayout()
        back = QPushButton("Back")
        back.setObjectName("quiet")
        back.clicked.connect(self.back.emit)
        top.addWidget(back)
        top.addSpacing(12)
        top.addWidget(_label("HOW GOOD IS THE AI", "hero", wrap=False), 1)
        top.addWidget(_label("RESULTS FROM", "caps", wrap=False))
        self.picker = QComboBox()
        self.picker.setMinimumWidth(420)
        self.picker.currentIndexChanged.connect(self._picked)
        top.addWidget(self.picker)
        outer.addLayout(top)
        self.lead = _label("", "lead")
        outer.addWidget(self.lead)
        outer.addSpacing(8)

        # The model in use, in four numbers
        self._stats = self._grid()
        self._stat_labels = []
        for column in range(4):
            card, inside = _card()
            card.setFixedHeight(132)
            value, text = _label("–", "stat"), _label("", "body")
            inside.addWidget(value)
            inside.addWidget(text)
            inside.addStretch(1)
            self._stat_labels.append((value, text))
            self._stats.addWidget(card, 0, column)
        outer.addLayout(self._stats)
        outer.addSpacing(6)

        # The models side by side: table (left) and charts (right), on the same grid
        board = self._grid()
        table_card, table_box = _card()
        row = QHBoxLayout()
        row.addWidget(_label("EVERY MODEL, SAME TEST", "caps", wrap=False))
        row.addStretch(1)
        row.addWidget(_label("click a model to pick it out in the chart", "body", wrap=False))
        table_box.addLayout(row)
        self.table = QTableWidget(0, len(COLUMNS) + 1)
        for column, title in enumerate(["Model"] + [c[1] for c in COLUMNS]):
            item = QTableWidgetItem(title)
            item.setToolTip(TIPS[title])
            self.table.setHorizontalHeaderItem(column, item)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setShowGrid(False)
        self.table.setMouseTracking(True)
        self.table.verticalHeader().setDefaultSectionSize(36)
        self.table.setMinimumHeight(6 * 36 + 46)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setFixedHeight(44)
        self.table.setStyleSheet("QHeaderView::section { letter-spacing: 0px; padding: 6px 2px; font-size: 13px; }")
        self.table.itemSelectionChanged.connect(self._row_picked)
        table_box.addWidget(self.table, 1)
        self.explain = _label("", "body")
        self.explain.setMinimumHeight(44)
        table_box.addWidget(self.explain)
        board.addWidget(table_card, 0, 0, 1, 2)

        chart_card, chart_box = _card()
        tabs = QHBoxLayout()
        tabs.setSpacing(0)
        tabs.addWidget(_label("THE CHART", "caps", wrap=False))
        tabs.addSpacing(18)
        self.chart_buttons = {}
        for name in ("Scores", "Deciding alone"):
            button = QPushButton(name)
            button.setObjectName("tab")
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, n=name: self.set_chart(n))
            self.chart_buttons[name] = button
            tabs.addWidget(button)
        tabs.addStretch(1)
        chart_box.addLayout(tabs)
        self.charts = QStackedWidget()
        self.scores_plot = self._plot()
        self.alone_plot = self._plot()
        self.charts.addWidget(self.scores_plot)
        self.charts.addWidget(self.alone_plot)
        chart_box.addWidget(self.charts, 1)
        self.chart_note = _label("", "body")
        self.chart_note.setMinimumHeight(44)
        chart_box.addWidget(self.chart_note)
        board.addWidget(chart_card, 0, 2, 1, 2)
        outer.addLayout(board, 1)
        outer.addSpacing(6)

        # Train and compare
        train_card, train_box = _card()
        row = QHBoxLayout()
        row.addWidget(_label("TRAIN AND COMPARE", "caps", wrap=False))
        row.addSpacing(24)
        self.checks = {}
        for name in ORDER:
            box = QCheckBox(NAMES[name])
            box.setChecked(True)
            box.setStyleSheet(f"QCheckBox {{ color: {theme.TEXT}; font-size: 14px; }}")
            box.toggled.connect(lambda on, n=name: self._checks_changed(n, on))
            self.checks[name] = box
            row.addWidget(box)
            row.addSpacing(10)
        row.addStretch(1)
        train_box.addLayout(row)
        row = QHBoxLayout()
        row.setSpacing(0)
        self.mode_buttons = {}
        for name, hint in (("Quick trial", "600 recordings, 3 folds: under a minute"),
                           ("Full run", "all recordings, 5 folds: about 2-5 minutes")):
            button = QPushButton(name)
            button.setObjectName("tab")
            button.setCheckable(True)
            button.setToolTip(hint)
            button.clicked.connect(lambda _=False, n=name: self._set_mode(n))
            self.mode_buttons[name] = button
            row.addWidget(button)
        row.addSpacing(18)
        self.mode_hint = _label("", "body", wrap=False)
        row.addWidget(self.mode_hint)
        row.addStretch(1)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedSize(260, 6)
        self.progress.setStyleSheet(f"QProgressBar {{ background: {theme.RAISED}; border: none; }}"
                                    f"QProgressBar::chunk {{ background: {theme.ACCENT}; }}")
        row.addWidget(self.progress)
        row.addSpacing(18)
        self.go = QPushButton("Train and compare  →")
        self.go.setObjectName("big")
        self.go.clicked.connect(self.start_or_stop)
        row.addWidget(self.go)
        train_box.addLayout(row)
        self.status = _label("Every model is trained on the same recordings and tested on ground stations it "
                             "never saw, so the comparison is fair.", "body")
        train_box.addWidget(self.status)
        outer.addWidget(train_card)

        self._set_mode("Quick trial")
        self.set_chart("Scores")
        self.reload()

    # -- layout helpers ----------------------------------------------------
    @staticmethod
    def _grid() -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(18)
        for column in range(4):
            grid.setColumnStretch(column, 1)
        return grid

    @staticmethod
    def _plot() -> pg.PlotWidget:
        plot = pg.PlotWidget(background=theme.PANEL)
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.setMouseEnabled(False, False)
        for name in ("left", "bottom"):
            axis = plot.getAxis(name)
            axis.setPen(pg.mkPen(theme.FAINT))
            axis.setTextPen(pg.mkPen(theme.DIM))
            font = QFont(theme.SANS)
            font.setPixelSize(13)
            axis.setStyle(tickFont=font)
            axis.enableAutoSIPrefix(False)
        plot.setMinimumHeight(260)
        return plot

    # -- runs ----------------------------------------------------------------
    def reload(self, select: Path | None = None) -> None:
        """Read the saved comparisons: the report's, then every run made in the app (newest first)."""
        runs = []
        paths = sorted(self._runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) \
            if self._runs_dir.exists() else []
        for path in paths + ([REPORT] if REPORT.exists() else []):
            try:
                runs.append(Run(path))
            except (OSError, ValueError):
                continue
        self._runs = runs
        self.picker.blockSignals(True)
        self.picker.clear()
        for run in runs:
            self.picker.addItem(run.title)
        self.picker.blockSignals(False)
        if not runs:
            self.lead.setText("No comparison yet. Press Train and compare below.")
            return
        # Newest full run first; a quick trial only if it was just made.
        index = 0
        if select is not None:
            index = next((i for i, r in enumerate(runs) if r.path == select), 0)
        else:
            index = next((i for i, r in enumerate(runs) if not r.limit), 0)
        self.picker.setCurrentIndex(index)
        self._picked(index)

    def _picked(self, index: int) -> None:
        if not 0 <= index < len(self._runs):
            return
        self._run = self._runs[index]
        run = self._run
        count = {6: "Six", 5: "Five", 4: "Four", 3: "Three", 2: "Two"}.get(len(run.models()), str(len(run.models())))
        kinds = "kind" if len(run.models()) == 1 else "kinds"
        trial = "  A quick trial: rougher numbers." if run.limit else ""
        self.lead.setText(f"{count} {kinds} of model, trained on {run.n:,} recordings volunteers had checked and "
                          f"tested on ground stations they never saw.{trial}")
        self._fill_stats()
        self._fill_table()
        self._draw()

    # -- the four numbers --------------------------------------------------
    def _fill_stats(self) -> None:
        run = self._run
        in_run = self._in_use in run.results
        model = self._in_use if in_run else run.best("auroc", True)
        r = run.results.get(model, {})
        name = NAMES.get(model, model)
        who = "the model the app uses" if in_run else "the best in this run"
        numbers = [
            (f"{r.get('accuracy', 0):.0%}", f"of recordings it gets right  ·  {name}, {who}"),
            (f"{r.get('decide_alone_5pc', 0):.0%}", "it can check on its own while staying under 5% wrong"),
            (f"{r.get('auroc', 0) * 100:.0f}%", "chance it rates a recording with a signal above one without"),
            (f"{len(run.models())}", "kinds of model compared on exactly the same test"),
        ]
        for (value, text), (v, t) in zip(self._stat_labels, numbers):
            if value.text() != v:
                count_up(value, v)
            text.setText(t)

    # -- the table ------------------------------------------------------------
    def _fill_table(self) -> None:
        run = self._run
        models = run.models()
        best = {key: run.best(key, higher) for key, _, higher, _ in COLUMNS}
        self.table.blockSignals(True)
        self.table.setRowCount(len(models))
        for row, model in enumerate(models):
            r = run.results[model]
            name = QTableWidgetItem(NAMES[model] + ("   in use" if model == self._in_use else ""))
            name.setForeground(QColor(COLOURS[model]))
            name.setToolTip(KINDS[model])
            font = QFont(theme.SANS_SEMIBOLD if model == self._in_use else theme.SANS)
            font.setPixelSize(14)
            name.setFont(font)
            self.table.setItem(row, 0, name)
            for column, (key, _, _, fmt) in enumerate(COLUMNS, start=1):
                value = r.get(key)
                text = "–" if value is None else (_duration(value) if key == "train_seconds" else fmt.format(value))
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if best[key] == model and key != "train_seconds":
                    item.setForeground(QColor(theme.SIGNAL))      # the best in its column
                    bold = QFont(theme.SANS_BOLD)
                    bold.setPixelSize(14)
                    item.setFont(bold)
                self.table.setItem(row, column, item)
        self.table.blockSignals(False)
        self.explain.setText("Green: the best in its column.  Alone 5% / 10%: how many recordings it can check "
                             "without a person, keeping mistakes under 5% or 10%.  Hover a heading for more.")

    def _row_picked(self) -> None:
        row = self.table.currentRow()
        models = self._run.models() if self._run else []
        self._focus = models[row] if 0 <= row < len(models) else None
        if self._focus:
            self.chart_note.setText(f"{NAMES[self._focus]}: {KINDS[self._focus]}.")
        self._draw()

    # -- the charts -----------------------------------------------------------
    def set_chart(self, name: str) -> None:
        self.charts.setCurrentWidget(self.alone_plot if name == "Deciding alone" else self.scores_plot)
        for key, button in self.chart_buttons.items():
            button.setChecked(key == name)
        self._chart = name
        self._draw()

    def _draw(self) -> None:
        run = self._run
        if run is None:
            return
        models = run.models()
        dim = lambda m: self._focus is not None and m != self._focus  # noqa: E731
        # Scores: right (solid) and ranking score (outline) per model
        plot = self.scores_plot
        plot.clear()
        x = np.arange(len(models))
        for i, m in enumerate(models):
            colour = QColor(COLOURS[m])
            colour.setAlpha(70 if dim(m) else 255)
            plot.addItem(pg.BarGraphItem(x=[i - 0.18], height=[run.results[m]["accuracy"]], width=0.34,
                                         brush=colour, pen=pg.mkPen(None)))
            outline = QColor(colour)
            plot.addItem(pg.BarGraphItem(x=[i + 0.18], height=[run.results[m]["auroc"]], width=0.34,
                                         brush=QColor(0, 0, 0, 0), pen=pg.mkPen(outline, width=2)))
        plot.getAxis("bottom").setTicks([[(i, NAMES[m].split(" (")[0].replace("Support-vector machine", "SVM")
                                           .replace("Gradient boosting", "Boosting")
                                           .replace("Logistic regression", "Logistic"))
                                          for i, m in enumerate(models)]])
        plot.setYRange(0.6, 0.95, padding=0)
        plot.setXRange(-0.6, len(models) - 0.4, padding=0)
        plot.setLabel("left", "score")
        # Deciding alone: the risk-coverage curves
        plot = self.alone_plot
        plot.clear()
        if run.curves:
            for m in models:
                curve = run.curves.get(m)
                if not curve:
                    continue
                colour = QColor(COLOURS[m])
                colour.setAlpha(60 if dim(m) else 255)
                width = 3.5 if m == self._focus or (self._focus is None and m == self._in_use) else 2
                plot.plot(curve["coverage"], curve["risk"], pen=pg.mkPen(colour, width=width))
            plot.addItem(pg.InfiniteLine(pos=0.05, angle=0, pen=pg.mkPen(theme.DIM, width=1, style=Qt.DashLine)))
            plot.setXRange(0, 1, padding=0)
            plot.setYRange(0, 0.25, padding=0)
            plot.setLabel("bottom", "share of recordings it checks on its own")
            plot.setLabel("left", "share of those it gets wrong")
        else:
            text = pg.TextItem("This run has no curves saved. Train and compare again to draw them.",
                               color=theme.DIM, anchor=(0.5, 0.5))
            plot.addItem(text)
            text.setPos(0.5, 0.5)
            plot.setXRange(0, 1)
            plot.setYRange(0, 1)
        if self._focus is None:
            self.chart_note.setText(
                "Solid bar: share right.  Outline: ranking score." if self._chart == "Scores" else
                "Lower and further right is better: it checks more recordings on its own and gets fewer wrong.  "
                "Dashed line: 5% wrong.")

    # -- training -----------------------------------------------------------
    def _set_mode(self, name: str) -> None:
        self._mode = name
        for key, button in self.mode_buttons.items():
            button.setChecked(key == name)
        self.mode_hint.setText(self.mode_buttons[name].toolTip())

    def _checks_changed(self, name: str, on: bool) -> None:
        # The combined model needs both of its parts: ticking it ticks them, unticking a part unticks it.
        if name == "cnn+forest" and on:
            for part in ("cnn", "forest"):
                self.checks[part].setChecked(True)
        elif name in ("cnn", "forest") and not on:
            self.checks["cnn+forest"].setChecked(False)
        self.go.setEnabled(any(box.isChecked() for box in self.checks.values()) or self._process is not None)

    def chosen_models(self) -> list[str]:
        return [m for m in ORDER if self.checks[m].isChecked()]

    def command(self) -> list[str]:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        quick = self._mode == "Quick trial"
        args = [str(SCRIPT), "--models", *self.chosen_models(), "--out", str(self._runs_dir),
                "--name", f"run_{stamp}_{'quick' if quick else 'full'}"]
        return args + (list(QUICK) if quick else [])

    def start_or_stop(self) -> None:
        if self._process is not None:
            self._process.kill()
            return
        args = self.command()
        self._expected = self._runs_dir / f"{args[args.index('--name') + 1]}.json"
        self._folds = int(QUICK[QUICK.index("--folds") + 1]) if self._mode == "Quick trial" else 5
        process = QProcess(self)
        process.setWorkingDirectory(str(ROOT))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._output)
        process.finished.connect(self._finished)
        self._process = process
        self._log = ""
        self.progress.setValue(2)
        self.go.setText("Stop")
        self.status.setText("Starting: reading the recordings' measurements…")
        for box in self.checks.values():
            box.setEnabled(False)
        process.start(sys.executable, args)

    def _output(self) -> None:
        text = bytes(self._process.readAllStandardOutput()).decode("utf-8", "replace")
        self._log += text
        for line in text.splitlines():
            line = line.strip()
            if not line or "Warning" in line or line.startswith("warnings."):
                continue
            fold = re.search(r"fold (\d+)/(\d+) done \((\d+) s\)", line)
            if fold:
                k, n = int(fold.group(1)), int(fold.group(2))
                self.progress.setValue(int(10 + 85 * k / n))
                self.status.setText(f"Trained and tested on {k} of {n} groups of ground stations "
                                    f"({fold.group(3)} s so far)…")
            elif line.startswith("Loading the waterfalls"):
                self.progress.setValue(6)
                self.status.setText("Loading the waterfall pictures for the CNN…")
            elif "recordings," in line and "stations" in line:
                self.status.setText(line)

    def _finished(self, code: int, _status) -> None:
        self._process = None
        self.go.setText("Train and compare  →")
        for box in self.checks.values():
            box.setEnabled(True)
        if code == 0 and self._expected.exists():
            self.progress.setValue(100)
            self.status.setText("Done. The new results are shown above; earlier runs stay in the list.")
            self.reload(select=self._expected)
        else:
            self.progress.setValue(0)
            last = [l for l in self._log.strip().splitlines() if l.strip()][-1:] or ["stopped"]
            self.status.setText(f"Stopped before the end ({last[0][:140]}).")
