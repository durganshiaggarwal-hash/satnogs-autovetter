"""The "Compare models" tab: different kinds of AI, side by side.

Six kinds of model were trained on exactly the same RML24 recordings and
tested on exactly the same held-out recordings (by
``scripts/compare_models.py``).  This tab shows:

* **one accuracy curve per model**, on one chart, so it is easy to see
  which model wins at which signal quality;
* **a table**: accuracy overall and on clean signals, size, how long
  it took to train, and how fast it answers;
* **which types the selected model mixes up**;
* a **"Use this model"** button, so the whole app can run on any of them.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...gui import theme
from ...gui.motion import PillButton, fade_in
from ...gui.pages.base import Page
from ...gui.widgets import card
from .report import ConfusionMatrix

#: The command that makes the results, shown when there are none yet.
COMMAND = 'python scripts/compare_models.py "path to RML24_IQdata"'
QUICK_COMMAND = 'python scripts/compare_models.py "path to RML24_IQdata" --per-cell 60 --epochs 8'


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min"


def _size(n: int | None, name: str) -> str:
    if not n:
        return "–"
    word = "decisions" if name == "forest" else "weights"
    return f"{n / 1e3:,.0f}k {word}" if n < 1e6 else f"{n / 1e6:.1f}M {word}"


class ComparePage(Page):
    """Which kind of AI is best at naming the signal?"""

    #: The left column is 300 px wide; a card's padding takes theme.PAD from each side.
    LEFT_TEXT_WIDTH = 300 - 2 * theme.PAD

    def _fit_labels(self) -> None:
        """Give each wrapped label the height its text needs at the column's width.

        Without this the column's layout squeezed them and the text was clipped
        top and bottom (wrapped labels in a fixed-width column do not report
        their height to it on their own).
        """
        for label in (self.chosen_name, self.chosen_about, self.in_use, self.fair):
            label.ensurePolished()   # measure with the stylesheet's font, not the default one
            label.setMinimumHeight(label.heightForWidth(self.LEFT_TEXT_WIDTH) + 2 if label.text() else 0)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, self._fit_labels)   # again once on screen, when styles are certainly applied

    COLUMNS = ["Model", "Overall", "+10 dB and above", "Size", "Training time", "Speed (one signal)", "In use"]

    def __init__(self, client, parent: QWidget | None = None) -> None:
        super().__init__(client, parent)
        self._listing: dict | None = None
        self._selected: str | None = None

        # -- left: the selected model, and the button -------------------
        chosen_card, chosen = card("The selected model", "Click a row in the table to select a model.")
        self.chosen_name = QLabel("–")
        self.chosen_name.setStyleSheet(f"font-family: '{theme.SANS_SEMIBOLD}'; font-size: 12pt;")
        self.chosen_name.setWordWrap(True)
        self.chosen_about = QLabel("")
        self.chosen_about.setObjectName("Hint")
        self.chosen_about.setWordWrap(True)
        self.use_button = PillButton("Use this model", "primary")
        self.use_button.setEnabled(False)
        self.use_button.setToolTip("From now on, every tab uses this model to name the signal. "
                                   "The app remembers the choice next time.")
        self.use_button.clicked.connect(self.use_selected)
        self.in_use = QLabel("")
        self.in_use.setObjectName("Muted")
        self.in_use.setWordWrap(True)
        for widget in (self.chosen_name, self.chosen_about, self.use_button, self.in_use):
            chosen.addWidget(widget)
        # Wrapped labels in a fixed-width column were squeezed below their text
        # height and clipped; never let them shrink below what the text needs.
        from PySide6.QtWidgets import QSizePolicy

        for label in (self.chosen_name, self.chosen_about, self.in_use):
            label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        fair_card, fair = card("A fair test")
        self.fair = QLabel("Every model learns from the same recordings and is tested on the same "
                           "recordings it never saw. Only the model changes.")
        self.fair.setObjectName("Hint")
        self.fair.setWordWrap(True)
        self.fair.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        fair.addWidget(self.fair)
        self.refresh_button = PillButton("Reload results", "secondary")
        self.refresh_button.clicked.connect(self.refresh)
        fair.addWidget(self.refresh_button)
        self._fit_labels()

        left = QWidget()
        left.setFixedWidth(300)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(theme.GAP)
        left_layout.addWidget(chosen_card)
        left_layout.addWidget(fair_card)
        left_layout.addStretch(1)

        # -- when only the main model exists: what was saved with it --------
        # The other models have to be trained on RML24 itself; without the
        # dataset, this shows the model the app uses and how its training went.
        self.empty_card, empty = card(
            "The model this app uses",
            "Trained once on the RML24 dataset; the app needs only the saved model, not the dataset.")
        self.card_facts = QLabel("")
        self.card_facts.setTextFormat(Qt.RichText)
        self.card_facts.setWordWrap(True)
        empty.addWidget(self.card_facts)
        self.training_curve = pg.PlotWidget()
        self.training_curve.setMinimumHeight(320)
        self.training_curve.setMouseEnabled(False, False)
        self.training_curve.setMenuEnabled(False)
        self.training_curve.setLabel("left", "loss (lower is better)")
        self.training_curve.setLabel("bottom", "training pass (epoch)")
        self.training_curve.showGrid(x=True, y=True, alpha=0.2)
        empty.addWidget(self.training_curve)
        self.curve_key = QLabel("")
        self.curve_key.setTextFormat(Qt.RichText)
        self.curve_key.setWordWrap(True)
        empty.addWidget(self.curve_key)
        self.empty_text = QLabel(
            "The other kinds of model (plain CNN, LSTM, Transformer, ResNet without expert features, random "
            "forest) learn from RML24 itself, so they appear here only once they have been trained on the "
            f"dataset with <code>{COMMAND}</code>. To see how well the model in use does on new signals, "
            "open <b>How good is the AI?</b>.")
        self.empty_text.setTextFormat(Qt.RichText)
        self.empty_text.setObjectName("Hint")
        self.empty_text.setWordWrap(True)
        empty.addWidget(self.empty_text)

        # -- right: the curves, the table, the mix-ups -------------------
        curve_card, curve_layout = card("Accuracy at every signal quality, one line per model",
                                        "Higher is better. The dashed line is guessing (1 in 6).")
        self.curve = pg.PlotWidget()
        self.curve.setMinimumHeight(380)
        self.curve.setMouseEnabled(False, False)
        self.curve.setMenuEnabled(False)
        self.curve.setYRange(0, 100)
        self.curve.setLabel("left", "named correctly (%)")
        self.curve.setLabel("bottom", "signal quality, SNR label (dB)")
        self.curve.showGrid(x=True, y=True, alpha=0.2)
        curve_layout.addWidget(self.curve)
        # The key sits under the chart, so it never hides a line.
        self.key = QLabel("")
        self.key.setTextFormat(Qt.RichText)
        self.key.setWordWrap(True)
        curve_layout.addWidget(self.key)
        self.winner = QLabel("")
        self.winner.setWordWrap(True)
        self.winner.setStyleSheet(f"font-family: '{theme.SANS_SEMIBOLD}';")
        curve_layout.addWidget(self.winner)

        table_card, table_layout = card("The models side by side",
                                        "Size: how much the model has to learn. Speed: time to name one "
                                        "signal on this computer. Best value in each column in green.")
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        table_layout.addWidget(self.table)

        matrix_card, matrix_layout = card("Which types the selected model mixes up",
                                          "Recordings at 0 dB and above. Green: right. Red: the mix-ups.")
        self.matrix_title = QLabel("")
        self.matrix_title.setObjectName("Muted")
        matrix_layout.addWidget(self.matrix_title)
        self.matrix = ConfusionMatrix()
        matrix_layout.addWidget(self.matrix)

        self.results_box = QWidget()
        results = QVBoxLayout(self.results_box)
        results.setContentsMargins(0, 0, 0, 0)
        results.setSpacing(theme.GAP)
        results.addWidget(curve_card)
        results.addWidget(table_card)
        results.addWidget(matrix_card)

        right = QVBoxLayout()
        right.setSpacing(theme.GAP)
        right.addWidget(self.empty_card)
        right.addWidget(self.results_box)
        right.addStretch(1)
        top = QHBoxLayout()
        top.setSpacing(theme.GAP)
        top.addWidget(left, 0, Qt.AlignTop)
        top.addLayout(right, 1)
        content = QWidget()
        stack = QVBoxLayout(content)
        stack.setContentsMargins(0, 0, 14, 8)
        stack.addLayout(top)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 18, 14, 10)
        layout.setSpacing(10)
        layout.addWidget(self.heading(self.title, "The AI this app uses, and how its training went. Other kinds of "
                                                  "AI join it here once they are trained on RML24."))
        layout.addWidget(self.scrolling(content), 1)
        self.empty_card.hide()
        self.results_box.hide()

    @property
    def title(self) -> str:
        return "Compare models"

    # -- loading ----------------------------------------------------------
    def refresh(self) -> None:
        self.tasks.run(self.client.models, self.show_listing, self.report_error)
        self.tasks.run(self.client.model_card, self.show_model_card, lambda _error: None)

    def show_model_card(self, saved: dict) -> None:
        """The model in use, as saved: kind, size, RML24 test score, and its training curve."""
        size = saved.get("n_parameters")
        accuracy = saved.get("test_accuracy")
        facts = [f"<b>{(saved.get('architecture') or 'model').upper()}</b> with expert features, "
                 f"naming {len(saved.get('classes') or [])} types: {', '.join(saved.get('classes') or [])}."]
        if size:
            facts.append(f"{size / 1e3:,.0f} thousand weights learned.")
        if accuracy is not None:
            facts.append(f"Tested on RML24 recordings it never saw: <b>{100 * accuracy:.1f}%</b> named correctly, "
                         "across every signal quality from -20 to +20 dB.")
        self.card_facts.setText("  ".join(facts))
        epochs, loss, val = saved.get("epochs") or [], saved.get("loss") or [], saved.get("val_loss") or []
        self.training_curve.clear()
        if loss:
            self.training_curve.plot(epochs[:len(loss)], loss, pen=pg.mkPen(theme.MAUVE, width=2.5))
        if val:
            self.training_curve.plot(epochs[:len(val)], val, pen=pg.mkPen(theme.GOOD, width=2.5))
        self.curve_key.setText(
            f"<span style='color:{theme.MAUVE}'>&#8212;</span> on the recordings it learned from &nbsp;&nbsp; "
            f"<span style='color:{theme.GOOD}'>&#8212;</span> on recordings kept aside to check it "
            f"({len(loss)} passes)." + (" The two lines end close together, so it learned the types rather than "
                                        "memorising its examples." if loss and val and val[-1] <= 1.1 * loss[-1]
                                        else ""))

    def show_listing(self, listing: dict) -> None:
        self._listing = listing
        has = bool(listing.get("has_results"))
        self.empty_card.setVisible(not has)
        self.results_box.setVisible(has)
        setup = listing.get("setup") or {}
        if setup:
            self.fair.setText(
                f"Every model learned from the same {setup.get('train', 0):,} recordings "
                f"(plus {setup.get('validation', 0):,} to decide when to stop) and was tested on the same "
                f"{setup.get('test', 0):,} recordings it never saw. At most {setup.get('epochs')} passes each. "
                "Only the model changes.")
        self._draw_curves(listing)
        self._fill_table(listing)
        active = next((m for m in listing["models"] if m["active"]), None)
        self.in_use.setText(f"The app is using: {active['label']}" if active else "")
        self._fit_labels()
        # Keep the selection; else the model in use; else the top row (the best on clean signals).
        for keep in (self._selected, listing.get("active")):
            if keep is not None and self._row_of(keep) is not None:
                self._select(keep)
                break
        else:
            first = self.table.item(0, 0)
            self._select(first.data(Qt.UserRole) if first is not None else None)

    def _compared(self, listing: dict) -> list[dict]:
        return [m for m in listing["models"] if m.get("results")]

    def _colour(self, name: str) -> str:
        names = [m["name"] for m in (self._listing or {}).get("models", []) if m["name"] != "main"]
        return theme.SERIES[names.index(name) % len(theme.SERIES)] if name in names else theme.INK_3

    def _draw_curves(self, listing: dict) -> None:
        self.curve.clear()
        compared = self._compared(listing)
        self.key.setText("&nbsp;&nbsp;&nbsp;".join(
            f"<span style='color:{self._colour(m['name'])}; font-size:14pt;'>&#9632;</span>&nbsp;{m['label']}"
            for m in compared))
        if not compared:
            self.winner.setText("")
            return
        self.curve.addItem(pg.InfiniteLine(100 / 6, angle=0, pen=pg.mkPen(theme.INK_3, width=1, style=Qt.DashLine)))
        for model in compared:
            points = sorted((float(s), a) for s, a in model["results"]["accuracy_by_snr"].items())
            colour = self._colour(model["name"])
            width = 3.2 if model["name"] == "resnet" else 2.0
            self.curve.plot([p[0] for p in points], [100 * p[1] for p in points], name=model["label"],
                            pen=pg.mkPen(colour, width=width), symbol="o", symbolSize=5,
                            symbolBrush=colour, symbolPen=None)
        clean = [m for m in compared if m["results"].get("accuracy_10db_plus") is not None]
        if clean:
            best = max(clean, key=lambda m: m["results"]["accuracy_10db_plus"])
            worst = min(clean, key=lambda m: m["results"]["accuracy_10db_plus"])
            text = (f"Best on clean signals: {best['label']}, "
                    f"{100 * best['results']['accuracy_10db_plus']:.1f}% at +10 dB and above")
            if worst is not best:
                text += (f" — the weakest, {worst['label']}, gets "
                         f"{100 * worst['results']['accuracy_10db_plus']:.1f}%.")
            self.winner.setText(text)
            fade_in(self.winner)

    def _fill_table(self, listing: dict) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        rows = [m for m in listing["models"] if m.get("results") or (m["name"] == "main" and m["available"])]
        compared = self._compared(listing)
        best = {}
        if compared:
            pick = lambda key, low=False: (min if low else max)(  # noqa: E731
                (m for m in compared if m["results"].get(key) is not None),
                key=lambda m: m["results"][key], default=None)
            for column, key, low in ((1, "accuracy", False), (2, "accuracy_10db_plus", False),
                                     (3, "n_parameters", True), (4, "train_seconds", True),
                                     (5, "ms_per_recording", True)):
                winner = pick(key, low)
                if winner is not None:
                    best[column] = winner["name"]
        for model in sorted(rows, key=lambda m: -((m.get("results") or {}).get("accuracy_10db_plus") or -1)):
            r = model.get("results") or {}
            row = self.table.rowCount()
            self.table.insertRow(row)
            cells = [
                model["label"],
                f"{100 * r['accuracy']:.1f}%" if r.get("accuracy") is not None else "–",
                f"{100 * r['accuracy_10db_plus']:.1f}%" if r.get("accuracy_10db_plus") is not None else "–",
                _size(r.get("n_parameters"), model["name"]),
                _duration(r.get("train_seconds")),
                f"{r['ms_per_recording']:.1f} ms" if r.get("ms_per_recording") is not None else "–",
                "in use" if model["active"] else ("" if model["available"] else "not saved"),
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, model["name"])
                if column == 0 and model["name"] != "main":
                    item.setForeground(QColor(self._colour(model["name"])))
                if best.get(column) == model["name"]:
                    item.setForeground(QColor(theme.GOOD))
                    item.setText(text + "  (best)")
                if column == 6 and model["active"]:
                    item.setForeground(QColor(theme.GOOD))
                self.table.setItem(row, column, item)
        self.table.blockSignals(False)
        # Tall enough for every row, no empty space and no scrolling.
        height = self.table.horizontalHeader().height() + 6
        height += sum(self.table.rowHeight(r) for r in range(self.table.rowCount()))
        self.table.setFixedHeight(max(height, 120))

    # -- selecting and using ----------------------------------------------
    def _row_of(self, name: str | None) -> int | None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.data(Qt.UserRole) == name:
                return row
        return None

    def _select(self, name: str | None) -> None:
        row = self._row_of(name)
        if row is not None:
            self.table.selectRow(row)
        self._show_selected(name if row is not None else None)

    def _selection_changed(self) -> None:
        items = self.table.selectedItems()
        self._show_selected(items[0].data(Qt.UserRole) if items else None)

    def model(self, name: str | None) -> dict | None:
        return next((m for m in (self._listing or {}).get("models", []) if m["name"] == name), None)

    def _show_selected(self, name: str | None) -> None:
        self._selected = name
        model = self.model(name)
        if model is None:
            self.chosen_name.setText("–")
            self.chosen_about.setText("")
            self.use_button.setEnabled(False)
            return
        self.chosen_name.setText(model["label"])
        self.chosen_about.setText(model["about"])
        self._fit_labels()
        self.use_button.setEnabled(bool(model["available"]) and not model["active"])
        self.use_button.setText("In use" if model["active"] else "Use this model")
        r = model.get("results")
        classes = ((self._listing or {}).get("setup") or {}).get("classes")
        if r and r.get("confusion_0db_plus") and classes:
            self.matrix.show_matrix(classes, r["confusion_0db_plus"])
            self.matrix_title.setText(model["label"])
        elif model["name"] == "main":
            self.matrix_title.setText("The main model was not part of this comparison; "
                                      "its results are in the \"How good is the AI?\" tab.")

    def use_selected(self) -> None:
        name = self._selected
        model = self.model(name)
        if model is None:
            return
        self.use_button.setEnabled(False)

        def done(listing: dict) -> None:
            self.show_listing(listing)
            self.notify.emit(f"Every tab now uses: {model['label']}.", "success")

        def failed(error) -> None:
            self.use_button.setEnabled(True)
            self.report_error(error)

        self.tasks.run(lambda: self.client.use_model(name), done, failed)


__all__ = ["ComparePage"]
