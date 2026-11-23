"""The replay's Messages view: the data packets the station decoded from this pass.

A list of packets (time, format, who sent it to whom, size) and, for the
selected one, the header taken apart, any readable text, the bytes in hex
and the first bits -- in the Lab analysis's "recovered message" style.
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..frames import FrameStore, describe
from . import theme


class _Relay(QObject):
    done = Signal(int, object)     # obs id, list of frame descriptions
    failed = Signal(int, str)


class MessagesPanel(QWidget):
    """Packets on top, the selected packet's anatomy below."""

    counted = Signal(int)   # how many packets this pass has (for the tab title)

    def __init__(self, store: FrameStore | None, parent=None) -> None:
        super().__init__(parent)
        self._store = store
        self._relay = _Relay()
        self._relay.done.connect(self._loaded)
        self._relay.failed.connect(self._failed)
        self._current: int | None = None
        self._rows: list[dict] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.status = QLabel("")
        self.status.setObjectName("body")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["RECEIVED", "FORMAT", "FROM → TO", "BYTES"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setMaximumHeight(170)
        self.table.itemSelectionChanged.connect(self._selected)
        layout.addWidget(self.table)
        self.detail = QLabel("")
        self.detail.setTextFormat(Qt.RichText)
        self.detail.setWordWrap(True)
        self.detail.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        frame = QFrame()
        frame.setObjectName("card")
        inside = QVBoxLayout(frame)
        inside.setContentsMargins(14, 12, 14, 12)
        inside.addWidget(self.detail)
        layout.addWidget(frame, 1)
        self.note = QLabel("Decoded by the volunteer's station (gr-satellites); read and taken apart here.")
        self.note.setObjectName("body")
        layout.addWidget(self.note)

    # ------------------------------------------------------------------
    def show_observation(self, obs) -> None:
        self._current = obs.id
        self._rows = []
        self.table.setRowCount(0)
        self.detail.setText("")
        if obs.n_decoded_frames == 0:
            self.status.setText("The station decoded no data packets from this pass. Many stations record "
                                "a satellite without having its decoder, and some passes are too weak to decode.")
            self.counted.emit(0)
            return
        if self._store is None:
            self.status.setText(f"{obs.n_decoded_frames} packets were decoded, but there is no connection to fetch them.")
            self.counted.emit(obs.n_decoded_frames)
            return
        self.status.setText(f"Fetching the {obs.n_decoded_frames} packets the station decoded…")
        self.counted.emit(obs.n_decoded_frames)
        relay, store, obs_id = self._relay, self._store, obs.id

        def work() -> None:
            try:
                rows = [describe(frame) for frame in store.frames(obs_id)]
                relay.done.emit(obs_id, rows)
            except Exception as error:  # noqa: BLE001 - no network must not break the replay
                try:
                    relay.failed.emit(obs_id, f"{type(error).__name__}: {error}")
                except RuntimeError:
                    pass

        threading.Thread(target=work, name="satnogs-frames", daemon=True).start()

    def _failed(self, obs_id: int, message: str) -> None:
        if obs_id == self._current:
            self.status.setText(f"Could not fetch the packets: {message}")

    def _loaded(self, obs_id: int, rows: list[dict]) -> None:
        if obs_id != self._current:
            return
        self._rows = rows
        ax25 = sum(r["kind"].startswith("AX.25") for r in rows)
        self.status.setText(f"{len(rows)} data packets decoded from this pass"
                            + (f" · {ax25} in the AX.25 packet-radio format" if ax25 else ""))
        self.counted.emit(len(rows))
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            route = f"{row['source']} → {row['destination']}" if row["source"] else "–"
            for column, value in enumerate((row["time"], row["kind"].split(" (")[0], route, str(row["size"]))):
                item = QTableWidgetItem(value)
                item.setForeground(Qt.white if column != 1 else Qt.lightGray)
                self.table.setItem(i, column, item)
        if rows:
            self.table.selectRow(0)

    def _selected(self) -> None:
        row = self.table.currentRow()
        if not 0 <= row < len(self._rows):
            return
        r = self._rows[row]
        dim, mono = theme.DIM, theme.MONO
        lines = [f"<span style='color:{theme.ACCENT}; font-size:13px'><b>{r['kind']}</b></span>"]
        if r["source"]:
            route = " → ".join([r["source"]] + r["repeaters"] + [r["destination"]])
            lines.append(f"<span style='color:{dim}'>from → to</span>&nbsp; <b>{route}</b>")
        lines.append(f"<span style='color:{dim}'>size</span>&nbsp; {r['size']} bytes"
                     + (f" ({r['payload_size']} after the header)" if r["source"] else ""))
        if r["text"]:
            lines.append(f"<span style='color:{dim}'>readable text</span>&nbsp; "
                         f"<span style='font-family:\"{mono}\"'>{_escape(r['text'][:160])}</span>")
        bits = " ".join(r["bits"][i:i + 8] for i in range(0, len(r["bits"]), 8))
        lines.append(f"<span style='color:{dim}'>first bits</span><br><span style='font-family:\"{mono}\"; "
                     f"font-size:10px; color:{theme.SIGNAL}'>{bits}</span>")
        hex_block = "<br>".join(_escape(line) for line in r["hex"])
        lines.append(f"<span style='color:{dim}'>bytes</span><br><span style='font-family:\"{mono}\"; "
                     f"font-size:10px'>{hex_block}</span>")
        self.detail.setText("<br>".join(lines))


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
