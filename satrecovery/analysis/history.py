"""Every analysis, kept: the History tab's memory.

:class:`AnalysisHistory` saves each analysis in a small SQLite database
inside the app's data folder, so an earlier result can be found and
opened again exactly as it was, without analysing it again.

Each row keeps a one-line summary (shown in the list) and the whole
result, compressed, so reopening needs no recomputation.

DESIGN NOTE: ENCAPSULATION
--------------------------
Nobody outside this class knows there is a database, what its columns
are, or that the results are compressed.  The rest of the app only
calls :meth:`add`, :meth:`entries`, :meth:`get` and :meth:`clear`.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
import zlib
from pathlib import Path
from typing import Any

from ..core.exceptions import ValidationError


class AnalysisHistory:
    """A list of past analyses, newest first."""

    #: The oldest entries are dropped beyond this many.
    LIMIT = 500

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS analyses ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT, saved_at REAL NOT NULL, kind TEXT NOT NULL,"
                " source TEXT NOT NULL, ai_label TEXT NOT NULL, confidence REAL NOT NULL,"
                " verdict TEXT NOT NULL, bits TEXT NOT NULL, result BLOB NOT NULL)")

    @property
    def path(self) -> Path:
        return self._path

    @contextlib.contextmanager
    def _connect(self):
        """One connection per call: committed, then closed (so Windows never keeps the file locked)."""
        db = sqlite3.connect(self._path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    # -- what is shown in the list ---------------------------------------
    @staticmethod
    def summarise(result: dict[str, Any]) -> dict[str, Any]:
        """The one-line summary of an analysis."""
        source = result.get("input", {})
        kind = source.get("kind", "?")
        speed = f"{source.get('symbol_rate', 0) / 1000:.0f} kbaud" if source.get("symbol_rate") else ""
        if kind == "recorded":
            label = source.get("snr_label_db")
            text = f"RML24 #{source.get('row', 0):,}" + (f", {label:+.0f} dB" if label is not None else "")
        elif kind == "simulated":
            snr = source.get("conditions", {}).get("snr_db")
            text = "Simulated" + (f", {snr:+.0f} dB" if snr is not None else "")
        elif kind == "file":
            text = f"File {source.get('name', '')}"
        else:
            text = kind
        if speed:
            text += f", {speed}"
        c = result["classification"]
        if c.get("true_label") is None:
            verdict = "unknown"
        else:
            verdict = "right" if c.get("correct") else f"wrong (really {c['true_label']})"
        check = result["recovery"].get("check", {})
        bits = (f"{check['bit_errors']:,} of {check['n_bits']:,} wrong" if check.get("available")
                else "not checked")
        level = result.get("trust", {}).get("level")
        label = c["label"] + (f" (⚠ {level})" if level in ("unsure", "unfamiliar") else "")
        return {"kind": kind, "source": text, "ai_label": label, "confidence": float(c["confidence"]),
                "verdict": verdict, "bits": bits}

    # -- the list -------------------------------------------------------
    def add(self, result: dict[str, Any]) -> int:
        """Keep one analysis; returns its number in the history."""
        if not isinstance(result, dict) or "classification" not in result or "recovery" not in result:
            raise ValidationError("result", type(result).__name__, "expected the result of one analysis")
        summary = self.summarise(result)
        blob = zlib.compress(json.dumps(result, separators=(",", ":")).encode("utf-8"), 6)
        with self._lock, self._connect() as db:
            cursor = db.execute(
                "INSERT INTO analyses (saved_at, kind, source, ai_label, confidence, verdict, bits, result)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), summary["kind"], summary["source"], summary["ai_label"], summary["confidence"],
                 summary["verdict"], summary["bits"], blob))
            entry = int(cursor.lastrowid)
            db.execute("DELETE FROM analyses WHERE id NOT IN (SELECT id FROM analyses ORDER BY id DESC LIMIT ?)",
                       (self.LIMIT,))
        return entry

    def entries(self, query: str = "", limit: int = 200) -> list[dict[str, Any]]:
        """Summaries, newest first; ``query`` keeps those mentioning it anywhere."""
        sql = "SELECT id, saved_at, kind, source, ai_label, confidence, verdict, bits FROM analyses"
        args: list[Any] = []
        words = [w for w in str(query).split() if w]
        if words:
            sql += " WHERE " + " AND ".join(
                "(source LIKE ? OR ai_label LIKE ? OR verdict LIKE ? OR bits LIKE ?)" for _ in words)
            for word in words:
                args += [f"%{word}%"] * 4
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(max(1, min(int(limit), self.LIMIT)))
        with self._lock, self._connect() as db:
            rows = db.execute(sql, args).fetchall()
        return [{"id": r[0], "saved_at": r[1], "kind": r[2], "source": r[3], "ai_label": r[4],
                 "confidence": r[5], "verdict": r[6], "bits": r[7]} for r in rows]

    def get(self, entry: int) -> dict[str, Any]:
        """The whole analysis, exactly as it was."""
        with self._lock, self._connect() as db:
            row = db.execute("SELECT result, saved_at FROM analyses WHERE id = ?", (int(entry),)).fetchone()
        if row is None:
            raise ValidationError("entry", entry, "not in the history")
        result = json.loads(zlib.decompress(row[0]).decode("utf-8"))
        result["history"] = {"id": int(entry), "saved_at": row[1]}
        return result

    def clear(self) -> int:
        """Forget everything; returns how many entries there were."""
        with self._lock, self._connect() as db:
            count = db.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            db.execute("DELETE FROM analyses")
        return int(count)

    def __len__(self) -> int:
        with self._lock, self._connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM analyses").fetchone()[0])


__all__ = ["AnalysisHistory"]
