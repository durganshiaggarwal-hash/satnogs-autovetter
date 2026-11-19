"""Keeping benchmark results so nobody has to run them twice.

A full benchmark takes minutes.  The GUI should open instantly and show
the last results; the API should answer "how did the hybrid do on 16QAM
at 6 dB?" without recomputing anything; and a result quoted in the
write-up should be traceable to the run that produced it.

All three need the results on disk, in something queryable.  SQLite is
the right size: it ships with Python, it is one file, and it needs no
server -- which matters for a project two students have to install on
their own laptops.

DESIGN NOTE: ENCAPSULATION OF A RESOURCE
----------------------------------------
:class:`ResultStore` hides everything about SQL.  Callers save a
:class:`~satrecovery.evaluation.runner.BenchmarkResult` and get back
plain dictionaries; no SQL string ever leaves this module, so the schema
can change without touching the GUI or the API.

It is also a context manager, so the database connection is closed even
when an experiment fails halfway::

    with ResultStore("results.db") as store:
        run_id = store.save(result, "phase6 full grid")
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..core.exceptions import ValidationError
from .runner import BenchmarkResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL,
    created   REAL NOT NULL,
    source    TEXT NOT NULL,
    notes     TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS cells (
    run_id          INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    receiver        TEXT NOT NULL,
    modulation      TEXT NOT NULL,
    snr_db          REAL NOT NULL,
    profile         TEXT NOT NULL,
    source          TEXT NOT NULL,
    n_frames        INTEGER NOT NULL,
    bits            INTEGER NOT NULL,
    errors          INTEGER NOT NULL,
    ber             REAL NOT NULL,
    ber_low         REAL NOT NULL,
    ber_high        REAL NOT NULL,
    ece             REAL NOT NULL,
    overconfidence  REAL NOT NULL,
    ranking_quality REAL NOT NULL,
    coverage_1e3    REAL NOT NULL,
    lost_frames     INTEGER NOT NULL,
    triage_quality  REAL,
    seconds         REAL NOT NULL,
    frame_errors    TEXT NOT NULL,
    frame_bits      TEXT NOT NULL,
    frame_conf      TEXT NOT NULL,
    PRIMARY KEY (run_id, receiver, modulation, snr_db, profile)
);
"""

#: Columns returned by :meth:`ResultStore.cells`, in order.  The per-frame
#: arrays are stored as JSON text and returned as lists.
_CELL_COLUMNS = (
    "receiver", "modulation", "snr_db", "profile", "source", "n_frames",
    "bits", "errors", "ber", "ber_low", "ber_high", "ece", "overconfidence",
    "ranking_quality", "coverage_1e3", "lost_frames", "triage_quality",
    "seconds", "frame_errors", "frame_bits", "frame_conf",
)


class ResultStore:
    """A small SQLite database of benchmark runs.

    Parameters
    ----------
    path:
        Database file.  Created if missing.  ``":memory:"`` gives a
        throwaway database, which is what the tests use.
    """

    def __init__(self, path: str | Path = "results.db") -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._connection: sqlite3.Connection | None = sqlite3.connect(self._path)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __enter__(self) -> "ResultStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the database. Safe to call more than once."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    @property
    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            raise ValidationError("store", self._path, "has been closed")
        return self._connection

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def save(self, result: BenchmarkResult, name: str, notes: str = "") -> int:
        """Store one run. Returns its id.

        Written in a single transaction: either every cell of the run is
        saved or none is, so an interrupted save never leaves half a
        result looking like a whole one.
        """
        if not isinstance(result, BenchmarkResult):
            raise ValidationError(
                "result", type(result).__name__, "expected a BenchmarkResult"
            )
        name = str(name).strip()
        if not name:
            raise ValidationError("name", name, "a run needs a name")

        with self._db:
            cursor = self._db.execute(
                "INSERT INTO runs (name, created, source, notes) VALUES (?, ?, ?, ?)",
                (name, time.time(), result.source, str(notes)),
            )
            run_id = int(cursor.lastrowid)
            for cell in result.cells:
                row = cell.as_dict()
                triage = row["triage_quality"]
                self._db.execute(
                    f"INSERT INTO cells VALUES ({', '.join('?' * 22)})",
                    (
                        run_id, row["receiver"], row["modulation"], row["snr_db"],
                        row["profile"], row["source"], row["n_frames"], row["bits"],
                        row["errors"], row["ber"], row["ber_low"], row["ber_high"],
                        row["ece"], row["overconfidence"], row["ranking_quality"],
                        row["coverage_at_1e-3"], row["lost_frames"],
                        None if triage != triage else triage,  # NaN -> NULL
                        row["seconds"],
                        json.dumps(cell.frame_errors.tolist()),
                        json.dumps(cell.frame_bits.tolist()),
                        json.dumps([round(v, 8) for v in cell.frame_confidence.tolist()]),
                    ),
                )
        return run_id

    def delete(self, run_id: int) -> None:
        """Remove a run and all its cells."""
        with self._db:
            self._db.execute("DELETE FROM runs WHERE id = ?", (int(run_id),))

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def runs(self) -> list[dict[str, Any]]:
        """Every stored run, newest first, with how many cells each holds."""
        rows = self._db.execute(
            """SELECT r.id, r.name, r.created, r.source, r.notes, COUNT(c.run_id)
               FROM runs r LEFT JOIN cells c ON c.run_id = r.id
               GROUP BY r.id ORDER BY r.created DESC, r.id DESC"""
        ).fetchall()
        return [
            {"id": r[0], "name": r[1], "created": r[2], "source": r[3],
             "notes": r[4], "cells": r[5]}
            for r in rows
        ]

    def cells(
        self,
        run_id: int,
        receiver: str | None = None,
        modulation: str | None = None,
        profile: str | None = None,
    ) -> list[dict[str, Any]]:
        """Cells of one run, filtered, ordered by scenario then receiver."""
        clauses, values = ["run_id = ?"], [int(run_id)]
        for column, value in (
            ("receiver", receiver), ("modulation", modulation), ("profile", profile)
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                values.append(str(value))
        rows = self._db.execute(
            f"SELECT {', '.join(_CELL_COLUMNS)} FROM cells WHERE {' AND '.join(clauses)} "
            "ORDER BY modulation, profile, snr_db, receiver",
            values,
        ).fetchall()

        out = []
        for row in rows:
            record = dict(zip(_CELL_COLUMNS, row))
            for key in ("frame_errors", "frame_bits", "frame_conf"):
                record[key] = json.loads(record[key])
            if record["triage_quality"] is None:
                record["triage_quality"] = float("nan")
            out.append(record)
        return out

    def latest(self, name: str | None = None) -> int | None:
        """Id of the most recent run, optionally with a given name."""
        for run in self.runs():
            if name is None or run["name"] == name:
                return int(run["id"])
        return None

    def __repr__(self) -> str:
        state = "closed" if self._connection is None else f"{len(self.runs())} runs"
        return f"ResultStore({self._path!r}, {state})"


__all__ = ["ResultStore"]
