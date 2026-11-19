"""A one-click PDF report of an analysis, like a lab report for a signal.

:class:`AnalysisReportWriter` turns one analysis -- the dictionary the
backend returns -- plus pictures of the app's own cards into a tidy A4
PDF: what the signal was, what the AI said and how sure it was, what the
receiver read, and whether the bits could be checked.

It uses only Qt, which the app already needs: an HTML page laid out by
``QTextDocument`` and printed to a ``QPdfWriter``.  Nothing to install.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QMarginsF, QSizeF, Qt, QUrl
from PySide6.QtGui import QPageLayout, QPageSize, QPdfWriter, QPixmap, QTextDocument

from ..analyser import STAGES

#: Page width available for pictures, in the writer's pixels (96 per inch).
PICTURE_WIDTH = 640


class AnalysisReportWriter:
    """Writes one analysis as a PDF."""

    TITLE = "Satellite Signal Analysis Report"

    def __init__(self, result: dict, pictures: list[tuple[str, QPixmap]] | None = None) -> None:
        if not isinstance(result, dict) or "classification" not in result or "recovery" not in result:
            raise ValueError("expected the result of one analysis")
        self._result = result
        # Pictures are kept at twice the size they are shown at, so they
        # stay sharp when printed, and no larger, so the PDF stays small.
        self._pictures = [
            (title, picture if picture.width() <= 2 * PICTURE_WIDTH
             else picture.scaledToWidth(2 * PICTURE_WIDTH, Qt.SmoothTransformation))
            for title, picture in (pictures or []) if not picture.isNull()]

    # -- the words -------------------------------------------------------
    @staticmethod
    def describe_input(source: dict) -> list[tuple[str, str]]:
        """Where the signal came from, as label/value rows."""
        kind = source.get("kind")
        rows: list[tuple[str, str]] = []
        if kind == "recorded":
            rows.append(("Source", f"Real recording from RML24 (#{source.get('row', 0):,})"))
            if source.get("snr_label_db") is not None:
                rows.append(("Signal quality", f"{source['snr_label_db']:+.0f} dB (dataset label)"))
        elif kind == "simulated":
            conditions = source.get("conditions", {})
            rows.append(("Source", "Simulated signal, made by the app"))
            if "snr_db" in conditions:
                rows.append(("Signal quality", f"{conditions['snr_db']:+.0f} dB"))
            if conditions.get("freq_offset_hz"):
                rows.append(("Doppler shift", f"{conditions['freq_offset_hz']:+.0f} Hz"))
        elif kind == "file":
            rows.append(("Source", f"Your file: {source.get('name', '')}"))
            rows.append(("Samples analysed", f"{source.get('samples_used', 0):,} of {source.get('samples_in_file', 0):,}"))
        if source.get("symbol_rate"):
            rows.append(("Speed", f"{source['symbol_rate'] / 1000:.0f} kbaud "
                                  f"({source.get('samples_per_symbol', '?')} samples per symbol)"))
        return rows

    def html(self) -> str:
        r = self._result
        c, rec = r["classification"], r["recovery"]
        check = rec.get("check", {})
        e = html.escape

        if c.get("true_label") is None:
            verdict = "The file does not say what the signal really is, so this answer cannot be marked right or wrong."
            colour = "#555555"
        elif c.get("correct"):
            verdict, colour = f"Right: it really is {c['true_label']}.", "#2e7d32"
        else:
            verdict, colour = f"Wrong: it is really {c['true_label']}.", "#c62828"

        trust = r.get("trust")
        trust_html = ""
        if trust:
            trust_colour = {"confident": "#2e7d32", "unsure": "#9a5b14"}.get(trust["level"], "#c62828")
            trust_html = f"<p style='color:{trust_colour}'><b>{e(trust['title'])}</b></p>" + "".join(
                f"<p class='muted'>• {e(reason)}</p>" for reason in trust.get("reasons", []))

        if check.get("available"):
            wrong, total = check["bit_errors"], check["n_bits"]
            outcome = (f"<b>Checked: {wrong:,} of {total:,} bits wrong ({100 * check['ber']:.2f}%)</b>")
        else:
            outcome = "<b>Not checked bit by bit.</b>"

        source_rows = "".join(f"<tr><td class='k'>{e(k)}</td><td>{e(v)}</td></tr>"
                              for k, v in self.describe_input(r.get("input", {})))
        options = "".join(f"<tr><td class='k'>{e(p['name'])}</td><td>{100 * p['p']:.1f}%</td></tr>"
                          for p in c["probabilities"][:4])
        steps = ", ".join(STAGES.get(s["name"], (s["name"],))[0] for s in rec.get("stages", []))
        bits = rec.get("bits_preview", "")
        bits = " ".join(bits[i : i + 8] for i in range(0, len(bits), 8))
        receiver = "hybrid (classical + AI)" if rec.get("receiver") == "hybrid" else rec.get("receiver", "classical")

        pictures = "".join(
            f"<p class='cap'>{e(title)}</p><p><img src='picture{i}' width='{shown_w}' height='{shown_h}'></p>"
            for i, (title, picture) in enumerate(self._pictures)
            for shown_w, shown_h in [self._shown_size(picture)])

        return f"""
<html><head><style>
  body {{ font-family: Arial, Helvetica, sans-serif; font-size: 10pt; color: #222222; }}
  h1 {{ font-size: 17pt; margin-bottom: 0; }}
  h2 {{ font-size: 12pt; margin-top: 14pt; margin-bottom: 4pt; color: #8a4b3c; }}
  td {{ padding: 2pt 10pt 2pt 0; vertical-align: top; }}
  td.k {{ color: #666666; }}
  .muted {{ color: #777777; font-size: 9pt; }}
  .cap {{ color: #666666; font-size: 9pt; margin-top: 10pt; }}
  .bits {{ font-family: 'Courier New', monospace; font-size: 9pt; }}
</style></head><body>
<h1>{self.TITLE}</h1>
<p class='muted'>Made by the Satellite Signal Analyser on {datetime.now():%d %B %Y, %H:%M}.</p>

<h2>1. The signal</h2>
<table>{source_rows}</table>

<h2>2. What kind of signal is it? (the AI classifier)</h2>
<p>The AI says <b>{e(c['label'])}</b>, and is <b>{100 * c['confidence']:.0f}% sure</b>.</p>
<p style='color:{colour}'><b>{e(verdict)}</b></p>
{trust_html}
<p class='muted'>The AI's top choices:</p>
<table>{options}</table>

<h2>3. What does it say? (the receiver)</h2>
<table>
  <tr><td class='k'>Receiver used</td><td>{e(receiver)}, reading it as {e(rec.get('modulation', ''))}</td></tr>
  <tr><td class='k'>Bits recovered</td><td>{rec.get('n_bits', 0):,}</td></tr>
  <tr><td class='k'>Receiver is sure of</td><td>{100 * rec.get('trusted_fraction', 0):.0f}% of the bits</td></tr>
  <tr><td class='k'>Cleaning steps</td><td>{e(steps)}</td></tr>
</table>
<p>{outcome}</p>
<p class='muted'>{e(check.get('detail', ''))}</p>
<p class='muted'>The first recovered bits:</p>
<p class='bits'>{e(bits)}</p>

{("<h2 style='page-break-before: always'>4. Pictures from the app</h2>" + pictures) if pictures else ''}
</body></html>"""

    @staticmethod
    def _shown_size(picture: QPixmap) -> tuple[int, int]:
        """The size a picture is shown at: at most the page width."""
        width = min(picture.width(), PICTURE_WIDTH)
        return width, round(picture.height() * width / max(picture.width(), 1))

    # -- the file --------------------------------------------------------
    def write(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix.lower() != ".pdf":
            path = path.with_suffix(".pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = QPdfWriter(str(path))
        writer.setResolution(96)
        writer.setTitle(self.TITLE)
        writer.setCreator("Satellite Signal Analyser")
        writer.setPageLayout(QPageLayout(QPageSize(QPageSize.A4), QPageLayout.Portrait,
                                         QMarginsF(16, 16, 16, 16), QPageLayout.Millimeter))
        document = QTextDocument()
        for i, (_, picture) in enumerate(self._pictures):
            document.addResource(QTextDocument.ImageResource, QUrl(f"picture{i}"), picture.toImage())
        document.setHtml(self.html())
        # Lay the page out in the writer's own units (96 per inch), so a
        # picture 640 wide is 640 of the page's ~670, as intended.
        document.setPageSize(QSizeF(writer.width(), writer.height()))
        document.print_(writer)
        return path


__all__ = ["AnalysisReportWriter"]
