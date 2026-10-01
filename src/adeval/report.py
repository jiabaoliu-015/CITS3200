from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path


@dataclass
class EvaluationResult:
    model: str
    tensors: int
    parameters: int


def _find_repo_root() -> Path:
    """Walk up from this file until we find the folder containing pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parent


REPORTS_DIR = _find_repo_root() / "outputs" / "reports"


def new_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_report_path(ext: str, stamp: str | None = None) -> Path:
    """Absolute path like <repo>/outputs/reports/evaluation_report_20261001_145400.html.

    Pass the same `stamp` for the HTML and PDF so the pair share a filename.
    """
    stamp = stamp or new_timestamp()
    return REPORTS_DIR / f"evaluation_report_{stamp}.{ext}"


def export_html(results: list[EvaluationResult], path: Path | None = None) -> Path:
    path = (path or default_report_path("html")).resolve()

    rows = "\n".join(
        f"<tr><td>{escape(r.model)}</td><td>{r.tensors}</td>"
        f"<td>{r.parameters:,}</td></tr>"
        for r in results
    )
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AdEval Evaluation Report</title>
<style>
  body {{ font-family: sans-serif; margin: 2rem; color: #1e2327; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 720px; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  caption {{ text-align: left; margin-bottom: 0.5rem; color: #666; }}
</style>
</head>
<body>
  <h1>AdEval Evaluation Report</h1>
  <table>
    <caption>Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</caption>
    <thead><tr><th>Model</th><th>Tensors</th><th>Parameters</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def export_pdf(results: list[EvaluationResult], path: Path | None = None) -> Path:
    from fpdf import FPDF

    path = (path or default_report_path("pdf")).resolve()

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "AdEval Evaluation Report", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        8,
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(4)

    col_widths = (90, 40, 50)
    headers = ("Model", "Tensors", "Parameters")

    pdf.set_font("Helvetica", "B", 11)
    for header, width in zip(headers, col_widths):
        pdf.cell(width, 8, header, border=1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    for result in results:
        pdf.cell(col_widths[0], 8, result.model, border=1)
        pdf.cell(col_widths[1], 8, str(result.tensors), border=1)
        pdf.cell(col_widths[2], 8, f"{result.parameters:,}", border=1)
        pdf.ln()

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    return path