import csv
import os
import pickle
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import torch
from fpdf import FPDF
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from adeval.console import console
from adeval.menus.base_menu import Menu
from adeval.menus.menu_names import MenuNames


class ReportMenu(Menu):
    def __init__(self):
        self.PROJECT_ROOT = Path(__file__).resolve().parents[3]
        self.OUTPUT_DIR = self.PROJECT_ROOT / "outputs"
        self.VALID_SUFFIXES = (
            "results.csv",
            "results.pkl",
            "AdEval_Report.html",
            "AdEval_Report.pdf",
        )
        self.IGNORED_DIRS = ["ImageSets", ".cache"]

    def run(self) -> MenuNames | None:
        console.print(
            Panel(
                "Report Menu",
                style="bold cyan",
            )
        )

        options = [
            "[0] Exit",
            "[1] Go to Main Menu",
            "[2] View Garage Evaluation Report",
            "[3] View Object Detection Evaluation Report",
            "[4] View HTML/PDF Report",
            "[5] Clear All Reports",
        ]
        for opt in options:
            console.print(opt)

        choice = Prompt.ask(
            "Select an option", choices=list(map(str, range(len(options))))
        )

        if choice == "0":
            return None
        elif choice == "1":
            return MenuNames.MainMenu
        elif choice == "2":
            paths = self.__find_valid_evaluation_paths("garage-evaluation")

            if len(paths) == 0:
                console.print("[red]No runs found")
                return MenuNames.ReportMenu

            path = self.__format_all_dirs_and_choose("Garage Evaluation", paths)
            self.__display_selected_run(path)
            self.__convert_to_pdf_html(path)

        elif choice == "3":
            dataset_name = self.__choose_dataset()
            if not dataset_name:
                return MenuNames.ReportMenu

            paths = self.__find_valid_evaluation_paths(
                f"openpcdet-evaluation/{dataset_name}"
            )
            if len(paths) == 0:
                console.print("[red]No runs found")
                return MenuNames.ReportMenu

            path = self.__format_all_dirs_and_choose(
                "Object Detection Evaluation", paths
            )
            self.__display_selected_run(path)
            self.__convert_to_pdf_html(path)
        elif choice == "4":
            paths = self.__find_valid_evaluation_paths("reports")
            if len(paths) == 0:
                console.print("[red]No runs found")
                return MenuNames.ReportMenu

            path = self.__format_all_dirs_and_choose("Report Generation", paths)
            files = [
                str(f.resolve())
                for f in path.iterdir()
                if f.is_file() and f.name.endswith(self.VALID_SUFFIXES)
            ]
            for p in files:
                if p.endswith("pdf"):
                    console.print(f"PDF Path: {p!s}")
                else:
                    console.print(f"HTML Path: {p!s}")
        elif choice == "5":
            self.__clear_reports()

        return MenuNames.ReportMenu

    def __find_valid_evaluation_paths(self, path: str):

        out_dir = self.OUTPUT_DIR / Path(path)

        if not out_dir.is_dir():
            return []

        valid_evaluations = [
            eval
            for eval in out_dir.iterdir()
            if eval.is_dir()
            and any(
                f.is_file() and f.name.endswith(self.VALID_SUFFIXES)
                for f in eval.iterdir()
            )
        ]

        return sorted(valid_evaluations, reverse=True)

    def __format_all_dirs_and_choose(self, table_name, paths: list[Path]):
        table = Table(title=f"{table_name}", show_lines=True)
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Run", style="green")
        table.add_column("Result files")

        for i, path in enumerate(paths, start=1):
            files = [
                str(f.resolve())
                for f in path.iterdir()
                if f.is_file() and f.name.endswith(self.VALID_SUFFIXES)
            ]
            table.add_row(str(i), path.name, ", ".join(files))

        console.print(table)

        choice = IntPrompt.ask(
            "Select a run",
            choices=[str(i) for i in range(1, len(paths) + 1)],
            show_choices=False,
        )

        return paths[choice - 1]

    def __choose_dataset(self):
        console.print(
            Panel(
                "Dataset Selection",
                style="bold cyan",
            )
        )

        options = [
            "[0] Go back",
            "[1] AV2",
            # "[2] nuScene",
        ]
        for opt in options:
            console.print(opt)

        choice = Prompt.ask(
            "Select an option", choices=list(map(str, range(len(options))))
        )

        if choice == "0":
            return ""
        elif choice == "1":
            return "av2"
        # elif choice == "2":
        #     return "nuscene"

    def __display_selected_run(self, path):
        files = [
            f.resolve()
            for f in Path(path).iterdir()
            if f.is_file() and f.name.endswith(self.VALID_SUFFIXES)
        ]

        for f in files:
            if f.name.endswith("csv"):
                console.print(self.__csv_to_table(f, 10))
            else:
                data = self.__read_pickle_file(f)
                console.print(self.__display_model_metrics_from_pkl_data(data))

        for f in files:
            console.print(f"Output path: {f!s}")

    def __csv_to_table(self, csv_path, max_rows=None) -> Table:
        with csv_path.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            table = Table(title=csv_path.name, show_lines=True)

            for column in header:
                table.add_column(column)

            for i, row in enumerate(reader):
                if max_rows is not None and i >= max_rows:
                    table.caption = f"Showing first {max_rows} rows"
                    break
                table.add_row(*row)

        return table

    def __read_pickle_file(self, path: Path):
        # Prevent bad pkl and only load numpy
        class SafeUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.split(".")[0] in ("numpy", "collections"):
                    return super().find_class(module, name)
                raise pickle.UnpicklingError(f"blocked global {module}.{name}")

        with open(path, "rb") as f:
            return SafeUnpickler(f).load()

    def __display_model_metrics_from_pkl_data(self, data) -> Table:
        scores_by_class = defaultdict(list)
        for frame in data:
            for name, score in zip(frame["name"], frame["score"]):
                scores_by_class[str(name)].append(float(score))

        all_scores = np.concatenate([np.array(s) for s in scores_by_class.values()])
        total = len(all_scores)

        table = Table(
            title="Per-class breakdown",
            header_style="bold green",
            show_footer=True,
        )
        num = {"justify": "right", "no_wrap": True}
        table.add_column("#", style="dim", **num)
        table.add_column(
            "Class", style="bold", footer="TOTAL", max_width=18, overflow="fold"
        )
        table.add_column("Count", footer=f"{total:,}", min_width=9, **num)
        table.add_column("% of detections", footer="100%", min_width=9, **num)
        table.add_column(
            "Mean score", footer=f"{all_scores.mean():.3f}", min_width=10, **num
        )
        table.add_column(
            "Max score", footer=f"{all_scores.max():.3f}", min_width=9, **num
        )
        table.add_column(
            "≥ 0.5", footer=f"{(all_scores >= 0.5).sum():,}", min_width=9, **num
        )

        rows = sorted(scores_by_class.items(), key=lambda kv: -len(kv[1]))
        for i, (name, s) in enumerate(rows, 1):
            s = np.array(s)
            table.add_row(
                str(i),
                name,
                f"{len(s):,}",
                f"{len(s) / total:.1%}",
                f"{s.mean():.3f}",
                f"{s.max():.3f}",
                f"{(s >= 0.5).sum():,}",
            )

        return table

    def __convert_to_pdf_html(self, path):
        if Confirm.ask("Convert to PDF and HTML?", default=False):
            timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
            out_directory = Path(self.OUTPUT_DIR) / f"reports/{timestamp}"
            out_directory.mkdir(parents=True, exist_ok=False)

            files = [
                f.resolve()
                for f in Path(path).iterdir()
                if f.is_file() and f.name.endswith(self.VALID_SUFFIXES)
            ]

            tables = []

            for f in files:
                if f.name.endswith("csv"):
                    with f.open(newline="") as fh:
                        tables.append(list(csv.reader(fh)))
                else:
                    data = self.__read_pickle_file(f)
                    scores_by_class = defaultdict(list)
                    for frame in data:
                        for name, score in zip(frame["name"], frame["score"]):
                            scores_by_class[str(name)].append(float(score))

                    total = sum(len(s) for s in scores_by_class.values())
                    table = [
                        [
                            "#",
                            "Class",
                            "Count",
                            "% of detections",
                            "Mean",
                            "Max",
                            ">=0.5",
                        ]
                    ]
                    rows = sorted(scores_by_class.items(), key=lambda kv: -len(kv[1]))
                    for i, (name, raw_scores) in enumerate(rows, 1):
                        scores = np.array(raw_scores)
                        table.append(
                            [
                                str(i),
                                name,
                                f"{len(scores):,}",
                                f"{len(scores) / total:.1%}",
                                f"{scores.mean():.3f}",
                                f"{scores.max():.3f}",
                                f"{(scores >= 0.5).sum():,}",
                            ]
                        )

                    all_scores = np.concatenate(
                        [np.asarray(s) for s in scores_by_class.values()]
                    )
                    table.append(
                        [
                            "",
                            "Total",
                            f"{total:,}",
                            "100.0%",
                            f"{all_scores.mean():.3f}",
                            f"{all_scores.max():.3f}",
                            f"{(all_scores >= 0.5).sum():,}",
                        ]
                    )
                    tables.append(table)

            # console.print(tables)

            # PDF
            pdf = FPDF()
            pdf.add_page()

            pdf.set_font("Helvetica", "B", 20)
            pdf.cell(0, 12, "Adeval Report", new_x="LMARGIN", new_y="NEXT")
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)

            pdf.set_font("Helvetica", size=10)
            pdf.cell(0, 8, f"Generated: {timestamp}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

            for rows in tables:
                with pdf.table() as table:  # first row is rendered as the header
                    for row in rows:
                        pdf_row = table.row()
                        for value in row:
                            pdf_row.cell(str(value))
                pdf.ln(6)

            pdf_path = out_directory / "AdEval_Report.pdf"
            pdf.output(str(pdf_path))

            # HTML

            html_tables = []
            for rows in tables:
                if not rows:
                    continue
                header = "".join(f"<th>{escape(str(v))}</th>" for v in rows[0])
                body = "".join(
                    "<tr>"
                    + "".join(f"<td>{escape(str(v))}</td>" for v in row)
                    + "</tr>"
                    for row in rows[1:]
                )
                html_tables.append(
                    f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
                )

            html = f"""<!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <title>Adeval Report</title>
        <style>
        body {{ font-family: Helvetica, Arial, sans-serif; margin: 2rem; }}
        table {{ border-collapse: collapse; margin-bottom: 1.5rem; }}
        th, td {{ border: 1px solid #999; padding: 4px 10px; text-align: left; }}
        th {{ background: #eee; }}
        </style>
        </head>
        <body>
        <h1>Adeval Report</h1>
        <hr>
        <p>Generated: {timestamp}</p>
        {"".join(html_tables)}
        </body>
        </html>
        """
            html_path = out_directory / "AdEval_Report.html"
            html_path.write_text(html, encoding="utf-8")

            console.print("[green]Operation successful")
            console.print(f"PDF Path: {pdf_path}")
            console.print(f"HTML Path: {html_path}")

    def __real_dir_size(self, root: Path) -> int:
        total = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in self.IGNORED_DIRS]
            for filename in filenames:
                file = Path(dirpath) / filename
                if file.is_file():
                    total += file.stat().st_size
        return total

    def __clear_reports(self):
        report_dirs = [
            self.OUTPUT_DIR / "garage-evaluation",
            self.OUTPUT_DIR / "openpcdet-evaluation",
            self.OUTPUT_DIR / "reports",
        ]

        freed_bytes = 0
        removed_any = False

        for folder in report_dirs:
            if not folder.is_dir():
                continue

            freed_bytes += self.__real_dir_size(folder)

            for entry in folder.iterdir():
                if entry.name == ".gitkeep":
                    continue
                removed_any = True
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()

        if removed_any:
            console.print(
                f"All reports have been cleared ({freed_bytes / 1024**3:.2f}GB freed)"
            )
        else:
            console.print("All reports are empty")
