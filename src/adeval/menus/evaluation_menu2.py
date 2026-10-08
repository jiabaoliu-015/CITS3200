import csv
import os
import pickle
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from adeval.console import console
from adeval.menus.base_menu import Menu
from adeval.menus.menu_names import MenuNames


class DatasetPaths(TypedDict):
    AV2: Path
    nuPlan: Path


class EvaluationMenu(Menu):
    def __init__(self):
        self.PROJECT_ROOT = Path(__file__).resolve().parents[3]
        self.DATASET_PATHS: DatasetPaths = {
            "AV2": self.PROJECT_ROOT / "third_party/OpenPCDet/data/argo2",
            "nuPlan": self.__py123d_data_root() / "nuplan",
        }

        # For AV2
        self.OPENPCDET_DIR = self.PROJECT_ROOT / "third_party/OpenPCDet"
        self.OPENPCDET_TOOLS_DIR = self.OPENPCDET_DIR / "tools"
        self.OPENPCDET_TEST_FILE = self.OPENPCDET_DIR / "tools/test.py"
        self.OPENPCDET_AV2_YAML = self.OPENPCDET_DIR / "tools/cfgs/argo2_models"
        self.AV2_MODEL_MAPPINGS = {"VoxelNeXt_Argo2": "cbgs_voxel01_voxelnext"}

        # For nuPlan
        self.VENV_GARAGE_PYTHON = self.PROJECT_ROOT / ".venv-garage/bin/python"
        self.VENV_GARAGE_CHECKPOINT = (
            self.PROJECT_ROOT
            / "tasks/nuplan_eval_task/checkpoints/resnet34_v0.1.0/model_0014.pth"
        )
        self.OUTPUT_DIR = self.PROJECT_ROOT / "outputs"

    def run(self) -> MenuNames | None:
        console.print(
            Panel(
                "Evaluation Menu",
                style="bold cyan",
            )
        )

        options = [
            "[0] Exit",
            "[1] Go to Main Menu",
            "[2] nuPlan Open-Loop Planning",
            "[3] 3D Object Detection Evaluation [GPU REQUIRED]",
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
            max_num_scenes = 0
            log_names = ""
            if Confirm.ask("Limit the number of scenes?", default=False):
                max_num_scenes = self.__ask_num_logs(
                    "How many scenes to limit?", high=147
                )

            if Confirm.ask("Evaluate specific logs?", default=False):
                user_logs = Prompt.ask(
                    "Log names (comma-separated, blank for all)", default=""
                )
                log_names = [
                    name.strip() for name in user_logs.split(",") if name.strip()
                ]

            self.__run_open_loop_evaluation(max_num_scenes, log_names)
        elif choice == "3":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if device == "cpu":
                console.print("GPU not found.")
                return MenuNames.EvaluationMenu

            dataset_name = self.__choose_dataset()
            self.__run_object_detection(dataset_name)

        return MenuNames.EvaluationMenu

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
            "[2] nuScene",
        ]
        for opt in options:
            console.print(opt)

        choice = Prompt.ask(
            "Select an option", choices=list(map(str, range(len(options))))
        )

        if choice == "0":
            return MenuNames.EvaluationMenu
        elif choice == "1":
            return "av2"
        elif choice == "2":
            return "nuscene"

    def __run_object_detection(self, dataset_name):
        if dataset_name == "av2":
            converted_paths = self.__convert_raw_models_into_openpcdet(dataset_name)
            batch_size = 1
            num_workers = 0
            if Confirm.ask("Increase the batch size?", default=False):
                batch_size = self.__ask_num_logs("Batch size: ", high=8)
            if Confirm.ask("Increase the number of workers?", default=False):
                num_workers = self.__ask_num_logs("Num of workers: ", low=0, high=8)

            build_cmd = [
                sys.executable,
                "-m",
                "pcdet.datasets.argo2.argo2_dataset",
                f"--root_path={self.DATASET_PATHS['AV2'] / 'sensor'!s}",
                f"--output_dir={self.DATASET_PATHS['AV2']!s}",
            ]

            run_cmds = []
            for yaml_path in self.OPENPCDET_AV2_YAML.iterdir():
                for model_path in converted_paths:
                    if self.AV2_MODEL_MAPPINGS[model_path.stem] == yaml_path.stem:
                        cmd = [
                            sys.executable,
                            str(self.OPENPCDET_TEST_FILE),
                            f"--cfg_file={yaml_path.relative_to(self.OPENPCDET_TOOLS_DIR)}",
                            f"--ckpt={model_path}",
                            f"--batch_size={batch_size}",
                            f"--workers={num_workers}",
                            "--set",
                            "DATA_CONFIG.DATA_PATH",
                            str(self.DATASET_PATHS["AV2"]),
                        ]
                        run_cmds.append((model_path.stem, cmd))

            progress = Progress(
                SpinnerColumn(finished_text="[green]✓"),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )

            TOTAL_RE = re.compile(r"Total samples for .+? dataset: (\d+)")
            EVAL_RE = re.compile(r"eval:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)")
            SAVED_RE = re.compile(r"Result is saved to (\S+)")
            METRICS_HEADER_RE = re.compile(r"\bAP\s+ATE\s+ASE\s+AOE\b")
            METRICS_ROW_RE = re.compile(
                r"^([A-Z_]+)\s+(-?\d+(?:\.\d+)?(?:\s+-?\d+(?:\.\d+)?)*)$"
            )
            METRICS_HEADER_CONT_RE = re.compile(r"[A-Z]+(?:\s+[A-Z]+)*")

            output_paths = []
            metrics_by_model: dict[str, dict[str, dict[str, float]]] = {}

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            # The build cmd
            with subprocess.Popen(
                build_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as proc:
                for line in proc.stdout:
                    console.print(
                        line.rstrip(),
                        markup=False,
                        highlight=False,
                        style="dim",
                    )

            if proc.returncode:
                raise subprocess.CalledProcessError(proc.returncode, build_cmd)

            # The run cmds
            with progress:
                for name, cmd in run_cmds:
                    task = progress.add_task(f"Loading {name}", total=None)
                    samples = 1
                    output_path = None

                    # Per-model metrics table state
                    metrics_header: list[str] = []
                    metrics_rows: list[list] = []
                    capturing_metrics = False

                    with subprocess.Popen(
                        cmd,
                        env=env,
                        cwd=self.OPENPCDET_DIR / "tools",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    ) as proc:
                        for line in proc.stdout:
                            stripped = line.strip()

                            # --- Metrics table capture (doesn't skip printing) ---
                            if not capturing_metrics:
                                if hm := METRICS_HEADER_RE.search(stripped):
                                    # Drop the timestamp/INFO prefix, keep the column names
                                    metrics_header = stripped[hm.start() :].split()
                                    metrics_rows = []
                                    capturing_metrics = True
                            elif rm := METRICS_ROW_RE.match(stripped):
                                row_name = rm.group(1)
                                metrics_rows.append(
                                    [row_name, *map(float, rm.group(2).split())]
                                )
                                if row_name == "AVERAGE_METRICS":
                                    capturing_metrics = False
                            elif not metrics_rows and METRICS_HEADER_CONT_RE.fullmatch(
                                stripped
                            ):
                                # Header wrapped onto the next line (e.g. "CDS")
                                metrics_header.extend(stripped.split())

                            # --- Progress handling ---
                            if m := EVAL_RE.search(line):
                                done, samples = map(int, m.groups())
                                progress.update(
                                    task,
                                    completed=done,
                                    total=samples,
                                    description=f"Evaluating {name}",
                                )
                                continue  # the bar replaces tqdm's own output
                            elif m := TOTAL_RE.search(line):
                                samples = int(m.group(1))
                                progress.update(
                                    task,
                                    total=samples,
                                    description=f"Evaluating {name}",
                                )
                            elif "Convert predictions to Argoverse 2 format" in line:
                                t = next(t for t in progress.tasks if t.id == task)
                                t.total = None
                                t.finished_time = None
                                progress.update(task, description=f"Scoring {name}")
                            elif m := SAVED_RE.search(line):
                                output_path = m.group(1)

                            progress.console.print(
                                line.rstrip(),
                                markup=False,
                                highlight=False,
                                style="dim",
                            )

                    if proc.returncode == 0:
                        progress.update(
                            task,
                            completed=samples,
                            total=samples,
                            description=f"[green]Evaluation {name} complete",
                        )
                        output_paths.append((name, output_path))
                        metrics_by_model[name] = {
                            row_name: dict(zip(metrics_header, values))
                            for row_name, *values in metrics_rows
                        }
                    else:
                        progress.update(task, description=f"[red]{name} failed")
                        raise subprocess.CalledProcessError(proc.returncode, cmd)

            # console.print(output_paths)
            # console.print(metrics_by_model)
            out_directory = self.__create_and_display_metrics(
                output_paths, metrics_by_model
            )
            console.print(f"Output path: {out_directory!s}")

    def __create_and_display_metrics(self, output_paths, metrics_by_model):
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        out_directory = Path(self.OUTPUT_DIR) / f"openpcdet-evaluation/av2/{timestamp}"
        out_directory.mkdir(parents=True, exist_ok=False)

        for name, path in output_paths:
            data = self.__read_pickle_file(Path(path) / "result.pkl")
            shutil.copy2(
                Path(path) / "result.pkl", out_directory / f"{name}_results.pkl"
            )
            self.__save_model_metrics(
                metrics_by_model, out_directory / f"{name}_results.csv"
            )

            table1 = self.__display_model_metrics_from_pkl_data(data)
            table2 = self.__display_model_metrics_from_csv(
                name, out_directory / f"{name}_result.csv"
            )

            console.print(table1)
            console.print(table2)

        return out_directory

    def __save_model_metrics(
        self,
        metrics_by_model,
        output_path: Path,
        summary_key="AVERAGE_METRICS",
    ) -> None:

        # Collect every metric name across all models, keeping first-seen order
        metric_names: list[str] = []
        for categories in metrics_by_model.values():
            for metrics in categories.values():
                for m in metrics:
                    if m not in metric_names:
                        metric_names.append(m)

        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["category", *metric_names])

            for categories in metrics_by_model.values():
                ordered = sorted(k for k in categories if k != summary_key)
                if summary_key in categories:
                    ordered.append(summary_key)  # keep the average last

                for category in ordered:
                    metrics = categories[category]
                    writer.writerow(
                        [
                            category,
                            *(metrics.get(m, "") for m in metric_names),
                        ]
                    )

    def __display_model_metrics_from_csv(
        self,
        model_name,
        csv_path: Path,
        summary_key="AVERAGE_METRICS",
    ) -> Table:
        with Path(csv_path).open(newline="") as f:
            reader = csv.DictReader(f)
            metric_names = [
                c for c in reader.fieldnames or [] if c not in ("model", "category")
            ]
            categories = {
                row["category"]: row for row in reader if row["model"] == model_name
            }

        if not categories:
            raise ValueError(f"no rows for model '{model_name}' in {csv_path}")

        def fmt(value: str) -> str:
            return f"{float(value):.3f}" if value else "-"

        table = Table(title=model_name, header_style="bold cyan")
        table.add_column("Category", style="bold")
        for metric in metric_names:
            table.add_column(metric, justify="right")

        for name in sorted(k for k in categories if k != summary_key):
            table.add_row(name, *(fmt(categories[name][m]) for m in metric_names))

        if summary_key in categories:
            table.add_section()
            table.add_row(
                "AVERAGE",
                *(fmt(categories[summary_key][m]) for m in metric_names),
                style="bold yellow",
            )
        return table

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
        table.add_column("% of dets", footer="100%", min_width=9, **num)
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

    def __read_pickle_file(self, path: Path):
        # Prevent bad pkl and only load numpy
        class SafeUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.split(".")[0] in ("numpy", "collections"):
                    return super().find_class(module, name)
                raise pickle.UnpicklingError(f"blocked global {module}.{name}")

        with open(path, "rb") as f:
            return SafeUnpickler(f).load()

    def __convert_raw_models_into_openpcdet(self, dataset_name) -> list[Path]:
        models_dir = self.PROJECT_ROOT / f"tasks/object_detection/{dataset_name}"
        out_dir = self.PROJECT_ROOT / f"tasks/object_detection/{dataset_name}/converted"
        out_dir.mkdir(parents=True, exist_ok=True)

        converted_paths = []

        for ckpt_path in sorted(models_dir.glob("*.pth")):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            if isinstance(ckpt, dict) and "model_state" in ckpt:
                wrapped = ckpt  # already OpenPCDet format
            else:
                wrapped = {"model_state": ckpt, "epoch": None}

            out_path = out_dir / ckpt_path.name
            torch.save(wrapped, out_path)
            converted_paths.append(out_path)

        return converted_paths

    def __run_open_loop_evaluation(self, max_num_scenes=0, log_names=""):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu"

        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        run_directory = Path(self.OUTPUT_DIR) / f"garage-evaluation/{timestamp}"
        run_directory.mkdir(parents=True, exist_ok=False)

        log_names_value = log_names or "null"
        max_num_scenes_value = max_num_scenes if max_num_scenes > 0 else "null"

        env = os.environ.copy()
        env["PY123D_GARAGE_DATA_ROOT"] = self.__py123d_data_root()

        cmd = [
            str(self.VENV_GARAGE_PYTHON),
            "-m",
            "py123d_garage.evaluation.open_loop.evaluate",
            f"hydra.run.dir={run_directory}",
            f"policy_config.evaluation_checkpoint_file={self.VENV_GARAGE_CHECKPOINT}",
            "+offline_data_sources/ltf_nuplan@benchmark_offline_data_sources.nuplan_test=nuplan_test",
            "benchmark_offline_data_sources.nuplan_test.cache_root=null",
            f"benchmark_offline_data_sources.nuplan_test.data_root={self.DATASET_PATHS['nuPlan']}",
            f"++benchmark_offline_data_sources.nuplan_test.garage_scene_filter.log_names={log_names_value}",
            "++benchmark_offline_data_sources.nuplan_test.garage_scene_filter.shuffle=false",
            f"++benchmark_offline_data_sources.nuplan_test.garage_scene_filter.max_num_scenes={max_num_scenes_value}",
            "save_visualizations=true",
            f"parallelization_config.device={device}",
        ]

        if device == "cpu":
            cmd.extend(
                [
                    "parallelization_config.max_workers=1",
                    "parallelization_config.inference_batch_size=1",
                    "parallelization_config.num_shards=1",
                    "parallelization_config.shard_index=0",
                ]
            )

        progress = Progress(
            SpinnerColumn(finished_text="[green]✓"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        log_path = run_directory / "evaluation.log"

        with progress, log_path.open("w") as log:
            task = progress.add_task("Getting ready to evaluate", total=None)

            with subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as proc:
                for line in proc.stdout:  # text=True also splits on tqdm's "\r"
                    line = line.strip()
                    if not line:
                        continue

                    # tqdm's "Inference: 50%|...| 2/4" redraws drive the bar
                    if m := re.search(r"Inference:\s+\d+%\|.*?\|\s*(\d+)/(\d+)", line):
                        progress.update(task, completed=int(m[1]), total=int(m[2]))
                        continue

                    log.write(line + "\n")
                    is_problem = re.search(
                        r"\]\[(WARNING|ERROR|CRITICAL)\]|Traceback", line
                    )
                    progress.console.print(
                        line,
                        markup=False,
                        highlight=False,
                        style="yellow" if is_problem else "dim",
                    )

                    if "Path where all results are stored" in line:
                        progress.update(task, description="Building scenes")
                    elif m := re.search(r"(\d+) scenes passed the filter", line):
                        progress.update(task, description=f"Found {m[1]} scenes")
                    elif "Running Inference" in line:
                        progress.reset(
                            task, description="Running inference", total=None
                        )
                    elif "Running Scoring" in line:
                        progress.reset(task, description="Scoring", total=None)
                    elif "Saved shard results" in line:
                        progress.update(task, description="Merging results")

            if proc.returncode == 0:
                progress.update(
                    task, description="Evaluation complete", total=1, completed=1
                )
            else:
                progress.update(task, description="[red]Evaluation failed")
                progress.stop_task(task)

        if proc.returncode != 0:
            console.print(f"[red]Full log:[/] {log_path}")
            raise subprocess.CalledProcessError(proc.returncode, cmd)

        console.print(f"Output Path: {run_directory / 'results.csv'}")

    @staticmethod
    def __py123d_data_root() -> Path:
        return (
            Path(os.environ.get("PY123D_DATA_ROOT") or "py123d_data_root")
            .expanduser()
            .resolve()
        )

    def __ask_num_logs(self, question, low: int = 1, high: int = 150) -> int:
        while True:
            value = IntPrompt.ask(
                f"{question} [prompt.choices]\\[{low}-{high}][/prompt.choices]",
                default=1,
            )
            if low <= value <= high:
                return value
            console.print(f"[red]Please enter a number between {low} and {high}.")
