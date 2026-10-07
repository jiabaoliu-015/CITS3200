import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TypedDict

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

            env = os.environ.copy()
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
                            f"--cfg_file={yaml_path}",
                            f"--ckpt={model_path}",
                            f"--batch_size={batch_size}",
                            f"--workers={num_workers}",
                            "--set",
                            "DATA_CONFIG.DATA_PATH",
                            str(self.DATASET_PATHS["AV2"]),
                        ]
                        run_cmds.append(cmd)

            subprocess.run(build_cmd, check=True, env=env)
            for cmd in run_cmds:
                subprocess.run(
                    cmd, check=True, env=env, cwd=self.OPENPCDET_DIR / "tools"
                )

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
