import os
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from rich.panel import Panel
from rich.prompt import Prompt

from adeval.console import console
from adeval.menus.base_menu import Menu
from adeval.menus.menu_names import MenuNames


class EvaluationMenu(Menu):
    def __init__(self):
        self.PROJECT_ROOT = Path(__file__).resolve().parents[3]
        self.VENV_GARAGE_PYTHON = self.PROJECT_ROOT / ".venv-garage/bin/python"
        self.VENV_GARAGE_CHECKPOINT = (
            self.PROJECT_ROOT
            / "tasks/nuplan_eval_task/checkpoints/resnet34_v0.1.0/model_0014.pth"
        )
        self.OUTPUT_DIR = self.PROJECT_ROOT / "outputs"
        self.NUPLAN_DATASET_DIR = self.__py123d_data_root() / "nuplan"

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
            "[3] 3D Object Detection Evaluation",
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
            self.__run_open_loop_evaluation()

        return MenuNames.EvaluationMenu

    def __run_open_loop_evaluation(self, max_num_scenes=0, log_names=""):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu"

        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        run_directory = Path(self.OUTPUT_DIR) / f"garage-evaluation-{timestamp}"
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
            f"benchmark_offline_data_sources.nuplan_test.data_root={self.NUPLAN_DATASET_DIR}",
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

        subprocess.run(cmd, check=True, env=env)

    @staticmethod
    def __py123d_data_root() -> Path:
        return (
            Path(os.environ.get("PY123D_DATA_ROOT") or "py123d_data_root")
            .expanduser()
            .resolve()
        )
