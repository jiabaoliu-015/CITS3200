import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
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
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from adeval.console import console
from adeval.menus.base_menu import DatasetPaths, Menu
from adeval.menus.menu_names import MenuNames


class DownloadMenu(Menu):
    def __init__(self):
        self.PROJECT_ROOT = Path(__file__).resolve().parents[3]
        self.NUPLAN_LOG_INFO = self.__py123d_data_root() / "nuplan.parquet"
        self.DATASET_PATHS: DatasetPaths = {
            "AV2": self.PROJECT_ROOT / "third_party/OpenPCDet/data/argo2",
            "nuPlan": self.__py123d_data_root() / "nuplan",
            # "nuScenes": self.PROJECT_ROOT / "third_party/OpenPCDet/data/nuscenes",
        }
        self.IGNORED_DIRS = ["ImageSets", ".cache"]

    def run(self) -> str | None:
        console.print(Panel("Download Menu", style="bold cyan"))
        options = [
            "[0] Exit",
            "[1] Go to Main Menu",
            "[2] Download AV2",
            "[3] Download nuPlan",
            "[4] Check downloaded dataset",
            "[5] Clear TMPDIR",
            "[6] Clear AV2",
            "[7] Clear nuPlan",
            # "[8] Download nuScenes",
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
            num_logs = self.__ask_num_logs("How many logs to download?")
            self.__download_av2(num_logs)
        elif choice == "3":
            num_logs = self.__ask_num_logs("How many logs to download?", high=147)
            logs = self.__get_valid_nuplan_logs()
            self.__download_nuplan(logs[:num_logs])
        elif choice == "4":
            self.__check_downloaded_datasets()
        elif choice == "5":
            self.__clear_temp_dir()
        elif choice == "6":
            self.__clear_av2()
        elif choice == "7":
            self.__clear_nuplan()
        # elif choice == "8":
        #     if (
        #         os.environ.get("NUSCENES_EMAIL") is None
        #         or os.environ.get("NUSCENES_PASSWORD") is None
        #     ):
        #         console.print("nuScene Email / Password is not set")

        #     size = Prompt.ask(
        #         "Select nuscene size [bold](s)[/]mall, [bold](m)[/]edium or [bold](l)[/]arge",
        #         choices=["s", "m", "l"],
        #         default="s",
        #         case_sensitive=False,
        #     )
        #     self.__download_nuscenes(size)

        return MenuNames.DownloadMenu

    def __download_av2(self, num_logs=1):
        av2_path = self.DATASET_PATHS["AV2"]
        av2_path.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["AV2_DATA_ROOT"] = str(av2_path)
        env["PYTHONUNBUFFERED"] = "1"
        cli = Path(sys.executable).parent / "py123d-download"

        cmd = [
            str(cli),
            "dataset=av2-sensor",
            "dataset.downloader.splits=[av2-sensor_val]",
            f"dataset.downloader.num_logs={num_logs}",
        ]

        TOTAL_RE = re.compile(r"AV2 sensor objects:\s+(\d+)")
        DONE_RE = re.compile(r"downloaded (\d+) / (\d+) objects")

        progress = Progress(
            SpinnerColumn(finished_text="[green]✓"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            speed_estimate_period=3600,
        )

        with progress:
            task = progress.add_task("Preparing Download", total=None)

            with subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as proc:
                for line in proc.stdout:
                    if m := DONE_RE.search(line):
                        done, total = map(int, m.groups())
                        progress.update(
                            task,
                            completed=done,
                            total=total,
                            description="Downloading AV2",
                        )
                    elif m := TOTAL_RE.search(line):
                        progress.update(
                            task, total=int(m.group(1)), description="Downloading AV2"
                        )

                    progress.console.print(
                        line.rstrip(), markup=False, highlight=False, style="dim"
                    )

            if proc.returncode == 0:
                progress.update(
                    task,
                    description="[green]Download complete",
                )
            else:
                progress.update(task, description="[red]Download failed")

        if proc.returncode == 0:
            # OPENPCDET NEEDS EMPTY TRAIN FOLDER
            train_dir = av2_path / "sensor/train"
            train_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"Output path: {av2_path!s}")
        else:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

    def __download_nuplan(self, log_names: list[str]):
        nuplan_path = self.DATASET_PATHS["nuPlan"]
        nuplan_path.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        # env["PYTHONUNBUFFERED"] = "1"
        # env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        cli = self.PROJECT_ROOT / ".venv-garage/bin/hf"

        cmd = [
            str(cli),
            "download",
            "kesai-labs/nuplan",
            "--repo-type=dataset",
            f"--local-dir={nuplan_path}",
            *[f"--include=logs/nuplan_test/{log}/*" for log in log_names],
            "--include=maps/*",
            "--exclude=*/camera.pcam_b0.arrow",
            "--exclude=*/camera.pcam_l1.arrow",
            "--exclude=*/camera.pcam_l2.arrow",
            "--exclude=*/camera.pcam_r1.arrow",
            "--exclude=*/camera.pcam_r2.arrow",
            "--exclude=*/lidar.*",
        ]

        subprocess.run(cmd, env=env, check=True)

        console.print(f"Output path: {nuplan_path!s}")

    # def __download_nuscenes(self, size):
    #     PRESETS = {"s": "mini", "m": "trainval_one", "l": "full"}
    #     nuscenes_path = self.DATASET_PATHS["nuScenes"]
    #     nuscenes_path.mkdir(parents=True, exist_ok=True)

    #     env = os.environ.copy()
    #     env["NUSCENES_DATA_ROOT"] = str(nuscenes_path)
    #     env["PYTHONUNBUFFERED"] = "1"
    #     cli = Path(sys.executable).parent / "py123d-download"

    #     cmd = [
    #         str(cli),
    #         "dataset=nuscenes",
    #         f"+downloader.preset={PRESETS[size]}",
    #     ]

    #     TOTAL_RE = re.compile(r"nuScene sensor objects:\s+(\d+)")
    #     DONE_RE = re.compile(r"downloaded (\d+) / (\d+) objects")

    #     progress = Progress(
    #         SpinnerColumn(finished_text="[green]✓"),
    #         TextColumn("[bold blue]{task.description}"),
    #         BarColumn(),
    #         MofNCompleteColumn(),
    #         TimeElapsedColumn(),
    #         TimeRemainingColumn(),
    #         speed_estimate_period=3600,
    #     )

    #     with progress:
    #         task = progress.add_task("Preparing Download", total=None)

    #         with subprocess.Popen(
    #             cmd,
    #             env=env,
    #             stdout=subprocess.PIPE,
    #             stderr=subprocess.STDOUT,
    #             text=True,
    #             bufsize=1,
    #         ) as proc:
    #             for line in proc.stdout:
    #                 if m := DONE_RE.search(line):
    #                     done, total = map(int, m.groups())
    #                     progress.update(
    #                         task,
    #                         completed=done,
    #                         total=total,
    #                         description="Downloading nuScene",
    #                     )
    #                 elif m := TOTAL_RE.search(line):
    #                     progress.update(
    #                         task,
    #                         total=int(m.group(1)),
    #                         description="Downloading nuScene",
    #                     )

    #                 progress.console.print(
    #                     line.rstrip(), markup=False, highlight=False, style="dim"
    #                 )

    #         if proc.returncode == 0:
    #             progress.update(
    #                 task,
    #                 description="[green]Download complete",
    #             )
    #         else:
    #             progress.update(task, description="[red]Download failed")

    #     if proc.returncode == 0:
    #         console.print(f"Output path: {nuscenes_path!s}")
    #     else:
    #         raise subprocess.CalledProcessError(proc.returncode, cmd)

    def __ask_num_logs(self, question, low: int = 1, high: int = 150) -> int:
        while True:
            value = IntPrompt.ask(
                f"{question} [prompt.choices]\\[{low}-{high}][/prompt.choices]",
                default=1,
            )
            if low <= value <= high:
                return value
            console.print(f"[red]Please enter a number between {low} and {high}.")

    @staticmethod
    def __py123d_data_root() -> Path:
        return (
            Path(os.environ.get("PY123D_DATA_ROOT") or "py123d_data_root")
            .expanduser()
            .resolve()
        )

    def __clear_temp_dir(self):
        temp_dir = os.environ.get("TMPDIR")
        if not temp_dir:
            console.print(
                "TMPDIR is not set. Set it to your project temporary directory "
                "(e.g. ./tmp) before clearing it."
            )
            return
        try:
            temp_folder = Path(temp_dir).expanduser()

            for item in temp_folder.iterdir():
                if item.name == ".gitkeep":
                    continue

                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

            console.print("TMPDIR has been cleared")

        except FileNotFoundError:
            console.print("TMPDIR does not exist; nothing to clear.")

    def __get_valid_nuplan_logs(self):
        df = pd.read_parquet(self.NUPLAN_LOG_INFO)
        logs = df[(df["split"] == "nuplan_test") & (df["has_sensors"])]
        return logs["log_name"].sort_values().tolist()

    def __real_dir_size(self, root: Path) -> int:
        total = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in self.IGNORED_DIRS]
            for filename in filenames:
                file = Path(dirpath) / filename
                if file.is_file():
                    total += file.stat().st_size
        return total

    def __check_downloaded_datasets(self) -> None:
        table = Table(title="Downloaded Datasets")
        table.add_column("Dataset")
        table.add_column("Size (GB)", justify="right")
        table.add_column("Path", overflow="fold")

        for name, path in self.DATASET_PATHS.items():
            size_bytes = self.__real_dir_size(path) if path.is_dir() else 0
            if size_bytes > 0:
                size = f"{size_bytes / 1024**3:.2f}"
            else:
                size = "[red]Not Found / Not Downloaded[/red]"
            table.add_row(name, size, str(path))

        console.print(table)
        Prompt.ask(
            "[dim]Press Enter to return to download menu[/dim]",
            default="",
            show_default=False,
        )

    def __clear_av2(self):
        path = self.DATASET_PATHS["AV2"] / "sensor"
        size_bytes = self.__real_dir_size(path) if path.is_dir() else 0
        if path.exists():
            shutil.rmtree(path)
            console.print(
                f"AV2 Downloads have been cleared ({size_bytes / 1024**3:.2f} GB freed)"
            )
            return

        console.print("AV2 Downloads are empty")

    def __clear_nuplan(self):
        path = self.DATASET_PATHS["nuPlan"]
        size_bytes = self.__real_dir_size(path) if path.is_dir() else 0
        if path.exists():
            shutil.rmtree(path)
            console.print(
                f"nuPlan Downloads have been cleared ({size_bytes / 1024**3:.2f} GB freed)"
            )
            return

        console.print("nuPlan Downloads are empty")
