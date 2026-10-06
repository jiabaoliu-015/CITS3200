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
from adeval.menus.base_menu import Menu
from adeval.menus.menu_names import MenuNames


class DownloadMenu(Menu):
    def run(self) -> str | None:
        console.print(Panel("Download Menu", style="bold cyan"))
        options = [
            "[0] Exit",
            "[1] Go to Main Menu",
            "[2] Download AV2",
            "[3] Download nuPlan",
            "[4] Check downloaded dataset",
            "[5] Clear TMPDIR",
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
            num_logs = self.__ask_num_logs()
            self.__download_av2(num_logs)
        elif choice == "3":
            num_logs = self.__ask_num_logs(high=147)
            logs = self.__get_valid_nuplan_logs()
            self.__download_nuplan(logs[:num_logs])
        elif choice == "4":
            self.__check_downloaded_datasets()
        elif choice == "5":
            self.__clear_temp_dir()

        return MenuNames.DownloadMenu

    def __download_av2(self, num_logs=1):
        av2_path = self.__py123d_data_root() / "av2"
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
            progress.console.print(f"Output path: {av2_path!s}")
        else:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

    def __download_nuplan(self, log_names: list[str]):
        nuplan_path = self.__py123d_data_root() / "nuplan"
        nuplan_path.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        # env["PYTHONUNBUFFERED"] = "1"
        # env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        cli = Path(sys.prefix).parent / ".venv-garage" / "bin" / "hf"

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

    def __download_nuscene(self):
        pass

    def __ask_num_logs(self, low: int = 1, high: int = 150) -> int:
        while True:
            value = IntPrompt.ask(
                f"How many logs to download? [{low}-{high}]", default=1
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
        df = pd.read_parquet(self.__py123d_data_root() / "nuplan.parquet")
        logs = df[(df["split"] == "nuplan_test") & (df["has_sensors"])]
        return logs["log_name"].sort_values().tolist()

    def __check_downloaded_datasets(self) -> None:
        root = Path(self.__py123d_data_root())
        paths = {
            "AV2": root / "av2",
            "nuPlan": root / "nuplan",
        }

        table = Table(title="Downloaded Datasets")
        table.add_column("Dataset")
        table.add_column("Size (GB)", justify="right")
        table.add_column("Path", overflow="fold")

        for name, path in paths.items():
            if path.is_dir():
                size_bytes = sum(
                    f.stat().st_size for f in path.rglob("*") if f.is_file()
                )
                size = f"{size_bytes / 1024**3:.2f}"
            else:
                size = "[red]Not Found / Not Downloaded[/red]"
            table.add_row(name, size, str(path))

        console.print(table)
        Prompt.ask(
            "[dim]Press Enter to return to menu[/dim]", default="", show_default=False
        )
