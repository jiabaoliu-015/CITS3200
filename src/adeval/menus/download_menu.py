import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.panel import Panel
from rich.prompt import Prompt

from adeval.console import console
from adeval.menus.base_menu import Menu
from adeval.menus.menu_names import MenuNames


class DownloadMenu(Menu):
    def run(self) -> str | None:
        console.print(Panel("Download Menu", style="bold cyan"))
        options = [
            "[4] Clear TMPDIR",
            "[3] Download simple nuplan",
            "[2] Download simple av2",
            "[1] Go to Main Menu",
            "[0] Exit",
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
            self._download_dataset(
                "av2-sensor-stream",
                "av2-sensor_val",
                "Av2ProgressDownloader",
                "dataset.parser.downloader.num_logs=1",
            )
        elif choice == "3":
            self._download_dataset(
                "nuplan-mini-stream",
                "nuplan-mini_val",
                "NuplanProgressDownloader",
            )
        elif choice == "4":
            self.__clear_temp_dir()

        return MenuNames.DownloadMenu

    def _download_dataset(
        self, dataset: str, split: str, downloader: str, *overrides: str
    ) -> None:
        # Resolve relative paths from the directory where adeval was started.
        # Without this default, py123d writes to a literal "None/logs" directory.
        data_root = Path(
            os.environ.get("PY123D_DATA_ROOT") or "py123d_data_root"
        ).expanduser().resolve()
        env = os.environ.copy()
        env["PY123D_DATA_ROOT"] = str(data_root)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "py123d.script.run_conversion",
                f"dataset={dataset}",
                f"dataset.parser.splits=[{split}]",
                f"dataset.parser.downloader._target_=adeval.downloaders.{downloader}",
                *overrides,
            ],
            check=True,
            env=env,
        )

        dataset_path = data_root / "logs" / split
        if dataset_path.is_dir():
            console.print("[green]Completed[/green]")
            console.print("Dataset path (converted):")
        else:
            console.print(
                "[yellow]Conversion exited, but the dataset directory was not found. "
                "Check the conversion logs. Expected path:[/yellow]"
            )
        console.print(
            str(dataset_path),
            markup=False,
            highlight=False,
            soft_wrap=True,
        )
        console.print("Output root (logs, maps and sensors):")
        console.print(str(data_root), markup=False, highlight=False, soft_wrap=True)

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
