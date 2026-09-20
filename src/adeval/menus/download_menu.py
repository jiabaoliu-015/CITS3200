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
            subprocess.run(
                [
                    sys.executable,
                    "-m", "py123d.script.run_conversion",
                    "dataset=av2-sensor-stream",
                    "dataset.parser.splits=[av2-sensor_val]",
                    "dataset.parser.downloader.num_logs=1",
                    "dataset.parser.downloader._target_=adeval.downloaders.Av2ProgressDownloader",
                ],
                check=True,
            )
            console.print("Completed")
        elif choice == "3":
            subprocess.run(
                [
                    sys.executable,
                    "-m", "py123d.script.run_conversion",
                    "dataset=nuplan-mini-stream",
                    "dataset.parser.splits=[nuplan-mini_val]",
                    "dataset.parser.downloader._target_=adeval.downloaders.NuplanProgressDownloader",
                ],
                check=True,
            )
            console.print("Completed")
        elif choice == "4":
            self.__clear_temp_dir()

        return MenuNames.DownloadMenu

    def __clear_temp_dir(self):
        try:
            temp_folder = Path(os.getenv("TMPDIR"))

            for item in temp_folder.iterdir():
                if item.name == ".gitkeep":
                    continue

                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

            console.print("TMPDIR has been cleared")

        except KeyError:
            console.print(
                'TMPDIR is not set. Please use [export TMPDIR="$HOME/CITS3200/tmp"] in local session or bashrc'
            )
