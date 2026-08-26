import os
import shutil
import subprocess
from pathlib import Path

from rich.panel import Panel
from rich.prompt import Prompt

from adeval.console import console
from adeval.menus.base_menu import Menu
from adeval.menus.menu_keys import DownloadMenuName, MainMenuName


class DownloadMenu(Menu):
    def run(self) -> str | None:
        console.print(Panel("Download Menu", style="bold cyan"))
        console.print("[4] Clear TMPDIR")
        console.print("[3] Download simple nuplan")
        console.print("[2] Download simple av2")
        console.print("[1] Go to Main Menu")
        console.print("[0] Exit")

        choice = Prompt.ask("Select an option", choices=["0", "1", "2", "3", "4"])

        if choice == "0":
            return None
        if choice == "1":
            return MainMenuName
        elif choice == "2":
            with console.status("av2 download + conversion\n"):
                subprocess.run(
                    [
                        "py123d-conversion",
                        "dataset=av2-sensor-stream",
                        "dataset.parser.splits=[av2-sensor_val]",
                        "dataset.parser.downloader.num_logs=1",
                    ],
                    check=True,
                )
            console.print("Completed")
        elif choice == "3":
            with console.status("nuplan download + conversion\n"):
                subprocess.run(
                    [
                        "py123d-conversion",
                        "dataset=nuplan-mini-stream",
                        "dataset.parser.splits=[nuplan-mini_val]",
                    ],
                    check=True,
                )
                console.print("Completed")
        elif choice == "4":
            self.__clear_temp_dir()

        return DownloadMenuName

    def __clear_temp_dir():
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
