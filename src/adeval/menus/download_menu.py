import os
import shutil
import subprocess
from pathlib import Path

from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from adeval.console import console
from adeval.menus.base_menu import Menu
from adeval.menus.menu_names import MenuNames

# download_menu.py -> repo root is three parents up.
MOTION_PLANNING_DIR = Path(__file__).resolve().parents[3] / "motion_planning"

# Matches the nuplan_mini entry in motion_planning/datasets/registry.yaml
# (data_path: data/nuplan/dataset/nuplan-v1.1/splits/mini,
#  map_path: data/nuplan/dataset/maps) so the benchmark can find it with no
# registry edits.
NUPLAN_DATA_ROOT = MOTION_PLANNING_DIR / "data" / "nuplan" / "dataset"


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
            with console.status("nuplan download (mini: train+val+test, + maps)\n"):
                subprocess.run(
                    [
                        "py123d-download",
                        "dataset=nuplan",
                        f"dataset.downloader.output_dir={NUPLAN_DATA_ROOT}",
                        "dataset.downloader.splits=[nuplan-mini_train,nuplan-mini_val,nuplan-mini_test]",
                        "dataset.downloader.include_maps=true",
                    ],
                    check=True,
                )
            console.print("Completed")

            if Confirm.ask(
                "Run the motion_planning benchmark test now?", default=True
            ):
                subprocess.run(
                    ["./benchmark", "--dataset", "nuplan_mini", "--profile", "quick", "--yes"],
                    cwd=MOTION_PLANNING_DIR,
                    check=True,
                )
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
