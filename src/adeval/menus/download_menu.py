import time

from rich.panel import Panel
from rich.prompt import Prompt

from adeval.console import console
from adeval.menus.base_menu import Menu

# import subprocess
# with console.status("av2 download\n"):
#     subprocess.run(
#         [
#             "py123d-conversion",
#             "dataset=av2-sensor-stream",
#             "dataset.parser.splits=[av2-sensor_val]",
#             "dataset.parser.downloader.num_logs=1",
#         ],
#         check=True,
#     )

# with console.status("nuplan download\n"):
#     subprocess.run(
#         [
#             "py123d-conversion",
#             "dataset=nuplan-mini-stream",
#             "dataset.parser.splits=[nuplan-mini_val]",
#         ],
#         check=True,
#     )


class DownloadMenu(Menu):
    def run(self) -> str | None:
        console.print(Panel("Main Menu", style="bold cyan"))
        console.print("[2] Simulate download")
        console.print("[1] Go to Main Menu")
        console.print("[0] Exit")

        choice = Prompt.ask("Select an option", choices=["0", "1", "2"])

        if choice == "0":
            return None
        if choice == "1":
            return "main_menu"
        elif choice == "2":
            with console.status("Simulate Download"):
                time.sleep(3)

            console.print("Done")

        return "download_menu"
