import time

from rich.panel import Panel
from rich.prompt import Prompt

from adeval.console import console
from adeval.menus.base_menu import Menu


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
