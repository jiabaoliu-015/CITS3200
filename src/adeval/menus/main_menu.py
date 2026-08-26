from rich.panel import Panel
from rich.prompt import Prompt

from adeval.console import console
from adeval.menus.base_menu import Menu
from adeval.menus.menu_keys import DownloadMenuName, MainMenuName


class MainMenu(Menu):
    def run(self) -> str | None:
        console.print(Panel("Main Menu", style="bold cyan"))
        console.print("[1] Go to Download Menu")
        console.print("[0] Exit")

        choice = Prompt.ask("Select an option", choices=["0", "1"])

        if choice == "0":
            return None
        if choice == "1":
            return DownloadMenuName

        return MainMenuName
