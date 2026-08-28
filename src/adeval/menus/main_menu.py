from rich.panel import Panel
from rich.prompt import Prompt

from adeval.console import console
from adeval.menus.base_menu import Menu
from adeval.menus.menu_names import MenuNames


class MainMenu(Menu):
    def run(self) -> str | None:
        console.print(Panel("Main Menu", style="bold cyan"))
        options = [
            "[1] Go to Download Menu",
            "[0] Exit",
        ]
        for opt in options:
            console.print(opt)

        choice = Prompt.ask(
            "Select an option", choices=list(map(str, range(len(options))))
        )

        if choice == "0":
            return None
        if choice == "1":
            return MenuNames.DownloadMenu

        return MenuNames.MainMenu
