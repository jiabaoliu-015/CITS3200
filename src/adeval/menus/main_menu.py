from rich.panel import Panel
from rich.prompt import Prompt

from adeval.console import console
from adeval.menus.base_menu import Menu
from adeval.menus.menu_names import MenuNames


class MainMenu(Menu):
    def run(self) -> str | None:
        console.print(Panel("Main Menu", style="bold cyan"))
        options = [
            "[0] Exit",
            "[1] Go to Download Menu",
            "[2] Go to Evaluation Menu",
            "[3] Go to Report Menu",
        ]
        for opt in options:
            console.print(opt)

        choice = Prompt.ask(
            "Select an option", choices=list(map(str, range(len(options))))
        )

        if choice == "0":
            return None
        elif choice == "1":
            return MenuNames.DownloadMenu
        elif choice == "2":
            return MenuNames.EvaluationMenu
        elif choice == "3":
            return MenuNames.ReportMenu

        return MenuNames.MainMenu
