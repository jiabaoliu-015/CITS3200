import os
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from rich.panel import Panel
from rich.prompt import Prompt

from adeval.console import console
from adeval.menus.base_menu import Menu
from adeval.menus.menu_names import MenuNames


class ReportMenu(Menu):
    def __init__(self):
        self.PROJECT_ROOT = Path(__file__).resolve().parents[3]

    def run(self) -> MenuNames | None:
        console.print(
            Panel(
                "Report Menu",
                style="bold cyan",
            )
        )

        options = [
            "[0] Exit",
            "[1] Go to Main Menu",
            "[2] View Garage Evaluation Report",
            "[3] View Object Detection Evaluation Report",
            "[4] View HTML/PDF Report",
            "[5] Clear All Reports",
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

        return MenuNames.ReportMenu
