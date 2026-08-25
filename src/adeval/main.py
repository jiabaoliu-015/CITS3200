from adeval.console import console
from adeval.menus.base_menu import Menu
from adeval.menus.download_menu import DownloadMenu
from adeval.menus.main_menu import MainMenu

MENUS: dict[str, Menu] = {"main_menu": MainMenu(), "download_menu": DownloadMenu()}


def run() -> None:
    current_menu = "main_menu1"
    try:
        while current_menu is not None:
            current_menu = MENUS[current_menu].run()
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted. Exiting.[/yellow]")
    except KeyError:
        console.print("[red]Invalid Menu Name. Exiting.[/red]")
        raise
    except Exception:
        console.print("[red]Something has gone wrong. Exiting.[/red]")
        raise
    else:
        console.print("[green]Bye bye. Exiting.[/green]")
