from adeval.console import console
from adeval.menus.menu_links import MENUS
from adeval.menus.menu_names import MenuNames


def run() -> None:
    current_menu = MenuNames.MainMenu
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
