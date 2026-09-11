from adeval.menus.base_menu import Menu
from adeval.menus.download_menu import DownloadMenu
from adeval.menus.evaluation_menu import EvaluationMenu
from adeval.menus.main_menu import MainMenu
from adeval.menus.menu_names import MenuNames

MENUS: dict[MenuNames, Menu] = {
    MenuNames.MainMenu: MainMenu(),
    MenuNames.DownloadMenu: DownloadMenu(),
    MenuNames.EvaluationMenu: EvaluationMenu(),
}
