from adeval.menus.base_menu import Menu
from adeval.menus.download_menu import DownloadMenu
from adeval.menus.main_menu import MainMenu
from adeval.menus.menu_names import MenuNames
from adeval.menus.nuplan_menu import NuplanMenu

MENUS: dict[MenuNames, Menu] = {
    MenuNames.MainMenu: MainMenu(),
    MenuNames.DownloadMenu: DownloadMenu(),
    MenuNames.NuplanMenu: NuplanMenu(),
}
