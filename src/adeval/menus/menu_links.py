from adeval.menus.base_menu import Menu
from adeval.menus.download_menu2 import DownloadMenu
from adeval.menus.evaluation_menu2 import EvaluationMenu
from adeval.menus.main_menu import MainMenu
from adeval.menus.menu_names import MenuNames
from adeval.menus.report_menu import ReportMenu

MENUS: dict[MenuNames, Menu] = {
    MenuNames.MainMenu: MainMenu(),
    MenuNames.DownloadMenu: DownloadMenu(),
    MenuNames.EvaluationMenu: EvaluationMenu(),
    MenuNames.ReportMenu: ReportMenu(),
}
