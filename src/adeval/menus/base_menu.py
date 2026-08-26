from abc import ABC, abstractmethod


class Menu(ABC):
    """
    Base class for a menu screen. run() returns the name of the next menu to go to, or None to exit the app.
    """

    @property
    @abstractmethod
    def menu_name(self) -> str:
        """
        Represents the menu name for linking in MENUS
        """

    @abstractmethod
    def run(self) -> str | None:
        pass
