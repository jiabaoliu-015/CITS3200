from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict


class Menu(ABC):
    """
    Base class for a menu screen. run() returns the name of the next menu to go to, or None to exit the app.
    """

    @abstractmethod
    def run(self) -> str | None:
        pass


class DatasetPaths(TypedDict):
    AV2: Path
    nuPlan: Path
    # nuScenes: Path
