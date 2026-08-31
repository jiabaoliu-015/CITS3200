import subprocess
from pathlib import Path

from rich.panel import Panel

from adeval.console import console
from adeval.menus.base_menu import Menu
from adeval.menus.menu_names import MenuNames

# src/adeval/menus/nuplan_menu.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[3]
MOTION_PLANNING_DIR = REPO_ROOT / "motion_planning"

# Matches the nuplan_mini entry in motion_planning/datasets/registry.yaml
# (data_path: data/nuplan/dataset/nuplan-v1.1/splits/mini,
#  map_path: data/nuplan/dataset/maps) so the benchmark can find it with no
# registry edits.
NUPLAN_DATA_ROOT = MOTION_PLANNING_DIR / "data" / "nuplan" / "dataset"


class NuplanMenu(Menu):
    def run(self) -> str | None:
        console.print(Panel("Nuplan: Download & Test", style="bold cyan"))

        console.print("Downloading nuPlan mini (train+val+test) + maps via py123d...")
        subprocess.run(
            [
                "py123d-download",
                "dataset=nuplan",
                f"dataset.downloader.output_dir={NUPLAN_DATA_ROOT}",
                "dataset.downloader.splits=[nuplan-mini_train,nuplan-mini_val,nuplan-mini_test]",
                "dataset.downloader.include_maps=true",
            ],
            check=True,
        )
        console.print("[green]Download complete.[/green]")

        console.print("Running motion_planning benchmark (IDMPlanner, nuplan_mini, quick profile)...")
        subprocess.run(
            ["./benchmark", "--dataset", "nuplan_mini", "--profile", "quick", "--yes"],
            cwd=MOTION_PLANNING_DIR,
            check=True,
        )

        return MenuNames.MainMenu
