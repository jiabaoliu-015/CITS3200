import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from adeval.console import console
from adeval.menus.base_menu import Menu
from adeval.menus.menu_names import MenuNames


class DownloadMenu(Menu):
    def run(self) -> str | None:
        console.print(Panel("Download Menu", style="bold cyan"))
        options = [
            "[0] Exit",
            "[1] Go to Main Menu",
            "[2] Download simple av2",
            "[3] Download simple nuplan",
            "[4] Check downloaded dataset",
            "[5] Clear TMPDIR",
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
        elif choice == "2":
            self._download_dataset(
                "av2-sensor-stream",
                "av2-sensor_val",
                "Av2ProgressDownloader",
                "dataset.parser.downloader.num_logs=1",
            )
        elif choice == "3":
            self._download_dataset(
                "nuplan-mini-stream",
                "nuplan-mini_val",
                "NuplanProgressDownloader",
            )
        elif choice == "4":
            self._check_downloaded_datasets()
        elif choice == "5":
            self.__clear_temp_dir()

        return MenuNames.DownloadMenu

    def _download_dataset(
        self, dataset: str, split: str, downloader: str, *overrides: str
    ) -> None:
        # Resolve relative paths from the directory where adeval was started.
        # Without this default, py123d writes to a literal "None/logs" directory.
        data_root = self._data_root()
        env = os.environ.copy()
        env["PY123D_DATA_ROOT"] = str(data_root)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "py123d.script.run_conversion",
                f"dataset={dataset}",
                f"dataset.parser.splits=[{split}]",
                f"dataset.parser.downloader._target_=adeval.downloaders.{downloader}",
                *overrides,
            ],
            check=True,
            env=env,
        )

        dataset_path = data_root / "logs" / split
        if dataset_path.is_dir():
            console.print("[green]Completed[/green]")
            console.print("Dataset path (converted):")
        else:
            console.print(
                "[yellow]Conversion exited, but the dataset directory was not found. "
                "Check the conversion logs. Expected path:[/yellow]"
            )
        console.print(
            str(dataset_path),
            markup=False,
            highlight=False,
            soft_wrap=True,
        )
        console.print("Output root (logs, maps and sensors):")
        console.print(str(data_root), markup=False, highlight=False, soft_wrap=True)

    @staticmethod
    def _data_root() -> Path:
        return Path(
            os.environ.get("PY123D_DATA_ROOT") or "py123d_data_root"
        ).expanduser().resolve()

    def _dataset_roots(self) -> list[Path]:
        roots = [self._data_root()]
        # Older runs without PY123D_DATA_ROOT wrote converted data under None/.
        legacy_root = Path("None").resolve()
        if legacy_root.is_dir() and legacy_root not in roots:
            roots.append(legacy_root)
        return roots

    def _check_downloaded_datasets(self) -> None:
        roots = [self._data_root()]
        table = Table()
        table.add_column("Dataset")
        table.add_column("Size (GB)", justify="right")
        table.add_column("Path", overflow="fold")
        for root in roots:
            logs_root = root / "logs"
            try:
                if not logs_root.is_dir():
                    continue
                for split_dir in sorted(logs_root.iterdir()):
                    if not split_dir.is_dir():
                        continue
                    # A nonempty sync index identifies a converted log. Empty
                    # directories and partial downloads are not listed as ready.
                    log_count = sum(
                        1
                        for log_dir in split_dir.iterdir()
                        if log_dir.is_dir()
                        and (log_dir / "sync.arrow").is_file()
                        and (log_dir / "sync.arrow").stat().st_size > 0
                    )
                    if not log_count:
                        continue
                    dataset = split_dir.name.rsplit("_", 1)[0]
                    name = {"av2-sensor": "AV2 Sensor", "nuplan-mini": "nuPlan mini"}.get(
                        dataset, dataset
                    )
                    try:
                        size_bytes = self._directory_size_bytes(split_dir)
                        size_gb = (
                            "<0.001" if 0 < size_bytes < 1_000_000
                            else f"{size_bytes / 1_000_000_000:.3f}"
                        )
                    except OSError:
                        size_gb = "Unavailable"
                    table.add_row(Text(name), size_gb, Text(str(split_dir)))
            except OSError as exc:
                console.print(f"Could not inspect {logs_root}: {exc}", markup=False)

        if table.row_count:
            console.print(table)
        else:
            console.print("No downloaded datasets found (no nonempty sync.arrow files).")
            console.print("Checked output roots:")
            for root in roots:
                console.print(str(root), markup=False, highlight=False, soft_wrap=True)

        console.print("[0] Go to Download Menu")
        console.print("[1] reset download dataset")
        choice = Prompt.ask("Select an option", choices=["0", "1"])
        if choice == "1":
            self._reset_downloaded_datasets(self._dataset_roots())

    @staticmethod
    def _directory_size_bytes(directory: Path) -> int:
        """Sum file sizes below this path, excluding symbolic-link targets."""
        def on_error(error: OSError) -> None:
            raise error

        total = 0
        for parent, _directories, filenames in os.walk(
            directory, followlinks=False, onerror=on_error
        ):
            for name in filenames:
                path = Path(parent) / name
                if not path.is_symlink():
                    total += path.stat().st_size
        return total

    def _reset_downloaded_datasets(self, roots: list[Path]) -> None:
        # Only remove py123d output folders; keep root-level files and TMPDIR.
        targets = []
        protected_roots = {Path.cwd().resolve(), Path.home().resolve()}
        for root in roots:
            if root in protected_roots or root == Path(root.anchor):
                console.print(
                    f"Cannot reset datasets: {root} is not a dedicated data directory.",
                    markup=False,
                )
                return
            for name in ("logs", "maps", "sensors"):
                path = root / name
                if path.exists() or path.is_symlink():
                    targets.append(path)

        if not targets:
            console.print("No downloaded datasets to reset.")
            return

        console.print("The following dataset folders will be permanently deleted:")
        for path in targets:
            console.print(str(path), markup=False, highlight=False, soft_wrap=True)
        if not Confirm.ask("Reset all listed datasets?", default=False, show_default=False):
            console.print("Reset cancelled.")
            return

        failed = False
        for path in targets:
            try:
                if path.is_symlink():
                    # Never follow a dataset-folder link into unrelated storage.
                    path.unlink()
                    console.print(f"Removed link only; target data kept: {path}", markup=False)
                elif path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except OSError as exc:
                failed = True
                console.print(f"Could not remove {path}: {exc}", markup=False)

        if failed:
            console.print("Reset incomplete. Some dataset files could not be removed.")
        else:
            console.print("Dataset reset completed.")

    def __clear_temp_dir(self):
        temp_dir = os.environ.get("TMPDIR")
        if not temp_dir:
            console.print(
                "TMPDIR is not set. Set it to your project temporary directory "
                "(e.g. ./tmp) before clearing it."
            )
            return
        try:
            temp_folder = Path(temp_dir).expanduser()

            for item in temp_folder.iterdir():
                if item.name == ".gitkeep":
                    continue

                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

            console.print("TMPDIR has been cleared")

        except FileNotFoundError:
            console.print("TMPDIR does not exist; nothing to clear.")
