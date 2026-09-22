from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from adeval.console import console
from adeval.menus.base_menu import Menu
from adeval.menus.menu_names import MenuNames

MODELS_DIRECTORY = Path("tasks/perception_task/models")
CHECKPOINT_WRAPPERS = ("model_state", "state_dict", "model")
KEY_PREFIXES = ("module.", "model.", "net.")


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Expected a mapping checkpoint, got {type(checkpoint).__name__}"
        )
    return checkpoint


def unwrap_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    wrappers = [key for key in CHECKPOINT_WRAPPERS if key in checkpoint]
    if len(wrappers) > 1:
        raise ValueError(f"Multiple checkpoint wrappers found: {wrappers}")

    if wrappers:
        wrapped = checkpoint[wrappers[0]]
        if not isinstance(wrapped, Mapping):
            raise TypeError(f"Wrapper {wrappers[0]!r} does not contain a mapping")
        state = dict(wrapped)
    else:
        state = dict(checkpoint)

    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Checkpoint state contains non-tensor entries")

    for prefix in KEY_PREFIXES:
        if all(key.startswith(prefix) for key in state):
            state = {key.removeprefix(prefix): value for key, value in state.items()}
            break

    return state


class EvaluationMenu(Menu):
    def run(self) -> MenuNames | None:
        console.print(Panel("Evaluation Menu", style="bold cyan"))
        model_paths = sorted(MODELS_DIRECTORY.glob("*.pth"))

        if not model_paths:
            console.print(
                f"[yellow]No model checkpoints found in {MODELS_DIRECTORY}.[/yellow]"
            )
            return MenuNames.MainMenu

        for index, path in enumerate(model_paths, start=1):
            console.print(f"[{index}] {path.name}")
        console.print("[0] Back to Main Menu")

        choice = Prompt.ask(
            "Select models to evaluate (a for all, 0 to go back)",
            choices=["a", "0", *(str(index) for index in range(1, len(model_paths) + 1))],
        )
        if choice == "0":
            return MenuNames.MainMenu

        selected_paths = model_paths if choice == "a" else [model_paths[int(choice) - 1]]
        self.evaluate(selected_paths)
        return MenuNames.EvaluationMenu

    def evaluate(self, model_paths: list[Path]) -> None:
        table = Table(title="Loaded Evaluation Models")
        table.add_column("Model", style="cyan")
        table.add_column("Tensors", style="green", justify="right")
        table.add_column("Parameters", style="green", justify="right")

        for path in model_paths:
            with console.status(f"Loading {path.name}..."):
                state = unwrap_checkpoint(load_checkpoint(path))
            parameter_count = sum(value.numel() for value in state.values())
            table.add_row(path.name, str(len(state)), f"{parameter_count:,}")

        console.print(table)
        console.print(
            "[green]Checkpoints loaded. Dataset evaluation can now run each model "
            "against the selected test split.[/green]"
        )