from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from adeval.console import console
from adeval.garage_runner import (
    GarageEvaluationResult,
    run_nuplan_garage_evaluation,
)
from adeval.menus.base_menu import Menu
from adeval.menus.menu_names import MenuNames


class EvaluationMenu(Menu):
    def run(self) -> MenuNames | None:
        console.print(
            Panel(
                "Evaluation Menu",
                style="bold cyan",
            )
        )

        tasks = self.get_tasks()

        if not tasks:
            console.print(
                "[yellow]No evaluation tasks are available.[/yellow]"
            )
            return MenuNames.MainMenu

        console.print("[bold]Available Tasks[/bold]")

        for index, task in enumerate(tasks, start=1):
            console.print(f"[{index}] {task['name']}")

        console.print("[0] Back to Main Menu")

        choice = Prompt.ask(
            "Select a task",
            choices=[
                str(index)
                for index in range(len(tasks) + 1)
            ],
        )

        if choice == "0":
            return MenuNames.MainMenu

        selected_task = tasks[int(choice) - 1]

        self.display_task(selected_task)

        run_choice = Prompt.ask(
            "Run evaluation?",
            choices=["y", "n"],
            default="y",
        )

        if run_choice == "y":
            result = self.evaluate(selected_task)

            if result is not None:
                self.display_metrics(result.metrics)

                console.print(
                    f"[dim]Results CSV: {result.results_csv}[/dim]"
                )

        return MenuNames.EvaluationMenu

    def get_tasks(self) -> list[dict[str, str]]:
        return [
            {
                "name": "nuPlan Open-Loop Planning",
                "model": "ResNet34 Latent TransFuser",
                "dataset": "nuPlan Test",
                "evaluation_type": "garage_nuplan",
            }
        ]

    def display_task(self, task: dict[str, str]) -> None:
        table = Table(
            title="Evaluation Configuration"
        )

        table.add_column(
            "Task",
            style="cyan",
        )

        table.add_column(
            "Model",
            style="green",
        )

        table.add_column(
            "Dataset",
            style="magenta",
        )

        table.add_row(
            task["name"],
            task["model"],
            task["dataset"],
        )

        console.print(table)

    def evaluate(
        self,
        task: dict[str, str],
    ) -> GarageEvaluationResult | None:
        if task["evaluation_type"] != "garage_nuplan":
            console.print(
                "[red]Unsupported evaluation type.[/red]"
            )
            return None

        console.print(
            "[bold green]Starting Garage evaluation...[/bold green]"
        )

        try:
            result = run_nuplan_garage_evaluation(
                max_num_scenes=5
            )
        except (OSError, RuntimeError, ValueError) as error:
            console.print(
                Panel(
                    str(error),
                    title="Evaluation Failed",
                    style="bold red",
                )
            )
            return None

        console.print(
            "[bold green]Evaluation completed successfully![/bold green]"
        )

        return result

    def display_metrics(
        self,
        metrics: dict[str, float],
    ) -> None:
        table = Table(
            title="Evaluation Metrics"
        )

        table.add_column(
            "Metric",
            style="cyan",
        )

        table.add_column(
            "Value",
            style="green",
        )

        for metric_name, metric_value in metrics.items():
            if metric_name == "Evaluated Scenes":
                displayed_value = str(int(metric_value))
            else:
                displayed_value = f"{metric_value:.6f}"

            table.add_row(
                metric_name,
                displayed_value,
            )

        console.print(table)