from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from adeval.console import console
from adeval.menus.base_menu import Menu
from adeval.menus.menu_names import MenuNames


class EvaluationMenu(Menu):
    def run(self) -> MenuNames | None:
        console.print(
            Panel(
                "Evaluation Menu",
                style="bold cyan"
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
            console.print(
                f"[{index}] {task['name']}"
            )

        console.print("[0] Back to Main Menu")

        choice = Prompt.ask(
            "Select a task",
            choices=[
                str(i)
                for i in range(len(tasks) + 1)
            ]
        )

        if choice == "0":
            return MenuNames.MainMenu

        selected_task = tasks[int(choice) - 1]

        self.display_task(selected_task)

        run_choice = Prompt.ask(
            "Run evaluation?",
            choices=["y", "n"],
            default="y"
        )

        if run_choice == "y":
            metrics = self.evaluate(selected_task)
            self.display_metrics(metrics)

        return MenuNames.EvaluationMenu


    def get_tasks(self):
        return [
            {
                "name": "Perception",
                "model": "Example Model",
                "dataset": "Example Dataset",
            }
        ]


    def display_task(self, task):
        table = Table(
            title="Evaluation Configuration"
        )

        table.add_column(
            "Task",
            style="cyan"
        )

        table.add_column(
            "Model",
            style="green"
        )

        table.add_column(
            "Dataset",
            style="magenta"
        )

        table.add_row(
            task["name"],
            task["model"],
            task["dataset"]
        )

        console.print(table)


    def evaluate(self, task):
        with console.status(
            f"[bold green]"
            f"Evaluating {task['model']} "
            f"on {task['dataset']}..."
        ):
            metrics = {
                "Accuracy": 0.91,
                "Precision": 0.88,
                "Recall": 0.86,
            }

        console.print(
            "[bold green]"
            "Evaluation completed!"
            "[/bold green]"
        )

        return metrics


    def display_metrics(self, metrics):
        table = Table(
            title="Evaluation Metrics"
        )

        table.add_column(
            "Metric",
            style="cyan"
        )

        table.add_column(
            "Value",
            style="green"
        )

        for metric, value in metrics.items():
            table.add_row(
                str(metric),
                str(value)
            )

        console.print(table)