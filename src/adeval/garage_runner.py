from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from adeval.console import console
from adeval.garage_results import read_garage_metrics

DEFAULT_LOG_NAME = "2021.05.25.14.24.08_veh-25_00934_01067"


@dataclass(frozen=True)
class GarageEvaluationResult:
    metrics: dict[str, float]
    results_csv: Path
    run_directory: Path


def run_nuplan_garage_evaluation(
    max_num_scenes: int = 5,
) -> GarageEvaluationResult:
    """Run Garage using the project-local runtime and data."""

    if max_num_scenes <= 0:
        raise ValueError("max_num_scenes must be greater than zero.")

    project_root = Path(__file__).resolve().parents[2]

    garage_python = project_root / ".venv-garage" / "bin" / "python"

    if not garage_python.is_file():
        raise FileNotFoundError(
            "The project-local Garage Python interpreter "
            f"was not found: {garage_python}"
        )

    if not os.access(garage_python, os.X_OK):
        raise PermissionError(
            f"The Garage Python interpreter is not executable: {garage_python}"
        )

    data_root_value = os.getenv("PY123D_DATA_ROOT")

    if data_root_value is None or not data_root_value.strip():
        raise RuntimeError("PY123D_DATA_ROOT is not configured.")

    data_root = Path(data_root_value.strip()).expanduser().resolve()

    if not data_root.is_dir():
        raise FileNotFoundError(
            f"PY123D_DATA_ROOT does not point to an existing directory: {data_root}"
        )

    checkpoint = project_root / "checkpoints" / "resnet34_v0.1.0" / "model_0014.pth"

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"The project-local Garage checkpoint was not found: {checkpoint}"
        )

    checkpoint_config = checkpoint.parent / "config.yaml"

    if not checkpoint_config.is_file():
        raise FileNotFoundError(
            "config.yaml must be located next to "
            f"the Garage checkpoint: {checkpoint_config}"
        )

    log_name = os.getenv(
        "ADEVAL_NUPLAN_LOG",
        DEFAULT_LOG_NAME,
    ).strip()

    if (
        re.fullmatch(
            r"[A-Za-z0-9._-]+",
            log_name,
        )
        is None
    ):
        raise ValueError(
            f"ADEVAL_NUPLAN_LOG contains unsupported characters: {log_name}"
        )

    device = os.getenv(
        "ADEVAL_DEVICE",
        "cpu",
    ).strip()

    if not device:
        raise ValueError("ADEVAL_DEVICE must not be empty.")

    nuplan_root = data_root / "nuplan" / "123D"

    if not nuplan_root.is_dir():
        raise FileNotFoundError(
            f"PY123D_DATA_ROOT must contain 'nuplan/123D': {data_root}"
        )

    log_directory = nuplan_root / "logs" / "nuplan_test" / log_name

    if not log_directory.is_dir():
        raise FileNotFoundError(
            f"The selected nuPlan log directory was not found: {log_directory}"
        )

    output_root = project_root / "outputs"

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_directory = Path(
        tempfile.mkdtemp(
            prefix="garage-evaluation-",
            dir=output_root,
        )
    )

    command = [
        str(garage_python),
        "-m",
        "py123d_garage.evaluation.open_loop.evaluate",
        f"hydra.run.dir={run_directory}",
        (f"policy_config.evaluation_checkpoint_file={checkpoint}"),
        (
            "+offline_data_sources/ltf_nuplan@"
            "benchmark_offline_data_sources."
            "nuplan_test=nuplan_test"
        ),
        ("benchmark_offline_data_sources.nuplan_test.cache_root=null"),
        (
            "++benchmark_offline_data_sources."
            "nuplan_test.garage_scene_filter."
            f"log_names=['{log_name}']"
        ),
        (
            "++benchmark_offline_data_sources."
            "nuplan_test.garage_scene_filter."
            "shuffle=true"
        ),
        (
            "++benchmark_offline_data_sources."
            "nuplan_test.garage_scene_filter."
            f"max_num_scenes={max_num_scenes}"
        ),
        f"parallelization_config.device={device}",
        "parallelization_config.max_workers=1",
        "parallelization_config.inference_batch_size=1",
        "parallelization_config.num_shards=1",
        "parallelization_config.shard_index=0",
    ]

    environment = os.environ.copy()
    environment["PY123D_DATA_ROOT"] = str(data_root)
    # Garage's current Hydra configuration still resolves this legacy
    # variable. Keep it internal so users only configure PY123D_DATA_ROOT.
    environment["PY123D_GARAGE_DATA_ROOT"] = str(data_root)
    environment["PYTHONUNBUFFERED"] = "1"

    stdout_log = run_directory / "garage_stdout.log"

    stderr_log = run_directory / "garage_stderr.log"

    inference_pattern = re.compile(r"Inference:.*?(\d+)/(\d+)")

    with (
        stdout_log.open(
            "w",
            encoding="utf-8",
            buffering=1,
        ) as stdout_file,
        stderr_log.open(
            "w",
            encoding="utf-8",
            buffering=1,
        ) as stderr_file,
    ):
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=environment,
            text=True,
            bufsize=1,
            stdout=stdout_file,
            stderr=subprocess.PIPE,
        )

        if process.stderr is None:
            process.terminate()
            process.wait()

            raise RuntimeError("Unable to read Garage process output.")

        current_line = ""
        progress_total = max_num_scenes

        try:
            with Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                progress_task = progress.add_task(
                    "Preparing scenes",
                    total=max_num_scenes,
                )

                while True:
                    character = process.stderr.read(1)

                    if character == "":
                        break

                    stderr_file.write(character)
                    stderr_file.flush()

                    if character in "\r\n":
                        match = inference_pattern.search(current_line)

                        if match is not None:
                            completed = int(match.group(1))

                            progress_total = int(match.group(2))

                            progress.update(
                                progress_task,
                                description=("Evaluating scenes"),
                                completed=completed,
                                total=progress_total,
                            )

                        current_line = ""
                    else:
                        current_line += character

                return_code = process.wait()

                if return_code == 0:
                    progress.update(
                        progress_task,
                        description=("Evaluation complete"),
                        completed=progress_total,
                        total=progress_total,
                    )
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stderr.close()

    if return_code != 0:
        stderr_text = stderr_log.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

        stdout_text = stdout_log.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

        error_message = (
            stderr_text or stdout_text or "Garage exited without an error message."
        )

        raise RuntimeError(
            "Garage evaluation failed.\n"
            f"Run directory: {run_directory}\n"
            f"Error: {error_message[-3000:]}"
        )

    results_csv = run_directory / "results.csv"

    if not results_csv.is_file():
        raise FileNotFoundError(
            "Garage completed but did not create "
            "results.csv. "
            f"Check the logs in: {run_directory}"
        )

    metrics = read_garage_metrics(results_csv)

    return GarageEvaluationResult(
        metrics=metrics,
        results_csv=results_csv,
        run_directory=run_directory,
    )
