from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
    """Run a Garage open-loop evaluation on a small nuPlan sample."""

    if max_num_scenes <= 0:
        raise ValueError("max_num_scenes must be greater than zero.")

    project_root = Path(__file__).resolve().parents[2]

    garage_python = Path(
        os.environ.get(
            "ADEVAL_GARAGE_PYTHON",
            project_root / ".venv-garage/bin/python",
        )
    )

    data_root = Path(
        os.environ.get(
            "PY123D_GARAGE_DATA_ROOT",
            project_root / "garage_data",
        )
    )

    checkpoint = Path(
        os.environ.get(
            "ADEVAL_NUPLAN_CHECKPOINT",
            project_root
            / "checkpoints"
            / "resnet34_v0.1.0"
            / "model_0014.pth",
        )
    )

    log_name = os.environ.get(
        "ADEVAL_NUPLAN_LOG",
        DEFAULT_LOG_NAME,
    )

    device = os.environ.get(
        "ADEVAL_DEVICE",
        "cpu",
    )

    log_directory = (
        data_root
        / "nuplan"
        / "123D"
        / "logs"
        / "nuplan_test"
        / log_name
    )

    if not garage_python.is_file():
        raise FileNotFoundError(
            f"Garage Python interpreter was not found: {garage_python}"
        )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Garage checkpoint was not found: {checkpoint}"
        )

    if not checkpoint.with_name("config.yaml").is_file():
        raise FileNotFoundError(
            "config.yaml must be located next to the Garage checkpoint."
        )

    if not log_directory.is_dir():
        raise FileNotFoundError(
            f"nuPlan log directory was not found: {log_directory}"
        )

    output_root = project_root / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)

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
        f"policy_config.evaluation_checkpoint_file={checkpoint}",
        (
            "+offline_data_sources/ltf_nuplan@"
            "benchmark_offline_data_sources.nuplan_test=nuplan_test"
        ),
        (
            "benchmark_offline_data_sources."
            "nuplan_test.cache_root=null"
        ),
        (
            "++benchmark_offline_data_sources.nuplan_test."
            f"garage_scene_filter.log_names=['{log_name}']"
        ),
        (
            "++benchmark_offline_data_sources.nuplan_test."
            f"garage_scene_filter.max_num_scenes={max_num_scenes}"
        ),
        f"parallelization_config.device={device}",
        "parallelization_config.max_workers=1",
        "parallelization_config.inference_batch_size=1",
        "parallelization_config.num_shards=1",
        "parallelization_config.shard_index=0",
    ]

    environment = os.environ.copy()
    environment["PY123D_GARAGE_DATA_ROOT"] = str(data_root)

    completed_process = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    stdout_log = run_directory / "garage_stdout.log"
    stderr_log = run_directory / "garage_stderr.log"

    stdout_log.write_text(
        completed_process.stdout,
        encoding="utf-8",
    )

    stderr_log.write_text(
        completed_process.stderr,
        encoding="utf-8",
    )

    if completed_process.returncode != 0:
        error_message = (
            completed_process.stderr.strip()
            or completed_process.stdout.strip()
            or "Garage exited without an error message."
        )

        raise RuntimeError(
            "Garage evaluation failed.\n"
            f"Run directory: {run_directory}\n"
            f"Error: {error_message[-3000:]}"
        )

    results_csv = run_directory / "results.csv"

    if not results_csv.is_file():
        raise FileNotFoundError(
            "Garage completed but did not create results.csv. "
            f"Check the logs in: {run_directory}"
        )

    metrics = read_garage_metrics(results_csv)

    return GarageEvaluationResult(
        metrics=metrics,
        results_csv=results_csv,
        run_directory=run_directory,
    )