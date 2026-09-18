import argparse
import csv
import math
from pathlib import Path


def read_garage_metrics(csv_path: str | Path) -> dict[str, float]:
    """Read and validate evaluation metrics from a Garage results CSV file."""

    required_columns = {
        "scene_uuid",
        "average_displacement_error_m",
        "final_displacement_error_m",
    }

    results_path = Path(csv_path)

    if not results_path.is_file():
        raise FileNotFoundError(
            f"Garage results file does not exist: {results_path}"
        )

    with results_path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "The results file does not contain the required metric columns."
            )

        rows = list(reader)

    scene_rows = [
        row for row in rows
        if row["scene_uuid"] != "average"
    ]

    average_rows = [
        row for row in rows
        if row["scene_uuid"] == "average"
    ]

    if not scene_rows:
        raise ValueError(
            "The results file does not contain any evaluated scenes."
        )

    if len(average_rows) != 1:
        raise ValueError(
            "The results file must contain exactly one average row."
        )

    metric_columns = (
        "average_displacement_error_m",
        "final_displacement_error_m",
    )

    for row in rows:
        scoring_error = (row.get("scoring_error") or "").strip()

        if scoring_error:
            raise ValueError(
                f"Scene {row['scene_uuid']} failed: {scoring_error}"
            )

        for column in metric_columns:
            try:
                value = float(row[column])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid value in column {column} "
                    f"for scene {row['scene_uuid']}."
                ) from error

            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"Invalid metric in column {column} "
                    f"for scene {row['scene_uuid']}."
                )

    average_row = average_rows[0]

    return {
        "ADE (m)": float(
            average_row["average_displacement_error_m"]
        ),
        "FDE (m)": float(
            average_row["final_displacement_error_m"]
        ),
        "Evaluated Scenes": float(len(scene_rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read metrics from a Garage evaluation results file."
    )

    parser.add_argument(
        "csv_path",
        help="Path to the Garage results.csv file.",
    )

    args = parser.parse_args()

    try:
        metrics = read_garage_metrics(args.csv_path)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Failed to read Garage results: {error}\n")

    print("Garage Evaluation Results")

    for metric_name, metric_value in metrics.items():
        if metric_name == "Evaluated Scenes":
            print(f"{metric_name}: {int(metric_value)}")
        else:
            print(f"{metric_name}: {metric_value:.6f}")


if __name__ == "__main__":
    main()