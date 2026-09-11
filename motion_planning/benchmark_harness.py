#!/usr/bin/env python3
"""Small, reproducible front-end for nuPlan IDM benchmarks."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import glob
import json
import os
import platform
import random
import re
import resource
import sqlite3
import shutil
import subprocess
import sys
import time
import threading
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
DEVKIT = ROOT / "nuplan-devkit"
RESULTS = ROOT / "results" / "benchmark"
REGISTRY = ROOT / "datasets" / "registry.yaml"
MINI_DB = ROOT / "data" / "nuplan" / "dataset" / "nuplan-v1.1" / "splits" / "mini"
DEFAULT_MAPS = ROOT / "data" / "nuplan" / "dataset" / "maps"
FIXED_TOKENS = [
    "0399f678142b5009", "285927171a4050d4", "e2f6f2d3fe4258ca",
    "6f8ebab04b105d63", "e2184a685a7c5e14", "dc6678527ae25380",
    "9a6f02ecdd625b7c", "01f387460c7655d0", "e55cc719e4405b03",
    "f18b0a7bcef65433",
]


def choose(prompt: str, choices: List[str], default: int = 0) -> int:
    print(f"\n{prompt}")
    for index, label in enumerate(choices, 1):
        print(f"  [{index}] {label}")
    while True:
        answer = input(f"Select [{default + 1}]: ").strip()
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return int(answer) - 1
        print("Please enter a listed number.")


def load_registry() -> List[Dict[str, Any]]:
    try:
        import yaml
    except ImportError as error:
        raise SystemExit("PyYAML is required to read datasets/registry.yaml") from error
    if not REGISTRY.exists():
        raise SystemExit(f"Dataset registry not found: {REGISTRY}")
    content = yaml.safe_load(REGISTRY.read_text(encoding="utf-8")) or {}
    datasets = content.get("datasets", [])
    if not isinstance(datasets, list):
        raise SystemExit(f"Invalid registry: 'datasets' must be a list in {REGISTRY}")
    required = {"id", "name", "data_path", "map_path"}
    result = []
    seen = set()
    for item in datasets:
        if not isinstance(item, dict) or not required.issubset(item):
            raise SystemExit(f"Invalid dataset entry in {REGISTRY}; required fields: {sorted(required)}")
        dataset_id = str(item["id"])
        if dataset_id in seen:
            raise SystemExit(f"Duplicate dataset id '{dataset_id}' in {REGISTRY}")
        seen.add(dataset_id)
        if item.get("enabled", True):
            result.append(item)
    return result


def resolve_registered_path(value: str) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (ROOT / expanded).resolve()


def registry_help(open_editor: bool = False) -> None:
    print(f"\nDataset registry: {REGISTRY}")
    print("Edit it with TextEdit, VS Code, or another plain-text editor.")
    print(f"macOS TextEdit: open -a TextEdit '{REGISTRY}'")
    print(f"VS Code: code '{REGISTRY}'")
    print("Change data_path to a directory containing nuPlan .db files.")
    print("Change map_path to the directory containing nuPlan map files and nuplan-maps-v1.0.json.")
    print("Sensor blobs are not required for the IDMPlanner benchmark.")
    if open_editor and sys.platform == "darwin":
        subprocess.run(["open", "-a", "TextEdit", str(REGISTRY)], check=False)


def resolve_db_files(path: Path) -> List[Path]:
    path = path.expanduser().resolve()
    if path.is_file() and path.suffix == ".db":
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.db"))
        if not files:
            files = sorted(path.rglob("*.db"))
    else:
        files = []
    if not files:
        raise SystemExit(f"No nuPlan .db files found at: {path}")
    return files


def normalize_location(value: Any) -> str:
    """Return a friendly city label while preserving unfamiliar providers' labels."""
    text = str(value or "").strip()
    key = text.lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "las-vegas": "Las Vegas", "us-nv-las-vegas-strip": "Las Vegas",
        "boston": "Boston", "us-ma-boston": "Boston",
        "pittsburgh": "Pittsburgh", "us-pa-pittsburgh-hazelwood": "Pittsburgh",
        "singapore": "Singapore", "sg-one-north": "Singapore",
    }
    return aliases.get(key, text or "Unknown")


def scan_dataset(
    db_files: List[Path], scenario_type: str, sample_limit: int, dataset_name: str
) -> Dict[str, Any]:
    """Read compatible SQLite metadata and retain deterministic token samples per location."""
    available: Counter[str] = Counter()
    reservoirs: Dict[str, List[str]] = defaultdict(list)
    log_locations: Dict[str, str] = {}
    log_datasets: Dict[str, str] = {}
    token_locations: Dict[str, str] = {}
    metadata_sources: Counter[str] = Counter()
    rng = random.Random(0)
    fixed_set = set(FIXED_TOKENS)
    official_query = """
        WITH ordered_scenes AS (
            SELECT token, ROW_NUMBER() OVER (ORDER BY name ASC) AS row_num FROM scene
        ), num_scenes AS (SELECT COUNT(*) AS cnt FROM scene),
        valid_scenes AS (
            SELECT o.token FROM ordered_scenes AS o CROSS JOIN num_scenes AS n
            WHERE o.row_num >= 3 AND o.row_num < n.cnt - 1
        )
        SELECT lower(hex(lp.token)) AS token
        FROM lidar_pc AS lp
        INNER JOIN scenario_tag AS st ON lp.token = st.lidar_pc_token
        INNER JOIN valid_scenes AS vs ON lp.scene_token = vs.token
        INNER JOIN scene AS goal_scene ON goal_scene.token = lp.scene_token
        INNER JOIN ego_pose AS goal_pose ON goal_scene.goal_ego_pose_token = goal_pose.token
        WHERE st.type = ?
        GROUP BY lp.token, lp.timestamp
        ORDER BY lp.timestamp ASC
    """
    fallback_query = """
        SELECT lower(hex(lidar_pc_token)) AS token
        FROM scenario_tag WHERE type = ? GROUP BY lidar_pc_token
    """
    for db_file in db_files:
        try:
            connection = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            table_names = {
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            location, source, logfile = "Unknown", "Unknown", db_file.stem
            if "log" in table_names:
                columns = {row[1] for row in connection.execute("PRAGMA table_info(log)")}
                selected = [name for name in ("logfile", "location", "map_version") if name in columns]
                if selected:
                    row = connection.execute(f"SELECT {','.join(selected)} FROM log LIMIT 1").fetchone()
                    values = dict(row) if row is not None else {}
                    logfile = str(values.get("logfile") or db_file.stem)
                    if values.get("location"):
                        location, source = normalize_location(values["location"]), "log.location"
                    elif values.get("map_version"):
                        location, source = normalize_location(values["map_version"]), "log.map_version"
            log_locations[logfile] = location
            log_locations[db_file.stem] = location
            log_datasets[logfile] = dataset_name
            log_datasets[db_file.stem] = dataset_name
            metadata_sources[source] += 1
            try:
                rows = connection.execute(official_query, (scenario_type,))
                iterator = rows
            except sqlite3.Error:
                iterator = connection.execute(fallback_query, (scenario_type,))
            for row in iterator:
                token = str(row["token"]).lower()
                available[location] += 1
                seen = available[location]
                bucket = reservoirs[location]
                if len(bucket) < sample_limit:
                    bucket.append(token)
                else:
                    replacement = rng.randrange(seen)
                    if replacement < sample_limit:
                        bucket[replacement] = token
                if token in fixed_set:
                    token_locations[token] = location
            connection.close()
        except sqlite3.Error:
            # A DB with compatible planning tables but no source metadata is still represented.
            metadata_sources["Unknown"] += 1
            log_locations[db_file.stem] = "Unknown"
            log_datasets[db_file.stem] = dataset_name
    for bucket in reservoirs.values():
        rng.shuffle(bucket)
    for location, bucket in reservoirs.items():
        for token in bucket:
            token_locations[token] = location
    return {
        "available": dict(available), "reservoirs": dict(reservoirs),
        "log_locations": log_locations, "log_datasets": log_datasets,
        "token_locations": token_locations,
        "metadata_sources": dict(metadata_sources),
    }


def balanced_tokens(reservoirs: Dict[str, List[str]], count: int) -> List[str]:
    """Round-robin locations, automatically redistributing unused location quotas."""
    locations = sorted((name for name, values in reservoirs.items() if values), key=str.casefold)
    offsets = {name: 0 for name in locations}
    selected: List[str] = []
    while len(selected) < count:
        added = False
        for name in locations:
            index = offsets[name]
            if index < len(reservoirs[name]) and len(selected) < count:
                selected.append(reservoirs[name][index])
                offsets[name] += 1
                added = True
        if not added:
            break
    return selected


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    lines = []
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f"  - {yaml_scalar(item)}" for item in value)
        elif isinstance(value, dict):
            lines.append(f"{key}:")
            lines.extend(f"  {yaml_scalar(item_key)}: {yaml_scalar(item_value)}" for item_key, item_value in value.items())
        else:
            lines.append(f"{key}: {yaml_scalar(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def capture_environment(path: Path) -> None:
    details = [
        f"created_at={dt.datetime.now().astimezone().isoformat()}",
        f"platform={platform.platform()}", f"machine={platform.machine()}",
        f"python={sys.version.replace(os.linesep, ' ')}", f"executable={sys.executable}",
    ]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=DEVKIT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        details.append(f"nuplan_commit={commit}")
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=DEVKIT, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) != 0
        details.append(f"nuplan_worktree_modified={str(dirty).lower()}")
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        packages = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.STDOUT
        ).strip()
        details.extend(["", "[python_packages]", packages])
    except (OSError, subprocess.CalledProcessError):
        pass
    path.write_text("\n".join(details) + "\n", encoding="utf-8")


def safe_number(value: Any) -> Optional[float]:
    try:
        if value is None or value != value:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize(
    raw: Path, runtime: float, peak_mb: float, config: Dict[str, Any],
    log_locations: Optional[Dict[str, str]] = None,
    log_datasets: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    import pandas as pd

    aggregate_files = list(raw.glob("**/aggregator_metric/*.parquet"))
    runner_files = list(raw.glob("**/runner_report.parquet"))
    runner_file = runner_files[0] if runner_files else raw / "runner_report.parquet"
    if not aggregate_files or not runner_file.exists():
        raise RuntimeError("nuPlan finished without the expected aggregator/runner parquet files")
    aggregate = pd.concat([pd.read_parquet(item) for item in aggregate_files], ignore_index=True)
    aggregate = aggregate[aggregate["scenario"].notna()].copy()
    # Aggregator parquet also contains scenario-type and final-score summary rows.
    # Real per-scenario rows are the ones whose num_scenarios field is empty.
    if "num_scenarios" in aggregate.columns:
        aggregate = aggregate[aggregate["num_scenarios"].isna()]
    else:
        aggregate = aggregate[aggregate["scenario"].astype(str).str.lower() != "final_score"]
    runner = pd.read_parquet(runner_file)
    runner = runner.rename(columns={"scenario_name": "scenario"})
    merged = aggregate.merge(runner, on="scenario", how="outer", suffixes=("", "_runner"))
    fields = {
        "score": "score",
        "progress": "ego_progress_along_expert_route",
        "ttc": "time_to_collision_within_bound",
        "comfort": "ego_is_comfortable",
        "collision_free": "no_ego_at_fault_collisions",
        "planner_latency_s": "compute_trajectory_runtimes_mean",
        "scenario_runtime_s": "duration",
    }
    records = []
    log_locations = log_locations or {}
    log_datasets = log_datasets or {}
    for _, row in merged.iterrows():
        log_name = row.get("log_name")
        if log_name is None or log_name != log_name:
            log_name = row.get("log_name_runner", "")
        record: Dict[str, Any] = {
            "scenario_token": str(row.get("scenario", "")),
            "dataset": log_datasets.get(str(log_name), str(config["dataset"])),
            "location": log_locations.get(str(log_name), "Unknown"),
            "scenario_type": str(row.get("scenario_type", config["scenario_type"])),
            "planner": str(row.get("planner_name", row.get("planner_name_runner", "IDMPlanner"))),
            "succeeded": bool(row.get("succeeded", False)),
        }
        for target, source in fields.items():
            record[target] = safe_number(row.get(source))
        records.append(record)
    records.sort(key=lambda item: item["scenario_token"])
    successful = sum(bool(item["succeeded"]) for item in records)
    location_counts = dict(sorted(Counter(item["location"] for item in records).items()))
    grouped_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for item in records:
        grouped_counts[item["dataset"]][item["location"]] += 1
    dataset_location_counts = {
        dataset: dict(sorted(counts.items())) for dataset, counts in sorted(grouped_counts.items())
    }
    def mean(name: str) -> Optional[float]:
        values = [item[name] for item in records if item[name] is not None]
        return sum(values) / len(values) if values else None
    summary = {
        "dataset": config["dataset"], "profile": config["profile"], "planner": "IDMPlanner",
        "requested_scenarios": config["requested_scenarios"],
        "planned_scenarios": config.get("planned_scenarios", config["requested_scenarios"]),
        "completed_scenarios": len(records),
        "available_location_counts": config.get("available_by_location", {}),
        "planned_location_counts": config.get("planned_by_location", {}),
        "location_counts": location_counts, "dataset_location_counts": dataset_location_counts,
        "successful_scenarios": successful, "success_rate": successful / len(records) if records else 0,
        "score": mean("score"), "progress": mean("progress"), "ttc": mean("ttc"),
        "comfort": mean("comfort"), "collision_free": mean("collision_free"),
        "planner_latency_s": mean("planner_latency_s"), "runtime_s": runtime,
        "peak_memory_mb": peak_mb,
    }
    (raw.parent / "metrics.json").write_text(
        json.dumps({"summary": summary, "scenarios": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    columns = list(records[0]) if records else ["scenario_token"]
    with (raw.parent / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(records)
    def fmt(name: str, percent: bool = False) -> str:
        value = summary[name]
        if value is None:
            return "n/a"
        return f"{value * 100:.2f}%" if percent else f"{value:.6f}"
    available_lines = "\n".join(
        f"    {name}: {value}" for name, value in config.get("available_by_location", {}).items()
    )
    planned_lines = "\n".join(
        f"    {name}: {value}" for name, value in config.get("planned_by_location", {}).items()
    )
    completed_lines = []
    for dataset, counts in dataset_location_counts.items():
        completed_lines.append(f"  {dataset}:")
        completed_lines.extend(f"    {name}: {value}" for name, value in counts.items())
    location_lines = "\n".join(completed_lines)
    if not location_lines:
        location_lines = f"  {summary['dataset']}:\n    Unknown: 0"
    report = f"""Motion Planning Benchmark Test version 1
========================================
Dataset: {summary['dataset']}
Profile: {summary['profile']}
Planner: IDMPlanner
Scenario type: {config['scenario_type']}
Sampling: {config.get('sampling', 'unspecified')}
Planned: {summary['planned_scenarios']} / {summary['requested_scenarios']}
Completed: {summary['completed_scenarios']} / {summary['requested_scenarios']}
Successful: {summary['successful_scenarios']} ({fmt('success_rate', True)})

Available eligible scenarios by dataset and location:
  {summary['dataset']}:
{available_lines or '    Unknown: 0'}

Planned by dataset and location:
  {summary['dataset']}:
{planned_lines or '    Unknown: 0'}

Completed by dataset and location:
{location_lines}

Score: {fmt('score')}
Progress: {fmt('progress')}
TTC within bound: {fmt('ttc')}
Comfort: {fmt('comfort')}
Collision free: {fmt('collision_free')}
Mean planner latency: {fmt('planner_latency_s')} s
Total runtime: {summary['runtime_s']:.2f} s
Peak memory: {summary['peak_memory_mb']:.2f} MB

Per-scenario values: metrics.csv / metrics.json
Raw nuPlan outputs: raw/
"""
    (raw.parent / "report.txt").write_text(report, encoding="utf-8")
    return records


def export_run(run_dir: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    required = ["report.txt", "metrics.csv", "metrics.json", "config.yaml", "environment.txt"]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise SystemExit(f"Cannot export; missing: {', '.join(missing)}")
    output = run_dir.with_name(run_dir.name + "_share.zip")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in required:
            archive.write(run_dir / name, arcname=f"{run_dir.name}/{name}")
        for log in (run_dir / "logs").glob("*.log"):
            archive.write(log, arcname=f"{run_dir.name}/logs/{log.name}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Motion Planning Benchmark Test version 1")
    parser.add_argument("--dataset", help="registered dataset id, or 'custom'")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--map-path", type=Path)
    parser.add_argument("--profile", choices=["quick", "standard", "extended"])
    parser.add_argument(
        "--location", help="'balanced', 'fixed', or a detected city/location name"
    )
    parser.add_argument("--yes", action="store_true", help="run without confirmation")
    parser.add_argument("--export", type=Path, metavar="RUN_DIR", help="create a small shareable zip")
    args = parser.parse_args()
    if args.export:
        print(f"Export created: {export_run(args.export)}")
        return 0
    print("\nMotion Planning Benchmark Test version 1")
    registered = load_registry()
    entry: Optional[Dict[str, Any]] = None
    if args.dataset:
        if args.dataset == "mini":
            args.dataset = "nuplan_mini"  # Backward-compatible V1 alias.
        if args.dataset != "custom":
            entry = next((item for item in registered if str(item["id"]) == args.dataset), None)
            if entry is None:
                available = ", ".join(str(item["id"]) for item in registered)
                raise SystemExit(f"Unknown dataset '{args.dataset}'. Registered ids: {available}, custom")
        dataset = args.dataset
    else:
        labels = []
        for item in registered:
            candidate = resolve_registered_path(str(item["data_path"]))
            status = "" if candidate.exists() else " (not configured)"
            labels.append(str(item["name"]) + status)
        selection = choose(
            "Dataset",
            labels + ["Custom nuPlan Dataset", "Open dataset registry / registration help"],
        )
        if selection == len(registered) + 1:
            registry_help(open_editor=True)
            input("After saving the registry, press Enter and run ./benchmark again...")
            return 0
        if selection == len(registered):
            dataset = "custom"
        else:
            entry = registered[selection]
            dataset = str(entry["id"])
    if dataset == "custom":
        if args.data_path:
            data_path = args.data_path
        else:
            while True:
                custom_value = input("nuPlan .db file or directory: ").strip()
                if custom_value:
                    data_path = Path(custom_value).expanduser()
                    break
                print("Path cannot be empty. Enter a .db file or its containing directory.")
        dataset_name = f"Custom nuPlan Dataset ({Path(data_path).expanduser().name or 'Unknown source'})"
        default_maps = DEFAULT_MAPS
        use_fixed_tokens = False
    else:
        assert entry is not None
        data_path = resolve_registered_path(str(entry["data_path"]))
        dataset_name = str(entry["name"])
        default_maps = resolve_registered_path(str(entry["map_path"]))
        use_fixed_tokens = bool(entry.get("fixed_tokens", False))
        if not data_path.exists():
            registry_help()
            raise SystemExit(f"\nDataset path is not configured or does not exist: {data_path}")
    maps = (args.map_path or default_maps).expanduser().resolve()
    if not maps.exists():
        raise SystemExit(f"Map directory not found: {maps}")
    db_files = resolve_db_files(data_path)
    profiles = ["quick", "standard", "extended"]
    profile = args.profile or profiles[choose("Benchmark", ["Quick (1 scenario)", "Standard (10 scenarios)", "Extended (100 scenarios)"])]
    count = {"quick": 1, "standard": 10, "extended": 100}[profile]
    scenario_type = str(entry.get("scenario_type", "traversing_intersection")) if entry else "traversing_intersection"
    print(f"\nScanning {len(db_files)} DB file(s) for location and scenario metadata...")
    scan = scan_dataset(db_files, scenario_type, count, dataset_name)
    available: Dict[str, int] = scan["available"]
    if not available:
        raise SystemExit(
            f"No compatible '{scenario_type}' scenarios were found. "
            "The DB schema may not expose nuPlan-compatible scenario metadata."
        )
    print("Available eligible scenarios grouped by dataset:")
    print(f"  {dataset_name}:")
    for name, value in sorted(available.items(), key=lambda item: item[0].casefold()):
        print(f"    {name}: {value}")
    fixed_allowed = use_fixed_tokens and profile in ("quick", "standard")
    options: List[tuple[str, str]] = []
    if fixed_allowed:
        options.append(("fixed", "V1 fixed Mini tokens (original comparable baseline)"))
    options.append(("balanced", f"All locations — balanced ({sum(available.values())} available)"))
    options.extend(
        (name, f"{name} — {dataset_name} ({value} available)")
        for name, value in sorted(available.items(), key=lambda item: item[0].casefold())
    )
    if args.location:
        requested_location = args.location.lower()
        if requested_location in ("balanced", "fixed"):
            sampling = requested_location
        else:
            normalized = normalize_location(args.location)
            sampling = next((name for name in available if name.casefold() == normalized.casefold()), "")
        if not sampling or sampling == "fixed" and not fixed_allowed:
            valid = ", ".join(item[0] for item in options)
            raise SystemExit(f"Unknown/unavailable location mode '{args.location}'. Choices: {valid}")
    elif args.yes:
        sampling = "fixed" if fixed_allowed else "balanced"
    else:
        sampling = options[choose("Location / sampling", [item[1] for item in options])][0]
    if sampling == "fixed":
        requested_fixed = FIXED_TOKENS[:1] if profile == "quick" else FIXED_TOKENS
        tokens = [token for token in requested_fixed if token in scan["token_locations"]]
        missing = len(requested_fixed) - len(tokens)
        if missing:
            print(f"Warning: {missing} fixed token(s) were not found in this dataset.")
        sampling_label = "V1 fixed Mini tokens"
    elif sampling == "balanced":
        tokens = balanced_tokens(scan["reservoirs"], count)
        sampling_label = "All locations — balanced"
    else:
        tokens = scan["reservoirs"].get(sampling, [])[:count]
        sampling_label = f"{sampling} only"
    if not tokens:
        raise SystemExit("No scenario tokens are available for the selected location/sampling mode.")
    planned_counts = Counter(scan["token_locations"].get(token, "Unknown") for token in tokens)
    print("Planned scenarios grouped by dataset:")
    print(f"  {dataset_name}:")
    for name, value in sorted(planned_counts.items(), key=lambda item: item[0].casefold()):
        print(f"    {name}: {value}")
    print(
        f"\nDataset: {dataset_name}\nDB files: {len(db_files)}\nProfile: {profile} "
        f"({count} requested, {len(tokens)} planned)\nSampling: {sampling_label}\nPlanner: IDMPlanner"
    )
    if not args.yes and input("Start benchmark? [Y/n] ").strip().lower() not in ("", "y", "yes"):
        print("Cancelled."); return 0
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS / f"{timestamp}_{dataset}_{profile}_idm"
    raw, logs = run_dir / "raw", run_dir / "logs"
    nuplan_output = raw / "closed_loop_nonreactive_agents"
    raw.mkdir(parents=True); logs.mkdir()
    config: Dict[str, Any] = {
        "harness_version": 1, "dataset": dataset_name, "dataset_path": str(data_path.resolve()),
        "map_path": str(maps), "planner": "IDMPlanner", "profile": profile,
        "scenario_type": scenario_type, "requested_scenarios": count,
        "planned_scenarios": len(tokens), "sampling": sampling_label,
        "available_by_location": dict(sorted(available.items())),
        "planned_by_location": dict(sorted(planned_counts.items())),
        "metadata_sources": dict(sorted(scan["metadata_sources"].items())),
        "uses_fixed_tokens": sampling == "fixed",
        "scenario_tokens": tokens,
    }
    write_yaml(run_dir / "config.yaml", config)
    capture_environment(run_dir / "environment.txt")
    command = [
        sys.executable, "-m", "nuplan.planning.script.run_simulation",
        "+simulation=closed_loop_nonreactive_agents", "planner=idm_planner",
        "scenario_builder=nuplan_mini", "scenario_filter=all_scenarios",
        "worker=sequential", "enable_profiling=false", "verbose=true",
        f"output_dir={nuplan_output}", f"scenario_builder.db_files=[{','.join(str(item) for item in db_files)}]",
        f"scenario_filter.scenario_types=[{scenario_type}]",
        f"scenario_filter.num_scenarios_per_type={count}", "scenario_filter.remove_invalid_goals=true",
    ]
    if tokens:
        command.append(f"scenario_filter.scenario_tokens=[{','.join(tokens)}]")
    env = os.environ.copy()
    env.update({"NUPLAN_DATA_ROOT": str(data_path.resolve()), "NUPLAN_MAPS_ROOT": str(maps),
                "NUPLAN_EXP_ROOT": str(raw), "PYTHONPATH": str(DEVKIT) + os.pathsep + env.get("PYTHONPATH", "")})
    start = time.monotonic()
    print(f"\nRunning {count} scenario(s). Detailed output: {logs / 'nuplan.log'}")
    progress = {"percent": 0}
    progress_pattern = re.compile(rb"Sequential:\s*(\d+)%")
    env["PYTHONUNBUFFERED"] = "1"
    with (logs / "nuplan.log").open("wb") as log:
        process = subprocess.Popen(
            command, cwd=DEVKIT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        def collect_output() -> None:
            recent = b""
            assert process.stdout is not None
            while True:
                chunk = process.stdout.read(256)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                recent = (recent + chunk)[-4096:]
                matches = progress_pattern.findall(recent)
                if matches:
                    progress["percent"] = min(100, int(matches[-1]))

        reader = threading.Thread(target=collect_output, daemon=True)
        reader.start()
        while process.poll() is None:
            elapsed = time.monotonic() - start
            percent = progress["percent"]
            filled = round(percent / 5)
            print(
                f"\rProgress: [{'#' * filled}{'-' * (20 - filled)}] "
                f"{percent:3d}% | Elapsed: {elapsed:6.1f}s",
                end="", flush=True,
            )
            time.sleep(0.25)
        reader.join()
        return_code = process.wait()
    elapsed = time.monotonic() - start
    final_percent = 100 if return_code == 0 else progress["percent"]
    filled = round(final_percent / 5)
    print(
        f"\rProgress: [{'#' * filled}{'-' * (20 - filled)}] "
        f"{final_percent:3d}% | Elapsed: {elapsed:6.1f}s"
    )
    runtime = time.monotonic() - start
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_mb = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    if return_code:
        (run_dir / "report.txt").write_text(
            f"Benchmark failed (exit code {return_code}).\nSee logs/nuplan.log\n", encoding="utf-8"
        )
        print(f"Benchmark failed. Report: {run_dir / 'report.txt'}")
        return return_code
    records = summarize(
        raw, runtime, peak_mb, config, scan["log_locations"], scan["log_datasets"]
    )
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["summary"]
    print(f"\nBenchmark completed.\nScore: {metrics['score']:.6f}\nSuccess: {metrics['successful_scenarios']} / {len(records)}\nReport: {run_dir / 'report.txt'}")
    print(f"Share later with: ./benchmark --export {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
