# Motion Planning Benchmark Test version 1

Run from this directory:

```bash
./benchmark
```

Choose a registered nuPlan dataset or a custom directory/file containing nuPlan `.db` data, then choose Quick (1 fixed scenario), Standard (the fixed 10 `traversing_intersection` tokens), or Extended (up to 100 `traversing_intersection` scenarios). IDMPlanner and the closed-loop non-reactive benchmark are used in V1.

## Registering a large nuPlan dataset

Select **Open dataset registry / registration help** in `./benchmark`, or edit `datasets/registry.yaml` directly. Set `data_path` to a directory containing nuPlan `.db` files and `map_path` to the maps directory containing `nuplan-maps-v1.0.json`. Absolute paths, `~`, environment variables, and project-relative paths are supported. Subdirectories are searched recursively. Camera/LiDAR sensor blobs are not needed by IDMPlanner.

For a non-interactive run:

```bash
./benchmark --dataset nuplan_mini --profile quick --yes
```

Every run is isolated under `results/benchmark/<timestamp>_.../` and contains `report.txt`, `metrics.csv`, `metrics.json`, `config.yaml`, `environment.txt`, `logs/`, and `raw/`. The raw folder contains nuPlan's parquet and simulation artifacts. Input datasets are only read.

## Location detection and balanced sampling

Before a run, the harness reads each compatible SQLite DB's `log.location` field, falling back to `log.map_version`. Unknown or vendor-compatible schemas without either value are labeled `Unknown`. It displays and saves results hierarchically as `dataset → location → count`, so cities from the same dataset stay together and separately registered datasets remain distinct. Each row in `metrics.csv` includes both `dataset` and `location`. It supports:

- the original fixed Mini tokens for comparable Quick/Standard V1 results;
- all detected locations with balanced round-robin sampling;
- one selected city/location only.

Extended runs default to balanced sampling. Reports and per-scenario CSV/JSON files include completed counts and the detected location. Non-interactive examples:

```bash
./benchmark --dataset nuplan_mini --profile extended --location balanced --yes
./benchmark --dataset nuplan_mini --profile standard --location "Boston" --yes
```

Create a small shareable archive without raw simulation data:

```bash
./benchmark --export results/benchmark/<run-directory>
```

`./benchmark` runs `benchmark_harness.py` with the Python interpreter that has the nuPlan environment installed. It does not assume any particular machine layout, so point it at your environment with `NUPLAN_PYTHON`:

```bash
export NUPLAN_PYTHON=/path/to/your/nuplan/env/bin/python
./benchmark --dataset nuplan_mini --profile quick --yes
```

If `NUPLAN_PYTHON` is unset, `./benchmark` falls back to `python3`/`python` on `PATH`. If it is set but does not point to an executable, `./benchmark` exits with an error explaining how to fix it, so switching environments (or unsetting the variable) never leaves you with a silent failure. Existing macOS ARM compatibility changes inside `nuplan-devkit` are intentionally untouched.
