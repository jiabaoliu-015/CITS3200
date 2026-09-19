# AdEval (Autonomous Driving Evaluator)
> The goal of the project is to compare multiple autonomous driving models against each other in various tasks

# Development Setup (Linux)
1. Clone the repository
    ```
    git clone https://github.com/jiabaoliu-015/CITS3200 && cd CITS3200
    ```

1. Setup [pyenv](https://github.com/pyenv/pyenv/) and have the right [build environment](https://github.com/pyenv/pyenv/wiki#suggested-build-environment) according to official docs 
    ```
    pyenv install 3.10
    pyenv local 3.10
    ```

1. Add exports to bashrc
    > This switches TMPDIR from RAM to disk storage because large datasets may not fit in /tmp. For one time use of application, set TMPDIR temporarily for session.
    ```
    export PY123D_DATA_ROOT="$HOME/CITS3200/py123d_data_root"
    export TMPDIR="$HOME/CITS3200/tmp"
    ```

1. Create a virtual environment in the project directory
    ```
    python -m venv .venv
    ```

1. Activate the virtual environment
    ```
    source .venv/bin/activate
    ```

1. Install libraries and makes it into a module (Just once unless new packages are added)
    ```
    pip install --upgrade pip && pip install -e ".[dev]"
    ```

1. Run file
    ```
    python -m adeval
    ```
# VSCode extensions to help during development
1. WSL
1. Python
1. Code Spell Checker
1. Even Better TOML
1. Git Graph
1. Ruff
1. Error Lens、

Py123D Garage nuPlan Evaluation

The Evaluation Menu integrates [Py123D Garage](https://github.com/kesai-labs/py123d_garage) to evaluate nuPlan data using a pretrained ResNet34 Latent TransFuser model. It displays the model and dataset names, runs evaluation, and prints ADE, FDE, and the evaluated scene count.

The current version randomly selects **up to five eligible scenes from one nuPlan test log per run**. This behavior requires the Garage runner to enable `garage_scene_filter.shuffle=true` before applying the scene limit. It is intended to verify the integration. Full-dataset benchmarking and HTML/PDF reports are not implemented.

Complete the development setup above first. Run the following commands from the repository root in a Linux or WSL terminal, using a branch that contains the Garage integration.

## 1. Install Garage

Garage uses a separate virtual environment. The menu runs in the existing `.venv` and automatically starts Garage using `.venv-garage/bin/python`.

```bash
.venv/bin/python -m venv .venv-garage
.venv-garage/bin/python -m pip install --upgrade pip
.venv-garage/bin/python -m pip install "py123d-garage @ git+https://github.com/kesai-labs/py123d_garage.git@v0.1.1"
.venv-garage/bin/python -m pip install "huggingface_hub[cli]>=0.34,<1"
.venv-garage/bin/python -m pip check
```

## 2. Download the Model

Download the official checkpoint using the revision referenced by the [Garage v0.1.1 release](https://github.com/kesai-labs/py123d_garage/releases/tag/v0.1.1):

```bash
.venv-garage/bin/hf download \
  kesai-labs/py123d_garage_pretrained_checkpoints \
  --revision 5516eddafcfa4b622c3df59200e5b189ef328182 \
  --include "resnet34_v0.1.0/*" \
  --local-dir "$PWD/checkpoints"
```

Keep `model_0014.pth` and its `config.yaml` together in `checkpoints/resnet34_v0.1.0/`.

## 3. Download Sample Data

This downloads one pre-converted nuPlan test log and the maps, retaining the three cameras required by the model:

```bash
export PY123D_GARAGE_DATA_ROOT="$PWD/garage_data"
export ADEVAL_NUPLAN_LOG="2021.05.25.14.24.08_veh-25_00934_01067"

.venv-garage/bin/hf download \
  kesai-labs/nuplan \
  --repo-type dataset \
  --revision 484d9d14e18fdf712dbc70962997ba2d53617bce \
  --local-dir "$PY123D_GARAGE_DATA_ROOT/nuplan/123D" \
  --include "logs/nuplan_test/$ADEVAL_NUPLAN_LOG/*" "maps/*" \
  --exclude \
    "*/camera.pcam_b0.arrow" \
    "*/camera.pcam_l1.arrow" \
    "*/camera.pcam_l2.arrow" \
    "*/camera.pcam_r1.arrow" \
    "*/camera.pcam_r2.arrow" \
    "*/lidar.*"
```

`PY123D_GARAGE_DATA_ROOT` is separate from the existing `PY123D_DATA_ROOT`. Garage expects `nuplan/123D` beneath its data root; data downloaded through the existing Download Menu is not automatically connected to this evaluation task.

## 4. Run Evaluation

Start the application using its existing environment:

```bash
.venv/bin/python -m adeval
```

Select `Evaluation Menu`, then `nuPlan Open-Loop Planning`, and confirm `Run evaluation?` with `y`.

The runner defaults to the checkpoint and sample log above, the project-local `garage_data` directory, and CPU execution. Optional environment variables can override these settings before starting the menu:

| Variable | Purpose |
| --- | --- |
| `ADEVAL_GARAGE_PYTHON` | Absolute path to the Garage environment's Python executable. |
| `PY123D_GARAGE_DATA_ROOT` | Absolute path to the directory containing `nuplan/123D`. |
| `ADEVAL_NUPLAN_CHECKPOINT` | Absolute path to a compatible checkpoint with an adjacent `config.yaml`. |
| `ADEVAL_NUPLAN_LOG` | Name of the nuPlan test log to evaluate. |
| `ADEVAL_DEVICE` | `cpu` by default; `cuda` requires a working CUDA-enabled Garage environment. |

Environment variables apply to the current terminal session. Set custom values again when opening a new terminal. The maximum of five scenes is currently set in the menu code.

Random sampling shuffles the eligible scenes from the selected log before taking up to five. Repeated runs may select different scene UUIDs and produce different ADE/FDE values. Sampling remains limited to that log; it does not select from every downloaded log. Runs may overlap or select the same scenes by chance. If five or fewer scenes are eligible, all of them are selected, so only their order may change.

## 5. Test Additional nuPlan Data

For another compatible nuPlan test log that is already converted to Py123D Arrow format, place the complete converted log and its maps under this layout:

```text
garage_data/
└── nuplan/
    └── 123D/
        ├── logs/
        │   └── nuplan_test/
        │       └── your_log_name/
        │           ├── sync.arrow
        │           ├── ego_state_se3.arrow
        │           ├── route_position.arrow
        │           ├── camera.pcam_l0.arrow
        │           ├── camera.pcam_f0.arrow
        │           ├── camera.pcam_r0.arrow
        │           └── ...
        └── maps/
            └── nuplan/
                └── ...
```

Preserve metadata, supporting files, and any referenced sensor assets, including their relative paths. The selected model requires three camera views, ego state, route information, and four seconds of future ego trajectory. Not every log contains scenes that pass the evaluation filters.

Replace `your_log_name` with the actual log directory name, then run:

```bash
export PY123D_GARAGE_DATA_ROOT="$PWD/garage_data"
export ADEVAL_NUPLAN_LOG="your_log_name"
.venv/bin/python -m adeval
```

The menu randomly selects up to five eligible scenes from the selected log. Adding files does not automatically enable multiple-log or full-dataset evaluation.

Raw nuPlan data must first be converted according to the [Py123D nuPlan conversion instructions](https://github.com/kesai-labs/py123d/blob/main/docs/datasets/nuplan.rst) for the installed version. Other splits, including nuPlan-mini, require matching configuration; do not rename them to `nuplan_test`. Arbitrary self-collected datasets require a custom parser and compatible Garage/model configuration and are not currently supported.

## 6. Evaluation Results

The menu displays:

- **ADE (m):** average position error across the predicted trajectory.
- **FDE (m):** position error at the final predicted point.
- **Evaluated Scenes:** the number of evaluated scenes, excluding the CSV average row.

Lower ADE and FDE mean smaller trajectory errors for the evaluated scenes. These small-sample results do not represent the full nuPlan test split.

Metrics can change when different scenes are sampled, but a different metric value is not guaranteed. Check the scene UUIDs to determine whether the selected scenes changed. When comparing models, evaluate them on the same scene UUIDs; independent random samples do not provide a controlled comparison.

Each run creates a new `outputs/garage-evaluation-<run-id>/` directory containing `results.csv`, `garage_stdout.log`, and `garage_stderr.log`. The menu prints the CSV path. If evaluation fails, check the error message and the logs in that run's directory. The result reader reports an error if any scene has failed scoring or invalid metrics.

Keep `.venv-garage/`, `garage_data/`, `checkpoints/`, and `outputs/` excluded through `.gitignore`. Commit code and documentation, and store datasets and model weights separately.

## 7. Check Random Scene Sampling

Run the evaluation twice and note the two CSV paths printed by the menu. Set the variables below to the actual directories for those runs, then compare the scene UUID sets:

```bash
RUN1="/absolute/path/to/outputs/garage-evaluation-first"
RUN2="/absolute/path/to/outputs/garage-evaluation-second"

diff -u \
  <(awk -F, 'NR > 1 && $1 != "average" {print $1}' "$RUN1/results.csv" | sort) \
  <(awk -F, 'NR > 1 && $1 != "average" {print $1}' "$RUN2/results.csv" | sort)
```

Differences indicate that the selected scene sets changed. No output means the sets are the same. Sorting ignores row-order changes, and the filter excludes the CSV header and average row. Random sampling does not guarantee a new set on every run; if repeated tests always select the same set, check that the runner passes `shuffle=true`, that more than five scenes pass the filters, and whether a fixed random seed is being applied.
