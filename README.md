# AdEval (Autonomous Driving Evaluator)

> Compare autonomous driving models across different evaluation tasks.

## Development Setup

AdEval requires Linux or WSL and Python 3.10.

```bash
git clone https://github.com/jiabaoliu-015/CITS3200
cd CITS3200

python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run AdEval:

```bash
python -m adeval
```

## nuPlan Evaluation with Py123D Garage

AdEval uses [Py123D Garage](https://github.com/kesai-labs/py123d_garage) to evaluate the nuPlan open-loop planning task with the pretrained ResNet34 Latent TransFuser model.

The evaluation randomly selects up to five eligible scenes and displays ADE, FDE, the evaluated scene count, and the results file location.

### 1. Install Garage

Run these commands from the CITS3200 directory:

```bash
python3.10 -m venv .venv-garage

.venv-garage/bin/python -m pip install --upgrade pip
.venv-garage/bin/python -m pip install \
    "py123d-garage @ git+https://github.com/kesai-labs/py123d_garage.git@v0.1.1"
.venv-garage/bin/python -m pip install \
    "huggingface_hub[cli]>=0.34,<1"
```

Verify the installation:

```bash
.venv-garage/bin/python -c \
    "import py123d_garage; print('Garage: OK')"
```

### 2. Download the Model

```bash
.venv-garage/bin/hf download \
    kesai-labs/py123d_garage_pretrained_checkpoints \
    --revision 5516eddafcfa4b622c3df59200e5b189ef328182 \
    --include "resnet34_v0.1.0/*" \
    --local-dir "$PWD/checkpoints"
```

The following files must exist:

```text
checkpoints/resnet34_v0.1.0/model_0014.pth
checkpoints/resnet34_v0.1.0/config.yaml
```

### 3. Download Sample nuPlan Data

Skip this step if compatible Garage data is already available.

```bash
export PY123D_GARAGE_DATA_ROOT="$PWD/garage_data"
export ADEVAL_NUPLAN_LOG="2021.05.25.14.24.08_veh-25_00934_01067"

.venv-garage/bin/hf download \
    kesai-labs/nuplan \
    --repo-type dataset \
    --revision 484d9d14e18fdf712dbc70962997ba2d53617bce \
    --local-dir "$PY123D_GARAGE_DATA_ROOT/nuplan/123D" \
    --include \
        "logs/nuplan_test/$ADEVAL_NUPLAN_LOG/*" \
        "maps/*" \
    --exclude \
        "*/camera.pcam_b0.arrow" \
        "*/camera.pcam_l1.arrow" \
        "*/camera.pcam_l2.arrow" \
        "*/camera.pcam_r1.arrow" \
        "*/camera.pcam_r2.arrow" \
        "*/lidar.*"
```

### 4. Run the Evaluation

```bash
source .venv/bin/activate

export ADEVAL_DEVICE=cpu
export TMPDIR="$PWD/tmp"
mkdir -p "$TMPDIR"

python -m adeval
```

Select:

```text
Evaluation Menu
→ nuPlan Open-Loop Planning
→ Run evaluation: y
```

Results are saved to:

```text
outputs/garage-evaluation-<run-id>/results.csv
```

## Optional External Garage Environment

Garage can be installed outside the CITS3200 directory. Set these variables before starting AdEval:

```bash
export GARAGE_RUNTIME="$HOME/garage-runtime"
export ADEVAL_GARAGE_PYTHON="$GARAGE_RUNTIME/.venv/bin/python"
export PY123D_GARAGE_DATA_ROOT="$GARAGE_RUNTIME/data"
export ADEVAL_NUPLAN_CHECKPOINT="$GARAGE_RUNTIME/checkpoints/resnet34_v0.1.0/model_0014.pth"
export ADEVAL_GARAGE_OUTPUT_ROOT="$GARAGE_RUNTIME/outputs"

source .venv/bin/activate
python -m adeval
```

These variables are optional. Without them, AdEval uses `.venv-garage`, `garage_data`, `checkpoints`, and `outputs` inside the CITS3200 directory.

## Additional Data

Compatible nuPlan logs must use the Py123D Garage Arrow format and be placed under:

```text
garage_data/nuplan/123D/logs/nuplan_test/<log-name>/
```

Select another log with:

```bash
export ADEVAL_NUPLAN_LOG="<log-name>"
```

Raw nuPlan or self-collected data must first be converted to a Garage-compatible format.

## WSL Memory

If WSL exits while loading the model, increase `%USERPROFILE%\.wslconfig` and restart WSL:

```ini
[wsl2]
memory=10GB
swap=8GB
processors=4
```

```powershell
wsl --shutdown
```

Do not commit virtual environments, datasets, checkpoints, generated outputs, or temporary files.

## Recommended VS Code Extensions

- WSL
- Python
- Ruff
- Even Better TOML
- Git Graph
- Error Lens
