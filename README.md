# AdEval (Autonomous Driving Evaluator)

> Compare autonomous driving models across different evaluation tasks.

## Development Setup

AdEval requires Linux or WSL. 

1. Install [pyenv](https://github.com/pyenv/pyenv#linuxunix) and its required [build dependencies](https://github.com/pyenv/pyenv/wiki#suggested-build-environment)

1. Clone the repository
    ```
    git clone --recurse-submodules https://github.com/jiabaoliu-015/CITS3200 && cd CITS3200
    ```

1. Install the python version (verify using `python -V`)
    ```
    pyenv install -s
    ```

1. Create a AdEval virtual environment
    ```
    python -m venv .venv
    ```

1. Activate the environment
    ```
    source .venv/bin/activate
    ```

1. Install the AdEval packages into the environment
    ```
    python -m pip install --upgrade pip && python -m pip install -e ".[dev]"
    ```

1. Add exports into `~/.bashrc` for Bash or `~/.zshrc` for Zsh. 
    ```bash
    export CUDA_HOME=/usr/local/cuda
    export PATH=$PATH:$CUDA_HOME/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
    export MAX_JOBS=4
    export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"

    export PY123D_DATA_ROOT="$HOME/CITS3200/py123d_data_root"
    export TMPDIR="$HOME/CITS3200/tmp"
    ```
    > Note: `python -c "import torch; print(torch.cuda.get_device_capability())"` can be ran in the environment to get `TORCH_CUDA_ARCH_LIST`

---
    
1. Install the OpenPCDet
    ```
    python -m pip install --no-build-isolation -e third_party/OpenPCDet
    ```

---

1. Create py123d garage virtual environment
    ```
    python -m venv .venv-garage
    ```

1. Upgrade pip
    ```
    .venv-garage/bin/python -m pip install --upgrade pip
    ```

1. Install py123d garage packages
    ```bash
    .venv-garage/bin/python -m pip install \
        "py123d-garage @ git+https://github.com/kesai-labs/py123d_garage.git"
    ```

1. Verify py123d garage installation
    ```bash
    .venv-garage/bin/python -c \
        "import py123d_garage; print('Garage: OK')"
    ```
1. (Optionally) To increase download speed from hugging face
    ```
    .venv-garage/bin/hf auth login
    ```

1. Install Resnet pre-trained
    ```bash
    .venv-garage/bin/hf download \
        kesai-labs/py123d_garage_pretrained_checkpoints \
        --local-dir "$HOME/CITS3200/tasks/nuplan_eval_task/checkpoints"
    ```

1. Verify resnet is present
    ```bash
    [ -f "$HOME/CITS3200/tasks/nuplan_eval_task/checkpoints/resnet34_v0.1.0/model_0014.pth" ] \
        && [ -f "$HOME/CITS3200/tasks/nuplan_eval_task/checkpoints/resnet34_v0.1.0/config.yaml" ] \
        && echo "Checkpoint and config exist" \
        || echo "Checkpoint or config missing"
    ```
---

1. Run AdEval
    ```
    python -m adeval
    ```

## nuPlan Evaluation with Py123D Garage

AdEval uses [Py123D Garage](https://github.com/kesai-labs/py123d_garage) to evaluate the nuPlan open-loop planning task with the pre-trained ResNet34 Latent TransFuser model.

### 1. Install Garage

Run each step separately from the CITS3200 directory:

1. Create the Garage environment with `python -m venv .venv-garage`.

2. Upgrade its pip installation with `.venv-garage/bin/python -m pip install --upgrade pip`.

3. Install Garage:

    ```bash
    .venv-garage/bin/python -m pip install \
        "py123d-garage @ git+https://github.com/kesai-labs/py123d_garage.git"
    ```

4. Verify the Garage installation:

    ```bash
    .venv-garage/bin/python -c \
        "import py123d_garage; print('Garage: OK')"
    ```

### 2. Download the Model

Download the pretrained checkpoints with `.venv-garage/bin/hf download kesai-labs/py123d_garage_pretrained_checkpoints --local-dir "$HOME/CITS3200/checkpoints"`.

Verify the downloaded files with `[ -f checkpoints/resnet34_v0.1.0/model_0014.pth ] && [ -f checkpoints/resnet34_v0.1.0/config.yaml ] && echo "Checkpoint and config exist" || echo "Checkpoint or config missing"`.

### 3. Download Sample nuPlan Data

Skip this section if compatible Garage data is already available.

1. Optionally authenticate with Hugging Face to improve download rate limits and speed using `.venv-garage/bin/hf auth login`.

2. Download the default sample log:

    ```bash
    .venv-garage/bin/hf download \
        kesai-labs/nuplan \
        --repo-type dataset \
        --revision 484d9d14e18fdf712dbc70962997ba2d53617bce \
        --local-dir "$PY123D_DATA_ROOT/nuplan/123D" \
        --include "logs/nuplan_test/2021.05.25.14.24.08_veh-25_00934_01067/*" \
        --include "maps/*" \
        --exclude "*/camera.pcam_b0.arrow" \
        --exclude "*/camera.pcam_l1.arrow" \
        --exclude "*/camera.pcam_l2.arrow" \
        --exclude "*/camera.pcam_r1.arrow" \
        --exclude "*/camera.pcam_r2.arrow" \
        --exclude "*/lidar.*"
    ```

### 4. Run the Evaluation

1. Activate AdEval with `source .venv/bin/activate`.

2. Set a disk-backed temporary directory with `export TMPDIR="$HOME/CITS3200/tmp"`.

3. Create that directory with `mkdir -p "$TMPDIR"`.

4. Start AdEval with `python -m adeval`.

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

## Persisting Environment Variables

Variables set with `export` only apply to the current shell session. To keep them after restarting the terminal, add the required lines to `~/.bashrc` for Bash or `~/.zshrc` for Zsh.

For the default project-local setup, add:

```bash
export PY123D_DATA_ROOT="$HOME/CITS3200/py123d_data_root"
export TMPDIR="$HOME/CITS3200/tmp"
```

Reload the configuration with `source ~/.bashrc` or `source ~/.zshrc`.

## Additional Data

Compatible nuPlan logs must use the Py123D Garage Arrow format and be placed under:

```text
py123d_data_root/nuplan/123D/logs/nuplan_test/<log-name>/
```

Select another log with `export ADEVAL_NUPLAN_LOG="<log-name>"`.

Raw nuPlan or self-collected data must first be converted to a Garage-compatible format.

## WSL Memory

If WSL exits while loading the model, increase `%USERPROFILE%\.wslconfig` and restart WSL:

```ini
[wsl2]
memory=10GB
swap=8GB
processors=4
```

Apply the change from Windows PowerShell with `wsl --shutdown`.

## Recommended VS Code Extensions

- WSL
- Python
- Ruff
- Even Better TOML
- Git Graph
- Error Lens

.venv-garage/bin/hf download kesai-labs/nuplan \
    --repo-type dataset \
    --local-dir "$PY123D_DATA_ROOT/nuplan" \
    --include "logs/nuplan_test/2021.05.25.14.16.10_veh-35_01690_02183/*" \
    --include "maps/*" \
    --exclude "*/camera.pcam_b0.arrow" \
    --exclude "*/camera.pcam_l1.arrow" \
    --exclude "*/camera.pcam_l2.arrow" \
    --exclude "*/camera.pcam_r1.arrow" \
    --exclude "*/camera.pcam_r2.arrow" \
    --exclude "*/lidar.*"

.venv-garage/bin/python -m py123d_garage.evaluation.open_loop.evaluate \
    policy_config.evaluation_checkpoint_file=/home/timeanomaly/CITS3200/tasks/nuplan_eval_task/checkpoints/resnet34_v0.1.0/model_0014.pth \
    +offline_data_sources/ltf_nuplan@benchmark_offline_data_sources.nuplan_test=nuplan_test \
    benchmark_offline_data_sources.nuplan_test.cache_root=null \
    benchmark_offline_data_sources.nuplan_test.data_root="$PY123D_DATA_ROOT/nuplan" \
    'benchmark_offline_data_sources.nuplan_test.garage_scene_filter.split_names=[nuplan_test]' \
    parallelization_config.device=cuda

.venv-garage/bin/hf download kesai-labs/nuplan index.parquet --repo-type dataset --local-dir .

HELP Tips
.venv-garage/bin/python -m py123d_garage.evaluation.open_loop.evaluate --help

https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu

conda create -n adeval -c conda-forge python=3.10 -y
conda activate adeval
conda env config vars set PY123D_DATA_ROOT="$HOME/CITS3200/py123d_data_root" TMPDIR="$HOME/CITS3200/tmp"
conda deactivate && conda activate adeval

pip install --upgrade pip && pip install -e ".[dev]"