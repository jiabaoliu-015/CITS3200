# AdEval (Autonomous Driving Evaluator)

> Compare autonomous driving models across different evaluation tasks.

AdEval requires **Linux 24.04** or lower. 

## Conda Setup

1. Instal [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install)

1. Install [Cuda Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64)

1. Add exports into `~/.bashrc` for Bash or `~/.zshrc` for Zsh. 
    ```bash
    export CUDA_HOME=/usr/local/cuda
    export PATH=$PATH:$CUDA_HOME/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
    ```

1. Clone the repository into home directory
    ```
    git clone --recurse-submodules https://github.com/jiabaoliu-015/CITS3200 && cd CITS3200
    ```

1. Create conda environment (if haven't been created yet)
    ```
    conda create -n adeval -c conda-forge python=3.10 -y
    ```

1. Activate environment
    ```
    conda activate adeval
    ```

1. Add environment variables to conda
    ```bash
    conda env config vars set \
        PY123D_DATA_ROOT="$HOME/CITS3200/py123d_data_root" \
        TMPDIR="$HOME/CITS3200/tmp" \
        PYTHONWARNINGS="ignore::RuntimeWarning:runpy"
    ```

1. Reactivate your conda environment
    ```
    conda deactivate && conda activate adeval
    ```

1. Verify the environment variables are set
    ```
    conda env config vars list
    ```

1. Build AdEval
    ```
    python -m pip install --upgrade pip && python -m pip install -e ".[dev]"
    ```

1. Run AdEval
    ```
    python -m adeval
    ```

## Pyenv Setup
1. Install [pyenv](https://github.com/pyenv/pyenv#linuxunix) and its required [build dependencies](https://github.com/pyenv/pyenv/wiki#suggested-build-environment)

1. Install [Cuda Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64)

1. Add exports into `~/.bashrc` for Bash or `~/.zshrc` for Zsh. 
    ```bash
    export CUDA_HOME=/usr/local/cuda
    export PATH=$PATH:$CUDA_HOME/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
    export PYTHONWARNINGS="ignore::RuntimeWarning:runpy"
    export PY123D_DATA_ROOT="$HOME/CITS3200/py123d_data_root"
    export TMPDIR="$HOME/CITS3200/tmp"
    ```

1. Clone the repository into home directory
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

1. Build AdEval
    ```
    python -m pip install --upgrade pip && python -m pip install -e ".[dev]"
    ```

1. Run AdEval
    ```
    python -m adeval
    ```

## nuPlan Evaluation with Py123D Garage

> AdEval uses [Py123D Garage](https://github.com/kesai-labs/py123d_garage) to evaluate the nuPlan open-loop planning task with the pre-trained ResNet34 Latent TransFuser model

1. While active in conda environment, run the garage setup
    ```
    bash setup_garage.sh
    ```

1. (Optionally) Once completed, to increase download speed from hugging face, run the command to login
    ```
    .venv-garage/bin/hf auth login
    ```

## Object Detection with OpenPCDet [GPU REQUIRED]

> AdEval uses [OpenPCDet](https://github.com/open-mmlab/openPCDet) to evaluate various models against popular dataset based on their object detection capability

1. While active in conda environment, run the install command
    ```
    python -m pip install --no-build-isolation -e third_party/OpenPCDet
    ```

# Troubleshooting
## For WSL

1. If WSL has memory issues when running AdEval:
    1. Open Run Dialog (Win + R)
    1. Enter the command which will either ask you to create the file or open an existing wslconfig
        ```
        notepad %USERPROFILE%\.wslconfig
        ```
    1. Increase the amount of memory or increase swap if not enough memory
        ```ini
        [wsl2]
        memory=10GB
        swap=10GB
        processors=2
        ```
    1. Restart WSL
        ```
        wsl --shutdown
        ```

# For Development
## Recommended VS Code Extensions
- WSL
- Python
- Ruff
- Even Better TOML
- Git Graph
- Error Lens

## Help commands for Py123D Garage
```
.venv-garage/bin/python -m py123d_garage.evaluation.open_loop.evaluate --help
```

## Reasoning for Ubuntu 24.04 and CUDA 12.8
OpenPCDet relies on older math functions that have been replaced in Ubuntu 26.04. Additionally, OpenPCDet relies on CUDA 12.8 or lower.

## Reasoning for separate Py123D garage instead of installing within the same environment
nuPlan Devkit replies on NumPy 1.x and Py123D garage relies on NumPy 2.x which makes both versions incompatible when building the project in the same environment