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

1. Add exports to bashrc, then reload the shell so they take effect immediately
    > This switches TMPDIR from RAM to disk storage because large datasets may not fit in /tmp (on many systems /tmp is a small tmpfs, and downloads will fail with "Disk quota exceeded" once they exceed it). For one time use of application, set TMPDIR temporarily for session.
    ```
    mkdir -p "$HOME/CITS3200/tmp"
    echo 'export PY123D_DATA_ROOT="$HOME/CITS3200/py123d_data_root"' >> ~/.bashrc
    echo 'export TMPDIR="$HOME/CITS3200/tmp"' >> ~/.bashrc
    source ~/.bashrc
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

# Motion Planning Benchmark Setup (Linux)
The `motion_planning/nuplan-devkit` benchmark (`./benchmark`, and the "Download simple nuplan" option in `python -m adeval`) needs `nuplan-devkit` and its dependencies installed. These are pinned to old, fixed versions (Python 3.9, `torch==1.9.0`, `pytorch-lightning==1.3.8`, etc.) that conflict with the modern dependencies used above, so this needs its **own, separate** Python environment — it cannot share the `.venv` from the steps above.

1. Install Python 3.9 with pyenv and create a second virtual environment inside `nuplan-devkit/`
    ```
    cd motion_planning/nuplan-devkit
    pyenv install -s 3.9
    pyenv local 3.9
    python -m venv .venv-nuplan
    source .venv-nuplan/bin/activate
    ```

1. Install the pinned dependencies
    > `pip>=23` is required for the resolver to finish in reasonable time; `pip<24.1` is required because `hydra-core==1.1.0rc1` has malformed version metadata that pip started rejecting in 24.1.
    ```
    pip install "pip>=23,<24.1"
    pip install -r requirements_torch.txt -r requirements.txt
    ```

1. Point `./benchmark` at this environment and reload the shell
    ```
    deactivate
    cd ../..
    echo "export NUPLAN_PYTHON=\"$(pwd)/motion_planning/nuplan-devkit/.venv-nuplan/bin/python\"" >> ~/.bashrc
    source ~/.bashrc
    ```

1. Download the dataset and run the benchmark through the main program (back in the top-level `.venv` from the steps above, not `.venv-nuplan`)
    ```
    cd ~/CITS3200
    source .venv/bin/activate
    python -m adeval
    ```
    In the menu, go to **Download Menu → [3] Download simple nuplan**. This downloads the nuPlan Mini dataset on first run (~11 GB, skipped on later runs once it's on disk), then asks `Run the motion_planning benchmark test now?` — answer yes. It shells out to `./benchmark` under the hood, which picks up the `NUPLAN_PYTHON` set above to run in the `nuplan-devkit` environment.

    To run the benchmark directly instead of through the menu (e.g. for non-interactive flags like `--profile`/`--location`), see `motion_planning/README_BENCHMARK.md`. If you'd rather use conda instead of a second pyenv/venv for `nuplan-devkit`, `conda env create -n cits3200-motion -f nuplan-devkit/environment.yml` works too — set `NUPLAN_PYTHON` to that env's `bin/python` instead.

# VSCode extensions to help during development
1. WSL
1. Python
1. Code Spell Checker
1. Even Better TOML
1. Git Graph
1. Ruff
1. Error Lens