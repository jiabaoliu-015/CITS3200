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

1. Install system packages needed to build `motion_planning/nuplan-devkit`'s dependencies
    ```
    sudo apt install -y gdal-bin libgdal-dev
    ```
    (`Fiona` needs the `gdal-config` binary from this at build time.)

1. `motion_planning/nuplan-devkit` pins old, fixed dependency versions (Python 3.9, `torch==1.9.0`, `pytorch-lightning==1.3.8`, etc.) that conflict with the modern dependencies just installed above, so it needs its own, separate Python environment. The steps below are for **Linux ARM64 (aarch64)**; see the note after the next step if you're on Linux x86_64.
    ```
    cd motion_planning/nuplan-devkit
    pyenv install -s 3.9
    pyenv local 3.9
    python -m venv .venv-nuplan
    source .venv-nuplan/bin/activate
    pip install "pip>=23,<24.1"
    ```
    > `pip>=23` is required for the resolver to finish in reasonable time; `pip<24.1` is required because `hydra-core==1.1.0rc1` has malformed version metadata that pip started rejecting in 24.1.

1. Install `torch`/`torchvision`/etc., then `torch_scatter` in its own pass
    ```
    sed '/^torch_scatter/d' requirements_torch.txt > /tmp/requirements_torch_no_scatter.txt
    pip install -r /tmp/requirements_torch_no_scatter.txt
    pip install --no-build-isolation torch_scatter==2.0.9
    ```
    > `torch_scatter` has no prebuilt wheel for aarch64, so it builds from source — but pip's isolated build environment can't see the `torch` installed by this step's first command, so it must be installed separately, with build isolation off, *after* `torch` is already present. **On Linux x86_64**, skip this split: `torch_scatter` has a prebuilt wheel there, so a single `pip install -r requirements_torch.txt` works.

1. Install the rest of `nuplan-devkit`'s dependencies
    ```
    PIP_CONSTRAINT=build-constraints.txt pip install -r requirements.txt
    ```
    > A few of the old pinned packages (e.g. `control==0.9.1`) still `import pkg_resources` at build time, which recent setuptools versions no longer ship. `build-constraints.txt` pins the isolated build environments to an older setuptools that still has it.

1. Point `./benchmark` at this environment, then switch back to the top-level environment
    ```
    deactivate
    cd ../..
    export NUPLAN_PYTHON="$(pwd)/motion_planning/nuplan-devkit/.venv-nuplan/bin/python"
    source .venv/bin/activate
    ```

1. Run file
    ```
    python -m adeval
    ```
    To also download the nuPlan Mini dataset and run the motion_planning benchmark, go to **Download Menu → [3] Download simple nuplan**. This downloads the dataset on first run (~11 GB, skipped on later runs once it's on disk), then asks `Run the motion_planning benchmark test now?` — answer yes. It shells out to `./benchmark` under the hood, which uses the `NUPLAN_PYTHON` set above.

    To run the benchmark directly instead of through the menu (e.g. for non-interactive flags like `--profile`/`--location`), see `motion_planning/README_BENCHMARK.md`. If you'd rather use conda instead of a second pyenv/venv for `nuplan-devkit`, `conda env create -n cits3200-motion -f nuplan-devkit/environment.yml` works too — set `NUPLAN_PYTHON` to that env's `bin/python` instead.

# VSCode extensions to help during development
1. WSL
1. Python
1. Code Spell Checker
1. Even Better TOML
1. Git Graph
1. Ruff
1. Error Lens
