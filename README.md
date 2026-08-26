# AdEval (Autonomous Driving Evaluator)
> The goal of the project is to compare multiple autonomous driving models against each other in various tasks

# Development Setup (Linux)
1. Clone the repository
    ```
    git clone https://github.com/jiabaoliu-015/CITS3200 && cd CITS3200
    ```

1. Setup pyenv according to official docs
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
1. Python
1. Code Spell Checker
1. Even Better TOML
1. Git Graph
1. Ruff
1. Error Lens