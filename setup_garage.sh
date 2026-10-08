#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv-garage"
python -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install \
    "py123d-garage @ git+https://github.com/kesai-labs/py123d_garage.git"
"$VENV_DIR/bin/hf" download \
        kesai-labs/py123d_garage_pretrained_checkpoints \
        --local-dir "$HOME/CITS3200/tasks/nuplan_eval_task/checkpoints"

echo "Done. py123d-garage is setup for open-loop planning."