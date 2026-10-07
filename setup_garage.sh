#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv-garage"
python -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install \
    "py123d-garage @ git+https://github.com/kesai-labs/py123d_garage.git"

echo "Done. py123d-garage is setup for open-loop planning."