#!/usr/bin/env bash
# Sets up a local Python virtual environment for the JCAS simulator and
# installs its dependencies (numpy/scipy/shapely/matplotlib/seaborn, Jupyter,
# pytest, and Streamlit for the interactive app).
#
# Usage:
#   ./setup.sh
#
# Override the interpreter used to create the venv with PYTHON_BIN, e.g.:
#   PYTHON_BIN=python3.12 ./setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "error: '$PYTHON_BIN' was not found on PATH. Install Python 3.10+ and retry." >&2
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $SCRIPT_DIR/$VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Reusing existing virtual environment in $SCRIPT_DIR/$VENV_DIR."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Installing dependencies from requirements.txt ..."
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

cat <<EOF

Setup complete. The virtual environment lives in $VENV_DIR/.

Next steps:
  source $VENV_DIR/bin/activate
  streamlit run simulator_interface.py  # interactive web app
  jupyter notebook main.ipynb  # original notebook
  pytest                       # run the test suite

EOF
