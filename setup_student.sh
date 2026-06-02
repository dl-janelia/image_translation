#!/usr/bin/env -S bash -i
#
# Image-translation exercise — STUDENT setup.
#
# Creates a conda env, installs the exercise dependencies into it, registers
# a Jupyter kernel, and downloads the data (via download_data.sh).
#
# Run from the exercise folder:
#   cd 06_image_translation
#   bash setup_student.sh
#
# DATA_ROOT is the PARENT directory; data lands in $DATA_ROOT/$KERNEL_NAME.
# If a TA pre-staged data on a shared mount, point DATA_ROOT at the same
# parent to skip the 14 GB download:
#   DATA_ROOT=/mnt/efs/dlmbl/data bash setup_student.sh
#
# Requires conda on PATH (install Miniconda if missing:
# https://docs.conda.io/en/latest/miniconda.html).

set -euo pipefail

START_DIR=$(pwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-06_image_translation}"
KERNEL_NAME="${KERNEL_NAME:-$ENV_NAME}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
DATA_ROOT="${DATA_ROOT:-$HOME/data}"   # parent dir; $KERNEL_NAME is appended
DATA_DIR="$DATA_ROOT/$KERNEL_NAME"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not on your PATH. Install Miniconda first:" >&2
    echo "  https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
fi

# When this exercise is checked out inside the VisCy monorepo (four levels up),
# install cytoland in editable mode against the local source. Otherwise install
# the pinned git ref from pyproject.toml.
MONOREPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." 2>/dev/null && pwd || true)"
if [[ -n "$MONOREPO_ROOT" && -f "$MONOREPO_ROOT/pyproject.toml" ]] \
        && grep -q '^name = "viscy"' "$MONOREPO_ROOT/pyproject.toml"; then
    INSTALL_MODE="workspace"
else
    INSTALL_MODE="pypi"
    MONOREPO_ROOT=""
fi
echo "Install mode: $INSTALL_MODE"

# --- 1. Create + activate the conda env ------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' already exists — reusing it."
else
    echo "Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION ..."
    conda create -n "$ENV_NAME" -y "python=$PYTHON_VERSION"
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "Active env: $CONDA_DEFAULT_ENV (python: $(python --version))"

# --- 2. Install dependencies into the active conda env ---------------------
# Use `uv pip install --python <conda python>` (NOT `uv sync`, which is
# project-mode and creates its own ./.venv ignoring CONDA_PREFIX). This
# installs straight into the conda env's site-packages.
command -v uv >/dev/null 2>&1 || python -m pip install uv
CONDA_PY="$(which python)"
echo "Installing dependencies into $CONDA_PY (uv $(uv --version)) ..."
uv pip install --python "$CONDA_PY" -r "$SCRIPT_DIR/pyproject.toml"

if [[ "$INSTALL_MODE" == "workspace" ]]; then
    echo "Workspace mode: installing editable cytoland from $MONOREPO_ROOT ..."
    uv pip install --python "$CONDA_PY" -e "$MONOREPO_ROOT/applications/cytoland[metrics]"
fi

# --- 3. Register the Jupyter kernel ----------------------------------------
python -m ipykernel install --user \
    --name "$KERNEL_NAME" --display-name "Python ($KERNEL_NAME)"
echo "Registered Jupyter kernel: $KERNEL_NAME"

# --- 4. Download data + checkpoints ----------------------------------------
DATA_ROOT="$DATA_ROOT" KERNEL_NAME="$KERNEL_NAME" bash "$SCRIPT_DIR/download_data.sh"

cd "$START_DIR"

cat <<EOF

--------------------------------------------------------------------
Student setup complete.
  - conda env:      $ENV_NAME
  - jupyter kernel: $KERNEL_NAME
  - data:           $DATA_DIR

To start the exercise:
  1. conda activate $ENV_NAME
  2. Launch Jupyter (jupyter lab) or open solution.py in VSCode.
  3. Select the "Python ($KERNEL_NAME)" kernel.
  4. Run cells top to bottom.
--------------------------------------------------------------------
EOF
