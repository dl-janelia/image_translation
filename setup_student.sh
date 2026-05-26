#!/usr/bin/env -S bash -i


#
# Image-translation exercise — STUDENT setup.
#
# This script:
#   1. Creates a conda environment "06_image_translation" with Python 3.13.
#   2. Installs uv inside that env (pip install uv) and uses it to install
#      cytoland + viscy plus the tutorial extras declared in pyproject.toml.
#      If run from inside a checkout of the VisCy monorepo, installs the
#      local cytoland workspace package in editable mode. Otherwise installs
#      from the git ref pinned in pyproject.toml.
#   3. Registers the env as a Jupyter kernel named "06_image_translation"
#      so students see it in VSCode / JupyterLab.
#   4. Downloads the training / test OME-Zarr datasets and the VSCyto2D
#      pretrained checkpoint into $DATA_ROOT (default ~/data/06_image_translation),
#      ONLY IF the data is not already there. If a TA has pre-staged data
#      on a shared filesystem, point DATA_ROOT at it to skip the download:
#
#        DATA_ROOT=/mnt/shared/image_translation bash setup_student.sh
#
# Requires conda to be on PATH (true on the course AWS images and on most
# laptops via miniconda/anaconda). If conda is missing, install miniconda
# from https://docs.conda.io/en/latest/miniconda.html first.
#
# Run this from the exercise folder:
#   cd 06_image_translation
#   bash setup_student.sh

set -euo pipefail

START_DIR=$(pwd)
ENV_NAME="${ENV_NAME:-06_image_translation}"
KERNEL_NAME="${KERNEL_NAME:-$ENV_NAME}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# --- Detect optional VisCy monorepo root (four levels up from this script) -
# When this exercise lives inside a viscy clone, install cytoland in editable
# mode against the local workspace. Otherwise fall back to PyPI.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONOREPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." 2>/dev/null && pwd || true)"
if [[ -n "${MONOREPO_ROOT:-}" && -f "$MONOREPO_ROOT/pyproject.toml" ]] \
        && grep -q '^name = "viscy"' "$MONOREPO_ROOT/pyproject.toml"; then
    INSTALL_MODE="workspace"
else
    INSTALL_MODE="pypi"
    MONOREPO_ROOT=""
fi
echo "Install mode: $INSTALL_MODE"

# --- 1. Create conda env ---------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    cat <<'ERR' >&2

ERROR: conda is not on your PATH.

This exercise uses conda for environment management (consistent with the
other DL@Janelia course exercises). Install Miniconda first:

  https://docs.conda.io/en/latest/miniconda.html

Then re-run this script.
ERR
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' already exists — reusing it."
else
    echo "Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION ..."
    conda create -n "$ENV_NAME" -y "python=$PYTHON_VERSION"
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "Failed to activate conda env '$ENV_NAME'." >&2
    exit 1
fi
echo "Active env: $CONDA_DEFAULT_ENV (python: $(python --version))"

# --- 2. Install dependencies into the active conda env ---------------------
# Use uv inside the env for fast resolution. We deliberately use
# `uv pip install --python $(which python)` rather than `uv sync`:
#   - uv sync is project-mode and always creates its own ./.venv, even with
#     --active; it does not respect CONDA_PREFIX.
#   - `uv pip install --python ...` installs into the targeted Python's
#     site-packages directly — which, with the conda env activated, is the
#     conda env's site-packages. That is what we want.
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv into the conda env ..."
    python -m pip install uv
fi
echo "Using uv: $(uv --version)"

CONDA_PY="$(which python)"
echo "Installing dependencies into $CONDA_PY ..."
uv pip install --python "$CONDA_PY" -r "$SCRIPT_DIR/pyproject.toml"

# Workspace override: if this exercise is checked out inside the VisCy
# monorepo, swap the installed cytoland for the local editable copy so
# changes to the upstream source are picked up live.
if [[ "$INSTALL_MODE" == "workspace" ]]; then
    echo "Workspace mode: replacing cytoland with editable install from $MONOREPO_ROOT ..."
    uv pip install --python "$CONDA_PY" -e "$MONOREPO_ROOT/applications/cytoland[metrics]"
fi

# --- 3. Register the env as a Jupyter kernel -------------------------------
python -m ipykernel install --user \
    --name "$KERNEL_NAME" \
    --display-name "Python ($KERNEL_NAME)"
echo "Registered Jupyter kernel: $KERNEL_NAME"

# --- 4. Download data + pretrained checkpoints (skip if already present) ----
DATA_ROOT="${DATA_ROOT:-$HOME/data/$KERNEL_NAME}"

# Shared-mount fast path: on the course AWS instances, TAs pre-stage the
# data at /mnt/efs/dlmbl/data/06_image_translation. If the user is using
# the default DATA_ROOT, that mount exists, and they haven't already got
# their own $DATA_ROOT, symlink $HOME/data/06_image_translation -> the
# shared mount instead of re-downloading 14 GB per student.
# Set DATA_ROOT explicitly to opt out (e.g. on a non-AWS laptop).
SHARED_DATA="/mnt/efs/dlmbl/data/$KERNEL_NAME"
if [[ "$DATA_ROOT" == "$HOME/data/$KERNEL_NAME" ]] \
        && [[ -d "$SHARED_DATA" ]] \
        && [[ ! -e "$DATA_ROOT" ]]; then
    echo "Found shared data at $SHARED_DATA — symlinking $DATA_ROOT to it."
    mkdir -p "$(dirname "$DATA_ROOT")"
    ln -s "$SHARED_DATA" "$DATA_ROOT"
fi

TRAINING_ZARR="$DATA_ROOT/training/a549_hoechst_cellmask_train_val.zarr"
TEST_ZARR="$DATA_ROOT/test/a549_hoechst_cellmask_test.zarr"
CHECKPOINT="$DATA_ROOT/pretrained_models/VSCyto2D/epoch=399-step=23200.ckpt"
FLUOR2PHASE_CKPT="$DATA_ROOT/pretrained_models/DLCourse/fluor2phase_step668.ckpt"

mkdir -p "$DATA_ROOT/training" "$DATA_ROOT/test" "$DATA_ROOT/pretrained_models"

if [[ -d "$TRAINING_ZARR" && -d "$TEST_ZARR" && -f "$CHECKPOINT" && -f "$FLUOR2PHASE_CKPT" ]]; then
    echo "Data already present at $DATA_ROOT — skipping download."
else
    echo "Downloading data + checkpoints to $DATA_ROOT ..."
    cd "$DATA_ROOT/training"
    wget -m -np -nH --cut-dirs=6 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/training/zarrv3/a549_hoechst_cellmask_train_val.zarr/"

    cd "$DATA_ROOT/test"
    wget -m -np -nH --cut-dirs=6 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/test/zarrv3/a549_hoechst_cellmask_test.zarr/"

    cd "$DATA_ROOT/pretrained_models"
    wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto2D/VSCyto2D/epoch=399-step=23200.ckpt"
    # Part 2.5 reverse model (fluorescence -> phase), hosted under the
    # dl_at_janelia/ tree because it's a DL@Janelia course model.
    mkdir -p "$DATA_ROOT/pretrained_models/DLCourse"
    cd "$DATA_ROOT/pretrained_models/DLCourse"
    wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/dl_at_janelia/DLCourse/pretrained_models/fluor2phase_step668.ckpt"
fi

cd "$START_DIR"

cat <<EOF

--------------------------------------------------------------------
Student setup complete.

  - conda env:      $ENV_NAME
  - jupyter kernel: $KERNEL_NAME
  - data:           $DATA_ROOT

To start the exercise:
  1. conda activate $ENV_NAME
  2. Launch Jupyter (jupyter lab) or open solution.py in VSCode.
  3. Select the "Python ($KERNEL_NAME)" kernel.
  4. Run cells top to bottom.
--------------------------------------------------------------------
EOF
