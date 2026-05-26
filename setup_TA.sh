#!/usr/bin/env -S bash -i
#
# Image-translation exercise — TA setup.
#
# This script supports three phases. Run them in order before the course,
# or pass --all to do everything in one shot.
#
#   1. STAGE  (default)  Download ~14 GB of OME-Zarr data + checkpoints
#                        into $DATA_ROOT. Skip files already present.
#   2. INSTALL           Run setup_student.sh end-to-end in a throwaway
#                        location to validate that pyproject.toml resolves
#                        cleanly against current PyPI / GitHub state.
#   3. SMOKE             Run a short fast_dev_run + 2-epoch limited training
#                        through solution.py on a GPU to confirm the
#                        notebook actually works on a course node.
#
# Usage:
#
#   # Just stage data (~20-40 min)
#   bash setup_TA.sh
#   bash setup_TA.sh --stage
#
#   # Validate install only (assumes data is already staged)
#   bash setup_TA.sh --install
#
#   # Smoke-test the notebook on a GPU (assumes install is done)
#   bash setup_TA.sh --smoke
#
#   # Do everything (recommended ~1 week before the course)
#   bash setup_TA.sh --all
#
#   # Stage onto a shared mount (recommended for courses):
#   DATA_ROOT=/mnt/efs/image_translation bash setup_TA.sh --all
#
# Once this finishes, students point setup_student.sh at the same DATA_ROOT
# and skip the download:
#
#   DATA_ROOT=/mnt/efs/image_translation bash setup_student.sh

set -euo pipefail

# -----------------------------------------------------------------------------
# Phase flags
# -----------------------------------------------------------------------------
DO_STAGE=false
DO_INSTALL=false
DO_SMOKE=false

if [[ $# -eq 0 ]]; then
    DO_STAGE=true
else
    for arg in "$@"; do
        case "$arg" in
            --stage)   DO_STAGE=true ;;
            --install) DO_INSTALL=true ;;
            --smoke)   DO_SMOKE=true ;;
            --all)     DO_STAGE=true; DO_INSTALL=true; DO_SMOKE=true ;;
            -h|--help)
                grep '^#' "$0" | sed 's/^# \?//'
                exit 0
                ;;
            *)
                echo "Unknown flag: $arg (try --help)" >&2
                exit 2
                ;;
        esac
    done
fi

START_DIR=$(pwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_NAME="${KERNEL_NAME:-06_image_translation}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/$KERNEL_NAME}"

echo "================================================================"
echo "TA setup — phases: stage=$DO_STAGE  install=$DO_INSTALL  smoke=$DO_SMOKE"
echo "  DATA_ROOT: $DATA_ROOT"
echo "  repo:      $SCRIPT_DIR"
echo "================================================================"

# -----------------------------------------------------------------------------
# Phase 1: STAGE — download data + checkpoints
# -----------------------------------------------------------------------------
if [[ "$DO_STAGE" == "true" ]]; then
    echo
    echo "### [1/3] Staging data + checkpoints into $DATA_ROOT ..."

    SENTINEL="$DATA_ROOT/.staged_ok"
    if [[ -f "$SENTINEL" && "${FORCE_DOWNLOAD:-0}" != "1" ]]; then
        echo "    Found sentinel $SENTINEL — staging already completed successfully."
        echo "    (Pass FORCE_DOWNLOAD=1 to re-run wget anyway, e.g. if the"
        echo "    hosted assets have been updated.)"
    else
        echo "    First run: 20-40 min. Re-runs: wget -m walks every file in"
        echo "    the OME-Zarr trees (per-file HEAD checks against the server)"
        echo "    so even a no-op re-stage takes several minutes. We write a"
        echo "    sentinel file ($SENTINEL) at the end of a successful run so"
        echo "    subsequent --stage calls can fast-skip cleanly."

        mkdir -p "$DATA_ROOT/training" "$DATA_ROOT/test" \
                 "$DATA_ROOT/pretrained_models" \
                 "$DATA_ROOT/pretrained_models/DLCourse"

        cd "$DATA_ROOT/training"
        wget -m -np -nH --cut-dirs=6 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/training/zarrv3/a549_hoechst_cellmask_train_val.zarr/"

        cd "$DATA_ROOT/test"
        wget -m -np -nH --cut-dirs=6 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/test/zarrv3/a549_hoechst_cellmask_test.zarr/"

        cd "$DATA_ROOT/pretrained_models"
        wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto2D/VSCyto2D/epoch=399-step=23200.ckpt"
        # Part 2.5 reverse model (fluorescence -> phase). Hosted under the
        # dl_at_janelia/ tree because it's a DL@Janelia course model, not a
        # general-purpose VisCy release.
        cd "$DATA_ROOT/pretrained_models/DLCourse"
        wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/dl_at_janelia/DLCourse/pretrained_models/fluor2phase_step668.ckpt"

        cd "$START_DIR"
    fi

    # Verify everything we expect actually exists
    echo "    Verifying staged files ..."
    MISSING=0
    for f in \
        "$DATA_ROOT/training/a549_hoechst_cellmask_train_val.zarr" \
        "$DATA_ROOT/test/a549_hoechst_cellmask_test.zarr" \
        "$DATA_ROOT/pretrained_models/VSCyto2D/epoch=399-step=23200.ckpt" \
        "$DATA_ROOT/pretrained_models/DLCourse/fluor2phase_step668.ckpt"
    do
        if [[ -e "$f" ]]; then
            echo "      OK    $f"
        else
            echo "      MISS  $f" >&2
            MISSING=$((MISSING + 1))
        fi
    done
    if [[ "$MISSING" -gt 0 ]]; then
        echo "    [1/3] FAILED — $MISSING expected file(s) missing." >&2
        exit 1
    fi
    # Drop the sentinel so subsequent --stage calls fast-skip the wget walk.
    touch "$DATA_ROOT/.staged_ok"
    echo "    [1/3] OK."
fi

# -----------------------------------------------------------------------------
# Phase 2: INSTALL — validate pyproject.toml resolves cleanly
# -----------------------------------------------------------------------------
# Runs setup_student.sh end-to-end against a throwaway conda env named
# "$ENV_NAME-validate" so we don't disturb the TA's own working env. The
# script downloads no data because $DATA_ROOT is already populated by --stage.
ENV_NAME="${ENV_NAME:-06_image_translation}"
VALIDATE_ENV_NAME="${VALIDATE_ENV_NAME:-${ENV_NAME}-validate}"
if [[ "$DO_INSTALL" == "true" ]]; then
    echo
    echo "### [2/3] Validating install in a throwaway conda env ($VALIDATE_ENV_NAME) ..."
    if ! command -v conda >/dev/null 2>&1; then
        echo "    conda not on PATH; cannot validate install." >&2
        exit 1
    fi
    # Wipe any prior validation env so we always test a clean install.
    conda env remove -n "$VALIDATE_ENV_NAME" -y >/dev/null 2>&1 || true
    ENV_NAME="$VALIDATE_ENV_NAME" DATA_ROOT="$DATA_ROOT" \
        bash "$SCRIPT_DIR/setup_student.sh"
    echo "    [2/3] OK — pyproject.toml + setup_student.sh produce a working env."
    echo "    (Throwaway env left as '$VALIDATE_ENV_NAME'; remove with:"
    echo "       conda env remove -n $VALIDATE_ENV_NAME -y)"
fi

# -----------------------------------------------------------------------------
# Phase 3: SMOKE — run a short training pass through solution.py
# -----------------------------------------------------------------------------
# This catches problems that install-time can't: bad GPU drivers, OOM on the
# course node's specific hardware, broken solution.py logic, missing data
# fields, etc. We use fast_dev_run + a 2-epoch limited run, which finishes in
# a few minutes on any course-grade GPU.
if [[ "$DO_SMOKE" == "true" ]]; then
    echo
    echo "### [3/3] Smoke-testing solution.py on GPU ..."
    if ! command -v conda >/dev/null 2>&1; then
        echo "    conda not on PATH; cannot run smoke test." >&2
        exit 1
    fi
    # Prefer the validate env if we just built one, else the regular env.
    SMOKE_ENV="$ENV_NAME"
    if conda env list | awk '{print $1}' | grep -qx "$VALIDATE_ENV_NAME"; then
        SMOKE_ENV="$VALIDATE_ENV_NAME"
    fi
    if ! conda env list | awk '{print $1}' | grep -qx "$SMOKE_ENV"; then
        echo "    No usable conda env '$SMOKE_ENV' found. Run --install first (or --all)." >&2
        exit 1
    fi
    SMOKE_SCRIPT="$SCRIPT_DIR/scripts/ta_smoke_test.py"
    if [[ ! -f "$SMOKE_SCRIPT" ]]; then
        echo "    Smoke test script not found at $SMOKE_SCRIPT" >&2
        echo "    (this should be committed alongside setup_TA.sh)" >&2
        exit 1
    fi
    DATA_ROOT="$DATA_ROOT" conda run -n "$SMOKE_ENV" --no-capture-output \
        python "$SMOKE_SCRIPT"
    echo "    [3/3] OK — solution.py runs end-to-end on this node."
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
cat <<EOF

--------------------------------------------------------------------
TA setup complete.

  - data:           $DATA_ROOT
  - phases run:     stage=$DO_STAGE install=$DO_INSTALL smoke=$DO_SMOKE

Tell students to run:
  DATA_ROOT=$DATA_ROOT bash setup_student.sh

This will create their per-user venv + jupyter kernel and reuse the
pre-staged data (no re-download).
--------------------------------------------------------------------
EOF
