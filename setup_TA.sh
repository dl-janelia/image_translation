#!/usr/bin/env -S bash -i
#
# Image-translation exercise — TA setup.
#
# Three phases, run in order before the course (or --all for everything):
#
#   1. STAGE   (default)  Download ~14 GB of data + checkpoints into
#                         $DATA_ROOT (delegates to download_data.sh).
#   2. INSTALL            Run setup_student.sh in a throwaway conda env to
#                         confirm pyproject.toml still resolves cleanly.
#   3. SMOKE              Run a short training pass through solution.py on a
#                         GPU to confirm the notebook works on a course node.
#
# Usage:
#   bash setup_TA.sh                  # stage data only (default)
#   bash setup_TA.sh --stage          # stage data
#   bash setup_TA.sh --install        # validate install (data already staged)
#   bash setup_TA.sh --smoke          # smoke-test notebook on GPU
#   bash setup_TA.sh --all            # all three (recommended ~1 week before)
#
# Export DATA_ROOT first (the data folder, holding training/test/
# pretrained_models). Stage onto a shared mount, then point students at the
# same DATA_ROOT:
#   export DATA_ROOT=/mnt/efs/dl_jrc/data/06_image_translation  
#   bash setup_TA.sh --all
#   # students: export DATA_ROOT=<same> && bash setup_student.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-06_image_translation}"
VALIDATE_ENV_NAME="${VALIDATE_ENV_NAME:-${ENV_NAME}-validate}"

if [[ -z "${DATA_ROOT:-}" ]]; then
    echo "ERROR: DATA_ROOT is not set. Export the data folder first, e.g.:" >&2
    echo "  export DATA_ROOT=/mnt/efs/dlmbl/data/06_image_translation" >&2
    exit 1
fi

require_conda() {
    command -v conda >/dev/null 2>&1 && return 0
    echo "conda not on PATH; cannot continue." >&2
    exit 1
}

env_exists() {
    conda env list | awk '{print $1}' | grep -qx "$1"
}

# --- Parse phase flags -------------------------------------------------------
DO_STAGE=false DO_INSTALL=false DO_SMOKE=false
if [[ $# -eq 0 ]]; then
    DO_STAGE=true
else
    for arg in "$@"; do
        case "$arg" in
            --stage)   DO_STAGE=true ;;
            --install) DO_INSTALL=true ;;
            --smoke)   DO_SMOKE=true ;;
            --all)     DO_STAGE=true; DO_INSTALL=true; DO_SMOKE=true ;;
            -h|--help) grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
            *)         echo "Unknown flag: $arg (try --help)" >&2; exit 2 ;;
        esac
    done
fi

echo "================================================================"
echo "TA setup — stage=$DO_STAGE install=$DO_INSTALL smoke=$DO_SMOKE"
echo "  DATA_ROOT: $DATA_ROOT"
echo "================================================================"

# --- Phase 1: STAGE ----------------------------------------------------------
if [[ "$DO_STAGE" == "true" ]]; then
    echo
    echo "### [1/3] Staging data + checkpoints ..."
    DATA_ROOT="$DATA_ROOT" bash "$SCRIPT_DIR/download_data.sh"
fi

# --- Phase 2: INSTALL --------------------------------------------------------
# Run setup_student.sh against a throwaway env so we don't disturb the TA's
# working env. No data is downloaded (DATA_ROOT is already staged).
if [[ "$DO_INSTALL" == "true" ]]; then
    echo
    echo "### [2/3] Validating install in throwaway env '$VALIDATE_ENV_NAME' ..."
    require_conda
    conda env remove -n "$VALIDATE_ENV_NAME" -y >/dev/null 2>&1 || true
    ENV_NAME="$VALIDATE_ENV_NAME" DATA_ROOT="$DATA_ROOT" \
        bash "$SCRIPT_DIR/setup_student.sh"
    echo "    Remove the throwaway env with: conda env remove -n $VALIDATE_ENV_NAME -y"
fi

# --- Phase 3: SMOKE ----------------------------------------------------------
# Catches what install-time can't: GPU drivers, OOM, broken solution.py logic.
if [[ "$DO_SMOKE" == "true" ]]; then
    echo
    echo "### [3/3] Smoke-testing solution.py on GPU ..."
    require_conda

    # Prefer the validate env if it exists, else the regular env.
    SMOKE_ENV="$ENV_NAME"
    env_exists "$VALIDATE_ENV_NAME" && SMOKE_ENV="$VALIDATE_ENV_NAME"
    if ! env_exists "$SMOKE_ENV"; then
        echo "    No conda env '$SMOKE_ENV'. Run --install first (or --all)." >&2
        exit 1
    fi

    DATA_ROOT="$DATA_ROOT" conda run -n "$SMOKE_ENV" --no-capture-output \
        python "$SCRIPT_DIR/scripts/ta_smoke_test.py"
fi

cat <<EOF

--------------------------------------------------------------------
TA setup complete.
  - data:       $DATA_ROOT
  - phases run: stage=$DO_STAGE install=$DO_INSTALL smoke=$DO_SMOKE

Tell students to run (same DATA_ROOT reuses the staged data):
  export DATA_ROOT=$DATA_ROOT
  bash setup_student.sh
--------------------------------------------------------------------
EOF
