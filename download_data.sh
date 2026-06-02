#!/usr/bin/env bash
#
# Image-translation exercise — data + checkpoint downloader.
#
# Single source of truth for WHERE the data lives and HOW it is fetched.
# Both setup_TA.sh and setup_student.sh call this; you can also run it
# directly to just stage the data:
#
#   DATA_ROOT=/mnt/efs/dlmbl/data bash download_data.sh
#
# DATA_ROOT is the *parent* directory; the exercise data always lives in a
# per-kernel subdirectory under it ($DATA_ROOT/$KERNEL_NAME). A TA who stages
# with DATA_ROOT=/mnt/efs/dlmbl/data and a student who runs with the same
# DATA_ROOT therefore resolve to the same data dir.
#
# Re-runs are cheap: if all four expected artifacts are already present, the
# download is skipped. Pass FORCE_DOWNLOAD=1 to wget anyway (e.g. if the
# hosted assets were updated).

set -euo pipefail

KERNEL_NAME="${KERNEL_NAME:-06_image_translation}"
DATA_ROOT="${DATA_ROOT:-$HOME/data}"
DATA_DIR="$DATA_ROOT/$KERNEL_NAME"

TRAINING_ZARR="$DATA_DIR/training/a549_hoechst_cellmask_train_val.zarr"
TEST_ZARR="$DATA_DIR/test/a549_hoechst_cellmask_test.zarr"
VSCYTO2D_CKPT="$DATA_DIR/pretrained_models/VSCyto2D/epoch=399-step=23200.ckpt"
FLUOR2PHASE_CKPT="$DATA_DIR/pretrained_models/DLCourse/fluor2phase_step668.ckpt"

echo "Data root: $DATA_ROOT  (parent; \$KERNEL_NAME is appended)"
echo "Data dir:  $DATA_DIR"

# Shared-mount fast path: on the course AWS instances, TAs pre-stage the data
# at /mnt/efs/dlmbl/data/06_image_translation. If the user is on the default
# DATA_ROOT, that mount exists, and they don't already have their own
# $DATA_DIR, symlink $DATA_DIR -> the shared mount instead of re-downloading
# 14 GB per student. Set DATA_ROOT explicitly to opt out (e.g. on a laptop).
SHARED_DATA="/mnt/efs/dlmbl/data/$KERNEL_NAME"
if [[ "$DATA_ROOT" == "$HOME/data" ]] \
        && [[ -d "$SHARED_DATA" ]] \
        && [[ ! -e "$DATA_DIR" ]]; then
    echo "Found shared data at $SHARED_DATA — symlinking $DATA_DIR to it."
    mkdir -p "$(dirname "$DATA_DIR")"
    ln -s "$SHARED_DATA" "$DATA_DIR"
fi

mkdir -p "$DATA_DIR/training" "$DATA_DIR/test" \
         "$DATA_DIR/pretrained_models" \
         "$DATA_DIR/pretrained_models/DLCourse"

if [[ "${FORCE_DOWNLOAD:-0}" != "1" ]] \
        && [[ -d "$TRAINING_ZARR" && -d "$TEST_ZARR" \
              && -f "$VSCYTO2D_CKPT" && -f "$FLUOR2PHASE_CKPT" ]]; then
    echo "Data already present at $DATA_DIR — skipping download."
    echo "(Pass FORCE_DOWNLOAD=1 to re-fetch anyway.)"
    exit 0
fi

echo "Downloading data + checkpoints to $DATA_DIR ..."

cd "$DATA_DIR/training"
wget -m -np -nH --cut-dirs=6 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/training/zarrv3/a549_hoechst_cellmask_train_val.zarr/"

cd "$DATA_DIR/test"
wget -m -np -nH --cut-dirs=6 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/test/zarrv3/a549_hoechst_cellmask_test.zarr/"

cd "$DATA_DIR/pretrained_models"
wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto2D/VSCyto2D/epoch=399-step=23200.ckpt"

# Part 2.5 reverse model (fluorescence -> phase). Hosted under the
# dl_at_janelia/ tree because it's a DL@Janelia course model, not a
# general-purpose VisCy release.
cd "$DATA_DIR/pretrained_models/DLCourse"
wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/dl_at_janelia/DLCourse/pretrained_models/fluor2phase_step668.ckpt"

echo "Done. Data staged at $DATA_DIR"
