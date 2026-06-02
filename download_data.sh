#!/usr/bin/env bash
#
# Download the exercise data + checkpoints into $DATA_ROOT.
# $DATA_ROOT is the folder that will hold training/, test/, pretrained_models/.
#
#   export DATA_ROOT=/mnt/efs/dlmbl/data/06_image_translation
#   bash download_data.sh
#
# Skips the download if the data is already present (FORCE_DOWNLOAD=1 overrides).

set -euo pipefail

if [[ -z "${DATA_ROOT:-}" ]]; then
    echo "ERROR: export DATA_ROOT first, e.g. /mnt/efs/dlmbl/data/06_image_translation" >&2
    exit 1
fi

mkdir -p "$DATA_ROOT"/{training,test,pretrained_models/DLCourse}

if [[ "${FORCE_DOWNLOAD:-0}" != "1" \
      && -d "$DATA_ROOT/training/a549_hoechst_cellmask_train_val.zarr" \
      && -d "$DATA_ROOT/test/a549_hoechst_cellmask_test.zarr" \
      && -f "$DATA_ROOT/pretrained_models/VSCyto2D/epoch=399-step=23200.ckpt" \
      && -f "$DATA_ROOT/pretrained_models/DLCourse/fluor2phase_step668.ckpt" ]]; then
    echo "Data already present at $DATA_ROOT — skipping (FORCE_DOWNLOAD=1 to re-fetch)."
    exit 0
fi

echo "Downloading data + checkpoints to $DATA_ROOT ..."
BASE="https://public.czbiohub.org/comp.micro"

cd "$DATA_ROOT/training"
wget -m -np -nH --cut-dirs=6 -R "index.html*" "$BASE/viscy/VS_datasets/VSCyto2D/training/zarrv3/a549_hoechst_cellmask_train_val.zarr/"

cd "$DATA_ROOT/test"
wget -m -np -nH --cut-dirs=6 -R "index.html*" "$BASE/viscy/VS_datasets/VSCyto2D/test/zarrv3/a549_hoechst_cellmask_test.zarr/"

cd "$DATA_ROOT/pretrained_models"
wget -m -np -nH --cut-dirs=4 -R "index.html*" "$BASE/viscy/VS_models/VSCyto2D/VSCyto2D/epoch=399-step=23200.ckpt"

# Part 2.5 reverse model (fluorescence -> phase), under the dl_at_janelia/ tree.
cd "$DATA_ROOT/pretrained_models/DLCourse"
wget -m -np -nH --cut-dirs=4 -R "index.html*" "$BASE/dl_at_janelia/DLCourse/pretrained_models/fluor2phase_step668.ckpt"

echo "Done. Data staged at $DATA_ROOT"
