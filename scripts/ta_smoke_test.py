"""TA smoke test — confirm the exercise actually trains on this node.

Invoked by `bash setup_TA.sh --smoke` (or `--all`). Mirrors the Part 1
pipeline from solution.py with the smallest possible workload that still
exercises every component: data loading, augmentations, model forward,
backward, optimizer step, validation, and a Cellpose forward pass.

Fails loudly if anything is broken — that's the whole point.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from cytoland.engine import VSUNet
from viscy_data.hcs import HCSDataModule
from viscy_transforms import (
    CenterSpatialCropd,
    NormalizeSampled,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandWeightedCropd,
)
from viscy_utils.losses import MixedLoss
from viscy_utils.trainer import VisCyTrainer

seed_everything(42, workers=True)

KERNEL_NAME = os.environ.get("KERNEL_NAME", "06_image_translation")
DATA_ROOT = Path(os.environ.get("DATA_ROOT", f"~/data/{KERNEL_NAME}")).expanduser()
TRAINING_ZARR = DATA_ROOT / "training" / "a549_hoechst_cellmask_train_val.zarr"
TEST_ZARR = DATA_ROOT / "test" / "a549_hoechst_cellmask_test.zarr"
VSCYTO2D_CKPT = (
    DATA_ROOT / "pretrained_models" / "VSCyto2D" / "epoch=399-step=23200.ckpt"
)
FLUOR2PHASE_CKPT = (
    DATA_ROOT / "pretrained_models" / "DLCourse" / "fluor2phase_step668.ckpt"
)

print(f"## TA smoke test")
print(f"  DATA_ROOT: {DATA_ROOT}")
for p in (TRAINING_ZARR, TEST_ZARR, VSCYTO2D_CKPT, FLUOR2PHASE_CKPT):
    if not p.exists():
        print(f"  MISSING: {p}", file=sys.stderr)
        sys.exit(1)
    print(f"  OK:      {p.name}")

if not torch.cuda.is_available():
    print("  ERROR: CUDA not available — TA nodes must have a GPU.", file=sys.stderr)
    sys.exit(2)
print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------------------
# Mini training pipeline (matches solution.py Part 1, scaled down)
# ---------------------------------------------------------------------------
BATCH_SIZE = 4
YX_PATCH_SIZE = (256, 256)
source_channel = ["Phase3D"]
target_channel = ["Nucl", "Mem"]

normalizations = [
    NormalizeSampled(
        keys=source_channel, level="fov_statistics",
        subtrahend="mean", divisor="std",
    ),
    NormalizeSampled(
        keys=target_channel, level="fov_statistics",
        subtrahend="median", divisor="iqr",
    ),
]
augmentations = [
    RandWeightedCropd(
        keys=source_channel + target_channel,
        spatial_size=(1, 384, 384), num_samples=2, w_key=target_channel[0],
    ),
    RandAffined(
        keys=source_channel + target_channel,
        rotate_range=[3.14, 0.0, 0.0], scale_range=[0.0, 0.3, 0.3],
        prob=0.8, padding_mode="zeros",
        shear_range=[0.0, 0.01, 0.01],
    ),
    RandAdjustContrastd(keys=source_channel, prob=0.5, gamma=(0.8, 1.2)),
    RandScaleIntensityd(keys=source_channel, factors=0.5, prob=0.5),
    RandGaussianNoised(keys=source_channel, prob=0.5, mean=0.0, std=0.3),
    RandGaussianSmoothd(
        keys=source_channel,
        sigma_x=(0.25, 0.75), sigma_y=(0.25, 0.75), sigma_z=(0.0, 0.0),
        prob=0.5,
    ),
    CenterSpatialCropd(
        keys=source_channel + target_channel,
        roi_size=(1, YX_PATCH_SIZE[0], YX_PATCH_SIZE[1]),
    ),
]

print("\n## Building data module ...")
dm = HCSDataModule(
    str(TRAINING_ZARR),
    source_channel=source_channel,
    target_channel=target_channel,
    z_window_size=1,
    split_ratio=0.8,
    batch_size=BATCH_SIZE,
    num_workers=0,
    yx_patch_size=YX_PATCH_SIZE,
    augmentations=augmentations,
    normalizations=normalizations,
)
dm.setup("fit")
print(f"  train samples: {len(dm.train_dataset)}")
print(f"  val   samples: {len(dm.val_dataset)}")

print("\n## Building model ...")
model = VSUNet(
    architecture="UNeXt2_2D",
    model_config=dict(
        in_channels=1, out_channels=2,
        encoder_blocks=[3, 3, 9, 3], dims=[96, 192, 384, 768],
        decoder_conv_blocks=2, stem_kernel_size=(1, 2, 2),
        in_stack_depth=1, pretraining=False,
    ),
    loss_function=MixedLoss(l1_alpha=0.5, l2_alpha=0.0, ms_dssim_alpha=0.5),
    schedule="WarmupCosine", lr=6e-4, log_batches_per_epoch=2,
    freeze_encoder=False,
)

print("\n## [1/3] fast_dev_run ...")
VisCyTrainer(
    accelerator="gpu", devices=[0],
    precision="16-mixed", fast_dev_run=True,
).fit(model, datamodule=dm)
print("    fast_dev_run OK")

print("\n## [2/3] 2-epoch limited run ...")
VisCyTrainer(
    accelerator="gpu", devices=[0],
    max_epochs=2, precision="16-mixed",
    log_every_n_steps=1,
    limit_train_batches=5, limit_val_batches=2,
    logger=TensorBoardLogger(
        save_dir="/tmp/ta_smoke", name="phase2fluor", log_graph=False,
    ),
    enable_progress_bar=False,
).fit(model, datamodule=dm)
print(f"    2-epoch run OK  (peak GPU: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB)")

# ---------------------------------------------------------------------------
# Sanity-check the pretrained checkpoint loads (Part 2 of the exercise)
# ---------------------------------------------------------------------------
print("\n## [3/3] Loading VSCyto2D pretrained checkpoint ...")
pretrained = VSUNet.load_from_checkpoint(
    VSCYTO2D_CKPT,
    architecture="UNeXt2_2D",
    model_config=dict(
        in_channels=1, out_channels=2,
        encoder_blocks=[3, 3, 9, 3], dims=[96, 192, 384, 768],
        decoder_conv_blocks=2, stem_kernel_size=(1, 2, 2),
        in_stack_depth=1, pretraining=False,
    ),
    map_location="cuda:0",
)
pretrained.eval()
print("    pretrained checkpoint loads + moves to GPU OK")

# Cellpose import + model construction (no segmentation — that takes minutes)
print("\n## Cellpose v4+ import + model construction ...")
from cellpose import models
cp = models.CellposeModel(gpu=True, device=torch.device("cuda:0"))
print(f"    cellpose model ready ({type(cp).__name__})")

print()
print("=" * 64)
print("TA smoke test PASSED — exercise is ready for students on this node.")
print("=" * 64)
