"""Visualization and TensorBoard helpers for the image-translation exercise.

These functions are pulled out of solution.py / exercise.ipynb to keep the
notebook focused on the pedagogically interesting parts (data loading,
augmentations, model, loss, training, evaluation). Nothing here is a
research idea — it's all matplotlib / TensorBoard / RGB-compositing
plumbing.

Open this file if you want to see how a batch is rendered or how PCA
features are mapped to RGB.
"""
from __future__ import annotations

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from cmap import Colormap
from skimage.exposure import rescale_intensity


# ---------------------------------------------------------------------------
# TensorBoard plumbing
# ---------------------------------------------------------------------------
def find_free_port() -> int:
    """Return an OS-assigned free TCP port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def launch_tensorboard(log_dir: str):
    """Launch a TensorBoard process pointed at `log_dir`.

    Returns the subprocess handle so the caller can terminate it.
    """
    import subprocess

    port = find_free_port()
    tensorboard_cmd = f"tensorboard --logdir={log_dir} --port={port}"
    process = subprocess.Popen(tensorboard_cmd, shell=True)
    print(
        f"TensorBoard started at http://localhost:{port}. \n"
        "If you are using VSCode remote session, forward the port using the "
        "PORTS tab next to TERMINAL."
    )
    return process


def log_batch_tensorboard(batch, batchno, writer, card_name):
    """Log one (phase, nuclei, membrane) batch as a 3-column grid in TensorBoard.

    Parameters
    ----------
    batch : dict
        Dict with "source" (phase, ``B, 1, Z, Y, X``) and "target"
        (nuc/mem, ``B, 2, Z, Y, X``).
    batchno : int
        Step number for the TensorBoard event.
    writer : torch.utils.tensorboard.SummaryWriter
        Where to log.
    card_name : str
        The image card name in TensorBoard.
    """
    batch_phase = batch["source"][:, :, 0, :, :]
    batch_membrane = batch["target"][:, 1, 0, :, :].unsqueeze(1)
    batch_nuclei = batch["target"][:, 0, 0, :, :].unsqueeze(1)

    p1, p99 = np.percentile(batch_membrane, (0.1, 99.9))
    batch_membrane = np.clip((batch_membrane - p1) / (p99 - p1), 0, 1)

    p1, p99 = np.percentile(batch_nuclei, (0.1, 99.9))
    batch_nuclei = np.clip((batch_nuclei - p1) / (p99 - p1), 0, 1)

    p1, p99 = np.percentile(batch_phase, (0.1, 99.9))
    batch_phase = np.clip((batch_phase - p1) / (p99 - p1), 0, 1)

    [N, C, H, W] = batch_phase.shape
    interleaved_images = torch.zeros((3 * N, C, H, W), dtype=batch_phase.dtype)
    interleaved_images[0::3, :] = batch_phase
    interleaved_images[1::3, :] = batch_nuclei
    interleaved_images[2::3, :] = batch_membrane

    grid = torchvision.utils.make_grid(interleaved_images, nrow=3)
    writer.add_image(card_name, grid, batchno)


def log_batch_jupyter(batch, channel_names: list[str]):
    """Render one batch inline in Jupyter as a 3-column figure.

    Per-channel colormaps follow the standard fluorescence convention:
    phase = gray, nuclei = green, membrane = magenta.

    Parameters
    ----------
    batch : dict
        Dict with "source" and "target" as in :func:`log_batch_tensorboard`.
    channel_names : list of str
        Channel labels for the column titles, e.g. ``["Phase3D", "Nucl", "Mem"]``.
    """
    batch_phase = batch["source"][:, :, 0, :, :]
    batch_size = batch_phase.shape[0]
    batch_membrane = batch["target"][:, 1, 0, :, :].unsqueeze(1)
    batch_nuclei = batch["target"][:, 0, 0, :, :].unsqueeze(1)

    p1, p99 = np.percentile(batch_membrane, (0.1, 99.9))
    batch_membrane = np.clip((batch_membrane - p1) / (p99 - p1), 0, 1)

    p1, p99 = np.percentile(batch_nuclei, (0.1, 99.9))
    batch_nuclei = np.clip((batch_nuclei - p1) / (p99 - p1), 0, 1)

    p1, p99 = np.percentile(batch_phase, (0.1, 99.9))
    batch_phase = np.clip((batch_phase - p1) / (p99 - p1), 0, 1)

    n_channels = batch["target"].shape[1] + batch["source"].shape[1]
    plt.figure()
    fig, axes = plt.subplots(
        batch_size, n_channels, figsize=(n_channels * 2, batch_size * 2)
    )
    phase_cmap = Colormap("gray").to_mpl()
    nuclei_cmap = Colormap("green").to_mpl()
    membrane_cmap = Colormap("magenta").to_mpl()
    [N, C, H, W] = batch_phase.shape
    for sample_id in range(batch_size):
        axes[sample_id, 0].imshow(batch_phase[sample_id, 0], cmap=phase_cmap)
        axes[sample_id, 1].imshow(batch_nuclei[sample_id, 0], cmap=nuclei_cmap)
        axes[sample_id, 2].imshow(batch_membrane[sample_id, 0], cmap=membrane_cmap)
        for i in range(n_channels):
            ax = axes[sample_id, i]
            ax.set_xticks([0, W - 1])
            ax.set_yticks([0, H - 1])
            ax.tick_params(axis="both", labelsize=7, length=2, pad=1)
            ax.set_title(channel_names[i])
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Image post-processing for visualization
# ---------------------------------------------------------------------------
def process_image(image: np.ndarray) -> np.ndarray:
    """Clip an image to its [0.5, 99.5] percentile range (display rescaling)."""
    p_low, p_high = np.percentile(image, (0.5, 99.5))
    return np.clip(image, p_low, p_high)


# ---------------------------------------------------------------------------
# RGB composite helpers (used in the Part 2 visualization cells)
# ---------------------------------------------------------------------------
class Color(NamedTuple):
    r: float
    g: float
    b: float


# Standard palettes for two-channel fluorescence composites.
BOP_ORANGE = Color(0.972549, 0.6784314, 0.1254902)
BOP_BLUE = Color(BOP_ORANGE.b, BOP_ORANGE.g, BOP_ORANGE.r)
GREEN = Color(0.0, 1.0, 0.0)
MAGENTA = Color(1.0, 0.0, 1.0)


def rescale_clip(image: torch.Tensor) -> np.ndarray:
    """Rescale a 2D image to [0, 1] and broadcast to 3 channels for compositing."""
    return rescale_intensity(image, out_range=(0, 1))[..., None].repeat(3, axis=-1)


def composite_nuc_mem(
    image: torch.Tensor, nuc_color: Color, mem_color: Color
) -> np.ndarray:
    """Blend a (nuc, mem) two-channel image into a single RGB composite."""
    c_nuc = rescale_clip(image[0]) * nuc_color
    c_mem = rescale_clip(image[1]) * mem_color
    return rescale_intensity(c_nuc + c_mem, out_range=(0, 1))


def clip_p(image: np.ndarray) -> np.ndarray:
    """Clip to [1, 99] percentile and rescale to [0, 1] for display."""
    return rescale_intensity(image.clip(*np.percentile(image, [1, 99])))


def clip_highlight(image: np.ndarray) -> np.ndarray:
    """Clip to [0, 99.5] percentile and rescale to [0, 1] for display."""
    return rescale_intensity(image.clip(0, np.percentile(image, 99.5)))
