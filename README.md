# Exercise: Image translation (Virtual Staining)

Written by Eduardo Hirata-Miyasaki, Ziwen Liu, and Shalin Mehta, CZ Biohub San Francisco, with many inputs and bugfixes from present and past TAs of the DL course (Diane Adjavon, Albert Dominguez Mantes, Trang Le, and Irene Mancebo Laguna).

## Overview

In this exercise, we will predict fluorescence images of nuclei and plasma membrane markers from quantitative phase images of cells, i.e., we will *virtually stain* the nuclei and plasma membrane visible in the phase image.
This is an example of an image translation task. We will apply spatial and intensity augmentations to train robust models and evaluate their performance. Finally, we will explore the opposite process of predicting a phase image from a fluorescence membrane label.

A549 phase to fluorescence timelapse

## Goals

### Part 1: Learn to use iohub (I/O library), VisCy dataloaders, and TensorBoard.

- Use a OME-Zarr dataset of 34 FOVs of adenocarcinomic human alveolar basal epithelial cells (A549),
each FOV has 3 channels (phase, nuclei, and cell membrane).
The nuclei were stained with DAPI and the cell membrane with Cellmask.
- Explore OME-Zarr using [iohub](https://czbiohub-sf.github.io/iohub/main/index.html)
and the high-content-screen (HCS) format.
- Use [MONAI](https://monai.io/) to implement data augmentations.

### Part 2: Train and evaluate the model to translate phase into fluorescence, and vice versa.

- Train a 2D UNeXt2 model to predict nuclei and membrane from phase images.
- Compare the performance of the trained model and a pre-trained model.
- Evaluate the model using pixel-level and instance-level metrics.

Checkout [VisCy](https://github.com/mehta-lab/VisCy) and the [cytoland](https://github.com/mehta-lab/VisCy/tree/main/applications/cytoland)
application — our deep learning pipeline for training and deploying computer vision models for image-based phenotyping, including the robust virtual
staining of landmark organelles. VisCy exploits recent advances in data and metadata formats ([OME-zarr](https://www.nature.com/articles/s41592-021-01326-w))
and DL frameworks ([PyTorch Lightning](https://lightning.ai/) and [MONAI](https://monai.io/)).

## Setup

This exercise lives as a submodule of
[dl-janelia/DL-MBL-2025](https://github.com/dl-janelia/DL-MBL-2025).
From your clone of the course repo:

```bash
git submodule update --init --recursive --remote
cd 04_image_translation
export DATA_ROOT=<TODO: course data folder>   # e.g. /mnt/efs/.../04_image_translation
bash setup_student.sh
```

`setup_student.sh` creates the `04_image_translation` conda env, installs
[pyproject.toml](pyproject.toml) into it, and registers the matching Jupyter
kernel. It does **not** download data — that's pre-staged by your TA. If you
need to fetch it yourself, point `DATA_ROOT` at the folder where you want
the data to live, then run the downloader:

```bash
export DATA_ROOT=/where/you/want/the/data   # this folder gets training/, test/, ...
bash download_data.sh
```

Requires `conda` on your PATH; install
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) first if you
don't have it.

## Run the exercise

Activate the env and launch Jupyter Lab (keep `DATA_ROOT` exported in the
same shell so the notebook can find the data):

```bash
export DATA_ROOT=<TODO: course data folder>   # e.g. /mnt/efs/.../04_image_translation
conda activate 04_image_translation
jupyter lab
```

Open `**exercise.ipynb**`, pick the **Python (04_image_translation)**
kernel, and run cells with `Shift+Enter`. Peek at `**solution.ipynb`**
when you get stuck.

VSCode users can open `exercise.ipynb` directly, or run `solution.py`
block-by-block in
[cell mode](https://code.visualstudio.com/docs/python/jupyter-support-py).

---

## For TAs

Stage data, validate the install, and smoke-test the notebook with one
command:

```bash
export DATA_ROOT=<TODO: course data folder>   # e.g. /mnt/efs/.../04_image_translation
bash setup_TA.sh --all
```

See `bash setup_TA.sh --help` for individual phases. To only stage the data:

```bash
export DATA_ROOT=<TODO: course data folder>   # e.g. /mnt/efs/.../04_image_translation
bash download_data.sh
```

Rebuild the notebooks after editing `solution.py`:

```bash
bash prepare-exercise.sh
```

Re-register the Jupyter kernel if it disappears from the dropdown:

```bash
conda activate 04_image_translation
python -m ipykernel install --user \
    --name 04_image_translation \
    --display-name "Python (04_image_translation)"
```

## References

- [Liu, Z. and Hirata-Miyasaki, E. et al. (2025) Robust virtual staining of landmark organelles with Cytoland. *Nature Machine Intelligence](https://www.nature.com/articles/s42256-025-01046-2)*
- [Guo et al. (2020) Revealing architectural order with quantitative label-free imaging and deep learning. *eLife](https://elifesciences.org/articles/55502)*