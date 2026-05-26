# Exercise: Image translation (Virtual Staining)

Written by Eduardo Hirata-Miyasaki, Ziwen Liu, and Shalin Mehta, CZ Biohub San Francisco, with many inputs and bugfixes from present and past TAs of the DL@MBL course (Diane Adjavon, Albert Dominguez Mantes, and others).

## Overview

In this exercise, we will predict fluorescence images of nuclei and plasma membrane markers from quantitative phase images of cells, i.e., we will *virtually stain* the nuclei and plasma membrane visible in the phase image.
This is an example of an image translation task. We will apply spatial and intensity augmentations to train robust models and evaluate their performance. Finally, we will explore the opposite process of predicting a phase image from a fluorescence membrane label.

![A549 phase to fluorescence timelapse](https://public.czbiohub.org/comp.micro/dl_at_janelia/DLCourse/assets/a549_timelapse.gif)

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

## Setup — Students

This exercise is included as a submodule of the main
[dl-janelia/DL-MBL-2025](https://github.com/dl-janelia/DL-MBL-2025)
course repo. From your clone of the course repo:

```bash
# Fetch the latest version of every exercise:
git submodule update --init --recursive --remote

# Then enter this exercise and run setup:
cd 06_image_translation
bash setup_student.sh
```

That's it — the script creates a `./.venv`, installs everything in
[pyproject.toml](pyproject.toml), registers a Jupyter kernel called
`06_image_translation`, and downloads the data (about 14 GB) into
`$DATA_ROOT` (default `$HOME/data/06_image_translation/`).

If your TA pre-staged the data on a shared mount, point `DATA_ROOT` at it to
skip the download:

```bash
DATA_ROOT=/path/to/shared/image_translation bash setup_student.sh
```

Everything is self-contained inside this folder.

## Setup — TAs

Pre-staging data, validating the install, and smoke-testing the notebook
on a course node are all handled by [setup_TA.sh](setup_TA.sh). See its
header comment for usage:

```bash
bash setup_TA.sh --help
```

Typical pre-course workflow on a shared mount:

```bash
DATA_ROOT=/path/to/shared/image_translation bash setup_TA.sh --all
```

Then tell students to run:

```bash
DATA_ROOT=/path/to/shared/image_translation bash setup_student.sh
```

## Run the exercise

After `setup_student.sh` finishes, you have three equivalent entry points.
Pick whichever you're most comfortable with — they all use the same kernel
and the same data.

### Option A — Jupyter Lab (recommended for this course)

```bash
./.venv/bin/jupyter lab            # opens in your browser
```

In the file browser, double-click **`exercise.ipynb`** (the version with
TODOs). At the top right, pick the **Python (06_image_translation)** kernel,
then run cells top-to-bottom with `Shift+Enter`. If you get stuck, peek at
**`solution.ipynb`** for the filled-in answer to that block.

### Option B — VSCode (notebook or cell mode)

Install VSCode plus the Python + Jupyter extensions. Then either:

- Open **`exercise.ipynb`** directly (it behaves like Jupyter Lab inside
  VSCode), or
- Open **`solution.py`** and use
[cell mode](https://code.visualstudio.com/docs/python/jupyter-support-py)
to run each `# %%` block interactively (`Ctrl+Enter` / `Shift+Enter`).

Either way, pick **Python (06_image_translation)** as the kernel from the
top-right kernel selector.

### Regenerating the notebooks

`exercise.ipynb` and `solution.ipynb` are regenerated from `solution.py` by a
GitHub Action on every push. To rebuild them locally (after editing
`solution.py`):

```bash
bash prepare-exercise.sh
```

### Re-registering the Jupyter kernel

If the kernel is missing from the dropdown (e.g. you reinstalled the venv):

```bash
./.venv/bin/python -m ipykernel install --user \
    --name 06_image_translation \
    --display-name "Python (06_image_translation)"
```

## References

- [Liu, Z. and Hirata-Miyasaki, E. et al. (2025) Robust virtual staining of landmark organelles with Cytoland. *Nature Machine Intelligence*](https://www.nature.com/articles/s42256-025-01046-2)
- [Guo et al. (2020) Revealing architectural order with quantitative label-free imaging and deep learning. *eLife*](https://elifesciences.org/articles/55502)

