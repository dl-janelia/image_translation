# Exercise 6: Image translation

Written by Eduardo Hirata-Miyasaki, Ziwen Liu, and Shalin Mehta, CZ Biohub San Francisco with many inputs from Diane Adjavon for the AI@MBL2025 course

## Overview

In this exercise, we will _virtually stain_ the nuclei and plasma membrane from the quantitative phase image (QPI), i.e., translate QPI images into fluoresence images of nuclei and plasma membranes.
QPI encodes multiple cellular structures and virtual staining decomposes these structures. After the model is trained, one only needs to acquire label-free QPI data.
This strategy solves the problem as "multi-spectral imaging", but is more compatible with live cell imaging and high-throughput screening.
Virtual staining is often a step towards multiple downstream analyses: segmentation, tracking, and cell state phenotyping.

In this exercise, you will:
- Train a model to predict the fluorescence images of nuclei and plasma membranes from QPI images
- Make it robust to variations in imaging conditions using data augmentions
- Segment the cells
- Use regression and segmentation metrics to evalute the models
- Visualize the image transform learned by the model
- Understand the failure modes of the trained model

[![HEK293T](https://raw.githubusercontent.com/mehta-lab/VisCy/main/docs/figures/svideo_1.png)](https://github.com/mehta-lab/VisCy/assets/67518483/d53a81eb-eb37-44f3-b522-8bd7bddc7755)
(Click on image to play video)

### Goals

#### Part 1: Train a virtual staining model

  - Explore OME-Zarr using [iohub](https://czbiohub-sf.github.io/iohub/main/index.html)
  and the high-content-screen (HCS) format.
  - Use our `viscy.data.HCSDataloader()` dataloader and explore the  3 channel (phase, fluoresecence nuclei and cell membrane) 
  A549 cell dataset. 
  - Implement data augmentations [MONAI](https://monai.io/) to train a robust model to imaging parameters and conditions. 
  - Use tensorboard to log the augmentations, training and validation losses and batches
  - Start the training of the UNeXt2 model to predict nuclei and membrane from phase images.

#### Part 2:Evaluate the model to translate phase into fluorescence.
  - Compare the performance of your trained model with the _VSCyto2D_ pre-trained model.
  - Evaluate the model using pixel-level and instance-level metrics.

#### Part 3: Visualize the image transforms learned by the model and explore the model's regime of validity
  - Visualize the first 3 principal componets mapped to a color space in each encoder and decoder block.
  - Explore the model's regime of validity by applying blurring and scaling transforms to the input phase image.

#### For more information:
Checkout [VisCy](https://github.com/mehta-lab/VisCy),
our deep learning pipeline for training and deploying computer vision models
for image-based phenotyping including the robust virtual staining of landmark organelles.

VisCy exploits recent advances in data and metadata formats
([OME-zarr](https://www.nature.com/articles/s41592-021-01326-w)) and DL frameworks,
[PyTorch Lightning](https://lightning.ai/) and [MONAI](https://monai.io/).


## Setup

Make sure that you are inside of the `image_translation` folder by using the `cd` command to change directories if needed.

Run the setup script to create the environment for this exercise and download the dataset.
```bash
sh setup_student.sh

```
Activate your environment
```bash
conda activate 06_image_translation
```

## Use vscode

Install vscode, install jupyter extension inside vscode, and setup [cell mode](https://code.visualstudio.com/docs/python/jupyter-support-py). Open [exercise.py](exercise.py) and run the script interactively.

## Use Jupyter Notebook / Lab

The matching exercise and solution notebooks can be found [here](https://github.com/dlmbl/image_translation/tree/28e0e515b4a8ad3f392a69c8341e105f730d204f) on the course repository.

Launch a jupyter lab environment

```
jupyter lab
```

...and continue with the instructions in the notebook.

If `06_image_translation` is not available as a kernel in jupyter, run:

```
python -m ipykernel install --user --name=06_image_translation
```

### References

- [Liu, Z. and Hirata-Miyasaki, E. et al. (2024) Robust Virtual Staining of Cellular Landmarks](https://www.nature.com/articles/s42256-025-01046-2)
- [Guo et al. (2020) Revealing architectural order with quantitative label-free imaging and deep learning. eLife](https://elifesciences.org/articles/55502)
