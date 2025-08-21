#!/usr/bin/env -S bash -i

START_DIR=$(pwd)

# Create conda environment
conda create -y --name 06_image_translation python=3.11

# Install ipykernel in the environment.
conda install -y ipykernel nbformat nbconvert black jupytext ipywidgets --name 06_image_translation
# Specifying the environment explicitly.
# conda activate sometimes doesn't work from within shell scripts.

# install viscy and its dependencies`s in the environment using pip.
# Find path to the environment - conda activate doesn't work from within shell scripts.
ENV_PATH=$(conda info --envs | grep 06_image_translation | awk '{print $NF}')
$ENV_PATH/bin/pip install "viscy[metrics,visual]==0.4.0a2"
$ENV_PATH/bin/pip install "jupyterlab"

# Create the directory structure
# NOTE: This takes about 20-40 minutes to run. This should be done before the exercise starts
output_dir=/mnt/efs/aimbl_2025
mkdir -p "$output_dir"/data/06_image_translation/training
mkdir -p "$output_dir"/data/06_image_translation/test
mkdir -p "$output_dir"/data/06_image_translation/pretrained_models
# ln -s "$output_dir"/data ~/data

# Change to the target directory
cd "$output_dir"/data/06_image_translation/training
# Download the OME-Zarr dataset recursively
wget -m -np -nH --cut-dirs=5 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/training/a549_hoechst_cellmask_train_val.zarr/"
cd "$output_dir"/data/06_image_translation/test
wget -m -np -nH --cut-dirs=5 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/test/a549_hoechst_cellmask_test.zarr/"
cd "$output_dir"/data/06_image_translation/pretrained_models
wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto2D/VSCyto2D/epoch=399-step=23200.ckpt"

# Change back to the starting directory
cd $START_DIR
