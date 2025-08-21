#!/usr/bin/env -S bash -i

START_DIR=$(pwd)
ENV_NAME="04_image_translation"

# Create conda environment
conda create -y --name $ENV_NAME python=3.11

# Install ipykernel in the environment.
conda install -y ipykernel nbformat nbconvert black jupytext ipywidgets --name $ENV_NAME
# Specifying the environment explicitly.
# conda activate sometimes doesn't work from within shell scripts.

# install viscy and its dependencies`s in the environment using pip.
# Find path to the environment - conda activate doesn't work from within shell scripts.
ENV_PATH=$(conda info --envs | grep $ENV_NAME | awk '{print $NF}')
$ENV_PATH/bin/pip install "viscy[metrics,visual]==0.4.0a2"
$ENV_PATH/bin/pip install "jupyterlab"

# Change back to the starting directory
cd $START_DIR
