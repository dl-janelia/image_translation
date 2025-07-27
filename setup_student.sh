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
$ENV_PATH/bin/pip install "viscy[metrics,visual]>=0.3.0,<0.4.0"
$ENV_PATH/bin/pip install "jupyterlab"

# Create the directory structure
# output_dir=/mnt/efs/dlmbl
# Assuming that we ran the setup_TA.sh that downloads the data to nfs shared storage
# Creating the simlink to the NFS partition
# ln -s "$output_dir"/data ~/data

# Change back to the starting directory
cd $START_DIR
