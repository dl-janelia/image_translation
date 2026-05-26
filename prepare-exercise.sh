#!/usr/bin/env bash
# Regenerate exercise.ipynb and solution.ipynb from solution.py.
#
# Picks the first jupytext / jupyter on $PATH. Conda users:
# `conda activate 06_image_translation` first. The GitHub Actions
# build-notebooks workflow installs jupytext directly with pip, so plain
# $PATH is fine there too.
set -euo pipefail

# "cell_metadata_filter": "all" preserves cell tags including our solution tags
jupytext --to ipynb --update-metadata '{"jupytext":{"cell_metadata_filter":"all"}}' --update solution.py
jupytext --to ipynb --update-metadata '{"jupytext":{"cell_metadata_filter":"all"}}' --update solution.py --output exercise.ipynb
jupyter nbconvert solution.ipynb --ClearOutputPreprocessor.enabled=True --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags task --to notebook --output solution.ipynb
jupyter nbconvert exercise.ipynb --ClearOutputPreprocessor.enabled=True --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags solution --to notebook --output exercise.ipynb
