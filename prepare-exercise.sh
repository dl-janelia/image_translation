#!/usr/bin/env bash
# Regenerate exercise.ipynb and solution.ipynb from solution.py.
#
# Uses the project venv (./.venv) if it exists; otherwise falls back to
# whatever jupytext/jupyter are on $PATH (matches the GitHub Actions
# build-notebooks workflow).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$SCRIPT_DIR/.venv/bin/jupytext" ]]; then
    export PATH="$SCRIPT_DIR/.venv/bin:$PATH"
fi

# "cell_metadata_filter": "all" preserves cell tags including our solution tags
jupytext --to ipynb --update-metadata '{"jupytext":{"cell_metadata_filter":"all"}}' --update solution.py
jupytext --to ipynb --update-metadata '{"jupytext":{"cell_metadata_filter":"all"}}' --update solution.py --output exercise.ipynb
jupyter nbconvert solution.ipynb --ClearOutputPreprocessor.enabled=True --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags task --to notebook --output solution.ipynb
jupyter nbconvert exercise.ipynb --ClearOutputPreprocessor.enabled=True --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags solution --to notebook --output exercise.ipynb
