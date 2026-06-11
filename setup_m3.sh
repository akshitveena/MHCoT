#!/bin/bash
# setup_m3.sh — clean rebuild of the `mhcot` conda env for Apple Silicon M3.
#
# Run from the project root:
#     bash setup_m3.sh
#
# Installs everything we need in ONE pip call so version drift can't happen.
# All pins are mutually compatible and known to work as of June 2026.

set -e   # exit on first error
set -u   # error on unset variables

echo "==> [1/6] removing old mhcot env (if exists)"
conda deactivate 2>/dev/null || true
conda env remove -n mhcot -y 2>/dev/null || true

echo "==> [2/6] creating fresh env: python=3.11 + pip"
conda create -n mhcot -y python=3.11 pip

# Activate inside this script (conda activate doesn't propagate to subshells
# without sourcing the conda init script first)
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate mhcot

echo "==> [3/6] verifying we're in the right env"
PY_BIN="$(which python)"
PIP_BIN="$(which pip)"
echo "python: ${PY_BIN}"
echo "pip:    ${PIP_BIN}"
python --version
if [[ "${PY_BIN}" != *"envs/mhcot/bin/python"* ]]; then
    echo "[ERROR] python is not inside the mhcot env. Aborting."
    exit 1
fi

echo "==> [4/6] installing all dependencies in one pip call"
pip install --upgrade pip
pip install \
    "torch>=2.4,<2.8" "torchvision" "torchaudio" \
    "numpy<2" "pyarrow>=15,<17" "pandas<2.2" \
    "pytz" "python-dateutil" "tzdata" \
    "transformers>=4.45,<5" "huggingface_hub>=0.24,<0.30" \
    "datasets>=3.0,<4" "accelerate>=0.34,<1" \
    "sentencepiece" "tqdm" "matplotlib" "scikit-learn" "scipy" \
    "openai<1.0" "graph_of_thoughts==0.0.2" \
    "pytest" "ipython"

echo "==> [5/6] freezing requirements.txt for reproducibility"
pip freeze > requirements.txt

echo "==> [6/6] smoke checks (imports + MPS + Besta)"
python - <<'PYEOF'
import torch, transformers, datasets, numpy as np
print(f"torch         {torch.__version__}")
print(f"MPS available {torch.backends.mps.is_available()}")
print(f"numpy         {np.__version__}")
print(f"transformers  {transformers.__version__}")
print(f"datasets      {datasets.__version__}")

from graph_of_thoughts import controller, operations, parser, prompter
print("graph_of_thoughts OK")

# Tiny MPS sanity check — confirms real matmul works
if torch.backends.mps.is_available():
    x = torch.randn(100, 100, device="mps")
    y = (x @ x).sum()
    print(f"MPS matmul check: sum = {y.item():.2f}")

print("--- setup complete ---")
PYEOF

echo ""
echo "All set. Activate the env in any new shell with:"
echo "    conda activate mhcot"
