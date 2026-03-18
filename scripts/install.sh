#!/bin/bash
set -e

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# Create venv
if [[ ! -d "$ROOT/.venv" ]]; then
  echo "Creating .venv..."
  python3 -m venv "$ROOT/.venv"
fi
echo "Using .venv at $ROOT/.venv"
source "$ROOT/.venv/bin/activate"

echo "Initializing submodules..."
git submodule update --init --recursive

echo "Installing golden-snitch and dependencies..."
pip install -e ".[dev]"

# PyBullet: optional, for high-fidelity sim
echo ""
echo "Installing PyBullet (optional)..."
if pip install pybullet 2>/dev/null; then
  echo "PyBullet installed."
  pip install -e simulators/pybullet-drones --no-deps
else
  echo "PyBullet not available (common on Apple Silicon / Python 3.14+)."
  echo "The lightweight DroneRacingEnv works without PyBullet."
  echo ""
  echo "For PyBullet support, use conda with Python 3.12:"
  echo "  conda create -n drones python=3.12 && conda activate drones"
  echo "  conda install -c conda-forge pybullet"
  echo "  pip install -e '.[dev]'"
  echo "  pip install -e simulators/pybullet-drones --no-deps"
fi

echo ""
echo "Verifying installation..."
python -m pytest tests/ -q

echo ""
echo "Done. Activate the venv before running scripts:"
echo "  source .venv/bin/activate"
echo ""
echo "Quick start:"
echo "  bash scripts/train.sh"
echo ""
echo "Optional:"
echo "  pip install wandb && wandb login  # experiment tracking"
echo "  pip install -r requirements-airsim.txt  # AirSim support"
