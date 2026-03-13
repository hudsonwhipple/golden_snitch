#!/bin/bash
set -e

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# Use project venv so all deps land in one place and `python` sees them.
if [[ ! -d "$ROOT/.venv" ]]; then
  echo "Creating .venv..."
  python3 -m venv "$ROOT/.venv"
fi
echo "Using .venv at $ROOT/.venv"
source "$ROOT/.venv/bin/activate"

echo "Initializing submodules..."
git submodule update --init --recursive

echo "Installing core dependencies..."
pip install -r requirements.txt

# PyBullet: on Apple Silicon pip often has no wheel and building from source fails.
# (Conda is not used here so that all packages stay inside .venv.)
echo "Installing PyBullet..."
if pip install pybullet 2>/dev/null; then
  echo "PyBullet installed."
  echo "Installing gym-pybullet-drones (editable) from submodule..."
  pip install -e simulators/pybullet-drones
else
  if [[ "$(uname -m)" == "arm64" ]] && [[ "$(uname -s)" == "Darwin" ]]; then
    echo "PyBullet has no pip wheel on Apple Silicon; skipping gym-pybullet-drones."
    echo "Use the in-repo PyBulletRacingEnv (stub) for training, or for full sim use a conda env:"
    echo "  conda create -n drones python=3.12 && conda activate drones"
    echo "  conda install -c conda-forge pybullet && pip install -r requirements.txt"
    echo "  pip install -e simulators/pybullet-drones --no-deps"
    echo "  pip install 'pytest>=7.3,<10' scipy transforms3d matplotlib control"
  else
    echo "PyBullet install failed. Check errors above."
    exit 1
  fi
fi

echo ""
echo "Done. Activate the venv before running scripts:"
echo "  source .venv/bin/activate"
echo "Optional AirSim: pip install --no-build-isolation -r requirements-airsim.txt"
