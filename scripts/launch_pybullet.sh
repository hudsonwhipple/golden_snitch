#!/bin/bash
# Quick sanity check: env loads and reset() works. No GUI or simulation — just a one-off test.
# For a visual PyBullet window, use gym-pybullet-drones examples in simulators/pybullet-drones.
cd "$(dirname "$0")/.."
PYTHON=python
[[ -f .venv/bin/python ]] && PYTHON=.venv/bin/python
$PYTHON -c "
from environments.pybullet_racing_env import PyBulletRacingEnv
env = PyBulletRacingEnv()
obs, info = env.reset()
print('PyBullet racing env OK. Obs shape:', obs.shape)
env.close()
print('(Headless check only; no window. For visuals, run gym-pybullet-drones examples.)')
"
