#!/bin/bash
cd "$(dirname "$0")/.."
# Use venv if present so same Python has gymnasium, sb3, etc.
if [[ -f .venv/bin/python ]]; then
  exec .venv/bin/python learning/rl_algorithms/train_ppo.py
fi
exec python learning/rl_algorithms/train_ppo.py