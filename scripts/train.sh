#!/bin/bash
# Multi-stage training orchestration for Golden Snitch
set -e

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# Activate venv
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi

TRACK="${TRACK:-simple_oval}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cpu}"

echo "=== Golden Snitch Training Pipeline ==="
echo "Track: $TRACK | Seed: $SEED | Device: $DEVICE"
echo ""

# Stage 1: Expert Data Generation
echo "--- Stage 1: Generating Expert Data ---"
python -m planning.expert_policy \
    --track "$TRACK" \
    --episodes 500 \
    --speed 2.0 \
    --perturbation 0.3 \
    --seed "$SEED"

EXPERT_DIR=$(ls -dt outputs/expert_data/expert_${TRACK}_* | head -1)
echo "Expert data: $EXPERT_DIR"
echo ""

# Stage 2: Behavior Cloning
echo "--- Stage 2: Behavior Cloning ---"
python -m learning.training.train_bc \
    --data "$EXPERT_DIR/expert_data.npz" \
    --track "$TRACK" \
    --epochs 50 \
    --batch-size 512 \
    --lr 0.001 \
    --seed "$SEED" \
    --device "$DEVICE"

BC_DIR=$(ls -dt outputs/bc/bc_${TRACK}_* | head -1)
echo "BC model: $BC_DIR"
echo ""

# Stage 3: PPO Fine-Tuning
echo "--- Stage 3: PPO Training ---"
python -m learning.training.train_ppo \
    --track "$TRACK" \
    --timesteps "${TIMESTEPS:-5000000}" \
    --n-envs "${N_ENVS:-4}" \
    --seed "$SEED" \
    --device "$DEVICE"

PPO_DIR=$(ls -dt outputs/ppo/ppo_${TRACK}_* | head -1)
echo "PPO model: $PPO_DIR"
echo ""

# Stage 4: Evaluation
echo "--- Stage 4: Evaluation ---"
python -m learning.evaluation.report eval \
    --model "$PPO_DIR/best_model.zip" \
    --track "$TRACK" \
    --episodes 100 \
    --output-dir "$PPO_DIR/eval"

echo ""
echo "=== Training Complete ==="
echo "Results: $PPO_DIR/eval/"
