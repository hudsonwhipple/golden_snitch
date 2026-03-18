# Golden Snitch - Training Guide

Step-by-step instructions for training a drone racing policy from scratch.

## Prerequisites

```bash
# Clone and set up
git clone <repo-url> && cd golden_snitch
git submodule update --init --recursive
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Verify setup:
```bash
python -m pytest tests/ -v
```

## Quick Start (Automated)

Run the full pipeline in one command:

```bash
bash scripts/train.sh
```

Override defaults with environment variables:

```bash
TRACK=figure_eight TIMESTEPS=1000000 N_ENVS=4 SEED=42 bash scripts/train.sh
```

## Stage-by-Stage Training

### Stage 1: Generate Expert Data

Generate demonstration trajectories by tracking optimal paths through gates:

```bash
python -m planning.expert_policy \
    --track simple_oval \
    --episodes 500 \
    --speed 2.0 \
    --perturbation 0.3 \
    --seed 42
```

Output: `outputs/expert_data/expert_simple_oval_<timestamp>/expert_data.npz`

**What to check:**
- Total samples should be ~40-60K for 500 episodes
- Successful episodes should be >90%
- Each episode should contain 100-300 timesteps

### Stage 2: Behavior Cloning

Train a policy to imitate the expert demonstrations:

```bash
python -m learning.training.train_bc \
    --data outputs/expert_data/<run_id>/expert_data.npz \
    --track simple_oval \
    --epochs 50 \
    --batch-size 512 \
    --lr 0.001 \
    --device cpu \
    --seed 42
```

Output: `outputs/bc/bc_simple_oval_<timestamp>/best_model.pt`

**What to check:**
- Validation loss should decrease steadily
- Final val loss should be < 0.01 (ideally < 0.005)
- Training and validation curves should converge (no overfitting)

### Stage 3: PPO Fine-Tuning

Refine the policy with reinforcement learning:

```bash
python -m learning.training.train_ppo \
    --track simple_oval \
    --timesteps 5000000 \
    --n-envs 4 \
    --seed 42 \
    --device cpu
```

Output: `outputs/ppo/ppo_simple_oval_<timestamp>/`

**Key flags:**
- `--bc-model`: Path to BC model for weight initialization (optional)
- `--resume`: Path to checkpoint to resume training
- `--no-curriculum`: Disable curriculum learning

**What to check (via TensorBoard):**
```bash
tensorboard --logdir outputs/ppo/<run_id>/tb_logs
```

- `rollout/ep_rew_mean`: Should increase over training
- `rollout/ep_len_mean`: Should decrease (faster episodes)
- Eval metrics logged every 250K steps: gate success rate should increase

### Stage 4: Evaluation

Evaluate any saved model:

```bash
python -m learning.evaluation.evaluate \
    --model outputs/ppo/<run_id>/best_model.zip \
    --track simple_oval \
    --episodes 100 \
    --output-dir outputs/ppo/<run_id>/eval
```

Generate full report with trajectory plots:

```bash
python -m learning.evaluation.report eval \
    --model outputs/ppo/<run_id>/best_model.zip \
    --track simple_oval \
    --episodes 100 \
    --output-dir outputs/ppo/<run_id>/eval
```

Compare two models:

```bash
python -m learning.evaluation.report compare \
    --model-a outputs/bc/<run_id>/best_model.zip \
    --model-b outputs/ppo/<run_id>/best_model.zip \
    --name-a "BC Only" \
    --name-b "PPO Fine-tuned" \
    --track simple_oval \
    --output-dir outputs/comparison
```

## Training Tips

### Compute Estimates (Apple Silicon M5pro)

| Stage                      | Duration   | Memory |
|----------------------------|-----------|--------|
| Expert generation (500 ep) | ~30 min   | 2 GB   |
| Behavior cloning (50 ep)   | ~10 min   | 1 GB   |
| PPO (5M steps, 4 envs)    | ~2-3 hrs  | 4 GB   |

### Hyperparameter Tuning

The most impactful parameters to tune:

1. **Reward weights** (`configs/default.yaml` → `reward:`): If the drone circles without passing gates, increase `gate_bonus`. If it crashes too much, increase `collision_penalty`.

2. **Learning rate**: Start with 3e-4, decrease if training is unstable.

3. **Number of environments**: More = faster wallclock, but more memory. 4-8 is the sweet spot.

4. **Curriculum thresholds**: If stuck at a level, lower `promotion_threshold` to 0.70.

### Troubleshooting

**Policy crashes immediately**: The expert data quality is critical. Verify expert trajectories are smooth by checking the expert metadata (successful episodes count). Increase expert PD gains if trajectories are too aggressive.

**Training plateaus**: Check reward breakdown in evaluation. If `R_progress` is high but `R_gate` is zero, the policy approaches gates but doesn't fly through them -- increase gate bonus or add approach waypoints to the trajectory.

**High crash rate after curriculum promotion**: The level jump may be too steep. Add intermediate curriculum levels or lower the promotion threshold.

## Artifact Structure

```
outputs/
├── expert_data/
│   └── expert_simple_oval_<ts>/
│       ├── expert_data.npz
│       └── metadata.json
├── bc/
│   └── bc_simple_oval_<ts>/
│       ├── best_model.pt
│       ├── final_model.pt
│       ├── obs_mean.npy
│       ├── obs_std.npy
│       ├── eval_results.json
│       └── training_config.yaml
└── ppo/
    └── ppo_simple_oval_<ts>/
        ├── best_model.zip
        ├── final_model.zip
        ├── training_config.yaml
        ├── curriculum_history.json
        ├── checkpoints/
        │   ├── step_250000.zip
        │   └── ...
        ├── tb_logs/
        │   └── PPO_1/events.out.tfevents...
        └── eval/
            ├── eval_results.json
            ├── trajectories.npz
            └── figures/
                └── trajectory_overlay_simple_oval.png
```

## Configuration

All parameters are in `configs/default.yaml`. Override with:
- Stage-specific YAML files (merged on top of defaults)
- CLI arguments (highest priority)

Key config sections: `env`, `reward`, `expert`, `bc`, `ppo`, `curriculum`, `wandb`.
