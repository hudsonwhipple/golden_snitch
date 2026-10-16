# Usage

Full setup, training and evaluation reference. For the project overview and results, see the [README](../README.md).

## Quick Start

### Setup

```bash
git clone https://github.com/hudsonwhipple/golden_snitch.git && cd golden_snitch
git submodule update --init --recursive
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Verify installation:

```bash
python -m pytest tests/ -v
```

### Train a Policy

Run the full multi-stage pipeline:

```bash
bash scripts/train.sh
```

Override track, device, or timesteps:

```bash
TRACK=figure_eight TIMESTEPS=5000000 DEVICE=mps bash scripts/train.sh
```

#### Train Script Flags

| Flag | Description |
|------|-------------|
| `--model <name>` | Resume PPO from `saved_models/<name>.zip` (implies `--skip-expert --skip-bc`) |
| `--model-name <name>` | Name to save the resulting model as (default: `<track>_<timestamp>`) |
| `--skip-expert` | Skip Stage 1, reuse latest existing expert data |
| `--skip-bc` | Skip Stage 2, reuse latest existing BC model |
| `--no-domain-rand` | Disable physics and sensor domain randomization during PPO training |
| `--track <name>` | Track name (default: `simple_oval`) |
| `--seed <int>` | Random seed (default: `42`) |
| `--device <str>` | Torch device: `cpu`, `mps`, or `cuda` (default: `cpu`) |

#### Resume Training

Each completed run saves its best model to `saved_models/`. To continue training from a previous run:

```bash
# Resume from a saved model (skips expert + BC automatically)
bash scripts/train.sh --model simple_oval_20260320_162152

# Resume with a custom name for the new checkpoint
bash scripts/train.sh --model simple_oval_20260320_162152 --model-name simple_oval_v2

# List available saved models
ls saved_models/*.zip
```

#### Skip Upstream Stages

If expert data and a BC model already exist, skip the expensive first two stages:

```bash
bash scripts/train.sh --skip-expert --skip-bc
```

#### Run Stages Individually

```bash
# 1. Generate expert demonstrations
python -m planning.expert_policy --track simple_oval --episodes 500

# 2. Train behavior cloning policy
python -m learning.training.train_bc --data outputs/expert_data/<run_id>/expert_data.npz

# 3. Fine-tune with PPO
python -m learning.training.train_ppo \
    --track simple_oval \
    --timesteps 5000000 \
    --bc-model outputs/bc/<run_id>/best_model.pt \
    --resume saved_models/<name>.zip \          # optional: resume from checkpoint
    --curriculum-state saved_models/<name>_curriculum.json \  # optional: restore curriculum
    --no-domain-rand                            # optional: disable domain randomization

# 4. Evaluate
python -m learning.evaluation.report eval \
    --model outputs/ppo/<run_id>/best_model.zip \
    --track simple_oval \
    --episodes 100 \
    --output-dir outputs/ppo/<run_id>/eval
```

### Evaluate a Policy

```bash
python -m learning.evaluation.report eval \
    --model outputs/ppo/<run_id>/best_model.zip \
    --track simple_oval \
    --episodes 100 \
    --output-dir outputs/ppo/<run_id>/eval
```

## Project Structure

```
golden_snitch/
├── configs/                    # Configuration files
│   ├── default.yaml           # Master config (all hyperparameters)
│   └── tracks/                # Race track definitions
│       ├── simple_oval.yaml   # 4-gate beginner track
│       ├── figure_eight.yaml  # 6-gate intermediate track
│       └── competition.yaml   # 8-gate advanced track
├── control/                   # Flight controllers
│   └── pid_controller.py      # Cascaded PID with anti-windup
├── environments/              # Gymnasium environments
│   ├── racing_env.py          # Primary training env (lightweight dynamics)
│   ├── quadrotor_dynamics.py  # Pure numpy quadrotor simulation
│   ├── pybullet_racing_env.py # High-fidelity env (requires pybullet)
│   ├── gate_detection.py      # Gate passage detection
│   ├── reward_functions/      # Composite reward system
│   ├── tracks/                # Track loading utilities
│   └── wrappers/              # Observation and curriculum wrappers
├── learning/                  # ML training infrastructure
│   ├── policies/              # Neural network architectures
│   │   └── state_policy.py    # MLP racing policy (256x256)
│   ├── training/              # Training scripts
│   │   ├── train_bc.py        # Behavior cloning trainer
│   │   ├── train_ppo.py       # PPO training with checkpoints
│   │   └── curriculum.py      # Curriculum level manager
│   ├── evaluation/            # Eval and reporting
│   │   ├── evaluate.py        # Evaluation harness
│   │   ├── metrics.py         # Racing metrics computation
│   │   └── report.py          # Report generation + comparison
│   └── utils/                 # Model I/O helpers
├── planning/                  # Classical planning
│   ├── trajectory_generator.py # Min-jerk spline trajectories
│   └── expert_policy.py       # Expert data generation
├── utils/                     # Shared utilities
│   ├── config.py              # YAML config loader
│   ├── math_utils.py          # Quaternion/rotation helpers
│   ├── tracking.py            # W&B + TensorBoard helpers
│   └── visualization.py       # Matplotlib plot utilities
├── tests/                     # Unit tests
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md        # System architecture details
│   └── TRAINING_GUIDE.md      # Step-by-step training guide
├── scripts/                   # Shell scripts
│   ├── train.sh               # Full pipeline orchestration
│   └── replay_trajectory.py   # Trajectory visualization
├── saved_models/              # Named model checkpoints for resuming (gitignored)
└── outputs/                   # Training artifacts (gitignored)
```

## Training Pipeline

```
Expert Trajectories → Behavior Cloning → PPO Fine-Tuning → Curriculum Learning
     (Stage 1)           (Stage 2)          (Stage 3)          (Stage 4)
```

**Stage 1** generates optimal trajectories through gates using minimum-jerk splines tracked by a PD controller.

**Stage 2** trains a neural network to imitate the expert via supervised learning (MSE loss on actions).

**Stage 3** initializes PPO from BC weights and optimizes the policy via reinforcement learning with a composite reward function.

**Stage 4** progressively increases difficulty: single gate → full track → added noise.

## Tracks

| Track | Gates | Description |
|-------|-------|-------------|
| simple_oval | 4 | Oval layout, constant altitude |
| figure_eight | 6 | Crossing paths, altitude changes |
| competition | 8 | Complex layout with tight turns |

## Configuration

All hyperparameters live in `configs/default.yaml`. Key sections:

- `env`: Track, physics/policy rates, parallel envs
- `reward`: Gate bonus, collision penalty, component weights
- `ppo`: Learning rate, batch size, clip range, entropy coefficient
- `curriculum`: Level definitions, promotion/demotion thresholds

## Compute Requirements

Designed for laptop-class compute (Apple Silicon M5pro):

| Stage | Duration | Memory |
|-------|----------|--------|
| Expert data (500 episodes) | ~30 min | 2 GB |
| Behavior cloning (50 epochs) | ~10 min | 1 GB |
| PPO (5M steps, 4 envs) | ~2-3 hrs | 4 GB |

## Monitoring

Training metrics are logged to TensorBoard:

```bash
tensorboard --logdir outputs/ppo/<run_id>/tb_logs
```

Optional W&B support:

```bash
pip install wandb && wandb login
```

## Tests

```bash
python -m pytest tests/ -v
```

Tests cover: gate detection, reward function, PID controller, track loading.

## Documentation

- [Architecture](ARCHITECTURE.md) - System design and component breakdown
- [Training Guide](TRAINING_GUIDE.md) - Detailed training instructions

## License

MIT License - see [LICENSE](../LICENSE).
