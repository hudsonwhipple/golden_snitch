# Golden Snitch - Autonomous Drone Racing

A competition-grade autonomous drone racing system using hierarchical reinforcement learning. Designed for both simulated competition and real-world deployment.

## System Overview

Golden Snitch uses a hybrid modular architecture:

- **Learned high-level policy** (PPO) selects racing commands
- **Classical low-level controller** (PID) handles flight stability
- **Multi-stage training**: Expert demonstrations → Behavior cloning → RL fine-tuning → Curriculum scaling

The system achieves its performance through a carefully designed training pipeline that progresses from imitation to optimization, with curriculum learning to handle increasing difficulty.

## Quick Start

### Setup

```bash
git clone <repo-url> && cd golden_snitch
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

Or run stages individually:

```bash
# 1. Generate expert demonstrations
python -m planning.expert_policy --track simple_oval --episodes 500

# 2. Train behavior cloning policy
python -m learning.training.train_bc --data outputs/expert_data/<run_id>/expert_data.npz

# 3. Fine-tune with PPO
python -m learning.training.train_ppo --track simple_oval --timesteps 5000000

# 4. Evaluate
python -m learning.evaluation.evaluate --model outputs/ppo/<run_id>/best_model.zip --track simple_oval
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

- [Architecture](docs/ARCHITECTURE.md) - System design and component breakdown
- [Training Guide](docs/TRAINING_GUIDE.md) - Detailed training instructions

## License

MIT License - see [LICENSE](LICENSE).
