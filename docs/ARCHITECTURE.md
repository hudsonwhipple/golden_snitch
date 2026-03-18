# Golden Snitch - System Architecture

## Overview

Golden Snitch is an autonomous drone racing system that uses hierarchical reinforcement learning to achieve minimum lap times and maximum gate completion rates. The system follows a **hybrid modular + learned** architecture where classical methods handle well-understood problems (control, state estimation) and learning is applied where it provides clear advantages (perception, racing strategy).

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                         │
│                                                                   │
│  ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌────────────┐  │
│  │  Expert   │──>│  Behavior │──>│   PPO    │──>│ Curriculum  │  │
│  │Trajectory │   │  Cloning  │   │Fine-Tune │   │  Learning   │  │
│  └──────────┘   └───────────┘   └──────────┘   └────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Runtime Architecture                         │
│                                                                   │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌────────────┐  │
│  │Telemetry │──>│  State   │──>│ High-Level│──>│  Low-Level │  │
│  │  (IMU)   │   │  Vector  │   │  Policy   │   │    PID     │  │
│  └──────────┘   │  (24-dim)│   │ (learned) │   │ Controller │  │
│                  └──────────┘   └───────────┘   └────────────┘  │
│                       ▲                               │          │
│  ┌──────────┐        │                               ▼          │
│  │   RGB    │──> [Future:                      ┌────────────┐  │
│  │  Camera  │    Perception]                   │  Actuator  │  │
│  └──────────┘                                  │  Commands  │  │
│                                                 └────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Environments

**`environments/racing_env.py`** - Primary training environment

- Lightweight quadrotor dynamics simulation (pure numpy, no pybullet dependency)
- 240 Hz physics, 48 Hz policy (5x decimation)
- Gymnasium-compatible (passes SB3 `check_env`)
- Flat 24-dim observation space, 4-dim action space

**`environments/quadrotor_dynamics.py`** - Physics simulation

- Rigid-body quadrotor model (Mellinger & Kumar, 2011)
- Crazyflie 2.x physical parameters
- Internal attitude PD controller for stability
- Configurable mass, drag, inertia parameters

**`environments/pybullet_racing_env.py`** - High-fidelity environment (optional)

- BaseAviary subclass from gym-pybullet-drones
- DSLPIDControl for velocity-to-RPM conversion
- Requires pybullet (Python 3.12 via conda)
- Used for validation when higher fidelity is needed

### 2. Observation Space (24 dimensions)

| Index  | Content                          | Frame       |
|--------|----------------------------------|-------------|
| 0:3    | Position (x, y, z)              | World       |
| 3:6    | Velocity (vx, vy, vz)           | Body        |
| 6:9    | Orientation (roll, pitch, yaw)  | Euler (rad) |
| 9:12   | Angular velocity (wx, wy, wz)   | Body        |
| 12:16  | Previous action                  | Normalized  |
| 16:19  | Relative pos to next gate        | Gate local  |
| 19:22  | Relative pos to lookahead gate   | Gate local  |
| 22     | Distance to next gate (norm)     | Scalar      |
| 23     | Gate progress fraction           | Scalar      |

### 3. Action Space (4 dimensions)

All actions in [-1, 1]:

| Index | Content       | Mapped to                    |
|-------|---------------|------------------------------|
| 0     | Thrust        | Hover thrust * (1 + action)  |
| 1     | Roll command  | Target roll angle            |
| 2     | Pitch command | Target pitch angle           |
| 3     | Yaw rate      | Target yaw rate              |

### 4. Reward Function

Multi-component reward (see `environments/reward_functions/composite_reward.py`):

- **Gate passage** (+10.0): Sparse bonus for passing through next gate in sequence
- **Progress** (weight 1.0): Dense signal for approaching the next gate
- **Velocity** (weight 0.3): Reward for speed directed toward the gate
- **Smoothness** (weight 0.05): Penalty for jerky control inputs
- **Collision** (-50.0): Terminal penalty for crashing
- **Time** (-0.01): Small per-step penalty to incentivize speed

### 5. Gate Detection

Gate passage uses oriented plane intersection (`environments/gate_detection.py`):

1. Check if drone crossed gate's plane between timesteps (front-to-back only)
2. Interpolate exact crossing point
3. Verify crossing point is within gate bounds (width x height + margin)
4. Sequential enforcement: only the next gate in order can be passed

### 6. Training Pipeline

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions.

**Stage 1: Expert Data** - Min-jerk trajectory generation + PD tracking controller

**Stage 2: Behavior Cloning** - Supervised learning on expert (obs, action) pairs

**Stage 3: PPO** - Reinforcement learning initialized from BC weights

**Stage 4: Curriculum** - Progressive difficulty (gates, speed, noise)

### 7. Curriculum System

Levels are defined in `configs/default.yaml`:

| Level | Gates | Max Speed | Noise |
|-------|-------|-----------|-------|
| 0     | 1     | 2.0 m/s   | No    |
| 1     | 3     | 3.5 m/s   | No    |
| 2     | All   | 5.0 m/s   | No    |
| 3     | All   | 5.0 m/s   | Yes   |

Promotion: gate success > 85% over 100 episodes. Demotion: gate success < 50%.

### 8. Track System

Tracks are defined as YAML files in `configs/tracks/`. Each track specifies:

- Start position and orientation
- Gate sequence: position, orientation (euler), size (width, height)

Available tracks: `simple_oval` (4 gates), `figure_eight` (6 gates), `competition` (8 gates).

## Key Design Decisions

1. **Lightweight sim over PyBullet**: Pure numpy dynamics enable training on any platform (including Python 3.14) with faster iteration than full physics engines.

2. **Hierarchical control**: The policy outputs high-level commands (thrust/angle), not motor RPMs. An internal PD controller handles attitude stability, dramatically reducing sample complexity.

3. **Gate-relative observations**: Providing gate position in the gate's local frame gives the policy a coordinate-invariant representation that generalizes across tracks.

4. **Multi-stage training**: BC warm-start avoids the RL exploration problem (random policies crash immediately), curriculum prevents catastrophic forgetting.

## Future Architecture (Phases D-E)

- **Perception Module**: MobileNetV3-Small gate detector with keypoint regression
- **State Estimation**: Extended Kalman Filter fusing PnP estimates with telemetry
- **Domain Randomization**: Visual + physics randomization for sim-to-real transfer
- **AirSim Integration**: High-fidelity RGB rendering for vision training
