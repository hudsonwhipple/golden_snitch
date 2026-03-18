# Golden Snitch - Task Tracker

## Phase A: Core Infrastructure
- [x] pyproject.toml and config system
- [x] Math utilities (quaternion, rotation, coordinate transforms)
- [x] Track system (YAML definitions + loader)
- [x] PID controller (cascaded, anti-windup, rate limiting)
- [x] Gate passage detection (oriented plane intersection)
- [x] Lightweight quadrotor dynamics simulation
- [x] Racing environment (Gymnasium, 24-dim obs, 4-dim action, passes check_env)
- [x] Observation wrapper (24-dim state vector)
- [x] Composite reward function (gate, progress, velocity, smoothness, collision, time)
- [x] Artifact management (model_io, tracking)

## Phase B: Expert + Behavior Cloning
- [x] Trajectory generator (min-jerk splines)
- [x] Expert data generation pipeline (PD controller tracking trajectories)
- [x] Behavior cloning trainer (MLP, MSE loss, cosine LR, checkpointing)
- [x] Expert data schema (observations, actions, episode_ids, gate_indices, timestamps)

## Phase C: PPO + Evaluation
- [x] PPO training with DummyVecEnv (checkpoints, eval callbacks, TensorBoard)
- [x] Curriculum manager (promotion/demotion, level definitions, state serialization)
- [x] Curriculum wrapper (gate count, noise)
- [x] Evaluation harness + metrics computation
- [x] Report generation (metrics table, trajectory overlay plots)
- [x] Model comparison tool

## Phase D: Perception
- [x] Gate detector (MobileNetV3-Small + keypoint regression head)
- [x] Feature extractor (64-dim visual features for policy)
- [x] PnP pose estimator (solvePnP from keypoints)
- [x] Vision policy (CNN + MLP combined, 88-dim obs)
- [x] Vision encoder training script (gate detector + feature encoder)

## Phase E: Domain Transfer
- [x] Domain randomization wrappers (physics, sensor noise, visual)
- [x] AirSim racing environment (full RGB rendering)
- [x] Extended Kalman Filter (fuse PnP + IMU)
- [x] Gate tracker (confidence-based recovery behavior)

## Documentation
- [x] ARCHITECTURE.md
- [x] TRAINING_GUIDE.md
- [x] README.md (complete rewrite)
