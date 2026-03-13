# Drone Racing AI

Research stack for autonomous drone racing.

**Simulators:** gym-pybullet-drones (PyBullet, default) · AirSim (optional, photorealistic)

---

## How to use this repo

**1. One-time setup**

```bash
cd golden_snitch
bash scripts/install.sh
```

**2. Every time you open a terminal for this project**

```bash
cd golden_snitch
source .venv/bin/activate
```

**3. What you can do**

| Goal | Command |
|------|---------|
| **Train a policy** (PPO on the in-repo racing env) | `bash scripts/train.sh` |
| **Check the env loads** | `bash scripts/launch_pybullet.sh` |
| **Run tests** | `pytest` |
| **See the PyBullet visual simulator** | `python scripts/run_pybullet_gui.py` (requires PyBullet installed) |
| **Use AirSim** (optional) | Install client (see below), start the AirSim binary, then run your script that uses `AirSimRacingEnv` |

Training uses the in-repo `PyBulletRacingEnv` by default. Right now that env is a **stub** (no real physics): it returns zero observations and zero rewards so the policy doesn’t learn anything useful. To get real behavior you can:

- Swap in a gym-pybullet-drones env (if PyBullet installed), or  
- Implement real physics/observations/rewards in `environments/pybullet_racing_env.py`, or  
- Use `AirSimRacingEnv` once the AirSim simulator and client are set up.

**4. Where to extend**

- **Envs:** `environments/` — e.g. `pybullet_racing_env.py`, `airsim_racing_env.py`
- **Training:** `learning/rl_algorithms/train_ppo.py`, `training/configs/ppo_racing.yaml`
- **Rewards:** `environments/reward_functions/`

---

## Setup (one-time)

### 1. Clone and install

```bash
git clone --recurse-submodules <repo>
cd golden_snitch
bash scripts/install.sh
```

This creates a `.venv`, installs core deps (torch, stable-baselines3, gymnasium, etc.), and on supported platforms installs PyBullet + the gym-pybullet-drones submodule. On Apple Silicon, PyBullet often has no pip wheel; the script will finish with core only and print optional conda instructions for the full sim.

### 2. Optional: AirSim Python client

To use AirSim environments and scripts:

```bash
source .venv/bin/activate
pip install --no-build-isolation -r requirements-airsim.txt
```

You still need the AirSim binary/simulator for your OS (see [AirSim](https://github.com/microsoft/AirSim)).

### 3. Apple Silicon (M1/M2/M3/M5)

If `install.sh` could not install PyBullet, you can either:

- Use the in-repo **stub** `PyBulletRacingEnv` (no real physics) with the venv, or  
- Use a conda env for the full gym-pybullet-drones stack (see the message printed by `install.sh`).

---

## Running things (every time)

Activate the project venv first so `python` and scripts see the right packages:

```bash
cd golden_snitch
source .venv/bin/activate
```

Then use the sections below.

### PyBullet

**How PyBullet works:** PyBullet is a physics engine with a Python API. It can run in two ways:

- **GUI mode** — `pybullet.connect(pybullet.GUI)` opens a 3D window (OpenGL). You see the scene, run the sim step-by-step, and can interact.
- **Direct (headless) mode** — no window; useful for training or servers.

**In this repo:** The in-repo `PyBulletRacingEnv` is a **stub** (no PyBullet, no GUI). To actually *see* a visual simulator you need either the script below or the gym-pybullet-drones examples.

**Quick check (no window):** Confirms the stub env loads.

```bash
bash scripts/launch_pybullet.sh
```

**Visual simulator (real PyBullet window):** Opens a 3D window with a ground plane and a cube. PyBullet must be installed in the Python you use:

```bash
python scripts/run_pybullet_gui.py
```

**Apple Silicon + conda:** If you installed PyBullet with `conda install -c conda-forge pybullet`, it lives in conda’s environment, not in the project `.venv`. So with the venv activated, `python` won’t see it. Use conda’s Python to run the GUI script:

```bash
conda run -n base python scripts/run_pybullet_gui.py
```

Or `deactivate` the venv, then `python scripts/run_pybullet_gui.py` (uses conda base). For **drones** in the GUI (quadcopters, tasks), use the gym-pybullet-drones submodule — run their examples from `simulators/pybullet-drones` with an environment where PyBullet is available (conda base or a venv with pybullet).

### AirSim

1. Start the AirSim simulator (binary / Unreal build) and load your environment.
2. With the venv activated, run your script or client, e.g.:

```bash
python -c "
from environments.airsim_racing_env import AirSimRacingEnv
env = AirSimRacingEnv()
# ...
"
```

To build the AirSim plugin from the submodule:

```bash
bash scripts/launch_airsim.sh
```

### Training

With the venv activated:

```bash
bash scripts/train.sh
```

Or run the trainer directly:

```bash
python learning/rl_algorithms/train_ppo.py
```

### Tests

With the venv activated, run the test suite:

```bash
pytest
```

To run a specific file or directory:

```bash
pytest path/to/tests/
```

---

## Summary

| What              | Command (after `source .venv/bin/activate`)     |
|-------------------|--------------------------------------------------|
| PyBullet (check)  | `bash scripts/launch_pybullet.sh`               |
| PyBullet (GUI)    | `python scripts/run_pybullet_gui.py`            |
| AirSim            | Start AirSim binary, then run your script       |
| Training          | `bash scripts/train.sh`                         |
| Tests             | `pytest`                                        |
