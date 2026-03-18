"""Expert data generation by tracking optimal trajectories through gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from environments.racing_env import DroneRacingEnv
from environments.tracks.track_loader import load_track
from planning.trajectory_generator import generate_racing_trajectory
from learning.utils.model_io import save_expert_data
from utils.config import load_config, generate_run_id, save_config
from utils.math_utils import gate_forward_vector


class ExpertController:
    """PD controller that tracks a reference trajectory.

    Outputs actions in [-1, 1] matching the DroneRacingEnv action space:
    [thrust, roll, pitch, yaw_rate].
    """

    def __init__(self, kp_pos: float = 3.0, kd_pos: float = 1.5, kp_yaw: float = 2.0):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_yaw = kp_yaw

    def compute_action(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        current_yaw: float,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
    ) -> np.ndarray:
        """Compute control action to track a reference position/velocity."""
        pos_error = target_pos - current_pos
        vel_error = target_vel - current_vel

        desired_accel = self.kp_pos * pos_error + self.kd_pos * vel_error

        thrust = np.clip(desired_accel[2] / 5.0, -1, 1)
        pitch = np.clip(desired_accel[0] / 5.0, -1, 1)
        roll = np.clip(-desired_accel[1] / 5.0, -1, 1)

        target_yaw = np.arctan2(target_vel[1], target_vel[0]) if np.linalg.norm(target_vel[:2]) > 0.1 else current_yaw
        yaw_error = target_yaw - current_yaw
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        yaw_cmd = np.clip(self.kp_yaw * yaw_error / np.pi, -1, 1)

        return np.array([thrust, roll, pitch, yaw_cmd], dtype=np.float32)


def generate_expert_data(
    track_name: str = "simple_oval",
    n_episodes: int = 500,
    avg_speed: float = 2.0,
    perturbation: float = 0.3,
    output_dir: str = "outputs/expert_data",
    seed: int = 42,
) -> Path:
    """Generate expert demonstration data by tracking optimal trajectories.

    Parameters
    ----------
    track_name : name of the track to generate data for
    n_episodes : number of demonstration episodes
    avg_speed : target speed for trajectory generation
    perturbation : random perturbation of start position (meters)
    output_dir : base output directory
    seed : random seed

    Returns
    -------
    Path to saved expert_data.npz
    """
    rng = np.random.default_rng(seed)
    track = load_track(track_name)
    controller = ExpertController()

    gate_positions = np.array([g.position for g in track.gates])
    gate_fwd = np.array([gate_forward_vector(g.orientation) for g in track.gates])

    all_obs = []
    all_actions = []
    all_episode_ids = []
    all_gate_indices = []
    all_timestamps = []

    env = DroneRacingEnv(
        track_name=track_name,
        max_episode_steps=2000,
        initial_perturbation=0.0,
    )

    successful = 0
    for ep in range(n_episodes):
        start_perturb = rng.uniform(-perturbation, perturbation, size=3)
        start_perturb[2] = 0  # don't perturb altitude
        start_pos = track.start_position + start_perturb

        positions, velocities, timestamps = generate_racing_trajectory(
            gate_positions=gate_positions,
            start_position=start_pos,
            avg_speed=avg_speed,
            dt=1.0 / 48,
            gate_forward_vectors=gate_fwd,
        )

        obs, info = env.reset(seed=seed + ep)
        env._sim.reset(start_pos.copy(), track.start_orientation.copy())
        env.racing_state.prev_pos = start_pos.copy()
        obs = env._get_obs()

        episode_obs = []
        episode_actions = []
        episode_gate_idx = []
        episode_ts = []

        for t_idx in range(min(len(positions) - 1, env.max_episode_steps)):
            target_pos = positions[t_idx + 1]
            target_vel = velocities[t_idx + 1] if t_idx + 1 < len(velocities) else np.zeros(3)

            current_pos = env._sim.state.position
            current_vel = env._sim.state.velocity
            current_yaw = env._sim.state.orientation[2]

            action = controller.compute_action(
                current_pos, current_vel, current_yaw, target_pos, target_vel
            )

            episode_obs.append(obs.copy())
            episode_actions.append(action.copy())
            episode_gate_idx.append(env.racing_state.next_gate_idx)
            episode_ts.append(timestamps[t_idx])

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                break

        if len(episode_obs) > 10:
            all_obs.extend(episode_obs)
            all_actions.extend(episode_actions)
            all_episode_ids.extend([ep] * len(episode_obs))
            all_gate_indices.extend(episode_gate_idx)
            all_timestamps.extend(episode_ts)
            if info.get("gates_passed", 0) > 0:
                successful += 1

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}, samples: {len(all_obs)}, "
                  f"successful: {successful}/{ep + 1}")

    env.close()

    run_id = generate_run_id("expert", track_name)
    save_dir = Path(output_dir) / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    data_path = save_dir / "expert_data.npz"
    save_expert_data(
        observations=np.array(all_obs),
        actions=np.array(all_actions),
        episode_ids=np.array(all_episode_ids),
        gate_indices=np.array(all_gate_indices),
        timestamps=np.array(all_timestamps),
        path=data_path,
        metadata={
            "track_name": track_name,
            "n_episodes": n_episodes,
            "n_samples": len(all_obs),
            "successful_episodes": successful,
            "avg_speed": avg_speed,
            "perturbation": perturbation,
            "seed": seed,
        },
    )

    print(f"\nExpert data saved to {data_path}")
    print(f"  Total samples: {len(all_obs)}")
    print(f"  Successful episodes: {successful}/{n_episodes}")

    return data_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate expert demonstration data")
    parser.add_argument("--track", default="simple_oval")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--perturbation", type=float, default=0.3)
    parser.add_argument("--output-dir", default="outputs/expert_data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_expert_data(
        track_name=args.track,
        n_episodes=args.episodes,
        avg_speed=args.speed,
        perturbation=args.perturbation,
        output_dir=args.output_dir,
        seed=args.seed,
    )
