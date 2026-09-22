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
from utils.math_utils import gate_forward_vector, normalize_angle

# How far past the reference trajectory's own duration an episode may run.
REFERENCE_TIME_MARGIN = 1.25

# Std of the exploration noise added to executed actions during data collection, so the
# expert is seen recovering from states a cloned policy will actually reach.
DEFAULT_ACTION_NOISE = 0.1


class ExpertController:
    """Trajectory-tracking controller that outputs DroneRacingEnv actions.

    The outer loop is a PD on position with reference-acceleration feedforward and
    linear-drag compensation. The resulting world-frame acceleration is rotated into
    the drone's yaw frame before being inverted into roll/pitch commands: tilt acts in
    the body frame, and the yaw law below turns the drone to face its velocity, so
    skipping that rotation mis-aims the thrust by the whole heading angle. On a closed
    track the heading sweeps 360 degrees, which is why an unrotated controller clears
    only the first (straight-ahead) gate.

    Outputs actions in [-1, 1] matching the DroneRacingEnv action space:
    [thrust, roll, pitch, yaw_rate].
    """

    def __init__(
        self,
        kp_pos: float = 8.0,
        kd_pos: float = 6.0,
        kp_yaw: float = 2.0,
        gravity: float = 9.81,
        drag_coeff: float = 0.5,
        max_angle: float = np.pi / 6,
    ):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_yaw = kp_yaw
        self.gravity = gravity
        self.drag_coeff = drag_coeff
        self.max_angle = max_angle

    def compute_action(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        current_yaw: float,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_accel: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute control action to track a reference position/velocity/acceleration."""
        if target_accel is None:
            target_accel = np.zeros(3)

        desired_accel = (
            target_accel
            + self.kp_pos * (target_pos - current_pos)
            + self.kd_pos * (target_vel - current_vel)
            + self.drag_coeff * current_vel  # cancel the model's linear drag
        )

        # Thrust must also hold the drone up; keep it positive so the inversion below
        # stays well conditioned.
        accel_up = max(desired_accel[2] + self.gravity, 0.5 * self.gravity)

        # Rotate the desired horizontal acceleration into the drone's yaw frame.
        cy, sy = np.cos(current_yaw), np.sin(current_yaw)
        accel_fwd = cy * desired_accel[0] + sy * desired_accel[1]
        accel_left = -sy * desired_accel[0] + cy * desired_accel[1]

        pitch = np.clip(np.arctan2(accel_fwd, accel_up), -self.max_angle, self.max_angle)
        roll = np.clip(
            -np.arctan2(accel_left * np.cos(pitch), accel_up), -self.max_angle, self.max_angle
        )

        # Tilting reduces the vertical component of thrust; compensate for it. The sim
        # takes 0 as hover, so body-frame accel T maps to the command T / g - 1.
        thrust = accel_up / max(np.cos(roll) * np.cos(pitch), 0.3)
        thrust_cmd = np.clip(thrust / self.gravity - 1.0, -1, 1)

        target_yaw = np.arctan2(target_vel[1], target_vel[0]) if np.linalg.norm(target_vel[:2]) > 0.1 else current_yaw
        yaw_cmd = np.clip(self.kp_yaw * normalize_angle(target_yaw - current_yaw) / np.pi, -1, 1)

        return np.array(
            [thrust_cmd, roll / self.max_angle, pitch / self.max_angle, yaw_cmd],
            dtype=np.float32,
        )


def generate_expert_data(
    track_name: str = "simple_oval",
    n_episodes: int = 500,
    avg_speed: float = 2.0,
    perturbation: float = 0.3,
    action_noise: float = DEFAULT_ACTION_NOISE,
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
    action_noise : std of the Gaussian noise added to the *executed* action. The
        recorded label stays the clean expert action, so the dataset gains
        (off-trajectory state -> corrective action) pairs. Without this the demos
        only ever show the on-trajectory states reachable from a 0.3 m start
        offset, a cloned policy's small per-step errors compound with nothing in
        the data telling it how to recover, and it leaves the demonstrated
        distribution entirely. See `tasks/lessons.md`.
    output_dir : base output directory
    seed : random seed

    Returns
    -------
    Path to saved expert_data.npz
    """
    rng = np.random.default_rng(seed)
    track = load_track(track_name)

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

    # Take the plant constants the controller compensates for from the sim itself, so
    # the feedforward stays correct if the quadrotor parameters change.
    controller = ExpertController(
        gravity=env._sim.params.gravity,
        drag_coeff=env._sim.params.drag_coeff,
        max_angle=env._sim.max_angle,
    )

    successful = 0
    for ep in range(n_episodes):
        start_perturb = rng.uniform(-perturbation, perturbation, size=3)
        start_perturb[2] = 0  # don't perturb altitude
        start_pos = track.start_position + start_perturb

        positions, velocities, accelerations, timestamps = generate_racing_trajectory(
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

        # Allow a margin past the reference duration so a slightly lagging run can still
        # finish the lap; the reference is held at its final sample once it runs out.
        last = len(positions) - 1
        n_steps = min(int(last * REFERENCE_TIME_MARGIN), env.max_episode_steps)

        for t_idx in range(n_steps):
            k = min(t_idx + 1, last)
            target_pos = positions[k]
            target_vel = velocities[k]
            target_accel = accelerations[k]

            current_pos = env._sim.state.position
            current_vel = env._sim.state.velocity
            current_yaw = env._sim.state.orientation[2]

            action = controller.compute_action(
                current_pos, current_vel, current_yaw, target_pos, target_vel, target_accel
            )

            # Label the visited state with the CLEAN expert action, then execute a noisy
            # one. The next state is therefore off-trajectory, and the label recorded
            # there is the expert's correction for it.
            episode_obs.append(obs.copy())
            episode_actions.append(action.copy())
            episode_gate_idx.append(env.racing_state.next_gate_idx)
            episode_ts.append(timestamps[min(t_idx, last)])

            executed = action
            if action_noise > 0:
                executed = np.clip(action + rng.normal(0, action_noise, size=action.shape), -1, 1)

            obs, reward, terminated, truncated, info = env.step(executed)

            # truncated fires once every gate is passed; stepping on past that wraps
            # next_gate_idx modulo the gate count and poisons the observations.
            if terminated or truncated:
                break

        # Only demonstrate complete laps -- cloning partial runs teaches the policy to
        # stop at whichever gate the controller happened to reach.
        completed = info.get("gates_passed", 0) >= len(track.gates)
        if completed and len(episode_obs) > 10:
            all_obs.extend(episode_obs)
            all_actions.extend(episode_actions)
            all_episode_ids.extend([ep] * len(episode_obs))
            all_gate_indices.extend(episode_gate_idx)
            all_timestamps.extend(episode_ts)
            successful += 1

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}, samples: {len(all_obs)}, "
                  f"complete laps: {successful}/{ep + 1}")

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
            # Episodes that passed EVERY gate. Only these are kept in the dataset.
            "completed_lap_episodes": successful,
            "avg_speed": avg_speed,
            "perturbation": perturbation,
            "seed": seed,
        },
    )

    print(f"\nExpert data saved to {data_path}")
    print(f"  Total samples: {len(all_obs)}")
    print(f"  Complete laps kept: {successful}/{n_episodes}")

    return data_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate expert demonstration data")
    parser.add_argument("--track", default="simple_oval")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--perturbation", type=float, default=0.3)
    parser.add_argument("--action-noise", type=float, default=DEFAULT_ACTION_NOISE,
                        help="std of exploration noise on executed actions (labels stay clean)")
    parser.add_argument("--output-dir", default="outputs/expert_data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_expert_data(
        track_name=args.track,
        n_episodes=args.episodes,
        avg_speed=args.speed,
        perturbation=args.perturbation,
        action_noise=args.action_noise,
        output_dir=args.output_dir,
        seed=args.seed,
    )
