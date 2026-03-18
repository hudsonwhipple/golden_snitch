"""Evaluation harness for drone racing policies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from environments.racing_env import DroneRacingEnv
from learning.evaluation.metrics import (
    EpisodeMetrics,
    AggregateMetrics,
    aggregate_metrics,
)


def evaluate_policy(
    model_path: str,
    track_name: str = "simple_oval",
    n_episodes: int = 100,
    max_episode_steps: int = 1000,
    seed: int = 2000,
    deterministic: bool = True,
    record_trajectories: bool = True,
) -> tuple[AggregateMetrics, list[EpisodeMetrics]]:
    """Evaluate a trained policy on a racing track.

    Parameters
    ----------
    model_path : path to saved SB3 PPO model (.zip)
    track_name : track to evaluate on
    n_episodes : number of evaluation episodes
    max_episode_steps : max steps per episode
    seed : base random seed
    deterministic : use deterministic actions
    record_trajectories : whether to record full position/action histories

    Returns
    -------
    (aggregate_metrics, episode_metrics_list)
    """
    model = PPO.load(model_path)
    env = DroneRacingEnv(
        track_name=track_name,
        max_episode_steps=max_episode_steps,
    )

    episodes = []
    dt = 1.0 / 48  # policy timestep

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_metrics = EpisodeMetrics(total_gates=len(env.track.gates))

        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            ep_metrics.total_reward += reward

            if record_trajectories:
                ep_metrics.positions.append(info.get("position", np.zeros(3)).copy())
                ep_metrics.velocities.append(info.get("velocity", np.zeros(3)).copy())
                ep_metrics.actions.append(action.copy())

        ep_metrics.gates_passed = info.get("gates_passed", 0)
        ep_metrics.collided = info.get("collided", False)
        ep_metrics.steps = step
        ep_metrics.lap_time = step * dt if ep_metrics.completed_lap else float("inf")

        episodes.append(ep_metrics)

    env.close()
    agg = aggregate_metrics(episodes)
    return agg, episodes


def save_evaluation_results(
    agg: AggregateMetrics,
    episodes: list[EpisodeMetrics],
    save_dir: str | Path,
    track_name: str = "",
) -> None:
    """Save evaluation results to files."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "eval_results.json", "w") as f:
        results = agg.to_dict()
        results["track_name"] = track_name
        json.dump(results, f, indent=2, default=str)

    if any(len(ep.positions) > 0 for ep in episodes):
        max_steps = max(len(ep.positions) for ep in episodes)
        n_eps = len(episodes)

        positions = np.full((n_eps, max_steps, 3), np.nan, dtype=np.float32)
        velocities = np.full((n_eps, max_steps, 3), np.nan, dtype=np.float32)
        actions = np.full((n_eps, max_steps, 4), np.nan, dtype=np.float32)

        for i, ep in enumerate(episodes):
            if ep.positions:
                t = len(ep.positions)
                positions[i, :t] = np.array(ep.positions)
            if ep.velocities:
                t = len(ep.velocities)
                velocities[i, :t] = np.array(ep.velocities)
            if ep.actions:
                t = len(ep.actions)
                actions[i, :t] = np.array(ep.actions)

        outcomes = [
            "completed" if ep.completed_lap else ("crashed" if ep.collided else "timeout")
            for ep in episodes
        ]

        np.savez_compressed(
            save_dir / "trajectories.npz",
            positions=positions,
            velocities=velocities,
            actions=actions,
            outcomes=np.array(outcomes),
            gates_passed=np.array([ep.gates_passed for ep in episodes]),
        )


def print_metrics_table(agg: AggregateMetrics, track_name: str = "") -> None:
    """Pretty-print evaluation metrics."""
    header = f"Evaluation Results{f' - {track_name}' if track_name else ''}"
    print(f"\n{'=' * 60}")
    print(f"  {header}")
    print(f"{'=' * 60}")
    print(f"  Episodes:           {agg.n_episodes}")
    print(f"  Gate Success Rate:  {agg.gate_success_rate_mean:.1%} +/- {agg.gate_success_rate_std:.1%}")
    print(f"  Avg Gates Passed:   {agg.avg_gates_passed:.1f}")
    print(f"  Crash Rate:         {agg.crash_rate:.1%}")
    print(f"  Completed Laps:     {agg.n_completed}")
    if agg.lap_time_mean < float("inf"):
        print(f"  Lap Time:           {agg.lap_time_mean:.2f}s +/- {agg.lap_time_std:.2f}s")
    print(f"  Control Smoothness: {agg.control_smoothness_mean:.4f} +/- {agg.control_smoothness_std:.4f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained drone racing policy")
    parser.add_argument("--model", required=True, help="Path to saved model (.zip)")
    parser.add_argument("--track", default="simple_oval")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args()

    agg, episodes = evaluate_policy(
        model_path=args.model,
        track_name=args.track,
        n_episodes=args.episodes,
        seed=args.seed,
    )
    print_metrics_table(agg, args.track)

    if args.output_dir:
        save_evaluation_results(agg, episodes, args.output_dir, args.track)
        print(f"Results saved to {args.output_dir}")
