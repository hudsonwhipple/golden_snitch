"""Report generation: metrics tables, trajectory plots, model comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from learning.evaluation.evaluate import evaluate_policy, save_evaluation_results, print_metrics_table
from learning.evaluation.metrics import AggregateMetrics
from utils.visualization import (
    plot_trajectory_overlay,
    plot_reward_breakdown,
    plot_episode_statistics,
    plot_speed_altitude_profiles,
    plot_metrics_comparison,
)
from environments.tracks.track_loader import load_track


def generate_eval_report(
    model_path: str,
    track_name: str = "simple_oval",
    n_episodes: int = 100,
    output_dir: str | Path | None = None,
    seed: int = 2000,
    deterministic: bool = True,
    initial_perturbation: float = 0.3,
) -> AggregateMetrics:
    """Run evaluation and generate full report with plots.

    Parameters
    ----------
    model_path : path to saved model
    track_name : track to evaluate on
    n_episodes : number of episodes
    output_dir : where to save results and plots
    seed : random seed
    deterministic : use deterministic actions (False adds policy stochasticity)
    initial_perturbation : random start position offset in meters (0 = fixed start)

    Returns
    -------
    Aggregate metrics
    """
    agg, episodes = evaluate_policy(
        model_path=model_path,
        track_name=track_name,
        n_episodes=n_episodes,
        seed=seed,
        deterministic=deterministic,
        initial_perturbation=initial_perturbation,
        record_trajectories=True,
    )

    print_metrics_table(agg, track_name)

    if output_dir:
        output_dir = Path(output_dir)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        save_evaluation_results(agg, episodes, output_dir, track_name)

        track = load_track(track_name)
        gate_positions = np.array([g.position for g in track.gates])
        gate_orientations = np.array([g.orientation for g in track.gates])
        gate_sizes = np.array([g.size for g in track.gates])

        # Build trajectory data for ALL episodes
        trajectories = []
        velocities_list = []
        outcomes = []
        for ep in episodes:
            if ep.positions:
                trajectories.append(np.array(ep.positions))
                velocities_list.append(
                    np.array(ep.velocities) if ep.velocities else np.zeros((len(ep.positions), 3))
                )
                if ep.completed_lap:
                    outcomes.append("completed")
                elif ep.collided:
                    outcomes.append("crashed")
                else:
                    outcomes.append("timeout")

        stats_annotation = {
            "n_episodes": agg.n_episodes,
            "gate_success_rate": agg.gate_success_rate_mean,
            "crash_rate": agg.crash_rate,
            "avg_gates_passed": agg.avg_gates_passed,
            "mean_episode_steps": agg.mean_episode_steps,
        }

        # 1. Trajectory overlay (all episodes)
        if trajectories:
            plot_trajectory_overlay(
                trajectories=trajectories,
                gate_positions=gate_positions,
                gate_orientations=gate_orientations,
                gate_sizes=gate_sizes,
                outcomes=outcomes,
                title=f"Flight Trajectories - {track_name} ({len(trajectories)} episodes)",
                save_path=figures_dir / f"trajectory_overlay_{track_name}.png",
                stats=stats_annotation,
            )

        # 2. Episode statistics (histograms + scatter)
        episodes_data = []
        for ep in episodes:
            outcome = "completed" if ep.completed_lap else ("crashed" if ep.collided else "timeout")
            episodes_data.append({
                "gates_passed": ep.gates_passed,
                "steps": ep.steps,
                "total_reward": ep.total_reward,
                "outcome": outcome,
            })

        plot_episode_statistics(
            episodes_data=episodes_data,
            title=f"Episode Statistics - {track_name} ({n_episodes} episodes)",
            save_path=figures_dir / f"episode_statistics_{track_name}.png",
        )

        # 3. Reward breakdown
        reward_components = {
            "gate": [ep.reward_breakdown.gate for ep in episodes],
            "progress": [ep.reward_breakdown.progress for ep in episodes],
            "velocity": [ep.reward_breakdown.velocity for ep in episodes],
            "smoothness": [ep.reward_breakdown.smoothness for ep in episodes],
            "collision": [ep.reward_breakdown.collision for ep in episodes],
            "time": [ep.reward_breakdown.time for ep in episodes],
        }

        plot_reward_breakdown(
            reward_components=reward_components,
            title=f"Reward Breakdown - {track_name}",
            save_path=figures_dir / f"reward_breakdown_{track_name}.png",
        )

        # 4. Speed & altitude profiles
        if trajectories and velocities_list:
            plot_speed_altitude_profiles(
                trajectories=trajectories,
                velocities=velocities_list,
                outcomes=outcomes,
                title=f"Speed & Altitude - {track_name}",
                save_path=figures_dir / f"speed_altitude_{track_name}.png",
            )

        # 5. Write comprehensive text summary
        _write_eval_summary(agg, episodes, track_name, output_dir / "eval_summary.txt")

    return agg


def _write_eval_summary(
    agg: AggregateMetrics,
    episodes: list,
    track_name: str,
    path: Path,
) -> None:
    """Write a comprehensive text summary of the evaluation."""
    lines = []
    lines.append(f"{'=' * 68}")
    lines.append(f"  EVALUATION SUMMARY - {track_name}")
    lines.append(f"{'=' * 68}")
    lines.append("")
    lines.append(f"  Total Episodes:     {agg.n_episodes}")
    lines.append(f"  Completed:          {agg.n_completed}")
    lines.append(f"  Crashed:            {agg.n_crashed}")
    lines.append(f"  Timeout:            {agg.n_timeout}")
    lines.append("")
    lines.append(f"  Gate Success Rate:  {agg.gate_success_rate_mean:.1%} +/- {agg.gate_success_rate_std:.1%}")
    lines.append(f"  Avg Gates Passed:   {agg.avg_gates_passed:.2f}")
    lines.append(f"  Crash Rate:         {agg.crash_rate:.1%}")
    lines.append(f"  Timeout Rate:       {agg.timeout_rate:.1%}")
    lines.append("")
    if agg.lap_time_mean < float("inf"):
        lines.append(f"  Lap Time:           {agg.lap_time_mean:.2f}s +/- {agg.lap_time_std:.2f}s")
    lines.append(f"  Mean Episode Steps: {agg.mean_episode_steps:.1f}")
    lines.append(f"  Control Smoothness: {agg.control_smoothness_mean:.4f} +/- {agg.control_smoothness_std:.4f}")
    lines.append("")
    lines.append(f"  Flight Statistics:")
    lines.append(f"    Mean Speed:       {agg.mean_speed:.2f} m/s")
    lines.append(f"    Max Speed:        {agg.max_speed:.2f} m/s")
    lines.append(f"    Mean Altitude:    {agg.mean_altitude:.2f} m")
    lines.append("")
    lines.append(f"  Reward Breakdown (mean cumulative per episode):")
    if agg.reward_breakdown_mean:
        for key, val in agg.reward_breakdown_mean.items():
            lines.append(f"    R_{key:12s}: {val:+.2f}")
    lines.append("")
    lines.append(f"{'=' * 68}")
    lines.append("")
    lines.append("  Per-Episode Details:")
    lines.append(f"  {'Ep':>4s}  {'Gates':>5s}  {'Steps':>5s}  {'Reward':>8s}  {'Outcome':>9s}")
    lines.append(f"  {'─' * 40}")
    for i, ep in enumerate(episodes):
        outcome = "completed" if ep.completed_lap else ("crashed" if ep.collided else "timeout")
        lines.append(f"  {i:4d}  {ep.gates_passed:5d}  {ep.steps:5d}  {ep.total_reward:8.1f}  {outcome:>9s}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def compare_models(
    model_a_path: str,
    model_b_path: str,
    name_a: str = "Model A",
    name_b: str = "Model B",
    track_name: str = "simple_oval",
    n_episodes: int = 100,
    output_dir: str | Path | None = None,
    seed: int = 2000,
) -> dict:
    """Compare two models side-by-side.

    Returns
    -------
    Dict with metrics for both models and deltas
    """
    print(f"\nEvaluating {name_a}...")
    agg_a, _ = evaluate_policy(model_a_path, track_name, n_episodes, seed=seed)
    print_metrics_table(agg_a, f"{track_name} ({name_a})")

    print(f"\nEvaluating {name_b}...")
    agg_b, _ = evaluate_policy(model_b_path, track_name, n_episodes, seed=seed)
    print_metrics_table(agg_b, f"{track_name} ({name_b})")

    dict_a = agg_a.to_dict()
    dict_b = agg_b.to_dict()

    deltas = {}
    for key in dict_a:
        va, vb = dict_a[key], dict_b[key]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if va != float("inf") and vb != float("inf"):
                deltas[key] = vb - va

    print("\n--- Delta Analysis ---")
    for key, delta in deltas.items():
        sign = "+" if delta > 0 else ""
        print(f"  {key}: {sign}{delta:.4f}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        comparison = {"model_a": dict_a, "model_b": dict_b, "deltas": deltas}
        with open(output_dir / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2, default=str)

        plot_metrics_comparison(
            {k: v for k, v in dict_a.items() if isinstance(v, (int, float)) and v != float("inf")},
            {k: v for k, v in dict_b.items() if isinstance(v, (int, float)) and v != float("inf")},
            name_a=name_a,
            name_b=name_b,
            title=f"Model Comparison - {track_name}",
            save_path=output_dir / "figures" / f"comparison_{track_name}.png",
        )

    return {"model_a": dict_a, "model_b": dict_b, "deltas": deltas}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation reports")
    subparsers = parser.add_subparsers(dest="command")

    eval_parser = subparsers.add_parser("eval", help="Evaluate a single model")
    eval_parser.add_argument("--model", required=True)
    eval_parser.add_argument("--track", default="simple_oval")
    eval_parser.add_argument("--episodes", type=int, default=100)
    eval_parser.add_argument("--output-dir", required=True)
    eval_parser.add_argument("--stochastic", action="store_true",
                             help="Use stochastic (non-deterministic) actions")
    eval_parser.add_argument("--perturbation", type=float, default=0.3,
                             help="Random start position offset in meters (0 = fixed start)")

    compare_parser = subparsers.add_parser("compare", help="Compare two models")
    compare_parser.add_argument("--model-a", required=True)
    compare_parser.add_argument("--model-b", required=True)
    compare_parser.add_argument("--name-a", default="Model A")
    compare_parser.add_argument("--name-b", default="Model B")
    compare_parser.add_argument("--track", default="simple_oval")
    compare_parser.add_argument("--episodes", type=int, default=100)
    compare_parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    if args.command == "eval":
        generate_eval_report(
            model_path=args.model,
            track_name=args.track,
            n_episodes=args.episodes,
            output_dir=args.output_dir,
            deterministic=not args.stochastic,
            initial_perturbation=args.perturbation,
        )
    elif args.command == "compare":
        compare_models(
            model_a_path=args.model_a,
            model_b_path=args.model_b,
            name_a=args.name_a,
            name_b=args.name_b,
            track_name=args.track,
            n_episodes=args.episodes,
            output_dir=args.output_dir,
        )
