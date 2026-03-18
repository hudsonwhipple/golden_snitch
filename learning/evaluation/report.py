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
    plot_metrics_comparison,
)
from environments.tracks.track_loader import load_track


def generate_eval_report(
    model_path: str,
    track_name: str = "simple_oval",
    n_episodes: int = 100,
    output_dir: str | Path | None = None,
    seed: int = 2000,
) -> AggregateMetrics:
    """Run evaluation and generate full report with plots.

    Parameters
    ----------
    model_path : path to saved model
    track_name : track to evaluate on
    n_episodes : number of episodes
    output_dir : where to save results and plots
    seed : random seed

    Returns
    -------
    Aggregate metrics
    """
    agg, episodes = evaluate_policy(
        model_path=model_path,
        track_name=track_name,
        n_episodes=n_episodes,
        seed=seed,
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

        n_plot = min(10, len(episodes))
        trajectories = []
        outcomes = []
        for ep in episodes[:n_plot]:
            if ep.positions:
                trajectories.append(np.array(ep.positions))
                if ep.completed_lap:
                    outcomes.append("completed")
                elif ep.collided:
                    outcomes.append("crashed")
                else:
                    outcomes.append("timeout")

        if trajectories:
            plot_trajectory_overlay(
                trajectories=trajectories,
                gate_positions=gate_positions,
                gate_orientations=gate_orientations,
                gate_sizes=gate_sizes,
                outcomes=outcomes,
                title=f"Flight Trajectories - {track_name}",
                save_path=figures_dir / f"trajectory_overlay_{track_name}.png",
            )

    return agg


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
