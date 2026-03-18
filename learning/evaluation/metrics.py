"""Quantitative evaluation metrics for drone racing policies."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics for a single evaluation episode."""

    gates_passed: int = 0
    total_gates: int = 0
    lap_time: float = float("inf")
    collided: bool = False
    steps: int = 0
    total_reward: float = 0.0
    positions: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    velocities: list = field(default_factory=list)

    @property
    def gate_success_rate(self) -> float:
        return self.gates_passed / max(self.total_gates, 1)

    @property
    def completed_lap(self) -> bool:
        return self.gates_passed >= self.total_gates


@dataclass
class AggregateMetrics:
    """Aggregated metrics across multiple evaluation episodes."""

    gate_success_rate_mean: float = 0.0
    gate_success_rate_std: float = 0.0
    lap_time_mean: float = float("inf")
    lap_time_std: float = 0.0
    crash_rate: float = 0.0
    gate_normalized_time_mean: float = float("inf")
    control_smoothness_mean: float = 0.0
    control_smoothness_std: float = 0.0
    velocity_efficiency_mean: float = 0.0
    velocity_efficiency_std: float = 0.0
    avg_gates_passed: float = 0.0
    n_episodes: int = 0
    n_completed: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def compute_control_smoothness(actions: np.ndarray) -> float:
    """Mean absolute action change across an episode."""
    if len(actions) < 2:
        return 0.0
    deltas = np.abs(np.diff(actions, axis=0))
    return float(np.mean(deltas))


def compute_velocity_efficiency(
    velocities: np.ndarray,
    positions: np.ndarray,
    gate_positions: np.ndarray,
    gate_indices: np.ndarray,
    max_speed: float = 5.0,
) -> float:
    """Mean fraction of velocity directed toward the target gate."""
    if len(velocities) < 1:
        return 0.0

    efficiencies = []
    for i in range(len(velocities)):
        if i >= len(gate_indices):
            break
        gate_idx = int(gate_indices[i]) % len(gate_positions)
        gate_pos = gate_positions[gate_idx]
        direction = gate_pos - positions[i]
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            continue
        direction_norm = direction / dist
        speed_toward = np.dot(velocities[i], direction_norm)
        efficiencies.append(speed_toward / max_speed)

    return float(np.mean(efficiencies)) if efficiencies else 0.0


def aggregate_metrics(episodes: list[EpisodeMetrics]) -> AggregateMetrics:
    """Compute aggregate statistics across evaluation episodes."""
    n = len(episodes)
    if n == 0:
        return AggregateMetrics()

    gate_rates = [ep.gate_success_rate for ep in episodes]
    crashes = [1.0 if ep.collided else 0.0 for ep in episodes]
    completed = [ep for ep in episodes if ep.completed_lap]

    smoothness_vals = []
    for ep in episodes:
        if len(ep.actions) >= 2:
            smoothness_vals.append(compute_control_smoothness(np.array(ep.actions)))

    lap_times = [ep.lap_time for ep in completed]

    gate_norm_times = []
    for ep in episodes:
        if ep.gates_passed > 0 and ep.lap_time < float("inf"):
            gate_norm_times.append(ep.lap_time / ep.gates_passed)

    return AggregateMetrics(
        gate_success_rate_mean=float(np.mean(gate_rates)),
        gate_success_rate_std=float(np.std(gate_rates)),
        lap_time_mean=float(np.mean(lap_times)) if lap_times else float("inf"),
        lap_time_std=float(np.std(lap_times)) if lap_times else 0.0,
        crash_rate=float(np.mean(crashes)),
        gate_normalized_time_mean=float(np.mean(gate_norm_times)) if gate_norm_times else float("inf"),
        control_smoothness_mean=float(np.mean(smoothness_vals)) if smoothness_vals else 0.0,
        control_smoothness_std=float(np.std(smoothness_vals)) if smoothness_vals else 0.0,
        avg_gates_passed=float(np.mean([ep.gates_passed for ep in episodes])),
        n_episodes=n,
        n_completed=len(completed),
    )
