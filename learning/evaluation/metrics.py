"""Quantitative evaluation metrics for drone racing policies."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RewardBreakdown:
    """Cumulative reward components for an episode."""

    gate: float = 0.0
    progress: float = 0.0
    velocity: float = 0.0
    smoothness: float = 0.0
    collision: float = 0.0
    time: float = 0.0

    def accumulate(self, step_breakdown: dict) -> None:
        self.gate += step_breakdown.get("gate", 0.0)
        self.progress += step_breakdown.get("progress", 0.0)
        self.velocity += step_breakdown.get("velocity", 0.0)
        self.smoothness += step_breakdown.get("smoothness", 0.0)
        self.collision += step_breakdown.get("collision", 0.0)
        self.time += step_breakdown.get("time", 0.0)

    def to_dict(self) -> dict:
        return {
            "gate": self.gate,
            "progress": self.progress,
            "velocity": self.velocity,
            "smoothness": self.smoothness,
            "collision": self.collision,
            "time": self.time,
        }


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
    reward_breakdown: RewardBreakdown = field(default_factory=RewardBreakdown)
    speeds: list = field(default_factory=list)

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
    timeout_rate: float = 0.0
    gate_normalized_time_mean: float = float("inf")
    control_smoothness_mean: float = 0.0
    control_smoothness_std: float = 0.0
    velocity_efficiency_mean: float = 0.0
    velocity_efficiency_std: float = 0.0
    avg_gates_passed: float = 0.0
    n_episodes: int = 0
    n_completed: int = 0
    n_crashed: int = 0
    n_timeout: int = 0
    mean_episode_steps: float = 0.0
    mean_speed: float = 0.0
    max_speed: float = 0.0
    mean_altitude: float = 0.0
    reward_breakdown_mean: dict = field(default_factory=dict)

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
    crashes = [ep for ep in episodes if ep.collided]
    completed = [ep for ep in episodes if ep.completed_lap]
    timeouts = [ep for ep in episodes if not ep.collided and not ep.completed_lap]

    smoothness_vals = []
    for ep in episodes:
        if len(ep.actions) >= 2:
            smoothness_vals.append(compute_control_smoothness(np.array(ep.actions)))

    lap_times = [ep.lap_time for ep in completed]

    gate_norm_times = []
    for ep in episodes:
        if ep.gates_passed > 0 and ep.lap_time < float("inf"):
            gate_norm_times.append(ep.lap_time / ep.gates_passed)

    # Flight statistics
    all_speeds = []
    all_altitudes = []
    for ep in episodes:
        if ep.speeds:
            all_speeds.extend(ep.speeds)
        if ep.positions:
            all_altitudes.extend([p[2] if len(p) > 2 else 0.0 for p in ep.positions])

    # Reward breakdown averages
    breakdown_keys = ["gate", "progress", "velocity", "smoothness", "collision", "time"]
    reward_breakdown_mean = {}
    for key in breakdown_keys:
        vals = [getattr(ep.reward_breakdown, key, 0.0) for ep in episodes]
        reward_breakdown_mean[key] = float(np.mean(vals))

    return AggregateMetrics(
        gate_success_rate_mean=float(np.mean(gate_rates)),
        gate_success_rate_std=float(np.std(gate_rates)),
        lap_time_mean=float(np.mean(lap_times)) if lap_times else float("inf"),
        lap_time_std=float(np.std(lap_times)) if lap_times else 0.0,
        crash_rate=len(crashes) / n,
        timeout_rate=len(timeouts) / n,
        gate_normalized_time_mean=float(np.mean(gate_norm_times)) if gate_norm_times else float("inf"),
        control_smoothness_mean=float(np.mean(smoothness_vals)) if smoothness_vals else 0.0,
        control_smoothness_std=float(np.std(smoothness_vals)) if smoothness_vals else 0.0,
        avg_gates_passed=float(np.mean([ep.gates_passed for ep in episodes])),
        n_episodes=n,
        n_completed=len(completed),
        n_crashed=len(crashes),
        n_timeout=len(timeouts),
        mean_episode_steps=float(np.mean([ep.steps for ep in episodes])),
        mean_speed=float(np.mean(all_speeds)) if all_speeds else 0.0,
        max_speed=float(np.max(all_speeds)) if all_speeds else 0.0,
        mean_altitude=float(np.mean(all_altitudes)) if all_altitudes else 0.0,
        reward_breakdown_mean=reward_breakdown_mean,
    )
