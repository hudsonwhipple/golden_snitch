"""Regenerate the README's results table and figures from two saved policies.

Both policies are evaluated in the current environment, where gates are solid frames,
over the same 100 seeded episodes and start-position spread as `learning.evaluation`:

    python scripts/make_readme_figures.py \
        --before models/ppo_before_solid_gates.zip \
        --after  models/ppo_solid_gates.zip

Each policy is also scored under the original thin-gate rules, so the table shows what
the first policy's 100/100 was actually measuring.

Writes docs/assets/{before_after.png, gate_offsets.png, flight.gif, results.json}.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO

from contextlib import contextmanager

from environments import base_env
from environments.base_env import DEFAULT_COLLISION_RADIUS
from environments.gate_detection import _plane_crossing, check_gate_passage
from environments.racing_env import DroneRacingEnv
from learning.evaluation.evaluate import DEFAULT_EVAL_PERTURBATION

POLICY_DT = 1.0 / 48
SEED = 2000  # same base seed as learning.evaluation.evaluate_policy
BEFORE_COLOR = "#e07a2f"
AFTER_COLOR = "#2e8b57"
GATE_COLOR = "#30343f"
APERTURE_COLOR = "#9aa0ac"


@dataclass
class Rollout:
    positions: np.ndarray
    gates_passed: int
    outcome: str  # "completed" | "crashed" | "timeout"
    cause: str | None
    steps: int
    crossing_offsets: list[float] = field(default_factory=list)


def rollout(model: PPO, env: DroneRacingEnv, seed: int) -> Rollout:
    obs, info = env.reset(seed=seed)
    positions = [info["position"].copy()]
    n_gates = len(env.track.gates)
    done, steps = False, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        positions.append(info["position"].copy())
        done = terminated or truncated
        steps += 1
    pos = np.array(positions)

    # Offset of every forward crossing of each gate's plane, in the gate's own frame.
    # d = max(|lateral|, |vertical|) is the quantity the collision check bands on.
    offsets = []
    for gate in env.track.gates:
        for a, b in zip(pos[:-1], pos[1:]):
            local, forward = _plane_crossing(a, b, gate)
            if local is not None and forward:
                d = float(np.max(np.abs(local)))
                if d < gate.size[0]:  # ignore crossings of the plane far from the gate
                    offsets.append(d)

    if info["gates_passed"] >= n_gates:
        outcome = "completed"
    elif info["collided"]:
        outcome = "crashed"
    else:
        outcome = "timeout"
    return Rollout(pos, info["gates_passed"], outcome, info.get("collision_cause"), steps, offsets)


def evaluate(model_path: Path, n_episodes: int) -> list[Rollout]:
    model = PPO.load(str(model_path), device="cpu")
    env = DroneRacingEnv(track_name="simple_oval", initial_perturbation=DEFAULT_EVAL_PERTURBATION)
    runs = [rollout(model, env, SEED + i) for i in range(n_episodes)]
    env.close()
    return runs, env


@contextmanager
def thin_gates():
    """Temporarily restore the original gate rules: no frame collision, and a 5 cm margin
    that *grew* each aperture. This is the environment the first policy was trained in."""
    orig_progress = base_env.RacingMixin._update_gate_progress
    orig_collision = base_env.RacingMixin._check_gate_collision

    def progress(self, prev_pos, curr_pos):
        passed = check_gate_passage(prev_pos, curr_pos, self._get_next_gate(), clearance=-0.05)
        if passed:
            self.racing_state.gates_passed += 1
            self.racing_state.next_gate_idx += 1
        return passed

    base_env.RacingMixin._update_gate_progress = progress
    base_env.RacingMixin._check_gate_collision = lambda self, a, b: False
    try:
        yield
    finally:
        base_env.RacingMixin._update_gate_progress = orig_progress
        base_env.RacingMixin._check_gate_collision = orig_collision


def summarize(runs: list[Rollout], n_gates: int) -> dict:
    laps = [r for r in runs if r.outcome == "completed"]
    strikes = sum(1 for r in runs if r.cause == "gate_frame")
    lap_times = [r.steps * POLICY_DT for r in laps]
    return {
        "episodes": len(runs),
        "laps_completed": len(laps),
        "gate_success_rate": float(np.mean([r.gates_passed / n_gates for r in runs])),
        "mean_lap_time_s": float(np.mean(lap_times)) if lap_times else None,
        "gate_frame_strikes": strikes,
        "other_crashes": sum(1 for r in runs if r.outcome == "crashed") - strikes,
        "crossing_offset_p95_m": float(np.percentile(sum((r.crossing_offsets for r in runs), []), 95)),
    }


def draw_gates(ax, track, label=True):
    for gate in track.gates:
        yaw = gate.orientation[2]
        lateral = np.array([-np.sin(yaw), np.cos(yaw)])
        center = np.asarray(gate.position[:2])
        half = gate.size[0] / 2
        outer = half + gate.frame_thickness
        a, b = center - half * lateral, center + half * lateral
        ax.plot(*zip(a, b), color=APERTURE_COLOR, lw=1.2, ls=(0, (2, 2)), zorder=2)
        for sign in (-1, 1):
            p, q = center + sign * half * lateral, center + sign * outer * lateral
            ax.plot(*zip(p, q), color=GATE_COLOR, lw=6, solid_capstyle="butt", zorder=3)
        if label:
            fwd = np.array([np.cos(yaw), np.sin(yaw)])
            ax.annotate("", xy=center + 0.35 * fwd, xytext=center,
                        arrowprops=dict(arrowstyle="-|>", color=GATE_COLOR, lw=1.2), zorder=3)


def style(ax, title):
    ax.set_title(title, fontsize=12, loc="left", pad=10)
    ax.set_aspect("equal")
    ax.set_xlim(-0.8, 3.8)
    ax.set_ylim(-0.8, 4.6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25)


def plot_before_after(before, after, track, s_before, s_after, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), dpi=150)
    panels = [
        (axes[0], before, BEFORE_COLOR, s_before, "Before: trained when gates were infinitely thin"),
        (axes[1], after, AFTER_COLOR, s_after, "After: retrained against solid gate frames"),
    ]
    for ax, runs, color, s, title in panels:
        draw_gates(ax, track)
        for r in runs:
            ok = r.outcome == "completed"
            ax.plot(r.positions[:, 0], r.positions[:, 1], color=color if ok else "#555b66",
                    alpha=0.2 if ok else 0.45, lw=1, zorder=1)
            if not ok:
                ax.scatter(*r.positions[-1, :2], marker="x", s=36, color="black", lw=1.5, zorder=4)
        style(ax, title)
        ax.text(0.02, 0.02,
                f"{s['laps_completed']}/{s['episodes']} laps    {s['gate_frame_strikes']} gate strikes"
                + (f"    {s['mean_lap_time_s']:.2f} s/lap" if s["mean_lap_time_s"] else ""),
                transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, alpha=0.9))
    fig.suptitle("Same 100 randomized starts, same solid-gate environment  (x = crash)",
                 fontsize=10, color="#555", y=0.02)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_gate_offsets(before, after, track, out):
    gate = track.gates[0]
    half = gate.size[0] / 2
    clean = half - DEFAULT_COLLISION_RADIUS
    outer = half + gate.frame_thickness + DEFAULT_COLLISION_RADIUS
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    bins = np.linspace(0, 0.8, 41)
    for runs, color, label in ((before, BEFORE_COLOR, "Before (thin-gate training)"),
                               (after, AFTER_COLOR, "After (solid-gate training)")):
        d = sum((r.crossing_offsets for r in runs), [])
        ax.hist(d, bins=bins, color=color, alpha=0.18)
        ax.hist(d, bins=bins, color=color, histtype="step", lw=2,
                label=f"{label}: 95th percentile {np.percentile(d, 95):.2f} m")
    ax.axvspan(clean, outer, color=GATE_COLOR, alpha=0.12, lw=0)
    ax.axvline(half, color=GATE_COLOR, lw=1.2, ls="--")
    top = ax.get_ylim()[1]
    ax.text(half + 0.008, top * 0.62, "aperture\nedge", fontsize=9, color=GATE_COLOR)
    ax.text(outer - 0.008, top * 0.93, "frame-strike zone\n(drone body touches frame)", fontsize=9,
            color=GATE_COLOR, ha="right", va="top")
    ax.set_xlabel("Distance from gate centre at the crossing, max(|lateral|, |vertical|)  (m)")
    ax.set_ylabel("Gate crossings")
    ax.set_title("Where each policy threads the gate", loc="left", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def make_gif(run: Rollout, track, out, lap_time: float):
    pos = run.positions
    frames = list(range(0, len(pos), 2)) + [len(pos) - 1] * 12
    fig, ax = plt.subplots(figsize=(4.8, 5.4), dpi=90)
    draw_gates(ax, track)
    style(ax, "")
    ax.set_title(f"Retrained policy, one lap ({lap_time:.2f} s)", fontsize=11, loc="left")
    trail, = ax.plot([], [], color=AFTER_COLOR, lw=2)
    dot = ax.scatter([pos[0, 0]], [pos[0, 1]], s=60, color=AFTER_COLOR, edgecolor="white", zorder=5)
    clock = ax.text(0.98, 0.02, "", transform=ax.transAxes, ha="right", fontsize=10, color="#555")

    def update(i):
        trail.set_data(pos[: i + 1, 0], pos[: i + 1, 1])
        dot.set_offsets(pos[i, :2])
        clock.set_text(f"t = {i * POLICY_DT:.2f} s")
        return trail, dot, clock

    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    anim.save(out, writer=PillowWriter(fps=24))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", type=Path, default=ROOT / "models/ppo_before_solid_gates.zip")
    ap.add_argument("--after", type=Path, default=ROOT / "models/ppo_solid_gates.zip")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/assets")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    before, env = evaluate(args.before, args.episodes)
    after, _ = evaluate(args.after, args.episodes)
    track = env.track
    n_gates = len(track.gates)
    s_before, s_after = summarize(before, n_gates), summarize(after, n_gates)

    plot_before_after(before, after, track, s_before, s_after, args.out / "before_after.png")
    plot_gate_offsets(before, after, track, args.out / "gate_offsets.png")
    lap = next(r for r in after if r.outcome == "completed")
    make_gif(lap, track, args.out / "flight.gif", lap.steps * POLICY_DT)

    with thin_gates():
        thin_before, _ = evaluate(args.before, args.episodes)
        thin_after, _ = evaluate(args.after, args.episodes)

    results = {"perturbation_m": DEFAULT_EVAL_PERTURBATION, "seed": SEED,
               "solid_gates": {"before": s_before, "after": s_after},
               "thin_gates": {"before": summarize(thin_before, n_gates),
                              "after": summarize(thin_after, n_gates)}}
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
