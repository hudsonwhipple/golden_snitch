"""Curriculum learning manager for progressive difficulty scaling."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CurriculumLevel:
    name: str
    n_gates: int  # -1 means all gates from track
    max_speed: float
    obs_noise: bool = False


@dataclass
class CurriculumState:
    current_level: int = 0
    steps_at_level: int = 0
    total_steps: int = 0
    promotion_history: list[dict] = field(default_factory=list)


class CurriculumManager:
    """Manages progressive difficulty scaling during training.

    Levels are ordered by difficulty. The agent is promoted when gate success
    rate exceeds `promotion_threshold` and demoted when it drops below
    `demotion_threshold`.
    """

    def __init__(
        self,
        levels: list[CurriculumLevel] | None = None,
        promotion_threshold: float = 0.85,
        demotion_threshold: float = 0.50,
        min_steps_per_level: int = 500_000,
        eval_episodes: int = 100,
    ):
        self.levels = levels or [
            CurriculumLevel("single_gate", n_gates=1, max_speed=2.0),
            CurriculumLevel("three_gates", n_gates=3, max_speed=3.5),
            CurriculumLevel("full_track", n_gates=-1, max_speed=5.0),
            CurriculumLevel("full_track_noise", n_gates=-1, max_speed=5.0, obs_noise=True),
        ]
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold
        self.min_steps_per_level = min_steps_per_level
        self.eval_episodes = eval_episodes
        self.state = CurriculumState()

    @classmethod
    def from_config(cls, cfg: dict) -> CurriculumManager:
        curriculum_cfg = cfg.get("curriculum", {})
        levels = []
        for ldef in curriculum_cfg.get("levels", []):
            levels.append(CurriculumLevel(
                name=ldef["name"],
                n_gates=ldef["n_gates"],
                max_speed=ldef["max_speed"],
                obs_noise=ldef.get("obs_noise", False),
            ))
        return cls(
            levels=levels or None,
            promotion_threshold=curriculum_cfg.get("promotion_threshold", 0.85),
            demotion_threshold=curriculum_cfg.get("demotion_threshold", 0.50),
            min_steps_per_level=curriculum_cfg.get("min_steps_per_level", 500_000),
            eval_episodes=curriculum_cfg.get("eval_episodes", 100),
        )

    @property
    def current_level(self) -> CurriculumLevel:
        return self.levels[self.state.current_level]

    @property
    def level_index(self) -> int:
        return self.state.current_level

    @property
    def is_max_level(self) -> bool:
        return self.state.current_level >= len(self.levels) - 1

    def update(self, gate_success_rate: float, steps: int) -> str | None:
        """Evaluate whether to promote or demote based on performance.

        Parameters
        ----------
        gate_success_rate : fraction of gates passed (0-1)
        steps : current total training steps

        Returns
        -------
        "promoted", "demoted", or None
        """
        step_delta = steps - self.state.total_steps
        self.state.total_steps = steps
        self.state.steps_at_level += step_delta

        if self.state.steps_at_level < self.min_steps_per_level:
            return None

        if gate_success_rate >= self.promotion_threshold and not self.is_max_level:
            old_level = self.state.current_level
            self.state.current_level += 1
            self.state.steps_at_level = 0
            self.state.promotion_history.append({
                "step": steps,
                "from_level": old_level,
                "to_level": self.state.current_level,
                "action": "promoted",
                "gate_success_rate": gate_success_rate,
            })
            return "promoted"

        if gate_success_rate < self.demotion_threshold and self.state.current_level > 0:
            old_level = self.state.current_level
            self.state.current_level -= 1
            self.state.steps_at_level = 0
            self.state.promotion_history.append({
                "step": steps,
                "from_level": old_level,
                "to_level": self.state.current_level,
                "action": "demoted",
                "gate_success_rate": gate_success_rate,
            })
            return "demoted"

        return None

    def get_env_kwargs(self) -> dict[str, Any]:
        """Get environment kwargs for the current curriculum level."""
        level = self.current_level
        return {
            "n_active_gates": level.n_gates,
            "obs_noise_std": 0.02 if level.obs_noise else 0.0,
        }

    def to_dict(self) -> dict:
        return {
            "current_level": self.state.current_level,
            "steps_at_level": self.state.steps_at_level,
            "total_steps": self.state.total_steps,
            "promotion_history": self.state.promotion_history,
        }

    def load_state(self, state_dict: dict) -> None:
        self.state = CurriculumState(
            current_level=state_dict["current_level"],
            steps_at_level=state_dict["steps_at_level"],
            total_steps=state_dict["total_steps"],
            promotion_history=state_dict.get("promotion_history", []),
        )

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path: str | Path) -> None:
        with open(path) as f:
            self.load_state(json.load(f))
