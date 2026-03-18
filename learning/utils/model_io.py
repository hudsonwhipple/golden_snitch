"""Model saving and loading utilities for all training stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from utils.config import save_config


def save_bc_model(
    model: torch.nn.Module,
    path: str | Path,
    config: dict | None = None,
    metrics: dict | None = None,
) -> None:
    """Save a behavior cloning model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    if config:
        save_config(config, path.parent / "training_config.yaml")
    if metrics:
        with open(path.parent / "eval_results.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)


def load_bc_model(
    model: torch.nn.Module,
    path: str | Path,
) -> torch.nn.Module:
    """Load behavior cloning weights into an existing model."""
    path = Path(path)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model


def save_ppo_model(
    model: BaseAlgorithm,
    path: str | Path,
    config: dict | None = None,
    curriculum_state: dict | None = None,
) -> None:
    """Save a PPO model checkpoint with optional curriculum state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    if config:
        save_config(config, path.parent / "training_config.yaml")
    if curriculum_state:
        with open(str(path) + "_curriculum.json", "w") as f:
            json.dump(curriculum_state, f, indent=2)


def load_ppo_model(
    path: str | Path,
    env=None,
) -> PPO:
    """Load a PPO model from checkpoint."""
    path = Path(path)
    model_path = str(path)
    if not model_path.endswith(".zip"):
        model_path += ".zip"
    return PPO.load(model_path, env=env)


def load_checkpoint(
    path: str | Path,
    env=None,
) -> tuple[PPO, dict | None]:
    """Load a PPO checkpoint and its curriculum state if available."""
    path = Path(path)
    model = load_ppo_model(path, env=env)

    curriculum_path = str(path) + "_curriculum.json"
    curriculum_state = None
    if Path(curriculum_path).exists():
        with open(curriculum_path) as f:
            curriculum_state = json.load(f)

    return model, curriculum_state


def save_expert_data(
    observations: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    gate_indices: np.ndarray,
    timestamps: np.ndarray,
    path: str | Path,
    metadata: dict | None = None,
) -> None:
    """Save expert demonstration data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=observations.astype(np.float32),
        actions=actions.astype(np.float32),
        episode_ids=episode_ids.astype(np.int32),
        gate_indices=gate_indices.astype(np.int32),
        timestamps=timestamps.astype(np.float32),
    )
    if metadata:
        meta_path = path.parent / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)


def load_expert_data(path: str | Path) -> dict[str, np.ndarray]:
    """Load expert demonstration data."""
    data = np.load(path)
    return {key: data[key] for key in data.files}
