"""Configuration loading and seed management."""

from __future__ import annotations

import copy
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
TRACKS_DIR = CONFIGS_DIR / "tracks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: dict, overrides: dict) -> dict:
    """Deep-merge overrides into base config, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    overrides_path: str | Path | None = None,
    cli_overrides: dict | None = None,
) -> dict[str, Any]:
    """Load default config, merge optional overrides file and CLI args."""
    cfg = load_yaml(CONFIGS_DIR / "default.yaml")
    if overrides_path is not None:
        cfg = merge_configs(cfg, load_yaml(overrides_path))
    if cli_overrides:
        cfg = merge_configs(cfg, cli_overrides)
    return cfg


def set_global_seed(seed: int) -> None:
    """Set seeds for reproducibility across all RNG sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_run_id(stage: str, track: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stage}_{track}_{ts}"


def save_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
