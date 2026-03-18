"""Experiment tracking helpers for W&B and TensorBoard."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def init_wandb(
    run_id: str,
    cfg: dict[str, Any],
    tags: list[str] | None = None,
) -> Any | None:
    """Initialize a W&B run if enabled in config. Returns the run object or None."""
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    try:
        import wandb

        run = wandb.init(
            project=wandb_cfg.get("project", "golden-snitch"),
            name=run_id,
            config=cfg,
            tags=tags or [],
            reinit=True,
        )
        return run
    except ImportError:
        print("[WARN] wandb not installed, skipping experiment tracking")
        return None


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics to W&B if a run is active."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


def finish_wandb() -> None:
    """Finish the active W&B run."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass
