"""Behavior cloning trainer for drone racing policy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from learning.policies.state_policy import RacingMLP
from learning.utils.model_io import load_expert_data, save_bc_model
from utils.config import load_config, generate_run_id, save_config
from utils.tracking import init_wandb, log_metrics, finish_wandb


def train_bc(
    expert_data_path: str,
    output_dir: str = "outputs/bc",
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    validation_split: float = 0.2,
    hidden_dims: list[int] | None = None,
    device: str = "cpu",
    seed: int = 42,
    track_name: str = "simple_oval",
    cfg: dict | None = None,
) -> tuple[Path, dict]:
    """Train a behavior cloning policy from expert demonstrations.

    Parameters
    ----------
    expert_data_path : path to expert_data.npz
    output_dir : base output directory
    epochs : number of training epochs
    batch_size : mini-batch size
    lr : learning rate
    validation_split : fraction of data for validation
    hidden_dims : MLP hidden layer sizes
    device : torch device
    seed : random seed
    track_name : for run ID generation
    cfg : full config dict for logging

    Returns
    -------
    (model_path, metrics_dict)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = load_expert_data(expert_data_path)
    obs = data["observations"]
    actions = data["actions"]

    n_samples = len(obs)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    obs_mean = obs[train_idx].mean(axis=0)
    obs_std = obs[train_idx].std(axis=0) + 1e-8

    obs_norm = (obs - obs_mean) / obs_std

    train_dataset = TensorDataset(
        torch.tensor(obs_norm[train_idx], dtype=torch.float32),
        torch.tensor(actions[train_idx], dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(obs_norm[val_idx], dtype=torch.float32),
        torch.tensor(actions[val_idx], dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    hidden_dims = hidden_dims or [256, 256]
    model = RacingMLP(obs_dim=obs.shape[1], act_dim=actions.shape[1], hidden_dims=hidden_dims)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    run_id = generate_run_id("bc", track_name)
    save_dir = Path(output_dir) / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = init_wandb(run_id, cfg or {}, tags=["stage:bc", f"track:{track_name}"])

    best_val_loss = float("inf")
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    print(f"Training BC policy: {n_train} train, {n_val} val samples")
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            pred = model(obs_batch)
            loss = criterion(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                pred = model(obs_batch)
                loss = criterion(pred, act_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        log_metrics({
            "bc/train_loss": train_loss,
            "bc/val_loss": val_loss,
            "bc/learning_rate": current_lr,
        }, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_bc_model(model, save_dir / "best_model.pt")

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: train={train_loss:.6f}, "
                  f"val={val_loss:.6f}, lr={current_lr:.6f}")

    save_bc_model(model, save_dir / "final_model.pt")

    np.save(save_dir / "obs_mean.npy", obs_mean)
    np.save(save_dir / "obs_std.npy", obs_std)

    metrics = {
        "best_val_loss": float(best_val_loss),
        "best_epoch": best_epoch,
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        "n_parameters": sum(p.numel() for p in model.parameters()),
    }

    with open(save_dir / "eval_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    save_config(cfg or {}, save_dir / "training_config.yaml")

    finish_wandb()

    print(f"\nBC training complete. Best val loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"Model saved to {save_dir}")

    return save_dir / "best_model.pt", metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train behavior cloning policy")
    parser.add_argument("--data", required=True, help="Path to expert_data.npz")
    parser.add_argument("--output-dir", default="outputs/bc")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--track", default="simple_oval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config()
    train_bc(
        expert_data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        track_name=args.track,
        seed=args.seed,
        device=args.device,
        cfg=cfg,
    )
