"""Vision encoder training: teach CNN to extract gate-relative features from RGB.

Phase D training script. Requires:
1. A trained state-based policy (from Phase C)
2. Synthetic RGB renders with known gate positions (ground truth)

The vision encoder is trained to produce features that match the ground-truth
gate-relative observations, enabling the state policy to work from vision input.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from perception.feature_extractor import VisualFeatureExtractor
from perception.gate_detector import GateDetector
from utils.config import generate_run_id, save_config
from utils.tracking import init_wandb, log_metrics, finish_wandb


def train_gate_detector(
    images: np.ndarray,
    keypoints_gt: np.ndarray,
    confidences_gt: np.ndarray,
    output_dir: str = "outputs/perception",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> Path:
    """Train the gate keypoint detector on synthetic renders.

    Parameters
    ----------
    images : (N, H, W, 3) uint8 RGB images
    keypoints_gt : (N, 4, 2) ground truth normalized keypoints
    confidences_gt : (N,) ground truth confidence (1.0 if gate visible)
    output_dir : where to save the model
    epochs : training epochs
    batch_size : mini-batch size
    lr : learning rate
    device : torch device
    seed : random seed

    Returns
    -------
    Path to saved model
    """
    torch.manual_seed(seed)

    images_float = images.astype(np.float32) / 255.0
    images_tensor = torch.from_numpy(images_float).permute(0, 3, 1, 2)
    kp_tensor = torch.from_numpy(keypoints_gt.reshape(-1, 8).astype(np.float32))
    conf_tensor = torch.from_numpy(confidences_gt.astype(np.float32)).unsqueeze(1)

    n = len(images)
    n_val = int(n * 0.1)
    n_train = n - n_val

    indices = torch.randperm(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_ds = TensorDataset(images_tensor[train_idx], kp_tensor[train_idx], conf_tensor[train_idx])
    val_ds = TensorDataset(images_tensor[val_idx], kp_tensor[val_idx], conf_tensor[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = GateDetector(input_channels=3, pretrained=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    kp_criterion = nn.MSELoss()
    conf_criterion = nn.BCELoss()

    run_id = generate_run_id("perception", "gate_detector")
    save_dir = Path(output_dir) / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for imgs, kps, confs in train_loader:
            imgs, kps, confs = imgs.to(device), kps.to(device), confs.to(device)
            pred_kp, pred_conf = model(imgs)
            loss = kp_criterion(pred_kp, kps) + 0.5 * conf_criterion(pred_conf, confs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, kps, confs in val_loader:
                imgs, kps, confs = imgs.to(device), kps.to(device), confs.to(device)
                pred_kp, pred_conf = model(imgs)
                loss = kp_criterion(pred_kp, kps) + 0.5 * conf_criterion(pred_conf, confs)
                val_losses.append(loss.item())

        scheduler.step()
        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "best_detector.pt")

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: train={np.mean(train_losses):.6f}, val={val_loss:.6f}")

    print(f"Gate detector saved to {save_dir}")
    return save_dir / "best_detector.pt"


def train_feature_encoder(
    images: np.ndarray,
    gate_features_gt: np.ndarray,
    detector_path: str | None = None,
    output_dir: str = "outputs/perception",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> Path:
    """Train the visual feature extractor to match ground-truth gate features.

    The extractor learns to produce 64-dim features that, when combined with
    telemetry, allow the state policy to function from vision input.

    Parameters
    ----------
    images : (N, n_stack, H, W, 3) stacked RGB frames
    gate_features_gt : (N, 64) ground-truth feature targets
        (typically: gate_relative_pos, gate_keypoints, gate_confidence, padded)
    detector_path : optional pre-trained gate detector weights
    output_dir : where to save
    """
    torch.manual_seed(seed)

    n, n_stack, h, w, c = images.shape
    images_flat = images.reshape(n, h, w, n_stack * c).astype(np.float32) / 255.0
    images_tensor = torch.from_numpy(images_flat).permute(0, 3, 1, 2)
    features_tensor = torch.from_numpy(gate_features_gt.astype(np.float32))

    n_val = int(n * 0.1)
    train_ds = TensorDataset(images_tensor[:n - n_val], features_tensor[:n - n_val])
    val_ds = TensorDataset(images_tensor[n - n_val:], features_tensor[n - n_val:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = VisualFeatureExtractor(
        feature_dim=gate_features_gt.shape[1],
        input_channels=n_stack * c,
        image_size=(h, w),
    )

    if detector_path:
        detector = GateDetector(input_channels=3, pretrained=False)
        detector.load_state_dict(torch.load(detector_path, map_location="cpu", weights_only=True))
        model.load_detector_weights(detector)
        print("Loaded pre-trained detector backbone weights")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    run_id = generate_run_id("perception", "feature_encoder")
    save_dir = Path(output_dir) / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        for imgs, feats in train_loader:
            imgs, feats = imgs.to(device), feats.to(device)
            pred = model(imgs)
            loss = criterion(pred, feats)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, feats in val_loader:
                imgs, feats = imgs.to(device), feats.to(device)
                loss = criterion(model(imgs), feats)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_dir / "best_encoder.pt")

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: val_loss={val_loss:.6f}")

    print(f"Feature encoder saved to {save_dir}")
    return save_dir / "best_encoder.pt"
