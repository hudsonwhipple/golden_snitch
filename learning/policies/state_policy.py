"""MLP policy for state-based drone racing, compatible with SB3 and standalone BC."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from environments.base_env import OBS_DIM, ACT_DIM


class RacingMLP(nn.Module):
    """Two-layer MLP with tanh activation and LayerNorm for racing control.

    Used both as a standalone BC policy and to initialize SB3 PPO weights.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]

        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.Tanh(),
            ])
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.action_head = nn.Linear(in_dim, act_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(obs)
        return torch.tanh(self.action_head(features))

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Inference helper for evaluation loops."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            action = self.forward(obs_t)
            return action.squeeze(0).cpu().numpy()
