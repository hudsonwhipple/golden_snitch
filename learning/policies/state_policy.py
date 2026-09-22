"""MLP policy for state-based drone racing, compatible with SB3 and standalone BC."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from environments.base_env import OBS_DIM, ACT_DIM


class RacingMLP(nn.Module):
    """Two-layer tanh MLP for racing control.

    Used both as a standalone BC policy and to initialize SB3 PPO weights, so the
    architecture deliberately mirrors SB3's ``MlpPolicy`` with
    ``net_arch=[256, 256], activation_fn=Tanh``:

        mlp_extractor.policy_net : Linear -> Tanh -> Linear -> Tanh
        action_net               : Linear(hidden, act_dim)

    Two things follow from that and must not be "improved" independently:

    - **No LayerNorm.** SB3's extractor has none, and LayerNorm re-normalizes activations
      at run time, so weights trained with it produce entirely different outputs once
      transplanted into a network without it.
    - **No output squashing.** SB3 uses ``action_net`` output directly as the mean of a
      diagonal Gaussian (``squash_output=False``), so BC must regress the action itself
      rather than a pre-tanh logit. Targets are already in [-1, 1]; ``predict`` clips.
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
        return self.action_head(self.backbone(obs))

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Inference helper for evaluation loops."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            action = self.forward(obs_t).clamp(-1.0, 1.0)
            return action.squeeze(0).cpu().numpy()
