"""CNN+MLP combined policy for vision-based drone racing.

Takes raw RGB frames + telemetry state and outputs racing commands.
Total observation: 24 (state) + 64 (visual features) = 88 dimensions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from learning.policies.state_policy import RacingMLP
from perception.feature_extractor import VisualFeatureExtractor


class VisionRacingPolicy(nn.Module):
    """Combined vision + state policy for deployment.

    In training:
    1. State-based policy is trained first (Phases A-C)
    2. Visual feature extractor is trained to reproduce gate-relative features
    3. Combined policy is fine-tuned end-to-end with PPO

    Architecture:
        RGB frames → VisualFeatureExtractor → 64-dim features
        [64-dim visual + 24-dim state] → MLP → 4-dim actions
    """

    VISUAL_DIM = 64
    STATE_DIM = 24
    COMBINED_DIM = STATE_DIM + VISUAL_DIM  # 88

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        image_size: tuple[int, int] = (96, 128),
        n_frame_stack: int = 3,
    ):
        super().__init__()
        self.image_size = image_size
        self.n_frame_stack = n_frame_stack

        self.vision = VisualFeatureExtractor(
            feature_dim=self.VISUAL_DIM,
            input_channels=3 * n_frame_stack,
            image_size=image_size,
        )

        hidden_dims = hidden_dims or [256, 256]
        self.policy = RacingMLP(
            obs_dim=self.COMBINED_DIM,
            act_dim=4,
            hidden_dims=hidden_dims,
        )

    def load_state_policy(self, state_policy: RacingMLP) -> None:
        """Initialize policy MLP from pre-trained state-based policy.

        The state-based policy has input dim 24; we expand the first layer
        to accept the combined 88-dim input while preserving learned weights.
        """
        first_layer = self.policy.backbone[0]
        old_first_layer = state_policy.backbone[0]

        with torch.no_grad():
            first_layer.weight[:, :24] = old_first_layer.weight
            first_layer.weight[:, 24:] = 0.01 * torch.randn_like(first_layer.weight[:, 24:])
            first_layer.bias.copy_(old_first_layer.bias)

        for i in range(1, len(self.policy.backbone)):
            src = state_policy.backbone[i]
            dst = self.policy.backbone[i]
            if hasattr(src, "weight"):
                dst.load_state_dict(src.state_dict())

        self.policy.action_head.load_state_dict(state_policy.action_head.state_dict())

    def forward(
        self,
        frames: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        frames : (B, C, H, W) stacked RGB frames
        state : (B, 24) telemetry state vector

        Returns
        -------
        (B, 4) action commands
        """
        visual_features = self.vision(frames)
        combined = torch.cat([state, visual_features], dim=1)
        return self.policy(combined)

    def predict(
        self,
        frames: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        """Numpy inference helper."""
        with torch.no_grad():
            frames_t = torch.as_tensor(frames, dtype=torch.float32)
            state_t = torch.as_tensor(state, dtype=torch.float32)

            if frames_t.dim() == 3:
                frames_t = frames_t.unsqueeze(0)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)

            action = self.forward(frames_t, state_t)
            return action.squeeze(0).cpu().numpy()
