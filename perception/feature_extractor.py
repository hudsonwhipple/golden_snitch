"""Visual feature extractor for policy integration.

Shares the MobileNetV3-Small backbone with the gate detector,
producing a compact 64-dim feature vector for the vision policy.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from perception.gate_detector import GateDetector


class VisualFeatureExtractor(nn.Module):
    """Extract visual features from RGB frames for policy input.

    Uses the gate detector's backbone and adds a projection to 64 dims.
    Can optionally share weights with a pre-trained gate detector.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        input_channels: int = 9,  # 3 frames stacked
        image_size: tuple[int, int] = (96, 128),
    ):
        super().__init__()
        self.feature_dim = feature_dim

        self.detector = GateDetector(
            input_channels=input_channels,
            pretrained=False,
            image_size=image_size,
        )

        self.projection = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim),
        )

    def load_detector_weights(self, detector: GateDetector) -> None:
        """Load pre-trained gate detector backbone weights."""
        self.detector.features.load_state_dict(detector.features.state_dict())

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract features from stacked RGB frames.

        Parameters
        ----------
        frames : (B, C, H, W) tensor (C=9 for 3-frame stack)

        Returns
        -------
        (B, feature_dim) feature vector
        """
        features = self.detector.features(frames)
        pooled = self.detector.pool(features).flatten(1)
        return self.projection(pooled)

    def extract(self, frames: np.ndarray) -> np.ndarray:
        """Numpy inference helper.

        Parameters
        ----------
        frames : (H, W, C) where C=9 for stacked frames, or (3, H, W, 3) unstacked

        Returns
        -------
        (feature_dim,) feature vector
        """
        with torch.no_grad():
            if frames.ndim == 4:
                frames = np.concatenate([f for f in frames], axis=-1)
            if frames.dtype == np.uint8:
                frames = frames.astype(np.float32) / 255.0

            h, w = self.detector.image_size
            import cv2
            resized = cv2.resize(frames, (w, h))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()
            features = self.forward(tensor)
            return features[0].cpu().numpy()
