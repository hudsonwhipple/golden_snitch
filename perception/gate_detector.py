"""MobileNetV3-Small gate keypoint detector for monocular vision.

Detects racing gate corners in RGB images, outputting 4 keypoint coordinates
and a confidence score. Designed for <5ms inference on Apple Silicon MPS.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class GateKeypointHead(nn.Module):
    """Regression head that predicts 4 gate corner keypoints + confidence."""

    def __init__(self, in_features: int = 576):
        super().__init__()
        self.keypoint_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 8),  # 4 corners x 2 (u, v) normalized to [0, 1]
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keypoints = torch.sigmoid(self.keypoint_head(features))
        confidence = self.confidence_head(features)
        return keypoints, confidence


class GateDetector(nn.Module):
    """MobileNetV3-Small backbone with keypoint regression head.

    Input: RGB image tensor (B, 3, H, W) -- 3-frame stacking uses (B, 9, H, W)
    Output: keypoints (B, 8), confidence (B, 1)
    """

    def __init__(
        self,
        input_channels: int = 3,
        pretrained: bool = True,
        image_size: tuple[int, int] = (96, 128),
    ):
        super().__init__()
        self.image_size = image_size

        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )

        if input_channels != 3:
            old_conv = backbone.features[0][0]
            backbone.features[0][0] = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = GateKeypointHead(in_features=576)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        keypoints, confidence = self.head(pooled)
        return keypoints, confidence

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Run inference on a single RGB frame.

        Parameters
        ----------
        frame : (H, W, 3) uint8 or float32 RGB image

        Returns
        -------
        keypoints : (4, 2) normalized corner coordinates
        confidence : float in [0, 1]
        """
        with torch.no_grad():
            if frame.dtype == np.uint8:
                frame = frame.astype(np.float32) / 255.0

            h, w = self.image_size
            import cv2
            resized = cv2.resize(frame, (w, h))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()

            keypoints, confidence = self.forward(tensor)
            kp = keypoints[0].cpu().numpy().reshape(4, 2)
            conf = float(confidence[0, 0].cpu().numpy())
            return kp, conf
