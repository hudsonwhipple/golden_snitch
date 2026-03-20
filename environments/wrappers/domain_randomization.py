"""Domain randomization wrappers for visual and physics robustness.

Visual randomization is applied to camera observations.
Physics randomization is applied to the quadrotor dynamics parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import gymnasium as gym


@dataclass
class VisualRandomizationConfig:
    """Parameters for visual domain randomization."""

    brightness_range: tuple[float, float] = (-0.3, 0.3)
    contrast_range: tuple[float, float] = (0.7, 1.3)
    saturation_range: tuple[float, float] = (0.5, 1.5)
    blur_sigma_range: tuple[float, float] = (0.0, 2.0)
    noise_sigma_range: tuple[float, float] = (0.0, 0.05)
    enabled: bool = True


@dataclass
class PhysicsRandomizationConfig:
    """Parameters for physics domain randomization."""

    mass_range: tuple[float, float] = (0.85, 1.15)
    drag_range: tuple[float, float] = (0.8, 1.2)
    inertia_range: tuple[float, float] = (0.9, 1.1)
    thrust_range: tuple[float, float] = (0.9, 1.1)
    latency_steps_range: tuple[int, int] = (0, 2)
    enabled: bool = True


@dataclass
class SensorNoiseConfig:
    """Parameters for sensor noise injection."""

    position_sigma: float = 0.02
    velocity_sigma: float = 0.05
    orientation_sigma_deg: float = 0.5
    enabled: bool = True


class PhysicsRandomizationWrapper(gym.Wrapper):
    """Randomize physics parameters at each episode reset.

    Modifies the underlying quadrotor dynamics parameters by sampling
    multiplicative factors from the configured ranges.
    """

    def __init__(
        self,
        env: gym.Env,
        config: PhysicsRandomizationConfig | None = None,
    ):
        super().__init__(env)
        self.config = config or PhysicsRandomizationConfig()
        self._rng = np.random.default_rng()
        self._action_buffer: list[np.ndarray] = []
        self._latency_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.config.enabled and hasattr(self.env, '_sim'):
            sim = self.env._sim
            params = sim.params

            mass_factor = self._rng.uniform(*self.config.mass_range)
            params.mass *= mass_factor

            drag_factor = self._rng.uniform(*self.config.drag_range)
            params.drag_coeff *= drag_factor

            inertia_factor = self._rng.uniform(*self.config.inertia_range)
            params.inertia *= inertia_factor

            thrust_factor = self._rng.uniform(*self.config.thrust_range)
            params.max_thrust *= thrust_factor

            sim.hover_thrust = params.mass * params.gravity

            self._latency_steps = self._rng.integers(*self.config.latency_steps_range)
            self._action_buffer = [np.zeros(4) for _ in range(self._latency_steps)]

        return obs, info

    def step(self, action):
        if self.config.enabled and self._latency_steps > 0:
            self._action_buffer.append(np.array(action))
            delayed_action = self._action_buffer.pop(0)
            return self.env.step(delayed_action)
        return self.env.step(action)


class SensorNoiseWrapper(gym.ObservationWrapper):
    """Add Gaussian noise to state observations, simulating real sensor noise."""

    def __init__(
        self,
        env: gym.Env,
        config: SensorNoiseConfig | None = None,
    ):
        super().__init__(env)
        self.config = config or SensorNoiseConfig()
        self._rng = np.random.default_rng()

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if not self.config.enabled:
            return obs

        noisy = obs.copy()

        # Position noise (indices 0:3)
        noisy[0:3] += self._rng.normal(0, self.config.position_sigma, size=3).astype(np.float32)

        # Velocity noise (indices 3:6)
        noisy[3:6] += self._rng.normal(0, self.config.velocity_sigma, size=3).astype(np.float32)

        # Orientation noise (indices 6:9)
        orient_sigma = np.deg2rad(self.config.orientation_sigma_deg)
        noisy[6:9] += self._rng.normal(0, orient_sigma, size=3).astype(np.float32)

        return noisy


class VisualRandomizationWrapper(gym.ObservationWrapper):
    """Apply visual augmentations to image observations.

    Only active when the environment provides image observations
    (e.g., in vision training mode). For state-based training this is a no-op.
    """

    def __init__(
        self,
        env: gym.Env,
        config: VisualRandomizationConfig | None = None,
    ):
        super().__init__(env)
        self.config = config or VisualRandomizationConfig()
        self._rng = np.random.default_rng()

    def observation(self, obs):
        if not self.config.enabled:
            return obs

        if isinstance(obs, np.ndarray) and obs.ndim >= 3 and obs.shape[-1] in (3, 4):
            return self._augment_image(obs)
        return obs

    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply random visual augmentations to an image."""
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        brightness = self._rng.uniform(*self.config.brightness_range)
        img = np.clip(img + brightness, 0, 1)

        contrast = self._rng.uniform(*self.config.contrast_range)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = np.clip((img - mean) * contrast + mean, 0, 1)

        noise_sigma = self._rng.uniform(*self.config.noise_sigma_range)
        if noise_sigma > 0:
            noise = self._rng.normal(0, noise_sigma, size=img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)

        if image.dtype == np.uint8:
            return (img * 255).astype(np.uint8)
        return img
