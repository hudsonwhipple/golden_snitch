"""Multi-gate tracking with confidence-based recovery behavior."""

from __future__ import annotations

import numpy as np

from environments.gate_detection import Gate


class GateTracker:
    """Tracks the current target gate and provides recovery behavior
    when gate detection confidence drops.

    Recovery strategy (from plan Section 11):
    1. Confidence drops below threshold → maintain last velocity for 0.25s
    2. Gradually reduce speed
    3. Increase yaw rate to scan for gate
    4. If no detection for 1.0s → slow hover + 360 scan
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        recovery_timeout: float = 1.0,
        dt: float = 1.0 / 48,
    ):
        self.confidence_threshold = confidence_threshold
        self.recovery_timeout = recovery_timeout
        self.dt = dt

        self._last_confident_gate_pos: np.ndarray | None = None
        self._last_confident_velocity: np.ndarray | None = None
        self._low_confidence_duration: float = 0.0
        self._in_recovery: bool = False

    def reset(self) -> None:
        self._last_confident_gate_pos = None
        self._last_confident_velocity = None
        self._low_confidence_duration = 0.0
        self._in_recovery = False

    def update(
        self,
        gate_relative_pos: np.ndarray | None,
        confidence: float,
        drone_velocity: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Update gate tracking state.

        Parameters
        ----------
        gate_relative_pos : (3,) gate position relative to drone, or None
        confidence : detection confidence [0, 1]
        drone_velocity : (3,) current drone velocity

        Returns
        -------
        estimated_gate_pos : (3,) best estimate of gate-relative position
        is_recovering : whether we're in recovery mode
        """
        if confidence >= self.confidence_threshold and gate_relative_pos is not None:
            self._last_confident_gate_pos = gate_relative_pos.copy()
            self._last_confident_velocity = drone_velocity.copy()
            self._low_confidence_duration = 0.0
            self._in_recovery = False
            return gate_relative_pos, False

        self._low_confidence_duration += self.dt
        self._in_recovery = True

        if self._last_confident_gate_pos is not None:
            estimated = self._last_confident_gate_pos.copy()
            if self._last_confident_velocity is not None:
                estimated -= self._last_confident_velocity * self._low_confidence_duration
            return estimated, True

        return np.zeros(3), True

    @property
    def should_scan(self) -> bool:
        """Whether the drone should enter full scan mode."""
        return self._low_confidence_duration >= self.recovery_timeout

    @property
    def recovery_speed_factor(self) -> float:
        """Speed reduction factor during recovery (1.0 = normal, 0.3 = minimum)."""
        if not self._in_recovery:
            return 1.0
        t = min(self._low_confidence_duration / self.recovery_timeout, 1.0)
        return max(1.0 - t * 0.7, 0.3)
