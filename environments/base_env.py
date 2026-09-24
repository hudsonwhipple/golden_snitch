"""Base class for drone racing environments with shared gate tracking logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environments.gate_detection import (
    Gate,
    check_gate_frame_collision,
    check_gate_passage,
    compute_gate_relative_position,
)
from environments.reward_functions.composite_reward import CompositeReward, RewardConfig, RewardInfo
from environments.tracks.track_loader import Track


OBS_DIM = 24
ACT_DIM = 4

# Body radius of the drone, used both to shrink a gate's usable aperture and to decide when
# it has struck the frame. Roughly a Crazyflie 2.x footprint: QuadrotorParams.arm_length is
# 0.0397 m centre-to-motor, plus propeller clearance.
DEFAULT_COLLISION_RADIUS = 0.08


@dataclass
class RacingState:
    """Mutable racing state tracked across steps."""

    next_gate_idx: int = 0
    gates_passed: int = 0
    prev_pos: np.ndarray | None = None
    prev_action: np.ndarray | None = None
    step_count: int = 0
    episode_reward: float = 0.0
    collided: bool = False


class RacingMixin:
    """Shared racing logic for gate tracking and reward computation.

    Mixed into concrete environment classes (PyBullet, AirSim, etc.)
    """

    def _init_racing(
        self,
        track: Track,
        reward_cfg: dict | None = None,
        collision_radius: float = DEFAULT_COLLISION_RADIUS,
    ) -> None:
        self.track = track
        self.reward_fn = CompositeReward(
            RewardConfig(**reward_cfg) if reward_cfg else RewardConfig()
        )
        self.racing_state = RacingState()
        self.max_episode_steps = 1000
        self.collision_radius = collision_radius

    def _reset_racing(self) -> None:
        self.racing_state = RacingState(
            prev_action=np.zeros(ACT_DIM, dtype=np.float32),
        )

    def _get_active_gates(self) -> list[Gate]:
        """Get the active gates for the current curriculum level.

        When a CurriculumWrapper sets ``_active_gate_count``, only the first
        N gates from the track are used.
        """
        n = getattr(self, "_active_gate_count", None)
        if n is not None and 0 < n < len(self.track.gates):
            return self.track.gates[:n]
        return self.track.gates

    def _get_next_gate(self) -> Gate:
        gates = self._get_active_gates()
        idx = self.racing_state.next_gate_idx % len(gates)
        return gates[idx]

    def _get_lookahead_gate(self) -> Gate:
        """Gate after next, for anticipation."""
        gates = self._get_active_gates()
        idx = (self.racing_state.next_gate_idx + 1) % len(gates)
        return gates[idx]

    def _update_gate_progress(self, prev_pos: np.ndarray, curr_pos: np.ndarray) -> bool:
        """Check and update gate passage. Returns True if a gate was passed."""
        gate = self._get_next_gate()
        passed = check_gate_passage(prev_pos, curr_pos, gate, clearance=self.collision_radius)
        if passed:
            self.racing_state.gates_passed += 1
            self.racing_state.next_gate_idx += 1
        return passed

    def _check_gate_collision(self, prev_pos: np.ndarray, curr_pos: np.ndarray) -> bool:
        """True if the swept segment struck any gate's frame.

        Checked against every gate on the track, not ``_get_active_gates()``: a gate the
        curriculum has deactivated for scoring is still a solid object in the world.
        """
        return any(
            check_gate_frame_collision(prev_pos, curr_pos, gate, clearance=self.collision_radius)
            for gate in self.track.gates
        )

    def _compute_racing_reward(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        action: np.ndarray,
        gate_passed: bool,
        collided: bool,
    ) -> RewardInfo:
        prev_pos = self.racing_state.prev_pos
        if prev_pos is None:
            prev_pos = pos
        prev_action = self.racing_state.prev_action
        if prev_action is None:
            prev_action = np.zeros(ACT_DIM, dtype=np.float32)

        gate = self._get_next_gate()
        return self.reward_fn.compute(
            drone_pos=pos,
            prev_drone_pos=prev_pos,
            drone_vel=vel,
            gate_pos=gate.position,
            action=action,
            prev_action=prev_action,
            gate_passed=gate_passed,
            collided=collided,
        )

    def _build_obs(
        self,
        pos: np.ndarray,
        vel_body: np.ndarray,
        rpy: np.ndarray,
        ang_vel: np.ndarray,
    ) -> np.ndarray:
        """Construct the 24-dim observation vector.

        Layout:
            [0:3]   position (x, y, z) world frame
            [3:6]   velocity (vx, vy, vz) body frame
            [6:9]   orientation (roll, pitch, yaw)
            [9:12]  angular velocity (wx, wy, wz) body frame
            [12:16] previous action
            [16:19] relative position to next gate (gate local frame)
            [19:22] relative position to lookahead gate (gate local frame)
            [22]    distance to next gate (normalized)
            [23]    gate progress fraction
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        obs[0:3] = pos
        obs[3:6] = vel_body
        obs[6:9] = rpy
        obs[9:12] = ang_vel

        prev_act = self.racing_state.prev_action
        obs[12:16] = prev_act if prev_act is not None else 0.0

        next_gate = self._get_next_gate()
        gate_rel = compute_gate_relative_position(pos, next_gate)
        obs[16:19] = gate_rel

        lookahead = self._get_lookahead_gate()
        obs[19:22] = compute_gate_relative_position(pos, lookahead)

        dist = np.linalg.norm(pos - next_gate.position)
        obs[22] = dist / 10.0  # normalize by ~track scale

        gates = self._get_active_gates()
        obs[23] = self.racing_state.gates_passed / max(len(gates), 1)

        return obs
