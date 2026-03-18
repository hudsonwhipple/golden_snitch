"""Tests for PID controller."""

import numpy as np
import pytest

from control.pid_controller import PIDController, PIDGains, CascadedPIDController


class TestPIDController:
    def test_zero_error_zero_output(self):
        pid = PIDController(PIDGains(kp=1.0, ki=0.0, kd=0.0))
        output = pid.compute(0.0, dt=0.01)
        assert output == 0.0

    def test_proportional_response(self):
        pid = PIDController(PIDGains(kp=2.0, ki=0.0, kd=0.0), output_limit=100.0, rate_limit=0.0)
        output = pid.compute(1.0, dt=0.01)
        assert output == pytest.approx(2.0)

    def test_output_clamped(self):
        pid = PIDController(PIDGains(kp=10.0, ki=0.0, kd=0.0), output_limit=1.0)
        output = pid.compute(5.0, dt=0.01)
        assert abs(output) <= 1.0

    def test_integral_accumulates(self):
        pid = PIDController(PIDGains(kp=0.0, ki=1.0, kd=0.0), output_limit=100.0)
        for _ in range(10):
            output = pid.compute(1.0, dt=0.1)
        assert output > 0.5

    def test_anti_windup(self):
        pid = PIDController(
            PIDGains(kp=0.0, ki=10.0, kd=0.0),
            output_limit=100.0,
            integral_limit=0.5,
        )
        for _ in range(100):
            pid.compute(1.0, dt=0.1)
        assert abs(pid._integral) <= 0.5

    def test_reset(self):
        pid = PIDController(PIDGains(kp=1.0, ki=1.0, kd=1.0))
        pid.compute(1.0, dt=0.01)
        pid.reset()
        assert pid._integral == 0.0
        assert pid._prev_error == 0.0
        assert pid._initialized is False


class TestCascadedPIDController:
    def test_zero_error_zero_output(self):
        cpid = CascadedPIDController()
        output = cpid.compute(
            vel_target=np.zeros(3),
            vel_current=np.zeros(3),
            yaw_rate_target=0.0,
            yaw_rate_current=0.0,
            dt=0.01,
        )
        np.testing.assert_allclose(output, np.zeros(4), atol=1e-10)

    def test_output_shape(self):
        cpid = CascadedPIDController()
        output = cpid.compute(
            vel_target=np.array([1.0, 0.0, 0.5]),
            vel_current=np.zeros(3),
            yaw_rate_target=0.5,
            yaw_rate_current=0.0,
            dt=0.02,
        )
        assert output.shape == (4,)

    def test_output_bounded(self):
        cpid = CascadedPIDController()
        output = cpid.compute(
            vel_target=np.array([10.0, -10.0, 10.0]),
            vel_current=np.zeros(3),
            yaw_rate_target=5.0,
            yaw_rate_current=0.0,
            dt=0.02,
        )
        assert np.all(np.abs(output) <= 1.0)

    def test_reset(self):
        cpid = CascadedPIDController()
        cpid.compute(np.ones(3), np.zeros(3), 1.0, 0.0, 0.02)
        cpid.reset()
        output = cpid.compute(np.zeros(3), np.zeros(3), 0.0, 0.0, 0.02)
        np.testing.assert_allclose(output, np.zeros(4), atol=1e-10)
