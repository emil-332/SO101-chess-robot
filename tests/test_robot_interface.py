"""Tests for the robot interface: action schema, observations, mock + gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chess_robot.robot.actions import (
    action_to_motor_dict,
    load_action_schema,
    so101_action_schema,
)
from chess_robot.robot.observations import STATE_KEY, build_robot_observation
from chess_robot.robot.so101_interface import (
    MockRobot,
    RobotNotReadyError,
    SO101Robot,
)
from chess_robot.safety.limits import SafetyLimits
from chess_robot.safety.safety_layer import SafetyLayer, SafetyViolationError

_SO101_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "robot" / "so101.yaml"
)


def _safety() -> SafetyLayer:
    return SafetyLayer(SafetyLimits())  # minimal: status checks on, numeric unset


# --- action schema ------------------------------------------------------------


def test_so101_schema_layout() -> None:
    schema = so101_action_schema()
    assert schema.action_dim == 6
    assert schema.gripper_index == 5
    assert schema.arm_joint_indices == (0, 1, 2, 3, 4)


def test_load_action_schema_from_config() -> None:
    schema = load_action_schema(_SO101_CONFIG)
    assert schema.action_dim == 6
    assert schema.joint_names[-1] == "gripper"
    assert schema.gripper_index == 5


def test_action_to_motor_dict() -> None:
    motors = action_to_motor_dict([1, 2, 3, 4, 5, 6], so101_action_schema())
    assert motors["shoulder_pan"] == 1.0
    assert motors["gripper"] == 6.0


# --- observations -------------------------------------------------------------


def test_build_robot_observation() -> None:
    frames = {"observation.images.side": np.zeros((2, 2, 3), dtype=np.uint8)}
    obs = build_robot_observation([0, 1, 2, 3, 4, 5], frames, schema=so101_action_schema())
    assert obs[STATE_KEY].shape == (6,)
    assert "observation.images.side" in obs


def test_build_observation_rejects_wrong_state_dim() -> None:
    with pytest.raises(ValueError):
        build_robot_observation([0, 1, 2], {}, schema=so101_action_schema())


# --- mock robot: safety routing -----------------------------------------------


def test_mock_robot_records_safe_action() -> None:
    robot = MockRobot(_safety(), so101_action_schema())
    robot.connect()
    robot.send_action([0.0] * 6)
    assert robot.sent_actions == [[0.0] * 6]
    assert robot.is_connected


def test_mock_robot_send_rejects_nan() -> None:
    robot = MockRobot(_safety(), so101_action_schema())
    robot.connect()
    with pytest.raises(SafetyViolationError):
        robot.send_action([float("nan")] * 6)
    assert robot.sent_actions == []  # nothing reached the hardware


def test_send_action_blocked_when_disconnected() -> None:
    robot = MockRobot(_safety(), so101_action_schema())  # not connected
    with pytest.raises(SafetyViolationError):
        robot.send_action([0.0] * 6)


def test_mock_read_observation_follows_commanded_action() -> None:
    frames = {"observation.images.side": np.zeros((2, 2, 3), dtype=np.uint8)}
    robot = MockRobot(_safety(), so101_action_schema(), frames=frames)
    robot.connect()
    robot.send_action([0.5] * 6)
    obs = robot.read_observation()
    assert obs[STATE_KEY].tolist() == [0.5] * 6


# --- real robot: hardware gate ------------------------------------------------


def test_so101_connect_refuses_when_safety_not_ready() -> None:
    robot = SO101Robot(_safety(), so101_action_schema(), follower_port="<PORT>")
    with pytest.raises(RobotNotReadyError):
        robot.connect()  # gate closed: limits unconfigured -> never touches LeRobot
