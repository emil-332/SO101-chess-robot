"""Tests for the pi0.5 inference wrappers"""

import numpy as np
import pytest

from chess_robot.policies.pi05_policy import MockPi05Policy, RemotePi05Policy


def test_mock_policy_returns_action_of_right_dim() -> None:
    policy = MockPi05Policy(6)
    action = policy.select_action(
        {"observation.state": np.zeros(6)}, "move knight from b1 to c3"
    )
    assert action.shape == (6,)
    assert np.all(action == 0)


def test_remote_payload_sends_state_no_images_by_default() -> None:
    policy = RemotePi05Policy("http://localhost:8000", 6)
    payload = policy.build_payload(
        {
            "observation.state": [1.0, 2.0],
            "observation.images.overhead": np.zeros((2, 2, 3)),
        },
        "instr",
    )
    assert payload["instruction"] == "instr"
    assert payload["state"] == [1.0, 2.0]
    assert "images" not in payload


def test_remote_payload_with_image_encoder() -> None:
    policy = RemotePi05Policy(
        "http://localhost:8000", 6, image_encoder=lambda img: list(img.shape)
    )
    payload = policy.build_payload(
        {"observation.images.overhead": np.zeros((4, 5, 3))}, "instr"
    )
    assert payload["images"]["observation.images.overhead"] == [4, 5, 3]


def test_remote_parse_action_validates_shape() -> None:
    policy = RemotePi05Policy("http://localhost:8000", 3)
    assert np.allclose(policy.parse_action({"action": [1, 2, 3]}), [1, 2, 3])
    with pytest.raises(ValueError):
        policy.parse_action({"action": [1, 2]})
