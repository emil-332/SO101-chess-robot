"""Tests for the safety layer (  1.6, skeleton)."""

from pathlib import Path

import pytest

from chess_robot.safety.limits import SafetyLimits, load_limits
from chess_robot.safety.safety_layer import (
    SafetyLayer,
    SafetyViolationError,
)

_DEFAULT_LIMITS_YAML = (
    Path(__file__).resolve().parents[1] / "configs" / "safety" / "default_limits.yaml"
)


def _ready_limits() -> SafetyLimits:
    """A fully-configured limits object (all numeric limits set)."""
    return SafetyLimits(
        max_abs_action=1.0,
        gripper_min=0.0,
        gripper_max=1.0,
        max_velocity=1.0,
        max_delta=0.1,
        episode_timeout_s=30.0,
        max_observation_age_s=0.2,
        joint_limits_configured=True,
        workspace_bounds_configured=True,
    )


# --- limits loading -----------------------------------------------------------


def test_default_config_loads_with_tbd_limits_unconfigured() -> None:
    limits = load_limits(_DEFAULT_LIMITS_YAML)
    problems = limits.unconfigured_numeric_limits()
    assert "action_magnitude.max_abs_action" in problems
    assert "max_velocity" in problems
    assert "max_observation_age_s" in problems
    # Status checks remain enabled even though numeric limits are <TBD>.
    assert limits.reject_nan_inf is True
    assert limits.detect_camera_dropout is True


def test_layer_with_default_config_is_not_hardware_ready() -> None:
    layer = SafetyLayer(load_limits(_DEFAULT_LIMITS_YAML))
    assert layer.is_hardware_ready() is False
    assert layer.hardware_readiness_problems()  # non-empty


def test_fully_configured_limits_are_hardware_ready() -> None:
    layer = SafetyLayer(_ready_limits(), expected_action_dim=6)
    assert layer.is_hardware_ready() is True
    assert layer.hardware_readiness_problems() == []


def test_unset_expected_action_dim_blocks_readiness() -> None:
    # Configured limits but no expected_action_dim => shape check is inert, so
    # the gate must report it rather than claim "ready".
    layer = SafetyLayer(_ready_limits())
    assert layer.is_hardware_ready() is False
    assert any("expected_action_dim" in p for p in layer.hardware_readiness_problems())


def test_action_shape_skip_is_surfaced_when_dim_unset() -> None:
    result = SafetyLayer(_ready_limits()).check_action([0.0, 0.0])
    assert any("action_shape" in s for s in result.skipped)


# --- enforced checks ----------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nan_and_inf_actions(bad: float) -> None:
    layer = SafetyLayer(_ready_limits())
    result = layer.check_action([0.1, bad, 0.2])
    assert result.ok is False
    assert any("NaN/inf" in v for v in result.violations)


def test_accepts_finite_action() -> None:
    layer = SafetyLayer(_ready_limits())
    result = layer.check_action([0.1, 0.2, 0.3])
    assert result.ok is True
    assert result.violations == ()


def test_rejects_wrong_action_shape() -> None:
    layer = SafetyLayer(_ready_limits(), expected_action_dim=6)
    result = layer.check_action([0.0, 0.0, 0.0])
    assert result.ok is False
    assert any("dim" in v for v in result.violations)


def test_camera_dropout_and_disconnect_are_violations() -> None:
    layer = SafetyLayer(_ready_limits())
    assert layer.check_action([0.0], camera_ok=False).ok is False
    assert layer.check_action([0.0], robot_connected=False).ok is False


# --- stale observation --------------------------------------------------------


def test_stale_observation_violation_when_too_old() -> None:
    layer = SafetyLayer(_ready_limits())  # max_observation_age_s = 0.2
    fresh = layer.check_action([0.0], observation_age_s=0.05)
    stale = layer.check_action([0.0], observation_age_s=1.0)
    assert fresh.ok is True
    assert stale.ok is False
    assert any("stale" in v for v in stale.violations)


def test_stale_observation_skipped_when_unconfigured() -> None:
    limits = SafetyLimits(max_observation_age_s=None)
    result = SafetyLayer(limits).check_action([0.0], observation_age_s=1.0)
    assert any("stale_observation" in s for s in result.skipped)
    # not a violation, because the limit is unconfigured (recorded as skipped)
    assert all("stale" not in v for v in result.violations)


# --- episode timeout ----------------------------------------------------------


def test_episode_timeout_enforced_when_configured() -> None:
    layer = SafetyLayer(_ready_limits())  # episode_timeout_s = 30
    assert layer.check_episode_timeout(10.0).ok is True
    assert layer.check_episode_timeout(45.0).ok is False


def test_episode_timeout_skipped_when_unconfigured() -> None:
    layer = SafetyLayer(SafetyLimits(episode_timeout_s=None))
    result = layer.check_episode_timeout(10_000.0)
    assert result.ok is True
    assert any("episode_timeout" in s for s in result.skipped)


def test_check_action_includes_timeout_when_elapsed_given() -> None:
    layer = SafetyLayer(_ready_limits(), expected_action_dim=1)  # timeout 30s
    assert layer.check_action([0.0], elapsed_s=10.0).ok is True
    timed_out = layer.check_action([0.0], elapsed_s=45.0)
    assert timed_out.ok is False
    assert any("timeout" in v for v in timed_out.violations)


def test_enforce_fails_closed_on_timeout() -> None:
    layer = SafetyLayer(_ready_limits(), expected_action_dim=1)
    assert layer.enforce([0.0], elapsed_s=5.0) == [0.0]
    with pytest.raises(SafetyViolationError):
        layer.enforce([0.0], elapsed_s=45.0)


# --- numeric-magnitude checks are NOT enforced yet (skeleton) ------------------


def test_numeric_magnitude_checks_are_stubbed_not_enforced() -> None:
    # A wildly out-of-range (but finite) action passes, because magnitude/joint/
    # workspace checks are not enforced yet — and that gap is recorded, not hidden.
    layer = SafetyLayer(_ready_limits())
    result = layer.check_action([1e9, -1e9, 1e9])
    assert result.ok is True
    assert any("action_magnitude" in s for s in result.skipped)
    assert any("joint_limits" in s for s in result.skipped)


def test_config_disabled_check_is_surfaced_not_silent() -> None:
    # A check turned off in config must appear in skipped + disabled_safety_checks,
    # never vanish silently.
    limits = SafetyLimits(workspace_bounds_enabled=False)
    layer = SafetyLayer(limits)
    assert "workspace_bounds" in layer.disabled_safety_checks()
    result = layer.check_action([0.0])
    assert any("workspace_bounds: DISABLED" in s for s in result.skipped)


# --- enforce() fail-closed ----------------------------------------------------


def test_enforce_returns_action_when_safe() -> None:
    layer = SafetyLayer(_ready_limits())
    action = [0.1, 0.2]
    assert layer.enforce(action) is action


def test_enforce_raises_on_violation() -> None:
    layer = SafetyLayer(_ready_limits())
    with pytest.raises(SafetyViolationError):
        layer.enforce([float("nan")])
