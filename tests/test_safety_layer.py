"""Tests for the safety layer (Milestone 1.6 + numeric-magnitude enforcement)."""

from pathlib import Path

import pytest

from chess_robot.safety.limits import SafetyLimits, load_limits
from chess_robot.safety.safety_layer import SafetyLayer, SafetyViolationError

_DEFAULT_LIMITS_YAML = (
    Path(__file__).resolve().parents[1] / "configs" / "safety" / "default_limits.yaml"
)

# A valid in-range 6-DOF action (5 arm joints + gripper) for the status checks.
_OK = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _ready_limits() -> SafetyLimits:
    """A fully-configured limits object (all numeric limits set) for SO-101 (6-DOF)."""
    return SafetyLimits(
        max_abs_action=2.0,
        gripper_min=0.0,
        gripper_max=1.0,
        gripper_index=5,
        max_velocity=10.0,
        max_delta=0.5,
        episode_timeout_s=30.0,
        max_observation_age_s=0.2,
        joint_limits_values=((-1.0, 1.0),) * 5,
        workspace_bounds_configured=True,
    )


# --- limits loading -----------------------------------------------------------


def test_default_config_loads_with_tbd_limits_unconfigured() -> None:
    limits = load_limits(_DEFAULT_LIMITS_YAML)
    problems = limits.unconfigured_numeric_limits()
    assert "action_magnitude.max_abs_action" in problems
    assert "gripper_range.min/max/index" in problems
    assert "max_velocity" in problems
    assert "max_observation_age_s" in problems
    assert limits.reject_nan_inf is True


def test_default_config_disables_workspace_bounds() -> None:
    # Workspace bounds need forward kinematics; disabled by config, surfaced not silent.
    layer = SafetyLayer(load_limits(_DEFAULT_LIMITS_YAML))
    assert "workspace_bounds" in layer.disabled_safety_checks()


def test_load_parses_joint_and_gripper(tmp_path: Path) -> None:
    cfg = tmp_path / "limits.yaml"
    cfg.write_text(
        "safety:\n"
        "  joint_limits: {enabled: true, values: [[-1, 1], [-2, 2]]}\n"
        "  gripper_range: {enabled: true, index: 5, min: 0.0, max: 1.0}\n",
        encoding="utf-8",
    )
    limits = load_limits(cfg)
    assert limits.joint_limits_values == ((-1.0, 1.0), (-2.0, 2.0))
    assert limits.gripper_index == 5
    assert limits.joint_limits_configured is True


def test_layer_with_default_config_is_not_hardware_ready() -> None:
    layer = SafetyLayer(load_limits(_DEFAULT_LIMITS_YAML))
    assert layer.is_hardware_ready() is False
    assert layer.hardware_readiness_problems()


def test_fully_configured_limits_are_hardware_ready() -> None:
    layer = SafetyLayer(_ready_limits(), expected_action_dim=6)
    assert layer.is_hardware_ready() is True
    assert layer.hardware_readiness_problems() == []


def test_unset_expected_action_dim_blocks_readiness() -> None:
    layer = SafetyLayer(_ready_limits())
    assert layer.is_hardware_ready() is False
    assert any("expected_action_dim" in p for p in layer.hardware_readiness_problems())


# --- status / shape checks ----------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nan_and_inf_actions(bad: float) -> None:
    result = SafetyLayer(_ready_limits()).check_action([0.0, bad, 0.0, 0.0, 0.0, 0.0])
    assert result.ok is False
    assert any("NaN/inf" in v for v in result.violations)


def test_accepts_valid_action() -> None:
    result = SafetyLayer(_ready_limits()).check_action(_OK)
    assert result.ok is True
    assert result.violations == ()


def test_rejects_wrong_action_shape() -> None:
    result = SafetyLayer(_ready_limits(), expected_action_dim=6).check_action([0.0, 0.0, 0.0])
    assert result.ok is False
    assert any("dim" in v for v in result.violations)


def test_camera_dropout_and_disconnect_are_violations() -> None:
    layer = SafetyLayer(_ready_limits())
    assert layer.check_action(_OK, camera_ok=False).ok is False
    assert layer.check_action(_OK, robot_connected=False).ok is False


# --- stale observation --------------------------------------------------------


def test_stale_observation_violation_when_too_old() -> None:
    layer = SafetyLayer(_ready_limits())  # max_observation_age_s = 0.2
    assert layer.check_action(_OK, observation_age_s=0.05).ok is True
    stale = layer.check_action(_OK, observation_age_s=1.0)
    assert stale.ok is False
    assert any("stale" in v for v in stale.violations)


def test_stale_observation_skipped_when_unconfigured() -> None:
    result = SafetyLayer(SafetyLimits()).check_action([0.0], observation_age_s=1.0)
    assert any("stale_observation" in s for s in result.skipped)
    assert all("stale" not in v for v in result.violations)


# --- episode timeout ----------------------------------------------------------


def test_episode_timeout_enforced_when_configured() -> None:
    layer = SafetyLayer(_ready_limits())  # episode_timeout_s = 30
    assert layer.check_episode_timeout(10.0).ok is True
    assert layer.check_episode_timeout(45.0).ok is False


def test_check_action_includes_timeout_when_elapsed_given() -> None:
    layer = SafetyLayer(_ready_limits(), expected_action_dim=6)
    assert layer.check_action(_OK, elapsed_s=10.0).ok is True
    timed_out = layer.check_action(_OK, elapsed_s=45.0)
    assert timed_out.ok is False
    assert any("timeout" in v for v in timed_out.violations)


def test_enforce_fails_closed_on_timeout() -> None:
    layer = SafetyLayer(_ready_limits(), expected_action_dim=6)
    assert layer.enforce(_OK, elapsed_s=5.0) == _OK
    with pytest.raises(SafetyViolationError):
        layer.enforce(_OK, elapsed_s=45.0)


# --- numeric-magnitude enforcement --------------------------------------------


def test_joint_limit_enforced() -> None:
    limits = SafetyLimits(joint_limits_values=((-1.0, 1.0), (-1.0, 1.0)))
    result = SafetyLayer(limits).check_action([5.0, 0.0])
    assert result.ok is False
    assert any("joint[0]" in v for v in result.violations)


def test_joint_limit_passes_in_range() -> None:
    limits = SafetyLimits(joint_limits_values=((-1.0, 1.0), (-1.0, 1.0)))
    assert SafetyLayer(limits).check_action([0.5, -0.5]).ok is True


def test_action_magnitude_enforced() -> None:
    result = SafetyLayer(SafetyLimits(max_abs_action=1.0)).check_action([0.5, 2.0])
    assert result.ok is False
    assert any("exceeds" in v for v in result.violations)


def test_gripper_range_enforced() -> None:
    limits = SafetyLimits(gripper_min=0.0, gripper_max=1.0, gripper_index=1)
    result = SafetyLayer(limits).check_action([0.0, 5.0])
    assert result.ok is False
    assert any("gripper" in v for v in result.violations)


def test_delta_enforced_with_previous_action() -> None:
    layer = SafetyLayer(SafetyLimits(max_delta=0.1))
    assert layer.check_action([0.05], previous_action=[0.0]).ok is True
    bad = layer.check_action([1.0], previous_action=[0.0])
    assert bad.ok is False
    assert any("delta" in v for v in bad.violations)


def test_delta_skipped_without_previous_action() -> None:
    result = SafetyLayer(SafetyLimits(max_delta=0.1)).check_action([1.0])
    assert any("velocity_delta" in s for s in result.skipped)
    assert result.ok is True


def test_velocity_enforced_with_dt() -> None:
    # delta 0.5 over dt 0.1 == 5.0 > max_velocity 1.0
    result = SafetyLayer(SafetyLimits(max_velocity=1.0)).check_action(
        [0.5], previous_action=[0.0], dt_s=0.1
    )
    assert result.ok is False
    assert any("velocity" in v for v in result.violations)


def test_unconfigured_numeric_checks_are_skipped_not_passed() -> None:
    # With no numeric limits set, a wild action passes but every gap is recorded.
    result = SafetyLayer(SafetyLimits()).check_action([1e9, -1e9, 1e9])
    assert result.ok is True
    assert any("action_magnitude" in s for s in result.skipped)
    assert any("joint_limits" in s for s in result.skipped)


def test_config_disabled_check_is_surfaced_not_silent() -> None:
    layer = SafetyLayer(SafetyLimits(workspace_bounds_enabled=False))
    assert "workspace_bounds" in layer.disabled_safety_checks()
    result = layer.check_action([0.0])
    assert any("workspace_bounds: DISABLED" in s for s in result.skipped)


# --- enforce() fail-closed ----------------------------------------------------


def test_enforce_returns_action_when_safe() -> None:
    layer = SafetyLayer(_ready_limits())
    assert layer.enforce(_OK) is _OK


def test_enforce_raises_on_violation() -> None:
    with pytest.raises(SafetyViolationError):
        SafetyLayer(_ready_limits()).enforce([float("nan")])
