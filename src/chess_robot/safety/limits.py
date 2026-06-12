"""Load and represent safety limits from config.

Parses ``configs/safety/default_limits.yaml`` into a :class:`SafetyLimits`.
Numeric limits ship as ``<TBD>`` placeholders and are mapped to ``None``
(unconfigured). :meth:`SafetyLimits.unconfigured_numeric_limits` reports which
enabled numeric checks are still unconfigured — the safety layer uses this to
report hardware readiness. **A check is never silently disabled by leaving its
limit unset; an unset limit means that check is not yet enforceable.**
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TBD_SENTINEL = "<TBD>"


def _is_unset(value: object) -> bool:
    return value is None or value == TBD_SENTINEL


def _as_float(value: object) -> float | None:
    """Coerce a numeric config value to float, or None if unset/non-numeric."""
    if isinstance(value, bool):  # avoid treating True/False as 1/0
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _as_int(value: object) -> int | None:
    """Coerce a config value to int, or None if unset/non-integer."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    return None


def _parse_pairs(value: object) -> tuple[tuple[float, float], ...] | None:
    """Parse a list of ``[min, max]`` pairs, or None if unset/malformed."""
    if not isinstance(value, list) or not value:
        return None
    pairs: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, list | tuple) or len(item) != 2:
            return None
        low, high = _as_float(item[0]), _as_float(item[1])
        if low is None or high is None:
            return None
        pairs.append((low, high))
    return tuple(pairs)


def _section(data: Mapping[Any, Any], key: str) -> Mapping[Any, Any]:
    value = data.get(key)
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class SafetyLimits:
    """Parsed safety limits. ``None`` numeric fields are unconfigured (``<TBD>``)."""

    # Check toggles (a check is enabled unless explicitly turned off in config).
    joint_limits_enabled: bool = True
    action_magnitude_enabled: bool = True
    gripper_range_enabled: bool = True
    workspace_bounds_enabled: bool = True

    # Numeric limits (None == unconfigured / <TBD>).
    max_abs_action: float | None = None
    gripper_min: float | None = None
    gripper_max: float | None = None
    gripper_index: int | None = None
    max_velocity: float | None = None
    max_delta: float | None = None
    episode_timeout_s: float | None = None
    max_observation_age_s: float | None = None
    # Per-joint [min, max] bounds in action order (None == unconfigured).
    joint_limits_values: tuple[tuple[float, float], ...] | None = None
    # Workspace bounds are Cartesian (need forward kinematics); not enforced on
    # joint-space actions. Tracked only as configured-or-not.
    workspace_bounds_configured: bool = False

    @property
    def joint_limits_configured(self) -> bool:
        return self.joint_limits_values is not None

    # Status checks (no numeric config required).
    reject_nan_inf: bool = True
    reject_stale_observations: bool = True
    detect_camera_dropout: bool = True
    detect_robot_disconnect: bool = True

    def unconfigured_numeric_limits(self) -> list[str]:
        """Enabled numeric checks whose limit is still unset (``<TBD>``)."""
        problems: list[str] = []
        if self.joint_limits_enabled and not self.joint_limits_configured:
            problems.append("joint_limits")
        if self.action_magnitude_enabled and self.max_abs_action is None:
            problems.append("action_magnitude.max_abs_action")
        if self.gripper_range_enabled and (
            self.gripper_min is None
            or self.gripper_max is None
            or self.gripper_index is None
        ):
            problems.append("gripper_range.min/max/index")
        if self.workspace_bounds_enabled and not self.workspace_bounds_configured:
            problems.append("workspace_bounds")
        if self.max_velocity is None:
            problems.append("max_velocity")
        if self.max_delta is None:
            problems.append("max_delta")
        if self.episode_timeout_s is None:
            problems.append("episode_timeout_s")
        if self.reject_stale_observations and self.max_observation_age_s is None:
            problems.append("max_observation_age_s")
        return problems

    def disabled_checks(self) -> list[str]:
        """Checks explicitly turned off in config (must be surfaced, not silent)."""
        toggles = (
            ("joint_limits", self.joint_limits_enabled),
            ("action_magnitude", self.action_magnitude_enabled),
            ("gripper_range", self.gripper_range_enabled),
            ("workspace_bounds", self.workspace_bounds_enabled),
            ("reject_nan_inf", self.reject_nan_inf),
            ("reject_stale_observations", self.reject_stale_observations),
            ("detect_camera_dropout", self.detect_camera_dropout),
            ("detect_robot_disconnect", self.detect_robot_disconnect),
        )
        return [name for name, enabled in toggles if not enabled]


def load_limits(path: str | Path) -> SafetyLimits:
    """Load :class:`SafetyLimits` from a YAML config file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    safety = _section(raw, "safety") if isinstance(raw, Mapping) else {}

    joint = _section(safety, "joint_limits")
    action_mag = _section(safety, "action_magnitude")
    gripper = _section(safety, "gripper_range")
    workspace = _section(safety, "workspace_bounds")
    checks = _section(safety, "checks")

    return SafetyLimits(
        joint_limits_enabled=bool(joint.get("enabled", True)),
        action_magnitude_enabled=bool(action_mag.get("enabled", True)),
        gripper_range_enabled=bool(gripper.get("enabled", True)),
        workspace_bounds_enabled=bool(workspace.get("enabled", True)),
        max_abs_action=_as_float(action_mag.get("max_abs_action")),
        gripper_min=_as_float(gripper.get("min")),
        gripper_max=_as_float(gripper.get("max")),
        gripper_index=_as_int(gripper.get("index")),
        max_velocity=_as_float(safety.get("max_velocity")),
        max_delta=_as_float(safety.get("max_delta")),
        episode_timeout_s=_as_float(safety.get("episode_timeout_s")),
        max_observation_age_s=_as_float(checks.get("max_observation_age_s")),
        joint_limits_values=_parse_pairs(joint.get("values")),
        workspace_bounds_configured=not _is_unset(workspace.get("values")),
        reject_nan_inf=bool(checks.get("reject_nan_inf", True)),
        reject_stale_observations=bool(checks.get("reject_stale_observations", True)),
        detect_camera_dropout=bool(checks.get("detect_camera_dropout", True)),
        detect_robot_disconnect=bool(checks.get("detect_robot_disconnect", True)),
    )
