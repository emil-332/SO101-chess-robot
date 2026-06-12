"""Safety layer: validates every real-robot action; blocks unsafe execution.

All real-robot actions must pass through this
layer. Fail-closed: :meth:`SafetyLayer.check_action` returns a
:class:`SafetyResult` and logs any violation; :meth:`SafetyLayer.enforce` raises
:class:`SafetyViolationError` so a caller cannot continue a rollout silently.

Enforced: NaN/inf in the action, action shape, stale observations, camera
dropout, robot disconnect, episode timeout, and the numeric-magnitude checks
(per-joint limits, action magnitude, gripper range, per-step delta / velocity).
Each numeric check enforces only when its limit is configured; while a limit is
``<TBD>`` the check is recorded in ``skipped`` (never silently passed) and
:meth:`SafetyLayer.is_hardware_ready` reports it, so nothing runs on hardware
with a false sense of safety. Workspace bounds are Cartesian and need forward
kinematics, so they are not enforced on joint-space actions — disable that check
in config and rely on the per-joint limits, which bound the reachable space.

Checks turned off in config are surfaced via
:meth:`SafetyLayer.disabled_safety_checks` and logged at construction — a safety
check is never disabled silently. Pass ``elapsed_s`` to :meth:`enforce` to
fail-close on episode timeout, and ``previous_action`` (+ optional ``dt_s``) to
enforce the per-step delta / velocity limits in the same call.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

from chess_robot.safety.limits import SafetyLimits

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SafetyResult:
    """Outcome of a safety check.

    ``ok`` is True only if no *active* check was violated. ``skipped`` lists
    checks that did not run (unconfigured limit, or a not-yet-enforced stub) so
    the gap is never silent.
    """

    ok: bool
    violations: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()


class SafetyViolationError(RuntimeError):
    """Raised by :meth:`SafetyLayer.enforce` when an action is unsafe."""

    def __init__(self, result: SafetyResult) -> None:
        super().__init__("; ".join(result.violations) or "safety violation")
        self.result = result


def _has_nonfinite(action: Sequence[float]) -> bool:
    try:
        return any(not math.isfinite(float(a)) for a in action)
    except (TypeError, ValueError):
        return True  # non-numeric content is unsafe


class SafetyLayer:
    """Validate actions against configured limits before execution."""

    def __init__(
        self, limits: SafetyLimits, *, expected_action_dim: int | None = None
    ) -> None:
        self._limits = limits
        self._expected_action_dim = expected_action_dim
        disabled = limits.disabled_checks()
        if disabled:
            logger.warning(
                "Safety checks DISABLED by config: %s", ", ".join(disabled)
            )

    @property
    def limits(self) -> SafetyLimits:
        return self._limits

    def disabled_safety_checks(self) -> list[str]:
        """Checks turned off in config (deployment should log/confirm these)."""
        return self._limits.disabled_checks()

    def hardware_readiness_problems(self) -> list[str]:
        """Reasons the layer is not ready to gate real-robot execution.

        Covers enabled numeric checks that are unconfigured (``<TBD>``) and an
        unset ``expected_action_dim`` (without it, action-shape validation is
        inert). Non-empty means do NOT run on hardware until resolved.
        """
        problems = list(self._limits.unconfigured_numeric_limits())
        if self._expected_action_dim is None:
            problems.append("expected_action_dim (action shape not validated)")
        return problems

    def is_hardware_ready(self) -> bool:
        return not self.hardware_readiness_problems()

    def check_action(
        self,
        action: Sequence[float],
        *,
        observation_age_s: float | None = None,
        camera_ok: bool = True,
        robot_connected: bool = True,
        elapsed_s: float | None = None,
        previous_action: Sequence[float] | None = None,
        dt_s: float | None = None,
    ) -> SafetyResult:
        """Check an action (and observation/episode status) against active checks.

        ``elapsed_s`` includes the episode-timeout check; ``previous_action`` (and
        optional ``dt_s``) include the per-step delta / velocity checks, so a single
        :meth:`enforce` call can fail-close on all of them.
        """
        violations: list[str] = []
        skipped: list[str] = []

        if self._limits.reject_nan_inf and _has_nonfinite(action):
            violations.append("action contains NaN/inf")

        if self._expected_action_dim is None:
            skipped.append("action_shape: expected_action_dim unset")
        elif len(action) != self._expected_action_dim:
            violations.append(
                f"action dim {len(action)} != expected {self._expected_action_dim}"
            )

        if self._limits.reject_stale_observations:
            if self._limits.max_observation_age_s is None:
                skipped.append("stale_observation: max_observation_age_s unconfigured")
            elif observation_age_s is None:
                skipped.append("stale_observation: no observation_age_s provided")
            elif observation_age_s > self._limits.max_observation_age_s:
                violations.append(
                    f"stale observation: age {observation_age_s}s > "
                    f"{self._limits.max_observation_age_s}s"
                )

        if self._limits.detect_camera_dropout and not camera_ok:
            violations.append("camera dropout")
        if self._limits.detect_robot_disconnect and not robot_connected:
            violations.append("robot disconnected")

        if elapsed_s is not None:
            timeout_violations, timeout_skipped = self._timeout_outcome(elapsed_s)
            violations.extend(timeout_violations)
            skipped.extend(timeout_skipped)

        # Numeric-magnitude checks. Each enforces only when its limit is set; an
        # unconfigured limit or a config-disabled check is recorded in `skipped`,
        # never silently passed.
        numeric_checks = (
            ("joint_limits", self._limits.joint_limits_enabled, self._check_joint_limits),
            (
                "action_magnitude",
                self._limits.action_magnitude_enabled,
                self._check_action_magnitude,
            ),
            (
                "gripper_range",
                self._limits.gripper_range_enabled,
                self._check_gripper_range,
            ),
            (
                "workspace_bounds",
                self._limits.workspace_bounds_enabled,
                self._check_workspace_bounds,
            ),
        )
        for name, enabled, check in numeric_checks:
            if not enabled:
                skipped.append(f"{name}: DISABLED by config")
                continue
            outcome = check(action)
            if outcome is None:
                skipped.append(f"{name}: not enforced (limit <TBD>)")
            else:
                violations.extend(outcome)

        velocity_violations, velocity_skipped = self._velocity_outcome(
            action, previous_action, dt_s
        )
        violations.extend(velocity_violations)
        skipped.extend(velocity_skipped)

        result = SafetyResult(
            ok=not violations, violations=tuple(violations), skipped=tuple(skipped)
        )
        if not result.ok:
            logger.warning(
                "SafetyLayer blocked action: %s", "; ".join(result.violations)
            )
        return result

    def _timeout_outcome(self, elapsed_s: float) -> tuple[list[str], list[str]]:
        """Return (violations, skipped) for the episode-timeout check."""
        timeout = self._limits.episode_timeout_s
        if timeout is None:
            return [], ["episode_timeout: unconfigured"]
        if elapsed_s > timeout:
            return [f"episode timeout: {elapsed_s}s > {timeout}s"], []
        return [], []

    def check_episode_timeout(self, elapsed_s: float) -> SafetyResult:
        """Check elapsed episode time against the configured timeout."""
        violations, skipped = self._timeout_outcome(elapsed_s)
        result = SafetyResult(
            ok=not violations, violations=tuple(violations), skipped=tuple(skipped)
        )
        if not result.ok:
            logger.warning(
                "SafetyLayer episode timeout: %ss > %ss",
                elapsed_s,
                self._limits.episode_timeout_s,
            )
        return result

    def enforce(
        self,
        action: Sequence[float],
        *,
        observation_age_s: float | None = None,
        camera_ok: bool = True,
        robot_connected: bool = True,
        elapsed_s: float | None = None,
        previous_action: Sequence[float] | None = None,
        dt_s: float | None = None,
    ) -> Sequence[float]:
        """Return ``action`` if safe, else raise :class:`SafetyViolationError`.

        Pass ``elapsed_s`` to also fail-close on episode timeout, and
        ``previous_action`` (+ optional ``dt_s``) on the delta / velocity limits.
        """
        result = self.check_action(
            action,
            observation_age_s=observation_age_s,
            camera_ok=camera_ok,
            robot_connected=robot_connected,
            elapsed_s=elapsed_s,
            previous_action=previous_action,
            dt_s=dt_s,
        )
        if not result.ok:
            raise SafetyViolationError(result)
        return action

    # --- Numeric-magnitude checks --------------------------------------------
    # Each returns a list of violation strings (empty == passed), or None when its
    # limit is unconfigured (recorded as skipped, not silently passed).

    def _check_joint_limits(self, action: Sequence[float]) -> list[str] | None:
        bounds = self._limits.joint_limits_values
        if bounds is None:
            return None
        violations: list[str] = []
        for index, (low, high) in enumerate(bounds):
            if index >= len(action):
                violations.append(f"joint_limits: action has no index {index}")
                continue
            value = float(action[index])
            if value < low or value > high:
                violations.append(f"joint[{index}]={value} outside [{low}, {high}]")
        return violations

    def _check_action_magnitude(self, action: Sequence[float]) -> list[str] | None:
        limit = self._limits.max_abs_action
        if limit is None:
            return None
        return [
            f"action[{i}]={float(a)} exceeds |{limit}|"
            for i, a in enumerate(action)
            if abs(float(a)) > limit
        ]

    def _check_gripper_range(self, action: Sequence[float]) -> list[str] | None:
        low, high, index = (
            self._limits.gripper_min,
            self._limits.gripper_max,
            self._limits.gripper_index,
        )
        if low is None or high is None or index is None:
            return None
        if index >= len(action):
            return [f"gripper_range: action has no index {index}"]
        value = float(action[index])
        if value < low or value > high:
            return [f"gripper[{index}]={value} outside [{low}, {high}]"]
        return []

    def _check_workspace_bounds(self, action: Sequence[float]) -> list[str] | None:
        # Cartesian bound; needs forward kinematics. Not enforceable on a
        # joint-space action — disable in config and rely on joint_limits.
        return None

    def _velocity_outcome(
        self,
        action: Sequence[float],
        previous_action: Sequence[float] | None,
        dt_s: float | None,
    ) -> tuple[list[str], list[str]]:
        """Per-step delta (and velocity, if ``dt_s`` given) against the limits."""
        max_delta = self._limits.max_delta
        max_velocity = self._limits.max_velocity
        if max_delta is None and max_velocity is None:
            return [], ["velocity_delta: max_delta/max_velocity unconfigured"]
        if previous_action is None:
            return [], ["velocity_delta: no previous_action provided"]
        if len(previous_action) != len(action):
            return ["velocity_delta: previous_action shape mismatch"], []

        deltas = [abs(float(a) - float(p)) for a, p in zip(action, previous_action, strict=True)]
        violations: list[str] = []
        skipped: list[str] = []
        if max_delta is not None:
            violations += [
                f"delta[{i}]={d} > max_delta {max_delta}"
                for i, d in enumerate(deltas)
                if d > max_delta
            ]
        else:
            skipped.append("max_delta unconfigured")
        if max_velocity is not None:
            if dt_s is None or dt_s <= 0:
                skipped.append("max_velocity: no dt_s provided")
            else:
                violations += [
                    f"velocity[{i}]={d / dt_s} > max_velocity {max_velocity}"
                    for i, d in enumerate(deltas)
                    if d / dt_s > max_velocity
                ]
        return violations, skipped
