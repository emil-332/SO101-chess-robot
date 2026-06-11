"""Safety layer: validates every real-robot action; blocks unsafe execution.

**Skeleton** All real-robot actions must pass through this
layer. Fail-closed: :meth:`SafetyLayer.check_action` returns a
:class:`SafetyResult` and logs any violation; :meth:`SafetyLayer.enforce` raises
:class:`SafetyViolationError` so a caller cannot continue a rollout silently.

Enforced now: NaN/inf in the action, action shape, stale observations, camera
dropout, robot disconnect, and episode timeout (the time-based checks skip when
their limit is unconfigured). **Numeric-magnitude checks (joint limits, action
magnitude, gripper range, workspace bounds, velocity/delta) are NOT enforced
yet** — they are explicit stubs returning ``None`` ("not enforced") until the
``<TBD>`` limits in ``configs/safety/default_limits.yaml`` are filled and
reviewed (1.6 follow-up). :meth:`SafetyLayer.is_hardware_ready` reports this
(including an unset ``expected_action_dim``) so nothing runs on hardware with a
false sense of safety. Checks turned off in config are surfaced via
:meth:`SafetyLayer.disabled_safety_checks` and logged at construction — a safety
check is never disabled silently. Pass ``elapsed_s`` to :meth:`enforce` to
fail-close on episode timeout in the same call.
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
    ) -> SafetyResult:
        """Check an action (and observation/episode status) against active checks.

        If ``elapsed_s`` is provided, the episode-timeout check is included so a
        single :meth:`enforce` call can also fail-close on timeout.
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

        # Numeric-magnitude checks: not enforced yet (skeleton). Disabled checks
        # and not-yet-enforced stubs are both recorded in `skipped`, never hidden.
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
            ("velocity_delta", True, self._check_velocity_delta),
        )
        for name, enabled, check in numeric_checks:
            if not enabled:
                skipped.append(f"{name}: DISABLED by config")
                continue
            outcome = check(action)
            if outcome is None:
                skipped.append(f"{name}: not enforced yet (limit <TBD>)")
            else:
                violations.extend(outcome)

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
    ) -> Sequence[float]:
        """Return ``action`` if safe, else raise :class:`SafetyViolationError`.

        Pass ``elapsed_s`` to also fail-close on episode timeout in one call.
        """
        result = self.check_action(
            action,
            observation_age_s=observation_age_s,
            camera_ok=camera_ok,
            robot_connected=robot_connected,
            elapsed_s=elapsed_s,
        )
        if not result.ok:
            raise SafetyViolationError(result)
        return action

    # --- Numeric-magnitude checks: TODO(1.6 follow-up) ------------------------
    # Each returns None while unenforced (skeleton). Once implemented, return a
    # list of violation strings (empty list == passed) using configured limits.

    def _check_joint_limits(self, action: Sequence[float]) -> list[str] | None:
        # TODO(1.6 follow-up): enforce per-joint [min, max] once configured.
        return None

    def _check_action_magnitude(self, action: Sequence[float]) -> list[str] | None:
        # TODO(1.6 follow-up): enforce |action_i| <= max_abs_action.
        return None

    def _check_gripper_range(self, action: Sequence[float]) -> list[str] | None:
        # TODO(1.6 follow-up): enforce gripper command within [min, max].
        return None

    def _check_workspace_bounds(self, action: Sequence[float]) -> list[str] | None:
        # TODO(1.6 follow-up): enforce target pose within workspace bounds.
        return None

    def _check_velocity_delta(self, action: Sequence[float]) -> list[str] | None:
        # TODO(1.6 follow-up): enforce max velocity / max per-step delta (needs
        # previous action / state history).
        return None
