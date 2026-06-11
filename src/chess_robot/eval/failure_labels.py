"""Stable failure-type labels + perception/manipulation attribution.

``FailureType`` is defined once in ``data.schema`` (the dataset's source of truth)
and re-exported here. Evaluation must be able to attribute a failed rollout to
**perception** vs **manipulation** (see ``docs/evaluation.md``).
"""

from __future__ import annotations

from chess_robot.data.schema import FailureType

# Failures caused by the perception preprocessing rather than the policy/robot.
PERCEPTION_FAILURES: frozenset[FailureType] = frozenset(
    {FailureType.PERCEPTION_ERROR, FailureType.CAMERA_FAILURE}
)


def is_perception_failure(failure_type: FailureType | None) -> bool:
    return failure_type in PERCEPTION_FAILURES


def is_manipulation_failure(failure_type: FailureType | None) -> bool:
    return failure_type is not None and failure_type not in PERCEPTION_FAILURES


def failure_category(failure_type: FailureType | None) -> str:
    """``"none"`` | ``"perception"`` | ``"manipulation"``."""
    if failure_type is None:
        return "none"
    return "perception" if failure_type in PERCEPTION_FAILURES else "manipulation"


__all__ = [
    "PERCEPTION_FAILURES",
    "FailureType",
    "failure_category",
    "is_manipulation_failure",
    "is_perception_failure",
]
