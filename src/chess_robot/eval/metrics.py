"""Manipulation / capture / residual evaluation metrics.

Standardized so the three approaches (pi0.5 supervised, +residual RL, +HIL-RL)
compare fairly (see ``docs/evaluation.md``). :func:`evaluate_episodes` aggregates
a list of :class:`EvalEpisode` into a :class:`MetricReport`. Rates with no
applicable data report ``None`` rather than a misleading ``0.0``. This is the
*manipulation* harness; perception is scored separately in
``eval/perception_metrics.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from chess_robot.data.schema import FailureType


@dataclass
class EvalEpisode:
    """Outcome of one evaluation rollout.

    ``success`` and ``safety_violation`` are always known; the rest default to
    ``None`` / 0 when not labeled (and are then excluded from their metric).
    """

    success: bool
    failure_type: FailureType | None = None
    is_capture: bool = False
    grasp_success: bool | None = None
    dropped: bool | None = None
    release_failure: bool | None = None
    collision: bool | None = None
    intervention_count: int = 0
    target_error_cm: float | None = None
    episode_time_s: float | None = None
    safety_violation: bool = False
    # capture / submove
    capture_success: bool | None = None
    removal_success: bool | None = None
    placement_success: bool | None = None
    capture_split_correct: bool | None = None
    # residual learning
    mean_residual_norm: float | None = None
    residual_saturated: bool | None = None


def _bool_rate(values: Iterable[bool | None]) -> float | None:
    applicable = [value for value in values if value is not None]
    if not applicable:
        return None
    return sum(1 for value in applicable if value) / len(applicable)


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _failure_rate(
    episodes: list[EvalEpisode], failure_type: FailureType
) -> float | None:
    if not episodes:
        return None
    return sum(1 for e in episodes if e.failure_type is failure_type) / len(episodes)


@dataclass
class MetricReport:
    """Aggregated manipulation/capture/residual metrics (``None`` == no data)."""

    num_episodes: int
    num_capture_episodes: int
    success_rate: float | None
    wrong_square_rate: float | None
    wrong_piece_rate: float | None
    mean_target_error_cm: float | None
    grasp_success_rate: float | None
    drop_rate: float | None
    release_failure_rate: float | None
    collision_rate: float | None
    intervention_rate: float | None
    number_of_interventions: int
    mean_episode_time_s: float | None
    safety_violation_rate: float | None
    capture_success_rate: float | None
    removal_submove_success_rate: float | None
    placement_submove_success_rate: float | None
    capture_split_correctness: float | None
    mean_residual_norm: float | None
    residual_action_saturation_rate: float | None


def evaluate_episodes(episodes: Iterable[EvalEpisode]) -> MetricReport:
    """Aggregate evaluation episodes into a :class:`MetricReport`."""
    items = list(episodes)
    total = len(items)
    captures = [e for e in items if e.is_capture]

    intervention_rate = (
        sum(1 for e in items if e.intervention_count > 0) / total if total else None
    )

    return MetricReport(
        num_episodes=total,
        num_capture_episodes=len(captures),
        success_rate=_bool_rate(e.success for e in items),
        wrong_square_rate=_failure_rate(items, FailureType.WRONG_SQUARE),
        wrong_piece_rate=_failure_rate(items, FailureType.WRONG_PIECE),
        mean_target_error_cm=_mean(e.target_error_cm for e in items),
        grasp_success_rate=_bool_rate(e.grasp_success for e in items),
        drop_rate=_bool_rate(e.dropped for e in items),
        release_failure_rate=_bool_rate(e.release_failure for e in items),
        collision_rate=_bool_rate(e.collision for e in items),
        intervention_rate=intervention_rate,
        number_of_interventions=sum(e.intervention_count for e in items),
        mean_episode_time_s=_mean(e.episode_time_s for e in items),
        safety_violation_rate=_bool_rate(e.safety_violation for e in items),
        capture_success_rate=_bool_rate(e.capture_success for e in captures),
        removal_submove_success_rate=_bool_rate(e.removal_success for e in captures),
        placement_submove_success_rate=_bool_rate(
            e.placement_success for e in captures
        ),
        capture_split_correctness=_bool_rate(e.capture_split_correct for e in items),
        mean_residual_norm=_mean(e.mean_residual_norm for e in items),
        residual_action_saturation_rate=_bool_rate(
            e.residual_saturated for e in items
        ),
    )


def improvement_over_base(base: MetricReport, variant: MetricReport) -> float | None:
    """Variant success-rate minus base success-rate (``None`` if either missing)."""
    if base.success_rate is None or variant.success_rate is None:
        return None
    return variant.success_rate - base.success_rate
