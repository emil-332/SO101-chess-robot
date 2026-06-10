"""Tests for the manipulation evaluation metrics + failure labels"""

import pytest

from chess_robot.data.schema import FailureType
from chess_robot.eval.failure_labels import (
    failure_category,
    is_manipulation_failure,
    is_perception_failure,
)
from chess_robot.eval.metrics import (
    EvalEpisode,
    evaluate_episodes,
    improvement_over_base,
)


def test_success_and_failure_rates() -> None:
    report = evaluate_episodes(
        [
            EvalEpisode(success=True),
            EvalEpisode(success=False, failure_type=FailureType.WRONG_SQUARE),
            EvalEpisode(success=False, failure_type=FailureType.WRONG_PIECE),
            EvalEpisode(success=True),
        ]
    )
    assert report.num_episodes == 4
    assert report.success_rate == pytest.approx(0.5)
    assert report.wrong_square_rate == pytest.approx(0.25)
    assert report.wrong_piece_rate == pytest.approx(0.25)


def test_intervention_and_safety_rates() -> None:
    report = evaluate_episodes(
        [
            EvalEpisode(success=True, intervention_count=0),
            EvalEpisode(success=False, intervention_count=2, safety_violation=True),
            EvalEpisode(success=True, intervention_count=1),
        ]
    )
    assert report.intervention_rate == pytest.approx(2 / 3)
    assert report.number_of_interventions == 3
    assert report.safety_violation_rate == pytest.approx(1 / 3)


def test_optional_metrics_skip_none() -> None:
    report = evaluate_episodes(
        [
            EvalEpisode(success=True, grasp_success=True, target_error_cm=1.0),
            EvalEpisode(success=False, grasp_success=False, target_error_cm=3.0),
            EvalEpisode(success=True),  # grasp/target not labeled -> excluded
        ]
    )
    assert report.grasp_success_rate == pytest.approx(0.5)
    assert report.mean_target_error_cm == pytest.approx(2.0)


def test_capture_metrics_over_capture_episodes() -> None:
    report = evaluate_episodes(
        [
            EvalEpisode(
                success=True,
                is_capture=True,
                capture_success=True,
                removal_success=True,
                placement_success=True,
                capture_split_correct=True,
            ),
            EvalEpisode(
                success=False,
                is_capture=True,
                capture_success=False,
                removal_success=True,
                placement_success=False,
                capture_split_correct=True,
            ),
            EvalEpisode(success=True, is_capture=False, capture_split_correct=True),
        ]
    )
    assert report.num_capture_episodes == 2
    assert report.capture_success_rate == pytest.approx(0.5)
    assert report.removal_submove_success_rate == pytest.approx(1.0)
    assert report.placement_submove_success_rate == pytest.approx(0.5)
    assert report.capture_split_correctness == pytest.approx(1.0)


def test_empty_returns_none_rates() -> None:
    report = evaluate_episodes([])
    assert report.num_episodes == 0
    assert report.success_rate is None
    assert report.wrong_square_rate is None
    assert report.intervention_rate is None


def test_residual_metrics_and_improvement_over_base() -> None:
    base = evaluate_episodes([EvalEpisode(success=False), EvalEpisode(success=True)])
    variant = evaluate_episodes(
        [
            EvalEpisode(success=True, mean_residual_norm=0.2, residual_saturated=False),
            EvalEpisode(success=True, mean_residual_norm=0.4, residual_saturated=True),
        ]
    )
    assert variant.mean_residual_norm == pytest.approx(0.3)
    assert variant.residual_action_saturation_rate == pytest.approx(0.5)
    assert improvement_over_base(base, variant) == pytest.approx(0.5)


def test_failure_label_attribution() -> None:
    assert is_perception_failure(FailureType.PERCEPTION_ERROR)
    assert is_perception_failure(FailureType.CAMERA_FAILURE)
    assert not is_perception_failure(FailureType.BAD_GRASP)
    assert is_manipulation_failure(FailureType.BAD_GRASP)
    assert not is_manipulation_failure(None)
    assert failure_category(FailureType.PERCEPTION_ERROR) == "perception"
    assert failure_category(FailureType.BAD_GRASP) == "manipulation"
    assert failure_category(None) == "none"
