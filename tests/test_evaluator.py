"""Tests for the evaluation runner"""

from pathlib import Path

import pytest

from chess_robot.chess.move_resolver import SubmoveRole
from chess_robot.data.schema import FailureType
from chess_robot.eval.evaluator import (
    eval_episode_from_rollout,
    evaluate_rollout_file,
    load_eval_config,
)
from chess_robot.utils.logging import RolloutLogger

_EVAL_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "eval" / "chess_eval.yaml"
)


def test_load_eval_config() -> None:
    config = load_eval_config(_EVAL_CONFIG)
    assert config.occupancy_source == "metadata"
    assert "pi05_supervised" in config.compare
    assert config.num_episodes is None  # "<TBD>" -> None


def test_eval_episode_from_rollout_maps_fields() -> None:
    record = {
        "success_label": True,
        "failure_type": None,
        "is_capture": True,
        "steps": [
            {"intervention_flag": True, "safety_violation_flag": False, "residual_norm": 1.0},
            {"intervention_flag": False, "safety_violation_flag": True, "residual_norm": 3.0},
        ],
    }
    episode = eval_episode_from_rollout(record)
    assert episode.success is True
    assert episode.is_capture is True
    assert episode.intervention_count == 1
    assert episode.safety_violation is True
    assert episode.mean_residual_norm == pytest.approx(2.0)


def test_evaluate_rollout_file_end_to_end(tmp_path: Path) -> None:
    logger = RolloutLogger(tmp_path)
    logger.start_episode(0, "move knight from b1 to c3")
    logger.log_step(
        submove_index=0, submove_role=SubmoveRole.MOVE, residual_action=[3.0, 4.0]
    )
    logger.end_episode(success_label=True)
    logger.start_episode(1, "move pawn from e2 to e4")
    logger.log_step(submove_index=0, submove_role=SubmoveRole.MOVE)
    logger.end_episode(success_label=False, failure_type=FailureType.WRONG_SQUARE)

    report = evaluate_rollout_file(logger.path)
    assert report.num_episodes == 2
    assert report.success_rate == pytest.approx(0.5)
    assert report.wrong_square_rate == pytest.approx(0.5)
    assert report.mean_residual_norm == pytest.approx(5.0)  # sqrt(9 + 16)
