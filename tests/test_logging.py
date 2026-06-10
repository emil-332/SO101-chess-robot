"""Tests for the rollout logger"""

from pathlib import Path

import pytest

from chess_robot.chess.board_state import BoardState, PieceType
from chess_robot.chess.move_resolver import SubmoveRole
from chess_robot.data.schema import FailureType
from chess_robot.utils.logging import RolloutLogger, read_rollouts


def test_logs_episode_with_steps(tmp_path: Path) -> None:
    logger = RolloutLogger(tmp_path)
    logger.start_episode(
        0,
        "move knight from b1 to c3",
        piece_type=PieceType.KNIGHT,
        start_square="b1",
        target_square="c3",
    )
    logger.log_step(
        submove_index=0,
        submove_role=SubmoveRole.MOVE,
        base_action=[0.1, 0.2],
        residual_action=[3.0, 4.0],
        final_action=[3.1, 4.2],
        reward=1.0,
        board_state=BoardState.standard_starting_position(),
    )
    episode = logger.end_episode(success_label=True)

    records = read_rollouts(logger.path)
    assert len(records) == 1
    record = records[0]
    assert record["instruction"] == "move knight from b1 to c3"
    assert record["piece_type"] == "knight"
    assert record["success_label"] is True
    assert len(record["steps"]) == 1

    step = record["steps"][0]
    assert step["submove_role"] == "move"
    assert step["residual_norm"] == pytest.approx(5.0)  # sqrt(9 + 16)
    assert step["board_state_fen"].startswith("rnbqkbnr/")
    assert episode.episode_index == 0


def test_log_step_before_start_raises(tmp_path: Path) -> None:
    logger = RolloutLogger(tmp_path)
    with pytest.raises(RuntimeError):
        logger.log_step(submove_index=0)


def test_appends_episodes_with_labels(tmp_path: Path) -> None:
    logger = RolloutLogger(tmp_path)

    logger.start_episode(0, "move pawn from e2 to e4")
    logger.log_step(
        submove_index=0,
        intervention_flag=True,
        safety_violation_flag=True,
        failure_type=FailureType.BAD_GRASP,
    )
    logger.end_episode(success_label=False, failure_type=FailureType.BAD_GRASP)

    logger.start_episode(1, "move pawn from d2 to d4")
    logger.log_step(submove_index=0)
    logger.end_episode(success_label=True)

    records = read_rollouts(logger.path)
    assert len(records) == 2
    assert records[0]["failure_type"] == "bad_grasp"
    assert records[0]["steps"][0]["intervention_flag"] is True
    assert records[0]["steps"][0]["safety_violation_flag"] is True
    assert records[1]["success_label"] is True
