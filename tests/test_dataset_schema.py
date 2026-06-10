"""Tests for the dataset schema and validation"""

from typing import Any

import pytest

from chess_robot.chess.board_state import BoardState, PieceType
from chess_robot.chess.move_resolver import SubmoveRole
from chess_robot.data.schema import (
    CAMERA_KEYS,
    REQUIRED_FIELDS,
    EpisodeRecord,
    FailureType,
    Feedback,
)
from chess_robot.data.validation import (
    find_missing_required_fields,
    is_valid_episode_record,
    validate_episode_record,
    validate_feedback,
)

_DOCUMENTED_FAILURE_TYPES = {
    "bad_grasp",
    "missed_piece",
    "dropped_piece",
    "wrong_square",
    "wrong_piece",
    "release_failure",
    "collision",
    "unsafe_motion",
    "capture_removal_failure",
    "timeout",
    "perception_error",
    "camera_failure",
    "unknown",
}


def _record(**overrides: Any) -> EpisodeRecord:
    base: dict[str, Any] = {
        "instruction": "move knight from b1 to c3",
        "piece_type": PieceType.KNIGHT,
        "start_square": "b1",
        "target_square": "c3",
    }
    base.update(overrides)
    return EpisodeRecord(**base)


def _complete_row() -> dict[str, Any]:
    return {
        "instruction": "move knight from b1 to c3",
        "observation.images.overhead": object(),
        "observation.state": object(),
        "action": object(),
        "timestamp": 0.0,
        "episode_index": 0,
        "piece_type": "knight",
        "start_square": "b1",
        "target_square": "c3",
    }


# --- required fields ----------------------------------------------------------


def test_complete_row_has_no_missing_required_fields() -> None:
    assert find_missing_required_fields(_complete_row()) == []


def test_missing_required_fields_are_reported() -> None:
    row = _complete_row()
    del row["instruction"]
    del row["action"]
    del row["observation.images.overhead"]  # remove the only camera
    missing = find_missing_required_fields(row)
    assert "instruction" in missing
    assert "action" in missing
    assert any("observation.images" in m for m in missing)


def test_required_fields_constant_includes_instruction() -> None:
    assert "instruction" in REQUIRED_FIELDS


# --- well-formed records validate clean --------------------------------------


def test_non_capture_record_is_valid() -> None:
    record = _record(board_state=BoardState.standard_starting_position())
    assert validate_episode_record(record) == []
    assert is_valid_episode_record(record)


def test_capture_submoves_are_valid() -> None:
    remove = _record(
        is_capture=True,
        submove_index=0,
        submove_role=SubmoveRole.REMOVE,
        captured_piece_type=PieceType.PAWN,
    )
    place = _record(
        is_capture=True,
        submove_index=1,
        submove_role=SubmoveRole.PLACE,
        captured_piece_type=PieceType.PAWN,
    )
    assert validate_episode_record(remove) == []
    assert validate_episode_record(place) == []


# --- instruction / metadata match --------------------------------------------


def test_instruction_metadata_mismatch_is_flagged() -> None:
    record = _record(piece_type=PieceType.QUEEN)  # instruction says knight
    problems = validate_episode_record(record)
    assert any("piece" in p for p in problems)


def test_unparseable_instruction_is_flagged() -> None:
    record = _record(instruction="please move the horse")
    assert any("does not parse" in p for p in validate_episode_record(record))


def test_empty_instruction_is_flagged() -> None:
    record = _record(instruction="   ")
    assert any("mandatory" in p for p in validate_episode_record(record))


# --- submove field consistency -----------------------------------------------


def test_move_role_with_capture_is_flagged() -> None:
    record = _record(
        is_capture=True, submove_role=SubmoveRole.MOVE, captured_piece_type=PieceType.PAWN
    )
    assert any("move" in p for p in validate_episode_record(record))


def test_place_role_requires_index_one() -> None:
    record = _record(
        is_capture=True,
        submove_index=0,  # wrong: place must be index 1
        submove_role=SubmoveRole.PLACE,
        captured_piece_type=PieceType.PAWN,
    )
    assert any("place" in p and "index 1" in p for p in validate_episode_record(record))


def test_remove_role_requires_capture() -> None:
    record = _record(submove_role=SubmoveRole.REMOVE, submove_index=0)
    assert any("remove" in p for p in validate_episode_record(record))


def test_negative_submove_index_is_flagged() -> None:
    record = _record(submove_index=-1)
    assert any("submove_index" in p for p in validate_episode_record(record))


def test_capture_without_captured_piece_is_flagged() -> None:
    record = _record(
        is_capture=True, submove_role=SubmoveRole.REMOVE, captured_piece_type=None
    )
    assert any("captured_piece_type" in p for p in validate_episode_record(record))


# --- labels: success + failure types -----------------------------------------


def test_failure_type_names_match_documented_set() -> None:
    assert {f.value for f in FailureType} == _DOCUMENTED_FAILURE_TYPES


@pytest.mark.parametrize("success", [True, False, None])
def test_success_label_values_are_accepted(success: bool | None) -> None:
    record = _record(success_label=success)
    assert validate_episode_record(record) == []


def test_success_true_with_failure_type_is_flagged() -> None:
    record = _record(success_label=True, failure_type=FailureType.WRONG_SQUARE)
    assert any("failure_type" in p for p in validate_episode_record(record))


# --- Feedback -----------------------------------------------------------------


def test_feedback_defaults_and_validation() -> None:
    feedback = Feedback(
        success_label=None,
        scalar_reward=None,
        intervention_flag=False,
        corrected_action=None,
        safety_violation_flag=False,
        failure_type=None,
    )
    assert feedback.submove_index == 0
    assert feedback.preference_label is None
    assert validate_feedback(feedback) == []


def test_feedback_negative_submove_index_is_flagged() -> None:
    feedback = Feedback(
        success_label=False,
        scalar_reward=-1.0,
        intervention_flag=True,
        corrected_action=None,
        safety_violation_flag=False,
        failure_type=FailureType.BAD_GRASP,
        submove_index=-2,
    )
    assert any("submove_index" in p for p in validate_feedback(feedback))


def test_camera_keys_are_stable() -> None:
    assert CAMERA_KEYS == (
        "observation.images.overhead",
        "observation.images.side",
        "observation.images.wrist",
    )
