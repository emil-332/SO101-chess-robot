"""Dataset schema and curation validation.

Two layers:

- :func:`find_missing_required_fields` checks a raw LeRobot row (a mapping) for
  the required keys (instruction, observation/action data, move metadata, and at
  least one camera image).
- :func:`validate_episode_record` checks a structured :class:`EpisodeRecord` for
  semantic consistency: the instruction must match the move metadata, squares
  must be valid, and the capture/submove/label fields must be mutually
  consistent.

Each validator returns a list of human-readable problems (empty == valid) so
callers (and the data-agent) can report all issues at once.
"""

from __future__ import annotations

from collections.abc import Mapping

from chess_robot.chess.board_state import Square
from chess_robot.chess.command_parser import CommandParseError, parse_command
from chess_robot.data.schema import (
    CAMERA_KEY_PREFIX,
    REQUIRED_FIELDS,
    EpisodeRecord,
    Feedback,
    SubmoveRole,
)


def find_missing_required_fields(row: Mapping[str, object]) -> list[str]:
    """Return the required keys absent from a raw dataset ``row``."""
    missing = [key for key in REQUIRED_FIELDS if key not in row]
    if not any(key.startswith(CAMERA_KEY_PREFIX) for key in row):
        missing.append(f"{CAMERA_KEY_PREFIX}* (at least one camera image)")
    return missing


def validate_episode_record(record: EpisodeRecord) -> list[str]:
    """Return semantic problems with an :class:`EpisodeRecord` (empty == valid)."""
    problems: list[str] = []
    problems += _check_instruction_matches_metadata(record)
    problems += _check_squares(record)
    problems += _check_capture_and_submove(record)
    problems += _check_labels(record)
    return problems


def is_valid_episode_record(record: EpisodeRecord) -> bool:
    return not validate_episode_record(record)


def validate_feedback(feedback: Feedback) -> list[str]:
    """Return problems with a :class:`Feedback` object (empty == valid)."""
    problems: list[str] = []
    if feedback.submove_index < 0:
        problems.append(f"submove_index must be >= 0, got {feedback.submove_index}")
    if feedback.success_label is True and feedback.failure_type is not None:
        problems.append("success_label True must not carry a failure_type")
    return problems


def _check_instruction_matches_metadata(record: EpisodeRecord) -> list[str]:
    if not record.instruction or not record.instruction.strip():
        return ["instruction is empty (it is mandatory)"]
    try:
        parsed = parse_command(record.instruction)
    except CommandParseError as exc:
        return [f"instruction does not parse: {exc}"]

    problems: list[str] = []
    if parsed.piece_type is not record.piece_type:
        problems.append(
            f"instruction piece {parsed.piece_type.value!r} != "
            f"metadata piece_type {record.piece_type.value!r}"
        )
    if parsed.start_square != record.start_square:
        problems.append(
            f"instruction start {parsed.start_square!r} != "
            f"metadata start_square {record.start_square!r}"
        )
    if parsed.target_square != record.target_square:
        problems.append(
            f"instruction target {parsed.target_square!r} != "
            f"metadata target_square {record.target_square!r}"
        )
    return problems


def _check_squares(record: EpisodeRecord) -> list[str]:
    problems: list[str] = []
    for label, name in (
        ("start_square", record.start_square),
        ("target_square", record.target_square),
    ):
        try:
            Square.from_name(name)
        except ValueError as exc:
            problems.append(f"{label} invalid: {exc}")
    if record.start_square == record.target_square:
        problems.append("start_square and target_square are identical")
    return problems


def _check_capture_and_submove(record: EpisodeRecord) -> list[str]:
    problems: list[str] = []
    if record.is_capture and record.captured_piece_type is None:
        problems.append("is_capture is True but captured_piece_type is None")
    if not record.is_capture and record.captured_piece_type is not None:
        problems.append("is_capture is False but captured_piece_type is set")
    if record.submove_index < 0:
        problems.append(f"submove_index must be >= 0, got {record.submove_index}")

    role = record.submove_role
    if role is SubmoveRole.MOVE:
        if record.is_capture:
            problems.append("submove_role 'move' must not be a capture")
        if record.submove_index != 0:
            problems.append("submove_role 'move' must have submove_index 0")
    elif role is SubmoveRole.REMOVE:
        if not record.is_capture:
            problems.append("submove_role 'remove' requires is_capture True")
        if record.submove_index != 0:
            problems.append("submove_role 'remove' must have submove_index 0")
    elif role is SubmoveRole.PLACE:
        if not record.is_capture:
            problems.append("submove_role 'place' requires is_capture True")
        if record.submove_index != 1:
            problems.append("submove_role 'place' must have submove_index 1")
    return problems


def _check_labels(record: EpisodeRecord) -> list[str]:
    if record.success_label is True and record.failure_type is not None:
        return ["success_label True must not carry a failure_type"]
    return []
