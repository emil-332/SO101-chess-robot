"""Tests for the command parser"""

import pytest

from chess_robot.chess.command_parser import (
    CommandParseError,
    ParsedCommand,
    PieceType,
    parse_command,
)


@pytest.mark.parametrize("piece", list(PieceType))
def test_parses_each_piece_type(piece: PieceType) -> None:
    result = parse_command(f"move {piece.value} from b1 to c3")
    assert result.piece_type is piece
    assert result.start_square == "b1"
    assert result.target_square == "c3"


def test_returns_parsed_command_dataclass() -> None:
    assert parse_command("move pawn from e2 to e4") == ParsedCommand(
        PieceType.PAWN, "e2", "e4"
    )


@pytest.mark.parametrize(
    "text",
    [
        "move knight from b1 to c3",
        "  move   knight  from  b1   to c3 ",  # extra / surrounding whitespace
        "Move Knight From B1 To C3",  # mixed case
    ],
)
def test_normalizes_whitespace_and_case(text: str) -> None:
    assert parse_command(text) == ParsedCommand(PieceType.KNIGHT, "b1", "c3")


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "empty"),
        ("move knight b1 c3", "missing from/to"),
        ("move knight from b1 to", "missing target"),
        ("knight from b1 to c3", "missing 'move'"),
        ("move dragon from b1 to c3", "invalid piece"),
        ("move knight from i1 to c3", "invalid file"),
        ("move knight from b9 to c3", "invalid rank"),
        ("move knight from b1 to c3 please", "trailing junk"),
        ("move knight from b1 to b1", "source == target"),
    ],
)
def test_rejects_invalid_commands(text: str, reason: str) -> None:
    with pytest.raises(CommandParseError):
        parse_command(text)


def test_error_is_value_error_subclass() -> None:
    assert issubclass(CommandParseError, ValueError)
