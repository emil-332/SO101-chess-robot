"""Tests for the move resolver (  1.4)."""

import pytest

from chess_robot.chess.board_state import (
    BoardState,
    Piece,
    PieceColor,
    PieceType,
    Square,
)
from chess_robot.chess.command_parser import ParsedCommand
from chess_robot.chess.move_resolver import (
    MoveResolutionError,
    MoveResolver,
    OffBoardLocation,
    SubmoveRole,
)

TRAY = OffBoardLocation(name="capture_tray")


def _board(pieces: dict[Square, Piece]) -> BoardState:
    return BoardState.from_map(pieces)


def _white(piece_type: PieceType) -> Piece:
    return Piece(piece_type, PieceColor.WHITE)


def _black(piece_type: PieceType) -> Piece:
    return Piece(piece_type, PieceColor.BLACK)


def test_non_capture_is_single_move_segment() -> None:
    board = _board({Square("b", 1): _white(PieceType.KNIGHT)})
    resolved = MoveResolver(TRAY).resolve(
        ParsedCommand(PieceType.KNIGHT, "b1", "c3"), board
    )

    assert resolved.is_capture is False
    assert resolved.captured_piece_type is None
    assert resolved.start_square == Square("b", 1)
    assert resolved.target_square == Square("c", 3)
    assert len(resolved.submoves) == 1

    only = resolved.submoves[0]
    assert only.index == 0
    assert only.role is SubmoveRole.MOVE
    assert only.piece_type is PieceType.KNIGHT
    assert only.source == Square("b", 1)
    assert only.destination == Square("c", 3)
    assert only.highlighted_squares == (Square("b", 1), Square("c", 3))


def test_capture_splits_into_ordered_remove_then_place() -> None:
    board = _board(
        {
            Square("b", 1): _white(PieceType.KNIGHT),
            Square("c", 3): _black(PieceType.PAWN),
        }
    )
    resolved = MoveResolver(TRAY).resolve(
        ParsedCommand(PieceType.KNIGHT, "b1", "c3"), board
    )

    assert resolved.is_capture is True
    assert resolved.captured_piece_type is PieceType.PAWN
    assert [s.role for s in resolved.submoves] == [
        SubmoveRole.REMOVE,
        SubmoveRole.PLACE,
    ]
    assert [s.index for s in resolved.submoves] == [0, 1]

    remove, place = resolved.submoves
    assert remove.piece_type is PieceType.PAWN  # the captured piece
    assert remove.source == Square("c", 3)
    assert remove.destination == TRAY
    assert remove.highlighted_squares == (Square("c", 3),)  # off-board excluded

    assert place.piece_type is PieceType.KNIGHT  # the instructed piece
    assert place.source == Square("b", 1)
    assert place.destination == Square("c", 3)
    assert place.highlighted_squares == (Square("b", 1), Square("c", 3))


def test_off_board_location_is_configurable() -> None:
    board = _board(
        {
            Square("b", 1): _white(PieceType.KNIGHT),
            Square("c", 3): _black(PieceType.PAWN),
        }
    )
    custom = OffBoardLocation(name="left_bin")
    resolver = MoveResolver(custom)
    resolved = resolver.resolve(ParsedCommand(PieceType.KNIGHT, "b1", "c3"), board)

    assert resolver.off_board_location == custom
    assert resolved.submoves[0].destination == custom


def test_capture_is_legality_agnostic_for_same_colour() -> None:
    # Capturing a same-colour piece is illegal chess but still splits — the
    # resolver does no legality reasoning (out of scope).
    board = _board(
        {
            Square("b", 1): _white(PieceType.KNIGHT),
            Square("c", 3): _white(PieceType.PAWN),
        }
    )
    resolved = MoveResolver(TRAY).resolve(
        ParsedCommand(PieceType.KNIGHT, "b1", "c3"), board
    )
    assert resolved.is_capture is True
    assert resolved.captured_piece_type is PieceType.PAWN


def test_rejects_empty_source_square() -> None:
    with pytest.raises(MoveResolutionError):
        MoveResolver(TRAY).resolve(
            ParsedCommand(PieceType.KNIGHT, "b1", "c3"), BoardState.empty()
        )


def test_rejects_piece_type_mismatch_against_occupancy() -> None:
    board = _board({Square("b", 1): _white(PieceType.KNIGHT)})
    with pytest.raises(MoveResolutionError):
        MoveResolver(TRAY).resolve(ParsedCommand(PieceType.QUEEN, "b1", "c3"), board)


def test_rejects_identical_source_and_target() -> None:
    board = _board({Square("b", 1): _white(PieceType.KNIGHT)})
    with pytest.raises(MoveResolutionError):
        MoveResolver(TRAY).resolve(ParsedCommand(PieceType.KNIGHT, "b1", "b1"), board)
