"""Tests for board-square representation and occupancy (  1.2)."""

import pytest

from chess_robot.chess.board_state import (
    NUM_SQUARES,
    BoardState,
    Piece,
    PieceColor,
    PieceType,
    Square,
)


def test_square_name_and_index_roundtrip() -> None:
    for index in range(NUM_SQUARES):
        square = Square.from_index(index)
        assert square.index == index
        assert Square.from_name(square.name) == square


def test_orientation_corners_are_fixed() -> None:
    assert Square("a", 1).index == 0
    assert Square("h", 1).index == 7
    assert Square("a", 8).index == 56
    assert Square("h", 8).index == 63


def test_from_name_normalizes_case_and_whitespace() -> None:
    assert Square.from_name("E4") == Square("e", 4)
    assert Square.from_name(" b1 ") == Square("b", 1)


@pytest.mark.parametrize("bad", ["i1", "a9", "a0", "aa", "e", "e44", "", "11"])
def test_from_name_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError):
        Square.from_name(bad)


@pytest.mark.parametrize("bad_index", [-1, 64, 100])
def test_from_index_rejects_out_of_range(bad_index: int) -> None:
    with pytest.raises(ValueError):
        Square.from_index(bad_index)


def test_direct_square_construction_rejects_multichar_file() -> None:
    # "ab" is a substring of FILES — guard against accepting it.
    with pytest.raises(ValueError):
        Square("ab", 1)


def test_piece_symbol_is_fen_style() -> None:
    assert Piece(PieceType.KNIGHT, PieceColor.WHITE).symbol == "N"
    assert Piece(PieceType.KNIGHT, PieceColor.BLACK).symbol == "n"
    assert Piece(PieceType.PAWN, PieceColor.BLACK).symbol == "p"


def test_empty_board_is_unoccupied() -> None:
    board = BoardState.empty()
    assert len(board) == 0
    assert not board.is_occupied(Square("e", 4))
    assert board.piece_at(Square("e", 4)) is None


def test_occupancy_query() -> None:
    knight = Piece(PieceType.KNIGHT, PieceColor.WHITE)
    board = BoardState.from_map({Square("b", 1): knight})
    assert board.is_occupied(Square("b", 1))
    assert board.piece_at(Square("b", 1)) == knight
    assert not board.is_occupied(Square("c", 3))


def test_updates_are_immutable() -> None:
    knight = Piece(PieceType.KNIGHT, PieceColor.WHITE)
    base = BoardState.empty()
    placed = base.with_piece(Square("b", 1), knight)
    assert len(base) == 0  # original untouched
    assert placed.is_occupied(Square("b", 1))

    removed = placed.without_piece(Square("b", 1))
    assert not removed.is_occupied(Square("b", 1))
    assert placed.is_occupied(Square("b", 1))  # prior state untouched


def test_standard_starting_position() -> None:
    board = BoardState.standard_starting_position()
    assert len(board) == 32
    assert board.piece_at(Square("e", 1)) == Piece(PieceType.KING, PieceColor.WHITE)
    assert board.piece_at(Square("d", 8)) == Piece(PieceType.QUEEN, PieceColor.BLACK)
    assert board.piece_at(Square("a", 2)) == Piece(PieceType.PAWN, PieceColor.WHITE)
    assert not board.is_occupied(Square("e", 4))  # middle ranks empty
    assert board.occupied_squares()[0] == Square("a", 1)


def test_board_equality_by_content() -> None:
    knight = Piece(PieceType.KNIGHT, PieceColor.WHITE)
    a = BoardState.from_map({Square("b", 1): knight})
    b = BoardState.empty().with_piece(Square("b", 1), knight)
    assert a == b
    assert a != BoardState.empty()


def test_piece_type_reexported_from_command_parser() -> None:
    from chess_robot.chess.command_parser import PieceType as ReexportedPieceType

    assert ReexportedPieceType is PieceType


def test_from_fen_standard_start() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert BoardState.from_fen(fen) == BoardState.standard_starting_position()


def test_from_fen_partial_position() -> None:
    board = BoardState.from_fen("8/8/8/8/4P3/8/8/8")
    assert len(board) == 1
    assert board.piece_at(Square("e", 4)) == Piece(PieceType.PAWN, PieceColor.WHITE)


@pytest.mark.parametrize(
    "bad",
    [
        "8/8/8/8/8/8/8",  # 7 ranks
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNX",  # invalid piece X
        "9/8/8/8/8/8/8/8",  # rank too long
    ],
)
def test_from_fen_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError):
        BoardState.from_fen(bad)


def test_to_fen_round_trips() -> None:
    board = BoardState.standard_starting_position()
    assert board.to_fen().startswith(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    )
    assert BoardState.from_fen(board.to_fen()) == board
    partial = BoardState.from_fen("8/8/8/8/4P3/8/8/8")
    assert partial.to_fen() == "8/8/8/8/4P3/8/8/8"
    assert BoardState.from_fen(partial.to_fen()) == partial
