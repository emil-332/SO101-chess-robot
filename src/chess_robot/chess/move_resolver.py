"""Deterministic move resolution: source/target, capture detection, split-moves.

Given a parsed command and a board occupancy (metadata- or perception-supplied),
this resolves the source/target squares, detects captures (target occupied), and
splits a capturing move into two ordered submoves:

- submove 0 (``remove``): move the captured piece from the target square to a
  configurable off-board location;
- submove 1 (``place``): move the instructed piece from its source to the target.

A non-capturing move is a single submove (``move``). Each submove carries the
board squares to highlight in its observation. This is capture *sequencing*,
deterministic preprocessing — **not** collision-aware path planning, and it does
no chess-legality reasoning (a same-colour "capture" still splits). The submove
index/role/captured-piece fields mirror the dataset schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from chess_robot.chess.board_mapper import BoardPoint
from chess_robot.chess.board_state import BoardState, PieceType, Square
from chess_robot.chess.command_parser import ParsedCommand


class MoveResolutionError(ValueError):
    """Raised when a command cannot be resolved against the board occupancy."""


class SubmoveRole(str, Enum):
    """Role of a submove. Values match the dataset ``submove_role`` field."""

    MOVE = "move"
    REMOVE = "remove"
    PLACE = "place"


@dataclass(frozen=True)
class OffBoardLocation:
    """A configurable destination off the board (e.g. a capture tray).

    The physical-coordinate convention is still an open decision (see plan.md);
    ``coordinate`` is optional so a name-only handle works as a bootstrap.
    """

    name: str
    coordinate: BoardPoint | None = None


# A submove endpoint is either a board square or the off-board location.
SubmoveEndpoint = Square | OffBoardLocation


@dataclass(frozen=True)
class Submove:
    """One manipulation segment handed to the policy in sequence."""

    index: int
    role: SubmoveRole
    piece_type: PieceType
    source: SubmoveEndpoint
    destination: SubmoveEndpoint

    @property
    def highlighted_squares(self) -> tuple[Square, ...]:
        """Board squares to highlight for this submove (off-board excluded)."""
        endpoints = (self.source, self.destination)
        return tuple(loc for loc in endpoints if isinstance(loc, Square))


@dataclass(frozen=True)
class ResolvedMove:
    """The fully resolved move: metadata plus the ordered submoves."""

    piece_type: PieceType
    start_square: Square
    target_square: Square
    is_capture: bool
    captured_piece_type: PieceType | None
    submoves: tuple[Submove, ...]


class MoveResolver:
    """Resolve commands into ordered submoves using board occupancy.

    The off-board location is injected (configurable; never hard-coded).
    """

    def __init__(self, off_board_location: OffBoardLocation) -> None:
        self._off_board = off_board_location

    @property
    def off_board_location(self) -> OffBoardLocation:
        return self._off_board

    def resolve(self, command: ParsedCommand, board: BoardState) -> ResolvedMove:
        """Resolve ``command`` against ``board`` occupancy.

        Raises:
            MoveResolutionError: if source and target are identical, the source
                square is empty, or the piece named by the command does not match
                the piece the board reports on the source square.
        """
        start = Square.from_name(command.start_square)
        target = Square.from_name(command.target_square)
        if start == target:
            raise MoveResolutionError(
                f"source and target are the same square ({start.name})"
            )

        source_piece = board.piece_at(start)
        if source_piece is None:
            raise MoveResolutionError(f"no piece on source square {start.name!r}")
        if source_piece.piece_type is not command.piece_type:
            raise MoveResolutionError(
                f"command names {command.piece_type.value} on {start.name} but the "
                f"board reports {source_piece.piece_type.value} there"
            )

        target_piece = board.piece_at(target)
        is_capture = target_piece is not None

        captured_piece_type: PieceType | None
        submoves: tuple[Submove, ...]
        if target_piece is not None:
            captured_piece_type = target_piece.piece_type
            submoves = (
                Submove(
                    index=0,
                    role=SubmoveRole.REMOVE,
                    piece_type=captured_piece_type,
                    source=target,
                    destination=self._off_board,
                ),
                Submove(
                    index=1,
                    role=SubmoveRole.PLACE,
                    piece_type=command.piece_type,
                    source=start,
                    destination=target,
                ),
            )
        else:
            captured_piece_type = None
            submoves = (
                Submove(
                    index=0,
                    role=SubmoveRole.MOVE,
                    piece_type=command.piece_type,
                    source=start,
                    destination=target,
                ),
            )

        return ResolvedMove(
            piece_type=command.piece_type,
            start_square=start,
            target_square=target,
            is_capture=is_capture,
            captured_piece_type=captured_piece_type,
            submoves=submoves,
        )
