"""Parse natural-language chess move commands into structured fields.

The supported grammar is the constrained pattern from
``docs/data_collection.md``::

    move {piece_type} from {start_square} to {target_square}

e.g. ``move knight from b1 to c3``. Piece types are the six standard chess
pieces; squares are standard algebraic coordinates (files a-h, ranks 1-8).

This module performs **no** chess-legality reasoning (move legality, turns, and
capture handling are out of scope here / handled by downstream deterministic
logic — see ``docs/architecture.md``). It only turns a command string into
validated fields, or raises :class:`CommandParseError`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from chess_robot.chess.board_state import PieceType

# PieceType is defined in board_state (piece identity is a board-state concern)
# and re-exported here so the parser's public import path stays stable.
__all__ = ["CommandParseError", "ParsedCommand", "PieceType", "parse_command"]


class CommandParseError(ValueError):
    """Raised when a command string does not match the supported grammar."""


@dataclass(frozen=True)
class ParsedCommand:
    """Structured result of parsing a move command.

    Squares are validated algebraic strings (e.g. ``"b1"``); the richer board
    representation is a separate concern (``board_state.py``).
    """

    piece_type: PieceType
    start_square: str
    target_square: str


_PIECE_ALTERNATION = "|".join(piece.value for piece in PieceType)
_COMMAND_RE = re.compile(
    rf"move\s+(?P<piece>{_PIECE_ALTERNATION})\s+"
    r"from\s+(?P<start>[a-h][1-8])\s+to\s+(?P<target>[a-h][1-8])"
)
_EXPECTED = (
    "expected 'move <piece> from <square> to <square>' "
    "(piece in pawn|knight|bishop|rook|queen|king, squares a1-h8)"
)


def parse_command(text: str) -> ParsedCommand:
    """Parse a natural-language move command into a :class:`ParsedCommand`.

    Leading/trailing whitespace is stripped, internal whitespace is collapsed,
    and matching is case-insensitive.

    Args:
        text: command string, e.g. ``"move knight from b1 to c3"``.

    Returns:
        The parsed piece type and source/target squares.

    Raises:
        CommandParseError: if ``text`` does not match the supported grammar, or
            if the source and target squares are identical.
    """
    normalized = re.sub(r"\s+", " ", text.strip()).lower()
    match = _COMMAND_RE.fullmatch(normalized)
    if match is None:
        raise CommandParseError(f"could not parse command {text!r}; {_EXPECTED}")

    start = match.group("start")
    target = match.group("target")
    if start == target:
        raise CommandParseError(
            f"source and target squares are identical ({start!r}) in command {text!r}"
        )

    return ParsedCommand(
        piece_type=PieceType(match.group("piece")),
        start_square=start,
        target_square=target,
    )
