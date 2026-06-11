"""Board-square representation and per-square occupancy / piece identity.

Coordinate conventions (fixed — do not change silently; shared by the board
mapper, perception, and move resolver):

- Files are ``a``-``h`` left-to-right; ranks are ``1``-``8`` bottom-to-top.
- ``a1`` is the bottom-left square; ``h8`` is the top-right.
- ``Square.index`` is rank-major: ``index = rank_index * 8 + file_index`` so
  ``a1 == 0``, ``h1 == 7``, ``a2 == 8`` ... ``h8 == 63``.

``PieceType`` lives here (piece identity is a board-state concern);
``command_parser`` re-imports it so its public import path is unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

FILES = "abcdefgh"
RANKS = (1, 2, 3, 4, 5, 6, 7, 8)
NUM_FILES = 8
NUM_RANKS = 8
NUM_SQUARES = NUM_FILES * NUM_RANKS


class PieceType(str, Enum):
    """The six standard chess piece types."""

    PAWN = "pawn"
    KNIGHT = "knight"
    BISHOP = "bishop"
    ROOK = "rook"
    QUEEN = "queen"
    KING = "king"


class PieceColor(str, Enum):
    """Piece color on a fully populated board."""

    WHITE = "white"
    BLACK = "black"


# FEN-style letters: uppercase for white, lowercase for black.
_PIECE_LETTER = {
    PieceType.PAWN: "p",
    PieceType.KNIGHT: "n",
    PieceType.BISHOP: "b",
    PieceType.ROOK: "r",
    PieceType.QUEEN: "q",
    PieceType.KING: "k",
}


@dataclass(frozen=True)
class Piece:
    """A piece's identity: type plus color."""

    piece_type: PieceType
    color: PieceColor

    @property
    def symbol(self) -> str:
        """FEN-style letter — uppercase if white, lowercase if black."""
        letter = _PIECE_LETTER[self.piece_type]
        return letter.upper() if self.color is PieceColor.WHITE else letter


@dataclass(frozen=True)
class Square:
    """A board square in algebraic coordinates (canonical lowercase file).

    Use :meth:`from_name` to parse arbitrary-case strings like ``"E4"``.
    """

    file: str
    rank: int

    def __post_init__(self) -> None:
        if len(self.file) != 1 or self.file not in FILES:
            raise ValueError(f"invalid file {self.file!r}; expected one of {FILES!r}")
        if self.rank not in RANKS:
            raise ValueError(f"invalid rank {self.rank!r}; expected 1-8")

    @classmethod
    def from_name(cls, name: str) -> Square:
        """Parse a square name such as ``"e4"`` (case- and whitespace-tolerant)."""
        text = name.strip().lower()
        if len(text) != 2 or not text[1].isdigit():
            raise ValueError(f"invalid square name {name!r}; expected like 'e4'")
        return cls(file=text[0], rank=int(text[1]))

    @classmethod
    def from_index(cls, index: int) -> Square:
        """Build a square from its rank-major index (``0``-``63``)."""
        if not 0 <= index < NUM_SQUARES:
            raise ValueError(f"invalid square index {index}; expected 0-{NUM_SQUARES - 1}")
        return cls(file=FILES[index % NUM_FILES], rank=index // NUM_FILES + 1)

    @property
    def name(self) -> str:
        return f"{self.file}{self.rank}"

    @property
    def file_index(self) -> int:
        return FILES.index(self.file)

    @property
    def rank_index(self) -> int:
        return self.rank - 1

    @property
    def index(self) -> int:
        return self.rank_index * NUM_FILES + self.file_index


_BACK_RANK = (
    PieceType.ROOK,
    PieceType.KNIGHT,
    PieceType.BISHOP,
    PieceType.QUEEN,
    PieceType.KING,
    PieceType.BISHOP,
    PieceType.KNIGHT,
    PieceType.ROOK,
)

# Inverse of _PIECE_LETTER: FEN letter (lowercase) -> piece type.
_LETTER_TO_TYPE = {letter: piece_type for piece_type, letter in _PIECE_LETTER.items()}


class BoardState:
    """Immutable per-square occupancy map.

    Empty squares are simply absent from the mapping. Build via the
    classmethods; the update helpers return new instances and never mutate.
    """

    def __init__(self, pieces: Mapping[Square, Piece] | None = None) -> None:
        self._pieces: dict[Square, Piece] = dict(pieces) if pieces is not None else {}

    @classmethod
    def empty(cls) -> BoardState:
        return cls()

    @classmethod
    def from_map(cls, pieces: Mapping[Square, Piece]) -> BoardState:
        return cls(pieces)

    @classmethod
    def standard_starting_position(cls) -> BoardState:
        """The standard 32-piece opening setup (a "full board, all pieces")."""
        pieces: dict[Square, Piece] = {}
        for file_index, file in enumerate(FILES):
            back = _BACK_RANK[file_index]
            pieces[Square(file, 1)] = Piece(back, PieceColor.WHITE)
            pieces[Square(file, 2)] = Piece(PieceType.PAWN, PieceColor.WHITE)
            pieces[Square(file, 7)] = Piece(PieceType.PAWN, PieceColor.BLACK)
            pieces[Square(file, 8)] = Piece(back, PieceColor.BLACK)
        return cls(pieces)

    @classmethod
    def from_fen(cls, fen: str) -> BoardState:
        """Build occupancy from the placement field of a FEN string.

        Only the piece-placement field is read (everything after the first space
        — side to move, castling, etc. — is ignored). Uppercase letters are
        white, lowercase black; digits are runs of empty squares.
        """
        placement = fen.strip().split(" ", 1)[0]
        ranks = placement.split("/")
        if len(ranks) != NUM_RANKS:
            raise ValueError(f"invalid FEN {fen!r}: expected {NUM_RANKS} ranks")
        pieces: dict[Square, Piece] = {}
        for row, rank_str in enumerate(ranks):
            rank = NUM_RANKS - row  # FEN lists rank 8 first
            file_index = 0
            for char in rank_str:
                if char.isdigit():
                    file_index += int(char)
                    continue
                piece_type = _LETTER_TO_TYPE.get(char.lower())
                if piece_type is None:
                    raise ValueError(f"invalid FEN piece {char!r} in {fen!r}")
                if file_index >= NUM_FILES:
                    raise ValueError(f"FEN rank too long: {rank_str!r}")
                color = PieceColor.WHITE if char.isupper() else PieceColor.BLACK
                pieces[Square(FILES[file_index], rank)] = Piece(piece_type, color)
                file_index += 1
            if file_index != NUM_FILES:
                raise ValueError(f"FEN rank wrong length: {rank_str!r}")
        return cls(pieces)

    def to_fen(self) -> str:
        """Serialize occupancy to the FEN piece-placement field (no game state).

        Round-trips with :meth:`from_fen`.
        """
        rows: list[str] = []
        for rank in range(NUM_RANKS, 0, -1):
            row = ""
            empties = 0
            for file in FILES:
                piece = self._pieces.get(Square(file, rank))
                if piece is None:
                    empties += 1
                    continue
                if empties:
                    row += str(empties)
                    empties = 0
                row += piece.symbol
            if empties:
                row += str(empties)
            rows.append(row)
        return "/".join(rows)

    def piece_at(self, square: Square) -> Piece | None:
        return self._pieces.get(square)

    def is_occupied(self, square: Square) -> bool:
        return square in self._pieces

    def occupied_squares(self) -> list[Square]:
        """Occupied squares, ordered by rank-major index."""
        return sorted(self._pieces, key=lambda square: square.index)

    def pieces(self) -> dict[Square, Piece]:
        """A copy of the occupancy mapping."""
        return dict(self._pieces)

    def with_piece(self, square: Square, piece: Piece) -> BoardState:
        updated = dict(self._pieces)
        updated[square] = piece
        return BoardState(updated)

    def without_piece(self, square: Square) -> BoardState:
        updated = dict(self._pieces)
        updated.pop(square, None)
        return BoardState(updated)

    def __len__(self) -> int:
        return len(self._pieces)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoardState):
            return NotImplemented
        return self._pieces == other._pieces

    # Defining __eq__ makes BoardState unhashable (Python sets __hash__ = None),
    # which is intended — equality is by content.

    def __repr__(self) -> str:
        return f"BoardState({len(self._pieces)} pieces)"
