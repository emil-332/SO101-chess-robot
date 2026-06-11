"""Map square names to image regions / board coordinates via a grounded grid.

Consumes a *grounded grid* (square -> image region and/or board coordinate),
produced by board perception or by deterministic calibration as a
bootstrap. This module does **not** ground squares from pixels — that is
perception's job — it only resolves a square against an already-grounded grid.

Board-coordinate conventions follow ``board_state``: files advance along +x
(a -> h), ranks along +y (1 -> 8), with a1 at the origin corner. Image regions
use pixel coordinates (origin top-left, x right, y down).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from chess_robot.chess.board_state import FILES, RANKS, Square

# An image-space point (pixel coordinates, origin top-left, x right, y down).
Point = tuple[float, float]


class SquareNotGroundedError(LookupError):
    """Raised when a square lacks the requested grounding in the grid."""


@dataclass(frozen=True)
class ImageRegion:
    """Axis-aligned pixel bounding box of a square in an image.

    A perspective view of a square is really a quadrilateral; this AABB is the
    first-cut abstraction used for square highlighting and region lookup.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError(
                f"degenerate ImageRegion ({self.x_min}, {self.y_min})-"
                f"({self.x_max}, {self.y_max}); need x_max > x_min and y_max > y_min"
            )

    @property
    def center(self) -> tuple[float, float]:
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


@dataclass(frozen=True)
class SquareQuad:
    """The four image-space corners of a square (its projected quadrilateral).

    Corners are in the square's own cell order:
    ``(file, rank)``, ``(file+1, rank)``, ``(file+1, rank+1)``, ``(file, rank+1)``.
    Unlike :class:`ImageRegion` (an axis-aligned bbox) this preserves the true
    perspective shape, so adjacent squares do not blur together under oblique
    views. Use :meth:`bounding_region` for the AABB.
    """

    corners: tuple[Point, Point, Point, Point]

    @property
    def center(self) -> Point:
        xs = [px for px, _ in self.corners]
        ys = [py for _, py in self.corners]
        return sum(xs) / 4.0, sum(ys) / 4.0

    def bounding_region(self) -> ImageRegion:
        xs = [px for px, _ in self.corners]
        ys = [py for _, py in self.corners]
        return ImageRegion(min(xs), min(ys), max(xs), max(ys))


@dataclass(frozen=True)
class BoardPoint:
    """A point in the board's physical frame (metres).

    +x along files (a -> h), +y along ranks (1 -> 8); z is height above the
    board plane.
    """

    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class GroundedGrid:
    """Per-square grounding: image regions and/or board coordinates.

    Either map may be partial or absent; the mapper raises
    :class:`SquareNotGroundedError` for a square missing the requested grounding.
    ``quads`` carries the exact projected quadrilaterals (perspective-accurate);
    ``regions`` are their axis-aligned bounding boxes (convenience).
    """

    regions: Mapping[Square, ImageRegion] | None = None
    coordinates: Mapping[Square, BoardPoint] | None = None
    quads: Mapping[Square, SquareQuad] | None = None

    @classmethod
    def from_regular_board_coordinates(
        cls,
        a1_center: BoardPoint,
        square_size: float,
    ) -> GroundedGrid:
        """Deterministic bootstrap: a regular 8x8 lattice in the board plane.

        Places every square's centre on a regular grid with spacing
        ``square_size`` (metres), a1 at ``a1_center``, files along +x and ranks
        along +y (matching ``board_state``). Image regions are left unset.
        """
        if square_size <= 0:
            raise ValueError(f"square_size must be positive, got {square_size}")
        coordinates: dict[Square, BoardPoint] = {}
        for file_index, file in enumerate(FILES):
            for rank in RANKS:
                coordinates[Square(file, rank)] = BoardPoint(
                    x=a1_center.x + file_index * square_size,
                    y=a1_center.y + (rank - 1) * square_size,
                    z=a1_center.z,
                )
        return cls(coordinates=coordinates)


class BoardMapper:
    """Resolve squares against a grounded grid."""

    def __init__(self, grid: GroundedGrid) -> None:
        self._grid = grid

    @property
    def grid(self) -> GroundedGrid:
        return self._grid

    @staticmethod
    def _as_square(square: Square | str) -> Square:
        return square if isinstance(square, Square) else Square.from_name(square)

    def region_for(self, square: Square | str) -> ImageRegion:
        """Image region for a square.

        Raises:
            ValueError: if ``square`` is an invalid square name.
            SquareNotGroundedError: if the square has no image-region grounding.
        """
        resolved = self._as_square(square)
        regions = self._grid.regions
        if regions is None or resolved not in regions:
            raise SquareNotGroundedError(f"no image region grounded for {resolved.name!r}")
        return regions[resolved]

    def coordinate_for(self, square: Square | str) -> BoardPoint:
        """Board coordinate for a square.

        Raises:
            ValueError: if ``square`` is an invalid square name.
            SquareNotGroundedError: if the square has no coordinate grounding.
        """
        resolved = self._as_square(square)
        coordinates = self._grid.coordinates
        if coordinates is None or resolved not in coordinates:
            raise SquareNotGroundedError(
                f"no board coordinate grounded for {resolved.name!r}"
            )
        return coordinates[resolved]

    def has_region(self, square: Square | str) -> bool:
        resolved = self._as_square(square)
        return self._grid.regions is not None and resolved in self._grid.regions

    def has_coordinate(self, square: Square | str) -> bool:
        resolved = self._as_square(square)
        return self._grid.coordinates is not None and resolved in self._grid.coordinates
