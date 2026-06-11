"""Camera helpers for perception data collection (  1b.2 tooling).

Two jobs:

- Turn four clicked corner points into :class:`BoardCorners`
  (:func:`corners_from_points`), for the corner-annotation tool.
- Extract per-square crops from a (side/oblique) image given a grounded grid
  (:func:`extract_square_crops`), with **vertical padding** so a tall piece that
  rises above its square's board-plane footprint is captured. Combined with a
  known position (FEN -> ``BoardState``), this auto-labels classifier crops
  (:func:`labeled_square_crops`) — no per-piece hand annotation.

Image arrays are numpy ``HxWx[C]`` with origin top-left (x right, y down). The
upward padding assumes a roughly upright camera, i.e. pieces extend toward
smaller y; adjust ``top_pad_ratio`` per rig.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from chess_robot.chess.board_mapper import GroundedGrid, ImageRegion, Point
from chess_robot.chess.board_state import BoardState, Square
from chess_robot.perception.square_grounding import BoardCorners

PixelBox = tuple[int, int, int, int]  # (x0, y0, x1, y1)


def corners_from_points(points: list[Point]) -> BoardCorners:
    """Build :class:`BoardCorners` from 4 clicked points, ordered a1, h1, h8, a8."""
    if len(points) != 4:
        raise ValueError(f"expected 4 corner points (a1, h1, h8, a8), got {len(points)}")
    a1, h1, h8, a8 = points
    return BoardCorners(a1=a1, h1=h1, h8=h8, a8=a8)


def crop_box(
    region: ImageRegion,
    image_width: int,
    image_height: int,
    *,
    top_pad_ratio: float = 1.0,
    side_pad_ratio: float = 0.0,
) -> PixelBox:
    """Pixel crop box for a square, padded upward (and optionally sideways).

    ``top_pad_ratio`` extends the crop toward smaller y by that fraction of the
    region height (to capture the piece above the square). The box is rounded to
    integers and clipped to the image bounds.
    """
    width = region.width
    height = region.height
    x0 = region.x_min - side_pad_ratio * width
    x1 = region.x_max + side_pad_ratio * width
    y0 = region.y_min - top_pad_ratio * height
    y1 = region.y_max
    return (
        max(0, round(x0)),
        max(0, round(y0)),
        min(image_width, round(x1)),
        min(image_height, round(y1)),
    )


def extract_square_crops(
    image: np.ndarray,
    grid: GroundedGrid,
    *,
    top_pad_ratio: float = 1.0,
    side_pad_ratio: float = 0.0,
) -> dict[Square, np.ndarray]:
    """Crop each grounded square's (padded) region out of ``image``.

    Degenerate boxes (clipped to zero area) are skipped.
    """
    regions = grid.regions
    if not regions:
        return {}
    image_height, image_width = image.shape[0], image.shape[1]
    crops: dict[Square, np.ndarray] = {}
    for square, region in regions.items():
        x0, y0, x1, y1 = crop_box(
            region,
            image_width,
            image_height,
            top_pad_ratio=top_pad_ratio,
            side_pad_ratio=side_pad_ratio,
        )
        if x1 > x0 and y1 > y0:
            crops[square] = image[y0:y1, x0:x1]
    return crops


def _stamp(out: np.ndarray, x: int, y: int, color: tuple[int, int, int], thickness: int) -> None:
    height, width = out.shape[0], out.shape[1]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + thickness)
    y1 = min(height, y + thickness)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = color


def _draw_line(
    out: np.ndarray, p0: Point, p1: Point, color: tuple[int, int, int], thickness: int
) -> None:
    """Bresenham line, stamping a thickness x thickness block at each step."""
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        _stamp(out, x0, y0, color, thickness)
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * err
        if doubled >= dy:
            err += dy
            x0 += sx
        if doubled <= dx:
            err += dx
            y0 += sy


def _draw_polygon(
    out: np.ndarray, points: tuple[Point, ...], color: tuple[int, int, int], thickness: int
) -> None:
    count = len(points)
    for i in range(count):
        _draw_line(out, points[i], points[(i + 1) % count], color, thickness)


def highlight_squares(
    image: np.ndarray,
    grid: GroundedGrid,
    squares: Iterable[Square],
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Return a copy of ``image`` with the given ``squares`` outlined.

    Marks the squares relevant to the current (sub)move so the policy attends to
    them — the observation-highlighting preprocessing. Draws the exact projected
    **quad** when the grid has one (perspective-accurate, avoids neighbour
    overlap), else the axis-aligned region border. Assumes a 3-channel image;
    squares missing from the grid are skipped.
    """
    out = image.copy()
    regions = grid.regions or {}
    quads = grid.quads or {}
    height, width = out.shape[0], out.shape[1]
    for square in squares:
        quad = quads.get(square)
        if quad is not None:
            _draw_polygon(out, quad.corners, color, thickness)
            continue
        region = regions.get(square)
        if region is None:
            continue
        x0 = max(0, int(region.x_min))
        y0 = max(0, int(region.y_min))
        x1 = min(width, int(region.x_max))
        y1 = min(height, int(region.y_max))
        if x1 <= x0 or y1 <= y0:
            continue
        out[y0:min(y0 + thickness, y1), x0:x1] = color
        out[max(y1 - thickness, y0):y1, x0:x1] = color
        out[y0:y1, x0:min(x0 + thickness, x1)] = color
        out[y0:y1, max(x1 - thickness, x0):x1] = color
    return out


def square_label(board: BoardState, square: Square) -> str:
    """13-class label for a square: ``"empty"`` or ``"{color}_{type}"``."""
    piece = board.piece_at(square)
    if piece is None:
        return "empty"
    return f"{piece.color.value}_{piece.piece_type.value}"


# eq disabled: holds a numpy array (array equality is not a plain bool).
@dataclass(frozen=True, eq=False)
class LabeledCrop:
    """A square crop paired with its auto-derived class label."""

    square: Square
    image: np.ndarray
    label: str


def labeled_square_crops(
    image: np.ndarray,
    grid: GroundedGrid,
    board: BoardState,
    *,
    top_pad_ratio: float = 1.0,
    side_pad_ratio: float = 0.0,
) -> list[LabeledCrop]:
    """Per-square crops labeled from a known position (FEN -> ``board``).

    This is the auto-labeling path: a known board plus the grounded grid yields
    labeled training crops for the per-square piece classifier without per-piece
    hand annotation.
    """
    crops = extract_square_crops(
        image, grid, top_pad_ratio=top_pad_ratio, side_pad_ratio=side_pad_ratio
    )
    return [
        LabeledCrop(square=square, image=crop, label=square_label(board, square))
        for square, crop in crops.items()
    ]
