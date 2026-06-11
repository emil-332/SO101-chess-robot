"""Synthetic board renderer + a colour-based stand-in classifier (Tier B).

Renders an 8x8 board from a :class:`BoardState` into a numpy image using the
**same homography geometry** as grounding (`grid_from_corners`), with optional
perspective (oblique views) and pixel noise. Pieces are drawn in 12 unique
colours; :class:`ColorPieceClassifier` reads occupancy + type + colour back from
the crop pixel colours, implementing the :class:`PieceClassifier` interface so it
plugs into `ComposedBoardPerception` and `evaluate_perception`.

**What this validates:** the geometry -> crop -> classify -> evaluation loop, and
empirically how accuracy degrades under oblique views (AABB crops overlap
neighbours) and noise — producing *real* perception metrics with no GPU/hardware.

**What it does NOT validate:** reading *photorealistic* pieces. The piece colours
are a controlled stand-in, not a real appearance model, so this does not exercise
the learned CNN's zero-shot challenge (unseen piece *shapes/textures*). Training
that CNN still needs rendered/real data on the cloud GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chess_robot.chess.board_mapper import GroundedGrid, Point
from chess_robot.chess.board_state import (
    FILES,
    RANKS,
    BoardState,
    Piece,
    PieceColor,
    PieceType,
    Square,
)
from chess_robot.perception.board_perception import DEFAULT_GROUNDING_CAMERA, CameraFrames
from chess_robot.perception.camera_utils import extract_square_crops
from chess_robot.perception.piece_locator import PieceClassifier
from chess_robot.perception.square_grounding import BoardCorners, grid_from_corners

Color = tuple[int, int, int]
Palette = tuple[Color, Color]  # (light square, dark square)

DEFAULT_PALETTE: Palette = ((210, 210, 180), (110, 110, 90))
ALT_PALETTE: Palette = ((180, 200, 220), (60, 80, 120))
_BACKGROUND: Color = (30, 30, 30)

_TYPES = (
    PieceType.PAWN,
    PieceType.KNIGHT,
    PieceType.BISHOP,
    PieceType.ROOK,
    PieceType.QUEEN,
    PieceType.KING,
)
# 12 saturated, pairwise-distant colours, all far from the greyish board palette.
_WHITE_RGB: tuple[Color, ...] = (
    (255, 0, 0),
    (255, 128, 0),
    (255, 255, 0),
    (128, 255, 0),
    (0, 255, 0),
    (0, 255, 128),
)
_BLACK_RGB: tuple[Color, ...] = (
    (0, 255, 255),
    (0, 128, 255),
    (0, 0, 255),
    (128, 0, 255),
    (255, 0, 255),
    (255, 0, 128),
)
PIECE_RGB: dict[tuple[PieceType, PieceColor], Color] = {}
for _index, _type in enumerate(_TYPES):
    PIECE_RGB[(_type, PieceColor.WHITE)] = _WHITE_RGB[_index]
    PIECE_RGB[(_type, PieceColor.BLACK)] = _BLACK_RGB[_index]


@dataclass
class RenderedBoard:
    """A rendered board image with its ground-truth corners and occupancy."""

    image: np.ndarray
    corners: BoardCorners
    board: BoardState


def _fill_quad(image: np.ndarray, points: tuple[Point, ...], color: Color) -> None:
    """Fill a convex quad (image-space corners) with a solid colour, in place."""
    height, width = image.shape[0], image.shape[1]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0 = max(0, int(np.floor(min(xs))))
    x1 = min(width, int(np.ceil(max(xs))) + 1)
    y0 = max(0, int(np.floor(min(ys))))
    y1 = min(height, int(np.ceil(max(ys))) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    grid_x, grid_y = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    px = grid_x + 0.5
    py = grid_y + 0.5
    inside_pos = np.ones(px.shape, dtype=bool)
    inside_neg = np.ones(px.shape, dtype=bool)
    count = len(points)
    for i in range(count):
        ax, ay = points[i]
        bx, by = points[(i + 1) % count]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        inside_pos &= cross >= -1e-9
        inside_neg &= cross <= 1e-9
    inside = inside_pos | inside_neg
    image[y0:y1, x0:x1][inside] = color


def _shrink_quad(points: tuple[Point, ...], inset: float) -> tuple[Point, ...]:
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return tuple(
        (cx + (p[0] - cx) * (1.0 - inset), cy + (p[1] - cy) * (1.0 - inset))
        for p in points
    )


def render_board(
    board: BoardState,
    corners: BoardCorners,
    *,
    image_size: tuple[int, int],
    palette: Palette = DEFAULT_PALETTE,
    inset: float = 0.5,
    noise_std: float = 0.0,
    background: Color = _BACKGROUND,
    rng: np.random.Generator | None = None,
) -> RenderedBoard:
    """Render ``board`` into an image bounded by ``corners`` (using the homography)."""
    height, width = image_size
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = background
    grid = grid_from_corners(corners)
    quads = grid.quads
    assert quads is not None
    light, dark = palette
    for file_index, file in enumerate(FILES):
        for rank in RANKS:
            square = Square(file, rank)
            corners_xy = quads[square].corners
            square_color = light if (file_index + (rank - 1)) % 2 == 0 else dark
            _fill_quad(image, corners_xy, square_color)
            piece = board.piece_at(square)
            if piece is not None:
                _fill_quad(
                    image,
                    _shrink_quad(corners_xy, inset),
                    PIECE_RGB[(piece.piece_type, piece.color)],
                )
    if noise_std > 0:
        generator = rng if rng is not None else np.random.default_rng()
        noisy = image.astype(float) + generator.normal(0.0, noise_std, image.shape)
        image = np.clip(noisy, 0, 255).astype(np.uint8)
    return RenderedBoard(image=image, corners=corners, board=board)


class ColorPieceClassifier(PieceClassifier):
    """Read occupancy + piece identity from crop pixel colours (Tier B stand-in).

    For each grounded square it crops the camera frame and, if enough pixels match
    one of the :data:`PIECE_RGB` colours, labels the square with that piece. Empty
    squares (board colours only) match nothing. This is the per-square-crop path of
    the real classifier, with a trivial colour reader instead of a trained CNN.
    """

    def __init__(
        self,
        *,
        camera: str = DEFAULT_GROUNDING_CAMERA,
        tolerance: int = 60,
        min_pixels: int = 4,
    ) -> None:
        self._camera = camera
        self._tolerance = tolerance
        self._min_pixels = min_pixels

    def classify(self, frames: CameraFrames, grid: GroundedGrid) -> BoardState:
        image = np.asarray(frames[self._camera])
        crops = extract_square_crops(image, grid, top_pad_ratio=0.0)
        pieces: dict[Square, Piece] = {}
        for square, crop in crops.items():
            piece = self._classify_crop(crop)
            if piece is not None:
                pieces[square] = piece
        return BoardState.from_map(pieces)

    def _classify_crop(self, crop: np.ndarray) -> Piece | None:
        flat = crop.reshape(-1, 3).astype(int)
        best: Piece | None = None
        best_count = self._min_pixels - 1
        for (piece_type, color), rgb in PIECE_RGB.items():
            distance = np.abs(flat - np.array(rgb)).sum(axis=1)
            count = int(np.count_nonzero(distance <= self._tolerance))
            if count > best_count:
                best_count = count
                best = Piece(piece_type, color)
        return best
