"""Map chesscog rendered-dataset labels to our board-corner convention.

The chesscog dataset (OSF 10.17605/OSF.IO/XF3KA) ships one JSON per rendered
image with three fields we use: ``corners`` (the four board outer-corner pixel
points, **unordered**), ``fen`` (full FEN), and ``white_turn`` (camera side).

Our pipeline labels corners by board square (a1/h1/h8/a8) so the homography in
``grid_from_corners`` aligns the auto-labelled crops with the FEN. The load-bearing
step is orientation: chesscog encodes it via ``white_turn``, so the geometric
corner that is "a1" differs between the two camera sides. Getting this wrong
mirrors/rotates every label and silently poisons training, so it is tested.

Derived from chesscog's own code (``occupancy_classifier/create_dataset.py``):
its perspective warp maps the sorted corners to ``[TL, TR, BR, BL]`` and indexes
squares with ``row, col = 7-rank, file`` (white) or ``rank, 7-file`` (black), with
grid origin (row 0, col 0) at TL. That gives:

- white camera: a1=BL, h1=BR, h8=TR, a8=TL
- black camera: a1=TR, h1=TL, h8=BL, a8=BR
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chess_robot.chess.board_mapper import Point
from chess_robot.perception.square_grounding import BoardCorners


def sort_corner_points(points: Any) -> np.ndarray:
    """Order 4 corner points as ``[top-left, top-right, bottom-right, bottom-left]``.

    Matches chesscog's warp source-point order. Splits by y (top/bottom), then by
    x within each pair (left/right). Assumes a roughly upright board quad.
    """
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (4, 2):
        raise ValueError(f"expected 4 corner points of shape (4, 2), got {array.shape}")
    by_y = array[np.argsort(array[:, 1])]
    top = by_y[:2]
    bottom = by_y[2:]
    top = top[np.argsort(top[:, 0])]  # [top-left, top-right]
    bottom = bottom[np.argsort(bottom[:, 0])]  # [bottom-left, bottom-right]
    return np.array([top[0], top[1], bottom[1], bottom[0]])


def _point(row: np.ndarray) -> Point:
    return (float(row[0]), float(row[1]))


def labeled_corners(points: Any, white_turn: bool) -> BoardCorners:
    """Map chesscog ``corners`` + ``white_turn`` to our labelled :class:`BoardCorners`."""
    top_left, top_right, bottom_right, bottom_left = sort_corner_points(points)
    if white_turn:
        return BoardCorners(
            a1=_point(bottom_left),
            h1=_point(bottom_right),
            h8=_point(top_right),
            a8=_point(top_left),
        )
    return BoardCorners(
        a1=_point(top_right),
        h1=_point(top_left),
        h8=_point(bottom_left),
        a8=_point(bottom_right),
    )


def manifest_entry(image: str, points: Any, white_turn: bool, fen: str) -> dict[str, Any]:
    """Build one ``prepare_piece_dataset.py`` manifest entry from a chesscog record."""
    corners = labeled_corners(points, white_turn)
    return {
        "image": image,
        "corners": {
            "a1": list(corners.a1),
            "h1": list(corners.h1),
            "h8": list(corners.h8),
            "a8": list(corners.a8),
        },
        "fen": fen,
    }
